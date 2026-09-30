"""로짓 스코어링 · 순열×이미지 TTA · 2단계 계단식 추론 · 중단 후 재개."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from data import CHOICES, PERMS, build_prompt, image_variant, load_image, options_of
from modeling import choice_scores, choice_token_groups, last_logits, make_inputs, temp_max_pixels
from stitch import STITCH_NOTE, make_stitched

PROB_COLS = [f"p_{c}" for c in CHOICES]


class Predictor:
    def __init__(self, model, processor, cfg, ocr_cache: Optional[Dict] = None):
        self.model, self.processor, self.cfg = model, processor, cfg
        self.groups = choice_token_groups(processor.tokenizer, cfg.letters)
        self.ocr_cache = ocr_cache or {}

    def _ocr_context(self, rid: str) -> str:
        if not self.cfg.use_ocr:
            return ""
        from ocr import build_ocr_context
        return build_ocr_context(self.ocr_cache.get(str(rid)))

    @torch.no_grad()
    def probs(self, image, row, perms: Sequence = PERMS[:1], img_kinds: Sequence[str] = ("orig",),
              preamble: str = "", max_pixels: Optional[int] = None) -> torch.Tensor:
        """한 문항의 a~d 확률. 보기 내용 기준으로 평균 (위치 편향 상쇄)."""
        opts = options_of(row)
        qtype = row.get("qtype") if self.cfg.use_type_hints else None
        ocr_ctx = self._ocr_context(row["id"])
        jobs = []
        for kind in img_kinds:
            img = image_variant(image, kind)
            for perm in perms:
                prompt = build_prompt(row["question"], [opts[j] for j in perm], qtype, ocr_ctx,
                                      self.cfg.letters, preamble)
                jobs.append((img, prompt, perm))

        total = torch.zeros(4)
        bs = self.cfg.infer_batch
        with temp_max_pixels(self.processor, max_pixels):
            for i in range(0, len(jobs), bs):
                chunk = jobs[i:i + bs]
                inputs = make_inputs(self.processor, [j[0] for j in chunk], [j[1] for j in chunk],
                                     self.cfg.chat_template_kwargs).to(self.model.device)
                p = choice_scores(last_logits(self.model, inputs).float(), self.groups).softmax(-1).cpu()
                for row_p, (_, _, perm) in zip(p, chunk):
                    for pos, j in enumerate(perm):
                        total[j] += row_p[pos]
        return total / len(jobs)


def margin_of(p: np.ndarray) -> np.ndarray:
    s = np.sort(p, axis=-1)
    return s[..., -1] - s[..., -2]


def _to_frame(records: List[dict]) -> pd.DataFrame:
    return pd.DataFrame(records, columns=["id", *PROB_COLS, "margin", "stage"])


def _resume(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path, dtype={"id": str})
        print(f"[resume] {path.name}: {len(df)}문항 이미 처리됨")
        return df
    return _to_frame([])


def run_fast_pass(pred: Predictor, df: pd.DataFrame, out_path: Path, save_every: int = 100) -> pd.DataFrame:
    """1단계: 단일 모델 + 순열 TTA. 전 문항 처리."""
    done = _resume(out_path)
    seen = set(done["id"])
    recs = done.to_dict("records")
    todo = df[~df["id"].isin(seen)]
    t0 = time.time()
    for n, (_, row) in enumerate(todo.iterrows(), 1):
        p = pred.probs(load_image(pred.cfg, row["path"]), row, PERMS[:pred.cfg.fast_perms]).numpy()
        recs.append({"id": row["id"], **dict(zip(PROB_COLS, p)), "margin": float(margin_of(p)), "stage": "fast"})
        if n % save_every == 0:
            _to_frame(recs).to_csv(out_path, index=False)
            print(f"  {n}/{len(todo)}  {(time.time() - t0) / n:.2f}s/문항")
    res = _to_frame(recs)
    res.to_csv(out_path, index=False)
    return res


def run_deep_pass(pred: Predictor, df: pd.DataFrame, fast: pd.DataFrame, adapters: Sequence[str],
                  out_path: Path, use_stitch: bool = True, save_every: int = 20) -> pd.DataFrame:
    """
    2단계: margin <= thr 문항만. 어댑터(fold 모델)별 × 순열 × 이미지 TTA 확률을 단순 평균하고,
    OCR 박스가 있으면 stitched 이미지 재추론 결과도 같은 가중치로 평균에 포함.
    adapters: 모델에 load_adapter로 올려 둔 어댑터 이름 목록 (단순 산술 평균, 성적순 선택 금지).
    """
    cfg = pred.cfg
    hard_ids = set(fast.loc[fast["margin"] <= cfg.margin_thr, "id"])
    print(f"[deep] 대상 {len(hard_ids)}/{len(fast)}문항 ({len(hard_ids) / max(len(fast), 1):.1%})")
    done = _resume(out_path)
    recs = done.to_dict("records")
    seen = set(done["id"])
    todo = df[df["id"].isin(hard_ids - seen)]
    t0 = time.time()
    for n, (_, row) in enumerate(todo.iterrows(), 1):
        img = load_image(cfg, row["path"])
        stitched = None
        if use_stitch and pred.ocr_cache.get(row["id"]):
            stitched = make_stitched(img, pred.ocr_cache[row["id"]]["items"])
        parts = []
        for name in adapters:
            pred.model.set_adapter(name)
            parts.append(pred.probs(img, row, PERMS[:cfg.deep_perms], cfg.image_tta))
            if stitched is not None:
                parts.append(pred.probs(stitched, row, PERMS[:cfg.deep_perms], ("orig",),
                                        preamble=STITCH_NOTE, max_pixels=cfg.stitch_max_pixels))
        p = torch.stack(parts).mean(0).numpy()
        recs.append({"id": row["id"], **dict(zip(PROB_COLS, p)), "margin": float(margin_of(p)), "stage": "deep"})
        if n % save_every == 0:
            _to_frame(recs).to_csv(out_path, index=False)
            print(f"  {n}/{len(todo)}  {(time.time() - t0) / n:.2f}s/문항")
    deep = _to_frame(recs)
    deep.to_csv(out_path, index=False)
    merged = fast.set_index("id")
    merged.update(deep.set_index("id"))
    return merged.reset_index()


def load_fold_adapters(base_model, adapter_dirs: Sequence[str]):
    """
    베이스 모델(어댑터 없이 로드) 하나에 여러 fold 어댑터를 올리고 set_adapter로 전환 (메모리 절약).
    반환: (PeftModel, 어댑터 이름 목록). 첫 어댑터가 활성 상태.
    """
    from peft import PeftModel
    names = [f"fold{k}" for k in range(len(adapter_dirs))]
    model = PeftModel.from_pretrained(base_model, str(adapter_dirs[0]), adapter_name=names[0])
    for name, d in zip(names[1:], adapter_dirs[1:]):
        model.load_adapter(str(d), adapter_name=name)
    model.set_adapter(names[0])
    model.eval()
    return model, names
