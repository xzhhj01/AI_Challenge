"""LoRA 학습: choice CE + 보기 셔플 + OCR 드롭아웃 + 나머지 스텝 포함 grad accum + best 체크포인트."""
from __future__ import annotations

import json
import math
import random
import time
from typing import Dict, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from data import CHOICES, build_prompt, load_image, options_of, seed_everything, shuffle_options, stratified_sample
from modeling import (attach_lora, canonical_ids, choice_token_groups, compute_loss, last_logits,
                      load_model, make_inputs, set_pixels)


class VQADataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg, ocr_cache: Optional[Dict] = None):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.ocr_cache = ocr_cache or {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        cfg, row = self.cfg, self.df.iloc[i]
        rng = random.Random(cfg.seed * 1_000_003 + i + 7919 * torch.initial_seed() % 1_000_000)
        opts, label = shuffle_options(options_of(row), CHOICES.index(row["answer"]), rng)
        ocr_ctx = ""
        if cfg.use_ocr and rng.random() >= cfg.ocr_dropout:
            from ocr import build_ocr_context
            ocr_ctx = build_ocr_context(self.ocr_cache.get(str(row["id"])))
        prompt = build_prompt(row["question"], opts, row["qtype"] if cfg.use_type_hints else None,
                              ocr_ctx, cfg.letters)
        return {"image": load_image(cfg, row["path"]), "prompt": prompt, "label": label}


def _worker_init(wid):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)


def train(cfg, train_df: pd.DataFrame, eval_df: pd.DataFrame, run_name: str,
          ocr_cache: Optional[Dict] = None, eval_fn=None) -> Dict:
    """
    eval_fn(model, processor, df) -> accuracy. 학습 중 cfg.eval_every 스텝마다 호출해
    최고 정확도일 때만 어댑터를 Drive에 저장.
    """
    seed_everything(cfg.seed)
    if cfg.n_train:
        train_df = stratified_sample(train_df, cfg.n_train, cfg.seed)
    model, processor = load_model(cfg, for_training=True)
    groups = choice_token_groups(processor.tokenizer, cfg.letters)
    canon = canonical_ids(processor.tokenizer, cfg.letters)
    model = attach_lora(model, cfg)

    g = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(VQADataset(train_df, cfg, ocr_cache), batch_size=1, shuffle=True,
                        collate_fn=lambda b: b[0], num_workers=2, generator=g, worker_init_fn=_worker_init)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_micro = len(loader) * cfg.epochs
    total_opt = math.ceil(total_micro / cfg.grad_accum)
    from transformers import get_cosine_schedule_with_warmup
    sched = get_cosine_schedule_with_warmup(opt, int(total_opt * cfg.warmup_ratio), total_opt)
    last_group = (total_micro - 1) // cfg.grad_accum

    out_dir = cfg.out("adapters", run_name, "x").parent
    cfg.save(out_dir / "config.json")
    eval_sub = stratified_sample(eval_df, cfg.eval_n, cfg.seed)
    history, best, micro, step, t0 = [], -1.0, 0, 0, time.time()
    running = 0.0

    model.train()
    for epoch in range(cfg.epochs):
        for ex in loader:
            inputs = make_inputs(processor, [ex["image"]], [ex["prompt"]],
                                 cfg.chat_template_kwargs).to(model.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = last_logits(model, inputs).float()
            loss = compute_loss(logits, torch.tensor([ex["label"]], device=logits.device),
                                groups, canon, cfg.loss_type)
            # 마지막 누적 구간이 grad_accum보다 짧아도 평균이 맞도록 실제 구간 길이로 나눔
            window = cfg.grad_accum if micro // cfg.grad_accum < last_group else total_micro - last_group * cfg.grad_accum
            (loss / window).backward()
            running += loss.item()
            micro += 1

            if micro % cfg.grad_accum == 0 or micro == total_micro:   # 나머지 스텝도 반영
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step == 5:
                    per = (time.time() - t0) / 5
                    print(f"[예상] 스텝당 {per:.1f}s × {total_opt}스텝 ≈ {per * total_opt / 60:.0f}분")
                if step % 10 == 0:
                    print(f"step {step}/{total_opt}  loss {running / (10 * cfg.grad_accum):.4f}  "
                          f"lr {sched.get_last_lr()[0]:.2e}")
                    running = 0.0
                if eval_fn and (step % cfg.eval_every == 0 or micro == total_micro):
                    model.eval()
                    set_pixels(processor, cfg.min_pixels, cfg.infer_max_pixels)
                    acc = eval_fn(model, processor, eval_sub)
                    set_pixels(processor, cfg.min_pixels, cfg.train_max_pixels)
                    model.train()
                    history.append({"step": step, "acc": acc})
                    print(f"  [eval] step {step}  acc {acc:.4f}  (best {max(best, acc):.4f})")
                    if acc > best:
                        best = acc
                        model.save_pretrained(out_dir)
                        (out_dir / "best.json").write_text(json.dumps({"step": step, "acc": acc}))

    if not eval_fn:
        model.save_pretrained(out_dir)
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    del model
    torch.cuda.empty_cache()
    return {"run": run_name, "best_acc": best, "adapter": str(out_dir), "minutes": (time.time() - t0) / 60}
