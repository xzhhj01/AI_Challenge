"""정확도 측정 하네스: 유형별 정확도, 오답 표, dev 보조 지표, 캘리브레이션, 라벨 오류 후보."""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd
import torch

from data import CHOICES, PERMS, dev_consensus, load_image, options_of, stratified_sample
from infer import PROB_COLS, Predictor, margin_of


def predict_frame(pred: Predictor, df: pd.DataFrame, n_perms: int = 1, img_kinds=("orig",)) -> pd.DataFrame:
    recs, t0 = [], time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for _, row in df.iterrows():
        p = pred.probs(load_image(pred.cfg, row["path"]), row, PERMS[:n_perms], img_kinds).numpy()
        recs.append({"id": row["id"], **dict(zip(PROB_COLS, p))})
    res = pd.DataFrame(recs)
    probs = res[PROB_COLS].to_numpy()
    res["pred"] = [CHOICES[i] for i in probs.argmax(1)]
    res["margin"] = margin_of(probs)
    res.attrs["sec_per_item"] = (time.time() - t0) / max(len(df), 1)
    res.attrs["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    return res


def evaluate(pred: Predictor, df: pd.DataFrame, gold_col: str = "answer", n: Optional[int] = None,
             n_perms: int = 1, name: str = "", show_wrong: int = 20, verbose: bool = True) -> pd.DataFrame:
    sub = stratified_sample(df, n, pred.cfg.seed) if n else df
    res = predict_frame(pred, sub, n_perms)
    meta = dict(res.attrs)   # merge 시 pandas 버전에 따라 attrs가 사라지므로 보관 후 복원
    res = res.merge(sub[["id", "question", "qtype", *CHOICES, gold_col]].rename(columns={gold_col: "gold"}), on="id")
    res.attrs.update(meta)
    res["correct"] = res["pred"] == res["gold"]
    if verbose:
        print(f"[{name}] acc {res['correct'].mean():.4f}  (n={len(res)}, "
              f"{res.attrs.get('sec_per_item', 0):.2f}s/문항, peak {res.attrs.get('peak_gb', 0):.1f}GB)")
        print(res.groupby("qtype")["correct"].agg(["mean", "count"]).sort_values("count", ascending=False).round(3))
        if show_wrong:
            cols = ["id", "qtype", "question", *CHOICES, "pred", "gold", "margin"]
            print(res.loc[~res["correct"], cols].head(show_wrong).to_string(max_colwidth=30))
    return res


def make_eval_fn(cfg, ocr_cache=None, n_perms: int = 1):
    """train()에 넘길 빠른 평가 함수."""
    def fn(model, processor, df):
        pred = Predictor(model, processor, cfg, ocr_cache)
        return float(evaluate(pred, df, n_perms=n_perms, verbose=False)["correct"].mean())
    return fn


def dev_hard_set(dev_df: pd.DataFrame) -> pd.DataFrame:
    """dev 3표 이상 + 동률 없는 문항 = '어려운 문항' 보조 지표용 (주 지표 아님)."""
    cons = dev_consensus(dev_df)
    hard = cons[(cons["top_votes"] >= 3) & (~cons["tie"])].copy()
    hard["answer"] = hard["gold"]
    return hard


def calibration_report(holdout_res: pd.DataFrame, dev_res: pd.DataFrame) -> pd.DataFrame:
    """사람이 헷갈린 dev 문항에서 모델 margin이 낮아지는지 확인 (불확실성 신호 점검)."""
    rows = []
    for name, r in (("train 홀드아웃", holdout_res), ("dev 어려운 문항", dev_res)):
        m = r["margin"]
        rows.append({"set": name, "mean_margin": m.mean(), "margin>0.95": (m > 0.95).mean(),
                     "margin<=0.8": (m <= 0.8).mean(),
                     "acc(margin>0.8)": r.loc[m > 0.8, "correct"].mean() if "correct" in r else np.nan,
                     "acc(margin<=0.8)": r.loc[m <= 0.8, "correct"].mean() if "correct" in r else np.nan})
    out = pd.DataFrame(rows)
    print(out.round(3).to_string(index=False))
    return out


def find_label_errors(res: pd.DataFrame, thr: float = 0.95) -> pd.DataFrame:
    """train에서 모델이 thr 이상 확신으로 다른 답을 고른 문항 → 라벨 오류 후보 (육안 확인용)."""
    top = res[PROB_COLS].max(axis=1)
    cand = res[(res["pred"] != res["gold"]) & (top >= thr)]
    print(f"라벨 오류 후보 {len(cand)}개")
    return cand.sort_values("margin", ascending=False)
