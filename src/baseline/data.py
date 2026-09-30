"""데이터 로딩·프롬프트·질문 유형·fold 분할·dev 합의 계산."""
from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps

CHOICES = ["a", "b", "c", "d"]

# 고정 순열 → TTA 재현성 (서로 다른 4지선다 순열은 24개가 상한)
PERMS = [(0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2),
         (3, 2, 1, 0), (0, 2, 1, 3), (1, 0, 3, 2), (2, 1, 0, 3)]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# 로딩
# ---------------------------------------------------------------------------
def extract_dataset(zip_path, dest="/content/data", force: bool = False) -> Path:
    """
    Drive의 대회 zip을 Colab 로컬 디스크로 해제하고, train.csv가 있는 폴더를 반환.
    - 로컬 디스크가 Drive보다 이미지 읽기가 훨씬 빠르므로 /content 아래로 풂
    - 이미 해제돼 있으면 건너뜀 (세션이 새로 시작되면 다시 풀림)
    - 한글 파일명이 깨지지 않도록 UTF-8 플래그가 없는 항목은 cp949로 복원
    """
    import shutil
    import zipfile

    zip_path, dest = Path(zip_path), Path(dest)
    if not zip_path.exists():
        raise FileNotFoundError(f"zip 없음: {zip_path}")
    found = list(dest.rglob("train.csv")) if dest.exists() else []
    if found and not force:
        print(f"[skip] 이미 해제됨: {found[0].parent}")
        return found[0].parent
    if dest.exists() and force:
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        for info in members:
            name = info.filename
            if not info.flag_bits & 0x800:          # UTF-8 플래그 없음 → 윈도우 한글 압축
                try:
                    name = name.encode("cp437").decode("cp949")
                except (UnicodeEncodeError, UnicodeDecodeError):
                    pass
            if name.startswith("__MACOSX/") or name.endswith(".DS_Store"):
                continue
            target = dest / name
            if not str(target.resolve()).startswith(str(dest.resolve())):
                raise ValueError(f"잘못된 경로: {name}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    print(f"해제 완료: {len(members)}개 항목 → {dest}")

    found = sorted(dest.rglob("train.csv"), key=lambda p: len(p.parts))
    if not found:
        raise FileNotFoundError("압축 안에서 train.csv를 찾지 못함")
    return found[0].parent


def check_paths(cfg, names=("train", "dev", "test"), n: int = 20) -> None:
    """csv의 path 컬럼이 data_dir 기준으로 실제 파일을 가리키는지 확인."""
    for name in names:
        df = pd.read_csv(Path(cfg.data_dir) / f"{name}.csv")
        paths = df["path"].astype(str).head(n)
        miss = [p for p in paths if not (Path(cfg.data_dir) / p).exists()]
        status = "OK" if not miss else f"누락 {len(miss)}/{len(paths)} 예: {miss[0]}"
        print(f"{name}.csv ({len(df)}행) 이미지 경로 {status}")


def load_csv(cfg, name: str) -> pd.DataFrame:
    df = pd.read_csv(Path(cfg.data_dir) / f"{name}.csv")
    df["id"] = df["id"].astype(str)
    if "answer" in df.columns:
        df["answer"] = df["answer"].astype(str).str.strip().str.lower()
    df["qtype"] = df.apply(lambda r: question_type(r["question"], options_of(r)), axis=1)
    return df


def options_of(row) -> List[str]:
    return ["" if pd.isna(row[c]) else str(row[c]).strip() for c in CHOICES]


def load_image(cfg, path: str) -> Image.Image:
    p = Path(str(path))
    if not p.is_absolute():
        p = Path(cfg.data_dir) / p
    with Image.open(p) as im:
        return ImageOps.exif_transpose(im).convert("RGB")   # EXIF 회전 보정은 필수 전처리


def image_variant(img: Image.Image, kind: str) -> Image.Image:
    """글자를 망가뜨리지 않는 약한 이미지 TTA만 허용 (반전·회전 금지)."""
    if kind == "orig":
        return img
    if kind == "bright":
        return ImageEnhance.Brightness(img).enhance(1.15)
    if kind == "contrast":
        return ImageEnhance.Contrast(img).enhance(1.15)
    if kind == "scale":
        w, h = img.size
        return img.resize((int(w * 1.15), int(h * 1.15)), Image.BICUBIC)
    raise ValueError(f"허용되지 않은 이미지 TTA: {kind}")


# ---------------------------------------------------------------------------
# 질문 유형 (프롬프트 힌트·층화 추출·유형별 분석에 공통 사용)
# ---------------------------------------------------------------------------
TYPE_RULES = [
    ("phone", r"전화|연락처|휴대폰"),
    ("price", r"가격|얼마|요금|비용|금액"),
    ("english", r"영어|영문|알파벳|english"),
    ("direction", r"어디로|방향|화살표|향하"),
    ("position", r"왼쪽|오른쪽|위쪽|아래쪽|상단|하단에|가운데|옆에"),
    ("color", r"색깔|색상|무슨 색|어떤 색"),
    ("number", r"몇|숫자|수치|개수|거리|번호|층|시간|날짜|%"),
    ("name", r"상호|가게|간판|이름|브랜드|회사|메뉴"),
]
_NUMERIC = re.compile(r"[\d\s,.\-:~%원/()a-zA-Z]*\d[\d\s,.\-:~%원/()a-zA-Z]*")

TYPE_HINTS = {
    "phone": "숫자를 한 자리씩 보기와 대조하고, 비슷한 숫자(1/7, 3/8, 0/6)를 구분하세요.",
    "price": "가격 숫자와 단위를 정확히 읽고, 해당 품목 줄의 가격인지 확인하세요.",
    "english": "영문 철자를 한 글자씩 대조하세요.",
    "direction": "화살표가 가리키는 쪽의 글자를 확인하세요.",
    "position": "질문이 가리키는 위치의 글자만 보세요.",
    "color": "글자나 배경의 실제 색을 확인하세요.",
    "number": "숫자와 단위를 한 자리씩 보기와 대조하세요.",
    "name": "가장 크게 쓰인 상호·표제 글자를 한 자씩 보기와 대조하세요.",
}


def question_type(question: str, options: Sequence[str]) -> str:
    q = str(question).lower()
    for t, pat in TYPE_RULES:
        if re.search(pat, q):
            return t
    if options and all(_NUMERIC.fullmatch(str(o) or "x") for o in options):
        return "number"
    return "other"


# ---------------------------------------------------------------------------
# 프롬프트 (학습·추론이 반드시 이 함수 하나를 공유)
# ---------------------------------------------------------------------------
def build_prompt(question: str, options: Sequence[str], qtype: Optional[str] = None,
                 ocr_context: str = "", letters: str = "ABCD", preamble: str = "") -> str:
    lines = []
    if preamble:
        lines += [preamble, ""]
    lines.append(f"질문: {str(question).strip()}")
    hint = TYPE_HINTS.get(qtype or "")
    if hint:
        lines.append(f"힌트: {hint}")
    if ocr_context:
        lines += ["", ocr_context]
    lines.append("")
    lines += [f"({L}) {o}" for L, o in zip(letters, options)]
    lines += ["", "정답 알파벳 하나로 답하세요."]
    return "\n".join(lines)


def shuffle_options(options: Sequence[str], label: int, rng: random.Random):
    """보기 순서 셔플 + 정답 인덱스 동기화."""
    perm = list(range(4))
    rng.shuffle(perm)
    return [options[j] for j in perm], perm.index(label)


# ---------------------------------------------------------------------------
# 분할·샘플링
# ---------------------------------------------------------------------------
def _strata(df: pd.DataFrame, n_min: int) -> pd.Series:
    s = df["qtype"].copy()
    small = s.value_counts()[lambda v: v < n_min].index
    return s.where(~s.isin(small), "other")


def add_folds(df: pd.DataFrame, n_folds: int, seed: int) -> pd.DataFrame:
    from sklearn.model_selection import StratifiedKFold
    df = df.copy()
    df["fold"] = -1
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, va) in enumerate(skf.split(df, _strata(df, n_folds))):
        df.iloc[va, df.columns.get_loc("fold")] = k
    return df


def stratified_sample(df: pd.DataFrame, n: Optional[int], seed: int) -> pd.DataFrame:
    """질문 유형 비율을 유지한 고정 표본 (희귀 유형이 빠지지 않게)."""
    if n is None or n >= len(df):
        return df
    strata = _strata(df, 2)
    frac = n / len(df)
    parts = [g.sample(max(1, round(len(g) * frac)), random_state=seed) for _, g in df.groupby(strata)]
    out = pd.concat(parts)
    if len(out) > n:
        out = out.sample(n, random_state=seed)
    elif len(out) < n:  # 반올림으로 부족하면 나머지에서 채움
        out = pd.concat([out, df.drop(out.index).sample(n - len(out), random_state=seed)])
    return out.sort_index()


# ---------------------------------------------------------------------------
# dev 합의 (응답1~5 / answer1~5 컬럼 자동 인식)
# ---------------------------------------------------------------------------
def dev_vote_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if re.fullmatch(r"(answer|응답)\s*_?\d+", str(c).strip().lower())]
    if not cols:
        raise ValueError(f"dev 응답 컬럼을 찾지 못함: {list(df.columns)}")
    return cols


def dev_consensus(df: pd.DataFrame) -> pd.DataFrame:
    cols = dev_vote_columns(df)
    recs = []
    for _, row in df.iterrows():
        votes = [str(row[c]).strip().lower() for c in cols if pd.notna(row[c])]
        votes = [v for v in votes if v in CHOICES]
        cnt = {c: votes.count(c) for c in CHOICES}
        top_n = max(cnt.values()) if votes else 0
        tops = [c for c in CHOICES if cnt[c] == top_n and top_n > 0]
        recs.append({
            "gold": tops[0] if len(tops) == 1 else None,   # 동률이면 gold 없음
            "top_votes": top_n,
            "n_votes": len(votes),
            "tie": len(tops) > 1,
            **{f"votes_{c}": cnt[c] for c in CHOICES},
        })
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(recs)], axis=1)
