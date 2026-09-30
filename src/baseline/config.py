"""모든 설정을 한 곳에서 관리 (학습·추론 불일치 방지 + 재현성)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class Config:
    # ---- 경로 ---------------------------------------------------------------
    zip_path: Path = Path("data/dataset.zip")  # Drive의 대회 데이터 zip (노트북과 동일)
    extract_dir: Path = Path("data")                     # 로컬 디스크에 해제 (Drive보다 빠름)
    data_dir: Path = Path("data")                        # 해제 후 train.csv가 있는 폴더로 자동 설정
    work_dir: Path = Path("runs")  # Drive: 세션이 끊겨도 산출물 유지

    # ---- 모델 ---------------------------------------------------------------
    # 2단계 zero-shot 비교에서 확정한 주력 모델 (세션이 끊겨 cfg를 새로 만들어도 유지되도록 기본값으로 고정)
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    chat_template_kwargs: dict = field(default_factory=lambda: {"enable_thinking": False})  # Qwen3 계열 사고 모드 끔

    # ---- 해상도 (픽셀 수, 학습/추론 분리) ------------------------------------
    min_pixels: int = 256 * 28 * 28
    train_max_pixels: int = 1024 * 1024
    infer_max_pixels: int = 1600 * 1600    # 2단계 비교 결과 'native' 예산으로 확정
    stitch_max_pixels: int = 1792 * 1792   # 이어붙인 이미지는 캔버스가 커지므로 상향

    # ---- 프롬프트 -----------------------------------------------------------
    letters: str = "ABCD"          # 모델에는 대문자로 제시, 제출은 소문자
    use_type_hints: bool = True
    use_ocr: bool = False          # ablation에서 이득 확인 후 True
    ocr_dropout: float = 0.2       # 학습 시 OCR 블록을 빼는 비율

    # ---- LoRA ---------------------------------------------------------------
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    vision_lora: bool = False

    # ---- 학습 ---------------------------------------------------------------
    loss_type: str = "choice_ce"   # "choice_ce" | "lm"
    lr: float = 1e-4
    epochs: int = 1
    grad_accum: int = 8
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    eval_every: int = 100          # optimizer step 단위
    eval_n: int = 300              # 학습 중 빠른 평가 문항 수
    n_train: Optional[int] = None  # None = 전체, 500 → 2000 → 전체 순으로 확대

    # ---- 검증 ---------------------------------------------------------------
    n_folds: int = 5
    holdout_fold: int = 0          # 빠른 탐색용 고정 홀드아웃 = fold 0

    # ---- 추론 ---------------------------------------------------------------
    fast_perms: int = 4            # 1단계 순열 수
    deep_perms: int = 4            # 2단계 순열 수 (× 이미지 TTA)
    image_tta: Tuple[str, ...] = ("orig", "bright", "scale")
    margin_thr: float = 0.8        # 1단계 확정 기준 (train 검증으로만 조정)
    infer_batch: int = 8

    # ---- dev 선별 투입 -------------------------------------------------------
    dev_margin_thr: float = 0.95
    dev_min_votes: int = 2
    dev_max_ratio: float = 0.2

    seed: int = 42

    def out(self, *parts: str) -> Path:
        p = Path(self.work_dir).joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def save(self, path: Path) -> None:
        d = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(self).items()}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")