"""모델·프로세서 로딩, 해상도 설정, 선지 토큰 그룹, LoRA 부착, 로짓 계산."""
from __future__ import annotations

import contextlib
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoProcessor

try:  # transformers 4.5x+ / 5.x
    from transformers import AutoModelForImageTextToText as AutoVLM
except ImportError:  # 구버전 대비
    from transformers import AutoModelForVision2Seq as AutoVLM

VISION_KEYS = ("visual", "vision_tower", "vision_model")
LANG_LEAVES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
VISION_LEAVES = ("qkv", "proj", "fc1", "fc2", "gate_proj", "up_proj", "down_proj", "linear_fc1", "linear_fc2")


def _attn_impl() -> str:
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def set_pixels(processor, min_pixels: int, max_pixels: int) -> None:
    """
    해상도 예산을 processor에 기록. 실제 적용은 make_inputs에서 PIL로 직접 리사이즈하므로
    transformers 버전별 인자 차이와 무관하게 확실히 적용됨.
    프로세서 자체의 상한은 넉넉하게 열어 두어 우리가 맞춘 크기를 다시 줄이지 않게 함.
    """
    processor._vqa_pixels = (int(min_pixels), int(max_pixels))
    ip = processor.image_processor
    hi, lo = 64 * 1024 * 1024, 32 * 32
    for name, val in (("min_pixels", lo), ("max_pixels", hi)):
        if hasattr(ip, name):
            setattr(ip, name, val)
    if isinstance(getattr(ip, "size", None), dict):
        ip.size = {**ip.size, "shortest_edge": lo, "longest_edge": hi}


def get_pixels(processor):
    return getattr(processor, "_vqa_pixels", None)


def fit_pixels(img, min_pixels: int, max_pixels: int):
    """종횡비를 유지한 채 전체 픽셀 수를 [min_pixels, max_pixels] 범위로 맞춤."""
    from PIL import Image
    w, h = img.size
    area = w * h
    if area > max_pixels:
        s = (max_pixels / area) ** 0.5
    elif area < min_pixels:
        s = (min_pixels / area) ** 0.5
    else:
        return img
    return img.resize((max(28, int(w * s)), max(28, int(h * s))), Image.BICUBIC)


@contextlib.contextmanager
def temp_max_pixels(processor, max_pixels: Optional[int]):
    """stitched 이미지처럼 캔버스가 큰 입력에만 일시적으로 해상도 상향."""
    before = get_pixels(processor)
    if max_pixels is None or before is None:
        yield
        return
    processor._vqa_pixels = (before[0], max_pixels)
    try:
        yield
    finally:
        processor._vqa_pixels = before


def load_model(cfg, adapter_path: Optional[str] = None, for_training: bool = False):
    kwargs = dict(attn_implementation=_attn_impl(), device_map="cuda")
    try:
        model = AutoVLM.from_pretrained(cfg.model_id, dtype=torch.bfloat16, **kwargs)
    except TypeError:  # 구버전은 torch_dtype
        model = AutoVLM.from_pretrained(cfg.model_id, torch_dtype=torch.bfloat16, **kwargs)
    processor = AutoProcessor.from_pretrained(cfg.model_id)
    processor.tokenizer.padding_side = "left"   # 배치 추론 시 마지막 위치가 실제 토큰이 되도록
    set_pixels(processor, cfg.min_pixels, cfg.train_max_pixels if for_training else cfg.infer_max_pixels)

    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path, adapter_name="default",
                                          is_trainable=for_training)
    if not for_training:
        model.eval()
    return model, processor


def attach_lora(model, cfg):
    from peft import LoraConfig, get_peft_model
    targets = []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        leaf = name.split(".")[-1]
        if any(k in name for k in VISION_KEYS):
            if cfg.vision_lora and leaf in VISION_LEAVES:
                targets.append(name)
        elif leaf in LANG_LEAVES:
            targets.append(name)
    if not targets:
        raise RuntimeError("LoRA 대상 모듈을 찾지 못함 — model.named_modules()를 확인하세요.")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lcfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                      target_modules=targets, bias="none")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# 선지 토큰: "A", " A", "a", " a" 중 단일 토큰을 모두 모아 logsumexp로 합산
# ---------------------------------------------------------------------------
def choice_token_groups(tokenizer, letters: str = "ABCD") -> List[List[int]]:
    groups = []
    for L in letters:
        ids = set()
        for s in (L, " " + L, L.lower(), " " + L.lower()):
            t = tokenizer.encode(s, add_special_tokens=False)
            if len(t) == 1:
                ids.add(t[0])
        if not ids:
            raise ValueError(f"'{L}'의 단일 토큰을 찾지 못함")
        groups.append(sorted(ids))
    flat = [i for g in groups for i in g]
    assert len(flat) == len(set(flat)), "선지 토큰 그룹이 겹침"
    return groups


def canonical_ids(tokenizer, letters: str = "ABCD") -> List[int]:
    return [tokenizer.encode(L, add_special_tokens=False)[0] for L in letters]


def choice_scores(logits: torch.Tensor, groups: Sequence[Sequence[int]]) -> torch.Tensor:
    """logits [B, V] → [B, 4]"""
    return torch.stack([torch.logsumexp(logits[:, list(g)], dim=-1) for g in groups], dim=-1)


def make_inputs(processor, images, prompts, chat_kwargs=None):
    chat_kwargs = chat_kwargs or {}
    budget = get_pixels(processor)
    if budget:
        images = [fit_pixels(im, *budget) for im in images]
    texts = []
    for p in prompts:
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]}]
        texts.append(processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                   **chat_kwargs))
    return processor(text=texts, images=list(images), padding=True, return_tensors="pt")


def check_resolution(processor, image, budgets=(768 * 768, 1024 * 1024, 1280 * 1280)):
    """해상도 설정이 실제로 먹는지 확인: 예산별 이미지 토큰 수가 달라져야 정상."""
    before = get_pixels(processor)
    for mx in budgets:
        set_pixels(processor, before[0] if before else 256 * 28 * 28, mx)
        x = make_inputs(processor, [image], ["확인"])
        grid = x.get("image_grid_thw")
        n_tok = int(grid.prod(-1).sum()) if grid is not None else -1
        print(f"max_pixels {mx:>9,} → 입력 크기 {fit_pixels(image, 1, mx).size}, 이미지 패치 수 {n_tok}")
    if before:
        set_pixels(processor, *before)


def last_logits(model, inputs) -> torch.Tensor:
    """마지막 위치 로짓만 계산 (메모리 절약). 지원 안 하는 버전이면 전체 계산."""
    try:
        out = model(**inputs, logits_to_keep=1)
    except TypeError:
        out = model(**inputs)
    return out.logits[:, -1, :]


def compute_loss(logits: torch.Tensor, labels: torch.Tensor, groups, canon, kind: str) -> torch.Tensor:
    """
    choice_ce: a~d 4개 점수에만 CE → 추론(로짓 비교)과 학습이 정확히 일치.
               프롬프트만 입력하고 마지막 위치를 보므로 별도 라벨 마스킹이 필요 없음.
    lm       : 전체 어휘 중 정답 문자 토큰에 CE (비교군).
    """
    if kind == "choice_ce":
        return F.cross_entropy(choice_scores(logits, groups), labels)
    if kind == "lm":
        return F.cross_entropy(logits, torch.tensor(canon, device=logits.device)[labels])
    raise ValueError(kind)
