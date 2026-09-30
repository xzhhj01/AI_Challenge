"""Train the final Qwen3.5-9B language LoRA with choice CE on fold != 0.

The model weights are fetched from the public Hugging Face model ID. The
competition train.csv and images are supplied by --data-dir. This script
never trains on fold 0 or on images identical to fold-0 images.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import random
import re
from pathlib import Path

from baseline import data, modeling
from baseline.config import Config

MODEL_ID = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
MICROBATCH = 4
GRAD_ACCUM = 2


def image_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_data(frame, data_dir: Path, fold: int = 0):
    """Recreate the stratified fold and remove cross-fold identical images."""
    if frame.id.duplicated().any() or set(frame.answer) - set("abcd"):
        raise ValueError("Duplicate IDs or invalid answer labels in train.csv")
    root = data_dir.resolve()
    hashes = []
    for raw in frame.path:
        image = (root / str(raw)).resolve()
        if not image.is_relative_to(root) or not image.is_file():
            raise ValueError(f"Missing or escaped training image: {raw}")
        hashes.append(image_hash(image))
    tagged = frame.copy()
    tagged["image_sha256"] = hashes
    train = tagged[tagged.fold != fold].copy()
    holdout = tagged[tagged.fold == fold].copy()
    overlap = set(holdout.image_sha256)
    removed = train[train.image_sha256.isin(overlap)].copy()
    train = train[~train.image_sha256.isin(overlap)].copy()
    if set(train.image_sha256) & overlap:
        raise AssertionError("Identical holdout images remain in training")
    return train, holdout, removed


def lora_targets(model, torch):
    """Select the exact language attention/MLP Linear modules used by training."""
    def eligible(name):
        if any(word in name for word in ("visual", "vision_tower", "vision_model")):
            return False
        leaf = name.rsplit(".", 1)[-1]
        return ((".mlp." in name and leaf in {"gate_proj", "up_proj", "down_proj"})
                or (".self_attn." in name and leaf in {"q_proj", "k_proj", "v_proj", "o_proj"})
                or (".linear_attn." in name and leaf in
                    {"in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"}))

    targets = [name for name, module in model.named_modules()
               if isinstance(module, torch.nn.Linear) and eligible(name)]
    by_layer = {index: set() for index in range(32)}
    for name in targets:
        match = re.search(r"(?:^|\.)layers\.(\d+)\.(.*)$", name)
        if match:
            by_layer[int(match.group(1))].add(match.group(2))
    mlp = {f"mlp.{part}" for part in ("gate_proj", "up_proj", "down_proj")}
    for index in range(32):
        attention = ({f"self_attn.{part}" for part in ("q_proj", "k_proj", "v_proj", "o_proj")}
                     if index % 4 == 3 else
                     {f"linear_attn.{part}" for part in
                      ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")})
        if by_layer[index] != mlp | attention:
            raise RuntimeError(f"Unexpected language modules in layer {index}")
    if len(targets) != 248:
        raise RuntimeError(f"Expected 248 language LoRA modules, got {len(targets)}")
    return targets


class ChoiceDataset:
    def __init__(self, frame, cfg, torch):
        self.frame = frame.reset_index(drop=True)
        self.cfg = cfg
        self.torch = torch

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        rng = random.Random(self.cfg.seed * 1_000_003 + index
                            + 7919 * self.torch.initial_seed() % 1_000_000)
        options, label = data.shuffle_options(data.options_of(row),
                                               data.CHOICES.index(row.answer), rng)
        prompt = data.build_prompt(row.question, options, row.qtype, "", self.cfg.letters)
        return {"image": data.load_image(self.cfg, row.path),
                "prompt": prompt, "label": label}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing train.csv and image paths")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="New output directory for adapter_final")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--check-data", action="store_true",
                        help="Check folds and duplicate images without loading a model")
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError("Output directory must be empty")
    cfg = Config(zip_path=args.data_dir, extract_dir=args.data_dir,
                 data_dir=args.data_dir, work_dir=args.out_dir, model_id=args.model_id,
                 train_max_pixels=768 * 768, min_pixels=256 * 28 * 28,
                 infer_max_pixels=1600 * 1600, seed=42, epochs=1, lr=1e-4,
                 lora_r=16, lora_alpha=32, lora_dropout=0.05,
                 grad_accum=GRAD_ACCUM, loss_type="choice_ce", vision_lora=False,
                 use_type_hints=True, use_ocr=False)
    frame = data.add_folds(data.load_csv(cfg, "train"), cfg.n_folds, cfg.seed)
    train, holdout, removed = split_data(frame, args.data_dir, cfg.holdout_fold)
    if len(frame) == 6714 and (len(train), len(holdout), len(removed)) != (5369, 1343, 2):
        raise RuntimeError("Competition split differs from the recorded 5369/1343/2 split")
    print(f"train={len(train)} holdout={len(holdout)} duplicate_images_removed={len(removed)}")
    if args.check_data:
        return
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoProcessor, get_cosine_schedule_with_warmup

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for training")
    data.seed_everything(cfg.seed)
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(cfg.seed)).tolist()
    train = train.iloc[order].reset_index(drop=True)
    model = modeling.AutoVLM.from_pretrained(
        args.model_id, revision=args.revision, dtype=torch.bfloat16,
        attn_implementation=modeling._attn_impl(), device_map="cuda")
    loaded_revision = getattr(model.config, "_commit_hash", None)
    if loaded_revision and loaded_revision != args.revision:
        raise RuntimeError("Loaded model revision differs")
    processor = AutoProcessor.from_pretrained(args.model_id, revision=args.revision)
    processor.tokenizer.padding_side = "left"
    modeling.set_pixels(processor, cfg.min_pixels, cfg.train_max_pixels)
    targets = lora_targets(model, torch)
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, target_modules=targets, bias="none"))
    if any(p.requires_grad and any(key in name for key in
           ("visual", "vision_tower", "vision_model"))
           for name, p in model.named_parameters()):
        raise RuntimeError("Vision module became trainable")
    dataset = ChoiceDataset(train, cfg, torch)
    loader = DataLoader(dataset, batch_size=MICROBATCH, shuffle=False,
                        collate_fn=lambda examples: examples, num_workers=0)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = math.ceil(len(loader) / cfg.grad_accum)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(steps * cfg.warmup_ratio), steps)
    groups = modeling.choice_token_groups(processor.tokenizer, cfg.letters)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    for micro, batch in enumerate(loader):
        inputs = modeling.make_inputs(
            processor, [item["image"] for item in batch],
            [item["prompt"] for item in batch], cfg.chat_template_kwargs).to(model.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(**inputs, logits_to_keep=1)
            scores = modeling.choice_scores(output.logits[:, -1, :].float(), groups)
        labels = torch.tensor([item["label"] for item in batch], device=scores.device)
        window_start = (micro // cfg.grad_accum) * MICROBATCH * cfg.grad_accum
        window_count = min(MICROBATCH * cfg.grad_accum, len(dataset) - window_start)
        (F.cross_entropy(scores, labels, reduction="sum") / window_count).backward()
        if (micro + 1) % cfg.grad_accum == 0 or micro + 1 == len(loader):
            torch.nn.utils.clip_grad_norm_(parameters, cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
    adapter = args.out_dir / "adapter_final"
    adapter.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(adapter)
    print(f"Saved final one-epoch adapter to {adapter}")


if __name__ == "__main__":
    main()
