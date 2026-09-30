"""Predict the question-routed test subset with Qwen3.5 and four cyclic orders.

Supply the competition test images and the trained language LoRA adapter.
The output probabilities are in the ORIGINAL a/b/c/d choice order. This
program does not need labels, baseline predictions, audit files, or an API.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from baseline import data, modeling
from baseline.config import Config
from question_routes import route_question

MODEL_ID = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
PROB = ("p_a", "p_b", "p_c", "p_d")
PERMS = ((0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2))


def predict_one(row, model, processor, cfg, groups, torch) -> dict:
    """Batch four views of one original image, then undo each choice order."""
    image = data.load_image(cfg, row["path"])
    choices = data.options_of(row)
    prompts = [data.build_prompt(
        row["question"], [choices[index] for index in order],
        row["qtype"] if cfg.use_type_hints else None, "", cfg.letters)
        for order in PERMS]
    inputs = modeling.make_inputs(
        processor, [image] * 4, prompts, cfg.chat_template_kwargs).to(model.device)
    with torch.inference_mode():
        logits = modeling.last_logits(model, inputs).float()
        position_probs = modeling.choice_scores(logits, groups).softmax(-1).cpu().numpy()
    if position_probs.shape != (4, 4):
        raise RuntimeError(f"Unexpected probability shape for {row['id']}")
    mapped = np.zeros((4, 4), dtype=float)
    for perm_index, order in enumerate(PERMS):
        for position, original in enumerate(order):
            mapped[perm_index, original] = position_probs[perm_index, position]
    mean = mapped.mean(axis=0)
    if (not np.isfinite(mean).all() or np.any(mean < 0)
            or not math.isclose(float(mean.sum()), 1, abs_tol=1e-4)):
        raise RuntimeError(f"Invalid four-TTA probability for {row['id']}")
    return {"id": row["id"], **dict(zip(PROB, mean.tolist()))}


def read_partial(path: Path, selected_ids: list[str]) -> list[dict]:
    if not path.exists():
        return []
    frame = pd.read_csv(path, dtype={"id": str})
    if list(frame.columns) != ["id", *PROB] or frame.id.tolist() != selected_ids[:len(frame)]:
        raise ValueError("Partial predictions do not match the selected test prefix")
    values = frame[list(PROB)].to_numpy(dtype=float)
    if (not np.isfinite(values).all() or np.any(values < 0)
            or not np.allclose(values.sum(axis=1), 1, atol=1e-4)):
        raise ValueError("Partial predictions contain invalid probabilities")
    return frame.to_dict("records")


def save_partial(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(".tmp")
    pd.DataFrame(records, columns=["id", *PROB]).to_csv(temporary, index=False)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing test.csv and its images")
    parser.add_argument("--adapter-dir", type=Path, required=True,
                        help="Directory containing adapter_model.safetensors and adapter_config.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args()
    if not (args.adapter_dir / "adapter_model.safetensors").is_file():
        raise FileNotFoundError("LoRA adapter_model.safetensors is missing")
    if not (args.adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError("LoRA adapter_config.json is missing")
    cfg = Config(zip_path=args.data_dir, extract_dir=args.data_dir,
                 data_dir=args.data_dir, work_dir=args.out_dir, model_id=args.model_id,
                 min_pixels=1024 * 1024, infer_max_pixels=1600 * 1600,
                 infer_batch=4, use_type_hints=True, use_ocr=False)
    test = data.load_csv(cfg, "test")
    if test.id.duplicated().any():
        raise ValueError("Duplicate test IDs")
    selected = test[test.question.map(route_question).ne("unchanged")].copy()
    if len(test) == 6714 and len(selected) != 2792:
        raise RuntimeError("Question route differs from the recorded 2792 of 6714 test rows")
    ids = selected.id.tolist()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    final = args.out_dir / "q35_pattern_probs.csv"
    partial = args.out_dir / "q35_pattern_probs.partial.csv"
    if final.exists():
        raise FileExistsError(f"Completed prediction exists: {final}")
    records = read_partial(partial, ids)
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for Qwen3.5 inference")
    model = modeling.AutoVLM.from_pretrained(
        args.model_id, revision=args.revision, dtype=torch.bfloat16,
        attn_implementation=modeling._attn_impl(), device_map="cuda")
    loaded_revision = getattr(model.config, "_commit_hash", None)
    if loaded_revision and loaded_revision != args.revision:
        raise RuntimeError("Loaded model revision differs")
    processor = AutoProcessor.from_pretrained(args.model_id, revision=args.revision)
    processor.tokenizer.padding_side = "left"
    modeling.set_pixels(processor, cfg.min_pixels, cfg.infer_max_pixels)
    model = PeftModel.from_pretrained(model, args.adapter_dir, is_trainable=False)
    model.eval()
    groups = modeling.choice_token_groups(processor.tokenizer, cfg.letters)
    for _, row in selected.iloc[len(records):].iterrows():
        records.append(predict_one(row, model, processor, cfg, groups, torch))
        if len(records) % 25 == 0:
            save_partial(partial, records)
            print(f"Predicted {len(records)}/{len(selected)}", flush=True)
    if [record["id"] for record in records] != ids:
        raise AssertionError("Final prediction ID order differs from question route")
    save_partial(partial, records)
    os.replace(partial, final)
    print(f"Saved {len(records)} four-TTA question-routed predictions to {final}")


if __name__ == "__main__":
    main()
