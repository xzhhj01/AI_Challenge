"""Train or run the Qwen3-VL + InternVL3.5 baseline cascade."""
from __future__ import annotations

import argparse
import dataclasses
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'baseline'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['train', 'predict'], required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--adapter-dir', type=Path)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError('Use a new output directory to avoid mixing runs')
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    import torch
    import config, data, modeling, infer

    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required')
    cfg = config.Config(data_dir=args.data_dir, work_dir=args.out_dir,
                        model_id='Qwen/Qwen3-VL-8B-Instruct',
                        min_pixels=256 * 28 * 28, train_max_pixels=1024**2,
                        infer_max_pixels=1600**2, use_type_hints=True,
                        use_ocr=False, vision_lora=False, fast_perms=4,
                        deep_perms=4, margin_thr=0.8)
    data.seed_everything(cfg.seed)
    cfg.save(args.out_dir / 'config.json')
    if args.mode == 'train':
        import train, evaluate
        all_train = data.add_folds(data.load_csv(cfg, 'train'), cfg.n_folds, cfg.seed)
        holdout = all_train[all_train.fold == cfg.holdout_fold]
        train_part = all_train[all_train.fold != cfg.holdout_fold]
        train.train(cfg, train_part, holdout, 'holdout_best', {},
                    eval_fn=evaluate.make_eval_fn(cfg, {}))
        gc.collect()
        torch.cuda.empty_cache()
        train.train(cfg, all_train, holdout, 'full_train', {}, eval_fn=None)
        return

    if args.adapter_dir is None:
        parser.error('--adapter-dir is required in predict mode')
    adapters = [args.adapter_dir / name for name in ('holdout_best', 'full_train')]
    for adapter in adapters:
        if not (adapter / 'adapter_model.safetensors').is_file():
            raise FileNotFoundError(adapter / 'adapter_model.safetensors')
    test = data.load_csv(cfg, 'test')
    base, processor = modeling.load_model(cfg)
    model, names = infer.load_fold_adapters(base, [str(p) for p in adapters])
    predictor = infer.Predictor(model, processor, cfg, {})
    model.set_adapter(names[0])
    fast = infer.run_fast_pass(predictor, test, args.out_dir / 'test_fast.csv')
    hard = test[test.id.isin(fast.loc[fast.margin <= cfg.margin_thr, 'id'])]
    records = []

    def collect(pred, source, views):
        for _, row in hard.iterrows():
            original = data.load_image(cfg, row.path)
            for view in views:
                image = (original if view == 'orig' else
                         modeling.fit_pixels(original, 1280**2, 10**9))
                probs = pred.probs(image, row, data.PERMS[:4], ('orig',)).numpy()
                records.append({'id': row.id, 'source': source, 'view': view,
                                **dict(zip(infer.PROB_COLS, probs))})
        pd.DataFrame(records).to_csv(args.out_dir / 'test_components.csv', index=False)

    for name, adapter in zip(names, adapters):
        model.set_adapter(name)
        collect(predictor, adapter.name, ['orig', 'up1280'])
    del predictor, model, base, processor
    gc.collect()
    torch.cuda.empty_cache()
    intern_cfg = dataclasses.replace(cfg, model_id='OpenGVLab/InternVL3_5-8B-HF')
    model, processor = modeling.load_model(intern_cfg)
    collect(infer.Predictor(model, processor, intern_cfg, {}), 'zs_internvl3_5', ['orig'])
    print(f'Completed baseline: {len(test)} total, {len(hard)} hard items')


if __name__ == '__main__':
    main()
