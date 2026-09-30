"""Combine the baseline cascade and question-selected Qwen3.5 probabilities."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from question_routes import route_question

PROB_COLS = ['p_a', 'p_b', 'p_c', 'p_d']
COMPONENTS = {
    ('holdout_best', 'orig'), ('holdout_best', 'up1280'),
    ('full_train', 'orig'), ('full_train', 'up1280'),
    ('zs_internvl3_5', 'orig'),
}
Q35_WEIGHT = 0.75
BASELINE_TEMPERATURE = 1.2804923323695767
Q35_TEMPERATURE = 1.3320988046333513


def check_probabilities(frame: pd.DataFrame) -> None:
    p = frame[PROB_COLS].to_numpy(dtype=float)
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('Invalid probabilities')
    if not np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-5):
        raise ValueError('Probabilities do not sum to one')


def baseline_cascade(fast: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    if not fast.id.is_unique:
        raise ValueError('Duplicate fast-pass IDs')
    check_probabilities(fast)
    out = fast[['id', *PROB_COLS]].copy().set_index('id')
    hard = set(fast.loc[fast.margin <= 0.8, 'id'])
    selected = components[components.apply(
        lambda row: (row['source'], row['view']) in COMPONENTS and row['id'] in hard, axis=1)].copy()
    if selected.duplicated(['id', 'source', 'view']).any():
        raise ValueError('Duplicate cascade components')
    if set(selected.id) != hard or not selected.groupby('id').size().eq(5).all():
        raise ValueError('Every hard item requires all five components')
    check_probabilities(selected)
    means = selected.groupby('id')[PROB_COLS].mean()
    out.loc[means.index, PROB_COLS] = means.to_numpy()
    return out.reset_index()


def temperature(p: np.ndarray, value: float) -> np.ndarray:
    logits = np.log(np.maximum(p, 1e-12)) / value
    maximum = logits.max(axis=1, keepdims=True)
    normalizer = maximum + np.log(np.exp(logits - maximum).sum(axis=1, keepdims=True))
    return np.exp(logits - normalizer)


def combine(baseline: pd.DataFrame, q35: pd.DataFrame, test: pd.DataFrame,
            sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    for name, frame in [('baseline', baseline), ('qwen35', q35), ('test', test), ('sample', sample)]:
        if not frame.id.is_unique:
            raise ValueError(f'Duplicate IDs: {name}')
    ids = sample.id.tolist()
    if set(ids) != set(test.id) or set(ids) != set(baseline.id):
        raise ValueError('Test, sample and baseline IDs differ')
    test = test.set_index('id').loc[ids]
    gate = test.question.map(lambda q: route_question(q) != 'unchanged').to_numpy()
    selected_ids = test.index[gate].tolist()
    if set(q35.id) != set(selected_ids):
        raise ValueError('Qwen3.5 predictions must match the question-selected IDs')
    check_probabilities(baseline)
    check_probabilities(q35)
    base = baseline.set_index('id').loc[ids, PROB_COLS].to_numpy(dtype=float)
    extra = q35.set_index('id').loc[selected_ids, PROB_COLS].to_numpy(dtype=float)
    merged = base.copy()
    merged[gate] = ((1 - Q35_WEIGHT) * temperature(base[gate], BASELINE_TEMPERATURE)
                    + Q35_WEIGHT * temperature(extra, Q35_TEMPERATURE))
    if not np.array_equal(merged[~gate], base[~gate]):
        raise AssertionError('Probabilities outside the selected IDs changed')
    answers = np.array(list('abcd'))[merged.argmax(axis=1)]
    submission = pd.DataFrame({'id': ids, 'answer': answers})
    probabilities = pd.DataFrame(merged, columns=PROB_COLS)
    probabilities.insert(0, 'id', ids)
    return submission, probabilities


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fast', type=Path, required=True)
    parser.add_argument('--components', type=Path, required=True)
    parser.add_argument('--qwen35', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--sample', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    read = lambda p: pd.read_csv(p, dtype={'id': str})
    baseline = baseline_cascade(read(args.fast), read(args.components))
    sub, _ = combine(baseline, read(args.qwen35), read(args.test), read(args.sample))
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False)
    print(f'Saved {len(sub)} answers to {args.out}')


if __name__ == '__main__':
    main()
