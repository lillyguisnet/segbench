"""PixMo-Points-Eval scoring, separate from segbench's published dot scores.

Matches Ai2 molmo2 olmo/eval/evaluators.py PointingEval at
f3cb1085fbb97c4a4d7fdcadd77cf871bf37a88a: minimum Euclidean-distance
assignment to reference dots, then membership in the assigned object's mask.
This is NOT maximum mask-hit matching. Scores are means of per-question
precision, recall and F1, not a pooled F1 or F1 of the mean P/R.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

DATASET = "allenai/pixmo-points-eval"
DATA_REVISION = "e237f0491ca49bf8dec04e06fc0c3807a37ef157"
EVAL_REVISION = "f3cb1085fbb97c4a4d7fdcadd77cf871bf37a88a"
METRIC = "pixmo-pointing-eval-distance-assignment-mask-hit-v1"


def score_points(points, reference_percent, masks) -> dict:
    """Score predictions in the mask's pixel grid; reference points are 0..100.

    Keep duplicates and off-image points as predictions. Never clip or repair
    them. Malformed/nonfinite coordinates must be handled before this function.
    """
    masks = np.asarray(masks, dtype=bool)
    n_ref = len(reference_percent)
    if n_ref and len(masks) != n_ref:
        raise ValueError("one mask is required per reference point")
    pred = np.asarray(points, dtype=float).reshape(-1, 2)
    if not np.isfinite(pred).all():
        raise ValueError("nonfinite prediction")
    n_pred = len(pred)
    if n_ref == 0:
        p = r = f1 = float(n_pred == 0)
        hits = 0
    elif n_pred == 0:
        p = r = f1 = 0.0
        hits = 0
    else:
        if masks.ndim != 3:
            raise ValueError("masks must have shape [objects, height, width]")
        h, w = masks.shape[1:]
        refs = np.asarray(reference_percent, dtype=float) * [w / 100, h / 100]
        rows, cols = linear_sum_assignment(cdist(pred, refs))
        hits = 0
        for i, j in zip(rows, cols):
            x, y = (round(float(v)) for v in pred[i])
            hits += int(0 <= x < w and 0 <= y < h and masks[j, y, x])
        p, r = hits / n_pred, hits / n_ref
        f1 = 0.0 if not p or not r else 2 * p * r / (p + r + 1e-10)
    return {"precision": p, "recall": r, "f1": f1, "hits": hits,
            "predicted": n_pred, "reference": n_ref}
