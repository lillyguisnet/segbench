"""Smoke test + latency + throughput for RF-DETR Segmentation (2XL, Apache 2.0).

uv run run.py

Pretrained on COCO's 80 categories: it finds "dog", "truck", "handbag", not
"wheel" or "paper bag". Weights download to weights/rfdetr/ on first use.

Optimized as the authors recommend for serving: model.inference() traces the
network for a fixed batch size in fp16 (redone for each batch size, untimed).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import harness  # noqa: E402

WEIGHT_DIR = harness.ROOT / "weights" / "rfdetr"
WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(WEIGHT_DIR)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rfdetr import RFDETRSeg2XLarge  # noqa: E402
from rfdetr.assets.coco_classes import COCO_CLASSES  # noqa: E402  (COCO category id -> name)

BATCHES = (1, 2, 4, 8, 16, 32)


def prepare(model, bs):
    model.remove_optimized_model()
    model.inference(compile=True, batch_size=bs, dtype=torch.float16)
    return model


def predict(model, ims):
    dets = model.predict(ims, threshold=0.5)
    dets = dets if isinstance(dets, list) else [dets]
    out = []
    for d in dets:
        masks = d.mask if d.mask is not None else np.zeros((0,))
        out.append({"masks": np.asarray(masks), "labels": [COCO_CLASSES[int(c)] for c in d.class_id]})
    return out


def summarize(r):
    counts = {}
    for n in r["labels"]:
        counts[n] = counts.get(n, 0) + 1
    return {"masks": int(len(r["masks"])), "labels": counts}


if __name__ == "__main__":
    harness.run(
        model_id="rfdetr-seg-2xl",
        config={"weights": "RFDETRSeg2XLarge (COCO, Apache 2.0)", "precision": "fp16, traced (model.inference)",
                "threshold": 0.5},
        load=RFDETRSeg2XLarge, prepare=prepare, predict=predict, summarize=summarize,
        batch_sizes=BATCHES,
        notes="Fixed COCO categories. Batch size is fixed at trace time, so each batch size is re-traced (untimed).",
    )
