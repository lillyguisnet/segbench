"""RF-DETR Seg 2XL on the benchmark photos, where a task is one of its 80 COCO categories.

uv run bench.py --name specialists-1

It can only find what COCO names. Entered: cows (find, whole), and the
trick questions whose absent thing is a COCO category (cow, car, person,
cat; not tractor). Not entered: logs, fig leaves, red foliage, the road,
and dirty / clean dishes (COCO has "bowl" and "cup" but no plate and no
dirty or clean). Threshold 0.5, as in the smoke test; fp16, traced.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import harness  # noqa: E402  (pins the GPU before torch loads)
import answers as A  # noqa: E402

WEIGHT_DIR = harness.ROOT / "weights" / "rfdetr"
os.chdir(WEIGHT_DIR)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rfdetr import RFDETRSeg2XLarge  # noqa: E402
from rfdetr.assets.coco_classes import COCO_CLASSES  # noqa: E402

from segbench.tasks import BY_KEY  # noqa: E402

COCO = set(COCO_CLASSES.values()) if isinstance(COCO_CLASSES, dict) else set(COCO_CLASSES)
CATEGORY = {"cows": "cow"}  # task -> COCO category
CONFIG = {"weights": "RFDETRSeg2XLarge (COCO)", "threshold": 0.5, "precision": "fp16, traced"}


def main(name: str) -> None:
    writer = A.Writer(name)
    model = RFDETRSeg2XLarge()
    model.inference(compile=True, batch_size=1, dtype=torch.float16)
    factor = A.load_factor("rfdetr-seg-2xl")

    def ask(category):
        def fn(im):
            d = model.predict(im, threshold=0.5)
            d = d[0] if isinstance(d, list) else d
            if d.mask is None:
                return [], []
            keep = [i for i, c in enumerate(d.class_id) if COCO_CLASSES[int(c)] == category]
            return [np.asarray(d.mask[i]) for i in keep], [float(d.confidence[i]) for i in keep]
        return fn

    jobs = [(BY_KEY[t], "task", c) for t, c in CATEGORY.items()]
    jobs += [(task, "trick", A.TRICK_PHRASES[task.key]) for task in BY_KEY.values()
             if A.TRICK_PHRASES[task.key] in COCO]
    data0, _ = A.photo(jobs[0][0].photo)
    A.timed(ask("cow"), data0, repeats=3)  # warm-up
    for task, kind, category in jobs:
        data, sha = A.photo(task.photo)
        (masks, scores), walls, models = A.timed(ask(category), data)
        ans = A.to_answers(task, masks, [None] * len(masks), scores)
        tracks = ("find", "whole") if kind == "task" else ("trick",)
        for track in tracks:
            text = ans["whole"] if track == "whole" else ans["find"]
            writer.write(A.record(model="rfdetr-seg-2xl", track=track, task=task, text=text, walls=walls,
                                  models=models, factor=factor, prompt=f"COCO category: {category}", sha=sha,
                                  extra={"config": CONFIG}), ans["masks"] if track != "find" else None)
        print(f"rfdetr {task.key:7} {kind:5} '{category}': {len(masks)}  {np.median(walls):.2f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    main(ap.parse_args().name)
