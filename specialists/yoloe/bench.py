"""YOLOE-26x (text prompts) on the six benchmark photos: find, whole and trick tracks.

uv run bench.py --name specialists-1

Same phrases as SAM 3 (common/answers.py). Run as Ultralytics ships it:
image size 640, confidence 0.25, fp16, full-resolution masks. With two
labels (dishes), one mask per object (class-agnostic suppression) so a dish
is not both dirty and clean.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import harness  # noqa: E402  (pins the GPU before torch loads)
import answers as A  # noqa: E402

WEIGHT_DIR = harness.ROOT / "weights" / "yoloe"
os.chdir(WEIGHT_DIR)

import numpy as np  # noqa: E402
from ultralytics import YOLOE  # noqa: E402

from segbench.tasks import TASKS  # noqa: E402

CONFIG = {"weights": "yoloe-26x-seg.pt", "imgsz": 640, "conf": 0.25, "precision": "fp16", "retina_masks": True}


def main(name: str, tasks=TASKS) -> None:
    writer = A.Writer(name)
    model = YOLOE("yoloe-26x-seg.pt")
    factor = A.load_factor("yoloe-26x-text")

    def ask(phrases):
        model.set_classes(phrases)

        def fn(im):
            r = model.predict(im, half=True, retina_masks=True, imgsz=640, conf=0.25, agnostic_nms=len(phrases) > 1,
                              verbose=False)[0]
            if r.masks is None:
                return [], [], []
            masks = list(r.masks.data.cpu().numpy() > 0.5)
            labels = [r.names[int(c)] for c in r.boxes.cls.tolist()]
            return masks, labels, r.boxes.conf.cpu().numpy().tolist()
        return fn

    data0, _ = A.photo(TASKS[0].photo)
    A.timed(ask(["cow"]), data0, repeats=3)  # warm-up
    for task in tasks:
        data, sha = A.photo(task.photo)
        (masks, phrases, scores), walls, models = A.timed(ask(A.PHRASES[task.key]), data)
        labels = [A.LABEL_OF_PHRASE.get(p) for p in phrases]
        ans = A.to_answers(task, masks, labels, scores)
        for track in ("find", "whole"):
            if track == "find" and task.kind == "region":
                continue
            writer.write(A.record(model="yoloe-26x", track=track, task=task, text=ans[track], walls=walls,
                                  models=models, factor=factor, prompt=" | ".join(A.PHRASES[task.key]), sha=sha,
                                  extra={"config": CONFIG}), ans["masks"] if track == "whole" else None)
        print(f"yoloe {task.key:7} {len(masks):3} masks  {np.median(walls):.2f}s", flush=True)
        phrase = A.TRICK_PHRASES[task.key]
        (masks, _, scores), walls, models = A.timed(ask([phrase]), data)
        ans = A.to_answers(task, masks, [None] * len(masks), scores)
        writer.write(A.record(model="yoloe-26x", track="trick", task=task, text=ans["find"], walls=walls,
                              models=models, factor=factor, prompt=phrase, sha=sha, extra={"config": CONFIG}),
                     ans["masks"])
        print(f"yoloe {task.key:7} trick '{phrase}': {len(masks)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--batch", type=int, choices=sorted(A.BATCHES), default=1, help="photo batch (2: added 2026-10-14)")
    args = ap.parse_args()
    main(args.name, A.BATCHES[args.batch])
