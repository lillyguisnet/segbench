"""RF-DETR Seg 2XL on the four point tasks, with whatever of its 80 COCO categories fit each request.

uv run points.py --name rfdetr-3      ->  results/run-rfdetr-3.jsonl

Maxime, 2026-10-10: enter RF-DETR on every task; failing where its fixed
vocabulary has no word for the thing is part of the result. Categories,
fixed before running:
  logs    none (no COCO category for a log or a log end)  -> no dots
  cows    cow
  fig     none ("potted plant" is a whole plant, not a leaf) -> no dots
  dishes  bowl, cup (the COCO categories inside our dish definition; COCO
          has no plate, pot, pan, lid or cutting board; dishes are scored
          on location alone, so its missing dirty/clean label costs nothing)
Same settings as specialists/rfdetr/bench.py (threshold 0.5, fp16, traced);
one dot per mask at its deepest point, like SAM 3 and YOLOE.

Timing: answers are the same on a busy or a free GPU, so they are computed
now; seconds and cost come from the clean run on the free GPU 0
(results/run-specialists-2.jsonl, 2026-10-09: the same model on the same
photo; RF-DETR resizes every photo to one input size, and categories are
only filtered afterwards). Each record names its source in timing_source.
Today's own seconds (GPU shared) are kept as seconds_this_run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import harness  # noqa: E402  (pins the GPU before torch loads)
import answers as A  # noqa: E402

os.chdir(harness.ROOT / "weights" / "rfdetr")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rfdetr import RFDETRSeg2XLarge  # noqa: E402
from rfdetr.assets.coco_classes import COCO_CLASSES  # noqa: E402

from segbench.tasks import BY_KEY  # noqa: E402

CATEGORIES = {"logs": [], "cows": ["cow"], "fig": [], "dishes": ["bowl", "cup"]}
CONFIG = {"weights": "RFDETRSeg2XLarge (COCO)", "threshold": 0.5, "precision": "fp16, traced"}
CLEAN = harness.ROOT / "results" / "run-specialists-2.jsonl"


def clean_timing() -> dict:
    """photo -> (line number, record) of RF-DETR's clean run on that photo."""
    out = {}
    for i, line in enumerate(CLEAN.read_text().splitlines(), 1):
        r = json.loads(line)
        if r["model"] == "rfdetr-seg-2xl" and r["image"] not in out:
            out[r["image"]] = (i, r)
    return out


def main(name: str) -> None:
    out = harness.ROOT / "results" / f"run-{name}.jsonl"
    if out.exists():
        sys.exit(f"{out} exists: choose a new --name")
    clean = clean_timing()
    model = RFDETRSeg2XLarge()
    model.inference(compile=True, batch_size=1, dtype=torch.float16)

    def ask(categories):
        def fn(im):
            d = model.predict(im, threshold=0.5)
            d = d[0] if isinstance(d, list) else d
            if d.mask is None:
                return [], []
            keep = [i for i, c in enumerate(d.class_id) if COCO_CLASSES[int(c)] in categories]
            return [np.asarray(d.mask[i]) for i in keep], [float(d.confidence[i]) for i in keep]
        return fn

    data0, _ = A.photo(BY_KEY["cows"].photo)
    A.timed(ask(["cow"]), data0, repeats=3)  # warm-up
    with out.open("w") as f:
        for key, cats in CATEGORIES.items():
            task = BY_KEY[key]
            data, sha = A.photo(task.photo)
            (masks, scores), walls, models = A.timed(ask(cats), data)
            ans = A.to_answers(task, masks, [None] * len(masks), scores)
            line, src = clean[task.photo]
            prompt = ("COCO categories: " + ", ".join(cats)) if cats else "none of its 80 COCO categories fit: no dots"
            rec = A.record(model="rfdetr-seg-2xl", track="find", task=task, text=ans["find"], walls=walls,
                           models=models, factor=1.0, prompt=prompt, sha=sha,
                           extra={"config": CONFIG, "categories": cats})
            rec.update({
                "seconds_this_run": rec["seconds"], "gpu_others_this_run": rec["gpu_others"],
                "seconds": src["seconds"], "model_seconds": src["model_seconds"], "cost_usd": src["cost_usd"],
                "cost_method": src["cost_method"],
                "timing_source": {"record": f"results/run-specialists-2.jsonl:{line}",
                                  "started_at": src["started_at"], "gpu_others": src["gpu_others"]},
                "timing_trustworthy": True, "gpu_others": src["gpu_others"],
            })
            f.write(json.dumps(rec) + "\n")
            print(f"rfdetr {key:7} {prompt:45} {len(masks):3} dots  (clean {src['seconds']:.3f}s, "
                  f"today {rec['seconds_this_run']:.3f}s)", flush=True)
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    main(ap.parse_args().name)
