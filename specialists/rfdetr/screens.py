"""RF-DETR Seg 2XL on the screen-click questions (segbench/screens.py, both rounds).

uv run screens.py --name screens-local-1-rfdetr

Same rule as on the photos (points.py; Maxime, 2026-10-10: failing where its
fixed vocabulary has no word for the thing is part of the result). RF-DETR
only knows COCO's 80 categories; none of them is a button, checkbox, folder,
news story, group header or label, so no category fits any question (fixed
before running): it is run and timed on each screenshot, and gives no click.
Same settings as on the photos (threshold 0.5, fp16, traced). Time = median
of 3 warm runs, opening the screenshot included; cost = $0.22/GPU-hour x
model seconds x its 2026-10-08 load factor (measured on photos).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import harness  # noqa: E402  (pins the GPU before torch loads)
import answers as A  # noqa: E402

ROOT = harness.ROOT
os.chdir(ROOT / "weights" / "rfdetr")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rfdetr import RFDETRSeg2XLarge  # noqa: E402
from rfdetr.assets.coco_classes import COCO_CLASSES  # noqa: E402

sys.path.insert(0, str(ROOT))
from segbench.screens import PROMPT_VERSION, ROUNDS  # noqa: E402

CATEGORIES: list[str] = []  # none of the 80 fits any screen question
CONFIG = {"weights": "RFDETRSeg2XLarge (COCO)", "threshold": 0.5, "precision": "fp16, traced"}


def main(name: str) -> None:
    out = ROOT / "results" / f"run-{name}.jsonl"
    if out.exists():
        sys.exit(f"{out} exists: choose a new --name")
    model = RFDETRSeg2XLarge()
    model.inference(compile=True, batch_size=1, dtype=torch.float16)
    factor = A.load_factor("rfdetr-seg-2xl")

    def fn(im):
        d = model.predict(im, threshold=0.5)
        d = d[0] if isinstance(d, list) else d
        if d.mask is None:
            return []
        return [np.asarray(d.mask[i]) for i, c in enumerate(d.class_id) if COCO_CLASSES[int(c)] in CATEGORIES]

    tasks = [t for r in sorted(ROUNDS) for t in ROUNDS[r]]
    with out.open("a") as f:
        for i, t in enumerate(tasks):
            data = (ROOT / "images" / "screenshots" / t.image).read_bytes()
            if i == 0:
                A.timed(fn, data, repeats=3)  # warm-up
            masks, walls, models = A.timed(fn, data)
            W, H = A.decode(data).size
            clicks = [{"x": round(p[0] * 1000 / W, 2), "y": round(p[1] * 1000 / H, 2), "form": "model"}
                      for p in (A.deepest_point(m) for m in masks) if p is not None]
            model_s = statistics.median(models)
            f.write(json.dumps({
                "model": "rfdetr-seg-2xl", "level": None, "track": "screen", "task": t.key, "repeat": 0,
                "prompt_version": PROMPT_VERSION + "-local", "categories": CATEGORIES,
                "prompt": "none of its 80 COCO categories fit: no clicks",
                "image": "screenshots/" + t.image, "image_sha256": hashlib.sha256(data).hexdigest(), "image_size": [W, H],
                "provider": "local-gpu", "route": "local",
                "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "seconds": round(statistics.median(walls), 4), "model_seconds": round(model_s, 4),
                "cost_usd": 0.22 / 3600 * model_s * factor,
                "cost_method": f"$0.22/GPU-hour x model seconds x load factor {factor:.3f} (2026-10-08 throughput test, photos)",
                "readable": True, "read_as": "model", "clicks": clicks, "point_from": ["mask_deepest_point"],
                "deterministic": True, "config": CONFIG, "gpu_others": A.other_programs(),
            }) + "\n")
            print(f"rfdetr {t.key:17} {len(clicks):3} clicks  {np.median(walls):.3f}s", flush=True)
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    main(ap.parse_args().name)
