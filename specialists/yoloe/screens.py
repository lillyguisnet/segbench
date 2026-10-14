"""YOLOE-26x on the screen-click questions (segbench/screens.py, both rounds).

uv run screens.py --name screens-local-1-yoloe

The instruction, as the language models get it, is YOLOE's one text class.
Same settings as on the photos (bench.py): image size 640, confidence 0.25,
fp16, full-resolution masks. One click per mask, at the point deepest inside
it. Deterministic: one answer, scored for every try. Time = median of 3 warm
runs, opening the screenshot included; cost = $0.22/GPU-hour x model seconds
x the 2026-10-08 load factor (measured on photos). Writes
results/run-<name>.jsonl in scripts/run_screens.py's format.
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
os.chdir(ROOT / "weights" / "yoloe")

import numpy as np  # noqa: E402
from ultralytics import YOLOE  # noqa: E402

sys.path.insert(0, str(ROOT))
from segbench.screens import PROMPT_VERSION, ROUNDS  # noqa: E402

CONFIG = {"weights": "yoloe-26x-seg.pt", "imgsz": 640, "conf": 0.25, "precision": "fp16", "retina_masks": True}


def main(name: str) -> None:
    out = ROOT / "results" / f"run-{name}.jsonl"
    if out.exists():
        sys.exit(f"{out} exists: choose a new --name")
    model = YOLOE("yoloe-26x-seg.pt")
    factor = A.load_factor("yoloe-26x-text")

    def ask(phrase):
        model.set_classes([phrase])

        def fn(im):
            r = model.predict(im, half=True, retina_masks=True, imgsz=640, conf=0.25, verbose=False)[0]
            return [] if r.masks is None else list(r.masks.data.cpu().numpy() > 0.5)
        return fn

    tasks = [t for r in sorted(ROUNDS) for t in ROUNDS[r]]
    with out.open("a") as f:
        for i, t in enumerate(tasks):
            path = ROOT / "images" / "screenshots" / t.image
            data = path.read_bytes()
            if i == 0:
                A.timed(ask(t.instruction), data, repeats=3)  # warm-up
            masks, walls, models = A.timed(ask(t.instruction), data)
            W, H = A.decode(data).size
            clicks = []
            for m in masks:
                p = A.deepest_point(m)
                if p is not None:
                    clicks.append({"x": round(p[0] * 1000 / W, 2), "y": round(p[1] * 1000 / H, 2), "form": "model"})
            model_s = statistics.median(models)
            f.write(json.dumps({
                "model": "yoloe-26x", "level": None, "track": "screen", "task": t.key, "repeat": 0,
                "prompt_version": PROMPT_VERSION + "-local", "prompt": t.instruction,
                "image": "screenshots/" + t.image, "image_sha256": hashlib.sha256(data).hexdigest(), "image_size": [W, H],
                "provider": "local-gpu", "route": "local",
                "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "seconds": round(statistics.median(walls), 4), "model_seconds": round(model_s, 4),
                "cost_usd": 0.22 / 3600 * model_s * factor,
                "cost_method": f"$0.22/GPU-hour x model seconds x load factor {factor:.3f} (2026-10-08 throughput test, photos)",
                "readable": True, "read_as": "model", "clicks": clicks, "point_from": ["mask_deepest_point"],
                "deterministic": True, "config": CONFIG, "gpu_others": A.other_programs(),
            }) + "\n")
            print(f"yoloe {t.key:17} {len(clicks):3} clicks  {np.median(walls):.2f}s", flush=True)
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    main(ap.parse_args().name)
