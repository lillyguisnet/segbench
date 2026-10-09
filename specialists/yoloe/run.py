"""Smoke test + latency + throughput for YOLOE-26 (largest size, x).

uv run run.py MODE   (text | prompt-free | all)

Weights and the MobileCLIP text encoder download into weights/yoloe/ on
first use (Ultralytics saves them in the working directory).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import harness  # noqa: E402

WEIGHT_DIR = harness.ROOT / "weights" / "yoloe"
WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(WEIGHT_DIR)

import numpy as np  # noqa: E402
from ultralytics import YOLOE  # noqa: E402

BATCHES = (1, 2, 4, 8, 16, 32, 64)
SIZE = "x"


def predict(model, ims):
    results = model.predict(ims, half=True, retina_masks=True, batch=len(ims), verbose=False)
    out = []
    for r in results:
        masks = r.masks.data.cpu().numpy() if r.masks is not None else np.zeros((0,))
        names = [r.names[int(c)] for c in r.boxes.cls.tolist()] if r.boxes is not None else []
        out.append({"masks": masks, "labels": names})
    return out


def summarize(r):
    counts = {}
    for n in r["labels"]:
        counts[n] = counts.get(n, 0) + 1
    return {"masks": int(len(r["masks"])), "labels": dict(sorted(counts.items(), key=lambda kv: -kv[1])[:6])}


def text():
    vocab = sorted(set(harness.SMOKE_PROMPTS.values()))

    def load():
        model = YOLOE(f"yoloe-26{SIZE}-seg.pt")
        model.set_classes(vocab)
        return model

    harness.run(
        model_id=f"yoloe-26{SIZE}-text",
        config={"weights": f"yoloe-26{SIZE}-seg.pt (ultralytics assets v8.4.0)", "precision": "fp16",
                "classes": vocab, "imgsz": 640, "retina_masks": True},
        load=load, predict=predict, summarize=summarize, batch_sizes=BATCHES,
        notes="One class list for all smoke images; the benchmark will set classes per task.",
    )


def prompt_free():
    harness.run(
        model_id=f"yoloe-26{SIZE}-prompt-free",
        config={"weights": f"yoloe-26{SIZE}-seg-pf.pt", "precision": "fp16", "imgsz": 640,
                "retina_masks": True, "vocabulary": "built-in 4,585 names"},
        load=lambda: YOLOE(f"yoloe-26{SIZE}-seg-pf.pt"),
        predict=predict, summarize=summarize, batch_sizes=BATCHES,
    )


MODES = {"text": text, "prompt-free": prompt_free}

if __name__ == "__main__":
    mode = sys.argv[1]
    for name in (MODES if mode == "all" else [mode]):
        print(f"== {name}", flush=True)
        MODES[name]()
