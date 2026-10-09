"""Smoke test + latency + throughput for gen2seg (ICLR 2026), SD and MAE-H variants.

uv run run.py MODE   (sd-fp32 | sd-fp16 | mae | all)

gen2seg does not return masks directly: it paints each object a distinct
colour (an "instance colouring" image). Turning colours into separate masks
(clustering, or a click per object) is a pipeline step still to write, so
these timings cover the model only. The smoke summary counts colour groups
covering at least 0.5 % of the image, a rough stand-in for "objects found".
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import harness  # noqa: E402

sys.path.insert(0, str(HERE / "vendor"))  # github.com/UCDvision/gen2seg @ a8d3ffa6

import numpy as np  # noqa: E402
import torch  # noqa: E402

BATCHES = (1, 2, 4, 8, 16)


def colour_groups(feature: np.ndarray) -> dict:
    q = (np.asarray(feature).reshape(-1, 3).astype(np.int32) // 32)
    keys = q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2]
    counts = np.bincount(keys, minlength=512) / len(keys)
    return {"colour_groups_over_0.5pct": int((counts >= 0.005).sum())}


def sd(dtype: torch.dtype):
    from gen2seg_sd_pipeline import gen2segSDPipeline

    def load():
        return gen2segSDPipeline.from_pretrained(
            "reachomk/gen2seg-sd", use_safetensors=True, torch_dtype=dtype
        ).to("cuda")

    def predict(pipe, ims):
        # The pipeline needs equal-sized images per call: group by size, keep order.
        out: list = [None] * len(ims)
        groups: dict = {}
        for i, im in enumerate(ims):
            groups.setdefault(im.size, []).append(i)
        for idx in groups.values():
            with torch.inference_mode():
                pred = np.asarray(pipe([ims[i] for i in idx], batch_size=len(idx)).prediction)
            pred = pred.reshape(len(idx), *pred.shape[-3:])
            for j, i in enumerate(idx):
                out[i] = pred[j]
        return out

    name = "fp16" if dtype == torch.float16 else "fp32"
    harness.run(
        model_id=f"gen2seg-sd-{name}",
        config={"weights": "reachomk/gen2seg-sd", "precision": name, "processing_resolution": 768},
        load=load, predict=predict, summarize=colour_groups, batch_sizes=BATCHES,
        notes="Model only: instance colouring image; colour-to-mask step not yet included.",
    )


def mae():
    from gen2seg_mae_pipeline import gen2segMAEInstancePipeline
    from transformers import AutoImageProcessor

    def load():
        processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-huge")
        return gen2segMAEInstancePipeline(model="reachomk/gen2seg-mae-h", image_processor=processor).to("cuda")

    def predict(pipe, ims):
        small = [im.resize((224, 224)) for im in ims]  # as in the authors' inference_mae.py
        with torch.inference_mode():
            pred = pipe(small).prediction
        return [np.asarray(p).squeeze() for p in pred]

    harness.run(
        model_id="gen2seg-mae-h",
        config={"weights": "reachomk/gen2seg-mae-h", "precision": "fp32", "input": "224×224 (authors' setting)"},
        load=load, predict=predict, summarize=colour_groups, batch_sizes=BATCHES + (32, 64),
        notes="Model only; 224×224 input is the authors' setting, so fine detail is limited.",
    )


MODES = {"sd-fp32": lambda: sd(torch.float32), "sd-fp16": lambda: sd(torch.float16), "mae": mae}

if __name__ == "__main__":
    mode = sys.argv[1]
    for name in (MODES if mode == "all" else [mode]):
        print(f"== {name}", flush=True)
        MODES[name]()
