"""Specialist answers on the benchmark photos, in the same records as the language models.

A specialist answers with masks. This module turns them into what the
tracks ask for, so the viewer, the consensus and the scorers read one
format for everyone:

- whole: one outline (polygon, coordinates 0..1000) per mask: its largest
  piece; regions (tree, road): every piece of the union of the masks;
- find: one dot per mask, at the point deepest inside it (README: the centre
  of a curved shape can fall outside it);
- the full-resolution masks themselves, run-length encoded (COCO's
  uncompressed RLE, column-major), in a side file: scoring uses those, the
  polygons are for display and voting.

Cost of one call = GPU rental price x this call's model seconds x the
model's load factor, where the load factor (cost per image under load /
cost per image at batch 1) comes from the throughput test on 2026-10-08
(results/specialists/). Speed (bubble size) = this call's wall-clock
seconds, picture decoding included.

Needs numpy, opencv and pillow (in every specialist environment).
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from segbench.tasks import TASKS  # noqa: E402  (no third-party imports there)

import harness  # noqa: E402  (same folder; already imported by the caller, pins the GPU)

GRID = 1000  # longest side of the grid polygons and dots are computed on
MIN_PIECE = 0.00002  # pieces smaller than this share of the photo are dropped from outlines
PROMPT_VERSION = "spec-v1"  # 2026-10-09

# One short phrase per task, written once (2026-10-09) before seeing any
# specialist answer, and never tuned on these photos: tuning on the test
# photos would turn the benchmark into a training set. Open-vocabulary
# models take a noun phrase, not a sentence, and cannot be told "not the
# spider plant"; that is part of what is measured.
PHRASES = {
    "logs": ["log end"],
    "cows": ["cow"],
    "fig": ["fig leaf"],
    "tree": ["red leaves"],
    "road": ["gravel road"],
    "dishes": ["dirty dish", "clean dish"],  # one prompt per label
}
LABEL_OF_PHRASE = {"dirty dish": "dirty", "clean dish": "clean"}
TRICK_PHRASES = {t.key: t.absent.removeprefix("each ") for t in TASKS}


def load_factor(model_id: str) -> float:
    rows = [json.loads(line) for line in (ROOT / "results/specialists/2026-10-08.jsonl").read_text().splitlines()]
    r = [x for x in rows if x.get("model_id") == model_id and "cost" in x and "usd_per_image_at_batch_1" in x["cost"]][-1]
    return r["cost"]["usd_per_image_under_load"] / r["cost"]["usd_per_image_at_batch_1"]


def photo(name: str) -> tuple[bytes, str]:
    data = (ROOT / "images" / name).read_bytes()
    return data, hashlib.sha256(data).hexdigest()


def decode(data: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(data))
    if im.getexif().get(0x0112) not in (None, 1):
        raise SystemExit("photo with an EXIF rotation tag: see scripts/prepare_annotation_set.py")
    return im.convert("RGB")


# ---------- masks -> shared format ----------


def rle(mask: np.ndarray) -> dict:
    flat = np.asarray(mask, dtype=bool).ravel(order="F")
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    bounds = np.concatenate([[0], change, [flat.size]])
    counts = np.diff(bounds).tolist()
    if flat[0]:
        counts = [0] + counts
    return {"size": list(mask.shape), "counts": counts}


def _small(mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    h, w = mask.shape
    s = GRID / max(h, w)
    small = cv2.resize(mask.astype(np.uint8) * 255, (max(1, round(w * s)), max(1, round(h * s))),
                       interpolation=cv2.INTER_AREA) >= 128
    return small, 1000 / small.shape[1], 1000 / small.shape[0]


def _polygons(small: np.ndarray, sx: float, sy: float) -> list[list[list[int]]]:
    contours, _ = cv2.findContours(small.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) < MIN_PIECE * small.size and out:
            continue
        c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
        if len(c) >= 3:
            out.append([[round(float(x) * sx), round(float(y) * sy)] for x, y in c])
    return out


def deepest_point(mask: np.ndarray) -> list[int] | None:
    small, sx, sy = _small(mask)
    if not small.any():
        return None
    d = cv2.distanceTransform(np.pad(small, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]
    y, x = np.unravel_index(int(d.argmax()), d.shape)
    return [round((x + 0.5) * sx), round((y + 0.5) * sy)]


def to_answers(task, masks: list[np.ndarray], labels: list[str | None], scores: list[float]) -> dict:
    """{"find": ..., "whole": ...} answer texts (the JSON a language model would reply) plus the masks."""
    keep = [i for i, m in enumerate(masks) if m.any()]
    masks, labels, scores = [masks[i] for i in keep], [labels[i] for i in keep], [scores[i] for i in keep]
    find, whole = [], []
    for m, label in zip(masks, labels):
        dot = deepest_point(m)
        polys = _polygons(*_small(m))
        if dot is None or not polys:
            continue
        find.append({"point": dot, **({"label": label} if label else {})})
        whole.append({"polygon": polys[0], **({"label": label} if label else {})})
    if task.kind == "region":
        union = np.zeros_like(masks[0], dtype=bool) if masks else None
        for m in masks:
            union |= m
        regions = [{"polygon": p} for p in _polygons(*_small(union))] if union is not None else []
        whole_text = json.dumps({"regions": regions})
    else:
        whole_text = json.dumps({"objects": whole})
    return {"find": json.dumps({"objects": find}), "whole": whole_text,
            "masks": [{"rle": rle(m), "label": lb, "score": round(float(s), 4)} for m, lb, s in zip(masks, labels, scores)]}


def dedupe(masks, labels, scores, iou: float = 0.5):
    """Keep the higher-scoring of two masks that cover the same thing (e.g. one dish as both dirty and clean)."""
    order = np.argsort(scores)[::-1]
    kept: list[int] = []
    for i in order:
        if all(_iou(masks[i], masks[j]) < iou for j in kept):
            kept.append(int(i))
    kept.sort()
    return [masks[i] for i in kept], [labels[i] for i in kept], [scores[i] for i in kept]


def _iou(a, b) -> float:
    inter = np.logical_and(a, b).sum()
    return float(inter / (np.logical_or(a, b).sum() or 1))


# ---------- timing and records ----------


def other_programs() -> dict:
    try:
        return {str(k): v for k, v in harness.other_gpu_programs().items()}
    except Exception as error:  # nvidia-smi missing: say so, do not guess
        return {"unknown": str(error)}


def timed(fn, data: bytes, repeats: int = 3):
    """Run fn(decoded image) `repeats` times. Returns (last result, wall seconds list, model seconds list)."""
    import torch

    walls, models = [], []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        im = decode(data)
        t1 = time.perf_counter()
        result = fn(im)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        walls.append(t2 - t0)
        models.append(t2 - t1)
    return result, walls, models


def record(*, model: str, track: str, task, text: str, walls, models, factor: float, prompt: str,
           sha: str, extra: dict | None = None) -> dict:
    model_s = statistics.median(models)
    r = {
        "model": model, "level": None, "track": track, "task": task.key, "task_number": task.number,
        "repeat": 0, "prompt_version": PROMPT_VERSION, "prompt": prompt, "image": task.photo, "image_sha256": sha,
        "provider": "local-gpu", "route": "local", "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seconds": round(statistics.median(walls), 4), "seconds_all": [round(x, 4) for x in walls],
        "model_seconds": round(model_s, 4),
        "cost_usd": harness.PRICE_PER_GPU_HOUR / 3600 * model_s * factor,
        "cost_method": f"${harness.PRICE_PER_GPU_HOUR}/GPU-hour x model seconds x load factor {factor:.3f} (2026-10-08 throughput test)",
        "text": text, "gpu_others": other_programs(),
    }
    r["timing_trustworthy"] = not r["gpu_others"]
    if extra:
        r.update(extra)
    return r


class Writer:
    """Appends records to results/run-<name>.jsonl and their masks to run-<name>.masks.jsonl.gz."""

    def __init__(self, name: str):
        self.path = ROOT / "results" / f"run-{name}.jsonl"
        self.masks = ROOT / "results" / f"run-{name}.masks.jsonl.gz"

    def write(self, rec: dict, masks: list | None = None) -> None:
        rec["mask_id"] = hashlib.sha1(json.dumps([rec["model"], rec["track"], rec["task"], rec.get("parts")]).encode()).hexdigest()[:12]
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        if masks is not None:
            with gzip.open(self.masks, "at") as f:
                f.write(json.dumps({"mask_id": rec["mask_id"], "masks": masks}) + "\n")


def gpu_note() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout
        return "\n".join(line for line in out.splitlines() if harness.GPU_UUID in line)
    except Exception:
        return ""
