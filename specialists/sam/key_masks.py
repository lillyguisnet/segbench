"""Outlines of the answer-key objects, for sorting mistakes (scripts/classify_errors.py).

    uv run key_masks.py  ->  ../../results/error-kinds/support/key-masks-sam2.1.npz

For every box of the reviewed answer key (annotations/boxes/answer-key-boxes-v1.json),
SAM 2.1 Large is given that box, plus the object's reference dot as a click
when the dot lies inside the box (the point key and the box key share ids).
One outline per object (SAM's advice when the prompt is not ambiguous), cut
to the box. Where SAM's outline covers under 5% of the box, the box itself is
kept as the outline and the object is listed in the file's "fallback".

Why: a box is a rectangle; a point inside it can still be on the background
(between a bird's wings, between the lobes of a leaf). The outline says
whether a point is on the object. SAM 2.1 because it is not an entrant
(as for draft-sam2.1.json). The outlines are not reviewed by a person: they
are a measuring aid for the error sorting, never part of a score.

Saved: one packed bit array per object (numpy packbits of the full-photo
mask), key "<task>/<id>", plus "_meta" (JSON) with sizes and SAM scores.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import harness  # noqa: E402,F401  (pins the GPU before torch loads)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
KEY = ROOT / "annotations/boxes/answer-key-boxes-v1.json"
DOTS = ["annotations/reviews/segbench-review-6187f623-approved.json",
        "annotations/reviews/segbench-review-5682a4be-approved.json"]
OUT = ROOT / "results/error-kinds/support/key-masks-sam2.1.npz"


def main() -> None:
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if OUT.exists():
        sys.exit(f"{OUT} exists")
    key = json.loads(KEY.read_text())
    dots, photos = {}, {}
    for f in DOTS:
        raw = json.loads((ROOT / f).read_text())
        for s in raw["source_snapshot"]["tasks"]:
            photos[s["key"]] = Path(s["photo"]).name
        for t in raw["tasks"].values():
            for o in t.get("objects", []):
                if o["status"] == "accepted":
                    dots[o["id"]] = (o["x"], o["y"])
    sam = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large")
    arrays, meta = {}, {"model": "facebook/sam2.1-hiera-large", "key": KEY.name, "method": __doc__.strip(),
                        "tasks": {}, "fallback": []}
    for task, t in key["tasks"].items():
        im = ImageOps.exif_transpose(Image.open(ROOT / "images" / photos[task])).convert("RGB")
        assert im.size == (t["width"], t["height"]), task
        objs = [o for o in t["objects"] if o["status"] == "accepted"]
        meta["tasks"][task] = {"width": im.width, "height": im.height, "objects": {}}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            sam.set_image(np.asarray(im))
            for o in objs:
                x1, y1, x2, y2 = o["box"]
                kw = {"box": np.array([x1, y1, x2, y2], dtype=np.float32)}
                d = dots.get(o["id"])
                if d and x1 <= d[0] <= x2 and y1 <= d[1] <= y2:
                    kw.update(point_coords=np.array([d], dtype=np.float32), point_labels=np.array([1]))
                masks, scores, _ = sam.predict(multimask_output=False, **kw)
                m = masks[0] > 0
                clip = np.zeros_like(m)
                xi1, yi1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
                xi2, yi2 = min(im.width, int(np.ceil(x2)) + 1), min(im.height, int(np.ceil(y2)) + 1)
                clip[yi1:yi2, xi1:xi2] = True
                m &= clip
                box_area = max(1, (xi2 - xi1) * (yi2 - yi1))
                fill = m.sum() / box_area
                if fill < 0.05:
                    m = clip
                    meta["fallback"].append(o["id"])
                arrays[f"{task}/{o['id']}"] = np.packbits(m)
                meta["tasks"][task]["objects"][o["id"]] = {"box": o["box"], "sam_score": round(float(scores[0]), 4),
                                                           "box_fill": round(float(fill), 3), "dot_click": "point_coords" in kw}
        fills = [v["box_fill"] for v in meta["tasks"][task]["objects"].values()]
        print(f"{task:11} {len(objs):3} outlines, median share of the box covered {np.median(fills):.2f}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, _meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8), **arrays)
    print("saved", OUT, "fallback:", meta["fallback"])


if __name__ == "__main__":
    main()
