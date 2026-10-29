"""Draft answer-key boxes from the approved reference dots, for a person to review.

    SEGBENCH_GPU_UUID=<uuid> uv run draft_boxes.py  ->  ../../annotations/boxes/draft-sam2.1.json

For every accepted reference dot of the eleven point tasks, SAM 2.1 Large
is given that dot as a click on the object, plus "not this" clicks on the
neighbouring reference dots of the same task (the ten nearest within three
times the distance to the nearest one), so that a log end's outline does
not run into the next log. The draft box is the smallest rectangle around
the resulting outline (pieces under 5% of the largest piece dropped).

Why SAM 2.1 and not SAM 3: SAM 3 is an entrant of the benchmark; an answer
key drafted by it could favour its way of drawing edges at the strict
overlap levels (90%, 95%). SAM 2.1 is not an entrant. The boxes are drafts:
a person checks every one (apps/box-review) and that review is the answer key.

Single click with no neighbour: SAM's three candidate outlines, the one it
rates best (the standard use of one click). With neighbours: one outline
(SAM's advice when the clicks are not ambiguous).
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import harness  # noqa: E402,F401  (pins the GPU before torch loads)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REFERENCES = ["annotations/reviews/segbench-review-6187f623-approved.json",
              "annotations/reviews/segbench-review-5682a4be-approved.json"]
TASKS = ["logs", "cows", "fig", "dishes", "ants", "beetles", "birds", "mushrooms", "jars", "pinkleaves", "whitecow"]
NEIGHBOURS, RADIUS_FACTOR, MIN_PIECE = 10, 3.0, 0.05


def tight_box(mask: np.ndarray) -> tuple[list[int] | None, int]:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None, 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = [i + 1 for i, a in enumerate(areas) if a >= MIN_PIECE * areas.max()]
    x1 = min(stats[i, cv2.CC_STAT_LEFT] for i in keep)
    y1 = min(stats[i, cv2.CC_STAT_TOP] for i in keep)
    x2 = max(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] for i in keep)
    y2 = max(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] for i in keep)
    return [int(x1), int(y1), int(x2), int(y2)], len(keep)


def main() -> None:
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    refs = {}
    for f in REFERENCES:
        raw = json.loads((ROOT / f).read_text())
        for key, t in raw["tasks"].items():
            if key in TASKS:
                refs[key] = (t, f, raw["source_snapshot"])
    sam = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large")
    out = {"schema": "segbench-box-draft/v1", "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "model": "facebook/sam2.1-hiera-large", "method": __doc__.strip(), "coordinate_system": "photo pixels, [x1, y1, x2, y2]",
           "tasks": {}}
    for key in TASKS:
        t, source, snap = refs[key]
        photo = next(s["photo"] for s in snap["tasks"] if s["key"] == key).split("/")[-1]
        im = Image.open(ROOT / "images" / photo).convert("RGB")
        assert im.size == (t["width"], t["height"]), key
        dots = [o for o in t["objects"] if o["status"] == "accepted"]
        xy = np.array([[o["x"], o["y"]] for o in dots], dtype=np.float32)
        objects = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            sam.set_image(np.asarray(im))
            for i, o in enumerate(dots):
                d = np.hypot(*(xy - xy[i]).T)
                d[i] = np.inf
                near = d.min() if len(dots) > 1 else math.inf
                neg = [j for j in np.argsort(d)[:NEIGHBOURS] if d[j] <= RADIUS_FACTOR * near]
                pts = np.concatenate([xy[i:i + 1], xy[neg]]) if neg else xy[i:i + 1]
                labels = np.array([1] + [0] * len(neg))
                masks, scores, _ = sam.predict(point_coords=pts, point_labels=labels, multimask_output=not neg)
                b = int(np.argmax(scores))
                mask = masks[b] > 0
                box, pieces = tight_box(mask)
                inside = bool(mask[min(int(o["y"]), im.height - 1), min(int(o["x"]), im.width - 1)])
                objects.append({"id": o["id"], "x": o["x"], "y": o["y"], "label": o.get("label"), "box": box,
                                "sam_score": round(float(scores[b]), 4), "pieces": pieces, "dot_inside": inside,
                                "negative_clicks": len(neg)})
        out["tasks"][key] = {"photo": photo, "width": im.width, "height": im.height, "reference_file": source,
                             "objects": objects}
        bad = sum(1 for o in objects if not o["dot_inside"] or o["box"] is None)
        print(f"{key:11} {len(objects):3} boxes, {bad} where the dot is not on SAM's outline", flush=True)
    dest = ROOT / "annotations/boxes/draft-sam2.1.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print("saved", dest)


if __name__ == "__main__":
    main()
