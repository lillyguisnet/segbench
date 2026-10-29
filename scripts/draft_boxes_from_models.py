"""Draft answer-key boxes from the top pointing models' box answers, for a person to review.

    uv run --with pillow scripts/draft_boxes_from_models.py  ->  annotations/boxes/draft-models-v1.json

Maxime, 2026-10-15: SAM 2.1's drafts (draft-sam2.1.json) were too often off
to be a quick starting point; start from the box answers of the best
pointing models instead.

Models: the five best of point benchmark v3 that have box answers (GPT-6
Astra, GPT-6.1 Sol, Gemini 3.8 Flash, Gemini 4 Argon, Muse Wonder (API));
the two Qwen models rank 6th and 7th but have no box answers yet (OpenRouter
limit). Every answer: both prompts, all three tries (up to 30 per photo),
read with the frozen rules (segbench/box_read.py).

For each approved reference dot (the objects are fixed by the dots; models
only propose edges):
- in each answer, the boxes that contain the dot; of those, a box is given
  to this dot only if this dot is, of all the reference dots inside the
  box, the one nearest its centre (a box around a group, or around a
  neighbour, is not this object's box); if several boxes qualify, the one
  whose centre is nearest the dot;
- the draft = the median of each side over the answers that gave one
  (a median per side resists one wild box);
- support = how many answers gave one; agreement = the median overlap
  (IoU) of those boxes with the draft.
With fewer than 3 supporting answers the SAM 2.1 draft is kept instead and
marked so. Nothing here is ground truth: a person checks every box.

Known bias, stated in docs/box-benchmark.md: a key that starts from these
five models' boxes may favour them (and models that draw like them) at the
strict overlap levels, if the reviewer keeps their edges.
"""
from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from segbench.box_read import iou, read_box_a, read_box_free  # noqa: E402
from segbench.tasks import BY_KEY  # noqa: E402

MODELS = ["astra", "sol", "gemini-flash", "gemini-argon", "muse-wonder-api"]
RUNS = ["run-boxes-box-a.jsonl", "run-boxes-box-a-resent.jsonl", "run-boxes-box-free.jsonl",
        "run-boxes-box-free-subs.jsonl", "run-boxes-box-free-resent.jsonl"]
MIN_SUPPORT = 3


def answers():
    """task -> [(model, prompt, try, boxes in photo pixels)]."""
    out: dict[str, list] = {}
    sizes = {}
    for name in RUNS:
        path = ROOT / "results" / name
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r["model"] not in MODELS or "error" in r:
                continue
            t = BY_KEY[r["task"]]
            W, H = sizes.setdefault(t.photo, Image.open(ROOT / "images" / t.photo).size)
            if r["prompt_version"] == "box-a":
                boxes, _ = read_box_a(r["text"])
            else:
                boxes, _ = read_box_free(r["text"], r["model"], W, H)
            px = [[b["box"][0] * W / 1000, b["box"][1] * H / 1000, b["box"][2] * W / 1000, b["box"][3] * H / 1000]
                  for b in boxes]
            out.setdefault(r["task"], []).append((r["model"], r["prompt_version"], r["repeat"], px))
    return out


def main():
    sam = json.loads((ROOT / "annotations/boxes/draft-sam2.1.json").read_text())
    by_task = answers()
    draft = {"schema": "segbench-box-draft/v1", "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "source": "top pointing models' box answers (median per side), SAM 2.1 where fewer than 3 agree",
             "models": MODELS, "runs": RUNS, "method": __doc__.strip(),
             "coordinate_system": "photo pixels, [x1, y1, x2, y2]", "tasks": {}}
    for key, t in sam["tasks"].items():
        dots = t["objects"]
        objs = []
        for o in dots:
            mine = []
            for model, prompt, rep, boxes in by_task.get(key, []):
                cands = []
                for b in boxes:
                    if not (b[0] <= o["x"] <= b[2] and b[1] <= o["y"] <= b[3]):
                        continue
                    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                    inside = [d for d in dots if b[0] <= d["x"] <= b[2] and b[1] <= d["y"] <= b[3]]
                    nearest = min(inside, key=lambda d: (d["x"] - cx) ** 2 + (d["y"] - cy) ** 2)
                    if nearest["id"] == o["id"]:
                        cands.append(((o["x"] - cx) ** 2 + (o["y"] - cy) ** 2, b))
                if cands:
                    mine.append((model, prompt, rep, min(cands)[1]))
            row = {"id": o["id"], "x": o["x"], "y": o["y"], "label": o.get("label"), "support": len(mine),
                   "models": sorted({m for m, *_ in mine}), "sam_box": o["box"]}
            if len(mine) >= MIN_SUPPORT:
                box = [round(statistics.median(b[i] for *_, b in mine)) for i in range(4)]
                row.update(box=box, source="models",
                           agreement=round(statistics.median(iou(b, box) for *_, b in mine), 3))
            else:
                row.update(box=o["box"], source="sam2.1 (fewer than 3 model boxes)", agreement=None)
            row["iou_with_sam"] = round(iou(row["box"], o["box"]), 3) if o["box"] else None
            objs.append(row)
        draft["tasks"][key] = {k: t[k] for k in ("photo", "width", "height", "reference_file")} | {"objects": objs}
        m = [x for x in objs if x["source"] == "models"]
        print(f"{key:11} {len(objs):3} objects: {len(m):3} from models (median support "
              f"{statistics.median(x['support'] for x in m) if m else 0:.0f}, median agreement "
              f"{statistics.median(x['agreement'] for x in m) if m else 0:.2f}), {len(objs) - len(m)} kept from SAM")
    out = ROOT / "annotations/boxes/draft-models-v1.json"
    out.write_text(json.dumps(draft, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
