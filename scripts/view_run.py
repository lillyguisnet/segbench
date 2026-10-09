# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow>=10", "numpy>=1.26", "lm15>=1.2.1"]
# ///
"""See what the models answered: every reply drawn on its photo, with measurements.

    uv run scripts/view_run.py results/run-pilot-1.jsonl

Writes results/views/<run>/index.html (open it in a browser) and small copies
of the photos next to it. Nothing here is a score: without the answer key we
can show what each model drew and the numbers it implies (counts, areas, the
road's width), not whether they are right.

Measurements are taken on the model's outlines filled in on the photo's
pixel grid (overlapping outlines are counted once):
- logs:   count, and each log end's diameter (a circle of the same area), in photo pixels
- cows:   count
- fig:    count, and total fig-leaf area as % of the photo
- tree:   reddish area as % of the photo (not yet % of the crown: that needs the crown's mask)
- road:   left edge, right edge and width at 6 heights, and the highest point the road reaches
- dishes: dirty and clean counts
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # this script runs in its own environment (above)

from segbench import parse  # noqa: E402
from segbench.models import BY_KEY as MODELS  # noqa: E402
from segbench.tasks import BY_KEY as TASKS  # noqa: E402

THUMB = 1400  # longest side of the photos in the viewer
ROAD_HEIGHTS = (950, 850, 750, 650, 550, 450)  # y, 0..1000 from the top


def mask(polygons, w, h) -> np.ndarray:
    im = Image.new("1", (w, h), 0)
    d = ImageDraw.Draw(im)
    for poly in polygons:
        d.polygon([(x * w / 1000, y * h / 1000) for x, y in poly], fill=1)
    return np.array(im, dtype=bool)


def measure(task, track, items, photo_size) -> dict:
    W, H = photo_size
    m: dict = {"count": len(items)}
    labels = [i.get("label") for i in items]
    if task.labels:
        for label in task.labels:
            m[label] = labels.count(label)
    if track != "whole":
        return m
    w, h = (1000, round(1000 * H / W)) if W >= H else (round(1000 * W / H), 1000)
    union = mask([i["polygon"] for i in items], w, h)
    m["area_pct"] = round(100 * union.mean(), 2)
    if task.key == "logs":
        px_per_cell = (W / w) * (H / h)
        diameters = [2 * math.sqrt(mask([i["polygon"]], w, h).sum() * px_per_cell / math.pi) for i in items]
        diameters = [d for d in diameters if d > 0]
        if diameters:
            m["diameter_px"] = {"median": round(statistics.median(diameters)), "min": round(min(diameters)),
                                "max": round(max(diameters))}
    if task.key == "road":
        rows = []
        for y in ROAD_HEIGHTS:
            row = union[min(h - 1, round(y * h / 1000))]
            xs = np.flatnonzero(row)
            rows.append({"y": y, "left": round(xs[0] * 1000 / w), "right": round(xs[-1] * 1000 / w),
                         "width_pct": round(100 * len(xs) / w, 1)} if len(xs) else {"y": y})
        m["road"] = rows
        ys = np.flatnonzero(union.any(axis=1))
        m["reaches_y"] = round(ys[0] * 1000 / h) if len(ys) else None
    return m


def main(path: str) -> None:
    run = Path(path)
    out = ROOT / "results" / "views" / run.stem.removeprefix("run-")
    (out / "img").mkdir(parents=True, exist_ok=True)
    photos = {}
    records = []
    for line in run.read_text().splitlines():
        r = json.loads(line)
        task = TASKS[r["task"]]
        if task.photo not in photos:
            with Image.open(ROOT / "images" / task.photo) as im:
                photos[task.photo] = im.size
                thumb = im.convert("RGB")
                thumb.thumbnail((THUMB, THUMB))
                thumb.save(out / "img" / task.photo, quality=82)
        rec = {"model": r["model"], "label": MODELS[r["model"]].label, "level": r["level"], "track": r["track"],
               "task": r["task"], "repeat": r["repeat"], "seconds": r.get("seconds"), "cost": r.get("cost_usd"),
               "thinking": (r.get("usage") or {}).get("reasoning_tokens")}
        if "error" in r:
            rec["error"] = r["error"][:300]
        else:
            try:  # read with today's reader, so reader fixes apply to old runs
                answer = parse.read(r["track"], task.kind, r["text"], task.labels)
                rec["items"] = [{k: ([[round(v) for v in p] for p in val] if k == "polygon" else
                                     [round(v) for v in val] if k == "point" else val)
                                 for k, val in it.items()} for it in answer["items"]]
                rec["m"] = measure(task, r["track"], answer["items"], photos[task.photo])
            except Exception as error:
                rec["unreadable"] = f"{type(error).__name__}: {error}"[:200]
                rec["text"] = r["text"][:600]
        records.append(rec)

    tasks = [{"key": t.key, "number": t.number, "photo": t.photo, "kind": t.kind, "target": t.target,
              "absent": t.absent, "w": photos[t.photo][0], "h": photos[t.photo][1]}
             for t in TASKS.values() if t.photo in photos]
    data = json.dumps({"run": run.stem, "tasks": tasks, "records": records}, separators=(",", ":"))
    html = (ROOT / "scripts" / "view_run.html").read_text().replace("/*DATA*/null", data)
    (out / "index.html").write_text(html)
    print(f"wrote {out / 'index.html'} ({len(html) // 1024} KB, {len(records)} replies)")


if __name__ == "__main__":
    main(sys.argv[1])
