"""Jev zooms in on an address, one map at a time (Maxime, 2026-10-16).

Find TypeSafe AI, 255 California St, San Francisco, starting from a map of most of
California. Each step Jev gets one map (no mark of the answer on it) and boxes where
the address is; the next map is centred on the middle of that box and zoomed in so the
box about fills it, until the closest zoom of the map.

    set -a; source .env; set +a
    uv run --with osmnx==2.1.1 --with pillow scripts/jev_zoom.py --tries 3 --name zoom1

Rules (set before the first run):
- Map: OpenStreetMap's standard map (tile.openstreetmap.org), 1024 x 1024 px, north up,
  nothing drawn on it. First map: zoom 7 centred on central California (37.2, -119.5),
  so San Francisco is neither in the middle nor marked.
- Question: one Jev box (lm15 bounding_box): the address, and "if it is not on the map,
  the part of the map closest to it".
- Next map: centred on the box's middle; zoom raised by floor(log2(1024 / the box's
  longer side in px)), at least 1 and at most 3 levels, never above MAX_Z (19, the
  map's closest zoom). It never zooms back out: a wrong box is not corrected.
- Measured (never shown to Jev): the address's true position (Nominatim, 2026-10-16:
  the building at 255 California St), whether it is on each map and inside each box,
  and the distance from the box's middle to it.

Data: map pictures (c) OpenStreetMap contributors (ODbL).
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from lm15 import bounding_box, judgments
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from jev_drive import View, font, meters  # noqa: E402
from segbench.call import call  # noqa: E402
from segbench.models import BY_KEY  # noqa: E402

MODEL = BY_KEY["jev"]
TARGET = (37.7929589, -122.3995978)   # Nominatim, 2026-10-16
ADDRESS = "255 California Street, San Francisco, CA 94111 (the office of TypeSafe AI)"
START = (37.2, -119.5, 7)
SIZE, MAX_Z, MAX_STEP = 1024, 19, 3


def unproject(view: View, x, y):
    """Pixel on the view -> (lat, lon)."""
    n = 256 * 2 ** view.z
    wx, wy = view.x0 + x, view.y0 + y
    lon = wx / n * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * wy / n))))
    return lat, lon


def question():
    return judgments(box=bounding_box(
        f"where {ADDRESS} is on this street map. If the address itself is not on the map, box "
        f"the part of the map closest to it."))


def ask(img):
    b = io.BytesIO()
    img.save(b, "PNG")
    q = question()
    for attempt in range(5):
        rec = call(MODEL, b.getvalue(), "image/png", None, level="medium", response_format=q)
        if not rec.get("error") or not rec.get("retryable"):
            break
        time.sleep(min(30, 2 ** attempt))
    if rec.get("error"):
        raise RuntimeError(f"Jev call failed: {rec['error']}")
    return rec, q


def draw(img, view, box, title, nxt=None):
    """For us: Jev's box (pink), the next map's frame (dashed green), the true spot (blue cross)."""
    img = img.copy()
    d = ImageDraw.Draw(img, "RGBA")
    if nxt:
        d.rectangle(nxt, outline=(0, 150, 80, 255), width=3)
    d.rectangle(box, outline=(255, 0, 200, 255), width=5)
    tx, ty = view.px(*TARGET)
    if -20 < tx < view.w + 20 and -20 < ty < view.h + 20:
        d.line((tx - 14, ty, tx + 14, ty), fill=(30, 90, 230, 255), width=5)
        d.line((tx, ty - 14, tx, ty + 14), fill=(30, 90, 230, 255), width=5)
    d.rectangle((0, 0, view.w, 34), fill=(255, 255, 255, 225))
    d.text((10, 6), title, fill=(0, 0, 0, 255), font=font(20))
    return img


def run(out: Path, log):
    out.mkdir(parents=True, exist_ok=True)
    lat, lon, z = START
    steps = []
    while True:
        view = View(lat, lon, z)
        img = view.picture()
        img.save(out / f"map-{len(steps) + 1:02d}-z{z}.png")
        rec, q = ask(img)
        b = rec["data"]["box"]
        box = [b["left"] * SIZE, b["top"] * SIZE, b["right"] * SIZE, b["bottom"] * SIZE]
        mid = unproject(view, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        tx, ty = view.px(*TARGET)
        side = max(box[2] - box[0], box[3] - box[1], 1)
        up = 0 if z >= MAX_Z else min(MAX_STEP, MAX_Z - z, max(1, int(math.floor(math.log2(SIZE / side)))))
        m_per_px = 156543.03 * math.cos(math.radians(TARGET[0])) / 2 ** z
        e = {"step": len(steps) + 1, "zoom": z, "centre": [lat, lon], "map_width_m": round(SIZE * m_per_px),
             "box_px": [round(v, 1) for v in box],
             "box_confidence": rec["response"]["provider_data"]["typesafe"]["answers"]["box"].get("confidence"),
             "on_map": 0 <= tx <= SIZE and 0 <= ty <= SIZE,
             "in_box": box[0] <= tx <= box[2] and box[1] <= ty <= box[3],
             "error_m": round(meters(mid, TARGET)), "next_zoom": z + up,
             "seconds": rec["seconds"], "cost_usd": rec["cost_usd"], "question": q}
        steps.append(e)
        nxt = None
        if up:
            half = SIZE / 2 ** up / 2
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            nxt = [cx - half, cy - half, cx + half, cy + half]
        title = (f"{e['step']:02d}  zoom {z} ({e['map_width_m'] / 1000:.3g} km across)  "
                 f"box middle {e['error_m']:,} m off  {'inside box' if e['in_box'] else 'NOT in box'}")
        draw(img, view, box, title, nxt).save(out / f"step-{e['step']:02d}.png")
        log(title + f"  conf {e['box_confidence']:.2f}")
        if not up:
            break
        lat, lon, z = mid[0], mid[1], z + up
    return steps


def sheet(out: Path, n):
    s, cols = 512, 4
    rows = (n + cols - 1) // cols
    img = Image.new("RGB", (cols * s, rows * s), "white")
    for i in range(n):
        img.paste(Image.open(out / f"step-{i + 1:02d}.png").resize((s, s), Image.LANCZOS), ((i % cols) * s, (i // cols) * s))
    img.save(out / "all-steps.jpg", quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--name", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    args = ap.parse_args()
    base = ROOT / "results" / "probe-jev-zoom" / args.name
    for t in range(1, args.tries + 1):
        out = base / f"try-{t}"
        steps = run(out, lambda s: print(f"[try {t}] {s}", flush=True))
        summary = {"maps": len(steps), "final_error_m": steps[-1]["error_m"],
                   "first_miss_step": next((s["step"] for s in steps if not s["in_box"]), None),
                   "cost_usd": sum(s["cost_usd"] or 0 for s in steps), "model": MODEL.route,
                   "when": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        (out / "steps.json").write_text(json.dumps({"summary": summary, "steps": steps}, indent=1))
        sheet(out, len(steps))
        print(f"[try {t}] {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
