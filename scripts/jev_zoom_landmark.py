"""Jev finds the Golden Gate Bridge on a map with no words at all (Maxime, 2026-10-16).

After scripts/jev_zoom.py (an address on a labelled map: Jev boxed the printed names,
and one wrong box could never be undone). Here the map has no text, so only shapes
(coastline, bay, roads, parks) can guide it; the target is a landmark; Jev may say the
target is not on the map, and the search then zooms back out.

    set -a; source .env; set +a
    uv run --with osmnx==2.1.1 --with pillow --with shapely --with mapbox-vector-tile scripts/jev_zoom_landmark.py --tries 3 --name gg1
    uv run ... scripts/jev_zoom_landmark.py --target corner --tries 3 --name corner1

Targets:
- bridge (gg1): the Golden Gate Bridge on the wordless map, found at zoom 15.
- corner (corner1, Maxime 2026-10-16, after gg1): the corner of California Street and
  Battery Street (TypeSafe AI's office), on OpenStreetMap's standard map WITH its names
  (tile.openstreetmap.org), found at zoom 18 (about 480 m across: both street names and
  the crossing are drawn). The corner = the street network's node where the two meet
  (OSM node 65295279, 37.7932159, -122.3999769), 44 m from the 255 California St point.
  "Box touches the target" = the corner is inside the box. Same search rules otherwise.

Starts: --start california (default; gg1, corner1): zoom 7 on central California.
--start us (corner-us1, Maxime 2026-10-16): zoom 4 on the middle of the lower 48 states
(39.5, -98.35, Kansas), about 7,800 km across: the whole country, San Francisco near the
left edge. Zoom-outs then stop at zoom 4 (the start's zoom) instead of 5.
--start world (corner-world1, Maxime 2026-10-16): zoom 2 centred on (0, 0), the whole
world in one 1024 px map (zoom 2 is 1024 px around); zoom-outs stop at zoom 2.

Rules (set before the first run):
- Map: drawn by scripts/wordless_map.py from OpenStreetMap shapes (water, land cover,
  parks, roads; no layer that holds a name is read, so no text at all), 1024 x 1024 px,
  north up, nothing drawn on it. (CARTO's no-label tiles now need a key; Esri's
  "no labels" base still prints "Golden Gate" and "GOLDEN GATE BRG".) First map: zoom 7
  centred on central California (37.2, -119.5), as in jev_zoom.py.
- One request per map, two judgments: yes/no "is the bridge (or, too zoomed out to
  draw it, the place where it stands) on this map?" and a box of it.
- Yes: the next map is centred on the box's middle, zoomed in by floor(log2(1024 / the
  box's longer side)), at least 1 and at most 3 levels, never past END_Z.
  Yes at END_Z (zoom 15, about 3.9 km across: the whole bridge fits): found, stop.
- No: zoom out one level around the same centre (never below zoom 5).
- At most MAX_MAPS maps.
- Measured (never shown to Jev): the bridge's outline from OpenStreetMap (relation
  21407754, through Nominatim 2026-10-16); whether it is on each map; whether Jev's box
  touches it; the distance from the box's middle to the bridge's centre line; at the
  end, the overlap (IoU) of Jev's box with the rectangle around the bridge.

Data: map (c) OpenStreetMap contributors (ODbL), OpenMapTiles schema, via OpenFreeMap.
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

from lm15 import bounding_box, judgments, yes_no
from PIL import Image, ImageDraw
from shapely.geometry import LineString, Point, Polygon, box as sbox

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from jev_drive import CACHE, View, font, iou, meters  # noqa: E402
from jev_zoom import unproject  # noqa: E402
from segbench.call import call  # noqa: E402
from segbench.models import BY_KEY  # noqa: E402
import wordless_map  # noqa: E402

MODEL = BY_KEY["jev"]
STARTS = {"california": (37.2, -119.5, 7), "us": (39.5, -98.35, 4), "world": (0.0, 0.0, 2)}
START = STARTS["california"]
SIZE, END_Z, MIN_Z, MAX_STEP, MAX_MAPS = 1024, 15, 5, 3, 25

_gg = json.loads((CACHE / "golden-gate.json").read_text())
OUTLINE = next(r for r in _gg if r["osm_id"] == 21407754)["geojson"]["coordinates"][0]  # [lon, lat]
_lats = [p[1] for p in OUTLINE]
SOUTH, NORTH = OUTLINE[_lats.index(min(_lats))], OUTLINE[_lats.index(max(_lats))]
CENTRE_LINE = [(SOUTH[1], SOUTH[0]), (NORTH[1], NORTH[0])]  # (lat, lon), south to north end


CORNER = (37.7932159, -122.3999769)
TARGETS = {
    "bridge": {"end_z": 15, "map": "wordless (scripts/wordless_map.py)",
               "yes": "Is the Golden Gate Bridge on this map? Answer yes also when the map is too "
                      "zoomed out to draw the bridge but the place where it stands is on the map.",
               "box": "the Golden Gate Bridge on this map. If the map is too zoomed out to draw the "
                      "bridge, box the place where it stands."},
    "corner": {"end_z": 18, "map": "OpenStreetMap standard (with names)",
               "yes": "Is the corner of California Street and Battery Street, San Francisco, on this "
                      "map? Answer yes also when the map is too zoomed out to draw those streets but "
                      "the place where they cross is on the map.",
               "box": "the corner of California Street and Battery Street, San Francisco (where the two "
                      "streets cross), on this map. If the map is too zoomed out to draw those streets, "
                      "box the place where they cross."},
}
T = {"name": "bridge"}


def picture(view: View) -> Image.Image:
    return wordless_map.draw(view) if T["name"] == "bridge" else view.picture()


def questions():
    t = TARGETS[T["name"]]
    return judgments(on_map=yes_no(t["yes"]), box=bounding_box(t["box"]))


def ask(img):
    b = io.BytesIO()
    img.save(b, "PNG")
    q = questions()
    for attempt in range(5):
        rec = call(MODEL, b.getvalue(), "image/png", None, level="medium", response_format=q)
        if not rec.get("error") or not rec.get("retryable"):
            break
        time.sleep(min(30, 2 ** attempt))
    if rec.get("error"):
        raise RuntimeError(f"Jev call failed: {rec['error']}")
    return rec, q


def bridge_px(view):
    if T["name"] == "corner":
        return Point(view.px(*CORNER))
    poly = Polygon([view.px(lat, lon) for lon, lat in OUTLINE])
    return poly if poly.is_valid else poly.buffer(0)


def target_m(p):
    return meters(p, CORNER) if T["name"] == "corner" else seg_m(p, *CENTRE_LINE)


def seg_m(p, a, b):
    k = 111_320 * math.cos(math.radians(p[0]))
    P, A, B = ((q[1] * k, q[0] * 111_320) for q in (p, a, b))
    return LineString([A, B]).distance(Point(P))


def draw(img, view, box, title, said_yes, nxt=None):
    """For us: Jev's box (pink if yes, grey if no), the next map's frame (green), the bridge (blue)."""
    img = img.copy()
    d = ImageDraw.Draw(img, "RGBA")
    poly = bridge_px(view)
    c = poly.centroid
    if -50 < c.x < view.w + 50 and -50 < c.y < view.h + 50:
        if poly.geom_type == "Polygon" and poly.length > 40:
            d.polygon(list(poly.exterior.coords), outline=(30, 90, 230, 255), width=4)
        else:
            d.line((c.x - 14, c.y, c.x + 14, c.y), fill=(30, 90, 230, 255), width=5)
            d.line((c.x, c.y - 14, c.x, c.y + 14), fill=(30, 90, 230, 255), width=5)
    if nxt:
        d.rectangle(nxt, outline=(0, 150, 80, 255), width=3)
    d.rectangle(box, outline=(255, 0, 200, 255) if said_yes else (120, 120, 120, 255), width=5)
    d.rectangle((0, 0, view.w, 34), fill=(255, 255, 255, 225))
    d.text((10, 6), title, fill=(0, 0, 0, 255), font=font(20))
    return img


def run(out: Path, log):
    out.mkdir(parents=True, exist_ok=True)
    lat, lon, z = START
    min_z = min(MIN_Z, START[2])
    steps, status = [], "out of maps"
    while len(steps) < MAX_MAPS:
        n = len(steps) + 1
        view = View(lat, lon, z)
        img = picture(view)
        img.save(out / f"map-{n:02d}-z{z}.png")
        rec, q = ask(img)
        b, yes = rec["data"]["box"], bool(rec["data"]["on_map"])
        box = [b["left"] * SIZE, b["top"] * SIZE, b["right"] * SIZE, b["bottom"] * SIZE]
        mid = unproject(view, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        poly = bridge_px(view)
        on_map = poly.intersects(sbox(0, 0, SIZE, SIZE))
        e = {"step": n, "zoom": z, "centre": [lat, lon],
             "map_width_m": round(SIZE * 156543.03 * math.cos(math.radians(lat)) / 2 ** z),
             "said_on_map": yes, "p_on_map": (rec.get("probabilities") or {}).get("on_map"),
             "really_on_map": on_map, "box_px": [round(v, 1) for v in box],
             "box_confidence": rec["response"]["provider_data"]["typesafe"]["answers"]["box"].get("confidence"),
             "box_touches_bridge": poly.intersects(sbox(*box)), "error_m": round(target_m(mid)),
             "bridge_iou": round(iou(box, list(poly.bounds)), 3) if T["name"] == "bridge" else None,
             "seconds": rec["seconds"], "cost_usd": rec["cost_usd"], "question": q}
        nxt, verdict = None, ""
        end_z = TARGETS[T["name"]]["end_z"]
        if yes and z >= end_z:
            status, verdict = "found", "STOP (found)"
        elif yes:
            side = max(box[2] - box[0], box[3] - box[1], 1)
            up = min(MAX_STEP, end_z - z, max(1, int(math.floor(math.log2(SIZE / side)))))
            half = SIZE / 2 ** up / 2
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            nxt = [cx - half, cy - half, cx + half, cy + half]
            lat, lon, z, verdict = mid[0], mid[1], z + up, f"zoom in to {z + up}"
        else:
            verdict = f"zoom out to {max(min_z, z - 1)}"
            z = max(min_z, z - 1)
        e["then"] = verdict
        steps.append(e)
        title = (f"{n:02d}  zoom {e['zoom']} ({e['map_width_m'] / 1000:.3g} km)  says {'YES' if yes else 'NO'}"
                 f" (really {'yes' if on_map else 'no'})  box {e['error_m']:,} m off  -> {verdict}")
        draw(img, view, box, title, yes, nxt).save(out / f"step-{n:02d}.png")
        log(title + f"  p(yes)={(e['p_on_map'] or {}).get('true')}  in box: {e['box_touches_bridge']}")
        if status == "found":
            break
    return steps, status


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
    ap.add_argument("--target", choices=list(TARGETS), default="bridge")
    ap.add_argument("--start", choices=list(STARTS), default="california")
    args = ap.parse_args()
    T["name"] = args.target
    global START
    START = STARTS[args.start]
    base = ROOT / "results" / "probe-jev-zoom-landmark" / args.name
    for t in range(1, args.tries + 1):
        out = base / f"try-{t}"
        steps, status = run(out, lambda s: print(f"[try {t}] {s}", flush=True))
        last = steps[-1]
        summary = {"status": status, "maps": len(steps), "final_zoom": last["zoom"], "final_error_m": last["error_m"],
                   "final_box_touches_bridge": last["box_touches_bridge"], "final_bridge_iou": last["bridge_iou"],
                   "yes_no_right": sum(s["said_on_map"] == s["really_on_map"] for s in steps),
                   "zoom_outs": sum(not s["said_on_map"] for s in steps),
                   "cost_usd": sum(s["cost_usd"] or 0 for s in steps), "model": MODEL.route,
                   "target": args.target, "start": args.start, "map": TARGETS[args.target]["map"],
                   "when": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        (out / "steps.json").write_text(json.dumps({"summary": summary, "steps": steps}, indent=1))
        sheet(out, len(steps))
        print(f"[try {t}] {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
