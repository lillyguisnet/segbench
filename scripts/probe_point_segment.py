"""Outline what a red dot is on: a 4 x 4 grid of dots, one request per dot (Maxime, 2026-10-16).

uv run --with pillow --with numpy scripts/probe_point_segment.py MODEL [--image PATH] [--go]
    ->  results/probe-point-segment/<image>/<model>-<time>.jsonl  (every call's record)
        results/probe-point-segment/<image>/<model>-<time>.json   (dots, outlines read, cost)
        results/probe-point-segment/<image>/<model>-<time>-*.jpg  (pictures)

Without --go: prints the plan and the first prompt, sends nothing.

What each request holds:
- the photo with ONE red dot drawn on it (a white ring around it, so it shows
  on red and dark things too), and
- the dot's coordinates in pixels of that photo, in words.

The 16 dots sit at the centres of a 4 x 4 grid of equal cells (12.5 %,
37.5 %, 62.5 %, 87.5 % of the width and of the height): equal spacing, and no
dot on the photo's edge. One dot per request rather than 16 numbered dots
in one: no dot can be confused with another, and no outline can borrow from
the answer to another dot. It costs 16 times the photo's input tokens.

The prompt asks for the outline as a polygon in pixels: the coordinate
system has to be fixed to read the answer at all; nothing else about the
format is asked (no JSON, no field names). Replies are read generously:
a code block or the whole text, JSON or plain "(x, y)" pairs, one polygon or
several, or Gemini's own masks (box_2d + base64 PNG) if it chooses them.
Nothing here is scored: there is no answer key for these dots.

Thinking: each model's "medium" (the benchmark's level). Image detail: the
most each provider offers (scripts/run.py IMAGE_SETTINGS_V2).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from segbench.call import call, pool  # noqa: E402
from segbench.models import BY_KEY  # noqa: E402

GRID = 4
WORKERS = {"openai-codex": 4, "claude-code": 3, "muse-code": 3}  # plans are shared with the family
IMAGE_SETTINGS = {"gemini": {"media_resolution": "MEDIA_RESOLUTION_ULTRA_HIGH"}}

_WHERE_PX = ("There is a red dot on this picture, at x = {x}, y = {y} in pixels "
             "(the picture is {w} x {h} pixels; x counts from the left edge, y from the top edge).\n\n")
_WHERE_1000 = ("There is a red dot on this picture, at x = {xn}, y = {yn} on a 0-1000 scale "
               "(x = 0 is the left edge and 1000 the right edge; y = 0 is the top edge and 1000 the bottom edge).\n\n")
_WHAT = ("Segment the thing the red dot is on: outline the whole object it is on, or, if it is on "
         "a background or a surface, that region. ")

# Five ways of asking (Maxime, 2026-10-16: "try five different ways"), written
# before any of them was sent to GPT-6.1 Sol. Each changes one thing from the
# one before it, except E, which asks for a mask instead of an outline:
#   A  the first prompt (Gemini 3.8 Flash's run): polygon, (x, y) pairs, pixels
#   B  A, with every point named {"x": .., "y": ..} in JSON (the point
#      benchmark's fix for x/y swaps, docs/point-benchmark-v2.md)
#   C  B on a 0-1000 scale instead of pixels (dot given on that scale too)
#   D  B, after first saying what exactly is under the dot: an object, or the
#      background between objects (Flash outlined a nearby butterfly when the
#      dot was on the board next to it)
#   E  a mask: the bounding box in pixels, then a 20 x 20 grid of 0/1 cells
#      over the box (the shape of Gemini's own segmentation answers; a grid can
#      have holes, a polygon cannot)
PROMPTS = {
    "A": (_WHERE_PX + _WHAT + "Give the outline as a polygon: the list of (x, y) points, in pixels of this "
          "picture, going around its edge.", "poly-px"),
    "B": (_WHERE_PX + _WHAT + "Give the outline as a polygon going around its edge, in pixels of this picture, "
          'as JSON: [{{"x": .., "y": ..}}, {{"x": .., "y": ..}}, ...]', "poly-px-named"),
    "C": (_WHERE_1000 + _WHAT + "Give the outline as a polygon going around its edge, on the same 0-1000 scale, "
          'as JSON: [{{"x": .., "y": ..}}, {{"x": .., "y": ..}}, ...]', "poly-1000"),
    "D": (_WHERE_PX + "First look closely at exactly what is under the red dot: is it on an object, or on the "
          "background or surface between objects? Say in one sentence what it is on. Then segment that thing: "
          "outline the whole object, or, if the dot is on a background or a surface, that region. "
          "Give the outline as a polygon going around its edge, in pixels of this picture, "
          'as JSON: [{{"x": .., "y": ..}}, {{"x": .., "y": ..}}, ...]', "poly-px-named"),
    "E": (_WHERE_PX + _WHAT + "Give the segmentation as a mask: first the thing's bounding box in pixels of this "
          "picture, then a 20 x 20 grid laid over that box, one row per line from top to bottom, with 1 where "
          "a cell is mostly on the thing and 0 where it is not. As JSON: "
          '{{"box": {{"x_min": .., "y_min": .., "x_max": .., "y_max": ..}}, "mask": ["0011...", ...]}} '
          "(20 strings of 20 characters).", "box-grid"),
}
GRID_CELLS = 20
PROMPT = PROMPTS["A"][0]  # the first run's words


def prompt_text(version: str, x: int, y: int, w: int, h: int) -> str:
    return PROMPTS[version][0].format(x=x, y=y, w=w, h=h, xn=round(x / w * 1000), yn=round(y / h * 1000))


def grid_points(w: int, h: int, n: int = GRID) -> list[tuple[int, int]]:
    """Centres of an n x n grid of equal cells, row by row from the top left."""
    return [(round((i + 0.5) * w / n), round((j + 0.5) * h / n)) for j in range(n) for i in range(n)]


def dot_radius(w: int, h: int) -> int:
    return max(4, round(min(w, h) * 0.008))


def with_dot(photo: Image.Image, x: int, y: int) -> bytes:
    img = photo.copy()
    d = ImageDraw.Draw(img)
    r = dot_radius(*img.size)
    ring = max(1, r // 3)
    d.ellipse((x - r - ring, y - r - ring, x + r + ring, y + r + ring), fill=(255, 255, 255))
    d.ellipse((x - r, y - r, x + r, y + r), fill=(230, 0, 0))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=95)
    return buf.getvalue()


# --- reading the reply ------------------------------------------------------

NUM = r"-?\d+(?:\.\d+)?"
PAIR = re.compile(rf"[\[(]\s*({NUM})\s*,\s*({NUM})\s*[\])]")
XY = re.compile(rf"""["']?x["']?\s*[:=]\s*({NUM})\s*,\s*["']?y["']?\s*[:=]\s*({NUM})""", re.I)


def _polygons_from_json(obj) -> list[list[tuple[float, float]]]:
    """Every list of >= 3 points found in a JSON value, as polygons."""
    def as_point(v):
        if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(c, (int, float)) for c in v):
            return (float(v[0]), float(v[1]))
        if isinstance(v, dict) and "x" in v and "y" in v:
            return (float(v["x"]), float(v["y"]))
        if isinstance(v, dict):  # {"point": [x, y], "label": ...}: the one pair-valued field
            pairs = [as_point(u) for u in v.values() if isinstance(u, (list, tuple))]
            pairs = [p for p in pairs if p is not None]
            return pairs[0] if len(pairs) == 1 else None
        return None

    found = []
    if isinstance(obj, list):
        pts = [as_point(v) for v in obj]
        if len(obj) >= 3 and all(p is not None for p in pts):
            return [pts]
        flat = obj
        if len(obj) >= 6 and len(obj) % 2 == 0 and all(isinstance(v, (int, float)) for v in obj):
            return [[(float(obj[i]), float(obj[i + 1])) for i in range(0, len(obj), 2)]]
        for v in flat:
            found += _polygons_from_json(v)
    elif isinstance(obj, dict):
        for v in obj.values():
            found += _polygons_from_json(v)
    return found


def _gemini_masks(obj, w: int, h: int) -> list[np.ndarray]:
    """Gemini's own segmentation format: [{"box_2d": [y0,x0,y1,x1] (0..1000), "mask": "data:image/png;base64,..."}]."""
    items = obj if isinstance(obj, list) else [obj]
    masks = []
    for it in items:
        if not (isinstance(it, dict) and "box_2d" in it and isinstance(it.get("mask"), str)):
            continue
        y0, x0, y1, x1 = it["box_2d"]
        x0, x1 = round(x0 / 1000 * w), round(x1 / 1000 * w)
        y0, y1 = round(y0 / 1000 * h), round(y1 / 1000 * h)
        if x1 <= x0 or y1 <= y0:
            continue
        png = base64.b64decode(it["mask"].split(",", 1)[-1])
        crop = Image.open(io.BytesIO(png)).convert("L").resize((x1 - x0, y1 - y0))
        full = np.zeros((h, w), bool)
        full[y0:y1, x0:x1] = np.asarray(crop) > 127
        masks.append(full)
    return masks


def read_reply(text: str, w: int, h: int) -> dict:
    """{"polygons": [...], "masks": [...], "how": str}; empty when nothing is readable."""
    blocks = re.findall(r"```(?:\w+)?\s*(.*?)```", text, re.S)
    candidates = list(reversed(blocks)) + [text]
    for chunk in candidates:
        try:
            obj = json.loads(chunk)
        except (ValueError, TypeError):
            obj = None
        if obj is not None:
            masks = _gemini_masks(obj, w, h)
            if masks:
                return {"polygons": [], "masks": masks, "how": "gemini masks"}
            polys = _polygons_from_json(obj)
            if polys:
                return {"polygons": polys, "masks": [], "how": "json"}
        pts = [(float(a), float(b)) for a, b in PAIR.findall(chunk)] or \
              [(float(a), float(b)) for a, b in XY.findall(chunk)]
        if len(pts) >= 3:
            return {"polygons": [pts], "masks": [], "how": "pairs in text"}
    return {"polygons": [], "masks": [], "how": "unreadable"}


def read_box_grid(text: str, w: int, h: int) -> dict:
    """Prompt E: {"box": {...}, "mask": [rows of 0/1]} -> a full-picture mask.

    Rows: every run of 0/1 characters of the longest common length (>= 4),
    in order; a grid that is not 20 x 20 is used at the size given. Box: the
    four named corners, else the first four numbers before the mask, x first
    (as asked)."""
    rows = re.findall(r"(?<![0-9])[01]{4,}(?![0-9])", text)
    if rows:
        n = max(set(map(len, rows)), key=lambda L: (sum(len(r) == L for r in rows), L))
        rows = [r for r in rows if len(r) == n]
    box = []
    for key in ("x_min", "y_min", "x_max", "y_max"):
        m = re.search(rf'["\']?{key}["\']?\s*[:=]\s*({NUM})', text)
        if m:
            box.append(float(m.group(1)))
    if len(box) != 4:
        head = text.split(rows[0])[0] if rows else text
        nums = [float(v) for v in re.findall(NUM, head)]
        box = nums[-4:] if len(nums) >= 4 else []
    if len(box) != 4 or len(rows) < 2:
        return {"polygons": [], "masks": [], "how": "unreadable"}
    x0, x1 = sorted((box[0], box[2]))
    y0, y1 = sorted((box[1], box[3]))
    grid = np.array([[c == "1" for c in r] for r in rows], np.uint8) * 255
    X0, Y0 = max(0, round(x0)), max(0, round(y0))
    X1, Y1 = min(w, round(x1)), min(h, round(y1))
    full = np.zeros((h, w), bool)
    if X1 > X0 and Y1 > Y0:
        cells = Image.fromarray(grid).resize((round(x1) - round(x0), round(y1) - round(y0)), Image.NEAREST)
        a = np.asarray(cells) > 127
        full[Y0:Y1, X0:X1] = a[Y0 - round(y0):Y1 - round(y0), X0 - round(x0):X1 - round(x0)]
    return {"polygons": [], "masks": [full], "how": f"box + {len(rows)}x{len(rows[0])} grid",
            "box": [x0, y0, x1, y1]}


def read_as(version: str, text: str, w: int, h: int) -> dict:
    kind = PROMPTS[version][1]
    if kind == "box-grid":
        return read_box_grid(text, w, h)
    read = read_reply(text, w, h)
    if kind == "poly-1000":  # as asked; not rescaled when the numbers say otherwise
        read = {**read, "polygons": [[(a * w / 1000, b * h / 1000) for a, b in poly] for poly in read["polygons"]]}
    if kind == "poly-px":  # unnamed pairs only: named x/y are never swapped
        read = axis_order(read, w, h)
    return read


def axis_order(read: dict, w: int, h: int) -> dict:
    """x first, as the prompt asks, unless that is impossible.

    Gemini 3.8 Flash wrote some outlines y first (its habit with points),
    without saying so. Rule, fixed from the numbers alone (no answer key, and
    not from whether the outline holds the dot): when the first numbers go
    beyond the picture's width while the second stay within it, the first
    are y. Otherwise x first. On a tall picture this cannot catch a y-first
    outline that stays in the top `w` pixels; such a case is shown as read.
    """
    if h <= w or not read["polygons"]:
        return read
    pts = [q for poly in read["polygons"] for q in poly]
    a, b = max(q[0] for q in pts), max(q[1] for q in pts)
    if a > w and b <= w:
        return {**read, "polygons": [[(y, x) for x, y in poly] for poly in read["polygons"]],
                "how": read["how"] + ", y first (numbers beyond the width)"}
    return read


def to_mask(read: dict, w: int, h: int) -> np.ndarray:
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    for poly in read["polygons"]:
        if len(poly) >= 3:
            d.polygon(poly, fill=255)
    mask = np.asarray(m) > 0
    for extra in read["masks"]:
        mask = mask | extra
    return mask


# --- pictures -----------------------------------------------------------------

COLOURS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
           (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
           (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195)]


def tile(photo: Image.Image, x: int, y: int, read: dict, mask: np.ndarray) -> Image.Image:
    img = photo.convert("RGBA")
    a = np.zeros((*mask.shape, 4), np.uint8)
    a[mask] = (0, 200, 255, 90)
    if read["masks"]:
        edge = mask & ~(np.roll(mask, 1, 0) & np.roll(mask, -1, 0) & np.roll(mask, 1, 1) & np.roll(mask, -1, 1))
        a[edge] = (0, 200, 255, 255)
    over = Image.fromarray(a, "RGBA")
    img = Image.alpha_composite(img, over)
    d = ImageDraw.Draw(img)
    lw = max(2, round(min(img.size) * 0.004))
    for poly in read["polygons"]:
        if len(poly) >= 2:
            d.line(poly + [poly[0]], fill=(0, 200, 255, 255), width=lw)
    if read.get("box"):
        d.rectangle(read["box"], outline=(255, 255, 255, 200), width=max(1, lw // 2))
    r = dot_radius(*img.size)
    d.ellipse((x - r - 2, y - r - 2, x + r + 2, y + r + 2), fill=(255, 255, 255, 255))
    d.ellipse((x - r, y - r, x + r, y + r), fill=(230, 0, 0, 255))
    return img.convert("RGB")


def sheet(tiles: list[Image.Image], n: int = GRID, width: int = 2400) -> Image.Image:
    w, h = tiles[0].size
    tw = width // n
    th = round(h * tw / w)
    out = Image.new("RGB", (tw * n, th * n), "white")
    for k, t in enumerate(tiles):
        out.paste(t.resize((tw - 6, th - 6)), ((k % n) * tw + 3, (k // n) * th + 3))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--image", default=str(ROOT / "images/butterfly_display_case.jpg"))
    ap.add_argument("--level", default="medium")
    ap.add_argument("--prompt", default="A", choices=sorted(PROMPTS), help="way of asking (see PROMPTS)")
    ap.add_argument("--only", type=int, nargs="*", help="dot numbers (0..15) to send; default all")
    ap.add_argument("--via-openrouter", metavar="SLUG/HOST",
                    help="call the same model through OpenRouter, pinned to one host, no fallback "
                         "(e.g. google/gemini-3.8-flash@google-ai-studio); prices stay the model's own")
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--reread", metavar="RUN.jsonl", help="no calls: read and draw a saved run again")
    args = ap.parse_args()
    if args.reread:
        return reread(args.reread, args.image)

    model = BY_KEY[args.model]
    if args.via_openrouter:
        # 2026-10-16: Google's key for segbench is at its monthly spending cap.
        slug, host = args.via_openrouter.split("@")
        model = replace(model, route=f"openrouter:{slug}",
                        extensions={"provider": {"order": [host], "allow_fallbacks": False}})
    photo = Image.open(args.image).convert("RGB")
    w, h = photo.size
    pts = grid_points(w, h)
    todo = args.only if args.only is not None else list(range(len(pts)))
    settings = IMAGE_SETTINGS.get(model.route.split(":")[0], {})
    print(f"{model.label} ({model.route}, billing {model.billing}), level {args.level}, settings {settings}")
    print(f"{Path(args.image).name}: {w} x {h}; {len(todo)} requests; dots {[pts[k] for k in todo]}")
    print(f"\nprompt {args.prompt}:\n" + prompt_text(args.prompt, *pts[todo[0]], w, h))
    if not args.go:
        print("\n(plan only; add --go to send)")
        return

    out_dir = ROOT / "results/probe-point-segment" / Path(args.image).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = out_dir / f"{model.key}-{args.prompt}-{stamp}"
    lock = threading.Lock()
    sem = threading.Semaphore(WORKERS.get(pool(model), 6))
    records: dict[int, dict] = {}

    def one(k: int) -> None:
        x, y = pts[k]
        with sem:
            rec = call(model, with_dot(photo, x, y), "image/jpeg", prompt_text(args.prompt, x, y, w, h),
                       level=args.level, **settings)
        rec.update({"dot": k, "x": x, "y": y, "image": Path(args.image).name, "prompt_version": args.prompt,
                    "prompt": prompt_text(args.prompt, x, y, w, h)})
        with lock:
            records[k] = rec
            with open(f"{base}.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"dot {k:2d} ({x},{y}): {rec.get('seconds')} s, ${rec.get('cost_usd')}, "
                  f"{'ERROR ' + rec['error'][:120] if rec.get('error') else (rec.get('usage') or {}).get('output_tokens')}")

    t0 = time.time()
    with ThreadPoolExecutor(len(todo)) as ex:
        list(ex.map(one, todo))
    wall = time.time() - t0

    report(model, records, photo, pts, todo, wall, base, args.image, args.level)


def report(model, records: dict, photo: Image.Image, pts, todo, wall, base, image: str, level: str) -> None:
    w, h = photo.size
    tiles, summary = [], []
    for k in todo:
        rec = records[k]
        x, y = pts[k]
        read = read_as(rec.get("prompt_version", "A"), rec.get("text", ""), w, h) if not rec.get("error") else {"polygons": [], "masks": [], "how": "error"}
        mask = to_mask(read, w, h)
        tiles.append(tile(photo, x, y, read, mask))
        u = rec.get("usage") or {}
        summary.append({"dot": k, "x": x, "y": y, "how": read["how"], "n_polygons": len(read["polygons"]),
                        "n_vertices": sum(len(p) for p in read["polygons"]), "area_px": int(mask.sum()),
                        "dot_inside": bool(mask[y, x]) if mask.any() else False,
                        "seconds": rec.get("seconds"), "cost_usd": rec.get("cost_usd"),
                        "input_tokens": u.get("input_tokens"), "output_tokens": u.get("output_tokens"),
                        "reasoning_tokens": u.get("reasoning_tokens"), "error": rec.get("error"),
                        "polygons": read["polygons"]})
        tiles[-1].save(f"{base}-dot{k:02d}.jpg", quality=88)
    if len(tiles) == GRID * GRID:
        sheet(tiles).save(f"{base}-sheet.jpg", quality=88)
        # Close-ups: a square around each dot, a third of the short side wide.
        half = min(w, h) // 3 // 2
        crops = []
        for t, k in zip(tiles, todo):
            x, y = pts[k]
            x0, y0 = min(max(0, x - half), w - 2 * half), min(max(0, y - half), h - 2 * half)
            crops.append(t.crop((x0, y0, x0 + 2 * half, y0 + 2 * half)))
        sheet(crops, width=1600).save(f"{base}-closeups.jpg", quality=88)
    costs = [s["cost_usd"] for s in summary if s["cost_usd"] is not None]
    secs = [s["seconds"] for s in summary if s["seconds"] is not None]
    total = {"model": model.key, "label": model.label,
             "prompt_version": records[todo[0]].get("prompt_version", "A"), "route": model.route, "billing": model.billing,
             "level": level, "image": Path(image).name, "size": [w, h], "requests": len(todo),
             "errors": sum(1 for s in summary if s["error"]), "unreadable": sum(1 for s in summary if s["how"] == "unreadable"),
             "cost_usd_total": sum(costs), "cost_usd_per_dot": sum(costs) / len(costs) if costs else None,
             "seconds_per_dot_mean": sum(secs) / len(secs) if secs else None, "wall_seconds": wall,
             "dots": summary}
    Path(f"{base}.json").write_text(json.dumps(total, indent=1))
    print(json.dumps({k: v for k, v in total.items() if k != "dots"}, indent=1))


def reread(jsonl: str, image: str) -> None:
    """Draw and sum up a saved run again (no calls), with the current reading rules."""
    recs = [json.loads(l) for l in open(jsonl)]
    photo = Image.open(image).convert("RGB")
    pts = grid_points(*photo.size)
    records = {r["dot"]: r for r in recs}
    model = replace(BY_KEY[recs[0]["model"]], route=recs[0]["route"])
    base = Path(jsonl).with_suffix("")
    old = Path(f"{base}.json")
    wall = json.loads(old.read_text()).get("wall_seconds") if old.exists() else None
    report(model, records, photo, pts, sorted(records), wall, base, image, recs[0]["level"])


if __name__ == "__main__":
    main()
