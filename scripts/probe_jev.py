"""What Jev 1.14's new image input and Box question can do, on synthetic pictures only.

uv run --with pillow --with requests scripts/probe_jev.py  ->  results/probe-jev-<time>.json

Same seven red ovals and four blue squares as scripts/check_boxes.py, drawn
at 1568x883 (Jev takes 256..1568 px per side and never resizes). Nothing here
touches a benchmark photo or the answer key. Questions asked:

1. one oval alone: is the Box accurate? (and its centre, as a point)
2. seven ovals, one Box "the red oval": which one does it pick, or a box around all?
3. a Choice none / one / several next to it
4. one Box per oval, each told apart in words (leftmost, ...): the docs' way
5. find, white out, ask again, until a Noul says no oval is left: a loop that
   gives several instances
6. something absent (a green triangle): the box it still returns, and the Noul
"""
import io
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import requests
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_boxes import OVALS, SQUARES  # noqa: E402
from segbench.box_read import iou  # noqa: E402

MODEL = "jev-1.14.0"
URL = "https://api.typesafe.ai/v1/systemone"
W, H = 1568, 883


def draw(ovals, w=W, h=H):
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    s = min(w, h)
    for fx, fy, r in SQUARES:
        d.rectangle((fx * w - r * s, fy * h - r * s, fx * w + r * s, fy * h + r * s), fill=(40, 70, 210))
    truth = []
    for fx, fy, hw, hh in ovals:
        box = (round((fx - hw) * w), round((fy - hh) * h), round((fx + hw) * w), round((fy + hh) * h))
        d.ellipse(box, fill=(215, 30, 30), outline=(150, 10, 10), width=max(1, s // 300))
        truth.append([box[0] * 1000 / w, box[1] * 1000 / h, box[2] * 1000 / w, box[3] * 1000 / h])
    return im, truth


def ask(im, questions):
    """One request: the picture in the state as `photo`; returns (answers, usage, seconds)."""
    b = io.BytesIO()
    im.save(b, "PNG")
    ref = str(uuid.uuid4())
    body = {"model": MODEL, "state": {"photo": ref},
            "assets": {ref: {"type": "image", "part": "photo"}}, "questions": questions}
    t = time.time()
    r = requests.post(URL, headers={"Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}",
                                    "TypeSafe-Danger-Zone-Images": "true"},
                      files={"request": (None, json.dumps(body), "application/json"),
                             "photo": ("photo.png", b.getvalue(), "image/png")}, timeout=120)
    dt = time.time() - t
    if r.status_code != 200:
        raise RuntimeError(f"{r.status_code}: {r.text[:500]}")
    j = r.json()
    return j["answers"], j.get("usage"), dt, j.get("model")


def as1000(a):
    b = a["box"]
    return [b["left"] * 1000, b["top"] * 1000, b["right"] * 1000, b["bottom"] * 1000]


def best(box, truth):
    return max((iou(box, t), i) for i, t in enumerate(truth))


def centre_inside(box, t):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return t[0] <= cx <= t[2] and t[1] <= cy <= t[3]


def main():
    log = {"model": MODEL, "size": [W, H], "probes": {}}

    # 1. one oval alone, each of the seven in turn
    rows = []
    for k, ov in enumerate(OVALS):
        im, truth = draw([ov])
        ans, use, dt, served = ask(im, {"b": {"type": "box", "instructions": "the red oval in `photo`"}})
        box = as1000(ans["b"])
        rows.append({"oval": k, "box": box, "truth": truth[0], "iou": round(iou(box, truth[0]), 3),
                     "centre_inside": centre_inside(box, truth[0]), "confidence": ans["b"]["confidence"],
                     "seconds": round(dt, 2), "usage": use})
        log["served_by"] = served
    log["probes"]["single"] = rows

    im, truth = draw(OVALS)
    log["truth"] = truth

    # 2 + 3 + 6. all ovals; one Box, a count Choice, presence Nouls, an absent target
    ans, use, dt, _ = ask(im, {
        "box": {"type": "box", "instructions": "the red oval in `photo`"},
        "count": {"type": "choice", "instructions": "How many red ovals are in `photo`?",
                  "criteria": {"none": "No red oval.", "one": "Exactly one red oval.",
                               "several": "Two or more red ovals."}},
        "has_oval": {"type": "noul", "instructions": "Is there a red oval in `photo`?"},
        "has_triangle": {"type": "noul", "instructions": "Is there a green triangle in `photo`?"},
        "triangle": {"type": "box", "instructions": "the green triangle in `photo`"},
    })
    box = as1000(ans["box"])
    log["probes"]["many"] = {"answers": ans, "box_1000": box, "best_iou": best(box, truth),
                             "triangle_1000": as1000(ans["triangle"]), "seconds": round(dt, 2), "usage": use}

    # 4. one Box per oval, told apart in words (all in one request)
    words = {"leftmost": "the leftmost red oval in `photo`",
             "rightmost": "the rightmost red oval in `photo`",
             "topmost": "the red oval nearest the top edge of `photo`",
             "smallest": "the smallest red oval in `photo`",
             "tall_thin": "the tall, thin red oval in `photo`",
             "wide_flat": "the wide, flat red oval in `photo`"}
    ans, use, dt, _ = ask(im, {k: {"type": "box", "instructions": v} for k, v in words.items()})
    log["probes"]["described"] = {k: {"box_1000": as1000(a), "best_iou": best(as1000(a), truth),
                                      "confidence": a["confidence"]} for k, a in ans.items()}
    log["probes"]["described_meta"] = {"seconds": round(dt, 2), "usage": use}

    # 5. find, white out, ask again
    work = im.copy()
    found, steps = [], []
    for _ in range(12):
        ans, use, dt, _ = ask(work, {
            "left": {"type": "noul", "instructions": "Is there a red oval in `photo`?"},
            "box": {"type": "box", "instructions": "a red oval in `photo`"}})
        box = as1000(ans["box"])
        steps.append({"p_left": ans["left"]["noul"] if "noul" in ans["left"] else ans["left"],
                      "box_1000": box, "best_iou": best(box, truth), "confidence": ans["box"]["confidence"],
                      "seconds": round(dt, 2), "usage": use})
        p = ans["left"].get("noul", ans["left"].get("probability"))
        if p is not None and p < 0.5:
            break
        found.append(box)
        pad = 0.01  # a little past the box, so no red rim is left behind
        x0, y0 = (box[0] / 1000 - pad) * W, (box[1] / 1000 - pad) * H
        x1, y1 = (box[2] / 1000 + pad) * W, (box[3] / 1000 + pad) * H
        ImageDraw.Draw(work).rectangle((x0, y0, x1, y1), fill="white")
    used, hits = set(), 0
    for t in truth:
        cand = [(iou(b, t), i) for i, b in enumerate(found) if i not in used]
        if cand and max(cand)[0] >= 0.5:
            used.add(max(cand)[1])
            hits += 1
    log["probes"]["loop"] = {"steps": steps, "found": len(found), "matched_iou50": hits, "of": len(truth)}

    out = Path("results") / f"probe-jev-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(log, indent=1))
    print(out)




# ---- second round (same day): two other ways to get several instances ----

def loop_marked(im, truth, rounds=12):
    """Find one, draw a thick green frame round it (nothing erased), ask for an unframed one."""
    work, found, steps = im.copy(), [], []
    for _ in range(rounds):
        ans, use, dt, _ = ask(work, {
            "left": {"type": "noul", "instructions": "Is there a red oval in `photo` that has no green frame drawn around it?"},
            "box": {"type": "box", "instructions": "a red oval in `photo` that has no green frame drawn around it"}})
        box = as1000(ans["box"])
        steps.append({"p_left": ans["left"]["noul"], "box_1000": box, "best_iou": best(box, truth),
                      "confidence": ans["box"]["confidence"], "seconds": round(dt, 2)})
        if ans["left"]["noul"] < 0.5:
            break
        found.append(box)
        ImageDraw.Draw(work).rectangle((box[0] * W / 1000, box[1] * H / 1000, box[2] * W / 1000, box[3] * H / 1000),
                                       outline=(0, 200, 0), width=6)
    return found, steps


def tiles(im, truth, x0=0.0, y0=0.0, x1=1.0, y1=1.0, depth=0, calls=None, max_depth=3):
    """Ask how many in this region: none -> stop; one -> Box; several -> split in four (10% overlap)."""
    calls = calls if calls is not None else []
    w, h = im.size
    crop = im.crop((round(x0 * w), round(y0 * h), round(x1 * w), round(y1 * h)))
    s = min(1568 / max(crop.size), 1.0) if max(crop.size) > 1568 else max(1.0, 280 / min(crop.size))
    crop = crop.resize((round(crop.size[0] * s), round(crop.size[1] * s)))
    ans, use, dt, _ = ask(crop, {
        "count": {"type": "choice", "instructions": "How many whole or partial red ovals are in `photo`?",
                  "criteria": {"none": "No red oval.", "one": "Exactly one red oval.",
                               "several": "Two or more red ovals."}},
        "box": {"type": "box", "instructions": "the red oval in `photo`"}})
    calls.append(dt)
    c = ans["count"]["choice"]
    if c == "none":
        return [], calls
    if c == "one" or depth == max_depth:
        b = ans["box"]["box"]
        return [[(x0 + b["left"] * (x1 - x0)) * 1000, (y0 + b["top"] * (y1 - y0)) * 1000,
                 (x0 + b["right"] * (x1 - x0)) * 1000, (y0 + b["bottom"] * (y1 - y0)) * 1000]], calls
    mx, my, o = (x0 + x1) / 2, (y0 + y1) / 2, 0.05 * (x1 - x0), 
    oy = 0.05 * (y1 - y0)
    out = []
    for a, b_, c_, d in [(x0, y0, mx + o, my + oy), (mx - o, y0, x1, my + oy),
                         (x0, my - oy, mx + o, y1), (mx - o, my - oy, x1, y1)]:
        found, _ = tiles(im, truth, a, b_, c_, d, depth + 1, calls, max_depth)
        out += found
    return out, calls


def merge(boxes, thr=0.3):
    """Two boxes that overlap a lot, or one inside the other, are one object cut by a tile edge: join them."""
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
                small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
                if small and inter / small > thr:
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del boxes[j]
                    changed = True
                    break
            if changed:
                break
    return boxes


def matched(found, truth, thr):
    used, hits = set(), 0
    for t in truth:
        cand = [(iou(b, t), i) for i, b in enumerate(found) if i not in used]
        if cand and max(cand)[0] >= thr:
            used.add(max(cand)[1])
            hits += 1
    return hits


def main2():
    im, truth = draw(OVALS)
    found, steps = loop_marked(im, truth)
    raw, calls = tiles(im, truth)
    merged = merge(raw)
    log = {"model": MODEL, "truth": truth,
           "marked_loop": {"steps": steps, "found": len(found),
                           "matched": {t: matched(found, truth, t) for t in (0.5, 0.75)}},
           "tiles": {"raw": raw, "merged": merged, "calls": len(calls), "seconds": round(sum(calls), 2),
                     "matched": {t: matched(merged, truth, t) for t in (0.5, 0.75)}}}
    out = Path("results") / f"probe-jev-round2-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(log, indent=1))
    print(out)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("found", "matched", "calls", "seconds")}
                      for k, v in log.items() if isinstance(v, dict)}, indent=1))
    for s in steps:
        print(" marked", s["p_left"], [round(x) for x in s["box_1000"]], round(s["best_iou"][0], 2), s["best_iou"][1])
    print(" tiles merged", [[round(x) for x in b] for b in merged])


def grid_points(im, truth, cols=8, rows=5):
    """One request: every cell of a grid is its own picture in the state, one Noul per cell.
    Joined yes-cells (touching sides) are one object; its point is the middle of its cells."""
    w, h = im.size
    state, assets, files, qs = {"cells": {}}, {}, {}, {}
    for r in range(rows):
        for c in range(cols):
            cell = im.crop((c * w // cols, r * h // rows, (c + 1) * w // cols, (r + 1) * h // rows))
            k = max(1.0, 280 / min(cell.size))
            cell = cell.resize((round(cell.size[0] * k), round(cell.size[1] * k)))
            b = io.BytesIO(); cell.save(b, "PNG")
            ref, part = str(uuid.uuid4()), f"r{r}c{c}"
            state["cells"][part] = ref
            assets[ref] = {"type": "image", "part": part}
            files[part] = (part + ".png", b.getvalue(), "image/png")
            qs[part] = {"type": "noul", "instructions": f"Does `cells.{part}` show a red oval, or part of one?"}
    body = {"model": MODEL, "state": state, "assets": assets, "questions": qs}
    t = time.time()
    resp = requests.post(URL, headers={"Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}",
                                       "TypeSafe-Danger-Zone-Images": "true"},
                         files={"request": (None, json.dumps(body), "application/json"), **files}, timeout=300)
    dt = time.time() - t
    if resp.status_code != 200:
        raise RuntimeError(f"{resp.status_code}: {resp.text[:500]}")
    j = resp.json()
    yes = {(r, c) for r in range(rows) for c in range(cols) if j["answers"][f"r{r}c{c}"]["noul"] >= 0.5}
    truth_cells = set()
    for t_ in truth:
        for r in range(rows):
            for c in range(cols):
                if t_[0] < (c + 1) * 1000 / cols and t_[2] > c * 1000 / cols and t_[1] < (r + 1) * 1000 / rows and t_[3] > r * 1000 / rows:
                    truth_cells.add((r, c))
    comps, seen = [], set()
    for cell in sorted(yes):
        if cell in seen:
            continue
        stack, comp = [cell], []
        seen.add(cell)
        while stack:
            r, c = stack.pop(); comp.append((r, c))
            for n in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if n in yes and n not in seen:
                    seen.add(n); stack.append(n)
        comps.append(comp)
    pts = [(sum(c + 0.5 for _, c in comp) / len(comp) * 1000 / cols, sum(r + 0.5 for r, _ in comp) / len(comp) * 1000 / rows) for comp in comps]
    used, hits = set(), 0
    for t_ in truth:
        for i, (x, y) in enumerate(pts):
            if i not in used and t_[0] <= x <= t_[2] and t_[1] <= y <= t_[3]:
                used.add(i); hits += 1; break
    return {"yes_cells": sorted(yes), "cells_with_oval": sorted(truth_cells),
            "cell_errors": len(yes ^ truth_cells), "objects": len(comps), "points": pts,
            "points_on_an_oval": hits, "of": len(truth), "seconds": round(dt, 2), "usage": j.get("usage")}


def main3():
    im, truth = draw(OVALS)
    res = grid_points(im, truth)
    out = Path("results") / f"probe-jev-grid-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(res, indent=1))
    print(out)
    print({k: v for k, v in res.items() if k not in ("yes_cells", "cells_with_oval", "points")})


if __name__ == "__main__":
    {"--round2": main2, "--grid": main3}.get(sys.argv[1] if len(sys.argv) > 1 else "", main)()
