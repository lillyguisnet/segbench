"""Jev 1.14 finding several objects by halving the picture (Maxime, 2026-10-16).

uv run --with pillow --with numpy scripts/probe_jev_split.py
    ->  results/probe-jev-split-<time>.json (the search) and .jsonl (every call's record)

Every request goes through segbench.call.call (lm15 >= 1.5.0, model "jev" of
segbench/models.py), so each one is an ordinary call record: tokens, seconds,
cost from prices.json, errors. The first runs (results/probe-jev-split-
20261015-*.json) called TypeSafe's API directly, with the picture as a state
field named `photo`; through lm15 the picture is the state itself, so the
questions say "the image" instead.

Synthetic pictures only (the seven red ovals and four blue squares of
scripts/check_boxes.py, at the four benchmark sizes); no benchmark photo, no
answer key.

The search, for each piece of the picture (the first piece is the whole):
  ask a Choice none / one / several (ovals cut off by the edge count)
  - none     -> stop
  - one      -> ask a Box in that piece; that is one object
  - several  -> cut the piece in two and do the same for each half

Choices made here (each one a trade-off, said to Maxime):
- Cut across the longer side, so pieces stay roughly square (a thin strip
  would be stretched by Jev's 256 px minimum).
- Each piece is cut from the full-size picture, then shrunk to Jev's 1568 px
  limit if needed: halving is also zooming in, which is the point.
- Halves do not overlap. An oval on a cut line is seen in both halves and
  boxed twice, in part; two boxes are joined when both touch the same cut
  line and overlap along it. (Overlapping halves were the other choice: no
  object is cut, but every object near a line is counted twice and must be
  told apart by overlap alone, which fails for touching objects.)
- The Box is asked in the same request as the Choice (they run in parallel
  and cost a few tokens), and used only when the answer is "one".
- At MAX_DEPTH a "several" is boxed anyway and flagged.
- All pieces of one depth are asked at the same time (threads): wall seconds are the
  real waiting time; request seconds are their sum.

The truth used to grade (which ovals are in which piece) is computed from
the drawing itself: synthetic, so it is not an answer key.
"""
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_boxes import OVALS, SIZES, SQUARES  # noqa: E402
from lm15 import bounding_box, choice, judgments  # noqa: E402
from segbench.box_read import iou  # noqa: E402
from segbench.call import call  # noqa: E402
from segbench.models import BY_KEY  # noqa: E402

MAX_DEPTH = 7
POOL = ThreadPoolExecutor(16)

JEV = BY_KEY["jev"]
COUNT_TEXT = ("How many red ovals are in the image? Count an oval that is cut off by the edge of the image too. "
              "The blue squares are not ovals.")
COUNT_OPTIONS = {"none": "No red oval, not even part of one.",
                 "one": "Exactly one red oval, whole or cut off by the edge.",
                 "several": "Two or more red ovals, whole or cut off by the edge."}
BOX_TEXT = "the red oval in the image (only its visible part)"
QUESTIONS = judgments(count=choice(COUNT_TEXT, COUNT_OPTIONS), box=bounding_box(BOX_TEXT))
RECORDS: list = []  # every call record, in the order they came back


def ask_jev(piece):
    """One request: the piece is the state; returns the call record (three tries if retryable)."""
    b = io.BytesIO()
    piece.save(b, "PNG")
    for attempt in range(3):
        rec = call(JEV, b.getvalue(), "image/png", None, level="medium", response_format=QUESTIONS)
        rec["sent_size"] = list(piece.size)
        rec["attempt"] = attempt
        RECORDS.append(rec)
        if not rec.get("error"):
            return rec
        if not rec.get("retryable"):
            break
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Jev call failed on a {piece.size} piece: {rec['error']}")


def draw(w, h):
    """The picture, each oval's box (pixels) and each oval's mask (to know which pieces it is in)."""
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    s = min(w, h)
    for fx, fy, r in SQUARES:
        d.rectangle((fx * w - r * s, fy * h - r * s, fx * w + r * s, fy * h + r * s), fill=(40, 70, 210))
    boxes, masks = [], []
    for fx, fy, hw, hh in OVALS:
        b = (round((fx - hw) * w), round((fy - hh) * h), round((fx + hw) * w), round((fy + hh) * h))
        d.ellipse(b, fill=(215, 30, 30), outline=(150, 10, 10), width=max(1, s // 300))
        m = Image.new("L", (w, h), 0)
        ImageDraw.Draw(m).ellipse(b, fill=255)
        boxes.append(list(b))
        masks.append(np.asarray(m) > 0)
    return im, boxes, masks


def fit(crop):
    """Into Jev's limits, keeping the shape: shrink so the long side is <= 1568, enlarge so the
    short side is >= 280; a piece too thin for both is padded with grey bars (letterbox).
    The docs say sides 256..1568; the API rejects sides under 280 (tried 2026-10-16:
    279x500 and 457x256 rejected, 280x500 accepted).
    Returns the picture sent and (pad_x, pad_y, inner_w, inner_h) to map its answers back."""
    w, h = crop.size
    k = min(1.0, 1568 / max(w, h))
    if min(w, h) * k < 280:
        k = min(280 / min(w, h), 1568 / max(w, h))
    iw, ih = max(1, round(w * k)), max(1, round(h * k))
    if (iw, ih) != (w, h):
        crop = crop.resize((iw, ih), Image.LANCZOS)
    W_, H_ = max(280, iw), max(280, ih)
    if (W_, H_) == (iw, ih):
        return crop, (0, 0, iw, ih)
    canvas = Image.new("RGB", (W_, H_), (128, 128, 128))
    px, py = (W_ - iw) // 2, (H_ - ih) // 2
    canvas.paste(crop, (px, py))
    return canvas, (px, py, iw, ih)


def unpad(b, sent, pad):
    """A Box answer on the sent picture -> fractions of the piece itself (clamped to it)."""
    px, py, iw, ih = pad
    W_, H_ = sent.size
    cl = lambda v: min(1.0, max(0.0, v))  # noqa: E731
    return {"left": cl((b["left"] * W_ - px) / iw), "right": cl((b["right"] * W_ - px) / iw),
            "top": cl((b["top"] * H_ - py) / ih), "bottom": cl((b["bottom"] * H_ - py) / ih)}


def ask_piece(im, masks, region, depth):
    x0, y0, x1, y1 = region
    piece, pad = fit(im.crop(region))
    rec = ask_jev(piece)
    data = rec["data"]
    box = unpad(data["box"], piece, pad)
    box_conf = rec["response"]["provider_data"]["typesafe"]["answers"]["box"].get("confidence")
    inside = sum(int(m[y0:y1, x0:x1].any()) for m in masks)
    node = {"depth": depth, "region": region, "sent_size": piece.size, "choice": data["count"],
            "p": rec["probabilities"]["count"], "truth": inside,
            "truth_word": "none" if inside == 0 else "one" if inside == 1 else "several",
            "seconds": rec["seconds"], "tokens": rec["usage"]["input_tokens"], "cost_usd": rec["cost_usd"]}
    return {"box": {"box": box, "confidence": box_conf}}, node


def search(im, masks, region, nodes, cuts):
    """Level by level: every piece of one depth is asked at the same time.
    region = (x0, y0, x1, y1) in pixels of the full picture. Returns found boxes (pixels)."""
    found, level, depth = [], [region], 0
    while level:
        results = list(POOL.map(lambda r: ask_piece(im, masks, r, depth), level))
        nxt = []
        for r, (ans, node) in zip(level, results):
            nodes.append(node)
            x0, y0, x1, y1 = r
            choice = node["choice"]
            if choice == "none":
                continue
            if choice == "one" or depth == MAX_DEPTH:
                b = ans["box"]["box"]
                w, h = x1 - x0, y1 - y0
                box = [x0 + b["left"] * w, y0 + b["top"] * h, x0 + b["right"] * w, y0 + b["bottom"] * h]
                node.update(box=box, box_confidence=ans["box"]["confidence"], forced=choice == "several")
                found.append({"box": box, "region": r})
                continue
            if x1 - x0 >= y1 - y0:
                m = (x0 + x1) // 2
                nxt += [(x0, y0, m, y1), (m, y0, x1, y1)]
                cuts.append(("x", m, y0, y1))
            else:
                m = (y0 + y1) // 2
                nxt += [(x0, y0, x1, m), (x0, m, x1, y1)]
                cuts.append(("y", m, x0, x1))
        print(f"  depth {depth}: {len(level)} pieces, "
              f"{sum(1 for _, n in results if n['choice'] == 'several')} several", flush=True)
        level, depth = nxt, depth + 1
    return found


def join_across_cuts(found, cuts, min_along=0.5, with_parts=False):
    """Parts of one oval cut by the halving are joined back into one box.

    Two raw boxes are linked when their pieces share a cut line, each box
    touches that line (within 2% of its piece: Jev's edges come in 1% steps),
    and their spans along the line agree (overlap >= min_along of the shorter
    span). Linked boxes are grouped (an oval cut into four parts is joined
    through its neighbours) and each group becomes the rectangle around it.
    Links are made between the raw parts, never between already joined boxes,
    so the order does not matter. Two different ovals that touch on a cut line
    with similar spans are joined wrongly: a known limit."""
    n = len(found)
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            A, B = found[i], found[j]
            ra, rb, a, b = A["region"], B["region"], A["box"], B["box"]
            for k in (0, 1):  # k = 0: a vertical cut line at x = pos (a on its left); 1: horizontal
                pos, o = ra[k + 2], 1 - k
                if rb[k] != pos or min(ra[o + 2], rb[o + 2]) <= max(ra[o], rb[o]):
                    continue
                if a[k + 2] < pos - 0.02 * (ra[k + 2] - ra[k]) or b[k] > pos + 0.02 * (rb[k + 2] - rb[k]):
                    continue
                along = min(a[o + 2], b[o + 2]) - max(a[o], b[o])
                shorter = min(a[o + 2] - a[o], b[o + 2] - b[o])
                if shorter > 0 and along / shorter >= min_along:
                    parent[root(i)] = root(j)
    groups = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(found[i]["box"])
    joined = [([min(x[0] for x in g), min(x[1] for x in g), max(x[2] for x in g), max(x[3] for x in g)], len(g))
              for g in groups.values()]
    return joined if with_parts else [b for b, _ in joined]


def recheck_joined(im, masks, joined, nodes, rounds=2):
    """A joined box is asked none / one / several on its own. "several" means two touching
    ovals were joined: halve that box again (its cut lines fall elsewhere) and join again.
    Up to `rounds` times; the last answer is kept."""
    w, h = im.size
    out = []
    for box, parts in joined:
        if parts == 1:
            out.append(box)
            continue
        region = (max(0, int(box[0])), max(0, int(box[1])), min(w, int(box[2]) + 1), min(h, int(box[3]) + 1))
        ans, node = ask_piece(im, masks, region, -1)
        node["recheck"] = True
        nodes.append(node)
        if node["choice"] != "several" or rounds == 0:
            out.append(box)
            continue
        sub = search(im, masks, region, nodes, [])
        out += recheck_joined(im, masks, join_across_cuts(sub, [], with_parts=True), nodes, rounds - 1)
    return out


def grade(found, truth, masks):
    out = {"boxes": len(found), "ovals": len(truth)}
    for thr in (0.5, 0.75, 0.9):
        used, hits = set(), 0
        for t in truth:
            cand = [(iou(b, t), i) for i, b in enumerate(found) if i not in used]
            if cand and max(cand)[0] >= thr:
                used.add(max(cand)[1])
                hits += 1
        out[f"iou{int(thr * 100)}"] = hits
    # as points: the middle of each box, right if on an oval no earlier point has claimed
    used, hits = set(), 0
    for b in found:
        x, y = int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)
        for i, m in enumerate(masks):
            if i not in used and 0 <= y < m.shape[0] and 0 <= x < m.shape[1] and m[y, x]:
                used.add(i)
                hits += 1
                break
    out["points_on_oval"] = hits
    p, r = hits / max(1, len(found)), hits / len(truth)
    out["point_f1"] = round(2 * p * r / (p + r), 3) if hits else 0.0
    return out


def main():
    log = {"model": JEV.route, "max_depth": MAX_DEPTH, "questions": QUESTIONS, "runs": {}}
    for name, (w, h) in SIZES.items():
        im, truth, masks = draw(w, h)
        nodes, cuts = [], []
        t = time.time()
        raw = search(im, masks, (0, 0, w, h), nodes, cuts)
        wall = time.time() - t
        first = join_across_cuts(raw, cuts)
        g_first = grade(first, truth, masks)
        found = recheck_joined(im, masks, join_across_cuts(raw, cuts, with_parts=True), nodes)
        tokens = sum(n["tokens"] for n in nodes)
        dollars = sum(n["cost_usd"] or 0 for n in nodes)
        g = grade(found, truth, masks)
        wrong = [n for n in nodes if n["choice"] != n["truth_word"]]  # recheck pieces included
        log["runs"][name] = {"size": [w, h], "truth": truth, "found": found, "raw": [f["box"] for f in raw],
                             "cuts": cuts, "nodes": nodes, "grade": g, "grade_before_recheck": g_first, "requests": len(nodes),
                             "wall_seconds": round(wall, 2),
                             "request_seconds": round(sum(n["seconds"] for n in nodes), 2),
                             "tokens": tokens, "dollars": round(dollars, 5),
                             "count_wrong": len(wrong)}
        print(f"{name:11s} before recheck: {g_first}")
        print(f"{name:11s} {w}x{h}: {g} | {len(nodes)} requests, {wall:.1f}s wall, "
              f"{tokens} tokens (${dollars:.4f}); counts wrong in {len(wrong)}/{len(nodes)} pieces")
        for n in wrong:
            print(f"    depth {n['depth']} {n['region']}: said {n['choice']} "
                  f"({max(n['p'].values()):.2f}), truth {n['truth']}")
    out = Path("results") / f"probe-jev-split-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(log, indent=1))
    out.with_suffix(".jsonl").write_text("".join(json.dumps(r) + "\n" for r in RECORDS))
    print(out)


if __name__ == "__main__":
    main()
