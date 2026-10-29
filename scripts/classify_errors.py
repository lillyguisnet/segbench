"""Sort every mistake of the photo benchmarks into a kind: points (point v3) and boxes (box v1).

    uv run --with numpy scripts/classify_errors.py  ->  results/error-kinds/v1/

Diagnostic only: no model calls, no score changes. Reads the records behind
the charts (results/point-benchmark-v3/scored.jsonl, results/box-benchmark/v1/
scored.jsonl), the raw replies they came from, the point key, the reviewed box
key, and two measuring aids made for this (results/error-kinds/support/, not
reviewed by a person): SAM 2.1 outlines of the key objects from their reviewed
boxes (specialists/sam/key_masks.py) and SAM 3 outlines of look-alikes
(scripts/find_lookalikes.py). Method and reasons: docs/error-kinds.md.

Every wrong point or box, and every object nobody found, gets one kind: the
first rule that fits, in the order below.

Whole reply (explains all its mistakes)
  no answer: stuck        no points or boxes; the thinking or the answer
                          repeats itself, or the reply ran out of room
  no answer: unreadable   coordinates in the reply, our reader got none
  no answer: empty        a reply with nothing to place ("none", an empty
                          list, a sentence without coordinates)
  coordinate mix-up       the reply scores far better read another way:
                          x and y swapped, or another scale (0-1, 0-100, pixels)
  shifted or stretched    the reply scores far better after one shift and
                          one stretch per axis: the right layout, misplaced
                          (6+ answers; four numbers are fitted)
  The stretch only for replies with most answers off every object outline
  (box centres for boxes); "far better": F1 at the widest ring (points) or 50%
  (boxes) up by 0.25 or more and at least doubled.
Each point or box that earned no credit
  doom loop               part of a mechanical pattern: one answer repeated
                          5+ times, or 12+ in a row whose steps repeat
                          (a ruler or a grid laid on the photo)
  off image               outside the photo
  group box  (boxes)      one box around 2+ key objects (half of each inside)
  on the object, off-centre (points)  on the outline of an object nobody
                          found, outside the scorer's ring around its dot
  mispositioned box (boxes)  overlaps (IoU >= 0.1) an object nobody found,
                          below 50%: "part" (box inside the object), "too big"
                          (object inside the box) or "off" (neither)
  duplicate               on an object already found (points: on its outline;
                          boxes: overlapping it by 10%+)
  wrong object            on another thing: a look-alike outlined by SAM 3
                          (the black cows, the butterflies, other leaves...),
                          or a spot where 3+ other models also made a mistake
                          that is on no key object ("shared": something there
                          fools models; found in a first pass over all replies;
                          within 2% of the photo diagonal for points, overlap
                          50%+ for boxes)
  near miss  (points)     off every outline but within a quarter of an
                          unfound object's size of its box
  in a gap  (points)      off every outline, as close to an object already
                          found: an extra point in a crowd (between logs)
  maybe missing from key  (points) on an object of the reviewed box key that
                          the point key lacks (one fig leaf drawn by hand in
                          the box review)
  background              on nothing we know of
Objects
  missed                  nobody found it (or the reply-level kind, or the
                          group box that swallowed it)

"Claims" are one-to-one: one off-centre point (or mispositioned box) per
unfound object, closest first; further ones on it are duplicates.

Score each kind costs (TIDE, Bolya et al. 2020, single class): the model's
score, as charted, if only that kind were fixed: wrong points removed, missed
objects found, off-centre points and mispositioned boxes made exact, a group
box split into its objects, a mix-up or shift read the better way. Also
"partial credit" (points matched only at the wider rings) and "loose fit"
(boxes matched at 50% but not at stricter levels): fixing them gives full
credit at every level. Fixes interact; the gains do not add up.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from segbench.box_read import iou  # noqa: E402
from segbench.point_score import maximum_matching, score as point_score  # noqa: E402

POINTS_DIR = ROOT / "results/point-benchmark-v3"   # the published photo answers: photos, and the crowd for "shared"
CLASSIFY_DIR = POINTS_DIR                          # the point answers sorted (--points-dir)
BOXES_DIR = ROOT / "results/box-benchmark/v1"
BOX_KEY = ROOT / "annotations/boxes/answer-key-boxes-v1.json"
SUPPORT = ROOT / "results/error-kinds/support"
LEVELS = (0.50, 0.75, 0.90, 0.95)
LOOP_RUN, SAME = 12, 5
NEAR = 0.25          # near miss: within NEAR x sqrt(box area) of the box
# Answer keys checked over the whole photo by a person although the review file
# says otherwise (its full_photo_checked flag was never set). The SAM 3
# "cut log end" outlines outside the key are then things that look like log
# ends but were not accepted: look-alikes, not gaps in the key.
CHECKED_BY_HAND = {"logs": "Maxime, 2026-10-19"}
SKIP_BOXES = False
SHARED = 3          # wrong object: this many other models made a mistake on the same spot
FIRST_PASS = {}     # (benchmark, task) -> [(entrant, x, y, box)] of off-object mistakes, filled by a first pass
MIX_GAIN = 0.25      # a better reading counts when F1 (loose / 50%) rises by this much and at least doubles
NO_ANSWER = ("no answer: stuck", "no answer: unreadable", "no answer: empty")
REPLY_KINDS = NO_ANSWER + ("coordinate mix-up", "shifted or stretched")
POINT_KINDS = REPLY_KINDS + ("doom loop", "off image", "on the object, off-centre", "duplicate", "wrong object",
                             "near miss", "in a gap", "maybe missing from key", "background", "missed")
BOX_KINDS = REPLY_KINDS + ("doom loop", "off image", "group box", "mispositioned box", "duplicate", "wrong object",
                           "maybe missing from key", "background", "missed")


# ------------------------------------------------------------------ photo data

def rle_decode(m):
    h, w = m["size"]
    flat = np.zeros(h * w, dtype=bool)
    pos, val = 0, False
    for c in m["counts"]:
        if val:
            flat[pos:pos + c] = True
        pos += c
        val = not val
    return flat.reshape(w, h).T


def paint(shape, masks):
    """Label image: index of the mask covering each pixel (-1: none); smaller masks on top."""
    lab = np.full(shape, -1, dtype=np.int16)
    for i in sorted(range(len(masks)), key=lambda i: -int(masks[i].sum())):
        lab[masks[i]] = i
    return lab


def load_photos():
    """Per task: the point key's dots (with the scorer's rings), the box key, outlines, look-alikes."""
    man = json.loads((POINTS_DIR / "manifest.json").read_text())
    tasks = {}
    for name in man["references"]:
        raw = json.loads((ROOT / "annotations/reviews" / name).read_text())
        for s in raw["source_snapshot"]["tasks"]:
            if s["key"] in man["tasks"]:
                t = raw["tasks"][s["key"]]
                tasks[s["key"]] = {"W": t["width"], "H": t["height"], "photo": s["photo"],
                                   "dots": [o for o in t["objects"] if o["status"] == "accepted"],
                                   "checked": t.get("full_photo_checked")}
    boxkey = json.loads(BOX_KEY.read_text())["tasks"]
    npz = np.load(SUPPORT / "key-masks-sam2.1.npz")
    look = json.loads((SUPPORT / "lookalikes-sam3.json").read_text())
    for k, T in tasks.items():
        W, H = T["W"], T["H"]
        boxes = {o["id"]: o["box"] for o in boxkey[k]["objects"] if o["status"] == "accepted"}
        ids = [o["id"] for o in T["dots"]] + [i for i in boxes if i not in {o["id"] for o in T["dots"]}]
        dot = {o["id"]: (o["x"], o["y"]) for o in T["dots"]}
        T["objects"] = [{"id": i, "dot": dot.get(i), "box": boxes.get(i), "in_point_key": i in dot,
                         "in_box_key": i in boxes} for i in ids]
        masks = []
        for o in T["objects"]:
            arr = npz.get(f"{k}/{o['id']}") if f"{k}/{o['id']}" in npz.files else None
            masks.append(np.unpackbits(arr)[:W * H].reshape(H, W).astype(bool) if arr is not None
                         else np.zeros((H, W), bool))
        T["keylab"] = paint((H, W), masks)
        T["mask_area"] = [int(m.sum()) for m in masks]
        del masks
        keyboxes = [o["box"] for o in T["objects"] if o["box"]]
        T["look"], T["gap"] = [], []
        lmasks, gmasks = [], []
        for rec in look["records"]:
            if rec["task"] != k:
                continue
            for obj in rec["objects"]:
                m = rle_decode(obj["mask"])
                area = int(m.sum())
                if not area:
                    continue
                inside = max((m[int(b[1]):int(math.ceil(b[3])) + 1, int(b[0]):int(math.ceil(b[2])) + 1].sum() / area
                              for b in keyboxes), default=0)
                if inside >= 0.5:  # mostly inside one key box: that object (or part of it), not another thing
                    continue
                as_look = rec["role"] == "lookalike" or k in CHECKED_BY_HAND
                (T["look"] if as_look else T["gap"]).append({"query": rec["query"], "box": obj["box"]})
                (lmasks if as_look else gmasks).append(m)
        T["looklab"] = paint((H, W), lmasks)
        T["gaplab"] = paint((H, W), gmasks)
        tasks[k] = T
    return tasks


def at(lab, x, y):
    H, W = lab.shape
    xi, yi = int(x), int(y)
    return int(lab[yi, xi]) if 0 <= xi < W and 0 <= yi < H else -1


def box_dist(x, y, b):
    return math.hypot(max(b[0] - x, 0, x - b[2]), max(b[1] - y, 0, y - b[3]))


def base_model(entrant):
    """The model behind an entrant, whatever its thinking level or route (muse-wonder-api@medium -> muse-wonder)."""
    m = entrant.split("@")[0]
    return m[:-4] if m.endswith("-api") else m


def shared(bench, task, entrant, x, y, box, T):
    cap = 0.02 * math.hypot(T["W"], T["H"])
    # other models only: the same model at another thinking level or by another route does not count
    others = {e for e, xx, yy, bb in FIRST_PASS.get((bench, task), ()) if base_model(e) != base_model(entrant) and
              (iou(box, bb) >= 0.5 if box else math.hypot(x - xx, y - yy) <= cap)}
    return len(others)


# ------------------------------------------------------------------ loops

def _lines(text):
    return [x.strip().lower() for x in re.split(r"[\n.!?]+|\}\s*,\s*\{|</point>|</point_box>", text) if len(x.strip()) > 3]


def text_loops(text, limit):
    """As scripts/mk15_loops.py: a line written over and over, a pair 20+ times, circles."""
    why = []
    ls = _lines(text)
    if ls:
        n = collections.Counter(re.sub(r"-?\d+", "#", x) for x in ls).most_common(1)[0][1]
        if n > limit:
            why.append(f"{n}x the same line")
    pairs = re.findall(r"[\[(]\s*(-?\d+)\s*,\s*(-?\d+)\s*[\])]", text)
    if pairs and collections.Counter(pairs).most_common(1)[0][1] >= 20:
        why.append("one pair 20+ times")
    plain = [x for x in ls if not re.search(r"\d", x)]
    if len(plain) >= 100 and len(set(plain)) / len(plain) < 0.4:
        why.append("going round in circles")
    return why


def mechanical(items):
    """Indices in a mechanical pattern: the same item SAME+ times (all but the first),
    or LOOP_RUN+ items in a row whose steps repeat with a period of 1 to 6."""
    pts = [tuple(round(v) for v in p) for p in items]
    hit, why = set(), []
    seen = collections.defaultdict(list)
    for i, p in enumerate(pts):
        seen[p].append(i)
    for p, idx in seen.items():
        if len(idx) >= SAME:
            hit.update(idx[1:])
            why.append(f"{len(idx)}x the same")
    steps = [tuple(b - a for a, b in zip(p, q)) for p, q in zip(pts, pts[1:])]
    best = 0
    for period in range(1, 7):
        start = None
        for i in range(period, len(steps) + 1):
            same = i < len(steps) and any(steps[i]) and all(abs(u - v) <= 1 for u, v in zip(steps[i], steps[i - period]))
            if same and start is None:
                start = i - period
            if not same and start is not None:
                if i - start + 1 >= LOOP_RUN:
                    hit.update(range(start, i + 1))
                    best = max(best, i - start + 1)
                start = None
    if best:
        why.append(f"{best} in a repeating step pattern")
    return hit, why


def reply_texts(raw):
    parts = ((raw.get("response") or {}).get("message") or {}).get("parts") or []
    thinking = "".join(p.get("text", "") for p in parts if p.get("type") == "thinking")
    return raw.get("text") or "", thinking


def no_answer_kind(raw, readable, n_objects):
    answer, thinking = reply_texts(raw)
    limit = max(20, 3 * n_objects)
    if text_loops(answer, limit):
        return "no answer: stuck"
    if not readable and re.search(r"\d+\s*,\s*\d+", answer):
        return "no answer: unreadable"
    if text_loops(thinking, limit) or raw.get("finish_reason") == "length":
        return "no answer: stuck"
    return "no answer: unreadable" if not readable and answer.strip() else "no answer: empty"


# ------------------------------------------------------------------ other readings

ALTERNATIVES = ("swapped", "0-1", "0-100", "pixels", "swapped pixels")


def reread(xy, how, W, H):
    """xy on the 0..1000 grid as read; returns it on the grid under another reading."""
    x, y = xy
    if how == "swapped":
        return y, x
    if how == "0-1":
        return x * 1000, y * 1000
    if how == "0-100":
        return x * 10, y * 10
    if how == "pixels":
        return x * 1000 / W, y * 1000 / H
    return y * 1000 / W, x * 1000 / H  # swapped pixels


def fits(how, top):
    return not ((how == "0-1" and top > 1) or (how == "0-100" and top > 100))


def fit_affine(src, dst_all):
    """Per-axis stretch and shift (x' = ax + b, y' = cy + d) that pulls src onto its
    nearest dst, iterated (ICP), stretch limited to 0.6-1.6. Returns (a, b, c, d)."""
    a, b, c, d = 1.0, 0.0, 1.0, 0.0
    src, dst_all = np.asarray(src, float), np.asarray(dst_all, float)
    for _ in range(20):
        q = src * [a, c] + [b, d]
        dist = np.hypot(q[:, None, 0] - dst_all[None, :, 0], q[:, None, 1] - dst_all[None, :, 1])
        nn = dist.argmin(1)
        keep = dist[np.arange(len(q)), nn] <= 2 * np.median(dist[np.arange(len(q)), nn]) + 1e-9
        if keep.sum() < 3:
            break
        new = []
        for ax in (0, 1):
            s, t = src[keep, ax], dst_all[nn[keep], ax]
            if s.std() > 1e-6:
                k = float(np.clip(np.cov(s, t, bias=True)[0, 1] / s.var(), 0.6, 1.6))
            else:
                k = 1.0
            new += [k, float(t.mean() - k * s.mean())]
        if np.allclose(new, [a, b, c, d], atol=1e-3):
            break
        a, b, c, d = new
    return a, b, c, d


# ------------------------------------------------------------------ points

def score_points(points, T):
    refs = [{"id": o["id"], "x": o["dot"][0], "y": o["dot"][1]} for o in T["objects"] if o["in_point_key"]]
    return point_score(points, refs, T["W"], T["H"])


def classify_point_reply(rec, raw, T, entrant=None):
    W, H = T["W"], T["H"]
    objs = T["objects"]
    pkey = [j for j, o in enumerate(objs) if o["in_point_key"]]           # scorer's reference order
    points = rec.get("predictions") or []
    n_ref = len(pkey)
    s = score_points(points, T) if points else None
    base = rec["score"]
    if s:
        assert abs(s["score"] - base) < 1e-9, (rec["entrant"], rec["task"], s["score"], base)
    reply_kind, better, note = None, None, ""
    if raw.get("error"):
        reply_kind = "no answer: unreadable"
    elif not points:
        reply_kind = no_answer_kind(raw, rec.get("readable"), n_ref)
    loop_idx, loop_why = mechanical([p["point"] for p in points])
    px = [(p["point"][0] * W / 1000, p["point"][1] * H / 1000) for p in points]
    on_share = sum(at(T["keylab"], *q) >= 0 for q in px) / len(px) if px else 1
    if points and not loop_idx:
        loose = s["tolerances"]["loose"]["f1"]
        top = max(max(abs(v) for v in p["point"]) for p in points)
        cands = []
        for how in ALTERNATIVES:
            if fits(how, top):
                alt = [{"point": list(reread(p["point"], how, W, H))} for p in points]
                cands.append(("coordinate mix-up", how, score_points(alt, T)))
        if len(points) >= 6 and on_share < 0.5:  # four numbers fitted: only for answers mostly off every object
            src = [(p["point"][0] * W / 1000, p["point"][1] * H / 1000) for p in points]
            dots = [objs[j]["dot"] for j in pkey]
            a, b, c, d = fit_affine(src, dots)
            alt = [{"point": [(x * a + b) * 1000 / W, (y * c + d) * 1000 / H]} for x, y in src]
            cands.append(("shifted or stretched", f"x*{a:.2f}{b:+.0f}px, y*{c:.2f}{d:+.0f}px", score_points(alt, T)))
        for kind, how, alt in cands:
            f = alt["tolerances"]["loose"]["f1"]
            if f >= loose + MIX_GAIN and f >= 2 * loose and (better is None or f > better[2]["tolerances"]["loose"]["f1"]):
                better = (kind, how, alt)
        if better:
            reply_kind, note = better[0], better[1]
    index = {o["id"]: j for j, o in enumerate(objs)}
    match = {lv: ({m["prediction"]: index[m["reference_id"]] for m in s["tolerances"][lv]["matches"]} if s else {})
             for lv in ("tight", "medium", "loose")}
    found = set(match["loose"].values())
    kind, target, claimed = {}, {}, {}
    for i in match["loose"]:
        kind[i], target[i] = "correct", match["loose"][i]
    rest = [i for i in range(len(points)) if i not in kind]
    for i in rest:
        x, y = px[i]
        if reply_kind in ("coordinate mix-up", "shifted or stretched"):
            kind[i] = reply_kind
        elif i in loop_idx:
            kind[i] = "doom loop"
        elif not (0 <= x <= W and 0 <= y <= H):
            kind[i] = "off image"
    # on an outline
    on = []
    for i in rest:
        if i in kind:
            continue
        j = at(T["keylab"], *px[i])
        if j >= 0:
            o = objs[j]
            cx, cy = o["dot"] or ((o["box"][0] + o["box"][2]) / 2, (o["box"][1] + o["box"][3]) / 2)
            on.append((math.hypot(px[i][0] - cx, px[i][1] - cy), i, j))
    for _, i, j in sorted(on):
        target[i] = j
        if not objs[j]["in_point_key"]:
            kind[i] = "maybe missing from key"   # an object of the box key the point key lacks
        elif j not in found and j not in claimed:
            kind[i], claimed[j] = "on the object, off-centre", i
        else:
            kind[i] = "duplicate"
    # on a look-alike
    for i in rest:
        if i not in kind and (l := at(T["looklab"], *px[i])) >= 0:
            kind[i], target[i] = "wrong object", "look:" + T["look"][l]["query"]
    # next to an object
    near = []
    for i in rest:
        if i in kind:
            continue
        for j, o in enumerate(objs):
            if o["box"]:
                b = o["box"]
                r = box_dist(*px[i], b) / (NEAR * math.sqrt(max(1, (b[2] - b[0]) * (b[3] - b[1]))))
            elif o["dot"]:  # in the point key only (one bird the box review removed): twice the ring
                r = math.dist(px[i], o["dot"]) / (2 * T["radius"][o["id"]])
            else:
                continue
            if r <= 1:
                near.append((r, i, j))
    for r, i, j in sorted(near):
        if i in kind:
            continue
        if objs[j]["in_point_key"] and j not in found and j not in claimed:
            kind[i], target[i], claimed[j] = "near miss", j, i
    for r, i, j in sorted(near):
        if i not in kind:  # off every outline, next to an object already found or claimed
            kind[i], target[i] = "in a gap", j
    for i in rest:
        if i not in kind:
            if at(T["gaplab"], *px[i]) >= 0:
                kind[i] = "maybe missing from key"
            elif (n := shared("points", rec["task"], entrant, *px[i], None, T)) >= SHARED:
                kind[i], target[i] = "wrong object", f"shared:{n}"
            else:
                kind[i] = "background"
    rows = []
    for i in range(len(points)):
        j = target.get(i)
        rows.append({"item": "point", "index": i, "x": round(px[i][0], 1), "y": round(px[i][1], 1), "kind": kind[i],
                     "object": objs[j]["id"] if isinstance(j, int) else (j or ""),
                     "full_credit": i in match["tight"], "detail": ""})
    miss = reply_kind or "missed"
    for j in pkey:
        if j not in found and j not in claimed:
            o = objs[j]
            rows.append({"item": "object", "index": j, "x": o["dot"][0], "y": o["dot"][1], "kind": miss,
                         "object": o["id"], "full_credit": False, "detail": ""})
    # TIDE-style fixes on the charted score (mean F1 over the three rings)
    tps = {lv: len(match[lv]) for lv in ("tight", "medium", "loose")}
    gains = fix_gains(rows, tps, len(points), n_ref, base, better[2]["score"] if better else None,
                      fix_in_place=("on the object, off-centre", "near miss"), partial="partial credit")
    flags = {"reply_kind": reply_kind or "", "reading": note, "loop": "; ".join(loop_why),
             "better_score": round(better[2]["score"], 4) if better else ""}
    return rows, gains, flags


def fix_gains(rows, tps, n, m, base, better_score, fix_in_place, partial):
    def sc(tp_by_level, n_):
        return statistics.mean(2 * t / (n_ + m) if n_ + m else 1.0 for t in tp_by_level.values())
    wrong = collections.Counter(r["kind"] for r in rows if r["item"] != "object" and r["kind"] != "correct")
    missed = collections.Counter(r["kind"] for r in rows if r["item"] == "object")
    gains = {}
    for k in set(wrong) | set(missed):
        if k in ("coordinate mix-up", "shifted or stretched"):
            new = better_score if better_score is not None else base
        elif k in fix_in_place:
            new = sc({lv: t + wrong[k] for lv, t in tps.items()}, n)
        elif k == "group box":
            new = sc({lv: t + missed[k] for lv, t in tps.items()}, n - wrong[k] + missed[k])
        else:
            new = sc({lv: t + missed[k] for lv, t in tps.items()}, n - wrong[k] + missed[k])
        gains[k] = new - base
    lowest = list(tps.values())[-1] if partial == "partial credit" else list(tps.values())[0]
    gains[partial] = sc({lv: lowest for lv in tps}, n) - base
    return gains


# ------------------------------------------------------------------ boxes

def match_boxes(px, refs, level):
    ov = [[iou(p, q) for q in refs] for p in px]
    edges = [[j for j, o in sorted(enumerate(row), key=lambda x: -x[1]) if o >= level - 1e-9] for row in ov]
    return dict(maximum_matching(edges)), ov


def box_score(px, refs):
    tps = {lv: len(match_boxes(px, refs, lv)[0]) for lv in LEVELS}
    n, m = len(px), len(refs)
    return statistics.mean(2 * t / (n + m) if n + m else 1.0 for t in tps.values()), tps


def classify_box_reply(rec, raw, T, entrant=None):
    W, H = T["W"], T["H"]
    objs = [o for o in T["objects"] if o["in_box_key"]]
    refs = [o["box"] for o in objs]
    m = len(refs)
    boxes = [b["box"] for b in rec["boxes"]]
    px = [[b[0] * W / 1000, b[1] * H / 1000, b[2] * W / 1000, b[3] * H / 1000] for b in boxes]
    base, tps = box_score(px, refs)
    assert abs(base - rec["scores"]["score"]) < 1e-9, (rec["model"], rec["task"])
    readable = rec["how"].get("read_as") != "unreadable"
    reply_kind, better, note = None, None, ""
    if not px:
        reply_kind = no_answer_kind(raw or {}, readable, m)
    loop_idx, loop_why = mechanical(boxes)
    match50, ov = match_boxes(px, refs, 0.5)
    on_share = sum(at(T["keylab"], (b[0] + b[2]) / 2, (b[1] + b[3]) / 2) >= 0 for b in px) / len(px) if px else 1
    if px and not loop_idx:
        f50 = 2 * len(match50) / (len(px) + m)
        top = max(max(abs(v) for v in b) for b in boxes)
        cands = []
        for how in ALTERNATIVES:
            if fits(how, top):
                alt = []
                for b in boxes:
                    (x1, y1), (x2, y2) = reread(b[:2], how, W, H), reread(b[2:], how, W, H)
                    alt.append([min(x1, x2) * W / 1000, min(y1, y2) * H / 1000, max(x1, x2) * W / 1000, max(y1, y2) * H / 1000])
                cands.append(("coordinate mix-up", how, alt))
        if len(px) >= 6 and on_share < 0.5:
            centres = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in px]
            a, bb, c, d = fit_affine(centres, [((r[0] + r[2]) / 2, (r[1] + r[3]) / 2) for r in refs])
            alt = [[b[0] * a + bb, b[1] * c + d, b[2] * a + bb, b[3] * c + d] for b in px]
            cands.append(("shifted or stretched", f"x*{a:.2f}{bb:+.0f}px, y*{c:.2f}{d:+.0f}px", alt))
        for kind, how, alt in cands:
            sc, t = box_score(alt, refs)
            f = 2 * t[0.5] / (len(alt) + m)
            if f >= f50 + MIX_GAIN and f >= 2 * f50 and (better is None or f > better[3]):
                better = (kind, how, sc, f)
        if better:
            reply_kind, note = better[0], better[1]
    found = set(match50.values())
    kind, target, detail, claimed, group_of = {}, {}, {}, {}, {}
    for i, j in match50.items():
        kind[i], target[i] = "correct", j
        detail[i] = f"overlap {ov[i][j]:.2f}"
    rest = [i for i in range(len(px)) if i not in kind]

    def area(b):
        return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    def inter(a, b):
        return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))

    for i in rest:
        b = px[i]
        if reply_kind in ("coordinate mix-up", "shifted or stretched"):
            kind[i] = reply_kind
        elif i in loop_idx:
            kind[i] = "doom loop"
        elif b[2] <= 0 or b[3] <= 0 or b[0] >= W or b[1] >= H:
            kind[i] = "off image"
        else:
            inside = [j for j, r in enumerate(refs) if inter(b, r) >= 0.5 * area(r) and ov[i][j] < 0.5]
            if len(inside) >= 2:
                kind[i], detail[i] = "group box", f"{len(inside)} objects"
                group_of[i] = inside
    loc = sorted(((ov[i][j], i, j) for i in rest if i not in kind for j in range(m)
                  if ov[i][j] >= 0.1 and j not in found), reverse=True)
    for o, i, j in loc:
        if i in kind or j in claimed:
            continue
        b, r = px[i], refs[j]
        part = inter(b, r) >= 0.8 * area(b) and area(b) < area(r)
        big = inter(b, r) >= 0.8 * area(r) and area(b) > area(r)
        kind[i], target[i], claimed[j] = "mispositioned box", j, i
        detail[i] = f"{'part' if part else 'too big' if big else 'off'}, overlap {o:.2f}"
    for i in rest:
        if i in kind:
            continue
        b = px[i]
        best = max(range(m), key=lambda j: ov[i][j]) if m else None
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        if best is not None and ov[i][best] >= 0.1:
            kind[i], target[i] = "duplicate", best
        elif (l := at(T["looklab"], cx, cy)) >= 0 or any(iou(b, x["box"]) >= 0.5 for x in T["look"]):
            q = T["look"][l]["query"] if l >= 0 else next(x["query"] for x in T["look"] if iou(b, x["box"]) >= 0.5)
            kind[i], target[i] = "wrong object", "look:" + q
        elif at(T["gaplab"], cx, cy) >= 0 or any(iou(b, x["box"]) >= 0.5 for x in T["gap"]):
            kind[i] = "maybe missing from key"
        elif (n := shared("boxes", rec["task"], entrant, cx, cy, b, T)) >= SHARED:
            kind[i], target[i] = "wrong object", f"shared:{n}"
        else:
            kind[i] = "background"
    rows = []
    for i in range(len(px)):
        j = target.get(i)
        rows.append({"item": "box", "index": i, "x": round((px[i][0] + px[i][2]) / 2, 1),
                     "y": round((px[i][1] + px[i][3]) / 2, 1), "kind": kind[i],
                     "object": objs[j]["id"] if isinstance(j, int) else (j or ""),
                     "full_credit": kind[i] == "correct" and ov[i][j] >= 0.95 - 1e-9,
                     "detail": detail.get(i, ""), "box": [round(v, 1) for v in px[i]]})
    grouped = {j for g in group_of.values() for j in g}
    for j, o in enumerate(objs):
        if j not in found and j not in claimed:
            k = "group box" if j in grouped and not reply_kind else (reply_kind or "missed")
            r = o["box"]
            rows.append({"item": "object", "index": j, "x": (r[0] + r[2]) / 2, "y": (r[1] + r[3]) / 2, "kind": k,
                         "object": o["id"], "full_credit": False, "detail": "", "box": r})
    gains = fix_gains(rows, tps, len(px), m, base, better[2] if better else None,
                      fix_in_place=("mispositioned box",), partial="loose fit")
    flags = {"reply_kind": reply_kind or "", "reading": note, "loop": "; ".join(loop_why),
             "better_score": round(better[2], 4) if better else ""}
    return rows, gains, flags


# ------------------------------------------------------------------ reports

def per_model(per_reply, kinds, extra):
    models = collections.defaultdict(lambda: collections.defaultdict(list))
    labels = {}
    for who, base, gains, counts in per_reply:
        models[who["entrant"]][who["task"]].append((base, gains, counts))
        labels[who["entrant"]] = who["label"]
    table = []
    for e, by_task in models.items():
        def mean(fn):
            return statistics.mean(statistics.mean(fn(x) for x in v) for v in by_task.values())
        row = {"entrant": e, "label": labels[e], "score": mean(lambda x: x[0])}
        for k in kinds + (extra,):
            row[f"gain:{k}"] = mean(lambda x, k=k: x[1].get(k, 0.0))
        tot = collections.Counter()
        for v in by_task.values():
            for x in v:
                tot.update(x[2])
        for k in ("correct",) + kinds:
            row[f"n:{k}"] = tot[k]
        table.append(row)
    return sorted(table, key=lambda r: -r["score"])


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def section(name, items, table, kinds, extra, unit):
    wrong = collections.Counter(r["kind"] for r in items if r["item"] != "object")
    unfound = collections.Counter(r["kind"] for r in items if r["item"] == "object")
    tw, tu = sum(wrong[k] for k in kinds), sum(unfound.values())
    out = [f"## {name}", "", f"| kind | wrong {unit} | share | objects left unfound | share |", "|---|---:|---:|---:|---:|"]
    for k in kinds:
        if wrong[k] or unfound[k]:
            out.append(f"| {k} | {wrong[k] or ''} | {f'{wrong[k] / tw:.1%}' if wrong[k] else ''} | "
                       f"{unfound[k] or ''} | {f'{unfound[k] / tu:.1%}' if unfound[k] else ''} |")
    out.append(f"| total | {tw} | | {tu} | |")
    full = sum(1 for r in items if r["kind"] == "correct" and r["full_credit"] in (True, "True"))
    out += ["", f"Correct {unit}: {wrong['correct']} (full credit at the strictest level: {full}). "
            "Objects left unfound: the reply-level kind when there is one, the group box that swallowed them, "
            "or missed.", "",
            f"### Score each kind costs, per model (points out of 100, as charted)", "",
            "| model | score | " + " | ".join(kinds + (extra,)) + " |", "|---|---:|" + "---:|" * (len(kinds) + 1)]
    for r in table:
        out.append(f"| {r['label']} | {100 * r['score']:.0f} | " + " | ".join(
            f"{100 * r['gain:' + k]:.1f}" if r["gain:" + k] >= 0.0005 else "" for k in kinds + (extra,)) + " |")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "results/error-kinds/v1")
    ap.add_argument("--points-dir", type=Path, default=POINTS_DIR,
                    help="scored point answers to sort (default: point v3). The crowd behind 'wrong object' "
                         "(SHARED) stays the published point v3 answers, so adding answers changes no earlier kind")
    ap.add_argument("--skip-boxes", action="store_true", help="points only")
    a = ap.parse_args()
    global CLASSIFY_DIR, SKIP_BOXES
    CLASSIFY_DIR, SKIP_BOXES = a.points_dir.resolve(), a.skip_boxes
    pool = {json.loads(l)["entrant"] for l in (POINTS_DIR / "scored.jsonl").read_text().splitlines()}
    if a.out.exists():
        sys.exit(f"{a.out} exists; choose a new --out (results are never overwritten)")
    photos = load_photos()
    for T in photos.values():
        refs = [{"id": o["id"], "x": o["dot"][0], "y": o["dot"][1]} for o in T["objects"] if o["in_point_key"]]
        from segbench.point_score import reference_distances, DIAGONAL_CAP, TOLERANCES
        nn = reference_distances(refs, T["W"], T["H"])
        T["radius"] = {r["id"]: min(TOLERANCES["loose"] * d, DIAGONAL_CAP * math.hypot(T["W"], T["H"]))
                       for r, d in zip(refs, nn)}

    for pass_no in (1, 2):
        out = run_all(photos)
        if pass_no == 1:
            for bench, items in (("points", out[0]), ("boxes", out[3])):
                for r in items:
                    if bench == "points" and r["entrant"] not in pool:
                        continue
                    if r["item"] != "object" and r["kind"] in ("background", "wrong object"):
                        box = json.loads(r["box"]) if bench == "boxes" else None
                        FIRST_PASS.setdefault((bench, r["task"]), []).append((r["entrant"], r["x"], r["y"], box))
    p_items, p_replies, p_per, b_items, b_replies, b_per = out
    write_all(a, p_items, p_replies, p_per, b_items, b_replies, b_per)


def run_all(photos):
    # points
    scored = [json.loads(l) for l in (CLASSIFY_DIR / "scored.jsonl").read_text().splitlines()]
    src = {n: (ROOT / "results" / n).read_text().splitlines() for n in {r["source_file"] for r in scored}}
    p_items, p_replies, p_per = [], [], []
    for rec in scored:
        raw = json.loads(src[rec["source_file"]][rec["source_line"] - 1])
        assert (raw["task"], raw["model"]) == (rec["task"], rec["model"])
        rows, gains, flags = classify_point_reply(rec, raw, photos[rec["task"]], rec["entrant"])
        who = {"entrant": rec["entrant"], "label": rec["label"], "task": rec["task"], "repeat": rec["repeat"],
               "prompt": raw.get("prompt_version", "")}
        p_items += [{**who, **r} for r in rows]
        c = collections.Counter(r["kind"] for r in rows)
        p_replies.append({**who, "score": round(rec["score"], 4), "points": rec.get("predicted_count") or 0,
                          "objects": sum(o["in_point_key"] for o in photos[rec["task"]]["objects"]),
                          **{k: c[k] for k in ("correct",) + POINT_KINDS}, **flags})
        p_per.append((who, rec["score"], gains, c))

    # boxes
    if SKIP_BOXES:
        return p_items, p_replies, p_per, [], [], []
    bscored = [json.loads(l) for l in (BOXES_DIR / "scored.jsonl").read_text().splitlines()]
    from segbench.chart_data import describe
    raws = {}
    for name in {r["source_file"] for r in bscored}:
        for line in (ROOT / "results" / name).read_text().splitlines():
            r = json.loads(line)
            if r.get("track") != "find" or "error" in r:
                continue
            k = (name, r["model"], r.get("level"), r.get("prompt_version"), r["task"], r["repeat"])
            raws[k] = r
    b_items, b_replies, b_per = [], [], []
    for rec in bscored:
        raw = raws.get((rec["source_file"], rec["model"], rec.get("level"), rec["prompt_version"], rec["task"], rec["repeat"]))
        assert raw is not None, rec["model"]
        entrant = f"{rec['model']}@{rec['level']}" if rec.get("level") else rec["model"]
        rows, gains, flags = classify_box_reply(rec, raw, photos[rec["task"]], entrant)
        who = {"entrant": entrant, "label": describe(rec["model"])[0], "task": rec["task"], "repeat": rec["repeat"],
               "prompt": rec["prompt_version"]}
        b_items += [{**who, **{k: v for k, v in r.items() if k != "box"}, "box": json.dumps(r["box"])} for r in rows]
        c = collections.Counter(r["kind"] for r in rows)
        b_replies.append({**who, "score": round(rec["scores"]["score"], 4), "boxes": len(rec["boxes"]),
                          "objects": rec["scores"]["reference"],
                          **{k: c[k] for k in ("correct",) + BOX_KINDS}, **flags})
        b_per.append((who, rec["scores"]["score"], gains, c))
    return p_items, p_replies, p_per, b_items, b_replies, b_per


def write_all(a, p_items, p_replies, p_per, b_items, b_replies, b_per):
    a.out.mkdir(parents=True)
    benches = [("points", p_items, p_replies, p_per, POINT_KINDS, "partial credit")]
    if not SKIP_BOXES:
        benches.append(("boxes", b_items, b_replies, b_per, BOX_KINDS, "loose fit"))
    for name, items, replies, per, kinds, extra in benches:
        (a.out / name).mkdir()
        write_csv(a.out / name / "items.csv", items)
        write_csv(a.out / name / "replies.csv", replies)
        table = per_model(per, kinds, extra)
        write_csv(a.out / name / "by-model.csv", table)
    lines = ["# Kinds of mistakes on the eleven photos", "",
             "Made by scripts/classify_errors.py (rules in its header, reasons in docs/error-kinds.md). "
             "Scores unchanged. Outlines of the key objects (SAM 2.1, from the reviewed boxes) and of look-alikes "
             "(SAM 3) are measuring aids, not reviewed by a person.", ""]
    source = "point benchmark v3" if CLASSIFY_DIR == POINTS_DIR else CLASSIFY_DIR.relative_to(ROOT).as_posix()
    lines += section(f"Points ({source}, {len(p_replies)} replies)", p_items,
                     per_model(p_per, POINT_KINDS, "partial credit"), POINT_KINDS, "partial credit", "points")
    if not SKIP_BOXES:
        lines += [""] + section(f"Boxes (box benchmark v1, {len(b_replies)} replies)", b_items,
                                per_model(b_per, BOX_KINDS, "loose fit"), BOX_KINDS, "loose fit", "boxes")
        sub = collections.Counter(r["detail"].split(",")[0] for r in b_items if r["kind"] == "mispositioned box")
        lines += ["", "Mispositioned boxes: " + ", ".join(f"{k} {v}" for k, v in sub.most_common()) + "."]
    lines += ["", "Each figure: how much higher that model's charted score would be if that one kind alone were fixed. "
              "Fixes interact; the figures do not add up."]
    (a.out / "summary.md").write_text("\n".join(lines) + "\n")
    print((a.out / "summary.md").read_text())


if __name__ == "__main__":
    main()
