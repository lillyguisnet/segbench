"""Move GPT-6 Astra's measurement clicks onto SAM 3's outline of the part (drafts only, 2026-10-16).

uv run measure_refine.py DRAFTS.json OUT.json      (from scripts/draft_measure_clicks.py refine)

Runs on GPU 1 (GPU 0 is the benchmark's, and busy): drafts, not timed.

For each of Astra's instances:
- SAM 3 (its point-prompt mode, "instance interactivity") is given one click
  in the middle of Astra's clicks and returns three candidate outlines (part,
  larger part, whole). Kept: the one containing the click whose extent along
  Astra's direction is closest to Astra's own length.
- span: each end moves to the outline's farthest pixel along Astra's
  direction, within a band around Astra's line (a tenth of its length, at
  least 4 px, each side).
- line: the two end clicks move outwards along the line to the outline's
  edge (or inwards to it, if they are outside it); the middle clicks move to
  the deepest point of the outline within a fifth of the length around them.
- Not believable -> Astra's clicks are kept (origin "astra"): the click is
  not inside the outline, the outline covers over 30 % of the photo, or the
  length changes by more than a factor 1.6.
- The dot: the middle of the clicks; when that is not on the outline (a
  wingspan's middle can fall between the wings), the deepest point of the
  outline instead.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-73bb8110-4d4b-f645-dce4-724b0baf767f"  # GPU 1

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def length(p):
    return sum(math.dist(p[i - 1], p[i]) for i in range(1, len(p)))


def middle(p):
    half, s = length(p) / 2, 0.0
    for i in range(1, len(p)):
        d = math.dist(p[i - 1], p[i])
        if s + d >= half and d > 0:
            f = (half - s) / d
            return [p[i - 1][0] + f * (p[i][0] - p[i - 1][0]), p[i - 1][1] + f * (p[i][1] - p[i - 1][1])]
        s += d
    return list(p[0])


def inside(mask, x, y):
    h, w = mask.shape
    return bool(mask[min(h - 1, max(0, int(y))), min(w - 1, max(0, int(x)))])


def span_on(mask, a, b):
    """Ends of the mask along a->b, within a band around the line; None if empty."""
    a, b = np.array(a, float), np.array(b, float)
    L = np.linalg.norm(b - a)
    if L < 1:
        return None
    u = (b - a) / L
    n = np.array([-u[1], u[0]])
    ys, xs = np.nonzero(mask)
    rel = np.stack([xs, ys], 1) - a
    t, off = rel @ u, rel @ n
    band = np.abs(off) <= max(4.0, 0.1 * L)
    if band.sum() < 3:
        return None
    t, off = t[band], off[band]
    i, j = np.argmin(t), np.argmax(t)
    return [list(a + t[i] * u + off[i] * n), list(a + t[j] * u + off[j] * n)]


def march(mask, frm, to, limit):
    """From `to`, step along frm->to while inside the mask (or back while outside); the last inside point."""
    d = np.array(to, float) - np.array(frm, float)
    if np.linalg.norm(d) < 1:
        return list(to)
    d /= np.linalg.norm(d)
    p = np.array(to, float)
    if inside(mask, *p):
        for _ in range(int(limit)):
            q = p + d
            if not inside(mask, *q):
                break
            p = q
    else:
        for _ in range(int(limit)):
            p = p - d
            if inside(mask, *p):
                break
        else:
            return list(to)
    return list(p)


def line_on(mask, pts, dist):
    L = length(pts)
    out = [list(p) for p in pts]
    out[0] = march(mask, pts[1], pts[0], L * 0.6)
    out[-1] = march(mask, pts[-2], pts[-1], L * 0.6)
    r = max(3, int(0.1 * L))
    h, w = mask.shape
    for k in range(1, len(pts) - 1):
        x, y = int(pts[k][0]), int(pts[k][1])
        x0, x1, y0, y1 = max(0, x - r), min(w, x + r + 1), max(0, y - r), min(h, y + r + 1)
        win = dist[y0:y1, x0:x1]
        if win.size and win.max() > 0:
            yy, xx = np.unravel_index(np.argmax(win), win.shape)
            out[k] = [x0 + xx, y0 + yy]
    return out


def extent(mask, pts):
    s = span_on(mask, pts[0], pts[-1])
    return math.dist(*s) if s else 0.0


def main():
    src, dst = sys.argv[1], sys.argv[2]
    drafts = json.loads(Path(src).read_text())
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(enable_inst_interactivity=True)
    proc = Sam3Processor(model)
    out = {}
    by_photo = {}
    for key, d in drafts.items():
        by_photo.setdefault(d["photo"], []).append(key)
    for photo, keys in by_photo.items():
        im = Image.open(ROOT / "images" / photo).convert("RGB")
        W, H = im.size
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = proc.set_image(im)
            for key in keys:
                d, res = drafts[key], []
                for pts in d["instances"]:
                    mid = middle(pts)
                    masks, scores, _ = model.predict_inst(state, point_coords=np.array([mid], np.float32),
                                                          point_labels=np.array([1]), multimask_output=True)
                    L0 = length(pts) if d["kind"] == "line" else math.dist(pts[0], pts[1])
                    best, why = None, "no outline holds the middle click"
                    cands = [(abs(extent(m > 0, pts) - math.dist(pts[0], pts[-1])), i) for i, m in enumerate(masks)
                             if inside(m > 0, *mid)]
                    if cands:
                        m = masks[min(cands)[1]] > 0
                        if m.mean() > 0.30:
                            why = f"outline covers {m.mean():.0%} of the photo"
                        else:
                            if d["kind"] == "span":
                                new = span_on(m, pts[0], pts[1])
                            else:
                                dist = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
                                new = line_on(m, pts, dist)
                            L1 = (length(new) if d["kind"] == "line" else math.dist(*new)) if new else 0
                            if new and 1 / 1.6 <= L1 / max(L0, 1e-6) <= 1.6:
                                best = (new, m)
                            else:
                                why = f"length {L0:.0f} -> {L1:.0f} px"
                    if best:
                        new, m = best
                        new = [[min(W, max(0, x)), min(H, max(0, y))] for x, y in new]
                        dot = None
                        if not inside(m, *middle(new)):
                            dist = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
                            yy, xx = np.unravel_index(np.argmax(dist), dist.shape)
                            dot = [float(xx), float(yy)]
                        res.append({"points": new, "origin": "astra+sam3", "astra_points": pts, "dot": dot})
                    else:
                        res.append({"points": pts, "origin": "astra", "astra_points": pts, "dot": None, "kept_because": why})
                n = sum(r["origin"] == "astra+sam3" for r in res)
                print(f"{key:20} {len(res):2} drafts, {n} moved onto SAM 3's outline", flush=True)
                out[key] = {"photo": photo, "kind": d["kind"], "instances": res}
    Path(dst).write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
