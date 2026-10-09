"""A draft answer key from many models' answers, and how much to trust each model.

This is NOT the answer key and never scores the benchmark. Models that share
a blind spot agree on the same mistake (every OpenAI model "found" a cat in
the frosted glass of the kitchen cabinets), and a lone model that is right
is outvoted. What a consensus is good for:

1. a first draft for the person making the answer key, who then only has to
   check the objects the models disagree about (the review queue);
2. a provisional ranking while the answer key is being made, scored with a
   leave-maker-out rule: a model is judged only against the consensus of the
   *other* makers' models, so it cannot vote for itself and its siblings
   (four GPT models, three Geminis, two Claudes) cannot carry it.

Methods (standard ones, so others can check them):
- Objects (dots, and the inside points of outlines): dots from different
  answers are grouped when they are close (no group holds two dots of the
  same answer), then a Dawid-Skene style EM estimates, together, whether
  each group is a real object and each answer's recall (share of real
  objects it marks) and false-mark rate. Better answers end up weighing more.
- Labels (dirty / clean): the same EM, with each answer's label accuracy.
- Regions (red foliage, road): STAPLE (Warfield et al., 2004), the standard
  way to merge several segmentations of one picture: per answer a
  sensitivity and specificity, and per pixel the chance it is in the region.

Coordinates here are photo pixels.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

# ---------- geometry ----------


def polygon_area(poly) -> float:
    p = np.asarray(poly, dtype=float)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _inside(pt, poly) -> bool:
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            inside = not inside
    return inside


def interior_point(poly) -> list[float]:
    """A point inside the outline: its area centroid, or the deepest grid point if that falls outside."""
    p = np.asarray(poly, dtype=float)
    x, y = p[:, 0], p[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    a = cross.sum() / 2
    if abs(a) > 1e-9:
        c = [float(((x + np.roll(x, -1)) * cross).sum() / (6 * a)), float(((y + np.roll(y, -1)) * cross).sum() / (6 * a))]
        if _inside(c, poly):
            return c
    x0, y0, x1, y1 = x.min(), y.min(), x.max(), y.max()
    w = max(1, int(x1 - x0) + 1)
    h = max(1, int(y1 - y0) + 1)
    s = max(w, h) / 32
    m = Image.new("1", (max(1, int(w / s)), max(1, int(h / s))), 0)
    ImageDraw.Draw(m).polygon([((px - x0) / s, (py - y0) / s) for px, py in poly], fill=1)
    arr = np.array(m)
    if not arr.any():
        return [float(x.mean()), float(y.mean())]
    # depth = distance to the nearest outside cell (small grid: brute force is fine)
    ins = np.argwhere(arr)
    outs = np.argwhere(~np.pad(arr, 1))  - 1
    d = ((ins[:, None, :] - outs[None, :, :]) ** 2).sum(-1).min(1)
    r, c = ins[d.argmax()]
    return [float(x0 + (c + 0.5) * s), float(y0 + (r + 0.5) * s)]


def rasterize(polygons, size: tuple[int, int], scale: float) -> np.ndarray:
    """Fill polygons (photo pixels) on a grid `scale` times smaller than the photo."""
    w, h = size
    im = Image.new("1", (max(1, round(w / scale)), max(1, round(h / scale))), 0)
    d = ImageDraw.Draw(im)
    for poly in polygons:
        d.polygon([(px / scale, py / scale) for px, py in poly], fill=1)
    return np.array(im, dtype=bool)


# ---------- objects: group dots, then EM ----------


def group(dots: list[tuple[str, int, float, float]], radius: float) -> list[dict]:
    """Group dots (answer, index, x, y) of different answers that mark the same object.

    Closest pairs first; a group never holds two dots of one answer, and no two
    of its dots are more than 1.5 x radius apart.
    """
    if not dots:
        return []
    xy = np.array([[d[2], d[3]] for d in dots])
    who = [d[0] for d in dots]
    dist = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    parent = list(range(len(dots)))
    members = {i: [i] for i in range(len(dots))}
    answers = {i: {who[i]} for i in range(len(dots))}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    ii, jj = np.nonzero(np.triu(dist < radius, 1))
    for k in np.argsort(dist[ii, jj], kind="stable"):
        a, b = root(ii[k]), root(jj[k])
        if a == b or answers[a] & answers[b]:
            continue
        merged = members[a] + members[b]
        if dist[np.ix_(merged, merged)].max() > 1.5 * radius:
            continue
        parent[b] = a
        members[a], answers[a] = merged, answers[a] | answers[b]
        del members[b], answers[b]
    out = []
    for m in members.values():
        out.append({"center": np.median(xy[m], axis=0).tolist(),
                    "votes": {who[i]: dots[i][1] for i in m}})
    return out


def tempering(makers: list[str]) -> float:
    """How many answers there are per independent source (one company = one source).

    EM assumes answers are independent witnesses. They are not: two answers
    of one model, and models of one company, make the same mistakes. Treated
    as independent, 26 answers make every verdict 0% or 100% and nothing goes
    to review. Dividing the evidence by answers-per-company is a standard,
    crude correction (likelihood tempering) that keeps the chances honest.
    """
    return max(1.0, len(makers) / max(1, len(set(makers))))


def em_objects(groups: list[dict], answers: list[str], iterations: int = 200, temper: float = 1.0):
    """Dawid-Skene for detection. Returns (P(real) per group, recall, false-mark rate, precision) per answer."""
    X = np.array([[a in g["votes"] for a in answers] for g in groups], dtype=float)
    if X.size == 0:
        return np.zeros(0), {}, {}, {}
    p = X.mean(1)
    for _ in range(iterations):
        recall = (X.T @ p + 1) / (p.sum() + 2)
        false = (X.T @ (1 - p) + 1) / ((1 - p).sum() + 2)
        prior = np.clip(p.mean(), 0.05, 0.95)
        logit = (np.log(prior / (1 - prior))
                 + (X @ np.log(recall / false) + (1 - X) @ np.log((1 - recall) / (1 - false))) / temper)
        new = 1 / (1 + np.exp(-np.clip(logit, -50, 50)))
        if np.abs(new - p).max() < 1e-6:
            p = new
            break
        p = new
    marks = X.sum(0)
    precision = np.where(marks > 0, (X.T @ p) / np.maximum(marks, 1), np.nan)
    return (p, dict(zip(answers, recall)), dict(zip(answers, false)), dict(zip(answers, precision)))


def em_labels(groups, p_real, labels: dict[str, list[str]], classes: tuple[str, str], iterations: int = 100,
              temper: float = 1.0):
    """One-coin Dawid-Skene for a two-way label. Returns (P(first class) per group, accuracy per answer)."""
    first = classes[0]
    votes = [[(a, labels[a][i] == first) for a, i in g["votes"].items()] for g in groups]
    q = np.array([np.mean([v for _, v in vs]) if vs else 0.5 for vs in votes])
    answers = sorted({a for vs in votes for a, _ in vs})
    acc = {}
    for _ in range(iterations):
        for a in answers:
            num = den = 0.0
            for c, vs in enumerate(votes):
                for b, v in vs:
                    if b == a:
                        num += p_real[c] * (q[c] if v else 1 - q[c])
                        den += p_real[c]
            acc[a] = (num + 1) / (den + 2)
        rho = np.clip(np.average(q, weights=p_real + 1e-9), 0.05, 0.95)
        new = []
        for vs in votes:
            lo = np.log(rho / (1 - rho)) + sum(np.log(acc[a] / (1 - acc[a])) * (1 if v else -1) for a, v in vs) / temper
            new.append(1 / (1 + np.exp(-lo)))
        new = np.array(new)
        if np.abs(new - q).max() < 1e-6:
            q = new
            break
        q = new
    return q, acc


def match(points, centers, radius) -> list[tuple[int, int]]:
    """Closest-first one-to-one matching of points to centers within radius."""
    if not len(points) or not len(centers):
        return []
    P, C = np.asarray(points), np.asarray(centers)
    d = np.sqrt(((P[:, None, :] - C[None, :, :]) ** 2).sum(-1))
    pairs, used_p, used_c = [], set(), set()
    for k in np.argsort(d, axis=None, kind="stable"):
        i, j = divmod(int(k), len(C))
        if d[i, j] >= radius:
            break
        if i not in used_p and j not in used_c:
            pairs.append((i, j))
            used_p.add(i)
            used_c.add(j)
    return pairs


# ---------- regions: STAPLE ----------


def staple(masks: dict[str, np.ndarray], iterations: int = 100, temper: float = 1.0):
    """Returns (P(in region) per pixel, sensitivity, specificity) per answer."""
    names = list(masks)
    D = np.stack([masks[n].ravel() for n in names]).astype(float)
    W = D.mean(0)
    for _ in range(iterations):
        sens = (D @ W + 1) / (W.sum() + 2)
        spec = ((1 - D) @ (1 - W) + 1) / ((1 - W).sum() + 2)
        prior = np.clip(W.mean(), 0.01, 0.99)
        la = np.log(prior) + np.log(sens) @ D + np.log(1 - sens) @ (1 - D)
        lb = np.log(1 - prior) + np.log(1 - spec) @ D + np.log(spec) @ (1 - D)
        new = 1 / (1 + np.exp(np.clip((lb - la) / temper + (1 - 1 / temper) * np.log((1 - prior) / prior), -50, 50)))
        if np.abs(new - W).max() < 1e-5:
            W = new
            break
        W = new
    shape = next(iter(masks.values())).shape
    return W.reshape(shape), dict(zip(names, sens)), dict(zip(names, spec))
