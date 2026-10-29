"""Jev + halving: finding every object of a kind with a model that gives one box per question.

Jev 1.14 (TypeSafe) answers judgments: here a Choice none / one / several and
a Box. It has no "list every object". The search around it (Maxime,
2026-10-16), first written and tuned on synthetic ovals only
(scripts/probe_jev_split.py, docs/jev.md) and copied here unchanged in its
rules before any benchmark photo was sent:

1. Ask about a piece of the picture (the first piece is the whole picture):
   how many targets are in it (none / one / several; one cut off by the edge
   counts), and the box of "the target" in it, both in one request.
   - none     -> stop;
   - one      -> the box is one object;
   - several  -> cut the piece in two across its longer side, ask both halves.
   All the pieces of one depth are asked at the same time. At MAX_DEPTH a
   "several" is boxed anyway (flagged `forced`).
2. Each piece is cut from the full-size picture and fitted to Jev's limits
   (long side <= 1568, short side >= 280; grey bars if both cannot hold), so
   halving also zooms in.
3. Parts of one object cut by a line are joined: two boxes from pieces that
   share a cut line, each touching it (within 2% of its piece), whose spans
   along the line overlap by half the shorter span or more.
4. Each joined box is asked again on its own; "several" halves it once more
   (up to two rounds).

What it returns: boxes in pixels of the full picture; the point of an object
is the middle of its box (Jev gives no points).

The rules differ from the probe in one way only, which changes no answer:
the joined boxes of step 4 are checked at the same time instead of one after
another (timing).
"""
from __future__ import annotations

import io
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from lm15 import bounding_box, choice, judgments
from PIL import Image

from .call import call
from .models import BY_KEY

MODEL = BY_KEY["jev"]
MAX_DEPTH = 7
RECHECK_ROUNDS = 2
MIN_SIDE, MAX_SIDE = 280, 1568  # the API's real minimum (the docs say 256), and its maximum
# Enlarge (Lanczos) every piece whose long side is shorter than this, keeping its shape;
# None = only as much as MIN_SIDE needs (as run jev-halving-1). Set by run_jev_halving --min-long.
MIN_LONG: int | None = None
ATTEMPTS = 5  # per request, for answers never given (rate limit, dropped connection)
VERSION = "jev-halving-v1"


def questions(target: str, note: str = ""):
    """The two judgments for one task, from the task's own words ("each cow", ...)."""
    thing = target[5:] if target.startswith("each ") else target
    extra = f" {note}" if note else ""
    count = choice(f"Find {target} in the image.{extra} How many are there? "
                   "Count one that is cut off by the edge of the image too.",
                   {"none": "None, not even part of one.",
                    "one": "Exactly one, whole or cut off by the edge.",
                    "several": "Two or more, whole or cut off by the edge."})
    box = bounding_box(f"the {thing} in the image (only its visible part).{extra}")
    return judgments(count=count, box=box)


def fit(crop: Image.Image):
    """Into Jev's limits, keeping the shape. Returns (picture sent, (pad_x, pad_y, inner_w, inner_h))."""
    w, h = crop.size
    k = min(1.0, MAX_SIDE / max(w, h))
    if MIN_LONG and max(w, h) * k < MIN_LONG:
        k = min(MIN_LONG, MAX_SIDE) / max(w, h)
    if min(w, h) * k < MIN_SIDE:
        k = min(MIN_SIDE / min(w, h), MAX_SIDE / max(w, h))
    iw, ih = max(1, round(w * k)), max(1, round(h * k))
    if (iw, ih) != (w, h):
        crop = crop.resize((iw, ih), Image.LANCZOS)
    W, H = max(MIN_SIDE, iw), max(MIN_SIDE, ih)
    if (W, H) == (iw, ih):
        return crop, (0, 0, iw, ih)
    canvas = Image.new("RGB", (W, H), (128, 128, 128))
    px, py = (W - iw) // 2, (H - ih) // 2
    canvas.paste(crop, (px, py))
    return canvas, (px, py, iw, ih)


def unpad(b: dict, sent: Image.Image, pad) -> dict:
    """A box on the picture sent -> fractions of the piece itself (clamped to it)."""
    px, py, iw, ih = pad
    W, H = sent.size
    cl = lambda v: min(1.0, max(0.0, v))  # noqa: E731
    return {"left": cl((b["left"] * W - px) / iw), "right": cl((b["right"] * W - px) / iw),
            "top": cl((b["top"] * H - py) / ih), "bottom": cl((b["bottom"] * H - py) / ih)}


@dataclass
class Search:
    """One search on one picture: its call records and its tree of pieces."""
    im: Image.Image
    q: dict
    workers: int = 16
    records: list = field(default_factory=list)
    nodes: list = field(default_factory=list)
    cuts: list = field(default_factory=list)
    media_type: str = "image/jpeg"

    def __post_init__(self):
        self.pool = ThreadPoolExecutor(self.workers)
        self.im = self.im.convert("RGB")

    def _send(self, piece: Image.Image) -> dict:
        b = io.BytesIO()
        if self.media_type == "image/png":
            piece.save(b, "PNG")
        else:
            piece.save(b, "JPEG", quality=95)
        rec = None
        for attempt in range(ATTEMPTS):
            rec = call(MODEL, b.getvalue(), self.media_type, None, level="medium", response_format=self.q)
            rec.update(sent_size=list(piece.size), attempt=attempt)
            rec.pop("traceback", None)
            self.records.append(rec)
            if not rec.get("error") or not rec.get("retryable"):
                break
            time.sleep(min(30, 2 ** attempt))
        if rec.get("error"):
            raise RuntimeError(f"Jev call failed on a {piece.size} piece: {rec['error']}")
        return rec

    def ask(self, region, depth: int) -> dict:
        x0, y0, x1, y1 = region
        piece, pad = fit(self.im.crop(region))
        rec = self._send(piece)
        data = rec["data"]
        b = unpad(data["box"], piece, pad)
        w, h = x1 - x0, y1 - y0
        node = {"depth": depth, "region": list(region), "sent_size": list(piece.size), "choice": data["count"],
                "p": (rec.get("probabilities") or {}).get("count"), "seconds": rec["seconds"],
                "tokens": rec["usage"]["input_tokens"], "cost_usd": rec["cost_usd"],
                "box": [x0 + b["left"] * w, y0 + b["top"] * h, x0 + b["right"] * w, y0 + b["bottom"] * h],
                "box_confidence": rec["response"]["provider_data"]["typesafe"]["answers"]["box"].get("confidence")}
        self.nodes.append(node)
        return node

    def halve(self, region, cuts) -> list:
        """Level by level from `region`; returns [{"box", "region", "forced"}] in pixels."""
        found, level, depth = [], [tuple(region)], 0
        while level:
            nodes = list(self.pool.map(lambda r: self.ask(r, depth), level))
            nxt = []
            for r, node in zip(level, nodes):
                x0, y0, x1, y1 = r
                if node["choice"] == "none":
                    continue
                if node["choice"] == "one" or depth == MAX_DEPTH:
                    node["forced"] = node["choice"] == "several"
                    found.append({"box": node["box"], "region": list(r), "forced": node["forced"]})
                    continue
                if x1 - x0 >= y1 - y0:
                    m = (x0 + x1) // 2
                    nxt += [(x0, y0, m, y1), (m, y0, x1, y1)]
                    cuts.append(("x", m, y0, y1))
                else:
                    m = (y0 + y1) // 2
                    nxt += [(x0, y0, x1, m), (x0, m, x1, y1)]
                    cuts.append(("y", m, x0, x1))
            level, depth = nxt, depth + 1
        return found

    def recheck(self, joined, rounds=RECHECK_ROUNDS) -> list:
        """Joined boxes asked again, all at once; "several" -> halve that box again and join."""
        w, h = self.im.size

        def one(item):
            box, parts = item
            if parts == 1:
                return [box]
            region = (max(0, int(box[0])), max(0, int(box[1])), min(w, int(box[2]) + 1), min(h, int(box[3]) + 1))
            node = self.ask(region, -1)
            node["recheck"] = True
            if node["choice"] != "several" or rounds == 0:
                return [box]
            sub = self.halve(region, [])
            return self.recheck(join_across_cuts(sub, with_parts=True), rounds - 1)

        with ThreadPoolExecutor(max(1, len(joined))) as groups:
            return [b for bs in groups.map(one, joined) for b in bs]

    def run(self) -> list:
        """Boxes (pixels of the full picture), after joining and rechecking."""
        w, h = self.im.size
        raw = self.halve((0, 0, w, h), self.cuts)
        self.raw = raw
        return self.recheck(join_across_cuts(raw, with_parts=True))


def join_across_cuts(found, min_along=0.5, with_parts=False):
    """Parts of one object cut by the halving, joined back (see the module's step 3).
    Linked between raw parts only, grouped (an object cut in four is joined through
    its neighbours); each group becomes the rectangle around it. Two different
    objects that touch on a cut line with similar spans are joined wrongly."""
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
            ra, rb, a, b = found[i]["region"], found[j]["region"], found[i]["box"], found[j]["box"]
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
    groups: dict = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(found[i]["box"])
    joined = [([min(x[0] for x in g), min(x[1] for x in g), max(x[2] for x in g), max(x[3] for x in g)], len(g))
              for g in groups.values()]
    return joined if with_parts else [b for b, _ in joined]
