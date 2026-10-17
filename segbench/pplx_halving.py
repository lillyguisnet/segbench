"""pplx-flash + halving: finding every object of a kind with a local decision model that gives no boxes.

Perplexity's pplx-decider-flash-v1.2 (4B, Apache 2.0; run by ~/Projects/deciders on
our own GPU) answers only judgments: choice, yes/no, score. No box, no point. The
search around it (2026-10-17), written and checked on the synthetic ovals only
before any benchmark photo was sent:

1. Count by halving, as Jev + halving does, with one change: an object is counted
   in the piece that holds its **middle** (Jev: every piece that shows part of it).
   Each object then belongs to one piece only, so no parts have to be joined back
   (joining needed boxes), and a piece showing only edges of neighbours can say
   "none" (Jev's splitting never stopped on the log pile for that reason).
   - none     -> stop;
   - one      -> locate it (step 2);
   - several  -> cut the piece in two across its longer side, ask both halves.
   All pieces of one depth are asked together. At MAX_DEPTH a "several" is located
   once anyway (flagged `forced`).
2. Locate by zooming: show a window (first the piece itself), ask which ninth of
   it (top left ... bottom right, in words; nothing is drawn on the picture) holds
   the object's middle, then show a window half as wide and high centred on that
   ninth (kept inside the piece), and ask again. Stop when a ninth is at most
   STOP_FRACTION of the photo's diagonal; the point is the middle of the last
   ninth chosen. The first question names the object whose middle is in the
   piece; the later ones the object nearest the centre of the window.
3. Every picture is the crop of the full-size photo, as a lossless PNG; the
   model's own processor resizes it (at most 1.6 MP, at least 65,536 pixels).

What it returns: points in pixels of the full picture (no boxes).

Requests are sent to one `deciders decide - --lines` process (the model's own
reference path, bf16), a list of requests per line, so all pieces of a depth
form one batch. One question per request (the model card's advice for this
model).
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

DECIDERS = Path.home() / "Projects" / "deciders" / "deciders"
MODEL_KEY = "pplx-decider-flash-v1.2"
ROUTE = "local:perplexity-ai/pplx-decider-flash-v1.2@6d771ad7b5ecff5e93b8fc7b90d4c5ccc341954d"
VERSION = "pplx-halving-v1"
MAX_DEPTH = 12
STOP_FRACTION = 0.01  # a ninth this small (of the photo's diagonal) ends the zoom
MAX_STEPS = 8         # zoom steps per object, a backstop only

NINTHS = [("top left", 0, 0), ("top middle", 1, 0), ("top right", 2, 0),
          ("middle left", 0, 1), ("centre", 1, 1), ("middle right", 2, 1),
          ("bottom left", 0, 2), ("bottom middle", 1, 2), ("bottom right", 2, 2)]


def _thing(target: str) -> str:
    return target[5:] if target.startswith("each ") else target


def count_question(target: str, note: str = "") -> dict:
    extra = f" {note}" if note else ""
    return {"type": "choice",
            "instructions": (f"Find {target} in the image.{extra} How many of them have their middle inside "
                             "this image? One cut off by the edge counts only if its middle is inside the image."),
            "criteria": {"none": "None has its middle inside the image.",
                         "one": "Exactly one has its middle inside the image.",
                         "several": "Two or more have their middle inside the image."}}


def locate_question(target: str, note: str = "", first: bool = True) -> dict:
    extra = f" {note}" if note else ""
    which = ("the one whose middle is inside this image" if first
             else "the one nearest the centre of this image")
    return {"type": "choice",
            "instructions": (f"Find {target} in the image.{extra} Take {which}. "
                             f"In which ninth of the image is the middle of that {_thing(target)}?"),
            "criteria": {name: f"the {name} ninth of the image (3 by 3 grid)" for name, _, _ in NINTHS}}


class Decider:
    """One `deciders decide - --lines` process: send a list of requests, get the list of responses."""

    def __init__(self, gpu: str = "0", log: Path | None = None):
        self.log = open(log or os.devnull, "a")
        self.proc = subprocess.Popen([str(DECIDERS), "decide", "-", "--lines", "--gpu", gpu],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                     text=True, bufsize=1)
        self.tmp = tempfile.TemporaryDirectory(prefix="pplx-halving-")
        self.n = 0

    def save(self, im: Image.Image) -> str:
        self.n += 1
        path = os.path.join(self.tmp.name, f"{self.n}.png")
        im.save(path, "PNG", compress_level=1)
        return path

    def decide(self, requests: list[dict]) -> list[dict]:
        if not requests:
            return []
        self.proc.stdin.write(json.dumps(requests) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"the decider process ended (exit {self.proc.poll()}); see its log")
        return json.loads(line)

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait(timeout=60)
        self.tmp.cleanup()
        self.log.close()


@dataclass
class Search:
    """One search on one picture: its requests, its tree of pieces and its zoom steps."""
    decider: Decider
    im: Image.Image
    target: str
    note: str = ""
    nodes: list = field(default_factory=list)
    cuts: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    requests: int = 0
    tokens: int = 0

    def __post_init__(self):
        self.im = self.im.convert("RGB")
        self.diag = math.hypot(*self.im.size)

    def _ask(self, regions, questions) -> list[dict]:
        reqs = [{"state": "An image.", "images": [self.decider.save(self.im.crop(tuple(map(round, r))))],
                 "questions": {"q": q}} for r, q in zip(regions, questions)]
        out = self.decider.decide(reqs)
        self.requests += len(reqs)
        self.tokens += sum(o["usage"]["input_tokens"] for o in out)
        return [o["answers"]["q"] for o in out]

    def halve(self) -> list[dict]:
        w, h = self.im.size
        level, depth, found = [(0, 0, w, h)], 0, []
        q = count_question(self.target, self.note)
        while level:
            answers = self._ask(level, [q] * len(level))
            nxt = []
            for r, a in zip(level, answers):
                x0, y0, x1, y1 = r
                self.nodes.append({"depth": depth, "region": list(r), "choice": a["choice"],
                                   "p": a["probabilities"]})
                if a["choice"] == "none":
                    continue
                if a["choice"] == "one" or depth == MAX_DEPTH or min(x1 - x0, y1 - y0) < 2:
                    found.append({"region": list(r), "forced": a["choice"] == "several"})
                    continue
                if x1 - x0 >= y1 - y0:
                    m = (x0 + x1) // 2
                    nxt += [(x0, y0, m, y1), (m, y0, x1, y1)]
                    self.cuts.append(("x", m, y0, y1))
                else:
                    m = (y0 + y1) // 2
                    nxt += [(x0, y0, x1, m), (x0, m, x1, y1)]
                    self.cuts.append(("y", m, x0, x1))
            level, depth = nxt, depth + 1
        return found

    def locate(self, found: list[dict]) -> list[dict]:
        """All objects zoomed in step by step together (one batch per step)."""
        live = [{**f, "window": list(f["region"]), "trail": []} for f in found]
        for step in range(MAX_STEPS):
            todo = [o for o in live if "point" not in o]
            if not todo:
                break
            q_first, q_next = locate_question(self.target, self.note, True), locate_question(self.target, self.note, False)
            answers = self._ask([o["window"] for o in todo], [q_first if step == 0 else q_next for _ in todo])
            for o, a in zip(todo, answers):
                x0, y0, x1, y1 = o["window"]
                ww, wh = x1 - x0, y1 - y0
                name = a["choice"]
                c, r = next((c, r) for n, c, r in NINTHS if n == name)
                cx, cy = x0 + (c + 0.5) * ww / 3, y0 + (r + 0.5) * wh / 3
                o["trail"].append({"window": [x0, y0, x1, y1], "ninth": name, "p": a["probabilities"][name]})
                if max(ww, wh) / 3 <= STOP_FRACTION * self.diag or step == MAX_STEPS - 1:
                    o["point"] = [cx, cy]
                    continue
                px0, py0, px1, py1 = o["region"]
                nw, nh = max(ww / 2, 2), max(wh / 2, 2)
                nx0 = min(max(cx - nw / 2, px0), px1 - nw)
                ny0 = min(max(cy - nh / 2, py0), py1 - nh)
                o["window"] = [nx0, ny0, nx0 + nw, ny0 + nh]
        self.steps = [{"region": o["region"], "forced": o["forced"], "trail": o["trail"], "point": o["point"]}
                      for o in live]
        return self.steps

    def run(self) -> list[dict]:
        """[{"point": [x, y] in pixels, "region", "forced", "trail"}]."""
        return self.locate(self.halve())
