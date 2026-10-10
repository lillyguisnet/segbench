"""Point answers (find and trick tracks), ready to draw: one answer per model and photo.

Shared by the point overlays (scripts/draw_points.py), the points site
(scripts/build_site.py) and the points video (scripts/make_video.py), so the
three read answers the same way and give every model the same colour.
No third-party packages.

Replies are read exactly as the scorer reads them (segbench/point_score.py
read_points: strict JSON, else the coordinates as written). A reply that
cannot be read even so is kept as `unreadable`, never guessed at. Points are kept as
written, 0..1000 of the photo's width and height: no [y, x] swapping, no
clipping (a point off the photo is counted in `off_photo`).
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from segbench.point_score import read_points, timed_on_free_gpu
from segbench.brand import BRAND, OTHER
from segbench.chart_data import describe
from segbench.models import LEVELS
from segbench.tasks import BY_KEY as TASKS

ROOT = Path(__file__).resolve().parents[1]
# The answers the published chart is scored from (point benchmark v2.2,
# scripts/select_best_prompt.py): each language model at its better prompt,
# 3 tries; specialists timed on the free GPU. Find track only.
RUNS = ("results/run-points-best.jsonl",)

MAKER_ORDER = list(BRAND)
# One maker's models are told apart by shades of its colour, darker to
# lighter in legend order; the lightest still carries a white logo.
SHADES = (-0.40, 0.28)


@dataclass
class Answer:
    key: str  # model key, plus "@level" when one model ran at several levels
    model: str
    level: str | None
    label: str  # name to show
    maker: str
    kind: str  # "general" or "specialist"
    task: str
    points: list[dict] = field(default_factory=list)  # {"x", "y"} in 0..1000, plus "label" for dishes
    seconds: float | None = None
    cost_usd: float | None = None
    timing_trustworthy: bool = True  # False: our GPU was shared during the call (specialists, 2026-10-09)
    unreadable: str = ""  # why the reply could not be read; "" when it was
    raw: str = ""  # the start of an unreadable reply, to show it
    prompt: str = ""  # what the model was sent (a specialist: its text prompt)
    started_at: str = ""

    @property
    def off_photo(self) -> int:
        return sum(not (0 <= p["x"] <= 1000 and 0 <= p["y"] <= 1000) for p in self.points)


def shade(colour: str, t: float) -> str:
    """t < 0 mixes toward black, t > 0 toward white."""
    rgb = [int(colour[i:i + 2], 16) for i in (1, 3, 5)]
    target = 0 if t < 0 else 255
    return "#" + "".join(f"{round(c + (target - c) * abs(t)):02x}" for c in rgb)


def order_key(a: Answer) -> tuple:
    rank = {lv: i for i, lv in enumerate(LEVELS)}
    return (MAKER_ORDER.index(a.maker) if a.maker in MAKER_ORDER else len(MAKER_ORDER), a.label,
            rank.get(a.level or "", 0))


def colours(answers: list[Answer]) -> dict[str, str]:
    """key -> disc colour, the same on every photo: brand colour, shaded within a maker."""
    by_maker: dict[str, list[str]] = defaultdict(list)
    for a in sorted(answers, key=order_key):
        if a.key not in by_maker[a.maker]:
            by_maker[a.maker].append(a.key)
    out = {}
    for maker, keys in by_maker.items():
        base = BRAND.get(maker, OTHER)
        for i, key in enumerate(keys):
            out[key] = base if len(keys) == 1 else shade(base, SHADES[0] + (SHADES[1] - SHADES[0]) * i / (len(keys) - 1))
    return out


def load(runs=RUNS, track: str = "find", repeat: int = 0, only: set[str] | None = None) -> list[Answer]:
    """Every point answer in the call records, in legend order. Pairs are left out: their finder answers alone."""
    rows = []
    for run in runs:
        path = Path(run) if Path(run).is_absolute() else ROOT / run
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r["track"] != track or r.get("repeat", 0) != repeat or "+" in r["model"]:
                continue
            if only and r["model"] not in only:
                continue
            rows.append(r)
    levels = defaultdict(set)
    for r in rows:
        levels[r["model"]].add(r.get("level"))

    answers = []
    for r in rows:
        task = TASKS[r["task"]]
        label, maker, kind = describe(r["model"])
        level = r.get("level")
        several = len(levels[r["model"]]) > 1
        a = Answer(key=f"{r['model']}@{level}" if several else r["model"], model=r["model"], level=level,
                   label=f"{label} ({level})" if several else label, maker=maker, kind=kind, task=r["task"],
                   seconds=r.get("seconds"), cost_usd=r.get("cost_usd"),
                   timing_trustworthy=r.get("timing_trustworthy", True) is not False or timed_on_free_gpu(r),
                   prompt=r.get("prompt", ""),
                   started_at=r.get("started_at", ""))
        if "error" in r or not r.get("text"):
            a.unreadable = "no reply"
        else:
            try:
                labelled = r["track"] == "find" and bool(task.labels)
                for item in read_points(r["text"], labelled=labelled, parser="lenient")[0]:
                    x, y = item["point"]
                    a.points.append({"x": x, "y": y, **({"label": item["label"]} if "label" in item else {})})
            except (ValueError, KeyError, TypeError):
                a.unreadable = "reply not readable"
                a.raw = r["text"][:400]
        answers.append(a)
    return sorted(answers, key=lambda a: (TASKS[a.task].number, order_key(a)))
