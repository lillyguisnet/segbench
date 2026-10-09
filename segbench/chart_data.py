"""From one record per call to one point per bubble on the charts.

No third-party packages here, so the benchmark runner can import it too.
The drawing itself is in segbench/chart.py.

What a call record must hold (one JSON line in results/, written by the
runner; extra fields are ignored):

    track      "find" | "outline" | "whole"
    task       task name, e.g. "cows" (the tasks of a track are all the
               task names seen in that track)
    model      model key: segbench.models key ("gemini-flash") or a
               specialist key from SPECIALISTS below ("sam2.1-large")
    level      thinking level ("min", "medium", "max"), or null
    parts      pairs only: [{"model": ..., "level": ...}, {"model": ...}]
               in order finder, outliner; "model" may then be omitted
    repeat     0, 1, 2, ...
    score      0..1, or null for a timing-only call (specialists are
               scored once and timed several times)
    readable   false when the reply could not be read: scores 0
    cost_usd   cost of this call (all parts of a pair), or null if unknown
    seconds    wall-clock seconds for this call (all parts of a pair)
    synthetic  true for made-up demo data: the chart then says so

How a bubble is computed (README, "Scores" and "The charts"):

    quality   mean over the track's tasks of the mean score over repeats;
              a task the entrant did not attempt counts as 0 and its
              label gets a mark
    spread    lowest and highest quality over repeats (the same mean,
              taken one repeat at a time)
    cost      mean cost per call, i.e. per image
    time      median seconds per call
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from segbench.models import BY_KEY, LEVELS

TRACKS = ("find", "outline", "whole")

# Models run on our own GPU: chart label and maker.
SPECIALISTS: dict[str, tuple[str, str]] = {
    "sam1-vit-h": ("SAM 1", "Meta"),
    "sam2.1-large": ("SAM 2.1", "Meta"),
    "sam3": ("SAM 3", "Meta"),
    "dinov3": ("DINOv3", "Meta"),
    "yoloe-26x": ("YOLOE-26x", "Ultralytics"),
    "rfdetr-seg-2xl": ("RF-DETR Seg", "Roboflow"),
    "gen2seg-sd": ("gen2seg", "UC Davis"),
}

LEVEL_SHORT = {"min": "min", "medium": "med", "max": "max"}


@dataclass
class Point:
    track: str
    entrant: str  # unique within a track, e.g. "gemini-flash@medium" or "gemini-flash@medium+sam2.1-large"
    label: str  # shown on the chart
    kind: str  # "general" | "specialist" | "pair"
    maker: str  # of the model, or of the finder for a pair
    model: str  # model key; for a pair, the finder's
    level: str  # thinking level, "" when none
    outliner: str  # pairs only: the outliner's model key
    outliner_maker: str
    quality: float  # 0..1
    quality_low: float
    quality_high: float
    cost_usd: float  # per image
    seconds: float  # median per image
    calls: int
    repeats: int
    tasks_done: int
    tasks_total: int
    synthetic: bool


def describe(model: str) -> tuple[str, str, str]:
    """(label, maker, kind) of a model key."""
    if model in BY_KEY:
        m = BY_KEY[model]
        return m.label, m.maker, "general"
    if model in SPECIALISTS:
        label, maker = SPECIALISTS[model]
        return label, maker, "specialist"
    return model, "Other", "specialist"


def _key(model: str, level: str | None) -> str:
    return f"{model}@{level}" if level else model


def entrant_of(r: dict) -> tuple[str, dict]:
    """The entrant key of a call record and what to show for it."""
    parts = r.get("parts")
    if parts:
        (finder, outliner) = parts
        f_label, f_maker, _ = describe(finder["model"])
        o_label, o_maker, _ = describe(outliner["model"])
        level = finder.get("level") or ""
        key = _key(finder["model"], level) + "+" + _key(outliner["model"], outliner.get("level"))
        label = f"{f_label} + {o_label}"
        return key, dict(label=label, kind="pair", maker=f_maker, model=finder["model"], level=level,
                         outliner=outliner["model"], outliner_maker=o_maker)
    label, maker, kind = describe(r["model"])
    level = r.get("level") or ""
    return _key(r["model"], level), dict(label=label, kind=kind, maker=maker, model=r["model"], level=level,
                                         outliner="", outliner_maker="")


def read_calls(paths: list[Path]) -> list[dict]:
    calls = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("track") in TRACKS:
                    calls.append(r)
    return calls


def _score(r: dict) -> float | None:
    if r.get("readable") is False:
        return 0.0
    return r.get("score")


def aggregate(calls: list[dict]) -> list[Point]:
    """One Point per (track, entrant)."""
    tasks = defaultdict(set)  # track -> task names
    groups = defaultdict(list)  # (track, entrant) -> calls
    info = {}
    for r in calls:
        tasks[r["track"]].add(r["task"])
        key, meta = entrant_of(r)
        groups[r["track"], key].append(r)
        info[key] = meta

    points = []
    for (track, key), rs in groups.items():
        track_tasks = sorted(tasks[track])
        by_task = defaultdict(list)  # task -> [(repeat, score)]
        for r in rs:
            s = _score(r)
            if s is not None:
                by_task[r["task"]].append((r.get("repeat", 0), s))
        if not by_task:
            continue  # timing only: nothing to place on the quality axis

        task_means = [statistics.fmean(s for _, s in by_task[t]) if by_task[t] else 0.0 for t in track_tasks]
        quality = statistics.fmean(task_means)

        # Spread: the same mean, one repeat at a time. A task with fewer
        # repeats (a deterministic specialist scored once) keeps its mean.
        repeats = sorted({rep for t in by_task for rep, _ in by_task[t]})
        per_repeat = []
        for rep in repeats:
            vals = []
            for t, mean in zip(track_tasks, task_means, strict=True):
                at = [s for rp, s in by_task.get(t, []) if rp == rep]
                vals.append(statistics.fmean(at) if at else mean)
            per_repeat.append(statistics.fmean(vals))

        costs = [r["cost_usd"] for r in rs if r.get("cost_usd") is not None]
        secs = [r["seconds"] for r in rs if r.get("seconds") is not None]
        if not costs or not secs:
            continue  # cannot be placed on the cost axis or sized

        points.append(Point(
            track=track, entrant=key, **info[key],
            quality=quality, quality_low=min(per_repeat), quality_high=max(per_repeat),
            cost_usd=statistics.fmean(costs), seconds=statistics.median(secs),
            calls=len(rs), repeats=len(repeats),
            tasks_done=sum(1 for t in track_tasks if by_task.get(t)), tasks_total=len(track_tasks),
            synthetic=any(r.get("synthetic") for r in rs),
        ))
    level_rank = {lv: i for i, lv in enumerate(LEVELS)}
    points.sort(key=lambda p: (TRACKS.index(p.track), p.kind, p.model, level_rank.get(p.level, -1), p.entrant))
    return points


def frontier(points: list[Point]) -> list[Point]:
    """Best-value points: no other point is at least as cheap and better.

    Ties in cost keep the better one; ties in quality keep the cheaper one.
    Returned from cheapest to most expensive.
    """
    best = []
    top = float("-inf")
    for p in sorted(points, key=lambda p: (p.cost_usd, -p.quality)):
        if p.quality > top:
            best.append(p)
            top = p.quality
    return best


def write_points(points: list[Point], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[fl.name for fl in fields(Point)])
        w.writeheader()
        for p in points:
            w.writerow(asdict(p))


def read_points(path: Path) -> list[Point]:
    types = {fl.name: fl.type for fl in fields(Point)}
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            vals = {}
            for k, v in row.items():
                t = types[k]
                vals[k] = (v == "True") if t == "bool" else int(v) if t == "int" else float(v) if t == "float" else v
            out.append(Point(**vals))
    return out
