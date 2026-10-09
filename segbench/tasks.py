"""The six tasks and the exact words sent to the models, per track.

Prompts are versioned: changing any word below means a new PROMPT_VERSION,
never an edit in place, because every result line records the version it was
asked with. Nothing here comes from the answer key, so tracks "find",
"whole" and "trick" can run before the answer key exists; their replies are
scored later. Track "outline" needs the answer key's dots and is not here yet.

Tracks (README, "three tracks"):
- find:  one point on each object (with a label for dishes).      tasks 1, 2, 3, 6
- whole: outlines (polygons), no hints.                            all six tasks
- trick: the find request for something absent; right answer: [].  all six photos

Coordinates are 0..1000 of the photo's width and height, written [x, y].
Gemini models tend to write [y, x] anyway (README, open decision 2); replies
are stored raw, so either reading can be scored later.
"""

from __future__ import annotations

from dataclasses import dataclass

PROMPT_VERSION = "v1"  # 2026-10-09

COORDINATES = """Coordinates are integers from 0 to 1000 over the whole photo: x = 0 is the
left edge and 1000 the right edge, y = 0 is the top edge and 1000 the bottom
edge. Write every point as [x, y]."""

DISH_DEFINITION = ("Dishes here means plates, bowls, mugs, pots, pans, baking trays, lids, "
                   "cutting boards and food containers; not cutlery, knives, bottles, "
                   "appliances or cloths.")


@dataclass(frozen=True)
class Task:
    number: int
    key: str
    photo: str  # file in images/
    target: str  # what to segment, as written in the prompt
    kind: str  # "objects" (separate things, counted) or "region" (one area, maybe in pieces)
    absent: str  # the trick question's target: clearly not in the photo
    note: str = ""  # extra sentence after the request (definitions)
    labels: tuple[str, ...] = ()  # labels the model must choose from, if any


TASKS: tuple[Task, ...] = (
    Task(1, "logs", "log_ends_closeup_1.jpg", "each cut log end", "objects", "each cow"),
    Task(2, "cows", "field_with_cows.jpg", "each cow", "objects", "each tractor"),
    Task(3, "fig", "fig_plant.jpg",
         "each fig leaf (not the leaves of the spider plant or of the purple plant)", "objects", "each car"),
    Task(4, "tree", "autumn_tree_orange_1.jpg",
         "the reddish foliage of the tree (not the yellow foliage)", "region", "each person"),
    Task(5, "road", "farm_road_1.jpg", "the gravel road", "region", "each cow"),
    Task(6, "dishes", "kitchen_counter_dishes.jpg", "each dish", "objects", "each cat",
         note=DISH_DEFINITION, labels=("dirty", "clean")),
)
BY_KEY = {t.key: t for t in TASKS}

TRACKS = ("find", "whole", "trick")

# ---------------- prompt v2 (2026-10-09): points only ----------------
# Written to measure pointing, not format-following (README; docs/point-benchmark-v1.md):
# - coordinates named ("x", "y") instead of an unlabelled pair: in v1 two
#   Gemini models wrote [y, x] although asked for [x, y];
# - "mark every one ... do not stop early": in v1 three models stopped at
#   exactly 50 of 83 log ends with room left to answer;
# - the same words for every model; trick questions use the same template.
# Checked on synthetic pictures only (scripts/check_points_v2.py) before any
# benchmark photo was sent.
PROMPT_VERSION_V2 = "v2"

COORDINATES_V2 = """x and y are integers from 0 to 1000 over the whole photo: x is the horizontal
position (0 = left edge, 1000 = right edge) and y the vertical position
(0 = top edge, 1000 = bottom edge). They are not pixels: whatever the photo's
size, the right edge is x = 1000 and the bottom edge is y = 1000."""
# The last sentence was added after the synthetic check (scripts/check_points_v2.py):
# DeepSeek Flash estimated positions on 0..1000, then divided them again as if
# they were pixels. Added for every model, checked again on synthetic pictures.


def prompt_v2(track: str, task: Task) -> str:
    if track not in ("find", "trick"):
        raise ValueError("prompt v2 covers the find and trick tracks only")
    target = task.target if track == "find" else task.absent
    labelled = track == "find" and bool(task.labels)
    rule = _label_rule(task) if labelled else ""
    note = f"\n{task.note}" if labelled and task.note else ""
    item = '{"x": 0, "y": 0, "label": "..."}' if labelled else '{"x": 0, "y": 0}'
    return f"""Find {target} in this photo. Put one point on each one, inside it.{rule}{note}
Mark every one you can see, however many there are. Do not stop early or give only a sample.

Reply with JSON only, in this shape:
{{"objects": [{item}, ...]}}
If there are none, reply {{"objects": []}}.

{COORDINATES_V2}"""


def applies(track: str, task: Task) -> bool:
    """Regions have nothing separate to find: they are only asked whole."""
    return not (track == "find" and task.kind == "region")


def _label_rule(task: Task) -> str:
    if not task.labels:
        return ""
    names = " or ".join(f'"{x}"' for x in task.labels)
    return f" Label each one {names}."


def prompt(track: str, task: Task) -> str:
    if track in ("find", "trick"):
        target = task.target if track == "find" else task.absent
        labelled = track == "find" and task.labels
        rule = _label_rule(task) if labelled else ""
        note = f"\n{task.note}" if track == "find" and task.note else ""
        item = '{"point": [x, y], "label": "..."}' if labelled else '{"point": [x, y]}'
        return f"""Find {target} in this photo. Put one point on each one, inside it.{rule}{note}

Reply with JSON only, no other text, in this shape:
{{"objects": [{item}, ...]}}
If there are none, reply {{"objects": []}}.

{COORDINATES}"""

    if track == "whole" and task.kind == "objects":
        rule = _label_rule(task)
        note = f"\n{task.note}" if task.note else ""
        item = '{"polygon": [[x, y], [x, y], ...], "label": "..."}' if task.labels else '{"polygon": [[x, y], [x, y], ...]}'
        return f"""Segment {task.target} in this photo: trace the outline of each one.{rule}{note}

Reply with JSON only, no other text, in this shape:
{{"objects": [{item}, ...]}}
Use at least 8 points per polygon. If there are none, reply {{"objects": []}}.

{COORDINATES}"""

    if track == "whole" and task.kind == "region":
        return f"""Segment {task.target} in this photo: trace its outline.

Reply with JSON only, no other text, in this shape:
{{"regions": [{{"polygon": [[x, y], [x, y], ...]}}, ...]}}
If it is in several separate pieces, give one polygon per piece. Use at least
8 points per polygon. If there is none, reply {{"regions": []}}.

{COORDINATES}"""

    raise ValueError(f"no prompt for track {track!r} on task {task.key!r}")
