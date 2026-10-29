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

# ---------------- second photo batch (2026-10-14): draft questions ----------------
# Photos added 2026-10-14 (images/README.md). NOT in TASKS: the published
# benchmark (v2.x) keeps its four point tasks and runs that default to "all
# tasks" are unchanged. Asked with --tasks KEY... to build a draft answer key
# from model consensus (run consensus-2), for a person to review. The cow
# question was chosen in the 2026-10-14 conversation; the others are proposals,
# approved the same day.
# The mineral photo has no question: nothing in it to point at one by one.
TASKS_BATCH_2: tuple[Task, ...] = (
    Task(7, "ants", "leafcutter_ants_branch.jpg", "each leaf piece carried by an ant", "objects", "each spider"),
    Task(8, "beetles", "butterfly_display_case.jpg", "each beetle (not the butterflies)", "objects", "each bird"),
    Task(9, "birds", "mounted_birds_display.jpg", "each bird", "objects", "each butterfly"),
    Task(10, "mushrooms", "oyster_mushrooms_hands.jpg", "each mushroom cap", "objects", "each strawberry"),
    Task(11, "jars", "strawberry_coulis_jars.jpg",
         "each jar of strawberry coulis (not the jars of other preserves)", "objects", "each mushroom"),
    Task(12, "pinkleaves", "living_plant_wall.jpg", "each pink leaf", "objects", "each cat"),
    Task(13, "whitecow", "cows_under_pine_tree.jpg", "each white cow with a bit of red", "objects", "each horse"),
)
BY_KEY = {t.key: t for t in TASKS + TASKS_BATCH_2}

# The local models' short phrases for the second batch, as for the first
# (specialists/common/answers.py PHRASES): one noun phrase per task, the
# question without "each". Written 2026-10-14 before any local model answered
# on the benchmark (Moondream saw them once, to draft the answer key).
# Open-vocabulary models cannot be told "not the butterflies"; that is part
# of what is measured. Plurals for the models whose documentation asks for
# all instances that way (segbench/local_models.py plural=True).
PHRASES_BATCH_2 = {
    "ants": "leaf piece carried by an ant",
    "beetles": "beetle",
    "birds": "bird",
    "mushrooms": "mushroom cap",
    "jars": "jar of strawberry coulis",
    "pinkleaves": "pink leaf",
    "whitecow": "white cow with a bit of red",
}
PLURAL_BATCH_2 = {
    "leaf piece carried by an ant": "the leaf pieces carried by ants", "beetle": "the beetles", "bird": "the birds",
    "mushroom cap": "the mushroom caps", "jar of strawberry coulis": "the jars of strawberry coulis",
    "pink leaf": "the pink leaves", "white cow with a bit of red": "the white cows with a bit of red",
    # trick questions (TASKS_BATCH_2 absent, without "each")
    "spider": "the spiders", "butterfly": "the butterflies", "strawberry": "the strawberries",
    "mushroom": "the mushrooms", "cat": "the cats", "horse": "the horses",
}

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


# ---------------- prompt "tags" (2026-10-12): Perceptron's own answer format ----------------
# For Perceptron Mk1.5 only (Maxime, 2026-10-12: try it with and without its
# own format). Prompt v2's task words unchanged; only the reply paragraph is
# swapped for the annotation tags Perceptron's docs ask for ("Return one
# point annotation per ...", mention= for labels, "if none are visible, say
# so without inventing" from its detection guide). No coordinate paragraph:
# the tags' 0..1000 grid is the model's own. Written once, checked on the
# synthetic pictures before any benchmark photo. Read by segbench/perceptron.py.
PROMPT_VERSION_TAGS = "tags"


def prompt_tags(track: str, task: Task) -> str:
    if track not in ("find", "trick"):
        raise ValueError("prompt tags covers the find and trick tracks only")
    target = task.target if track == "find" else task.absent
    labelled = track == "find" and bool(task.labels)
    rule = _label_rule(task) if labelled else ""
    note = f"\n{task.note}" if labelled and task.note else ""
    mention = (" Give each label as its mention: " + " or ".join(f'mention="{x}"' for x in task.labels) + "."
               if labelled else "")
    return f"""Find {target} in this photo. Put one point on each one, inside it.{rule}{note}
Mark every one you can see, however many there are. Do not stop early or give only a sample.

Return one point annotation per object.{mention}
If there are none, say so without inventing a point."""


# ---------------- box prompts (2026-10-15): the picture benchmark with boxes ----------------
# Maxime, 2026-10-15: rerun the whole picture benchmark asking for boxes
# instead of points; two prompts per model, each model keeps the better one
# (as for points); no trick questions. Both prompts say "visible part" (the
# answer-key boxes are drawn around the visible part only: hidden parts
# cannot be checked) and ask for one box per object, never one around a
# group (the box form of "one point on each").
# - "box-a": named corners on the 0..1000 grid (named keys because unnamed
#   pairs were read [y, x] by two Gemini models in point v1), JSON reply;
# - "box-free": no answer format and no coordinate system at all (Maxime:
#   "let them choose on their own"); every reply is read by the rules in
#   segbench/box_read.py, fixed on synthetic pictures before any benchmark
#   photo was sent.
# Same words for every model. Checked on synthetic pictures only
# (scripts/check_boxes.py) before any benchmark photo.
PROMPT_VERSION_BOX_A = "box-a"
PROMPT_VERSION_BOX_FREE = "box-free"
BOX_VERSIONS = (PROMPT_VERSION_BOX_A, PROMPT_VERSION_BOX_FREE)

COORDINATES_BOX_A = """x_min, y_min, x_max and y_max are integers from 0 to 1000 over the whole
photo: x is the horizontal position (0 = left edge, 1000 = right edge) and y
the vertical position (0 = top edge, 1000 = bottom edge). They are not pixels:
whatever the photo's size, the right edge is x = 1000 and the bottom edge is
y = 1000."""


def _box_request(task: Task, verb: str) -> str:
    rule = _label_rule(task)
    note = f"\n{task.note}" if task.labels and task.note else ""
    return (f"Find {task.target} in this photo. {verb}: the smallest rectangle that "
            f"contains all of its visible part. Leave out any part that is hidden behind something else. "
            f"Give each object its own box, even when objects touch or overlap; never one box around a group."
            f"{rule}{note}\n"
            "Mark every one you can see, however many there are. Do not stop early or give only a sample.")


def prompt_box(version: str, task: Task) -> str:
    """The box question for one task (find track only: no trick questions in the box benchmark)."""
    if task.kind != "objects":
        raise ValueError(f"{task.key} is a region: nothing to box one by one")
    if version == PROMPT_VERSION_BOX_FREE:
        return _box_request(task, "Give the bounding box of each one")
    if version != PROMPT_VERSION_BOX_A:
        raise ValueError(f"unknown box prompt {version!r}")
    corners = '"x_min": 0, "y_min": 0, "x_max": 0, "y_max": 0'
    item = f'{{{corners}, "label": "..."}}' if task.labels else f"{{{corners}}}"
    return f"""{_box_request(task, "Draw one box around each one")}

Reply with JSON only, in this shape:
{{"objects": [{item}, ...]}}
If there are none, reply {{"objects": []}}.

{COORDINATES_BOX_A}"""


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
