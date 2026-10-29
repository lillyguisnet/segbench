"""Outline from a dot, scored by one measurement (draft v1, 2026-10-16).

Each question names a part ("the wing") on one photo. Maxime clicks the
reference in apps/measure-clicks: a line along the middle of the part
("line") or two points ("span"). The red dot the models will see is placed
in the middle of those clicks (half-way along a line, the midpoint of a span)
unless he moves it. The models are then asked, with prompt D of
scripts/probe_point_segment.py plus the part's name, to outline the part the
dot is on; a fixed rule turns their outline into the same measurement.
Ideas and reasons: docs/segment-measure-ideas.md.

Left out of v1 on purpose:
- the faucet itself: the cloth hides its bend, so its visible part is two
  pieces and has no single centre line (its lever handle is kept);
- the swan's neck: white head on a white neck, no clear end;
- coulis height / jar height: two measurements in one question (later).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    key: str
    photo: str
    title: str
    part: str  # the words in the prompt: "Outline the {part} the red dot is on"
    kind: str  # "line": along the middle, 2+ clicks; "span": exactly 2 clicks
    how: str  # how to click the reference
    suggest: int  # how many instances we would like
    note: str = ""


QUESTIONS: tuple[Question, ...] = (
    # Mounted birds: crisp shapes on a white wall.
    Question("birds-wing", "mounted_birds_display.jpg", "Bird wings", "wing", "line",
             "Along the middle of the wing, from where it joins the body to the tip of the longest feather. "
             "4 to 8 clicks, then Enter.", 14,
             "Only wings you can see from base to tip."),
    Question("birds-neck", "mounted_birds_display.jpg", "Goose necks", "neck", "line",
             "Along the middle of the neck, from just behind the head (the white chin strap) to where the black "
             "meets the brown breast. 3 to 6 clicks, then Enter.", 2),
    Question("birds-beak", "mounted_birds_display.jpg", "Beaks", "beak", "span",
             "Where the feathers start at the base of the beak, then the beak's tip.", 5,
             "Small on this photo: zoom in."),
    Question("birds-wingspan", "mounted_birds_display.jpg", "Wingspans", "bird", "span",
             "The two wing tips.", 8,
             "Only birds with both wing tips visible. The dot starts between the tips: drag it onto the bird's body."),
    # Kitchen.
    Question("kitchen-handle", "kitchen_counter_dishes.jpg", "Faucet lever", "lever handle of the faucet", "line",
             "Along the middle of the lever, from where it leaves the faucet's body to its tip. 3 to 5 clicks, then Enter.", 1),
    Question("kitchen-knife", "kitchen_counter_dishes.jpg", "Knives", "knife", "span",
             "The tip of the blade, then the end of the handle.", 2),
    Question("kitchen-dish-width", "kitchen_counter_dishes.jpg", "Bowl and plate widths", "dish", "span",
             "The rim's leftmost point, then its rightmost point.", 6,
             "Only dishes whose whole rim is visible."),
    # Butterfly case.
    Question("butterfly-wingspan", "butterfly_display_case.jpg", "Butterfly wingspans", "butterfly", "span",
             "The tips of the two front wings.", 16),
    Question("butterfly-tail", "butterfly_display_case.jpg", "Swallowtail tails", "tail of the hind wing", "line",
             "Along the middle of the tail, from where it leaves the wing to its tip. 2 to 4 clicks, then Enter.", 8,
             "About 20 px long: a hard set on purpose."),
    # Ants.
    Question("ants-leaf", "leafcutter_ants_branch.jpg", "Leaf pieces", "leaf piece", "span",
             "The two points of the leaf piece farthest apart.", 13),
    Question("ants-branch", "leafcutter_ants_branch.jpg", "Side branch", "branch", "line",
             "Along the middle of the thin lower branch, from where it leaves the main branch to its end. "
             "4 to 8 clicks, then Enter.", 1),
    # Farm road.
    Question("road1-pole", "farm_road_1.jpg", "Utility poles (portrait)", "utility pole", "span",
             "Along the pole's middle: where it enters the ground, then its top (or where it leaves the photo).", 2,
             "The angle is measured too (how much it leans)."),
    Question("road1-wire", "farm_road_1.jpg", "Overhead wires", "wire", "line",
             "Along the wire, from where it enters the photo (or a pole) to where it leaves (or the next pole). "
             "5 to 10 clicks, then Enter.", 3,
             "Very thin: a stress test. Skip a wire you cannot follow."),
    Question("road2-pole", "farm_road_2.jpg", "Utility poles (landscape)", "utility pole", "span",
             "Along the pole's middle: where it enters the ground, then its top (or where it leaves the photo).", 2),
    # Cows under the pine.
    Question("pine-branch", "cows_under_pine_tree.jpg", "Leaning dead branch", "dead branch", "line",
             "Along the middle of the pale dead branch leaning from the trunk, from the trunk to its end. "
             "4 to 8 clicks, then Enter.", 1),
    Question("pine-cow", "cows_under_pine_tree.jpg", "Cow lengths", "cow", "span",
             "The nose, then the base of the tail (not the tail's tip).", 4,
             "Only cows seen from the side."),
    # Logs.
    Question("logs-diameter", "log_ends_closeup_1.jpg", "Log-end diameters", "cut end of the log", "span",
             "Across the cut end through its centre, at its widest: outer edge to outer edge, bark ring included.", 20),
    Question("barn-log-length", "log_pile_barn_1.jpg", "Loose log lengths", "log", "span",
             "The log's two ends, on its middle line.", 8,
             "Only logs lying lengthwise in the loose pile on the right, with both ends visible."),
    # Jars, mushrooms, leaves.
    Question("jars-lid", "strawberry_coulis_jars.jpg", "Jar lids", "lid", "span",
             "The lid's leftmost point, then its rightmost point.", 6),
    Question("mushroom-cap", "oyster_mushrooms_hands.jpg", "Mushroom caps", "mushroom cap", "span",
             "The cap's widest width, edge to edge.", 10,
             "Only caps whose whole edge you can see."),
    Question("wall-pinkleaf", "living_plant_wall.jpg", "Pink leaves", "pink leaf", "span",
             "The leaf's tip, then where the leaf meets its stalk.", 12),
    Question("fig-leaf", "fig_plant.jpg", "Fig leaves", "fig leaf", "span",
             "The tip of the middle lobe, then where the leaf meets its stalk.", 8,
             "Only leaves seen flat enough that both points are visible."),
)

BY_KEY = {q.key: q for q in QUESTIONS}

PROMPT = (
    "There is a red dot on this picture, at x = {x}, y = {y} in pixels "
    "(the picture is {w} x {h} pixels; x counts from the left edge, y from the top edge).\n\n"
    "First look closely at exactly what is under the red dot, and say in one sentence what it is on. "
    "Then segment the {part} the red dot is on: outline it, the visible part only. "
    "Give the outline as a polygon going around its edge, in pixels of this picture, "
    'as JSON: [{{"x": .., "y": ..}}, {{"x": .., "y": ..}}, ...]'
)
