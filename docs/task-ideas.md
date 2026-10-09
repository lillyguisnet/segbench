# Task ideas for the 18 photos (draft, 2026-10-09)

Each task is asked as a segmentation ("segment every …"). A small scoring
function then turns the masks into a **real-world number** and compares it
with a ground truth we make by hand. Proposals; nothing here is built yet.

## How ground truth is made (cheap and unambiguous)

- **One click per object** for everything we count (a dot in the middle of
  each log end, each cow, each leaf). Fast to make, and it lets us score
  more than the count: each click must fall inside exactly one predicted
  mask. A model that says "14 cows" by drawing 14 blobs in the grass gets
  the count right and the matching wrong. Scores: count error, and
  precision / recall of the matching.
- **Colour rules** where the truth is physical: blue sky seen through a
  tree is sky, whatever a model draws. A pixel rule (written once, checked
  by eye) gives a pixel-exact reference.
- **A line or a few points** for measurements (the axis of a pole, the
  edge of a road).
- **Same scene, several photos** (logs ×4 + ×2 close-ups, tree ×2, road
  ×2, sunset ×2): the answer should be the same in each. That gives a
  **consistency** score that needs no ground truth at all.

## Tasks

| photo(s) | ask the model to segment | real-world number | ground truth | difficulty |
|---|---|---|---|---|
| `log_ends_closeup_1/2` | each cut log end | **number of logs**; size distribution (each end's diameter relative to the largest) | click per log end (~70) | many touching circles |
| `log_ends_closeup_1/2` | the white fungus on the log ends | **share of logs with rot** (firewood quality) | per-log yes/no from the clicks above | small, scattered |
| `log_pile_barn_1–4` | each log end in the big pile | **log count** (firewood stock); same pile in 4 photos, 2 phones, 2 lights | clicks on the clearest photo; consistency across the 4 | hundreds of small ends |
| `log_pile_barn_1–4` | each log in the loose pile on the right | **count** (~15) | clicks | logs lie lengthwise, overlap |
| `field_with_cows` | each cow | **head count** (livestock counting); one brown cow among black ones | clicks (~15) | tiny, distant |
| `fig_plant` | each fig leaf (not the spider plant, not the purple plant) | **fig leaf count**; leaf area share (plant growth) | clicks per fig leaf | three plant species mixed |
| `kitchen_counter_dishes` | each dirty dish (counter and sink, not the drying rack) | **items left to wash** | clicks, labelled dirty / in rack | needs judgement: clean vs dirty |
| `fallen_leaves_on_grass` | each fallen leaf | **leaf count**, and **share of the lawn covered** (when to rake) | clicks (~150) | many small leaves |
| `fallen_leaves_on_grass` | the dead, straw-coloured grass | **share of lawn that is dead** (fuzzy edges) | careful polygon, with a tolerance | gradual boundary |
| `autumn_tree_orange_1/2` | the tree's leaves and branches | **canopy cover**: share of the crown that blocks the sky (forestry's "gap fraction") | colour rule: blue-sky pixels inside the crown | fine edges, holes |
| `autumn_tree_orange_1/2` | the reddish (not yellow) foliage | **% of the crown turned red** (fall-colour progress) | colour rule on hue, agreed once; consistency between the 2 phones | fuzzy, the task you asked for |
| `maple_leaves_dusk` | the yellowed leaves | **% of leaves turned yellow** | colour rule + polygon check | low light |
| `maple_leaves_dusk` | each cup of the weather vane (anemometer) at the bottom | **cup count = 3** (two overlap) | known | the mechanical-part task, small and dark |
| `farm_road_1/2` | each utility pole | **pole count**, and **tilt angle** of the near pole (utilities inspect leaning poles) | clicks; a line along each pole | thin, tall |
| `farm_road_1/2` | the overhead wires | **number of wires** | count by eye | very thin structures |
| `farm_road_1/2` | the gravel road | **where the road goes** (left edge, right edge at three heights) | a few points per edge | soft edges |
| `red_house_driveway` | each window of the house | **window count**; **window-to-wall ratio** (an energy-audit number) | polygons (few, easy) | lit and dark windows |
| `sunset_field_1/2` | each lone tree on the horizon | **count** (5–6) | clicks | silhouettes, tiny |
| `sunset_field_1/2`, `field_with_cows` | the clouds | **cloud cover** (meteorology's oktas: eighths of sky covered) | judged by eye in eighths, tolerance ±1 | fuzzy |

**Trick questions (made-up objects).** Ask for something absent: "segment
every cow" in `farm_road_1`, "segment every car" in `fig_plant`. The right
answer is nothing. This catches models that draw what they are asked for
whether or not it is there (DeepSeek Pro did exactly that with a picture
it could not see).

## Still missing from the plan

Horns on an animal, an engine with a named part, a computer screenshot,
and a stack of ~30 dishes (the kitchen photo is a cluttered counter, not a
stack). The anemometer can stand in for "a mechanical part" until an
engine photo exists.

## Before anything is published

`red_house_driveway.jpg` shows a **readable licence plate** and two people
on the porch, in a repository whose photos are CC BY 4.0. Blurring the
plate (and noting it in `images/README.md`) changes a small, documented
patch of pixels; no task above uses it.
