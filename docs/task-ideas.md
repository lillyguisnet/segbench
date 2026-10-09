# Task ideas for the 18 photos (draft, 2026-10-09)

Each task is asked as a segmentation ("segment every …"). A small scoring
function then turns the masks into a **real-world number** and compares it
with a ground truth we make by hand. Proposals; nothing here is built yet.

## Shortlist of six (proposal, 2026-10-09)

Chosen so each task tests a different skill, and each has an answer key
that is quick to make and hard to argue with.

| # | photo | skill tested | ask the model to segment | find track | outline track (given one dot per object) |
|---|---|---|---|---|---|
| 1 | `log_ends_closeup_1` | many touching, similar objects | each cut log end | log count (~70) + each dot matched | diameter of each log end vs. a two-click measurement (subset of ~20) |
| 2 | `field_with_cows` | tiny, distant objects; one odd one out | each cow | head count (~13) + matching; is the brown cow found | mask stays on the cow: overlap with a quick polygon (edges are crisp here, so overlap is fair) |
| 3 | `fig_plant` | telling look-alike plants apart | each fig leaf (not the spider plant or the purple plant) | fig leaf count (~12) + no leaves of the other plants | total fig-leaf area vs. polygons (a growth measure) |
| 4 | `autumn_tree_orange_1` | fuzzy, gradual regions | the reddish (not yellow) foliage | none: nothing separate to find | % of the crown that is red vs. a hue rule |
| 5 | `farm_road_1` | following a long, soft-edged region; direction | the gravel road | none: one road, nothing to count | the road's path: centre line and width at ~6 heights vs. clicks on both edges; which way it bends; where it disappears |
| 6 | `kitchen_counter_dishes` | meaning, not just shape: dirty vs. clean | each dish, labelled dirty or clean | dirty count (~10) and clean count (~10), each dot matched **with the right label** | outline each dish (overlap with polygons; stacked bowls in the rack are the hard part) |

The whole-task track runs the same six with no hints.

**Task 4: one photo.** `autumn_tree_orange_1` (Android): the clearest
split between red (upper right of the crown) and yellow (lower left), and
no wires across the crown as in `_2`.

**Task 5: no lanes.** It is a one-track gravel farm road with no painted
lines, so there are no lanes to find. What can be measured is where the
road goes: from the bottom of the picture it runs up, bends gently right,
then left past the barn, and disappears near the far pole.
`farm_road_1` (portrait) shows the longest stretch of road.

**Task 6: what counts as a dish, and as dirty.** Dishes = plates, bowls,
mugs, pots, pans, baking trays, lids, cutting board, food containers. Not
counted: cutlery and knives (too thin; thin shapes are not this task's
point), bottles, appliances, cloths. Answer key per item: *dirty* (on the
counter or in the sink, or food visible), *clean* (in the drying rack),
or *unsure* (not scored). The two clues agree for almost every item here;
items where they disagree are marked unsure rather than argued over. The
prompt asks for the label but does not explain the rule: deciding is the
test.

**Free trick question on every task:** also ask for something absent
(cows in the log picture, cars in the fig picture). The right answer is
nothing. Costs one extra call per task and needs no answer key.

**Why the outline track gives a dot, not a box:** for logs, a box would
give away the size we are measuring.

**Left out, and why:**

- **Utility poles:** replaced by the road task (Maxime, 2026-10-09).
- **Second tree photo:** one is enough (Maxime, 2026-10-09).
- **Big barn log pile:** hundreds of ends to click; the close-up already
  tests the same skill.
- **Fallen leaves, clouds, wires, sunset trees:** each overlaps a chosen
  skill or has an arguable answer.
- **Windows** (`red_house_driveway`): readable licence plate and people;
  not worth the privacy step for a task like the others.
- **Anemometer cups:** only 3, small and dark; too little signal.
- **Horns, engine part, screenshot:** no photo yet.

Answer-key work, roughly: ~120 dots (20 with a dirty/clean label), ~20
two-click diameters, ~45 small polygons (cows, fig leaves, dishes), one
crown outline, ~12 road-edge clicks, one hue rule. About 2 hours.

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
