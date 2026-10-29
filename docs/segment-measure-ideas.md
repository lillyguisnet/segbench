# Outline from a dot, scored by a measurement (ideas, 2026-10-16)

This is a draft. No model has been asked yet. The 22 questions of v1 are in
`segbench/measure_tasks.py`. The reference is clicked in `apps/measure-clicks`:
start it with `uv run scripts/serve_measure_clicks.py` and open
http://192.168.2.24/measure/ (home network) or http://lambda.tail69222b.ts.net/measure/. The clicks are saved in
`annotations/measure/clicks-v1.json`.

The red dot the models see is placed automatically in the middle of the
clicks: half-way along a line, or half-way between the two points of a span.
The person clicking moves it when it is not clearly on the part. Every click
is made from scratch, with no model's draft to start from, so the reference
cannot lean towards any model in the comparison.

**The idea (Maxime, 2026-10-16).** Put a dot on a part ("the horn") and ask
the model to outline that part. Turn the outline into one measurement, such
as the length of the horn's centre line, and score that number. The
reference is a few clicks by a person. Pixel overlap with a hand-drawn mask
is not used. This is track 2 of the README (outline, given a dot), with
questions chosen so the measurement is hard to argue with.

**How to ask.** Prompt D of `scripts/probe_point_segment.py`: first say what
is under the dot, then outline it. Here it also **names the part**, because
a dot on a goose's neck is also on the goose. Example: "Outline the neck of
the bird the red dot is on." The words name the part; the dot says which one.

## Five measurements, each with a fixed rule and a few clicks as the reference

| kind | taken from the outline (fixed algorithm) | reference (clicks) | score |
|---|---|---|---|
| **centre line of a curved part** ("horn") | skeleton → longest path, smoothed; its length and its two ends | a clicked line along the middle, from base to tip (6-10 clicks) | length error (%) and mean distance between the two lines (% of length) |
| **tip to tip** | the two points of the outline farthest apart along the clicked direction | 2 clicks | length error (%) and end-point distance |
| **width / diameter** | width of the outline across the clicked line (or largest width, for round ends) | 2 clicks across | width error (%) |
| **angle** | main axis of the outline | 2 clicks along the axis | degrees off |
| **ratio inside one object** | e.g. height of the coulis / height of the jar | 2-3 clicks | ratio error |

Why these hold up: the scores are relative (%), so the photo's size does
not matter. Each question measures only one property, so the outline may
wobble elsewhere without changing the score. The ends of the measured part
must have crisp edges. A part with a fuzzy end is not used.

## Candidates in the photos we already have

Best first: crisp edges, a clear start and end, a strong background contrast.

| photo | part (dot on it) | measurement | n | notes |
|---|---|---|---|---|
| `mounted_birds_display` | each wing, shoulder to tip | **centre line** (the closest thing to a horn we have) + tip position | ~16 | crisp on a white wall; the swan's wing is white on white (harder) |
| `mounted_birds_display` | Canada goose necks, swan neck | **centre line** | 3 | the black neck is clearly separate from the white chin strap and the brown body |
| `mounted_birds_display` | gannet's beak, swan's beak, geese beaks | tip to tip | ~5 | small (743 px photo) |
| `mounted_birds_display` | each bird | wingspan, tip to tip | ~10 | |
| `kitchen_counter_dishes` | the faucet's curved spout | **centre line** | 1 | horn-shaped; the yellow cloth hides part of it (outline the visible part only) |
| `kitchen_counter_dishes` | the faucet's lever handle | **centre line** | 1 | a small curved horn, crisp metal edges |
| `kitchen_counter_dishes` | the two knives | tip to tip + angle | 2 | |
| `kitchen_counter_dishes` | bowl and plate rims | width | ~6 | |
| `butterfly_display_case` | each butterfly | wingspan, tip to tip | ~40 | crisp on a white board; 743 px |
| `butterfly_display_case` | the tails of the four green swallowtails | **centre line** | ~8 | very small (~20 px): a hard set |
| `leafcutter_ants_branch` | the lower side branch | **centre line** + angle | 1 | tapering, on a blurred background |
| `leafcutter_ants_branch` | the main branch | width at 3 places + angle | 1 | |
| `farm_road_1` / `_2` | the near utility pole | angle (lean) + width | 2 | the "pole tilt" idea of `task-ideas.md` |
| `farm_road_1` / `_2` | the overhead wires | **centre line** (how much they sag) | ~4 | very thin: a stress test |
| `cows_under_pine_tree` | the long dead branch leaning from the trunk | **centre line** + angle | 1 | |
| `cows_under_pine_tree` | the near cows | nose to tail | ~4 | legs and heads overlap |
| `log_ends_closeup_1` / `_2` | each cut log end | diameter | ~80 | already planned for track 2 (README); click 2 points across each end |
| `log_pile_barn_1` | the loose logs lying lengthwise (right) | end to end + diameter | ~12 | |
| `strawberry_coulis_jars` | each lid / each jar | lid width; coulis height / jar height | ~6 | a ratio, so perspective matters less |
| `oyster_mushrooms_hands` | each cap | largest width | ~10 | caps overlap; the stems are white on white (not used) |
| `living_plant_wall` | each pink leaf | tip to base (length) | ~15 | crisp pink on green |
| `fig_plant` | the large fig leaves | tip to base | ~10 | |

Not used: `red_house_driveway` (readable licence plate, people), the
mineral, the sunsets, the dusk maple leaves and the fallen leaves (soft
edges or nothing elongated). **We have no photo of real horns.** A goat, a
cow with horns or an antelope would be the cleanest example of this idea
and is worth taking.

## Things to settle before building

- **The skeleton is fragile.** A noisy outline grows side branches. Rule:
  keep only the longest path, then smooth it. Fix the rule on synthetic
  shapes (a drawn horn with a jagged outline) before any model's answer is
  scored, as was done for the box readers.
- **Occlusion.** Outline the visible part only (as in the box benchmark),
  and click the reference on the visible part too.
- **Which models.** Language models with prompt D, plus SAM 1 / 2.1 / 3,
  which take a dot natively and cost almost nothing (README track 2).
- **Reference effort.** About 150-250 questions at 2-10 clicks each, in the
  annotation app. Far less work than masks.

## Drafts (Maxime, 2026-10-16: "so that I do more of a correction")

This replaces the "from scratch" rule above, at Maxime's request.
`scripts/draft_measure_clicks.py` ran GPT-6 Astra (first on point benchmark
v3, on our ChatGPT plan, medium thinking: 22 requests, $1.68 at API price,
not billed). Astra proposed the clicks of each question. SAM 3, given a click
in their middle, outlined each part on GPU 1, and the clicks were moved onto
that outline. This happened for 115 of 131 drafts; the other 16 kept Astra's
clicks because SAM's outline was not believable. In the app, drafts are
orange; only the items Maxime keeps count.

**The bias this brings:** the reference starts from the answers of two
models that will be compared. Each item keeps its draft (`draft_points`,
`draft_dot`, `origin`), so we know how far Maxime moved it. A check, as for
the box answer key: score Astra and SAM 3 separately on the items Maxime
added himself, and on the drafts he kept.
