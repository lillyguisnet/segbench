# Box benchmark (2026-10-15): the picture benchmark with boxes instead of points

Asked by Maxime on 2026-10-15: rerun the whole picture benchmark (the eleven
object tasks of point benchmark v3) asking each model for a box around each
object instead of a point on it. Decisions made with him that day:

- **Answer key: drafted boxes, checked by a person.** Every box is checked,
  moved, resized, removed or added by a person in `apps/box-review`
  (`uv run scripts/build_box_review.py`); the export of that review is the
  answer key. The objects themselves are fixed by the 262 approved
  reference dots; drafts only propose edges.
  - First draft (superseded): SAM 2.1 Large from each dot (the dot as a
    click, neighbouring dots as "not this" clicks), tightest rectangle
    around the outline (`specialists/sam/draft_boxes.py` ->
    `annotations/boxes/draft-sam2.1.json`). SAM 2.1 rather than SAM 3
    because SAM 3 is an entrant. Maxime found these too often off (parts
    of plates, one box around a butterfly instead of a beetle) to be a
    quick starting point.
  - **Draft used (Maxime, 2026-10-15): the top pointing models' own box
    answers** (`scripts/draft_boxes_from_models.py` ->
    `annotations/boxes/draft-models-v1.json`). The five best of point v3 with
    box answers (GPT-6 Astra, GPT-6.1 Sol, Gemini 3.8 Flash, Gemini 4 Argon,
    Muse Wonder (API)); both prompts, all tries: up to 30 answers per photo.
    Per dot, each answer's box around that dot (only if the dot is the
    reference dot nearest the box's centre, so group boxes and neighbours'
    boxes are not taken), then the median of each side. 252 of 262 boxes
    from the models (median 27-30 answers each, median overlap with the
    draft 0.76-0.98 by photo); 10 with fewer than 3 model boxes keep the
    SAM draft. The review app shows each box's support and agreement and
    can visit the least sure first (U).
  - **Known bias of this choice:** a key that starts from five entrants'
    boxes can favour those five, and models that draw edges like them, at
    the strict overlap levels (90%, 95%) wherever the reviewer keeps the
    draft's edges. At 50% it matters little. Mitigations: the median over
    30 answers is not any one model's box; the review changes what is
    wrong; how much each box was changed is in the review's audit. The
    five drafting models' strict-level scores are to be read with this in
    mind and reported with the method.
- **Boxes are around the visible part only**, said in both prompts and
  applied in the review: hidden parts cannot be checked by anyone.
- **Scoring at four overlap levels: 50%, 75%, 90%, 95%** (intersection over
  union). A model box and an answer-key box match when they overlap at least
  that much; maximum one-to-one matching; F1 at each level. A task's box score
  is the mean of its four F1s; a model's, the mean over the eleven tasks.
  Dishes count by location only (as since point v2.2); the dirty/clean label
  is reported apart (`scripts/score_boxes.py`).
- **Two prompts per model, each model keeps the better** (as for points):
  - `box-a`: named corners (x_min, y_min, x_max, y_max) on the 0..1000 grid,
    JSON, the "not pixels" sentence of point prompt v2;
  - `box-free`: the same request with no answer format and no coordinate
    system: each model answers the way it chooses, and the reply is read the
    way it says it is written (below).
  Words in `segbench/tasks.py` (`prompt_box`). Same words for every model.
  Selection is made on the test photos, as for points: the chosen score
  includes a little luck, favouring erratic models slightly.
- **Reading leniency**: a box whose two x (or y) are written in the wrong
  order is the same rectangle and is read so; x and y are never swapped
  except where a format's documented order is y first (Gemini's box_2d).
- **No trick questions** in this benchmark.
- **MolmoPoint 8B is out**: it can only point.

## Checked on synthetic pictures first

`scripts/check_boxes.py`: seven red ovals (wide, tall, round, small, two
overlapping) and four blue squares, at four sizes of the benchmark photos
(4000x2252, 2252x4000, 1015x1015, 743x1319), every language model, both prompts.

- `box-a` (`results/check-boxes-box-a-20261015-153903.jsonl`): every reply
  used the named corners. Most models passed. GPT-6 Luna, GPT-5.6 Terra and
  Kimi K3 failed on the two large pictures and passed on the small ones:
  asked for 0..1000, they shrank or stretched the vertical axis (Luna's y
  about 0.82 of the truth on the wide picture), while in pixels (box-free)
  the same pictures were nearly perfect. That is what the second prompt is
  for; box-a was not changed.
- `box-free` (`results/check-boxes-box-free-20261015-153307.jsonl`): each
  model's own habit. GPT models: pixels of the photo. Claude models: pixels
  of the photo as Claude sees it, shrunk to 2576 px on its long side
  (4000x2252 -> 2576x1450; the small pictures unchanged). Gemini models:
  Google's `box_2d`, y first, 0..1000. Kimi K3: 0..1. Muse Wonder: named
  corners, 0..1000. GLM 5.3 Flash and DeepSeek Flash changed format between
  pictures, usually saying which.
- Qwen 3.8 27B, Qwen 3.8 Flash Next, Perceptron Mk1.5 and Mistral Large 4
  (OpenRouter; checked later the same day, once the key's spending limit was
  reset, before any of their benchmark replies existed):
  `results/check-boxes-box-a-20261015-183620.jsonl`,
  `results/check-boxes-box-free-20261015-183431.jsonl`. box-a: 15 of 16 pass;
  Mistral's 16th reply was empty (it spent all 4,454 output tokens thinking:
  a model failure, scored as no boxes). box-free: Qwen writes `bbox_2d`
  (0..1000, x first), Perceptron its own `<point_box>` tags, and Mistral
  writes the key `box_2d` with **x first**, not Google's y-first order.

## Reading box-free replies

Rules in `segbench/box_read.py` (`read_box_free`), written from the
synthetic replies only, and frozen before any box-free reply on a benchmark
photo was read: copy and hashes in
`results/box-benchmark/box_read.frozen-*.py` and `rules-frozen-*.sha256`.
In short: the boxes the reply gives; read as the reply says (order, scale);
else as its format is documented (box_2d); else as that model wrote on the
synthetic pictures; else from the numbers (all <= 1: 0..1; any > 1000:
pixels; else 0..1000). Replies the rules cannot read cleanly (no boxes, a
list written twice, mixed formats, boxes beyond the photo) are flagged and
read by a person; those readings go to `results/box-free-hand-readings.json`
with the reason. Nothing reads the answer key.

**Rules v2 (2026-10-15, frozen as `box_read.frozen-v2-*.py`)**, after reading
the 39 flagged benchmark replies and before any box-free reply was scored
against any answer key (the reviewed key did not exist yet):
- when a reply names the size of the picture its pixels refer to ("the
  4096 x 2304 image"), that size is the frame, even when it is not the
  photo's size. GPT and Claude models often say it: GPT-6 Luna 4096 x 2304
  for the 4000 x 2252 photos, Claude 2576 x 1449..1476, GPT-5.6 Terra once
  3393 x 3393 and once 3456 x 4608 (a portrait frame for a landscape photo);
- corner pairs also written "(a, b) -> (c, d)" and "top-left (a, b),
  bottom-right (c, d)" (Claude Opus wrote both).
**Rules v3** (frozen `box_read.frozen-v3-*.py`), from the four OpenRouter
models' synthetic replies, before their benchmark runs: their habits
(Qwen and Mistral 0..1000 x first, Perceptron tags), and `box_2d` read x
first for Mistral Large 4 only. **Rules v4** (`box_read.frozen-v4-*.py`):
Perceptron Mk1.5 answers prompt box-a in its own tags instead of the JSON
asked for; a box-a reply with no JSON at all is read from those tags (same
0..1000, x-first frame box-a asks for). Neither change reads any earlier
reply differently (checked: 495 box-free and 503 box-a replies).

With v2 every box-free reply is read by the rules; none needed a reading
by hand. Still flagged, and read as written (the model's own error, not a
reading problem): replies whose boxes run past the photo, e.g. GPT-5.6 Terra
on the cows writing pixel heights up to 3651 for a 2252-pixel-tall photo,
and DeepSeek Flash writing 0..1000 values up to 1780. Empty answers ("[]")
are answers of "none".

**The four OpenRouter models' replies** (264, no errors): every reply read by
the rules except two, read by hand (`results/box-free-hand-readings.json`,
with the reason): Qwen 3.8 Flash Next on the dishes, tries 0 and 2, wrote
pixel numbers (up to 3996 and 2560) although its habit is 0..1000, naming no
frame; read with rule 3's own default for numbers above 1000, pixels of the
photo. Uncertain: try 2's largest x is exactly 2560, so it may be in pixels
of a resized copy; the photo's pixels is the documented default and was
used. Empty or box-less replies, scored as no boxes: Mistral Large 4 twice
("I found 11 cows in this photo." and "Here are the pink leaves:" with
nothing after), and once out of its 32,000-token budget with no text (box-a,
fig); Perceptron Mk1.5 and Qwen Flash once each (no text after thinking,
box-free, fig). Perceptron twice repeated one box over and over on the jars
(a list written twice is counted once; the repeats stay and count as extra
boxes). Mistral's boxes a little past 1000 (up to 1024) are read as written.

## Runs

Language models: medium thinking, 3 tries, most image detail (as point v2),
both prompts: `results/run-boxes-box-a.jsonl`,
`results/run-boxes-box-free.jsonl` (API models) and
`results/run-boxes-box-free-subs.jsonl` (OpenAI and Anthropic plans). Muse
Wonder through its pay-per-token API (the plan's quota is used up until
2026-10-19), as in point v3. The four OpenRouter models were run the same
way into the same files once the key's limit was reset (2026-10-15).

Three GLM 5.3 Flash calls got HTTP 400 from Z.AI ("image input format /
parse error", "invalid API parameter") while the other tries on the same
photos were answered: the model never saw the question. They were sent again
(`results/run-boxes-box-a-resent.jsonl`, `run-boxes-box-free-resent.jsonl`);
the answer replaces the error.

Local models, 2026-10-15, on GPU 0 with the paper-rewriter service
(model-vllm-smr-9b) stopped (restarted afterwards):
- `scripts/time_finders_boxes.py` (scripts/time_finders.py with each model's
  `box` task, all eleven tasks, no trick questions; single-copy times and
  the load test on the first four tasks, as point v3's costs):
  LocateAnything 3B, Florence-2 Large (phrase grounding), Moondream 3.1,
  GroundingPI 4B in `results/run-finders-boxes-1.jsonl`; Rex-Omni 3B in
  `results/run-finders-boxes-2-rexomni.jsonl`, re-timed because an unknown
  program (10 GB) used GPU 0 during its first run (its records in
  run-finders-boxes-1 are not used). Load factors with boxes: LocateAnything
  1.0 (2 copies out of memory, as for points), Rex-Omni 0.566, Florence-2
  0.543, Moondream 1.0 (one copy fits), GroundingPI 0.206 (16 requests at
  once). A desktop overlay (130 MiB) and, for LocateAnything, a video job
  running on GPU 1 (276 MiB context) held memory on GPU 0 without computing
  there (0% use checked before).
- `specialists/common/boxes_native.py`: SAM 3, YOLOE-26x and RF-DETR Seg,
  the same calls as for points, answered with each model's own detection box
  (not the rectangle around its mask), the same objects the point answers
  kept; `results/run-specialists-boxes-1.jsonl`. Seconds as for points
  (median of 3), cost with their 2026-10-08 load factors.

## Known limits

- At 90% and 95% overlap the answer key's own precision matters: a box a few
  pixels off on each side of a small object (a distant cow) fails 95%.
  Small annotation differences move overlap a lot, most for small objects and
  strict levels (Murrugarra-Llerena et al. 2022, doi:10.1109/cvprw56347.2022.00528).
  Those two levels say more about agreeing with this answer key's edges than
  about finding objects.
- The draft may anchor the reviewer toward the drafting models' edges (see
  "Known bias of this choice" above).
- Cost and seconds, as in point v3, from the first four tasks only.

## The answer key (2026-10-16)

Reviewed by Maxime in `apps/box-review` on Lilly's computer; the last export
(`annotations/boxes/segbench-boxes-c795cd22-2026-10-16T15-19-37-741Z.json`,
copied over SSH, user `aurel` on lilly-pc) is the answer key, kept as
`annotations/boxes/answer-key-boxes-v1.json`. It belongs to
draft-models-v1 (same SHA-256), every photo marked checked, nothing left
pending: 262 boxes kept (261 drafts corrected or kept, one fig leaf drawn by
hand), one bird removed as not an object (15 birds instead of the point
key's 16; fig leaves 34 instead of 33). Only 27 boxes were kept exactly as
drafted; median overlap of the final box with its draft 0.88, 62 below 0.75.
The point scores are not changed by these differences (Maxime, 2026-10-16).

## Results v1 (2026-10-16)

`uv run --with pillow scripts/build_boxes.py` -> `results/box-benchmark/v1/`
(summary.csv, scored.jsonl, calls.jsonl, manifest.json with each model's
prompt); `uv run --with pillow scripts/plot_boxes.py` ->
`charts/box-benchmark-v1/` (with-local-models, language-models: score
against cost; overlap-levels: every model's score at 50/75/90/95%, one row
per model, instead of one bubble chart per level, since cost does not change
with the level). Rex-Omni's records of its disturbed first run are left out
(`results/run-finders-boxes-1-without-rexomni.jsonl` is run-finders-boxes-1
without them). Prompt ties would have gone to box-a (written in advance;
none happened).

Best scores (mean of the four levels): Gemini 4 Argon 62.7, Claude Opus 5.5
62.0, Claude Sonnet 5.5 61.6, GPT-6 Astra 56.8, Gemini 3.8 Flash 55.5. At
50% overlap the top models are close to 90%; at 95%, none reaches 30%.
Free format was the better prompt for 12 of 19 language models; for Claude
Sonnet by 21 points (40.3 -> 61.6) and Gemini 3.5 Flash Lite by 19. Hardest
photos: pink leaves (median 15.7), fig leaves (27.5), dishes (35.0), logs
(36.8). Box and point rankings differ: Claude Sonnet ranks 3rd with boxes
(17th with points); Muse Wonder 15th (5th with points), its boxes rarely
tight (95%: 7.7).

**Check of the drafting bias.** The same answers scored against the
unreviewed draft and against the reviewed key, at the strict levels (mean of
90% and 95%): the five drafting models lose 9.5 points on average from draft
to key; the 22 others lose 0.2. Against the draft they had an advantage at
the strict levels; the review removed it (their edges were often moved). The
published scores use the reviewed key only.

### More charts (2026-10-16)

`uv run scripts/plot_box_extra.py` (Maxime asked the same day):
- `charts/box-benchmark-v1/points-vs-boxes/`: each model's point score
  (point benchmark v3, round mark) and box score (square), joined by a line,
  logo inside each mark, box minus point at the right. Same eleven photos,
  but different tests and answer keys: the order is comparable, the gap's
  size only roughly. Where the two scores are within ~3 points the marks
  overlap.
- `charts/box-benchmark-v1/levels-logos/`: each model's box score at 50, 75,
  90 and 95% overlap, square marks joined by a line, darker = stricter,
  logo beside the name.
- `charts/box-benchmark-v1/original-4-photos/`: the main score-against-cost
  chart on the four original photos only (logs, cows, fig leaves, dishes).
  Each model keeps the prompt chosen on all eleven photos; only the photos
  scored change. Cost and seconds are the same as on the main chart (they
  already come from these four photos). Leaders: Claude Opus 5.5 55.7,
  Claude Sonnet 5.5 54.5, Gemini 4 Argon 50.7, GroundingPI 4B 50.2.
- `charts/box-benchmark-v1/points-vs-boxes-4-photos/`: the points-vs-boxes
  chart on the four original photos only. Point score = the mean of point v3's
  scores on those four photos (point v3's score is the plain mean of its
  per-photo scores; checked), so it uses point v3's prompt choice, made on
  eleven photos; it is not point v2.7's published four-photo score, whose
  prompt choice was made on these four photos alone. On these photos 25 of
  27 models score lower with boxes; Claude Sonnet 5.5 +17, GroundingPI 4B 0.
