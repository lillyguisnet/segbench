# Point-finding benchmark v2 (2026-10-09)

Goal (Maxime): measure **pointing**, not format-following, with every entrant
run as well as is reasonable. Scoring rules are unchanged from v1
(`docs/point-benchmark-v1.md`: same approved reference points, three
tolerances, maximum one-to-one matching, F1, equal task weights; dishes need
the right dirty/clean label). What changed is how answers are asked for,
read and timed.

## Language models: prompt v2, 3 tries, most image detail

- **Prompt v2** (`segbench/tasks.py` `prompt_v2`): named coordinates
  `{"x": .., "y": ..}` (v1's unlabelled pairs were swapped by two Gemini
  models); "Mark every one you can see, however many there are. Do not stop
  early or give only a sample." (three models stopped at exactly 50 of 83 log
  ends); "They are not pixels: whatever the photo's size, the right edge is
  x = 1000 and the bottom edge is y = 1000." Same words for every model;
  trick questions use the same template.
- **Checked on synthetic pictures only** before any benchmark photo
  (`scripts/check_points_v2.py`: six red circles, four blue squares, wide and
  tall). First check 24/26: DeepSeek Flash treated 0..1000 positions as
  pixels and divided them again, so the "not pixels" sentence was added for
  everyone; second check 25/26 (Kimi K3 scaled one answer by 0.9; it passed
  the same picture in the first check). Prompt then frozen.
- **Image detail** (`scripts/probe_detail.py`, input tokens of a synthetic
  4000x2252 picture): Gemini 3 takes `mediaResolution ULTRA_HIGH` on the
  image part (1,108 -> 2,213 tokens; HIGH is already its default; refused in
  generationConfig) and gets it. OpenAI's default already takes the most
  (10,674; "high" would *lower* it to ~2,950), so it stays default. lm15
  refuses OpenAI's newer "original" value, so it was not tried. GLM, DeepSeek,
  Kimi and Qwen ignore "high"; Anthropic has no setting.
- **3 tries** per model and question, medium thinking (min and max thinking
  not run in this version). Run `results/run-points-v2.jsonl`: 390 calls,
  all answered; $8.17 at list prices, $1.59 actually charged.

## Reading answers: the content, as written

Strict JSON first; if it fails, coordinates are recovered mechanically
(`segbench/point_score.py` `recover_points`): object by object, named x/y
where the keys are intact, otherwise the object's numbers in written order
(first = x, the prompt's order); numbers glued to letters ("x0") are not
coordinates. Never: reordering, swapping, rescaling, clipping, deduplicating,
or anything using the answer key. A reply strict JSON can read is never read
differently (tested). 10 of 390 v2 replies needed it, all checked by hand:
mostly one missing brace; Gemini Pro's log answers degrade mid-list into keys
like `"xecho"` while still pointing (2 of 3 tries), recovered from 25 and 9
points to the ~80 and ~69 written.

## Specialists: whole GPU 0, one rule for speed and cost

GPU 0 for the measurements: SAM 3.1 helper stopped, the scholarsreadinglist
embeddings job paused (SIGSTOP: holds 1 GB, no compute), browser and desktop
overlay < 0.1 GB, no compute; both restored afterwards.

- **Speed** (chart seconds), every specialist: one photo at a time, model
  loaded, from opening the photo to the points on the CPU; median per
  question. Dishes for the pointing models are two queries (dirty, clean),
  added. Loading the model is not counted (an API keeps its model loaded).
- **Cost under load**: $0.22 per GPU-hour (RunPod RTX 3090, 2026-10-08) x the
  question's single-copy time x a load factor. Pointing models
  (`scripts/time_finders.py`): K copies run at once on the benchmark photos;
  in the window where all copies work, load factor = window seconds / single-
  copy seconds of the work finished; best K kept (K = 1 gives 1).
  LocateAnything: 2 copies do not fit (out of memory on full-size photos);
  MolmoPoint (~19 GB): 1 copy. SAM 3, YOLOE, RF-DETR keep their 2026-10-08
  load factors (batch sweep on the test images, `results/specialists/`),
  re-timed on the free GPU in `results/run-specialists-2.jsonl` (same answers
  as before, about twice as fast).
- **Repeats**: LocateAnything samples (temperature 0.7, its card's setting):
  3 seeds, like the language models' 3 tries. The others are greedy or
  beam-search: answers repeat exactly (checked), so their tries are identical.
- **Wording** for the pointing models: protocol v2 of `scripts/run_finders.py`
  (each model's documented "all instances" form). Florence-2 is entered in
  both of its text modes (grounding, open-vocabulary detection): nothing
  documents which one means "every object", and picking after seeing scores
  would be choosing by result.

## Limits

- One thinking level (medium) for the language models.
- The API models' seconds include the network and the provider's queue; the
  specialists' do not include a network (they would add a little behind an
  API).
- The approved reference points were corrected from model proposals:
  shared omissions and anchoring are possible. Masks will replace the
  distance tolerance later.
- Six photos (four object tasks here) do not support broad claims.

## v2.1 (2026-10-09): each model keeps the better of two prompts (option B)

Maxime chose option B: every model shown at its best of two fixed setups,
with no note on the chart (the method is here and in the report folder).
- Language models: prompt v1's exact words re-run with v2's image settings
  ("v1-hd", `results/run-points-v1hd.jsonl`, 156 calls, 3 tries), so the two
  candidates differ only in wording. Per model, the prompt with the higher
  four-task score is kept for the whole model, never task by task
  (`scripts/select_best_prompt.py`). Kept v1-hd: Astra 68.6 (v2 66.9), Sol
  66.4 (63.5), Kimi 33.9 (30.7), GLM 18.2 (15.1), Luna 15.5 (11.7). Kept v2:
  the other eight.
- Florence-2 keeps its better text mode (phrase grounding 32.9 vs
  open-vocabulary detection 5.9) and is shown once, as "Florence-2 Large".
- **Known bias**: the choice is made on the test photos, so a chosen score
  includes a little luck (scores move up to ~26 points between tries on one
  photo). It favours the most erratic models slightly. The unbiased way
  (option A: choose on separate photos, then run the test once) was offered
  and not taken.
Report: `results/point-benchmark-v2.1`; chart: `charts/point-benchmark-v2.1`.

## v2.2 (2026-10-10): dishes by location; chart changes

- **Dishes count on location alone** (Maxime): a dot matches a dish by
  position only. Dirty/clean is kept as a separate number, the share of
  correct labels among the dishes found (`label_accuracy` in
  `scored.jsonl`; `scripts/score_points.py --dish-labels separate`). The
  prompt still asks for the label, unchanged. Each model's choice of prompt
  was re-made under this rule and did not change.
- Chart (`segbench/chart_editorial.py`): plain white background (the accent
  wash, meant for the lower right, was strongest at the upper right);
  y-axis labels with %; models off the best-value line show their score
  instead of their maker, at the same small size; logos for NVIDIA (Simple
  Icons, CC0), Microsoft and Ai2 (Lobe Icons, MIT); IDEA Research has no
  logo in either library and is written as text.
Report: `results/point-benchmark-v2.2`; chart: `charts/point-benchmark-v2.2`.

## v2.3 (2026-10-10): RF-DETR on all four tasks

Maxime: RF-DETR failing where its vocabulary has no word is part of the
result. It now answers every task with the COCO categories that fit, fixed
before running (`specialists/rfdetr/points.py`): logs none and fig none (no
dots, score 0), cows "cow", dishes "bowl" and "cup" (dishes are scored by
location). Result 23.5: cows 76.7, dishes 17.2, logs 0, fig 0. Answers were
computed on 2026-10-10 with GPU 0 busy (another project's vLLM evaluation,
not paused); seconds and cost are its clean 2026-10-09 measurements on the
same photos (`timing_source` in each record; today's busy-GPU seconds,
3-7x slower, kept as `seconds_this_run`). On the chart its disc covers
YOLOE's: 23.5 vs 23.2, $0.0028 vs $0.0028 per 1,000 images.
Report: `results/point-benchmark-v2.3`; chart: `charts/point-benchmark-v2.3`.

## v2.4 (2026-10-10): Qwen 3.8 Flash Next

Maxime: add Qwen 3.8 Flash Next on Alibaba's API. Reached through
OpenRouter's `qwen/qwen3.8-flash` (its model page link: Qwen/Qwen3.8-Flash-Next),
pinned to the Alibaba host with fallbacks refused, like Qwen 27B; every call
was served by Alibaba. That OpenRouter's "Flash" is Flash-Next rests on
OpenRouter's link, not on a statement from Alibaba. Price $0.15 / $0.47 per
million tokens (OpenRouter, 2026-10-10). Same checks as every model before
any benchmark photo: circle test passed; thinking off = 0 thinking tokens,
medium accepted; image detail: "high" ignored (2,513 input tokens either
way), so default; prompt v2 synthetic check passed wide and tall. Both
prompts run 3 times (42 calls, $0.04 at list price, $0.03 billed by
OpenRouter); v2 kept (71.6 vs 69.6). Two v1-hd answers needed coordinate
recovery (checked: 11 and 79 points as written). Result 71.6, first place:
logs 85.4, cows 100, fig 54.9, dishes 46.1 (by location).
Chart labels: models off the best-value line may now put their label just
below their disc when both sides are taken (they sat in a far column with
long crossing leader lines under the flat part of the line).
Report: `results/point-benchmark-v2.4`; chart: `charts/point-benchmark-v2.4`.

## v2.5 (2026-10-11): Moondream 3.1

Maxime: add Moondream 3.1 (M87 Labs, July 2026; 9B mixture of experts, about
2B active), run locally. Set up in `~/Projects/finders/models/moondream`:
Moondream's own engine (Photon, `moondream` 2.6.1 / `kestrel` 0.9.1),
weights pinned at `moondream/moondream3.1-9B-A2B@1da27e6c2d`, its `point`
skill, greedy decoding, the object limit raised to 1000 (the default would
stop near 383 points), Photon's reuse of a picture between questions off (a
reused picture gave points a few pixels off a fresh one, and repeats would
look ~6x faster). Answers repeat exactly (checked).
- **Wording**: protocol v2's rule (each model's documented form): singular
  phrases, like Rex-Omni, as Moondream's documentation points at "face" and
  detects "person": "log end", "cow", "fig leaf", "dirty dish" + "clean
  dish". Fixed before the run; one run, no alternatives tried.
- **GPU 1, not GPU 0**: timed by `scripts/time_finders.py --gpu 1` on the
  whole of GPU 1, the same card model (RTX 3090, 350 W, PCIe 4 x16), with its
  four programs stopped for the run (09:07-09:11) and restarted after.
  `results/run-finders-4.jsonl`, `.timing.json`.
- **Cost**: one copy only (Photon reserves ~17 GB of the card), so load
  factor 1, i.e. no discount for running copies side by side. Photon instead
  batches questions inside one copy; that was not measured, so Moondream's
  cost here is likely an overestimate, as for MolmoPoint.
- **Result 59.4**: logs 78.8, cows 70.2, fig 64.3 (the best fig score of any
  entrant), dishes 24.4 (by location; 0 of the dishes it found carry the
  right dirty/clean label). 0.56 s per question, $0.036 per 1,000 images: on
  the best-value line, between Florence-2 and Qwen 3.8 Flash Next. Trick
  questions: none invented except 10 "person" dots on the autumn tree photo.
Report: `results/point-benchmark-v2.5`; chart: `charts/point-benchmark-v2.5`.

## v2.6 (2026-10-12): Muse Wonder, Muse Wonder Full, Perceptron Mk1.5

Maxime: add Meta's Muse Wonder and Muse Wonder Full (released 2026-10-12)
and Perceptron Mk1.5. Same checks as every model before any benchmark photo
(circle test, prompt v2 synthetic check wide and tall, thinking levels,
image detail), same runs (prompts v2 and v1-hd, 3 tries, medium thinking,
plus the trick questions), same rule (each model keeps its better prompt,
`scripts/select_best_prompt.py`). Every earlier entrant's score is unchanged
(checked: the candidate scores and the 236 earlier selected records are
identical).

- **Muse Wonder and Muse Wonder Full** (Meta, direct API): called on our Muse
  Code plan (Power Usage) with the plan's own key, never the pay-per-token
  key (`segbench/muse_plan.py`; Full exists only on the plan). Meta's terms
  reserve that key for Muse Code; using it here was Maxime's decision. The
  plain model ids, not the "-contributor" copies Meta may train on. They
  cannot stop thinking (min = minimal); image detail: Meta already takes its
  most by default (4,096 tokens for a 4000x2252 picture; "high" changes
  nothing). Cost at Muse Wonder's API price, $4 / $20 per million tokens
  (OpenRouter, 2026-10-12), for both: Full has no per-token price anywhere,
  so its tokens are priced like Wonder's, as every plan-called model is
  priced at an API price; Full sold per token would likely cost more. Three
  Full calls on the log photo got no reply for 10 minutes on the first
  attempt and the batch was restarted; the same questions then took 37-69 s.
  Results: Muse Wonder **75.4** (prompt v1-hd; v2 74.4): logs 83.9, cows
  94.7, fig 72.5, dishes 50.7 (by location), 18.8 s, $47.81 per 1,000
  images. Muse Wonder Full **74.6** (v2; v1-hd 74.4): logs 85.6, cows 91.8,
  fig 73.2, dishes 48.0, 35.7 s, $55.64 per 1,000. First and second, ahead of
  Qwen 3.8 Flash Next (71.6); the two differ by less than one point. Trick
  questions: none invented. Plan use for all runs: 8% of the 5-hour window.
- **Perceptron Mk1.5** (closed weights; Perceptron's API through OpenRouter,
  pinned to Perceptron): our prompts like every model. **38.9** (v2; v1-hd
  36.7): logs 21.4, cows 98.0, fig 0.0, dishes 36.1, 49.0 s, $6.50 per 1,000.
  Perceptron caps a reply at 8,192 tokens, thinking included (the others
  had 32,000); at medium thinking 4 of its 12 v2 answers ran out while
  thinking and came back empty, which score 0 and are not asked again.
  Its own answer format (point tags, `prompt_version` "tags",
  `results/run-points-tags.jsonl`) scored 46.1, mostly by running out less
  often; not used on the chart (Maxime, 2026-10-12: the usual prompt, as for
  every model). OpenRouter does not pass Perceptron's own "answer with
  points" switch (identical replies with and without it). Two trick
  questions were blocked by the OpenRouter key's spending cap and rerun;
  the failed records are kept in side files.

Screen clicks, round 2 (`results/run-screens-2.jsonl`, untuned prompt, 3
tries, medium thinking): Muse Wonder Full 100, Muse Wonder 97, Perceptron
Mk1.5 45 (its own tag format: 52, `results/run-screens-2-tags.jsonl`, not
used). Every earlier model's score unchanged.

Candidate scores: `results/point-benchmark-v2-loc-2026-10-12`,
`results/point-benchmark-v1hd-loc-2026-10-12`. Report:
`results/point-benchmark-v2.6`; charts: `charts/point-benchmark-v2.6`,
`charts/screens-2`.

## v2.7 (2026-10-14): Gemini 4 Argon, Claude Haiku 5.5, Mistral Large 4, GroundingPI 4B

Maxime: add Gemini 4 Argon and Claude Haiku 5.5 (both asked for that day),
Mistral Large 4 (run 2026-10-13, held back until now) and GroundingPI 4B
(local, weights out 2026-10-14). Same checks and runs as every model before
(`docs/adding-a-model.md`), same rule (each language model keeps its better
prompt). Every earlier entrant's score is unchanged, on the photos and on the
screens (checked: the selection gained 12 records per new model and lost
none; the earlier rows of both score tables are identical).

- **Gemini 4 Argon** (Google API, `gemini-4-argon`, on the API since
  2026-10-13; no dated snapshot, so Google may change it under the same
  name). lm15 1.2.1 sends Google's thinking *levels* only to names starting
  "gemini-3" and gave Argon a token budget instead (8,192 for medium);
  `segbench/call.py` now sends levels to "gemini-4" names too. Argon takes
  low, medium and high (minimal refused), so min low, medium medium, max high,
  as Gemini 3.1 Pro and 3.8 Flash. At medium it mostly chooses not to think
  (305 thinking tokens over the 30 v2 answers). Ultra-high image detail
  doubles its image tokens, so it gets it, as every Gemini. $2 / $10 per
  million tokens. **71.8** (v2; v1-hd 65.9): logs 84.6, cows 92.6, fig 68.9,
  dishes 40.9, 10.1 s, $11.18 per 1,000 images. Third, behind Muse Wonder and
  Muse Wonder Full. It counts well and places loosely: on the dishes its
  counts were 23-25 of 26, but 1-5 points per try hit the tight mark. Trick
  questions: none invented (0 of 18).
- **Claude Haiku 5.5** (Claude plan, `claude-haiku-5-5`, released
  2026-10-07). Thinks by default; Anthropic's off for it is
  `thinking: disabled` (min). Priced at its API price, $0.10 / $0.50.
  **62.9** (v2; v1-hd 60.5): logs 58.2, cows 99.0, fig 50.3, dishes 44.1,
  7.7 s, $1.37 per 1,000: on the best-value line between Moondream 3.1 and
  Qwen 3.8 Flash Next. Trick questions: none invented.
- **Mistral Large 4** (OpenRouter, pinned to Mistral, `mistral-large-4-0`;
  "public preview": Mistral keeps training it under the same name). Its only
  thinking levels are off and high, so medium = high, no max. Charted at
  Mistral's regular price ($1.36 / $4.18), not the launch sale it was billed
  at. **50.8** (v2; v1-hd 45.5): logs 34.6, cows 80.6, fig 55.2, dishes 33.0,
  161 s per question, $46.39 per 1,000. One answer was read by the lenient
  reader (recovered 1). Trick questions: invented in 3 of 18 (a car on the
  fig photo, people in the tree, cows on the road).
- **GroundingPI 4B** (arXiv:2609.39601; `~/Projects/finders/models/groundingpi`,
  through its authors' own vLLM server, temperature 0, answers repeat
  exactly). Photos: its category-point template, singular ("Point to: cow."),
  like its examples. Timed on the whole of GPU 0 (its one program, the paper
  rewriter, stopped); load factor 0.3148 with 16 requests at once.
  **50.0**: logs 83.1, cows 93.9, fig 5.7, dishes 17.2, 2.1 s, $0.059 per
  1,000. Trick questions: one invented point on every one (6 of 6), e.g. a
  "cow" in the middle of the log photo. Not on the best-value line (Moondream
  3.1 is cheaper and better).

Screen clicks, round 2 (`results/run-screens-2.jsonl`;
`results/run-screens-local-2.jsonl` for GroundingPI): Gemini 4 Argon 100
(tied first with GPT-6.1 Sol, GPT-6 Astra and Muse Wonder Full), Claude
Haiku 5.5 98.5 (on the best-value line), Mistral Large 4 41.5 (225 s per
question), GroundingPI 4B 17.8 (on the best-value line, between Florence-2
in detection mode and MolmoPoint 8B). GroundingPI gets its makers' screen template ("Point to the UI
element to click for the following instruction: ..."), which asks for one
element: it always clicks once, and every question has 2 to 8 right
answers, so one click scores at most 41 here. Said on the screens page.

Charts: Mistral's logo (Simple Icons, CC0) and orange; GroundingPI has no
mark that fits a disc, so "GPI" in its purple (`assets/logos/SOURCES.md`).
Labels on the best-value line now keep a small margin between each other
(GroundingPI's ran into MolmoPoint's on the screens chart); every earlier
chart redraws identically with it (checked v2.5, v2.6 and screens-1).
Pinned labels (v3: MolmoPoint 8B lower left, Gemini 3.5 Flash Lite below,
Gemini 4 Argon and Claude Sonnet 5.5 lower right). Pinned labels: v2.7 Gemini 3.1 Pro left (and Qwen 3.8 27B left on the
language-model chart), `scripts/plot_point_pilot.py`; on screens-2,
Moondream's pin removed (GroundingPI's disc now sits where it pointed),
`scripts/plot_screens.py`.

Candidate scores: `results/point-benchmark-v2-loc-2026-10-14`,
`results/point-benchmark-v1hd-loc-2026-10-14`. Report:
`results/point-benchmark-v2.7`; charts: `charts/point-benchmark-v2.7`,
`charts/screens-2`.

## v3 (2026-10-14): seven more photos, eleven tasks

Seven tasks on the second photo batch (`images/README.md`, "Added
2026-10-14"; `segbench/tasks.py` TASKS_BATCH_2): leaf pieces carried by
ants, beetles (not the butterflies), birds, mushroom caps, jars of
strawberry coulis (not the other preserves), pink leaves, and the white cow
with a bit of red (the cow question was asked for in the 2026-10-14
conversation; the others were proposed there and approved the same day). The mineral photo has no task. These photos are
resized copies (0.5 to 1.2 megapixels), not phone originals like the first
eighteen.

Reference points: a draft from four models (Gemini 3.8 Flash, GPT-6.1 Sol,
Muse Wonder through the pay-per-use API because the plan's quota was used up,
Moondream 3.1; `results/run-consensus-2.jsonl`, `scripts/consensus.py`),
then reviewed dot by dot by the photos' author in the review app: 17 added, 4 moved, 2
rejected, every photo checked whole; then approved
(`annotations/reviews/segbench-review-5682a4be-approved.json`, 109 points).
The two reviews are separate files; `scripts/score_points.py --reference`
takes both.

Same treatment as v2.7 on all eleven tasks (`scripts/build_v3.py`): every
language model with both fixed prompts, 3 tries at medium thinking
(`results/run-points-v2-b2.jsonl`, `run-points-v1hd-b2.jsonl`), keeping
the prompt with the better eleven-task score. Three models change prompt
from v2.7 because of that (GPT-6 Astra to v2, Kimi K3 to v2, Mistral Large
4 to v1-hd), so their scores on the first four tasks differ from v2.7's.
Local models: their published runs on the first photos plus the second
batch (`run-finders-b2.jsonl`, `run-specialists-b2.jsonl`,
`run-rfdetr-b2.jsonl`), with the same kind of short phrase
(`segbench/tasks.py` PHRASES_BATCH_2). RF-DETR has COCO words only for
birds and cows (and cannot be told the colour of the cow).

Cost and seconds come from the first four tasks only, for every entrant:
the local models were timed on those photos only, and the new photos are
about ten times smaller, so cost would otherwise depend on which photos a
model happened to be timed on. Each new record keeps its measured values
as `seconds_measured` and `cost_usd_measured`.

Muse Wonder is in through Meta's pay-per-token API, with lm15's meta
provider (entrant `muse-wonder-api`, "Muse Wonder (API)"): the first
photos from its 2026-10-14 API reruns (`run-points-v2-muse-api-20261014.jsonl`,
`run-points-v1hd-muse-api-20261014.jsonl`), the new photos in the `-b2`
runs; both prompts, 3 tries at medium, as for every model (v2 kept, 72.3).
Its plan copy (`muse-wonder`, the one on v2.7) is out of quota until
2026-10-19. Not in v3: Muse Wonder Full, which exists only on the plan
(HTTP 404 on the pay-per-token key, checked again 2026-10-14). Muse Wonder's
cost on the chart is the API's per-token price. The white-cow task is easy for most models (one object,
found by nearly all); it mostly separates the few that miss it.

Report: `results/point-benchmark-v3`; candidates `-v2`, `-v1hd`; charts:
`charts/point-benchmark-v3`.

Error bars (`uv run scripts/plot_error_bars.py point-benchmark-v3`,
`charts/point-benchmark-v3-error-bars`): the same method as for v2.7
(run-to-run variation from the 3 tries, Student-t, Bonferroni over the
models on each chart: 28, and 19 language models), now with the published
chart's pinned labels for whichever version is drawn. On a chart with error
bars the bottom of the scale now also makes room for the lowest bar end
(`segbench/chart_editorial.py`): on v3's language-model chart, which starts
at 20% without bars, four bars ran off the bottom with no end mark. Charts
without error bars redraw identically (checked v2.7 and v3).

### v3 with error bars over the tasks (2026-10-14)

Asked for the same day: each model's best run, and error bars over the
tasks instead of over the tries (`uv run scripts/plot_task_error_bars.py
point-benchmark-v3`, `charts/point-benchmark-v3-task-error-bars`). A run
is one complete pass over the 11 tasks (language models 3, local models 1);
the dot is the run with the highest mean. The bar is a Student-t interval
over the run's 11 task scores, Bonferroni over the models on the chart (28,
or 19). The bars are long (about ±25 to ±50 points) because the tasks
differ in difficulty; that difficulty is shared by every model, so a
comparison of two models should use their per-task differences (paired):
for the top models about ±2 to ±5 points of standard error, and 17 of the
378 pairs differ under a Bonferroni correction over all pairs. Best-run
selection is made on the test photos and favours models with more runs and
more erratic answers (GLM 5.3 Flash +10 points over its mean of runs,
Mistral Large 4 +7). Drawn both ranked (one row per model) and against
cost; on the cost chart the scale now also leaves room for the highest bar
end (`segbench/chart_editorial.py`; charts without bars unchanged).

### v3: how much the ranking depends on the photos chosen (2026-10-14)

`uv run scripts/task_robustness.py point-benchmark-v3`,
`charts/point-benchmark-v3-task-robustness`. Asked: "if I had kept only x
of the y tasks, how robust is the relative ranking". Each model scored by
its best run, re-chosen within every subset; every subset of the 11 tasks
enumerated. Results: keeping 8 of 11, rank agreement with the full ranking
0.98 (Spearman, median), same top five in 92% of choices, same winner in
47%. Two disjoint sets of 5 tasks agree at 0.82, projected (Spearman-Brown)
to 0.91 for two disjoint sets of 11: the broad order is robust, the order
of neighbours is not. GPT-6 Astra wins on 8 of the 11 leave-one-out sets
(Gemini 4 Argon without ants or jars, GPT-6.1 Sol without fig leaves).
With the mean of runs instead of the best run the ranking is nearly the
same (0.98). This replaces the error-bar-over-tasks view
(`plot_task_error_bars.py`) as the answer to that question.
