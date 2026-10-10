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
