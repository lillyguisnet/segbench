# segbench

How good are today's AI models at **image segmentation** on messy, real
pictures, what does it cost, and how long does it take?

We send each model a small set of real photos and ask it to find or
segment something. Then, for each photo, a small piece of code turns the
answer into the number that actually matters for that picture (a count, a
size, the share of a tree that has turned red, where a road goes). We
score that number against an answer key we made by hand.

Why not score pixel overlap with a hand-drawn mask (IoU)? Because for many
of these pictures the "correct" mask is unclear: where does a reddish
clump of leaves end? A model can draw a reasonable outline that matches a
human mask poorly. The real metric asks the question we care about.

## The design: three tracks, three charts (locked 2026-10-09)

Segmenting a picture is two jobs: **finding** the objects and
**outlining** them. Some models can only do one of them (SAM 1 and 2
outline but cannot read a request; DINOv3 compares but cannot outline).
Others do both. A model that has to do both at once can lose points for
either reason, so we measure each job on its own, plus the whole task.

| track | the model gets | it answers with | scored on | entrants | cost and time counted |
|---|---|---|---|---|---|
| **1. Find** | the photo + the request ("put a dot on each cow") | one dot per object (and a label where the task needs one) | whether each dot lands inside an answer-key mask | language models, SAM 3, YOLOE-26, RF-DETR (its 80 categories only), DINOv3 look-alike search | the finder only |
| **2. Outline** | the photo + the request + **one dot on each object**, taken from the answer key | one outline per given dot | the task's measurement (size, area, overlap, path) | language models, SAM 1, SAM 2.1, SAM 3, gen2seg, DINOv3 (tree task) | the outliner only |
| **3. Whole task** | the photo + the request, no hints | outlines, labelled where needed | finding and measurement together | single models that do both, and **labelled pairs** ("Gemini Flash + SAM 2.1") | everything: all parts, end to end |

Why the outline track gets the answer key's dots, not the best finder's:
a finder's mistakes would then lower every outliner's score, and the
outline results would go stale every time a better finder appears. It
gets a **dot, not a box**, because a box would give away the size we
measure (log diameters).

**Pairs in track 3** are built from the best-value finders of track 1
(those on its frontier, below) combined with the best-value outliners of
track 2. Each pair is its own bubble, labelled with both parts.

**What we may and may not claim.** Each claim names its track. A model's
outlining skill is judged in track 2 only; a low track-3 score never
means "bad at outlining", because track 3 also asks it to find.

### The six tasks

Full detail and the reasons for each choice: `docs/task-ideas.md`.

| # | photo | skill tested | segment… | track 1 (find) | track 2 (outline) |
|---|---|---|---|---|---|
| 1 | `log_ends_closeup_1` | many touching, similar objects | each cut log end | ~70 logs | diameter of each log end |
| 2 | `field_with_cows` | tiny, distant objects; one odd one out | each cow | ~13 cows, incl. the brown one | outline stays on the cow |
| 3 | `fig_plant` | look-alike plants | each fig leaf, not the other plants | ~12 fig leaves | total fig-leaf area |
| 4 | `autumn_tree_orange_1` | fuzzy, gradual regions | the reddish (not yellow) foliage | — | % of the crown that is red |
| 5 | `farm_road_1` | long, soft-edged region; direction | the gravel road | — | the road's path and width at ~6 heights |
| 6 | `kitchen_counter_dishes` | meaning: dirty vs. clean | each dish, labelled dirty or clean | ~10 dirty + ~10 clean, label must be right | outline of each dish |

Tasks 4 and 5 have nothing separate to find: they are run once, and that
run counts in both track 2 and track 3.

**Trick questions.** Each photo is also asked for something that is not
there (cows in the log picture, cars in the fig picture). The right
answer is nothing. Reported as a separate "invents objects" score next to
each bubble, not mixed into quality.

### Scores

Every score is 0 to 1; a track's quality is the mean over its tasks.

- **The answer key is masks.** A dot on an object is right anywhere on
  the object, so the truth for "where is it" is the object's mask, not a
  dot. Every object of every task gets its own mask (plus a label for the
  dishes: dirty, clean or unsure); widths, areas and the road's path are
  measured on these masks. The dot given to outliners in track 2 is the
  point deepest inside each answer-key mask.
- **Finding:** an answer dot is right when it lands inside an answer-key
  mask that no earlier dot has claimed (each object can be found once).
  Score = F1, which punishes both missed objects and invented ones; the
  count error is reported too. For dishes, a dot counts only if its label
  (dirty or clean) is right; *unsure* items are left out.
- **Outlining:** the task's measurement, compared with the answer key and
  turned into 0..1 by a stated tolerance (written in each task's folder).
- **Whole task:** an answer-key dot is found when it falls inside exactly
  one of the model's outlines (with the right label for dishes). Score =
  the mean of that finding score and the outlining score on the objects
  found.
- **Specialists** answer with masks; for track 1 each mask becomes one dot
  at the point deepest inside the mask (the centre can fall outside a
  curved shape).

### How the answer key is made, and its one known bias

Masks are drawn in Lilly's annotation app **recorn** on the canonical
photos (`scripts/prepare_annotation_set.py`), with SAM 3.1 suggesting
outlines that a person accepts, corrects or redraws. That is fast, but
an answer key that starts from SAM's outlines can favour SAM-family
outliners in track 2. Three safeguards:

- **Every mask is checked by a person at full zoom**, and recorn records
  where each one came from (SAM text prompt, SAM click, or hand-drawn)
  and how many pixels the person changed. That record is kept with the
  answer key.
- **Most track-2 scores are tolerant measurements** (log diameter, leaf
  area, red share, the road's path), not pixel-by-pixel overlap, so edge
  details matter less.
- **A bias check:** for ~10 objects (cows and dishes, the two tasks scored
  by overlap) a second mask is drawn fully by hand. If SAM models score
  clearly better against the SAM-assisted masks than against the
  hand-drawn ones, that is reported next to the track-2 chart.

**One pixel grid.** All photos must be stored upright (no EXIF rotation
tag; the preparation script refuses any other photo). Answer keys,
models and scorers then all see the same pixels; model coordinates
(0..1000) are mapped onto that grid.

### The charts

Three bubble charts, one per track, in the style of
[Artificial Analysis](https://artificialanalysis.ai):

| axis | what it shows |
|---|---|
| **y** | quality: mean score over the track's tasks, 0 to 1 (spread over repeats in `points.csv`) |
| **x** | cost: US dollars per image (log scale), only the parts that track counts |
| **disc colour** | speed: median seconds per image, end to end; green = fast, red = slow |

Each model at each thinking level is one bubble; pairs have their own
marker. A line joins the **best-value models** (the Pareto frontier: no
other bubble is both better and cheaper). Top left is best.

Drawing them: `uv run scripts/draw_charts.py results/<run>.jsonl` (or
`--demo` for made-up data, stamped FAKE DATA). It writes a PNG to post, a
PDF to print, an SVG for the web and `points.csv` (the numbers behind
every bubble) to `charts/`. What a call record must hold and how a bubble
is computed: `segbench/chart_data.py`. Choices made for readability:

- **Quality is shown 0 to 100** (the 0..1 score × 100), easier to read.
- **One cost axis for all three charts**, in dollars per 1,000 images
  (per image, the GPU models would read $0.0000008).
- **Square image** (2160 × 2160): feeds show posts as squares.
- **Every model is a disc of the same size with its maker's logo in
  white** (the models' own mark where it is better known: Gemini, Qwen,
  Kimi; sources and licences: `assets/logos/SOURCES.md`). API models and models
  on our GPU look the same: anyone can rent the GPU, so their cost and
  speed compare fairly.
- **Speed is the disc colour**, green (fast) to red (slow), on a log scale
  from 1 s to 100 s, the same on every chart: the API models take 3 s to
  minutes, so that is where colour must tell them apart; everything under
  1 s is full green. The green is darker than the red so that red-green
  colour-blind readers still see a difference in lightness.
- **No title, caption or footnotes on the image**: the post that shares it
  says what it is. Made-up data gets a faint FAKE DATA stamp.
- **A pair** (one model finds, another outlines) shows the finder's logo
  with the outliner's logo on a small white badge.
- **The cost axis spans the data** plus a third of a decade each side,
  the same on all three charts.
- `--level min` (or medium, max) shows one thinking level only; with all
  levels, each label ends with its level.
- **A second style**, `--style editorial` (`segbench/chart_editorial.py`):
  a header (track, title, a one-line key), the best-value models as big
  blue-ringed discs joined by a smooth curve over a soft blue glow, every
  other model a small white disc with its logo in brand colour, and every
  model named with its maker and seconds per image (crowded ones in a
  column, with thin leader lines). The curve is monotone: it never dips or
  overshoots between two models, but it is still a guide for the eye.
- **The best-value line joins the frontier models with straight
  segments**, cheapest to best, as Artificial Analysis draws it. It is a
  guide for the eye: a point on a segment between two models is not a
  model. Their names are in bold. A model that skipped tasks (they count
  as 0) is never on it.

### Still to settle, in the pilot run

1. How a language model receives the dots in track 2: drawn as numbered
   markers on the photo, or written as coordinates. Both are tried on one
   photo; the better one is then used for **every** model.
2. Gemini answers with points as (down, across) even when asked for
   (across, down). Decide: score as written, or accept each family's own
   order.
3. The colour rule that defines "reddish" for task 4, shown on the photo
   before it is fixed.
4. Cutlery and knives in the dish count (current plan: left out).

## Models

### General models (vision + language, called through an API)

The list, with exact routes, lives in `segbench/models.py`; list prices in
`prices.json`. "Sees images" was observed on 2026-10-08 with
`scripts/check_models.py` (four red circles, two blue squares: a model
passes when it outlines exactly the four circles in the right places).

| model | exact model | route | sees images | 4-circle test |
|---|---|---|---|---|
| Luna | `gpt-6-luna` | ChatGPT plan (`openai-codex:`) | yes | pass |
| Terra | `gpt-5.6-terra` (no GPT-6 Terra exists) | ChatGPT plan (`openai-codex:`) | yes | pass |
| Sonnet | `claude-sonnet-5-5` | Claude plan (`claude-code:`) | yes | pass |
| Opus | `claude-opus-5-5` | Claude plan (`claude-code:`) | yes | pass |
| Astra | `gpt-6-astra` | ChatGPT plan (`openai-codex:`) | yes | pass (OpenAI "overloaded" once) |
| Sol | `gpt-6.1-sol` | ChatGPT plan (`openai-codex:`) | yes | pass |
| Gemini Pro | `gemini-3.1-pro-preview` | Google API | yes | pass |
| Gemini Flash | `gemini-3.8-flash` | Google API | yes | pass |
| Gemini Flash Lite | `gemini-3.5-flash-lite` | Google API | yes | right circles, but points written as [y, x] |
| GLM 5.3 Flash | `glm-5.3-flash` | Z.AI API | yes | pass |
| DeepSeek Flash | `deepseek-flash` (= V4.1 Flash) | DeepSeek API | yes | pass |
| Kimi K3 | `kimi-k3` | Moonshot API | yes | pass |
| Qwen 27B | `qwen/qwen3.8-27b` | OpenRouter, **Alibaba host only** | yes | pass on Alibaba, fail on Cerebras |
| ~~GLM 5.3~~ | `glm-5.3` | Z.AI API | **no**: rejected with an error | excluded |
| ~~DeepSeek Pro~~ | `deepseek-v4-pro` | DeepSeek API | **no**, and silently: the picture is replaced by "[Unsupported Image]" and the model answers anyway | excluded |

All calls go through [lm15](https://lm15.dev), one Python client for
every provider: `segbench/call.py` makes one call and returns one record
(reply, tokens, seconds, cost, or the error). A failed call is a record, not
a crash, so a run always finishes.

What the first checks taught us:

- **Pin the host of open-weight models.** OpenRouter offered 18 hosts for
  Qwen 3.8 27B, some with compressed (fp4/fp8) versions, at prices that
  differ up to 20×. The same model passed 3/3 on Alibaba's host and 0/3 on
  Cerebras's. We pin the maker's own host and refuse fallbacks.
- **A model can fail to see the picture without any error** (DeepSeek
  Pro). Every new model must pass the circle test before it is benchmarked.
- **Speed varies a lot from one call to the next.** Kimi K3 took 141 s and
  then 264 s on the same picture; GLM 5.3 Flash 141 s and then 22 s. One call
  per model says little: we keep the median of several.
- **Gemini models like [y, x].** Gemini Flash Lite found every circle but
  wrote coordinates as [y, x], Google's own convention, against the
  prompt. Scored as written it fails. Whether to also test each family with
  its own native format is an open decision.
- **Cost is computed from tokens × list price** (`segbench/cost.py`).
  Google counts thinking tokens apart from the answer but bills both; the
  code adds them back. Our estimate for Qwen matched what OpenRouter billed
  within 1 %.

### Thinking: minimum, medium, maximum

Every model is run at three thinking levels (two for Kimi K3 and GPT-6
Astra, whose "max" was dropped on 2026-10-09 as too expensive), each
translated into the
model's own settings in `segbench/models.py` (`THINKING`): **min** = off
where the model can stop thinking, else its lowest level; **max** = its
highest level that really reaches the provider; **medium** = the middle.
Checked on 2026-10-08 (`scripts/probe_thinking.py` reads what is sent;
`scripts/check_thinking.py` measures the effect on the circle test, all 60
calls at once, in about 6 minutes).

Thinking tokens on the circle test (two runs each, 2026-10-08):

| model | min | medium | max | slowest call at max |
|---|---|---|---|---|
| Luna (GPT-6) | 0 | ~240 | 900–1,500 | 18 s |
| Terra | 0 | ~300 | 7,000–14,000 | 402 s |
| Sonnet | 0 | 0 (chose not to) | ~2,700 | 25 s |
| Opus | 0 at low (cannot stop) | ~450 | ~6,200 | 58 s |
| Astra | 0 at low (cannot stop) | 0 | 1,600–4,300 | 119 s |
| Sol | 0 at low (cannot stop) | ~150 | ~2,500 | 60 s |
| Gemini Pro | 0 | ~1,600 | 1,800–9,000 | 73 s |
| Gemini Flash | 0 | ~550 | 2,000–3,100 | 30 s |
| Gemini Flash Lite | 0 | ~470 | ~520 | 9 s |
| GLM 5.3 Flash | 50–230 (cannot stop) | 50–250 | ~3,000 | 64 s |
| DeepSeek Flash | 0 | 1,800–3,700 | 2,200–2,600 | 18 s |
| Kimi K3 | 0 | 1,300–2,500 | 6,000–9,700 | 244 s |
| Qwen 27B | 0 | 180–530 | ~1,000 | 20 s |

What to know:

- **The dial works on every model**, but levels are not comparable across
  makers: Terra's "max" thinks 40 times longer than Qwen's.
- **Off costs accuracy for some**: with thinking off, Luna (5.6 and 6) and Kimi failed
  the circle test both times; at medium they passed.
- **Some cannot stop thinking**: Opus 5.5 (Anthropic refuses both of its
  "off" settings), GPT-6 Astra and GPT-6.1 Sol (OpenAI accepts only low and
  up). Their minimum is "low"; on the circle test they then chose not to
  think at all.
- **We use the newest version of every model.** Luna is GPT-6 Luna since
  2026-10-08 (the earlier rows of `results/` are GPT-5.6 Luna).
- **The ChatGPT plan has a weekly limit, shared with everyone using the
  account.** It was reached on 2026-10-08 and reset on 2026-10-09; GPT-6
  Luna was first checked through OpenRouter, then on the plan once it was
  back (same results: off fails the circle test, medium and max pass). See how much is left with the plan's usage
  endpoint (`https://chatgpt.com/backend-api/wham/usage`, the Codex login's
  token). A full benchmark run must check it before starting.
- **Hidden models on the ChatGPT plan**: `gpt-6-astra` is listed only to
  Codex 0.153 or newer (lm15 1.2.1 says it is 0.147), and `gpt-6.1-sol`
  and `gpt-6-luna` are listed to nobody; all three answer. Use
  `LM15_CODEX_CLIENT_VERSION=0.153.4` to see the full list.
- **Thinking runs vary a lot**: the same model and level can think twice as
  long on the next call (Luna max: 8,800 then 17,800 tokens).
- **Long calls get cut**: the ChatGPT-plan route dropped both Terra-max
  calls after 5–6 minutes ("connection closed mid-chunk"); the retry
  worked (3 and 7 minutes). The benchmark runner must retry dropped
  connections (and keep a record of the dropped attempt), but never retry
  a reply the model actually gave.
- **lm15 1.2.1 has gaps here** (to fix in lm15): `effort="off"` sends
  nothing to Sonnet 5.5, which thinks by default (we send Anthropic's
  `thinking: {"type": "between_tools"}` ourselves); and some words are
  quietly changed instead of refused, contrary to lm15's own rule
  ("minimal" becomes "low" on Sonnet and Kimi, "medium" becomes "low" on
  Kimi, "xhigh"/"max" become "high" on Gemini).

### Why lm15 and not FunctAI

[FunctAI](https://github.com/MaximeRivest/functai) (built on lm15) turns a
typed Python function into a model call. It is the right tool for a task
like "classify this", but not for a benchmark:

- **We must own the exact prompt.** FunctAI writes the prompt from the
  function (name, docstring, comments, types) and its own layout; a new
  FunctAI version can change the words sent, and so the scores.
- **Pictures are not in a release yet.** Picture inputs landed on
  2026-10-08 in unreleased code; in the published 1.2.0 a picture is sent
  as text and the model guesses.
- **One call must be one measurement.** FunctAI re-asks when a reply cannot
  be read and can answer from its disk cache; both are good for apps and
  wrong for us (an unreadable reply scores 0, and a cached reply has no
  real time or cost). Both can be turned off, but then little of FunctAI is
  left to use.
- **Different families want different output formats** (see Gemini's
  [y, x]); one typed return value hides that.

### OpenAI and Anthropic run on our subscriptions

Luna, Terra and Sonnet do not use API keys. lm15 reuses the login of the
Codex CLI (`~/.codex/auth.json`) and of Claude Code
(`~/.claude/.credentials.json`), so these calls are covered by our
ChatGPT and Claude plans instead of being billed per token.

- **Always write the provider prefix**: `openai-codex:gpt-5.6-luna`,
  `claude-code:claude-sonnet-5-5`. A bare `gpt-5.6-luna` could be sent to
  the pay-per-token API, because `.env` also holds `OPENAI_API_KEY` and
  `ANTHROPIC_API_KEY`.
- **The Codex route refuses a token limit** (`max_tokens`). Send an empty
  `Config()` there.
- **Check that it works**: `uv run scripts/check_models.py luna terra sonnet`
  (see above). `segbench/call.py` refuses a subscription model whose route
  does not resolve to a subscription login.
- To list the models a login offers:
  `uv run python -c "from lm15 import OpenAICodexLM as L; print([m.id for m in L().list_models()])"`
  (or `ClaudeCodeLM`).
- If lm15 says the login has expired, run `codex login` or `claude`, then
  `/login`, on the machine that runs the benchmark.

**What this means for the chart.** A subscription call costs us nothing
per call, but the x-axis must compare models fairly. So these models get
the **cost the same tokens would have at the public API price**, which is
what a reader would pay. Their speed can also differ from the paid API,
because it is a different server. The chart will say so.

### Specialist segmentation models (run on our own GPUs)

| model | what it needs as input |
|---|---|
| SAM 1 | points or boxes, not text |
| SAM 2 | points or boxes, not text |
| SAM 3 | a short text phrase ("horns", "fig leaf"); a SAM 3.1 video service already runs on lambda, see `docs/gpu-setup.md` |
| DINOv3 | an image backbone: needs a small head or text alignment to segment |

**Native text input is not required.** We accept automatic, spatially
prompted, example-based and fixed-category models, and build pipelines
around missing capabilities. General usefulness across our pictures is
the goal, not a particular prompting interface.

The first candidate survey is in `docs/model-survey.md` (8 October 2026).
It covers gen2seg, YOLOE-26, RF-DETR Segmentation, Mask-DAPE, SegMAN,
COSINE, SegNext and SegGPT, plus EoMT/PMT as concrete segmentation
implementations for DINO. It separates available implementations from
recent papers whose model downloads still need verification. These are
candidates, not installed or benchmarked models.

**Compare complete pipelines.** Name every component and count prompt
generation, mask selection, refinement and retries in total cost and
end-to-end time. Keep human-assisted results separate from automatic
runs. Ground-truth-derived clicks/boxes are oracle-assisted diagnostics,
not automatic benchmark inputs.

## How a run works

1. Load a task (photo, request, answer key, scorer) and a track.
2. Send the request to every entrant, all at once (they wait on remote
   servers, not on this machine); specialists run on GPU 0.
3. Read the answer in the track's shared format (dots, or outlines as
   polygons in coordinates 0..1000 of the image width and height;
   specialist masks are kept as masks). Score it with the task's scorer.
4. Save one line per call to `results/` (JSON Lines, never overwritten):
   track, task, model and exact ID, thinking level, prompt version, raw
   reply, tokens in / out / thinking, cost in USD, seconds, whether the
   reply could be read, and the score.

Rules that keep the numbers honest:

- **Same request for everyone** in a track and task, word for word, with
  a version number. Changing a prompt means a new version, never an edit
  in place.
- **Cost** = tokens × the provider's list price on the day of the run (the
  price table and its date are saved with the results). When a provider
  reports the cost itself (OpenRouter does), we keep both. Subscription
  models are charged at their public API price.
- **Cost of models on our own GPUs** (SAM, DINOv3, gen2seg, YOLOE,
  RF-DETR) = rental price of the GPU ($0.22/hour for an RTX 3090, RunPod
  Community Cloud, 8 Oct 2026) ÷ images per hour **under load** (best
  batch size or several workers), because API prices also assume busy,
  shared GPUs. Speed (disc colour) is still one image at a time. Method:
  `specialists/README.md`.
- **Time** = wall-clock seconds from request sent to full reply received.
  API times depend on the provider's load, so we keep the median of the
  repeats and record the date and time of day. A pair's time is the sum
  of its steps.
- **A reply we cannot read scores 0**, but its cost and time still count.
- **Repeats:** 3 per model, thinking level, task and track. Specialists
  give the same answer every time, so they are scored once and timed
  repeatedly.
- **Retries:** a dropped connection is retried (and the dropped attempt
  recorded); a real reply from the model is never retried.
- **Thinking levels:** min, medium and max for every model, except Kimi
  K3 and GPT-6 Astra (min and medium: their max cost up to $0.16 and
  $0.24 per call).
- **Answer-key hints are only used where the track says so** (track 2).
  Nothing derived from the answer key ever reaches a track-1 or track-3
  entrant.
- **GPU guard:** specialist timings are flagged if another program used
  GPU 0 during the run.

**Size of a full run:** 37 model-and-level combinations × 14 runs per
model (4 find, 6 outline, 4 whole-task; tasks 4 and 5 count in both track 2
and 3) × 3 repeats ≈ 1,550 calls, plus ≈ 650 for the trick questions.
The pilot measures the real cost and the ChatGPT-plan usage first.

## Setup

On `lambda` (the GPU server, where this project lives):

```bash
cd ~/Projects/segbench
uv sync                               # installs lm15 into .venv
source .env                           # loads the API keys into the shell
uv run scripts/check_models.py        # every remote model answers and sees the picture
```

API keys live in `.env`, which is **never committed** (`.gitignore`
blocks it). `.env.example` lists the names of the keys you need; ask Maxime
for the values.

## Layout

```
images/    the benchmark photos (with where each came from and its licence)
tasks/     one folder per task: ground truth + scoring function
scripts/   run models, score, draw the chart
results/   raw call records (JSON Lines), one file per run
charts/    the drawn charts (charts/demo/: made-up data, not committed)
assets/fonts/  Inter, the charts' font (SIL Open Font License, OFL.txt)
assets/logos/  makers' logos for the charts (trademarks; SOURCES.md)
docs/      research notes, task choices, candidate model survey
specialists/  one environment per specialist model family (see its README)
```

## Status

- [x] project set up, API keys in place
- [x] OpenAI and Anthropic through our subscriptions, images tested
- [x] initial specialist survey with primary sources and availability notes
- [x] GPU and installed-model inventory (`docs/gpu-setup.md`)
- [x] GPU 0 taken for the benchmark; SAM 1/2.1/3, YOLOE-26, RF-DETR Seg, gen2seg installed, smoke-tested and priced (`specialists/README.md`)
- [ ] DINOv3 + segmentation head
- [x] first batch of photos (18, location data removed)
- [x] six tasks chosen (`docs/task-ideas.md`); horns, engine and screenshot left out for now
- [x] three-track design locked (above)
- [ ] answer-key masks in recorn (SAM 3.1 service moved to GPU 0 for this; photos prepared)
- [ ] shared answer formats, prompts, and a scorer per task (tested on fake answers)
- [ ] adapters: specialists into the shared formats; finder + outliner pairs; DINOv3 text matching and look-alike search; gen2seg colours into separate outlines
- [x] pilot run 1 (2026-10-09): every model at medium thinking, tracks find / whole / trick, once each; 208 calls, all answered, $5.36 at list prices ($1.30 really billed), 3 % of the ChatGPT plan's week (`results/run-pilot-1.jsonl`)
- [ ] full run (all thinking levels, 3 repeats), after the open points above
- [x] confirm exact model IDs and which ones accept images (2 excluded)
- [x] remote calls with token, time and cost tracking (`segbench/call.py`)
- [x] SAM 1/2 and DINOv3 receive the task through the tracks (above)
- [x] runner for the tracks that need no answer key: `scripts/run.py` (find, whole, trick; resumable, retries dropped calls only); prompts in `segbench/tasks.py` (version v1); `scripts/summarize_run.py`
- [ ] track 2 (outline) in the runner: needs the answer key's dots
- [x] draft answer key from model consensus, for the person making the real one (`scripts/consensus.py`, methods in `segbench/consensus.py`): Dawid-Skene voting for objects and labels, STAPLE for regions, a review queue of contested objects, and a provisional ranking that judges each model only against other companies' models. Never used as the answer key: models that share a blind spot agree on the same mistake
- [x] chart drawing (`scripts/draw_charts.py`), checked on made-up data
- [x] every model's dots on its photo, each dot drawn as its maker's logo, one image per photo (`scripts/draw_points.py`, writes `results/views/points-find/`); models of one maker are told apart by shades of its colour
- [ ] the three charts, from real scores

## Licence

- **Code and text:** [Apache License 2.0](LICENSE).
- **Photos in `images/`:** [Creative Commons Attribution 4.0](images/LICENSE-CC-BY-4.0.txt)
  (CC BY 4.0). Anyone may use, share and change them, also commercially,
  as long as they credit "segbench authors" and link to the licence.
- **Font in `assets/fonts/`:** [Inter](https://github.com/rsms/inter) by
  The Inter Project Authors, under the [SIL Open Font License 1.1](assets/fonts/OFL.txt);
  cut down to Latin letters for the charts.
- **Logos in `assets/logos/`:** trademarks of their owners, used only to
  show who made each model; not covered by the Apache licence. Sources in
  `assets/logos/SOURCES.md`.
