# segbench

How good are today's AI models at **image segmentation** on messy, real
pictures, what does it cost, and how long does it take?

We send each model a small set of real photos and ask it to segment
something. The question is always phrased as a segmentation request (never
"how many logs are there?"). Then, for each image, a small piece of code
turns the model's segmentation into the number that actually matters for
that picture (a count, a fraction of the image, the right part found or
not). We score that number against the truth.

Why not score pixel overlap with a hand-drawn mask (IoU)? Because for many
of these pictures the "correct" mask is unclear: where does a reddish
clump of leaves end? A model can draw a reasonable outline that matches a
human mask poorly. The real metric asks the question we care about.

## The result: one chart

A bubble chart in the style of [Artificial Analysis](https://artificialanalysis.ai):

| axis | what it shows |
|---|---|
| **y** | quality: mean score over all tasks, 0 to 1 |
| **x** | cost: US dollars per image (log scale) |
| **bubble size** | inference time: median seconds per image |

Top left is best: high quality, low cost.

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

Every model is run at three thinking levels, each translated into the
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

## Images and tasks (first ideas)

The first photos are in `images/` (see `images/README.md` for what each
one shows). Planned pictures, each with the real metric we expect to score:

| image | what the model is asked to segment | real metric (draft) |
|---|---|---|
| a stack of ~30 dishes | each dish | count error |
| a log pile | each log (end) | count error |
| an animal with horns | the horns | horns found, tips in the right place |
| an engine | one named mechanical part | is the mask on the right part |
| autumn tree leaves | the reddish clumps (fuzzy edges) | red share of the image vs. a colour-based reference |
| a computer screenshot | named interface elements | elements found / missed / invented |
| a fig tree | each leaf | count error |

Each task gets its own folder in `tasks/` with the image's ground truth and
the small scoring function. Scores are mapped to 0..1 so tasks can be
averaged.

## How a run works

1. Load an image and its task.
2. Ask the model to segment, in **one shared output format** (draft: a JSON
   list of objects, each with a label and a polygon in coordinates 0..1000
   of the image width and height). Gemini can also return masks; any other
   format is converted to the shared one before scoring.
3. Turn the output into masks, then run the task's scoring function.
4. Save one line per call to `results/` (JSON Lines, never overwritten):
   model and exact ID, task, image, prompt version, raw reply, tokens in,
   tokens out, reasoning tokens, cost in USD, wall-clock seconds, whether
   the reply could be read, and the score.

Rules that keep the numbers honest:

- **Cost** = tokens × the provider's price on the day of the run (the
  price table and its date are saved with the results). When a provider
  reports the cost itself (OpenRouter does), we keep both.
- **Cost of models on our own GPUs** (SAM, DINOv3, a local Qwen) = GPU
  seconds × a stated price per GPU-hour. That price is a choice and is
  printed on the chart.
- **Time** = wall-clock seconds from request sent to full reply received.
  API times depend on the provider's load, so we keep the median of
  several runs and record the date and time of day.
- **A reply we cannot read scores 0** but its cost and time still count.
- **Repeats**: each model × image is run several times (start with 3) so
  we can show how much a score moves between runs.
- Every prompt has a version number; changing a prompt means a new
  version, never an edit in place.

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
docs/      research notes and candidate model survey
```

## Status

- [x] project set up, API keys in place
- [x] OpenAI and Anthropic through our subscriptions, images tested
- [x] initial specialist survey with primary sources and availability notes
- [x] GPU and installed-model inventory (`docs/gpu-setup.md`)
- [ ] free a GPU for runs, then install SAM 1/2/3, DINOv3, gen2seg, YOLOE-26, RF-DETR Seg
- [x] first batch of photos (18, location data removed)
- [ ] collect the remaining images (dishes stack, horns, engine, screenshot)
- [ ] write ground truth and a scoring function per task
- [x] confirm exact model IDs and which ones accept images (2 excluded)
- [x] remote calls with token, time and cost tracking (`segbench/call.py`)
- [ ] decide how SAM 1/2 and DINOv3 receive the task
- [ ] runner with cost and time tracking
- [ ] the chart

## Licence

- **Code and text:** [Apache License 2.0](LICENSE).
- **Photos in `images/`:** [Creative Commons Attribution 4.0](images/LICENSE-CC-BY-4.0.txt)
  (CC BY 4.0). Anyone may use, share and change them, also commercially,
  as long as they credit "segbench authors" and link to the licence.
