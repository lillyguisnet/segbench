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

Exact model IDs are confirmed the first time each one is called and
written here. "Vision?" means we still have to check that the model accepts
images; a model without image input drops out.

| model | maker | route (API key) | vision? |
|---|---|---|---|
| Qwen 27B | Alibaba | OpenRouter / DeepInfra, or local on lambda | check: need the vision variant |
| Luna (`gpt-5.6-luna`) | OpenAI | ChatGPT subscription: `openai-codex:` | yes, tested |
| Terra (`gpt-5.6-terra`) | OpenAI | ChatGPT subscription: `openai-codex:` | yes, tested |
| Sonnet (`claude-sonnet-5-5`) | Anthropic | Claude subscription: `claude-code:` | yes, tested |
| GLM 5.3 | Z.AI | `ZAI_API_KEY` | check: may need the "V" variant |
| GLM 5.3 Flash | Z.AI | `ZAI_API_KEY` | check |
| DeepSeek Pro | DeepSeek | `DEEPSEEK_API_KEY` | check |
| DeepSeek Flash | DeepSeek | `DEEPSEEK_API_KEY` | check |
| Kimi K3 | Moonshot | `MOONSHOTAI_API_KEY` (also Together, Fireworks, Parasail) | yes |
| Gemini Pro | Google | `GEMINI_API_KEY` | yes |
| Gemini Flash | Google | `GEMINI_API_KEY` | yes |
| Gemini Flash Lite | Google | `GEMINI_API_KEY` | yes |

All API calls go through [lm15](https://lm15.dev), one Python client for
every provider, so the request code is the same for every model.

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
- **Check that it works**: `uv run scripts/check_subscriptions.py` draws
  four red circles, asks each model to segment them, and checks that it
  finds them in the right places. That proves the image reached the model.
  The script refuses any model that is not routed through a subscription.
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
uv run scripts/check_subscriptions.py # OpenAI + Anthropic logins work
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
- [ ] confirm exact model IDs and which ones accept images
- [ ] decide how SAM 1/2 and DINOv3 receive the task
- [ ] runner with cost and time tracking
- [ ] the chart

## Licence

- **Code and text:** [Apache License 2.0](LICENSE).
- **Photos in `images/`:** [Creative Commons Attribution 4.0](images/LICENSE-CC-BY-4.0.txt)
  (CC BY 4.0). Anyone may use, share and change them, also commercially,
  as long as they credit "segbench authors" and link to the licence.
