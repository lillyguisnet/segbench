# Adding a model

Every model in the comparison got the same treatment. This page is that
treatment, written down, so a new model gets it too without anyone digging
through old conversations. Two kinds of models:

- **Remote**: called over the internet, with an API key, through OpenRouter,
  or on one of our subscriptions (ChatGPT, Claude, Muse Code).
- **Local**: run on our own GPUs, mostly through `~/Projects/finders`.

One script walks a remote model through the steps and refuses to skip one:

```bash
uv run scripts/add_model.py status KEY     # what is done, what is next
```

What it cannot decide for you (which host, which thinking level, which price)
is below, with the rule we have followed so far and where it came from.
The standard runs themselves are in `segbench/protocol.py`.

## A remote model

### 1. Register it

In `segbench/models.py`, add a `Model(...)` with `vision=None` (not checked
yet), and a dated comment: who asked for it, when, what it is.

- **Route**: an lm15 model string, always with its provider prefix
  (`gemini:`, `openrouter:`, `openai-codex:`...). Without the prefix lm15 may
  pick another provider for the same name, or the paid API instead of a plan.
- **Use the newest version** of the model, by its exact id, and say in the
  note whether it is a preview (Mistral Large 4: "Mistral keeps training it
  under the same name, so a score holds for its run date").
- **Through OpenRouter: pin one host, no fallback**:
  `extensions={"provider": {"order": ["HOST"], "allow_fallbacks": False}}`.
  Choose the maker's own host. On 2026-10-08 the same Qwen 27B found the test
  circles 3 times out of 3 on Alibaba's host and 0 out of 3 on Cerebras's
  (some hosts serve compressed copies).
- **Subscriptions, never the pay-per-token key** (Maxime, 2026-10-08 and
  2026-10-12): OpenAI models on `openai-codex:`, Anthropic on `claude-code:`,
  Meta's Muse on `meta:` with `credential="muse-code"` (`segbench/muse_plan.py`).
  `billing="subscription"`. A new plan with its own login needs its own small
  module like `muse_plan.py` that fails rather than falling back to the paid
  key, an entry in `segbench/call.py` `SUBSCRIPTION_CREDENTIALS`, and a
  concurrency limit in `scripts/run.py` `PER_PROVIDER` (plans are shared with
  the family's own use).
- **Plain model ids, not "-contributor" copies** a provider may train on: the
  screenshots show other people's names and messages.

In `prices.json`, add its price to the current snapshot (`cost.PRICES_DATE`),
with `listed_as` (where you read it) and a dated `note`:

- **The regular list price**, not a launch sale (Maxime, 2026-10-13: Mistral
  was on sale at half price; the chart uses the regular one, the note gives both).
- **The maker's own host's price** for open-weight models on OpenRouter.
- **A subscription model is charged at its API price** per token, like every
  other plan model, with no special label on the chart (Maxime, 2026-10-12).
  A model sold only inside a plan gets its closest sibling's API price, said
  in the note (Muse Wonder Full at Muse Wonder's price).
- **Thinking tokens are billed as output.** If the provider reports them
  outside `output_tokens` (Google does), add it to `segbench/cost.py`
  `THINKING_NOT_IN_OUTPUT`.

A new maker also needs its logo and colour in `segbench/brand.py`, the logo
file in `assets/logos/` and its source and licence in
`assets/logos/SOURCES.md`. Without them the charts show its initials in grey.

`uv run scripts/add_model.py lint KEY` checks all of this without calling anything.

### 2. Check it before any benchmark picture

```bash
uv run scripts/add_model.py check KEY
```

First round: the **circle test** (does it see the picture and point at the
four red circles?) and the **thinking probe** (which thinking levels it
accepts, and how much it thinks at each). Then decide:

- `vision=True` only if the circle test passed. A model can lose the picture
  without any error: DeepSeek V4 Pro silently replaced it with
  "[Unsupported Image]" and answered anyway. That is `vision=False` with a note.
- Its `THINKING` entry (in `models.py`, the comment above `THINKING` has
  every rule): min = off where it can stop thinking, else its lowest level;
  medium = the middle of the levels it really has; max = its highest level
  that is really sent. **Medium is what is benchmarked.** A model with only
  "off" and "on" puts "on" in medium and has no max (Kimi K3, Mistral Large 4).
  Leave max out when it costs far more than the others (Kimi K3 and GPT-6
  Astra, 2026-10-09).

Run `check KEY` again for the second round: the **image-detail probe** (does a
"high detail" setting raise the input tokens?) and the **answer-format check**
(six circles on a wide and a tall picture, with the benchmark's own prompt).

- If a setting raises the input tokens, add it to `scripts/run.py`
  `IMAGE_SETTINGS_V2` for that provider: every model gets the most detail its
  provider offers.
- If the answer-format check fails, **do not change the prompt for that
  model**: everyone gets the same words, and replies are read generously
  instead (we measure seeing, not following a format). Gemini Flash Lite
  failed it (it writes [y, x]) and was benchmarked anyway. Say so in the report.

### 3. Run it

```bash
uv run scripts/add_model.py run KEY        # the plan: how many calls, the prompts; sends nothing
uv run scripts/add_model.py run KEY --go   # sends them; logs in results/run-<run>.KEY.log
```

The standard runs (`segbench/protocol.py`): medium thinking, 3 tries of each
question, both photo prompts (v2 with the trick questions, and v1-hd), and
the harder round of screenshots (round 2). 57 calls. Answers are appended to
the shared files (`results/run-points-v2.jsonl`, `run-points-v1hd.jsonl`,
`run-screens-2.jsonl`). An interrupted run continues where it stopped when
started again; a call the provider never answered is retried, a real reply
never is.

A slow thinking model can take an hour (Mistral Large 4: 2½ minutes per
photo). On a subscription, check the plan's usage window first (Muse:
`uv run python -c 'from segbench import muse_plan; print(muse_plan.usage())'`).

### 4. Preview it

```bash
uv run scripts/add_model.py preview KEY
```

Scores it exactly as the published results are scored, keeps its better
photo prompt (the rule every model had), and ranks it against what is
published. Nothing published changes. `results/preview/KEY/points-*/index.html`
shows each of its answers on the photos. This is what the "here is how it
did, shall we publish?" message is made from. Look at its answers before
writing that message: odd failures (141 log ends where there are 83, one
click far from every label) are worth a sentence.

Then **ask before publishing**.

## Publish (after a yes)

Publishing redraws the shared charts. Commit any unrelated changes first, so
the publishing commit holds only this model.

1. Logo and colour, if the maker is new (step 1).
2. Score both photo prompts for every model, into dated folders:
   ```bash
   uv run scripts/score_points.py --inputs results/run-points-v2.jsonl results/run-specialists-2.jsonl results/run-finders-3.jsonl \
     --out results/point-benchmark-v2-loc-DATE --parser lenient --dish-labels separate --method docs/point-benchmark-v2.md
   uv run scripts/score_points.py --inputs results/run-points-v1hd.jsonl \
     --out results/point-benchmark-v1hd-loc-DATE --parser lenient --dish-labels separate --method docs/point-benchmark-v2.md
   ```
3. Each model's better prompt: `python3 scripts/select_best_prompt.py point-benchmark-v2-loc-DATE point-benchmark-v1hd-loc-DATE`.
   Check every earlier model is unchanged: `git diff results/run-points-best.jsonl`
   should show only the new model's lines.
4. The scores and the chart of the new version (v2.6 -> v2.7):
   ```bash
   uv run scripts/score_points.py --inputs results/run-points-best.jsonl --out results/point-benchmark-v2.7 \
     --parser lenient --dish-labels separate --method docs/point-benchmark-v2.md
   python3 scripts/plot_point_pilot.py point-benchmark-v2.7 point-benchmark-v2.7
   ```
   Copy `charts/point-benchmark-v2.6/index.html` to the new folder.
5. Screenshots:
   ```bash
   LOCAL=$(ls results/run-screens-local-*.jsonl | grep -v raw)
   python3 scripts/plot_screens.py annotations/screens/key-round2.json screens-2 "Can AI click every right thing on a screen?" results/run-screens-2.jsonl $LOCAL
   ```
   If labels overlap, pin them in `scripts/plot_screens.py` `LABEL_SPOTS`.
   Update the model counts written in `charts/screens-2/index.html`.
6. Look at both charts. Then set `PUBLISHED_PHOTOS` in `segbench/protocol.py`
   to the new version (the photo site follows it).
7. Write it down: a section in `docs/point-benchmark-v2.md` ("v2.7 (date):
   model": checks, choices made, anything odd) and a row in the README's
   model table.
8. `uv run python -m unittest discover -s tests -p 'test_*.py'`, commit (the
   repository commits as Lilly), push, then `scripts/publish_site.sh`.

## A model on our GPUs

Local models are timed, and timing needs the GPU to ourselves: start these
runs only when the GPU is free, by hand (Maxime's call each time).

1. **Add it to `~/Projects/finders`** (its README, "Adding a model"): its own
   pinned environment, its documented prompt template, `./finders smoke NAME`.
   It is then usable from any project, not only this one.
2. **One `LocalModel(...)` in `segbench/local_models.py`**: label, maker,
   finders name and task, `plural` if its documentation asks for all
   instances with plurals, and `photo_rounds` (3; 2 for a slow model).
   `photo_copies`: how many copies to try at once under load (what fits in 24 GB).
3. **Photos, timed under load**: `uv run --with pillow scripts/time_finders.py --name finders-N --setups KEY [--gpu 1]`
   (its header says which GPU programs to stop first, and to restart them after).
   Copy the `load_factor` it measured (`results/run-finders-N.timing.json`)
   into its `LocalModel`. Add `run-finders-N.jsonl` to `segbench/protocol.py`
   `LOCAL_PHOTO_RUNS`.
4. **Screenshots**: set `screen_seeds` (3 if the model samples, 1 if it
   always gives the same answer), then
   `uv run --with pillow scripts/run_screens_local.py --name screens-local-N --models KEY [--gpu 1]`.
5. `uv run scripts/add_model.py preview KEY`, ask, then publish as above.

Rules we kept for local models: a model that fails most questions stays in
(failing is a result: RF-DETR, 2026-10-10); a model that cannot read a
request can be paired with one that can (the pair is named on the chart);
cost is the GPU's rental price ($0.22 per hour, an RTX 3090) under load, not
one picture at a time.

## Where each thing lives

| what | file |
|---|---|
| remote models, routes, thinking levels | `segbench/models.py` |
| local models | `segbench/local_models.py` |
| prices | `prices.json` |
| logos and colours | `segbench/brand.py`, `assets/logos/` |
| the standard runs, what is published | `segbench/protocol.py` |
| one call, measured | `segbench/call.py` |
| the steps above | `scripts/add_model.py` |
| registry tests (offline) | `tests/test_registry.py` |
