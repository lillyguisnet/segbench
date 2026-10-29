# TypeLLM on the photo benchmarks (2026-10-17): preview, not published

Asked by Maxime on 2026-10-17: TypeLLM (typellm.ai, "the workflow model", a
Jev-like typed-answer model; key `TYPELLM_API_KEY` in `.env`) on the eleven
photo tasks, points and boxes, with thinking off and on. Not on OpenRouter
under that name. Nothing published was changed.

## How it was asked (`segbench/typellm.py`, `scripts/run_typellm.py`)

- Direct HTTP, `POST https://api.typellm.ai/v1/generate`, model `typellm-latest`
  (early access: the model behind that name can change; scores hold for the run date).
- The benchmark's own words, as every language model got them, minus only the
  "Reply with JSON only, in this shape" paragraph: the typed schema is the answer
  format. Words in the field's `instructions`, `context` empty, the photo file sent
  whole (TypeLLM shrinks it to about 2,048 image tokens and 2,048 px itself).
- Four prompts, like the language models' four: points `typellm-v2` and
  `typellm-v1-hd` (items {x, y}, integers 0..1000), boxes `typellm-box-a`
  (items {x_min, y_min, x_max, y_max}, 0..1000) and `typellm-box-free`
  (TypeLLM's own `box` type). Each entrant keeps its better prompt per benchmark
  (ties to v2 / box-a), chosen on the test photos as for every model.
- **The `box` type is capped at 50 per answer**, with no way to continue
  (maxItems above 50 and `continue_from` both refused). The logs have 83; box-free
  returned 50 every time. Arrays of objects allow 200 (the other three prompts).
  A `box` cannot sit inside an object, so box-free dishes have no dirty/clean label
  (only location is scored).
- **No thinking on a list.** `thinking_effort` is refused on an array, on its
  items and on the items' properties. So:
  - `typellm@none`: the list alone. TypeLLM's own entry.
  - `typellm@count-medium`, "TypeLLM + count first": our pipeline. A first field
    thinks at medium (2,048 tokens): "Find {target} in this photo. How many are
    there? ..." (`count_words`), and the list is asked after it (`depends_on`,
    TypeLLM's documented chaining). It sees the count, not the reasoning.
- Default temperature (0). Tries still differ (the docs: batching); 3 tries each.
- Checked first on the synthetic ovals (`--check`, `results/check-typellm-*.jsonl`):
  all four prompts, both modes, four sizes: 7 of 7, no extra, every time.
- Price (typellm.ai/pricing, 2026-10-17): $0.05 per million input tokens (images
  included), $0.50 per million thinking tokens, answers free. Not in `prices.json` yet.

## Run `typellm-1` (2026-10-17)

264 calls (11 tasks x 4 prompts x 2 modes x 3 tries), no errors, $0.13 in all.
`results/run-typellm-1-<prompt>.jsonl`. Scored with `scripts/score_typellm.py`
(v3's and box v1's own scorers and settings) -> `results/preview/typellm/typellm-1/`.

| | points (v3) | rank | boxes (v1) | rank | cost / photo | seconds |
|---|---|---|---|---|---|---|
| TypeLLM | 58.2 (v2) | 12 of 29 | 45.1 (box-free) | 14 of 28 | $0.00011 | 3.5 |
| TypeLLM + count first | 67.7 (v2) | 8 of 29 | 51.1 (box-a) | 10 of 28 | $0.00102 | 20-23 |

Box levels: TypeLLM 68.2 / 54.8 / 35.5 / 21.8 at 50/75/90/95%; with count first
79.4 / 64.7 / 37.8 / 22.3. For comparison, Jev + halving scored 23.4 on points.

What the answers show:
- **The white cow is never found without the count**: 0 points or boxes in all
  tries of every prompt (score 0 on that task, about 9 points of the total). With
  the count first, 1 every time.
- **The list follows the count, even when the count is wrong**: logs try 0 of
  count-first v2 counted 7 and gave 7 points (83 are there); logs fell from 86
  to 61 with the count. It helps where TypeLLM alone gives up (white cow,
  dishes) and hurts where the count is off.
- Cost is in the local models' range ($0.0001 per photo): TypeLLM alone is
  13 to 460 times cheaper than the language models scoring 55-61 on points
  (Claude Haiku 5.5 to Mistral Large 4); with the count first, 5 to 30 times
  cheaper than its neighbours (Qwen 3.8 27B, Claude Opus 5.5).

## Preview charts and pictures (2026-10-17)

- `uv run scripts/plot_typellm_preview.py` -> `charts/preview-typellm/{points,boxes}/...`:
  the published v3 and box v1 charts, every published bubble from the published
  chart's own records, plus the two TypeLLM entries (checked against the preview
  scores). Label pins added only where TypeLLM's labels collided (in the script).
  TypeLLM's maker has initials "TL" and a colour in `segbench/brand.py`, and its two
  chart names in `segbench/chart_data.py` `EXTRA_ENTRANTS`; no logo.
- `uv run --with pillow scripts/view_typellm.py` -> `results/preview/typellm/typellm-1/view/`:
  the charts, then all 132 pictures (dots and boxes, both entries, 3 tries), each
  entry with its kept prompt. Right = matched at the medium distance (dots) or 50%
  overlap (boxes); the answer key's unmatched objects drawn as missed. Right counts
  checked against the scores (0 mismatches).
- Plain TypeLLM's kept box prompt is box-free, the one capped at 50 boxes: on the
  logs (83) it can never box more than 50.
