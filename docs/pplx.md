# pplx-decider-flash-v1.2 (Perplexity's local decision model) on the photos, 2026-10-17

Maxime: install Perplexity's newest decision model, run it locally on the
picture benchmark (GPU 0; the 9B rewriter paused for it). Exploratory, like
Jev + halving: not part of any published chart.

## The model

`perplexity-ai/pplx-decider-flash-v1.2` @ `6d771ad7b5` (released 2026-10-15),
4B, distilled from the 27B pplx-decider-v1.2, Apache 2.0 (base Qwen3.5-4B,
Apache 2.0). Installed in `~/Projects/deciders` (weights on `/mnt/fast/deciders`),
the model's own reference code, bf16, one RTX 3090. It answers choice / yes-no /
score questions with calibrated probabilities. **No boxes and no points** (Jev
1.14 has a box question; this model does not). The 27B (52 GB) does not fit our
two 24 GB GPUs without unsupported quantisation and was not run.

## The search (`segbench/pplx_halving.py`, `scripts/run_pplx_halving.py`)

Written and checked on the synthetic ovals only, committed (544343f) before any
photo was sent; nothing changed after.

1. **Count by halving**, as Jev + halving, but an object counts only in the piece
   holding its **middle** (Jev: every piece showing part of it, which made the
   splitting never stop on the log pile). So nothing has to be joined back.
2. **Locate by zooming**, because the model has no box: "in which ninth (3 x 3) of
   the image is its middle?", then a window half as wide and high around that
   ninth, again, until a ninth is under 1% of the photo's diagonal. The point is
   the middle of the last ninth.
3. Crops of the full-size photo, lossless PNG, one question per request (the
   model card's advice for this model), all pieces of a depth in one batch.

Synthetic check (`--check`): 7, 7, 6, 5 of 7 ovals got a point, none off an oval.

## Run `pplx-halving-1` (eleven v3 photo tasks, 3 tries, task's own words)

Scored as v3 and as Jev + halving (`scripts/score_points.py`, lenient, both
approved references, cost and seconds from the first four tasks):
`results/point-benchmark-v3-pplx-halving` (with its viewer, `index.html`).

**20.5%**, 28th of 30 (Jev + halving 23.4% 27th; YOLOE-26x 15.5%, RF-DETR 13.6%
below). Tight / medium / loose: 8.7 / 23.0 / 29.8.

| task | refs | points | pplx | Jev + halving | v3 median |
|---|---|---|---|---|---|
| logs | 83 | 64 | 29.5 | 22.7 | 58.1 |
| cows | 11 | 6 | 58.8 | 40.2 | 89.9 |
| fig | 33 | 16 | 24.5 | 33.6 | 32.8 |
| dishes | 26 | 14 | 13.3 | 12.6 | 26.7 |
| ants | 13 | 2 | 0.0 | 36.5 | 64.4 |
| beetles | 22 | 9 | 47.3 | 33.6 | 90.5 |
| birds | 16 | 14 | 22.2 | 16.7 | 47.7 |
| mushrooms | 14 | 9 | 8.7 | 23.7 | 67.5 |
| jars | 6 | 4 | 0.0 | 16.1 | 36.7 |
| pinkleaves | 37 | 13 | 21.3 | 21.5 | 29.0 |
| whitecow | 1 | 0 | 0.0 | 0.0 | 100.0 |

- **The three tries are identical** (same pictures in, same probabilities out: a
  local model with no sampling). Three tries add nothing here; Jev's do vary.
- **It undercounts.** It stops at "one" while a piece still holds several: of the
  pieces it answered "one" for, a real object's middle is in most (beetles 9 of 9,
  logs 52 of 64), but exactly one in few (beetles 2 of 9, logs 33 of 64). Every
  photo ends with fewer points than objects; no piece was ever forced.
- **Its zoom is rough.** Its pick of a ninth averages 0.23-0.47 probability; the
  points mostly land within 5% of the photo's diagonal of a real object in their
  piece (logs 51 of 52, beetles 9 of 9) but often outside v3's tight radius
  (8.7% tight against 29.8% loose).
- **White cow: "none" for the whole photo** (0.76), so nothing was searched.
- Not done: a check of "one" pieces (ask again on halves), or the 27B model.
  Changing the search now and rerunning on these photos would be tuning on them.

## Speed and cost (provisional)

Cost 0 (own GPU; electricity not priced; segbench prices local models at
rental-equivalent GPU rates, not applied here). Median 36.8 s per photo on the
first four tasks, but **not a clean timing**: the reference path (not the fast
path), PNG encoding of large crops on the CPU, the first logs try includes a
one-time warm-up (98 s against 42-45 s), and a scholarsreadinglist job
(`fieldpick`) shared GPU 0 from 13:15 to 13:22 UTC (cows, fig try 0, dishes to
birds; `gpu_others` in each record). Peak GPU memory held: 13.7 GB.
