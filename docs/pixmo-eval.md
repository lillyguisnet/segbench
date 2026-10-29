# PixMo-Points-Eval: preparation and local tests

Separate from the published eleven-photo benchmark. No existing scores,
questions, prompts or annotation keys are changed.

## Dataset and source versions

- Dataset: https://huggingface.co/datasets/allenai/pixmo-points-eval
- Pinned revision: `e237f0491ca49bf8dec04e06fc0c3807a37ef157`, **test split only**.
- 436 questions, 431 distinct image URLs, 600 annotated target points. These
  are questions, not 436 distinct photographs. Some targets are absent.
- Images remain on third-party websites. We download original bytes and
  require the dataset's SHA256. Dead links, changed images and undecodable
  images remain explicit coverage failures, never silently accepted.
- Licence: ODC-BY-1.0 for annotations; this does not grant ownership or
  blanket redistribution permission for the images. Downloaded data are
  ignored by git under `cache/pixmo-points-eval/`.
- This is **evaluation data**, not the 2.4-million-example training corpus.
  Do not use this test split to optimize prompts or programs. Any future
  training sample must exclude evaluation image URLs AND hashes, not merely
  question rows (Ai2's loader explicitly filters evaluation URLs from training).

`prepare` also caches the authors' dataset loader and evaluator at molmo2
revision `f3cb1085fbb97c4a4d7fdcadd77cf871bf37a88a`. All three source
files are checksum-verified. They are evidence, not imported model code.

## Prepared snapshot (2026-10-22)

- 209/436 questions verified, on 206 images: 278 target points and 24
  absent-target questions.
- 227 questions unavailable: 143 original-byte hash mismatches; the rest
  mostly blocked/dead links, plus DNS/TLS/network failures.
- Original images occupy about 36.5 MB. All 436 reference annotations are
  cached, including those whose images are unavailable.
- The seven-model plan for these 209 questions is ~1.62 GPU-hours plus
  startup/scoring. No GPU evaluations have been started.
- This available subset is useful for comparing our own runs, **not** for
  claiming full-test scores comparable to the published 436-question results.

## Prepare (CPU/network only)

```sh
uv run scripts/pixmo_eval.py prepare
uv run scripts/pixmo_eval.py plan
```

Dependencies are pinned in the script's PEP 723 header, independent of both
segbench's environment and each finder's environment. No model weights are
loaded by `prepare`, `plan` or `score`.

The downloader uses eight workers and one retry by default; tune with
`--workers` and `--retries`. Re-running verifies cached images and retries
unavailable URLs. It does not replace changed images with unverified copies.
EXIF-rotated images are held for a manual mask-orientation check, rather
than silently rotating predictions relative to masks. The manifest includes
mask hashes; scoring rejects modified masks or manifests.

Coverage and individual download failures: `cache/pixmo-points-eval/provenance.json`.
No preparation success implies 436 usable images: inspect coverage explicitly.

## Run a pilot, then a full pass

```sh
# 20 randomly selected verified questions; fixed order, seed 0.
uv run scripts/pixmo_eval.py run --model molmopoint-8b --gpu 0 \
  --limit 20 --out results/pixmo/molmo-pilot

# All verified questions. Explicitly accept the incomplete dataset.
uv run scripts/pixmo_eval.py run --model molmopoint-8b --gpu 0 \
  --allow-partial --out results/pixmo/molmo-available

# An interrupted run keeps each completed prediction. Resume with identical options.
uv run scripts/pixmo_eval.py run --model molmopoint-8b --gpu 0 \
  --allow-partial --out results/pixmo/molmo-available --resume

# Re-score saved predictions without running a model.
uv run scripts/pixmo_eval.py score --run results/pixmo/molmo-available
```

Without `--limit`, incomplete datasets require `--allow-partial` so a full
benchmark cannot silently turn into an available-only run.

GPU selection is required. Runs refuse busy GPUs unless the user explicitly
passes `--allow-shared-gpu`; this records contention but does **not** make
shared-GPU timings trustworthy. No other jobs or services are stopped.
Use the same selected IDs for cross-model comparisons. If a model crashes,
resume explicitly; an incomplete prediction file is not a completed result.

Model runs use the ready `~/Projects/finders` wrappers in their pinned
separate environments, with one load for all selected questions. Labels
are passed **verbatim** to each model's native template, with no
reference counts, reference dots or masks sent to it. Empty predictions
are retained and scored, never retried for a better answer. Sampling seed
is fixed at 0 unless specified; this is one pass, not best-of-many.

Default seven models in the runtime plan:

| Model key | Output used for pointing | Prior 436-query estimate |
|---|---|---:|
| `sam3` | deepest interior point of each predicted mask | 2 min |
| `moondream-3.1` | native points | 4 min |
| `florence2-large-grounding` | predicted box centres | 5 min |
| `groundingpi-4b` | native points | 16 min |
| `rexomni-3b` | native points | 24 min |
| `locateanything-3b` | native points | 53 min |
| `molmopoint-8b` | native points | 100 min |

Florence's `florence2-large-ovd` mode is also available as an explicitly
separate entrant. RF-DETR's fixed COCO vocabulary is not wired into this
arbitrary-expression evaluation. SAM 1/2 require location hints and are
not standalone finders for these labels. Recursive decision-model pipelines
are separate experiments and are not included in these seven timings.

**Estimates, not measurements on PixMo:** multiply medians from
`results/point-benchmark-v3/summary.csv` by question count. Total ~3.4
GPU-hours sequentially on one free RTX 3090, before startup/scoring. The
photo benchmark is much denser: PixMo could finish substantially faster.
Some historical timings were provisional/shared-GPU. A short pilot should
replace these estimates with actual throughput. Both cards were busy when
this setup was prepared; no GPU evaluations were started.

## Scoring: reproduce the authors' rule, not our dot rule

The dataset page says pixels, but the actual loader multiplies reference
coordinates by `[mask_width, mask_height] / 100`: **references are 0..100
percentages**. Never infer this from test scores.

1. Convert finder predictions from their upright image pixel grid to the
   mask pixel grid. Some masks have a different resolution than the image.
2. Use Hungarian assignment minimizing Euclidean distance to reference dots.
3. An assigned prediction is correct only if its rounded pixel is inside
   that reference object's mask. Python's rounding matches the authors'.
4. Compute per-question precision, recall and F1; report their arithmetic
   means. Mean F1 is not F1 of mean precision/recall, nor pooled F1.
5. For an absent target, no predictions scores 1; any predictions score 0.

Importantly, the authors do **not** use maximum mask-hit matching. Preserve
that difference for comparability. Duplicates and out-of-image predictions
are not clipped, deleted or moved. Additional objects penalize precision.

`summary.json` reports verified/total/selected/scored questions separately,
with explicit `full_test_set` and `complete_selected_run` flags. Means on an
available subset are labelled partial, not a reproduction of published
whole-test results. Missing model records are not silently called empty
predictions. Raw answers, weights/settings, prompts, warm timing, startup
time and point provenance remain in `predictions.jsonl`; logs and GPU
snapshots accompany each attempt.

This reproduces the **metric**, not necessarily a paper's exact inference
setup. Our native prompts, numerical precision, generation settings and
available-image subset can differ. Molmo models were trained on related
PixMo data; this evaluation is useful but not wholly new-domain evidence.
No cost-under-load measurement is claimed by this sequential runner.

## Tests

```sh
uv run --frozen --with numpy==2.3.3 --with scipy==1.16.2 \
  --with pyarrow==21.0.0 --with pillow==11.3.0 \
  python -m unittest discover -s tests
```

38 tests pass, including 150 randomized comparisons against the pinned
official evaluator (if `prepare` cached its source), mask membership,
duplicates, negative examples, rounding, off-image predictions, the
difference between minimum-distance and maximum-hit matching, image/mask
resolution scaling, missing predictions, modified-file rejection and
busy-GPU refusal. PixMo tests skip in the minimal environment rather than
adding scientific dependencies to the existing baseline test suite.
