# Falcon Perception-HD — local setup (2026-10-23)

Requested: investigate and install Falcon, **do not run the benchmark**.

## Which Falcon

For this project's image localization tasks, the relevant model is
[TII Falcon Perception](https://huggingface.co/tiiuae/Falcon-Perception), not
the Falcon text-only language models or Falcon OCR. It accepts an image and
a natural-language object description, and returns multiple boxes and masks.
It is not a general chat or question-answering model.

Use the latest published RL post-trained **HD** release, revision
`19-08-2026`, pinned to `54916b3dec58565fafc6d82eb3051fe7246ab666`.
The smaller 300M detection-only sibling is unnecessary to fit this machine;
the full model has about 0.6B parameters and 2.53 GB of weights. Its sibling's
comparison table incorrectly says the full model has 7B; the full model's
own card, configuration and weight size are consistent with 0.6B instead.

Sources checked: the maker's model card, release refs and
[official inference repository](https://github.com/tiiuae/Falcon-Perception).
The card links reports arXiv:2603.27365 and arXiv:2608.18881. The local
scholarly index did not return the model's reports, so no performance claims
are inferred from that search. The maker's published benchmark results are
not measurements on our pictures.

## Reusable installation

Per `docs/adding-a-model.md`, installed separately in:

- `~/Projects/finders/models/falcon/find.py` — adapter and exact weight pin.
- `~/Projects/finders/models/falcon/pyproject.toml` and `uv.lock` — isolated,
  locked Python 3.12 environment, CUDA 12.8, PyTorch 2.11.0.
- TII native inference engine at commit
  `c457916c9974efbacfa91f0f6ecc2c49c6543e56`, installed as a pinned dependency.
- Weights in the existing shared Hugging Face cache, not duplicated here.

No system packages or existing model environments are changed. No permanent
server is needed; a command loads the model, answers its requests, and exits.

```sh
# Idempotent installation, downloads only:
~/Projects/finders/finders setup falcon

# For later use, on a picture of your choice (NOT needed to finish setup):
~/Projects/finders/finders run falcon photo.jpg "cow" --task segment
~/Projects/finders/finders run falcon photo.jpg "cow" --task box
```

From any project's Python environment (standard library only):

```python
import sys
sys.path.insert(0, "/home/maxime/Projects/finders")
import finders
records = finders.find_many("falcon", [
    {"image": "photo.jpg", "query": "cow", "task": "segment"},
])
```

## Choices and limitations

- **Native paged engine**, rather than the Hugging Face remote-code wrapper.
  The wrapper still forces coordinate deduplication and mask NMS; the HD
  model explicitly no longer needs them. Both are off here.
- **Original precision, no quantization:** float32, as the official example.
  This uses more memory than reduced precision but fits a 24 GB card.
- **3090-compatible attention:** the documented `--flex-attn-safe` settings
  (`BLOCK_M=64`, `BLOCK_N=64`, `num_stages=1`). Smaller kernel tiles avoid
  exhausting per-SM shared memory; this is not lower model precision.
- The authors' compiled execution and CUDA graphs are enabled. First-time
  compilation is setup overhead, not a meaningful steady-state speed result.
- Native image preprocessing: shortest-side floor 256, longest-side cap 1024.
  This sacrifices small details in very large pictures, matching the default
  official setup rather than silently adding crops or higher resolutions.
- Context 8192, output cap 2048, greedy decoding, mask threshold 0.3, official
  high-resolution upsampling settings. Stops and truncation are recorded.
- High-resolution feature reuse is off: each question starts from the image
  again, so repeated images cannot make later measurements look cheaper.
- `segment` returns real masks and native boxes. Its point is derived inside
  the mask, explicitly marked `mask_deepest_point`. `box` and `point` use
  detection-only inference; their points are **box centres**, not native
  independent pointing. Empty masks are retained, with explicit box fallback.
- No fabricated confidence scores or answer-key-based repairs. Raw geometry,
  generated tokens and exact prompt are retained in each record.

## Verified on this machine

On lambda, GPU 1 (RTX 3090, 24 GB), without stopping any existing services:

- Downloaded weights verified against the published SHA-256:
  `8c18e2ffbfedfedf22ca1d231a754459f945cca5e9ea72eb5f2e0b0f4748edfa`.
  Safetensors contains 632,372,288 parameters (2,529,514,176 bytes on disk).
- Standard finders smoke test, using **only** its separate dog/groceries/truck
  pictures: four grocery bags, one dog, four wheel predictions (count is
  ambiguous), no absent cat. Repeating the bag request returns identical points.
- Non-empty native segmentation masks, visually inspected on the dog.
  Peak PyTorch GPU allocation during smoke: **10.74 GiB**, plus driver overhead.
  This is a small-picture compatibility check, not a benchmark measurement or
  a guarantee for arbitrary large inputs.
- First load/graph compilation took about 43 seconds and the first image
  triggered about 55 seconds of further compilation. These are **not** normal
  response times and must not be used as benchmark latency.
- Public Python API check passes for `segment`, `box` and `point`. Masks have
  the original image dimensions and the derived point is inside the mask;
  switching to detection and back leaves repeat segmentation results identical.
- All 8 offline adapter tests pass. Segbench's 27 unit tests pass (2 existing
  optional tests skipped). Deprecation/compiler warnings in logs are non-fatal;
  no fallback model or reduced-precision weights were used.

Evidence: `~/Projects/finders/runs/falcon-setup-20261023/` (setup and smoke
logs, additional adapter checks), and `~/Projects/finders/runs/smoke/falcon*`.

## Photo benchmarks (2026-10-24, GPU 1)

Run with the user's go-ahead on the whole of GPU 1: its six services
(LightOnOCR, Parakeet, Chattering semantic search, Kokoro, MyVoice, Chattering
speaker) were stopped for the ~7-minute run and restarted afterwards. Every
timed record has `gpu_others: {}`. Script and log:
`results/falcon-gpu1-run-20261023.sh`, `results/falcon-gpu1-run-20261023.log`.

| run | file | how |
|---|---|---|
| points, first four photos | `results/run-finders-6.jsonl` (+ `.timing.json`) | `scripts/time_finders.py --gpu 1`: 3 rounds, trick questions, load test |
| points, second batch | `results/run-finders-b2-falcon.jsonl` | `scripts/run_finders_batch2.py --gpu 1`: answers only, as every local model |
| boxes, eleven photos | `results/run-finders-boxes-4-falcon.jsonl` (+ `.timing.json`) | `scripts/time_finders_boxes.py --gpu 1`, its detection mode |

**Box run redone.** The first box run (`results/superseded/run-finders-boxes-3-falcon.*`)
overlapped `lighton-ocr-retry.timer`, which restarts LightOnOCR every 10
minutes when it is down: it started on GPU 1 at 08:45:11 and failed a minute
later. The nvidia-smi checks before and after missed it (`gpu_others` was
empty), so its timing was not trustworthy. Rerun at 08:51 with the timer
paused too and GPU 1 watched every second
(`results/falcon-gpu1-boxes-rerun-20261024.*`): only Falcon on it. Same 33
answers (greedy); single-copy times within 1%; the two-copy load factor
changed from 0.951 (contaminated) to 1.004. **Pause that timer for any future
whole-GPU-1 timing.** The point runs (08:40-08:43) ended before it fired.

Choices: points use its `segment` task (point = deepest point inside each
mask); boxes use its own detection mode. Singular phrases, like its examples.
Greedy decoding, so the three rounds are identical.

**Warm-up (Falcon only):** it compiles on its first question in every process
(20-50 s), which is not answering time. `LocalModel.warmup_photos` gives it one
untimed question per photo first, in the single-copy pass and in each copy of
the load test (whose window starts after the warm-ups). Every other model keeps
exactly the single warm-up it was timed with (tests/test_falcon_setup.py).

**Load test:** points: two copies ran out of GPU memory (Falcon builds its 90
log-end masks at the photo's full 4000x2252 size), so cost uses one copy,
factor 1.0, the rule for a model that does not fit twice (MolmoPoint). Boxes
(no masks): two copies fit but gain nothing (1.004), so factor 1.0 too.

### Scores (preview, not published)

Scored exactly as the published versions, into
`results/preview/falcon-perception-hd/` (points-v3, boxes-v1).

| benchmark | score | rank among published | seconds / question | cost / question |
|---|---|---|---|---|
| points v3 (11 photos) | **38.0** | 22 of 29 (between SAM 3, 39.6, and Luna, 36.3) | 1.60 | $0.00018 |
| boxes v1 (11 photos) | **39.3** | 19 of 28 (between Mk1.5, 41.2, and Kimi K3, 37.5) | 1.05 | $0.00010 |

Per photo (points): cows 91, logs 74, ants 64, pink leaves 62, birds 51,
mushrooms 33, dishes 32, jars 10, and **0 on fig, beetles and white cow**: on
those three it answered "absent" (`<|absence|>`, a real answer, not a reading
failure). It answered every trick question correctly with nothing.
Local models for comparison (points v3): MolmoPoint 63.7, Moondream 62.4,
LocateAnything 55.7, Rex-Omni 52.1, GroundingPI 48.7, SAM 3 39.6.

Not done: screenshots (not asked; `screen_seeds` unset), publication (not in
`protocol.LOCAL_PHOTO_RUNS`, `build_v3.py` or `build_boxes.py` inputs), and a
TII logo/colour for the charts.

## Benchmark boundary

No published protocols, predictions, scores or charts are changed.

## Licence

The model card and inference package specify **Apache 2.0**, permitting
commercial use and modification subject to the licence. Retain TII's licence
and NOTICE on redistribution and acknowledge/cite the project in public work,
as requested in its NOTICE. Package dependencies keep their own licences.
