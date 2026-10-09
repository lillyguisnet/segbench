# GPU setup on lambda (checked 8 October 2026)

What the machine has today, what is already installed for segmentation, and
what is missing before the specialist models can run. A snapshot: GPU use
changes daily, so re-check with `nvidia-smi` before a run.

## Hardware

- 2 × NVIDIA RTX 3090, 24 GB each (compute capability 8.6), driver 610.57, CUDA 13.3.
- 48 CPU cores, 220 GB RAM, 206 GB free on disk (disk is 89 % full).
- No `nvcc` (CUDA compiler). PyTorch wheels bring their own CUDA, so that is
  fine unless a model needs custom GPU code compiled (none of our first picks do).

## Who uses the GPUs right now

| GPU | user | memory | notes |
|---|---|---|---|
| 0 | `model-vllm-smr-9b` (9B model for sciencemadereadable.com) | 21.7 GB | serves the public site; the GPU router never stops it. **Do not touch.** |
| 0 | desktop | ~0.4 GB | |
| 1 | `qwen3-embed-4b` (vLLM) feeding `openalex-embeddings` | 12.2 GB, **100 % busy** | user service, resumable job, 62.8 % done on 8 Oct at ~400 items/s → **about 2 more days** |
| 1 | parakeet speech-to-text, chattering semantic search, embeddings supervisor, kokoro TTS | ~9 GB | small always-on services |

Free: about **2 GB on GPU 0 and 3 GB on GPU 1**. That is not enough for SAM 3
or SAM 1 ViT-H, and any speed we measure on GPU 1 while the embedding job runs
would be wrong (the card is already at 100 %).

**A GPU router** (`model-router`, see `~/Projects/os/machines/hosts/lambda/NORTH-MINI-CODE.md`)
can load a big model on both cards on demand. When it does, it stops the small
services, SAM included, for at least 30 minutes. A benchmark run can be cut off
by this.

## The SAM service that already exists

- `sam31.service`, port 9300, on GPU 1. It is **SAM 3.1**: Meta's March 2026
  update of SAM 3 (`facebook/sam3.1`, file `sam3.1_multiplex.pt`, 3.3 GB, already
  downloaded).
- It is the **video** model. A photo must be sent as a one-frame "video" (a zip
  with one image). SAM 3.1's improvements are about tracking objects across
  video frames; for single photos the relevant model is SAM 3's image model
  (`facebook/sam3`, access granted, not downloaded).
- Code: Lilly's fork at `~/Projects/lilly/sam3` (uncommitted local changes),
  environment `~/family-ai/sam31/.venv` (Python 3.12, torch 2.13).
- The model loads on the first request (`/health` says `"loaded": false` now),
  capped at 99 % of GPU 1. With 3 GB free it would most likely fail to load today.
- Takes text, points and boxes (`POST /segment`), returns masks as PNGs.
- `sam-studio.service` (port 8501) is a web page on top of it for browsing
  images and clicking.

## Already on disk

From another project's model cache (`~/.cache/follicular-models`, read-only for us):

| model | what is there | usable for us? |
|---|---|---|
| SAM 1 | `sam_vit_b_01ec64.pth`, `sam_vit_h_4b8939.pth` (largest) | yes |
| SAM 2.1 | `sam2.1_hiera_small.pt` only | partly: we want the large one too |
| DINOv3 | source code and a version retrained for cell microscopy (Cellpose) | **no**: we need the general weights |
| RF-DETR | `~/.roboflow/models/rf-detr-medium.pth` | no: object boxes only, not masks |

Hugging Face login works (`maximerivest`) and has access to every gated model
we need: `facebook/sam3`, `facebook/sam3.1`, `facebook/sam2.1-hiera-large`,
`facebook/dinov3-*`, `reachomk/gen2seg-sd`, `reachomk/gen2seg-mae-h`.

## What is missing, model by model

| model | missing | rough download |
|---|---|---|
| SAM 1 | Python package (`segment-anything`) in our environment | 0 (weights on disk) |
| SAM 2.1 | package + large weights | ~0.9 GB |
| SAM 3 (photos) | image-model weights `facebook/sam3`; or use the service | ~3.5 GB |
| DINOv3 | general weights + a segmentation head (EoMT or PMT checkpoint) | ~1–2 GB |
| gen2seg | everything (code from GitHub, SD and MAE-H weights) | ~7 GB |
| YOLOE-26 | `ultralytics` ≥ 8.4.0 + weights (auto-download) | <0.5 GB |
| RF-DETR Seg | `rfdetr` package + segmentation weights (auto-download) | <0.5 GB |

About 15 GB in total; disk space is fine.

**Licences to note:** YOLOE (Ultralytics) is AGPL-3.0; RF-DETR's standard
models are Apache 2.0; SAM 1/2 Apache 2.0; SAM 3 and DINOv3 have Meta's own
licences. Fine for a benchmark, but worth knowing before publishing code
that bundles them.

## Plan agreed on 8 October: benchmark on GPU 0

**Done 8 Oct ~19:30:** `smr-worker-home` and `model-vllm-smr-9b` stopped; GPU 0
has ~22.7 GB free for the benchmark. They are still stopped; restart with the
commands below when the benchmark is done.

Maxime confirmed the sciencemadereadable 9B on GPU 0 has no users right now
(demo stage), so GPU 0 is ours for the benchmark. GPU 1 and its embedding
job stay as they are.

To take GPU 0, in this order:

```bash
systemctl --user stop smr-worker-home      # website job worker; otherwise it starts jobs that fail
sudo systemctl stop model-vllm-smr-9b      # frees ~22 GB on GPU 0
```

To give it back:

```bash
sudo systemctl start model-vllm-smr-9b
systemctl --user start smr-worker-home
```

Things that can take GPU 0 back from us during a run:

- **A system rebuild or a reboot** restarts the 9B (a manual stop does not
  persist). Avoid `nixos-rebuild` / `deploy` on lambda while a run is going.
- **The model router.** It refuses to load its big models while the 9B is
  running; once the 9B is stopped, a request for `qwen3.8-flash-next` or
  `north-mini-code` would load onto both GPUs. It has been idle for 5 days.
- **The desktop.** GPU 0 also drives the screen; a game would share it.
- The InkType retry timer is safe: it only restarts InkType after a skipped
  start, and InkType is paused in the config.

So the runner checks, before and after every timed call, that no other
program is computing on GPU 0, and marks the timing as unreliable if one is.

## Decisions needed (before the GPU 0 plan)

1. **GPU room for runs.** Options: pause `openalex-embeddings` and
   `qwen3-embed-4b` while we benchmark (the job resumes where it stopped), or
   wait ~2 days for it to finish. GPU 0 stays with the public site.
2. **Clean timing.** Speed is one axis of the chart, so specialist runs need a
   GPU nobody else is using at that moment, and the router should not load a
   big model during a run.
3. **SAM 3 through the service or our own runner.** The service is convenient
   but measures HTTP, video setup and Lilly's fork. Proposed: our own runner
   with the official SAM 3 image model, same weights cache, nothing of Lilly's
   changed.
4. **One environment or several.** These packages pin different versions
   (RF-DETR wants `transformers<5`, for example). Proposed: one small
   environment per model family under `specialists/`, so one model's
   requirements cannot break another's.
