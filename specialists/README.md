# Specialist models on our GPU

Each model family has its own small Python environment (their package
requirements conflict), all on GPU 0 of lambda, CUDA 12.8 PyTorch wheels.

| folder | models | run |
|---|---|---|
| `sam/` | SAM 1 ViT-H, SAM 2.1 Large, SAM 3 (image model) | `cd sam && uv run run.py all` (+ `auto-concurrent`) |
| `yoloe/` | YOLOE-26x, text-prompted and prompt-free | `cd yoloe && uv run run.py all` |
| `rfdetr/` | RF-DETR Seg 2XL (COCO categories) | `cd rfdetr && uv run run.py` |
| `gen2seg/` | gen2seg SD (fp16 and fp32) and MAE-H | `cd gen2seg && uv run run.py all` |

First run of `gen2seg/` needs the authors' code:
`git clone https://github.com/UCDvision/gen2seg.git gen2seg/vendor && git -C gen2seg/vendor checkout a8d3ffa6`.

Test pictures in `smoke_images/` (dog, groceries, truck) come from the
[segment-anything](https://github.com/facebookresearch/segment-anything)
repository's `notebooks/images/` (Apache 2.0); they only check that a model
runs, they are not benchmark photos.

Weights: SAM 1 files in `../weights/sam1/` (copied from the follicular cache,
md5 checked against the official names); YOLOE and RF-DETR download into
`../weights/yoloe|rfdetr/`; everything else is in the Hugging Face cache
(`~/.cache/huggingface`). `weights/` is not committed.

## How speed and cost are measured (`common/harness.py`)

- **Latency** (bubble size): one image at a time, warm model, median of 10,
  results copied back to the CPU inside the timing.
- **Throughput under load** (cost): batch sizes 1, 2, 4, … until memory runs
  out, 20 s each, rotating through all test images; for the "segment
  everything" modes, whose CPU clean-up leaves the GPU idle, also 2 processes
  sharing the GPU. The best one sets the cost.
- **Optimized like a real deployment**: TF32 on, each model's official
  half-precision setting where it has one (bf16 for SAM 2/3, fp16 for YOLOE,
  RF-DETR traced in fp16 with the authors' `inference()`), gen2seg fp16
  checked against fp32 (same result, 1.8× faster).
- **Cost per image** = $0.22 per GPU-hour (RunPod Community Cloud RTX 3090,
  8 Oct 2026; evidence in `docs/evidence/`) ÷ images per hour.
- **Guard**: the GPU is pinned by UUID; each record lists other programs on
  the GPU and is marked untrustworthy if one appeared during the run.

Not done yet (headroom, would make our GPU look even cheaper): TensorRT
export for YOLOE and RF-DETR, `torch.compile` for SAM 2/3, batched grounding
for SAM 3. These models are already 100–10,000× cheaper per image than an
LLM API call, so this does not change the chart's story.

## Still to build before the real benchmark

- Prompts from the task: SAM 1/2 need boxes or points (from another model or
  the "everything" mode + selection); gen2seg needs its colour image turned
  into separate masks (clustering or one click per object).
- DINOv3 + a segmentation head (EoMT or PMT).
