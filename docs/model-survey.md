# Segmentation candidate survey

Research checked: **8 October 2026**. This is a shortlist, not an exhaustive catalogue or a performance ranking. No candidate was installed or benchmarked in this survey. A release listing weights is not the same as a successful local inference test.

## Scope

Native text prompting is **not required**. Accept automatic segmentation, points, boxes, scribbles, example images/masks, fixed-category models, and backbones with segmentation components. We can build pipelines around them. SAM/DINO-based combinations remain eligible; identify their ingredients rather than presenting them as independent foundation models.

Prioritize broad use across ordinary photographs and screenshots. Do not mistake a new publication date or a recently updated repository for a new model release. Dates below explicitly distinguish papers, conferences, and software releases where verified.

## Candidates with published implementation and weight instructions

Pipeline ideas below are our proposals, not measured capabilities on our images.

| Candidate | Date evidence | What it supplies | Proposed pipeline / caveat | Availability evidence |
|---|---|---|---|---|
| **gen2seg (SD and MAE-H)** | First paper 21 May 2025; revised 2 April 2026; ICLR 2026 | Generalizable instance segmentation using generative representations | Generate segmentation features, extract object masks, then select relevant objects. Also offers point-based mask extraction. Try both released variants; don't assume generated colours are already reliable instance IDs. | Official inference code and two Hugging Face checkpoints [S1] |
| **YOLOE-26** | Family announced 14 January 2026 | Instance masks with text, example boxes, or a separate prompt-free checkpoint | Try example-box prompting for repeated objects as well as text. Prompt-free mode has a vocabulary, not unrestricted discovery. | Official documented segmentation checkpoints at multiple sizes [S2] |
| **RF-DETR Segmentation** | Preview 2 October 2025; full family 22 January 2026 (v1.4.0) | Fast automatic instance masks | Keep in the benchmark despite fixed pretrained categories. Try mask proposals plus selection; if it misses an unfamiliar object, relabelling cannot recover the missing mask. Adaptation may need separate training examples. | Official model family and package; uses a DINOv2 backbone for the usual segmentation models [S3] |
| **DAPE / Mask-DAPE** | Instance code released 7 January 2026; semantic/panoptic code 13 January; AAAI 2026 | Instance masks, semantic regions, and panoptic outputs through task-specific configurations | A recent conventional segmentation comparison. Fixed training categories require selection or adaptation; choose the right task checkpoint. | Official configs and checkpoints [S4] |
| **SegMAN** | CVPR 2025 | Semantic segmentation: labels regions across the image | An architecture distinct from the SAM approach. Good candidate to investigate for region/area tasks; splitting touching objects for counts is extra pipeline work. | Official small-to-large segmentation checkpoints and image demo [S5] |
| **COSINE** | ICCV 2025 | Text- and example-guided segmentation, including instance and semantic tasks | Try a reference example for unusual targets. Includes DINOv2 and SAM visual-prompt paths: name the complete configuration. | Official inference paths and ModelScope checkpoint instructions; download access still needs testing [S6] |
| **SegNext** | CVPR 2024, deliberately an older baseline | Interactive segmentation with spatial prompts | Reuse the automatic prompt generator proposed for SAM 1/2. Record corrections and all extra calls. This is UNC's SegNext, not the similarly named semantic model SegNeXt. | Official code, demo, and three weight downloads [S7] |
| **SegGPT** | Paper April 2023, deliberately an older baseline | Segmentation from example image + mask pairs | Test an example-driven route for leaves, parts, or screenshot regions. Reference examples must not disclose the answer for the test picture. | BAAI inference code and released checkpoint [S8] |

### Concrete ways to turn our existing DINO entry into a segmenter

- **EoMT:** CVPR 2025, with released DINOv2/DINOv3-based models. An encoder-only architecture for semantic, instance, and panoptic segmentation. It offers a concrete alternative to inventing an unspecified head. [S9]
- **PMT:** 2026 paper / CVPR 2026 Workshop. Adds a segmentation decoder to a frozen vision encoder; the official repository lists image checkpoints with DINOv3. Treat this as **DINOv3 + PMT**, not as an unrelated foundation model. [S10]

Both still require a suitable checkpoint and a way to select/adapt the target categories. A segmentation architecture is not automatically a model that knows every possible target.

## Recent research to keep on the watchlist

| Candidate | Date / status | Why retain it | What is not yet verified |
|---|---|---|---|
| **DiGSeg** | Paper 27 April 2026; project identifies ECCV 2026 | Diffusion-based semantic/open-vocabulary segmentation, evaluated across several domains | The project's Code link returned to its own page in this check. A usable official code/weight release was not confirmed. Cross-domain experiments do not prove a single checkpoint works without adaptation. [S11] |
| **Qwen3-VL-Seg** | Paper 8 May 2026 | Adds dense mask prediction to a vision-language model rather than calling an external SAM segmenter | Official released weights and an author-linked implementation were not confirmed from the paper page. [S12] |
| **EFPNet** | ICML, 6–11 July 2026 | Click-based segmentation with focused correction around the latest click | Paper verified; runnable code and pretrained weights not confirmed. [S13] |
| **OIS** | ICLR 2025 | Order-aware interactive segmentation for separating overlapping objects | Authors' project page has examples and paper links, but a public implementation/weight release was not confirmed. [S14] |

**An existing pipeline worth noting:** WOW-Seg (ICLR 2026) recognizes regions; its released demo combines its own checkpoint with SAM. It is eligible as a pipeline experiment, not a new independent mask generator. [S15]

## Proposed first integration pass

Start with **YOLOE-26, gen2seg, SegNext, SegGPT, and RF-DETR Segmentation**, plus **PMT or EoMT** to make the DINO entry concrete. Add **SegMAN or Mask-DAPE** for a conventional region-segmentation comparison, and investigate COSINE's download/setup next.

Trade-off: this order favors available implementations and different approaches over adding every newer paper. We keep unavailable recent models on the watchlist rather than quietly dropping them. The older SegNext and SegGPT are included to test spatial and example prompting, not presented as recent releases.

Before implementing each candidate, verify checkpoint provenance, code/weight licences, exact revision, inference memory, and required preprocessing on our hardware. Published training memory or vendor latency is not our measured inference memory/time.

## Proposed benchmark safeguards

- Each chart point names a **complete pipeline and configuration**. Log prompt generation, selection, refinement, retries, model calls, and local computation separately; also record total cost and end-to-end time.
- Separate automatic and human-assisted runs. Do not compare free human clicks with fully automatic systems without identifying that assistance.
- Never generate test-time clicks or boxes from ground truth in the automatic track. Such a run may be useful, but label it as an oracle-assisted diagnostic.
- Fine-tuning and reference-mask creation need an explicit separate data budget. Never train or choose thresholds on the held-out test answers.
- Preserve native masks and instance IDs. Do not force specialist masks through lossy polygon conversion merely because language models produce polygons.
- Do not infer object counts from connected components without checking touching/fragmented objects; count extraction is a named pipeline stage.

## Primary sources

URLs are recorded here for reproducibility. Primary repositories, papers, and release notes were fetched during this survey; download instructions were inspected, not every weight file downloaded. These are verified leads, not a claim that no newer model exists.

- **S1 — gen2seg:** `https://github.com/UCDvision/gen2seg`; `https://arxiv.org/abs/2505.15263`
- **S2 — YOLOE:** `https://docs.ultralytics.com/models/yoloe/`; release announcement `https://community.ultralytics.com/t/yolo26-available-now/1746`
- **S3 — RF-DETR:** `https://github.com/roboflow/rf-detr`; release chronology `https://github.com/roboflow/rf-detr/releases/tag/1.4.0` and `https://github.com/roboflow/rf-detr/releases/tag/1.3.0`
- **S4 — DAPE:** `https://github.com/xiuqhou/DAPE`
- **S5 — SegMAN:** `https://github.com/yunxiangfu2001/SegMAN`
- **S6 — COSINE:** `https://github.com/aim-uofa/COSINE`; see `SOURCE_LAYOUT.md` in that repository. Paper: `https://openaccess.thecvf.com/content/ICCV2025/html/Liu_Unified_Open-World_Segmentation_with_Multi-Modal_Prompts_ICCV_2025_paper.html`
- **S7 — SegNext:** `https://github.com/uncbiag/SegNext`
- **S8 — SegGPT:** `https://github.com/baaivision/Painter/blob/main/SegGPT/SegGPT_inference/README.md`; `https://arxiv.org/abs/2304.03284`
- **S9 — EoMT:** `https://github.com/tue-mps/eomt`
- **S10 — PMT:** `https://github.com/tue-mps/pmt`; `https://arxiv.org/abs/2603.25398`
- **S11 — DiGSeg:** `https://wang-haoxiao.github.io/DiGSeg/`; `https://arxiv.org/abs/2604.24575`
- **S12 — Qwen3-VL-Seg:** `https://arxiv.org/abs/2605.07141`
- **S13 — EFPNet:** `https://proceedings.mlr.press/v306/hu26au.html`
- **S14 — OIS:** `https://ukaukaaaa.github.io/projects/OIS/index.html`; `https://proceedings.iclr.cc/paper_files/paper/2025/hash/c9d7ec703f135a77b3be95e0f2548afb-Abstract-Conference.html`
- **S15 — WOW-Seg:** `https://github.com/AAwcAA/WOW-Seg-Meta`; particularly `demo/README.md`.
