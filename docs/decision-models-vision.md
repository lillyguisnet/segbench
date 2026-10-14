# Decision models (Jev-like) that accept images — survey

Searched 2026-10-12. This whole field is about four weeks old: Jev launched on
15 September 2026, and nearly every model below appeared after that. Expect this
page to go out of date within days.

**What counts here.** A *decision model* reads an input (the "state") plus questions
whose possible answers you define in the request, and returns a probability for
every allowed answer instead of writing text. The three question shapes are nearly
always Jev's: `choice` (pick one option), `noul` (probability of yes; OpenAI calls it
`predicate`) and `score` (ordered levels). Most services copy Jev's wire format,
`POST /v1/systemone`, so one client can talk to several of them.

**Jev itself does not take images.** Its model page says "Text only … No image,
audio, or video input" (jev-1.13.0, docs.typesafe.ai/models, read 2026-10-12).
TypeSafe has said publicly that it will build "new modalities" but has not shipped
any. Everything below comes from other companies or from the community.

## 1. Hosted APIs that take images

| Provider · model | Endpoint | Images | Price (input tokens; output is free everywhere) | Notes |
|---|---|---|---|---|
| **OpenAI** · `gpt-6-luna` (Decisions API) | `POST https://api.openai.com/v1/decisions`; official SDKs (`client.decisions.create`) | `input_image` parts, **inline base64 data URLs only** (no hosted URLs or file ids) | $0.10 / M | Public beta. Its own field names (`predicate`/`choice`/`score`, `input`, `answers`), not the Jev format. Supports ZDR/HIPAA and US/EU data residency. Also offered through OpenRouter and Vercel AI Gateway. |
| **Perplexity** · `pplx-decider-v1.1-27b` | `POST https://api.perplexity.ai/v1/decisions` | OpenAI-style image parts with base64 data URLs, placed inside a `state` array | $0.02 / M | Open weights too (Apache-2.0, Qwen3.8-27B fine-tune). Has a cookbook recipe, "Drive a browser with the Decisions API", that asks six questions about a numbered screenshot. |
| **Cloudflare Workers AI** · `clef` (27B), `clef-flash` (9B), `clef-omni` (30B MoE, 3B active; released 2026-10-09) | `POST …/accounts/{id}/ai/run/@cf/cloudflare/clef` (also clef-flash and clef-omni) | Images, plus video; clef-omni adds audio. Each image is shrunk to about 1 megapixel and capped at 1,024 tokens. | OpenRouter lists $0.042 / $0.021 / $0.15 per M | Uses the Jev format. All three are open weights (Apache-2.0) and serve on SGLang at `/v1/systemone`. |
| **Liquid AI** · `d1` (cloud) | `POST https://api.liquid.ai/decisions/v1/systemone` | Top-level `images` array. Their SDK can't send images yet, so use plain HTTP. | Not published on the pages I read | On OpenRouter, `liquid/d1` is **text only**. Open-weight versions are listed in §3. |
| **AlphaNimble** · `wity-1` ("System 1.5") | `POST https://api.wity.alphanimble.com/v1/systemone`; Python and TypeScript SDKs, a CLI | One top-level `image` field (photo, screenshot, scan, chart) | $0.042 / M flat; images and thinking are free; $5 of free credit for new accounts | Has an "auto" mode that thinks step by step on close calls, so it is slower: on ImageJevBench, about 2.7 s per decision in auto mode and 1.6 s with thinking off. Also has a `/v1/generate` endpoint for text, a 128K context, and can be installed on your own servers. Ranked #1 on ImageJevBench's official composite. |
| **decisionmodels.io** ("System1 Models") · `s1-vision` | `POST https://api.decisionmodels.io/v1/systemone`; hosted MCP server | **Exactly one image per request** (data URL or HTTPS link), at most 8,192 tokens in total, and only one question per request | $0.217 / M once an image is included (text-only requests $0.031 / M); an EU tier is available | Resells open models. s1-vision runs PixilabAI's Blink v0.3 26B-A4B in NVFP4 (4-bit). |
| **OpenRouter Decisions API** (one account, many providers) | `POST /api/alpha/decisions`; catalogue at `GET /api/v1/models?output_modalities=decisions` | Image-capable models today: `cloudflare/clef`, `clef-flash`, `clef-omni`, `openai/gpt-6-luna-decisions`, `perplexity/pplx-decider-v1.1-27b` | Provider price passed through | 21 decision models are listed; the other 16 are text only (Jev, Mercury Decide, Microsoft-Decision-1, Solar Decide, Drex, tev1, d1, …). |
| Autoloops · "Gemma 4 31B IT" | Not found | ImageJevBench lists it as a closed API that takes images | ~$0.083 per 1,000 decisions (from the board) | **Not verified.** autoloops.ai describes itself as serverless hosting for voice agents. I found no public docs for a decision endpoint. |

## 2. Device and browser SDKs

- **Google MediaPipe Decision Maker** (first commits 2026-10-02): runs on the device (Android, iOS, Web/WebGPU, desktop, Python). It takes text, RGB images and 16 kHz audio, both as input and as answer options. Its engines are EmbeddingGemma 2 `text_vision`, Gemma 4 E2B/E4B, or cross-encoders such as Laya and GLiNER2.5-Decide. It accepts Jev JSON requests.
- **Chrome `DecisionModel` web API** (landed 6–8 October 2026, behind `chrome://flags#decisions-api`, desktop only): **text only** for now.
- On-device builds of `decider-2b-vision`: CoreML (`coreai-community`, `mlboydaisuke`) and LiteRT (`litert-community`).

## 3. Open weights you run yourself

Scores come from two independent boards that rank differently:
**DI-V** = Jev Decision Index 0.3.2, Vision board (HF Space `multimodalart/jev-decision-index?mode=vision`, 2026-10-12; open weights only; tests include CV-Bench, BLINK, RealWorldQA, Mind2Web, CharXiv, Winoground, …; for reference, the stock 397B-parameter Qwen3.5 vision model scores 63.4).
**IJB** = Benchmark Heaven ImageJevBench v0.3.0 (2026-10-12; 57 systems including closed APIs; the "capability" score averages accuracy and calibration).

### Large (26–35B parameters): a 24 GB RTX 3090 needs a 4-bit version or both GPUs

| Model | Licence | Images / other | Board | Ways to get it |
|---|---|---|---|---|
| vLLM-SR **d3** 27B (vLLM Semantic Router) | Apache-2.0 | images, video | DI-V **#1, 71.3**, calibration error 0.016 | HF `vllm-sr/d3`; also the engine behind decisionmodels.io `s1-pro`, which serves it text only |
| Perplexity **pplx-decider-v1.1-27b** | Apache-2.0 | images | DI-V #2, 70.6 | HF, plus community GGUF / MLX-4bit / NF4 / NVFP4 / EXL3 versions; Perplexity API; OpenRouter |
| AutoTrust **JEV-27B-VL** (LoRA on Qwen3.8-27B) | Apache-2.0 | images; **never trained on images** (the decision head saw only text; vision comes unchanged from the base) | DI-V #3, 69.6 | HF (1.5M downloads); GGUF, AWQ, FP8, NVFP4, MLX, EXL3 versions; served with vLLM via the repo's `serve.sh` |
| Doccy Health **Solomon v1.1** 27B | Apache-2.0 (adapter) | images | DI-V #5, 66.8 | HF `DoccyHealth/Solomon`; bring your own Qwen3.8-27B base |
| Juspay **Xor 1.2** (35B, 3B active) | Apache-2.0 | images | DI-V #6, 64.2 | HF `juspay/xor`; patched SGLang |
| **RSI-Jev v6.1 VL** 27B | Apache-2.0 | images | DI-V #7, 64.0 | HF `shgao/rsi-jev-v6.1-vl-27b` |
| Cloudflare **clef** / **clef-omni** | Apache-2.0 | images, video (+ audio in omni) | clef-omni DI-V #11, 59.5 | HF `Cloudflare/*`; Workers AI; OpenRouter |
| **Surogate Rune** 26B-A4B v3 | see repo | images | IJB capability #2 | HF GGUF |
| **AutoJev-27B** | see repo | images | IJB capability #3 | HF `denis-pplx/autojev-27b` |
| **JEV-35B** (AutoTrust, 2026-10-11) | Apache-2.0 | card mentions images | not ranked yet | HF |
| Also: Matilda Jev v1.5 (its vision path is officially "not validated"), Jevify Gemma 4 26B-A4B, shisa-de-1/2, jev-vision-27b, PixilabAI Blink 26B-A4B (base of s1-vision), Bonsai-2-27B | | | | |

### Mid-size (8–12B): runs on one 3090 in 8-bit or bf16

- **vLLM-SR d3-flash** 9B: DI-V #4, 66.9; images and video; about 150 ms per image on an MI325X. Apache-2.0.
- **clef-flash** 9B: Apache-2.0; Workers AI and OpenRouter.
- **JPT-9B** (DI-V 61.7) and **imajev-9b** (IJB capability 67.1, but costs more than the board's "Jev-class" cap allows).
- **Winnow-12B** (Gemma 4 12B; DI-V 58.8); **Jev-Omni** 12B (text, image, audio, video; DI-V 56.1).
- **StandardOne-8B** (built on Ministral 3, keeps the Pixtral vision tower); Jev-Vision 8B (well-calibrated only in name: IJB calibration 36.5).

### Small (4B and under): fast on our hardware

- **imajev-4b**: IJB **#1 capability among Jev-class systems (68.0)**; Apache-2.0; demo on HF Spaces; local server at `/v1/systemone`.
- **vLLM-SR d3-mini** 4B: DI-V 63.2; also d3-nano 2B (54.7), d3-lite 0.8B, d3-edge 0.6B.
- **Intern-Decision** 4B / 2B / 0.8B (InternLM): DI-V 56.5 / 47.8 / 36.3; best calibration on the board (error 0.014); up to 8 images per request; Apache-2.0.
- **H2O-Lightning-4B**: images since v1.2; plain vLLM 0.30 plus a small shim; Hugging Face demo. Apache-2.0.
- **Liquid d1-3B** (SigLIP2 vision encoder; LFM 1.0 licence) and **d1-omni-600M** (images *or* up to 30 s of speech; 587M parameters).
- Others on the boards: JPT-4B, RSI-Jev 4B/3B, NeoHorse-Jev-4B, Reflex 4B, vjev-vision, OmniJev 0.8/2/4B, OneJev 0.8B, Mapika `decider-2b-vision`, Jevify Qwen3-VL-2B, Glance, Qevi-2B, PlayJev.
- **laya-vision** (201M, runs on a CPU, ~41 ms per answer on an L4 GPU): **non-commercial licence (CC BY-NC-SA)**.
- Diffusion-language-model variants (djev and others): poorly calibrated, near the bottom of both boards.

## 4. Closest to segbench: decision models that point or click

- **jev-spatial** (built on Molmo2-ER, Apache-2.0). It points by choosing one of 9 cells in a 3×3 grid, three rounds in a row, zooming into the chosen cell each time. The final grid is 27×27, so each axis is resolved to about 3.7% of the image. Its authors report +7 on Where2Place and +2 on RefSpatial-Bench compared with the base model writing coordinates as text, and up to 4× faster with 8 questions per image.
- **GUI-Decisions-31B / -2B** (infinitylogesh, Apache-2.0). Reads a click as a probability spread over 256 positions per axis, taken from vLLM's token probabilities, with no text generated. Authors report ScreenSpot clicks of 0.869 (31B) and 0.770 (2B, against 0.645 for its base model), about 190 ms per step. It can also output boxes and rough masks for ordinary objects.
- **CUA-S1 4B** (computer-use adapters on Qwen3.5-4B, one for text and one for images), and **Visual-Jev Browser 4B** (screenshot plus page-structure table).

## 5. Do-it-yourself and bridges

- Any vision-language model that exposes token probabilities can act as a decision model: list the options as letters and read the probability of each letter. Most community fine-tunes are built this way. Mapika's "decider-chat" models are stock checkpoints plus a temperature setting.
- **jev-media-bridge** (macOS only): runs local OCR and labelling on the image and sends only the extracted text to text-only Jev.

## Caveats (stated, not absorbed)

- **Search coverage.** I read both leaderboards in full, the OpenRouter catalogue, and Hugging Face name searches (`jev`, `decider`, `system-one`, `systemone`, `system1`, `typed-decision`, `decision-model`, `noul`, `laya`, filtered by image task type). Hugging Face holds several hundred decision-model repos, mostly tiny experiments. I listed only those with evidence of image input: a board entry, or a model card that says so. A model with an unusual name that is on neither board would be missed.
- **Scores were not checked by us.** The two boards disagree: Wity ranks #1 on IJB's composite, and the Decision Index has no closed APIs at all. Most model cards quote their own numbers. None of these were measured on segbench images.
- **Our hardware.** lambda's RTX 3090s cannot run NVFP4 (it needs newer Blackwell GPUs) and have no fast FP8, so 27B models need GGUF or AWQ 4-bit versions or both GPUs. Some compressed or alternative serving paths drop image support: the DI-V board excluded gevva and reflex-27b for that reason.
- **Licences vary.** laya-vision and StartLux are non-commercial; Liquid uses its own LFM licence; most others are Apache-2.0.
- **Not verified:** Autoloops' decision endpoint; Liquid's cloud price for images.

Sources: docs.typesafe.ai/models; developers.openai.com/api/docs/guides/decisions.md;
docs.perplexity.ai/docs/decisions/quickstart.md; developers.cloudflare.com/workers-ai/models/clef{,-omni}/;
docs.liquid.ai/lfm/models/d1; wity.alphanimble.com/llms.txt; system1models.ai/llms.txt;
openrouter.ai/api/v1/models?output_modalities=decisions; developers.google.com/edge/mediapipe/solutions/decision/decision_maker;
dejan.ai/blog/chrome-decisions-api-decisionmodel; benchmarkheaven.com/image-jev-bench;
huggingface.co/spaces/multimodalart/jev-decision-index (data/vision.json); the Hugging Face model cards named above.
