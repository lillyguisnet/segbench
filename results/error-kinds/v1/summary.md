# Kinds of mistakes on the eleven photos

Made by scripts/classify_errors.py (rules in its header, reasons in docs/error-kinds.md). Scores unchanged. Outlines of the key objects (SAM 2.1, from the reviewed boxes) and of look-alikes (SAM 3) are measuring aids, not reviewed by a person.

## Points (point benchmark v3, 770 replies)

| kind | wrong points | share | objects left unfound | share |
|---|---:|---:|---:|---:|
| no answer: stuck |  |  | 355 | 5.5% |
| no answer: unreadable |  |  | 158 | 2.4% |
| no answer: empty |  |  | 497 | 7.6% |
| coordinate mix-up | 245 | 5.2% | 219 | 3.4% |
| shifted or stretched | 90 | 1.9% | 101 | 1.6% |
| doom loop | 277 | 5.9% |  |  |
| off image | 27 | 0.6% |  |  |
| on the object, off-centre | 1207 | 25.6% |  |  |
| duplicate | 806 | 17.1% |  |  |
| wrong object | 440 | 9.3% |  |  |
| near miss | 607 | 12.8% |  |  |
| in a gap | 582 | 12.3% |  |  |
| maybe missing from key | 2 | 0.0% |  |  |
| background | 441 | 9.3% |  |  |
| missed |  |  | 5175 | 79.6% |
| total | 4724 | | 6505 | |

Correct points: 10656 (full credit at the strictest level: 6202). Objects left unfound: the reply-level kind when there is one, the group box that swallowed them, or missed.

### Score each kind costs, per model (points out of 100, as charted)

| model | score | no answer: stuck | no answer: unreadable | no answer: empty | coordinate mix-up | shifted or stretched | doom loop | off image | on the object, off-centre | duplicate | wrong object | near miss | in a gap | maybe missing from key | background | missed | partial credit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-6 Astra | 78 |  |  |  |  |  |  |  | 3.9 | 0.6 | 1.3 | 0.8 | 0.4 |  | 0.6 | 4.5 | 9.1 |
| GPT-6.1 Sol | 77 |  |  |  |  |  |  |  | 4.7 | 0.7 | 0.9 | 1.1 | 0.3 |  | 0.3 | 5.6 | 8.8 |
| Gemini 3.8 Flash | 74 |  |  |  |  |  |  |  | 4.5 | 0.2 | 0.3 | 0.1 | 0.1 |  |  | 12.4 | 9.5 |
| Gemini 4 Argon | 74 |  |  |  |  |  |  |  | 7.2 | 0.3 | 0.3 | 0.1 | 0.1 |  | 0.1 | 6.5 | 11.9 |
| Muse Wonder (API) | 72 |  |  |  |  |  |  |  | 4.6 | 0.5 | 0.7 | 2.5 | 0.2 |  | 0.3 | 6.6 | 13.3 |
| Qwen 3.8 Flash Next | 70 |  |  |  |  |  |  |  | 4.6 | 0.9 | 0.7 | 1.6 | 0.1 |  | 0.1 | 14.3 | 9.3 |
| Qwen 3.8 27B | 68 |  |  |  |  |  |  |  | 4.7 | 0.9 | 0.4 | 3.3 | 0.2 |  | 0.1 | 14.1 | 9.6 |
| Claude Opus 5.5 | 65 |  |  |  |  |  |  |  | 10.7 | 0.4 | 0.3 | 2.7 | 0.3 |  |  | 11.6 | 10.3 |
| MolmoPoint 8B | 64 |  |  |  |  |  |  |  | 6.6 | 0.7 | 0.3 | 0.3 |  |  | 0.1 | 20.4 | 8.4 |
| Moondream 3.1 | 62 |  |  |  |  |  |  |  | 10.6 | 0.5 | 0.3 |  | 0.1 |  |  | 18.3 | 8.2 |
| Gemini 3.1 Pro | 61 |  |  |  | 5.6 |  |  |  | 7.0 | 0.6 | 0.1 | 0.8 | 0.1 |  |  | 12.6 | 11.1 |
| Gemini 3.5 Flash Lite | 56 |  |  |  | 1.2 |  |  |  | 3.2 | 1.1 | 0.5 | 1.8 | 0.4 |  | 0.2 | 19.9 | 12.0 |
| Kimi K3 | 56 |  |  |  |  |  |  |  | 8.0 | 0.8 | 0.6 | 4.6 | 0.2 |  | 1.0 | 19.0 | 10.3 |
| Mistral Large 4 | 56 | 6.1 |  |  |  |  |  |  | 5.9 | 1.1 | 0.3 | 2.7 | 0.2 |  | 0.1 | 19.3 | 10.0 |
| Claude Haiku 5.5 | 56 |  |  | 9.1 |  |  |  |  | 10.6 | 0.5 | 0.3 | 1.2 | 0.2 | 0.1 |  | 15.8 | 8.5 |
| LocateAnything 3B | 56 |  |  |  |  |  |  |  | 9.0 | 1.2 | 0.8 | 2.4 | 0.1 |  | 0.1 | 23.0 | 8.5 |
| Claude Sonnet 5.5 | 55 |  |  |  |  |  |  | 0.2 | 11.3 | 2.0 | 0.1 | 3.7 | 0.6 |  | 0.4 | 15.5 | 8.7 |
| Rex-Omni 3B | 52 |  |  |  |  |  | 0.3 |  | 8.8 | 1.2 | 0.7 | 0.3 | 0.2 |  | 0.2 | 25.8 | 8.2 |
| GroundingPI 4B | 49 |  |  |  |  |  |  |  | 3.4 |  | 0.4 |  | 0.1 |  | 0.2 | 43.4 | 4.8 |
| Perceptron Mk1.5 | 40 | 18.2 | 18.2 |  |  |  | 0.4 |  | 3.5 | 0.1 | 0.1 | 0.2 |  |  |  | 11.6 | 6.6 |
| SAM 3 | 40 |  |  | 27.3 |  |  |  |  | 15.3 | 0.4 | 1.1 |  | 0.1 |  |  | 8.0 | 8.1 |
| GPT-6 Luna | 36 |  |  |  | 0.5 | 1.0 | 1.3 |  | 14.8 | 0.6 | 0.5 | 5.9 | 0.4 |  | 1.4 | 22.9 | 9.3 |
| GPT-5.6 Terra | 34 |  |  |  |  | 0.6 |  |  | 13.8 | 0.6 | 0.4 | 5.1 | 0.5 |  | 0.3 | 28.8 | 6.4 |
| DeepSeek Flash | 33 |  |  | 3.0 |  | 3.7 |  |  | 10.8 | 0.4 | 0.3 | 6.1 | 0.1 |  | 0.6 | 26.1 | 11.6 |
| GLM 5.3 Flash | 29 |  |  |  |  | 1.6 |  |  | 14.5 | 0.5 | 0.2 | 7.0 | 0.3 |  | 0.8 | 30.7 | 12.2 |
| Florence-2 Large | 26 |  |  |  |  |  |  |  | 4.6 | 1.2 | 1.0 | 4.0 |  |  |  | 56.4 | 3.6 |
| YOLOE-26x | 16 |  |  | 45.5 |  |  |  |  | 3.2 |  | 0.3 |  |  |  |  | 33.2 | 3.6 |
| RF-DETR Seg | 14 |  |  | 63.6 |  |  |  |  | 4.1 |  | 6.8 |  |  |  |  | 10.6 | 2.7 |

## Boxes (box benchmark v1, 825 replies)

| kind | wrong boxes | share | objects left unfound | share |
|---|---:|---:|---:|---:|
| no answer: empty |  |  | 546 | 7.7% |
| coordinate mix-up | 71 | 1.9% | 93 | 1.3% |
| shifted or stretched | 46 | 1.2% | 55 | 0.8% |
| doom loop | 61 | 1.6% |  |  |
| off image | 96 | 2.5% |  |  |
| group box | 553 | 14.6% | 1557 | 22.0% |
| mispositioned box | 1463 | 38.6% |  |  |
| duplicate | 506 | 13.4% |  |  |
| wrong object | 305 | 8.1% |  |  |
| background | 685 | 18.1% |  |  |
| missed |  |  | 4822 | 68.2% |
| total | 3786 | | 7073 | |

Correct boxes: 11114 (full credit at the strictest level: 1906). Objects left unfound: the reply-level kind when there is one, the group box that swallowed them, or missed.

### Score each kind costs, per model (points out of 100, as charted)

| model | score | no answer: stuck | no answer: unreadable | no answer: empty | coordinate mix-up | shifted or stretched | doom loop | off image | group box | mispositioned box | duplicate | wrong object | maybe missing from key | background | missed | loose fit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemini 4 Argon | 63 |  |  |  |  |  |  |  | 1.1 | 1.3 | 0.2 |  |  |  | 8.5 | 27.8 |
| Claude Opus 5.5 | 62 |  |  |  |  |  |  |  | 1.4 | 2.1 | 0.3 | 0.1 |  | 0.2 | 7.5 | 28.2 |
| Claude Sonnet 5.5 | 62 |  |  |  |  |  |  |  | 0.6 | 0.6 | 0.1 | 0.2 |  | 0.1 | 13.1 | 25.9 |
| GPT-6 Astra | 57 |  |  |  |  |  |  |  | 1.5 | 2.3 | 0.6 | 0.1 |  | 0.7 | 4.4 | 33.5 |
| Gemini 3.8 Flash | 56 |  |  |  |  |  |  |  | 0.3 | 0.9 | 0.2 | 0.1 |  |  | 15.3 | 30.8 |
| GPT-6.1 Sol | 53 |  |  |  |  |  |  |  | 1.9 | 3.3 | 0.9 | 0.1 |  | 0.5 | 5.3 | 35.0 |
| Qwen 3.8 Flash Next | 53 |  |  |  |  |  |  |  | 2.3 | 3.0 | 0.3 | 0.1 |  | 0.2 | 13.0 | 30.9 |
| GroundingPI 4B | 53 |  |  |  |  |  |  |  | 0.2 | 2.5 | 0.6 | 1.7 |  | 0.3 | 14.2 | 26.4 |
| Claude Haiku 5.5 | 51 |  |  |  |  |  |  |  | 2.1 | 3.0 | 0.2 | 0.9 |  |  | 14.2 | 31.6 |
| Gemini 3.1 Pro | 50 |  |  | 3.0 |  |  |  |  | 2.7 | 1.7 | 0.1 | 0.1 |  | 0.1 | 17.3 | 28.2 |
| Gemini 3.5 Flash Lite | 50 |  |  |  | 1.2 |  |  |  | 11.3 | 0.4 | 0.3 |  |  | 0.1 | 15.2 | 23.3 |
| Qwen 3.8 27B | 50 |  |  |  |  |  |  |  | 2.8 | 4.7 | 0.4 | 0.2 |  | 0.1 | 13.4 | 31.0 |
| LocateAnything 3B | 47 |  |  |  |  |  |  |  | 11.3 | 0.8 | 0.3 | 2.8 |  |  | 18.0 | 20.9 |
| Rex-Omni 3B | 44 |  |  |  |  |  |  |  | 4.2 | 0.7 | 0.7 | 0.2 |  |  | 26.5 | 26.4 |
| Muse Wonder (API) | 42 |  |  |  |  |  |  |  | 2.5 | 9.6 | 0.4 | 0.2 |  | 0.3 | 8.1 | 37.9 |
| Moondream 3.1 | 41 |  |  |  |  |  |  |  | 1.2 | 3.3 | 0.1 | 1.2 |  |  | 25.1 | 32.4 |
| SAM 3 | 41 |  |  | 27.3 |  |  |  |  | 0.8 | 0.5 | 1.5 | 0.3 |  |  | 7.4 | 21.3 |
| Perceptron Mk1.5 | 41 |  |  | 3.0 |  |  |  |  | 1.3 | 3.6 | 0.1 | 0.3 |  | 0.1 | 22.5 | 28.0 |
| Kimi K3 | 38 |  |  |  |  |  |  |  | 10.5 | 13.4 | 0.3 |  |  | 0.3 | 16.6 | 24.6 |
| Mistral Large 4 | 31 |  |  | 6.1 | 1.4 |  |  |  | 3.7 | 8.0 | 0.3 | 0.1 |  | 0.2 | 18.9 | 31.0 |
| GPT-6 Luna | 29 |  |  |  |  | 1.1 |  |  | 3.7 | 17.3 | 0.5 | 0.1 |  | 1.2 | 18.4 | 24.1 |
| Florence-2 Large | 24 |  |  |  |  |  |  |  | 32.0 | 0.6 | 1.3 | 1.4 |  |  | 27.6 | 10.8 |
| GPT-5.6 Terra | 24 |  |  |  |  |  |  |  | 10.2 | 12.1 | 0.4 | 0.1 |  | 0.1 | 29.3 | 19.4 |
| RF-DETR Seg | 14 |  |  | 63.6 |  |  |  |  |  |  |  | 5.1 |  |  | 9.8 | 6.5 |
| DeepSeek Flash | 13 |  |  |  |  | 0.2 |  |  | 9.3 | 26.0 | 0.1 | 0.1 |  | 0.2 | 30.2 | 18.1 |
| YOLOE-26x | 13 |  |  | 45.5 |  |  |  |  | 1.1 | 0.6 |  |  |  |  | 32.3 | 9.3 |
| GLM 5.3 Flash | 10 |  |  |  |  | 0.7 |  |  | 7.9 | 26.3 | 0.1 |  |  | 0.1 | 32.1 | 17.3 |

Mispositioned boxes: off 931, too big 373, part 159.

Each figure: how much higher that model's charted score would be if that one kind alone were fixed. Fixes interact; the figures do not add up.
