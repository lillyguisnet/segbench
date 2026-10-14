# Maker logos on the charts

Used only to show who made each model. The logos are trademarks of their
owners; they are **not** covered by this project's Apache 2.0 licence.

| file | owner | drawing from | its licence |
|---|---|---|---|
| `anthropic.svg`, `googlegemini.svg`, `deepseek.svg`, `qwen.svg`, `kimi.svg`, `zdotai.svg`, `meta.svg`, `ultralytics.svg`, `roboflow.svg` | Anthropic, Google (Gemini), DeepSeek, Alibaba (Qwen), Moonshot AI (Kimi), Z.ai, Meta, Ultralytics, Roboflow | [Simple Icons](https://simpleicons.org) 16.34.0 | CC0 1.0 |
| `openai.svg` | OpenAI | [Lobe Icons](https://github.com/lobehub/lobe-icons) (`@lobehub/icons-static-svg` 1.95.1); Simple Icons dropped OpenAI in version 16 | MIT |

| `nvidia.svg` | NVIDIA | [Simple Icons](https://simpleicons.org) 16.34.0 | CC0 1.0 |
| `microsoft.svg`, `ai2.svg` | Microsoft (Florence-2), Ai2 (MolmoPoint) | [Lobe Icons](https://github.com/lobehub/lobe-icons) (`@lobehub/icons-static-svg` 1.95.1); the Microsoft file is titled "Azure" there but is the four-square Microsoft mark | MIT |
| `perceptron.svg` | Perceptron (Mk1.5) | [Lobe Icons](https://github.com/lobehub/lobe-icons) (`@lobehub/icons-static-svg` 1.95.1), downloaded 2026-10-12; its holes rely on the even-odd fill rule, which the charts and the site do not use, so it was rewritten once with [picosvg](https://github.com/googlefonts/picosvg) into the same shape under the default rule (rendered pixel-identical at 240 px) | MIT |
| `mistralai.svg` | Mistral AI (Mistral Large 4) | [Simple Icons](https://simpleicons.org) 16.34.0, downloaded 2026-10-14, unchanged | CC0 1.0 |
| `moondream.svg` | M87 Labs (Moondream) | moondream.ai's own icon, `https://moondream.ai/safari-pinned-tab.svg` | the owner's trademark (in neither library) |

Downloaded 2026-10-09 (the last three on 2026-10-10), unchanged. Moondream (2026-10-11): its
icon is a moon face, a disc with two eyes and a smile all filled in one
colour (so the face does not show); we keep the eyes and smile only, in the
drawing's own coordinates, framed around the moon's centre (the face keeps
its place, a little below the middle), so the chart's disc is the moon. IDEA
Research (Rex-Omni) is in neither library: the chart writes "IDEA". Where the models are better known than
their maker, the chart uses the models' mark: the Gemini star (not Google's
G), Qwen (not Alibaba Cloud), Kimi (not Moonshot AI). UC Davis (gen2seg) has no logo here: the
chart shows its initials. GroundingPI (2026-10-14) is in neither library and
its only mark is a wide picture of its name: the chart writes "GPI". To add a maker: put its single-colour SVG here
and add it to `LOGOS` in `segbench/chart.py`.
