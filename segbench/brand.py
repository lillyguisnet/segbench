"""Who made each model, as drawn: logo files and brand colours. No third-party packages.

Used by the charts (segbench/chart.py), the point overlays, the points site
and the points video, so a maker looks the same everywhere.
"""

from __future__ import annotations

from pathlib import Path

LOGO_DIR = Path(__file__).resolve().parents[1] / "assets" / "logos"

# Maker -> logo file in assets/logos (sources there): the mark readers know
# the models by (the Gemini star, Kimi, Qwen), else the company's. A maker
# without a logo gets its initials.
LOGOS = {
    "OpenAI": "openai.svg",
    "Anthropic": "anthropic.svg",
    "Google": "googlegemini.svg",
    "DeepSeek": "deepseek.svg",
    "Alibaba": "qwen.svg",
    "Moonshot": "kimi.svg",
    "Z.AI": "zdotai.svg",
    "Meta": "meta.svg",
    "Ultralytics": "ultralytics.svg",
    "Roboflow": "roboflow.svg",
    "NVIDIA": "nvidia.svg",
    "Ai2": "ai2.svg",
    "Microsoft": "microsoft.svg",
    "Moondream": "moondream.svg",
    "Perceptron": "perceptron.svg",
    "Mistral": "mistralai.svg",
}
# No logo file yet for these (each needs its source and licence noted in
# assets/logos/SOURCES.md first): initials instead.
INITIALS = {"UC Davis": "UCD", "IDEA": "IDEA"}  # no logo in Simple Icons or Lobe Icons (checked 2026-10-10)
# GroundingPI (checked 2026-10-14): in neither library, and its only mark is
# a wide picture of its name (the README title), not a mark for a small disc.
INITIALS["GroundingPI"] = "GPI"
# TypeLLM (2026-10-17): no logo in either library, and its site's mark is its name in type.
INITIALS["TypeLLM"] = "TL"

# Brand colours (as on the 2026-10-09 brand-colour charts, commit 3aca946),
# used where colour shows the maker (the point overlays, site and video; the
# bubble charts colour by speed instead). Two makers may share a colour (Meta
# and Google are both blue): the logo tells makers apart.
BRAND = {
    "OpenAI": "#10a37f",  # the ChatGPT green
    "Anthropic": "#d97757",  # Claude's terracotta
    "Google": "#4285f4",
    "DeepSeek": "#26339c",
    "Alibaba": "#615ced",  # Qwen violet
    "Moonshot": "#1c1c1e",  # Kimi black
    "Z.AI": "#e5484d",  # Z.ai is black and white; black is Kimi's
    "Meta": "#0866ff",
    "Roboflow": "#a01ee6",
    "Ultralytics": "#f0507a",
    "NVIDIA": "#76b900",
    "Ai2": "#f0529c",
    "IDEA": "#2b6cb0",
    "Microsoft": "#737373",
    "Moondream": "#b8860b",  # moonlight gold; Moondream's own site is black and white
    "Perceptron": "#1e130e",  # its mark's dark brown (Lobe Icons' Perceptron avatar), near Kimi's black
    "Mistral": "#fa520f",  # Mistral orange (Simple Icons' colour for Mistral AI)
    "GroundingPI": "#7050ad",  # the purple of its name and of the first badge on its README
    "TypeLLM": "#2f6f5e",  # its site is black and white; a dark teal distinct from the others
}
OTHER = "#7a808a"


def initials(maker: str) -> str:
    return INITIALS.get(maker) or "".join(w[0] for w in maker.split())[:3]
