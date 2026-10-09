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
}
INITIALS = {"UC Davis": "UCD"}

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
}
OTHER = "#7a808a"


def initials(maker: str) -> str:
    return INITIALS.get(maker) or "".join(w[0] for w in maker.split())[:3]
