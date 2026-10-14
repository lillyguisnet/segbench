# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9", "svgelements>=1.9"]
# ///
"""The photo chart (point benchmark v2.7) with the four models added on 2026-10-14 in colour.

uv run scripts/plot_new_models.py  ->  charts/point-benchmark-v2.7-new-models/

Gemini 4 Argon, Claude Haiku 5.5, Mistral Large 4 and GroundingPI 4B keep
their colours; every other model is grey. Same scores as the published v2.7
chart (charts/point-benchmark-v2.7), with Maxime's choices of 2026-10-14 for
this one picture:
- Muse Wonder Full left out (its score stays in the published chart);
- the cost axis says it is a log scale;
- labels placed by hand: every best-value model's label on the white side
  of the red line, the rest moved around them (LABELS below; a spot beside
  the disc, then a nudge in points, x to the right and y up);
- the language-model chart's cost axis starts a little lower (WIDEN_LEFT).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from segbench.chart_data import aggregate, read_calls, write_points  # noqa: E402
from segbench.chart import x_range  # noqa: E402
from segbench.chart_editorial import draw  # noqa: E402

SOURCE = ROOT / "charts" / "point-benchmark-v2.7"
OUT = ROOT / "charts" / "point-benchmark-v2.7-new-models"
NEW = {"gemini-argon@medium", "haiku@medium", "mistral-large-4@medium", "groundingpi-4b"}
LEFT_OUT = {"muse-wonder-full"}  # by model
XLABEL = "Cost ($ per 1,000 images, log scale)"
LABELS = {
    "with-specialists": {
        # best-value line, left (white) side: a column above YOLOE's disc,
        # which RF-DETR's covers (same cost, 23.2 vs 23.5)
        "yoloe-26x": "above",
        "rfdetr-seg-2xl": "above@0,36!",
        "sam3": "ul@0,34",
        "florence2-large-grounding": "ul@0,53",
        "moondream-3.1": "above@-15,0!",
        "haiku@medium": "left@0,14!",
        # the rest
        "groundingpi-4b": "right@0,-7!",
        "rexomni-3b": "lr@40,0",
        "locateanything-3b": "right",
        "qwen-27b@medium": "below@-6,-3!",
        "gemini-flash@medium": "lr@3,-25",
        "sol@medium": "right@33,-8",
    },
    "language-models": {
        "haiku@medium": "left",
        "qwen-27b@medium": "ll",
        "gemini-flash@medium": "lr@-9,-4",
    },
}
# Room on the left of the cost axis, in powers of ten: on the language-model
# chart Claude Haiku 5.5 is the cheapest model, and its label needs the space
# left of its disc to stay on the white side of the line.
WIDEN_LEFT = {"language-models": 0.4}


def main() -> None:
    for variant, spots in LABELS.items():
        folder = OUT / variant
        folder.mkdir(parents=True, exist_ok=True)
        rows = [json.loads(line) for line in (SOURCE / variant / "calls.jsonl").read_text().splitlines()]
        calls = folder / "calls.jsonl"
        calls.write_text("".join(json.dumps(r) + "\n" for r in rows if r["model"] not in LEFT_OUT))
        points = aggregate(read_calls([calls]))
        write_points(points, folder / "points.csv")
        lo, hi = x_range(points)
        xlim = (lo / 10 ** WIDEN_LEFT.get(variant, 0), hi)
        for path in draw(points, "find", folder / "find", xlim=xlim, label_spots=spots,
                         highlight=NEW, xlabel=XLABEL):
            print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
