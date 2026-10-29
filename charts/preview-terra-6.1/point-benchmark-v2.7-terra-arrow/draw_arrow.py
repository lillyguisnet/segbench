# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9", "svgelements>=1.9"]
# ///
"""The 4-photo chart, every model greyed but the two Terras, with an arrow
from GPT-5.6 Terra to GPT-6.1 Terra (preview, 2026-10-26).

    LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH uv run charts/preview-terra-6.1/point-benchmark-v2.7-terra-arrow/draw_arrow.py [RAD]

Draws with scripts/draw_charts.py unchanged (same records, label pins and
highlight as ../point-benchmark-v2.7-terra-highlight/with-specialists); the
arrow and the axis's "log scale" are added just before saving. RAD bends the arrow (0 = straight).
"""
import csv
import sys
from pathlib import Path

import matplotlib.figure
from matplotlib import patheffects
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SRC = ROOT / "charts/preview-terra-6.1/point-benchmark-v2.7-terra-highlight/with-specialists"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from segbench.chart_editorial import BIG_PT, INK, SMALL_PT  # noqa: E402

RAD = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
pts = {r["entrant"]: r for r in csv.DictReader(open(SRC / "points.csv"))}
xy = lambda e: (float(pts[e]["cost_usd"]) * 1000, float(pts[e]["quality"]) * 100)  # noqa: E731
OLD, NEW = xy("terra@medium"), xy("terra-6.1@medium")

_save = matplotlib.figure.Figure.savefig


def savefig(fig, *a, **k):
    if not getattr(fig, "_terra_arrow", False):
        ax = fig.axes[0]
        ax.add_patch(FancyArrowPatch(OLD, NEW, connectionstyle=f"arc3,rad={RAD}", arrowstyle="-|>,head_length=9,head_width=5",
                                     mutation_scale=1.6, linewidth=2.6, color=INK,
                                     shrinkA=SMALL_PT * 0.75, shrinkB=BIG_PT * 0.75, zorder=9,
                                     path_effects=[patheffects.withStroke(linewidth=6, foreground="white")]))
        # The cost axis says it is a log scale, worded as in scripts/plot_new_models.py.
        ax.xaxis.label.set_text("Cost ($ per 1,000 images, log scale)")
        fig._terra_arrow = True
    return _save(fig, *a, **k)


matplotlib.figure.Figure.savefig = savefig
import draw_charts  # noqa: E402

sys.argv = ["draw_charts.py", str(SRC / "calls.jsonl"), "--style", "editorial", "--track", "find", "--names", "all",
            "--out", str(HERE), "--label-spot=gemini-pro@medium=left",
            "--highlight", "terra@medium", "--highlight", "terra-6.1@medium"]
draw_charts.main()
