# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9", "svgelements>=1.9"]
# ///
"""Box benchmark v1: every model's score at each overlap level, one row per model.

    uv run scripts/plot_box_levels.py   ->  charts/box-benchmark-v1/overlap-levels/levels.{png,pdf,svg}

Replaces one bubble chart per overlap level: cost does not change with the
level, so the levels are compared side by side in one ranking. Rows sorted
by the average over the four levels (the main charts' score). Same fonts
and colours as the editorial charts.
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from segbench.chart import _fonts  # noqa: E402
from segbench.chart_editorial import ACCENT, GRID, INK, INK_2, INK_3  # noqa: E402

LEVELS = [("50", "#f7a8ae"), ("75", "#f36d77"), ("90", ACCENT), ("95", "#8f1720")]


def main():
    rows = list(csv.DictReader(open(ROOT / "results/box-benchmark/v1/summary.csv")))
    rows.sort(key=lambda r: -float(r["score"]))
    f = _fonts()
    n = len(rows)
    fig = plt.figure(figsize=(10, 3.2 + 0.36 * n), dpi=216)
    top = 1 - 1.75 / fig.get_figheight()
    ax = fig.add_axes([0.30, 0.9 / fig.get_figheight(), 0.60, top - 0.9 / fig.get_figheight()])
    fig.text(0.04, 1 - 0.7 / fig.get_figheight(), "Box scores at each overlap level", fontproperties=f.bold,
             fontsize=24, color=INK, va="baseline")
    fig.text(0.04, 1 - 1.12 / fig.get_figheight(),
             "A box counts when it overlaps the right answer by at least 50%, 75%, 90% or 95%. "
             "Rows sorted by the average.", fontproperties=f.regular, fontsize=11.5, color=INK_2, va="baseline")
    for i, r in enumerate(rows):
        y = n - 1 - i
        vals = [float(r[f"f1_{lv}"]) * 100 for lv, _ in LEVELS]
        ax.plot([min(vals), max(vals)], [y, y], color=GRID, lw=3, zorder=1, solid_capstyle="round")
        for (lv, c), v in zip(LEVELS, vals):
            ax.scatter([v], [y], s=70, color=c, edgecolor="white", linewidth=1.2, zorder=3)
        local = r["kind"] == "specialist"
        ax.text(-2, y, r["label"], ha="right", va="center", fontproperties=f.semibold if not local else f.regular,
                fontsize=12, color=INK if not local else INK_2, transform=ax.transData, clip_on=False)
        ax.text(101.5, y, f"{float(r['score']) * 100:.1f}", ha="left", va="center", fontproperties=f.semibold,
                fontsize=11, color=INK_2, clip_on=False)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_yticks([])
    ax.set_xticks(range(0, 101, 20))
    ax.set_xticklabels([f"{x}%" for x in range(0, 101, 20)], fontproperties=f.medium, fontsize=11, color=INK_3)
    ax.grid(axis="x", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.text(101.5, n - 0.1, "avg", ha="left", va="bottom", fontproperties=f.medium, fontsize=10, color=INK_3)
    handles = [plt.Line2D([], [], marker="o", ls="", markersize=9, markerfacecolor=c, markeredgecolor="white",
                          label=f"{lv}% overlap") for lv, c in LEVELS]
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.30, 1 - 1.3 / fig.get_figheight()),
                     ncol=4, frameon=False, prop=f.medium, fontsize=11, handletextpad=0.3, columnspacing=1.4)
    for t in leg.get_texts():
        t.set_color(INK_2)
        t.set_fontsize(11)
    fig.text(0.04, 0.25 / fig.get_figheight(), "Grey names: models run on our own GPU.", fontproperties=f.regular,
             fontsize=10, color=INK_3)
    out = ROOT / "charts/box-benchmark-v1/overlap-levels"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"levels.{ext}", facecolor="white")
    print("saved", out)


if __name__ == "__main__":
    main()
