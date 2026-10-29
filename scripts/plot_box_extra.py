# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9", "svgelements>=1.9"]
# ///
"""Three more charts of box benchmark v1 (Maxime, 2026-10-16).

    uv run scripts/plot_box_extra.py   ->  charts/box-benchmark-v1/{points-vs-boxes,levels-logos,original-4-photos}/

1. points-vs-boxes: each model's point score (point benchmark v3, round mark)
   and box score (square mark), joined by a line; logos inside the marks.
2. levels-logos: each model's box score at 50, 75, 90 and 95% overlap,
   joined by a line, darker = stricter; the model's logo beside its name.
3. original-4-photos: the main bubble chart (score against cost) on the four
   original photos only (logs, cows, fig leaves, dishes); each model keeps
   the prompt chosen on all eleven photos (v1), so only the photos change.
4. points-vs-boxes-4-photos: chart 1 on those four photos only (point scores:
   point v3's per-photo scores, its prompt choice made on all eleven photos).
"""
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from segbench.chart import INITIALS, _fonts, logo_path  # noqa: E402
from segbench.chart_editorial import ACCENT, BRAND, GRID, INK, INK_2, INK_3  # noqa: E402  (the bubble charts' logo colours)

BOX = ROOT / "results/box-benchmark/v1"
POINT = ROOT / "results/point-benchmark-v3/summary.csv"
OUT = ROOT / "charts/box-benchmark-v1"
LEVELS = ("50", "75", "90", "95")
SHADES = (0.28, 0.5, 0.75, 1.0)  # share of the accent colour mixed into white, 50% -> 95%
F = None


def shade(t):
    a, w = to_rgb(ACCENT), (1, 1, 1)
    return tuple(w[i] + (a[i] - w[i]) * t for i in range(3))


def mark(ax, x, y, maker, d, shape, rim, rim_w=1.6, z=4, fill="white", logo=True):
    """A white disc (shape 'o') or square ('s') with the maker's logo, as on the bubble charts."""
    ax.scatter([x], [y], s=d ** 2, marker=shape, facecolors=[fill], edgecolors=[rim], linewidths=rim_w, zorder=z,
               clip_on=False)
    if not logo:
        return
    p, size, colour = logo_path(maker), d * 0.56, BRAND.get(maker, INK)
    if p is not None:
        ax.scatter([x], [y], s=size ** 2, marker=p, facecolors=[colour], edgecolors="none", zorder=z + 0.01,
                   clip_on=False)
    else:
        text = INITIALS.get(maker) or "".join(w[0] for w in maker.split())[:3]
        ax.text(x, y, text, ha="center", va="center_baseline", fontproperties=F.bold,
                fontsize=size * (0.6 if len(text) < 3 else 0.44), color=colour, zorder=z + 0.01)


def frame(n, title, subtitle):
    fig = plt.figure(figsize=(10, 2.9 + 0.42 * n), dpi=216)
    H = fig.get_figheight()
    ax = fig.add_axes([0.33, 0.85 / H, 0.56, 1 - 2.05 / H - 0.85 / H])
    fig.text(0.04, 1 - 0.72 / H, title, fontproperties=F.bold, fontsize=24, color=INK, va="baseline")
    fig.text(0.04, 1 - 1.12 / H, subtitle, fontproperties=F.regular, fontsize=11.5, color=INK_2, va="top",
             linespacing=1.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_yticks([])
    ax.set_xticks(range(0, 101, 20))
    ax.set_xticklabels([f"{x}%" for x in range(0, 101, 20)], fontproperties=F.medium, fontsize=11, color=INK_3)
    ax.grid(axis="x", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.text(0.04, 0.28 / H, "Grey names: models run on our own GPU.", fontproperties=F.regular, fontsize=10,
             color=INK_3)
    return fig, ax, H


def name(ax, y, r, x=-3):
    local = r["kind"] == "specialist"
    ax.text(x, y, r["label"], ha="right", va="center", fontproperties=F.regular if local else F.semibold,
            fontsize=12, color=INK_2 if local else INK, clip_on=False)


def save(fig, folder, stem):
    d = OUT / folder
    d.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(d / f"{stem}.{ext}", facecolor="white")
    plt.close(fig)
    print("saved", d / f"{stem}.png")


def points_vs_boxes(rows, tasks=None, box_scores=None, folder="points-vs-boxes",
                    subtitle="Same 11 photos, two different tests: compare the order, not the numbers."):
    """tasks=None: all eleven photos. Otherwise the point score is the mean of
    those photos' point scores (point v3's score is the plain mean of its
    per-photo scores) and box_scores gives the box score on the same photos."""
    prow = [r for r in csv.DictReader(open(POINT)) if r["score"]]
    if tasks is None:
        pts = {r["entrant"]: float(r["score"]) * 100 for r in prow}
        box = {r["entrant"]: float(r["score"]) * 100 for r in rows}
    else:
        pts = {r["entrant"]: statistics.mean(float(r[t]) for t in tasks) * 100 for r in prow}
        box = box_scores
    rows = [r for r in rows if r["entrant"] in pts]
    rows.sort(key=lambda r: -box[r["entrant"]])
    n = len(rows)
    fig, ax, H = frame(n, "Pointing at things vs. drawing boxes around them", subtitle)
    key_y = 1 - 1.62 / H
    for x, shape, rim, text in ((0.33, "o", INK_3, "point score"), (0.50, "s", ACCENT, "box score")):
        fig.lines.append(plt.Line2D([x], [key_y], marker=shape, ms=11, mfc="white", mec=rim, mew=1.6, ls="",
                                    transform=fig.transFigure, figure=fig))
        fig.text(x + 0.018, key_y, text, fontproperties=F.medium, fontsize=11, color=INK_2, va="center")
    up = "#1f9d63"
    for i, r in enumerate(rows):
        y = n - 1 - i
        p, b = pts[r["entrant"]], box[r["entrant"]]
        ax.plot([p, b], [y, y], color=GRID, lw=4.5, zorder=1, solid_capstyle="round")
        mark(ax, p, y, r["maker"], 19, "o", INK_3, rim_w=1.3, z=3)
        mark(ax, b, y, r["maker"], 19, "s", ACCENT, rim_w=1.8, z=4)
        name(ax, y, r)
        d = b - p
        d = round(d)
        ax.text(103, y, (f"{d:+d}" if d else "0").replace("-", "−"), ha="left", va="center", fontproperties=F.semibold, fontsize=11,
                color=up if d > 0 else INK_3, clip_on=False)
    ax.text(103, n - 0.15, "box − point", ha="left", va="bottom", fontproperties=F.medium, fontsize=9.5,
            color=INK_3, clip_on=False)
    save(fig, folder, folder)
    return {r["entrant"]: (pts[r["entrant"]], box[r["entrant"]]) for r in rows}


def levels_logos(rows):
    rows = sorted(rows, key=lambda r: -float(r["score"]))
    n = len(rows)
    fig, ax, H = frame(n, "Box scores, from loose to strict",
                       "A box counts when it overlaps the right answer by at least 50%, 75%, 90% or 95%. "
                       "Darker = stricter.")
    for i, r in enumerate(rows):
        y = n - 1 - i
        vals = [float(r[f"f1_{lv}"]) * 100 for lv in LEVELS]
        ax.plot(vals, [y] * 4, color=shade(0.35), lw=2.2, zorder=1, solid_capstyle="round")
        for v, t in zip(vals, SHADES):
            ax.scatter([v], [y], s=11 ** 2, marker="s", color=shade(t), edgecolors="white", linewidths=1.1, zorder=3)
        mark(ax, -2.6, y, r["maker"], 17, "o", GRID, rim_w=1.1, z=3)
        name(ax, y, r, x=-6)
        ax.text(103, y, f"{float(r['score']) * 100:.1f}", ha="left", va="center", fontproperties=F.semibold,
                fontsize=11, color=INK_2, clip_on=False)
    ax.text(103, n - 0.15, "average", ha="left", va="bottom", fontproperties=F.medium, fontsize=9.5, color=INK_3,
            clip_on=False)
    handles = [plt.Line2D([], [], marker="s", ls="", markersize=10, markerfacecolor=shade(t), markeredgecolor="white",
                          label=f"{lv}%") for lv, t in zip(LEVELS, SHADES)]
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.33, 1 - 1.3 / H), ncol=4, frameon=False,
                     prop=F.medium, handletextpad=0.3, columnspacing=1.6, title="overlap needed:",
                     title_fontproperties=F.medium)
    leg._legend_box.align = "left"
    for t in leg.get_texts():
        t.set_color(INK_2)
        t.set_fontsize(11)
    leg.get_title().set_color(INK_3)
    leg.get_title().set_fontsize(10.5)
    save(fig, "levels-logos", "levels-logos")


FOUR = ("logs", "cows", "fig", "dishes")
# Label pins for the bubble chart on the four photos, as in scripts/plot_boxes.py.
# Left to the placer, Gemini 4 Argon's label went far right on a long line
# and RF-DETR's line crossed Florence-2's disc.
SPOTS_FOUR: dict[str, str] = {"gemini-argon@medium": "lr@6,-7", "rfdetr-seg-2xl": "right@0,-9!"}


def original_four():
    calls = [json.loads(l) for l in (BOX / "calls.jsonl").read_text().splitlines()]
    four = [c for c in calls if c["task"] in FOUR]
    folder = OUT / "original-4-photos"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in four))
    spots = [f"--label-spot={e}={s}" for e, s in SPOTS_FOUR.items()]
    subprocess.run(["uv", "run", str(ROOT / "scripts/draw_charts.py"), str(folder / "calls.jsonl"), "--style",
                    "editorial", "--track", "find", "--names", "all", "--out", str(folder), *spots,
                    "--title", "How well can AI box things in the 4 original photos?"],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    # check: each plotted score = mean over the four tasks of the mean over tries
    want = {}
    for c in four:
        e = f"{c['model']}@{c['level']}" if c["level"] else c["model"]
        want.setdefault(e, {}).setdefault(c["task"], []).append(c["score"])
    plotted = list(csv.DictReader(open(folder / "points.csv")))
    for p in plotted:
        w = statistics.mean(statistics.mean(v) for v in want[p["entrant"]].values())
        assert abs(float(p["quality"]) - w) < 1e-12, p["entrant"]
    print(f"original-4-photos: {len(plotted)} points, checked")
    return {p["entrant"]: float(p["quality"]) * 100 for p in plotted}


def main():
    global F
    F = _fonts()
    rows = list(csv.DictReader(open(BOX / "summary.csv")))
    pv = points_vs_boxes(rows)
    levels_logos(rows)
    four = original_four()
    pv4 = points_vs_boxes(rows, FOUR, four, "points-vs-boxes-4-photos",
                          "The 4 original photos only (logs, cows, fig leaves, dishes). Two different tests: "
                          "compare the order, not the numbers.")
    (OUT / "extra-charts-data.json").write_text(json.dumps({"points_vs_boxes": pv, "original_4_photos": four,
                                                            "points_vs_boxes_4_photos": pv4}, indent=1))


if __name__ == "__main__":
    main()
