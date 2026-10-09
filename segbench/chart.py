"""Draw one bubble chart per track: quality up, cost across, time as size.

Needs matplotlib (not in the base install): run it through
scripts/draw_charts.py, which declares them.

Reading the chart:
- y: quality, 0 to 100 (the README's 0..1 score x 100), with a thin line
  for the spread over repeats.
- x: US dollars per 1,000 images, log scale (each step is x10).
- bubble size: median seconds per image. The diameter grows with the
  logarithm of the time (0.01 s to 1,000 s on the same scale in every
  chart), because times range from 0.02 s to several minutes: on a linear
  scale the GPU models would be invisible dots.
- colour: who made the model. Models on our own GPU share one colour, so
  the general-versus-specialist story reads at a glance.
- shade: thinking level (light = min, full = max), joined by a thin line.
- ring: a pair (finder + outliner); the ring is the outliner's colour.
- dashed staircase: the best-value frontier.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter  # noqa: E402

from segbench.chart_data import Point, frontier  # noqa: E402
from segbench.models import LEVELS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = ROOT / "assets" / "fonts"

# ---------- look ----------

INK = "#16181d"
INK_2 = "#4a4f5a"
INK_3 = "#8a8f99"
GRID = "#e8e9ec"
PAPER = "#ffffff"
FAKE_RED = "#d62839"

MAKER_COLOURS = {
    "OpenAI": "#10a37f",
    "Anthropic": "#d97757",
    "Google": "#3f7ee8",
    "Z.AI": "#8e5bd0",
    "DeepSeek": "#1fb3c9",
    "Moonshot": "#3a3d45",
    "Alibaba": "#e0a300",
}
GPU_COLOUR = "#d6336c"  # every model we run on our own GPU
OTHER_COLOUR = "#9aa0a6"

LEVEL_ALPHA = {"min": 0.28, "medium": 0.55, "max": 0.88, "": 0.80}

SECONDS_DOMAIN = (0.01, 1000.0)  # same bubble scale on every chart
DIAMETER_PT = (6.0, 40.0)
SIZE_LEGEND = (0.1, 1, 10, 100)  # seconds

TRACK_TEXT = {
    "find": ("Finding objects",
             "Track 1 of 3. The model gets a photo and a request (\u201cput a dot on each cow\u201d) and answers "
             "with one dot per object."),
    "outline": ("Outlining objects",
                "Track 2 of 3. The model is given one dot on each object and must outline it; "
                "scored on what the outline measures (size, area, share, path)."),
    "whole": ("The whole task",
              "Track 3 of 3. Photo and request only: find the objects and outline them. "
              "A ring marks a pair: one model finds, a GPU model outlines."),
}


@dataclass
class Fonts:
    regular: font_manager.FontProperties
    medium: font_manager.FontProperties
    semibold: font_manager.FontProperties
    bold: font_manager.FontProperties


def _fonts() -> Fonts:
    def load(name: str) -> font_manager.FontProperties:
        path = FONT_DIR / f"Inter-{name}.ttf"
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            return font_manager.FontProperties(fname=str(path))
        return font_manager.FontProperties(family="sans-serif", weight=name.lower())

    return Fonts(load("Regular"), load("Medium"), load("SemiBold"), load("Bold"))


def colour_of(maker: str, kind: str) -> str:
    if kind == "specialist":
        return GPU_COLOUR
    return MAKER_COLOURS.get(maker, OTHER_COLOUR)


def _darker(colour: str, f: float = 0.72) -> tuple[float, float, float]:
    r, g, b = to_rgb(colour)
    return (r * f, g * f, b * f)


def diameter(seconds: float) -> float:
    lo, hi = (math.log10(v) for v in SECONDS_DOMAIN)
    t = (math.log10(min(max(seconds, SECONDS_DOMAIN[0]), SECONDS_DOMAIN[1])) - lo) / (hi - lo)
    return DIAMETER_PT[0] + t * (DIAMETER_PT[1] - DIAMETER_PT[0])


def _money(v: float) -> str:
    if v >= 1:
        return f"${v:,.0f}"
    digits = max(0, -math.floor(math.log10(v)))
    return f"${v:.{digits}f}"


def _seconds(v: float) -> str:
    return f"{v:g} s"


def x_range(points: list[Point]) -> tuple[float, float]:
    """Whole decades around all costs, in $ per 1,000 images."""
    xs = [p.cost_usd * 1000 for p in points]
    return 10 ** math.floor(math.log10(min(xs)) - 0.15), 10 ** math.ceil(math.log10(max(xs)) + 0.15)


def draw(points: list[Point], track: str, out_base: Path, *, xlim: tuple[float, float] | None = None,
         footnote: str = "", stamp: str = "", dpi: int = 200) -> list[Path]:
    """Draw the points of one track; return the files written.

    out_base.png  to post (3200 x 2000 pixels at dpi=200)
    out_base.pdf  to print: vector, fonts embedded
    out_base.svg  for the web: vector, text drawn as shapes so it looks the
                  same without the font installed
    `stamp` is the date shown in the corner (the run's date), default today.
    """
    pts = [p for p in points if p.track == track]
    if not pts:
        raise ValueError(f"no points for track {track!r}")
    synthetic = any(p.synthetic for p in pts)
    fonts = _fonts()

    plt.rcParams.update({"svg.fonttype": "path", "pdf.fonttype": 42, "font.family": fonts.regular.get_name()})
    fig = plt.figure(figsize=(16, 10), facecolor=PAPER)
    ax = fig.add_axes((0.062, 0.150, 0.705, 0.680))
    ax.set_facecolor(PAPER)

    # ----- axes -----
    xlim = xlim or x_range(pts)
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    decades = [10.0 ** k for k in range(round(math.log10(xlim[0])), round(math.log10(xlim[1])) + 1)]
    ax.xaxis.set_major_locator(FixedLocator(decades))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10), numticks=100))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticklabels([_money(v) for v in decades])

    qs = [100 * p.quality_low for p in pts] + [100 * p.quality_high for p in pts]
    y0 = max(0, 10 * math.floor((min(qs) - 4) / 10))
    y1 = min(100, 10 * math.ceil((max(qs) + 4) / 10))
    ax.set_ylim(y0, y1)
    ax.set_yticks(range(int(y0), int(y1) + 1, 10))

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9ccd2")
    ax.grid(True, which="major", color=GRID, lw=1.0, zorder=0)
    ax.grid(True, which="minor", axis="x", color="#f3f4f6", lw=0.8, zorder=0)
    ax.tick_params(which="both", length=0, pad=8, labelcolor=INK_2, labelsize=12.5)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontproperties(fonts.regular)
        lab.set_fontsize(12.5)
    ax.set_xlabel("Cost per 1,000 images (US dollars, log scale)", fontproperties=fonts.medium, fontsize=13.5,
                  color=INK_2, labelpad=12)
    ax.set_ylabel("Quality score (0\u2013100)", fontproperties=fonts.medium, fontsize=13.5, color=INK_2, labelpad=12)
    ax.text(0.012, 0.985, "\u2196 better: higher quality, lower cost", transform=ax.transAxes, ha="left", va="top",
            fontproperties=fonts.medium, fontsize=11.5, color=INK_3)

    def X(p: Point) -> float:
        return p.cost_usd * 1000

    def Y(p: Point) -> float:
        return p.quality * 100

    # ----- best-value frontier: a staircase (between two points, nothing better is known) -----
    # Only entrants that attempted every task can be "best value": a model
    # that skipped tasks is cheap for a reason that the frontier would hide.
    best = frontier([p for p in pts if p.tasks_done == p.tasks_total])
    best_ids = {p.entrant for p in best}
    fx = [X(p) for p in best] + [xlim[1]]
    fy = [Y(p) for p in best] + [Y(best[-1])]
    stair = [(fx[0], fy[0])]
    for i in range(1, len(fx)):
        stair += [(fx[i], fy[i - 1]), (fx[i], fy[i])]
    ax.step(fx, fy, where="post", color=INK, lw=1.4, ls=(0, (5, 4)), alpha=0.55, zorder=1)

    # ----- spread over repeats -----
    for p in pts:
        if p.quality_high - p.quality_low > 1e-9:
            ax.vlines(X(p), 100 * p.quality_low, 100 * p.quality_high, color=colour_of(p.maker, p.kind),
                      lw=1.3, alpha=0.45, zorder=2)

    # ----- thinking levels of one model, joined -----
    chains = defaultdict(list)
    for p in pts:
        if p.kind == "general":
            chains[p.model].append(p)
    rank = {lv: i for i, lv in enumerate(LEVELS)}
    for chain in chains.values():
        chain.sort(key=lambda p: rank.get(p.level, 0))
        if len(chain) > 1:
            ax.plot([X(p) for p in chain], [Y(p) for p in chain], color=colour_of(chain[0].maker, "general"),
                    lw=1.3, alpha=0.5, zorder=2, solid_capstyle="round")

    # ----- bubbles, biggest first so small ones stay visible -----
    for p in sorted(pts, key=lambda p: -p.seconds):
        c = colour_of(p.maker, p.kind)
        d = diameter(p.seconds)
        alpha = LEVEL_ALPHA.get(p.level, 0.8)
        face = (*to_rgb(c), alpha)
        if p.kind == "pair":
            ring = colour_of(p.outliner_maker, "specialist")
            ax.scatter([X(p)], [Y(p)], s=(d + 7) ** 2, facecolors="none", edgecolors=ring, linewidths=3.2, zorder=3)
        ax.scatter([X(p)], [Y(p)], s=d ** 2, facecolors=[face], edgecolors=[_darker(c)], linewidths=1.1,
                        zorder=4)

    # ----- labels: one per model (at its most-thinking bubble), one per specialist or pair -----
    labelled: list[tuple[Point, str]] = []
    for chain in chains.values():
        last = chain[-1]
        labelled.append((last, last.label))
    for p in pts:
        if p.kind != "general":
            labelled.append((p, p.label + (f" ({p.level})" if p.level else "")))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px_per_pt = fig.dpi / 72
    circles = []  # (x, y, radius) in pixels, every bubble
    for p in pts:
        cx, cy = ax.transData.transform((X(p), Y(p)))
        circles.append((cx, cy, (diameter(p.seconds) / 2 + (5 if p.kind == "pair" else 0)) * px_per_pt))
    ab = ax.get_window_extent(renderer)
    frame = (ab.x0 + 4, ab.y0 + 4, ab.x1 - 4, ab.y1 - 34)  # keep clear of the "better" note at the top

    def priority(item):
        p, _ = item
        return (p.entrant not in best_ids and not _chain_on_frontier(p), p.kind == "general", -p.quality)

    def _chain_on_frontier(p: Point) -> bool:
        return p.kind == "general" and any(q.entrant in best_ids for q in chains[p.model])

    # Labels must not sit on the frontier line either: each of its segments is an obstacle.
    placed: list[tuple[float, float, float, float]] = []
    seg = [tuple(ax.transData.transform(xy)) for xy in stair]
    for (x0, y0), (x1, y1) in zip(seg, seg[1:]):
        placed.append((min(x0, x1) - 2, min(y0, y1) - 2, max(x0, x1) + 2, max(y0, y1) + 2))
    for p, text in sorted(labelled, key=priority):
        if p.tasks_done < p.tasks_total:
            text += " *"
        strong = p.entrant in best_ids or _chain_on_frontier(p)
        style = dict(fontproperties=fonts.semibold if strong else fonts.regular, fontsize=11.5,
                     color=INK if strong else INK_2, zorder=6)
        probe = ax.text(0, 0, text, **style)
        bb = probe.get_window_extent(renderer)
        probe.remove()
        w, h = bb.width + 4, bb.height + 2  # a little air between neighbouring labels
        cx, cy = ax.transData.transform((X(p), Y(p)))
        r = (diameter(p.seconds) / 2 + (5 if p.kind == "pair" else 0)) * px_per_pt
        box, ha, va, off, far = _place(cx, cy, r, w, h, placed, circles, frame, px_per_pt)
        placed.append(box)
        arrow = dict(arrowstyle="-", color=INK_3, lw=0.8, shrinkA=1, shrinkB=r / px_per_pt + 1) if far else None
        ax.annotate(text, (X(p), Y(p)), xytext=(off[0] / px_per_pt, off[1] / px_per_pt), textcoords="offset points",
                    ha=ha, va=va, arrowprops=arrow, annotation_clip=False, **style)

    # ----- header -----
    title, subtitle = TRACK_TEXT[track]
    tasks = max(p.tasks_total for p in pts)
    repeats = max(p.repeats for p in pts)
    fig.text(0.062, 0.935, f"Segmentation on real photos: {title.lower()}", fontproperties=fonts.bold,
             fontsize=25, color=INK, ha="left", va="baseline")
    fig.text(0.062, 0.895, subtitle, fontproperties=fonts.regular, fontsize=13.5, color=INK_2, ha="left",
             va="baseline")
    counts = f"{tasks} task{'s' * (tasks != 1)}, {repeats} run{'s' * (repeats != 1)} each"
    fig.text(0.062, 0.865, f"{counts}. Quality is the mean task score; cost and time are per image.",
             fontproperties=fonts.regular, fontsize=13.5, color=INK_2, ha="left",
             va="baseline")

    # ----- legend panel -----
    _legend(fig, fonts, pts)

    # ----- footer -----
    notes = ["Cost: tokens \u00d7 public API list price (2026-10-08), also for models we reach through a "
             "subscription. GPU models: RTX 3090 rental ($0.22/h) \u00f7 images per hour under load.",
             "Time: wall clock, one image at a time. Thin vertical lines: spread over runs. "
             "Dashed line: best value (no model is both cheaper and better)."]
    if any(p.tasks_done < p.tasks_total for p in pts):
        notes.append("* Did not attempt every task (outside what the model can be asked): "
                     "missing tasks count as 0, and it cannot be on the best-value line.")
    if footnote:
        notes.append(footnote)
    for i, n in enumerate(notes):
        fig.text(0.062, 0.058 - i * 0.021, n, fontproperties=fonts.regular, fontsize=10.5, color=INK_3,
                 ha="left", va="baseline")
    fig.text(0.985, 0.058, "segbench", fontproperties=fonts.bold, fontsize=13, color=INK, ha="right",
             va="baseline")
    fig.text(0.985, 0.037, f"github.com/lillyguisnet/segbench \u00b7 {stamp or date.today().isoformat()}",
             fontproperties=fonts.regular, fontsize=10.5, color=INK_3, ha="right", va="baseline")

    if synthetic:
        ax.text(0.5, 0.5, "FAKE DATA", transform=ax.transAxes, ha="center", va="center", rotation=22,
                fontproperties=fonts.bold, fontsize=120, color=FAKE_RED, alpha=0.07, zorder=0)
        fig.text(0.985, 0.935, "FAKE DATA \u2014 NOT RESULTS", fontproperties=fonts.bold, fontsize=13,
                 color=FAKE_RED, ha="right", va="baseline")

    out_base.parent.mkdir(parents=True, exist_ok=True)
    paths = [out_base.with_suffix(s) for s in (".png", ".pdf", ".svg")]
    fig.savefig(paths[0], dpi=dpi, facecolor=PAPER, metadata={"Software": None})
    fig.savefig(paths[1], facecolor=PAPER, metadata={"CreationDate": None, "Producer": None})
    fig.savefig(paths[2], facecolor=PAPER, metadata={"Date": None})
    plt.close(fig)
    return paths


def _legend(fig, fonts: Fonts, pts: list[Point]) -> None:
    lx = 0.800  # left edge of the panel, in figure fractions
    y = 0.785
    head = dict(fontproperties=fonts.semibold, fontsize=12, color=INK, ha="left", va="center")
    body = dict(fontproperties=fonts.regular, fontsize=11.5, color=INK_2, ha="left", va="center")
    step = 0.030

    def dot(y_: float, colour, size: float, alpha: float = 0.8, ring: str | None = None, x_: float = lx + 0.010):
        if ring:
            fig.add_artist(Line2D([x_], [y_], marker="o", ms=size + 6, mfc="none", mec=ring, mew=2.6,
                                  transform=fig.transFigure))
        fig.add_artist(Line2D([x_], [y_], marker="o", ms=size, mfc=(*to_rgb(colour), alpha),
                              mec=_darker(colour), mew=1.0, transform=fig.transFigure))

    # makers, in a fixed order, only those on this chart
    fig.text(lx, y, "Made by", **head)
    y -= step * 1.05
    makers = [m for m in MAKER_COLOURS if any(p.maker == m and p.kind != "specialist" for p in pts)]
    for m in makers:
        dot(y, MAKER_COLOURS[m], 11)
        fig.text(lx + 0.024, y, m, **body)
        y -= step
    if any(p.kind == "specialist" for p in pts):
        dot(y, GPU_COLOUR, 11)
        fig.text(lx + 0.024, y, "Run on our own GPU (specialists)", **body)
        y -= step
    if any(p.kind == "pair" for p in pts):
        dot(y, MAKER_COLOURS["Google"], 11, ring=GPU_COLOUR)
        fig.text(lx + 0.024, y, "Pair: finder + GPU outliner", **body)
        y -= step

    # thinking levels
    if any(p.level for p in pts if p.kind == "general"):
        y -= step * 0.6
        fig.text(lx, y, "Thinking level", **head)
        y -= step * 1.05
        for i, lv in enumerate(LEVELS):
            dot(y, INK_2, 13, alpha=LEVEL_ALPHA[lv], x_=lx + 0.010 + i * 0.062)
            fig.text(lx + 0.022 + i * 0.062, y, lv, **body)
        y -= step * 0.9
        fig.text(lx, y, "One model's levels are joined by a line;", **{**body, "fontsize": 10.5, "color": INK_3})
        y -= step * 0.62
        fig.text(lx, y, "the name sits at its highest level.", **{**body, "fontsize": 10.5, "color": INK_3})
        y -= step

    # time
    y -= step * 0.4
    fig.text(lx, y, "Time per image", **head)
    y -= step * 0.62
    fig.text(lx, y, "median, bubble grows \u00d710 per step", **{**body, "fontsize": 10.5, "color": INK_3})
    y -= step * 1.5
    x = lx + 0.004
    for i, s in enumerate(SIZE_LEGEND):
        d = diameter(s)
        x += (d / 2 / 72 / 16) + (0.016 if i else 0)  # points -> inches -> fraction of the 16-inch width
        fig.add_artist(Line2D([x], [y], marker="o", ms=d, mfc="#eef0f3", mec="#9aa0a6", mew=1.0,
                              transform=fig.transFigure))
        fig.text(x, y - 0.036, _seconds(s), **{**body, "ha": "center", "fontsize": 10.5})
        x += d / 2 / 72 / 16


def _overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def _place(cx, cy, r, w, h, placed, circles, frame, px_per_pt):
    """Greedy label placement around a bubble: the nearest spot that covers no
    other label and no bubble; failing that, the least-covered spot. Returns
    (box, ha, va, offset in pixels, needs a leader line)."""
    gap = 3 * px_per_pt
    best = None
    for ring in range(6):
        extra = ring * 9 * px_per_pt
        d = r + gap + extra
        # (anchor offset, ha, va): right, left, above, below, then the diagonals
        c = d * 0.72
        spots = [((d, 0), "left", "center"), ((-d, 0), "right", "center"),
                 ((0, d), "center", "bottom"), ((0, -d), "center", "top"),
                 ((c, c), "left", "bottom"), ((-c, c), "right", "bottom"),
                 ((c, -c), "left", "top"), ((-c, -c), "right", "top")]
        for i, ((ox, oy), ha, va) in enumerate(spots):
            x0 = cx + ox - (w if ha == "right" else w / 2 if ha == "center" else 0)
            y0 = cy + oy - (h if va == "top" else h / 2 if va == "center" else 0)
            box = (x0, y0, x0 + w, y0 + h)
            cost = ring * 40 + i * 4
            cost += sum(_overlap(box, b) for b in placed) * 50
            for (bx, by, br) in circles:
                if (bx, by) == (cx, cy):
                    continue
                cost += _overlap(box, (bx - br * 0.85, by - br * 0.85, bx + br * 0.85, by + br * 0.85)) * 3
            out = (max(0, frame[0] - box[0]) + max(0, box[2] - frame[2])
                   + max(0, frame[1] - box[1]) + max(0, box[3] - frame[3]))
            cost += out * h * 200
            if best is None or cost < best[0]:
                best = (cost, box, ha, va, (ox, oy), ring > 0)
            if cost == ring * 40 + i * 4:  # nothing in the way: take it
                return best[1:]
    return best[1:]
