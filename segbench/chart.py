"""Draw one bubble chart per track: quality up, cost across, time as size.

Needs matplotlib (not in the base install): run it through
scripts/draw_charts.py, which declares it.

The image carries no title, caption or footnotes: the post that shares it
says what it is. What the chart itself encodes:
- y: quality, 0 to 100 (the README's 0..1 score x 100). The spread over
  repeats is in points.csv, not drawn: 40 error bars made the chart
  unreadable.
- x: US dollars per 1,000 images, log scale.
- size: speed, so that bigger is better like higher is. The fastest
  model is the biggest bubble; the diameter shrinks with the logarithm of
  the median seconds per image (0.01 s to 1,000 s, same scale on every
  chart), because times range from 0.02 s to minutes. The legend reads in
  seconds, the unit people know.
- colour: the maker's brand colour (exceptions where two brands clash,
  see BRAND below).
- shape: circle = called through an API, square = run on our own GPU,
  dark ring = a pair (one model finds, another outlines).
- shade: thinking level (light = min), only when the chart shows more than
  one level; with one level every bubble is full colour.
- dashed line: the best-value (Pareto) frontier, straight segments from the
  cheapest model to the best one, as Artificial Analysis draws it; its
  names are in bold.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

from segbench.chart_data import Point, frontier  # noqa: E402
from segbench.models import LEVELS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = ROOT / "assets" / "fonts"

INK = "#16181d"
INK_2 = "#4a4f5a"
INK_3 = "#8a8f99"
GRID = "#ebecef"
PAPER = "#ffffff"
FAKE_RED = "#d62839"

# Brand colours. Where two brands share a colour, the less known one is
# moved to a nearby shade or, failing that, given a free colour:
BRAND = {
    # called through an API
    "OpenAI": "#10a37f",     # the ChatGPT green (OpenAI's own mark is black, which Kimi uses)
    "Anthropic": "#d97757",  # Claude's terracotta
    "Google": "#4285f4",     # Google blue
    "DeepSeek": "#26339c",   # DeepSeek's whale blue, darkened to stay apart from Google's
    "Alibaba": "#615ced",    # Qwen violet
    "Moonshot": "#1c1c1e",   # Kimi black
    "Z.AI": "#e5484d",       # exception: Z.ai's black and white is taken; a free colour
    # run on our own GPU (squares, so Meta blue is not read as Google)
    "Meta": "#0866ff",       # Meta blue
    "Roboflow": "#a01ee6",   # Roboflow purple
    "Ultralytics": "#f0507a",  # exception: Ultralytics' blue is taken; a free colour
    "UC Davis": "#daaa00",   # UC Davis gold
}
OTHER = "#9aa0a6"
PAIR_RING = INK_2

LEVEL_ALPHA = {"min": 0.28, "medium": 0.55, "max": 0.88, "": 0.85}

SECONDS_DOMAIN = (0.01, 1000.0)  # same bubble scale on every chart
DIAMETER_PT = (9.0, 40.0)  # slowest, fastest
SIZE_LEGEND = (100, 10, 1, 0.1)  # seconds, small (slow) to big (fast)

FIG_W, FIG_H = 16, 9  # inches; 3200 x 1800 pixels at dpi 200


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


def colour_of(maker: str) -> str:
    return BRAND.get(maker, OTHER)


def _darker(colour: str, f: float = 0.72) -> tuple[float, float, float]:
    r, g, b = to_rgb(colour)
    return (r * f, g * f, b * f)


def diameter(seconds: float) -> float:
    lo, hi = (math.log10(v) for v in SECONDS_DOMAIN)
    t = (math.log10(min(max(seconds, SECONDS_DOMAIN[0]), SECONDS_DOMAIN[1])) - lo) / (hi - lo)
    return DIAMETER_PT[1] - t * (DIAMETER_PT[1] - DIAMETER_PT[0])  # faster = bigger


def _money(v: float) -> str:
    if v >= 1:
        return f"${v:,.0f}"
    digits = max(0, -math.floor(math.log10(v)))
    return f"${v:.{digits}f}"


def x_range(points: list[Point]) -> tuple[float, float]:
    """Whole decades around all costs, in $ per 1,000 images."""
    xs = [p.cost_usd * 1000 for p in points]
    return 10 ** math.floor(math.log10(min(xs)) - 0.15), 10 ** math.ceil(math.log10(max(xs)) + 0.15)


def _marker(p: Point) -> str:
    return "s" if p.kind == "specialist" else "o"


def _radius_pt(p: Point) -> float:
    return diameter(p.seconds) / 2 + (5 if p.kind == "pair" else 0)


def draw(points: list[Point], track: str, out_base: Path, *, xlim: tuple[float, float] | None = None,
         dpi: int = 200) -> list[Path]:
    """Draw the points of one track; return the files written.

    out_base.png  to post (3200 x 1800 pixels at dpi=200)
    out_base.pdf  to print: vector, fonts embedded
    out_base.svg  for the web: vector, text drawn as shapes
    """
    pts = [p for p in points if p.track == track]
    if not pts:
        raise ValueError(f"no points for track {track!r}")
    fonts = _fonts()

    plt.rcParams.update({"svg.fonttype": "path", "pdf.fonttype": 42, "font.family": fonts.regular.get_name()})
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=PAPER)
    ax = fig.add_axes((0.065, 0.105, 0.715, 0.865))
    ax.set_facecolor(PAPER)

    # ----- axes -----
    xlim = xlim or x_range(pts)
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    decades = [10.0 ** k for k in range(round(math.log10(xlim[0])), round(math.log10(xlim[1])) + 1)]
    ax.xaxis.set_major_locator(FixedLocator(decades))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([_money(v) for v in decades])

    qs = [100 * p.quality for p in pts]
    y0 = max(0, 10 * math.floor((min(qs) - 4) / 10))
    y1 = min(100, 10 * math.ceil((max(qs) + 4) / 10))
    ax.set_ylim(y0, y1)
    ax.set_yticks(range(int(y0), int(y1) + 1, 10))

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9ccd2")
    ax.grid(True, which="major", color=GRID, lw=1.0, zorder=0)
    ax.tick_params(which="both", length=0, pad=8, labelcolor=INK_2)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontproperties(fonts.regular)
        lab.set_fontsize(13)
    ax.set_xlabel("Cost per 1,000 images", fontproperties=fonts.medium, fontsize=14, color=INK_2, labelpad=10)
    ax.set_ylabel("Quality", fontproperties=fonts.medium, fontsize=14, color=INK_2, labelpad=10)

    def X(p: Point) -> float:
        return p.cost_usd * 1000

    def Y(p: Point) -> float:
        return p.quality * 100

    # ----- best-value frontier: straight segments joining the frontier models -----
    # Only entrants that attempted every task can be "best value": a model
    # that skipped tasks is cheap for a reason that the frontier would hide.
    best = frontier([p for p in pts if p.tasks_done == p.tasks_total])
    best_ids = {p.entrant for p in best}
    line = [(X(p), Y(p)) for p in best]
    ax.plot(*zip(*line), color=INK, lw=1.4, ls=(0, (5, 4)), alpha=0.5, zorder=1, solid_capstyle="round")

    # ----- one model's thinking levels (labelled once, at the highest) -----
    chains = defaultdict(list)
    for p in pts:
        if p.kind == "general":
            chains[p.model].append(p)
    rank = {lv: i for i, lv in enumerate(LEVELS)}
    for chain in chains.values():
        chain.sort(key=lambda p: rank.get(p.level, 0))
    several_levels = len({p.level for p in pts if p.level}) > 1

    # ----- bubbles, biggest first so small ones stay visible -----
    for p in sorted(pts, key=lambda p: -diameter(p.seconds)):
        c = colour_of(p.maker)
        d = diameter(p.seconds)
        face = (*to_rgb(c), LEVEL_ALPHA.get(p.level, 0.85) if several_levels else 0.85)
        if p.kind == "pair":
            ax.scatter([X(p)], [Y(p)], s=(d + 9) ** 2, facecolors="none", edgecolors=PAIR_RING, linewidths=2.0,
                       zorder=3)
        side = d * math.sqrt(math.pi / 4) if p.kind == "specialist" else d  # square of the circle's area
        ax.scatter([X(p)], [Y(p)], s=side ** 2, marker=_marker(p), facecolors=[face], edgecolors=[_darker(c)],
                   linewidths=1.1, zorder=4)

    # ----- labels: one per model (at its most-thinking bubble), one per specialist or pair -----
    labelled: list[tuple[Point, str]] = [(chain[-1], chain[-1].label) for chain in chains.values()]
    labelled += [(p, p.label + (f" ({p.level})" if p.level and several_levels else "")) for p in pts
                 if p.kind != "general"]

    def on_frontier(p: Point) -> bool:
        return p.entrant in best_ids or (p.kind == "general" and any(q.entrant in best_ids for q in chains[p.model]))

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px_per_pt = fig.dpi / 72
    circles = [(*ax.transData.transform((X(p), Y(p))), _radius_pt(p) * px_per_pt) for p in pts]
    ab = ax.get_window_extent(renderer)
    frame = (ab.x0 + 4, ab.y0 + 4, ab.x1 - 4, ab.y1 - 4)

    # Labels must not sit on the frontier line either: small boxes every few
    # pixels along each (slanted) segment are obstacles.
    placed: list[tuple[float, float, float, float]] = []
    seg = [tuple(ax.transData.transform(xy)) for xy in line]
    for (x0, y0_), (x1, y1_) in zip(seg, seg[1:]):
        n = max(1, int(math.hypot(x1 - x0, y1 - y0_) / 8))
        for k in range(n + 1):
            x, y = x0 + (x1 - x0) * k / n, y0_ + (y1_ - y0_) * k / n
            placed.append((x - 4, y - 4, x + 4, y + 4))
    for p, text in sorted(labelled, key=lambda it: (not on_frontier(it[0]), it[0].kind == "general",
                                                    -it[0].quality)):
        strong = on_frontier(p)
        style = dict(fontproperties=fonts.semibold if strong else fonts.regular, fontsize=12,
                     color=INK if strong else INK_2, zorder=6)
        probe = ax.text(0, 0, text, **style)
        bb = probe.get_window_extent(renderer)
        probe.remove()
        cx, cy = ax.transData.transform((X(p), Y(p)))
        r = _radius_pt(p) * px_per_pt
        box, ha, va, off, far = _place(cx, cy, r, bb.width + 4, bb.height + 2, placed, circles, frame, px_per_pt)
        placed.append(box)
        arrow = dict(arrowstyle="-", color=INK_3, lw=0.8, shrinkA=1, shrinkB=r / px_per_pt + 1) if far else None
        ax.annotate(text, (X(p), Y(p)), xytext=(off[0] / px_per_pt, off[1] / px_per_pt), textcoords="offset points",
                    ha=ha, va=va, arrowprops=arrow, annotation_clip=False, **style)

    _legend(fig, fonts, pts)

    # Made-up numbers must never pass for results: a large faint stamp.
    if any(p.synthetic for p in pts):
        ax.text(0.5, 0.5, "FAKE DATA", transform=ax.transAxes, ha="center", va="center", rotation=22,
                fontproperties=fonts.bold, fontsize=120, color=FAKE_RED, alpha=0.07, zorder=0)

    out_base.parent.mkdir(parents=True, exist_ok=True)
    paths = [out_base.with_suffix(s) for s in (".png", ".pdf", ".svg")]
    fig.savefig(paths[0], dpi=dpi, facecolor=PAPER, metadata={"Software": None})
    fig.savefig(paths[1], facecolor=PAPER, metadata={"CreationDate": None, "Producer": None})
    fig.savefig(paths[2], facecolor=PAPER, metadata={"Date": None})
    plt.close(fig)
    return paths


def _legend(fig, fonts: Fonts, pts: list[Point]) -> None:
    lx = 0.815  # left edge of the panel, in figure fractions
    y = 0.93
    head = dict(fontproperties=fonts.semibold, fontsize=12.5, color=INK, ha="left", va="center")
    body = dict(fontproperties=fonts.regular, fontsize=12, color=INK_2, ha="left", va="center")
    step = 0.036
    sx = lx + 0.009  # swatch centre
    tx = lx + 0.024  # text after a swatch

    def swatch(y_: float, colour, size: float = 11, marker: str = "o", alpha: float = 0.85, x_: float = sx,
               ring: bool = False):
        if ring:
            fig.add_artist(Line2D([x_], [y_], marker="o", ms=size + 7, mfc="none", mec=PAIR_RING, mew=1.8,
                                  transform=fig.transFigure))
        fig.add_artist(Line2D([x_], [y_], marker=marker, ms=size, mfc=(*to_rgb(colour), alpha),
                              mec=_darker(colour), mew=1.0, transform=fig.transFigure))

    def makers(kinds: set[str]) -> list[str]:
        seen = {p.maker for p in pts if p.kind in kinds} | {p.outliner_maker for p in pts if "pair" in kinds}
        return [m for m in BRAND if m in seen] + sorted(seen - set(BRAND) - {""})

    api = makers({"general", "pair"}) if any(p.kind in ("general", "pair") for p in pts) else []
    api = [m for m in api if any(p.maker == m and p.kind != "specialist" for p in pts)]
    gpu = [m for m in makers({"specialist", "pair"})
           if any((p.maker == m and p.kind == "specialist") or p.outliner_maker == m for p in pts)]

    if api:
        fig.text(lx, y, "Through an API", **head)
        y -= step
        for m in api:
            swatch(y, colour_of(m))
            fig.text(tx, y, m, **body)
            y -= step
        y -= step * 0.35
    if gpu:
        fig.text(lx, y, "On our own GPU", **head)
        y -= step
        for m in gpu:
            swatch(y, colour_of(m), marker="s", size=10)
            fig.text(tx, y, m, **body)
            y -= step
        y -= step * 0.35
    if any(p.kind == "pair" for p in pts):
        swatch(y, colour_of("Google"), ring=True)
        fig.text(tx, y, "Pair: finder + outliner", **body)
        y -= step * 1.35

    if len({p.level for p in pts if p.level}) > 1:
        fig.text(lx, y, "Thinking", **head)
        y -= step
        for lv, dx in zip(LEVELS, (0, 0.049, 0.122)):  # room for each word
            x = sx + dx
            swatch(y, INK_2, size=12, alpha=LEVEL_ALPHA[lv], x_=x)
            fig.text(x + 0.011, y, lv, **body)
        y -= step * 1.35

    fig.text(lx, y, "Speed (time per image)", **head)
    y -= step * 1.45
    x = lx
    for i, s in enumerate(SIZE_LEGEND):
        d = diameter(s)
        x += d / 2 / 72 / FIG_W + (0.014 if i else 0)  # points -> inches -> fraction of the figure width
        fig.add_artist(Line2D([x], [y], marker="o", ms=d, mfc="#eef0f3", mec="#9aa0a6", mew=1.0,
                              transform=fig.transFigure))
        fig.text(x, y - 0.045, f"{s:g} s", **{**body, "ha": "center", "fontsize": 11})
        x += d / 2 / 72 / FIG_W


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
        d = r + gap + ring * 9 * px_per_pt
        c = d * 0.72
        # (anchor offset, ha, va): right, left, above, below, then the diagonals
        spots = [((d, 0), "left", "center"), ((-d, 0), "right", "center"),
                 ((0, d), "center", "bottom"), ((0, -d), "center", "top"),
                 ((c, c), "left", "bottom"), ((-c, c), "right", "bottom"),
                 ((c, -c), "left", "top"), ((-c, -c), "right", "top")]
        for i, ((ox, oy), ha, va) in enumerate(spots):
            x0 = cx + ox - (w if ha == "right" else w / 2 if ha == "center" else 0)
            y0 = cy + oy - (h if va == "top" else h / 2 if va == "center" else 0)
            box = (x0, y0, x0 + w, y0 + h)
            base = ring * 40 + i * 4
            cost = base + sum(_overlap(box, b) for b in placed) * 50
            for (bx, by, br) in circles:
                if (bx, by) == (cx, cy):
                    continue
                cost += _overlap(box, (bx - br * 0.85, by - br * 0.85, bx + br * 0.85, by + br * 0.85)) * 3
            out = (max(0, frame[0] - box[0]) + max(0, box[2] - frame[2])
                   + max(0, frame[1] - box[1]) + max(0, box[3] - frame[3]))
            cost += out * h * 200
            if best is None or cost < best[0]:
                best = (cost, box, ha, va, (ox, oy), ring > 0)
            if cost == base:  # nothing in the way: take it
                return best[1:]
    return best[1:]
