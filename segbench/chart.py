"""Draw one bubble chart per track: quality up, cost across, speed as colour.

Needs matplotlib and svgelements (not in the base install): run it through
scripts/draw_charts.py, which declares them.

The image carries no title, caption or footnotes: the post that shares it
says what it is. What the chart itself encodes:
- y: quality, 0 to 100 (the README's 0..1 score x 100). The spread over
  repeats is in points.csv, not drawn.
- x: US dollars per 1,000 images, log scale. API models and models on a
  rented GPU are drawn the same way: both can be bought by anyone.
- marker: a disc with the maker's logo in white, every disc the same size.
- disc colour: speed, median seconds per image, green (fast) through amber
  to red (slow), on a log scale from 1 s to 100 s, the same on every chart.
  The API models take 3 s to minutes, so that is where colour must tell
  them apart; GPU models (well under 1 s) are all full green.
- small white badge at the lower right: a pair (the big logo finds, the
  small one outlines).
- dashed line: the best-value (Pareto) frontier, straight segments from the
  cheapest model to the best one; its names are in bold.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402
from matplotlib.transforms import ScaledTranslation  # noqa: E402

from segbench.chart_data import Point, frontier  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = ROOT / "assets" / "fonts"
LOGO_DIR = ROOT / "assets" / "logos"

INK = "#16181d"
INK_2 = "#4a4f5a"
INK_3 = "#8a8f99"
GRID = "#ebecef"
PAPER = "#ffffff"
FAKE_RED = "#d62839"

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

# Speed: green (fast) -> amber -> red (slow), each deep enough to carry a
# white logo. The green is darker than the red, so red-green colour-blind
# readers still see a difference in lightness.
SPEED_CMAP = LinearSegmentedColormap.from_list("speed", ["#0f7a45", "#5a9a1e", "#d08a00", "#e0601c", "#cf2f2a"])
SPEED_DOMAIN = (1.0, 100.0)  # seconds; faster or slower is clipped to the ends
SPEED_TICKS = (1, 3, 10, 30, 100)

DISC_PT = 30.0  # diameter of every disc
LOGO_FRACTION = 0.58  # logo size inside the disc
BADGE_PT = 16.0  # the outliner's small disc on a pair

FIG_W, FIG_H = 10, 10  # inches; square, as feeds show images; 2160 x 2160 pixels at dpi 216


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


@cache
def logo_path(maker: str) -> MPath | None:
    """The maker's logo as a matplotlib path, centred, y up, or None."""
    import svgelements as se

    name = LOGOS.get(maker)
    if not name or not (LOGO_DIR / name).exists():
        return None
    svg = (LOGO_DIR / name).read_text()
    vb = [float(v) for v in re.search(r'viewBox="([^"]+)"', svg).group(1).split()]
    cx, cy = vb[0] + vb[2] / 2, vb[1] + vb[3] / 2
    verts, codes = [], []
    for d in re.findall(r'<path[^>]*\sd="([^"]+)"', svg):
        for s in se.Path(d):
            if isinstance(s, se.Move):
                verts.append(s.end)
                codes.append(MPath.MOVETO)
            elif isinstance(s, se.Close):
                verts.append(s.end)
                codes.append(MPath.CLOSEPOLY)
            elif isinstance(s, se.Line):
                verts.append(s.end)
                codes.append(MPath.LINETO)
            elif isinstance(s, se.QuadraticBezier):
                verts += [s.control, s.end]
                codes += [MPath.CURVE3] * 2
            elif isinstance(s, se.CubicBezier):
                verts += [s.control1, s.control2, s.end]
                codes += [MPath.CURVE4] * 3
            elif isinstance(s, se.Arc):
                for c in s.as_cubic_curves():
                    verts += [c.control1, c.control2, c.end]
                    codes += [MPath.CURVE4] * 3
    return MPath([(x - cx, cy - y) for x, y in verts], codes)


def speed_colour(seconds: float):
    lo, hi = (math.log10(v) for v in SPEED_DOMAIN)
    t = (math.log10(min(max(seconds, SPEED_DOMAIN[0]), SPEED_DOMAIN[1])) - lo) / (hi - lo)
    return SPEED_CMAP(t)


def _money(v: float) -> str:
    if v >= 1:
        return f"${v:,.0f}"
    digits = max(0, -math.floor(math.log10(v)))
    return f"${v:.{digits}f}"


def x_range(points: list[Point]) -> tuple[float, float]:
    """All costs plus a margin (a third of a decade each side), in $ per 1,000 images."""
    xs = [p.cost_usd * 1000 for p in points]
    return 10 ** (math.log10(min(xs)) - 0.35), 10 ** (math.log10(max(xs)) + 0.35)


def _badge_offset_pt(diameter: float = DISC_PT) -> float:
    return (diameter / 2) * 0.74  # along each axis: the badge sits on the rim, lower right


def _radius_pt(p: Point) -> float:
    r = DISC_PT / 2 + 1.5
    if p.kind == "pair":
        r = max(r, math.hypot(_badge_offset_pt(), _badge_offset_pt()) + BADGE_PT / 2)
    return r


def _disc(ax, x, y, maker: str, *, diameter: float, fill, logo_colour, fonts: Fonts, edge=PAPER,
          edge_width: float = 1.8, shadow: bool = True, transform=None, zorder: float = 4) -> None:
    """One solid disc with a logo (or initials) at (x, y) in `transform` (data by default).

    A white edge keeps touching discs apart; a soft shadow lifts the disc
    off the grid."""
    t = transform if transform is not None else ax.transData
    if shadow:
        drop = t + ScaledTranslation(0, -1.6 / 72, ax.figure.dpi_scale_trans)
        ax.scatter([x], [y], s=(diameter + 2.5) ** 2, marker="o", facecolors=[(0, 0, 0, 0.13)], edgecolors="none",
                   transform=drop, zorder=zorder - 0.004)
    ax.scatter([x], [y], s=diameter ** 2, marker="o", facecolors=[fill], edgecolors=[edge], linewidths=edge_width,
               transform=t, zorder=zorder)
    logo = logo_path(maker)
    size = diameter * LOGO_FRACTION
    if logo is not None:
        ax.scatter([x], [y], s=size ** 2, marker=logo, facecolors=[logo_colour], edgecolors="none", linewidths=0,
                   transform=t, zorder=zorder + 0.001)
    else:
        text = INITIALS.get(maker) or "".join(w[0] for w in maker.split())[:3]
        ax.text(x, y, text, ha="center", va="center_baseline", fontproperties=fonts.bold,
                fontsize=size * (0.62 if len(text) < 3 else 0.46), color=logo_colour, transform=t,
                zorder=zorder + 0.001)


def _pair_badge(ax, x, y, maker: str, fonts: Fonts, *, diameter: float = DISC_PT, transform=None,
                zorder: float = 4) -> None:
    o = _badge_offset_pt(diameter)
    t = (transform if transform is not None else ax.transData) + ScaledTranslation(o / 72, -o / 72,
                                                                                 ax.figure.dpi_scale_trans)
    _disc(ax, x, y, maker, diameter=BADGE_PT * diameter / DISC_PT, fill=PAPER, logo_colour=INK, edge=INK_3,
          edge_width=1.0, shadow=False, fonts=fonts, transform=t, zorder=zorder + 0.002)


def draw(points: list[Point], track: str, out_base: Path, *, xlim: tuple[float, float] | None = None,
         legend: bool = True, names: str = "all", dpi: int = 216) -> list[Path]:
    """Draw the points of one track; return the files written.

    legend=False drops the row above the chart; names="frontier" names only
    the best-value models (the logos say who made the others).

    out_base.png  to post (2160 x 2160 pixels at dpi=216)
    out_base.pdf  to print: vector, fonts embedded
    out_base.svg  for the web: vector, text drawn as shapes
    """
    pts = [p for p in points if p.track == track]
    if not pts:
        raise ValueError(f"no points for track {track!r}")
    fonts = _fonts()

    plt.rcParams.update({"svg.fonttype": "path", "pdf.fonttype": 42, "font.family": fonts.regular.get_name()})
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=PAPER)
    ax = fig.add_axes((0.105, 0.085, 0.85, 0.80 if legend else 0.885))
    ax.set_facecolor(PAPER)

    # ----- axes -----
    xlim = xlim or x_range(pts)
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    decades = [10.0 ** k for k in range(math.ceil(math.log10(xlim[0])), math.floor(math.log10(xlim[1])) + 1)]
    ax.xaxis.set_major_locator(FixedLocator(decades))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([_money(v) for v in decades])

    qs = [100 * p.quality for p in pts]
    y0 = max(0, 10 * math.floor((min(qs) - 5) / 10))
    y1 = min(100, 10 * math.ceil((max(qs) + 5) / 10))
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
        lab.set_fontsize(14)
    ax.set_xlabel("Cost per 1,000 images", fontproperties=fonts.semibold, fontsize=15.5, color=INK_2, labelpad=10)
    ax.set_ylabel("Quality", fontproperties=fonts.semibold, fontsize=15.5, color=INK_2, labelpad=10)

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
    ax.plot(*zip(*line), color=INK, lw=1.6, ls=(0, (5, 4)), alpha=0.55, zorder=1)

    # ----- discs: slowest first, so fast ones (often in crowded spots) stay on top -----
    for i, p in enumerate(sorted(pts, key=lambda p: -p.seconds)):
        z = 4 + i * 0.01
        _disc(ax, X(p), Y(p), p.maker, diameter=DISC_PT, fill=speed_colour(p.seconds), logo_colour=PAPER,
              fonts=fonts, zorder=z)
        if p.kind == "pair":
            _pair_badge(ax, X(p), Y(p), p.outliner_maker, fonts, zorder=z)

    # ----- labels -----
    several_levels = len({p.level for p in pts if p.level}) > 1
    labelled = [(p, p.label + (f" \u00b7 {p.level}" if p.level and several_levels else "")) for p in pts
                if names == "all" or p.entrant in best_ids]

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
        n = max(1, int(math.hypot(x1 - x0, y1_ - y0_) / 8))
        for k in range(n + 1):
            x, y = x0 + (x1 - x0) * k / n, y0_ + (y1_ - y0_) * k / n
            placed.append((x - 4, y - 4, x + 4, y + 4))
    for p, text in sorted(labelled, key=lambda it: (it[0].entrant not in best_ids, -it[0].quality)):
        strong = p.entrant in best_ids
        style = dict(fontproperties=fonts.bold if strong else fonts.medium, fontsize=13.5,
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

    if legend:
        _legend(fig, fonts, pts)

    # Made-up numbers must never pass for results: a large faint stamp.
    if any(p.synthetic for p in pts):
        ax.text(0.5, 0.5, "FAKE DATA", transform=ax.transAxes, ha="center", va="center", rotation=35,
                fontproperties=fonts.bold, fontsize=110, color=FAKE_RED, alpha=0.07, zorder=0)

    out_base.parent.mkdir(parents=True, exist_ok=True)
    paths = [out_base.with_suffix(s) for s in (".png", ".pdf", ".svg")]
    fig.savefig(paths[0], dpi=dpi, facecolor=PAPER, metadata={"Software": None})
    fig.savefig(paths[1], facecolor=PAPER, metadata={"CreationDate": None, "Producer": None})
    fig.savefig(paths[2], facecolor=PAPER, metadata={"Date": None})
    plt.close(fig)
    return paths


def _legend(fig, fonts: Fonts, pts: list[Point]) -> None:
    """One row above the chart: the speed colours, and the pair badge if any."""
    y = 0.945
    body = dict(fontproperties=fonts.medium, fontsize=13, color=INK_2, va="center")

    x = 0.105
    fig.text(x, y, "Speed", **{**body, "fontproperties": fonts.semibold, "color": INK, "ha": "left"})
    x += 0.075
    fig.text(x, y, "fast", **{**body, "ha": "left", "color": INK_3})
    x += 0.047
    width = 0.24
    bar = fig.add_axes((x, y - 0.009, width, 0.018))
    lo, hi = (math.log10(v) for v in SPEED_DOMAIN)
    bar.imshow([[k / 199 for k in range(200)]], cmap=SPEED_CMAP, aspect="auto", extent=(lo, hi, 0, 1))
    bar.set_xlim(lo, hi)
    bar.set_xticks([math.log10(t) for t in SPEED_TICKS])
    labels = [f"{t:g}" for t in SPEED_TICKS]
    labels[0], labels[-1] = f"\u2264{labels[0]}", f"\u2265{labels[-1]} s"
    bar.set_xticklabels(labels)
    bar.set_yticks([])
    for sp in bar.spines.values():
        sp.set_visible(False)
    bar.tick_params(length=0, pad=4, labelcolor=INK_3)
    for lab in bar.get_xticklabels():
        lab.set_fontproperties(fonts.regular)
        lab.set_fontsize(11)
    fig.text(x + width + 0.012, y, "slow", **{**body, "ha": "left", "color": INK_3})

    pair = next((p for p in pts if p.kind == "pair"), None)
    if pair:
        hold = fig.add_axes((0.70, y - 0.03, 0.06, 0.06))
        hold.set_xlim(-1, 1)
        hold.set_ylim(-1, 1)
        hold.axis("off")
        d = DISC_PT * 0.8
        _disc(hold, 0, 0, pair.maker, diameter=d, fill=INK_3, logo_colour=PAPER, shadow=False, fonts=fonts)
        _pair_badge(hold, 0, 0, pair.outliner_maker, fonts, diameter=d)
        fig.text(0.765, y, "Pair: finder + outliner", **{**body, "ha": "left"})


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
