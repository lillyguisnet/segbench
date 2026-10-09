"""The charts in an editorial style: a one-line headline (the ability measured), the best-value models
joined by a smooth blue curve over a soft fill, every model a white disc
with its logo in brand colour, and every model named, crowded ones in
neat columns with thin leader lines.

Same data, logos and fonts as segbench/chart.py; run it through
scripts/draw_charts.py --style editorial.

What it shows:
- y: quality, 0 to 100; x: US dollars per 1,000 images, log scale.
- big disc with a blue ring, name and score in blue: a best-value
  (Pareto) model. The curve between them is a guide for the eye, not
  more models: it is monotone (it never dips or overshoots between two
  models), computed in log-cost space. It runs on flat to the right edge
  (paying more never buys less), and the glow under it fades out left of
  the cheapest model (nothing is cheaper).
- small disc: any other model.
- under each name: the maker and the median seconds per image (speed is
  text in this style, not colour).
- a small badge at the lower right of a disc: a pair (the big logo finds,
  the badge outlines).
"""

from __future__ import annotations

import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402
from matplotlib.transforms import ScaledTranslation  # noqa: E402

from segbench.chart import INITIALS, Fonts, _fonts, _money, logo_path, x_range  # noqa: E402
from segbench.chart_data import Point, frontier  # noqa: E402

INK = "#0f1115"
INK_2 = "#4b5260"
INK_3 = "#8b93a1"
GRID = "#e4e8ef"
RIM = "#cfd5df"
ACCENT = "#1d5fe0"
FAKE_RED = "#d62839"

# Logo colours on white discs: each brand's own colour (nothing is encoded
# by them here, so two black logos are fine).
BRAND = {
    "OpenAI": "#0f1115",
    "Anthropic": "#d97757",
    "Google": "#3c7cf0",
    "DeepSeek": "#4d6bfe",
    "Alibaba": "#615ced",
    "Moonshot": "#0f1115",
    "Z.AI": "#0f1115",
    "Meta": "#0866ff",
    "Ultralytics": "#042aff",
    "Roboflow": "#6706ce",
    "UC Davis": "#022851",
}
MAKER_NAME = {"Moonshot": "Moonshot AI", "Z.AI": "Z.ai"}
# The headline of each chart: what ability it measures, in plain words.
TITLES = {
    "find": "How well can AI point at things in a photo?",
    "outline": "How well can AI outline things in a photo?",
    "whole": "How well can AI find and outline things in a photo?",
}

BIG_PT = 31.0  # best-value discs
SMALL_PT = 20.0  # every other disc
FIG = 10  # inches, square


def _secs(s: float) -> str:
    if s < 1:
        return f"{s:.2g} s"
    return f"{s:.1f} s" if s < 10 else f"{s:.0f} s"


def _pchip(xs: list[float], ys: list[float], n: int = 48) -> tuple[list[float], list[float]]:
    """Monotone cubic through the points (Fritsch-Carlson): no dips, no overshoot."""
    k = len(xs)
    if k < 2:
        return list(xs), list(ys)
    h = [xs[i + 1] - xs[i] for i in range(k - 1)]
    d = [(ys[i + 1] - ys[i]) / h[i] for i in range(k - 1)]
    m = [d[0]] + [0.0] * (k - 2) + [d[-1]]
    for i in range(1, k - 1):
        if d[i - 1] * d[i] > 0:
            w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    ox, oy = [], []
    for i in range(k - 1):
        for j in range(n):
            t = j / n
            h00, h10 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t
            h01, h11 = -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
            ox.append(xs[i] + t * h[i])
            oy.append(h00 * ys[i] + h10 * h[i] * m[i] + h01 * ys[i + 1] + h11 * h[i] * m[i + 1])
    return ox + [xs[-1]], oy + [ys[-1]]


def _disc(ax, x, y, maker: str, d: float, *, rim, rim_w: float, fonts: Fonts, z: float,
          transform=None, shadow: bool = False) -> None:
    t = transform if transform is not None else ax.transData
    if shadow:
        drop = t + ScaledTranslation(0, -2 / 72, ax.figure.dpi_scale_trans)
        ax.scatter([x], [y], s=(d + 4) ** 2, facecolors=[(*to_rgb(rim), 0.16)], edgecolors="none",
                   transform=drop, zorder=z - 0.004)
    ax.scatter([x], [y], s=d ** 2, facecolors="white", edgecolors=[rim], linewidths=rim_w, transform=t, zorder=z)
    logo = logo_path(maker)
    size = d * 0.56
    colour = BRAND.get(maker, INK)
    if logo is not None:
        ax.scatter([x], [y], s=size ** 2, marker=logo, facecolors=[colour], edgecolors="none", linewidths=0,
                   transform=t, zorder=z + 0.001)
    else:
        text = INITIALS.get(maker) or "".join(w[0] for w in maker.split())[:3]
        ax.text(x, y, text, ha="center", va="center_baseline", fontproperties=fonts.bold,
                fontsize=size * (0.6 if len(text) < 3 else 0.44), color=colour, transform=t, zorder=z + 0.001)


def _badge(ax, x, y, maker: str, d: float, fonts: Fonts, z: float) -> None:
    o = d / 2 * 0.74
    t = ax.transData + ScaledTranslation(o / 72, -o / 72, ax.figure.dpi_scale_trans)
    _disc(ax, x, y, maker, d * 0.56, rim=RIM, rim_w=0.9, fonts=fonts, z=z + 0.002, transform=t)


def _radius_pt(p: Point, d: float) -> float:
    r = d / 2 + 1.5
    if p.kind == "pair":
        o = d / 2 * 0.74
        r = max(r, math.hypot(o, o) + d * 0.28)
    return r


# ---------- label layout, in pixels ----------

def _overlap(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def _box(cx, cy, off, ha, va, w, h):
    x0 = cx + off[0] - (w if ha == "right" else w / 2 if ha == "center" else 0)
    y0 = cy + off[1] - (h if va == "top" else h / 2 if va == "center" else 0)
    return (x0, y0, x0 + w, y0 + h)


def _penalty(box, placed, circles, own, frame) -> float:
    cost = sum(_overlap(box, b) for b in placed) * 50
    for i, (bx, by, br) in enumerate(circles):
        if i != own:
            cost += _overlap(box, (bx - br, by - br, bx + br, by + br)) * 12
    out = (max(0, frame[0] - box[0]) + max(0, box[2] - frame[2])
           + max(0, frame[1] - box[1]) + max(0, box[3] - frame[3]))
    return cost + out * 5000


SPOTS = {"right": ((1, 0), "left", "center"), "left": ((-1, 0), "right", "center"),
         "above": ((0, 1), "center", "bottom"), "below": ((0, -1), "center", "top"),
         "ur": ((0.72, 0.72), "left", "bottom"), "ul": ((-0.72, 0.72), "right", "bottom"),
         "lr": ((0.72, -0.72), "left", "top"), "ll": ((-0.72, -0.72), "right", "top")}


def _place(cx, cy, r, w, h, prefer, rings, step, placed, circles, own, frame, gap):
    """Nearest free spot around a disc. Returns (box, align, needs a leader line, free)."""
    best = None
    for ring in range(rings):
        dist = r + gap + ring * step
        for i, key in enumerate(prefer):
            (ux, uy), ha, va = SPOTS[key]
            box = _box(cx, cy, (ux * dist, uy * dist), ha, va, w, h)
            pen = _penalty(box, placed, circles, own, frame)
            cost = ring * 40 + i * 4 + pen
            if best is None or cost < best[0]:
                best = (cost, box, ha, ring > 0, pen == 0)
            if pen == 0:
                return best[1:]
    return best[1:]


def draw(points: list[Point], track: str, out_base: Path, *, xlim: tuple[float, float] | None = None,
         header: bool = True, title: str | None = None, accent: str = ACCENT, names: str = "all",
         dpi: int = 216) -> list[Path]:
    """Draw the points of one track; return the files written (PNG 2160 x 2160, PDF, SVG).

    `title` replaces the track's headline (TITLES); `accent` is the colour of
    the best-value curve, rings, scores, glow and the background's tint."""
    pts = [p for p in points if p.track == track]
    if not pts:
        raise ValueError(f"no points for track {track!r}")
    fonts = _fonts()
    plt.rcParams.update({"svg.fonttype": "path", "pdf.fonttype": 42, "font.family": fonts.regular.get_name()})
    fig = plt.figure(figsize=(FIG, FIG))

    # Background: white, washing to a faint blue at the lower right.
    bg = fig.add_axes((0, 0, 1, 1), zorder=-10)
    yy, xx = np.mgrid[0:1:200j, 0:1:200j]
    t = np.clip((xx + (1 - yy)) / 2, 0, 1)[..., None]
    top = np.array(to_rgb("#ffffff"))
    bottom = top * 0.91 + np.array(to_rgb(accent)) * 0.09  # a faint wash of the accent
    bg.imshow(top * (1 - t) + bottom * t, extent=(0, 1, 0, 1), aspect="auto", origin="upper")
    bg.axis("off")

    ax_top = 0.855 if header else 0.93
    ax = fig.add_axes((0.065, 0.095, 0.905, ax_top - 0.095))
    ax.set_facecolor("none")

    xlim = xlim or x_range(pts)
    xlim = (xlim[0], xlim[1] * 10 ** 0.35)  # room on the right for labels
    qs = [100 * p.quality for p in pts]
    y0 = max(0, 10 * math.floor((min(qs) - 5) / 10))
    y1 = min(100, 10 * math.ceil((max(qs) + 6) / 10))

    def style_axes():
        ax.set_xscale("log")
        ax.set_xlim(*xlim)
        ax.set_ylim(y0, y1)
        decades = [10.0 ** k for k in range(math.ceil(math.log10(xlim[0])), math.floor(math.log10(xlim[1])) + 1)]
        ax.xaxis.set_major_locator(FixedLocator(decades))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([_money(v) for v in decades])
        ax.set_yticks(range(int(y0), int(y1) + 1, 10))

    style_axes()
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(True, axis="y", color=GRID, lw=1.0, zorder=0)
    ax.tick_params(length=0, pad=7, labelcolor=INK_3)
    for lab in ax.get_xticklabels():
        lab.set_fontproperties(fonts.semibold)
        lab.set_fontsize(12)
    for lab in ax.get_yticklabels():
        lab.set_fontproperties(fonts.medium)
        lab.set_fontsize(12)
    ax.set_xlabel("Cost ($ per 1,000 images)", fontproperties=fonts.semibold, fontsize=13, color=INK_2,
                  labelpad=12)

    def X(p: Point) -> float:
        return p.cost_usd * 1000

    def Y(p: Point) -> float:
        return p.quality * 100

    best = frontier([p for p in pts if p.tasks_done == p.tasks_total])
    best_ids = {p.entrant for p in best}
    others = [p for p in pts if p.entrant not in best_ids]

    # ----- the curve and its soft fill, across the whole cost axis -----
    # Right of the best model the curve runs on flat: paying more never buys
    # less, since the best model can still be bought. Left of the cheapest
    # model nothing exists, so there the glow fades out instead of claiming
    # a quality that no model reaches.
    lx, ly = _pchip([math.log10(X(p)) for p in best], [Y(p) for p in best])
    cx_, cy_ = [10 ** v for v in lx], ly
    curve = list(zip(cx_, cy_)) + [(xlim[1], cy_[-1])]
    poly = MPath([(xlim[0], cy_[0])] + curve + [(xlim[1], y0), (xlim[0], y0), (xlim[0], cy_[0])])
    clip = PathPatch(poly, transform=ax.transData, facecolor="none", edgecolor="none")
    ax.add_patch(clip)
    # Vertical: strongest at the top of the curve, gone 25 points below its
    # lowest point. Horizontal: full from the cheapest model rightwards.
    f_top, f_bottom = max(cy_), min(cy_) - 25
    frac = [(v - y0) / (y1 - y0) for v in (f_bottom, f_top)]
    lo, hi = math.log10(xlim[0]), math.log10(xlim[1])
    x_first = (math.log10(cx_[0]) - lo) / (hi - lo)
    cols = np.linspace(0, 1, 512)
    across = np.clip(cols / max(x_first, 1e-6), 0, 1) ** 1.6
    down = np.linspace(0.20, 0.0, 256)
    rgba = np.zeros((256, 512, 4))
    rgba[..., :3] = to_rgb(accent)
    rgba[..., 3] = down[:, None] * across[None, :]
    img = ax.imshow(rgba, extent=(0, 1, frac[0], frac[1]), transform=ax.transAxes, aspect="auto",
                    origin="upper", zorder=1)
    img.set_clip_path(clip)
    ax.plot(cx_, cy_, color=accent, lw=3.2, zorder=3, solid_capstyle="round", solid_joinstyle="round")
    ax.plot([cx_[-1], xlim[1]], [cy_[-1], cy_[-1]], color=accent, lw=3.2, alpha=0.45, zorder=3,
            solid_capstyle="round")
    style_axes()  # imshow may have touched the limits

    # ----- discs -----
    for i, p in enumerate(sorted(others, key=lambda p: -p.seconds)):
        _disc(ax, X(p), Y(p), p.maker, SMALL_PT, rim=RIM, rim_w=1.0, fonts=fonts, z=4 + i * 0.01)
        if p.kind == "pair":
            _badge(ax, X(p), Y(p), p.outliner_maker, SMALL_PT, fonts, 4 + i * 0.01)
    for i, p in enumerate(best):
        _disc(ax, X(p), Y(p), p.maker, BIG_PT, rim=accent, rim_w=2.8, fonts=fonts, z=6 + i * 0.01, shadow=True)
        if p.kind == "pair":
            _badge(ax, X(p), Y(p), p.outliner_maker, BIG_PT, fonts, 6 + i * 0.01)

    _labels(fig, ax, fonts, pts, best_ids, names, X, Y, curve, accent)
    if header:
        _header(fig, fonts, title or TITLES[track])
    if any(p.synthetic for p in pts):
        ax.text(0.5, 0.42, "FAKE DATA", transform=ax.transAxes, ha="center", va="center", rotation=35,
                fontproperties=fonts.bold, fontsize=110, color=FAKE_RED, alpha=0.06, zorder=0.5)

    out_base.parent.mkdir(parents=True, exist_ok=True)
    paths = [out_base.with_suffix(s) for s in (".png", ".pdf", ".svg")]
    fig.savefig(paths[0], dpi=dpi, metadata={"Software": None})
    fig.savefig(paths[1], metadata={"CreationDate": None, "Producer": None})
    fig.savefig(paths[2], metadata={"Date": None})
    plt.close(fig)
    return paths


def _labels(fig, ax, fonts: Fonts, pts, best_ids, names, X, Y, curve, accent: str) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ppt = fig.dpi / 72
    inv = ax.transData.inverted()
    several_levels = len({p.level for p in pts if p.level}) > 1

    def props(font, size, colour):
        return {"fontproperties": font, "fontsize": size, "color": colour}

    def width(text, pr) -> float:
        t = ax.text(0, 0, text, **pr)
        w = t.get_window_extent(renderer).width
        t.remove()
        return w

    def block(p: Point):
        """Lines of (text, props) runs, with sizes."""
        name = p.label + (f" \u00b7 {p.level}" if p.level and several_levels else "")
        if p.entrant in best_ids:
            lines = [[(name, props(fonts.bold, 13.5, INK))],
                     [(f"{100 * p.quality:.1f}", props(fonts.bold, 13.5, accent)),
                      (f"  {_secs(p.seconds)}", props(fonts.medium, 11, INK_3))]]
        else:
            maker = MAKER_NAME.get(p.maker, p.maker)
            if p.kind == "pair":
                maker += " + " + MAKER_NAME.get(p.outliner_maker, p.outliner_maker)
            lines = [[(name, props(fonts.semibold, 11, INK))],
                     [(f"{maker} \u00b7 {_secs(p.seconds)}", props(fonts.regular, 9.5, INK_3))]]
        sized = []
        for line in lines:
            ws = [width(t, pr) for t, pr in line]
            sized.append((line, ws, sum(ws), max(pr["fontsize"] for _, pr in line) * 1.22 * ppt))
        return sized, max(s[2] for s in sized), sum(s[3] for s in sized) + 1 * ppt * (len(sized) - 1)

    def render(sized, box, align):
        x0, _, x1, ytop = box
        y = ytop
        for line, ws, lw, lh in sized:
            x = x0 if align == "left" else x1 - lw if align == "right" else (x0 + x1 - lw) / 2
            for (text, pr), w in zip(line, ws):
                ax.text(*inv.transform((x, y)), text, ha="left", va="top", zorder=8, **pr)
                x += w
            y -= lh + 1 * ppt

    def leader(c, box, align, first_h):
        cx, cy, r = c
        if align == "left":
            ex, ey = box[0] - 3, box[3] - first_h / 2
        elif align == "right":
            ex, ey = box[2] + 3, box[3] - first_h / 2
        else:
            ex, ey = (box[0] + box[2]) / 2, box[1] - 2 if box[1] > cy else box[3] + 2
        dx, dy = ex - cx, ey - cy
        n = math.hypot(dx, dy) or 1
        sx, sy = cx + dx / n * (r - 2), cy + dy / n * (r - 2)
        a, b = inv.transform((sx, sy)), inv.transform((ex, ey))
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#b5bcc8", lw=0.8, zorder=3.5, solid_capstyle="round")

    circles = []
    for p in pts:
        cx, cy = ax.transData.transform((X(p), Y(p)))
        circles.append((cx, cy, _radius_pt(p, BIG_PT if p.entrant in best_ids else SMALL_PT) * ppt))
    ab = ax.get_window_extent(renderer)
    frame = (ab.x0 + 2, ab.y0 + 6, fig.bbox.width - 10, ab.y1 - 2)  # labels may use the right margin
    placed = []
    pix = [ax.transData.transform(xy) for xy in curve]
    for (ax0, ay0), (ax1, ay1) in zip(pix, pix[1:]):
        n = max(1, int(math.hypot(ax1 - ax0, ay1 - ay0) / 8))
        for k in range(n + 1):
            px, py = ax0 + (ax1 - ax0) * k / n, ay0 + (ay1 - ay0) * k / n
            placed.append((px - 5, py - 5, px + 5, py + 5))
    gap = 4 * ppt

    # 1. Best-value models: name and score above the disc, if there is room.
    index = {p.entrant: i for i, p in enumerate(pts)}
    for p in sorted((p for p in pts if p.entrant in best_ids), key=lambda p: -p.quality):
        i = index[p.entrant]
        sized, w, h = block(p)
        box, align, far, _ = _place(*circles[i], w, h, ["above", "ul", "ur", "left", "right", "below"], 6,
                                    10 * ppt, placed, circles, i, frame, gap)
        placed.append(box)
        render(sized, box, align)
        if far:
            leader(circles[i], box, align, sized[0][3])
    if names != "all":
        return

    # 2. Every other model: right next to its disc when that spot is free...
    others = sorted((p for p in pts if p.entrant not in best_ids), key=lambda p: -p.quality)
    blocks = {p.entrant: block(p) for p in others}
    crowded = []
    for p in others:
        i = index[p.entrant]
        sized, w, h = blocks[p.entrant]
        box, align, _, free = _place(*circles[i], w, h, ["right", "left"], 1, 0, placed, circles, i, frame, gap)
        if free:
            placed.append(box)
            render(sized, box, align)
        else:
            crowded.append(p)

    # 3. ...else in a column beside its cluster, in height order, with leader lines.
    near = 90 * ppt  # one cluster, one column
    groups: list[list[Point]] = []
    for p in crowded:
        c = circles[index[p.entrant]]
        hit = [g for g in groups if any(math.hypot(c[0] - circles[index[q.entrant]][0],
                                                   c[1] - circles[index[q.entrant]][1]) < near for q in g)]
        merged = [p] + [q for g in hit for q in g]
        groups = [g for g in groups if g not in hit] + [merged]
    def column(sub, side, taken):
        """Labels of `sub` stacked beside it on `side`, slid to the freest height."""
        sub = sorted(sub, key=lambda p: -circles[index[p.entrant]][1])
        cs = [circles[index[p.entrant]] for p in sub]
        tops, sizes = [], []
        for p, c in zip(sub, cs):
            sized, w, h = blocks[p.entrant]
            want = c[1] + sized[0][3] / 2
            if tops:
                want = min(want, tops[-1] - sizes[-1][1] - 3 * ppt)
            tops.append(want)
            sizes.append((w, h))
        shift = statistics.fmean(c[1] + blocks[p.entrant][0][0][3] / 2 - t for p, c, t in zip(sub, cs, tops))
        tops = [t + shift for t in tops]
        if side == "right":
            xc = max(c[0] + c[2] for c in cs) + 10 * ppt
            bxs = [(xc, t - h, xc + w, t) for t, (w, h) in zip(tops, sizes)]
        else:
            xc = min(c[0] - c[2] for c in cs) - 10 * ppt
            bxs = [(xc - w, t - h, xc, t) for t, (w, h) in zip(tops, sizes)]
        best_col = None
        for k in range(61):
            dy = (k + 1) // 2 * 5 * ppt * (1 if k % 2 else -1)
            moved = [(b[0], b[1] + dy, b[2], b[3] + dy) for b in bxs]
            pen = sum(_penalty(b, taken, circles, -1, frame) for b in moved) + abs(dy) * 3
            # long leader lines cross the chart: they cost too
            pen += 3 * sum(math.hypot((b[0] if side == "right" else b[2]) - c[0], b[3] - c[1])
                           for b, c in zip(moved, cs))
            if best_col is None or pen < best_col[0]:
                best_col = (pen, moved)
        return best_col[0], list(zip(sub, cs, best_col[1])), "left" if side == "right" else "right"

    for g in groups:
        options = [[(g, "right")], [(g, "left")]]
        if len(g) >= 3:  # split by cost: cheaper half to the left, dearer half to the right
            by_x = sorted(g, key=lambda p: circles[index[p.entrant]][0])
            half = len(g) // 2
            options.append([(by_x[:half], "left"), (by_x[half:], "right")])
        best_opt = None
        for opt in options:
            taken, total, parts = list(placed), 0.0, []
            for sub, side in opt:
                pen, rows, align = column(sub, side, taken)
                total += pen
                taken += [b for _, _, b in rows]
                parts.append((rows, align))
            if best_opt is None or total < best_opt[0]:
                best_opt = (total, parts)
        for rows, align in best_opt[1]:
            for p, c, b in rows:
                sized = blocks[p.entrant][0]
                placed.append(b)
                render(sized, b, align)
                leader(c, b, align, sized[0][3])


def _header(fig, fonts: Fonts, title: str) -> None:
    """The headline, at 30 pt or smaller if that is what it takes to fit the width."""
    t = fig.text(0.062, 0.915, title, fontproperties=fonts.bold, fontsize=30, color=INK, va="baseline")
    room = (0.97 - 0.062) * fig.bbox.width
    width = t.get_window_extent(fig.canvas.get_renderer()).width
    if width > room:
        t.set_fontsize(30 * room / width)
