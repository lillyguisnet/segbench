# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9", "svgelements>=1.9", "pillow>=10"]
# ///
"""Every model's points on its photo, each point drawn as the model's logo.

    uv run scripts/draw_points.py                                   # pilot-1 + specialists, find track
    uv run scripts/draw_points.py results/run-pilot-1.jsonl --track trick
    uv run scripts/draw_points.py --models sonnet,opus,sam3         # only these

Writes results/views/points-<track>/<task>.jpg (one image per photo; for a
task with labels, one image per label: dishes-dirty.jpg, dishes-clean.jpg).
Nothing here is a score: it shows where each model put its dots, not
whether they are right.

How a point is drawn:
- a disc centred exactly on the point, with the maker's logo in white
  (the same logos as the charts: segbench/chart.py, assets/logos);
- disc colour: the maker's brand colour. Models from the same maker get
  darker and lighter shades of it, darkest first in the legend. Two
  makers may share a colour (Meta and Google are both blue): the logo
  already tells makers apart, the colour only has to tell one maker's
  models apart;
- the view is zoomed to where the points are (all of them, with a margin)
  when that leaves out a good part of the photo: the cows are far away;
  --no-zoom shows the whole photo;
- the points of all models are drawn in a shuffled order (fixed seed), so
  no model is always on top and none is always buried.

Replies are read with today's reader (segbench/parse.py), like the answer
viewer. A reply that cannot be read is listed in the legend as
"unreadable", never guessed at. Points are drawn as written: no [y, x]
swapping (README, open decision 2).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # this script runs in its own environment (above)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_hex, to_rgb  # noqa: E402
from PIL import Image  # noqa: E402

from segbench import parse  # noqa: E402
from segbench.chart import INK, INK_2, INK_3, PAPER, _disc, _fonts  # noqa: E402
from segbench.chart_data import describe  # noqa: E402
from segbench.models import LEVELS  # noqa: E402
from segbench.tasks import BY_KEY as TASKS  # noqa: E402

RUNS = ("results/run-pilot-1.jsonl", "results/run-specialists-1.jsonl")

# Brand colours (as on the 2026-10-09 brand-colour charts, commit 3aca946).
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
MAKER_ORDER = list(BRAND)

LONG_SIDE = 3000  # pixels of the photo in the output
DISC = 0.0115  # disc diameter, as a share of the photo's longest side
DPI = 100
ZOOM_MARGIN = 0.04  # around the points, as a share of the photo's longest side
ZOOM_MIN_ASPECT = 0.33  # a zoomed view is never thinner than this (short side / long side)


@dataclass
class Entrant:
    key: str  # model key, plus the level when one model ran at several
    label: str
    maker: str
    points: dict[str, list[tuple[float, float]]] = field(default_factory=lambda: defaultdict(list))
    unreadable: str = ""
    colour: str = OTHER


def shade(colour: str, t: float) -> str:
    """t < 0 mixes toward black, t > 0 toward white."""
    r, g, b = to_rgb(colour)
    target = 0.0 if t < 0 else 1.0
    a = abs(t)
    return to_hex(tuple(c + (target - c) * a for c in (r, g, b)))


def give_colours(entrants: list[Entrant]) -> None:
    by_maker = defaultdict(list)
    for e in entrants:
        by_maker[e.maker].append(e)
    for maker, group in by_maker.items():
        base = BRAND.get(maker, OTHER)
        k = len(group)
        for i, e in enumerate(group):
            # from 40 % darker to 28 % lighter: lighter still carries a white logo
            e.colour = base if k == 1 else shade(base, -0.40 + 0.68 * i / (k - 1))


def load(runs: list[Path], track: str, repeat: int, only: set[str] | None) -> dict[str, list[Entrant]]:
    """task -> entrants, in legend order."""
    rows = []
    for run in runs:
        for line in run.read_text().splitlines():
            r = json.loads(line)
            if r["track"] != track or r.get("repeat", 0) != repeat or "+" in r["model"]:
                continue
            if only and r["model"] not in only:
                continue
            rows.append(r)
    levels = defaultdict(set)
    for r in rows:
        levels[r["model"]].add(r.get("level"))

    tasks: dict[str, list[Entrant]] = defaultdict(list)
    for r in rows:
        task = TASKS[r["task"]]
        label, maker, _ = describe(r["model"])
        several = len(levels[r["model"]]) > 1
        e = Entrant(key=f'{r["model"]}@{r.get("level")}' if several else r["model"],
                    label=f'{label} ({r.get("level")})' if several else label, maker=maker)
        if "error" in r or not r.get("text"):
            e.unreadable = "no reply"
        else:
            try:
                answer = parse.read(r["track"], task.kind, r["text"], task.labels)
                for item in answer["items"]:
                    e.points[item.get("label", "")].append(tuple(item["point"]))
            except (ValueError, KeyError, TypeError):
                e.unreadable = "reply not readable"
        tasks[r["task"]].append(e)

    level_rank = {lv: i for i, lv in enumerate(LEVELS)}
    for entrants in tasks.values():
        entrants.sort(key=lambda e: (MAKER_ORDER.index(e.maker) if e.maker in MAKER_ORDER else 99, e.label,
                                     level_rank.get(e.key.partition("@")[2], 0)))
        give_colours(entrants)
    return tasks


def zoom_box(points, W0: int, H0: int) -> tuple[int, int, int, int]:
    """The part of the photo holding all the points (0..1000), with a margin.

    Never thinner than ZOOM_MIN_ASPECT, and the whole photo when zooming
    would keep most of it anyway."""
    inside = [(x * W0 / 1000, y * H0 / 1000) for x, y in points if 0 <= x <= 1000 and 0 <= y <= 1000]
    if not inside:
        return 0, 0, W0, H0
    margin = ZOOM_MARGIN * max(W0, H0)
    x0, x1 = min(x for x, _ in inside) - margin, max(x for x, _ in inside) + margin
    y0, y1 = min(y for _, y in inside) - margin, max(y for _, y in inside) + margin
    w, h = x1 - x0, y1 - y0
    w, h = max(w, h * ZOOM_MIN_ASPECT), max(h, w * ZOOM_MIN_ASPECT)
    w, h = min(w, W0), min(h, H0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0 = min(max(cx - w / 2, 0), W0 - w)
    y0 = min(max(cy - h / 2, 0), H0 - h)
    if w * h > 0.8 * W0 * H0:
        return 0, 0, W0, H0
    return round(x0), round(y0), round(x0 + w), round(y0 + h)


def draw(task_key: str, label: str, entrants: list[Entrant], out: Path, title: str, seed: int,
         zoom: bool = True) -> None:
    task = TASKS[task_key]
    with Image.open(ROOT / "images" / task.photo) as im:
        photo = im.convert("RGB")  # upright pixel grid as stored, the one the models were sent
    W0, H0 = photo.size
    dots = [(e, p) for e in entrants for p in e.points.get(label, [])]
    box = zoom_box([p for _, p in dots], W0, H0) if zoom else (0, 0, W0, H0)
    zoomed = box != (0, 0, W0, H0)
    photo = photo.crop(box)
    scale = LONG_SIDE / max(photo.size)
    W, H = round(photo.size[0] * scale), round(photo.size[1] * scale)
    photo = photo.resize((W, H), Image.LANCZOS)

    def to_px(x: float, y: float) -> tuple[float, float]:
        return (x * W0 / 1000 - box[0]) * scale, (y * H0 / 1000 - box[1]) * scale

    fonts = _fonts()
    disc_px = DISC * max(W, H)
    disc_pt = disc_px * 72 / DPI

    # Legend below the photo: one cell per entrant, as many columns as fit.
    cell_w, row_h, pad = disc_px * 14, disc_px * 1.9, disc_px * 1.2
    cols = max(1, int((W - 2 * pad) // cell_w))
    rows = -(-len(entrants) // cols)
    head_h = disc_px * 2.2
    legend_h = head_h + rows * row_h + pad
    fig = plt.figure(figsize=(W / DPI, (H + legend_h) / DPI), dpi=DPI, facecolor=PAPER)
    total_h = H + legend_h

    ax = fig.add_axes((0, legend_h / total_h, 1, H / total_h))
    ax.imshow(photo, extent=(0, W, H, 0), interpolation="none")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    random.Random(seed).shuffle(dots)
    off_photo = defaultdict(int)
    for z, (e, (x, y)) in enumerate(dots):
        if not (0 <= x <= 1000 and 0 <= y <= 1000):
            off_photo[e.key] += 1
            continue
        _disc(ax, *to_px(x, y), e.maker, diameter=disc_pt, fill=e.colour, logo_colour=PAPER,
              fonts=fonts, edge_width=max(1.0, disc_pt * 0.06), zorder=4 + z * 0.01)

    lg = fig.add_axes((0, 0, 1, legend_h / total_h))
    lg.set_xlim(0, W)
    lg.set_ylim(legend_h, 0)
    lg.axis("off")
    fs = disc_pt * 0.62
    if zoomed:
        title += "  ·  zoomed to the points"
    lg.text(pad, head_h * 0.55, title, ha="left", va="center", fontproperties=fonts.semibold, fontsize=fs * 1.1,
            color=INK)
    for i, e in enumerate(entrants):
        c, r = divmod(i, rows)  # fill column by column, so one maker's shades sit together
        x0, y0 = pad + c * cell_w + disc_px / 2, head_h + r * row_h + row_h / 2
        _disc(lg, x0, y0, e.maker, diameter=disc_pt, fill=e.colour, logo_colour=PAPER, fonts=fonts,
              edge_width=max(1.0, disc_pt * 0.06))
        n = len(e.points.get(label, []))
        note = e.unreadable or (f"{n}" + (f", {off_photo[e.key]} off photo" if off_photo[e.key] else ""))
        lg.text(x0 + disc_px * 0.9, y0, e.label, ha="left", va="center_baseline", fontproperties=fonts.medium,
                fontsize=fs, color=INK if not e.unreadable else INK_3)
        lg.text(x0 + cell_w - disc_px * 1.4, y0, note, ha="right", va="center_baseline",
                fontproperties=fonts.regular, fontsize=fs * 0.9, color=INK_2 if not e.unreadable else INK_3)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, facecolor=PAPER, pil_kwargs={"quality": 90})
    plt.close(fig)
    shown = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    print(f"wrote {shown}  ({len(dots)} points, {len(entrants)} models)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", default=list(RUNS), help="call records (.jsonl)")
    ap.add_argument("--track", default="find", choices=("find", "trick"))
    ap.add_argument("--repeat", type=int, default=0, help="which repeat to draw (default 0)")
    ap.add_argument("--models", help="comma-separated model keys to draw (default: all)")
    ap.add_argument("--out", type=Path, default=None, help="folder (default results/views/points-<track>)")
    ap.add_argument("--no-zoom", action="store_true", help="always show the whole photo")
    ap.add_argument("--seed", type=int, default=0, help="seed of the drawing order")
    args = ap.parse_args()

    runs = [Path(r) if Path(r).is_absolute() else ROOT / r for r in args.runs]
    only = set(args.models.split(",")) if args.models else None
    out = args.out or ROOT / "results" / "views" / f"points-{args.track}"
    tasks = load(runs, args.track, args.repeat, only)
    if not tasks:
        sys.exit("no point answers found for this track")
    for key in (t for t in TASKS if t in tasks):
        task = TASKS[key]
        labels = task.labels if args.track == "find" and task.labels else ("",)
        for label in labels:
            what = task.target.split(" (")[0] + (f": {label}" if label else "")
            title = f"{task.number}. {what}  ·  {args.track} track, {', '.join(r.stem for r in runs)}"
            name = f"{task.number}-{key}" + (f"-{label}" if label else "")
            draw(key, label, tasks[key], out / f"{name}.jpg", title, args.seed, zoom=not args.no_zoom)


if __name__ == "__main__":
    main()
