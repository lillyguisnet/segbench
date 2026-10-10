# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow>=10", "numpy>=1.26", "matplotlib>=3.9", "svgelements>=1.9"]
# ///
"""A short square video of the point answers: same photo, same question, every model's dots.

    uv run scripts/make_video.py                    # results/views/video/segbench-points.mp4
    uv run scripts/make_video.py --preview          # a few still frames instead, to check the look

No title or end card: the video is the four photos, one after another.
For each photo: the question, then each model's dots popping in, one model
after another (its logo lights up below), then how far apart the counts
were. Nothing is scored: it shows what the models answered, not whether it
is right. Answers, colours and logos come from segbench/points.py and
segbench/brand.py, like the site and the overlays.

1080 x 1080, 30 frames a second, H.264 (plays everywhere). Frames are drawn
with Pillow and piped to ffmpeg (on PATH).
"""

from __future__ import annotations

import argparse
import io
import math
import random
import subprocess
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # this script runs in its own environment (above)

from segbench import points  # noqa: E402
from segbench.tasks import BY_KEY as TASKS  # noqa: E402

SIZE = 1080
FPS = 30
BG = (14, 15, 18)
WHITE = (255, 255, 255)
DIM = (150, 155, 165)
FAINT = (95, 100, 110)
FONT = ROOT / "assets" / "fonts"

PHOTO_BOX = (40, 190, 1040, 880)  # where the photo is fitted (x0, y0, x1, y1)
STRIP_Y = 945  # centre line of the row of logos

SCENE = 5.4  # seconds per photo
DOTS_FROM, DOTS_TO = 0.55, 3.55  # seconds into a scene: when dots start and finish popping in
POP = 0.30  # seconds for one dot to pop in

SCENES = (  # photo, the question as shown, what is counted, how far the camera pushes in
    ("cows", "Put a dot on each cow.", "cows", 2.3),  # the cows are far away: glide in on the herd, then back
    ("logs", "Put a dot on each cut log end.", "log ends", 1.12),  # elsewhere a slow push in
    ("fig", "Put a dot on each fig leaf.", "fig leaves", 1.12),
    ("dishes", "Put a dot on each dish.", "dishes", 1.12),
)
ZOOM_IN = (0.35, 2.1)  # seconds into a scene: a strong push in (> 1.5) eases in over this time ...
ZOOM_OUT = (3.55, 4.35)  # ... and back out over this one, so the counts are read on the whole photo


@cache
def font(weight: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT / f"Inter-{weight}.ttf"), size)


@cache
def disc_sprite(maker: str, colour: str, px: int = 160) -> Image.Image:
    """The maker's logo on a disc of its colour with a white edge, as the charts draw it, RGBA."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from segbench.chart import _disc, _fonts

    fig = plt.figure(figsize=(1, 1), dpi=px)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axis("off")
    d_pt = 72 * 0.86  # disc diameter in points: 86 % of the sprite, room for the edge and shadow
    _disc(ax, 0, 0, maker, diameter=d_pt, fill=colour, logo_colour="#ffffff", fonts=_fonts(),
          edge_width=d_pt * 0.07, shadow=False)
    buf = io.BytesIO()
    fig.savefig(buf, dpi=px, transparent=True)
    plt.close(fig)
    return Image.open(buf).convert("RGBA")


@cache
def sprite(maker: str, colour: str, px: int) -> Image.Image:
    return disc_sprite(maker, colour).resize((max(2, px), max(2, px)), Image.LANCZOS)


def ease_out_back(t: float) -> float:
    c = 1.9
    t -= 1
    return 1 + (c + 1) * t ** 3 + c * t ** 2


def ease(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


@dataclass
class Dot:
    key: str
    x: float  # photo pixels
    y: float
    start: float  # seconds into the scene


class Scene:
    def __init__(self, key: str, question: str, noun: str, zoom: float, answers: list[points.Answer],
                 colour: dict[str, str]):
        task = TASKS[key]
        self.question, self.noun = question, noun
        with Image.open(ROOT / "images" / task.photo) as im:
            photo = im.convert("RGB")  # the upright pixel grid the models were sent
        W, H = photo.size
        mine = [a for a in answers if a.task == key]
        self.models = [a.key for a in mine]
        self.makers = {a.key: a.maker for a in mine}
        self.colour = colour
        inside = [(a.key, p["x"] * W / 1000, p["y"] * H / 1000) for a in mine for p in a.points
                  if 0 <= p["x"] <= 1000 and 0 <= p["y"] <= 1000]

        # The photo fitted in PHOTO_BOX; a camera (zoom, centre) moves over it, see view().
        box_w, box_h = PHOTO_BOX[2] - PHOTO_BOX[0], PHOTO_BOX[3] - PHOTO_BOX[1]
        self.src = photo
        self.fit = min(box_w / W, box_h / H)  # screen pixels per photo pixel at zoom 1
        self.size = (round(W * self.fit), round(H * self.fit))
        self.pos = (PHOTO_BOX[0] + (box_w - self.size[0]) // 2, PHOTO_BOX[1] + (box_h - self.size[1]) // 2)
        self.zoom = zoom
        xs, ys = sorted(x for _, x, _ in inside), sorted(y for _, _, y in inside)
        # strong zoom: aim at the middle of the dots (the herd); slow push: the photo's centre
        self.focus = (xs[len(xs) // 2], ys[len(ys) // 2]) if zoom > 1.5 else (W / 2, H / 2)

        # Disc size from how crowded the photo is: logos stay readable where they can, never huge.
        area = self.size[0] * self.size[1]
        self.disc = int(min(36, max(17, 0.42 * math.sqrt(area / max(1, len(inside))))))

        # Models pop in one after another; within a model, its dots ripple in over a short moment.
        slot = (DOTS_TO - DOTS_FROM - POP) / max(1, len(self.models))
        self.model_start = {k: DOTS_FROM + i * slot for i, k in enumerate(self.models)}
        rnd = random.Random(key)
        self.dots = []
        for k, x, y in inside:
            self.dots.append(Dot(k, x, y, self.model_start[k] + rnd.uniform(0, slot * 1.6)))
        self.dots.sort(key=lambda d: d.start)

        counts = [len(a.points) for a in mine if not a.unreadable]
        unreadable = sum(bool(a.unreadable) for a in mine)
        self.summary = f"{len(mine)} models counted {min(counts)} to {max(counts)} {noun}"
        self.footnote = f"{unreadable} repl{'y' if unreadable == 1 else 'ies'} could not be read" if unreadable else ""
        self.unreadable = {a.key for a in mine if a.unreadable}

    def view(self, t: float) -> tuple[float, float, float, float]:
        """The part of the photo on screen at time t: (x0, y0, x1, y1) in photo pixels."""
        W, H = self.src.size
        if self.zoom > 1.5:
            k = ease((t - ZOOM_IN[0]) / (ZOOM_IN[1] - ZOOM_IN[0])) * (1 - ease((t - ZOOM_OUT[0]) / (ZOOM_OUT[1] - ZOOM_OUT[0])))
        else:
            k = ease(t / SCENE)
        z = 1 + (self.zoom - 1) * k
        w, h = W / z, H / z
        cx = W / 2 + (self.focus[0] - W / 2) * k
        cy = H / 2 + (self.focus[1] - H / 2) * k
        x0 = min(max(cx - w / 2, 0), W - w)
        y0 = min(max(cy - h / 2, 0), H - h)
        return x0, y0, x0 + w, y0 + h

    def frame(self, t: float) -> Image.Image:
        img = Image.new("RGB", (SIZE, SIZE), BG)
        fade = ease(t / 0.35) * ease((SCENE - t) / 0.3)
        v = self.view(t)
        img.paste(self.src.resize(self.size, Image.BICUBIC, box=v, reducing_gap=2.0), self.pos)
        canvas = img.convert("RGBA")
        sx = self.size[0] / (v[2] - v[0])
        zoom = sx / self.fit
        disc = self.disc * zoom ** 0.5  # dots grow a little as the camera comes in
        x_lo, y_lo = self.pos
        x_hi, y_hi = x_lo + self.size[0], y_lo + self.size[1]
        canvas_photo = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
        for d in self.dots:
            u = (t - d.start) / POP
            if u <= 0:
                continue
            k = ease_out_back(min(1.0, u))
            px = round(disc * k)
            X, Y = x_lo + (d.x - v[0]) * sx, y_lo + (d.y - v[1]) * sx
            if px < 3 or not (x_lo - px < X < x_hi + px and y_lo - px < Y < y_hi + px):
                continue
            sp = sprite(self.makers[d.key], self.colour[d.key], px)
            canvas_photo.alpha_composite(sp, (round(X - px / 2), round(Y - px / 2)))
        # dots stay inside the photo's frame
        frame_mask = Image.new("L", (SIZE, SIZE), 0)
        ImageDraw.Draw(frame_mask).rectangle((x_lo, y_lo, x_hi - 1, y_hi - 1), fill=255)
        canvas.alpha_composite(Image.composite(canvas_photo, Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0)), frame_mask))
        draw = ImageDraw.Draw(canvas)
        # the question
        draw.text((SIZE / 2, 112), self.question, font=font("SemiBold", 52), fill=WHITE, anchor="mm")
        # logos of the models, lighting up in turn
        n = len(self.models)
        gap = min(60, (SIZE - 120) / n)
        x0 = SIZE / 2 - gap * (n - 1) / 2
        for i, k in enumerate(self.models):
            on = ease((t - self.model_start[k]) / 0.25)
            bad = k in self.unreadable
            size = round(40 + 8 * on * (1 - ease((t - self.model_start[k] - 0.35) / 0.4)))
            sp = sprite(self.makers[k], self.colour[k], size).copy()
            alpha = 0.28 + 0.72 * on * (0.35 if bad else 1)
            sp.putalpha(sp.getchannel("A").point(lambda a, al=alpha: round(a * al)))
            canvas.alpha_composite(sp, (round(x0 + i * gap - size / 2), round(STRIP_Y - size / 2)))
        # counter, then the spread of counts
        shown = sum(1 for d in self.dots if d.start <= t)
        counter_off = ease((t - DOTS_TO - 0.1) / 0.25)  # the counter goes, then the summary comes
        summary_on = ease((t - DOTS_TO - 0.4) / 0.35)
        if counter_off < 1:
            a = round(255 * (1 - counter_off))
            draw.text((SIZE / 2, 1018), f"{shown} dots", font=font("Medium", 30), fill=(*DIM, a), anchor="mm")
        if summary_on > 0:
            a = round(255 * summary_on)
            draw.text((SIZE / 2, 1018), self.summary, font=font("Medium", 30), fill=(*WHITE, a), anchor="mm")
            if self.footnote:
                draw.text((SIZE / 2, 1054), self.footnote, font=font("Regular", 22), fill=(*FAINT, a), anchor="mm")
        out = canvas.convert("RGB")
        if fade < 1:
            out = Image.blend(Image.new("RGB", (SIZE, SIZE), BG), out, fade)
        return out


def frames(scenes: list[Scene]):
    for s in scenes:
        for i in range(round(SCENE * FPS)):
            yield s.frame(i / FPS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", default=list(points.RUNS), help="call records (.jsonl)")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "views" / "video" / "segbench-points.mp4")
    ap.add_argument("--preview", action="store_true", help="write a few still frames next to --out instead")
    args = ap.parse_args()

    answers = points.load(args.runs, "find")
    colour = points.colours(answers)
    scenes = [Scene(k, q, n, z, answers, colour) for k, q, n, z in SCENES]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.preview:
        stills = [(scenes[0], 2.6), (scenes[0], SCENE - 0.5), (scenes[1], SCENE - 0.5), (scenes[3], 2.2)]
        for i, (s, t) in enumerate(stills):
            p = args.out.with_name(f"preview-{i}.png")
            s.frame(t).save(p)
            print(f"wrote {p}")
        return

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{SIZE}x{SIZE}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(args.out)]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = 0
    for f in frames(scenes):
        ff.stdin.write(np.asarray(f, dtype=np.uint8).tobytes())
        n += 1
    ff.stdin.close()
    if ff.wait():
        sys.exit("ffmpeg failed")
    print(f"wrote {args.out} ({n / FPS:.1f} s, {args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
