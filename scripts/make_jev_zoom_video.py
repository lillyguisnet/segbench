# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow>=10"]
# ///
"""A video of Jev zooming in from the whole world to a San Francisco street corner (2026-10-16).

    uv run scripts/make_jev_zoom_video.py              # results/probe-jev-zoom-landmark/corner-world1/jev-zoom-world.mp4
    uv run scripts/make_jev_zoom_video.py --preview    # a few still frames instead, to check the look

Everything shown is the recorded search corner-world1, try 1 of 3 (the longest: it
drifted to Vancouver, zoomed out twice, then found the corner): the ten maps exactly as
Jev was sent them (map-NN-zK.png), its yes/no probabilities, its boxes, the decision the
search took, and the cost. Moves between maps are drawn from the two maps themselves:
the camera flies into the next map's area on the current one (or back out) while the
next map fades in at its true place and scale, so nothing is drawn that Jev did not see,
except our marks: Jev's box (pink), the next map's area (green), the real corner (a red pin,
never shown to Jev). The title and the side panel quote the two questions word for word,
read from the record. No sequence is text only: captions sit under the map.

1920 x 1080, 30 frames a second, H.264; frames drawn with Pillow and piped to ffmpeg,
like scripts/make_jev_pens_video.py.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from functools import cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "results" / "probe-jev-zoom-landmark" / "corner-world1" / "try-1"
FONT = ROOT / "assets" / "fonts"
CORNER = (37.7932159, -122.3999769)  # as in scripts/jev_zoom_landmark.py
MAP = 1024

W, H, FPS = 1920, 1080, 30
BG = (14, 15, 18)
WHITE = (255, 255, 255)
DIM = (150, 155, 165)
FAINT = (90, 95, 105)
PINK = (236, 40, 170)
GREEN = (40, 200, 120)
RED = (226, 32, 40)
GREY = (150, 150, 150)
YELLOW = (255, 210, 0)

VX, VY, VS = 60, 110, 870          # the map's viewport (square)
PX0, PX1 = 1010, 1860              # the side panel
CAP_Y = VY + VS + 46

# seconds per part of a step
T_ASK, T_BOX, T_DECIDE, T_MOVE = 1.0, 0.6, 0.9, 1.6
T_INTRO, T_FOUND, T_REWIND_EACH, T_GRID = 3.0, 4.0, 0.42, 4.5


@cache
def font(weight: str, size: int):
    return ImageFont.truetype(str(FONT / f"Inter-{weight}.ttf"), size)


def clamp01(t):
    return min(1.0, max(0.0, t))


def ease(t):
    t = clamp01(t)
    return t * t * (3 - 2 * t)


def ease_out_back(t):
    t = clamp01(t) - 1
    return 1 + 2.7 * t ** 3 + 1.7 * t ** 2


def pin(d, x, y, size=1.0, alpha=255):
    """A red map pin whose tip is at (x, y)."""
    r, h = 13 * size, 38 * size
    cx, cy = x, y - h + r
    a = alpha
    d.ellipse((x - 7 * size, y - 3 * size, x + 7 * size, y + 3 * size), fill=(0, 0, 0, round(a * 0.35)))
    for grow, col in ((3 * size, (255, 255, 255, a)), (0, RED + (a,))):
        rr = r + grow
        k = rr / r
        d.polygon([(x, y + grow * 0.6), (cx - rr * 0.82, cy + rr * 0.55), (cx + rr * 0.82, cy + rr * 0.55)], fill=col)
        d.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), fill=col)
    d.ellipse((cx - r * 0.42, cy - r * 0.42, cx + r * 0.42, cy + r * 0.42), fill=(255, 255, 255, a))


def wrap(text, f, width):
    lines, line = [], ""
    for w in text.split():
        t = (line + " " + w).strip()
        if f.getlength(t) <= width:
            line = t
        else:
            lines.append(line)
            line = w
    return lines + [line]


def world_px(lat, lon, z):
    n = 256 * 2 ** z
    return (lon + 180) / 360 * n, (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n


class Step:
    def __init__(self, s):
        self.n, self.z = s["step"], s["zoom"]
        self.lat, self.lon = s["centre"]
        self.yes = s["said_on_map"]
        self.p_yes = s["p_on_map"]["true"]
        self.box = s["box_px"]
        self.width_km = s["map_width_m"] / 1000
        self.cost = s["cost_usd"]
        self.error_m = s["error_m"]
        self.in_box = s["box_touches_bridge"]
        self.img = Image.open(DIR / f"map-{self.n:02d}-z{self.z}.png").convert("RGB")
        cx, cy = world_px(self.lat, self.lon, self.z)
        self.x0, self.y0 = cx - MAP / 2, cy - MAP / 2

    def px(self, lat, lon):
        x, y = world_px(lat, lon, self.z)
        return x - self.x0, y - self.y0

    def rect_of(self, other: "Step"):
        """The other map's square, in this map's pixels: (cx, cy, half)."""
        cx, cy = self.px(other.lat, other.lon)
        return cx, cy, MAP / 2 * 2 ** (self.z - other.z)


def km(v):
    return f"{v:,.0f} km" if v >= 10 else (f"{v:.1f} km" if v >= 1 else f"{v * 1000:,.0f} m")


class Movie:
    def __init__(self):
        data = json.loads((DIR / "steps.json").read_text())
        self.steps = [Step(s) for s in data["steps"]]
        props = data["steps"][0]["question"]["schema"]["properties"]
        assert all(st["question"]["schema"]["properties"] == props for st in data["steps"]), "one question for all maps"
        self.q_yes, self.q_box = props["on_map"]["description"], props["box"]["description"]
        self.n = len(self.steps)
        self.cum_cost = [sum(s.cost for s in self.steps[:i + 1]) for i in range(self.n)]
        self.timeline = []  # (start, end, kind, k)
        t = 0.0

        def add(d, kind, k):
            nonlocal t
            self.timeline.append((t, t + d, kind, k))
            t += d
        add(T_INTRO, "intro", 0)
        for k in range(self.n):
            add(T_ASK, "ask", k)
            add(T_BOX, "box", k)
            add(T_DECIDE, "decide", k)
            if k < self.n - 1:
                add(T_MOVE, "move", k)
        add(T_FOUND, "found", self.n - 1)
        for k in range(self.n - 2, -1, -1):
            add(T_REWIND_EACH, "rewind", k)
        add(T_GRID, "grid", 0)
        self.duration = t

    # --- drawing the map viewport -------------------------------------------------

    def paste(self, frame, img, cam, rect, alpha=1.0):
        """Draw `img` (whose square is `rect` = (cx, cy, half) in camera-map pixels) through
        the camera (cx, cy, half) into the viewport."""
        ccx, ccy, ch = cam
        k = VS / (2 * ch)
        rx, ry, rh = rect
        dx0, dy0 = VX + (rx - rh - (ccx - ch)) * k, VY + (ry - rh - (ccy - ch)) * k
        side = 2 * rh * k
        ix0, iy0 = max(VX, dx0), max(VY, dy0)
        ix1, iy1 = min(VX + VS, dx0 + side), min(VY + VS, dy0 + side)
        if ix1 - ix0 < 1 or iy1 - iy0 < 1 or alpha <= 0:
            return
        s = img.width / side
        src = ((ix0 - dx0) * s, (iy0 - dy0) * s, (ix1 - dx0) * s, (iy1 - dy0) * s)
        size = (round(ix1 - ix0), round(iy1 - iy0))
        if size[0] < 1 or size[1] < 1:
            return
        part = img.resize(size, Image.LANCZOS if side < img.width * 1.5 else Image.BICUBIC, box=src)
        pos = (round(ix0), round(iy0))
        if alpha >= 1:
            frame.paste(part, pos)
        else:
            frame.paste(Image.blend(frame.crop((*pos, pos[0] + size[0], pos[1] + size[1])), part, alpha), pos)

    @staticmethod
    def to_view(cam, x, y):
        ccx, ccy, ch = cam
        k = VS / (2 * ch)
        return VX + (x - (ccx - ch)) * k, VY + (y - (ccy - ch)) * k

    def mark_box(self, d, cam, box, colour, width=5, grow=1.0, alpha=255):
        x0, y0 = self.to_view(cam, box[0], box[1])
        x1, y1 = self.to_view(cam, box[2], box[3])
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        hw, hh = max(4, (x1 - x0) / 2 * grow), max(4, (y1 - y0) / 2 * grow)
        d.rectangle((cx - hw, cy - hh, cx + hw, cy + hh), outline=colour + (alpha,), width=width)

    def mark_truth(self, d, cam, step, alpha=255, pulse=0.0):
        x, y = self.to_view(cam, *step.px(*CORNER))
        inside = VX + 10 < x < VX + VS - 10 and VY + 10 < y < VY + VS - 10
        if inside:
            if pulse > 0:
                rr = 10 + 60 * pulse
                d.ellipse((x - rr, y - rr * 0.5, x + rr, y + rr * 0.5), outline=RED + (round(alpha * (1 - pulse)),), width=4)
            pin(d, x, y, 1.0, alpha)
        else:  # an arrow at the edge, toward it
            cx, cy = VX + VS / 2, VY + VS / 2
            a = math.atan2(y - cy, x - cx)
            ux, uy = math.cos(a), math.sin(a)
            kk = min((VS / 2 - 26) / abs(ux) if ux else 1e9, (VS / 2 - 26) / abs(uy) if uy else 1e9)
            tx, ty = cx + ux * kk, cy + uy * kk
            bx, by = tx - ux * 30, ty - uy * 30
            pts = [(tx, ty), (bx - uy * 14, by + ux * 14), (bx + uy * 14, by - ux * 14)]
            d.polygon(pts, fill=RED + (alpha,), outline=(255, 255, 255, alpha), width=3)

    def next_rect_box(self, k):
        a, b = self.steps[k], self.steps[k + 1]
        cx, cy, h = a.rect_of(b)
        return [cx - h, cy - h, cx + h, cy + h]

    def viewport(self, frame, kind, k, u):
        """The map viewport for part `kind` of step k at progress u (0..1)."""
        full = (MAP / 2, MAP / 2, MAP / 2)
        s = self.steps[k]
        ov = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        ov_pin = Image.new("RGBA", frame.size, (0, 0, 0, 0))  # under the boxes, so the pin never hides one
        dp = ImageDraw.Draw(ov_pin)
        if kind in ("move", "rewind"):
            e = ease(u) if kind == "move" else ease(1 - u)
            nxt = self.steps[k + 1]
            rx, ry, rh = s.rect_of(nxt)
            h = MAP / 2 * (rh / (MAP / 2)) ** e
            w = (MAP / 2 - h) / (MAP / 2 - rh) if abs(MAP / 2 - rh) > 1e-6 else e
            cam = (MAP / 2 + (rx - MAP / 2) * w, MAP / 2 + (ry - MAP / 2) * w, h)
            if rh < MAP / 2:  # zoom in: the next map sits inside, fades in on top
                self.paste(frame, s.img, cam, full)
                self.paste(frame, nxt.img, cam, (rx, ry, rh), clamp01((e - 0.12) / 0.45))
            else:             # zoom out: the next map is around, the current one fades out on top
                self.paste(frame, nxt.img, cam, (rx, ry, rh))
                self.paste(frame, s.img, cam, full, 1 - clamp01((e - 0.45) / 0.4))
            if kind == "move":
                fade = round(255 * (1 - clamp01(u / 0.35)))
                if s.yes:
                    self.mark_box(d, cam, self.next_rect_box(k), GREEN, 4, alpha=fade)
                self.mark_box(d, cam, s.box, PINK if s.yes else GREY, 5, alpha=fade)
                self.mark_truth(dp, cam, s if e < 0.5 else s, alpha=255)
        else:
            cam = full
            self.paste(frame, s.img, cam, full)
            if kind == "intro":
                pass
            elif kind == "ask":
                pass
            elif kind == "box":
                self.mark_box(d, cam, s.box, PINK if s.yes else GREY, 5, grow=ease_out_back(u / 0.8))
            elif kind == "decide":
                self.mark_box(d, cam, s.box, PINK if s.yes else GREY, 5)
                if s.yes and k < self.n - 1:
                    self.mark_box(d, cam, self.next_rect_box(k), GREEN, 4, grow=ease(u / 0.6), alpha=round(255 * clamp01(u / 0.3)))
                elif not s.yes:
                    # the map greys out a little: "not here"
                    d.rectangle((VX, VY, VX + VS, VY + VS), fill=(20, 20, 25, round(70 * clamp01(u / 0.4))))
            elif kind == "found":
                self.mark_box(d, cam, s.box, PINK, 6)
            if kind != "found":
                self.mark_truth(dp, cam, s)
            else:
                self.mark_truth(dp, cam, s, pulse=(u * 2.5) % 1 if u < 0.8 else 0)
        frame.paste(ov_pin, (0, 0), ov_pin)
        frame.paste(ov, (0, 0), ov)
        fd = ImageDraw.Draw(frame)
        fd.rectangle((VX - 1, VY - 1, VX + VS, VY + VS), outline=(60, 63, 70), width=1)
        fd.text((VX + VS - 8, VY + VS - 8), "© OpenStreetMap contributors", font=font("Medium", 15),
                fill=(70, 70, 70), anchor="rd")

    # --- the side panel -----------------------------------------------------------

    def panel(self, d, kind, k, u):
        s = self.steps[k]
        y = VY
        d.text((PX0, y), f"Map {s.n} of {self.n}", font=font("SemiBold", 44), fill=WHITE)
        y += 0
        d.text((PX1, y + 8), f"≈ {km(s.width_km)} across", font=font("Medium", 28), fill=DIM, anchor="ra")
        y += 76
        d.text((PX0, y), "Sent to Jev with the map, word for word:", font=font("Medium", 22), fill=FAINT)
        y += 38
        q_font = font("Medium", 22)
        on1 = kind in ("intro", "ask")
        on2 = kind == "box"
        d.text((PX0, y), "1  yes / no", font=font("SemiBold", 22), fill=PINK if on1 else DIM)
        for line in wrap(f"“{self.q_yes}”", q_font, PX1 - PX0 - 130):
            d.text((PX0 + 130, y), line, font=q_font, fill=WHITE if on1 else DIM)
            y += 29
        y += 12
        if kind == "intro":
            fill = 0.0
        elif kind == "ask":
            fill = ease(u / 0.8)
        else:
            fill = 1.0
        won = "yes" if s.yes else "no"
        for key, p in (("yes", s.p_yes), ("no", 1 - s.p_yes)):
            hi = key == won and fill >= 1
            d.text((PX0, y + 4), key, font=font("SemiBold" if hi else "Medium", 28), fill=WHITE if hi else DIM)
            bx0, bx1 = PX0 + 80, PX1 - 90
            d.rounded_rectangle((bx0, y, bx1, y + 32), 8, fill=(36, 38, 44))
            col = (PINK if key == "yes" else GREY) if hi or fill < 1 else (70, 72, 80)
            if p * fill > 0.005:
                d.rounded_rectangle((bx0, y, bx0 + (bx1 - bx0) * p * fill, y + 32), 8, fill=col)
            d.text((PX1, y + 16), f"{round(100 * p * fill)}%", font=font("Medium", 26), fill=WHITE if hi else DIM, anchor="rm")
            y += 44
        y += 18
        d.text((PX0, y), "2  box", font=font("SemiBold", 22), fill=PINK if on2 else DIM)
        for line in wrap(f"“{self.q_box}”", q_font, PX1 - PX0 - 130):
            d.text((PX0 + 130, y), line, font=q_font, fill=WHITE if on2 else DIM)
            y += 29
        y += 26
        # decision
        shown = kind in ("decide", "move", "found") or (kind == "box" and u > 0.6)
        if shown:
            if k == self.n - 1:
                txt, col = "Closest zoom: stop here", YELLOW
            elif s.yes:
                f = 2 ** (self.steps[k + 1].z - s.z)
                txt, col = f"Yes → zoom in {f}× on the box", PINK
            else:
                f = 2 ** (s.z - self.steps[k + 1].z)
                txt, col = f"No → zoom out {f}×", (200, 200, 200)
            d.text((PX0, y), txt, font=font("Bold", 34), fill=col)
        y += 64
        # legend
        lg = [(PINK, "box", "Jev's box"), (GREEN, "box", "next map"), (RED, "pin", "the real corner (Jev never sees it)")]
        x = PX0
        for col, kind_, label in lg:
            if kind_ == "box":
                d.rectangle((x, y + 4, x + 24, y + 26), outline=col, width=4)
            else:
                pin(d, x + 12, y + 30, 0.7)
            d.text((x + 34, y + 15), label, font=font("Medium", 22), fill=DIM, anchor="lm")
            x += 34 + font("Medium", 22).getlength(label) + 30
            if x > PX1 - 150:
                x, y = PX0, y + 38
        y += 62
        self.zoom_chart(d, y, k, kind, u)
        # cost
        done = k + 1 if kind not in ("intro", "ask") or (kind == "ask" and u > 0.5) else k
        d.text((PX0, VY + VS - 34), f"requests sent  {done}", font=font("Medium", 26), fill=DIM)
        d.text((PX1, VY + VS - 34), f"${self.cum_cost[done - 1] if done else 0:.5f}", font=font("Medium", 26), fill=DIM, anchor="ra")

    def zoom_chart(self, d, y0, k, kind, u):
        """How close each map was: one dot per map, world at the bottom, street at the top."""
        x0, x1, h = PX0 + 70, PX1 - 20, max(90, VY + VS - 80 - y0)
        y1 = y0 + h
        zmin, zmax = 2, 18
        d.text((PX0, y0 - 6), "street", font=font("Medium", 20), fill=FAINT)
        d.text((PX0, y1 - 18), "world", font=font("Medium", 20), fill=FAINT)
        d.line((x0, y1, x1, y1), fill=(50, 53, 60), width=2)
        d.line((x0, y0, x0, y1), fill=(50, 53, 60), width=2)

        def pt(i, z):
            return x0 + 20 + (x1 - x0 - 40) * i / (self.n - 1), y1 - (z - zmin) / (zmax - zmin) * (h - 10)
        shown = k + (1 if kind not in ("intro",) else 0)
        if kind == "grid" or kind == "rewind":
            shown = self.n
        pts = [pt(i, s.z) for i, s in enumerate(self.steps[:shown])]
        if kind == "move" and k + 1 < self.n:
            a, b = pt(k, self.steps[k].z), pt(k + 1, self.steps[k + 1].z)
            e = ease(u)
            pts.append((a[0] + (b[0] - a[0]) * e, a[1] + (b[1] - a[1]) * e))
        if len(pts) > 1:
            d.line(pts, fill=(110, 115, 125), width=3)
        for i, p in enumerate(pts[:shown]):
            s = self.steps[i]
            col = PINK if s.yes else GREY
            r = 9 if i == k and kind not in ("grid", "rewind") else 6
            d.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=col, outline=WHITE if r == 9 else None, width=2)

    # --- frames -----------------------------------------------------------------

    def part(self, t):
        for a, b, kind, k in self.timeline:
            if a <= t < b:
                return kind, k, (t - a) / (b - a)
        a, b, kind, k = self.timeline[-1]
        return kind, k, 1.0

    def caption(self, kind, k, u):
        s = self.steps[k]
        if kind == "intro":
            return "The first map: the whole world, nothing marked on it", WHITE
        if kind == "ask":
            return (f"Map {s.n}: Jev answers {'yes' if s.yes else 'no'}, {round(100 * (s.p_yes if s.yes else 1 - s.p_yes))}% sure"), WHITE
        if kind == "box":
            return ("Jev's box: where it puts the corner" if s.yes else "Jev's box is not used: it said the corner is not here"), (PINK if s.yes else DIM)
        if kind == "decide":
            if k == self.n - 1:
                return "The closest zoom of the map, and Jev still says yes: the search stops", YELLOW
            if s.yes:
                return "The next map: centred on the box, zoomed in so the box about fills it", GREEN
            return "Not here: zoom back out around the same spot", WHITE
        if kind == "move":
            nxt = self.steps[k + 1]
            return (f"Zooming {'in' if nxt.z > s.z else 'out'} to a map ≈ {km(nxt.width_km)} across"), DIM
        if kind == "found":
            return (f"Found: the box contains the real corner, its middle {s.error_m} m away"), YELLOW
        if kind == "rewind":
            return "The whole search again, from the street back to the world", DIM
        return (f"{self.n} maps, {self.n} requests, ${self.cum_cost[-1]:.4f}: zoomed in 7 times, out 2 times"), WHITE

    def grid(self, frame, u):
        cols, cell, gap = 5, 330, 18
        gx = (W - (cols * cell + (cols - 1) * gap)) / 2
        gy = 135
        d = ImageDraw.Draw(frame, "RGBA")
        for i, s in enumerate(self.steps):
            a = clamp01((u * 4.5 - i * 0.12))
            if a <= 0:
                continue
            x, y = gx + (i % cols) * (cell + gap), gy + (i // cols) * (cell + 95)
            im = s.img.resize((cell, cell), Image.LANCZOS)
            dd = ImageDraw.Draw(im)
            k = cell / MAP
            b = [v * k for v in s.box]
            dd.rectangle(b, outline=PINK if s.yes else GREY, width=4)
            tx, ty = (v * k for v in s.px(*CORNER))
            if 0 < tx < cell and 0 < ty < cell:
                pin(ImageDraw.Draw(im, "RGBA"), tx, ty, 0.6)
            pos = (round(x), round(y + (1 - ease(a)) * 30))
            if a < 1:
                im = Image.blend(frame.crop((*pos, pos[0] + cell, pos[1] + cell)), im, ease(a))
            frame.paste(im, pos)
            d.text((x, pos[1] + cell + 12), f"{s.n}  {'yes' if s.yes else 'no'}", font=font("SemiBold", 24),
                   fill=PINK if s.yes else DIM)
            d.text((x + cell, pos[1] + cell + 12), km(s.width_km), font=font("Medium", 22), fill=FAINT, anchor="ra")

    def frame(self, t) -> Image.Image:
        kind, k, u = self.part(t)
        frame = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(frame)
        d.text((VX, 52), "Jev 1.14 looks for “the corner of California Street and Battery Street, San Francisco”",
               font=font("SemiBold", 32), fill=WHITE, anchor="lm")
        if kind == "grid":
            self.grid(frame, u)
        else:
            self.viewport(frame, kind, k, u)
            self.panel(ImageDraw.Draw(frame), kind if kind != "rewind" else "found", k if kind != "rewind" else self.n - 1, u)
        text, col = self.caption(kind, k, u)
        x = VX + VS / 2 if kind != "grid" else W / 2
        ImageDraw.Draw(frame).text((x if kind == "grid" else W / 2, CAP_Y if kind != "grid" else 1010), text,
                                   font=font("SemiBold", 34), fill=col, anchor="mm")
        fade = min(clamp01(t / 0.5), clamp01((self.duration - t) / 0.6))
        return frame if fade >= 1 else Image.blend(Image.new("RGB", (W, H), BG), frame, fade)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--out", type=Path, default=DIR.parent / "jev-zoom-world.mp4")
    a = ap.parse_args()
    m = Movie()
    if a.preview:
        out = DIR.parent / "video-preview"
        out.mkdir(exist_ok=True)
        for st, en, kind, k in m.timeline:
            if (kind, k) in {("intro", 0), ("ask", 0), ("decide", 0), ("move", 1), ("decide", 2), ("move", 3),
                             ("found", 9), ("rewind", 4), ("grid", 0)}:
                t = st + (en - st) * (0.5 if kind != "grid" else 0.8)
                m.frame(t).save(out / f"{t:06.2f}-{kind}-{k}.png")
        print(f"previews in {out}, duration {m.duration:.1f} s")
        return
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(a.out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = round(m.duration * FPS)
    for i in range(n):
        p.stdin.write(m.frame(i / FPS).tobytes())
    p.stdin.close()
    if p.wait():
        sys.exit("ffmpeg failed")
    print(f"wrote {a.out} ({n / FPS:.1f} s, {a.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
