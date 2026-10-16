# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow>=10"]
# ///
"""A video of Jev + halving finding the pens on Maxime's floor photo (2026-10-16).

    uv run scripts/make_jev_pens_video.py              # results/probe-jev-pens/jev-halving-pens.mp4
    uv run scripts/make_jev_pens_video.py --preview    # a few still frames instead, to check the look

Everything shown is the recorded search (results/probe-jev-pens/run.json,
try 1 of 3): the pieces in the order they were asked, the picture Jev was
sent for each (fitted as segbench/jev_halving.fit does), its probabilities
for none / one / several, its boxes, the join across a cut line, the recheck,
the final boxes, then the three tries side by side. No sequence is text
only: captions sit under the picture.

1920 x 1080, 30 frames a second, H.264; frames drawn with Pillow and piped to
ffmpeg (on PATH), like scripts/make_video.py.
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
DIR = ROOT / "results" / "probe-jev-pens"
FONT = ROOT / "assets" / "fonts"

W, H, FPS = 1920, 1080, 30
BG = (14, 15, 18)
PANEL = (24, 26, 31)
WHITE = (255, 255, 255)
DIM = (150, 155, 165)
FAINT = (90, 95, 105)
PINK = (236, 40, 170)
YELLOW = (255, 210, 0)
CUT = (240, 240, 240)
ACTIVE = (255, 210, 0)

VX, VY, VW, VH = 60, 100, 1320, 742  # the photo's viewport (16:9)
PX0, PX1 = 1430, 1860  # the side panel


@cache
def font(weight: str, size: int):
    return ImageFont.truetype(str(FONT / f"Inter-{weight}.ttf"), size)


def clamp01(t):
    return min(1.0, max(0.0, t))


def ease(t):
    t = clamp01(t)
    return t * t * (3 - 2 * t)


def ease_out_back(t):
    t = clamp01(t)
    c = 1.7
    t -= 1
    return 1 + (c + 1) * t ** 3 + c * t ** 2


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_rect(a, b, t):
    return [lerp(x, y, t) for x, y in zip(a, b)]


def fit_sent(crop: Image.Image) -> Image.Image:
    """The picture Jev was sent for a piece (segbench/jev_halving.fit, copied: this script has its own environment)."""
    w, h = crop.size
    k = min(1.0, 1568 / max(w, h))
    if min(w, h) * k < 280:
        k = min(280 / min(w, h), 1568 / max(w, h))
    iw, ih = max(1, round(w * k)), max(1, round(h * k))
    if (iw, ih) != (w, h):
        crop = crop.resize((iw, ih), Image.LANCZOS)
    Wc, Hc = max(280, iw), max(280, ih)
    if (Wc, Hc) == (iw, ih):
        return crop
    canvas = Image.new("RGB", (Wc, Hc), (128, 128, 128))
    canvas.paste(crop, ((Wc - iw) // 2, (Hc - ih) // 2))
    return canvas


class Movie:
    def __init__(self):
        self.photo = Image.open(DIR / "pens.jpg").convert("RGB")
        self.PW, self.PH = self.photo.size
        self.tries = json.loads((DIR / "run.json").read_text())
        run = self.tries[0]
        n = run["nodes"]
        self.nodes = n
        full = [0, 0, self.PW, self.PH]
        # storyboard: (node index, camera rect, caption, outcome)
        right_half, top_right, top_mid = [480, 0, 960, 540], [480, 0, 960, 270], [480, 0, 720, 270]
        self.asks = []  # each: dict(t0, node, camera, caption)
        self.events = []
        self.cuts = []  # (t0, (x0, y0, x1, y1))
        self.boxes = []  # (t0, rect, t_end or None)
        self.nones = []  # (t0, rect)
        self.captions = []  # (t0, t1, text)
        self.camera_keys = []  # (t, rect)
        t = 0.0
        self.captions.append((0.4, 3.2, "Find each pen in this photo."))
        self.camera_keys += [(0.0, full)]
        t = 3.0
        ASK = 2.3  # seconds per question

        def ask(i, camera, caption, after=0.0):
            nonlocal t
            node = n[i]
            self.camera_keys.append((t, camera))
            self.asks.append({"t0": t, "node": node})
            self.captions.append((t + 0.1, t + ASK + after - 0.1, caption))
            outcome_t = t + 1.35
            r = node["region"]
            if node["choice"] == "several" and not node.get("recheck"):
                x0, y0, x1, y1 = r
                if x1 - x0 >= y1 - y0:
                    m = (x0 + x1) / 2
                    self.cuts.append((outcome_t, (m, y0, m, y1)))
                else:
                    m = (y0 + y1) / 2
                    self.cuts.append((outcome_t, (x0, m, x1, m)))
            elif node["choice"] == "one" and not node.get("recheck"):
                self.boxes.append([outcome_t, node["box"], None])
            elif node["choice"] == "none":
                self.nones.append((outcome_t, r))
            t += ASK + after

        ask(0, full, "The whole photo: none, one or several pens? Several, so cut it in half.")
        ask(2, full, "Left half: one pen. Jev draws its box.")
        ask(1, full, "Right half: several pens. Cut it again.")
        ask(4, [400, 180, 1040, 540], "Bottom right: one pen.")
        ask(3, top_right, "Top right: several. Each cut also zooms in: Jev sees the piece larger.")
        ask(6, top_right, "This piece holds no pen: stop here.")
        ask(5, [470, 0, 830, 202], "This one holds several: cut once more.")
        ask(8, [470, 0, 830, 202], "One pen.")
        ask(7, [490, 120, 770, 277], "One, but only the tip of the purple pen, cut by the line below.", after=0.4)
        # join: the tip's box and the lower box meet on the cut line y = 270
        self.join_t0 = t
        tip, low = self.boxes[-1], self.boxes[-3]  # boxes from node 7 and node 4
        self.tip, self.low = tip, low
        self.joined = [min(tip[1][0], low[1][0]), min(tip[1][1], low[1][1]),
                       max(tip[1][2], low[1][2]), max(tip[1][3], low[1][3])]
        tip[2] = low[2] = t + 2.2
        self.camera_keys.append((t, [470, 200, 750, 357]))
        self.captions.append((t + 0.1, t + 3.0, "Two parts that meet on the same cut line are joined into one pen."))
        t += 3.2
        self.boxes.append([self.join_t0 + 2.2, self.joined, None])
        self.joined_box = self.boxes[-1]
        ask(9, [470, 200, 750, 357], "The joined box is asked again on its own: one pen.")
        # results
        self.result_t0 = t
        self.camera_keys.append((t, full))
        self.captions.append((t + 0.2, t + 4.6, "Three pens. The highlighter, the stylus, the eraser and the rest are left out."))
        t += 5.0
        self.tries_t0 = t
        self.captions.append((t + 0.6, t + 6.2, "Three tries, the same three pens. About one second and $0.0004 each."))
        t += 6.8
        self.duration = t
        self.camera_keys.sort(key=lambda k: k[0])
        self.final = run["final"]
        self.cost = [sum(x["cost_usd"] for x in [a["node"] for a in self.asks[:k + 1]]) for k in range(len(self.asks))]
        self.sent = {id(a["node"]): fit_sent(self.photo.crop(tuple(a["node"]["region"]))) for a in self.asks}

    # ---------------------------------------------------------------- camera
    def camera_at(self, t):
        """Ease from one key to the next over 0.9 s, starting at the next key's time."""
        keys = self.camera_keys
        cur = self.aspect(keys[0][1])
        for i in range(1, len(keys)):
            k_t, k_r = keys[i]
            if t < k_t:
                break
            p = ease((t - k_t) / 0.9)
            cur = lerp_rect(cur, self.aspect(k_r), p)
        return cur

    def aspect(self, r):
        """The smallest 16:9 rect around r with a margin, kept inside the photo."""
        x0, y0, x1, y1 = r
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        w, h = (x1 - x0), (y1 - y0)
        if [x0, y0, x1, y1] != [0, 0, self.PW, self.PH]:
            w, h = w * 1.12, h * 1.12
        ar = VW / VH
        if w / h < ar:
            w = h * ar
        else:
            h = w / ar
        w, h = min(w, self.PW), min(h, self.PH)
        cx = min(max(cx, w / 2), self.PW - w / 2)
        cy = min(max(cy, h / 2), self.PH - h / 2)
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

    # ---------------------------------------------------------------- drawing
    def viewport(self, t, cam, show_cuts=1.0, dim_all=0.0):
        c0, c1, c2, c3 = cam
        img = self.photo.resize((VW, VH), Image.BICUBIC, box=(c0, c1, c2, c3))
        s = VW / (c2 - c0)

        def P(x, y):
            return ((x - c0) * s, (y - c1) * s)

        def R(r):
            a, b = P(r[0], r[1]), P(r[2], r[3])
            return [a[0], a[1], b[0], b[1]]

        over = Image.new("RGBA", (VW, VH), (0, 0, 0, 0))
        d = ImageDraw.Draw(over)
        # pieces with no pen: greyed
        for t0, r in self.nones:
            a = ease((t - t0) / 0.4) * show_cuts
            if a > 0:
                d.rectangle(R(r), fill=(10, 10, 14, int(150 * a)))
                rr = R(r)
                cx, cy, k = (rr[0] + rr[2]) / 2, (rr[1] + rr[3]) / 2, 0.12 * min(rr[2] - rr[0], rr[3] - rr[1])
                d.line((cx - k, cy - k, cx + k, cy + k), fill=(200, 200, 205, int(220 * a)), width=6)
                d.line((cx - k, cy + k, cx + k, cy - k), fill=(200, 200, 205, int(220 * a)), width=6)
        # the piece being asked: the rest dimmed, the piece outlined
        act = self.active(t)
        if act is not None:
            node, a = act
            rr = R(node["region"])
            mask = Image.new("L", (VW, VH), int(130 * a))
            ImageDraw.Draw(mask).rectangle(rr, fill=0)
            over = Image.alpha_composite(over, Image.merge("RGBA", [Image.new("L", (VW, VH), 0)] * 3 + [mask]))
            d = ImageDraw.Draw(over)
            d.rectangle(rr, outline=ACTIVE + (int(255 * a),), width=4)
        # cut lines, drawn from one end
        for t0, (x0, y0, x1, y1) in self.cuts:
            p = ease((t - t0) / 0.5)
            if p <= 0:
                continue
            a, b = P(x0, y0), P(x1, y1)
            e = (lerp(a[0], b[0], p), lerp(a[1], b[1], p))
            hot = t0 > self.join_t0 - 99 and self.join_glow(t, (x0, y0, x1, y1))
            col = (YELLOW if hot else CUT) + (int(230 * show_cuts),)
            d.line((a, e), fill=col, width=6 if hot else 3)
        # boxes
        for bx in self.boxes:
            t0, r, t_end = bx
            if t < t0 or (t_end is not None and t >= t_end):
                continue
            # the joined box takes over from its two parts at full size; every other box pops in
            p = 1.0 if bx is self.joined_box else ease_out_back((t - t0) / 0.45)
            rr = R(r)
            if t_end is not None and t_end - t < 2.2 and t >= self.join_t0 + 0.6:  # parts moving into the joined box
                rr = R(lerp_rect(r, self.joined, ease((t - (self.join_t0 + 1.2)) / 1.0)))
            cx, cy = (rr[0] + rr[2]) / 2, (rr[1] + rr[3]) / 2
            hw, hh = (rr[2] - rr[0]) / 2 * p, (rr[3] - rr[1]) / 2 * p
            pulse = 0
            if t_end is not None and self.join_t0 <= t < t_end:
                pulse = 0.5 + 0.5 * math.sin((t - self.join_t0) * 9)
            hw, hh = abs(hw), abs(hh)
            d.rectangle((cx - hw, cy - hh, cx + hw, cy + hh), outline=PINK + (255,), width=5 + int(3 * pulse))
        # final: dots at the middles
        if t >= self.result_t0 + 0.8:
            for k, b in enumerate(self.final):
                p = ease_out_back((t - (self.result_t0 + 0.8 + 0.25 * k)) / 0.4)
                if p <= 0:
                    continue
                cx, cy = P((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
                r = 11 * p
                d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=YELLOW + (255,), outline=(0, 0, 0, 255), width=2)
        if dim_all:
            over = Image.alpha_composite(over, Image.new("RGBA", (VW, VH), (0, 0, 0, int(255 * dim_all))))
        img = Image.alpha_composite(img.convert("RGBA"), over).convert("RGB")
        return img

    def join_glow(self, t, line):
        x0, y0, x1, y1 = line
        return y0 == y1 == 270 and self.join_t0 <= t < self.join_t0 + 3.0

    def active(self, t):
        for a in reversed(self.asks):
            if a["t0"] <= t < a["t0"] + 2.2:
                fade = ease((t - a["t0"]) / 0.3) * (1 - ease((t - a["t0"] - 1.9) / 0.3))
                return a["node"], fade
        return None

    def current_ask(self, t):
        cur = None
        for k, a in enumerate(self.asks):
            if a["t0"] <= t:
                cur = (k, a)
        return cur

    def panel(self, frame, t):
        d = ImageDraw.Draw(frame)
        d.rounded_rectangle((PX0 - 20, VY, PX1 + 20, VY + VH), radius=18, fill=PANEL)
        cur = self.current_ask(t)
        if t >= self.result_t0:
            self.panel_result(frame, t)
            return
        d.text((PX0, VY + 26), "What Jev sees", font=font("SemiBold", 26), fill=DIM)
        if cur is None:
            return
        k, a = cur
        node = a["node"]
        since = t - a["t0"]
        if t >= self.join_t0 and t < self.join_t0 + 3.2:
            since = 99  # keep the last answer shown during the join
        sent = self.sent[id(node)]
        box_w, box_h = PX1 - PX0, 300
        sc = min(box_w / sent.width, box_h / sent.height)
        th = sent.resize((max(1, int(sent.width * sc)), max(1, int(sent.height * sc))), Image.LANCZOS)
        ox, oy = PX0 + (box_w - th.width) // 2, VY + 70 + (box_h - th.height) // 2
        alpha = ease(since / 0.3)
        bg = frame.crop((ox, oy, ox + th.width, oy + th.height))
        frame.paste(Image.blend(bg, th, alpha), (ox, oy))
        d.rectangle((ox - 2, oy - 2, ox + th.width + 1, oy + th.height + 1), outline=ACTIVE, width=3)
        d.text((PX0 + box_w / 2, VY + 392), f"{sent.width} × {sent.height} px", font=font("Medium", 22), fill=FAINT, anchor="mm")
        d.text((PX0, VY + 420), "How many pens?", font=font("SemiBold", 32), fill=WHITE)
        fill_p = ease((since - 0.35) / 0.7)
        win = node["choice"]
        for j, key in enumerate(("none", "one", "several")):
            y = VY + 478 + j * 58
            p = node["p"][key]
            won = key == win and since >= 1.1
            d.text((PX0, y), key, font=font("SemiBold" if won else "Medium", 28), fill=WHITE if won else DIM)
            bx0, bx1 = PX0 + 130, PX1 - 80
            d.rounded_rectangle((bx0, y + 4, bx1, y + 30), radius=8, fill=(44, 47, 54))
            wfill = (bx1 - bx0) * p * fill_p
            if wfill > 2:
                d.rounded_rectangle((bx0, y + 4, bx0 + wfill, y + 30), radius=8, fill=PINK if won else (110, 115, 125))
            d.text((PX1, y + 17), f"{round(100 * p * fill_p)}%", font=font("Medium", 24), fill=WHITE if won else DIM, anchor="rm")
        d.text((PX0, VY + VH - 62), f"requests  {k + 1}", font=font("Medium", 26), fill=DIM)
        d.text((PX1, VY + VH - 62), f"${self.cost[k]:.5f}", font=font("Medium", 26), fill=DIM, anchor="ra")

    def panel_result(self, frame, t):
        d = ImageDraw.Draw(frame)
        since = t - self.result_t0
        d.text((PX0, VY + 26), "Pens found", font=font("SemiBold", 26), fill=DIM)
        order = sorted(self.final, key=lambda b: b[0])
        for k, b in enumerate(order):
            a = ease((since - 0.8 - 0.25 * k) / 0.4)
            if a <= 0:
                continue
            pad = 18
            crop = self.photo.crop((int(b[0] - pad), int(b[1] - pad), int(b[2] + pad), int(b[3] + pad)))
            box_w, box_h = PX1 - PX0, 190
            sc = min(box_w / crop.width, box_h / crop.height)
            th = crop.resize((max(1, int(crop.width * sc)), max(1, int(crop.height * sc))), Image.LANCZOS)
            ox, oy = PX0 + (box_w - th.width) // 2, VY + 80 + k * 215 + (box_h - th.height) // 2
            bg = frame.crop((ox, oy, ox + th.width, oy + th.height))
            frame.paste(Image.blend(bg, th, a), (ox, oy))
            d.rectangle((ox - 2, oy - 2, ox + th.width + 1, oy + th.height + 1), outline=PINK, width=3)

    def captions_at(self, frame, t):
        d = ImageDraw.Draw(frame)
        for t0, t1, text in self.captions:
            if t0 <= t < t1:
                a = ease((t - t0) / 0.3) * (1 - ease((t - t1 + 0.3) / 0.3))
                col = tuple(int(lerp(BG[i], WHITE[i], a)) for i in range(3))
                d.text((VX + VW / 2 if t < self.tries_t0 else W / 2, 930), text, font=font("SemiBold", 36), fill=col, anchor="mm")

    def header(self, frame, t):
        d = ImageDraw.Draw(frame)
        d.text((VX, 52), "Jev 1.14 + halving", font=font("SemiBold", 30), fill=WHITE, anchor="lm")
        d.text((VX + 300, 52), "finding every object with a model that draws one box per question",
               font=font("Medium", 24), fill=FAINT, anchor="lm")

    def tries_scene(self, t):
        """The main layout shrinks into the first of three tries, side by side."""
        since = t - self.tries_t0
        frame = Image.new("RGB", (W, H), BG)
        self.header(frame, t)
        tw = 580
        th = round(tw * self.PH / self.PW)
        gap = (W - 3 * tw) // 4
        d = ImageDraw.Draw(frame)
        for k, run in enumerate(self.tries):
            x = gap + k * (tw + gap)
            y = 300
            if k == 0:  # the big photo moving into place
                p = ease(since / 0.9)
                rx, ry, rw = lerp(VX, x, p), lerp(VY, y, p), lerp(VW, tw, p)
            else:
                p = ease((since - 0.6 - 0.35 * (k - 1)) / 0.5)
                if p <= 0:
                    continue
                rx, ry, rw = x, y + (1 - p) * 40, tw
            rh = rw * self.PH / self.PW
            im = self.photo.resize((int(rw), int(rh)), Image.LANCZOS)
            dd = ImageDraw.Draw(im)
            s = rw / self.PW
            for b in run["final"]:
                dd.rectangle((b[0] * s, b[1] * s, b[2] * s, b[3] * s), outline=PINK, width=max(2, int(4 * s)))
                cx, cy = (b[0] + b[2]) / 2 * s, (b[1] + b[3]) / 2 * s
                r = max(4, 9 * s)
                dd.ellipse((cx - r, cy - r, cx + r, cy + r), fill=YELLOW, outline=(0, 0, 0))
            if k > 0:
                im = Image.blend(Image.new("RGB", im.size, BG), im, p)
            frame.paste(im, (int(rx), int(ry)))
            if since > 1.2 + 0.35 * k:
                a = ease((since - 1.2 - 0.35 * k) / 0.4)
                col = tuple(int(lerp(BG[i], DIM[i], a)) for i in range(3))
                d.text((x + tw / 2, y + th + 40), f"try {k + 1}  ·  {len(run['final'])} pens  ·  {run['requests']} requests",
                       font=font("Medium", 28), fill=col, anchor="mm")
        self.captions_at(frame, t)
        fade = ease((self.duration - t) / 0.6)
        return frame if fade >= 1 else Image.blend(Image.new("RGB", (W, H), BG), frame, fade)

    def frame(self, t):
        if t >= self.tries_t0:
            return self.tries_scene(t)
        frame = Image.new("RGB", (W, H), BG)
        self.header(frame, t)
        cam = self.camera_at(t)
        show_cuts = 1 - ease((t - self.result_t0) / 0.8) if t >= self.result_t0 else 1.0
        vp = self.viewport(t, cam, show_cuts)
        intro = ease(t / 0.8)
        if intro < 1:
            vp = Image.blend(Image.new("RGB", vp.size, BG), vp, intro)
        frame.paste(vp, (VX, VY))
        self.panel(frame, t)
        self.captions_at(frame, t)
        return frame


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DIR / "jev-halving-pens.mp4")
    ap.add_argument("--preview", action="store_true")
    a = ap.parse_args()
    m = Movie()
    if a.preview:
        out = DIR / "video-preview"
        out.mkdir(exist_ok=True)
        times = [1.5, 4.4, 7.6, 10.0, 15.0, 19.6, 24.0, 26.3, 30.5, 33.0, m.tries_t0 + 0.5, m.tries_t0 + 4.0]
        for t in times:
            m.frame(t).save(out / f"t{t:05.1f}.png")
        print(out, f"duration {m.duration:.1f} s")
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
