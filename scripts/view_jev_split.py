"""Pictures of a probe_jev_split run: the final boxes over the synthetic picture.

uv run --with pillow --with numpy scripts/view_jev_split.py [results/probe-jev-split-*.json]
    ->  results/probe-jev-split-view/  (one PNG per picture size + index.html)
"""
import glob
import html
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_jev_split import draw  # noqa: E402
from segbench.box_read import iou  # noqa: E402

TRUTH, JEV, CUT, POINT = (0, 170, 60), (230, 0, 160), (150, 150, 150), (255, 210, 0)
OUT = Path("results/probe-jev-split-view")


def font(size):
    for f in ("DejaVuSans-Bold.ttf", "/run/current-system/sw/share/X11/fonts/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def render(run):
    w, h = run["size"]
    im, truth, masks = draw(w, h)
    im = im.convert("RGB")
    d = ImageDraw.Draw(im)
    s = max(w, h) / 1400  # line widths look the same at every size once shown ~1400 px
    for axis, pos, lo, hi in run["cuts"]:
        line = (pos, lo, pos, hi) if axis == "x" else (lo, pos, hi, pos)
        d.line(line, fill=CUT, width=max(1, round(1.5 * s)))
    for t in truth:
        d.rectangle(t, outline=TRUTH, width=max(1, round(3 * s)))
    f = font(round(22 * s))
    rows = []
    for k, b in enumerate(run["found"], 1):
        d.rectangle(b, outline=JEV, width=max(1, round(4 * s)))
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        r = 7 * s
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=POINT, outline=(0, 0, 0), width=max(1, round(s)))
        best, i = max((iou(b, t), i) for i, t in enumerate(truth))
        on = any(m[int(cy), int(cx)] for m in masks if 0 <= int(cy) < h and 0 <= int(cx) < w)
        d.text((b[0] + 4 * s, b[1] - 26 * s), str(k), fill=JEV, font=f, stroke_width=round(3 * s), stroke_fill="white")
        rows.append((k, best, i, on))
    out = OUT / f"{run['name']}.png"
    im.thumbnail((1600, 1600))
    im.save(out)
    return out.name, rows


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("results/probe-jev-split-2*.json"))[-1]
    log = json.load(open(src))
    OUT.mkdir(parents=True, exist_ok=True)
    cards = []
    for name, run in log["runs"].items():
        run["name"] = name
        png, rows = render(run)
        g = run["grade"]
        table = "".join(
            f"<tr><td>{k}</td><td>{best:.0%}</td><td>{'oval ' + str(i + 1) if best > 0 else '—'}</td>"
            f"<td>{'yes' if on else '<b>no</b>'}</td></tr>" for k, best, i, on in rows)
        cards.append(f"""<section><h2>{html.escape(name)} · {run['size'][0]}×{run['size'][1]}</h2>
<p>{g['boxes']} boxes for {g['ovals']} ovals · {g['points_on_oval']} middles on an oval ·
overlap ≥50%: {g['iou50']}, ≥75%: {g['iou75']}, ≥90%: {g['iou90']} ·
{run['requests']} requests, {run['wall_seconds']} s, ${run['dollars']:.4f}</p>
<a href="{png}"><img src="{png}" alt="{html.escape(name)}"></a>
<details><summary>each box</summary><table><tr><th>box</th><th>best overlap</th><th>with</th><th>middle on an oval</th></tr>{table}</table></details>
</section>""")
    (OUT / "index.html").write_text(f"""<!doctype html><meta charset="utf-8"><title>Jev 1.14 + halving</title>
<style>
body{{font:15px/1.45 system-ui,sans-serif;max-width:1500px;margin:1.5em auto;padding:0 1em;color:#222;background:#fafafa}}
section{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:.6em 1em 1em;margin:1.2em 0}}
img{{max-width:100%;max-height:85vh;border:1px solid #ccc}} h2{{margin:.2em 0}} p{{margin:.3em 0 .6em}}
.k span{{display:inline-block;width:1.6em;height:.8em;margin:0 .3em 0 1em;vertical-align:middle}}
table{{border-collapse:collapse;margin-top:.5em}} td,th{{border:1px solid #ddd;padding:2px 10px;text-align:left}}
</style>
<h1>Jev 1.14 + halving: final boxes</h1>
<p>Synthetic test picture (7 red ovals, 4 blue squares), no benchmark photo. Run: <code>{html.escape(src)}</code></p>
<p class="k"><span style="background:rgb{JEV}"></span>Jev's final box (numbered)
<span style="background:rgb{TRUTH}"></span>true box of each oval
<span style="background:rgb{POINT};border:1px solid #000;border-radius:50%;width:.8em"></span>middle of Jev's box (its point)
<span style="background:rgb{CUT};height:2px"></span>where the picture was cut in half</p>
{''.join(cards)}""")
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
