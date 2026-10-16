"""One benchmark photo with every halving cut of a Jev + halving try drawn on it.

uv run --with pillow scripts/view_jev_halving_cuts.py TASK [REPEAT] [RUN] [REPORT DIR]
    defaults: 0, jev-halving-1, results/point-benchmark-v3-jev-halving
    ->  results/<report dir>-view/TASK-cuts[-tryN].jpg

Cut lines: white, thicker for earlier cuts. Final pieces tinted by Jev's answer:
none (grey), one (left as is), several at the deepest level, boxed anyway (orange).
Dashed cyan: joined boxes asked again (the recheck). Pink: final boxes; yellow: their
middles; green dots: answer-key points (filled = matched, medium tolerance).
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
task = sys.argv[1]
rep = int(sys.argv[2]) if len(sys.argv) > 2 else 0
RUN = sys.argv[3] if len(sys.argv) > 3 else "jev-halving-1"
REPORT = ROOT / (sys.argv[4] if len(sys.argv) > 4 else "results/point-benchmark-v3-jev-halving")
OUT = REPORT.parent / (REPORT.name + "-view")
OUT.mkdir(parents=True, exist_ok=True)
PINK, YELLOW, GREEN, CYAN = (230, 0, 160), (255, 210, 0), (0, 200, 70), (0, 220, 255)
TINT = {"none": (40, 40, 40, 120), "one": None, "forced": (255, 140, 0, 90)}

tree = next(r for r in map(json.loads, open(ROOT / f"results/run-{RUN}.trees.jsonl"))
            if r["task"] == task and r["repeat"] == rep)
rec = next(r for r in map(json.loads, open(ROOT / f"results/run-{RUN}.jsonl"))
           if r["task"] == task and r["repeat"] == rep)
s = (REPORT / "data.js").read_text()
data = json.loads(s[s.index("{"):s.rindex("}") + 1])
t = data["tasks"][task]
sc = next(r for r in data["records"] if r["task"] == task and int(r["repeat"]) == rep)
matched = {m["reference_id"] for m in sc["tolerances"]["medium"]["matches"]}

im = Image.open(ROOT / t["photo"]).convert("RGB")
w, h = im.size
z = max(1.0, 1800 / max(w, h))  # draw at ~1800 px so thin lines stay visible
im = im.resize((round(w * z), round(h * z)), Image.LANCZOS)
k = max(im.size) / 1400

nodes = [n for n in tree["nodes"] if not n.get("recheck")]
regions = {tuple(n["region"]) for n in nodes}
over = Image.new("RGBA", im.size, (0, 0, 0, 0))
od = ImageDraw.Draw(over)
for n in nodes:
    r = n["region"]
    kids = any(tuple(m["region"]) != tuple(r) and m["depth"] == n["depth"] + 1
               and m["region"][0] >= r[0] and m["region"][2] <= r[2] and m["region"][1] >= r[1] and m["region"][3] <= r[3]
               for m in nodes)
    if kids:
        continue
    kind = "forced" if n["choice"] == "several" else n["choice"]
    if TINT[kind]:
        od.rectangle([v * z for v in r], fill=TINT[kind])
im = Image.alpha_composite(im.convert("RGBA"), over).convert("RGB")
d = ImageDraw.Draw(im)

depth_of = {}
for n in nodes:
    depth_of[tuple(n["region"])] = n["depth"]
def cut_depth(axis, pos, lo, hi):
    for n in nodes:
        x0, y0, x1, y1 = n["region"]
        if n["choice"] == "several" and (
                (axis == "x" and (y0, y1) == (lo, hi) and x0 < pos < x1) or (axis == "y" and (x0, x1) == (lo, hi) and y0 < pos < y1)):
            return n["depth"]
    return MAXD
MAXD = rec.get("max_depth", 7)
for axis, pos, lo, hi in sorted(tree["cuts"], key=lambda c: -cut_depth(*c)):
    line = (pos * z, lo * z, pos * z, hi * z) if axis == "x" else (lo * z, pos * z, hi * z, pos * z)
    wd = max(1.0, (7 - min(7, cut_depth(axis, pos, lo, hi))) * 0.9 + 1.2) * k
    d.line(line, fill=(0, 0, 0), width=max(2, round(wd + 2 * k)))
    d.line(line, fill=(255, 255, 255), width=max(1, round(wd)))

def dashed(b, col, wd, seg=10):
    x0, y0, x1, y1 = b
    for a, c in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        L = max(abs(c[0] - a[0]), abs(c[1] - a[1]))
        steps = max(1, int(L // (2 * seg)))
        for j in range(steps):
            p, q = j * 2 * seg / L, min(1, (j * 2 + 1) * seg / L) if L else 1
            d.line((a[0] + (c[0] - a[0]) * p, a[1] + (c[1] - a[1]) * p,
                    a[0] + (c[0] - a[0]) * q, a[1] + (c[1] - a[1]) * q), fill=col, width=wd)

for n in tree["nodes"]:
    if n.get("recheck"):
        dashed([v * z for v in n["region"]], CYAN, max(1, round(2 * k)), seg=6 * k)
for item in rec["answer"]["items"]:
    b = item["box"]
    d.rectangle((b[0] * w / 1000 * z, b[1] * h / 1000 * z, b[2] * w / 1000 * z, b[3] * h / 1000 * z),
                outline=PINK, width=max(1, round(3 * k)))
for ref in t["refs"]:
    r = 8 * k
    x, y = ref["x"] * z, ref["y"] * z
    d.ellipse((x - r, y - r, x + r, y + r), fill=GREEN if ref["id"] in matched else None, outline=GREEN, width=max(1, round(3 * k)))
for item in rec["answer"]["items"]:
    x, y = item["point"][0] * w / 1000 * z, item["point"][1] * h / 1000 * z
    r = 5 * k
    d.ellipse((x - r, y - r, x + r, y + r), fill=YELLOW, outline=(0, 0, 0))
out = OUT / (f"{task}-cuts.jpg" if rep == 0 else f"{task}-cuts-try{rep}.jpg")
im.save(out, quality=90)
leaves = {}
print(out, im.size, "cuts", len(tree["cuts"]), "requests", rec["requests"], "boxes", len(rec["answer"]["items"]),
      "matched", len(matched), "of", t["count"], "forced", rec.get("forced"), "max depth", rec.get("max_depth_reached"))
