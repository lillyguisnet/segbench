"""What the feedback loop (segbench/jev_verify.py) kept and threw away, on one photo and try.

uv run --with pillow scripts/view_jev_verify.py RUN REPORT_DIR TASK [REPEAT]
    ->  <REPORT_DIR>-view/TASK-verify[-tryN].jpg

Pink: kept (the check said yes). Grey with a cross: rejected. Orange: a box found
again after a rejection and kept. Yellow: middles of the kept boxes (the points
scored). Green: the answer key (filled = matched, medium tolerance). Thin white: the cuts.
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
run, report, task = sys.argv[1], ROOT / sys.argv[2], sys.argv[3]
rep = int(sys.argv[4]) if len(sys.argv) > 4 else 0
OUT = report.parent / (report.name + "-view")
OUT.mkdir(parents=True, exist_ok=True)
PINK, YELLOW, GREEN, GREY, ORANGE = (230, 0, 160), (255, 210, 0), (0, 200, 70), (90, 90, 90), (255, 140, 0)

tree = next(t for t in map(json.loads, open(ROOT / f"results/run-{run}.trees.jsonl")) if t["task"] == task and t["repeat"] == rep)
s = (report / "data.js").read_text()
data = json.loads(s[s.index("{"):s.rindex("}") + 1])
t = data["tasks"][task]
sc = next(r for r in data["records"] if r["task"] == task and int(r["repeat"]) == rep)
matched = {m["reference_id"] for m in sc["tolerances"]["medium"]["matches"]}

im = Image.open(ROOT / t["photo"]).convert("RGB")
w, h = im.size
z = max(1.0, 1800 / max(w, h))
im = im.resize((round(w * z), round(h * z)), Image.LANCZOS)
d = ImageDraw.Draw(im)
k = max(im.size) / 1400
Z = lambda b: [v * z for v in b]  # noqa: E731
for axis, pos, lo, hi in tree["cuts"]:
    d.line(Z((pos, lo, pos, hi) if axis == "x" else (lo, pos, hi, pos)), fill=(255, 255, 255), width=max(1, round(1.2 * k)))
kept_points = []
for e in tree["verify"]:
    b = Z(e["box"])
    if e["outcome"] == "kept":
        d.rectangle(b, outline=PINK, width=max(1, round(3 * k)))
        kept_points.append(b)
        continue
    d.rectangle(b, outline=GREY, width=max(1, round(3 * k)))
    d.line((b[0], b[1], b[2], b[3]), fill=GREY, width=max(1, round(2 * k)))
    d.line((b[0], b[3], b[2], b[1]), fill=GREY, width=max(1, round(2 * k)))
    if e["outcome"] == "replaced":
        nb = Z(e["retry_box"])
        d.rectangle(nb, outline=ORANGE, width=max(1, round(3 * k)))
        kept_points.append(nb)
for ref in t["refs"]:
    r, x, y = 8 * k, ref["x"] * z, ref["y"] * z
    d.ellipse((x - r, y - r, x + r, y + r), fill=GREEN if ref["id"] in matched else None, outline=GREEN, width=max(1, round(3 * k)))
for b in kept_points:
    x, y, r = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, 5 * k
    d.ellipse((x - r, y - r, x + r, y + r), fill=YELLOW, outline=(0, 0, 0))
out = OUT / (f"{task}-verify.jpg" if rep == 0 else f"{task}-verify-try{rep}.jpg")
im.save(out, quality=90)
print(out)
