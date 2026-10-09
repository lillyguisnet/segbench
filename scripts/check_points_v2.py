"""Prompt v2 on synthetic pictures, before any benchmark photo: does every model
answer with named x/y, in the right order, on both a wide and a tall picture?

uv run --with pillow scripts/check_points_v2.py  ->  results/check-points-v2-<time>.jsonl

Six red circles and four blue squares (distractors), at known places. Pass =
one point inside each circle and no other point. A placement chosen so that
reading x and y swapped puts the points off the circles.
"""
import io, json, sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from segbench.call import call
from segbench.models import BENCHMARKED, levels_for
from segbench.point_score import read_points
from segbench.tasks import BY_KEY, prompt_v2
from run import image_settings

def picture(w, h):
    circles = [(0.12, 0.20), (0.40, 0.75), (0.65, 0.30), (0.85, 0.85), (0.30, 0.45), (0.78, 0.55)]
    squares = [(0.20, 0.80), (0.55, 0.12), (0.90, 0.20), (0.50, 0.55)]
    r = 0.045 * min(w, h)
    im = Image.new("RGB", (w, h), "white"); d = ImageDraw.Draw(im)
    for fx, fy in squares:
        d.rectangle((fx * w - r, fy * h - r, fx * w + r, fy * h + r), fill=(40, 70, 210))
    for fx, fy in circles:
        d.ellipse((fx * w - r, fy * h - r, fx * w + r, fy * h + r), fill=(215, 30, 30))
    b = io.BytesIO(); im.save(b, "JPEG", quality=92)
    return b.getvalue(), [(fx * w, fy * h) for fx, fy in circles], r

task = replace(BY_KEY["cows"], target="each red circle", absent="each green triangle")
pics = {"wide": picture(1600, 900), "tall": picture(900, 1600)}
dims = {"wide": (1600, 900), "tall": (900, 1600)}

def grade(points, circles, r, w, h):
    pts = [(p["point"][0] * w / 1000, p["point"][1] * h / 1000) for p in points]
    hit = sum(any((x - cx) ** 2 + (y - cy) ** 2 <= (r * 1.15) ** 2 for x, y in pts) for cx, cy in circles)
    extra = sum(not any((x - cx) ** 2 + (y - cy) ** 2 <= (r * 1.15) ** 2 for cx, cy in circles) for x, y in pts)
    return hit, extra

def run(job):
    m, shape = job
    data, circles, r = pics[shape]
    rec = call(m, data, "image/jpeg", prompt_v2("find", task), level="medium" if "medium" in levels_for(m.key) else None,
               **image_settings(m, "v2"))
    rec.update(check="points-v2", shape=shape)
    if "error" not in rec:
        try:
            pts, how = read_points(rec["text"], parser="lenient")
            rec["read_as"] = how
            rec["hit"], rec["extra"] = grade(pts, circles, r, *dims[shape])
            rec["named"] = '"x"' in rec["text"]
        except Exception as e:
            rec["unreadable"] = str(e)
    return rec

only = set(sys.argv[1:])
jobs = [(m, s) for m in BENCHMARKED for s in pics if not only or m.key in only]
with ThreadPoolExecutor(16) as pool:
    recs = list(pool.map(run, jobs))
out = Path("results") / f"check-points-v2-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
out.write_text("".join(json.dumps(r) + "\n" for r in recs))
for r in recs:
    if "error" in r: s = "ERROR " + r["error"][:100]
    elif "unreadable" in r: s = "UNREADABLE " + r["unreadable"][:80]
    else: s = (f"{'PASS' if r['hit'] == 6 and r['extra'] == 0 else 'fail'}  {r['hit']}/6 circles, {r['extra']} extra, "
               f"{'named x/y' if r['named'] else 'NOT named'}, read {r['read_as']}, ${r.get('cost_usd') or 0:.4f}, {r['seconds']:.0f}s")
    print(f"{r['model']:18} {r['shape']:4} {s}")
print("saved", out)
