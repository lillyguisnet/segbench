"""Which image-detail settings each provider honours, measured by input tokens.

uv run scripts/probe_detail.py   ->  results/detail-probe-<time>.jsonl

A synthetic 4000x2252 picture (never a benchmark photo), prompt "Reply OK",
minimum thinking. A setting counts as honoured when it raises the image's
input tokens without an error. No pointing is scored here.
"""
import io, json, random, sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from segbench.call import call
from segbench.models import BENCHMARKED

rng = random.Random(7)
im = Image.new("RGB", (4000, 2252), (235, 235, 230))
d = ImageDraw.Draw(im)
for _ in range(400):
    x, y, r = rng.randrange(4000), rng.randrange(2252), rng.randrange(6, 40)
    d.ellipse((x - r, y - r, x + r, y + r), fill=(rng.randrange(256), rng.randrange(256), rng.randrange(256)))
buf = io.BytesIO(); im.save(buf, "JPEG", quality=90); data = buf.getvalue()

def settings(m):
    out = [("default", {})]
    p = m.route.split(":")[0]
    if p == "gemini":
        out += [("media ULTRA_HIGH", {"media_resolution": "MEDIA_RESOLUTION_ULTRA_HIGH"})]
    elif p == "claude-code":
        pass  # no image-detail setting on Anthropic's API
    else:
        out += [("detail high", {"detail": "high"})]
    return out

only = set(sys.argv[1:])
jobs = [(m, name, kw) for m in BENCHMARKED for name, kw in settings(m) if not only or m.key in only]
def run(job):
    m, name, kw = job
    r = call(m, data, "image/jpeg", "Reply with the single word OK.", level="min", **kw)
    r["probe"] = name
    return r
with ThreadPoolExecutor(16) as pool:
    recs = list(pool.map(run, jobs))
out = Path("results") / f"detail-probe-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
out.write_text("".join(json.dumps(r) + "\n" for r in recs))
for r in recs:
    u = r.get("usage") or {}
    print(f"{r['model']:18} {r['probe']:18} " + (f"ERROR {r['error'][:110]}" if "error" in r else f"input {u.get('input_tokens')}"))
print("saved", out)
