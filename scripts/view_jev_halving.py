"""Pictures of a Jev + halving run on the benchmark photos: its boxes, their middles, the answer key's dots.

uv run --with pillow scripts/view_jev_halving.py [run name] [report dir]
    defaults: jev-halving-1, results/point-benchmark-v3-jev-halving
    ->  results/<report dir>-view/ (one PNG per task, try 0, + index.html)

Pink: Jev's final boxes. Yellow: their middles (the points scored).
Green dots: the answer key's points (from the scoring report's data.js),
filled when some yellow point matched them at the medium tolerance.
"""
import html
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
run = sys.argv[1] if len(sys.argv) > 1 else "jev-halving-1"
report = ROOT / (sys.argv[2] if len(sys.argv) > 2 else "results/point-benchmark-v3-jev-halving")
OUT = report.parent / (report.name + "-view")
PINK, YELLOW, GREEN = (230, 0, 160), (255, 210, 0), (0, 200, 70)


def main():
    s = (report / "data.js").read_text()
    data = json.loads(s[s.index("{"):s.rindex("}") + 1])
    recs = {(r["task"], r["repeat"]): r for r in map(json.loads, (ROOT / f"results/run-{run}.jsonl").read_text().splitlines())}
    scored = {(r["task"], int(r["repeat"])): r for r in data["records"]}
    OUT.mkdir(parents=True, exist_ok=True)
    cards = []
    for task, t in data["tasks"].items():
        rec, sc = recs[(task, 0)], scored[(task, 0)]
        im = Image.open(ROOT / t["photo"]).convert("RGB")
        w, h = im.size
        d = ImageDraw.Draw(im)
        k = max(w, h) / 1400
        matched = {m["reference_id"] for m in sc["tolerances"]["medium"]["matches"]}
        for item in rec["answer"]["items"]:
            b = item["box"]
            d.rectangle((b[0] * w / 1000, b[1] * h / 1000, b[2] * w / 1000, b[3] * h / 1000), outline=PINK,
                        width=max(1, round(2.5 * k)))
        for ref in t["refs"]:
            r = 8 * k
            d.ellipse((ref["x"] - r, ref["y"] - r, ref["x"] + r, ref["y"] + r),
                      fill=GREEN if ref["id"] in matched else None, outline=GREEN, width=max(1, round(3 * k)))
        for item in rec["answer"]["items"]:
            x, y = item["point"][0] * w / 1000, item["point"][1] * h / 1000
            r = 5 * k
            d.ellipse((x - r, y - r, x + r, y + r), fill=YELLOW, outline=(0, 0, 0))
        im.thumbnail((1800, 1800))
        im.save(OUT / f"{task}.jpg", quality=88)
        med = sc["tolerances"]["medium"]
        cards.append(f"""<section><h2>{html.escape(task)}</h2>
<p>{len(rec['answer']['items'])} boxes for {t['count']} objects · matched {med['tp']} (medium tolerance) ·
score {float(sc['score']) * 100:.1f}% (try 0; the three tries: {', '.join(f"{float(scored[(task, i)]['score']) * 100:.0f}%" for i in range(3))}) ·
{rec['requests']} requests, {rec['seconds']:.1f} s, ${rec['cost_usd']:.4f}</p>
<a href="{task}.jpg"><img src="{task}.jpg"></a></section>""")
    (OUT / "index.html").write_text(f"""<!doctype html><meta charset="utf-8"><title>Jev 1.14 + halving on the photos</title>
<style>body{{font:15px/1.45 system-ui,sans-serif;max-width:1500px;margin:1.5em auto;padding:0 1em;color:#222;background:#fafafa}}
section{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:.6em 1em 1em;margin:1.2em 0}}
img{{max-width:100%;max-height:85vh;border:1px solid #ccc}} h2{{margin:.2em 0}} p{{margin:.3em 0 .6em}}
.k span{{display:inline-block;width:1.4em;height:.8em;margin:0 .3em 0 1em;vertical-align:middle}}</style>
<h1>Jev 1.14 + halving on the benchmark photos</h1>
<p>Run <code>results/run-{html.escape(run)}.jsonl</code>, first of three tries per photo. Overall score 23.4% (point benchmark v3 scoring).</p>
<p class="k"><span style="border:2px solid rgb{PINK}"></span>Jev's box
<span style="background:rgb{YELLOW};border-radius:50%;width:.8em"></span>its middle (the point scored)
<span style="background:rgb{GREEN};border-radius:50%;width:.8em"></span>answer-key point that was matched
<span style="border:2px solid rgb{GREEN};border-radius:50%;width:.6em"></span>answer-key point missed</p>
{''.join(cards)}""")
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
