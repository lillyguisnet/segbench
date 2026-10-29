"""Two Jev + halving runs side by side on the benchmark photos: every task, every try.

uv run --with pillow scripts/view_jev_compare.py
    defaults: jev-halving-1 (pieces enlarged to 280 px) vs jev-halving-long1080 (to 1080 px)
    ->  results/jev-compare-280-vs-1080/ (one JPEG per run, task and try, + index.html)

Each side opens on its own median-scoring try of that task (neither the best
nor the worst); buttons show the other tries. Marks as scripts/view_jev_halving.py:
pink = Jev's boxes, yellow = their middles (the points scored), green = answer-key
points, filled when matched at the medium tolerance.
"""
import html
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
RUNS = [("jev-halving-1", "results/point-benchmark-v3-jev-halving", "Before: small pieces enlarged to 280 px"),
        ("jev-halving-long1080", "results/point-benchmark-v3-jev-halving-long1080", "After: small pieces enlarged to 1080 px")]
OUT = ROOT / "results/jev-compare-280-vs-1080"
PINK, YELLOW, GREEN = (230, 0, 160), (255, 210, 0), (0, 200, 70)


def load(run, report):
    s = (ROOT / report / "data.js").read_text()
    data = json.loads(s[s.index("{"):s.rindex("}") + 1])
    recs = {(r["task"], r["repeat"]): r for r in map(json.loads, (ROOT / f"results/run-{run}.jsonl").read_text().splitlines())}
    scored = {(r["task"], int(r["repeat"])): r for r in data["records"]}
    overall = sum(float(r["score"]) for r in data["records"]) / len(data["records"])
    return data, recs, scored, overall


def draw(photo, t, rec, sc, path):
    im = Image.open(ROOT / photo).convert("RGB")
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
    im.thumbnail((1400, 1400))
    im.save(path, quality=85)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    loaded = [load(run, report) for run, report, _ in RUNS]
    tasks = loaded[0][0]["tasks"]
    sections = []
    for task, t in tasks.items():
        cols = []
        for side, ((run, _, title), (_, recs, scored, _)) in enumerate(zip(RUNS, loaded)):
            tries = sorted(i for (tk, i) in scored if tk == task)
            scores = {i: float(scored[(task, i)]["score"]) for i in tries}
            median = sorted(tries, key=lambda i: scores[i])[len(tries) // 2]
            views = []
            for i in tries:
                rec, sc = recs[(task, i)], scored[(task, i)]
                name = f"{task}-{side}-{i}.jpg"
                draw(t["photo"], t, rec, sc, OUT / name)
                med = sc["tolerances"]["medium"]
                views.append({"src": name, "label": f"Try {i + 1}: {scores[i] * 100:.0f}%",
                              "info": f"{len(rec['answer']['items'])} boxes for {t['count']} objects · "
                                      f"{med['tp']} found · {rec['requests']} questions, {rec['seconds']:.1f} s, "
                                      f"${rec['cost_usd']:.4f}"})
            mean = sum(scores.values()) / len(scores) * 100
            buttons = "".join(f'<button data-i="{j}"{" class=on" if i == median else ""}>{html.escape(v["label"])}</button>'
                              for j, (i, v) in enumerate(zip(tries, views)))
            start = views[tries.index(median)]
            cols.append(f"""<div class="col" data-views='{html.escape(json.dumps(views))}'>
<h3>{html.escape(title)} <span class="avg">average {mean:.1f}%</span></h3>
<div class="tries">{buttons}</div><p class="info">{html.escape(start['info'])}</p>
<a href="{start['src']}" target="_blank"><img src="{start['src']}" loading="lazy"></a></div>""")
        sections.append(f'<section><h2>{html.escape(task)}</h2><div class="pair">{"".join(cols)}</div></section>')
    overall = " → ".join(f"{o * 100:.1f}%" for *_, o in loaded)
    (OUT / "index.html").write_text(f"""<!doctype html><meta charset="utf-8"><title>Jev + halving: 280 px vs 1080 px pieces</title>
<style>body{{font:15px/1.45 system-ui,sans-serif;max-width:2000px;margin:1.2em auto;padding:0 1em;color:#222;background:#fafafa}}
section{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:.5em 1em 1em;margin:1em 0}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:1em}} @media(max-width:900px){{.pair{{grid-template-columns:1fr}}}}
img{{width:100%;max-height:80vh;object-fit:contain;border:1px solid #ccc;background:#eee}}
h2{{margin:.2em 0 .4em}} h3{{margin:.2em 0;font-size:1em}} .avg{{font-weight:normal;color:#666;margin-left:.5em}}
.info{{margin:.3em 0 .5em;color:#444;font-size:.92em}}
button{{font:inherit;font-size:.85em;margin:0 .3em .2em 0;padding:.15em .6em;border:1px solid #bbb;border-radius:5px;background:#fff;cursor:pointer}}
button.on{{background:#222;color:#fff;border-color:#222}}
.k span{{display:inline-block;width:1.4em;height:.8em;margin:0 .3em 0 1em;vertical-align:middle}}</style>
<h1>Jev 1.14 + halving: small pieces enlarged to 280 px (left) vs 1080 px (right)</h1>
<p>Overall score {overall} (point benchmark v3 scoring, 11 photos, 3 tries each). Each side starts on its middle-scoring try; the buttons show the others.</p>
<p class="k"><span style="border:2px solid rgb{PINK}"></span>Jev's box
<span style="background:rgb{YELLOW};border-radius:50%;width:.8em"></span>its middle (the point scored)
<span style="background:rgb{GREEN};border-radius:50%;width:.8em"></span>answer-key point found
<span style="border:2px solid rgb{GREEN};border-radius:50%;width:.6em"></span>answer-key point missed</p>
{''.join(sections)}
<script>
document.querySelectorAll('.col').forEach(col => {{
  const views = JSON.parse(col.dataset.views);
  col.querySelectorAll('button').forEach(b => b.onclick = () => {{
    const v = views[+b.dataset.i];
    col.querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
    col.querySelector('img').src = v.src; col.querySelector('a').href = v.src;
    col.querySelector('.info').textContent = v.info;
  }});
}});
</script>""")
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
