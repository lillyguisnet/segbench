"""Beetles, Jev + halving (up to 15 cuts) + feedback loop: a 2x2 page, 280 px vs 1080 px pieces,
before vs after the check, every try.

uv run --with pillow scripts/view_jev_beetles_grid.py   (after scripts/view_jev_verify.py on both runs, tries 0-2)
    ->  results/jev-beetles-grid-280-vs-1080/index.html

Rows: before the check (the halving's boxes), after it (scripts/view_jev_verify.py's pictures).
Columns: pieces enlarged to 280 px, to 1080 px. Every picture has its own try buttons; all start on
their column's median try after the check (so a column starts with the same try top and bottom).
"""
import html
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/jev-beetles-grid-280-vs-1080"
PINK, YELLOW, GREEN = (230, 0, 160), (255, 210, 0), (0, 200, 70)
COLS = [("Pieces enlarged to 280 px", "jev-halving-beetles-depth15-verify", "point-benchmark-v3-jev-beetles-depth15-verify"),
        ("Pieces enlarged to 1080 px", "jev-halving-beetles-depth15-verify-long1080",
         "point-benchmark-v3-jev-beetles-depth15-verify-long1080")]


def report(d):
    s = (ROOT / "results" / d / "data.js").read_text()
    data = json.loads(s[s.index("{"):s.rindex("}") + 1])
    return data["tasks"]["beetles"], {int(r["repeat"]): r for r in data["records"] if r["task"] == "beetles"}


def draw_before(run, before_report, rep, path):
    """The halving's boxes before the check, in the style of view_jev_verify (cuts, 1800 px)."""
    t, scored = report(before_report)
    tree = next(x for x in map(json.loads, open(ROOT / f"results/run-{run}.trees.jsonl")) if x["repeat"] == rep)
    matched = {m["reference_id"] for m in scored[rep]["tolerances"]["medium"]["matches"]}
    im = Image.open(ROOT / t["photo"]).convert("RGB")
    w, h = im.size
    z = max(1.0, 1800 / max(w, h))
    im = im.resize((round(w * z), round(h * z)), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    k = max(im.size) / 1400
    Z = lambda b: [v * z for v in b]  # noqa: E731
    for axis, pos, lo, hi in tree["cuts"]:
        d.line(Z((pos, lo, pos, hi) if axis == "x" else (lo, pos, hi, pos)), fill=(255, 255, 255), width=max(1, round(1.2 * k)))
    boxes = [Z(b) for b in tree["halving_boxes"]]
    for b in boxes:
        d.rectangle(b, outline=PINK, width=max(1, round(3 * k)))
    for ref in t["refs"]:
        r, x, y = 8 * k, ref["x"] * z, ref["y"] * z
        d.ellipse((x - r, y - r, x + r, y + r), fill=GREEN if ref["id"] in matched else None, outline=GREEN,
                  width=max(1, round(3 * k)))
    for b in boxes:
        x, y, r = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, 5 * k
        d.ellipse((x - r, y - r, x + r, y + r), fill=YELLOW, outline=(0, 0, 0))
    im.save(path, quality=85)


def caption(sc, rec):
    m = sc["tolerances"]["medium"]
    return (f"{sc['score'] * 100:.0f}% · {m['tp']} of 22 beetles found · {sc['predicted_count'] - m['tp']} extra · "
            f"{sc['predicted_count']} boxes")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cols = []
    for c, (title, run, rep_dir) in enumerate(COLS):
        _, after = report(rep_dir)
        _, before = report(rep_dir + "-before-check")
        recs = {r["repeat"]: r for r in map(json.loads, open(ROOT / f"results/run-{run}.jsonl"))}
        views = []
        for i in range(3):
            b_name, a_name = f"before-{c}-{i}.jpg", f"after-{c}-{i}.jpg"
            draw_before(run, rep_dir + "-before-check", i, OUT / b_name)
            src = ROOT / "results" / f"{rep_dir}-view" / ("beetles-verify.jpg" if i == 0 else f"beetles-verify-try{i}.jpg")
            Image.open(src).save(OUT / a_name, quality=85)
            r = recs[i]
            views.append({"before": b_name, "after": a_name, "label": f"Try {i + 1}",
                          "bcap": caption(before[i], r), "acap": caption(after[i], r),
                          "cost": f"whole try: {r['requests']} questions, {r['seconds']:.1f} s, ${r['cost_usd']:.4f}"})
        start = sorted(range(3), key=lambda i: after[i]["score"])[1]
        mb = sum(before[i]["score"] for i in range(3)) / 3 * 100
        ma = sum(after[i]["score"] for i in range(3)) / 3 * 100
        cols.append(dict(title=title, views=views, start=start, mb=mb, ma=ma))

    def cell(n, c, row):
        v = c["views"][c["start"]]
        avg = c["mb"] if row == "before" else c["ma"]
        buttons = "".join(f'<button data-i="{i}"{" class=on" if i == c["start"] else ""}>{x["label"]}</button>'
                          for i, x in enumerate(c["views"]))
        return (f'<div class="cell" data-col="{n}" data-row="{row}"><div>{buttons}</div>'
                f'<p class="cap"><b>average {avg:.1f}%</b> · shown: '
                f'<span>{html.escape(v["bcap" if row == "before" else "acap"])}</span></p>'
                f'<p class="cost">{html.escape(v["cost"])}</p>'
                f'<a href="{v[row]}" target="_blank"><img src="{v[row]}"></a></div>')

    head = "".join(f'<div class="colhead"><h2>{html.escape(c["title"])}</h2></div>' for c in cols)
    grid = (f'<div></div>{head}'
            f'<div class="rowhead"><h2>Before the check</h2><p>the halving\'s boxes</p></div>'
            + "".join(cell(n, c, "before") for n, c in enumerate(cols))
            + f'<div class="rowhead"><h2>After the check</h2><p>each box shown to Jev again close up: "is this a beetle?"</p></div>'
            + "".join(cell(n, c, "after") for n, c in enumerate(cols)))
    data = json.dumps([c["views"] for c in cols])
    (OUT / "index.html").write_text(f"""<!doctype html><meta charset="utf-8"><title>Jev on the beetles: 280 vs 1080 px, before vs after the check</title>
<style>body{{font:15px/1.45 system-ui,sans-serif;max-width:2100px;margin:1em auto;padding:0 1em;color:#222;background:#fafafa}}
h1{{font-size:1.35em;margin:.2em 0}} h2{{font-size:1.05em;margin:.1em 0}}
.grid{{display:grid;grid-template-columns:9em 1fr 1fr;gap:.8em;align-items:start}}
.colhead,.cell,.rowhead{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:.5em .8em}}
.rowhead{{align-self:stretch;display:flex;flex-direction:column;justify-content:center}} .rowhead p{{color:#555;font-size:.88em;margin:.3em 0}}
img{{width:100%;max-height:78vh;object-fit:contain;display:block;background:#eee;border:1px solid #ccc}}
.cap{{margin:.3em 0 .1em;font-size:.92em}} .cost{{margin:0 0 .4em;color:#555;font-size:.85em}}
button{{font:inherit;font-size:.85em;margin:.3em .3em 0 0;padding:.15em .7em;border:1px solid #bbb;border-radius:5px;background:#fff;cursor:pointer}}
button.on{{background:#222;color:#fff;border-color:#222}}
.k span{{display:inline-block;width:1.4em;height:.8em;margin:0 .3em 0 1em;vertical-align:middle}}
@media(max-width:1000px){{.grid{{grid-template-columns:1fr 1fr}} .grid>div:first-child,.rowhead{{display:none}}}}</style>
<h1>Jev + halving (up to 15 cuts) + feedback loop, beetles photo (22 beetles)</h1>
<p>Same search on both sides; only the size small pieces are enlarged to before being sent differs. Every picture has its own try buttons. All four start on their column's middle try after the check, so each column starts with the same try top and bottom (the check runs on that try's boxes).</p>
<p class="k"><span style="border:2px solid rgb{PINK}"></span>box (kept)
<span style="border:2px solid rgb(90,90,90)"></span>rejected by the check (grey, crossed)
<span style="border:2px solid rgb(255,140,0)"></span>found again after a rejection
<span style="background:rgb{YELLOW};border-radius:50%;width:.8em"></span>point scored
<span style="background:rgb{GREEN};border-radius:50%;width:.8em"></span>beetle found
<span style="border:2px solid rgb{GREEN};border-radius:50%;width:.6em"></span>beetle missed
· thin white lines: cuts</p>
<div class="grid">{grid}</div>
<script>
const V = {data};
document.querySelectorAll('.cell').forEach(c => {{
  const n = +c.dataset.col, row = c.dataset.row;
  c.querySelectorAll('button').forEach(b => b.onclick = () => {{
    const v = V[n][+b.dataset.i];
    c.querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
    c.querySelector('img').src = v[row]; c.querySelector('a').href = v[row];
    c.querySelector('.cap span').textContent = row === 'before' ? v.bcap : v.acap;
    c.querySelector('.cost').textContent = v.cost;
  }});
}});
</script>""")
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
