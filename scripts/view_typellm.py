"""Every TypeLLM answer of a run drawn on its photo: one page, dots and boxes, both entries, every try.

    uv run --with pillow scripts/view_typellm.py [--name typellm-1]
    ->  results/preview/typellm/<name>/view/index.html  (+ photos/, data.js)

Reads the preview scores of scripts/score_typellm.py; each entry is shown with
the prompt it kept. Dots: right = matched to an answer-key dot at the medium
tolerance (the published scoring's middle level). Boxes: right = matched to an
answer-key box at 50% overlap (one to one, maximum matching, as scored). The
answer key's objects nobody matched are drawn as "missed".
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from segbench.box_read import iou  # noqa: E402
from segbench.point_score import maximum_matching  # noqa: E402
from segbench.tasks import BY_KEY  # noqa: E402

TASKS = ["logs", "cows", "fig", "dishes", "ants", "beetles", "birds", "mushrooms", "jars", "pinkleaves", "whitecow"]
LEVELS = {"none": "TypeLLM", "count-medium": "TypeLLM + count first"}
LONG = 1600


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="typellm-1")
    a = ap.parse_args()
    pre = ROOT / "results/preview/typellm" / a.name
    out = pre / "view"
    (out / "photos").mkdir(parents=True, exist_ok=True)
    report = json.loads((pre / "report.json").read_text())
    key = json.loads((ROOT / "annotations/boxes/answer-key-boxes-v1.json").read_text())["tasks"]
    runs = {}
    for v in ("typellm-v2", "typellm-v1-hd", "typellm-box-a", "typellm-box-free"):
        for r in map(json.loads, (ROOT / f"results/run-{a.name}-{v}.jsonl").read_text().splitlines()):
            runs[(v, r["level"], r["task"], r["repeat"])] = r

    # photos and their answer keys
    s = (pre / "points-typellm-v2/data.js").read_text()
    pt_tasks = json.loads(s[s.index("{"):s.rindex("}") + 1])["tasks"]
    tasks = {}
    for t in TASKS:
        meta = pt_tasks[t]
        im = Image.open(ROOT / meta["photo"])
        im.load()
        assert im.size == (meta["width"], meta["height"]), t
        k = LONG / max(im.size)
        im.convert("RGB").resize((round(im.width * k), round(im.height * k)), Image.LANCZOS).save(
            out / "photos" / f"{t}.jpg", quality=85)
        tasks[t] = {"photo": f"photos/{t}.jpg", "w": im.width, "h": im.height, "question": BY_KEY[t].target,
                    "dots": [{"id": x["id"], "x": x["x"], "y": x["y"], "r": x["radii"]["medium"]} for x in meta["refs"]],
                    "boxes": [{"id": o["id"], "box": o["box"]} for o in key[t]["objects"] if o["status"] == "accepted"]}

    views = []
    for level, label in LEVELS.items():
        # dots
        prompt = report[f"points typellm@{level}"]["prompt"]
        for r in map(json.loads, (pre / f"points-{prompt}/scored.jsonl").read_text().splitlines()):
            if r["level"] != level:
                continue
            t = tasks[r["task"]]
            matches = {m["prediction"]: m["reference_id"] for m in r["tolerances"]["medium"]["matches"]}
            preds = [{"x": p["point"][0] * t["w"] / 1000, "y": p["point"][1] * t["h"] / 1000, "ok": i in matches}
                     for i, p in enumerate(r.get("predictions") or [])]
            run = runs[(prompt, level, r["task"], int(r["repeat"]))]
            views.append({"bench": "dots", "entry": label, "task": r["task"], "try": int(r["repeat"]),
                          "prompt": prompt, "score": r["score"], "preds": preds, "count": run.get("count"),
                          "hit": sorted(set(matches.values())), "seconds": run["seconds"]})
        # boxes
        prompt = report[f"boxes typellm@{level}"]["prompt"]
        for r in map(json.loads, (pre / "boxes/scored.jsonl").read_text().splitlines()):
            if r["level"] != level or r["prompt_version"] != prompt:
                continue
            t = tasks[r["task"]]
            pred = [[b["box"][0] * t["w"] / 1000, b["box"][1] * t["h"] / 1000, b["box"][2] * t["w"] / 1000,
                     b["box"][3] * t["h"] / 1000] for b in r["boxes"]]
            refs = [x["box"] for x in t["boxes"]]
            edges = [[j for j, q in sorted(enumerate(refs), key=lambda x: -iou(p, x[1])) if iou(p, q) >= 0.5 - 1e-9]
                     for p in pred]
            pairs = dict(maximum_matching(edges))
            run = runs[(prompt, level, r["task"], r["repeat"])]
            views.append({"bench": "boxes", "entry": label, "task": r["task"], "try": r["repeat"], "prompt": prompt,
                          "score": r["scores"]["score"], "count": run.get("count"), "seconds": run["seconds"],
                          "preds": [{"box": [round(v, 1) for v in p], "ok": i in pairs} for i, p in enumerate(pred)],
                          "hit": sorted(t["boxes"][j]["id"] for j in pairs.values())})
    assert len(views) == 2 * 2 * 11 * 3, len(views)
    (out / "data.js").write_text("window.VIEW=" + json.dumps({"tasks": tasks, "views": views}) + ";\n")
    import shutil
    (out / "charts").mkdir(exist_ok=True)
    for bench in ("points", "boxes"):
        for variant in ("with-specialists", "language-models", "with-local-models"):
            src = ROOT / "charts/preview-typellm" / bench / variant / "find.png"
            if src.exists():
                shutil.copy(src, out / "charts" / f"{bench}-{variant}.png")
    (out / "index.html").write_text(PAGE)
    print(f"{len(views)} pictures -> {out.relative_to(ROOT)}/index.html")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TypeLLM on the segbench photos</title>
<style>
:root{--ok:#18b45b;--bad:#ff2d6f;--miss:#ffd400;--bg:#f6f7f9;--fg:#15171a;--mut:#6b7280;--card:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#121417;--fg:#eceef1;--mut:#9aa1ab;--card:#1c1f24}}
body{margin:0;font:15px/1.4 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--fg)}
header{position:sticky;top:0;z-index:5;background:var(--bg);padding:14px 20px 10px;border-bottom:1px solid #8883}
h1{font-size:20px;margin:0 0 8px}
.row{display:flex;flex-wrap:wrap;gap:14px;align-items:center}
.seg{display:inline-flex;border:1px solid #8886;border-radius:8px;overflow:hidden}
.seg button{border:0;background:transparent;color:var(--fg);padding:6px 12px;font:inherit;cursor:pointer}
.seg button.on{background:var(--fg);color:var(--bg)}
.legend{color:var(--mut);font-size:13px;display:flex;gap:14px;flex-wrap:wrap}
.sw{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:5px;vertical-align:-1px}
.note{color:var(--mut);font-size:13px;margin-top:6px}
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:16px;padding:16px 20px 40px}
.card{background:var(--card);border-radius:10px;overflow:hidden;box-shadow:0 1px 3px #0002}
.card.big{grid-column:1/-1}
.pic{position:relative;cursor:zoom-in;line-height:0}
.card.big .pic{cursor:zoom-out}
.pic img{width:100%;height:auto;display:block}
.pic svg{position:absolute;inset:0;width:100%;height:100%}
.cap{padding:8px 12px 10px}
.cap b{font-size:15px}
.stats{color:var(--mut);font-size:13px;margin-top:2px}
.score{float:right;font-weight:700;font-size:16px}
</style></head><body>
<header>
<h1>TypeLLM's answers on the 11 benchmark photos</h1>
<div class="row">
 <span class="seg" id="bench"><button data-v="dots" class="on">Dots</button><button data-v="boxes">Boxes</button></span>
 <span class="seg" id="entry"><button data-v="TypeLLM" class="on">TypeLLM</button><button data-v="TypeLLM + count first">TypeLLM + count first</button></span>
 <span class="seg" id="try"><button data-v="0" class="on">Try 1</button><button data-v="1">Try 2</button><button data-v="2">Try 3</button></span>
 <span class="legend"><span><i class="sw" style="background:var(--ok)"></i>right</span><span><i class="sw" style="background:var(--bad)"></i>wrong or extra</span><span><i class="sw" style="background:transparent;border:2px dashed var(--miss)"></i>missed (answer key)</span></span>
</div>
<div class="note" id="note"></div>
</header>
<details open style="padding:14px 20px 0"><summary style="cursor:pointer;font-weight:600">The charts with TypeLLM added (preview, not published)</summary>
<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:16px;margin-top:12px">
<figure style="margin:0"><img src="charts/points-with-specialists.png" style="width:100%"><figcaption class="note">Dots, every model</figcaption></figure>
<figure style="margin:0"><img src="charts/points-language-models.png" style="width:100%"><figcaption class="note">Dots, online models only</figcaption></figure>
<figure style="margin:0"><img src="charts/boxes-with-local-models.png" style="width:100%"><figcaption class="note">Boxes, every model</figcaption></figure>
<figure style="margin:0"><img src="charts/boxes-language-models.png" style="width:100%"><figcaption class="note">Boxes, online models only</figcaption></figure>
</div><p class="note">"TypeLLM + count first" is our add-on: a thinking "how many?" question asked before the list (TypeLLM cannot think while writing a list).</p></details>
<main id="grid"></main>
<script src="data.js"></script>
<script>
const V=window.VIEW, st={bench:"dots",entry:"TypeLLM",try:0};
for(const id of ["bench","entry","try"]) document.getElementById(id).onclick=e=>{
  const b=e.target.closest("button"); if(!b) return;
  st[id]=id==="try"?+b.dataset.v:b.dataset.v;
  for(const x of e.currentTarget.children) x.classList.toggle("on",x===b); draw();};
function draw(){
  const vs=V.views.filter(v=>v.bench===st.bench&&v.entry===st.entry&&v.try===st.try);
  const order=Object.keys(V.tasks); vs.sort((a,b)=>order.indexOf(a.task)-order.indexOf(b.task));
  const mean=vs.reduce((s,v)=>s+v.score,0)/vs.length;
  document.getElementById("note").textContent=
    (st.bench==="dots"?"Right = within the scoring's middle distance of an answer-key dot. ":"Right = overlaps an answer-key box by at least 50%. ")+
    `Prompt kept: ${vs[0].prompt}. This try's mean score: ${(mean*100).toFixed(1)}% (the chart shows the mean of 3 tries`+
    (st.bench==="boxes"?", and box scores average 4 overlap levels, so they are stricter than the 50% shown here).":").")+
    (st.entry!=="TypeLLM"?" \"Counted\" = what TypeLLM answered to our first, thinking question \"how many?\".":"");
  const g=document.getElementById("grid"); g.innerHTML="";
  for(const v of vs){
    const t=V.tasks[v.task], sw=Math.max(t.w,t.h)/400, hit=new Set(v.hit);
    let s="";
    if(v.bench==="dots"){
      for(const d of t.dots) if(!hit.has(d.id)) s+=`<circle cx="${d.x}" cy="${d.y}" r="${d.r}" fill="none" stroke="#ffd400" stroke-width="${sw}" stroke-dasharray="${sw*3} ${sw*2}"/>`;
      for(const p of v.preds) s+=`<circle cx="${p.x}" cy="${p.y}" r="${sw*3.2}" fill="${p.ok?"#18b45b":"#ff2d6f"}" stroke="#fff" stroke-width="${sw}"/>`;
    }else{
      for(const b of t.boxes) if(!hit.has(b.id)){const [x0,y0,x1,y1]=b.box; s+=`<rect x="${x0}" y="${y0}" width="${x1-x0}" height="${y1-y0}" fill="none" stroke="#ffd400" stroke-width="${sw}" stroke-dasharray="${sw*3} ${sw*2}"/>`;}
      for(const p of v.preds){const [x0,y0,x1,y1]=p.box; s+=`<rect x="${x0}" y="${y0}" width="${Math.max(x1-x0,1)}" height="${Math.max(y1-y0,1)}" fill="none" stroke="${p.ok?"#18b45b":"#ff2d6f"}" stroke-width="${sw*1.3}"/>`;}
    }
    const right=v.preds.filter(p=>p.ok).length, n=st.bench==="dots"?t.dots.length:t.boxes.length;
    const c=document.createElement("div"); c.className="card";
    c.innerHTML=`<div class="pic"><img src="${t.photo}" loading="lazy"><svg viewBox="0 0 ${t.w} ${t.h}" preserveAspectRatio="none">${s}</svg></div>
      <div class="cap"><span class="score">${(v.score*100).toFixed(0)}%</span><b>${v.task}</b>: ${t.question}
      <div class="stats">${n} in the answer key · ${v.preds.length} given · ${right} right · ${v.preds.length-right} wrong/extra · ${n-hit.size} missed${v.count!=null?` · counted ${v.count}`:""} · ${v.seconds.toFixed(1)} s</div></div>`;
    c.querySelector(".pic").onclick=()=>{c.classList.toggle("big"); if(c.classList.contains("big")) c.scrollIntoView({behavior:"smooth",block:"start"});};
    g.appendChild(c);
  }
}
draw();
</script></body></html>
"""

if __name__ == "__main__":
    main()
