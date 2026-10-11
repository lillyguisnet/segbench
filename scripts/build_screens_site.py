# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow>=10"]
# ///
"""Build the screen-clicks page: pick a question and a model, see where it clicked.

    uv run scripts/build_screens_site.py --serve     # writes results/views/site-screens/, serves it
    uv run scripts/build_site.py                     # builds the photo site with this page in screens/

Static, no outside requests, same look as the points site (scripts/site.html):
scripts/screens_site.html with the data written in. Shows each question's
answer key (boxes and circles checked by a person) and every model's clicks,
each marked right, repeated (on a target already clicked) or wrong, with the
same matching as the scorer (scripts/score_screens.py). All 3 tries of each
model can be stepped through.

Screenshots are stored as lossless WebP: the pixels the models were sent.
Local models' runs (results/run-screens-local-*.jsonl) are included when
they exist.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from segbench import points  # noqa: E402
from segbench.chart_data import describe  # noqa: E402
from segbench.point_score import maximum_matching  # noqa: E402
from segbench.screens import BY_KEY as TASKS, ROUNDS, hits  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from build_site import REPO, logo  # noqa: E402

KEYS = {1: ROOT / "annotations/screens/key.json", 2: ROOT / "annotations/screens/key-round2.json"}
RUNS = ["results/run-screens-1.jsonl", "results/run-screens-2.jsonl"]
CHARTS = {1: ROOT / "charts/screens-1/language-models/find.png", 2: ROOT / "charts/screens-2/language-models/find.png"}
SHORT = {"chattering": "Fold Lilly’s group", "x": "Reply to Michael", "viewer": "Close the viewer",
         "favorites": "Add to favorites", "shopify": "Tick lamb-only",
         "fold-groups": "Fold all but segbench", "news-1000": "News > 1,000 posts",
         "onedrive-folders": "OneDrive folders", "dirty-labels": "D labels", "sold-out": "Nothing left to sell"}
THUMB_SIDE = 220


def runs_present() -> list[Path]:
    found = [ROOT / r for r in RUNS] + sorted(p for p in (ROOT / "results").glob("run-screens-local-*.jsonl")
                                             if ".raw-" not in p.name)
    return [p for p in found if p.exists()]


def build(out: Path) -> Path:
    keys = {r: json.loads(p.read_text()) for r, p in KEYS.items()}
    for r, k in keys.items():
        if not k["status"].startswith("accepted"):
            sys.exit(f"round {r} answer key is not accepted yet ({k['status']})")
    round_of = {t.key: r for r, ts in ROUNDS.items() for t in ts}

    rows = []
    for path in runs_present():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r.get("task") in round_of and "error" not in r:
                rows.append(r)

    if out.exists():
        shutil.rmtree(out)
    (out / "img").mkdir(parents=True)

    shots, tasks, answers, models = {}, [], {}, {}
    for rnd in sorted(ROUNDS):
        for t in ROUNDS[rnd]:
            key = keys[rnd]["tasks"][t.key]
            if t.image not in shots:
                with Image.open(ROOT / "images" / "screenshots" / t.image) as im:
                    rgb = im.convert("RGB")
                stem = t.image.removesuffix(".png")
                rgb.save(out / "img" / f"{stem}.webp", lossless=True, method=6)
                thumb = rgb.copy()
                thumb.thumbnail((THUMB_SIDE, THUMB_SIDE), Image.LANCZOS)
                thumb.save(out / "img" / f"{stem}.thumb.jpg", quality=80, optimize=True)
                shots[t.image] = {"img": f"img/{stem}.webp", "thumb": f"img/{stem}.thumb.jpg",
                                  "w": rgb.width, "h": rgb.height}
            targets = key["targets"]
            tasks.append({"key": t.key, "round": rnd, "name": SHORT.get(t.key, t.key), "ask": t.instruction,
                          "shot": t.image, "targets": [{k: g[k] for k in ("id", "box", "circle", "covered_by") if k in g}
                                                       for g in targets]})
            answers[t.key] = {}
            mine = sorted((r for r in rows if r["task"] == t.key), key=lambda r: (r["model"], r["repeat"]))
            for r in mine:
                W, H = r["image_size"]
                if [W, H] != key["image_size"]:
                    sys.exit(f"{t.key}: {r['model']} saw {W}x{H}, the key is for {key['image_size']}")
                clicks = [(c["x"] * W / 1000, c["y"] * H / 1000) for c in r.get("clicks", [])] if r.get("readable") else []
                edges = [hits(targets, x, y) for x, y in clicks]
                pairs = dict(maximum_matching(edges))  # click -> target
                drawn = []
                for i, (x, y) in enumerate(clicks):
                    state = "hit" if i in pairs else "dup" if edges[i] else "miss"
                    drawn.append([round(x, 1), round(y, 1), state] + ([pairs[i]] if i in pairs else []))
                tp, n, m = len(pairs), len(clicks), len(targets)
                entry = {"clicks": drawn, "right": tp, "f1": round(2 * tp / (n + m), 4) if n + m else 1.0}
                if r.get("seconds") is not None:
                    entry["seconds"] = round(r["seconds"], 2)
                if r.get("cost_usd") is not None:
                    entry["cost"] = r["cost_usd"]
                if r.get("provider") == "local-gpu":
                    entry["prompt"] = r.get("prompt")
                if not r.get("readable"):
                    entry["unreadable"] = r.get("unreadable_because", "reply not readable")
                answers[t.key].setdefault(r["model"], []).append(entry)
                label, maker, kind = describe(r["model"])
                models.setdefault(r["model"], {"label": label, "maker": maker, "kind": kind, "level": r.get("level"),
                                               "deterministic": bool(r.get("deterministic"))})

    stand_ins = [SimpleNamespace(key=k, maker=m["maker"], label=m["label"], level=m["level"]) for k, m in models.items()]
    colour = points.colours(stand_ins)
    for k in models:
        models[k]["colour"] = colour[k]
    order = [a.key for a in sorted(stand_ins, key=points.order_key)]
    # each model's score per round: mean over tries, then over questions (as on the charts)
    for k in models:
        models[k]["rounds"] = {}
        for rnd in sorted(ROUNDS):
            per_q = [sum(e["f1"] for e in answers[t.key][k]) / len(answers[t.key][k])
                     for t in ROUNDS[rnd] if k in answers[t.key]]
            if len(per_q) == len(ROUNDS[rnd]):
                models[k]["rounds"][rnd] = round(sum(per_q) / len(per_q), 4)

    charts = {}
    (out / "charts").mkdir()
    for rnd, png in CHARTS.items():
        if png.exists():
            shutil.copy(png, out / "charts" / f"round-{rnd}.png")
            charts[rnd] = f"charts/round-{rnd}.png"

    run_dates = [r["started_at"][:10] for r in rows if r.get("started_at")]
    data = {"repo": REPO, "run_date": min(run_dates), "shots": shots, "tasks": tasks, "order": order,
            "models": models, "answers": answers, "charts": charts,
            "reviewer": {r: k.get("reviewer") for r, k in keys.items()},
            "logos": {m: logo(m) for m in sorted({m["maker"] for m in models.values()})}}
    (out / "fonts").mkdir()
    for weight in ("Regular", "Medium", "SemiBold"):
        shutil.copy(ROOT / "assets" / "fonts" / f"Inter-{weight}.ttf", out / "fonts")
    shutil.copy(ROOT / "assets" / "fonts" / "OFL.txt", out / "fonts")
    html = (ROOT / "scripts" / "screens_site.html").read_text()
    (out / "index.html").write_text(html.replace("/*DATA*/null", json.dumps(data, separators=(",", ":"))))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"wrote {out} ({size / 1e6:.1f} MB: {len(tasks)} questions, {len(shots)} screenshots, {len(models)} models)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "views" / "site-screens")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=8791)
    args = ap.parse_args()
    out = build(args.out)
    if args.serve:
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(out))
        print(f"serving on http://localhost:{args.port}")
        http.server.ThreadingHTTPServer(("", args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
