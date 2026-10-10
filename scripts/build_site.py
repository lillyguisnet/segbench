# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow>=10"]
# ///
"""Build the points site: pick a model, see its dots on each photo.

    uv run scripts/build_site.py                 # writes results/views/site/
    uv run scripts/build_site.py --serve         # and serves it on http://localhost:8790
    scripts/publish_site.sh                      # builds, then publishes to GitHub Pages

A static site with no build tools and no outside requests: index.html (the
page, its code and the answers, from scripts/site.html), the photos made
smaller (img/), and the Inter font. Logos are written into the page from
assets/logos.

What it shows comes from segbench/points.py (the same reading and colours
as the point overlays). Time and cost are shown per call; a specialist's are left out unless it
was timed on the free GPU. Each model's first of its 3 tries is shown.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import re
import shutil
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # this script runs in its own environment (above)

from segbench import points  # noqa: E402
from segbench.brand import LOGO_DIR, LOGOS, initials  # noqa: E402
from segbench.tasks import BY_KEY as TASKS  # noqa: E402

PHOTO_SIDE = 2400  # longest side of the photos on the site
THUMB_SIDE = 360
SHORT = {"logs": "Log ends", "cows": "Cows", "fig": "Fig leaves", "tree": "Red foliage", "road": "Road",
         "dishes": "Dishes"}
REPO = "https://github.com/lillyguisnet/segbench"


def logo(maker: str) -> dict:
    """{"viewBox", "d": [...]} from assets/logos, or {"initials"} for a maker without one."""
    name = LOGOS.get(maker)
    path = LOGO_DIR / name if name else None
    if not path or not path.exists():
        return {"initials": initials(maker)}
    svg = path.read_text()
    return {"viewBox": re.search(r'viewBox="([^"]+)"', svg).group(1),
            "d": re.findall(r'<path[^>]*\sd="([^"]+)"', svg)}


def ask(prompt: str) -> str:
    """The request itself: the prompt's first line (the same in both prompt versions)."""
    return prompt.strip().split("\n")[0]


def build(out: Path, runs: list[str], track: str) -> Path:
    answers = points.load(runs, track)
    if not answers:
        sys.exit("no point answers found")
    colour = points.colours(answers)
    if out.exists():
        shutil.rmtree(out)
    (out / "img").mkdir(parents=True)
    (out / "fonts").mkdir()

    tasks, models, by_task = [], {}, {}
    for key in (k for k in TASKS if any(a.task == k for a in answers)):
        task = TASKS[key]
        with Image.open(ROOT / "images" / task.photo) as im:
            photo = im.convert("RGB")  # the upright pixel grid the models were sent
        W, H = photo.size
        small = photo.copy()
        small.thumbnail((PHOTO_SIDE, PHOTO_SIDE), Image.LANCZOS)
        small.save(out / "img" / task.photo, quality=80, optimize=True, progressive=True)
        thumb = photo.copy()
        thumb.thumbnail((THUMB_SIDE, THUMB_SIDE), Image.LANCZOS)
        thumb_name = task.photo.replace(".jpg", ".thumb.jpg")
        thumb.save(out / "img" / thumb_name, quality=78, optimize=True)
        mine = [a for a in answers if a.task == key]
        general = next((a for a in mine if a.kind == "general" and a.prompt), None)
        tasks.append({"key": key, "number": task.number, "name": SHORT.get(key, key), "photo": f"img/{task.photo}",
                      "thumb": f"img/{thumb_name}", "w": W, "h": H, "labels": list(task.labels),
                      "ask": ask(general.prompt) if general else task.target})
        by_task[key] = {}
        for a in mine:
            models.setdefault(a.key, {"label": a.label, "maker": a.maker, "kind": a.kind, "level": a.level,
                                      "colour": colour[a.key]})
            entry = {"points": [[round(p["x"], 1), round(p["y"], 1)] + ([p["label"]] if "label" in p else [])
                                for p in a.points],
                     "off": a.off_photo}
            if a.unreadable:
                entry["unreadable"], entry["raw"] = a.unreadable, a.raw
            if a.timing_trustworthy and a.seconds is not None:
                entry["seconds"] = round(a.seconds, 2)
                if a.cost_usd is not None:
                    entry["cost"] = a.cost_usd
            if a.kind == "specialist":
                entry["prompt"] = a.prompt
            by_task[key][a.key] = entry

    order = list(dict.fromkeys(a.key for a in sorted(answers, key=points.order_key)))
    makers = sorted({m["maker"] for m in models.values()})
    data = {"track": track, "runs": [Path(r).stem for r in runs],
            "run_date": min(a.started_at for a in answers if a.started_at)[:10],
            "repo": REPO, "tasks": tasks, "order": order, "models": models, "answers": by_task,
            "logos": {m: logo(m) for m in makers}}

    for weight in ("Regular", "Medium", "SemiBold"):
        shutil.copy(ROOT / "assets" / "fonts" / f"Inter-{weight}.ttf", out / "fonts")
    shutil.copy(ROOT / "assets" / "fonts" / "OFL.txt", out / "fonts")
    shutil.copy(ROOT / "images" / "LICENSE-CC-BY-4.0.txt", out / "img")
    html = (ROOT / "scripts" / "site.html").read_text()
    html = html.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    (out / "index.html").write_text(html)
    (out / ".nojekyll").write_text("")  # serve the files as they are
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"wrote {out} ({size / 1e6:.1f} MB: {len(tasks)} photos, {len(models)} models)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", default=list(points.RUNS), help="call records (.jsonl)")
    ap.add_argument("--track", default="find", choices=("find", "trick"))
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "views" / "site")
    ap.add_argument("--serve", action="store_true", help="serve the site on localhost after building it")
    ap.add_argument("--port", type=int, default=8790)
    args = ap.parse_args()
    out = build(args.out, args.runs, args.track)
    if args.serve:
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(out))
        print(f"serving on http://localhost:{args.port}")
        http.server.ThreadingHTTPServer(("", args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
