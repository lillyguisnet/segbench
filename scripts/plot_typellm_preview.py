"""Preview charts: the published point v3 and box v1 charts with TypeLLM's two entries added (2026-10-17).

    uv run scripts/plot_typellm_preview.py [--name typellm-1]  ->  charts/preview-typellm/

Every published bubble is drawn from the published chart's own call records
(charts/point-benchmark-v3/*/calls.jsonl, charts/box-benchmark-v1/*/calls.jsonl),
unchanged. TypeLLM's records come from scripts/score_typellm.py's preview,
each entry with the prompt it kept: "TypeLLM" (typellm@none) and
"TypeLLM + count first" (typellm@count-medium, our pipeline). Cost and seconds
from the first four photos, as for every entrant. Not published.
"""
import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "charts/preview-typellm"
FIRST4 = ("logs", "cows", "fig", "dishes")
KEYS = {"none": "typellm", "count-medium": "typellm-count"}
# The published charts' pins, plus TypeLLM's where the placer needs help (set after looking).
POINT_SPOTS = {"molmopoint-8b": "ll", "gemini-flash-lite@medium": "below", "gemini-argon@medium": "lr",
               "sonnet@medium": "lr"}
BOX_SPOTS = {"with-local-models": {"gemini-flash-lite@medium": "ll@-10,-55", "qwen-27b@medium": "ll@-20,-100",
                                   "mk1.5@medium": "right@-3,0!"},
             "language-models": {"gemini-flash-lite@medium": "ll@-10,-30", "qwen-27b@medium": "ll@-40,-100",
                                 "mk1.5@medium": "right"}}
EXTRA_SPOTS: dict[str, dict[str, str]] = {
    # "TypeLLM + count first" joins the best-value line just left of Qwen Flash Next; its label
    # sat on MolmoPoint 8B's pinned one. MolmoPoint's label moved right of its disc.
    "points/with-specialists": {"molmopoint-8b": "below@30,2!"},
    # boxes, language models: "+ count first" sits on Claude Haiku's disc; its label went onto
    # Gemini 3.5 Flash Lite's pinned one. Moved down-left into the free space, with a line.
    "boxes/language-models": {"typellm-count": "below@-75,-80"},
}


def draw(folder: Path, calls: list[dict], spots: dict, title: str | None, expect: dict):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "calls.jsonl"
    path.write_text("".join(json.dumps(c) + "\n" for c in calls))
    here = {(f"{c['model']}@{c['level']}" if c.get("level") else c["model"]) for c in calls}
    args = [f"--label-spot={e}={s}" for e, s in spots.items() if e in here]
    cmd = ["uv", "run", str(ROOT / "scripts/draw_charts.py"), str(path), "--style", "editorial", "--track", "find",
           "--names", "all", "--out", str(folder), *args] + (["--title", title] if title else [])
    subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    plotted = {r["entrant"]: r for r in csv.DictReader(open(folder / "points.csv"))}
    for e, (q, c, s) in expect.items():
        p = plotted[e]
        for got, want in ((p["quality"], q), (p["cost_usd"], c), (p["seconds"], s)):
            assert math.isclose(float(got), want, rel_tol=1e-9, abs_tol=1e-12), (folder, e, got, want)
    print(f"{folder.relative_to(ROOT)}: {len(plotted)} bubbles; TypeLLM's checked against its preview scores")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="typellm-1")
    a = ap.parse_args()
    pre = ROOT / "results/preview/typellm" / a.name
    report = json.loads((pre / "report.json").read_text())

    # points: the scored records of each entry's kept prompt
    pt_calls, pt_expect = [], {}
    for level, key in KEYS.items():
        rep = report[f"points typellm@{level}"]
        for line in (pre / f"points-{rep['prompt']}" / "scored.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["level"] == level:
                pt_calls.append({**r, "model": key, "level": None, "entrant": key})
        pt_expect[key] = (float(rep["score"]), float(rep["cost_usd"]), float(rep["seconds"]))
    for variant in ("with-specialists", "language-models"):
        pub = [json.loads(l) for l in (ROOT / f"charts/point-benchmark-v3/{variant}/calls.jsonl").read_text().splitlines()]
        draw(OUT / "points" / variant, pub + pt_calls, {**POINT_SPOTS, **EXTRA_SPOTS.get(f"points/{variant}", {})},
             None, pt_expect)

    # boxes: per-try scores of each entry's kept prompt
    bx_calls, bx_expect = [], {}
    for level, key in KEYS.items():
        rep = report[f"boxes typellm@{level}"]
        for r in csv.DictReader(open(pre / "boxes/per-try.csv")):
            if r["entrant"] == f"typellm@{level}" and r["prompt"] == rep["prompt"]:
                timed = r["task"] in FIRST4
                bx_calls.append({"track": "find", "task": r["task"], "model": key, "level": None,
                                 "repeat": int(r["repeat"]), "score": float(r["score"]), "readable": True,
                                 "cost_usd": float(r["cost_usd"]) if timed else None,
                                 "seconds": float(r["seconds"]) if timed else None, "prompt_version": r["prompt"]})
        bx_expect[key] = (float(rep["score"]), float(rep["cost_usd_per_question_first4"]), float(rep["median_seconds_first4"]))
    title = "How well can AI draw a box around each thing in a photo?"
    for variant in ("with-local-models", "language-models"):
        pub = [json.loads(l) for l in (ROOT / f"charts/box-benchmark-v1/{variant}/calls.jsonl").read_text().splitlines()]
        draw(OUT / "boxes" / variant, pub + bx_calls, {**BOX_SPOTS[variant], **EXTRA_SPOTS.get(f"boxes/{variant}", {})},
             title, bx_expect)


if __name__ == "__main__":
    main()
