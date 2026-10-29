"""Charts of box benchmark v1 with the project's editorial renderer.

    uv run --with pillow scripts/plot_boxes.py   ->  charts/box-benchmark-v1/

Main charts: the score averaged over the four overlap levels, with and
without the local models; then the per-level ranking (plot_box_levels.py). Checks every plotted point against summary.csv.
"""
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results/box-benchmark/v1"
OUT = ROOT / "charts/box-benchmark-v1"
TITLE = "How well can AI draw a box around each thing in a photo?"
# Labels pinned beside their disc, as for the point charts (chart -> entrant -> spot).
# Gemini 3.5 Flash Lite and Qwen 3.8 27B have nearly the same cost and score
# (their discs overlap); left to the placer, their labels went far away on
# long lines across the chart. Pinned, with nudges, into free space below.
LABEL_SPOTS: dict[str, dict[str, str]] = {
    "with-local-models": {"gemini-flash-lite@medium": "ll@-10,-55", "qwen-27b@medium": "ll@-20,-100",
                          "mk1.5@medium": "right@-3,0!"},
    "language-models": {"gemini-flash-lite@medium": "ll@-10,-30", "qwen-27b@medium": "ll@-40,-100",
                        "mk1.5@medium": "right"},
}


def draw(name, calls, title, expect_col):
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "calls.jsonl"
    path.write_text("".join(json.dumps(c) + "\n" for c in calls))
    spots = [f"--label-spot={e}={s}" for e, s in LABEL_SPOTS.get(name, {}).items()]
    subprocess.run(["uv", "run", str(ROOT / "scripts/draw_charts.py"), str(path), "--style", "editorial",
                    "--track", "find", "--names", "all", "--title", title, "--out", str(folder), *spots],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    summary = {r["entrant"]: r for r in csv.DictReader(open(SRC / "summary.csv"))}
    plotted = list(csv.DictReader(open(folder / "points.csv")))
    for p in plotted:
        s = summary[p["entrant"]]
        for a, b in (("quality", expect_col), ("cost_usd", "cost_usd"), ("seconds", "seconds")):
            assert math.isclose(float(p[a]), float(s[b]), abs_tol=1e-12), (name, p["entrant"], a)
    print(f"{name}: {len(plotted)} points, checked against summary.csv")


def main():
    calls = [json.loads(l) for l in (SRC / "calls.jsonl").read_text().splitlines()]
    local = {c["model"] for c in calls if c["level"] is None}
    draw("with-local-models", calls, TITLE, "score")
    draw("language-models", [c for c in calls if c["model"] not in local], TITLE, "score")
    # The overlap levels: one ranking chart (scripts/plot_box_levels.py), not
    # a bubble chart per level (cost does not change with the level, and four
    # crowded charts read worse than one comparison).
    subprocess.run(["uv", "run", str(ROOT / "scripts/plot_box_levels.py")], cwd=ROOT, check=True)


if __name__ == "__main__":
    sys.exit(main())
