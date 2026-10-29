# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9", "svgelements>=1.9", "scipy>=1.13"]
# ///
"""The photo chart with each model's best run and error bars over the tasks.

    uv run scripts/plot_task_error_bars.py [RUN]        # default point-benchmark-v3

Asked for on 2026-10-14: "I'm happy to take the best run of a model, but I
want error bars on the tasks." Reads the call records behind the chart
(charts/RUN/<variant>/calls.jsonl) and writes
charts/RUN-task-error-bars/<variant>/ (find.png/.pdf/.svg, points.csv,
intervals.json).

What a dot and a bar are:
- Run: one complete pass of a model over every task (one try per task,
  the tries sharing a repeat number). Language models have 3 runs; local
  models have one complete run (their extra tries cover the first four
  photos only, and are identical except LocateAnything's).
- Dot: the model's best run, the one with the highest mean over the tasks.
  Picked on the test photos themselves, so it flatters models with more
  runs and more erratic answers; the published chart keeps the mean of the
  runs.
- Bar: how much that score could move with a different set of tasks of the
  same kind. The tasks are treated as a sample from such tasks; the bar is
  a two-sided Student-t interval over the T task scores of the best run:
  mean +- t(1 - alpha / 2m, T - 1) * s / sqrt(T), s the standard deviation
  of the task scores. Bonferroni: m = models on the chart, so all bars on a
  chart hold together with at least 95% confidence. Cut to 0-100%.
- Not in the bar: luck between tries (the best run is taken as given).
- Each bar is for one model on its own. All models answered the SAME tasks,
  so two models whose bars overlap can still differ reliably: a comparison
  of two models should use the per-task differences (paired), which are
  much tighter than these bars.
- Cost and seconds: unchanged from the chart (all tries, first four tasks).
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import asdict, fields, replace
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from plot_point_pilot import LABEL_SPOTS  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from segbench.chart import _fonts, x_range  # noqa: E402
from segbench.chart_data import aggregate, read_calls  # noqa: E402
from segbench.chart_editorial import draw  # noqa: E402

ALPHA = 0.05
VARIANTS = ("with-specialists", "language-models")


def ranked(points, info, kinds, path_base: Path, m: int, T: int) -> list[Path]:
    """One row per model, best run first: the dot, its task bar, and the mean of runs as a tick."""
    fonts = _fonts()

    def sized(fp, size):
        fp = fp.copy()
        fp.set_size(size)
        return fp

    pts = sorted(points, key=lambda p: -p.quality)
    fig, ax = plt.subplots(figsize=(10, 1.6 + 0.36 * len(pts)), dpi=216)
    for i, p in enumerate(pts):
        c = info[p.entrant]
        y = len(pts) - 1 - i
        colour = "#e8384f" if kinds[p.entrant] == "general" else "#4a6fa5"
        ax.plot([100 * c["low"], 100 * c["high"]], [y, y], color="#a3abb8", lw=2.2, solid_capstyle="butt", zorder=1)
        for end in (c["low"], c["high"]):
            ax.plot([100 * end] * 2, [y - 0.18, y + 0.18], color="#a3abb8", lw=1.6, zorder=1)
        ax.scatter([100 * c["mean_of_runs"]], [y], marker="|", s=90, color="#202520", linewidths=1.4, zorder=2)
        ax.scatter([100 * p.quality], [y], s=70, color=colour, edgecolor="white", linewidths=1.2, zorder=3)
        ax.text(-2, y, p.label, ha="right", va="center", fontproperties=sized(fonts.medium, 10.5))
        ax.text(101.5, y, f"{100 * p.quality:.1f}", ha="left", va="center", fontproperties=sized(fonts.semibold, 10))
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.7, len(pts) - 0.3)
    ax.set_yticks([])
    ax.set_xticks(range(0, 101, 10))
    ax.set_xticklabels([f"{v}%" for v in range(0, 101, 10)], fontproperties=sized(fonts.regular, 9.5), color="#6b7280")
    ax.grid(axis="x", color="#e5e7eb", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#cdd4c8")
    height = 1.6 + 0.36 * len(pts)
    fig.text(0.03, 1 - 0.30 / height, "How well can AI point at things in a photo?", va="top",
             fontproperties=sized(fonts.bold, 18))
    fig.text(0.03, 1 - 0.72 / height,
             f"Dot: each model's best run over {T} tasks (red: language model, blue: local model). "
             f"Black tick: the mean of its runs.\nGrey bar: where the score could fall with other tasks of the same kind "
             f"(all {m} bars together hold with 95% confidence).", va="top", color="#4b5563",
             fontproperties=sized(fonts.regular, 9.5), linespacing=1.5)
    fig.subplots_adjust(left=0.25, right=0.94, top=1 - 1.45 / height, bottom=0.6 / height)
    paths = []
    for ext in ("png", "pdf", "svg"):
        path = path_base.with_suffix("." + ext)
        fig.savefig(path, facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def score_of(r: dict) -> float:
    return 0.0 if r.get("readable") is False else r["score"]


def main() -> None:
    run = sys.argv[1] if len(sys.argv) > 1 else "point-benchmark-v3"
    src, out = ROOT / "charts" / run, ROOT / "charts" / f"{run}-task-error-bars"
    for variant in VARIANTS:
        calls = read_calls([src / variant / "calls.jsonl"])
        published = {p.entrant: p for p in aggregate(calls)}
        tasks = sorted({r["task"] for r in calls})
        by = defaultdict(lambda: defaultdict(dict))  # entrant -> repeat -> task -> call
        for r in calls:
            by[r["entrant"]][r.get("repeat", 0)][r["task"]] = r
        best, info = [], {}
        m = len(published)
        for e, runs in by.items():
            complete = {rep: rs for rep, rs in runs.items() if set(rs) == set(tasks)}
            means = {rep: statistics.fmean(score_of(rs[t]) for t in tasks) for rep, rs in complete.items()}
            rep = max(means, key=lambda k: (means[k], -k))  # ties: the earlier run
            best += complete[rep].values()
            vals = [score_of(complete[rep][t]) for t in tasks]
            mean, sd = statistics.fmean(vals), statistics.stdev(vals)
            t = float(stats.t.ppf(1 - ALPHA / (2 * m), len(vals) - 1))
            half = t * sd / math.sqrt(len(vals))
            info[e] = {"best_run": rep, "runs": {k: round(v, 4) for k, v in means.items()}, "task_scores": dict(zip(tasks, vals)),
                       "mean": mean, "sd": sd, "T": len(vals), "t": t, "half_width": half, "m": m,
                       "confidence_each": 1 - ALPHA / m, "low": max(0.0, mean - half), "high": min(1.0, mean + half),
                       "mean_of_runs": published[e].quality}
        # dots = best run; cost and seconds as on the chart (all tries)
        points = [replace(p, cost_usd=published[p.entrant].cost_usd, seconds=published[p.entrant].seconds)
                  for p in aggregate(best)]
        for p in points:
            assert math.isclose(p.quality, info[p.entrant]["mean"], abs_tol=1e-12), p.entrant
        here = {p.entrant for p in points}
        spots = {e: v for e, v in {**LABEL_SPOTS.get(run, {}), **LABEL_SPOTS.get(f"{run}/{variant}", {})}.items()
                 if e in here}
        folder = out / variant
        paths = draw(points, "find", folder / "find", xlim=x_range(points), names="all", label_spots=spots,
                     errorbars={e: (c["low"], c["high"]) for e, c in info.items()})
        paths += ranked(points, info, {p.entrant: p.kind for p in points}, folder / "ranked", m, len(tasks))
        names = [f.name for f in fields(points[0])] + ["best_run", "mean_of_runs", "ci_low", "ci_high", "task_sd",
                                                       "t", "bonferroni_m"]
        with (folder / "points.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=names)
            w.writeheader()
            for p in points:
                c = info[p.entrant]
                w.writerow(asdict(p) | {"best_run": c["best_run"], "mean_of_runs": c["mean_of_runs"], "ci_low": c["low"],
                                        "ci_high": c["high"], "task_sd": c["sd"], "t": c["t"], "bonferroni_m": m})
        (folder / "intervals.json").write_text(json.dumps(
            {"source": str((src / variant / "calls.jsonl").relative_to(ROOT)), "alpha_family": ALPHA, "tasks": tasks,
             "method": __doc__.split("What a dot and a bar are:", 1)[1].strip(), "entrants": info}, indent=2) + "\n")
        print(f"{variant}: m = {m}, T = {len(tasks)}, each bar at {100 * (1 - ALPHA / m):.3f}% confidence")
        for p in sorted(points, key=lambda p: -p.quality):
            c = info[p.entrant]
            print(f"  {p.label:24s} best run {100 * p.quality:5.1f} (mean of runs {100 * c['mean_of_runs']:5.1f})  "
                  f"{100 * c['low']:5.1f} to {100 * c['high']:5.1f}  (±{100 * c['half_width']:.1f})")
        for path in paths:
            print("  ", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
