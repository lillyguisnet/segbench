# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9", "svgelements>=1.9", "scipy>=1.13"]
# ///
"""The photo chart with Bonferroni-corrected error bars.

    uv run scripts/plot_error_bars.py [RUN]        # default point-benchmark-v2.7

Reads the exact call records behind the published chart
(charts/RUN/<variant>/calls.jsonl), draws the same chart with a bar on
every model, and writes charts/RUN-error-bars/<variant>/ (find.png/.pdf/.svg,
points.csv with the intervals, intervals.json with how each was computed).

What a bar is (written for the page too, see METHOD below):
- A model's score is the mean over the tasks (4 in v2.7, 11 in v3) of its mean over tries.
  The bar covers run-to-run variation only: how much the score could move
  if the same photos were asked again. The tasks are fixed (they define the
  benchmark); the photos chosen are not a source of variation in the bar.
- Standard error: sqrt(sum over tasks of s_t^2 / n_t) / T (T tasks), with s_t the
  sample variance of the task's tries and n_t their number (stratified mean).
- Degrees of freedom: Welch-Satterthwaite over the tasks' variances.
- Bonferroni: each bar is a two-sided Student-t interval at confidence
  1 - 0.05 / m, m = number of models on that chart, so that all bars on the
  chart hold together with at least 95% confidence.
- Cut to 0-100%, the only possible scores.
- No bar: models whose tries gave identical answers (deterministic
  specialists, checked here: zero variance), or with one try.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import asdict, fields
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from segbench.chart import x_range  # noqa: E402
from segbench.chart_data import aggregate, read_calls  # noqa: E402
from segbench.chart_editorial import draw  # noqa: E402

ALPHA = 0.05
VARIANTS = ("with-specialists", "language-models")
sys.path.insert(0, str(ROOT / "scripts"))
from plot_point_pilot import LABEL_SPOTS as PUBLISHED_SPOTS  # noqa: E402


def label_spots(run: str, variant: str) -> dict:
    """The published chart's pinned labels (scripts/plot_point_pilot.py), for this run and variant."""
    return {**PUBLISHED_SPOTS.get(run, {}), **PUBLISHED_SPOTS.get(f"{run}/{variant}", {})}


def interval(by_task: dict[str, list[float]], m: int) -> dict:
    """Bonferroni-corrected t interval of the mean over tasks of the mean over tries."""
    T = len(by_task)
    mean = statistics.fmean(statistics.fmean(v) for v in by_task.values())
    parts = [(statistics.variance(v) / len(v) / T ** 2, len(v) - 1) for v in by_task.values() if len(v) > 1]
    var = sum(a for a, _ in parts)
    out = {"mean": mean, "tries": {t: len(v) for t, v in by_task.items()}, "m": m,
           "confidence_each": 1 - ALPHA / m}
    if var <= 0:
        return out | {"se": 0.0, "df": None, "t": None, "low": mean, "high": mean, "bar": False}
    df = var ** 2 / sum(a ** 2 / d for a, d in parts if a > 0)
    t = float(stats.t.ppf(1 - ALPHA / (2 * m), df))
    se = math.sqrt(var)
    return out | {"se": se, "df": df, "t": t, "half_width": t * se,
                  "low": max(0.0, mean - t * se), "high": min(1.0, mean + t * se), "bar": True}


def main() -> None:
    run = sys.argv[1] if len(sys.argv) > 1 else "point-benchmark-v2.7"
    src, out = ROOT / "charts" / run, ROOT / "charts" / f"{run}-error-bars"
    for variant in VARIANTS:
        calls = read_calls([src / variant / "calls.jsonl"])
        points = aggregate(calls)
        published = {r["entrant"]: float(r["quality"]) for r in csv.DictReader((src / variant / "points.csv").open())}
        assert {p.entrant: p.quality for p in points} == published, f"{variant}: not the published chart's scores"

        scores = defaultdict(lambda: defaultdict(list))
        for r in calls:
            scores[r["entrant"]][r["task"]].append(0.0 if r.get("readable") is False else r["score"])
        m = len(points)
        ci = {e: interval(t, m) for e, t in scores.items()}
        for p in points:
            assert math.isclose(ci[p.entrant]["mean"], p.quality, abs_tol=1e-12)

        xlim = x_range(points)  # per chart, as scripts/plot_point_pilot.py draws them
        folder = out / variant
        paths = draw(points, "find", folder / "find", xlim=xlim, names="all",
                     label_spots={e: v for e, v in label_spots(run, variant).items()
                                  if e in {p.entrant for p in points}},
                     errorbars={e: (c["low"], c["high"]) for e, c in ci.items() if c["bar"]})
        names = [fl.name for fl in fields(points[0])] + ["ci_low", "ci_high", "se", "df", "t", "bonferroni_m"]
        with (folder / "points.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=names)
            w.writeheader()
            for p in points:
                c = ci[p.entrant]
                w.writerow(asdict(p) | {"ci_low": c["low"], "ci_high": c["high"], "se": c["se"], "df": c["df"],
                                        "t": c["t"], "bonferroni_m": m})
        (folder / "intervals.json").write_text(json.dumps(
            {"source": str((src / variant / "calls.jsonl").relative_to(ROOT)), "alpha_family": ALPHA,
             "method": __doc__.split("What a bar is", 1)[1].strip(), "entrants": ci}, indent=2) + "\n")
        print(f"{variant}: m = {m}, each bar at {100 * (1 - ALPHA / m):.3f}% confidence")
        for p in sorted(points, key=lambda p: -p.quality):
            c = ci[p.entrant]
            bar = (f"{100 * c['low']:5.1f} to {100 * c['high']:5.1f}  (±{100 * c['half_width']:.1f}, df {c['df']:.1f})"
                   if c["bar"] else "no bar: " + ("one try" if max(c["tries"].values()) == 1 else "identical tries"))
            print(f"  {p.label:24s} {100 * p.quality:5.1f}  {bar}")
        for path in paths:
            print("  ", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
