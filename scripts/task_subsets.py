# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9"]
# ///
"""Leave-one-task-out check of the photo chart: every group of 3 of the 4 tasks.

    uv run scripts/task_subsets.py [RUN]        # default point-benchmark-v2.7

Reads the exact call records behind the published chart
(charts/RUN/with-specialists/calls.jsonl) and, for each of the 4 groups of
3 tasks (leave out logs, cows, fig or dishes), recomputes every model's
score the chart's way (mean over tasks of the mean over tries), its rank,
and the best-value line. Writes charts/RUN-task-subsets/: subsets.png,
subsets.csv, frontiers.json, index.html.

This checks what the error bars do not cover: how much the result depends
on which tasks are in the benchmark. With 4 tasks there are only 4 groups,
so it is a sensitivity check, not a confidence interval.
"""

from __future__ import annotations

import csv
import html
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from segbench.chart import _fonts  # noqa: E402

TASKS = ("logs", "cows", "fig", "dishes")
NAMES = {"logs": "log ends", "cows": "cows", "fig": "fig leaves", "dishes": "dishes"}
COLOURS = {"logs": "#8a5a2b", "cows": "#2f7d32", "fig": "#7b3fa0", "dishes": "#1f6fb5"}


def main() -> None:
    run = sys.argv[1] if len(sys.argv) > 1 else "point-benchmark-v2.7"
    src = ROOT / "charts" / run / "with-specialists"
    out = ROOT / "charts" / f"{run}-task-subsets"
    out.mkdir(parents=True, exist_ok=True)

    tries, label, cost = defaultdict(lambda: defaultdict(list)), {}, {}
    for line in (src / "calls.jsonl").read_text().splitlines():
        r = json.loads(line)
        tries[r["entrant"]][r["task"]].append(0.0 if r.get("readable") is False else r["score"])
        label[r["entrant"]] = r["label"]
    published = {r["entrant"]: r for r in csv.DictReader((src / "points.csv").open())}
    for e, r in published.items():
        cost[e] = float(r["cost_usd"])

    task = {e: {t: 100 * statistics.fmean(v[t]) for t in TASKS} for e, v in tries.items()}
    groups = {"all 4": TASKS, **{f"without {NAMES[d]}": tuple(t for t in TASKS if t != d) for d in TASKS}}
    score = {e: {g: statistics.fmean(task[e][t] for t in ts) for g, ts in groups.items()} for e in task}
    for e in score:  # the full score must be the published one
        assert abs(score[e]["all 4"] - 100 * float(published[e]["quality"])) < 1e-9, e
    rank = defaultdict(dict)
    for g in groups:
        for i, e in enumerate(sorted(score, key=lambda e: -score[e][g])):
            rank[e][g] = i + 1
    frontier = {}
    for g in groups:
        best, top = [], 0.0
        for e in sorted(cost, key=lambda e: (cost[e], -score[e][g])):
            if score[e][g] > top:
                best.append(e)
                top = score[e][g]
        frontier[g] = best

    order = sorted(score, key=lambda e: -score[e]["all 4"])
    loo = [g for g in groups if g != "all 4"]

    with (out / "subsets.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["entrant", "label", *[f"task_{t}" for t in TASKS],
                    *[f"score_{g.replace(' ', '_')}" for g in groups], *[f"rank_{g.replace(' ', '_')}" for g in groups]])
        for e in order:
            w.writerow([e, label[e], *[task[e][t] for t in TASKS], *[score[e][g] for g in groups],
                        *[rank[e][g] for g in groups]])
    (out / "frontiers.json").write_text(json.dumps({g: [label[e] for e in b] for g, b in frontier.items()},
                                                   indent=2) + "\n")

    # ----- figure: one row per model, its score with each task left out -----
    fonts = _fonts()
    plt.rcParams.update({"font.family": fonts.regular.get_name(), "svg.fonttype": "path"})
    n = len(order)
    fig, ax = plt.subplots(figsize=(10, 0.36 * n + 2.2))
    fig.subplots_adjust(left=0.24, right=0.86, top=1 - 1.25 / (0.36 * n + 2.2), bottom=0.55 / (0.36 * n + 2.2))
    for i, e in enumerate(order):
        y = n - 1 - i
        vals = [score[e][g] for g in loo]
        ax.plot([min(vals), max(vals)], [y, y], color="#d3d8e0", lw=5, solid_capstyle="round", zorder=1)
        for t, g in zip(TASKS, loo, strict=True):
            ax.scatter(score[e][g], y, s=34, color=COLOURS[t], zorder=3, edgecolors="white", linewidths=0.6)
        ax.scatter(score[e]["all 4"], y, marker="|", s=260, color="#0f1115", linewidths=2.2, zorder=4)
        rs = [rank[e][g] for g in groups]
        ax.text(101.5, y, f"#{rank[e]['all 4']}", va="center", fontsize=9.5, fontproperties=fonts.semibold)
        ax.text(108, y, f"{min(rs)}–{max(rs)}" if min(rs) != max(rs) else "same", va="center", fontsize=9.5,
                color="#4b5260")
    ax.set_yticks(range(n))
    ax.set_yticklabels([label[e] for e in reversed(order)], fontsize=10)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 10))
    ax.set_xticklabels([f"{v}%" for v in range(0, 101, 10)], fontsize=9, color="#8b93a1")
    ax.grid(True, axis="x", color="#e4e8ef", lw=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#e4e8ef")
    ax.tick_params(length=0)
    top = n - 0.3
    ax.text(101.5, top + 0.15, "rank", fontsize=9, color="#8b93a1", va="bottom")
    ax.text(108, top + 0.15, "rank range", fontsize=9, color="#8b93a1", va="bottom")
    handles = [plt.Line2D([], [], marker="|", ls="none", color="#0f1115", ms=12, mew=2.2, label="all 4 tasks (chart score)")]
    handles += [plt.Line2D([], [], marker="o", ls="none", color=COLOURS[t], ms=7, label=f"without {NAMES[t]}")
                for t in TASKS]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.24, 1 - 0.55 / (0.36 * n + 2.2)), ncol=5,
               frameon=False, fontsize=9, handletextpad=0.3, columnspacing=1.2)
    fig.text(0.24, 1 - 0.22 / (0.36 * n + 2.2), "Score with each task left out in turn", fontsize=14,
             fontproperties=fonts.bold, va="top")
    fig.savefig(out / "subsets.png", dpi=200, metadata={"Software": None})
    fig.savefig(out / "subsets.svg", metadata={"Date": None})
    plt.close(fig)

    # ----- page -----
    def cell(v):
        return f"<td>{v:.1f}</td>"

    head = "".join(f"<th>{html.escape(g)}</th>" for g in groups)
    rows = []
    for e in order:
        rs = [rank[e][g] for g in groups]
        rows.append(f"<tr><th>{html.escape(label[e])}</th>" + "".join(cell(task[e][t]) for t in TASKS) + "<td class=s></td>"
                    + "".join(f"<td>{score[e][g]:.1f} <span>#{rank[e][g]}</span></td>" for g in groups)
                    + f"<td>{min(rs)}–{max(rs)}</td></tr>")
    fronts = "".join(f"<li><b>{html.escape(g)}:</b> {' → '.join(html.escape(label[e]) for e in b)}</li>"
                     for g, b in frontier.items())
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Photo chart: every group of 3 tasks</title><style>
body{{margin:0;padding:16px;font:14px/1.5 var(--font-sans,system-ui);background:var(--color-background-primary,#fff);color:var(--color-text-primary,#16191d)}}
main{{max-width:1100px;margin:auto}}img{{width:100%;max-width:1000px;display:block;margin:12px 0;background:#fff}}
table{{border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums}}th,td{{padding:3px 8px;text-align:right;border-bottom:1px solid var(--color-border-tertiary,#e4e8ef)}}
th:first-child{{text-align:left}}td span{{color:var(--color-text-secondary,#8b93a1)}}td.s{{border:0;width:12px}}.wrap{{overflow-x:auto}}
p,li{{max-width:820px}}@media(prefers-color-scheme:dark){{body{{background:var(--color-background-primary,#16191d);color:var(--color-text-primary,#e8eaee)}}}}
</style></head><body><main>
<h2>Photo chart: what if one task were left out?</h2>
<p>Same data as the published v2.7 chart. Each model's score is recomputed on every group of 3 of the 4 tasks (4 groups), the chart's way: average over tasks of the average over tries.
The black tick is the chart's score; each coloured dot is the score without one task. A long grey band means the model's place depends on which tasks are in the benchmark.</p>
<img src="subsets.png" alt="Each model's score on the four groups of three tasks">
<h3>Best-value line in each group</h3><ul>{fronts}</ul>
<h3>All numbers</h3><div class="wrap"><table><tr><th>Model</th>{''.join(f'<th>{NAMES[t]}</th>' for t in TASKS)}<th class=s></th>{head}<th>rank range</th></tr>{''.join(rows)}</table></div>
<p>Drawn by <code>uv run scripts/task_subsets.py {run}</code>. Data: <a href="subsets.csv" download>subsets.csv</a>, <a href="frontiers.json" download>frontiers.json</a>.</p>
</main></body></html>"""
    (out / "index.html").write_text(page)
    for e in order:
        rs = [rank[e][g] for g in groups]
        print(f"{label[e]:24s} #{rank[e]['all 4']:<3d} range {min(rs)}-{max(rs)}")


if __name__ == "__main__":
    main()
