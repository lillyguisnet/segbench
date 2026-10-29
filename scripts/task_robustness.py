# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9", "numpy>=1.26", "scipy>=1.13"]
# ///
"""How robust is the ranking to the photos and tasks we chose?

    uv run scripts/task_robustness.py [RUN]        # default point-benchmark-v3

Asked on 2026-10-14: "if I had kept only x of the y tasks, how robust is the
relative ranking; how robust are the results to the idiosyncrasies of the
images and tasks we chose". Each task has its own photo, so a task here is
also a photo.

Score of a model on a set of tasks: its best run on those tasks (a run is
one complete pass over all tasks; language models have 3, local models 1),
re-chosen within each set, as the mean over the set's tasks. Ranks: 1 =
best, ties share the average rank. Every subset is enumerated (no sampling).

1. Keep x of the T tasks (x = 1 .. T-1, every subset): rank agreement with
   the ranking on all T tasks (Spearman's rho), whether the winner stays the
   winner, and each model's range of ranks. These subsets share tasks with
   the full set, so their agreement is optimistic, more so as x nears T.
2. Two disjoint sets (every split into two sets of x tasks, x <= T/2): rank
   agreement between the two rankings. This is the honest measure of "would
   other photos of the same kind give the same ranking". Projected to two
   disjoint sets of T tasks with the Spearman-Brown formula
   (r_T = k r / (1 + (k - 1) r), k = T / x), a standard approximation for
   the reliability of a longer test.
3. Per model, at x tasks kept: 5th to 95th percentile of its rank over all
   subsets, and the share of subsets in which it stays above the model just
   below it in the full ranking.

Writes charts/RUN-task-robustness/: agreement.png, ranks-<x>.png (x = 6 .. 10),
robustness.json, index.html.
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from segbench.chart import _fonts  # noqa: E402
from segbench.chart_data import aggregate, read_calls  # noqa: E402

RANK_XS = (6, 7, 8, 9, 10)


def sized(fp, size):
    fp = fp.copy()
    fp.set_size(size)
    return fp


def load(run: str):
    calls = read_calls([ROOT / "charts" / run / "with-specialists" / "calls.jsonl"])
    labels = {p.entrant: p.label for p in aggregate(calls)}
    kinds = {p.entrant: p.kind for p in aggregate(calls)}
    tasks = sorted({r["task"] for r in calls})
    by = defaultdict(lambda: defaultdict(dict))
    for r in calls:
        by[r["entrant"]][r.get("repeat", 0)][r["task"]] = 0.0 if r.get("readable") is False else r["score"]
    entrants = sorted(by)
    runs = max(len(v) for v in by.values())
    S = np.full((len(entrants), runs, len(tasks)), np.nan)  # model x run x task; NaN: no complete run
    for i, e in enumerate(entrants):
        complete = [rs for _, rs in sorted(by[e].items()) if set(rs) == set(tasks)]
        for j, rs in enumerate(complete):
            S[i, j] = [rs[t] for t in tasks]
    return entrants, labels, kinds, tasks, S


def scores(S, subset):
    return np.nanmax(S[:, :, list(subset)].mean(axis=2), axis=1)


def ranks(s):
    return stats.rankdata(-s, method="average")


def main() -> None:
    run = sys.argv[1] if len(sys.argv) > 1 else "point-benchmark-v3"
    out = ROOT / "charts" / f"{run}-task-robustness"
    out.mkdir(parents=True, exist_ok=True)
    entrants, labels, kinds, tasks, S = load(run)
    T, n = len(tasks), len(entrants)
    full = scores(S, range(T))
    full_rank = ranks(full)
    order = np.argsort(full_rank)  # best first
    winner = order[0]

    keep = {}
    rank_samples = {}
    for x in range(1, T):
        rhos, top1, top5 = [], [], []
        R = []
        for sub in itertools.combinations(range(T), x):
            r = ranks(scores(S, sub))
            R.append(r)
            rhos.append(stats.spearmanr(r, full_rank).statistic)
            top1.append(bool(r[winner] == r.min() and (r == r.min()).sum() == 1))
            top5.append(set(np.argsort(r)[:5]) == set(order[:5]))
        R = np.array(R)
        rank_samples[x] = R
        keep[x] = {"subsets": len(R), "rho_median": float(np.median(rhos)), "rho_p5": float(np.percentile(rhos, 5)),
                   "rho_p95": float(np.percentile(rhos, 95)), "same_winner": float(np.mean(top1)),
                   "same_top5": float(np.mean(top5))}

    disjoint = {}
    for x in range(1, T // 2 + 1):
        rhos = []
        for a in itertools.combinations(range(T), x):
            rest = [t for t in range(T) if t not in a]
            for b in itertools.combinations(rest, x):
                if a < b:
                    rhos.append(stats.spearmanr(ranks(scores(S, a)), ranks(scores(S, b))).statistic)
        r = float(np.median(rhos))
        k = T / x
        disjoint[x] = {"pairs": len(rhos), "rho_median": r, "rho_p5": float(np.percentile(rhos, 5)),
                       "rho_p95": float(np.percentile(rhos, 95)),
                       "projected_to_T": k * r / (1 + (k - 1) * r) if r > 0 else None}

    per_model = {}
    for x in RANK_XS:
        R = rank_samples[x]
        per_model[x] = {}
        for pos, i in enumerate(order):
            below = order[pos + 1] if pos + 1 < n else None
            per_model[x][entrants[i]] = {
                "full_rank": float(full_rank[i]), "rank_p5": float(np.percentile(R[:, i], 5)),
                "rank_p95": float(np.percentile(R[:, i], 95)), "rank_min": float(R[:, i].min()),
                "rank_max": float(R[:, i].max()),
                "stays_above_next": None if below is None else float(np.mean(R[:, i] < R[:, below]))}

    # the same with the mean of runs instead of the best run, as a check
    Sm = np.nanmean(S, axis=1, keepdims=True)
    mean_rank = ranks(scores(Sm, range(T)))
    check = {"rho_best_vs_mean_of_runs_all_tasks": float(stats.spearmanr(full_rank, mean_rank).statistic)}
    for x in (8,):
        rh = [stats.spearmanr(ranks(scores(Sm, sub)), mean_rank).statistic for sub in itertools.combinations(range(T), x)]
        check[f"mean_of_runs_keep_{x}_rho_median"] = float(np.median(rh))
        check[f"mean_of_runs_keep_{x}_same_winner"] = float(np.mean(
            [ranks(scores(Sm, sub))[np.argmin(mean_rank)] == 1 for sub in itertools.combinations(range(T), x)]))

    (out / "robustness.json").write_text(json.dumps(
        {"source": f"charts/{run}/with-specialists/calls.jsonl", "tasks": tasks, "entrants": entrants,
         "method": __doc__.split("Score of a model", 1)[1].strip(), "keep_x_of_T": keep, "disjoint_sets": disjoint,
         "per_model": per_model, "check_mean_of_runs": check}, indent=2) + "\n")

    fonts = _fonts()
    # ---- agreement chart
    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=216)
    xs = list(keep)
    ax.fill_between(xs, [keep[x]["rho_p5"] for x in xs], [keep[x]["rho_p95"] for x in xs], color="#e8384f", alpha=0.13, lw=0)
    ax.plot(xs, [keep[x]["rho_median"] for x in xs], color="#e8384f", lw=2.4, marker="o", ms=5,
            label=f"keep x of the {T} tasks, compared with all {T}")
    dx = list(disjoint)
    ax.fill_between(dx, [disjoint[x]["rho_p5"] for x in dx], [disjoint[x]["rho_p95"] for x in dx], color="#4a6fa5",
                    alpha=0.13, lw=0)
    ax.plot(dx, [disjoint[x]["rho_median"] for x in dx], color="#4a6fa5", lw=2.4, marker="o", ms=5,
            label="two separate sets of x tasks (no task shared), compared with each other")
    proj = disjoint[max(dx)]["projected_to_T"]
    ax.plot([max(dx), T], [disjoint[max(dx)]["rho_median"], proj], color="#4a6fa5", lw=1.6, ls=(0, (4, 3)))
    ax.scatter([T], [proj], s=60, color="white", edgecolor="#4a6fa5", linewidths=2, zorder=3)
    ax.text(T - 0.15, proj - 0.035, f"projected: two separate sets\nof {T} tasks, {proj:.2f}", ha="right", va="top",
            color="#4a6fa5", fontproperties=sized(fonts.medium, 9.5))
    ax.set_xlim(0.6, T + 0.4)
    ax.set_ylim(min(0, min(disjoint[1]["rho_p5"], keep[1]["rho_p5"])) - 0.02, 1.02)
    ax.set_xticks(range(1, T + 1))
    ax.set_xticklabels(range(1, T + 1), fontproperties=sized(fonts.regular, 10), color="#6b7280")
    yt = [v / 10 for v in range(0, 11, 2)]
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{v:.1f}" for v in yt], fontproperties=sized(fonts.regular, 10), color="#6b7280")
    ax.set_xlabel("Number of tasks (photos) used", fontproperties=sized(fonts.medium, 11), color="#374151")
    ax.set_ylabel("Ranking agreement (1 = identical order)", fontproperties=sized(fonts.medium, 11), color="#374151")
    ax.grid(color="#e5e7eb", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower right", frameon=False, prop=sized(fonts.regular, 10))
    fig.text(0.03, 0.965, "How much does the ranking depend on the photos we chose?", va="top",
             fontproperties=sized(fonts.bold, 17))
    fig.text(0.03, 0.905, f"{n} models, ranked by their best run. Line: median over every possible choice of tasks; "
             "band: 90% of choices.", va="top", color="#4b5563", fontproperties=sized(fonts.regular, 10))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.84, bottom=0.11)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out / f"agreement.{ext}", facecolor="white")
    plt.close(fig)

    # ---- per-model rank ranges
    for x in RANK_XS:
        pm = per_model[x]
        height = 1.7 + 0.36 * n
        fig, ax = plt.subplots(figsize=(10, height), dpi=216)
        for pos, i in enumerate(order):
            e, y = entrants[i], n - 1 - pos
            c = pm[e]
            colour = "#e8384f" if kinds[e] == "general" else "#4a6fa5"
            ax.plot([c["rank_p5"], c["rank_p95"]], [y, y], color="#a3abb8", lw=6, solid_capstyle="round", zorder=1)
            ax.scatter([c["full_rank"]], [y], s=60, color=colour, edgecolor="white", linewidths=1.2, zorder=3)
            ax.text(-0.3, y, labels[e], ha="right", va="center", fontproperties=sized(fonts.medium, 10.5))
            if c["stays_above_next"] is not None:
                share = c["stays_above_next"]
                ax.text(n + 1.6, y - 0.5, f"{100 * share:.0f}%", ha="right", va="center",
                        color="#202520" if share >= 0.9 else "#b42318", fontproperties=sized(fonts.semibold, 9.5))
        ax.set_xlim(0.4, n + 1.8)
        ax.set_ylim(-0.7, n - 0.3)
        ax.set_yticks([])
        ticks = [1, 5, 10, 15, 20, 25, n] if n > 25 else [1, 5, 10, 15, n]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"#{t}" for t in ticks], fontproperties=sized(fonts.regular, 9.5), color="#6b7280")
        ax.grid(axis="x", color="#e5e7eb", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.text(n + 1.6, n - 0.1, "stays\nabove\nnext", ha="right", va="bottom", color="#6b7280",
                fontproperties=sized(fonts.regular, 8.5), linespacing=1.1)
        fig.text(0.03, 1 - 0.30 / height, f"Where each model ranks with only {x} of the {T} tasks", va="top",
                 fontproperties=sized(fonts.bold, 17))
        fig.text(0.03, 1 - 0.70 / height,
                 f"Dot: its rank on all {T} tasks. Grey bar: its rank over every choice of {x} tasks "
                 f"({len(rank_samples[x])} choices; 5th to 95th percentile).\nRight: share of choices in which it "
                 "stays above the model just below it (red under 90%).", va="top", color="#4b5563",
                 fontproperties=sized(fonts.regular, 9.5), linespacing=1.5)
        fig.subplots_adjust(left=0.25, right=0.96, top=1 - 1.5 / height, bottom=0.55 / height)
        for ext in ("png", "svg", "pdf"):
            fig.savefig(out / f"ranks-{x}.{ext}", facecolor="white")
        plt.close(fig)

    print("keep x of T: x  rho(median, 5-95%)  same winner  same top 5")
    for x, k in keep.items():
        print(f"  {x:2d}  {k['rho_median']:.2f} ({k['rho_p5']:.2f}-{k['rho_p95']:.2f})  {100 * k['same_winner']:5.1f}%  "
              f"{100 * k['same_top5']:5.1f}%")
    print("disjoint sets: x  rho(median, 5-95%)  projected to T")
    for x, d in disjoint.items():
        print(f"  {x:2d}  {d['rho_median']:.2f} ({d['rho_p5']:.2f}-{d['rho_p95']:.2f})  {d['projected_to_T']:.2f}")
    print("check (mean of runs):", {k: round(v, 3) for k, v in check.items()})
    for x in (8,):
        print(f"per model, keep {x}:")
        for i in order:
            c = per_model[x][entrants[i]]
            s = "" if c["stays_above_next"] is None else f"{100 * c['stays_above_next']:.0f}%"
            print(f"  {labels[entrants[i]]:24s} #{c['full_rank']:<4g} {c['rank_p5']:g}-{c['rank_p95']:g}  above next {s}")


if __name__ == "__main__":
    main()
