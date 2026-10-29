# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9", "svgelements>=1.9"]
# ///
"""Thinking tokens per answer for each language model, on the answers of the photo chart (point v3).

    uv run scripts/plot_thinking_tokens.py  ->  charts/thinking-tokens-v1/thinking-tokens.{png,svg,pdf}, per-model.csv
                                            and thinking-tokens-box.{png,svg,pdf}, thinking-tokens-box-linear.{...}

Answers: the language-model rows of results/point-benchmark-v3/scored.jsonl (each model's charted
prompt, medium thinking, 11 photos x 3 tries = 33 answers), each traced to its raw record
(source_file, source_line) for the provider's own thinking-token count.

Counts are the providers' exact ones (OpenAI reasoning_tokens, Anthropic thinking_tokens, Gemini
thoughtsTokenCount, the chat APIs' reasoning_tokens). Gemini leaves thoughtsTokenCount out when no
thinking happened (its total is then question + answer exactly), so a missing count from Gemini is 0;
a missing count from any other provider stops the script.

Bar chart: average per answer; whisker one standard deviation over the 33 answers, the lower end cut
at 0 (token counts are never negative). Misleading for skewed models (Perceptron: 20 answers at 170-465,
10 at 2,500-7,400, so mean - sd < 0): kept for the record, the box charts replace it.

Box charts (Maxime, 2026-10-20): box = middle half of the answers (25th to 75th percentile, inclusive
method), line = median, whiskers = furthest answer within 1.5 box heights (Tukey), dots = answers beyond.
Sorted by median. The log version draws 0 at a floor below the smallest non-zero count, labelled "none",
with a break mark; quartiles are taken on the real counts, then placed.
"""
import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from plot_error_kinds import logo  # noqa: E402
from segbench.chart import _fonts  # noqa: E402
from segbench.chart_data import describe  # noqa: E402
from segbench.chart_editorial import GRID, INK, INK_2, INK_3  # noqa: E402

SCORED = ROOT / "results/point-benchmark-v3/scored.jsonl"
OUT = ROOT / "charts/thinking-tokens-v1"
BAR = "#0f8b8d"


def thinking_tokens():
    """{entrant: (label, maker, [tokens per answer])} for every language model on the chart."""
    lines = {}
    out = {}
    for row in map(json.loads, open(SCORED)):
        if "@" not in row["entrant"]:  # local specialists: no thinking
            continue
        f = ROOT / "results" / row["source_file"]
        if f not in lines:
            lines[f] = open(f).read().splitlines()
        raw = json.loads(lines[f][row["source_line"] - 1])
        assert raw["model"] == row["model"] and raw.get("task", row["task"]) == row["task"], row
        n = (raw.get("usage") or {}).get("reasoning_tokens")
        if n is None:
            if raw["provider"] != "gemini":
                raise SystemExit(f"no thinking count from {raw['provider']} for {row['entrant']} {row['task']}")
            n = 0
        label, maker, _ = describe(row["model"])
        out.setdefault(row["entrant"], (label, maker, []))[2].append(n)
    return out


FLOOR = 10  # where 0 is drawn on the log chart; every non-zero count is above it (checked)


def box_stats(v):
    q1, med, q3 = statistics.quantiles(v, n=4, method="inclusive")
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    inside = [x for x in v if lo <= x <= hi]
    return {"q1": q1, "med": med, "q3": q3, "whislo": min(inside), "whishi": max(inside),
            "fliers": [x for x in v if x < lo or x > hi]}


def boxes(data, fonts, log):
    rows = sorted(data.values(), key=lambda r: (-statistics.median(r[2]), -statistics.mean(r[2])))
    put = (lambda x: max(x, FLOOR)) if log else (lambda x: x)
    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_axes([0.095, 0.235, 0.885, 0.62])
    if log:
        ax.set_yscale("log")
    stats = []
    for label, _, v in rows:
        b = box_stats(v)
        stats.append({k: ([put(x) for x in val] if k == "fliers" else put(val)) for k, val in b.items()})
    ax.bxp(stats, positions=range(len(rows)), widths=0.62, patch_artist=True, showfliers=True,
           boxprops={"facecolor": BAR, "edgecolor": BAR, "linewidth": 1.2},
           medianprops={"color": "white", "linewidth": 2.4, "solid_capstyle": "butt"},
           whiskerprops={"color": INK_2, "linewidth": 1.3}, capprops={"color": INK_2, "linewidth": 1.3},
           flierprops={"marker": "o", "markersize": 5, "markerfacecolor": "white", "markeredgecolor": INK_2,
                       "markeredgewidth": 1.1}, zorder=3)
    xt = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for i, (label, _, v) in enumerate(rows):  # the median, in a row above the chart (Maxime, 2026-10-20)
        b = stats[i]
        if b["q1"] == b["q3"]:  # a box of zero height (most answers equal): a thick bar, or it vanishes
            ax.plot([i - 0.31, i + 0.31], [b["med"]] * 2, color=BAR, lw=5, solid_capstyle="butt", zorder=4)
        ax.text(i, 1.025, f"{statistics.median(v):,.0f}", transform=xt, ha="center", va="bottom",
                fontproperties=fonts.semibold, fontsize=10.5, color=INK_2)
    ax.text(-0.75, 1.025, "middle\nanswer", transform=xt, ha="right", va="bottom", fontproperties=fonts.regular,
            fontsize=9.5, color=INK_3, linespacing=1.1)
    if log:
        lo, hi = FLOOR / 1.6, 32000
        ax.set_ylim(lo, hi)
        ticks = [FLOOR, 100, 1000, 10000]
        ax.set_yticks(ticks)
        ax.set_yticklabels(["none", "100", "1,000", "10,000"])
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        for y in (100, 1000, 10000):
            ax.axhline(y, color=GRID, zorder=0, lw=1)
        ax.axhline(FLOOR, color=GRID, zorder=0, lw=1, ls=(0, (3, 3)))
        # break mark on the left edge between "none" and 100
        yb = FLOOR * 1.9
        for dy in (0.93, 1.07):
            ax.plot([-0.016, 0.004], [yb * dy / 1.06, yb * dy * 1.06], color=INK_3, lw=1.2, clip_on=False,
                    transform=matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData))
        logo_y = lo / 1.55
    else:
        top = max(max(v) for _, _, v in rows) * 1.07
        ax.set_ylim(0, top)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.grid(axis="y", color=GRID, zorder=0)
        logo_y = -top * 0.042
    for i, (_, maker, _) in enumerate(rows):
        logo(ax, i, logo_y, maker, 21, fonts)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[0] for r in rows], rotation=60, ha="right", rotation_mode="anchor")
    for lab in ax.get_xticklabels():
        lab.set_fontproperties(fonts.medium)
        lab.set_fontsize(12)
        lab.set_color(INK)
    for lab in ax.get_yticklabels():
        lab.set_fontproperties(fonts.regular)
        lab.set_fontsize(11)
        lab.set_color(INK_2)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.tick_params(axis="x", length=0, pad=38)
    for side in ax.spines.values():
        side.set_visible(False)
    fig.text(0.04, 0.965, "How much each model thinks before pointing", fontproperties=fonts.bold, fontsize=24,
             color=INK)
    fig.text(0.04, 0.937, "Thinking tokens per answer on the eleven photos. Box: the middle half of a model's "
             "answers; white line: the middle answer (number above).", fontproperties=fonts.regular, fontsize=13,
             color=INK_2)
    n = sum(len(v) for _, _, v in rows)
    scale = ("Log scale: each line up is ten times more. " if log else "")
    fig.text(0.04, 0.034, f"{n} answers: {len(rows)} language models, 11 photos, 3 tries, at the medium thinking "
             "setting of the photo chart (GLM, Kimi and Mistral: their high).\nWhiskers reach the furthest answer "
             "within 1.5 box heights; dots are answers beyond. " + scale + "Counts as each provider reports them.",
             fontproperties=fonts.regular, fontsize=10.5, color=INK_3, va="center", linespacing=1.5)
    return fig


def main():
    data = thinking_tokens()
    assert all(x == 0 or x > FLOOR * 1.5 for _, _, v in data.values() for x in v)
    rows = []
    for entrant, (label, maker, v) in data.items():
        rows.append({"entrant": entrant, "label": label, "maker": maker, "answers": len(v),
                     "mean": statistics.mean(v), "sd": statistics.stdev(v), "median": statistics.median(v),
                     "q1": box_stats(v)["q1"], "q3": box_stats(v)["q3"],
                     "min": min(v), "max": max(v), "no_thinking": sum(x == 0 for x in v)})
    rows.sort(key=lambda r: -r["mean"])

    fonts = _fonts()
    plt.rcParams.update({"svg.fonttype": "path", "pdf.fonttype": 42, "font.family": fonts.regular.get_name()})
    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_axes([0.095, 0.235, 0.885, 0.665])
    top = max(r["mean"] + r["sd"] for r in rows) * 1.08
    for i, r in enumerate(rows):
        m, sd = r["mean"], r["sd"]
        ax.bar(i, m, width=0.8, color=BAR, zorder=2)
        ax.plot([i, i], [max(0, m - sd), m + sd], color=INK, lw=1.4, zorder=3, solid_capstyle="butt")
        for y in (max(0, m - sd), m + sd):
            ax.plot([i - 0.16, i + 0.16], [y, y], color=INK, lw=1.4, zorder=3)
        ax.text(i, m + sd + top * 0.012, f"{m:,.0f}", ha="center", va="bottom", fontproperties=fonts.semibold,
                fontsize=10.5, color=INK_2, zorder=3)
        logo(ax, i, -top * 0.042, r["maker"], 21, fonts)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylim(0, top)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r["label"] for r in rows], rotation=60, ha="right", rotation_mode="anchor")
    for lab in ax.get_xticklabels():
        lab.set_fontproperties(fonts.medium)
        lab.set_fontsize(12)
        lab.set_color(INK)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    for lab in ax.get_yticklabels():
        lab.set_fontproperties(fonts.regular)
        lab.set_fontsize(11)
        lab.set_color(INK_2)
    ax.grid(axis="y", color=GRID, zorder=0)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.tick_params(axis="x", length=0, pad=38)
    for side in ax.spines.values():
        side.set_visible(False)

    fig.text(0.04, 0.965, "How much each model thinks before pointing", fontproperties=fonts.bold, fontsize=24,
             color=INK)
    fig.text(0.04, 0.937, "Thinking tokens per answer on the eleven photos. Bar: average. Line: how much it varies "
             "from one answer to the next.", fontproperties=fonts.regular, fontsize=13, color=INK_2)
    n = sum(r["answers"] for r in rows)
    fig.text(0.04, 0.034, f"{n} answers: {len(rows)} language models, 11 photos, 3 tries, at the medium thinking setting "
             "used on the photo chart (GLM, Kimi and Mistral:\ntheir high). Line: one standard deviation either side, cut "
             "at 0. Counts as each provider reports them. Local models were run without thinking.",
             fontproperties=fonts.regular, fontsize=10.5, color=INK_3, va="center", linespacing=1.5)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"thinking-tokens.{ext}", dpi=170 if ext == "png" else None, facecolor="white")
    plt.close(fig)
    for name, log in (("thinking-tokens-box", True), ("thinking-tokens-box-linear", False)):
        fig = boxes(data, fonts, log)
        for ext in ("png", "svg", "pdf"):
            fig.savefig(OUT / f"{name}.{ext}", dpi=170 if ext == "png" else None, facecolor="white")
        plt.close(fig)
        print(OUT / f"{name}.png")
    with open(OUT / "per-model.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: round(v, 1) if isinstance(v, float) else v for k, v in r.items()})
    print(OUT / "thinking-tokens.png")


if __name__ == "__main__":
    main()
