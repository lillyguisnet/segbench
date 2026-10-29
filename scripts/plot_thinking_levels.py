# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9", "svgelements>=1.9"]
# ///
"""Charts of the thinking-levels run (docs/thinking-levels.md).

    uv run scripts/plot_thinking_levels.py  ->  charts/thinking-levels-v1/{outcomes,thinking-tokens}.{png,svg,pdf},
                                                 per-level.csv, paired.csv

1. outcomes: for each model, three stacked bars (least, medium, most thinking): correct points and the four
   families of mistakes, shares as on the square error chart (within each answer, then over tries, then photos).
2. thinking-tokens: thinking tokens per answer at each level (bar: average; line: one standard deviation,
   cut at 0). Counts as each provider reports them.
paired.csv: each level's score and correct share minus medium's, with a 95% interval from resampling the
eleven photos (paired: the same photos at both levels), 10,000 resamples, seed 0.
"""
import csv
import json
import random
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import plot_error_kinds as pek  # noqa: E402
from segbench.chart import _fonts  # noqa: E402
from segbench.chart_data import describe  # noqa: E402
from segbench.chart_editorial import GRID, INK, INK_2, INK_3  # noqa: E402

RES = ROOT / "results/thinking-levels-v1"
KINDS = ROOT / "results/error-kinds/thinking-levels-v1"
OUT = ROOT / "charts/thinking-levels-v1"
LEVELS = ("min", "medium", "max")
GREEN = "#2f9e4f"
GROUPS = [("Correct point", GREEN)] + [(n, c) for n, c, _ in pek.FAMILIES]
NAMES = ["Correct point", "Aimed at the right object, landed off", "Extra point, no object there",
         "Object never found", "Whole answer broken (no answer, doom loop, wrong coordinates)"]
# what "least" means for each model (docs/thinking-levels.md)
LEAST = {"sonnet": "off", "haiku": "off", "terra": "off", "luna": "off",
         "opus": "low", "astra": "low", "sol": "low", "muse-wonder": "minimal"}
LEVEL_WORD = {"medium": "medium", "max": "max"}


def scored():
    return [json.loads(l) for l in (RES / "scored.jsonl").read_text().splitlines()]


def outcome_shares():
    pek.SRC = KINDS
    return pek.shares("points", lambda k: 0 if k == "correct" else pek.FAMILY_OF[k] + 1, len(GROUPS))


def task_means(records, key):
    """{entrant: {task: mean over tries of key(record)}}"""
    by = {}
    for r in records:
        by.setdefault(r["entrant"], {}).setdefault(r["task"], []).append(key(r))
    return {e: {t: statistics.mean(v) for t, v in ts.items()} for e, ts in by.items()}


def paired(a, b, n=10000, seed=0):
    """mean over tasks of a - b, and a 95% interval from resampling tasks."""
    tasks = sorted(a)
    d = [a[t] - b[t] for t in tasks]
    rng = random.Random(seed)
    boots = sorted(statistics.mean(rng.choices(d, k=len(d))) for _ in range(n))
    return statistics.mean(d), boots[int(0.025 * n)], boots[int(0.975 * n) - 1]


def models_in_order(share):
    """Models sorted by correct share at medium (as the published square chart)."""
    ms = {e.split("@")[0] for e in share}
    return sorted(ms, key=lambda m: -share[f"{m}@medium"][0])


def style_axes(ax, fonts):
    for side in ax.spines.values():
        side.set_visible(False)
    for lab in ax.get_yticklabels():
        lab.set_fontproperties(fonts.regular)
        lab.set_fontsize(11)
        lab.set_color(INK_2)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.tick_params(axis="x", length=0)


def group_labels(fig, ax, fonts, models, xs_of, top, logo_y):
    """Level word under each bar, then logo and model name under each group."""
    for m in models:
        xs = xs_of(m)
        for x, lv in zip(xs, LEVELS):
            word = LEAST[m] if lv == "min" else LEVEL_WORD[lv]
            ax.text(x, -top * 0.012, word, ha="right", va="top", rotation=60, rotation_mode="anchor",
                    fontproperties=fonts.regular, fontsize=9.5, color=INK_2, clip_on=False)
        label, maker, _ = describe(m)
        pek.logo(ax, xs[1], logo_y, maker, 22, fonts)
        ax.text(xs[1], logo_y - top * 0.045, label.replace("Claude ", "Claude\n"), ha="center", va="top",
                fontproperties=fonts.semibold, linespacing=1.15, fontsize=11.5, color=INK, clip_on=False)


def outcomes(fonts, share, models):
    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_axes([0.075, 0.2, 0.905, 0.585])
    xs_of = lambda m: [models.index(m) * 3.7 + k * 1.0 for k in range(3)]  # noqa: E731
    for m in models:
        for x, lv in zip(xs_of(m), LEVELS):
            v = share[f"{m}@{lv}"]
            bottom = 0
            for f, (_, colour) in enumerate(GROUPS):
                h = 100 * v[f] / sum(v)
                ax.bar(x, h, bottom=bottom, width=0.86, color=colour, edgecolor="white", linewidth=0.8, zorder=2)
                if h >= 8:
                    ax.text(x, bottom + h / 2, f"{h:.0f}", ha="center", va="center", fontproperties=fonts.semibold,
                            fontsize=10, color="white", zorder=3)
                bottom += h
    last = xs_of(models[-1])[-1]
    ax.set_xlim(-0.7, last + 0.7)
    ax.set_ylim(0, 100)
    ax.set_xticks([])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    style_axes(ax, fonts)
    group_labels(fig, ax, fonts, models, xs_of, 100, -13.5)
    fig.text(0.04, 0.965, "Does thinking change where the points land?", fontproperties=fonts.bold, fontsize=24,
             color=INK)
    fig.text(0.04, 0.937, "Each model at its least, medium and most thinking: every correct point, wrong point and "
             "unfound object, as a share.", fontproperties=fonts.regular, fontsize=13, color=INK_2)
    for k, ((_, colour), name) in enumerate(zip(GROUPS, NAMES)):
        x, y = (0.04, 0.895) if k == 0 else (0.04 + ((k - 1) % 2) * 0.39, 0.865 - ((k - 1) // 2) * 0.03)
        fig.patches.append(plt.Rectangle((x, y - 0.004), 0.016, 0.016, color=colour, transform=fig.transFigure,
                                         figure=fig))
        fig.text(x + 0.024, y + 0.004, name, va="center", fontproperties=fonts.medium, fontsize=13, color=INK)
    n = sum(1 for _ in scored())
    fig.text(0.04, 0.034, f"{n} answers: {len(models)} models x 3 thinking settings, 11 photos, 3 tries (one GPT-5.6 Terra "
             "answer at max never arrived). \"off\": thinking turned off; \"low\" or\n\"minimal\": the model cannot turn "
             "it off, its least. Medium is the setting on the published charts. Sorted by correct points at medium. "
             "Numbers under 8% not printed.",
             fontproperties=fonts.regular, fontsize=10.5, color=INK_3, va="center", linespacing=1.5)
    return fig


def thinking_tokens(fonts, models):
    tok = {}
    for r in scored():
        f = ROOT / "results" / r["source_file"]
        tok.setdefault(r["entrant"], []).append((f, r["source_line"]))
    lines = {}
    vals = {}
    for e, refs in tok.items():
        for f, n in refs:
            if f not in lines:
                lines[f] = f.read_text().splitlines()
            raw = json.loads(lines[f][n - 1])
            t = (raw.get("usage") or {}).get("reasoning_tokens")
            if t is None:
                raise SystemExit(f"no thinking count for {e} ({raw['provider']})")
            vals.setdefault(e, []).append(t)
    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_axes([0.095, 0.2, 0.885, 0.68])
    xs_of = lambda m: [models.index(m) * 3.7 + k * 1.0 for k in range(3)]  # noqa: E731
    colours = {"min": "#9fd3d4", "medium": "#3aa5a7", "max": "#0b6466"}
    top = max(statistics.mean(v) + statistics.pstdev(v) for v in vals.values()) * 1.1
    rows = []
    for m in models:
        for x, lv in zip(xs_of(m), LEVELS):
            v = vals[f"{m}@{lv}"]
            mean, sd = statistics.mean(v), statistics.stdev(v)
            rows.append({"entrant": f"{m}@{lv}", "mean": round(mean, 1), "sd": round(sd, 1),
                         "median": statistics.median(v), "max": max(v), "no_thinking": sum(t == 0 for t in v)})
            ax.bar(x, mean, width=0.86, color=colours[lv], zorder=2)
            if sd > 0:
                lo, hi = max(0, mean - sd), mean + sd
                ax.plot([x, x], [lo, hi], color=INK, lw=1.3, zorder=3)
                for y in (lo, hi):
                    ax.plot([x - 0.2, x + 0.2], [y, y], color=INK, lw=1.3, zorder=3)
            ax.text(x, (mean + sd if sd else mean) + top * 0.01, f"{mean:,.0f}", ha="center", va="bottom",
                    rotation=90, fontproperties=fonts.semibold, fontsize=9.5, color=INK_2, zorder=3)
    last = xs_of(models[-1])[-1]
    ax.set_xlim(-0.7, last + 0.7)
    ax.set_ylim(0, top)
    ax.set_xticks([])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(axis="y", color=GRID, zorder=0)
    style_axes(ax, fonts)
    group_labels(fig, ax, fonts, models, xs_of, top, -top * 0.135)
    fig.text(0.04, 0.965, "How much each model thought at each setting", fontproperties=fonts.bold, fontsize=24,
             color=INK)
    fig.text(0.04, 0.937, "Thinking tokens per answer on the eleven photos. Bar: average. Line: how much it varies "
             "from one answer to the next.", fontproperties=fonts.regular, fontsize=13, color=INK_2)
    fig.text(0.04, 0.034, "33 answers per bar (11 photos, 3 tries; Terra at max 32). Line: one standard deviation either "
             "side, cut at 0; none when every answer is the same.\nCounts as each provider reports them. A Claude answer may "
             "use at most 32,000 tokens, thinking included; the ChatGPT plan set no such limit.",
             fontproperties=fonts.regular, fontsize=10.5, color=INK_3, va="center", linespacing=1.5)
    return fig, rows


def main():
    fonts = _fonts()
    plt.rcParams.update({"svg.fonttype": "path", "pdf.fonttype": 42, "font.family": fonts.regular.get_name()})
    OUT.mkdir(parents=True, exist_ok=True)
    share = outcome_shares()
    models = models_in_order(share)
    fig = outcomes(fonts, share, models)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"outcomes.{ext}", dpi=170 if ext == "png" else None, facecolor="white")
    plt.close(fig)
    fig, tok_rows = thinking_tokens(fonts, models)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"thinking-tokens.{ext}", dpi=170 if ext == "png" else None, facecolor="white")
    plt.close(fig)

    recs = scored()
    score = task_means(recs, lambda r: r["score"])
    # correct share per answer, from the sorted mistakes (the chart's green), averaged per task
    per_answer = {}
    for r in csv.DictReader(open(KINDS / "points" / "items.csv")):
        k = (r["entrant"], r["task"], r["repeat"])
        c = per_answer.setdefault(k, [0, 0])
        c[0] += r["kind"] == "correct"
        c[1] += 1
    correct = {}
    for (e, t, _), (ok, n) in per_answer.items():
        correct.setdefault(e, {}).setdefault(t, []).append(ok / n)
    correct = {e: {t: statistics.mean(v) for t, v in ts.items()} for e, ts in correct.items()}
    summary = {r["entrant"]: r for r in csv.DictReader(open(RES / "summary.csv"))}
    out = []
    for m in models:
        for lv in LEVELS:
            e = f"{m}@{lv}"
            row = {"model": m, "label": describe(m)[0], "level": lv, "setting": LEAST[m] if lv == "min" else lv,
                   "score": round(100 * float(summary[e]["score"]), 1),
                   "correct_share": round(100 * statistics.mean(correct[e].values()), 1),
                   "seconds": round(float(summary[e]["seconds"]), 1)}
            if lv != "medium":
                for name, d in (("score", score), ("correct_share", correct)):
                    mid, lo, hi = paired(d[e], d[f"{m}@medium"])
                    row[f"{name}_vs_medium"] = round(100 * mid, 1)
                    row[f"{name}_vs_medium_95"] = f"{100 * lo:+.1f} to {100 * hi:+.1f}"
            out.append(row)
    fields = list(out[1].keys() | out[0].keys())
    order = ["model", "label", "level", "setting", "score", "score_vs_medium", "score_vs_medium_95",
             "correct_share", "correct_share_vs_medium", "correct_share_vs_medium_95", "seconds"]
    with open(OUT / "per-level.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in order if k in fields])
        w.writeheader()
        w.writerows(out)
    with open(OUT / "thinking-tokens.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tok_rows[0]))
        w.writeheader()
        w.writerows(tok_rows)
    print(OUT)


if __name__ == "__main__":
    main()
