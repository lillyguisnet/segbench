# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9", "svgelements>=1.9"]
# ///
"""Thinking tokens per answer, every language model at every thinking level we have on the eleven photos.

    uv run scripts/plot_thinking_levels_box.py  ->  charts/thinking-levels-v1/thinking-tokens-box.{png,svg,pdf},
                                                     thinking-tokens-box.csv

The box chart of scripts/plot_thinking_tokens.py (same box rules, log scale, "none" lane), one box per level:
- least, medium and max for the eight models of the thinking-levels run (results/thinking-levels-v1, Muse
  Wonder on our plan);
- medium only for the other eleven language models of point v3 (results/point-benchmark-v3), including
  Muse Wonder through the API. Their level label is the setting actually sent: GLM, Kimi and Mistral "high".
Perceptron Mk1.5 with thinking off exists on the first four photos only (2026-10-13): not comparable, left out.
Groups sorted by the median at medium, as the medium-only box chart.
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
from plot_thinking_tokens import box_stats  # noqa: E402
from segbench.chart import _fonts  # noqa: E402
from segbench.chart_data import describe  # noqa: E402
from segbench.chart_editorial import GRID, INK, INK_2, INK_3  # noqa: E402

SOURCES = (ROOT / "results/thinking-levels-v1/scored.jsonl", ROOT / "results/point-benchmark-v3/scored.jsonl")
OUT = ROOT / "charts/thinking-levels-v1"
LEVELS = ("min", "medium", "max")
COLOUR = {"min": "#9fd3d4", "medium": "#3aa5a7", "max": "#0b6466"}
MEDIAN = {"min": "#0b6466", "medium": "white", "max": "white"}
# the setting actually sent (segbench/models.py THINKING)
LEAST = {"sonnet": "off", "haiku": "off", "terra": "off", "luna": "off",
         "opus": "low", "astra": "low", "sol": "low", "muse-wonder": "minimal"}
MEDIUM_IS_HIGH = {"glm-5.3-flash", "kimi-k3", "mistral-large-4"}
FLOOR = 4  # where 0 is drawn; the smallest non-zero count is 11 (checked)


def load():
    """{model: {level: [thinking tokens]}}; an entrant's answers come from the first source that has it."""
    lines, data, seen = {}, {}, set()
    for src in SOURCES:
        here = set()
        for r in map(json.loads, open(src)):
            e = r["entrant"]
            if "@" not in e or e in seen:
                continue
            here.add(e)
            f = ROOT / "results" / r["source_file"]
            if f not in lines:
                lines[f] = f.read_text().splitlines()
            raw = json.loads(lines[f][r["source_line"] - 1])
            assert raw["model"] == r["model"] and raw["task"] == r["task"], r
            t = (raw.get("usage") or {}).get("reasoning_tokens")
            if t is None:
                if raw["provider"] != "gemini":  # Gemini omits the count when it did not think (checked)
                    raise SystemExit(f"no thinking count from {raw['provider']} for {e}")
                t = 0
            model, level = e.split("@")
            data.setdefault(model, {}).setdefault(level, []).append(t)
        seen |= here
    return data


def setting(model, level):
    if level == "min":
        return LEAST[model]
    if level == "medium" and model in MEDIUM_IS_HIGH:
        return "high"
    return level


def main():
    data = load()
    models = sorted(data, key=lambda m: (-statistics.median(data[m]["medium"]), -statistics.mean(data[m]["medium"])))
    fonts = _fonts()
    plt.rcParams.update({"svg.fonttype": "path", "pdf.fonttype": 42, "font.family": fonts.regular.get_name()})
    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_axes([0.085, 0.3, 0.9, 0.515])
    ax.set_yscale("log")
    put = lambda x: max(x, FLOOR)  # noqa: E731
    lo, hi = FLOOR / 1.6, 70000
    at = lambda frac: lo * (hi / lo) ** frac  # noqa: E731  (axes fraction -> data, log scale)

    x, boxes, rows, centres = 0.0, [], [], {}
    for m in models:
        xs = []
        for lv in LEVELS:
            if lv in data[m]:
                boxes.append((x, m, lv, data[m][lv]))
                xs.append(x)
                x += 1.0
        centres[m] = statistics.mean(xs)
        x += 0.75  # gap between models
    for pos, m, lv, v in boxes:
        b = box_stats(v)
        st = {k: ([put(y) for y in val] if k == "fliers" else put(val)) for k, val in b.items()}
        ax.bxp([st], positions=[pos], widths=0.74, patch_artist=True, showfliers=True,
               boxprops={"facecolor": COLOUR[lv], "edgecolor": COLOUR[lv], "linewidth": 1.0},
               medianprops={"color": MEDIAN[lv], "linewidth": 2.2, "solid_capstyle": "butt"},
               whiskerprops={"color": INK_2, "linewidth": 1.1}, capprops={"color": INK_2, "linewidth": 1.1},
               flierprops={"marker": "o", "markersize": 4, "markerfacecolor": "white", "markeredgecolor": INK_2,
                           "markeredgewidth": 1.0}, zorder=3)
        if st["q1"] == st["q3"]:  # zero-height box: draw a thick bar so it does not vanish
            ax.plot([pos - 0.37, pos + 0.37], [st["med"]] * 2, color=COLOUR[lv], lw=5, solid_capstyle="butt",
                    zorder=4)
        med = statistics.median(v)
        xt = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(pos, 1.015, f"{med:,.0f}", transform=xt, ha="center", va="bottom", rotation=90,
                fontproperties=fonts.semibold, fontsize=8.5, color=INK_2)
        ax.text(pos, at(-0.012), setting(m, lv), ha="right", va="top", rotation=60, rotation_mode="anchor",
                fontproperties=fonts.regular, fontsize=8.5, color=INK_2, clip_on=False)
        q1, _, q3 = statistics.quantiles(v, n=4, method="inclusive")
        rows.append({"model": m, "label": describe(m)[0], "level": lv, "setting": setting(m, lv), "answers": len(v),
                     "median": med, "q1": round(q1, 1), "q3": round(q3, 1), "min": min(v), "max": max(v),
                     "mean": round(statistics.mean(v), 1), "no_thinking": sum(t == 0 for t in v)})
    xt = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transAxes)
    ax.text(-0.012, 1.015, "middle\nanswer", transform=xt, ha="right", va="bottom", fontproperties=fonts.regular,
            fontsize=9, color=INK_3, linespacing=1.1)

    ax.set_ylim(lo, hi)
    ax.set_yticks([FLOOR, 10, 100, 1000, 10000])
    ax.set_yticklabels(["none", "10", "100", "1,000", "10,000"])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    for y in (10, 100, 1000, 10000):
        ax.axhline(y, color=GRID, zorder=0, lw=1)
    ax.axhline(FLOOR, color=GRID, zorder=0, lw=1, ls=(0, (3, 3)))
    yb = FLOOR * 1.55  # break mark between "none" and 10
    bt = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for dy in (0.95, 1.06):
        ax.plot([-0.016, 0.004], [yb * dy / 1.05, yb * dy * 1.05], color=INK_3, lw=1.2, clip_on=False, transform=bt)
    for m in models:
        logo(ax, centres[m], at(-0.115), describe(m)[1], 19, fonts)
        ax.text(centres[m] + 0.25, at(-0.16), describe(m)[0], ha="right", va="top", rotation=60, rotation_mode="anchor",
                fontproperties=fonts.medium, fontsize=11, color=INK, clip_on=False)
    ax.set_xlim(-0.7, x - 0.75 + 0.2)
    ax.set_xticks([])
    for lab in ax.get_yticklabels():
        lab.set_fontproperties(fonts.regular)
        lab.set_fontsize(11)
        lab.set_color(INK_2)
    ax.tick_params(axis="y", length=0, pad=6)
    for side in ax.spines.values():
        side.set_visible(False)

    fig.text(0.04, 0.965, "How much each model thinks, at each setting", fontproperties=fonts.bold, fontsize=24,
             color=INK)
    fig.text(0.04, 0.937, "Thinking tokens per answer on the eleven photos. Box: the middle half of the answers; "
             "line: the middle answer (number above).", fontproperties=fonts.regular, fontsize=13, color=INK_2)
    for k, (lv, name) in enumerate((("min", "Least thinking (off, or the lowest the model allows)"),
                                    ("medium", "Medium (the published charts)"), ("max", "Maximum"))):
        x0 = (0.04, 0.47, 0.75)[k]
        fig.patches.append(plt.Rectangle((x0, 0.894), 0.016, 0.016, color=COLOUR[lv], transform=fig.transFigure,
                                         figure=fig))
        fig.text(x0 + 0.024, 0.902, name, va="center", fontproperties=fonts.medium, fontsize=12, color=INK)
    n = sum(r["answers"] for r in rows)
    fig.text(0.04, 0.035, f"{n:,} answers, 11 photos, 3 tries per setting (Terra at max: 32). Eight models were run at "
             "three settings; the others have medium only (GLM, Kimi, Mistral: their\nhigh). Muse Wonder: our plan; "
             "Muse Wonder (API): the copy on the published charts. Log scale. Whiskers: furthest answer within 1.5 box "
             "heights,\ndots beyond. Counts as each provider reports them. A Claude answer may use at most 32,000 tokens.",
             fontproperties=fonts.regular, fontsize=10, color=INK_3, va="center", linespacing=1.5)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"thinking-tokens-box.{ext}", dpi=170 if ext == "png" else None, facecolor="white")
    plt.close(fig)
    with open(OUT / "thinking-tokens-box.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(OUT / "thinking-tokens-box.png", len(models), "models,", len(rows), "boxes")


if __name__ == "__main__":
    main()
