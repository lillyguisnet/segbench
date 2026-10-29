# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9", "svgelements>=1.9"]
# ///
"""Charts of the kinds of mistakes (results/error-kinds/v1, scripts/classify_errors.py).

    uv run scripts/plot_error_kinds.py   ->  charts/error-kinds-v1/{overview,by-model,families,families-{points,boxes}[-correct]}.{png,svg,pdf}

1. overview: for points and for boxes, how many mistakes of each kind, all
   models and tries together. Two bar colours: wrong answers (a point or box
   that earned nothing) and objects left unfound. F1 counts each of these the
   same, so their lengths compare. Kinds are grouped in four families.
3. families: for each model (x axis, sorted by its charted score), the share
   of its mistakes in each of the four families: shares within each answer,
   averaged over tries, then over photos with equal weight (as the score);
   plain counts would let the log pile (83 objects) decide most bars. Objects swallowed by a group box
   count as "aimed but off", like the box.
2. by-model: for each model (sorted by its charted score), how many points of
   that score each kind of mistake costs (fix that kind alone, rescore exactly
   as charted). Points left, boxes right. Fixes interact: the cells of a row do
   not add up to what the model misses of 100.
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from segbench.chart import INITIALS, _fonts, logo_path  # noqa: E402
from segbench.chart_data import describe  # noqa: E402
from segbench.chart_editorial import BRAND, GRID, INK, INK_2, INK_3  # noqa: E402

SRC = ROOT / "results/error-kinds/v1"
OUT = ROOT / "charts/error-kinds-v1"

FAMILIES = [  # (name, colour, {benchmark: [(kind, label)]})
    ("Aimed at the right object, landed off", "#e8890c", {
        "points": [("on the object, off-centre", "On the object, outside the accepted circle"),
                   ("near miss", "Just off an object nobody found")],
        "boxes": [("mispositioned box", "Overlaps an object, but under 50%"),
                  ("group box", "One box around several objects")]}),
    ("Extra answer, no object to earn", "#8a4fd6", {
        "points": [("duplicate", "On an object already found"), ("in a gap", "In a gap between found objects"),
                   ("wrong object", "On another thing (look-alike)"), ("background", "On nothing"),
                   ("off image", "Outside the photo")],
        "boxes": [("duplicate", "On an object already found"), ("wrong object", "On another thing (look-alike)"),
                  ("background", "On nothing"), ("off image", "Outside the photo")]}),
    ("Object never found", "#2f6fd6", {
        "points": [("missed", "Missed")], "boxes": [("missed", "Missed")]}),
    ("Whole answer broken", "#d23a3a", {
        "points": [("no answer: empty", "No answer: said none"), ("no answer: stuck", "No answer: stuck in a loop"),
                   ("no answer: unreadable", "No answer: our reader failed"),
                   ("doom loop", "Doom loop in the answer"), ("coordinate mix-up", "Coordinates mixed up (x/y swapped, scale)"),
                   ("shifted or stretched", "Whole answer shifted or stretched")],
        "boxes": [("no answer: empty", "No answer: said none"),
                  ("doom loop", "Doom loop in the answer"), ("coordinate mix-up", "Coordinates mixed up (x/y swapped, scale)"),
                  ("shifted or stretched", "Whole answer shifted or stretched")]}),
]
EXTRA = {"points": ("partial credit", "Correct, only at\nwider circles"),
         "boxes": ("loose fit", "Correct, edges\nnot tight")}
SHORT = {  # column heads of the by-model chart
    "on the object, off-centre": "On object,\noff-centre", "near miss": "Near\nmiss",
    "mispositioned box": "Overlap\nunder 50%", "group box": "Group\nbox", "duplicate": "Dupli-\ncate",
    "in a gap": "In a\ngap", "wrong object": "Wrong\nobject", "background": "On\nnothing", "off image": "Off\nphoto",
    "missed": "Missed", "no answer: empty": "Said\nnone", "no answer: stuck": "Stuck",
    "no answer: unreadable": "Not\nread", "doom loop": "Doom\nloop", "coordinate mix-up": "Coords\nmixed up",
    "shifted or stretched": "Shifted,\nstretched",
}


def lighter(colour, k):
    r, g, b = to_rgb(colour)
    return (1 - k * (1 - r), 1 - k * (1 - g), 1 - k * (1 - b))


def overview(fonts):
    fig, axes = plt.subplots(1, 2, figsize=(15, 8.4), gridspec_kw={"wspace": 0.62})
    for ax, bench, title in zip(axes, ("points", "boxes"), ("Points · 770 answers", "Boxes · 825 answers")):
        items = list(csv.DictReader(open(SRC / bench / "items.csv")))
        wrong, unfound = {}, {}
        for r in items:
            d = unfound if r["item"] == "object" else wrong
            d[r["kind"]] = d.get(r["kind"], 0) + 1
        rows = []  # (label, wrong, unfound, colour, family or None)
        for fam, colour, kinds in FAMILIES:
            present = [(k, lab) for k, lab in kinds[bench] if wrong.get(k) or unfound.get(k)]
            if not present:
                continue
            rows.append((fam, None, None, colour, True))
            rows += [(lab, wrong.get(k, 0), unfound.get(k, 0), colour, False) for k, lab in present]
        y = 0
        top = max(w + u for _, w, u, _, f in rows if not f)
        for lab, w, u, colour, is_fam in rows:
            if is_fam:
                y += 0.45 if y else 0
                ax.text(-0.012, y, lab, transform=ax.get_yaxis_transform(), ha="right", va="center",
                        fontproperties=fonts.semibold, fontsize=11.5, color=colour)
                y += 1
                continue
            ax.barh(y, w, height=0.72, color=colour, zorder=2)
            ax.barh(y, u, left=w, height=0.72, color=lighter(colour, 0.38), zorder=2, hatch="////",
                    edgecolor="white", linewidth=0)
            ax.text(-0.012, y, lab, transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontproperties=fonts.regular, fontsize=10.5, color=INK)
            ax.text(w + u + 70, y, f"{w + u:,}", va="center", fontproperties=fonts.medium,
                    fontsize=9.5, color=INK_2)
            y += 1
        ax.set_ylim(y - 0.4, -0.8)
        ax.set_xlim(0, 5900)  # same scale on both panels
        ax.set_yticks([])
        ax.set_title(title, loc="left", fontproperties=fonts.semibold, fontsize=14, color=INK, pad=12, x=-0.6)
        ax.grid(axis="x", color=GRID, zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="x", colors=INK_3, labelsize=9)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color="#777"), Patch(facecolor=lighter("#777", 0.38), hatch="////", edgecolor="white")],
               labels=["Wrong answer (a point or box that earned nothing)", "Object left unfound because of it"],
               loc="lower left", bbox_to_anchor=(0.02, -0.005), ncol=2, frameon=False, prop=fonts.regular, fontsize=10.5)
    fig.suptitle("What went wrong, on the eleven photos", x=0.02, ha="left", y=0.995,
                 fontproperties=fonts.bold, fontsize=18, color=INK)
    fig.text(0.02, 0.945, "Every model and every try. The score counts a wrong answer and an unfound object the same, "
             "so bar lengths compare.", fontproperties=fonts.regular, fontsize=11, color=INK_2)
    fig.subplots_adjust(left=0.235, right=0.985, top=0.86, bottom=0.08)
    return fig


def by_model(fonts):
    ncols = {}
    for bench in ("points", "boxes"):
        rows = list(csv.DictReader(open(SRC / bench / "by-model.csv")))
        ncols[bench] = 1 + sum(any(float(r[f"gain:{k}"]) >= 0.0005 for r in rows)
                               for _, _, kinds in FAMILIES for k, _ in kinds[bench])
    fig, axes = plt.subplots(1, 2, figsize=(24, 11.6),
                             gridspec_kw={"wspace": 0.22, "width_ratios": [ncols["points"], ncols["boxes"]]})
    cmap = LinearSegmentedColormap.from_list("loss", ["#ffffff", "#fde3c8", "#f6a35b", "#d9480f", "#7c1d06"])
    grey = LinearSegmentedColormap.from_list("fit", ["#ffffff", "#c9ced8", "#5b6270"])
    vmax = 30
    for ax, bench, title in zip(axes, ("points", "boxes"), ("Points", "Boxes")):
        rows = list(csv.DictReader(open(SRC / bench / "by-model.csv")))
        cols = [(k, colour) for _, colour, kinds in FAMILIES for k, _ in kinds[bench]
                if any(float(r[f"gain:{k}"]) >= 0.0005 for r in rows)]
        cols.append((EXTRA[bench][0], INK_3))
        for i, r in enumerate(rows):
            for j, (k, _) in enumerate(cols):
                v = 100 * float(r[f"gain:{k}"])
                x = j + (0.35 if j == len(cols) - 1 else 0)
                last = j == len(cols) - 1
                if v >= 1:
                    ax.add_patch(plt.Rectangle((x - 0.47, i - 0.45), 0.94, 0.9, zorder=2,
                                               color=(grey if last else cmap)(min(v, vmax) / vmax)))
                    ax.text(x, i, f"{int(v + 0.5)}", ha="center", va="center", fontproperties=fonts.medium,
                            fontsize=10, color="white" if v > 14 else INK, zorder=3)
                elif v >= 0.05:
                    ax.text(x, i, "·", ha="center", va="center", fontsize=10, color=INK_3, zorder=3)
        n = len(rows)
        ax.set_xlim(-0.6, len(cols) - 0.15)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_yticks(range(n))
        ax.set_yticklabels([f"{r['label']}  {100 * float(r['score']):.0f}" for r in rows])
        for lab in ax.get_yticklabels():
            lab.set_fontproperties(fonts.regular)
            lab.set_fontsize(10.5)
            lab.set_color(INK)
        ax.set_xticks([j + (0.35 if j == len(cols) - 1 else 0) for j in range(len(cols))])
        ax.set_xticklabels([SHORT.get(k, EXTRA[bench][1]) for k, _ in cols])
        ax.xaxis.tick_top()
        for lab, (_, colour) in zip(ax.get_xticklabels(), cols):
            lab.set_fontproperties(fonts.semibold)
            lab.set_fontsize(9.5)
            lab.set_color(colour)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        for i in range(n):
            ax.axhline(i + 0.5, color=GRID, lw=0.6, zorder=1)
        ax.axvline(len(cols) - 1.3, color=INK_3, lw=0.8, ls=(0, (2, 3)))
        ax._panel_title = f"{title}: model and its score"
    fig.suptitle("Score points each kind of mistake costs each model", x=0.012, ha="left", y=0.995,
                 fontproperties=fonts.bold, fontsize=18, color=INK)
    fig.text(0.012, 0.955, "How much higher the model's charted score (out of 100) would be if only that kind of "
             "mistake were fixed. Fixes overlap, so a row does not add up. "
             "Column colours: orange aimed but off · purple extra · blue missed · red broken answer. "
             "· under one point.", fontproperties=fonts.regular, fontsize=10.5, color=INK_2)
    fig.text(0.012, 0.935, "Last column (after the dashed line) is not a mistake: answers that already count, "
             "but would count more if exact.", fontproperties=fonts.regular, fontsize=10.5, color=INK_2)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.83, bottom=0.015)
    for ax, x in zip(axes, (0.012, None)):
        pos = ax.get_position()
        fig.text(x if x is not None else pos.x0 - 0.075, pos.y1 + 0.06, ax._panel_title, ha="left",
                 fontproperties=fonts.semibold, fontsize=14, color=INK)
    return fig


FAMILY_OF = {"group box": 0, "maybe missing from key": 1}
for _n, (_f, _c, _kinds) in enumerate(FAMILIES):
    for _b in _kinds.values():
        for _k, _ in _b:
            FAMILY_OF.setdefault(_k, _n)


def shares(bench, groups_of, n_groups):
    """Per model: share of each group, as the score averages: within each answer, then over its tries,
    then over the photos (equal weight). groups_of(kind) -> group index, or None to leave the item out."""
    per = {}
    for r in csv.DictReader(open(SRC / bench / "items.csv")):
        g = groups_of(r["kind"])
        if g is not None:
            per.setdefault((r["entrant"], r["task"], r["repeat"]), [0.0] * n_groups)[g] += 1
    by = {}
    for (e, t, _), v in per.items():
        tot = sum(v)
        by.setdefault(e, {}).setdefault(t, []).append([x / tot for x in v])
    out = {}
    for e, tasks in by.items():
        per_task = [[sum(col) / len(tries) for col in zip(*tries)] for tries in tasks.values()]
        out[e] = [sum(col) / len(per_task) for col in zip(*per_task)]
    return out


def logo(ax, x, y, maker, d, fonts):
    """The maker's logo in a white disc, as on the bubble charts (data x, y; d in points)."""
    ax.scatter([x], [y], s=d ** 2, facecolors="white", edgecolors=[GRID], linewidths=1.0, zorder=4, clip_on=False)
    p, size, colour = logo_path(maker), d * 0.56, BRAND.get(maker, INK)
    if p is not None:
        ax.scatter([x], [y], s=size ** 2, marker=p, facecolors=[colour], edgecolors="none", zorder=5, clip_on=False)
    else:
        text = INITIALS.get(maker) or "".join(w[0] for w in maker.split())[:3]
        ax.text(x, y, text, ha="center", va="center_baseline", fontproperties=fonts.bold,
                fontsize=size * (0.6 if len(text) < 3 else 0.44), color=colour, zorder=5, clip_on=False)


def families(fonts):
    fig, axes = plt.subplots(2, 1, figsize=(17, 12.2), gridspec_kw={"hspace": 0.7})
    for ax, bench, title in zip(axes, ("points", "boxes"), ("Points", "Boxes")):
        share = shares(bench, lambda k: None if k == "correct" else FAMILY_OF[k], len(FAMILIES))
        rows = list(csv.DictReader(open(SRC / bench / "by-model.csv")))  # sorted by score
        for i, r in enumerate(rows):
            v = share[r["entrant"]]
            tot = sum(v)
            bottom = 0
            for f, (name, colour, _) in enumerate(FAMILIES):
                h = 100 * v[f] / tot
                ax.bar(i, h, bottom=bottom, width=0.74, color=colour, zorder=2)
                if h >= 7:
                    ax.text(i, bottom + h / 2, f"{h:.0f}", ha="center", va="center", fontproperties=fonts.medium,
                            fontsize=8.5, color="white", zorder=3)
                bottom += h
            logo(ax, i, -5.5, describe(r["entrant"].split("@")[0])[1], 17, fonts)
            ax.text(i, 102, f"{100 * float(r['score']):.0f}", ha="center", va="bottom", fontproperties=fonts.semibold,
                    fontsize=9, color=INK_2)
        ax.text(-0.9, 102, "score", ha="right", va="bottom", fontproperties=fonts.regular, fontsize=9, color=INK_3)
        ax.set_xlim(-0.6, len(rows) - 0.4)
        ax.set_ylim(0, 100)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([r["label"] for r in rows], rotation=40, ha="right", rotation_mode="anchor")
        for lab in ax.get_xticklabels():
            lab.set_fontproperties(fonts.regular)
            lab.set_fontsize(10)
            lab.set_color(INK)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.tick_params(axis="y", colors=INK_3, labelsize=9, length=0)
        ax.tick_params(axis="x", length=0, pad=22)  # room for the logo between bar and name
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(f"{title}: share of each model's mistakes", loc="left", fontproperties=fonts.semibold,
                     fontsize=14, color=INK, pad=24)
    from matplotlib.patches import Patch
    names = [n + (" (no answer, doom loop, wrong coordinates)" if n == "Whole answer broken" else "")
             for n, _, _ in FAMILIES]
    fig.legend(handles=[Patch(color=c) for _, c, _ in FAMILIES], labels=names,
               loc="upper left", bbox_to_anchor=(0.045, 0.935), ncol=4, frameon=False, prop=fonts.medium, fontsize=11)
    fig.suptitle("Where each model's mistakes go", x=0.045, ha="left", y=0.995,
                 fontproperties=fonts.bold, fontsize=18, color=INK)
    fig.text(0.045, 0.952, "Models sorted by their charted score (number above each bar). Mistakes = wrong answers + "
             "objects left unfound; shares taken within each answer, then averaged over tries and photos, as the score is.",
             fontproperties=fonts.regular, fontsize=10.5, color=INK_2)
    fig.subplots_adjust(left=0.045, right=0.99, top=0.85, bottom=0.12)
    return fig


SQUARE = {  # words of the square charts, per benchmark
    "points": {
        "names": ["Correct point", "Aimed at the right object, landed off", "Extra point, no object there",
                  "Object never found", "Whole answer broken (no answer, doom loop, wrong coordinates)"],
        "title": ("What happened to each model's points and objects", "Where each model's pointing mistakes go"),
        "sub": ("Every correct point, wrong point and unfound object, as a share. Models sorted by correct points.",
                "Share of each model's mistakes in four kinds. Models in the same order as the correct-points chart."),
        "note": ("770 answers: 28 models, 11 photos, 3 tries (local models 1). Shares taken within each answer, then averaged "
                 "over tries and photos (every photo\ncounts the same). Numbers under 8% not printed. "
                 "Correct: within the accepted circle around a real object.",
                 "770 answers: 28 models, 11 photos, 3 tries (local models 1). Mistakes = wrong points + "
                 "objects left unfound. Shares taken within each\nanswer, then averaged over tries and photos "
                 "(every photo counts the same). Numbers under 8% not printed."),
    },
    "boxes": {  # Maxime, 2026-10-21: the same two charts for the box benchmark
        "names": ["Correct box", "Right object, box off or around several", "Extra box: on nothing, a look-alike or a repeat",
                  "Object never found", "Whole answer broken (no answer, doom loop, wrong coordinates)"],
        "title": ("What happened to each model's boxes and objects", "Where each model's box mistakes go"),
        "sub": ("Every correct box, wrong box and unfound object, as a share. Models sorted by correct boxes.",
                "Share of each model's mistakes in four kinds. Models in the same order as the correct-boxes chart."),
        "note": ("825 answers: 27 models, 11 photos, 3 tries (SAM 3, YOLOE, RF-DETR 1). Shares taken within each answer, then "
                 "averaged over tries and photos (every\nphoto counts the same). Numbers under 8% not printed. "
                 "Correct: the box and a real object's box share at least half of their combined area.",
                 "825 answers: 27 models, 11 photos, 3 tries (SAM 3, YOLOE, RF-DETR 1). Mistakes = wrong boxes + objects "
                 "left unfound. Shares taken within\neach answer, then averaged over tries and photos (every photo counts "
                 "the same). Numbers under 8% not printed. Orange includes one box around several objects."),
    },
}


def families_square(fonts, bench, with_correct=False):
    """One benchmark's half of chart 3 alone, square and larger (Maxime: points 2026-10-19, boxes 2026-10-21)."""
    words = SQUARE[bench]
    groups = ([("Correct", "#2f9e4f")] if with_correct else []) + [(n, c) for n, c, _ in FAMILIES]
    share = shares(bench, lambda k: (0 if with_correct else None) if k == "correct" else FAMILY_OF[k] + with_correct,
                   len(groups))
    # sorted by share of correct answers (points: widest circle; boxes: 50% overlap), same order on both square charts
    correct = shares(bench, lambda k: 0 if k == "correct" else 1, 2)
    rows = sorted(csv.DictReader(open(SRC / bench / "by-model.csv")), key=lambda r: -correct[r["entrant"]][0])
    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_axes([0.075, 0.235, 0.905, 0.54 if with_correct else 0.57])
    for i, r in enumerate(rows):
        v = share[r["entrant"]]
        bottom = 0
        for f, (_, colour) in enumerate(groups):
            h = 100 * v[f] / sum(v)
            ax.bar(i, h, bottom=bottom, width=0.8, color=colour, edgecolor="white", linewidth=0.8, zorder=2)
            if h >= 8:
                ax.text(i, bottom + h / 2, f"{h:.0f}", ha="center", va="center", fontproperties=fonts.semibold,
                        fontsize=10.5, color="white", zorder=3)
            bottom += h
        logo(ax, i, -4.2, describe(r["entrant"].split("@")[0])[1], 21, fonts)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylim(0, 100)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r["label"] for r in rows], rotation=60, ha="right", rotation_mode="anchor")
    for lab in ax.get_xticklabels():
        lab.set_fontproperties(fonts.medium)
        lab.set_fontsize(12)
        lab.set_color(INK)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    for lab in ax.get_yticklabels():
        lab.set_fontproperties(fonts.regular)
        lab.set_fontsize(11)
        lab.set_color(INK_2)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.tick_params(axis="x", length=0, pad=38)
    for side in ax.spines.values():
        side.set_visible(False)
    title, sub = words["title"][not with_correct], words["sub"][not with_correct]
    fig.text(0.04, 0.965, title, fontproperties=fonts.bold, fontsize=24, color=INK)
    fig.text(0.04, 0.937, sub, fontproperties=fonts.regular, fontsize=13, color=INK_2)
    names = words["names"][0 if with_correct else 1:]
    for k, ((_, colour), name) in enumerate(zip(groups, names)):
        if with_correct:  # five entries: correct alone on the first row
            x, y = (0.04, 0.895) if k == 0 else (0.04 + ((k - 1) % 2) * 0.39, 0.865 - ((k - 1) // 2) * 0.03)
        else:
            x, y = 0.04 + (k % 2) * 0.39, 0.895 - (k // 2) * 0.03
        fig.patches.append(plt.Rectangle((x, y - 0.004), 0.016, 0.016, color=colour, transform=fig.transFigure,
                                         figure=fig))
        fig.text(x + 0.024, y + 0.004, name, va="center", fontproperties=fonts.medium, fontsize=13, color=INK)
    note = words["note"][not with_correct]
    fig.text(0.04, 0.034, note, fontproperties=fonts.regular, fontsize=10.5, color=INK_3,
             va="center", linespacing=1.5)
    return fig


def main():
    fonts = _fonts()
    plt.rcParams.update({"svg.fonttype": "path", "pdf.fonttype": 42, "font.family": fonts.regular.get_name(),
                         "hatch.linewidth": 0.8})
    OUT.mkdir(parents=True, exist_ok=True)
    for name, make in (("overview", overview), ("by-model", by_model), ("families", families),
                       ("families-points", lambda f: families_square(f, "points")),
                       ("families-points-correct", lambda f: families_square(f, "points", True)),
                       ("families-boxes", lambda f: families_square(f, "boxes")),
                       ("families-boxes-correct", lambda f: families_square(f, "boxes", True))):
        fig = make(fonts)
        for ext in ("png", "svg", "pdf"):
            fig.savefig(OUT / f"{name}.{ext}", dpi=170 if ext == "png" else None, facecolor="white")
        plt.close(fig)
        print(OUT / f"{name}.png")


if __name__ == "__main__":
    main()
