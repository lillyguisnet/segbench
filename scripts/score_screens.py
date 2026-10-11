# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow>=10"]
# ///
"""Score the screen answers against the answer key, and draw every click.

    uv run scripts/score_screens.py results/run-screens-1.jsonl --key annotations/screens/key-draft.json

Writes results/screens-<run>/: one picture per screenshot (the key's boxes in
green, each click as a dot: green inside a target, red outside), scores.csv
(per model and task), and summary.md. A click counts once: one-to-one
matching of clicks to boxes (maximum matching), score = F1 per answer, then
the mean over a model's tries, then over the five tasks.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from segbench.chart_data import describe  # noqa: E402
from segbench.point_score import maximum_matching  # noqa: E402
from segbench.screens import BY_KEY as TASKS, hits  # noqa: E402


def score(clicks_px, targets):
    edges = [hits(targets, x, y) for x, y in clicks_px]
    tp = len(maximum_matching(edges))
    n, m = len(clicks_px), len(targets)
    return {"tp": tp, "fp": n - tp, "fn": m - tp, "f1": 2 * tp / (n + m) if n + m else 1.0, "hits": edges}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--key", type=Path, required=True)
    args = ap.parse_args()
    key = json.loads(args.key.read_text())
    out = ROOT / "results" / f"screens-{args.run.stem.removeprefix('run-')}"
    out.mkdir(parents=True, exist_ok=True)

    rows, dots = [], defaultdict(list)
    for line in args.run.read_text().splitlines():
        r = json.loads(line)
        if r["task"] not in key["tasks"]:
            continue  # a question this key does not answer (e.g. a retired one)
        t = key["tasks"][r["task"]]
        if r["image_size"] != t["image_size"]:
            raise ValueError(f"{r['task']}: image size {r['image_size']} differs from the key's")
        W, H = r["image_size"]
        clicks = [(c["x"] * W / 1000, c["y"] * H / 1000) for c in r.get("clicks", [])] if r.get("readable") else []
        s = score(clicks, t["targets"])
        label = describe(r["model"])[0]
        rows.append({"model": r["model"], "label": label, "task": r["task"], "repeat": r["repeat"],
                     "clicks": len(clicks), **{k: s[k] for k in ("tp", "fp", "fn", "f1")},
                     "readable": r.get("readable", False), "seconds": r.get("seconds"), "cost_usd": r.get("cost_usd")})
        for (x, y), hit in zip(clicks, s["hits"]):
            dots[r["task"]].append((x, y, bool(hit)))

    with (out / "scores.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["model"]][r["task"]].append(r["f1"])
    tasks = list(key["tasks"])
    table = []
    for m, per in by.items():
        task_means = {t: statistics.mean(per[t]) for t in tasks}
        cost = [r["cost_usd"] for r in rows if r["model"] == m and r["cost_usd"] is not None]
        secs = [r["seconds"] for r in rows if r["model"] == m]
        table.append((statistics.mean(task_means.values()), describe(m)[0], task_means,
                      statistics.median(secs), sum(cost) / len(cost) * 1000 if cost else None))
    table.sort(key=lambda x: -x[0])
    lines = [f"# Screen clicks, {args.run.name}, key {args.key.name} ({key['status'].split(':')[0]})", "",
             "| model | score | " + " | ".join(tasks) + " | seconds | $ / 1,000 screenshots |",
             "|---|---|" + "---|" * len(tasks) + "---|---|"]
    for total, label, tm, sec, cost in table:
        lines.append(f"| {label} | {total * 100:.0f} | " + " | ".join(f"{tm[t] * 100:.0f}" for t in tasks)
                     + f" | {sec:.0f} | {'' if cost is None else f'{cost:.2f}'} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    font = ImageFont.load_default(size=20)
    for t in (TASKS[k] for k in key["tasks"]):
        im = Image.open(ROOT / "images" / "screenshots" / t.image).convert("RGB")
        d = ImageDraw.Draw(im)
        for g in key["tasks"][t.key]["targets"]:
            if "circle" in g:
                cx, cy, r = g["circle"]
                d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(0, 170, 60), width=2)
            else:
                d.rectangle(g["box"], outline=(0, 170, 60), width=3)
        for x, y, hit in dots[t.key]:
            colour = (0, 190, 70) if hit else (230, 30, 30)
            d.ellipse((x - 6, y - 6, x + 6, y + 6), fill=colour, outline="white", width=2)
        n_hit = sum(h for *_, h in dots[t.key])
        d.rectangle((0, 0, 760, 34), fill=(255, 255, 255))
        d.text((8, 6), f"{t.instruction}   {n_hit} of {len(dots[t.key])} clicks inside a target", fill="black", font=font)
        im.save(out / f"{t.key}.png")
    print("Wrote", out)


if __name__ == "__main__":
    main()
