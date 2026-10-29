"""Score box answers against the answer-key boxes (box benchmark, 2026-10-15).

uv run scripts/score_boxes.py --reference annotations/boxes/<reviewed export>.json \
    --inputs results/run-boxes-box-a.jsonl results/run-boxes-box-free.jsonl --out results/box-benchmark/<name>

Matching: a model box and an answer-key box match when their overlap
(intersection over union, IoU) is at least the level; each answer-key box
can be matched once and each model box once, as many matches as possible
(maximum matching, not greedy). Score at a level = F1 (missed objects and
extra boxes both count). Levels 50%, 75%, 90%, 95% (Maxime, 2026-10-15).
The box score of a task = the mean of its four F1s (as COCO averages over
overlap levels); a model's score = the mean over the tasks, equal weights.

Dishes: location only, as in the point benchmark since v2.2; the share of
matched dishes with the right dirty/clean label is reported apart
(`label_accuracy`, at 50%).

Reading: box-a with segbench/box_read.read_box_a (lenient); box-free with
read_box_free, except where results/box-free-hand-readings.json holds a
reading made by hand (with its reason); local models: the boxes in their
records (0..1000 grid). Nothing reads the answer key.

Every try is scored. Summaries: the mean over tries (as published so far)
and each model's best complete try (`best_run`).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from segbench.box_read import iou, read_box_a, read_box_free  # noqa: E402
from segbench.point_score import maximum_matching  # noqa: E402
from segbench.tasks import BY_KEY  # noqa: E402

LEVELS = (0.50, 0.75, 0.90, 0.95)
TASKS = ["logs", "cows", "fig", "dishes", "ants", "beetles", "birds", "mushrooms", "jars", "pinkleaves", "whitecow"]
HAND = ROOT / "results/box-free-hand-readings.json"


def load_reference(path: Path) -> dict:
    data = json.loads(path.read_text())
    out = {}
    for key, t in data["tasks"].items():
        draft = data.get("schema") == "segbench-box-draft/v1"
        objs = [o for o in t["objects"] if o.get("box") and (draft or o["status"] == "accepted")]
        out[key] = {"width": t["width"], "height": t["height"],
                    "boxes": [{"id": o["id"], "box": o["box"], "label": o.get("label")} for o in objs]}
    return out


def record_key(r: dict) -> str:
    return f"{r['model']}|{r.get('level')}|{r['prompt_version']}|{r['task']}|{r['repeat']}"


def read(r: dict, ref: dict, hand: dict) -> tuple[list[dict], dict]:
    labelled = bool(BY_KEY[r["task"]].labels)
    if r["prompt_version"] == "box-a":
        try:
            boxes, how = read_box_a(r["text"], labelled=labelled)
        except ValueError as e:  # an empty reply, or one with no boxes in it: scored as no boxes
            return [], {"read_as": "unreadable", "why": str(e)}
        return boxes, {"read_as": how}
    if r["prompt_version"] == "box-free":
        if record_key(r) in hand:
            h = hand[record_key(r)]
            return h["boxes"], {"read_as": "by hand", "why": h["why"]}
        t = ref[r["task"]]
        boxes, how = read_box_free(r["text"], r["model"], t["width"], t["height"], labelled=labelled)
        return boxes, how
    return r["answer"]["items"], {"read_as": "local model record"}


def score_task(boxes: list[dict], ref: dict, labelled: bool) -> dict:
    w, h = ref["width"], ref["height"]
    pred = [[b["box"][0] * w / 1000, b["box"][1] * h / 1000, b["box"][2] * w / 1000, b["box"][3] * h / 1000] for b in boxes]
    refs = [x["box"] for x in ref["boxes"]]
    overlaps = [[iou(p, q) for q in refs] for p in pred]
    out = {"predicted": len(pred), "reference": len(refs), "levels": {}}
    for level in LEVELS:
        edges = [[j for j, o in sorted(enumerate(row), key=lambda x: -x[1]) if o >= level - 1e-9] for row in overlaps]
        pairs = maximum_matching(edges)
        tp = len(pairs)
        f1 = 2 * tp / (len(pred) + len(refs)) if pred or refs else 1.0
        out["levels"][f"{round(level * 100)}"] = {"tp": tp, "f1": f1}
        if level == LEVELS[0] and labelled:
            right = sum(1 for i, j in pairs if boxes[i].get("label") == ref["boxes"][j]["label"])
            out["label_accuracy"] = right / tp if tp else None
    out["score"] = statistics.mean(v["f1"] for v in out["levels"].values())
    best = [max(row) for row in overlaps if row]
    out["median_best_iou"] = statistics.median(best) if best else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--inputs", nargs="+", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    ref = load_reference(args.reference)
    hand = json.loads(HAND.read_text()) if HAND.exists() else {}
    args.out.mkdir(parents=True, exist_ok=True)
    rows, scored = [], []
    for path in args.inputs:
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r.get("track") != "find" or r["task"] not in TASKS or "error" in r:
                continue
            boxes, how = read(r, ref, hand)
            s = score_task(boxes, ref[r["task"]], bool(BY_KEY[r["task"]].labels))
            entrant = f"{r['model']}@{r['level']}" if r.get("level") else r["model"]
            row = {"entrant": entrant, "prompt": r["prompt_version"], "task": r["task"], "repeat": r["repeat"],
                   "score": s["score"], **{f"f1_{k}": v["f1"] for k, v in s["levels"].items()},
                   "predicted": s["predicted"], "reference": s["reference"], "label_accuracy": s.get("label_accuracy"),
                   "seconds": r.get("seconds"), "cost_usd": r.get("cost_usd")}
            rows.append(row)
            scored.append({**{k: r.get(k) for k in ("model", "level", "prompt_version", "task", "repeat")},
                           "source_file": path.name, "how": how, "boxes": boxes, "scores": s})
    with open(args.out / "per-try.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    (args.out / "scored.jsonl").write_text("".join(json.dumps(x) + "\n" for x in scored))
    # summaries per entrant and prompt
    by = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        by[(row["entrant"], row["prompt"])][row["repeat"]][row["task"]] = row
    summary = []
    for (entrant, prompt), runs in sorted(by.items()):
        complete = {k: v for k, v in runs.items() if set(v) >= set(TASKS)}
        if not complete:
            continue
        def mean_of(run, col):
            return statistics.mean(run[t][col] for t in TASKS)
        run_scores = {k: mean_of(v, "score") for k, v in complete.items()}
        best = max(run_scores, key=run_scores.get)
        line = {"entrant": entrant, "prompt": prompt, "tries": len(complete),
                "score": statistics.mean(run_scores.values()), "best_run": best, "best_run_score": run_scores[best]}
        for lv in ("50", "75", "90", "95"):
            line[f"f1_{lv}"] = statistics.mean(mean_of(v, f"f1_{lv}") for v in complete.values())
        cost = [complete[k][t]["cost_usd"] for k in complete for t in ("logs", "cows", "fig", "dishes")
                if complete[k][t]["cost_usd"] is not None]
        secs = [complete[k][t]["seconds"] for k in complete for t in ("logs", "cows", "fig", "dishes")
                if complete[k][t]["seconds"] is not None]
        line["cost_usd_per_question_first4"] = statistics.mean(cost) if cost else None
        line["median_seconds_first4"] = statistics.median(secs) if secs else None
        summary.append(line)
    with open(args.out / "summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(summary[0]))
        wr.writeheader()
        wr.writerows(sorted(summary, key=lambda x: -x["score"]))
    (args.out / "reference.sha256").write_text(hashlib.sha256(args.reference.read_bytes()).hexdigest() + "  "
                                               + str(args.reference) + "\n")
    for s in sorted(summary, key=lambda x: -x["score"]):
        print(f"{s['entrant']:28} {s['prompt']:9} {s['score']*100:5.1f}  "
              + "  ".join(f"{lv}% {s[f'f1_{lv}']*100:5.1f}" for lv in ("50", "75", "90", "95")))


if __name__ == "__main__":
    main()
