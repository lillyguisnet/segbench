"""Box benchmark v1 (2026-10-16): scores, each model's prompt, report and chart records.

    uv run --with pillow scripts/build_boxes.py   ->  results/box-benchmark/v1/
    then: uv run --with pillow scripts/plot_boxes.py   (the charts)

- scores: scripts/score_boxes.py against the reviewed answer key
  (annotations/boxes/answer-key-boxes-v1.json), every reply of every try;
- language models: both prompts (box-a, box-free), 3 tries at medium; each
  model keeps the prompt with the higher eleven-task score, for the whole
  model (ties to box-a, written in advance). Chosen on the test photos, as
  for points;
- local models: one setup each (their box mode);
- the score of a task: mean F1 over the four overlap levels (50, 75, 90,
  95%); a model's: mean over the eleven tasks, equal weights, mean over tries;
- cost and seconds from the first four tasks only (as point v3).
"""
import csv
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from segbench.chart_data import describe  # noqa: E402

R = ROOT / "results"
OUT = R / "box-benchmark" / "v1"
KEY = ROOT / "annotations/boxes/answer-key-boxes-v1.json"
INPUTS = ["run-boxes-box-a.jsonl", "run-boxes-box-a-resent.jsonl", "run-boxes-box-free.jsonl",
          "run-boxes-box-free-subs.jsonl", "run-boxes-box-free-resent.jsonl",
          "run-finders-boxes-1-without-rexomni.jsonl", "run-finders-boxes-2-rexomni.jsonl",
          "run-specialists-boxes-1.jsonl"]
TASKS = ["logs", "cows", "fig", "dishes", "ants", "beetles", "birds", "mushrooms", "jars", "pinkleaves", "whitecow"]
FIRST4 = ("logs", "cows", "fig", "dishes")
LEVELS = ("50", "75", "90", "95")


def main():
    if OUT.exists():
        sys.exit(f"{OUT} exists: move it away first (scores are never overwritten)")
    cand = R / "box-benchmark" / "v1-candidates"
    subprocess.run(["uv", "run", "--with", "pillow", "scripts/score_boxes.py", "--reference", str(KEY),
                    "--inputs", *[str(R / i) for i in INPUTS], "--out", str(cand)],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    summary = list(csv.DictReader(open(cand / "summary.csv")))
    by = defaultdict(dict)
    for s in summary:
        by[s["entrant"]][s["prompt"]] = s
    choice = {}
    for e, ps in by.items():
        if set(ps) == {"box-a", "box-free"}:
            choice[e] = "box-a" if float(ps["box-a"]["score"]) >= float(ps["box-free"]["score"]) else "box-free"
        else:
            (choice[e],) = ps
    scored = [json.loads(l) for l in (cand / "scored.jsonl").read_text().splitlines()]
    raw = {}
    for name in INPUTS:
        for line in (R / name).read_text().splitlines():
            r = json.loads(line)
            if r.get("track") == "find" and "error" not in r:
                raw[(r["model"], r.get("level"), r["prompt_version"], r["task"], r["repeat"])] = r
    keep, calls = [], []
    for s in scored:
        e = f"{s['model']}@{s['level']}" if s.get("level") else s["model"]
        if choice[e] != s["prompt_version"]:
            continue
        keep.append(s)
        r = raw[(s["model"], s.get("level"), s["prompt_version"], s["task"], s["repeat"])]
        timed = s["task"] in FIRST4
        calls.append({"track": "find", "task": s["task"], "model": s["model"], "level": s.get("level"),
                      "repeat": s["repeat"], "score": s["scores"]["score"], "readable": True,
                      **{f"f1_{lv}": s["scores"]["levels"][lv]["f1"] for lv in LEVELS},
                      "cost_usd": r.get("cost_usd") if timed else None, "seconds": r.get("seconds") if timed else None,
                      "cost_usd_measured": r.get("cost_usd"), "seconds_measured": r.get("seconds"),
                      "prompt_version": s["prompt_version"], "source_file": s["source_file"]})
    OUT.mkdir(parents=True)
    (OUT / "scored.jsonl").write_text("".join(json.dumps(x) + "\n" for x in keep))
    (OUT / "calls.jsonl").write_text("".join(json.dumps(x) + "\n" for x in calls))
    # report
    grp = defaultdict(lambda: defaultdict(list))
    for c in calls:
        e = f"{c['model']}@{c['level']}" if c["level"] else c["model"]
        grp[e][c["task"]].append(c)
    rows = []
    for e, ts in grp.items():
        assert set(ts) == set(TASKS), (e, set(TASKS) - set(ts))
        any_c = next(iter(ts.values()))[0]
        label, maker, kind = describe(any_c["model"])
        row = {"entrant": e, "label": label, "maker": maker, "kind": kind, "prompt": choice[e],
               "score": statistics.mean(statistics.mean(c["score"] for c in ts[t]) for t in TASKS)}
        for lv in LEVELS:
            row[f"f1_{lv}"] = statistics.mean(statistics.mean(c[f"f1_{lv}"] for c in ts[t]) for t in TASKS)
        for t in TASKS:
            row[t] = statistics.mean(c["score"] for c in ts[t])
        timed = [c for t in FIRST4 for c in ts[t]]
        row["cost_usd"] = statistics.mean(c["cost_usd"] for c in timed if c["cost_usd"] is not None)
        row["seconds"] = statistics.median(c["seconds"] for c in timed if c["seconds"] is not None)
        other = by[e].get("box-free" if choice[e] == "box-a" else "box-a")
        row["other_prompt_score"] = float(other["score"]) if other else None
        lab = [x["scores"].get("label_accuracy") for x in keep
               if x["task"] == "dishes" and (f"{x['model']}@{x['level']}" if x.get("level") else x["model"]) == e]
        lab = [v for v in lab if v is not None]
        row["dish_label_accuracy_50"] = statistics.mean(lab) if lab else None
        rows.append(row)
    rows.sort(key=lambda r: -r["score"])
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (OUT / "manifest.json").write_text(json.dumps({
        "answer_key": str(KEY.relative_to(ROOT)), "inputs": INPUTS, "tasks": TASKS, "timing_tasks": list(FIRST4),
        "overlap_levels": [int(x) / 100 for x in LEVELS], "prompt_choice": choice, "tie_rule": "box-a",
        "method": "docs/box-benchmark.md"}, indent=1))
    for r in rows:
        print(f"{r['label']:24} {r['prompt']:9} {r['score']*100:5.1f}  "
              + "  ".join(f"{lv}% {r[f'f1_{lv}']*100:5.1f}" for lv in LEVELS)
              + f"   ${r['cost_usd']*1000:7.2f}/1k  {r['seconds']:5.1f}s")


if __name__ == "__main__":
    main()
