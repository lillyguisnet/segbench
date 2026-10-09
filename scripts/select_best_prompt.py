"""Option B (Maxime, 2026-10-09): each language model keeps the better of two fixed prompts.

python3 scripts/select_best_prompt.py  ->  results/run-points-best.jsonl

Candidates, both 3 tries at medium thinking with the most image detail:
  v2     results/run-points-v2.jsonl     (scored: results/point-benchmark-v2)
  v1-hd  results/run-points-v1hd.jsonl   (scored: results/point-benchmark-v1hd)
Rule, fixed before looking at which model gains: per model, the prompt with
the higher four-task score (mean over tries and tasks), applied to the whole
model, never task by task (per-task picking would flatter scores more).
Ties go to v2. Chosen on the test photos themselves, so scores are slightly
flattering; stated in docs/point-benchmark-v2.md, not on the chart.

Specialists, same principle: Florence-2 keeps the better of its two text
modes (grounding vs open-vocabulary detection); the others had one setup.
Records are copied unchanged plus "derived_from" and "selected_prompt".
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def scores(run):
    return {r["entrant"]: float(r["score"]) for r in csv.DictReader(open(R / run / "summary.csv")) if r["score"]}


v2, v1 = scores("point-benchmark-v2"), scores("point-benchmark-v1hd")
choice = {e: ("v2" if v2[e] >= v1[e] else "v1-hd") for e in v2 if e in v1}
fl = {k: v2[k] for k in ("florence2-large-grounding", "florence2-large-ovd")}
florence = max(fl, key=fl.get)

out, table = [], []
for name, keep in (("run-points-v2.jsonl", "v2"), ("run-points-v1hd.jsonl", "v1-hd")):
    for i, line in enumerate((R / name).read_text().splitlines(), 1):
        r = json.loads(line)
        e = f"{r['model']}@{r['level']}"
        if r["track"] == "find" and choice.get(e) == keep:
            r.update(derived_from=f"results/{name}:{i}", selected_prompt=keep)
            out.append(r)
for name in ("run-specialists-2.jsonl", "run-finders-3.jsonl"):
    for i, line in enumerate((R / name).read_text().splitlines(), 1):
        r = json.loads(line)
        if r["track"] != "find" or (r["model"].startswith("florence2") and r["model"] != florence):
            continue
        r["derived_from"] = f"results/{name}:{i}"
        out.append(r)
(R / "run-points-best.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
for e in sorted(choice, key=lambda e: -max(v2[e], v1[e])):
    print(f"{e:26} v2 {v2[e]*100:5.1f}  v1-hd {v1[e]*100:5.1f}  -> {choice[e]}")
print(f"Florence-2: {florence} ({fl[florence]*100:.1f} vs {min(fl.values())*100:.1f})")
print(len(out), "records")
