"""Point benchmark v3 (2026-10-14): the four published tasks plus the seven of the second photo batch.

    uv run scripts/build_v3.py            ->  results/point-benchmark-v3/ (+ the two prompt candidates)
    python3 scripts/plot_point_pilot.py point-benchmark-v3 point-benchmark-v3   # the charts

Same treatment as v2.7 (scripts/select_best_prompt.py, segbench/protocol.py),
on 11 tasks instead of 4:
- language models: both fixed prompts (v2, v1-hd), 3 tries at medium; each
  model keeps the prompt with the higher 11-task score, for the whole model
  (ties to v2). Chosen on the test photos, as before (docs/point-benchmark-v2.md);
- local models: their published runs on the first photos (protocol
  LOCAL_PHOTO_RUNS, the same filters) plus their runs on the second batch;
  Florence-2 in its published mode (phrase grounding; the only one run on
  the second batch);
- scores: the two approved reference files, location only, lenient reading,
  equal weight per task;
- cost and seconds: from the first four tasks only, for every entrant (local
  models were timed on those photos only; the second batch's photos are about
  ten times smaller, so mixing would make cost depend on which photos a model
  was timed on).
An entrant without all 11 tasks has no score: Muse Wonder's and Muse Wonder
Full's plan copies (quota used up until 2026-10-19). Muse Wonder is in through
the pay-per-token API instead (muse-wonder-api); Full has no API.
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from segbench.protocol import LOCAL_PHOTO_RUNS  # noqa: E402
from segbench.tasks import TASKS_BATCH_2  # noqa: E402

R = ROOT / "results"
NAME = "point-benchmark-v3"
OLD_TASKS = ["logs", "cows", "fig", "dishes"]
TASKS = OLD_TASKS + [t.key for t in TASKS_BATCH_2]
REFERENCES = ["annotations/reviews/segbench-review-6187f623-approved.json",
              "annotations/reviews/segbench-review-5682a4be-approved.json"]
# Muse Wonder through Meta's pay-per-token API with lm15 (muse-wonder-api,
# segbench/models.py): its first-photo answers are the 2026-10-14 API reruns;
# its plan copy (muse-wonder) is out of quota until 2026-10-19. Muse Wonder
# Full has no API counterpart (HTTP 404, checked again 2026-10-14): not in v3.
PROMPTS = {"v2": ["run-points-v2.jsonl", "run-points-v2-muse-api-20261014.jsonl", "run-points-v2-b2.jsonl"],
           "v1-hd": ["run-points-v1hd.jsonl", "run-points-v1hd-muse-api-20261014.jsonl", "run-points-v1hd-b2.jsonl"]}
LOCAL_B2 = ["run-finders-b2.jsonl", "run-specialists-b2.jsonl", "run-rfdetr-b2.jsonl"]
FLORENCE = "florence2-large-grounding"  # v2.7's choice of mode


def score(inputs, out, timing=None):
    cmd = ["uv", "run", "scripts/score_points.py", "--parser", "lenient", "--dish-labels", "separate",
           "--method", "docs/point-benchmark-v2.md", "--reference", *REFERENCES, "--tasks", *TASKS,
           "--inputs", *[str(R / i) for i in inputs], "--out", str(R / out)]
    if timing:
        cmd += ["--timing-tasks", *timing]
    subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    return {r["entrant"]: float(r["score"]) for r in csv.DictReader(open(R / out / "summary.csv")) if r["score"]}


def main():
    for d in (NAME, f"{NAME}-v2", f"{NAME}-v1hd"):
        if (R / d).exists():
            sys.exit(f"results/{d} exists: move it away first (scores are never overwritten)")
    v2 = score(PROMPTS["v2"], f"{NAME}-v2")
    v1 = score(PROMPTS["v1-hd"], f"{NAME}-v1hd")
    choice = {e: ("v2" if v2[e] >= v1[e] else "v1-hd") for e in v2 if e in v1}
    out = []
    for keep, files in PROMPTS.items():
        for name in files:
            for i, line in enumerate((R / name).read_text().splitlines(), 1):
                r = json.loads(line)
                if r["track"] == "find" and choice.get(f"{r['model']}@{r['level']}") == keep:
                    r.update(derived_from=f"results/{name}:{i}", selected_prompt=keep)
                    out.append(r)
    for name in [*LOCAL_PHOTO_RUNS, *LOCAL_B2]:
        for i, line in enumerate((R / name).read_text().splitlines(), 1):
            r = json.loads(line)
            if r["track"] != "find" or (r["model"].startswith("florence2") and r["model"] != FLORENCE):
                continue
            if r["model"] == "rfdetr-seg-2xl" and name not in ("run-rfdetr-3.jsonl", "run-rfdetr-b2.jsonl"):
                continue
            r["derived_from"] = f"results/{name}:{i}"
            out.append(r)
    best = R / "run-points-best-v3.jsonl"
    best.write_text("".join(json.dumps(r) + "\n" for r in out))
    final = score([best.name], NAME, timing=OLD_TASKS)
    for e in sorted(choice, key=lambda e: -max(v2[e], v1[e])):
        print(f"{e:26} v2 {v2[e]*100:5.1f}  v1-hd {v1[e]*100:5.1f}  -> {choice[e]}")
    print(f"\n{len(out)} records -> {best.relative_to(ROOT)}; {len(final)} entrants scored on all {len(TASKS)} tasks")
    for e, s in sorted(final.items(), key=lambda x: -x[1]):
        print(f"  {e:28} {s*100:5.1f}")


if __name__ == "__main__":
    main()
