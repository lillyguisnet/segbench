"""Thinking levels on the eleven photos (2026-10-20): every Claude and GPT model, and Muse Wonder on
our plan, at no (or least) thinking, at medium and at maximum thinking.

    uv run scripts/build_thinking_levels.py   ->  results/thinking-levels-v1/ (scored like point v3)

Asked by Maxime on 2026-10-20: does thinking change pointing scores and the kinds of mistakes?

- min and max: results/run-thinking-levels-v2.jsonl and -v1hd.jsonl (scripts/run.py, find track,
  11 tasks, 3 tries). Levels as segbench/models.py THINKING: off where the model can stop thinking
  (Sonnet between_tools, Haiku disabled, Luna and Terra off), else its lowest (Opus, Astra, Sol low;
  Muse minimal); max = each model's "max".
- medium: the very records of point v3 (results/run-points-best-v3.jsonl), so medium rows equal the
  published ones; Muse Wonder's medium is its plan copy (run-points-v2.jsonl on the first four photos,
  run-thinking-levels-v2-muse-medium.jsonl on the other seven), not the API copy charted in v3.
- prompt: each model's prompt on the v3 chart (Luna and Sol v1-hd, the others v2; Muse Wonder v2, as
  its API copy on v3), at every level. Not re-chosen per level.
- scoring: point v3's (scripts/build_v3.py): both approved keys, location only, lenient reader,
  equal weight per task; cost and seconds from the first four tasks.
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from build_v3 import REFERENCES, TASKS, OLD_TASKS  # noqa: E402

R = ROOT / "results"
NAME = "thinking-levels-v1"
MODELS = ("sonnet", "opus", "haiku", "luna", "terra", "astra", "sol", "muse-wonder")
PROMPT = {"luna": "v1-hd", "sol": "v1-hd"}  # the others v2
NEW = ("run-thinking-levels-v2.jsonl", "run-thinking-levels-v2-gpt.jsonl", "run-thinking-levels-v2-muse.jsonl",
       "run-thinking-levels-v1hd.jsonl")
MUSE_MEDIUM = (("run-points-v2.jsonl", ("logs", "cows", "fig", "dishes")),
               ("run-thinking-levels-v2-muse-medium.jsonl", None))


def rows(name):
    for i, line in enumerate((R / name).read_text().splitlines(), 1):
        yield i, name, json.loads(line)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=NAME, help="results/<name>/ (a provisional look: another name)")
    ap.add_argument("--allow-short", action="store_true", help="score even if some model and level lacks answers")
    a = ap.parse_args()
    out_name = a.name
    if (R / out_name).exists():
        sys.exit(f"results/{out_name} exists: move it away first (scores are never overwritten)")
    out = []

    def keep(r, name, i):
        if r.get("track") != "find" or "error" in r or r["model"] not in MODELS or r["task"] not in TASKS:
            return
        assert r["prompt_version"] == PROMPT.get(r["model"], "v2"), (name, i, r["model"], r["prompt_version"])
        r.setdefault("derived_from", f"results/{name}:{i}")
        out.append(r)

    for i, name, r in rows("run-points-best-v3.jsonl"):  # medium, as published
        if r["model"] in MODELS and r["level"] == "medium":
            keep(r, name, i)
    for name, tasks in MUSE_MEDIUM:
        for i, _, r in rows(name):
            if r["model"] == "muse-wonder" and r["level"] == "medium" and (tasks is None or r["task"] in tasks):
                keep(r, name, i)
    for name in NEW:
        for i, _, r in rows(name):
            if r["level"] in ("min", "max"):
                keep(r, name, i)

    count = {}
    for r in out:
        count[(r["model"], r["level"])] = count.get((r["model"], r["level"]), 0) + 1
    short = {k: v for k, v in count.items() if v != 33}
    if (short or len(count) != 3 * len(MODELS)) and not a.allow_short:
        sys.exit(f"not 33 answers for every model and level: {short or sorted(count)}")

    if short:
        print(f"SHORT (scored anyway): {short}")
    sel = R / f"run-{out_name}.jsonl"
    sel.write_text("".join(json.dumps(r) + "\n" for r in out))
    subprocess.run(["uv", "run", "scripts/score_points.py", "--parser", "lenient", "--dish-labels", "separate",
                    "--method", "docs/thinking-levels.md", "--reference", *REFERENCES, "--tasks", *TASKS,
                    "--timing-tasks", *OLD_TASKS, "--inputs", str(sel), "--out", str(R / out_name)],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    s = {r["entrant"]: r for r in csv.DictReader(open(R / out_name / "summary.csv"))}
    print(f"{len(out)} answers -> {sel.relative_to(ROOT)}; scores in results/{out_name}/summary.csv\n")
    print(f"{'model':14} {'min':>6} {'medium':>7} {'max':>6}")
    for m in MODELS:
        print(f"{m:14} " + " ".join(f"{100 * float(s[f'{m}@{lv}']['score']):6.1f}" for lv in ("min", "medium", "max")))


if __name__ == "__main__":
    main()
