"""Summarize a run file: per model and level, what came back and what it cost.

    uv run scripts/summarize_run.py results/run-pilot-1.jsonl

No scoring (that needs the answer key): counts of items found per task,
readable replies, errors, cost and time. Trick questions show how many
objects were invented (the right answer is 0).
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict

from segbench import parse
from segbench.tasks import BY_KEY as TASKS

COUNTED = ("logs", "cows", "fig", "dishes")


def main(path: str) -> None:
    rows = [json.loads(line) for line in open(path)]
    for r in rows:  # read again with today's reader, so a reader fix applies to old runs
        if "error" in r:
            continue
        task = TASKS[r["task"]]
        r.pop("unreadable_because", None)
        try:
            r["answer"], r["readable"] = parse.read(r["track"], task.kind, r["text"], task.labels), True
        except Exception as error:
            r["readable"], r["unreadable_because"] = False, f"{type(error).__name__}: {error}"
    by = defaultdict(list)
    for r in rows:
        by[(r["model"], r["level"])].append(r)

    head = (f"{'model':18} {'level':6} {'calls':>5} {'read':>4} {'err':>3} {'$ total':>8} {'med s':>6} {'max s':>6} | "
            + " ".join(f"{t + ' f/w':>11}" for t in COUNTED) + " | invented")
    print(head)
    print("-" * len(head))
    for (model, level), rs in sorted(by.items(), key=lambda kv: sum(r.get("cost_usd") or 0 for r in kv[1])):
        ok = [r for r in rs if r.get("readable")]
        err = [r for r in rs if "error" in r]
        secs = [r["seconds"] for r in rs if "error" not in r]

        def n(track, task):
            hit = [r for r in rs if r["track"] == track and r["task"] == task]
            if not hit:
                return "-"
            r = hit[0]
            if "error" in r:
                return "E"
            return str(len(r["answer"]["items"])) if r.get("readable") else "?"

        counts = " ".join(f"{n('find', t) + '/' + n('whole', t):>11}" for t in COUNTED)
        invented = sum(len(r["answer"]["items"]) for r in rs if r["track"] == "trick" and r.get("readable"))
        print(f"{model:18} {level:6} {len(rs):5} {len(ok):4} {len(err):3} {sum(r.get('cost_usd') or 0 for r in rs):8.3f} "
              f"{statistics.median(secs) if secs else 0:6.1f} {max(secs, default=0):6.1f} | {counts} | {invented}")

    print("\nf/w = items found in track 'find' / outlined in track 'whole'; ? = unreadable reply, E = error")
    bad = [r for r in rows if "error" in r or not r.get("readable")]
    for r in bad:
        why = r.get("error") or r.get("unreadable_because")
        print(f"  {r['model']:18} {r['level']:6} {r['track']:5} {r['task']:6} {why[:140]}")
    total = sum(r.get("cost_usd") or 0 for r in rows)
    billed = sum(r.get("cost_usd") or 0 for r in rows if r["provider"] not in ("openai-codex", "claude-code"))
    print(f"\ntotal at list prices ${total:.2f}; really billed (not on a plan) ${billed:.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
