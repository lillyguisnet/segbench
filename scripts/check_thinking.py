"""Does each thinking level change how much a model thinks on a picture?

Runs the red-circles check (segbench/smoke.py) at our three levels (min,
medium, max; segbench/models.py THINKING) for every benchmarked model,
REPEATS times each, and prints thinking tokens, seconds, cost and pass/fail.

Run:   set -a; source .env; set +a
       uv run scripts/check_thinking.py [model keys...]
"""

from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from segbench import smoke
from segbench.call import call
from segbench.cost import billed_output_tokens
from segbench.models import BENCHMARKED, BY_KEY, levels_for

REPEATS = 2


def one(model, level: str) -> dict:
    r = call(model, smoke.png(), "image/png", smoke.PROMPT, level=level)
    r["check"] = "red-circles-v1"
    if "error" not in r:
        try:
            r["passed"], r["detail"] = smoke.grade(smoke.parse(r["text"]))
        except Exception as error:
            r["passed"], r["detail"] = False, f"unreadable ({type(error).__name__})"
    return r


def main(argv: list[str]) -> int:
    models = [BY_KEY[k] for k in argv] if argv else list(BENCHMARKED)
    jobs = [(m, level) for m in models for level in levels_for(m.key) for _ in range(REPEATS)]
    out = Path("results") / f"thinking-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    print(f"{len(jobs)} calls, all at once; each saved to {out} as it finishes", flush=True)
    records, lock = [], threading.Lock()
    # Every call at once: they wait on remote servers, not on this machine.
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool, out.open("w") as f:
        for done in as_completed([pool.submit(one, m, level) for m, level in jobs]):
            r = done.result()
            with lock:
                records.append(r)
                f.write(json.dumps(r) + "\n")
                f.flush()
            print(f"  {len(records):3}/{len(jobs)} {r['model']:18} {r['level']:6} {r.get('seconds', 0):6.1f}s "
                  f"{'error' if 'error' in r else ('pass' if r['passed'] else 'FAIL')}", flush=True)

    print(f"{'model':18} {'level':7} {'thinking tokens':>18} {'answer tokens':>14} {'seconds':>14} {'cost $':>16}  passed")
    for m in models:
        for level in levels_for(m.key):
            rs = [r for r in records if r["model"] == m.key and r["level"] == level]
            ok = [r for r in rs if "error" not in r]
            if not ok:
                print(f"{m.key:18} {level:7} ERROR {rs[0]['error'][:150]}")
                continue
            think = [r["usage"].get("reasoning_tokens") or 0 for r in ok]
            billed = [billed_output_tokens(r["provider"], r["usage"]) or 0 for r in ok]
            answer = [b - t if r["provider"] not in ("gemini", "xai") else b - t for b, t, r in zip(billed, think, ok)]
            secs = [r["seconds"] for r in ok]
            costs = [r["cost_usd"] or 0 for r in ok]
            fmt = lambda xs, f: "/".join(f.format(x) for x in xs)
            print(f"{m.key:18} {level:7} {fmt(think, '{}'):>18} {fmt(answer, '{}'):>14} {fmt(secs, '{:.0f}'):>14} "
                  f"{fmt(costs, '{:.4f}'):>16}  {sum(r['passed'] for r in ok)}/{len(rs)}"
                  + ("" if len(ok) == len(rs) else f"  ({len(rs) - len(ok)} errors)"))
    print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
