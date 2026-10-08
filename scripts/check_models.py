"""Check every remote model: reachable, receives the picture, answers in our format.

Sends the synthetic red-circles picture (segbench/smoke.py) to each model in
segbench/models.py, all at once, and prints one line per model. Every call is
saved, reply included, to results/checks-<time>.jsonl.

Run:   set -a; source .env; set +a
       uv run scripts/check_models.py              # every model
       uv run scripts/check_models.py kimi-k3 qwen-27b
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from segbench import smoke
from segbench.call import call
from segbench.models import BY_KEY, MODELS


def check(model) -> dict:
    record = call(model, smoke.png(), "image/png", smoke.PROMPT)
    record["check"] = "red-circles-v1"
    if "error" not in record:
        try:
            record["passed"], record["detail"] = smoke.grade(smoke.parse(record["text"]))
        except Exception as error:
            record["passed"], record["detail"] = False, f"unreadable reply ({type(error).__name__}: {error})"
    return record


def main(argv: list[str]) -> int:
    unknown = [k for k in argv if k not in BY_KEY]
    if unknown:
        print(f"unknown model keys: {unknown}; known: {list(BY_KEY)}")
        return 2
    models = [BY_KEY[k] for k in argv] if argv else list(MODELS)
    with ThreadPoolExecutor(max_workers=len(models)) as pool:
        records = list(pool.map(check, models))

    out = Path("results") / f"checks-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    out.parent.mkdir(exist_ok=True)
    with out.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    failures = 0
    for r in records:
        u = r.get("usage") or {}
        if "error" in r:
            failures += 1
            line = f"ERROR  {r['error'][:300]}"
        else:
            failures += not r["passed"]
            line = (f"{'OK   ' if r['passed'] else 'BAD  '} {r['detail']} | {r['seconds']:.1f}s | "
                    f"in {u.get('input_tokens')} out {u.get('output_tokens')} thinking {u.get('reasoning_tokens')} | "
                    f"${r['cost_usd'] or 0:.5f}" + (f" (OpenRouter billed ${r['provider_cost_usd']:.5f}, host {r.get('served_by')})"
                                                   if r.get('provider_cost_usd') is not None else ""))
        print(f"{r['model']:18} {line}")
    print(f"\nsaved {out}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
