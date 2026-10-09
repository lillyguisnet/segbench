"""Which thinking levels does each model accept, and does the level change anything?

Sends a short text-only arithmetic question (cheap, no picture) to every
benchmarked model at each of lm15's seven effort words, plus the default,
and prints the thinking tokens each one used. A level that is accepted but
gives the same thinking as every other level is probably ignored.

Run:   set -a; source .env; set +a
       uv run scripts/probe_thinking.py [model keys...]
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from segbench.call import call
from segbench.models import BENCHMARKED, BY_KEY

EFFORTS = [None, "off", "minimal", "low", "medium", "high", "xhigh", "max"]
QUESTION = ("A shop sells pens in packs of 7 and pencils in packs of 11. Ana buys 60 items in total, "
            "using whole packs only, more packs of pens than of pencils. How many packs of each? "
            "Reply with two numbers only.")


def main(argv: list[str]) -> int:
    models = [BY_KEY[k] for k in argv] if argv else list(BENCHMARKED)
    jobs = [(m, e) for m in models for e in EFFORTS]
    # One thread per model: a model's levels run one after another, so they do not compete.
    def run_model(m):
        return [call(m, None, "", QUESTION, effort=e) for e in EFFORTS]
    with ThreadPoolExecutor(max_workers=len(models)) as pool:
        records = [r for rs in pool.map(run_model, models) for r in rs]

    out = Path("results") / f"thinking-probe-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    with out.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"{'model':18}" + "".join(f"{str(e):>10}" for e in EFFORTS))
    for m in models:
        cells = []
        for r in (r for r in records if r["model"] == m.key):
            if "error" in r:
                cells.append("refused")
            else:
                u = r["usage"]
                think = u.get("reasoning_tokens")
                cells.append("?" if think is None else str(think))
        print(f"{m.key:18}" + "".join(f"{c:>10}" for c in cells))
    print("\nnumbers = thinking tokens reported; ? = provider reports none; refused = error (see file)")
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
