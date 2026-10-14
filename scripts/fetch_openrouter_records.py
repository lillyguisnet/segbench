"""Save OpenRouter's own record of every call we made through it (2026-10-12).

OpenRouter keeps, per call, the attempts it made at the host
(`provider_responses`: status and seconds of each), the host's time to first
token (`latency`) and its total (`generation_time`). Our clock cannot see
these. They showed that 61 of 65 Qwen 3.8 Flash Next calls were first refused
by Alibaba (HTTP 429, "too many requests"), costing a median 5.3 s, which
OpenRouter retried silently and our clock counted; calls we retry ourselves
are timed from the attempt that answered (scripts/run.py), so the two were not
timed alike. Reading the records costs nothing; they are saved here in case
OpenRouter stops serving them.

Run: uv run python scripts/fetch_openrouter_records.py results/run-points-best.jsonl ...
Writes results/openrouter-records.jsonl (one line per call, keyed by generation id).
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "openrouter-records.jsonl"


def key() -> str:
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    text = Path("~/Projects/lm15-dev/.env").expanduser().read_text()
    return re.findall(r"OPENROUTER_API_KEY=['\"]?([^'\"\n]+)", text)[-1]  # a shell keeps the last


def fetch(gen_id: str) -> dict | None:
    req = urllib.request.Request(f"https://openrouter.ai/api/v1/generation?id={gen_id}",
                                 headers={"Authorization": f"Bearer {key()}", "User-Agent": "segbench"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=30))["data"]
    except Exception:
        return None


def main(paths: list[str]) -> None:
    have = {json.loads(l)["id"] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    wanted = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            r = json.loads(line)
            gid = ((r.get("response") or {}).get("provider_data") or {}).get("id")
            if r.get("provider") == "openrouter" and gid and gid not in have:
                wanted.append((Path(p).name, r["model"], r["seconds"], gid))
                have.add(gid)
    with ThreadPoolExecutor(8) as ex:
        records = list(ex.map(lambda w: fetch(w[3]), wanted))
    saved = 0
    with OUT.open("a") as f:
        for (run, model, seconds, gid), d in zip(wanted, records, strict=True):
            if d is None:
                continue
            refused = [p for p in d.get("provider_responses") or [] if p.get("status") != 200]
            f.write(json.dumps({"id": gid, "run": run, "model": model, "our_seconds": seconds,
                                "refused_attempts": len(refused),
                                "refused_seconds": sum(p.get("latency") or 0 for p in refused) / 1000,
                                "record": d}) + "\n")
            saved += 1
    print(f"saved {saved} of {len(wanted)} new records to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1:])
