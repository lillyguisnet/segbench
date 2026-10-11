"""Ask the language models where they would click on the five screenshots.

    set -a; source .env; set +a
    uv run scripts/run_screens.py --name screens-1 --dry-run
    uv run scripts/run_screens.py --name screens-1          # again: only what is missing

Same settings as the photo benchmark's prompt v2 run (scripts/run.py):
every model that sees pictures, medium thinking, 3 tries, the most image
detail each provider offers, retries only when the provider never answered.
Writes results/run-<name>.jsonl, appended to, never rewritten. Tasks and
prompt: segbench/screens.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from lm15 import LMRouter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from run import ATTEMPTS, BACKOFF_SECONDS, IMAGE_SETTINGS_V2, PER_PROVIDER  # noqa: E402

from segbench.call import call  # noqa: E402
from segbench.models import BENCHMARKED, BY_KEY  # noqa: E402
from segbench.screens import BY_KEY as TASKS, PROMPT_VERSION, ROUNDS, prompt, read_clicks  # noqa: E402

LEVEL = "medium"


def key_of(r):
    return (r["task"], r["model"], r["level"], r["repeat"], r["prompt_version"])


def run_one(model, task, repeat, image, semaphores):
    text = prompt(task)
    extra = IMAGE_SETTINGS_V2.get(model.route.split(":")[0], {})
    failed = []
    with semaphores[LMRouter().resolve(model.route).provider]:
        for attempt in range(ATTEMPTS):
            record = call(model, image["bytes"], "image/png", text, level=LEVEL, **extra)
            if "error" not in record or not record.get("retryable") or attempt == ATTEMPTS - 1:
                break
            failed.append({k: record.get(k) for k in ("started_at", "seconds", "error")})
            time.sleep(BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)])
    record.update({"track": "screen", "task": task.key, "repeat": repeat, "prompt_version": PROMPT_VERSION,
                   "prompt": text, "image": "screenshots/" + task.image, "image_sha256": image["sha256"],
                   "image_size": image["size"], "failed_attempts": failed})
    if "error" not in record:
        try:
            clicks, how = read_clicks(record["text"])
            record.update(readable=True, read_as=how, clicks=clicks)
        except ValueError as error:
            record.update(readable=False, unreadable_because=str(error))
    return record


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--tasks", nargs="*", default=[])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--round", type=int, default=1, choices=sorted(ROUNDS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    models = [BY_KEY[k] for k in args.models] if args.models else list(BENCHMARKED)
    tasks = [TASKS[k] for k in args.tasks] if args.tasks else list(ROUNDS[args.round])
    out = ROOT / "results" / f"run-{args.name}.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            if "error" not in r or not r.get("retryable"):
                done.add(key_of(r))
    jobs = [(m, t, i) for m in models for t in tasks for i in range(args.repeats)]
    todo = [j for j in jobs if (j[1].key, j[0].key, LEVEL, j[2], PROMPT_VERSION) not in done]
    print(f"{len(jobs)} calls planned, {len(jobs) - len(todo)} already in {out.name}, {len(todo)} to send")
    if args.dry_run:
        print("\n" + prompt(tasks[0]))
        return 0

    images = {}
    for t in tasks:
        path = ROOT / "images" / "screenshots" / t.image
        data = path.read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n", path  # size from the PNG header: width, height
        size = [int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")]
        images[t.key] = {"bytes": data, "sha256": hashlib.sha256(data).hexdigest(), "size": size}
    semaphores = {p: threading.Semaphore(PER_PROVIDER[p])
                  for p in {LMRouter().resolve(m.route).provider for m in models}}
    start, lock = time.time(), threading.Lock()
    with ThreadPoolExecutor(max_workers=64) as pool, out.open("a") as f:
        futures = [pool.submit(run_one, m, t, i, images[t.key], semaphores) for m, t, i in todo]
        for n, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            with lock:
                f.write(json.dumps(r) + "\n")
                f.flush()
            status = (f"ERROR {r['error'][:100]}" if "error" in r else
                      f"unreadable: {r['text'][:80]!r}" if not r["readable"] else
                      f"{len(r['clicks'])} clicks ({r['read_as']}) " +
                      " ".join(f"({c['x']:.0f},{c['y']:.0f})" for c in r["clicks"][:4]))
            print(f"[{n}/{len(todo)} {time.time() - start:5.0f}s] {r['model']:18} {r['task']:10} "
                  f"{r.get('seconds', 0):6.1f}s ${r.get('cost_usd') or 0:.4f}  {status}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
