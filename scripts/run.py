"""Send the benchmark photos to the remote models and save every reply.

Runs the tracks that need no answer key (find, whole, trick; see
segbench/tasks.py). Replies are saved raw with tokens, seconds and cost, one
JSON line per call, and scored later once the answer key exists.

    set -a; source .env; set +a
    uv run scripts/run.py --name pilot-1 --levels medium --dry-run   # the plan and its size
    uv run scripts/run.py --name pilot-1 --levels medium              # run it
    uv run scripts/run.py --name pilot-1 --levels medium              # again: only what is missing

The output file results/run-<name>.jsonl is appended to, never rewritten.
Running the same command again skips every call that already has an answer
(or a final, non-retryable error), so an interrupted run just continues.

Retries: a call the provider never answered (rate limit, dropped connection,
overloaded) is sent again up to 3 times, and each failed attempt is kept in
the record. A reply the model actually gave is never sent again, even if it
cannot be read: it scores 0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from lm15 import LMRouter

from segbench import parse
from segbench.point_score import read_points
from segbench.call import call
from segbench.models import BENCHMARKED, BY_KEY, levels_for
from segbench.tasks import BY_KEY as TASKS_BY_KEY
from segbench.tasks import PROMPT_VERSION, PROMPT_VERSION_V2, TASKS, TRACKS, applies, prompt, prompt_v2

ROOT = Path(__file__).resolve().parents[1]
ATTEMPTS = 3
BACKOFF_SECONDS = (15, 60)
# Calls in flight at once per provider. Subscriptions are shared with the
# family's own use, so they get fewer.
PER_PROVIDER = defaultdict(lambda: 6, {"openai-codex": 4, "claude-code": 3})

# Prompt v2 also asks each provider for the most image detail it offers
# (scripts/probe_detail.py, 2026-10-09, input tokens of a 4000x2252 picture):
# Gemini ULTRA_HIGH on the image part doubles it (1,108 -> 2,213); OpenAI's
# default already takes the most (10,674; "high" would lower it to ~2,950);
# GLM, DeepSeek, Kimi and Qwen ignore "high"; Anthropic has no setting.
IMAGE_SETTINGS_V2 = {"gemini": {"media_resolution": "MEDIA_RESOLUTION_ULTRA_HIGH"}}


def image_settings(model, version: str) -> dict:
    if version != PROMPT_VERSION_V2:
        return {}
    return IMAGE_SETTINGS_V2.get(model.route.split(":")[0], {})


def prompt_for(version: str, track: str, task) -> str:
    return prompt_v2(track, task) if version == PROMPT_VERSION_V2 else prompt(track, task)


def key_of(r: dict) -> tuple:
    return (r["track"], r["task"], r["model"], r["level"], r["repeat"], r["prompt_version"])


def plan(models, levels, tasks, tracks, repeats):
    jobs = []
    for model in models:
        for level in levels_for(model.key):
            if levels and level not in levels:
                continue
            for task in tasks:
                for track in tracks:
                    if applies(track, task):
                        for repeat in range(repeats):
                            jobs.append((model, level, task, track, repeat))
    return jobs


def done_keys(path: Path) -> set:
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if "error" not in r or not r.get("retryable"):
                done.add(key_of(r))
    return done


def run_one(model, level, task, track, repeat, images, semaphores, version=PROMPT_VERSION) -> dict:
    image = images[task.photo]
    text = prompt_for(version, track, task)
    extra = image_settings(model, version)
    failed = []
    with semaphores[LMRouter().resolve(model.route).provider]:
        for attempt in range(ATTEMPTS):
            record = call(model, image["bytes"], "image/jpeg", text, level=level, **extra)
            if "error" not in record or not record.get("retryable") or attempt == ATTEMPTS - 1:
                break
            failed.append({k: record.get(k) for k in ("started_at", "seconds", "error")})
            time.sleep(BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)])
    record.update({
        "track": track, "task": task.key, "task_number": task.number, "repeat": repeat,
        "prompt_version": version, "prompt": text,
        "image": task.photo, "image_sha256": image["sha256"], "image_bytes": len(image["bytes"]),
        "failed_attempts": failed,
    })
    if "error" not in record:
        try:
            if version == PROMPT_VERSION_V2:  # named x/y; same reader as the scorer
                points, how = read_points(record["text"], labelled=track == "find" and bool(task.labels),
                                          parser="lenient")
                answer, record["read_as"] = {"items": points}, how
            else:
                answer = parse.read(track, task.kind, record["text"], task.labels)
            record["readable"], record["answer"] = True, answer
        except Exception as error:
            record["readable"], record["unreadable_because"] = False, f"{type(error).__name__}: {error}"[:300]
    return record


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="run name: results/run-<name>.jsonl")
    ap.add_argument("--models", nargs="*", default=[], help="model keys (default: every benchmarked model)")
    ap.add_argument("--levels", nargs="*", default=[], help="min medium max (default: all of each model's levels)")
    ap.add_argument("--tasks", nargs="*", default=[], help=f"task keys {list(TASKS_BY_KEY)} (default: all)")
    ap.add_argument("--tracks", nargs="*", default=list(TRACKS), help=f"{list(TRACKS)} (default: all)")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true", help="print the plan and one prompt per track, call nothing")
    ap.add_argument("--prompt-version", choices=(PROMPT_VERSION, PROMPT_VERSION_V2), default=PROMPT_VERSION,
                    help="v2: named x/y, 'mark every one', most image detail; find and trick tracks only")
    args = ap.parse_args(argv)

    models = [BY_KEY[k] for k in args.models] if args.models else list(BENCHMARKED)
    if bad := [m.key for m in models if not m.vision]:
        ap.error(f"these models do not see images: {bad}")
    tasks = [TASKS_BY_KEY[k] for k in args.tasks] if args.tasks else list(TASKS)
    jobs = plan(models, args.levels, tasks, args.tracks, args.repeats)
    out = ROOT / "results" / f"run-{args.name}.jsonl"
    done = done_keys(out)
    version = args.prompt_version
    if version == PROMPT_VERSION_V2 and set(args.tracks) - {"find", "trick"}:
        ap.error("prompt v2 covers --tracks find trick only")
    todo = [j for j in jobs if (j[3], j[2].key, j[0].key, j[1], j[4], version) not in done]
    print(f"{len(jobs)} calls planned, {len(jobs) - len(todo)} already in {out.name}, {len(todo)} to send")

    if args.dry_run:
        shown = set()
        for _, _, task, track, _ in todo:
            if (track, task.kind, bool(task.labels)) not in shown:
                shown.add((track, task.kind, bool(task.labels)))
                print(f"\n--- {track} / {task.key} ({version}) ---\n{prompt_for(version, track, task)}")
        return 0

    images = {}
    for task in tasks:
        data = (ROOT / "images" / task.photo).read_bytes()
        images[task.photo] = {"bytes": data, "sha256": hashlib.sha256(data).hexdigest()}
    semaphores = {p: threading.Semaphore(PER_PROVIDER[p])
                  for p in {LMRouter().resolve(m.route).provider for m in models}}

    lock = threading.Lock()
    start = time.time()
    n_ok = n_err = n_unreadable = 0
    with ThreadPoolExecutor(max_workers=64) as pool, out.open("a") as f:
        futures = [pool.submit(run_one, *job, images, semaphores, version) for job in todo]
        for i, future in enumerate(as_completed(futures), 1):
            r = future.result()
            with lock:
                f.write(json.dumps(r) + "\n")
                f.flush()
            if "error" in r:
                n_err += 1
                status = f"ERROR {r['error'][:120]}"
            elif not r["readable"]:
                n_unreadable += 1
                status = f"unreadable: {r['unreadable_because'][:100]}"
            else:
                n_ok += 1
                status = f"{len(r['answer']['items'])} items"
            print(f"[{i}/{len(todo)} {time.time() - start:6.0f}s] {r['model']:18} {r['level']:6} {r['track']:5} "
                  f"{r['task']:6} {r.get('seconds', 0):6.1f}s ${r.get('cost_usd') or 0:.4f}  {status}", flush=True)
    print(f"\n{n_ok} read, {n_unreadable} unreadable, {n_err} errors; saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
