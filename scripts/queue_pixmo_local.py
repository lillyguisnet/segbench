"""Run a small PixMo model queue when a GPU is free; never stop other jobs.

python3 scripts/queue_pixmo_local.py --out results/pixmo/local-DATE
Status and launcher output live in that directory. Preparation must already
be complete. The dataset manifest is frozen when the queue is created.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("moondream-3.1", "molmopoint-8b")


def timestamp():
    return datetime.now(UTC).isoformat()


def available_gpu(requested=None):
    """Match the evaluation runner's no-shared-GPU policy, with memory headroom."""
    cards = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True, timeout=20)
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
        text=True, timeout=20)
    busy = {line.split(",")[0].strip() for line in apps.splitlines() if line.strip()}
    for line in cards.splitlines():
        index, uuid, free, utilization = [s.strip() for s in line.split(",")]
        if requested is not None and requested not in (index, uuid):
            continue
        if uuid not in busy and int(free) >= 20480 and int(utilization) <= 5:
            return {"index": index, "uuid": uuid, "free_mib": int(free)}
    return None


def write_status(dest, status):
    status["updated_at"] = timestamp()
    tmp = dest / "status.json.tmp"
    tmp.write_text(json.dumps(status, indent=2) + "\n")
    tmp.replace(dest / "status.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--gpu", help="restrict the queue to this GPU index or UUID")
    parser.add_argument("--wait-hours", type=float, default=24)
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    if args.wait_hours <= 0 or args.poll_seconds <= 0:
        parser.error("wait duration and polling interval must be positive")
    dest = args.out.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    manifest = ROOT / "cache/pixmo-points-eval/examples.jsonl"
    original_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    questions = sum(json.loads(s)["status"] == "verified" for s in manifest.read_text().splitlines() if s.strip())
    status = {"state": "waiting_for_free_gpu", "pid": os.getpid(), "started_at": timestamp(),
              "models": list(MODELS), "questions_per_model": questions, "completed": [], "requested_gpu": args.gpu,
              "manifest_sha256": original_hash, "wait_limit_hours": args.wait_hours,
              "policy": "no compute processes, at least 20 GiB free, utilization <=5%; never stop other jobs"}
    write_status(dest, status)
    started = time.monotonic()
    try:
        for model in MODELS:
            while True:
                if hashlib.sha256(manifest.read_bytes()).hexdigest() != original_hash:
                    raise RuntimeError("dataset manifest changed while queued; refusing a different evaluation set")
                if time.monotonic() - started > args.wait_hours * 3600:
                    raise TimeoutError("queue wait limit reached; no other jobs were interrupted")
                gpu = available_gpu(args.gpu)
                if gpu is not None:
                    break
                status.update(state="waiting_for_free_gpu", next_model=model)
                write_status(dest, status)
                print(f"{timestamp()} waiting for a free GPU: next {model}", flush=True)
                time.sleep(args.poll_seconds)
            status.update(state="running", next_model=model, gpu=gpu)
            write_status(dest, status)
            command = ["uv", "run", "scripts/pixmo_eval.py", "run", "--model", model,
                       "--gpu", gpu["uuid"], "--allow-partial", "--out", str(dest / model)]
            print(f"{timestamp()} starting {model} on GPU {gpu['index']}", flush=True)
            with (dest / f"{model}.log").open("w") as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                status["child_pid"] = child.pid
                write_status(dest, status)
                try:
                    code = child.wait()
                finally:
                    if child.poll() is None:
                        child.terminate()
                        child.wait()
            if code:
                raise RuntimeError(f"{model} exited {code}; see {dest / (model + '.log')}; no automatic retry")
            summary = json.loads((dest / model / "summary.json").read_text())
            if not summary["complete_selected_run"]:
                raise RuntimeError(f"{model} has missing predictions")
            status["completed"].append({"model": model, "f1": summary["f1"],
                                        "questions": summary["scored_questions"]})
            status.pop("child_pid", None)
            write_status(dest, status)
        status.update(state="completed", finished_at=timestamp())
        write_status(dest, status)
        print(json.dumps(status, indent=2), flush=True)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        status.update(state="failed", error=str(exc))
        write_status(dest, status)
        print(str(exc), file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
