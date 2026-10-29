"""Ask the local pointing models (~/Projects/finders) about the second photo batch.

    uv run --with pillow scripts/run_finders_batch2.py --name finders-b2 --gpu 0 \
        --models locateanything-3b molmopoint-8b rexomni-3b florence2-large-grounding moondream-3.1 groundingpi-4b
    # the draft answer key (2026-10-14) was made with:
    uv run --with pillow scripts/run_finders_batch2.py --name consensus-2 --gpu 0 --models moondream-3.1 \
        --tracks find --prompt-version finders-draft-v1

Writes results/run-<name>.jsonl (appended; a model and task already there is
skipped), in the same record format as scripts/run_finders.py, so
scripts/score_points.py scores them like every other entrant.

Wording: segbench/tasks.py PHRASES_BATCH_2 (plurals for plural=True models,
segbench/local_models.py), in each model's own template inside finders, as
for the first photos. Trick questions: the absent thing, likewise.

No timing or cost here: these are the answers only. On the chart, a local
model's seconds and cost come from its timed run on the first photos
(segbench/local_models.py load factors; scripts/time_finders.py), where the
GPU was measured free and under load. Each record says so in timing_source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/home/maxime/Projects/finders")
import finders  # noqa: E402
from PIL import Image  # noqa: E402

from segbench.local_models import BY_KEY as LOCAL  # noqa: E402
from segbench.tasks import PHRASES_BATCH_2, PLURAL_BATCH_2, TASKS_BATCH_2  # noqa: E402


def gpu_uuid(index: int) -> str:
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout
    return {int(i): u.strip() for i, u in (line.split(",") for line in out.splitlines())}[index]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--models", nargs="+", required=True, choices=sorted(k for k, m in LOCAL.items() if m.finders))
    ap.add_argument("--tasks", nargs="+", default=[t.key for t in TASKS_BATCH_2],
                    choices=[t.key for t in TASKS_BATCH_2])
    ap.add_argument("--tracks", nargs="+", default=["find", "trick"], choices=["find", "trick"])
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--prompt-version", default="finders-phrase-b2")
    args = ap.parse_args()
    out = ROOT / "results" / f"run-{args.name}.jsonl"
    done = set()
    if out.exists():
        done = {(r["model"], r["task"], r["track"]) for r in map(json.loads, out.read_text().splitlines())
                if "error" not in r}
    tasks = [t for t in TASKS_BATCH_2 if t.key in args.tasks]
    uuid = gpu_uuid(args.gpu)
    for key in args.models:
        local = LOCAL[key]
        word = (lambda q: PLURAL_BATCH_2[q]) if local.plural else (lambda q: q)
        todo = [(t, track) for t in tasks for track in args.tracks if (key, t.key, track) not in done]
        if not todo:
            print(f"== {key}: nothing to do")
            continue
        phrase = {(t.key, tr): PHRASES_BATCH_2[t.key] if tr == "find" else t.absent.removeprefix("each ")
                  for t, tr in todo}
        jobs = [{"image": str(ROOT / "images" / t.photo), "query": word(phrase[t.key, tr]), "task": local.task,
                 "id": f"{t.key}|{tr}"} for t, tr in todo]
        print(f"== {key}: {len(jobs)} queries", flush=True)
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        recs = finders.find_many(local.finders, jobs, variant=local.variant, masks=False, gpu=uuid)
        with out.open("a") as f:
            for (task, track), job, rec in zip(todo, jobs, recs, strict=True):
                path = ROOT / "images" / task.photo
                W, H = rec["width"], rec["height"]
                with Image.open(path) as im:
                    if im.size != (W, H):
                        raise SystemExit(f"{task.key}: finders saw {W}x{H}, our photo is {im.size}")
                objects = [{"point": [round(o["point"][0] * 1000 / W, 2), round(o["point"][1] * 1000 / H, 2)]}
                           for o in rec["objects"] if o.get("point") is not None]
                f.write(json.dumps({
                    "model": key, "level": None, "track": track, "task": task.key, "task_number": task.number,
                    "repeat": 0, "prompt_version": args.prompt_version, "prompt": rec.get("prompt_sent") or job["query"],
                    "query": job["query"], "image": task.photo,
                    "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "provider": "local-gpu", "route": "local", "started_at": started,
                    "seconds": None, "cost_usd": None,
                    "timing_source": "its timed run on the first photos (segbench/local_models.py)",
                    "text": json.dumps({"objects": objects}), "raw_output": rec.get("raw_output"),
                    "finders": {"model": local.finders, "variant": local.variant, "task": local.task,
                                "weights": rec.get("weights"), "settings": rec.get("settings"),
                                "seconds_this_run": rec.get("seconds")},
                }) + "\n")
                print(f"   {task.key:11} {track:5} {len(objects):4} points", flush=True)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
