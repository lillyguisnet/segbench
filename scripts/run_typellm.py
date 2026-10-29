"""TypeLLM on the eleven photo tasks of point benchmark v3 and box benchmark v1 (Maxime, 2026-10-17).

    set -a; source .env; set +a
    uv run --with pillow scripts/run_typellm.py --check            # synthetic ovals only, no benchmark photo
    uv run --with pillow scripts/run_typellm.py --name typellm-1   # the photos

-> results/run-<name>-<prompt>.jsonl, one file per prompt (scripts/score_points.py
   and scripts/score_boxes.py score one prompt per file), both thinking levels in each.

The questions are segbench/typellm.py (the benchmark's own words; the typed
schema replaces the reply-format paragraph). As every language model: 3 tries
of each question, the same photo files sent whole (TypeLLM shrinks a photo to
about 2,048 image tokens and a 2,048 px long side itself: the most detail its
API takes). No trick questions (as the box benchmark; point v3's score has
none either). A record's seconds = the waiting time of its call; cost = its
billed tokens at TypeLLM's list price. Appending: a (prompt, level, task, try)
already in its file is skipped, so an interrupted run continues. Up to 4 calls
at once (the API allows 200 a minute).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from segbench import typellm as L  # noqa: E402
from segbench.box_read import iou  # noqa: E402
from segbench.tasks import BY_KEY, Task  # noqa: E402

V3_TASKS = ["logs", "cows", "fig", "dishes", "ants", "beetles", "birds", "mushrooms", "jars", "pinkleaves", "whitecow"]
REPEATS = 3
VERSIONS = L.POINT_VERSIONS + L.BOX_VERSIONS
LOCK = threading.Lock()


def commit() -> str:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "segbench/typellm.py", "scripts/run_typellm.py",
                            "segbench/tasks.py"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return head + ("-dirty" if dirty else "")


def check() -> None:
    """Seven red ovals and four blue squares (scripts/check_boxes.py) at the four photo sizes."""
    from check_boxes import SIZES, grade, picture
    oval = Task(0, "ovals", "", "each red oval (not the blue squares)", "objects", "each cow")
    out = ROOT / "results" / f"check-typellm-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    jobs = [(n, wh, v, lv) for n, wh in SIZES.items() for v in VERSIONS for lv in L.LEVELS]

    def one(job):
        name, (w, h), v, lv = job
        jpg, truth = picture(w, h)
        try:
            rec = L.ask(jpg, "image/jpeg", v, oval, lv)
        except Exception as e:  # noqa: BLE001
            return job, {"error": str(e)}, None
        if v in L.POINT_VERSIONS:
            pts = [(o["x"], o["y"]) for o in json.loads(rec["text"])["objects"]]
            inside = set()
            for x, y in pts:  # a point inside an oval's ellipse, each oval counted once
                for k, t in enumerate(truth):
                    cx, cy, rx, ry = (t[0] + t[2]) / 2, (t[1] + t[3]) / 2, (t[2] - t[0]) / 2, (t[3] - t[1]) / 2
                    if ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1:
                        inside.add(k)
            g = (len(inside), len(pts) - len(inside))
        else:
            g = grade(rec["answer"]["items"], truth)
        return job, rec, g

    with ThreadPoolExecutor(4) as ex:
        for (name, wh, v, lv), rec, g in ex.map(one, jobs):
            with out.open("a") as f:
                f.write(json.dumps({"size": name, "prompt_version": v, "level": lv, "grade": g, **rec}) + "\n")
            msg = rec.get("error") or (f"{g[0]}/7 found, {g[1]} extra | {rec['items_returned']} items, "
                                       f"{rec['seconds']:.1f} s, think {rec['usage'].get('thinking_tokens')}")
            print(f"{name:10s} {v:17s} {lv:6s} {msg}", flush=True)
    print(f"-> {out.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name")
    ap.add_argument("--tasks", nargs="+", default=V3_TASKS)
    ap.add_argument("--versions", nargs="+", default=list(VERSIONS))
    ap.add_argument("--levels", nargs="+", default=list(L.LEVELS))
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        return check()
    if not a.name:
        sys.exit("--name required")
    code = commit()
    jobs = []
    for v in a.versions:
        path = ROOT / "results" / f"run-{a.name}-{v}.jsonl"
        done = set()
        if path.exists():
            done = {(r["level"], r["task"], r["repeat"]) for r in map(json.loads, path.read_text().splitlines())
                    if "error" not in r}
        for key in a.tasks:
            for lv in a.levels:
                for rep in range(a.repeats):
                    if (lv, key, rep) not in done:
                        jobs.append((path, v, BY_KEY[key], lv, rep))
    photos = {}
    for t in {j[2] for j in jobs}:
        raw = (ROOT / "images" / t.photo).read_bytes()
        from PIL import Image
        photos[t.key] = (raw, hashlib.sha256(raw).hexdigest(), list(Image.open(io.BytesIO(raw)).size))
    print(f"{len(jobs)} calls to make", flush=True)

    def one(job):
        path, v, task, lv, rep = job
        raw, sha, size = photos[task.key]
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            rec = L.ask(raw, "image/jpeg", v, task, lv)
        except Exception as e:  # noqa: BLE001  never answered after retries: kept, visible, re-asked next run
            rec = {"error": str(e), "text": "", "answer": {"items": []}}
        record = {"model": L.MODEL_KEY, "route": "typellm:typellm-latest", "provider": "typellm", "level": lv,
                  "track": "find", "task": task.key, "task_number": task.number, "repeat": rep,
                  "prompt_version": v, "image": task.photo, "image_sha256": sha, "image_size": size,
                  "started_at": started, "price": L.PRICE, "code": code, **rec}
        with LOCK:
            with path.open("a") as f:
                f.write(json.dumps(record) + "\n")
            n = rec.get("items_returned")
            print(f"{v:17s} {lv:6s} {task.key:10s} try {rep}: "
                  + (f"ERROR {rec['error'][:120]}" if "error" in rec else
                     f"{n:3d} items, {rec['seconds']:5.1f} s, ${rec['cost_usd']:.5f}"), flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, jobs))


if __name__ == "__main__":
    main()
