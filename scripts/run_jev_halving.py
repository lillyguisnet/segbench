"""Jev + halving on the benchmark photos (Maxime, 2026-10-16): the eleven object tasks of point benchmark v3.

    set -a; source .env; set +a
    uv run --with numpy --with pillow scripts/run_jev_halving.py --check        # synthetic ovals only, no photo
    uv run --with numpy --with pillow scripts/run_jev_halving.py --name jev-halving-1

-> results/run-<name>.jsonl         one record per task and try, read by
                                    scripts/score_points.py (text: the box
                                    middles as points) and scripts/score_boxes.py
                                    (answer.items: the boxes), 0..1000 grid
   results/run-<name>.calls.jsonl   every request's call record (segbench.call)
   results/run-<name>.trees.jsonl   every search's pieces, answers and cuts

The search is segbench/jev_halving.py, fixed before any photo was sent. No
screenshots. Three tries per task, as every language model; tasks one after
another, so a search's seconds are not slowed by another one running. A
record's seconds = the whole search's waiting time; its cost = the sum of its
requests. The question words are the task's own (segbench/tasks.py target,
and the dish definition for dishes), the same as every model gets; dishes are
asked by location only (no dirty/clean, as scored since v2.2). Appending:
a task and try already in the file is skipped, so an interrupted run continues.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from segbench import cost  # noqa: E402
from segbench import jev_halving as jh  # noqa: E402
from segbench import jev_verify as jv  # noqa: E402
from segbench.tasks import BY_KEY as TASKS  # noqa: E402

V3_TASKS = ["logs", "cows", "fig", "dishes", "ants", "beetles", "birds", "mushrooms", "jars", "pinkleaves", "whitecow"]
REPEATS = 3


def commit() -> str:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "segbench/jev_halving.py", "segbench/jev_verify.py",
                            "scripts/run_jev_halving.py"],
                           cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return head + ("-dirty" if dirty else "")


def one(im: Image.Image, target: str, note: str = "", verify: bool = False) -> tuple[dict, jh.Search]:
    q = jh.questions(target, note)
    s = jh.Search(im, q)
    t = time.perf_counter()
    boxes = s.run()
    s.verify_log = None
    if verify:
        v = jv.Verifier(s, target, note)
        s.halving_boxes = boxes
        boxes = v.run(boxes)
        s.verify_log = v.log
    wall = time.perf_counter() - t
    w, h = s.im.size
    items = []
    for b in boxes:
        g = [b[0] * 1000 / w, b[1] * 1000 / h, b[2] * 1000 / w, b[3] * 1000 / h]
        g = [round(min(1000.0, max(0.0, v)), 1) for v in g]
        items.append({"box": g, "point": [round((g[0] + g[2]) / 2, 1), round((g[1] + g[3]) / 2, 1)]})
    calls = s.records
    out = {
        "seconds": round(wall, 3),
        "requests": len(calls),
        "failed_requests": sum(1 for r in calls if r.get("error")),
        "usage": {"input_tokens": sum(r["usage"]["input_tokens"] for r in calls if r.get("usage")), "output_tokens": 0},
        "cost_usd": sum(r["cost_usd"] or 0 for r in calls),
        "forced": sum(1 for n in s.nodes if n.get("forced")),
        "max_depth_reached": max((n["depth"] for n in s.nodes), default=0),
        "answer": {"items": items},
        "text": json.dumps({"objects": [{"x": i["point"][0], "y": i["point"][1]} for i in items]}),
        "questions": q,
    }
    return out, s


def check(verify=False):
    """The task wording on the synthetic ovals (the probe's own picture), before any photo."""
    from check_boxes import SIZES
    from probe_jev_split import draw, grade
    for name, (w, h) in SIZES.items():
        im, truth, masks = draw(w, h)
        rec, s = one(im, "each red oval (not the blue squares)", verify=verify)
        found = [[i["box"][0] * w / 1000, i["box"][1] * h / 1000, i["box"][2] * w / 1000, i["box"][3] * h / 1000]
                 for i in rec["answer"]["items"]]
        print(f"{name:11s} {grade(found, truth, masks)} | {rec['requests']} requests, {rec['seconds']} s, "
              f"${rec['cost_usd']:.4f}, forced {rec['forced']}"
              + (f", verify {[e['outcome'] for e in s.verify_log]}" if s.verify_log else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name")
    ap.add_argument("--tasks", nargs="+", default=V3_TASKS)
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--verify", action="store_true", help="feedback loop on the final boxes (segbench/jev_verify.py)")
    ap.add_argument("--max-depth", type=int, default=jh.MAX_DEPTH,
                    help=f"most halvings of one piece (default {jh.MAX_DEPTH}, as run jev-halving-1)")
    a = ap.parse_args()
    jh.MAX_DEPTH = a.max_depth
    if a.check:
        return check(a.verify)
    if not a.name:
        sys.exit("--name required")
    out = ROOT / "results" / f"run-{a.name}.jsonl"
    calls_out, trees_out = out.with_suffix(".calls.jsonl"), out.with_suffix(".trees.jsonl")
    done = set()
    if out.exists():
        done = {(r["task"], r["repeat"]) for r in map(json.loads, out.read_text().splitlines())}
    code = commit()
    for key in a.tasks:
        task = TASKS[key]
        path = ROOT / "images" / task.photo
        raw = path.read_bytes()
        im = Image.open(path)
        im.load()
        for repeat in range(a.repeats):
            if (key, repeat) in done:
                continue
            started = datetime.now(timezone.utc).isoformat(timespec="seconds")
            try:
                rec, s = one(im, task.target, task.note, verify=a.verify)
            except Exception as e:  # noqa: BLE001  a request that never got an answer: the try is recorded as failed
                rec, s = {"error": f"{type(e).__name__}: {e}", "text": "", "answer": {"items": []}}, None
            record = {"model": jh.MODEL.key, "route": jh.MODEL.route, "provider": "typesafe", "level": "medium",
                      "track": "find", "task": key, "task_number": task.number, "repeat": repeat,
                      "prompt_version": jh.VERSION + (f"+{jv.VERSION}" if a.verify else ""), "image": task.photo,
                      "image_sha256": hashlib.sha256(raw).hexdigest(), "image_size": list(im.size),
                      "started_at": started, "prices_date": cost.PRICES_DATE, "code": code,
                      "max_depth": jh.MAX_DEPTH, **rec}
            with out.open("a") as f:
                f.write(json.dumps(record) + "\n")
            if s is not None:
                with calls_out.open("a") as f:
                    for r in s.records:
                        f.write(json.dumps({"task": key, "repeat": repeat, **r}) + "\n")
                with trees_out.open("a") as f:
                    f.write(json.dumps({"task": key, "repeat": repeat, "image_size": list(s.im.size),
                                        "nodes": s.nodes, "cuts": s.cuts, "raw": s.raw,
                                        "verify": s.verify_log,
                                        "halving_boxes": getattr(s, "halving_boxes", None)}) + "\n")
            print(f"{key:11s} try {repeat}: {len(rec['answer']['items']):3d} objects, "
                  f"{rec.get('requests', 0):4d} requests, {rec.get('seconds', 0):6.1f} s, "
                  f"${rec.get('cost_usd', 0):.4f}, forced {rec.get('forced', 0)}"
                  + (f"  ERROR {rec['error']}" if rec.get("error") else ""), flush=True)


if __name__ == "__main__":
    main()
