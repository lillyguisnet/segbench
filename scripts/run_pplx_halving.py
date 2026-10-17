"""pplx-flash + halving on the benchmark photos (2026-10-17): the eleven object tasks of point benchmark v3.

    uv run --with numpy --with pillow scripts/run_pplx_halving.py --check      # synthetic ovals only, no photo
    uv run --with numpy --with pillow scripts/run_pplx_halving.py --name pplx-halving-1

The model runs on our GPU (default GPU 0) through ~/Projects/deciders; nothing is
sent anywhere. -> results/run-<name>.jsonl (one record per task and try, read by
scripts/score_points.py; points on the 0..1000 grid), run-<name>.trees.jsonl
(every piece, answer, cut and zoom step), run-<name>.decider.log (the model
process's log).

The search is segbench/pplx_halving.py, fixed before any photo was sent. Tasks,
words, tries and record format as scripts/run_jev_halving.py (jev-halving-1):
the task's own words, dishes by location only, three tries, tasks one after
another. A record's seconds = the whole search's waiting time on one RTX 3090
(model already loaded); cost 0 (own GPU, electricity not priced). Appending: a
task and try already in the file is skipped.
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
from segbench import pplx_halving as ph  # noqa: E402
from segbench.tasks import BY_KEY as TASKS  # noqa: E402

V3_TASKS = ["logs", "cows", "fig", "dishes", "ants", "beetles", "birds", "mushrooms", "jars", "pinkleaves", "whitecow"]
REPEATS = 3


def commit() -> str:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "segbench/pplx_halving.py", "scripts/run_pplx_halving.py"],
                           cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return head + ("-dirty" if dirty else "")


def gpu_others(gpu: str) -> dict:
    """Other programs on the GPU at the start of a search: pid -> MiB."""
    try:
        uuid = subprocess.run(["nvidia-smi", "-i", gpu, "--query-gpu=uuid", "--format=csv,noheader"],
                              capture_output=True, text=True, check=True).stdout.strip()
        rows = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        return {"unknown": str(e)}
    return {pid.strip(): int(mib) for u, pid, mib in (r.split(",") for r in rows.strip().splitlines()) if u.strip() == uuid}


def one(decider: ph.Decider, im: Image.Image, target: str, note: str = "") -> tuple[dict, ph.Search]:
    s = ph.Search(decider, im, target, note)
    t = time.perf_counter()
    found = s.run()
    wall = time.perf_counter() - t
    w, h = s.im.size
    items = [{"point": [round(min(1000.0, max(0.0, f["point"][0] * 1000 / w)), 1),
                        round(min(1000.0, max(0.0, f["point"][1] * 1000 / h)), 1)]} for f in found]
    out = {
        "seconds": round(wall, 3),
        "requests": s.requests,
        "failed_requests": 0,
        "usage": {"input_tokens": s.tokens, "output_tokens": 0},
        "cost_usd": 0.0,
        "cost_method": "own GPU (RTX 3090), electricity not priced",
        "forced": sum(1 for f in found if f["forced"]),
        "max_depth_reached": max((n["depth"] for n in s.nodes), default=0),
        "answer": {"items": items},
        "text": json.dumps({"objects": [{"x": i["point"][0], "y": i["point"][1]} for i in items]}),
        "questions": {"count": ph.count_question(target, note), "locate": ph.locate_question(target, note, True),
                      "locate_next": ph.locate_question(target, note, False)},
    }
    return out, s


def check(decider):
    """The task-style wording on the synthetic ovals (Jev's probe picture), before any photo."""
    from check_boxes import SIZES
    from probe_jev_split import draw
    for name, (w, h) in SIZES.items():
        im, truth, masks = draw(w, h)
        rec, s = one(decider, im, "each red oval (not the blue squares)")
        used, hits, extra = set(), 0, 0
        for it in rec["answer"]["items"]:
            x, y = int(it["point"][0] * w / 1000), int(it["point"][1] * h / 1000)
            hit = next((i for i, m in enumerate(masks) if i not in used and m[min(y, h - 1), min(x, w - 1)]), None)
            if hit is None:
                extra += 1
            else:
                used.add(hit)
                hits += 1
        print(f"{name:11s} points on a different oval {hits} of {len(truth)}, off any oval {extra} | "
              f"{rec['requests']} requests, {rec['seconds']} s, forced {rec['forced']}, "
              f"depth {rec['max_depth_reached']}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name")
    ap.add_argument("--tasks", nargs="+", default=V3_TASKS)
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not a.check and not a.name:
        sys.exit("--name required")
    log = ROOT / "results" / (f"run-{a.name}.decider.log" if a.name else "check-pplx-halving.decider.log")
    t = time.perf_counter()
    decider = ph.Decider(a.gpu, log)
    decider.decide([{"state": "warm-up", "questions": {"q": {"type": "noul"}}}])
    print(f"model loaded in {time.perf_counter() - t:.1f} s", flush=True)
    try:
        if a.check:
            return check(decider)
        out = ROOT / "results" / f"run-{a.name}.jsonl"
        trees_out = out.with_suffix(".trees.jsonl")
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
                others = gpu_others(a.gpu)
                rec, s = one(decider, im, task.target, task.note)
                record = {"model": ph.MODEL_KEY, "route": ph.ROUTE, "provider": "local", "level": None,
                          "track": "find", "task": key, "task_number": task.number, "repeat": repeat,
                          "prompt_version": ph.VERSION, "image": task.photo,
                          "image_sha256": hashlib.sha256(raw).hexdigest(), "image_size": list(im.size),
                          "started_at": started, "code": code, "gpu": a.gpu, "gpu_others": others,
                          "max_depth": ph.MAX_DEPTH, **rec}
                with out.open("a") as f:
                    f.write(json.dumps(record) + "\n")
                with trees_out.open("a") as f:
                    f.write(json.dumps({"task": key, "repeat": repeat, "image_size": list(s.im.size),
                                        "nodes": s.nodes, "cuts": s.cuts, "objects": s.steps}) + "\n")
                print(f"{key:11s} try {repeat}: {len(rec['answer']['items']):3d} objects, "
                      f"{rec['requests']:4d} requests, {rec['seconds']:6.1f} s, forced {rec['forced']}, "
                      f"depth {rec['max_depth_reached']}", flush=True)
    finally:
        decider.close()


if __name__ == "__main__":
    main()
