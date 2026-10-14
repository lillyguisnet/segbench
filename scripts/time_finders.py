"""Pointing models from ~/Projects/finders on the whole of GPU 0: answers, speed and cost under load.

uv run --with pillow scripts/time_finders.py --name finders-3
(stop the SAM 3.1 helper first: sudo systemctl stop sam31; start it after)

Same wording as protocol v2 (scripts/run_finders.py). Writes:
  results/run-<name>.jsonl              benchmark records (find x3 repeats, trick x1), scorable
  results/run-<name>.timing.json        every measurement behind speed and cost

Speed (the chart's seconds), per benchmark question: one photo at a time,
model loaded, from opening the photo to the points on the CPU; measured as the
gap between consecutive answers of one worker (the first answer is a warm-up
and is not used). Dishes are two queries (dirty, clean): their times add.

Cost under load: K copies of the model run at once on the GPU (as many as fit),
each cycling through the benchmark queries. In the window where all copies are
working, work done = the sum of the single-copy times of the queries finished.
Load factor = window seconds / work done (below 1 = the GPU does more under
load). The best K sets it (K = 1 gives 1). Cost of a question = $0.22 per
GPU-hour x its single-copy time x the load factor.

A model answered by a service (LocalModel.service, GroundingPI's vLLM): one
service runs for the whole measurement, allowed max(photo_copies) requests
at once. "K copies" are then K processes sending requests to it at once,
which is how a vLLM model is used under load; the single-copy times are
taken on the same service with one request at a time. Its own processes
are not counted as other programs on the GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from run_finders import LABEL_OF, PHRASES, PLURAL  # noqa: E402
from segbench.local_models import BY_KEY as LOCAL, photo_setups  # noqa: E402
from segbench.tasks import BY_KEY as TASKS, TASKS as ALL_TASKS  # noqa: E402

FINDERS = Path("/home/maxime/Projects/finders")
GPU_UUID = "GPU-ed411651-10df-4a1c-981a-9cf0787065b1"
PRICE = 0.22
POINT_TASKS = ("logs", "cows", "fig", "dishes")

# our key -> (finders model, task, variant, copies to try under load, rounds of answers),
# from segbench/local_models.py (each model's notes are there).
SETUPS = photo_setups()
# Asked with plurals ("the cows"), as their own documentation asks for all instances.
PLURAL_MODELS = {k for k in SETUPS if LOCAL[k].plural}
GPUS = {"0": ("GPU-ed411651-10df-4a1c-981a-9cf0787065b1",
              "whole GPU 0: SAM 3.1 helper stopped, scholarsreadinglist embeddings job paused (SIGSTOP, 1 GB held, "
              "no compute); browser and desktop overlay hold <0.1 GB, no compute"),
        "1": ("GPU-73bb8110-4d4b-f645-dce4-724b0baf767f",
              "whole GPU 1 (same RTX 3090 model as GPU 0, no display): its four programs stopped for the run "
              "(chattering-semantic, kokoro-tts, parakeet-server, qwen3-embed-4b), nothing else on it")}
TIMING_BASIS = GPUS["0"][1]


def gpu_others(ours: set[str] = frozenset()) -> dict:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=10).stdout
    return {pid.strip(): int(mem) for uuid, pid, mem in (l.split(",") for l in out.splitlines() if l.strip())
            if uuid.strip() == GPU_UUID and pid.strip() not in ours}


def descendants(pid: int) -> set[str]:
    """pid and every process started under it (from /proc)."""
    children: dict[str, list[str]] = {}
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat.read_text().rsplit(")", 1)[1].split()
        except OSError:
            continue
        children.setdefault(fields[1], []).append(stat.parent.name)
    out, todo = set(), [str(pid)]
    while todo:
        p = todo.pop()
        out.add(p)
        todo += children.get(p, [])
    return out


class Service:
    """One running model service (finders `find.py serve`), shared by every worker."""

    def __init__(self, fmodel: str, max_at_once: int, log_path: Path):
        cmd = ["uv", "run", "--frozen", "--quiet", "--directory", str(FINDERS / "models" / fmodel), "find.py", "serve",
               "--max-num-seqs", str(max_at_once), "--port", str(_free_port())]
        env = dict(os.environ, FINDERS_GPU=GPU_UUID)
        env.pop("VIRTUAL_ENV", None)
        self.max_at_once = max_at_once
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=open(log_path, "w"), text=True, env=env)
        self.url = self.proc.stdout.readline().strip()  # printed once it answers
        if not self.url.startswith("http"):
            self.proc.kill()
            raise SystemExit(f"{fmodel} service did not start (log: {log_path})")
        print(f"   service ready at {self.url} (up to {max_at_once} at once)", flush=True)

    def pids(self) -> set[str]:
        return descendants(self.proc.pid)

    def gpu_mib(self) -> int:
        ours = self.pids()
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return sum(int(m) for p, m in (l.split(",") for l in out.splitlines() if l.strip()) if p.strip() in ours)

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(60)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Worker:
    """One copy of a model: jobs in, (arrival time, record) out."""

    def __init__(self, fmodel, variant, jobs, log_path=None, service=None):
        cmd = ["uv", "run", "--frozen", "--quiet", "--directory", str(FINDERS / "models" / fmodel), "find.py",
               "--jobs", "-", "--no-masks"] + (["--variant", variant] if variant else [])
        env = dict(os.environ, FINDERS_GPU=GPU_UUID)
        if service is not None:
            env["FINDERS_SERVER_URL"] = service.url
        env.pop("VIRTUAL_ENV", None)
        self.log = open(log_path, "w") if log_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                     text=True, env=env)
        self.out: list[tuple[float, dict]] = []
        self.proc.stdin.write("".join(json.dumps(j) + "\n" for j in jobs))
        self.proc.stdin.close()
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        for line in self.proc.stdout:
            if line.strip():
                self.out.append((time.monotonic(), json.loads(line)))

    def wait(self):
        self.proc.wait()
        self.thread.join()
        if self.proc.returncode != 0:
            raise RuntimeError(f"worker exited {self.proc.returncode}")
        return self.out


def queries(key):
    """The benchmark's queries for one setup: [(task key, label query, sent query)]."""
    plural = key in PLURAL_MODELS
    return [(t, q, PLURAL[q] if plural else q) for t in POINT_TASKS for q in PHRASES[t]]


def job(key, ftask, task_key, q, sent, rnd, tag):
    return {"image": str(ROOT / "images" / TASKS[task_key].photo), "query": sent, "task": ftask,
            "seed": rnd, "id": f"{tag}|{task_key}|{q}|{rnd}"}


def measure(key, name_out):
    fmodel, ftask, variant, copies, rounds = SETUPS[key]
    qs = queries(key)
    warm = job(key, ftask, "cows", qs[1][1], qs[1][2], 99, "warmup")
    jobs = [warm] + [job(key, ftask, t, q, s, r, "single") for r in range(rounds) for t, q, s in qs]
    plural = key in PLURAL_MODELS
    trick = [{"image": str(ROOT / "images" / t.photo), "task": ftask, "seed": 0,
              "query": PLURAL.get(t.absent.removeprefix("each "), t.absent.removeprefix("each ")) if plural
              else t.absent.removeprefix("each "), "id": f"trick|{t.key}|{t.absent}|0"} for t in ALL_TASKS]
    others_before = gpu_others()
    print(f"== {key}: {len(jobs) - 1} timed queries, {len(trick)} trick", flush=True)
    t0 = time.monotonic()
    service = Service(fmodel, max(copies, default=1), name_out.with_name(f"{name_out.stem}.{key}.service.log")) \
        if LOCAL[key].service else None
    try:
        return _measure(key, name_out, fmodel, ftask, variant, copies, qs, jobs, trick, others_before, t0, service)
    finally:
        if service is not None:
            service.stop()


def _measure(key, name_out, fmodel, ftask, variant, copies, qs, jobs, trick, others_before, t0, service):
    ours = service.pids() if service else set()
    out = Worker(fmodel, variant, jobs + trick, service=service).wait()
    others = {**others_before, **gpu_others(ours)}
    # single-copy time of each answer = gap from the previous answer (opening the photo included)
    single = []
    for (ta, ra), (tb, rb) in zip(out, out[1:]):
        if rb["id"].startswith("single"):
            single.append({"id": rb["id"], "wall": tb - ta, "model_seconds": rb["seconds"], "rec": rb})
    per_query = {}
    for s in single:
        _, tkey, q, rnd = s["id"].split("|")
        per_query.setdefault((tkey, q), []).append(s["wall"])
    work_of = {k: statistics.median(v) for k, v in per_query.items()}

    # under load
    load = [{"copies": 1, "load_factor": 1.0}]
    for k in copies:
        cyc = [job(key, ftask, t, q, s, 100 + r, "load") for r in range(4) for t, q, s in qs]
        logs = [name_out.with_name(f"{name_out.stem}.{key}.copy{i}of{k}.log") for i in range(k)]
        ws = [Worker(fmodel, variant, cyc, lg, service) for lg in logs]
        try:
            outs = [w.wait() for w in ws]
        except RuntimeError as e:  # a copy that does not fit (out of memory) is a result, not a crash
            for w in ws:
                w.proc.kill()
            tail = " ".join(logs[-1].read_text()[-400:].split()) if logs[-1].exists() else ""
            load.append({"copies": k, "load_factor": None, "failed": f"{e}; {tail[-300:]}"})
            print(f"   {k} copies: did not run ({e})", flush=True)
            continue
        start = max(o[0][0] - o[0][1]["seconds"] for o in outs)  # all copies loaded and working
        end = min(o[-1][0] for o in outs)  # until the first copy runs out of work
        done = sum(work_of[(r["id"].split("|")[1], r["id"].split("|")[2])]
                   for o in outs for t, r in o if start < t <= end)
        factor = (end - start) / done if done else None
        load.append({"copies": k, "window_s": round(end - start, 2), "work_done_s": round(done, 2),
                     "load_factor": round(factor, 4) if factor else None, "peak_gib": max(r["gpu_peak_gib"] or 0 for o in outs for _, r in o),
                     **({"service_gpu_mib": service.gpu_mib(), "service_max_at_once": service.max_at_once} if service else {})})
        print(f"   {k} copies: load factor {factor:.3f} (window {end - start:.0f}s)", flush=True)
    best = min(l["load_factor"] for l in load if l["load_factor"])

    # benchmark records
    shas = {t.key: hashlib.sha256((ROOT / "images" / t.photo).read_bytes()).hexdigest() for t in ALL_TASKS}
    sizes = {t.key: Image.open(ROOT / "images" / t.photo).size for t in ALL_TASKS}
    recs = []
    by_round: dict[tuple, list] = {}
    for s in single:
        _, tkey, q, rnd = s["id"].split("|")
        by_round.setdefault((tkey, int(rnd)), []).append((q, s))
    for (tkey, rnd), parts in sorted(by_round.items()):
        recs.append(make_record(key, "find", tkey, rnd, [(q, s["rec"], s["wall"]) for q, s in parts], best, shas, sizes,
                                others, fmodel, variant, ftask))
    for _, r in out:
        if r["id"].startswith("trick"):
            _, tkey, absent, _ = r["id"].split("|")
            recs.append(make_record(key, "trick", tkey, 0, [(absent, r, r["seconds"])], best, shas, sizes, others,
                                    fmodel, variant, ftask))
    with name_out.open("a") as f:
        f.writelines(json.dumps(r) + "\n" for r in recs)
    for tkey in POINT_TASKS:
        rs = [r for r in recs if r["track"] == "find" and r["task"] == tkey]
        counts = [len(json.loads(r["text"])["objects"]) for r in rs]
        print(f"   {tkey:6} points per repeat {counts}  {statistics.median(r['seconds'] for r in rs):6.2f}s", flush=True)
    return {"setup": key, "finders_model": fmodel, "variant": variant, "task": ftask, "rounds": LOCAL[key].photo_rounds,
            **({"service": {"url": service.url, "max_at_once": service.max_at_once}} if service else {}),
            "single": [{k: v for k, v in s.items() if k != "rec"} for s in single], "query_seconds": {
                f"{t}|{q}": round(v, 4) for (t, q), v in work_of.items()}, "load": load, "load_factor": best,
            "gpu_others": others, "elapsed_s": round(time.monotonic() - t0, 1)}


def make_record(key, track, tkey, rnd, parts, factor, shas, sizes, others, fmodel, variant, ftask):
    task = TASKS[tkey]
    W, H = sizes[tkey]
    objects = []
    for q, rec, _ in parts:
        if (rec["width"], rec["height"]) != (W, H):
            raise SystemExit(f"{key} {tkey}: finders saw {rec['width']}x{rec['height']}, our photo is {W}x{H}")
        for o in rec["objects"]:
            if o.get("point") is None:
                continue
            row = {"point": [round(o["point"][0] * 1000 / W, 2), round(o["point"][1] * 1000 / H, 2)]}
            if track == "find" and task.labels:
                row["label"] = LABEL_OF.get(q)
            objects.append(row)
    seconds = sum(w for _, _, w in parts)
    return {"model": key, "level": None, "track": track, "task": tkey, "task_number": task.number, "repeat": rnd,
            "prompt_version": "finders-phrase-v2", "prompt": " | ".join(rec.get("prompt_sent") or "" for _, rec, _ in parts),
            "image": task.photo, "image_sha256": shas[tkey], "provider": "local-gpu", "route": "local",
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seconds": round(seconds, 4), "cost_usd": PRICE / 3600 * seconds * factor,
            "cost_method": f"${PRICE}/GPU-hour x single-copy seconds x load factor {factor:.3f} (this run, {TIMING_BASIS.split(':')[0]})",
            "text": json.dumps({"objects": objects}),
            "point_from": sorted({o.get("point_from") for _, rec, _ in parts for o in rec["objects"]} - {None}),
            "finders": {"model": fmodel, "variant": variant, "task": ftask, "weights": parts[0][1].get("weights"),
                        "settings": parts[0][1].get("settings"), "queries": [
                            {"query": q, "prompt_sent": rec.get("prompt_sent"), "raw_output": rec.get("raw_output"),
                             "model_seconds": rec["seconds"]} for q, rec, _ in parts]},
            "gpu_others": others, "timing_trustworthy": not others, "timing_basis": TIMING_BASIS, "gpu_uuid": GPU_UUID}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--setups", nargs="*", default=list(SETUPS))
    ap.add_argument("--gpu", choices=list(GPUS), default="0", help="which whole GPU to time on (default 0, as before)")
    a = ap.parse_args()
    global GPU_UUID, TIMING_BASIS
    GPU_UUID, TIMING_BASIS = GPUS[a.gpu]
    out = ROOT / "results" / f"run-{a.name}.jsonl"
    if out.exists():
        sys.exit(f"{out} exists")
    timing = []
    for key in a.setups:
        timing.append(measure(key, out))
        (ROOT / "results" / f"run-{a.name}.timing.json").write_text(json.dumps(timing, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
