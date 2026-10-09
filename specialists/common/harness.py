"""Shared measurement for the specialist models that run on our own GPU.

Two numbers per model configuration, measured separately on purpose:

* **Latency** (bubble size): one image at a time, warm model, median of
  several runs. What a single user waits for.
* **Throughput under load** (cost): the GPU kept busy with batches at the
  best batch size we can find. Cost per image = rental price per hour divided
  by images per hour. API prices already assume a provider that keeps its
  GPUs busy with many users' requests, so pricing our GPU at batch 1 would
  make it look worse than it is.

Import this module BEFORE torch: it pins the process to the benchmark GPU.
Every timed call returns results already copied to the CPU, so GPU work is
finished inside the timing.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Sequence

# --- pin the GPU before torch is imported -----------------------------------
# GPU 0 (PCI 01:00.0) on lambda. Addressed by UUID so CUDA's own ordering
# cannot pick the other card.
GPU_UUID = os.environ.get("SEGBENCH_GPU_UUID", "GPU-ed411651-10df-4a1c-981a-9cf0787065b1")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID

# --- price ------------------------------------------------------------------
# Rental equivalent of one RTX 3090, US$/hour. RunPod Community Cloud list
# price on 2026-10-08 (Secure Cloud: $0.50; Vast.ai median of 64 on-demand
# offers: $0.198). Evidence: docs/evidence/. Change it here and in docs only.
PRICE_PER_GPU_HOUR = 0.22
PRICE_SOURCE = "RunPod Community Cloud RTX 3090 on-demand, 2026-10-08"

ROOT = Path(__file__).resolve().parents[2]
SMOKE_DIR = ROOT / "specialists" / "smoke_images"
RESULTS_DIR = ROOT / "results" / "specialists"


def smoke_images():
    from PIL import Image

    out = []
    for p in sorted(SMOKE_DIR.glob("*.jpg")):
        im = Image.open(p).convert("RGB")
        im.info["segbench_name"] = p.name  # lets a runner pick a per-image prompt
        out.append((p.name, im))
    return out


# What each smoke image should be asked for (text-prompted models).
SMOKE_PROMPTS = {"truck.jpg": "wheel", "groceries.jpg": "paper bag", "dog.jpg": "dog"}


def prompt_for(im) -> str:
    return SMOKE_PROMPTS.get(im.info.get("segbench_name", ""), "object")


# --- GPU monitoring -----------------------------------------------------------
def _smi(*query: str) -> list[list[str]]:
    out = subprocess.run(
        ["nvidia-smi", *query, "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10,
    ).stdout
    return [[c.strip() for c in line.split(",")] for line in out.strip().splitlines() if line.strip()]


def other_gpu_programs() -> dict[int, int]:
    """Programs other than us computing on the benchmark GPU: pid -> MiB."""
    rows = _smi("--query-compute-apps=gpu_uuid,pid,used_memory")
    me = os.getpid()
    return {int(pid): int(mem) for uuid, pid, mem in rows if uuid == GPU_UUID and int(pid) != me}


class UtilizationSampler:
    """Samples GPU busy % every 0.25 s in the background."""

    def __init__(self) -> None:
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                rows = _smi("--query-gpu=uuid,utilization.gpu")
                self.samples += [int(u) for uuid, u in rows if uuid == GPU_UUID]
            except Exception:
                pass
            self._stop.wait(0.25)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def summary(self) -> dict[str, float | None]:
        s = self.samples
        if not s:
            return {"util_mean": None, "util_median": None}
        return {"util_mean": round(statistics.fmean(s), 1), "util_median": statistics.median(s)}


# --- the measurement ------------------------------------------------------------
def run(
    *,
    model_id: str,
    config: dict[str, Any],
    load: Callable[[], Any],
    predict: Callable[[Any, list], list],
    summarize: Callable[[Any], dict] | None = None,
    prepare: Callable[[Any, int], Any] | None = None,
    batch_sizes: Sequence[int] = (1,),
    latency_runs: int = 10,
    warmup: int = 3,
    load_seconds_per_point: float = 20.0,
    notes: str = "",
) -> dict:
    """Load, smoke-test, time and price one model configuration.

    load()                 -> state
    prepare(state, bs)     -> state  (optional, untimed: e.g. compile for a batch size)
    predict(state, images) -> one CPU result per image
    summarize(result)      -> small dict saved as the smoke-test evidence
    """
    import torch

    # TF32 on Ampere: tensor cores for fp32 matmuls/convs, a standard serving setting.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    images = smoke_images()
    pil = [im for _, im in images]
    others_before = other_gpu_programs()

    t = time.perf_counter()
    state = load()
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t
    if prepare:
        state = prepare(state, 1)

    def timed(batch: list) -> tuple[float, list]:
        t0 = time.perf_counter()
        out = predict(state, batch)
        torch.cuda.synchronize()
        return time.perf_counter() - t0, out

    for i in range(warmup):
        timed([pil[i % len(pil)]])

    # Smoke test: one output per image, summarized.
    smoke = {}
    for name, im in images:
        _, out = timed([im])
        smoke[name] = summarize(out[0]) if summarize else {"ok": out[0] is not None}

    latencies = [timed([pil[i % len(pil)]])[0] for i in range(latency_runs)]

    # Throughput: largest batch that fits, each run for a fixed wall time.
    sweep = []
    for bs in batch_sizes:
        batch = [pil[i % len(pil)] for i in range(bs)]
        try:
            if prepare:
                state = prepare(state, bs)
            torch.cuda.reset_peak_memory_stats()
            timed(batch)  # warm this batch size
            n, t0 = 0, time.perf_counter()
            with UtilizationSampler() as sampler:
                while time.perf_counter() - t0 < load_seconds_per_point:
                    # rotate through all images so content-dependent models are timed on every one
                    timed([pil[(n + i) % len(pil)] for i in range(bs)])
                    n += bs
            elapsed = time.perf_counter() - t0
            sweep.append({
                "batch_size": bs,
                "images_per_s": round(n / elapsed, 3),
                "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                **sampler.summary(),
            })
        except torch.OutOfMemoryError:
            sweep.append({"batch_size": bs, "out_of_memory": True})
            torch.cuda.empty_cache()
            break
        print(f"  {model_id} batch {bs}: {sweep[-1]}", flush=True)

    ok = [p for p in sweep if "images_per_s" in p]
    best = max(ok, key=lambda p: p["images_per_s"])
    others_after = other_gpu_programs()
    foreign = {pid: mem for pid, mem in others_after.items() if pid not in others_before}
    heavy_before = {pid: mem for pid, mem in others_before.items() if mem > 500}

    record = {
        "date": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_id": model_id,
        "config": config,
        "notes": notes,
        "load_seconds": round(load_s, 2),
        "latency_s": {
            "median": round(statistics.median(latencies), 4),
            "min": round(min(latencies), 4),
            "max": round(max(latencies), 4),
            "runs": latency_runs,
            "batch_size": 1,
        },
        "throughput": {"sweep": sweep, "best": best},
        "cost": {
            "usd_per_image_under_load": PRICE_PER_GPU_HOUR / 3600 / best["images_per_s"],
            "usd_per_image_at_batch_1": PRICE_PER_GPU_HOUR / 3600 * statistics.median(latencies),
            "usd_per_gpu_hour": PRICE_PER_GPU_HOUR,
            "price_source": PRICE_SOURCE,
        },
        "smoke": smoke,
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "uuid": GPU_UUID,
            "other_programs_before_mib": others_before,
            "new_programs_during_run_mib": foreign,
            "timing_trustworthy": not foreign and not heavy_before,
        },
        "software": {"tf32": True, "cudnn_benchmark": True, "torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version()},
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{_dt.date.today().isoformat()}.jsonl"
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(
        f"{model_id}: latency {record['latency_s']['median']:.3f}s | "
        f"best {best['images_per_s']} img/s at batch {best['batch_size']} "
        f"(GPU busy {best.get('util_mean')}%) | "
        f"${record['cost']['usd_per_image_under_load']:.6f}/image | smoke {smoke}",
        flush=True,
    )
    return record


# --- several workers sharing the GPU ------------------------------------------
def _worker(make_predict, seconds, start_barrier, result_queue, worker_index):
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    pil = [im for _, im in smoke_images()]
    predict = make_predict()
    for i in range(2):
        predict([pil[i % len(pil)]])
    torch.cuda.synchronize()
    start_barrier.wait()
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        predict([pil[(n + worker_index) % len(pil)]])
        n += 1
    torch.cuda.synchronize()
    result_queue.put((n, time.perf_counter() - t0))


def run_concurrent(*, model_id: str, config: dict, make_predict: Callable, workers: int,
                   seconds: float = 40.0, notes: str = "") -> dict:
    """Throughput with several processes sharing the GPU (for models whose
    CPU-side work leaves the GPU idle part of the time). make_predict must be
    a top-level function returning predict(images) -> CPU results."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    barrier, queue = ctx.Barrier(workers + 1), ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(make_predict, seconds, barrier, queue, i)) for i in range(workers)]
    for p in procs:
        p.start()
    try:
        # A worker that crashes (e.g. out of GPU memory) breaks the barrier or
        # never reports: fail loudly instead of waiting forever.
        barrier.wait(timeout=900)
        with UtilizationSampler() as sampler:
            results = [queue.get(timeout=seconds + 600) for _ in procs]
    finally:
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
    total = sum(n for n, _ in results)
    elapsed = max(t for _, t in results)
    ips = total / elapsed
    record = {
        "date": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_id": model_id,
        "kind": "concurrent_throughput",
        "config": config,
        "notes": notes,
        "workers": workers,
        "images_per_s": round(ips, 3),
        **sampler.summary(),
        "cost": {"usd_per_image_under_load": PRICE_PER_GPU_HOUR / 3600 / ips,
                 "usd_per_gpu_hour": PRICE_PER_GPU_HOUR, "price_source": PRICE_SOURCE},
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / f"{_dt.date.today().isoformat()}.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"{model_id}: {workers} workers -> {ips:.3f} img/s (GPU busy {record['util_mean']}%) "
          f"${record['cost']['usd_per_image_under_load']:.6f}/image", flush=True)
    return record
