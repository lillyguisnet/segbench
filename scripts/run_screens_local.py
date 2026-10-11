"""Ask the local pointing models (~/Projects/finders) where to click on the screenshots.

    uv run --with pillow scripts/run_screens_local.py --name screens-local-1

Both rounds' questions (segbench/screens.py ROUNDS), the same screenshots as
the language models. Writes results/run-<name>.jsonl in run_screens.py's
format (clicks in 0..1000), so score_screens.py and plot_screens.py read
both alike. Never appends to an existing file.

Wording: each model gets the instruction exactly as the language models do
("Fold every open conversation group except segbench."), inside its own
documented template (MolmoPoint "Point to {q}", LocateAnything "Point to:
{q}.", ...). Not rewritten into a noun phrase: that would be us solving the
task for them, and tuning on these screenshots.

Clicks: the point each model gives (MolmoPoint, LocateAnything, Rex-Omni
point); for Florence-2 the centre of each box; for SAM 3 the spot deepest
inside each outline (finders' `point_from` is kept per record).

Repeats: LocateAnything samples (temperature 0.7), so 3 seeds like the
language models' 3 tries; the others are deterministic: one answer, scored
for every try.

Time: warm model seconds per screenshot (a warm-up question first, not
recorded). Run only on the whole of GPU 0 (scripts/run_screens_local.sh
waits for it). Cost: $0.22 per GPU-hour x those seconds x each model's load
factor measured on the photos (scripts/time_finders.py, specialists
throughput test): not re-measured on screenshots.
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

from segbench.screens import PROMPT_VERSION, ROUNDS  # noqa: E402

GPU_UUID = "GPU-ed411651-10df-4a1c-981a-9cf0787065b1"  # GPU 0, the benchmark GPU
PRICE_PER_GPU_HOUR = 0.22
# key -> (finders model, task, variant, seeds, load factor measured on the photos)
MODELS = {
    "locateanything-3b": ("locateanything", "point", None, 3, 1.0),
    "molmopoint-8b": ("molmopoint", "point", None, 1, 1.0),
    "rexomni-3b": ("rexomni", "point", None, 1, 0.6254),
    "florence2-large-grounding": ("florence2", "box", "large", 1, 0.4041),
    "florence2-large-ovd": ("florence2", "ovd", "large", 1, 0.528),
    "sam3": ("sam3", "segment", None, 1, 1.022),
}


def gpu_others() -> dict:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=10).stdout
    return {pid.strip(): int(mem) for uuid, pid, mem in (l.split(",") for l in out.splitlines() if l.strip())
            if uuid.strip() == GPU_UUID}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    args = ap.parse_args()
    out = ROOT / "results" / f"run-{args.name}.jsonl"
    if out.exists():
        sys.exit(f"{out} exists: choose a new --name")
    tasks = [t for r in sorted(ROUNDS) for t in ROUNDS[r]]
    shots = {}
    for t in tasks:
        path = ROOT / "images" / "screenshots" / t.image
        data = path.read_bytes()
        shots[t.key] = {"path": path, "sha": hashlib.sha256(data).hexdigest(),
                        "size": [int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")]}

    for key in args.models:
        fmodel, ftask, variant, seeds, factor = MODELS[key]
        warm = {"image": str(shots[tasks[0].key]["path"]), "query": tasks[0].instruction, "task": ftask, "id": "warm-up"}
        jobs = [warm] + [{"image": str(shots[t.key]["path"]), "query": t.instruction, "task": ftask,
                          "seed": s, "id": f"{t.key}|{s}"} for t in tasks for s in range(seeds)]
        before = gpu_others()
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        print(f"== {key}: {len(jobs) - 1} questions", flush=True)
        recs = finders.find_many(fmodel, jobs, variant=variant, masks=False, gpu=GPU_UUID)
        others = {**before, **gpu_others()}
        out.with_name(out.stem + f".raw-{key}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        with out.open("a") as f:
            for job, rec in zip(jobs[1:], recs[1:], strict=True):
                tkey, seed = job["id"].split("|")
                W, H = rec["width"], rec["height"]
                if [W, H] != shots[tkey]["size"]:
                    raise SystemExit(f"{key} {tkey}: finders saw {W}x{H}, the screenshot is {shots[tkey]['size']}")
                clicks = [{"x": round(o["point"][0] * 1000 / W, 2), "y": round(o["point"][1] * 1000 / H, 2), "form": "model"}
                          for o in rec["objects"] if o.get("point") is not None]
                seconds = rec["seconds"]
                f.write(json.dumps({
                    "model": key, "level": None, "track": "screen", "task": tkey, "repeat": int(seed),
                    "prompt_version": PROMPT_VERSION + "-local", "prompt": rec.get("prompt_sent") or job["query"],
                    "image": "screenshots/" + shots[tkey]["path"].name, "image_sha256": shots[tkey]["sha"],
                    "image_size": [W, H], "provider": "local-gpu", "route": "local", "started_at": started,
                    "seconds": round(seconds, 4), "cost_usd": PRICE_PER_GPU_HOUR / 3600 * seconds * factor,
                    "cost_method": f"${PRICE_PER_GPU_HOUR}/GPU-hour x warm model seconds x load factor {factor} (measured on the photos)",
                    "readable": True, "read_as": "model", "clicks": clicks,
                    "point_from": sorted({o.get("point_from") for o in rec["objects"] if o.get("point_from")}),
                    "raw_output": rec.get("raw_output"),
                    "finders": {"model": fmodel, "variant": variant, "task": ftask, "seed": int(seed),
                                "weights": rec.get("weights"), "settings": rec.get("settings"),
                                "load_seconds": recs[0].get("load_seconds"), "gpu_peak_gib": rec.get("gpu_peak_gib")},
                    "deterministic": seeds == 1, "gpu_others": others,
                }) + "\n")
                print(f"   {tkey:17} seed {seed}  {len(clicks):3} clicks  {seconds:6.2f}s", flush=True)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
