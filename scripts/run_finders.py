"""Run the shared pointing models (~/Projects/finders) on the benchmark photos.

uv run --with pillow scripts/run_finders.py --name finders-1                 # headline: short phrases
uv run --with pillow scripts/run_finders.py --name finders-1-sentence --sentence locateanything molmopoint

Writes results/run-<name>.jsonl in the same record format as scripts/run.py
(answers as {"objects": [{"point": [x, y]}]} in 0..1000), so
scripts/score_points.py scores them like every other entrant. Never appends
to an existing file: a new run gets a new name.

Wording (fixed 2026-10-09, before any of these models saw a benchmark photo):
- headline: the same short phrases as SAM 3 and YOLOE
  (specialists/common/answers.py PHRASES, prompt version spec-v1), inside
  each model's own documented template. Dishes: one query per label
  ("dirty dish", "clean dish"); both answers are kept as given (these models
  give no score to choose between two labels on one dish).
- --sentence: the exact target the language models were asked for
  (segbench/tasks.py), for models that read sentences. A diagnostic, never
  the headline, so the wording is not picked after seeing scores.

SAM 3 is not rerun here: it is already in the benchmark (specialists/sam).

Time = the model's own seconds per picture, warm (load time excluded, as for
the other specialists); a task with two queries (dishes) adds both. Cost =
$0.22 per GPU-hour x those seconds, one picture at a time: no throughput test
under load was made for these models, so this overstates their cost compared
with the load-tested specialists. Every record lists other programs on the
GPU; timings are provisional when any were there.
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
from PIL import Image  # noqa: E402  (fail at start, not after GPU work)

from segbench.tasks import BY_KEY as TASKS, TASKS as ALL_TASKS  # noqa: E402

GPU_UUID = "GPU-ed411651-10df-4a1c-981a-9cf0787065b1"  # GPU 0, the benchmark GPU
PRICE_PER_GPU_HOUR = 0.22  # RunPod Community Cloud RTX 3090, 2026-10-08 (as the other specialists)
POINT_TASKS = ("logs", "cows", "fig", "dishes")

# Same as specialists/common/answers.py (spec-v1); copied, not imported, because
# that module pins torch-side settings on import.
PHRASES = {"logs": ["log end"], "cows": ["cow"], "fig": ["fig leaf"], "dishes": ["dirty dish", "clean dish"]}
LABEL_OF = {"dirty dish": "dirty", "clean dish": "clean"}

# Protocol v2 (2026-10-09, after reading v1's raw answers; v1 kept on file):
# each model asked for ALL instances the way its own documentation does it.
# - MolmoPoint: plural with "the" (card examples: "Point to the boats",
#   "Point to the penguins"); singular "Point to cow" returned one cow.
# - LocateAnything: its pointing template "Point to: {q}." with the plural
#   (no documented example either way; plural = "all of them", same rule).
# - Rex-Omni: unchanged; its README uses singular category names ("person").
# - Florence-2: open-vocabulary detection (<OPEN_VOCABULARY_DETECTION>), its
#   mode for finding every object of a kind, instead of caption phrase
#   grounding, which drew one box around the whole log pile in v1.
# The plurals are plain English, written once, the same for every model that
# takes plurals; they were not adjusted to any score.
PLURAL = {"log end": "the log ends", "cow": "the cows", "fig leaf": "the fig leaves",
          "dirty dish": "the dirty dishes", "clean dish": "the clean dishes",
          "tractor": "the tractors", "car": "the cars", "person": "the people", "cat": "the cats"}
PLURAL_MODELS = {"locateanything-3b", "molmopoint-8b"}
V2_TASK = {"florence2-large": "ovd"}

# Our key -> (finders model, task, variant). Defaults of the finders package.
MODELS = {
    "locateanything-3b": ("locateanything", "point", None),
    "molmopoint-8b": ("molmopoint", "point", None),
    "rexomni-3b": ("rexomni", "point", None),
    "florence2-large": ("florence2", "box", "large"),
}


def sentence_queries(task) -> list[str]:
    if task.labels:  # same object list and definition as the language models, one label per query
        return [f"each {lab} dish. {task.note}" for lab in task.labels]
    return [task.target]


def gpu_others() -> dict:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=10).stdout
    return {pid.strip(): int(mem) for uuid, pid, mem in (l.split(",") for l in out.splitlines() if l.strip())
            if uuid.strip() == GPU_UUID}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--protocol", choices=("v1", "v2"), default="v2",
                    help="v1: singular phrases, finders defaults (first run); v2: documented all-instance wording")
    ap.add_argument("--sentence", nargs="*", metavar="FINDERS_MODEL",
                    help="diagnostic: full-sentence wording for these finders models (no trick questions)")
    args = ap.parse_args()
    out = ROOT / "results" / f"run-{args.name}.jsonl"
    if out.exists():
        sys.exit(f"{out} exists: choose a new --name")
    keys = [k for k in args.models if args.sentence is None or MODELS[k][0] in args.sentence]

    photos = {}
    for t in ALL_TASKS:
        data = (ROOT / "images" / t.photo).read_bytes()
        photos[t.key] = {"path": ROOT / "images" / t.photo, "sha": hashlib.sha256(data).hexdigest()}

    for key in keys:
        fmodel, ftask, variant = MODELS[key]
        if args.protocol == "v2":
            ftask = V2_TASK.get(key, ftask)
        wording = (lambda q: PLURAL[q]) if args.protocol == "v2" and key in PLURAL_MODELS else (lambda q: q)
        jobs, meta = [], []
        for tkey in POINT_TASKS:
            task = TASKS[tkey]
            queries = sentence_queries(task) if args.sentence else PHRASES[tkey]
            for q in queries:
                jobs.append({"image": str(photos[tkey]["path"]), "query": q if args.sentence else wording(q), "task": ftask,
                             "id": f"{tkey}|find|{q}"})
                meta.append((tkey, "find", q))
        if args.sentence is None:
            for task in ALL_TASKS:  # trick questions on all six photos
                q = task.absent.removeprefix("each ")
                jobs.append({"image": str(photos[task.key]["path"]), "query": wording(q), "task": ftask,
                             "id": f"{task.key}|trick|{q}"})
                meta.append((task.key, "trick", q))
        before = gpu_others()
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        print(f"== {key}: {len(jobs)} queries", flush=True)
        recs = finders.find_many(fmodel, jobs, variant=variant, masks=False, gpu=GPU_UUID)
        after = gpu_others()
        raw_file = out.with_name(out.stem + f".raw-{key}.jsonl")  # the finders records as returned, saved first
        raw_file.write_text("".join(json.dumps(r) + "\n" for r in recs))
        # group queries of one task/track (dishes: two labels) into one benchmark call
        grouped: dict[tuple, list] = {}
        for (tkey, track, q), rec in zip(meta, recs, strict=True):
            grouped.setdefault((tkey, track), []).append((q, rec))
        with out.open("a") as f:
            for (tkey, track), parts in grouped.items():
                task = TASKS[tkey]
                objects, point_from, raw = [], set(), []
                for q, rec in parts:
                    W, H = rec["width"], rec["height"]
                    with Image.open(photos[tkey]["path"]) as im:  # same pixel grid as ours
                        if im.size != (W, H):
                            raise SystemExit(f"{key} {tkey}: finders saw {W}x{H}, our photo is {im.size}")
                    for o in rec["objects"]:
                        if o.get("point") is None:
                            continue
                        x, y = o["point"]
                        row = {"point": [round(x * 1000 / W, 2), round(y * 1000 / H, 2)]}
                        if track == "find" and task.labels:
                            label = LABEL_OF.get(q) or next((lab for lab in task.labels if q.startswith(f"each {lab} ")), None)
                            row["label"] = label
                        objects.append(row)
                        point_from.add(o.get("point_from"))
                    raw.append({"query": q, "prompt_sent": rec.get("prompt_sent"), "raw_output": rec.get("raw_output"),
                                "n_objects": len(rec["objects"])})
                seconds = sum(rec["seconds"] for _, rec in parts)
                others = {p: m for p, m in {**before, **after}.items()}
                f.write(json.dumps({
                    "model": key, "level": None, "track": track, "task": tkey, "task_number": task.number, "repeat": 0,
                    "prompt_version": "finders-sentence-v1" if args.sentence else f"finders-phrase-{args.protocol}",
                    "prompt": " | ".join(p["prompt_sent"] or p["query"] for p in raw),
                    "image": task.photo, "image_sha256": photos[tkey]["sha"], "provider": "local-gpu", "route": "local",
                    "started_at": started, "seconds": round(seconds, 4), "model_seconds": round(seconds, 4),
                    "cost_usd": PRICE_PER_GPU_HOUR / 3600 * seconds,
                    "cost_method": f"${PRICE_PER_GPU_HOUR}/GPU-hour x warm model seconds, one picture at a time (no load test)",
                    "text": json.dumps({"objects": objects}), "point_from": sorted(p for p in point_from if p),
                    "finders": {"model": fmodel, "variant": variant, "task": ftask, "weights": parts[0][1].get("weights"),
                                "settings": parts[0][1].get("settings"), "load_seconds": parts[0][1].get("load_seconds"),
                                "gpu_peak_gib": max(r.get("gpu_peak_gib") or 0 for _, r in parts), "queries": raw},
                    "gpu_others": others, "timing_trustworthy": not others,
                }) + "\n")
                n = len(objects)
                print(f"   {tkey:6} {track:5} {n:4} points  {seconds:6.2f}s  from {sorted(point_from - {None})}", flush=True)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
