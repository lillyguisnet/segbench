"""Local finders models asked for boxes (box benchmark, 2026-10-15): answers, speed and cost under load.

uv run --with pillow scripts/time_finders_boxes.py --name finders-boxes-1 --basis "..."

Exactly scripts/time_finders.py (same whole-GPU method, same load test, same
price), with three differences:
- each model's own box task in ~/Projects/finders instead of its point task
  (LocateAnything, Rex-Omni, Moondream, GroundingPI: `box`, "all instances";
  Florence-2 Large: phrase grounding, its chosen mode on the photos, which
  already answers with boxes);
- answers on all eleven object tasks (the first four and the second batch),
  with the same short phrases as for points (plurals for the models whose
  documentation asks that way); no trick questions (Maxime, 2026-10-15);
- speed and cost: single-copy times on the first four tasks only, and the
  load test on those four, as in point v3 (the second batch's photos are
  about ten times smaller; cost must not depend on which photos were timed).
  Records of the second batch keep their own measured seconds as
  seconds_measured.

MolmoPoint is not here: it can only point.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import time_finders as TF  # noqa: E402
from run_finders import LABEL_OF, PHRASES, PLURAL  # noqa: E402
from segbench.tasks import PHRASES_BATCH_2, PLURAL_BATCH_2, TASKS_BATCH_2  # noqa: E402

BOX_SETUPS = ["locateanything-3b", "rexomni-3b", "florence2-large-grounding", "moondream-3.1", "groundingpi-4b"]
BOX_TASK = {"locateanything-3b": "box", "rexomni-3b": "box", "florence2-large-grounding": "box",
            "moondream-3.1": "box", "groundingpi-4b": "box",
            # Added 2026-10-23: Falcon's own detection mode (native boxes, no masks).
            # Not in BOX_SETUPS: run by name (--setups falcon-perception-hd).
            "falcon-perception-hd": "box"}
ALL_PHRASES = {**PHRASES, **{k: [v] for k, v in PHRASES_BATCH_2.items()}}
ALL_PLURAL = {**PLURAL, **PLURAL_BATCH_2}
SECOND = [t.key for t in TASKS_BATCH_2]


def queries(key):
    plural = key in TF.PLURAL_MODELS
    return [(t, q, ALL_PLURAL[q] if plural else q) for t in TF.POINT_TASKS + tuple(SECOND) for q in ALL_PHRASES[t]]


def load_queries(key):
    return [x for x in queries(key) if x[0] in TF.POINT_TASKS]


def make_record(key, track, tkey, rnd, parts, factor, shas, sizes, others, fmodel, variant, ftask):
    task = TF.TASKS[tkey]
    W, H = sizes[tkey]
    objects, limit = [], False
    for q, rec, _ in parts:
        if (rec["width"], rec["height"]) != (W, H):
            raise SystemExit(f"{key} {tkey}: finders saw {rec['width']}x{rec['height']}, our photo is {W}x{H}")
        limit = limit or bool(rec.get("hit_object_limit"))
        for o in rec["objects"]:
            if o.get("box") is None:
                continue
            x1, y1, x2, y2 = o["box"]
            row = {"box": [round(x1 * 1000 / W, 2), round(y1 * 1000 / H, 2), round(x2 * 1000 / W, 2), round(y2 * 1000 / H, 2)]}
            if task.labels:
                row["label"] = LABEL_OF.get(q)
            objects.append(row)
    seconds = sum(w for _, _, w in parts)
    r = {"model": key, "level": None, "track": "find", "task": tkey, "task_number": task.number, "repeat": rnd,
         "prompt_version": "finders-box-v1", "prompt": " | ".join(rec.get("prompt_sent") or "" for _, rec, _ in parts),
         "image": task.photo, "image_sha256": shas[tkey], "provider": "local-gpu", "route": "local",
         "seconds": round(seconds, 4), "cost_usd": TF.PRICE / 3600 * seconds * factor,
         "cost_method": f"${TF.PRICE}/GPU-hour x single-copy seconds x load factor {factor:.3f} (this run, box task, "
                        f"{TF.TIMING_BASIS.split(':')[0]})",
         "text": json.dumps({"objects": objects}), "answer": {"items": objects}, "hit_object_limit": limit,
         "finders": {"model": fmodel, "variant": variant, "task": ftask, "weights": parts[0][1].get("weights"),
                     "settings": parts[0][1].get("settings"), "queries": [
                         {"query": q, "prompt_sent": rec.get("prompt_sent"), "raw_output": rec.get("raw_output"),
                          "model_seconds": rec["seconds"]} for q, rec, _ in parts]},
         "gpu_others": others, "timing_trustworthy": not others, "timing_basis": TF.TIMING_BASIS, "gpu_uuid": TF.GPU_UUID}
    from datetime import datetime, timezone
    r["started_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--setups", nargs="*", default=BOX_SETUPS)
    ap.add_argument("--basis", required=True, help="what was stopped to free the GPU (kept in every record)")
    ap.add_argument("--gpu", choices=list(TF.GPUS), default="0", help="which whole GPU to time on (default 0, as before)")
    a = ap.parse_args()
    TF.GPU_UUID, TF.TIMING_BASIS = TF.GPUS[a.gpu][0], a.basis
    out = ROOT / "results" / f"run-{a.name}.jsonl"
    if out.exists():
        sys.exit(f"{out} exists")
    # time_finders, pointed at boxes: its single-copy pass asks every query of queries(),
    # its load test cycles the queries given to job() for the load windows.
    orig_setups = dict(TF.SETUPS)
    TF.SETUPS = {k: (v[0], BOX_TASK[k], v[2], v[3], v[4]) for k, v in orig_setups.items() if k in BOX_TASK}
    TF.make_record = make_record  # looked up by name inside TF._measure
    TF.ALL_TASKS = tuple(TF.TASKS[t] for t in TF.POINT_TASKS + tuple(SECOND))  # photos' hashes and sizes
    timing = []
    for key in a.setups:
        timing.append(_run(key, out))
        (ROOT / "results" / f"run-{a.name}.timing.json").write_text(json.dumps(timing, indent=1))
    print("saved", out)


def _run(key, out):
    """TF.measure without trick questions."""
    fmodel, ftask, variant, copies, rounds = TF.SETUPS[key]
    qs = queries(key)
    # Warm-ups on the first four photos (the load test's), one per photo for warmup_photos models.
    jobs = TF.warmups(key, ftask, load_queries(key)) + [
        TF.job(key, ftask, t, q, s, r, "single") for r in range(rounds) for t, q, s in qs]
    others_before = TF.gpu_others()
    print(f"== {key} ({ftask}): {sum(j['id'].startswith('single') for j in jobs)} timed queries", flush=True)
    import time
    t0 = time.monotonic()
    service = TF.Service(fmodel, max(copies, default=1), out.with_name(f"{out.stem}.{key}.service.log")) \
        if TF.LOCAL[key].service else None
    try:
        # TF._measure uses its `qs` only for the load test: the first four tasks
        return TF._measure(key, out, fmodel, ftask, variant, copies, load_queries(key), jobs, [], others_before, t0,
                           service)
    finally:
        if service is not None:
            service.stop()


if __name__ == "__main__":
    main()
