# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.3.3", "scipy==1.16.2", "pyarrow==21.0.0", "pillow==11.3.0"]
# ///
"""Prepare, run and score PixMo-Points-Eval without changing photo baselines.

uv run scripts/pixmo_eval.py prepare
uv run scripts/pixmo_eval.py plan
uv run scripts/pixmo_eval.py run --model molmopoint-8b --gpu 0 --out results/pixmo/molmo-pilot --limit 20
uv run scripts/pixmo_eval.py score --run results/pixmo/molmo-pilot
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from http.client import HTTPException
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from segbench.local_models import BY_KEY
from segbench.pixmo import DATA_REVISION, DATASET, EVAL_REVISION, METRIC, score_points

DEFAULT_DATA = ROOT / "cache/pixmo-points-eval"
DEFAULT_FINDERS = Path.home() / "Projects/finders"
SOURCE_HASHES = {
    "test.parquet": "5a7bf84b8b06ccde4deff7857923686cae3540ecc39212183c010b3a9275c3e0",
    "pixmo_datasets.py": "d74d4e17913a5d5cb06cff357f84d15a26b8fa2772e2d878160128e203dcbb8d",
    "evaluators.py": "db774284f70f6e501c5d487b146dc2d92ba7df58f36c56b58d5a56fa57a5e081",
}
MODEL_KEYS = tuple(k for k, m in BY_KEY.items() if m.finders and k != "florence2-large")
DEFAULT_MODELS = ("sam3", "moondream-3.1", "florence2-large-grounding", "rexomni-3b",
                  "groundingpi-4b", "locateanything-3b", "molmopoint-8b")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def get_bytes(url: str, timeout=25) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "segbench-research/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = response.read(64 * 1024 * 1024 + 1)
    if len(data) > 64 * 1024 * 1024:
        raise ValueError("download exceeds 64 MiB limit")
    return data


def cached_source(url: str, path: Path) -> None:
    if not path.exists():
        data = get_bytes(url, timeout=60)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_bytes(data)
        temp.replace(path)
    if digest(path.read_bytes()) != SOURCE_HASHES[path.name]:
        raise ValueError(f"pinned source checksum mismatch: {path}")


def prepare(args) -> None:
    base = args.data.resolve()
    source = base / "source/test.parquet"
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{DATA_REVISION}/data/test-00000-of-00001.parquet"
    cached_source(url, source)
    for file in ("olmo/data/pixmo_datasets.py", "olmo/eval/evaluators.py"):
        cached_source(f"https://raw.githubusercontent.com/allenai/molmo2/{EVAL_REVISION}/{file}",
                      base / "source" / Path(file).name)
    parquet = pq.ParquetFile(source)
    if parquet.metadata.num_rows != 436:
        raise ValueError(f"expected pinned 436-question test split, found {parquet.metadata.num_rows}")
    # One nested-mask row at a time: converting the whole file to Python lists
    # at once needs several GB and is much slower than streaming Arrow batches.
    rows = (batch.to_pylist()[0] for batch in parquet.iter_batches(batch_size=1))
    (base / "masks").mkdir(parents=True, exist_ok=True)
    (base / "images").mkdir(exist_ok=True)
    examples, image_jobs = [], {}
    for i, row in enumerate(rows):
        ident = f"pixmo-test-{i:04d}"
        masks = np.asarray(row["masks"], dtype=bool)
        if masks.ndim != 3 or (row["points"] and masks.shape[0] != len(row["points"])):
            raise ValueError(f"invalid reference masks: {ident}")
        h, w = masks.shape[1:]
        mask_file = base / "masks" / f"{ident}.npz"
        np.savez_compressed(mask_file, masks=masks)
        sha = row["image_sha256"]
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("invalid image hash")
        image_jobs[(row["image_url"], sha)] = base / "images" / f"{sha}.image"
        examples.append({"id": ident, "row": i, "label": row["label"],
                         "image_url": row["image_url"], "image_sha256": sha,
                         "image": f"images/{sha}.image", "masks": f"masks/{ident}.npz",
                         "mask_width": w, "mask_height": h, "masks_sha256": digest(mask_file.read_bytes()),
                         "reference_percent": [[p["x"], p["y"]] for p in row["points"]]})

    def download(item):
        (image_url, sha), path = item
        error = None
        for attempt in range(args.retries + 1):
            try:
                raw = path.read_bytes() if path.exists() else get_bytes(image_url)
                if digest(raw) != sha:
                    raise ValueError("SHA256 mismatch: bytes differ from annotated original")
                with Image.open(io.BytesIO(raw)) as im:
                    im.load()
                    original = im.size
                    upright = ImageOps.exif_transpose(im).size
                    orientation = im.getexif().get(274, 1)
                if not path.exists():
                    path.write_bytes(raw)
                return (image_url, sha), {"status": "verified", "width": upright[0], "height": upright[1],
                                          "original_size": original, "exif_orientation": orientation, "bytes": len(raw)}
            except (OSError, ValueError, HTTPException, Image.DecompressionBombError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                if "SHA256 mismatch" in error:
                    break  # never accept a changed image as the original
                if attempt < args.retries:
                    time.sleep(attempt + 1)
        return (image_url, sha), {"status": "unavailable", "error": error}

    states = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download, item) for item in image_jobs.items()]
        for i, future in enumerate(as_completed(futures), 1):
            key, state = future.result()
            states[key] = state
            if i % 25 == 0 or i == len(futures):
                print(f"images {i}/{len(futures)}; verified {sum(s['status'] == 'verified' for s in states.values())}", flush=True)
    for ex in examples:
        ex.update(states[(ex["image_url"], ex["image_sha256"])])
        # Finders transposes EXIF. Masks have no transformation metadata. Don't
        # silently use an image in a different orientation than the answer key.
        if ex["status"] == "verified" and ex["exif_orientation"] != 1:
            ex.update(status="unavailable", error="EXIF-transformed image requires manual mask-orientation check")
    manifest = base / "examples.jsonl"
    manifest.write_text("".join(json.dumps(ex) + "\n" for ex in examples))
    good = [e for e in examples if e["status"] == "verified"]
    provenance = {"dataset": DATASET, "revision": DATA_REVISION, "split": "test", "metric": METRIC,
                  "evaluator_revision": EVAL_REVISION, "prepared_at": datetime.now(UTC).isoformat(),
                  "source_sha256": digest(source.read_bytes()), "manifest_sha256": digest(manifest.read_bytes()),
                  "total_questions": len(examples), "total_unique_urls": len(image_jobs),
                  "total_objects": sum(len(e["reference_percent"]) for e in examples),
                  "verified_questions": len(good), "verified_images": len({e["image_sha256"] for e in good}),
                  "downloaded_bytes": sum(s.get("bytes", 0) for s in states.values()),
                  "unavailable": [{"id": e["id"], "url": e["image_url"], "error": e["error"]}
                                  for e in examples if e["status"] != "verified"],
                  "reference_coordinates": "0..100 percentage, NOT pixels (per authors' loader)"}
    save_json(base / "provenance.json", provenance)
    print(json.dumps({k: v for k, v in provenance.items() if k != "unavailable"}, indent=2))
    print(f"Missing questions: {len(examples) - len(good)}. Details: {base / 'provenance.json'}")


def selected_examples(args):
    examples = read_jsonl(args.data / "examples.jsonl")
    eligible = [e for e in examples if e["status"] == "verified"]
    # Deterministic random ordering, never chosen by label, answer count or model score.
    random.Random(args.seed).shuffle(eligible)
    return examples, eligible[:args.limit] if args.limit is not None else eligible


def plan(args):
    examples, selected = selected_examples(args)
    print(f"Verified {sum(e['status'] == 'verified' for e in examples)}/{len(examples)} questions; selected {len(selected)}")
    prior = {r["entrant"]: float(r["seconds"]) for r in csv.DictReader(
        (ROOT / "results/point-benchmark-v3/summary.csv").open())}
    total = 0
    for key in DEFAULT_MODELS:
        seconds = len(selected) * prior[key]
        total += seconds
        ready = (args.finders / "models" / BY_KEY[key].finders / ".venv").is_dir()
        print(f"{key:28} {seconds / 60:7.1f} min   {'ready' if ready else 'environment missing'}")
    print(f"Total: {total / 3600:.2f} GPU-hours, one pass, models sequentially.")
    print("Estimate from our eleven-photo medians, not measured on PixMo; excludes startup and CPU scoring.")
    print("Fewer targets per question may shorten generation. Busy GPUs can greatly increase times.")


def gpu_state(gpu: str) -> dict:
    cards = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free,utilization.gpu",
                                     "--format=csv,noheader,nounits"], text=True)
    card = next((line.split(", ") for line in cards.splitlines()
                 if gpu in line.split(", ")[:2]), None)
    if card is None:
        raise ValueError(f"GPU {gpu!r} not found")
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                                    "--format=csv,noheader,nounits"], text=True)
    return {"index": card[0], "uuid": card[1], "free_mib": int(card[2]), "utilization": int(card[3]),
            "processes": [s for s in apps.splitlines() if s.startswith(card[1])]}


def run(args):
    base, dest = args.data.resolve(), args.out.resolve()
    examples, selected = selected_examples(args)
    if args.limit is None and len(selected) < len(examples) and not args.allow_partial:
        raise ValueError(f"only {len(selected)}/{len(examples)} questions verified; "
                         "use --allow-partial to explicitly evaluate the available subset, "
                         "or --limit for a pilot")
    if not selected:
        raise ValueError("no verified examples selected")
    model = BY_KEY[args.model]
    runner = args.finders.resolve() / "models" / model.finders
    if not (runner / ".venv").is_dir():
        raise ValueError(f"missing model environment: {runner}")
    config = {"dataset": DATASET, "revision": DATA_REVISION, "metric": METRIC,
              "manifest_sha256": digest((base / "examples.jsonl").read_bytes()),
              "data": str(base), "model": args.model, "task": model.task, "variant": model.variant,
              "seed": args.seed, "selected_ids": [e["id"] for e in selected],
              "wording": "label verbatim in finders native task template; no answer-key hints"}
    if dest.exists() and not args.resume:
        raise ValueError("output directory exists: use --resume or a new --out")
    if args.resume and json.loads((dest / "config.json").read_text()) != config:
        raise ValueError("run configuration changed; use a new output directory")
    done = read_jsonl(dest / "predictions.jsonl") if (dest / "predictions.jsonl").exists() else []
    ids = [r["id"] for r in done]
    if len(ids) != len(set(ids)) or set(ids) - set(config["selected_ids"]):
        raise ValueError("duplicate or unexpected saved prediction IDs")
    pending = [e for e in selected if e["id"] not in set(ids)]
    if not pending:
        print("All selected predictions already exist; rescoring without GPU work.")
        return score_run(argparse.Namespace(run=dest))
    gpu = gpu_state(args.gpu)
    if (gpu["processes"] or gpu["utilization"] > 5) and not args.allow_shared_gpu:
        raise ValueError(f"GPU is busy; won't disturb it: {gpu}. Wait, or explicitly use --allow-shared-gpu.")
    # Hash every selected input again. Never run on a corrupt/stale cache file.
    for ex in pending:
        if digest((base / ex["image"]).read_bytes()) != ex["image_sha256"]:
            raise ValueError(f"cached image changed: {ex['id']}")
    dest.mkdir(parents=True, exist_ok=True)
    save_json(dest / "config.json", config)
    attempt = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    save_json(dest / f"gpu-{attempt}.json", gpu)
    jobs = dest / f"jobs-{attempt}.jsonl"
    jobs.write_text("".join(json.dumps({"id": e["id"], "image": str(base / e["image"]),
                                       "query": e["label"], "task": model.task, "seed": args.seed}) + "\n"
                            for e in pending))
    cmd = ["uv", "run", "--frozen", "--quiet", "--directory", str(runner), "find.py",
           "--jobs", str(jobs), "--no-masks"]
    if model.variant:
        cmd += ["--variant", model.variant]
    env = dict(os.environ, FINDERS_GPU=gpu["uuid"])
    env.pop("VIRTUAL_ENV", None)
    env.pop("CUDA_VISIBLE_DEVICES", None)  # the explicit --gpu controls this run
    start = time.monotonic()
    expected = [e["id"] for e in pending]
    received = 0
    with (dest / f"model-{attempt}.log").open("w") as log, (dest / "predictions.jsonl").open("a") as output:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log, text=True, env=env)
        try:
            for line in proc.stdout:
                rec = json.loads(line)
                if received >= len(expected) or rec.get("id") != expected[received]:
                    raise ValueError("model returned an unexpected or out-of-order ID")
                output.write(json.dumps(rec, allow_nan=False) + "\n")
                output.flush()
                received += 1
                print(f"{args.model} {len(done) + received}/{len(selected)}: {rec['seconds']:.2f}s", flush=True)
            code = proc.wait()
            if code or received != len(expected):
                raise RuntimeError(f"model stopped: exit {code}, saved {received}/{len(expected)}; see {log.name}. Resume explicitly.")
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
    save_json(dest / f"elapsed-{attempt}.json", {"wall_seconds": time.monotonic() - start,
                                               "completed": received, "gpu_after": gpu_state(args.gpu)})
    score_run(argparse.Namespace(run=dest))


def score_run(args):
    dest = args.run.resolve()
    config = json.loads((dest / "config.json").read_text())
    base = Path(config["data"])
    if digest((base / "examples.jsonl").read_bytes()) != config["manifest_sha256"]:
        raise ValueError("dataset manifest changed since the run")
    examples = {e["id"]: e for e in read_jsonl(base / "examples.jsonl")}
    predictions = read_jsonl(dest / "predictions.jsonl")
    seen, scores = set(), []
    for rec in predictions:
        ident = rec["id"]
        if ident in seen or ident not in config["selected_ids"]:
            raise ValueError("duplicate or unexpected prediction")
        seen.add(ident)
        ex = examples[ident]
        w, h = rec["width"], rec["height"]
        if (w, h) != (ex["width"], ex["height"]):
            raise ValueError(f"image orientation/size mismatch: {ident}")
        points = []
        for obj in rec["objects"]:
            point = obj.get("point")
            if point is None:
                raise ValueError(f"finder object has no point: {ident}")
            if len(point) != 2 or not np.isfinite(point).all():
                raise ValueError(f"malformed point: {ident}")
            points.append([point[0] * ex["mask_width"] / w, point[1] * ex["mask_height"] / h])
        mask_path = base / ex["masks"]
        if digest(mask_path.read_bytes()) != ex["masks_sha256"]:
            raise ValueError(f"reference mask checksum mismatch: {ident}")
        with np.load(mask_path) as masks:
            metrics = score_points(points, ex["reference_percent"], masks["masks"])
        scores.append(dict(id=ident, label=ex["label"], **metrics, seconds=rec["seconds"],
                           point_from=sorted({o.get("point_from") or "unspecified" for o in rec["objects"]})))
    summary = {"model": config["model"], "metric": METRIC, "dataset_revision": DATA_REVISION,
               "total_questions": len(examples), "verified_questions": sum(e["status"] == "verified" for e in examples.values()),
               "selected_questions": len(config["selected_ids"]), "scored_questions": len(scores),
               "complete_selected_run": len(seen) == len(config["selected_ids"]),
               "full_test_set": len(seen) == len(examples),
               "missing_prediction_ids": [i for i in config["selected_ids"] if i not in seen],
               "note": "Partial/available-only runs are NOT full published-benchmark reproductions."}
    if scores:
        for key in ("precision", "recall", "f1"):
            summary[key] = float(np.mean([s[key] for s in scores]))
        summary["warm_seconds_total"] = sum(s["seconds"] for s in scores)
        summary["warm_seconds_median"] = float(np.median([s["seconds"] for s in scores]))
    save_json(dest / "summary.json", summary)
    (dest / "per-example.jsonl").write_text("".join(json.dumps(s) + "\n" for s in scores))
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="download annotations and hash-verified original images; no GPU")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--retries", type=int, default=1)
    for name in ("plan", "run"):
        p = sub.add_parser(name)
        p.add_argument("--data", type=Path, default=DEFAULT_DATA)
        p.add_argument("--finders", type=Path, default=DEFAULT_FINDERS)
        p.add_argument("--limit", type=int, help="fixed random subset size; omitted: all verified questions")
        p.add_argument("--seed", type=int, default=0)
        if name == "run":
            p.add_argument("--model", choices=MODEL_KEYS, required=True)
            p.add_argument("--gpu", required=True, help="GPU index or UUID; won't run on a busy GPU by default")
            p.add_argument("--out", type=Path, required=True)
            p.add_argument("--resume", action="store_true")
            p.add_argument("--allow-partial", action="store_true", help="explicitly accept missing dataset images")
            p.add_argument("--allow-shared-gpu", action="store_true")
    p = sub.add_parser("score", help="CPU-only scoring of saved predictions")
    p.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if getattr(args, "limit", None) is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.cmd == "prepare" and (args.workers < 1 or args.retries < 0):
        parser.error("--workers must be positive; --retries must be nonnegative")
    {"prepare": prepare, "plan": plan, "run": run, "score": score_run}[args.cmd](args)


if __name__ == "__main__":
    main()
