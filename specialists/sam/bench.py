"""SAM 3, and SAM 1 / SAM 2.1 paired with a finder, on the six benchmark photos.

uv run bench.py sam3 --name specialists-1
uv run bench.py pairs --name specialists-1 --finders ../../results/run-pilot-1.jsonl

sam3:  SAM 3 with a short phrase per task (answers.PHRASES); one call gives
       the masks for both the find track (a dot per mask) and the whole track
       (outlines). Trick questions ask for the absent thing.
pairs: whole track as "finder + outliner": each language model's find-track
       dots from a past run, given one by one to SAM 1 and SAM 2.1 as a
       click; SAM returns 3 candidate masks per click and the one it rates
       best is kept (the standard use of a single click). The finder's labels
       (dirty / clean) carry over. Cost and time = the finder's call + SAM's.
       SAM 1 and 2.1 cannot read a request, so they only enter as pairs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import harness  # noqa: E402,F401  (pins the GPU before torch loads)
import answers as A  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(A.ROOT))
from segbench import parse  # noqa: E402
from segbench.tasks import BY_KEY as TASKS, TASKS as ALL_TASKS  # noqa: E402

WEIGHTS = A.ROOT / "weights"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ---------- SAM 3, text ----------


def sam3(writer: A.Writer) -> None:
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    proc = Sam3Processor(build_sam3_image_model(), confidence_threshold=0.5)
    factor = A.load_factor("sam3-text")

    def ask(phrases):
        def fn(im):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                st = proc.set_image(im)
                out = []
                for phrase in phrases:
                    proc.reset_all_prompts(st)
                    st = proc.set_text_prompt(phrase, st)
                    out.append((phrase, st["masks"].cpu().numpy().reshape(-1, im.height, im.width),
                                st["scores"].float().cpu().numpy()))
            return out
        return fn

    data0, _ = A.photo(ALL_TASKS[0].photo)
    A.timed(ask(["cow"]), data0, repeats=2)  # warm-up, not recorded
    for task in ALL_TASKS:
        data, sha = A.photo(task.photo)
        result, walls, models = A.timed(ask(A.PHRASES[task.key]), data)
        masks, labels, scores = [], [], []
        for phrase, ms, ss in result:
            masks += list(ms)
            labels += [A.LABEL_OF_PHRASE.get(phrase)] * len(ms)
            scores += ss.tolist()
        if task.labels:
            masks, labels, scores = A.dedupe(masks, labels, scores)
        ans = A.to_answers(task, masks, labels, scores)
        prompt = " | ".join(A.PHRASES[task.key])
        for track in ("find", "whole"):
            if track == "find" and task.kind == "region":
                continue
            writer.write(A.record(model="sam3", track=track, task=task, text=ans[track], walls=walls, models=models,
                                  factor=factor, prompt=prompt, sha=sha,
                                  extra={"config": {"confidence_threshold": 0.5, "precision": "bf16 autocast"}}),
                         ans["masks"] if track == "whole" else None)
        print(f"sam3   {task.key:7} {len(masks):3} masks  {np.median(walls):.2f}s", flush=True)
        phrase = A.TRICK_PHRASES[task.key]
        result, walls, models = A.timed(ask([phrase]), data)
        _, ms, ss = result[0]
        ans = A.to_answers(task, list(ms), [None] * len(ms), ss.tolist())
        writer.write(A.record(model="sam3", track="trick", task=task, text=ans["find"], walls=walls, models=models,
                              factor=factor, prompt=phrase, sha=sha), ans["masks"])
        print(f"sam3   {task.key:7} trick '{phrase}': {len(ms)}", flush=True)


# ---------- SAM 1 / SAM 2.1 with a finder's dots ----------


def predictors():
    from segment_anything import SamPredictor, sam_model_registry
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    sam1 = SamPredictor(sam_model_registry["vit_h"](checkpoint=str(WEIGHTS / "sam1" / "sam_vit_h_4b8939.pth")).cuda().eval())
    sam2 = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large")

    def run1(im, points):
        with torch.inference_mode():
            sam1.set_image(np.asarray(im))
            out = []
            for chunk in range(0, len(points), 8):
                pts = torch.tensor(points[chunk:chunk + 8], dtype=torch.float, device="cuda")[:, None, :]
                pts = sam1.transform.apply_coords_torch(pts, (im.height, im.width))
                masks, iou, _ = sam1.predict_torch(pts, torch.ones(pts.shape[:2], device="cuda"), multimask_output=True)
                best = iou.argmax(1)
                out += [(m[b].cpu().numpy(), float(s[b])) for m, s, b in zip(masks, iou, best)]
            return out

    def run2(im, points):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            sam2.set_image(np.asarray(im))
            out = []
            for chunk in range(0, len(points), 8):
                pts = np.array(points[chunk:chunk + 8], dtype=np.float32)[:, None, :]
                masks, iou, _ = sam2.predict(point_coords=pts, point_labels=np.ones(pts.shape[:2]), multimask_output=True)
                masks = masks.reshape(len(pts), 3, im.height, im.width)
                iou = iou.reshape(len(pts), 3)
                best = iou.argmax(1)
                out += [(masks[i, b] > 0, float(iou[i, b])) for i, b in enumerate(best)]
            return out

    return {"sam1-vit-h": (run1, A.load_factor("sam1-vit-h-box-fp32"), "fp32 (official)"),
            "sam2.1-large": (run2, A.load_factor("sam2.1-large-box"), "bf16 autocast (official examples)")}


def pairs(writer: A.Writer, finders_path: Path) -> None:
    finds = [json.loads(line) for line in finders_path.read_text().splitlines()]
    finds = [r for r in finds if r["track"] == "find"]
    outliners = predictors()
    data0, _ = A.photo(ALL_TASKS[0].photo)
    for run, _, _ in outliners.values():  # warm-up
        A.timed(lambda im: run(im, [[100, 100]]), data0, repeats=1)
    done = set()
    if writer.path.exists():  # resume: skip pairs already written
        done = {(r["model"], r["task"]) for r in map(json.loads, writer.path.read_text().splitlines()) if r.get("parts")}
    for f in finds:
        task = TASKS[f["task"]]
        if all((f"{f['model']}+{k}", task.key) in done for k in outliners):
            continue
        data, sha = A.photo(task.photo)
        base = {"finder_record": {k: f.get(k) for k in ("model", "level", "started_at", "seconds", "cost_usd")}}
        for key, (run, factor, precision) in outliners.items():
            parts = [{"model": f["model"], "level": f["level"]}, {"model": key}]
            name = f"{f['model']}+{key}"
            if (name, task.key) in done:
                continue
            try:
                items = parse.read("find", task.kind, f["text"], task.labels)["items"] if "error" not in f else None
            except Exception as error:
                items, why = None, f"finder reply unreadable ({type(error).__name__})"
            else:
                why = f"finder error: {f.get('error', '')[:200]}" if items is None else ""
            if items is None:
                writer.write({"model": name, "parts": parts, "level": f["level"], "track": "whole", "task": task.key,
                              "task_number": task.number, "repeat": 0, "prompt_version": A.PROMPT_VERSION,
                              "provider": "pair", "image": task.photo, "image_sha256": sha,
                              "seconds": f.get("seconds"), "cost_usd": f.get("cost_usd"), "error": why, **base})
                continue
            with Image.open(A.ROOT / "images" / task.photo) as im:
                W, H = im.size
            inside = [it for it in items if 0 <= it["point"][0] <= 1000 and 0 <= it["point"][1] <= 1000]
            points = [[it["point"][0] * W / 1000, it["point"][1] * H / 1000] for it in inside]
            if points:
                result, walls, models = A.timed(lambda im: run(im, points), data)
            else:
                result, walls, models = [], [0.0], [0.0]
            masks = [m for m, _ in result]
            scores = [s for _, s in result]
            labels = [it.get("label") for it in inside]
            ans = A.to_answers(task, masks, labels, scores) if masks else {"whole": json.dumps({"objects": []}), "masks": []}
            rec = A.record(model=name, track="whole", task=task, text=ans["whole"], walls=walls, models=models,
                           factor=factor, prompt=f"{len(points)} clicks from {f['model']} ({f['level']})", sha=sha,
                           extra={"parts": parts, "level": f["level"], "provider": "pair",
                                  "config": {"outliner_precision": precision, "multimask": "best of 3 by predicted IoU"},
                                  "outliner_seconds": None, "outliner_cost_usd": None, **base})
            rec["outliner_seconds"], rec["outliner_cost_usd"] = rec["seconds"], rec["cost_usd"]
            rec["seconds"] = round((f.get("seconds") or 0) + rec["seconds"], 4)
            rec["cost_usd"] = (f.get("cost_usd") or 0) + rec["cost_usd"]
            writer.write(rec, ans["masks"])
            print(f"{name:28} {task.key:7} {len(points):3} clicks  SAM {rec['outliner_seconds']:.2f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sam3", "pairs"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--finders", type=Path)
    args = ap.parse_args()
    writer = A.Writer(args.name)
    sam3(writer) if args.mode == "sam3" else pairs(writer, args.finders)
