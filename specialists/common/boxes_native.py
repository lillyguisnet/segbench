"""SAM 3, YOLOE-26x and RF-DETR Seg asked for boxes (box benchmark, 2026-10-15).

Run inside each model's own environment, on the whole of GPU 0:
    cd specialists/sam    && uv run ../common/boxes_native.py sam3   --name specialists-boxes-1
    cd specialists/yoloe  && uv run ../common/boxes_native.py yoloe  --name specialists-boxes-1
    cd specialists/rfdetr && uv run ../common/boxes_native.py rfdetr --name specialists-boxes-1

The same call as for points (same phrases, categories, thresholds and
precision: specialists/sam/bench.py, specialists/yoloe/bench.py,
specialists/rfdetr/points.py), but each object is answered with the box the
model itself gives (its detection box, in photo pixels -> the 0..1000
grid), not a dot. Objects are those the point answers kept: detections
with an empty mask are dropped, and for SAM 3's dishes the higher-scoring
of two masks covering the same dish is kept (answers.dedupe), as before.
All eleven object tasks; no trick questions. Seconds and cost: the same
measurement as for points (answers.timed, median of 3, load factors of the
2026-10-08 throughput test); the chart uses the first four tasks only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402  (pins the GPU before torch loads)
import answers as A  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from segbench.tasks import BY_KEY  # noqa: E402

TASKS = ["logs", "cows", "fig", "dishes", "ants", "beetles", "birds", "mushrooms", "jars", "pinkleaves", "whitecow"]
RFDETR_CATEGORIES = {"logs": [], "cows": ["cow"], "fig": [], "dishes": ["bowl", "cup"], "ants": [], "beetles": [],
                     "birds": ["bird"], "mushrooms": [], "jars": [], "pinkleaves": [], "whitecow": ["cow"]}


def sam3():
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    proc = Sam3Processor(build_sam3_image_model(), confidence_threshold=0.5)

    def ask(task):
        phrases = A.PHRASES[task.key]

        def fn(im):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                st = proc.set_image(im)
                out = []
                for phrase in phrases:
                    proc.reset_all_prompts(st)
                    st = proc.set_text_prompt(phrase, st)
                    out.append((phrase, st["masks"].cpu().numpy().reshape(-1, im.height, im.width),
                                st["boxes"].float().cpu().numpy().reshape(-1, 4), st["scores"].float().cpu().numpy()))
                return out
        return fn, " | ".join(phrases)

    def objects(task, result):
        masks, labels, scores = [], [], []
        for phrase, ms, bs, ss in result:
            masks += list(ms)
            labels += [(A.LABEL_OF_PHRASE.get(phrase), b.tolist()) for b in bs]
            scores += ss.tolist()
        return masks, labels, scores, bool(task.labels)

    return ask, objects, A.load_factor("sam3-text"), "sam3", {"confidence_threshold": 0.5, "precision": "bf16 autocast",
                                                               "box": "SAM 3's own detection box"}


def yoloe():
    from ultralytics import YOLOE

    model = YOLOE("yoloe-26x-seg.pt")

    def ask(task):
        phrases = A.PHRASES[task.key]
        model.set_classes(phrases)

        def fn(im):
            r = model.predict(im, half=True, retina_masks=True, imgsz=640, conf=0.25, agnostic_nms=len(phrases) > 1,
                              verbose=False)[0]
            if r.masks is None:
                return [], [], [], []
            return (list(r.masks.data.cpu().numpy() > 0.5), [r.names[int(c)] for c in r.boxes.cls.tolist()],
                    r.boxes.xyxy.cpu().numpy().tolist(), r.boxes.conf.cpu().numpy().tolist())
        return fn, " | ".join(phrases)

    def objects(task, result):
        masks, phrases, boxes, scores = result
        return masks, [(A.LABEL_OF_PHRASE.get(p), b) for p, b in zip(phrases, boxes)], scores, False

    cfg = {"weights": "yoloe-26x-seg.pt", "imgsz": 640, "conf": 0.25, "precision": "fp16", "retina_masks": True,
           "box": "YOLOE's own detection box"}
    return ask, objects, A.load_factor("yoloe-26x-text"), "yoloe-26x", cfg


def rfdetr():
    os.chdir(harness.ROOT / "weights" / "rfdetr")
    from rfdetr import RFDETRSeg2XLarge
    from rfdetr.assets.coco_classes import COCO_CLASSES

    model = RFDETRSeg2XLarge()
    model.inference(compile=True, batch_size=1, dtype=torch.float16)

    def ask(task):
        cats = RFDETR_CATEGORIES[task.key]

        def fn(im):
            d = model.predict(im, threshold=0.5)
            d = d[0] if isinstance(d, list) else d
            if d.mask is None:
                return [], [], []
            keep = [i for i, c in enumerate(d.class_id) if COCO_CLASSES[int(c)] in cats]
            return ([np.asarray(d.mask[i]) for i in keep], [np.asarray(d.xyxy[i]).tolist() for i in keep],
                    [float(d.confidence[i]) for i in keep])
        return fn, ("COCO categories: " + ", ".join(cats)) if cats else "none of its 80 COCO categories fit: no boxes"

    def objects(task, result):
        masks, boxes, scores = result
        return masks, [(None, b) for b in boxes], scores, False

    cfg = {"weights": "RFDETRSeg2XLarge (COCO)", "threshold": 0.5, "precision": "fp16, traced",
           "box": "RF-DETR's own detection box"}
    return ask, objects, A.load_factor("rfdetr-seg-2xl"), "rfdetr-seg-2xl", cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=("sam3", "yoloe", "rfdetr"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--basis", required=True, help="what was stopped to free GPU 0")
    a = ap.parse_args()
    ask, objects, factor, key, cfg = {"sam3": sam3, "yoloe": yoloe, "rfdetr": rfdetr}[a.model]()
    out = harness.ROOT / "results" / f"run-{a.name}.jsonl"
    if any(json.loads(line)["model"] == key for line in (out.read_text().splitlines() if out.exists() else [])):
        sys.exit(f"{key} already in {out}")
    warm = BY_KEY["cows"]
    A.timed(ask(warm)[0], A.photo(warm.photo)[0], repeats=2)  # warm-up, not recorded
    with out.open("a") as f:
        for tkey in TASKS:
            task = BY_KEY[tkey]
            data, sha = A.photo(task.photo)
            fn, prompt = ask(task)
            result, walls, models = A.timed(fn, data)
            masks, labels, scores, dedupe = objects(task, result)
            keep = [i for i, m in enumerate(masks) if np.asarray(m).any()]  # as answers.to_answers
            masks, labels, scores = [masks[i] for i in keep], [labels[i] for i in keep], [scores[i] for i in keep]
            if dedupe:
                masks, labels, scores = A.dedupe(masks, labels, scores)
            W, H = A.decode(data).size
            items = []
            for label, (x1, y1, x2, y2) in labels:
                item = {"box": [round(x1 * 1000 / W, 2), round(y1 * 1000 / H, 2), round(x2 * 1000 / W, 2),
                                round(y2 * 1000 / H, 2)]}
                if task.labels:
                    item["label"] = label
                items.append(item)
            rec = A.record(model=key, track="find", task=task, text=json.dumps({"objects": items}), walls=walls,
                           models=models, factor=factor, prompt=prompt, sha=sha,
                           extra={"config": cfg, "timing_basis": a.basis})
            rec.update(prompt_version="spec-box-v1", answer={"items": items}, scores=[round(s, 4) for s in scores])
            f.write(json.dumps(rec) + "\n")
            print(f"{key:15} {tkey:10} {len(items):3} boxes  {rec['seconds']:.2f}s", flush=True)
    print("saved", out)


if __name__ == "__main__":
    main()
