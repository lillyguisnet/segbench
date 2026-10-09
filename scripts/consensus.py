# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow>=10", "numpy>=1.26", "lm15>=1.2.1"]
# ///
"""Merge every model's answers into a draft answer key, and weigh each model.

    uv run scripts/view_run.py results/run-pilot-1.jsonl    # first (makes the photo copies)
    uv run scripts/consensus.py results/run-pilot-1.jsonl

Writes:
- results/consensus-<run>.json         the draft key: each candidate object with its
                                       chance of being real, its votes and its label;
                                       the region masks' per-pixel chances (small grid)
- results/views/<run>/consensus.html   to look at it, with the review queue

A draft for a person to check, never the answer key: see segbench/consensus.py
for why, and for the methods. The provisional scores judge each model only
against the consensus of the other makers' models (leave-maker-out).
"""

from __future__ import annotations

import base64
import io
import json
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from segbench import consensus as cs  # noqa: E402
from segbench import parse  # noqa: E402
from segbench.models import BY_KEY as MODELS  # noqa: E402
from segbench.tasks import BY_KEY as TASKS  # noqa: E402

REAL, CONTESTED_LOW, CONTESTED_HIGH = 0.5, 0.2, 0.8
REGION_GRID = 300  # longest side of the grid regions are merged on


def load(run: Path):
    answers: dict[str, dict[str, dict]] = {}  # task -> answer name -> {model, track, items}
    for line in run.read_text().splitlines():
        r = json.loads(line)
        if r["track"] == "trick" or "error" in r:
            continue
        task = TASKS[r["task"]]
        try:
            items = parse.read(r["track"], task.kind, r["text"], task.labels)["items"]
        except Exception:
            continue  # unreadable: it gives no evidence (it still scores 0 in the benchmark)
        with Image.open(ROOT / "images" / task.photo) as im:
            W, H = im.size
        for it in items:  # 0..1000 -> photo pixels
            if "point" in it:
                it["point"] = [it["point"][0] * W / 1000, it["point"][1] * H / 1000]
            if "polygon" in it:
                it["polygon"] = [[x * W / 1000, y * H / 1000] for x, y in it["polygon"]]
        answers.setdefault(r["task"], {})[f"{r['model']}/{r['track']}"] = {
            "model": r["model"], "maker": MODELS[r["model"]].maker, "track": r["track"], "items": items, "size": (W, H)}
    return answers


def dots_of(ans: dict) -> list[list[float]]:
    return [it["point"] if "point" in it else cs.interior_point(it["polygon"]) for it in ans["items"]]


def merge_objects(answers: dict[str, dict], radius: float, labels: tuple[str, ...]):
    names = list(answers)
    dots = [(n, i, *p) for n in names for i, p in enumerate(dots_of(answers[n]))]
    groups = cs.group(dots, radius)
    temper = cs.tempering([answers[n]["maker"] for n in names])
    p, recall, false, precision = cs.em_objects(groups, names, temper=temper)
    q = acc = None
    if labels and groups:
        q, acc = cs.em_labels(groups, p, {n: [it.get("label") for it in answers[n]["items"]] for n in names}, labels,
                              temper=temper)
    return groups, p, recall, false, precision, q, acc


def object_task(task, answers):
    sizes = [2 * np.sqrt(cs.polygon_area(it["polygon"]) / np.pi)
             for a in answers.values() for it in a["items"] if "polygon" in it]
    radius = 0.5 * statistics.median(sizes)
    groups, p, recall, false, precision, q, acc = merge_objects(answers, radius, task.labels)
    objects = []
    for c, g in enumerate(groups):
        o = {"center": [round(v, 1) for v in g["center"]], "p_real": round(float(p[c]), 3),
             "votes": sorted(g["votes"])}
        if q is not None:
            o["p_" + task.labels[0]] = round(float(q[c]), 3)
        objects.append(o)
    # provisional scores, leave-maker-out
    scores = {}
    for name, ans in answers.items():
        others = {n: a for n, a in answers.items() if a["maker"] != ans["maker"]}
        g2, p2, *_, q2, _ = merge_objects(others, radius, task.labels)
        key = [k for k in range(len(g2)) if p2[k] > REAL]
        centers = [g2[k]["center"] for k in key]
        mine = dots_of(ans)
        pairs = cs.match(mine, centers, radius)
        tp = len(pairs)
        s = {"found": len(mine), "key": len(key), "tp": tp}
        s["precision"] = tp / len(mine) if mine else 0.0
        s["recall"] = tp / len(key) if key else 0.0
        s["f1"] = 2 * tp / (len(mine) + len(key)) if mine or key else 1.0
        if task.labels and q2 is not None:
            first = task.labels[0]
            ok = sum((ans["items"][i].get("label") == first) == (q2[key[j]] > 0.5) for i, j in pairs)
            s["f1_labelled"] = 2 * ok / (len(mine) + len(key)) if mine or key else 1.0
        scores[name] = s
    weights = {n: {"recall": round(float(recall[n]), 3), "false_rate": round(float(false[n]), 4),
                   "precision": None if np.isnan(precision[n]) else round(float(precision[n]), 3)}
               for n in answers}
    if acc:
        for n in weights:
            weights[n]["label_accuracy"] = round(float(acc[n]), 3) if n in acc else None
    return {"kind": "objects", "radius_px": round(radius, 1), "objects": objects, "weights": weights, "scores": scores}


def region_task(task, answers):
    W, H = next(iter(answers.values()))["size"]
    scale = max(W, H) / REGION_GRID
    masks = {n: cs.rasterize([it["polygon"] for it in a["items"]], (W, H), scale) for n, a in answers.items()}
    temper = cs.tempering([a["maker"] for a in answers.values()])
    prob, sens, spec = cs.staple(masks, temper=temper)
    scores = {}
    for name, ans in answers.items():
        others = {n: m for n, m in masks.items() if answers[n]["maker"] != ans["maker"]}
        key = cs.staple(others, temper=cs.tempering([answers[n]["maker"] for n in others]))[0] > REAL
        mine = masks[name]
        union = (mine | key).sum()
        scores[name] = {"iou": float((mine & key).sum() / union) if union else 1.0,
                        "area_pct": round(100 * float(mine.mean()), 2), "key_area_pct": round(100 * float(key.mean()), 2)}
    weights = {n: {"sensitivity": round(float(sens[n]), 3), "specificity": round(float(spec[n]), 3)} for n in answers}
    return {"kind": "region", "grid": list(prob.shape[::-1]), "prob": prob, "weights": weights, "scores": scores,
            "area_pct": round(100 * float((prob > REAL).mean()), 2)}


def heat_png(prob: np.ndarray) -> str:
    rgba = np.zeros((*prob.shape, 4), dtype=np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = 255, 45, 85
    rgba[..., 3] = (np.clip(prob, 0, 1) * 170).astype(np.uint8)
    edge = (prob > REAL) ^ np.roll(prob > REAL, 1, 0) | (prob > REAL) ^ np.roll(prob > REAL, 1, 1)
    rgba[edge] = (255, 255, 255, 255)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main(path: str) -> None:
    run = Path(path)
    name = run.stem.removeprefix("run-")
    answers = load(run)
    out, page = {"run": run.stem, "note": "DRAFT from model consensus, not an answer key", "tasks": {}}, []
    for key, task in TASKS.items():
        if key not in answers:
            continue
        res = object_task(task, answers[key]) if task.kind == "objects" else region_task(task, answers[key])
        W, H = next(iter(answers[key].values()))["size"]
        view = {"key": key, "number": task.number, "target": task.target, "photo": task.photo, "w": W, "h": H,
                "kind": task.kind, "labels": list(task.labels), **{k: v for k, v in res.items() if k != "prob"},
                "answers": {n: {"model": a["model"], "label": MODELS[a["model"]].label, "maker": a["maker"],
                                "track": a["track"], "dots": [[round(v) for v in d] for d in dots_of(a)]
                                if task.kind == "objects" else None,
                                "polys": [[[round(v) for v in p] for p in it["polygon"]] for it in a["items"]]
                                if task.kind == "region" else None}
                            for n, a in answers[key].items()}}
        if task.kind == "region":
            view["heat"] = heat_png(res["prob"])
            res = {**res, "prob": np.round(res["prob"], 3).tolist()}
        out["tasks"][key] = res
        page.append(view)
        n_rev = sum(CONTESTED_LOW < o["p_real"] < CONTESTED_HIGH for o in res.get("objects", []))
        print(f"{key:7} " + (f"{sum(o['p_real'] > REAL for o in res['objects'])} likely objects, {n_rev} to review, radius {res['radius_px']} px"
                             if task.kind == "objects" else f"consensus area {res['area_pct']}% of photo"))
    (ROOT / "results" / f"consensus-{name}.json").write_text(json.dumps(out, separators=(",", ":")))
    html = (ROOT / "scripts" / "consensus.html").read_text().replace("/*DATA*/null", json.dumps({"run": run.stem, "tasks": page}, separators=(",", ":")))
    dest = ROOT / "results" / "views" / name / "consensus.html"
    dest.write_text(html)
    print(f"wrote results/consensus-{name}.json and {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1])
