"""Falcon Perception-HD's answers drawn on the eleven benchmark photos (one page).

    uv run --with pillow --with numpy scripts/view_falcon.py [--gpu-uuid UUID]
    -> results/preview/falcon-perception-hd/overlay/index.html

The benchmark runs kept points and boxes but not Falcon's outlines (masks off,
to keep the records small). This asks Falcon the same questions again, once,
with outlines on (not timed), and checks its points are exactly the scored
ones: it decodes greedily, so the same question must give the same answer.

Layers: Falcon's outlines (filled, one colour per object); its points, green
when they hit an answer-key object (medium tolerance, as scored), red when
they hit nothing; answer-key objects it missed (yellow rings); its boxes
(detection mode) and the answer-key boxes, for the box benchmark.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, "/home/maxime/Projects/finders")
import finders  # noqa: E402

R = ROOT / "results"
PREVIEW = R / "preview" / "falcon-perception-hd"
OUT = PREVIEW / "overlay"
KEY = "falcon-perception-hd"
TASKS = ["logs", "cows", "fig", "dishes", "ants", "beetles", "birds", "mushrooms", "jars", "pinkleaves", "whitecow"]
SHOW_W = 1200  # small enough for Chattering's artifact history (the 1600 px page did not fit)
QUERY_COLORS = [(255, 120, 0), (190, 60, 255)]  # orange, purple
PALETTE = [(0, 150, 255), (255, 120, 0), (180, 70, 255), (0, 200, 170), (255, 60, 150), (120, 200, 0),
           (255, 200, 0), (90, 110, 255)]


def jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def rle_decode(rle):
    h, w = rle["size"]
    flat = np.zeros(h * w, dtype=bool)
    pos, val = 0, False
    for c in rle["counts"]:
        if val:
            flat[pos:pos + c] = True
        pos += c
        val = not val
    return flat.reshape((w, h)).T


def gather():
    """Per task: photo, Falcon's exact questions, scored points and matches, boxes, answer keys."""
    s = (PREVIEW / "points-v3" / "data.js").read_text()
    report = json.loads(s[s.index("{"):s.rindex("}") + 1])
    pts = {r["task"]: r for r in jsonl(PREVIEW / "points-v3" / "scored.jsonl") if r["repeat"] == 0}
    bxs = {r["task"]: r for r in jsonl(PREVIEW / "boxes-v1" / "scored.jsonl") if r["repeat"] == 0}
    box_key = json.load(open(ROOT / "annotations/boxes/answer-key-boxes-v1.json"))["tasks"]
    queries = {}
    for r in jsonl(R / "run-finders-6.jsonl"):
        if r["track"] == "find" and r["repeat"] == 0:
            queries[r["task"]] = [q["query"] for q in r["finders"]["queries"]]
    for r in jsonl(R / "run-finders-b2-falcon.jsonl"):
        if r["track"] == "find":
            queries[r["task"]] = [r["query"]]
    tasks = {}
    for t in TASKS:
        info = report["tasks"][t]
        W, H = int(info["width"]), int(info["height"])
        refs = info["refs"] if isinstance(info["refs"], list) else json.loads(info["refs"])
        p = pts[t]
        hit = {m["prediction"]: m["reference_id"] for m in p["tolerances"]["medium"]["matches"]}
        found_refs = set(hit.values())
        tasks[t] = {
            "photo": info["photo"].removeprefix("images/"), "W": W, "H": H, "queries": queries[t],
            "score": round(p["score"] * 100, 1), "box_score": round(bxs[t]["scores"]["score"] * 100, 1),
            "points": [{"x": q["point"][0] * W / 1000, "y": q["point"][1] * H / 1000, "hit": i in hit}
                       for i, q in enumerate(p["predictions"])],
            "missed": [{"x": r["x"], "y": r["y"], "r": r["radii"]["medium"]} for r in refs
                       if r["id"] not in found_refs],
            "n_refs": len(refs),
            "boxes": [[b["box"][0] * W / 1000, b["box"][1] * H / 1000, b["box"][2] * W / 1000, b["box"][3] * H / 1000]
                      for b in bxs[t]["boxes"]],
            "key_boxes": [o["box"] for o in box_key[t]["objects"] if o.get("status", "accepted") == "accepted"],
        }
    return tasks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu-uuid", default="GPU-73bb8110-4d4b-f645-dce4-724b0baf767f")
    a = ap.parse_args()
    tasks = gather()
    jobs = [{"image": str(ROOT / "images" / v["photo"]), "query": q, "task": "segment", "id": f"{t}|{k}"}
            for t, v in tasks.items() for k, q in enumerate(v["queries"])]
    # Outlines 10 at a time (FALCON_MASK_CHUNK, finders/models/falcon/find.py): the
    # 90 log-end outlines at full size do not fit beside GPU 1's services otherwise.
    os.environ["FALCON_MASK_CHUNK"] = "10"
    recs = finders.find_many("falcon", jobs, variant="hd", masks=True, gpu=a.gpu_uuid)
    by_task = {}
    for job, rec in zip(jobs, recs, strict=True):
        by_task.setdefault(job["id"].split("|")[0], []).append(rec)

    OUT.mkdir(parents=True, exist_ok=True)
    checks = []
    for t, v in tasks.items():
        rs = by_task[t]
        W, H = v["W"], v["H"]
        # Same answers as the scored run? (points, in the scored records' rounding)
        again = [[round(o["point"][0] * 1000 / W, 2), round(o["point"][1] * 1000 / H, 2)]
                 for r in rs for o in r["objects"] if o.get("point") is not None]
        scored = [[round(p["x"] * 1000 / W, 2), round(p["y"] * 1000 / H, 2)] for p in v["points"]]
        same = len(again) == len(scored) and all(abs(x1 - x2) <= 0.011 and abs(y1 - y2) <= 0.011
                                                 for (x1, y1), (x2, y2) in zip(again, scored))
        checks.append((t, same, len(again), len(scored)))
        # Display photo and a transparent layer of outlines, at SHOW_W wide.
        photo = ImageOps.exif_transpose(Image.open(ROOT / "images" / v["photo"])).convert("RGB")
        assert photo.size == (W, H), (t, photo.size, (W, H))
        w, h = SHOW_W, round(SHOW_W * H / W)
        photo.resize((w, h), Image.LANCZOS).save(OUT / f"{t}.webp", quality=72, method=6)
        layer = np.zeros((H, W, 4), dtype=np.uint8)
        k = 0
        for qi, r in enumerate(rs):
            for o in r["objects"]:
                if "mask" not in o:
                    continue
                m = rle_decode(o["mask"])
                # Two questions (dishes): one colour per question, chosen to stand out
                # from the teal dishes; one question: one colour per object.
                c = PALETTE[k % len(PALETTE)] if len(rs) == 1 else QUERY_COLORS[qi]
                edge = m & ~(np.roll(m, 3, 0) & np.roll(m, -3, 0) & np.roll(m, 3, 1) & np.roll(m, -3, 1))
                layer[m] = (*c, 105)
                layer[edge] = (*c, 235)
                k += 1
        Image.fromarray(layer, "RGBA").resize((w, h), Image.LANCZOS).save(OUT / f"{t}.masks.webp", quality=80, method=6)
        v["n_masks"] = k
        if len(rs) > 1:
            v["query_colors"] = ["rgb(%d,%d,%d)" % QUERY_COLORS[i] for i in range(len(rs))]
        v["mask_area_pct"] = round(100 * sum(o.get("mask_area", 0) for r in rs for o in r["objects"]) / (W * H), 1)

    page = (ROOT / "scripts" / "view_falcon.html").read_text().replace("__DATA__", json.dumps(tasks))
    (OUT / "index.html").write_text(page)
    for t, same, n1, n2 in checks:
        print(f"{t:11} same answers as the scored run: {'yes' if same else 'NO'} ({n1} vs {n2} points)")
    print(f"page: {OUT / 'index.html'}")
    if not all(c[1] for c in checks):
        sys.exit("some answers differ from the scored run: the page would not show what was scored")


if __name__ == "__main__":
    main()
