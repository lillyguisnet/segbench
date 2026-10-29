"""Draw one reply with every point or box coloured by its kind of mistake (scripts/classify_errors.py).

    uv run --with numpy --with pillow scripts/draw_error_kinds.py points "GPT-5.6 Terra" cows 2 out.png
    uv run --with numpy --with pillow scripts/draw_error_kinds.py --examples results/error-kinds/v1

Answer-key objects: white outline (SAM 2.1, from the reviewed box) and their box
in thin white; objects nobody found: blue box. Look-alikes (SAM 3): thin khaki
outline. Points: discs; boxes: rectangles; colour = kind (legend at the bottom).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from classify_errors import load_photos  # noqa: E402

COLOUR = {"correct": (40, 200, 60), "on the object, off-centre": (255, 170, 0), "mispositioned box": (255, 170, 0),
          "near miss": (255, 110, 0), "duplicate": (170, 60, 255), "in a gap": (120, 60, 160), "wrong object": (255, 40, 40),
          "background": (255, 0, 200), "group box": (255, 230, 0), "doom loop": (130, 130, 130),
          "off image": (0, 0, 0), "coordinate mix-up": (0, 235, 235), "shifted or stretched": (160, 255, 60),
          "maybe missing from key": (255, 255, 255)}
EXAMPLES = [  # (benchmark, model label, task, try): chosen to show each kind
    ("points", "SAM 3", "birds", 0), ("points", "Claude Sonnet 5.5", "jars", 0), ("points", "GPT-6 Astra", "fig", 0),
    ("points", "RF-DETR Seg", "whitecow", 0), ("points", "GPT-5.6 Terra", "cows", 2), ("points", "Kimi K3", "dishes", 0),
    ("points", "GPT-6 Luna", "beetles", 0), ("points", "Gemini 3.1 Pro", "beetles", 0),
    ("boxes", "Florence-2 Large", "logs", 0), ("boxes", "DeepSeek Flash", "birds", 0), ("boxes", "GPT-6 Luna", "cows", 1),
    ("boxes", "Kimi K3", "fig", 0), ("boxes", "Claude Opus 5.5", "dishes", 0), ("boxes", "LocateAnything 3B", "pinkleaves", 0),
    ("boxes", "RF-DETR Seg", "whitecow", 0), ("boxes", "Gemini 3.5 Flash Lite", "logs", 0),
]


def outline(mask):
    m = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(m.filter(ImageFilter.MaxFilter(5))) > 0, mask


def draw(photos, items, bench, label, task, rep, out, size=1100):
    T = photos[task]
    im = ImageOps.exif_transpose(Image.open(ROOT / T["photo"])).convert("RGB")
    a = np.asarray(im).copy()
    for lab, colour in ((T["looklab"], (190, 170, 110)), (T["keylab"], (255, 255, 255))):
        edge = lab != np.roll(lab, 1, 0)
        edge |= lab != np.roll(lab, 1, 1)
        edge &= (lab >= 0) | (np.roll(lab, 1, 0) >= 0)
        edge = np.asarray(Image.fromarray(edge.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(3))) > 0
        a[edge] = colour
    s = size / max(im.size)
    im = Image.fromarray(a).resize((int(im.width * s), int(im.height * s)))
    d = ImageDraw.Draw(im)
    sel = [r for r in items if r["label"] == label and r["task"] == task and r["repeat"] == str(rep)]
    missed = {r["object"] for r in sel if r["item"] == "object"}
    for o in T["objects"]:
        if o["box"] and (bench == "boxes" or o["in_point_key"]):
            b = [v * s for v in o["box"]]
            d.rectangle(b, outline=(0, 90, 255) if o["id"] in missed else (255, 255, 255), width=3 if o["id"] in missed else 1)
    for r in sel:
        if r["item"] == "object":
            continue
        c = COLOUR.get(r["kind"], (255, 255, 0))
        if bench == "boxes":
            d.rectangle([v * s for v in json.loads(r["box"])], outline=c, width=3)
        else:
            x, y = float(r["x"]) * s, float(r["y"]) * s
            d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=c, outline=(0, 0, 0))
    counts = {}
    for r in sel:
        k = "missed" if r["item"] == "object" else r["kind"]
        counts[k] = counts.get(k, 0) + 1
    legend = Image.new("RGB", (im.width, 26 + 22 * ((len(counts) + 2) // 3)), (25, 25, 25))
    ld = ImageDraw.Draw(legend)
    ld.text((8, 5), f"{label} · {task} · try {rep} · {bench}", fill=(255, 255, 255))
    for k, (kind, n) in enumerate(sorted(counts.items(), key=lambda x: -x[1])):
        x, y = 8 + (k % 3) * (im.width // 3), 26 + 22 * (k // 3)
        ld.rectangle([x, y + 2, x + 14, y + 16], fill=COLOUR.get(kind, (0, 90, 255)) if kind != "missed" else (0, 90, 255))
        ld.text((x + 20, y + 3), f"{kind}: {n}", fill=(255, 255, 255))
    page = Image.new("RGB", (im.width, im.height + legend.height))
    page.paste(im, (0, 0))
    page.paste(legend, (0, im.height))
    page.save(out)
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("args", nargs="*")
    ap.add_argument("--examples", type=Path)
    a = ap.parse_args()
    photos = load_photos()
    if a.examples:
        (a.examples / "examples").mkdir(exist_ok=True)
        cache = {}
        for bench, label, task, rep in EXAMPLES:
            items = cache.setdefault(bench, list(csv.DictReader(open(a.examples / bench / "items.csv"))))
            name = f"{bench}-{label.replace(' ', '_')}-{task}-{rep}.png"
            print(name, draw(photos, items, bench, label, task, rep, a.examples / "examples" / name))
        return
    bench, label, task, rep, out = a.args
    items = list(csv.DictReader(open(ROOT / "results/error-kinds/v1" / bench / "items.csv")))
    print(draw(photos, items, bench, label, task, int(rep), out))


if __name__ == "__main__":
    main()
