"""The two box prompts on synthetic pictures, before any benchmark photo.

uv run --with pillow scripts/check_boxes.py [--prompt box-a|box-free] [models]
    ->  results/check-boxes-<prompt>-<time>.jsonl

Seven red ovals (wide, tall, round, small, two that overlap) and four blue
squares (distractors) on white, at the sizes of the benchmark photos
(4000x2252, 2252x4000, 1015x1015, 743x1319). The truth of each oval is its
bounding rectangle, exactly.

What it is for:
- box-a: does every model answer with the named corners, on the 0..1000
  grid, x and y the right way round? Pass = every oval matched by one box
  at IoU >= 0.7 and no other box.
- box-free: no format is asked, so this shows which format and which
  coordinate system each model chooses on its own (0..1000, 0..1, pixels of
  which picture size, x or y first). The reading rules of
  segbench/box_read.py are fixed from these replies and the models' own
  words, never from a benchmark photo or its answer key.

Ovals placed so that reading x and y swapped, or a 0..1000 value as pixels,
puts the boxes off the ovals.
"""
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from segbench.box_read import iou, read_box_a  # noqa: E402
from segbench.call import call  # noqa: E402
from segbench.models import BY_KEY as MODELS, levels_for  # noqa: E402
from segbench.tasks import BY_KEY, PROMPT_VERSION_BOX_A, PROMPT_VERSION_BOX_FREE, prompt_box  # noqa: E402
from run import IMAGE_SETTINGS_V2  # noqa: E402

# Every language model of point benchmark v3 (Muse Wonder through its API:
# the plan's quota is used up until 2026-10-19).
BOX_MODELS = ["luna", "terra", "sonnet", "opus", "haiku", "astra", "sol", "gemini-pro", "gemini-flash",
              "gemini-flash-lite", "gemini-argon", "glm-5.3-flash", "deepseek-flash", "kimi-k3", "qwen-27b",
              "qwen-flash", "mk1.5", "muse-wonder-api", "mistral-large-4"]

# (centre x, centre y, half width, half height) as shares of the picture's width and height.
OVALS = [(0.14, 0.22, 0.07, 0.10), (0.42, 0.74, 0.12, 0.06), (0.68, 0.28, 0.035, 0.13),
         (0.86, 0.80, 0.05, 0.05), (0.30, 0.48, 0.02, 0.025), (0.60, 0.56, 0.06, 0.08), (0.69, 0.60, 0.05, 0.07)]
SQUARES = [(0.20, 0.82, 0.04), (0.55, 0.10, 0.04), (0.90, 0.18, 0.04), (0.42, 0.30, 0.03)]
SIZES = {"wide": (4000, 2252), "tall": (2252, 4000), "square": (1015, 1015), "small-tall": (743, 1319)}


def picture(w, h):
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    s = min(w, h)
    for fx, fy, r in SQUARES:
        d.rectangle((fx * w - r * s, fy * h - r * s, fx * w + r * s, fy * h + r * s), fill=(40, 70, 210))
    truth = []
    for fx, fy, hw, hh in OVALS:
        box = (round((fx - hw) * w), round((fy - hh) * h), round((fx + hw) * w), round((fy + hh) * h))
        d.ellipse(box, fill=(215, 30, 30), outline=(150, 10, 10), width=max(1, s // 300))
        truth.append([box[0] * 1000 / w, box[1] * 1000 / h, box[2] * 1000 / w, box[3] * 1000 / h])
    b = io.BytesIO()
    im.save(b, "JPEG", quality=92)
    return b.getvalue(), truth


def grade(boxes, truth, threshold=0.7):
    used, hit = set(), 0
    for t in truth:
        best = max(((iou(b["box"], t), i) for i, b in enumerate(boxes) if i not in used), default=(0, None))
        if best[0] >= threshold:
            used.add(best[1])
            hit += 1
    return hit, len(boxes) - len(used)


def main():
    args = sys.argv[1:]
    version = PROMPT_VERSION_BOX_A
    if "--prompt" in args:
        i = args.index("--prompt")
        version = args[i + 1]
        del args[i:i + 2]
    assert version in (PROMPT_VERSION_BOX_A, PROMPT_VERSION_BOX_FREE)
    task = replace(BY_KEY["cows"], target="each red oval")
    text = prompt_box(version, task)
    pics = {k: picture(*wh) for k, wh in SIZES.items()}
    models = [MODELS[k] for k in (args or BOX_MODELS)]

    def run(job):
        m, shape = job
        data, truth = pics[shape]
        extra = IMAGE_SETTINGS_V2.get(m.route.split(":")[0], {})
        rec = call(m, data, "image/jpeg", text, level="medium" if "medium" in levels_for(m.key) else None, **extra)
        rec.update(check=f"boxes-{version}", prompt_version=version, prompt=text, shape=shape,
                   size=SIZES[shape], truth=truth)
        if "error" not in rec and version == PROMPT_VERSION_BOX_A:
            try:
                boxes, how = read_box_a(rec["text"], parser="lenient")
                rec["read_as"] = how
                rec["hit"], rec["extra"] = grade(boxes, truth)
                rec["named"] = '"x_min"' in rec["text"]
            except Exception as e:
                rec["unreadable"] = str(e)
        return rec

    jobs = [(m, s) for m in models for s in pics]
    with ThreadPoolExecutor(16) as pool:
        recs = list(pool.map(run, jobs))
    out = Path("results") / f"check-boxes-{version}-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in recs))
    for r in recs:
        if "error" in r:
            s = "ERROR " + r["error"][:100]
        elif version == PROMPT_VERSION_BOX_FREE:
            s = f"{len(r['text'])} chars, ${r.get('cost_usd') or 0:.4f}, {r['seconds']:.0f}s"
        elif "unreadable" in r:
            s = "UNREADABLE " + r["unreadable"][:80]
        else:
            s = (f"{'PASS' if r['hit'] == len(OVALS) and r['extra'] == 0 else 'fail'}  {r['hit']}/{len(OVALS)} ovals, "
                 f"{r['extra']} extra, {'named' if r['named'] else 'NOT named'}, read {r['read_as']}, "
                 f"${r.get('cost_usd') or 0:.4f}, {r['seconds']:.0f}s")
        print(f"{r['model']:18} {r['shape']:10} {s}")
    print("saved", out)


if __name__ == "__main__":
    main()
