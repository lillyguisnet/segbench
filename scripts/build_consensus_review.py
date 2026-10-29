# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow>=10"]
# ///
"""Package a standalone human review app. No model calls or annotation writes.

uv run scripts/build_consensus_review.py
uv run scripts/build_consensus_review.py results/consensus-pilot-1.json
uv run scripts/build_consensus_review.py results/consensus-consensus-2.json results/views/consensus-2/review

Input is snapshotted and hashed; a new source has a separate browser save.
Full-resolution JPEGs retain their original pixel grid. This builds assets
only; the editable application source lives in apps/consensus-review/.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from segbench.tasks import BY_KEY


def build(source: Path, dest: Path):
    raw = source.read_bytes()
    data = json.loads(raw)
    seed = {"schema": 1, "source": source.name, "source_sha256": hashlib.sha256(raw).hexdigest(),
            "run": data["run"], "tasks": []}
    (dest / "images").mkdir(parents=True, exist_ok=True)
    titles = {"logs": "Log ends", "cows": "Cows", "fig": "Fig leaves", "tree": "Red foliage",
              "road": "Gravel road", "dishes": "Dishes", "ants": "Leaf pieces carried by ants",
              "beetles": "Beetles", "birds": "Birds", "mushrooms": "Mushroom caps", "jars": "Strawberry coulis jars",
              "pinkleaves": "Pink leaves", "whitecow": "White cow with a bit of red"}
    app = ROOT / "apps/consensus-review"
    if dest != app:  # another consensus: its own copy of the app, so each keeps its own seed
        for name in ("index.html", "app.js", "style.css"):
            shutil.copyfile(app / name, dest / name)
    for task in BY_KEY.values():
        if task.key not in data["tasks"]:
            continue
        photo = ROOT / "images" / task.photo
        with Image.open(photo) as im:
            if im.getexif().get(274) not in (None, 1):
                raise ValueError(f"Rotate {photo} before annotating")
            w, h = im.size
        shutil.copyfile(photo, dest / "images" / task.photo)
        source_task = data["tasks"][task.key]
        t = {"key": task.key, "title": titles[task.key], "target": task.target, "kind": task.kind,
             "photo": f"images/{task.photo}", "width": w, "height": h,
             "image_sha256": hashlib.sha256(photo.read_bytes()).hexdigest()}
        if task.kind == "objects":
            t["objects"] = [{"id": f"{task.key}-{i:04}", "x": o["center"][0], "y": o["center"][1],
                             "support": o["p_real"], "votes": o["votes"],
                             "label": ("dirty" if o["p_dirty"] > .5 else "clean") if "p_dirty" in o else None,
                             "label_support": o.get("p_dirty")}
                            for i, o in enumerate(source_task["objects"])]
        else:
            t["grid"] = source_task["grid"]
            t["mask"] = "".join("1" if p > .5 else "0" for row in source_task["prob"] for p in row)
        seed["tasks"].append(t)
    (dest / "seed.js").write_text("window.REVIEW_SEED = " + json.dumps(seed, separators=(",", ":")) + ";\n")
    print(f"Built {dest.relative_to(ROOT)}: {len(seed['tasks'])} tasks; source {seed['source_sha256'][:12]}")


if __name__ == "__main__":
    # A second argument builds into another folder (e.g. results/views/consensus-2/review).
    build(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/consensus-pilot-1.json",
          (ROOT / sys.argv[2]).resolve() if len(sys.argv) > 2 else ROOT / "apps/consensus-review")
