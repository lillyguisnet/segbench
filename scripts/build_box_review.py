# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Package the answer-key box review app (apps/box-review) from the draft (default: draft-models-v1.json).

uv run scripts/build_box_review.py [annotations/boxes/draft-models-v1.json]

Writes apps/box-review/seed.js and copies the photos into apps/box-review/images/
(both ignored by Git, like apps/consensus-review). The draft is snapshotted by
its SHA-256: the browser save and every export belong to that one draft.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from segbench.tasks import BY_KEY  # noqa: E402

TITLES = {"logs": "Log ends", "cows": "Cows", "fig": "Fig leaves", "dishes": "Dishes",
          "ants": "Leaf pieces carried by ants", "beetles": "Beetles", "birds": "Birds", "mushrooms": "Mushroom caps",
          "jars": "Strawberry coulis jars", "pinkleaves": "Pink leaves", "whitecow": "White cow with a bit of red"}


def main():
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "annotations/boxes/draft-models-v1.json"
    raw = source.read_bytes()
    draft = json.loads(raw)
    app = ROOT / "apps/box-review"
    (app / "images").mkdir(parents=True, exist_ok=True)
    seed = {"schema": 1, "source": source.name, "source_sha256": hashlib.sha256(raw).hexdigest(), "tasks": []}
    for key, t in draft["tasks"].items():
        photo = ROOT / "images" / t["photo"]
        shutil.copyfile(photo, app / "images" / t["photo"])
        seed["tasks"].append({"key": key, "title": TITLES[key], "target": BY_KEY[key].target,
                              "photo": f"images/{t['photo']}", "width": t["width"], "height": t["height"],
                              "image_sha256": hashlib.sha256(photo.read_bytes()).hexdigest(),
                              "objects": t["objects"]})
    (app / "seed.js").write_text("window.BOX_SEED = " + json.dumps(seed, separators=(",", ":")) + ";\n")
    print(f"Built apps/box-review: {len(seed['tasks'])} tasks, "
          f"{sum(len(t['objects']) for t in seed['tasks'])} boxes; draft {seed['source_sha256'][:12]}")


if __name__ == "__main__":
    main()
