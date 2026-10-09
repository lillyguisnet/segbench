"""Make the six task photos into an image set the recorn annotation app reads.

uv run --project specialists/sam scripts/prepare_annotation_set.py

Writes annotations/data/segbench/{canonical/*.png, manifest.json}, the same
layout as recorn's tools/prepare_set.py (Lilly's app).

Orientation rule (one pixel grid for everyone): a canonical image is the
decoded pixel array of the photo, exactly as stored. That is only the
picture a person sees if the photo carries no rotation tag (EXIF
orientation absent or 1). This script refuses any photo with another
orientation, so the answer key, the models and the scorers can never
disagree about which way is up. All coordinates in the answer keys are in
these canonical pixels; models receive the same photo, and their 0..1000
coordinates are mapped onto this grid.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TASK_PHOTOS = {  # task number -> photo (docs/task-ideas.md, locked 2026-10-09)
    1: "log_ends_closeup_1.jpg",
    2: "field_with_cows.jpg",
    3: "fig_plant.jpg",
    4: "autumn_tree_orange_1.jpg",
    5: "farm_road_1.jpg",
    6: "kitchen_counter_dishes.jpg",
}
SET = "segbench"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    out = ROOT / "annotations" / "data" / SET
    canon = out / "canonical"
    canon.mkdir(parents=True, exist_ok=True)
    images = []
    for task, name in TASK_PHOTOS.items():
        src = ROOT / "images" / name
        with Image.open(src) as im:
            orientation = im.getexif().get(0x0112)
            if orientation not in (None, 1):
                raise SystemExit(f"{name}: EXIF orientation {orientation}; rotate the pixels first (see docstring)")
            fmt, mode, rgb = im.format, im.mode, im.convert("RGB")
        dst = canon / f"{src.stem}.png"
        rgb.save(dst, format="PNG", compress_level=6)
        images.append({
            "id": src.stem, "file": dst.name, "width": rgb.width, "height": rgb.height, "task": task,
            "original": {"path": f"images/{name}", "format": fmt, "mode": mode,
                         "sha256": sha256(src), "exif_orientation": orientation},
        })
        print(f"task {task}: {name} {rgb.width}x{rgb.height} orientation={orientation}")
    manifest = {"set": SET, "description": "segbench task photos (CC BY 4.0), canonical pixels, upright",
                "count": len(images), "images": images}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{len(images)} images -> {out}")


if __name__ == "__main__":
    main()
