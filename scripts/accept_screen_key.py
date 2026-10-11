"""Turn an export of the screen review app into the screen answer key.

    python3 scripts/accept_screen_key.py ~/Downloads/segbench-screen-key-....json annotations/screens/key.json

Refuses unless every screenshot was accepted by a named person, the
screenshots are byte-for-byte the ones reviewed, and every box is a valid
rectangle on its screenshot. Never overwrites: a new key gets a new name.
The export itself is kept next to the key (annotations/screens/reviews/).
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from segbench.screens import BY_KEY as TASKS, shape_ok  # noqa: E402


def main(export: Path, out: Path):
    d = json.loads(export.read_text())
    if d.get("schema") != "segbench-screen-key/v1":
        sys.exit("not an export of the screen review app")
    if not d.get("reviewer"):
        sys.exit("no reviewer name in the export")
    if out.exists():
        sys.exit(f"{out} exists; choose a new name")
    problems = []
    order = [TASKS[k] for k in d["tasks"]]
    for t in order:
        k = d["tasks"].get(t.key)
        if not k:
            problems.append(f"{t.key}: missing")
            continue
        if not k.get("accepted"):
            problems.append(f"{t.key}: not accepted")
        path = ROOT / "images" / "screenshots" / t.image
        if hashlib.sha256(path.read_bytes()).hexdigest() != k["image_sha256"]:
            problems.append(f"{t.key}: the screenshot changed since it was reviewed")
        W, H = k["image_size"]
        ids = [g["id"] for g in k["targets"]]
        if len(set(ids)) != len(ids):
            problems.append(f"{t.key}: two boxes share a name")
        for g in k["targets"]:
            if not shape_ok(g, W, H):
                problems.append(f"{t.key}/{g['id']}: {g.get('box') or g.get('circle')} is not a shape on the {W}x{H} screenshot")
            if any(c not in ids for c in g.get("covered_by", ())):
                problems.append(f"{t.key}/{g['id']}: covered by a target that does not exist")
    if problems:
        sys.exit("refused:\n  " + "\n  ".join(problems))
    reviews = ROOT / "annotations" / "screens" / "reviews"
    reviews.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(export, reviews / export.name)
    key = {"schema": d["schema"], "status": f"accepted by {d['reviewer']}", "reviewer": d["reviewer"],
           "accepted_from": f"annotations/screens/reviews/{export.name}",
           "export_sha256": hashlib.sha256(export.read_bytes()).hexdigest(),
           "draft": d["draft"], "draft_sha256": d["draft_sha256"], "coordinates": d["coordinates"], "scoring": d["scoring"],
           "tasks": {k: {f: v[f] for f in ("image_size", "image_sha256", "accepted_at", "notes", "targets")}
                     for k, v in d["tasks"].items()}}
    out.write_text(json.dumps(key, indent=2) + "\n")
    for t in order:
        print(f"{t.key:11} {len(key['tasks'][t.key]['targets'])} box(es): "
              + ", ".join(f"{g['id']} {g.get('box') or g.get('circle')}" for g in key["tasks"][t.key]["targets"]))
    print("Wrote", out)


if __name__ == "__main__":
    main(Path(sys.argv[1]).expanduser().resolve(), Path(sys.argv[2]).resolve())
