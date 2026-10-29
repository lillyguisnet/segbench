"""Mark the things on each photo that a wrong answer could be pointing at (look-alikes).

    uv run scripts/find_lookalikes.py  ->  results/error-kinds/support/lookalikes-sam3.json

For sorting mistakes (scripts/classify_errors.py): a point or box that is on
no answer-key object is either on some other thing a model could confuse with
the target ("wrong object": the black cows next to the white one, the
butterflies around the beetles, the other plants' leaves) or on nothing
("background"). SAM 3 (through ~/Projects/finders) outlines the things named
below on each photo, with its default confidence threshold.

The phrases were fixed on 2026-10-19 before this sorting was run, from each
question's wording (its explicit exclusions) and the broader kind of thing the
target is. Logs have none: the close-up is nothing but logs, so "on a log" would
mean nothing. Logs get the target phrase instead, because their answer key
was never checked over the whole photo: a mistake on a SAM 3 "cut log end"
that is in no key box may be a real log end missing from the key.

SAM 3 is an entrant of the benchmark. Here it only names other things on the
photo; it never decides whether an answer is right, and nothing here changes a
score. The outlines are not reviewed by a person.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, "/home/maxime/Projects/finders")
import finders  # noqa: E402

OUT = ROOT / "results/error-kinds/support/lookalikes-sam3.json"
LOOKALIKES = {
    "logs": [],
    "cows": ["animal"],
    "fig": ["leaf"],
    "dishes": ["dish", "cup", "bowl", "pot", "kitchen utensil"],
    "ants": ["ant", "leaf"],
    "beetles": ["butterfly", "insect"],
    "birds": ["bird", "shadow"],
    "mushrooms": ["mushroom", "hand"],
    "jars": ["jar"],
    "pinkleaves": ["leaf"],
    "whitecow": ["cow"],
}
KEY_GAP = {"logs": ["cut log end"]}  # answer key not checked over the whole photo


def main() -> None:
    if OUT.exists():
        sys.exit(f"{OUT} exists")
    photos = {}
    for f in ["annotations/reviews/segbench-review-6187f623-approved.json",
              "annotations/reviews/segbench-review-5682a4be-approved.json"]:
        for s in json.loads((ROOT / f).read_text())["source_snapshot"]["tasks"]:
            photos[s["key"]] = ROOT / "images" / Path(s["photo"]).name
    jobs = [{"image": str(photos[t]), "query": q, "id": f"{t}|{role}|{q}"}
            for role, table in (("lookalike", LOOKALIKES), ("key_gap", KEY_GAP))
            for t, qs in table.items() for q in qs]
    recs = finders.find_many("sam3", jobs, gpu="0")
    out = {"model": recs[0]["weights"], "method": __doc__.strip(), "lookalikes": LOOKALIKES, "key_gap": KEY_GAP,
           "records": []}
    for j, r in zip(jobs, recs):
        task, role, q = j["id"].split("|")
        out["records"].append({"task": task, "role": role, "query": q, "width": r["width"], "height": r["height"],
                               "settings": r.get("settings"),
                               "objects": [{k: o[k] for k in ("box", "score", "mask_area", "mask")} for o in r["objects"]]})
        print(f"{task:11} {role:9} {q:16} {len(r['objects']):3} found", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out))
    print("saved", OUT)


if __name__ == "__main__":
    main()
