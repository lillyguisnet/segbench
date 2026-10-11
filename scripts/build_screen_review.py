"""Package the screen answer-key review app (apps/screen-key-review/).

    python3 scripts/build_screen_review.py results/run-screens-1.jsonl annotations/screens/key-draft.json

Copies the screenshots next to the app and writes seed.js: each task's
instruction, screenshot, draft boxes and every model click (in screenshot
pixels). Both are rebuilt, never committed (the screenshots are private).
A person fixes and accepts the boxes in the app and exports a JSON file;
scripts/accept_screen_key.py turns that file into the answer key.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from segbench.chart_data import describe  # noqa: E402
from segbench.screens import BY_KEY as TASKS  # noqa: E402

APP = ROOT / "apps" / "screen-key-review"


def main(run: Path, draft: Path):
    key = json.loads(draft.read_text())
    order = [TASKS[k] for k in key["tasks"]]
    clicks = {t.key: [] for t in order}
    for line in run.read_text().splitlines():
        r = json.loads(line)
        if r["task"] not in clicks:
            continue
        W, H = r["image_size"]
        for i, c in enumerate(r.get("clicks", []) if r.get("readable") else []):
            clicks[r["task"]].append({"x": round(c["x"] * W / 1000, 1), "y": round(c["y"] * H / 1000, 1),
                                      "model": r["model"], "label": describe(r["model"])[0],
                                      "try": r["repeat"] + 1, "n": i + 1})
    answers = {}
    for line in run.read_text().splitlines():
        r = json.loads(line)
        if r["task"] not in clicks:
            continue
        answers.setdefault(r["task"], []).append({"model": r["model"], "label": describe(r["model"])[0],
                                                   "try": r["repeat"] + 1})
    (APP / "images").mkdir(parents=True, exist_ok=True)
    tasks = []
    for t in order:
        src = ROOT / "images" / "screenshots" / t.image
        shutil.copyfile(src, APP / "images" / t.image)
        d = key["tasks"][t.key]
        tasks.append({"key": t.key, "instruction": t.instruction, "what": t.what, "image": f"images/{t.image}",
                      "image_size": d["image_size"], "image_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
                      "targets": d["targets"],
                      "accepted": bool(d.get("accepted")), "accepted_at": d.get("accepted_at"), "notes": d.get("notes", ""),
                      "clicks": clicks[t.key], "answers": answers[t.key]})
    raw = draft.read_bytes()
    seed = {"schema": "segbench-screen-review-seed/v1", "draft": str(draft.relative_to(ROOT)),
            "draft_sha256": hashlib.sha256(raw).hexdigest(), "run": run.name, "tasks": tasks}
    (APP / "seed.js").write_text("window.SCREEN_SEED = " + json.dumps(seed, separators=(",", ":")) + ";\n")
    print(f"Built {APP.relative_to(ROOT)}: {len(tasks)} screenshots, "
          f"{sum(len(c) for c in clicks.values())} clicks; draft {seed['draft_sha256'][:12]}")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
