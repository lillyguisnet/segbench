"""Draft the measurement clicks, for Maxime to correct (2026-10-16).

uv run --with pillow --with numpy scripts/draft_measure_clicks.py ask      # 1. GPT-6 Astra proposes the clicks
uv run --with pillow --with numpy scripts/draft_measure_clicks.py refine   # 2. SAM 3 outlines each one; the clicks are moved onto that outline
uv run --with pillow --with numpy scripts/draft_measure_clicks.py merge    # 3. into annotations/measure/clicks-v1.json, as drafts to check

Maxime, 2026-10-16: "use the best pointing model and SAM 3 to get me started
so that I do more of a correction". GPT-6 Astra is first on point benchmark
v3 (77.6 %), called on our ChatGPT plan at medium thinking.

1. ask: one request per question: the photo and the question's own
   clicking instructions; Astra answers with the clicks of up to `suggest`
   instances, in pixels. Its records: annotations/measure/draft/astra.jsonl.
2. refine (specialists/sam/measure_refine.py, on GPU 1): SAM 3, given one
   click in the middle of Astra's clicks, outlines the part (of its three
   candidate outlines, the one whose length along Astra's clicks is closest
   to Astra's); span ends go to the outline's ends along Astra's direction,
   line points to the middle of the outline. When SAM's outline is not
   believable (it misses the middle click, covers over 30 % of the photo, or
   changes the length by more than a factor 1.6), Astra's clicks are kept.
3. merge: every draft becomes an item with status "draft"; Maxime keeps,
   moves or deletes each one. Questions he already clicked are left alone.

The bias this brings, said to Maxime before running: the reference starts
from GPT-6 Astra's and SAM 3's answers, and both will be compared. Each item
keeps its draft (`draft_points`, `origin`), so how far Maxime moved it is
known, and the two can be scored separately on the items he drew himself.
"""
from __future__ import annotations

import io
import json
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from segbench.call import call  # noqa: E402
from segbench.measure_tasks import QUESTIONS  # noqa: E402
from segbench.models import BY_KEY  # noqa: E402

DRAFT = ROOT / "annotations/measure/draft"
ASTRA = DRAFT / "astra.jsonl"
REFINED = DRAFT / "refined.json"
STATE = ROOT / "annotations/measure/clicks-v1.json"
MODEL = "astra"

ASK = (
    "This photo is {w} x {h} pixels (x counts from the left edge, y from the top edge).\n\n"
    "Find each {part} in this photo that can be measured clearly{note}. Give at most {n}, the clearest first.\n\n"
    "{what}\n\n"
    'Answer as JSON, in pixels of this photo: [{{"points": [{{"x": .., "y": ..}}, ...]}}, ...]'
)


def instructions(q) -> str:
    how = re.sub(r"\s*\d+ to \d+ clicks, then Enter\.?", "", q.how).strip()
    how = how[0].lower() + how[1:]
    if q.kind == "span":
        return f"For each one, give exactly two points, in this order: {how}"
    return f"For each one, give 3 to 8 points in order, {how}"


def prompt(q, w: int, h: int) -> str:
    kept = " ".join(x for x in re.split(r"(?<=\.)\s+", q.note) if x and "dot" not in x and "zoom" not in x.lower()
                    and "hard set" not in x and "stress test" not in x and "measured too" not in x).rstrip(".")
    note = f" ({kept[0].lower() + kept[1:]})" if kept else ""
    return ASK.format(w=w, h=h, part=q.part, note=note, n=q.suggest, what=instructions(q))


def read(text: str) -> list[list[list[float]]]:
    """[[x, y], ...] per instance; generous: the last JSON array in the reply."""
    blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, re.S) or [text[text.find("["):text.rfind("]") + 1]]
    for chunk in reversed(blocks):
        try:
            data = json.loads(chunk)
        except ValueError:
            continue
        out = []
        for inst in data if isinstance(data, list) else []:
            pts = inst.get("points") if isinstance(inst, dict) else inst
            pts = [[float(p["x"]), float(p["y"])] for p in pts or [] if isinstance(p, dict) and "x" in p and "y" in p]
            if len(pts) >= 2:
                out.append(pts)
        return out
    return []


def ask() -> None:
    DRAFT.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["q"] for l in ASTRA.open()} if ASTRA.exists() else set()
    model, lock = BY_KEY[MODEL], threading.Lock()

    def one(q):
        photo = ROOT / "images" / q.photo
        w, h = Image.open(photo).size
        text = prompt(q, w, h)
        rec = call(model, photo.read_bytes(), "image/jpeg", text, level="medium")
        rec.update({"q": q.key, "prompt": text, "size": [w, h]})
        inst = read(rec.get("text", "")) if not rec.get("error") else []
        with lock:
            with ASTRA.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"{q.key:20} {len(inst):2} drafts  {rec.get('seconds')} s  ${rec.get('cost_usd')}"
                  f"{'  ERROR ' + rec['error'][:100] if rec.get('error') else ''}", flush=True)

    todo = [q for q in QUESTIONS if q.key not in done]
    with ThreadPoolExecutor(4) as ex:  # our ChatGPT plan is shared with the family
        list(ex.map(one, todo))


def astra_drafts() -> dict:
    out = {}
    for line in ASTRA.open():
        rec = json.loads(line)
        if rec.get("error"):
            continue
        q = next(q for q in QUESTIONS if q.key == rec["q"])
        w, h = rec["size"]
        inst = [p for p in read(rec["text"]) if all(0 <= x <= w and 0 <= y <= h for x, y in p)]
        if q.kind == "span":
            inst = [p[:2] for p in inst]
        out[q.key] = {"photo": q.photo, "kind": q.kind, "instances": inst}
    return out


def refine() -> None:
    src = DRAFT / "astra-drafts.json"
    src.write_text(json.dumps(astra_drafts(), indent=1))
    subprocess.run(["uv", "run", "measure_refine.py", str(src), str(REFINED)], cwd=ROOT / "specialists/sam", check=True)


def middle(pts):
    import math
    total = sum(math.dist(pts[i - 1], pts[i]) for i in range(1, len(pts)))
    s = 0.0
    for i in range(1, len(pts)):
        d = math.dist(pts[i - 1], pts[i])
        if s + d >= total / 2 and d > 0:
            f = (total / 2 - s) / d
            return [round(pts[i - 1][0] + f * (pts[i][0] - pts[i - 1][0]), 1), round(pts[i - 1][1] + f * (pts[i][1] - pts[i - 1][1]), 1)]
        s += d
    return list(pts[0])


def merge() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from serve_measure_clicks import build, check
    seed = build()
    cur = json.loads(STATE.read_text()) if STATE.exists() else {"revision": 0, "data": None}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    data = cur["data"] or {
        "schema": "segbench-measure-clicks/v1", "questions_sha256": seed["questions_sha256"],
        "coordinate_system": "original upright photo pixels, [x, y]",
        "purpose": "reference measurements for outline-from-a-dot questions: a line along the middle of a part, or two points; dot = what the models are shown",
        "created_at": now, "updated_at": now, "items": [],
        "questions": {q.key: {"done": False, "dropped": False, "notes": ""} for q in QUESTIONS}, "audit": []}
    have = {it["q"] for it in data["items"]}
    refined = json.loads(REFINED.read_text())
    added = 0
    for key, r in refined.items():
        if key in have:
            print(f"{key}: already clicked by hand, drafts not added")
            continue
        for i, d in enumerate(r["instances"]):
            pts = [[round(x, 1), round(y, 1)] for x, y in d["points"]]
            dot = d.get("dot") or middle(pts)
            data["items"].append({"id": f"{key}-draft{i:02d}", "q": key, "points": pts, "dot": dot,
                                  "dot_moved": bool(d.get("dot")), "status": "draft", "origin": d["origin"],
                                  "draft_points": pts, "draft_dot": dot, "created_at": now, "edited_at": None})
            added += 1
    data["updated_at"] = now
    data["audit"].append({"at": now, "action": "drafts added", "count": added,
                          "source": "GPT-6 Astra clicks, moved onto SAM 3 outlines (scripts/draft_measure_clicks.py)"})
    check(data, seed)
    new = {"revision": cur["revision"] + 1, "saved_at": now, "data": data}
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(new, indent=1))
    print(f"{added} drafts added; revision {new['revision']}")


if __name__ == "__main__":
    {"ask": ask, "refine": refine, "merge": merge}[sys.argv[1]]()
