"""Ask Jev whether each outline of the dot probe outlines the thing the red dot is on (Maxime, 2026-10-16).

uv run --with pillow --with numpy scripts/jev_judge_segments.py
    ->  results/probe-point-segment/butterfly_display_case/jev-judge-<time>.jsonl  (every request)
        results/probe-point-segment/butterfly_display_case/jev-judge-<time>.json   (verdicts)
        results/probe-point-segment/butterfly_display_case/jev-judge-<time>-sheet.jpg

The six runs of scripts/probe_point_segment.py (Gemini 3.8 Flash with prompt A,
GPT-6.1 Sol with prompts A-E), 16 dots each: 96 outlines. Each is drawn as in
the pictures Maxime saw (photo, outline shaded with a cyan edge, the red dot
with its white ring) and sent to Jev 1.14 twice:
- "full": the whole photo (743 x 1319, inside Jev's 280-1568 px limits);
- "zoom": the outline's and the dot's box with a margin of 25 % of its size
  (at least 40 px), enlarged to Jev's limits. A small outline on the full
  photo is a few dozen pixels; the zoom shows it.
Two judgments per request (they cost nothing extra): yes/no "is the outline
correct", and a choice of four verdicts. Written before any of them was sent.

No answer key: these dots were never annotated. Jev's verdicts are compared
by eye with the pictures, not scored.
"""
from __future__ import annotations

import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from lm15 import choice, judgments, yes_no
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from probe_point_segment import grid_points, read_as, tile, to_mask  # noqa: E402
from segbench.call import call  # noqa: E402
from segbench.jev_halving import ATTEMPTS, MODEL, fit  # noqa: E402

DIR = ROOT / "results/probe-point-segment/butterfly_display_case"
PHOTO = ROOT / "images/butterfly_display_case.jpg"
RUNS = {"Gemini Flash, A": "gemini-flash-20261016-190813", "Sol, A": "sol-A-20261016-191822",
        "Sol, B": "sol-B-20261016-191857", "Sol, C": "sol-C-20261016-191935",
        "Sol, D": "sol-D-20261016-192013", "Sol, E": "sol-E-20261016-192054"}

INTRO = ("In this photo, a red dot with a white ring marks one spot. The area shaded with a cyan edge is "
         "someone's outline of the thing the red dot is on: the object it is on or, if the dot is on a "
         "background or a surface, that region. ")
QUESTIONS = judgments(
    ok=yes_no(INTRO + "Is the outline correct: does it cover the thing the red dot is on, most of it, and not much else?"),
    verdict=choice(INTRO + "What best describes the outline?", {
        "correct": "It outlines the thing the red dot is on, closely.",
        "rough": "It is on the thing the red dot is on, but misses a large part of it or spills well beyond it.",
        "wrong_thing": "It outlines something else, not the thing the red dot is on.",
        "broken": "It is not the shape of any one thing: zigzags or scrambled lines.",
    }),
)


def zoom(img: Image.Image, mask: np.ndarray, x: int, y: int) -> Image.Image:
    ys, xs = np.nonzero(mask)
    x0, x1 = (min(xs.min(), x), max(xs.max(), x)) if len(xs) else (x, x)
    y0, y1 = (min(ys.min(), y), max(ys.max(), y)) if len(ys) else (y, y)
    mx, my = max(40, 0.25 * (x1 - x0)), max(40, 0.25 * (y1 - y0))
    W, H = img.size
    return img.crop((max(0, int(x0 - mx)), max(0, int(y0 - my)), min(W, int(x1 + mx) + 1), min(H, int(y1 + my) + 1)))


def ask(picture: Image.Image) -> dict:
    sent, _ = fit(picture)
    b = io.BytesIO()
    sent.save(b, "JPEG", quality=95)
    for attempt in range(ATTEMPTS):
        rec = call(MODEL, b.getvalue(), "image/jpeg", None, level="medium", response_format=QUESTIONS)
        rec.update(sent_size=list(sent.size), attempt=attempt)
        rec.pop("traceback", None)
        if not rec.get("error") or not rec.get("retryable"):
            return rec
        time.sleep(min(30, 2 ** attempt))
    return rec


def build_jobs() -> list:
    """(run, dot, view, picture) for every outline: the pictures Jev was sent (before fitting)."""
    photo = Image.open(PHOTO).convert("RGB")
    w, h = photo.size
    pts = grid_points(w, h)
    jobs = []
    for run, base in RUNS.items():
        recs = {json.loads(l)["dot"]: json.loads(l) for l in open(DIR / f"{base}.jsonl")}
        for k in range(16):
            r = recs[k]
            read = read_as(r.get("prompt_version", "A"), r.get("text", ""), w, h)
            mask = to_mask(read, w, h)
            pic = tile(photo, *pts[k], read, mask)
            jobs += [(run, k, "full", pic), (run, k, "zoom", zoom(pic, mask, *pts[k]))]
    return jobs


def main() -> None:
    jobs = build_jobs()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = DIR / f"jev-judge-{stamp}"

    def one(job):
        run, k, view, pic = job
        rec = ask(pic)
        rec.update(run=run, dot=k, view=view)
        return rec

    t0 = time.time()
    with ThreadPoolExecutor(16) as ex:
        recs = list(ex.map(one, jobs))
    with open(f"{out}.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    rows = []
    for (run, k, view, pic), r in zip(jobs, recs):
        p = r.get("probabilities") or {}
        rows.append({"run": run, "dot": k, "view": view, "error": r.get("error"),
                     "p_ok": (p.get("ok") or {}).get("true"), "verdict": (r.get("data") or {}).get("verdict"),
                     "p_verdict": p.get("verdict"), "seconds": r.get("seconds"), "cost_usd": r.get("cost_usd")})
    summary = {"requests": len(recs), "errors": sum(1 for r in rows if r["error"]),
               "cost_usd": sum(r["cost_usd"] or 0 for r in rows), "wall_seconds": time.time() - t0,
               "questions": QUESTIONS, "rows": rows}
    Path(f"{out}.json").write_text(json.dumps(summary, indent=1, default=str))

    # sheet: one row per run, zoom pictures, Jev's verdicts under each
    S, L, T = 200, 150, 54
    font = ImageFont.load_default(size=13)
    big = ImageFont.load_default(size=16)
    sheet = Image.new("RGB", (L + 16 * S, 30 + len(RUNS) * (S + T)), "white")
    d = ImageDraw.Draw(sheet)
    for i in range(16):
        d.text((L + i * S + 80, 8), f"dot {i}", fill="black", font=big)
    by = {(r["run"], r["dot"], r["view"]): r for r in rows}
    pics = {(run, k, view): pic for run, k, view, pic in jobs}
    colour = {"correct": (30, 140, 60), "rough": (200, 130, 0), "wrong_thing": (200, 30, 30), "broken": (140, 0, 140)}
    for j, run in enumerate(RUNS):
        y = 30 + j * (S + T)
        d.text((6, y + S // 2), run, fill="black", font=big)
        for k in range(16):
            z = pics[(run, k, "zoom")].copy()
            z.thumbnail((S - 6, S - 6))
            sheet.paste(z, (L + k * S + 3 + (S - 6 - z.width) // 2, y + 3))
            for n, view in enumerate(("full", "zoom")):
                r = by[(run, k, view)]
                txt = "error" if r["error"] else f"{view}: {r['p_ok']:.2f} {r['verdict']}"
                d.text((L + k * S + 4, y + S + 4 + n * 17), txt, fill=colour.get(r["verdict"], "black"), font=font)
    sheet.save(f"{out}-sheet.jpg", quality=88)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("rows", "questions")}))
    print(out)


if __name__ == "__main__":
    main()
