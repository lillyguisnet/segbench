"""The same two questions Jev was asked, put to Claude Opus 5.5 (Maxime, 2026-10-18).

uv run --with pillow --with numpy scripts/opus_judge_segments.py JEV_RUN
    JEV_RUN: results/probe-point-segment/butterfly_display_case/jev-judge-<time>  (no suffix)
    ->  .../opus-judge-<time>.jsonl / .json, and .../jev-vs-opus-<time>-sheet.jpg

A separate Opus call per picture, on our Claude plan (claude-code), at the
benchmark's medium thinking. It gets exactly the picture Jev got (the same
bytes after Jev's size fitting) and the same words: Jev's yes/no and its
four-way choice, written as text, with a fixed last two lines to answer in.
Replies are read generously (the last "correct:" and "verdict:" lines, any
case); a reply that gives neither is kept as unread, never guessed.
"""
from __future__ import annotations

import io
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from jev_judge_segments import DIR, INTRO, RUNS, build_jobs  # noqa: E402
from segbench.call import call  # noqa: E402
from segbench.jev_halving import fit  # noqa: E402
from segbench.models import BY_KEY  # noqa: E402

MODEL = BY_KEY["opus"]
VERDICTS = {
    "correct": "It outlines the thing the red dot is on, closely.",
    "rough": "It is on the thing the red dot is on, but misses a large part of it or spills well beyond it.",
    "wrong_thing": "It outlines something else, not the thing the red dot is on.",
    "broken": "It is not the shape of any one thing: zigzags or scrambled lines.",
}
PROMPT = (INTRO + "\n\n1. Is the outline correct: does it cover the thing the red dot is on, most of it, and not much else? (yes or no)\n"
          "2. What best describes the outline?\n" + "\n".join(f"   - {k}: {v}" for k, v in VERDICTS.items()) +
          "\n\nEnd your answer with exactly these two lines:\ncorrect: yes or no\nverdict: correct, rough, wrong_thing or broken")


def read(text: str) -> dict:
    ok = re.findall(r"correct\s*[:=]\s*\**\s*(yes|no)\b", text, re.I)
    verdict = re.findall(r"verdict\s*[:=]\s*\**\s*(correct|rough|wrong[_ ]thing|broken)\b", text, re.I)
    return {"ok": ok[-1].lower() == "yes" if ok else None,
            "verdict": verdict[-1].lower().replace(" ", "_") if verdict else None}


def main() -> None:
    jev_base = Path(sys.argv[1])
    jev = {(r["run"], r["dot"], r["view"]): r for r in json.loads(Path(f"{jev_base}.json").read_text())["rows"]}
    jobs = build_jobs()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = DIR / f"opus-judge-{stamp}"

    def one(job):
        run, k, view, pic = job
        sent, _ = fit(pic)
        b = io.BytesIO()
        sent.save(b, "JPEG", quality=95)
        for attempt in range(4):
            rec = call(MODEL, b.getvalue(), "image/jpeg", PROMPT, level="medium")
            if not rec.get("error") or not rec.get("retryable"):
                break
            time.sleep(10 * (attempt + 1))
        rec.pop("traceback", None)
        rec.update(run=run, dot=k, view=view, sent_size=list(sent.size), prompt=PROMPT)
        print(f"{run:16} dot {k:2} {view}: {read(rec.get('text', '')) if not rec.get('error') else rec['error'][:80]}", flush=True)
        return rec

    t0 = time.time()
    with ThreadPoolExecutor(3) as ex:  # our Claude plan is shared with the family
        recs = list(ex.map(one, jobs))
    with open(f"{out}.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    rows = []
    for (run, k, view, _), r in zip(jobs, recs):
        a = read(r.get("text", "")) if not r.get("error") else {"ok": None, "verdict": None}
        j = jev[(run, k, view)]
        rows.append({"run": run, "dot": k, "view": view, "opus_ok": a["ok"], "opus_verdict": a["verdict"],
                     "jev_p_ok": j["p_ok"], "jev_ok": j["p_ok"] >= 0.5, "jev_verdict": j["verdict"],
                     "error": r.get("error"), "seconds": r.get("seconds"), "cost_usd": r.get("cost_usd")})
    both = [r for r in rows if r["opus_ok"] is not None]
    summary = {"requests": len(rows), "errors": sum(1 for r in rows if r["error"]),
               "unread": sum(1 for r in rows if r["opus_ok"] is None and not r["error"]),
               "cost_usd_api_price": sum(r["cost_usd"] or 0 for r in rows), "wall_seconds": time.time() - t0,
               "agree_ok": sum(r["opus_ok"] == r["jev_ok"] for r in both), "compared": len(both),
               "jev_run": jev_base.name, "rows": rows}
    Path(f"{out}.json").write_text(json.dumps(summary, indent=1))
    draw_sheet(jobs, rows, DIR / f"jev-vs-opus-{stamp}-sheet.jpg")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}))


def draw_sheet(jobs, rows, path) -> None:
    S, L, T = 200, 150, 58
    font, big = ImageFont.load_default(size=13), ImageFont.load_default(size=16)
    sheet = Image.new("RGB", (L + 16 * S, 70 + len(RUNS) * (S + T)), "white")
    d = ImageDraw.Draw(sheet)
    d.text((8, 8), "Under each outline: whole photo / zoom.   yes = says correct, no = says not correct.   "
                   "Green border: Jev and Opus agree on both views; red: they disagree on at least one.", fill="black", font=big)
    for i in range(16):
        d.text((L + i * S + 80, 40), f"dot {i}", fill="black", font=big)
    by = {(r["run"], r["dot"], r["view"]): r for r in rows}
    pics = {(run, k, view): pic for run, k, view, pic in jobs}
    mark = lambda v: "?" if v is None else ("yes" if v else "no")  # noqa: E731
    for j, run in enumerate(RUNS):
        y = 66 + j * (S + T)
        d.text((6, y + S // 2), run, fill="black", font=big)
        for k in range(16):
            f, z = by[(run, k, "full")], by[(run, k, "zoom")]
            agree = all(r["opus_ok"] == r["jev_ok"] for r in (f, z))
            x0 = L + k * S
            d.rectangle((x0 + 1, y + 1, x0 + S - 2, y + S - 2), outline=(40, 160, 70) if agree else (220, 40, 40), width=3)
            im = pics[(run, k, "zoom")].copy()
            im.thumbnail((S - 10, S - 10))
            sheet.paste(im, (x0 + 5 + (S - 10 - im.width) // 2, y + 5))
            for n, (who, key) in enumerate((("Jev ", "jev_ok"), ("Opus", "opus_ok"))):
                d.text((x0 + 6, y + S + 4 + n * 18), f"{who}  {mark(f[key])} / {mark(z[key])}", fill="black", font=big)
    sheet.save(path, quality=88)


if __name__ == "__main__":
    main()
