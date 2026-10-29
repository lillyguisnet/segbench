"""Score TypeLLM's run exactly as point benchmark v3 and box benchmark v1 were scored; rank it among them.

    uv run --with pillow scripts/score_typellm.py --name typellm-1   ->  results/preview/typellm/<name>/

Nothing published is changed or rescored: the published summaries are read as they are.
- points: scripts/score_points.py with v3's settings (both approved references, lenient reading,
  location only for dishes, cost and seconds from the first four tasks), one prompt per file;
  each entrant keeps its better prompt over the eleven tasks (ties to v2), as v3;
- boxes: scripts/score_boxes.py against the reviewed answer key, one prompt per file; each entrant
  keeps its better prompt (ties to box-a), as box v1.
Entrants: typellm@none (TypeLLM) and typellm@count-medium (TypeLLM + count first, our pipeline).
"""
import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from build_v3 import OLD_TASKS, REFERENCES, TASKS  # noqa: E402

R = ROOT / "results"
KEY = ROOT / "annotations/boxes/answer-key-boxes-v1.json"
NAMES = {"typellm@none": "TypeLLM", "typellm@count-medium": "TypeLLM + count first"}


def points(name, out):
    res = {}
    for v, tie in (("typellm-v2", 1), ("typellm-v1-hd", 0)):
        d = out / f"points-{v}"
        subprocess.run(["uv", "run", "scripts/score_points.py", "--parser", "lenient", "--dish-labels", "separate",
                        "--method", "docs/point-benchmark-v2.md", "--reference", *REFERENCES, "--tasks", *TASKS,
                        "--timing-tasks", *OLD_TASKS, "--inputs", str(R / f"run-{name}-{v}.jsonl"), "--out", str(d)],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        for r in csv.DictReader(open(d / "summary.csv")):
            if r["score"]:
                res.setdefault(r["entrant"], []).append((float(r["score"]), tie, v, r))
    return {e: max(c) for e, c in res.items()}


def boxes(name, out):
    d = out / "boxes"
    subprocess.run(["uv", "run", "--with", "pillow", "scripts/score_boxes.py", "--reference", str(KEY), "--inputs",
                    *[str(R / f"run-{name}-{v}.jsonl") for v in ("typellm-box-a", "typellm-box-free")],
                    "--out", str(d)], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    per = {}
    for r in csv.DictReader(open(d / "per-try.csv")):
        per.setdefault((r["entrant"], r["prompt"]), {}).setdefault(r["task"], []).append(r)
    res = {}
    for r in csv.DictReader(open(d / "summary.csv")):
        tie = 1 if r["prompt"] == "typellm-box-a" else 0
        tasks = {t: statistics.mean(float(x["score"]) for x in rows) for t, rows in per[(r["entrant"], r["prompt"])].items()}
        res.setdefault(r["entrant"], []).append((float(r["score"]), tie, r["prompt"], {**r, **tasks}))
    return {e: max(c) for e, c in res.items()}


def rank(score, published):
    return 1 + sum(1 for s in published if s > score), len(published) + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    out = R / "preview" / "typellm" / a.name
    if out.exists():
        sys.exit(f"{out} exists: move it away first (scores are never overwritten)")
    out.mkdir(parents=True)
    pub_p = list(csv.DictReader(open(R / "point-benchmark-v3/summary.csv")))
    pub_b = list(csv.DictReader(open(R / "box-benchmark/v1/summary.csv")))
    report = {}
    print("POINTS (point benchmark v3: 11 photos; published entrants: %d)" % len(pub_p))
    for e, (s, _, v, r) in sorted(points(a.name, out).items()):
        rk = rank(s, [float(x["score"]) for x in pub_p])
        print(f"  {NAMES[e]:22} {s*100:5.1f}  rank {rk[0]} of {rk[1]}  prompt {v}  "
              f"${float(r['cost_usd'] or 0):.5f}/photo  {float(r['seconds'] or 0):.1f} s")
        print("     " + "  ".join(f"{t} {float(r[t])*100:.0f}" for t in TASKS))
        report[f"points {e}"] = {"score": s, "rank": rk, "prompt": v, **r}
    print("BOXES (box benchmark v1: 11 photos; published entrants: %d)" % len(pub_b))
    for e, (s, _, v, r) in sorted(boxes(a.name, out).items()):
        rk = rank(s, [float(x["score"]) for x in pub_b])
        print(f"  {NAMES[e]:22} {s*100:5.1f}  rank {rk[0]} of {rk[1]}  prompt {v}  "
              + "  ".join(f"{lv}% {float(r['f1_' + lv])*100:.1f}" for lv in ("50", "75", "90", "95"))
              + f"  ${float(r['cost_usd_per_question_first4'] or 0):.5f}/photo  {float(r['median_seconds_first4'] or 0):.1f} s")
        print("     " + "  ".join(f"{t} {r[t]*100:.0f}" for t in TASKS))
        report[f"boxes {e}"] = {"score": s, "rank": rk, "prompt": v, **r}
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
