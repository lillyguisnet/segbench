"""Draw the screen-click results with the photo benchmark's editorial chart.

    python3 scripts/plot_screens.py results/run-screens-1.jsonl annotations/screens/key-draft.json screens-1

Scores each answer against the key exactly as scripts/score_screens.py does
(F1 of clicks inside the key's boxes), writes one call record per answer in
the chart's format (segbench/chart_data.py), then calls
scripts/draw_charts.py --style editorial. Output: charts/<name>/. The records
hold scores, times and costs only, never the screenshots or the replies.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from segbench.chart_data import describe  # noqa: E402
from segbench.point_score import maximum_matching  # noqa: E402
from segbench.screens import hits  # noqa: E402

TITLE = "How well can AI click the right thing on a screen?"


def f1(clicks, targets):
    tp = len(maximum_matching([hits(targets, x, y) for x, y in clicks]))
    return 2 * tp / (len(clicks) + len(targets)) if clicks or targets else 1.0


def main(run: Path, key_path: Path, name: str, title: str = TITLE):
    key = json.loads(key_path.read_text())
    out = ROOT / "charts" / name / "language-models"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for line in run.read_text().splitlines():
        r = json.loads(line)
        if r["task"] not in key["tasks"]:
            continue  # a question this key does not answer (e.g. a retired one)
        W, H = r["image_size"]
        assert [W, H] == key["tasks"][r["task"]]["image_size"], r["task"]
        clicks = [(c["x"] * W / 1000, c["y"] * H / 1000) for c in r.get("clicks", [])] if r.get("readable") else []
        label, maker, kind = describe(r["model"])
        rows.append({"track": "find", "task": r["task"], "model": r["model"], "level": r["level"],
                     "repeat": r["repeat"], "entrant": f"{r['model']}@{r['level']}", "label": label,
                     "maker": maker, "kind": kind, "seconds": r.get("seconds"), "cost_usd": r.get("cost_usd"),
                     "readable": bool(r.get("readable")) and "error" not in r,
                     "score": f1(clicks, key["tasks"][r["task"]]["targets"])})
    calls = out / "calls.jsonl"
    calls.write_text("".join(json.dumps(r) + "\n" for r in rows))
    subprocess.run(["uv", "run", str(ROOT / "scripts/draw_charts.py"), str(calls), "--style", "editorial",
                    "--track", "find", "--names", "all", "--title", title, "--out", str(out)], cwd=ROOT, check=True)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    (out.parent / "manifest.json").write_text(json.dumps({
        "source": str(run.relative_to(ROOT)), "source_sha256": digest(run),
        "key": str(key_path.relative_to(ROOT)), "key_sha256": digest(key_path), "key_status": key["status"],
        "renderer": "scripts/draw_charts.py --style editorial --track find --names all --title ...",
        "metric": "F1 of clicks inside the key's targets (one click per target), mean over 3 tries, then over the tasks",
        "caveats": ["Answer key is a draft until a person checks it.",
                    "Five screenshots: seven models score 100, so the top is not separated.",
                    "Language models only: the local pointing models are not run yet.",
                    "The smooth curve is a visual guide, not measured intermediate models."],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), sys.argv[3], *sys.argv[4:5])
