"""Draw the screen-click results with the photo benchmark's editorial chart.

    python3 scripts/plot_screens.py annotations/screens/key.json screens-1 "TITLE" results/run-screens-1.jsonl results/run-screens-local-1*.jsonl

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
# Labels pinned beside their logo, per chart (Maxime, 2026-10-11): on round 2,
# SAM 3's label beside its logo on the 0% line, which frees the spot just above
# Moondream's logo; Florence-2 Large's just above its logo (else it touches
# SAM 3's by 2 px and the placer sends it far off with a long line).
LABEL_SPOTS = {"screens-2": {"sam3": "right", "moondream-3.1": "above", "florence2-large-grounding": "above"}}


def f1(clicks, targets):
    tp = len(maximum_matching([hits(targets, x, y) for x, y in clicks]))
    return 2 * tp / (len(clicks) + len(targets)) if clicks or targets else 1.0


def main(runs: list[Path], key_path: Path, name: str, title: str = TITLE):
    key = json.loads(key_path.read_text())
    rows = []
    for run in runs:
        for line in run.read_text().splitlines():
            r = json.loads(line)
            if r["task"] not in key["tasks"] or "error" in r:
                continue  # a question this key does not answer (e.g. a retired one)
            W, H = r["image_size"]
            assert [W, H] == key["tasks"][r["task"]]["image_size"], r["task"]
            clicks = [(c["x"] * W / 1000, c["y"] * H / 1000) for c in r.get("clicks", [])] if r.get("readable") else []
            label, maker, kind = describe(r["model"])
            rows.append({"track": "find", "task": r["task"], "model": r["model"], "level": r["level"],
                         "repeat": r["repeat"], "entrant": r["model"] + (f"@{r['level']}" if r["level"] else ""),
                         "label": label, "maker": maker, "kind": kind, "seconds": r.get("seconds"),
                         "cost_usd": r.get("cost_usd"), "readable": bool(r.get("readable")),
                         "score": f1(clicks, key["tasks"][r["task"]]["targets"])})
    variants = {"language-models": [r for r in rows if r["kind"] == "general"]}
    if any(r["kind"] == "specialist" for r in rows):
        variants["with-specialists"] = rows
    for variant, vrows in variants.items():
        out = ROOT / "charts" / name / variant
        out.mkdir(parents=True, exist_ok=True)
        calls = out / "calls.jsonl"
        calls.write_text("".join(json.dumps(r) + "\n" for r in vrows))
        here = {r["entrant"] for r in vrows}
        spots = [f"--label-spot={e}={s}" for e, s in LABEL_SPOTS.get(name, {}).items() if e in here]
        subprocess.run(["uv", "run", str(ROOT / "scripts/draw_charts.py"), str(calls), "--style", "editorial",
                        "--track", "find", "--names", "all", "--title", title, "--out", str(out), *spots],
                       cwd=ROOT, check=True)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    (ROOT / "charts" / name / "manifest.json").write_text(json.dumps({
        "sources": {str(r.relative_to(ROOT)): digest(r) for r in runs},
        "key": str(key_path.relative_to(ROOT)), "key_sha256": digest(key_path), "key_status": key["status"],
        "renderer": "scripts/draw_charts.py --style editorial --track find --names all --title ...",
        "label_spots": LABEL_SPOTS.get(name, {}),
        "metric": "F1 of clicks on the key's targets (one click per target), mean over tries, then over the questions",
        "caveats": ["Five screenshots cannot support broad claims.",
                    "Language-model seconds include the internet and the provider's queue; local models' do not.",
                    "Local models: cost uses each model's load factor measured on the photos, not on screenshots.",
                    "Local models timed on GPU 0 while it also drew the desktop (window manager, 20-30% busy); "
                    "Moondream 3.1 (added 2026-10-11) on the whole of GPU 1, the same card model, with nothing else on it.",
                    "The smooth curve is a visual guide, not measured intermediate models."],
    }, indent=2) + "\n")


if __name__ == "__main__":
    # plot_screens.py KEY NAME "TITLE" RUN [RUN ...]
    main([Path(a).resolve() for a in sys.argv[4:]], Path(sys.argv[1]).resolve(), sys.argv[2], sys.argv[3])
