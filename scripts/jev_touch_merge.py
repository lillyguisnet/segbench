"""Jev + halving: merge boxes whose sides touch (Maxime, 2026-10-16), from the saved run, no new request.

    uv run --with pillow scripts/jev_touch_merge.py [run name]   (default jev-halving-1)
    ->  results/point-benchmark-v3-jev-touch-<rule>-<pct>/

The halving cuts pieces side by side (never overlapping), so two parts of one
object can only meet along a cut, never overlap. Rule, written before any
score was seen, applied to the halving's boxes ("parts": nothing joined, no
recheck; only the halving's requests are counted):

Two boxes touch when a side of one faces a side of the other (left/right or
top/bottom) with a gap of at most TOL of the photo's size along that axis
(slight overlap counts as touching), and the two facing sides share a length.
They are merged when that shared length is at least PCT of
- "either": the shorter of the two facing sides (one box's side is mostly
  against the other), or
- "both": the longer of the two (both boxes' sides are mostly against each other).
Merged boxes are grouped (an object cut in four joins through its parts)
and each group becomes the rectangle around it.

PCT in 25, 50, 75, 90; TOL = 0.5% of the photo (20 px across the 4000 px
photos; Jev's edges come in 1% steps of a piece, and a piece is at most the
whole photo). All eight are reported; picking one by its score would tune on
the test photos.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from jev_merge_variants import REFERENCES, TASKS  # noqa: E402

TOL = 0.005
PCTS = (25, 50, 75, 90)
RULES = ("either", "both")


def touch_merge(boxes, W, H, pct, rule):
    n = len(boxes)
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            a, b = boxes[i], boxes[j]
            for k, size in ((0, W), (1, H)):  # k = 0: sides facing left/right; 1: top/bottom
                o = 1 - k
                gap = min(abs(a[k + 2] - b[k]), abs(b[k + 2] - a[k]))
                if gap > TOL * size:
                    continue
                shared = min(a[o + 2], b[o + 2]) - max(a[o], b[o])
                la, lb = a[o + 2] - a[o], b[o + 2] - b[o]
                denom = min(la, lb) if rule == "either" else max(la, lb)
                if shared > 0 and denom > 0 and shared / denom >= pct / 100:
                    parent[root(i)] = root(j)
    groups = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(boxes[i])
    return [[min(x[0] for x in g), min(x[1] for x in g), max(x[2] for x in g), max(x[3] for x in g)]
            for g in groups.values()]


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "jev-halving-1"
    trees = {(t["task"], t["repeat"]): t for t in map(json.loads, (ROOT / f"results/run-{run}.trees.jsonl").read_text().splitlines())}
    records = [json.loads(l) for l in (ROOT / f"results/run-{run}.jsonl").read_text().splitlines()]
    for rule in RULES:
        for pct in PCTS:
            name = f"touch-{rule}-{pct}"
            dest = ROOT / f"results/point-benchmark-v3-jev-{name}"
            if dest.exists():
                sys.exit(f"{dest} exists: move it away first")
            out = []
            for r in records:
                t = trees[(r["task"], r["repeat"])]
                W, H = t["image_size"]
                first_recheck = next((i for i, n in enumerate(t["nodes"]) if n.get("recheck")), len(t["nodes"]))
                halving = t["nodes"][:first_recheck]
                items = []
                for b in touch_merge([f["box"] for f in t["raw"]], W, H, pct, rule):
                    g = [round(min(1000, max(0, v)), 1) for v in (b[0] * 1000 / W, b[1] * 1000 / H, b[2] * 1000 / W, b[3] * 1000 / H)]
                    items.append({"box": g, "point": [round((g[0] + g[2]) / 2, 1), round((g[1] + g[3]) / 2, 1)]})
                out.append(dict(r, model=f"jev-{name}", merge_variant=name, answer={"items": items},
                                text=json.dumps({"objects": [{"x": i["point"][0], "y": i["point"][1]} for i in items]}),
                                requests=len(halving), cost_usd=sum(n["cost_usd"] or 0 for n in halving),
                                seconds=round(r["seconds"] * len(halving) / max(1, len(t["nodes"])), 3)))
            path = ROOT / f"results/run-{run}-merge-{name}.jsonl"
            path.write_text("".join(json.dumps(v) + "\n" for v in out))
            subprocess.run(["uv", "run", "scripts/score_points.py", "--parser", "lenient", "--dish-labels", "separate",
                            "--method", "docs/point-benchmark-v2.md", "--reference", *REFERENCES, "--tasks", *TASKS,
                            "--timing-tasks", "logs", "cows", "fig", "dishes", "--inputs", str(path), "--out", str(dest)],
                           cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
            print(dest.name, flush=True)


if __name__ == "__main__":
    main()
