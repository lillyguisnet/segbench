"""Jev + halving: the run's cut-line join with its "touches the cut" tolerance in photo pixels (2026-10-16).

    uv run --with pillow scripts/jev_join_photo_tol.py  ->  results/point-benchmark-v3-jev-join-cut-photo-tol/

Why: on the dishes photo (try 0) the left plate's parts were not joined: one
part stopped 8 px short of the cut at x = 500, and the run's tolerance is 2%
of the piece (5 px for a 250 px piece). Here the tolerance is 0.5% of the
photo's side (20 px on a 4000 px photo), everything else as the run's join
(segbench/jev_halving.join_across_cuts), on the halving's boxes (no recheck).
Found by looking at a test photo: exploratory, not a candidate rule.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from jev_merge_variants import REFERENCES, TASKS  # noqa: E402

TOL = 0.005


def join_px(found, W, H, tol=TOL, min_along=0.5):
    n = len(found)
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ra, rb, a, b = found[i]["region"], found[j]["region"], found[i]["box"], found[j]["box"]
            for k, size in ((0, W), (1, H)):
                pos, o = ra[k + 2], 1 - k
                if rb[k] != pos or min(ra[o + 2], rb[o + 2]) <= max(ra[o], rb[o]):
                    continue
                if a[k + 2] < pos - tol * size or b[k] > pos + tol * size:
                    continue
                along = min(a[o + 2], b[o + 2]) - max(a[o], b[o])
                shorter = min(a[o + 2] - a[o], b[o + 2] - b[o])
                if shorter > 0 and along / shorter >= min_along:
                    parent[root(i)] = root(j)
    g = {}
    for i in range(n):
        g.setdefault(root(i), []).append(found[i]["box"])
    return [[min(x[0] for x in v), min(x[1] for x in v), max(x[2] for x in v), max(x[3] for x in v)] for v in g.values()]


def main():
    trees = {(t["task"], t["repeat"]): t for t in map(json.loads, open(ROOT / "results/run-jev-halving-1.trees.jsonl"))}
    out = []
    for r in map(json.loads, open(ROOT / "results/run-jev-halving-1.jsonl")):
        t = trees[(r["task"], r["repeat"])]
        W, H = t["image_size"]
        fr = next((i for i, n in enumerate(t["nodes"]) if n.get("recheck")), len(t["nodes"]))
        hv = t["nodes"][:fr]
        items = []
        for b in join_px(t["raw"], W, H):
            g = [round(min(1000, max(0, v)), 1) for v in (b[0] * 1000 / W, b[1] * 1000 / H, b[2] * 1000 / W, b[3] * 1000 / H)]
            items.append({"box": g, "point": [round((g[0] + g[2]) / 2, 1), round((g[1] + g[3]) / 2, 1)]})
        out.append(dict(r, model="jev-join-px", merge_variant="join-cut-photo-tol", answer={"items": items},
                        text=json.dumps({"objects": [{"x": i["point"][0], "y": i["point"][1]} for i in items]}),
                        requests=len(hv), cost_usd=sum(n["cost_usd"] or 0 for n in hv),
                        seconds=round(r["seconds"] * len(hv) / max(1, len(t["nodes"])), 3)))
    p = ROOT / "results/run-jev-halving-1-merge-join-cut-photo-tol.jsonl"
    p.write_text("".join(json.dumps(v) + "\n" for v in out))
    subprocess.run(["uv", "run", "scripts/score_points.py", "--parser", "lenient", "--dish-labels", "separate",
                    "--method", "docs/point-benchmark-v2.md", "--reference", *REFERENCES, "--tasks", *TASKS,
                    "--timing-tasks", "logs", "cows", "fig", "dishes", "--inputs", str(p),
                    "--out", str(ROOT / "results/point-benchmark-v3-jev-join-cut-photo-tol")], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
