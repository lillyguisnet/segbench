"""How merging changes Jev + halving's score (Maxime, 2026-10-16), from the saved run: no new request.

    uv run --with pillow scripts/jev_merge_variants.py [run name]   (default jev-halving-1)
    ->  results/run-<run>-merge-<variant>.jsonl and results/point-benchmark-v3-jev-merge-<variant>/

Variants, written down before any of them was scored:
- parts:          the boxes the halving found, nothing joined;
- joined:         parts joined across cut lines with the run's rule
                  (segbench/jev_halving.join_across_cuts: both touch the same
                  cut line within 2% of their piece, spans along it overlap
                  >= half the shorter), no recheck;
- final:          what the run returned (joined, then rechecked with new
                  requests: "several" halves a joined box again);
- overlap:        parts merged when two boxes overlap by >= half of the
                  smaller one (the same object seen twice), until stable;
- joined+overlap: joined, then overlap.

Seconds and cost: the requests the variant needed. Every variant except
"final" uses only the halving's requests (those before the first recheck
request in the search's log); "final" uses them all. Their seconds are an
estimate (the search's waiting time x their share of its requests). Scored as v3
(scripts/score_points.py, lenient, both references, cost and seconds from the
first four tasks). Chosen with knowledge of the run's pictures but not of any
variant's score; still, this is the test set: a merge rule picked from these
scores would be tuned on it.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from segbench.jev_halving import join_across_cuts  # noqa: E402

TASKS = ["logs", "cows", "fig", "dishes", "ants", "beetles", "birds", "mushrooms", "jars", "pinkleaves", "whitecow"]
REFERENCES = ["annotations/reviews/segbench-review-6187f623-approved.json",
              "annotations/reviews/segbench-review-5682a4be-approved.json"]


def overlap_merge(boxes, share=0.5):
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
                small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
                if small > 0 and inter >= share * small:
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del boxes[j]
                    changed = True
                    break
            if changed:
                break
    return boxes


def variants(tree, record):
    raw = tree["raw"]
    parts = [f["box"] for f in raw]
    joined = join_across_cuts(raw)
    W, H = tree["image_size"]
    final = [[i["box"][0] * W / 1000, i["box"][1] * H / 1000, i["box"][2] * W / 1000, i["box"][3] * H / 1000]
             for i in record["answer"]["items"]]
    return {"parts": parts, "joined": joined, "final": final, "overlap": overlap_merge(parts),
            "joined+overlap": overlap_merge(joined)}


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "jev-halving-1"
    trees = {(t["task"], t["repeat"]): t for t in map(json.loads, (ROOT / f"results/run-{run}.trees.jsonl").read_text().splitlines())}
    records = [json.loads(l) for l in (ROOT / f"results/run-{run}.jsonl").read_text().splitlines()]
    out = {}
    for r in records:
        t = trees[(r["task"], r["repeat"])]
        W, H = t["image_size"]
        first_recheck = next((i for i, n in enumerate(t["nodes"]) if n.get("recheck")), len(t["nodes"]))
        halving = t["nodes"][:first_recheck]
        for name, boxes in variants(t, r).items():
            items = []
            for b in boxes:
                g = [round(min(1000, max(0, v)), 1) for v in (b[0] * 1000 / W, b[1] * 1000 / H, b[2] * 1000 / W, b[3] * 1000 / H)]
                items.append({"box": g, "point": [round((g[0] + g[2]) / 2, 1), round((g[1] + g[3]) / 2, 1)]})
            v = dict(r, model=f"jev-merge-{name}", answer={"items": items}, merge_variant=name,
                     text=json.dumps({"objects": [{"x": i["point"][0], "y": i["point"][1]} for i in items]}))
            if name != "final":
                # seconds: an estimate, the search's waiting time in proportion to the requests
                # (the run timed the whole search, not its halving phase alone)
                v.update(requests=len(halving), cost_usd=sum(n["cost_usd"] or 0 for n in halving),
                         seconds=round(r["seconds"] * len(halving) / max(1, len(t["nodes"])), 3))
            out.setdefault(name, []).append(v)
    for name, recs in out.items():
        safe = name.replace("+", "-")
        path = ROOT / f"results/run-{run}-merge-{safe}.jsonl"
        path.write_text("".join(json.dumps(v) + "\n" for v in recs))
        dest = ROOT / f"results/point-benchmark-v3-jev-merge-{safe}"
        if dest.exists():
            sys.exit(f"{dest} exists: move it away first")
        subprocess.run(["uv", "run", "scripts/score_points.py", "--parser", "lenient", "--dish-labels", "separate",
                        "--method", "docs/point-benchmark-v2.md", "--reference", *REFERENCES, "--tasks", *TASKS,
                        "--timing-tasks", "logs", "cows", "fig", "dishes", "--inputs", str(path), "--out", str(dest)],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        print(dest.name)


if __name__ == "__main__":
    main()
