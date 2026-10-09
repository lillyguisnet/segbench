"""Table of the specialist-model measurements (latest record per model).

uv run scripts/summarize_specialists.py [results/specialists/DATE.jsonl ...]

Cost uses the best throughput found: largest useful batch, or several
workers sharing the GPU when that was faster.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    files = sys.argv[1:] or sorted(glob.glob(str(ROOT / "results" / "specialists" / "*.jsonl")))
    runs, concurrent = {}, {}
    for f in files:
        for line in open(f):
            r = json.loads(line)
            if r.get("kind") == "concurrent_throughput":
                best = concurrent.get(r["model_id"])
                if best is None or r["images_per_s"] > best["images_per_s"]:
                    concurrent[r["model_id"]] = r
            else:
                runs[r["model_id"]] = r  # later lines win

    print("| model | latency, 1 image | best throughput | how | GPU busy | $ per 1,000 images | trustworthy timing |")
    print("|---|---|---|---|---|---|---|")
    rows = []
    for mid, r in runs.items():
        best = r["throughput"]["best"]
        ips, how, util = best["images_per_s"], f"batch {best['batch_size']}", best.get("util_mean")
        c = concurrent.get(mid)
        if c and c["images_per_s"] > ips:
            ips, how, util = c["images_per_s"], f"{c['workers']} workers", c.get("util_mean")
        usd = r["cost"]["usd_per_gpu_hour"] / 3600 / ips
        rows.append((usd, f"| {mid} | {r['latency_s']['median']:.3f} s | {ips:.2f} img/s | {how} | "
                          f"{util}% | ${usd * 1000:.4f} | {'yes' if r['gpu']['timing_trustworthy'] else 'NO'} |"))
    for _, row in sorted(rows):
        print(row)
    price = next(iter(runs.values()))["cost"] if runs else {}
    print(f"\nPrice: ${price.get('usd_per_gpu_hour')}/GPU-hour ({price.get('price_source')}).")


if __name__ == "__main__":
    main()
