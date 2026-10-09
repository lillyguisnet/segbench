# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9"]
# ///
"""Draw the three bubble charts (one per track) from call records.

    uv run scripts/draw_charts.py --demo                 # made-up data, to work on the look
    uv run scripts/draw_charts.py results/run-*.jsonl    # real call records

Writes to charts/ (charts/demo/ with --demo), for each track:
    <track>.png   to post (3200 x 2000 pixels)
    <track>.pdf   to print (vector, fonts embedded)
    <track>.svg   for the web (vector)
    points.csv    the numbers behind every bubble, all tracks

The three charts share one cost axis, so a bubble's position can be
compared between them. What a call record must hold, and how a bubble is
computed: segbench/chart_data.py.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # this script runs in its own environment (above), not the project's

from segbench.chart import draw, x_range  # noqa: E402
from segbench.chart_data import TRACKS, aggregate, read_calls, write_points  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("calls", nargs="*", type=Path, help="JSON Lines files of call records")
    ap.add_argument("--demo", action="store_true", help="make up demo data first and draw it")
    ap.add_argument("--out", type=Path, help="output folder (default charts/, or charts/demo/ with --demo)")
    ap.add_argument("--track", choices=TRACKS, action="append", help="only this track (repeatable)")
    ap.add_argument("--note", default="", help="one more footnote line, e.g. the run's date and photo count")
    args = ap.parse_args()

    out = args.out or ROOT / "charts" / ("demo" if args.demo else "")
    calls_files = args.calls
    if args.demo:
        demo = out / "calls.jsonl"
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_demo_calls.py"), str(demo)], check=True)
        calls_files = [demo]
    if not calls_files:
        ap.error("give call-record files, or --demo")

    calls = read_calls(calls_files)
    points = aggregate(calls)
    if not points:
        sys.exit("no scored calls found in those files")
    write_points(points, out / "points.csv")
    xlim = x_range(points)  # one cost axis for all three charts
    days = sorted(r["started_at"][:10] for r in calls if r.get("started_at"))
    stamp = (days[0] if days[0] == days[-1] else f"{days[0]} to {days[-1]}") if days else ""
    for track in args.track or TRACKS:
        if not any(p.track == track for p in points):
            print(f"{track}: no data, skipped")
            continue
        for path in draw(points, track, out / track, xlim=xlim, footnote=args.note, stamp=stamp):
            print(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


if __name__ == "__main__":
    main()
