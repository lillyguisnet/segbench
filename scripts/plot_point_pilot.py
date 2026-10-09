"""Render the point pilot with the project's existing editorial plotting code.

python scripts/plot_point_pilot.py

Does not modify the shared renderer. Filters incomplete entrants before the
old aggregator can turn unattempted tasks into zeroes. Writes one API-only
chart and one including specialists' explicitly provisional cost estimates.
Calls uv run scripts/draw_charts.py for both, then checks its aggregates
against the original point report (quality, cost and median time).
"""
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/point-benchmark-v1.2/scored.jsonl'
SUMMARY = ROOT / 'results/point-benchmark-v1.2/summary.csv'
OUT = ROOT / 'charts/point-pilot-editorial'


def main():
    records = [json.loads(line) for line in SOURCE.read_text().splitlines()]
    with SUMMARY.open() as f:
        expected = {r['entrant']: r for r in csv.DictReader(f)}
    complete = {k for k, r in expected.items() if int(r['coverage']) == 4}
    variants = {
        'language-models': [r for r in records if r['entrant'] in complete and r['kind'] == 'general'],
        'with-specialists': [r for r in records if r['entrant'] in complete],
    }
    for variant, rows in variants.items():
        folder = OUT / variant
        folder.mkdir(parents=True, exist_ok=True)
        calls = folder / 'calls.jsonl'
        calls.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        subprocess.run(['uv', 'run', str(ROOT / 'scripts/draw_charts.py'), str(calls),
                        '--style', 'editorial', '--track', 'find', '--names', 'all', '--out', str(folder)],
                       cwd=ROOT, check=True)
        with (folder / 'points.csv').open() as f:
            plotted = list(csv.DictReader(f))
        assert {r['entrant'] for r in plotted} == {r['entrant'] for r in rows}
        for point in plotted:
            ref = expected[point['entrant']]
            for actual, want in [('quality', 'score'), ('cost_usd', 'cost_usd'), ('seconds', 'seconds')]:
                assert math.isclose(float(point[actual]), float(ref[want]), abs_tol=1e-12), (variant, point['entrant'], actual)
        print(f'Checked {variant}: {len(plotted)} chart points match the scored report exactly.')
    manifest = {
        'source': str(SOURCE.relative_to(ROOT)), 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'renderer': 'scripts/draw_charts.py --style editorial --track find --names all',
        'renderer_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                            for name in ['scripts/draw_charts.py', 'segbench/chart_editorial.py', 'segbench/chart.py', 'segbench/chart_data.py']},
        'metric': 'point-localisation-v1; tolerance-average F1; equal weighting of four tasks',
        'excluded_incomplete': [k for k in expected if k not in complete],
        'caveats': ['Single medium-thinking pilot, not repeat-validated.',
                    'Reference-point proximity is not point-inside-mask correctness.',
                    'Specialist cost estimates and shared-GPU timings are provisional.',
                    'The smooth curve is a visual guide, not measured intermediate models.'],
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
