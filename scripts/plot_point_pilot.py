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
import sys
RUN = sys.argv[1] if len(sys.argv) > 1 else 'point-benchmark-v1.2'
SOURCE = ROOT / 'results' / RUN / 'scored.jsonl'
SUMMARY = ROOT / 'results' / RUN / 'summary.csv'
OUT = ROOT / 'charts' / (sys.argv[2] if len(sys.argv) > 2 else 'point-pilot-editorial')

# Labels pinned beside their disc, per chart, as in scripts/plot_screens.py
# (keys "<chart>" or "<chart>/<variant>"); the automatic placer puts the rest
# around them. Charts not listed are drawn exactly as before.
# 2026-10-14, v2.7: Gemini 3.1 Pro's left of its disc. Left to the placer, it
# went far to the right with a line through Claude Opus's and Mistral Large
# 4's labels, and Gemini 3.8 Flash's (whose disc sits under Gemini 4 Argon's)
# crossed GPT-6.1 Sol's. On the language-model chart, Qwen 3.8 27B's left of
# its disc (it ran into Claude Haiku 5.5's; labels on the best-value line,
# like Haiku's, cannot be pinned).
# v3 (2026-10-14): with Muse Wonder (API) added, the placer sent MolmoPoint's,
# Gemini 3.5 Flash Lite's, Gemini 4 Argon's and Claude Sonnet's labels far
# away on long crossing lines; pinned beside their discs (positions chosen so
# the pinned labels do not cover each other: pins are not checked against
# one another, only the automatic labels avoid them).
LABEL_SPOTS = {'point-benchmark-v3': {'molmopoint-8b': 'll', 'gemini-flash-lite@medium': 'below',
                                      'gemini-argon@medium': 'lr', 'sonnet@medium': 'lr'},
               'point-benchmark-v2.7': {'gemini-pro@medium': 'left'},
               'point-benchmark-v2.7/language-models': {'qwen-27b@medium': 'left'},
               'point-benchmark-v2.8': {'gemini-pro@medium': 'left'},
               'point-benchmark-v2.8/language-models': {'qwen-27b@medium': 'left'}}


def main():
    records = [json.loads(line) for line in SOURCE.read_text().splitlines()]
    with SUMMARY.open() as f:
        expected = {r['entrant']: r for r in csv.DictReader(f)}
    n_tasks = len(json.loads((SOURCE.parent / 'manifest.json').read_text()).get('tasks') or [0] * 4)
    complete = {k for k, r in expected.items() if int(r['coverage']) == n_tasks}
    variants = {
        'language-models': [r for r in records if r['entrant'] in complete and r['kind'] == 'general'],
        'with-specialists': [r for r in records if r['entrant'] in complete],
    }
    for variant, rows in variants.items():
        folder = OUT / variant
        folder.mkdir(parents=True, exist_ok=True)
        calls = folder / 'calls.jsonl'
        calls.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        here = {r['entrant'] for r in rows}
        pinned = {**LABEL_SPOTS.get(OUT.name, {}), **LABEL_SPOTS.get(f'{OUT.name}/{variant}', {})}
        spots = [f'--label-spot={e}={s}' for e, s in pinned.items() if e in here]
        subprocess.run(['uv', 'run', str(ROOT / 'scripts/draw_charts.py'), str(calls),
                        '--style', 'editorial', '--track', 'find', '--names', 'all', '--out', str(folder), *spots],
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
        **({'label_spots': spots} if (spots := {k: v for k, v in LABEL_SPOTS.items()
                                                 if k == OUT.name or k.startswith(OUT.name + '/')}) else {}),
        'metric': f'point-localisation-v1; tolerance-average F1; equal weighting of {n_tasks} tasks',
        'excluded_incomplete': [k for k in expected if k not in complete],
        'caveats': ['Single medium-thinking pilot, not repeat-validated.',
                    'Reference-point proximity is not point-inside-mask correctness.',
                    'Specialist cost estimates and shared-GPU timings are provisional.',
                    'The smooth curve is a visual guide, not measured intermediate models.'],
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
