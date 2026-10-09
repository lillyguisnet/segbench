"""Score saved finder calls against approved human points; build a reviewable report.

python scripts/score_points.py --out results/point-benchmark-v1

No model calls. New output directory required, so an earlier report is never
silently overwritten. Input exports, replies and source images are untouched.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from segbench.point_score import VERSION, TOLERANCES, DIAGONAL_CAP, parse_points, reference_distances, score
from segbench.chart_data import describe

TASKS = ('logs', 'cows', 'fig', 'dishes')


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def mean_known(values):
    return statistics.mean(values) if values and all(v is not None for v in values) else None


def build(reference, inputs, out):
    raw = json.loads(reference.read_text())
    if raw['schema'] != 'segbench-consensus-review/v1':
        raise ValueError('Unsupported reference format')
    seed = {t['key']: t for t in raw['source_snapshot']['tasks']}
    refs, task_meta = {}, {}
    for k in TASKS:
        t = raw['tasks'][k]
        if t['kind'] != 'objects':
            raise ValueError(f'{k} is not an object task')
        accepted = [o for o in t['objects'] if o['status'] == 'accepted']
        if any(o.get('label') not in ('dirty', 'clean') for o in accepted) and k == 'dishes':
            raise ValueError('Unsure dish references need an explicit ignore-region policy before scoring')
        if not accepted:
            raise ValueError(f'{k}: no accepted references')
        image_path = ROOT / 'images' / Path(seed[k]['photo']).name
        if digest(image_path) != t['image_sha256'] or t['image_sha256'] != seed[k]['image_sha256']:
            raise ValueError(f'{k}: image hash mismatch')
        nn = reference_distances(accepted, t['width'], t['height'])
        refs[k] = accepted
        task_meta[k] = {key: t[key] for key in ('width', 'height', 'image_sha256')}
        task_meta[k].update({'photo': 'images/' + image_path.name, 'count': len(accepted),
                            'labels': dict(Counter(o['label'] for o in accepted)) if k == 'dishes' else {},
                            'refs': [{**{key: o[key] for key in ('id', 'x', 'y', 'label')},
                                      'radii': {level: min(f * d, DIAGONAL_CAP * math.hypot(t['width'], t['height']))
                                                for level, f in TOLERANCES.items()}} for o, d in zip(accepted, nn)]})
    records = []
    seen = set()
    for path in inputs:
        source_hash = digest(path)
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            r = json.loads(line)
            if r.get('track') != 'find' or r.get('task') not in TASKS or r.get('parts'):
                continue
            k = r['task']
            identity = (r['model'], r.get('level'), k, r.get('repeat', 0))
            if identity in seen:
                raise ValueError(f'Duplicate call identity {identity}; select attempts explicitly')
            seen.add(identity)
            if r.get('image_sha256') != task_meta[k]['image_sha256']:
                raise ValueError(f'{identity}: call image hash mismatch')
            label, maker, kind = describe(r['model'])
            key = r['model'] + ('@' + r['level'] if r.get('level') else '')
            rec = {field: r.get(field) for field in ('model', 'level', 'task', 'repeat', 'seconds', 'cost_usd', 'provider')}
            rec.update({'entrant': key, 'label': label, 'maker': maker, 'kind': kind, 'track': 'find',
                        'metric': VERSION, 'reference_sha256': digest(reference), 'source_file': path.name,
                        'source_line': line_number, 'source_sha256': source_hash,
                        'timing_provisional': kind == 'specialist',
                        'timing_trustworthy': r.get('timing_trustworthy'), 'cost_method': r.get('cost_method'),
                        'gpu_others': r.get('gpu_others'), 'predictions': []})
            try:
                if r.get('error'):
                    raise ValueError(r['error'])
                points = parse_points(r.get('text', ''), labelled=k == 'dishes')
                scored = score(points, refs[k], task_meta[k]['width'], task_meta[k]['height'], labelled=k == 'dishes')
                rec.update(scored)
                rec.update({'readable': True, 'predictions': points,
                            'predicted_labels': dict(Counter(p['label'] for p in points)) if k == 'dishes' else {}})
            except (ValueError, KeyError, TypeError) as e:
                rec.update({'score': 0, 'readable': False, 'error': str(e), 'predicted_count': None,
                            'count_error': None, 'reference_count': len(refs[k]), 'predicted_labels': {},
                            'tolerances': {level: {'f1': 0, 'matches': []} for level in TOLERANCES}})
            records.append(rec)
    groups = defaultdict(list)
    for r in records:
        groups[r['entrant']].append(r)
    summary = []
    for key, rows in groups.items():
        by_task = {k: [r for r in rows if r['task'] == k] for k in TASKS}
        cover = sum(bool(rs) for rs in by_task.values())
        levels = {level: statistics.mean(statistics.mean(r['tolerances'][level]['f1'] for r in rs) for rs in by_task.values())
                  if cover == 4 else None for level in TOLERANCES}
        times = [r['seconds'] for r in rows]
        summary.append({'entrant': key, 'label': rows[0]['label'], 'maker': rows[0]['maker'], 'kind': rows[0]['kind'],
                        'level': rows[0]['level'], 'coverage': cover, 'calls': len(rows),
                        'task_scores': {k: statistics.mean(r['score'] for r in rs) if rs else None for k, rs in by_task.items()},
                        'tolerances': levels, 'score': statistics.mean(levels.values()) if cover == 4 else None,
                        'seconds': statistics.median(times) if all(v is not None for v in times) else None,
                        'cost_usd': mean_known([r['cost_usd'] for r in rows]),
                        'timing_provisional': any(r['timing_provisional'] for r in rows),
                        'unreadable': sum(not r['readable'] for r in rows)})
    for s in summary:
        ranks = [1 + sum(other['tolerances'][level] > s['tolerances'][level] + 1e-12 for other in summary if other['coverage'] == 4)
                 for level in TOLERANCES] if s['coverage'] == 4 else []
        s['rank_range'] = [min(ranks), max(ranks)] if ranks else None
    summary.sort(key=lambda s: -(s['score'] if s['score'] is not None else -1))
    manifest = {'metric': VERSION, 'created_at': datetime.now(timezone.utc).isoformat(),
                'reference': reference.name, 'reference_sha256': digest(reference),
                'input_files': {p.name: digest(p) for p in inputs}, 'tolerances': TOLERANCES,
                'diagonal_cap': DIAGONAL_CAP, 'scope': 'Point localisation only. Not point-inside-object or segmentation.',
                'aggregation': 'Equal mean over three F1 tolerances, then four tasks. Complete coverage required.',
                'specification_sha256': digest(ROOT / 'docs/point-benchmark-v1.md'),
                'scorer_sha256': digest(ROOT / 'segbench/point_score.py')}
    out.mkdir(parents=True, exist_ok=False)
    (out / 'images').mkdir()
    for k in TASKS:
        name = Path(task_meta[k]['photo']).name
        shutil.copyfile(ROOT / 'images' / name, out / 'images' / name)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (out / 'scored.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    payload = {'manifest': manifest, 'tasks': task_meta, 'summary': summary, 'records': records}
    (out / 'data.js').write_text('window.POINT_REPORT=' + json.dumps(payload, separators=(',', ':')) + ';\n')
    shutil.copyfile(ROOT / 'scripts/point_report.html', out / 'index.html')
    shutil.copyfile(ROOT / 'docs/point-benchmark-v1.md', out / 'method.md')
    fields = ['entrant', 'label', 'level', 'coverage', 'score', *TASKS, *TOLERANCES, 'seconds', 'cost_usd', 'timing_provisional', 'unreadable']
    with (out / 'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields);writer.writeheader()
        for s in summary:
            row = {k: s.get(k) for k in fields};row.update(s['task_scores']);row.update(s['tolerances']);writer.writerow(row)
    with (out / 'per-task.csv').open('w') as f:
        fields = ['entrant', 'task', 'repeat', 'score', 'predicted_count', 'reference_count', 'count_error', 'readable', 'error', *TOLERANCES]
        writer = csv.DictWriter(f, fieldnames=fields);writer.writeheader()
        for r in records:
            row = {k: r.get(k) for k in fields};row.update({level:r['tolerances'][level]['f1'] for level in TOLERANCES});writer.writerow(row)
    for s in summary:
        val=f"{s['score']*100:.1f}" if s['score'] is not None else 'partial'
        print(f"{s['label']:22} {val:>7}  coverage {s['coverage']}/4  ranks {s['rank_range']}  unreadable {s['unreadable']}")
    print('Wrote', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference', type=Path, default=ROOT / 'annotations/reviews/segbench-review-6187f623-approved.json')
    ap.add_argument('--inputs', nargs='+', type=Path, default=[ROOT / 'results/run-pilot-1.jsonl', ROOT / 'results/run-specialists-1.jsonl'])
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    build(args.reference, args.inputs, args.out)
