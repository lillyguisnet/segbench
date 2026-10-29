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
from segbench.perceptron import read_tag_answer
from segbench.point_score import (VERSION, TOLERANCES, DIAGONAL_CAP, read_points, reference_distances, score,
                                  timed_on_free_gpu)
from segbench.chart_data import describe

TASKS = ('logs', 'cows', 'fig', 'dishes')


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def mean_known(values):
    return statistics.mean(values) if values and all(v is not None for v in values) else None


def build(references, inputs, out, parser='strict', method='docs/point-benchmark-v1.md', dish_labels='required',
          tasks=TASKS, timing_tasks=None):
    # Several reference files (2026-10-14): the second photo batch has its own
    # review (annotations/reviews/segbench-review-5682a4be-approved.json); each
    # task is read from the one file that has it.
    if isinstance(references, Path):
        references = [references]
    tasks = tuple(tasks)
    # Cost and seconds from these tasks only (default: all). v3 (2026-10-14):
    # the original four, for every entrant alike: local models were timed on
    # those photos only, and the second batch's photos are ~10x smaller.
    timing_tasks = tuple(timing_tasks or tasks)
    raws, seed = {}, {}
    for reference in references:
        raw = json.loads(reference.read_text())
        if raw['schema'] != 'segbench-consensus-review/v1':
            raise ValueError('Unsupported reference format')
        for t in raw['source_snapshot']['tasks']:
            if t['key'] in raws:
                raise ValueError(f"{t['key']} is in more than one reference file")
            raws[t['key']], seed[t['key']] = (raw, reference), t
    missing = [k for k in tasks if k not in raws]
    if missing:
        raise ValueError(f'no reference for tasks {missing}')
    ref_hash = {k: digest(raws[k][1]) for k in tasks}
    refs, task_meta = {}, {}
    for k in tasks:
        t = raws[k][0]['tasks'][k]
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
            if r.get('track') != 'find' or r.get('task') not in tasks or r.get('parts'):
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
                        'metric': VERSION, 'parser': parser, 'reference_sha256': ref_hash[k], 'source_file': path.name,
                        'source_line': line_number, 'source_sha256': source_hash,
                        'timing_provisional': kind == 'specialist' and not timed_on_free_gpu(r),
                        'timing_trustworthy': r.get('timing_trustworthy'), 'cost_method': r.get('cost_method'),
                        'gpu_others': r.get('gpu_others'), 'predictions': []})
            try:
                if r.get('error'):
                    raise ValueError(r['error'])
                if r.get('prompt_version') == 'tags':  # Perceptron's own tags (segbench/perceptron.py)
                    points, how = read_tag_answer(r.get('text', ''), ('dirty', 'clean') if k == 'dishes' else ())
                else:
                    points, how = read_points(r.get('text', ''), labelled=k == 'dishes', parser=parser)
                need_label = k == 'dishes' and dish_labels == 'required'
                scored = score(points, refs[k], task_meta[k]['width'], task_meta[k]['height'], labelled=need_label)
                if k == 'dishes' and dish_labels == 'separate':
                    # dirty/clean right among the dishes found by location; not part of the score
                    ref_label = {o['id']: o['label'] for o in refs[k]}
                    scored['label_accuracy'] = {
                        lvl: (sum(points[m['prediction']].get('label') == ref_label[m['reference_id']]
                                  for m in v['matches']) / len(v['matches']) if v['matches'] else None)
                        for lvl, v in scored['tolerances'].items()}
                rec.update(scored)
                rec.update({'readable': True, 'read_as': how, 'predictions': points,
                            'predicted_labels': dict(Counter(p['label'] for p in points)) if k == 'dishes' else {}})
            except (ValueError, KeyError, TypeError) as e:
                rec.update({'score': 0, 'readable': False, 'error': str(e), 'predicted_count': None,
                            'count_error': None, 'reference_count': len(refs[k]), 'predicted_labels': {},
                            'tolerances': {level: {'f1': 0, 'matches': []} for level in TOLERANCES}})
            if k not in timing_tasks:
                rec['seconds_measured'], rec['cost_usd_measured'] = rec['seconds'], rec['cost_usd']
                rec['seconds'] = rec['cost_usd'] = None
                rec['timing_note'] = f'not used: cost and seconds come from tasks {list(timing_tasks)}'
            records.append(rec)
    groups = defaultdict(list)
    for r in records:
        groups[r['entrant']].append(r)
    summary = []
    for key, rows in groups.items():
        by_task = {k: [r for r in rows if r['task'] == k] for k in tasks}
        full = len(tasks)
        cover = sum(bool(rs) for rs in by_task.values())
        levels = {level: statistics.mean(statistics.mean(r['tolerances'][level]['f1'] for r in rs) for rs in by_task.values())
                  if cover == full else None for level in TOLERANCES}
        timed = [r for r in rows if r['task'] in timing_tasks]
        times = [r['seconds'] for r in timed]
        summary.append({'entrant': key, 'label': rows[0]['label'], 'maker': rows[0]['maker'], 'kind': rows[0]['kind'],
                        'level': rows[0]['level'], 'coverage': cover, 'calls': len(rows),
                        'task_scores': {k: statistics.mean(r['score'] for r in rs) if rs else None for k, rs in by_task.items()},
                        'tolerances': levels, 'score': statistics.mean(levels.values()) if cover == full else None,
                        'seconds': statistics.median(times) if all(v is not None for v in times) else None,
                        'cost_usd': mean_known([r['cost_usd'] for r in timed]),
                        'timing_provisional': any(r['timing_provisional'] for r in rows),
                        'unreadable': sum(not r['readable'] for r in rows),
                        'recovered': sum(r.get('read_as') == 'recovered' for r in rows)})
    for s in summary:
        ranks = [1 + sum(other['tolerances'][level] > s['tolerances'][level] + 1e-12 for other in summary if other['coverage'] == len(tasks))
                 for level in TOLERANCES] if s['coverage'] == len(tasks) else []
        s['rank_range'] = [min(ranks), max(ranks)] if ranks else None
    summary.sort(key=lambda s: -(s['score'] if s['score'] is not None else -1))
    manifest = {'metric': VERSION, 'parser': parser, 'dish_labels': dish_labels, 'created_at': datetime.now(timezone.utc).isoformat(),
                **({'reference': references[0].name, 'reference_sha256': digest(references[0])} if len(references) == 1 else
                   {'references': {r.name: digest(r) for r in references}}),
                'tasks': list(tasks), 'timing_tasks': list(timing_tasks),
                'input_files': {p.name: digest(p) for p in inputs}, 'tolerances': TOLERANCES,
                'diagonal_cap': DIAGONAL_CAP, 'scope': 'Point localisation only. Not point-inside-object or segmentation.',
                'aggregation': f'Equal mean over three F1 tolerances, then {len(tasks)} tasks. Complete coverage required.',
                'specification': method, 'specification_sha256': digest(ROOT / method),
                'scorer_sha256': digest(ROOT / 'segbench/point_score.py')}
    out.mkdir(parents=True, exist_ok=False)
    (out / 'images').mkdir()
    for k in tasks:
        name = Path(task_meta[k]['photo']).name
        shutil.copyfile(ROOT / 'images' / name, out / 'images' / name)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (out / 'scored.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    payload = {'manifest': manifest, 'tasks': task_meta, 'summary': summary, 'records': records}
    (out / 'data.js').write_text('window.POINT_REPORT=' + json.dumps(payload, separators=(',', ':')) + ';\n')
    shutil.copyfile(ROOT / 'scripts/point_report.html', out / 'index.html')
    shutil.copyfile(ROOT / method, out / 'method.md')
    fields = ['entrant', 'label', 'level', 'coverage', 'score', *tasks, *TOLERANCES, 'seconds', 'cost_usd', 'timing_provisional', 'unreadable', 'recovered']
    with (out / 'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields);writer.writeheader()
        for s in summary:
            row = {k: s.get(k) for k in fields};row.update(s['task_scores']);row.update(s['tolerances']);writer.writerow(row)
    with (out / 'per-task.csv').open('w') as f:
        fields = ['entrant', 'task', 'repeat', 'score', 'predicted_count', 'reference_count', 'count_error', 'readable', 'read_as', 'error', *TOLERANCES]
        writer = csv.DictWriter(f, fieldnames=fields);writer.writeheader()
        for r in records:
            row = {k: r.get(k) for k in fields};row.update({level:r['tolerances'][level]['f1'] for level in TOLERANCES});writer.writerow(row)
    for s in summary:
        val=f"{s['score']*100:.1f}" if s['score'] is not None else 'partial'
        print(f"{s['label']:22} {val:>7}  coverage {s['coverage']}/{len(tasks)}  ranks {s['rank_range']}  unreadable {s['unreadable']}")
    print('Wrote', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference', type=Path, nargs='+', default=[ROOT / 'annotations/reviews/segbench-review-6187f623-approved.json'])
    ap.add_argument('--tasks', nargs='+', default=list(TASKS), help=f'task keys (default: the published four, {TASKS})')
    ap.add_argument('--timing-tasks', nargs='+', help='cost and seconds from these tasks only (default: all scored tasks)')
    ap.add_argument('--inputs', nargs='+', type=Path, default=[ROOT / 'results/run-pilot-1.jsonl', ROOT / 'results/run-specialists-1.jsonl'])
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--parser', choices=('strict', 'lenient'), default='strict',
                    help='lenient: if strict JSON fails, recover [x, y] pairs mechanically (see point_score.recover_points)')
    ap.add_argument('--method', default='docs/point-benchmark-v1.md', help='method document copied into the report')
    ap.add_argument('--dish-labels', choices=('required', 'separate'), default='required',
                    help='separate: dishes scored on location alone; dirty/clean accuracy reported apart')
    args = ap.parse_args()
    build(args.reference, args.inputs, args.out, args.parser, args.method, args.dish_labels, args.tasks, args.timing_tasks)
