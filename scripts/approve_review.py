"""Record explicit human approval of visible proposals on fully checked photos.

This is not automatic consensus acceptance. Run only after a person has
confirmed that visible, non-rejected dots on checked photos are approved.
Hidden low-support proposals and rejected objects remain unchanged. Neither
this operation nor full-photo approval creates full-resolution object masks.

python scripts/approve_review.py INPUT OUTPUT --confirm-visible-approved

The original export is never overwritten. Approval uses the review app's
normal visibility threshold (support >= 0.2), not the consensus 0.5 cutoff.
Temporary selections and the 'show low-support' display toggle are not saved
in the export and are not used. Objects already accepted remain accepted.
"""
import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def approve(data, timestamp):
    if data.get('schema') != 'segbench-consensus-review/v1':
        raise ValueError('Unsupported review schema')
    seed = data['source_snapshot']
    if seed['source_sha256'] != data['source_sha256']:
        raise ValueError('Source snapshot mismatch')
    changed = {}
    for original in seed['tasks']:
        key = original['key']
        task = data['tasks'][key]
        if task['kind'] != 'objects' or not task['full_photo_checked']:
            continue
        sources = {o['id']: o for o in original['objects']}
        ids = []
        for obj in task['objects']:
            if obj['status'] not in ('pending', 'unsure'):
                continue
            # Unsure objects are visible regardless of model support.
            visible = obj['status'] == 'unsure' or sources.get(obj['id'], {}).get('support', 1) >= .2
            if not visible:
                continue
            if not (0 <= obj['x'] <= task['width'] and 0 <= obj['y'] <= task['height']):
                raise ValueError(f'Cannot approve out-of-photo object {obj["id"]}')
            obj['status'] = 'accepted'
            obj['edited_at'] = timestamp
            ids.append(obj['id'])
        changed[key] = ids
        data['audit'].append({'at': timestamp, 'task': key, 'action': 'human_bulk_approval',
                              'ids': ids, 'basis': 'Explicit user confirmation: yes all approved',
                              'scope': 'Visible non-rejected dots on fully checked photos; hidden low-support proposals excluded'})
    data['updated_at'] = timestamp
    data['exported_at'] = timestamp
    return changed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--confirm-visible-approved', action='store_true', required=True)
    args = ap.parse_args()
    raw = args.input.read_bytes()
    data = json.loads(raw)
    changed = approve(data, datetime.now(timezone.utc).isoformat())
    data['approval_provenance'] = {'original_export': args.input.name,
                                   'original_export_sha256': hashlib.sha256(raw).hexdigest(),
                                   'original_export_preserved': True}
    with args.output.open('x') as f:
        json.dump(data, f, indent=2)
        f.write('\n')
    for key, task in data['tasks'].items():
        if task['kind'] == 'objects':
            accepted = [o for o in task['objects'] if o['status'] == 'accepted']
            labels = dict(Counter(o['label'] for o in accepted)) if key == 'dishes' else {}
            print(key, len(accepted), 'accepted;', len(changed.get(key, [])), 'newly approved;', labels)
        else:
            print(key, 'region unchanged; checked:', task['full_photo_checked'])
    assert args.input.read_bytes() == raw
    print('Original export unchanged; wrote', args.output)


if __name__ == '__main__':
    main()
