"""Point-localisation pilot, not point-inside-mask scoring. See docs/point-benchmark-v1.md.

No model calls or third-party dependencies. Geometry uses original image
pixels. Maximum-cardinality matching prevents greedy-order scoring errors.
"""
from __future__ import annotations

import json
import math
import re

VERSION = 'point-localisation-v1'
TOLERANCES = {'tight': .10, 'medium': .25, 'loose': .50}
DIAGONAL_CAP = .02


def _unique(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f'Duplicate JSON key: {k}')
        out[k] = v
    return out


def _nonfinite(s):
    raise ValueError(f'Non-finite JSON number: {s}')


def parse_points(text: str, labelled: bool = False) -> list[dict]:
    fence = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.S)
    start, end = text.find('{'), text.rfind('}')
    if not fence and (start < 0 or end <= start):
        raise ValueError('No JSON object')
    candidate = fence.group(1) if fence else text[start:end + 1]
    data = json.loads(candidate, object_pairs_hook=_unique, parse_constant=_nonfinite)
    if not isinstance(data, dict) or not isinstance(data.get('objects'), list):
        raise ValueError('Expected an objects list')
    out = []
    for obj in data['objects']:
        if not isinstance(obj, dict):
            raise ValueError('Object must be a dictionary')
        if 'point' in obj:  # prompt v1: "point": [x, y]
            p = obj.get('point')
        elif 'x' in obj or 'y' in obj:  # prompt v2: named coordinates
            p = [obj.get('x'), obj.get('y')]
        else:
            p = None
        if not isinstance(p, list) or len(p) != 2 or not all(type(v) in (int, float) and math.isfinite(v) for v in p):
            raise ValueError('Expected two finite numeric coordinates')
        row = {'point': [float(v) for v in p]}
        if labelled:
            label = obj.get('label')
            if not isinstance(label, str) or label.strip().lower() not in ('dirty', 'clean'):
                raise ValueError('Expected dirty or clean label')
            row['label'] = label.strip().lower()
        out.append(row)
    return out


_NUM = r'-?\d+(?:\.\d+)?'
# A pair: '[a, b' followed by ']' or by a comma and something that is not a
# number (a broken '[188, 30, "label"'). Never a triple like [a, b, c].
_PAIR = re.compile(r'\[\s*(' + _NUM + r')\s*,\s*(' + _NUM + r')\s*(?=\]|,\s*[^\d\s\-.])')
_LABEL = re.compile(r'\b(dirty|clean)\b', re.I)
# Prompt v2's named form: "x": n ... "y": n inside one object, either order.
_NAMED = re.compile(r'"x"\s*:\s*(' + _NUM + r')\s*,\s*"y"\s*:\s*(' + _NUM + r')'
                    r'|"y"\s*:\s*(' + _NUM + r')\s*,\s*"x"\s*:\s*(' + _NUM + r')')


def recover_points(text: str, labelled: bool = False) -> list[dict]:
    """Mechanical fallback for replies strict JSON cannot read.

    Every [number, number] pair, in reply order, as written: no reordering,
    deduplication, clipping, axis swapping or rescaling. For labels, the
    first 'dirty'/'clean' after a pair and before the next pair; a pair
    without one keeps no label and so can only count as a miss in labelled
    scoring. Used only when parse_points fails, so a well-formed reply is
    never read differently.
    """
    if re.search(r'"[xy]"\s*:', text):  # prompt v2's named form, possibly garbled
        found = _recover_named(text)
    else:
        found = [(m.start(), m.end(), m.group(1), m.group(2)) for m in _PAIR.finditer(text)]
    if not found:
        raise ValueError('No coordinate pairs found')
    out = []
    for i, (start, end_, x, y) in enumerate(found):
        row = {'point': [float(x), float(y)]}
        if labelled:
            end = found[i + 1][0] if i + 1 < len(found) else len(text)
            lab = _LABEL.search(text, end_, end)
            row['label'] = lab.group(1).lower() if lab else None
        out.append(row)
    return out


_NUMTOK = re.compile(r'(?<![A-Za-z0-9_.])-?\d+(?:\.\d+)?(?![A-Za-z0-9_.])')


def _recover_named(text: str) -> list[tuple]:
    """Object by object (text between two '{'), as written: named x/y where the
    keys are intact (either order); every other number in the object paired in
    written order, first = x (the prompt's order). Numbers glued to letters
    ("x0") are not coordinates. Returns (start, end, x, y)."""
    cuts = [m.start() for m in re.finditer(r'\{', text)] + [len(text)]
    out = []
    for a, b in zip(cuts, cuts[1:]):
        chunk, used = text[a:b], []
        for m in _NAMED.finditer(chunk):
            x, y = (m.group(1), m.group(2)) if m.group(1) is not None else (m.group(4), m.group(3))
            out.append((a + m.start(), a + m.end(), x, y))
            used.append((m.start(), m.end()))
        rest = [n for n in _NUMTOK.finditer(chunk) if not any(s <= n.start() < e for s, e in used)]
        for n1, n2 in zip(rest[0::2], rest[1::2]):
            out.append((a + n1.start(), a + n2.end(), n1.group(0), n2.group(0)))
    return sorted(out)


def read_points(text: str, labelled: bool = False, parser: str = 'strict') -> tuple[list[dict], str]:
    """(points, how they were read): 'strict', or 'recovered' (lenient parser only)."""
    try:
        return parse_points(text, labelled), 'strict'
    except (ValueError, KeyError, TypeError):
        if parser != 'lenient':
            raise
    return recover_points(text, labelled), 'recovered'


def reference_distances(refs: list[dict], width: int, height: int) -> list[float]:
    coords = [(r['x'], r['y']) for r in refs]
    if len(set(coords)) != len(coords):
        raise ValueError('Duplicate reference coordinates')
    if any(not math.isfinite(x) or not math.isfinite(y) or not (0 <= x <= width and 0 <= y <= height) for x, y in coords):
        raise ValueError('Invalid reference coordinates')
    diagonal = math.hypot(width, height)
    return [min((math.hypot(x - xx, y - yy) for j, (xx, yy) in enumerate(coords) if i != j), default=diagonal)
            for i, (x, y) in enumerate(coords)]


def maximum_matching(edges: list[list[int]]) -> list[tuple[int, int]]:
    """Augmenting paths, maximum cardinality (not greedy). prediction -> reference."""
    assigned: dict[int, int] = {}

    def augment(p, visited):
        for r in edges[p]:
            if r in visited:
                continue
            visited.add(r)
            if r not in assigned or augment(assigned[r], visited):
                assigned[r] = p
                return True
        return False

    for p in range(len(edges)):
        augment(p, set())
    return sorted((p, r) for r, p in assigned.items())


def score_at(points, refs, width, height, radii, labelled=False):
    predicted = [(p['point'][0] * width / 1000, p['point'][1] * height / 1000) for p in points]
    edges = []
    for i, (x, y) in enumerate(predicted):
        if not (0 <= x <= width and 0 <= y <= height):
            edges.append([])
            continue
        valid = [(math.hypot(x - r['x'], y - r['y']), j) for j, r in enumerate(refs)
                 if (not labelled or points[i].get('label') == r['label']) and
                 math.hypot(x - r['x'], y - r['y']) <= radii[j] + 1e-9]
        edges.append([j for _, j in sorted(valid)])
    pairs = maximum_matching(edges)
    tp, n, m = len(pairs), len(points), len(refs)
    return {'tp': tp, 'fp': n - tp, 'fn': m - tp,
            'precision': tp / n if n else (1.0 if not m else 0.0),
            'recall': tp / m if m else 1.0,
            'f1': 2 * tp / (n + m) if n + m else 1.0,
            'matches': [{'prediction': i, 'reference_id': refs[j]['id'],
                         'distance_px': math.dist(predicted[i], (refs[j]['x'], refs[j]['y'])),
                         'radius_px': radii[j]} for i, j in pairs]}


def score(points, refs, width, height, labelled=False):
    nn = reference_distances(refs, width, height)
    levels, geometry = {}, {}
    for key, factor in TOLERANCES.items():
        radii = [min(factor * d, DIAGONAL_CAP * math.hypot(width, height)) for d in nn]
        levels[key] = score_at(points, refs, width, height, radii, labelled)
        if labelled:
            geometry[key] = score_at(points, refs, width, height, radii, False)['f1']
    return {'score': sum(v['f1'] for v in levels.values()) / len(levels), 'tolerances': levels,
            'geometry_only_f1': geometry or None, 'count_error': len(points) - len(refs),
            'predicted_count': len(points), 'reference_count': len(refs)}


# Processes that held a little GPU 0 memory but did no computing during the
# 2026-10-09 clean measurements (docs/point-benchmark-v2.md): the browser,
# the desktop overlay, and the scholarsreadinglist embeddings job, paused.
IDLE_GPU_HOLDERS = {'603447', '1925731', '753117'}


def timed_on_free_gpu(r: dict) -> bool:
    """A specialist record timed on the whole of GPU 0 (only idle holders present)."""
    others = r.get('gpu_others')
    return isinstance(others, dict) and set(others) <= IDLE_GPU_HOLDERS and r.get('started_at', '') >= '2026-10-09T20'
