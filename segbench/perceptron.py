"""Reading Perceptron's own answer format: annotation tags inside the text.

Perceptron's models (Mk1, Mk1.5) are trained to ground answers with tags
(https://docs.perceptron.inc/perceptron-mk1.5/concepts/annotations):

    <point mention="cup" asset_idx="0"> (120,200) </point>
    <point_box mention="cup"> (100,150) (300,350) </point_box>
    <collection mention="cups"> <point> (1,2) </point> ... </collection>

on a 0..1000 grid of the picture, x first. Perceptron's API has a switch
(vision_config.annotation_format) that asks for them; OpenRouter does not
pass it (checked 2026-10-12), so prompt version "tags" asks for them in
words instead (segbench/tasks.py prompt_tags, segbench/screens.py
prompt_tags). Only the answer format differs from prompt v2: the task's
words are the same.

Reading rules, fixed before any benchmark photo was sent:
- <point>: its (x, y), as written.
- <point_box>: the centre of its two corners (it answers "where", like a
  point; the screen reader treats boxes the same way).
- <polygon>: not read (never asked for); counted in `skipped`.
- Label: the tag's own mention, else the enclosing collection's; it is
  the label when it is exactly one of the allowed words (any case, spaces
  trimmed), otherwise kept as written (it then cannot match a labelled
  reference).
- No rescaling, no swapping, no clipping, nothing from the answer key.
- No tag at all: the reply is read as prompt v2's JSON when it holds that
  (lenient reader); otherwise it is an answer with no points (the prompt
  says "if there are none, say so").
"""

from __future__ import annotations

import re

_NUM = r"-?\d+(?:\.\d+)?"
_PAIR = re.compile(r"\(\s*(" + _NUM + r")\s*,\s*(" + _NUM + r")\s*\)")
_TAG = re.compile(r"<(collection|point_box|point|polygon)\b([^>]*)>(.*?)</\1\s*>", re.S | re.I)
_ATTR = re.compile(r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\')')


def _attrs(s: str) -> dict:
    return {m.group(1).lower(): m.group(2) if m.group(2) is not None else m.group(3) for m in _ATTR.finditer(s)}


def _label(mention: str | None, labels: tuple[str, ...]) -> str | None:
    if mention is None:
        return None
    word = mention.strip().lower()
    return word if word in labels else mention


def read_tags(text: str, labels: tuple[str, ...] = ()) -> tuple[list[dict], dict]:
    """(points as [{"point": [x, y], "label"?: ..., "form": ...}], how read).

    `how` = {"read_as": "tags" | "none", "skipped": n polygons, "tags": n}.
    """
    out: list[dict] = []
    skipped = n_tags = 0

    def take(kind: str, attrs: dict, body: str, inherited: str | None):
        nonlocal skipped, n_tags
        n_tags += 1
        mention = attrs.get("mention", inherited)
        pairs = [(float(a), float(b)) for a, b in _PAIR.findall(body)]
        if kind == "point" and len(pairs) == 1:
            x, y = pairs[0]
        elif kind == "point_box" and len(pairs) == 2:
            (x1, y1), (x2, y2) = pairs
            x, y = (x1 + x2) / 2, (y1 + y2) / 2
        else:
            skipped += 1
            return
        item = {"point": [x, y], "form": kind}
        if labels:
            item["label"] = _label(mention, labels)
        out.append(item)

    for m in _TAG.finditer(text or ""):
        kind, attrs, body = m.group(1).lower(), _attrs(m.group(2)), m.group(3)
        if kind == "collection":
            for inner in _TAG.finditer(body):
                take(inner.group(1).lower(), _attrs(inner.group(2)), inner.group(3), attrs.get("mention"))
        else:
            take(kind, attrs, body, None)
    return out, {"read_as": "tags" if n_tags else "none", "skipped": skipped, "tags": n_tags}


def read_tag_answer(text: str, labels: tuple[str, ...] = ()) -> tuple[list[dict], str]:
    """The photo benchmark's reader for prompt version "tags": (points, how)."""
    points, how = read_tags(text, labels)
    if how["read_as"] == "tags":
        return points, "tags" + (f" ({how['skipped']} skipped)" if how["skipped"] else "")
    from .point_score import read_points  # a reply in prompt v2's JSON after all
    try:
        return read_points(text, labelled=bool(labels), parser="lenient")
    except Exception:
        return [], "none"
