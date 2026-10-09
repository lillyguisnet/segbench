"""Read a model's reply into the shared answer format, or say why it cannot be read.

This only checks shape (is there JSON, are the points numbers in 0..1000);
it does not score. Raw replies are always kept, so a looser or stricter
reader can be applied later without calling any model again.
"""

from __future__ import annotations

import json
import re


def _json(text: str):
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("no JSON object in the reply")
        candidate = text[start:end + 1]
    return json.loads(candidate)


def _point(p) -> list[float]:
    if not (isinstance(p, (list, tuple)) and len(p) == 2 and all(isinstance(v, (int, float)) for v in p)):
        raise ValueError(f"not a point: {str(p)[:60]}")
    # A point outside 0..1000 is kept, not refused: it lands on no object, so
    # the scorer counts it as a miss (or clips an outline to the photo).
    return [float(p[0]), float(p[1])]


def _polygon(poly) -> list[list[float]]:
    if not isinstance(poly, list) or len(poly) < 3:
        raise ValueError("polygon with fewer than 3 points")
    return [_point(p) for p in poly]


def read(track: str, kind: str, text: str, labels: tuple[str, ...] = ()) -> dict:
    """{"items": [...]} in the shared format, or raises ValueError."""
    data = _json(text)
    if not isinstance(data, dict):
        raise ValueError("JSON is not an object")
    if track in ("find", "trick"):
        out = []
        for obj in data["objects"]:
            item = {"point": _point(obj["point"])}
            if track == "find" and labels:
                label = str(obj.get("label", "")).strip().lower()
                if label not in labels:
                    raise ValueError(f"label {label!r} not one of {labels}")
                item["label"] = label
            out.append(item)
        return {"items": out}
    if kind == "region":
        return {"items": [{"polygon": _polygon(r["polygon"])} for r in data["regions"]]}
    out = []
    for obj in data["objects"]:
        item = {"polygon": _polygon(obj["polygon"])}
        if labels:
            label = str(obj.get("label", "")).strip().lower()
            if label not in labels:
                raise ValueError(f"label {label!r} not one of {labels}")
            item["label"] = label
        out.append(item)
    return {"items": out}
