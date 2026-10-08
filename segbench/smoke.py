"""A tiny synthetic segmentation test: four red circles and two blue squares.

Used to check that a model is reachable, really receives the picture, and
can answer in our polygon format. Not part of the benchmark score.
"""

from __future__ import annotations

import json
import re
import struct
import zlib

W, H = 640, 480
CIRCLES = [(120, 110, 45), (450, 90, 35), (300, 330, 55), (540, 380, 40)]  # x, y, radius in pixels
SQUARES = [(200, 180, 50), (40, 380, 45)]  # x, y, half side: blue distractors

PROMPT = """Segment every red circle in this image.

Reply with JSON only, no other text, in this shape:
{"objects": [{"label": "red circle", "polygon": [[x, y], [x, y], ...]}]}

Coordinates are integers from 0 to 1000: x = 0 is the left edge and 1000 the
right edge, y = 0 is the top edge and 1000 the bottom edge. Use at least 8
points per polygon, tracing the outline."""


def png() -> bytes:
    """The test picture as PNG bytes (built with the standard library only)."""
    rows = []
    for y in range(H):
        row = bytearray(b"\x00")  # PNG filter byte: none
        for x in range(W):
            px = (255, 255, 255)
            if any(abs(x - sx) <= s and abs(y - sy) <= s for sx, sy, s in SQUARES):
                px = (40, 70, 210)
            if any((x - cx) ** 2 + (y - cy) ** 2 <= r * r for cx, cy, r in CIRCLES):
                px = (215, 30, 30)
            row += bytes(px)
        rows.append(bytes(row))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    ihdr = struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(b"".join(rows))) + chunk(b"IEND", b"")


def parse(text: str) -> list[dict]:
    """The objects in a reply. Accepts a JSON object wrapped in prose or a code fence."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("no JSON object in the reply")
    return json.loads(match.group(0))["objects"]


def grade(objects: list[dict]) -> tuple[bool, str]:
    """Pass when each circle is matched by exactly one polygon centred inside it, with no extras."""
    centres = []
    for obj in objects:
        pts = obj["polygon"]
        centres.append((sum(p[0] for p in pts) / len(pts) * W / 1000, sum(p[1] for p in pts) / len(pts) * H / 1000))
    unmatched = list(centres)
    misses = 0
    for tx, ty, r in CIRCLES:
        best = min(unmatched, key=lambda c: (c[0] - tx) ** 2 + (c[1] - ty) ** 2, default=None)
        if best is None or ((best[0] - tx) ** 2 + (best[1] - ty) ** 2) ** 0.5 > r:
            misses += 1
        else:
            unmatched.remove(best)
    ok = len(objects) == len(CIRCLES) and misses == 0
    return ok, f"{len(objects)} found / {len(CIRCLES)} true, {misses} missed, {len(unmatched)} extra"
