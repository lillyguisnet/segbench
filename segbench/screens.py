"""Screen tasks: where would you click? A separate benchmark from the photos.

Five screenshots taken by Lilly on 2026-10-10 (images/screenshots/, not in
the public repo: they show private accounts). Each task is one instruction,
in Lilly's words; the model answers with the point(s) it would click.

Decided with Maxime on 2026-10-10:
- Scored apart from the photo benchmark, on its own chart.
- Prompt written once, plainly, and never tuned on these screenshots: the
  same shape as the photo prompt v2 (segbench/tasks.py), with "screenshot"
  for "photo" and clicks for objects.
- Answers are read generously (read_clicks below): we measure seeing, not
  following a reply format.
- The answer key comes after the answers: model agreement gives a draft,
  a person corrects it (as for the photos).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

PROMPT_VERSION = "screen-v1"  # 2026-10-10; any word changed below means a new version


@dataclass(frozen=True)
class ScreenTask:
    key: str
    image: str  # file in images/screenshots/
    instruction: str  # Lilly's words, as given
    what: str  # what the screenshot shows (for people, never sent)


SCREEN_TASKS: tuple[ScreenTask, ...] = (
    ScreenTask("chattering", "screen_chattering_chart.png", "Collapse Lilly's conversation group.",
               "Chattering with the pointing chart open"),
    ScreenTask("x", "screen_x_feed.png", "Open a reply to Michael.",
               "X home feed"),
    ScreenTask("viewer", "screen_slack_photo_viewer.png", "Close the photo viewer.",
               "Photos viewer over Slack, showing a screenshot of itself (several close buttons)"),
    ScreenTask("favorites", "screen_review_app_dishes.png", "Add to favorites.",
               "Edge with the segbench review app"),
    ScreenTask("shopify", "screen_shopify_products.png", "Tick lamb-only products.",
               "Shopify product list"),
)
# Round 2 (2026-10-10, chosen by Maxime): several right clicks per question,
# so a score can be partial; each has a decoy. Same screenshots, same prompt.
SCREEN_TASKS_2: tuple[ScreenTask, ...] = (
    ScreenTask("fold-groups", "screen_chattering_chart.png", "Fold every open conversation group except segbench.",
               "Chattering sidebar; one group is already folded"),
    ScreenTask("news-1000", "screen_x_feed.png", "Open every news story with more than 1,000 posts.",
               "X 'Today's News' panel: 20.5K, 9,853 and 378 posts"),
    ScreenTask("onedrive-folders", "screen_slack_photo_viewer.png", "Open every folder inside OneDrive.",
               "Explorer sidebar; Documents, Music, Pictures also under This PC"),
    ScreenTask("dirty-labels", "screen_review_app_dishes.png", "Click every D (dirty) label.",
               "Review app: kitchen photo with round D (dirty) and C (clean) labels"),
    ScreenTask("sold-out", "screen_shopify_products.png", "Tick every product that has no more items to sell.",
               "Shopify list; stock 0, -1 and positive"),
)
# Asked once, then replaced (2026-10-10, Lilly): "dish marked dirty" means the
# whole dish, which would need a mask of each dish; the question was changed
# to the round label itself, a sharper target. Its answers stay on record.
RETIRED: tuple[ScreenTask, ...] = (
    ScreenTask("dirty-dishes", "screen_review_app_dishes.png", "Click every dish marked dirty.",
               "Review app: kitchen photo with D (dirty) and C (clean) markers"),
)
ROUNDS = {1: SCREEN_TASKS, 2: SCREEN_TASKS_2}
BY_KEY = {t.key: t for t in SCREEN_TASKS + SCREEN_TASKS_2 + RETIRED}


# ---------------- answer-key shapes ----------------
# A target is a box {"box": [left, top, right, bottom]} or a circle
# {"circle": [cx, cy, r]}, in screenshot pixels, edges included. "covered_by":
# ids of targets drawn on top of it: a click there belongs to them, not to it.

def in_shape(t: dict, x: float, y: float) -> bool:
    if "circle" in t:
        cx, cy, r = t["circle"]
        return (x - cx) ** 2 + (y - cy) ** 2 <= r * r
    left, top, right, bottom = t["box"]
    return left <= x <= right and top <= y <= bottom


def hits(targets: list[dict], x: float, y: float) -> list[int]:
    """Indexes of the targets a click at (x, y) counts for."""
    by_id = {t["id"]: t for t in targets}
    return [i for i, t in enumerate(targets) if in_shape(t, x, y)
            and not any(in_shape(by_id[c], x, y) for c in t.get("covered_by", ()) if c in by_id)]


def shape_ok(t: dict, width: int, height: int) -> bool:
    if "circle" in t:
        cx, cy, r = t["circle"]
        return r > 0 and 0 <= cx < width and 0 <= cy < height
    left, top, right, bottom = t["box"]
    return 0 <= left <= right < width and 0 <= top <= bottom < height

COORDINATES = """x and y are integers from 0 to 1000 over the whole screenshot: x is the horizontal
position (0 = left edge, 1000 = right edge) and y the vertical position
(0 = top edge, 1000 = bottom edge). They are not pixels: whatever the screenshot's
size, the right edge is x = 1000 and the bottom edge is y = 1000."""


def prompt(task: ScreenTask) -> str:
    return f"""This is a screenshot of a computer screen. Task: {task.instruction}
Where would you click to do it? Give one point for each click needed.

Reply with JSON only, in this shape:
{{"clicks": [{{"x": 0, "y": 0}}, ...]}}

{COORDINATES}"""


# ---------------- reading answers, generously ----------------
# Any of these counts as a click, wherever it sits in the reply (JSON,
# code fence, prose): {"x": a, "y": b} (either order), "point"/"click"/
# "coordinates": [a, b], a box [x1, y1, x2, y2] or {"x1"..} (its centre),
# a bare [a, b] or (a, b). Kept as written: no axis swap, no rescaling, no
# clipping, nothing from the answer key. A model that writes pixels or
# [y, x] is read as it wrote, and misses: that is a seeing-the-screen
# question we cannot answer for it without guessing.

_NUM = r"-?\d+(?:\.\d+)?"


def _num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)) and math.isfinite(v):
        return float(v)
    if isinstance(v, str) and re.fullmatch(r"\s*" + _NUM + r"\s*", v):
        return float(v)
    return None


def _from_json(node, out):
    """Walk any JSON value; collect clicks in reading order."""
    if isinstance(node, dict):
        low = {str(k).lower(): v for k, v in node.items()}
        x, y = _num(low.get("x")), _num(low.get("y"))
        if x is not None and y is not None:
            out.append((x, y, "xy"))
            return
        box = [_num(low.get(k)) for k in ("x1", "y1", "x2", "y2")]
        if all(v is not None for v in box):
            out.append(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, "box"))
            return
        for k in ("point", "click", "coordinates", "coords", "position", "location", "center", "centre"):
            v = low.get(k)
            if isinstance(v, (list, dict)):
                before = len(out)
                _from_json(v, out)
                if len(out) > before:
                    return
        for k in ("bbox", "box", "bounding_box"):
            v = low.get(k)
            if isinstance(v, list) and len(v) == 4 and all(_num(a) is not None for a in v):
                a = [_num(t) for t in v]
                out.append(((a[0] + a[2]) / 2, (a[1] + a[3]) / 2, "box"))
                return
        for v in node.values():
            if isinstance(v, (list, dict)):
                _from_json(v, out)
        return
    if isinstance(node, list):
        nums = [_num(v) for v in node]
        if len(node) == 2 and all(n is not None for n in nums):
            out.append((nums[0], nums[1], "pair"))
            return
        if len(node) == 4 and all(n is not None for n in nums):
            out.append(((nums[0] + nums[2]) / 2, (nums[1] + nums[3]) / 2, "box"))
            return
        for v in node:
            _from_json(v, out)


def _json_candidates(text: str):
    for m in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.S):
        yield m.group(1)
    a, b = text.find("{"), text.rfind("}")
    if 0 <= a < b:
        yield text[a:b + 1]
    if text.strip().startswith("["):  # a bare JSON list; a [a, b] inside prose is read with the prose
        yield text[text.find("["):text.rfind("]") + 1]


_NAMED = re.compile(r'"?x"?\s*[:=]\s*(' + _NUM + r')\s*,\s*"?y"?\s*[:=]\s*(' + _NUM + r')'
                    r'|"?y"?\s*[:=]\s*(' + _NUM + r')\s*,\s*"?x"?\s*[:=]\s*(' + _NUM + r')', re.I)
_PAIR = re.compile(r'[\[(]\s*(' + _NUM + r')\s*,\s*(' + _NUM + r')\s*[\])]')


def read_clicks(text: str) -> tuple[list[dict], str]:
    """(clicks as [{"x", "y"}] in 0..1000 as written, how they were read).

    'json': a JSON value in the reply held them. 'text': found in the words
    (named x/y first, else bare pairs). Raises ValueError if no click at all.
    """
    parsed = False
    for cand in _json_candidates(text or ""):
        try:
            data = json.loads(cand)
        except ValueError:
            continue
        parsed = True
        out: list = []
        _from_json(data, out)
        if out:
            return [{"x": x, "y": y, "form": f} for x, y, f in out], "json"
    found = []
    for m in _NAMED.finditer(text or ""):
        x, y = (m.group(1), m.group(2)) if m.group(1) is not None else (m.group(4), m.group(3))
        found.append({"x": float(x), "y": float(y), "form": "xy"})
    if not found:
        found = [{"x": float(a), "y": float(b), "form": "pair"} for a, b in _PAIR.findall(text or "")]
    if not found:
        if parsed:  # well-formed JSON with no click in it: the model chose not to click
            return [], "json"
        raise ValueError("no click in the reply")
    return found, "text"
