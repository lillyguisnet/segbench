"""Reading boxes out of model replies (box benchmark, 2026-10-15).

Every box is returned as {"box": [x_min, y_min, x_max, y_max]} on the
0..1000 grid of the photo (x across, y down), plus "label" for dishes.

Rules, fixed before any benchmark photo was sent, and never using the
answer key:
- A box whose two x (or two y) are written in the wrong order is the same
  rectangle: min and max are taken (Maxime, 2026-10-15: "lenience: yes").
- x and y are never swapped, except where a format's own documented order
  is y first (Gemini's "box_2d": [y_min, x_min, y_max, x_max]).
- No rescaling of prompt box-a's replies (it asks for 0..1000); no
  clipping; no deduplication.

Prompt box-a (named corners, JSON):
  strict JSON first; if it fails, object by object (text between two "{"):
  the four named corners where the keys are intact; else a known array key
  (box_2d: y first; bbox_2d, bbox, box, bounding_box: x first); else the
  object's first four numbers in written order (x_min, y_min, x_max, y_max,
  the prompt's order).
"""

from __future__ import annotations

import json
import math
import re

_NUM = r"-?\d+(?:\.\d+)?"
_NUMTOK = re.compile(r"(?<![A-Za-z0-9_.])-?\d+(?:\.\d+)?(?![A-Za-z0-9_.])")
_LABEL = re.compile(r"\b(dirty|clean)\b", re.I)
CORNERS = ("x_min", "y_min", "x_max", "y_max")
_ALIASES = {"x_min": ("x_min", "xmin", "x1", "left"), "y_min": ("y_min", "ymin", "y1", "top"),
            "x_max": ("x_max", "xmax", "x2", "right"), "y_max": ("y_max", "ymax", "y2", "bottom")}
Y_FIRST_KEYS = ("box_2d",)  # Gemini's documented order
X_FIRST_KEYS = ("bbox_2d", "bbox", "box", "bounding_box", "boundingbox", "bounding box")


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def ordered(x1, y1, x2, y2) -> list[float]:
    vals = [float(v) for v in (x1, y1, x2, y2)]
    if not all(math.isfinite(v) for v in vals):
        raise ValueError("non-finite coordinate")
    return [min(vals[0], vals[2]), min(vals[1], vals[3]), max(vals[0], vals[2]), max(vals[1], vals[3])]


def _label_of(obj_or_text, labelled):
    if not labelled:
        return {}
    if isinstance(obj_or_text, dict):
        lab = obj_or_text.get("label")
        lab = lab.strip().lower() if isinstance(lab, str) else None
        return {"label": lab if lab in ("dirty", "clean") else lab}
    m = _LABEL.search(obj_or_text)
    return {"label": m.group(1).lower() if m else None}


def _from_dict(obj: dict):
    low = {str(k).strip().lower(): v for k, v in obj.items()}
    vals = {}
    for corner, names in _ALIASES.items():
        for n in names:
            if isinstance(low.get(n), (int, float)) and not isinstance(low.get(n), bool):
                vals[corner] = low[n]
                break
    if len(vals) == 4:
        return ordered(*(vals[c] for c in CORNERS)), "named"
    for k in Y_FIRST_KEYS:
        v = low.get(k)
        if isinstance(v, list) and len(v) == 4 and all(isinstance(x, (int, float)) for x in v):
            return ordered(v[1], v[0], v[3], v[2]), k
    for k in X_FIRST_KEYS:
        v = low.get(k)
        if isinstance(v, list) and len(v) == 4 and all(isinstance(x, (int, float)) for x in v):
            return ordered(*v), k
    return None, None


def _json_candidate(text: str):
    fence = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.S)
    if fence:
        return fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object")
    return text[start:end + 1]


def parse_box_a(text: str, labelled: bool = False) -> list[dict]:
    data = json.loads(_json_candidate(text))
    if not isinstance(data, dict) or not isinstance(data.get("objects"), list):
        raise ValueError("expected an objects list")
    out = []
    for obj in data["objects"]:
        if not isinstance(obj, dict):
            raise ValueError("object must be a dictionary")
        box, _ = _from_dict(obj)
        if box is None:
            raise ValueError("expected four corners")
        out.append({"box": box, **_label_of(obj, labelled)})
    return out


_KEYED = re.compile(r'"?(' + "|".join(n for ns in _ALIASES.values() for n in ns) + r')"?\s*:\s*(' + _NUM + ")", re.I)
_ARRAY_KEY = re.compile(r'"?(' + "|".join(Y_FIRST_KEYS + X_FIRST_KEYS) + r')"?\s*:\s*\[\s*(' + _NUM + r')\s*,\s*(' + _NUM
                        + r')\s*,\s*(' + _NUM + r')\s*,\s*(' + _NUM + r')\s*\]', re.I)


def recover_box_a(text: str, labelled: bool = False) -> list[dict]:
    cuts = [m.start() for m in re.finditer(r"\{", text)] + [len(text)]
    out = []
    for a, b in zip(cuts, cuts[1:]):
        chunk = text[a:b]
        named = {}
        for m in _KEYED.finditer(chunk):
            for corner, names in _ALIASES.items():
                if m.group(1).lower() in names and corner not in named:
                    named[corner] = float(m.group(2))
        if len(named) == 4:
            box = ordered(*(named[c] for c in CORNERS))
        elif m := _ARRAY_KEY.search(chunk):
            v = [float(m.group(i)) for i in range(2, 6)]
            box = ordered(v[1], v[0], v[3], v[2]) if m.group(1).lower() in Y_FIRST_KEYS else ordered(*v)
        else:
            nums = [float(n.group(0)) for n in _NUMTOK.finditer(chunk)]
            if len(nums) < 4:
                continue
            box = ordered(*nums[:4])
        out.append({"box": box, **_label_of(chunk, labelled)})
    if not out:
        raise ValueError("no boxes found")
    return out


def read_box_a(text: str, labelled: bool = False, parser: str = "lenient") -> tuple[list[dict], str]:
    """(boxes, 'strict' | 'recovered'). A reply strict JSON can read is never read differently."""
    try:
        return parse_box_a(text, labelled), "strict"
    except (ValueError, KeyError, TypeError):
        if parser != "lenient":
            raise
    try:
        data = json.loads(_json_candidate(text))
        if isinstance(data, dict) and data.get("objects") == []:
            return [], "strict"
    except (ValueError, TypeError):
        pass
    try:
        return recover_box_a(text, labelled), "recovered"
    except ValueError:
        pass
    # Rules v4 (2026-10-15): a reply with no JSON at all but Perceptron's own
    # box tags (<point_box> (x1,y1) (x2,y2), 0..1000, x first: the frame box-a
    # asks for) is read from its tags. Perceptron Mk1.5 answers box-a that
    # way; reading it is reading what it wrote, not repairing it.
    boxes, how = read_box_free(text, "mk1.5", 1000, 1000, labelled)
    if how.get("source") == "tags" and boxes:
        return boxes, "tags"
    raise ValueError("no boxes found")


# ---------------------------------------------------------------------------
# Prompt box-free: no answer format and no coordinate system were asked for.
#
# Rules written 2026-10-15 from the replies on synthetic pictures only
# (results/check-boxes-box-free-20261015-153307.jsonl, scripts/check_boxes.py),
# before any box-free reply on a benchmark photo was read. A person reading
# a reply would do the same: take the boxes the model gave as its answer,
# read them the way the model says they are written, and where it says
# nothing, the way that model writes them.
#
# 1. The boxes, in the first form found, in this order:
#    JSON items with named corners or a known array key; Perceptron tags
#    (<point_box> (x1,y1) (x2,y2)); "(a, b) to (c, d)" corner pairs;
#    groups of exactly four numbers in [ ] or ( ); "x a-b, y c-d" ranges.
#    A list written out twice in the same reply (a final "answer" repeating
#    the boxes above it) counts once.
# 2. Order, x or y first: the reply's own words ("[ymin, xmin, ...]",
#    "(left, top, right, bottom)", named keys); else the format's documented
#    order (Gemini's box_2d: y first); else the model's habit (MODEL_HABITS);
#    else x first.  "(x, y, width, height)" in the words: corner and size.
# 3. Scale: the reply's own words ("normalized" / "0-1", "0-1000",
#    "pixels"); else the format's documented scale (box_2d: 0..1000; tags:
#    0..1000); else the model's habit; else from the numbers: all <= 1 ->
#    0..1, any > 1000 -> pixels of the photo, else 0..1000.
#    Pixels are those of the picture the model saw: for Claude models the
#    photo shrunk to at most 2576 px on its long side (measured on the
#    synthetic pictures: 4000x2252 -> 2576x1450; 1015x1015 and 743x1319
#    unchanged); for every other model the photo itself.
# 3b. (added 2026-10-15, before any box-free reply was scored against any
#    answer key, after reading the flagged benchmark replies) Pixels: when
#    the reply names the size of the picture its pixels refer to ("the
#    4096 x 2304 image"), that size is the frame. GPT models and Claude
#    models often say it (GPT-6 Luna: 4096 x 2304 for a 4000 x 2252 photo;
#    GPT-5.6 Terra once 3393 x 3393). Read as stated, even when the stated
#    size is not the photo's: the reply says so.
#    Corner pairs also as "(a, b) -> (c, d)" and "top-left (a, b),
#    bottom-right (c, d)" (Claude Opus wrote both).
# 4. Dish labels: JSON items, their own label; otherwise "dirty" or "clean"
#    on the box's own line, else the nearest one above it (a heading such
#    as "Dirty dishes:" over a list).
# Replies these rules cannot read cleanly are flagged (`needs_hand_check`)
# and read by a person; those readings are kept in
# results/box-free-hand-readings.json with the reason, never using the
# answer key.
# ---------------------------------------------------------------------------

CLAUDE_LONG_SIDE = 2576
# Observed on the synthetic pictures (four sizes each), used only when a
# reply does not say how its numbers are written.
MODEL_HABITS = {
    "luna": ("xy", "px"), "terra": ("xy", "px"), "astra": ("xy", "px"), "sol": ("xy", "px"),
    "sonnet": ("xy", "px"), "opus": ("xy", "px"), "haiku": ("xy", "px"),
    "gemini-pro": ("yx", "1000"), "gemini-flash": ("yx", "1000"), "gemini-flash-lite": ("yx", "1000"),
    "gemini-argon": ("yx", "1000"),
    "kimi-k3": ("xy", "1"), "muse-wonder-api": ("xy", "1000"),
    # Added 2026-10-15 (rules v3) from their synthetic replies
    # (results/check-boxes-box-free-20261015-183431.jsonl), before any of their
    # benchmark replies existed: Qwen writes bbox_2d on 0..1000, x first;
    # Mistral Large 4 writes "box_2d" but x first (not Google's order);
    # Perceptron writes its own tags (0..1000, x first).
    "qwen-27b": ("xy", "1000"), "qwen-flash": ("xy", "1000"), "mistral-large-4": ("xy", "1000"),
    "mk1.5": ("xy", "1000"),
    # glm-5.3-flash and deepseek-flash: no single habit (each changed format
    # between pictures, usually saying which): their words, else the numbers.
}
CLAUDE_MODELS = {"sonnet", "opus", "haiku"}
# Models that write the key "box_2d" with x first (observed on all four
# synthetic pictures): for them box_2d does not mean Google's y-first order.
BOX_2D_X_FIRST = {"mistral-large-4"}

_Y_FIRST_WORDS = re.compile(r"\[?\(?\s*`?(y_?min|y1|ymin|top)\s*,\s*(x_?min|x1|xmin|left)", re.I)
_X_FIRST_WORDS = re.compile(r"(x_?min|x1|xmin|left)\s*,\s*(y_?min|y1|ymin|top)", re.I)
_XYWH_WORDS = re.compile(r"\bx\s*,\s*y\s*,\s*(w|width)\s*,\s*(h|height)\b", re.I)
_S_UNIT = re.compile(r"normali[sz]ed|\b0\s*(?:-|–|to)\s*1(?:\.0)?\b(?!\d)|\bfractions?\b", re.I)
_S_1000 = re.compile(r"\b0\s*(?:-|–|to)\s*1,?000\b|\b(?:scaled?|normali[sz]ed)\s+to\s+(?:a\s+)?(?:0\s*(?:-|–|to)\s*)?1,?000\b|\b1000\s*[x×]\s*1000\b|per\s*mille", re.I)
_S_PX = re.compile(r"\bpixels?\b|\bpx\b", re.I)
_GROUP4 = re.compile(r"[\[\(]\s*(" + _NUM + r")\s*,\s*(" + _NUM + r")\s*,\s*(" + _NUM + r")\s*,\s*(" + _NUM + r")\s*[\]\)]")
_CORNER_PAIRS = re.compile(r"\(\s*(" + _NUM + r")\s*,\s*(" + _NUM + r")\s*\)\s*(?:to|-|–|—|→|->|,|and)?\s*"
                           r"(?:,?\s*(?:and\s+)?bottom[- ]right:?\s*)?\(\s*(" + _NUM + r")\s*,\s*(" + _NUM + r")\s*\)", re.I)
_FRAME = re.compile(r"(?<![\d.])(\d{3,5})\s*[×xX]\s*(\d{3,5})(?![\d.])")
_RANGES = re.compile(r"\bx\s*[:=]?\s*(" + _NUM + r")\s*(?:-|–|to)\s*(" + _NUM + r")\s*,?\s*(?:and\s+)?y\s*[:=]?\s*("
                     + _NUM + r")\s*(?:-|–|to)\s*(" + _NUM + r")", re.I)


def _json_items(text: str):
    """Boxes from JSON in the reply: [(4 numbers, order or None, start, end)] or None."""
    found = []
    for m in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.S):
        found.append((m.group(1), m.start(1)))
    if not found:
        s = min([i for i in (text.find("["), text.find("{")) if i >= 0], default=-1)
        if s >= 0:
            found.append((text[s:max(text.rfind("]"), text.rfind("}")) + 1], s))
    for body, at in found:
        try:
            data = json.loads(body)
        except (ValueError, TypeError):
            continue
        items = data
        if isinstance(data, dict):
            lists = [v for v in data.values() if isinstance(v, list)]
            items = lists[0] if len(lists) == 1 else [data]
        if not isinstance(items, list):
            continue
        out = []
        for it in items:
            if isinstance(it, dict):
                low = {str(k).strip().lower(): v for k, v in it.items()}
                named = {}
                for corner, names in _ALIASES.items():
                    for n in names:
                        if isinstance(low.get(n), (int, float)) and not isinstance(low.get(n), bool):
                            named[corner] = low[n]
                            break
                if len(named) == 4:
                    out.append(([named[c] for c in CORNERS], "xy", it))
                    continue
                if all(isinstance(low.get(k), (int, float)) for k in ("x", "y", "width", "height")):
                    out.append(([low["x"], low["y"], low["x"] + low["width"], low["y"] + low["height"]], "xy", it))
                    continue
                arr = next(((k, low[k]) for k in Y_FIRST_KEYS + X_FIRST_KEYS
                            if isinstance(low.get(k), list) and len(low[k]) == 4), None)
                if arr and all(isinstance(v, (int, float)) for v in arr[1]):
                    out.append((list(arr[1]), "yx" if arr[0] in Y_FIRST_KEYS else None, it))
            elif isinstance(it, list) and len(it) == 4 and all(isinstance(v, (int, float)) for v in it):
                out.append((list(it), None, None))
        if out:
            return out, at
    return None


def _text_label(text: str, m) -> str:
    """The dish label for a box written in text: on its own line, else the nearest one above."""
    a = text.rfind("\n", 0, m.start()) + 1
    b = text.find("\n", m.end())
    line = text[a: b if b >= 0 else len(text)]
    found = _LABEL.findall(line) or _LABEL.findall(text[:a])[-1:]
    return found[0] if found else ""


def _drop_repeat(boxes: list) -> tuple[list, bool]:
    """A list written out twice (the same boxes again, in the same order) counts once."""
    n = len(boxes)
    if n >= 2 and n % 2 == 0 and [b[0] for b in boxes[: n // 2]] == [b[0] for b in boxes[n // 2:]]:
        return boxes[: n // 2], True
    return boxes, False


def read_box_free(text: str, model: str, width: int, height: int, labelled: bool = False) -> tuple[list[dict], dict]:
    """(boxes on the 0..1000 grid, how they were read). See the rules above."""
    from .perceptron import _TAG, _PAIR as _TAG_PAIR, _attrs

    how: dict = {"flags": []}
    raw = []  # (numbers, order from the format or None, label source)
    js = _json_items(text)
    if js:
        items, _ = js
        how["source"] = "json"
        for nums, order, it in items:
            raw.append((nums, order, it))
    else:
        tags = [m for m in _TAG.finditer(text) if m.group(1).lower() in ("point_box", "collection")]
        for m in tags:
            bodies = ([(i.group(2), i.group(3)) for i in _TAG.finditer(m.group(3)) if i.group(1).lower() == "point_box"]
                      if m.group(1).lower() == "collection" else [(m.group(2), m.group(3))])
            for attrs, body in bodies:
                pairs = _TAG_PAIR.findall(body)
                if len(pairs) == 2:
                    raw.append(([float(pairs[0][0]), float(pairs[0][1]), float(pairs[1][0]), float(pairs[1][1])], "xy",
                                _attrs(attrs).get("mention", "") or _attrs(m.group(2)).get("mention", "")))
        if raw:
            how["source"] = "tags"
        else:
            for rx, name in ((_CORNER_PAIRS, "corner pairs"), (_GROUP4, "groups of four"), (_RANGES, "x/y ranges")):
                ms = list(rx.finditer(text))
                if ms:
                    how["source"] = name
                    for m in ms:
                        nums = [float(g) for g in m.groups()]
                        if rx is _RANGES:
                            nums = [nums[0], nums[2], nums[1], nums[3]]
                        raw.append((nums, "xy" if rx is not _GROUP4 else None, _text_label(text, m)))
                    break
    if not raw:
        how["source"] = "none"
        how["flags"].append("no boxes found")
        return [], how
    raw, repeated = _drop_repeat(raw)
    if repeated:
        how["flags"].append("list written twice, counted once")

    words = text
    habit = MODEL_HABITS.get(model)
    # order
    fmt_orders = {o for _, o, _ in raw}
    if _XYWH_WORDS.search(words) and how["source"] != "json":
        order, why_o = "xywh", "words"
    elif fmt_orders == {"yx"} and model in BOX_2D_X_FIRST:
        order, why_o = "xy", "habit (box_2d written x first)"
    elif fmt_orders == {"yx"}:
        order, why_o = "yx", "format (box_2d)"
    elif fmt_orders == {"xy"}:
        order, why_o = "xy", "format"
    elif _Y_FIRST_WORDS.search(words):
        order, why_o = "yx", "words"
    elif _X_FIRST_WORDS.search(words):
        order, why_o = "xy", "words"
    elif habit:
        order, why_o = habit[0], "habit"
    else:
        order, why_o = "xy", "default"
    if len(fmt_orders - {None}) > 1:
        how["flags"].append("mixed formats")
    # scale
    values = [v for nums, _, _ in raw for v in nums]
    if _S_UNIT.search(words) and max(values) <= 1.0 + 1e-9:
        scale, why_s = "1", "words"
    elif _S_1000.search(words):
        scale, why_s = "1000", "words"
    elif _S_PX.search(words):
        scale, why_s = "px", "words"
    elif how["source"] == "tags" or fmt_orders == {"yx"}:
        scale, why_s = "1000", "format"
    elif habit:
        scale, why_s = habit[1], "habit"
    elif max(values) <= 1.0:
        scale, why_s = "1", "numbers"
    elif max(values) > 1000:
        scale, why_s = "px", "numbers"
    else:
        scale, why_s = "1000", "numbers"
    if scale == "1" and max(values) > 1.0 + 1e-9:
        how["flags"].append("0..1 scale but numbers above 1")
    fw, fh = float(width), float(height)
    frames = {(int(a), int(b)) for a, b in _FRAME.findall(words)} - {(1000, 1000)}
    if scale == "px" and len(frames) == 1:
        fw, fh = map(float, frames.pop())
        how["stated_frame"] = [int(fw), int(fh)]
    elif scale == "px" and len(frames) > 1:
        how["flags"].append(f"several picture sizes named: {sorted(frames)}")
    elif scale == "px" and model in CLAUDE_MODELS and max(width, height) > CLAUDE_LONG_SIDE:
        k = CLAUDE_LONG_SIDE / max(width, height)
        fw, fh = width * k, height * k
        how["claude_frame"] = [round(fw), round(fh)]
    sx, sy = {"1": (1000.0, 1000.0), "1000": (1.0, 1.0), "px": (1000.0 / fw, 1000.0 / fh)}[scale]
    out = []
    for nums, _, lab in raw:
        a, b, c, d = nums
        if order == "yx":
            a, b, c, d = b, a, d, c
        elif order == "xywh":
            c, d = a + c, b + d
        box = ordered(a * sx, b * sy, c * sx, d * sy)
        item = {"box": box}
        if labelled:
            item.update(_label_of(lab if isinstance(lab, dict) else str(lab or ""), True))
        out.append(item)
    how.update(order=order, order_from=why_o, scale=scale, scale_from=why_s)
    if any(b["box"][2] > 1000.5 or b["box"][3] > 1000.5 or b["box"][0] < -0.5 or b["box"][1] < -0.5 for b in out):
        how["flags"].append("boxes beyond the photo")
    return out, how
