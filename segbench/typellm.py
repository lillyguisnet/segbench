"""TypeLLM (typellm.ai) on the photo benchmarks: its questions, one call, its price (Maxime, 2026-10-17).

TypeLLM answers typed fields instead of text: each question is a JSON-schema
field with instructions, and the answer comes back in that type. So there is
no reply to parse, and no answer-format paragraph to write: the schema is the
format. The words of each request are the benchmark's own (segbench/tasks.py),
the same as every language model gets, minus only the "Reply with JSON only,
in this shape" paragraph that the schema replaces. They go in the field's
`instructions`; `context` is empty (the photo is the context).

Four prompts, matching the four the language models had:
- points "typellm-v2":    prompt v2's request and coordinate paragraph; items {x, y} integers on 0..1000;
- points "typellm-v1-hd": prompt v1's request and coordinate paragraph (without "write every point as
                          [x, y]": the schema names x and y); same items;
- boxes "typellm-box-a":  box-a's request and coordinate paragraph; items {x_min, y_min, x_max, y_max}
                          integers on 0..1000;
- boxes "typellm-box-free": box-free's request (no coordinate system), items of TypeLLM's own "box" type
                          (edges as fractions of the photo), the closest thing to "answer the way you
                          choose". That type is capped at 50 items per answer by the API, with no
                          continuation (checked 2026-10-17 on 90 synthetic dots: 50 returned; maxItems
                          above 50 and continue_from both refused): on the logs (83) it cannot reach all.
                          Arrays of objects allow 200, so the other three prompts ask for maxItems 200.
Dishes: the item also gets "label", one of "dirty" / "clean", as the prompts ask (scored apart).

Thinking. TypeLLM cannot think while it writes a list: the API refuses
thinking_effort on an array, on its items and on the items' properties
(checked 2026-10-17: "set it on its items", then "on its properties", then "not
supported inside array items"; a "box" item takes nothing but its type). So:
- level "none": the list asked alone, no thinking. The only way TypeLLM answers
  this question by itself: its entry.
- level "count-medium" (an experiment, a pipeline of ours): a first question
  that thinks at medium effort (up to 2,048 tokens), "how many ... are in this
  photo?" in the task's own words, and the list asked after it with depends_on,
  TypeLLM's documented way to chain fields (it sees the count, not the
  reasoning). The count question is ours, not the benchmark's: its result is
  reported as "TypeLLM + count first", never as TypeLLM's thinking level.
The API's default temperature (0, the most likely answer) is kept, as every
model keeps its provider's default sampling.
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request

from segbench import tasks as T

ENDPOINT = "https://api.typellm.ai/v1/generate"
MODEL_KEY = "typellm"
# typellm.ai/pricing, read 2026-10-17: input (context, questions, images) $0.05 per million
# tokens, thinking $0.50 per million, typed answers free. New accounts get $5.
PRICE = {"input": 0.05, "thinking": 0.50, "listed_as": "typellm-latest", "read": "2026-10-17 typellm.ai/pricing"}
LEVELS = ("none", "count-medium")
TIMEOUT = 90  # the API's largest (60 by default)
POINT_VERSIONS = ("typellm-v2", "typellm-v1-hd")
BOX_VERSIONS = ("typellm-box-a", "typellm-box-free")
MAX_OBJECT_ITEMS = 200  # the API's limit for an array of objects
REPLY_PARAGRAPH = "\n\nReply with JSON only"


def _request_words(version: str, task: T.Task) -> str:
    """The benchmark prompt's words with its reply-format paragraph removed (the schema is the format)."""
    if version == "typellm-v2":
        full = T.prompt_v2("find", task)
    elif version == "typellm-v1-hd":
        full = T.prompt("find", task)
    elif version == "typellm-box-a":
        full = T.prompt_box(T.PROMPT_VERSION_BOX_A, task)
    elif version == "typellm-box-free":
        return T.prompt_box(T.PROMPT_VERSION_BOX_FREE, task)
    else:
        raise ValueError(version)
    head, sep, rest = full.partition(REPLY_PARAGRAPH)
    if not sep:
        raise ValueError(f"{version}: no reply paragraph to remove")
    coords = rest.split("\n\n", 2)[-1]  # the coordinate paragraph after the JSON shape and "If there are none"
    if version == "typellm-v1-hd":
        coords = coords.replace("\nedge. Write every point as [x, y].", "\nedge.")
        assert "[x, y]" not in coords, coords
    return f"{head}\n\n{coords}"


def question(version: str, task: T.Task) -> dict:
    """The one field ("objects") TypeLLM is asked for this task."""
    label = {"label": {"type": "string", "enum": list(task.labels)}} if task.labels else {}
    if version in POINT_VERSIONS:
        items = {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}, **label}}
        q = {"type": "array", "maxItems": MAX_OBJECT_ITEMS, "items": items}
    elif version == "typellm-box-a":
        corners = {k: {"type": "integer"} for k in ("x_min", "y_min", "x_max", "y_max")}
        q = {"type": "array", "maxItems": MAX_OBJECT_ITEMS,
             "items": {"type": "object", "properties": {**corners, **label}}}
    elif version == "typellm-box-free":
        # A box cannot sit inside an object, so this prompt's dishes come without a dirty/clean
        # label (the words still ask for it, unchanged); location is all that is scored.
        q = {"type": "array", "items": {"type": "box"}}
    else:
        raise ValueError(version)
    q["instructions"] = _request_words(version, task)
    return q


def count_words(task: T.Task) -> str:
    """The thinking question of level count-medium: the task's own target and definition."""
    note = f"\n{task.note}" if task.note else ""
    return (f"Find {task.target} in this photo. How many are there? Count every one you can see, "
            f"even partly hidden ones, however many there are.{note}")


def questions(version: str, task: T.Task, level: str) -> dict:
    q = question(version, task)
    if level == "none":
        return {"objects": q}
    if level == "count-medium":
        return {"count": {"type": "integer", "thinking_effort": "medium", "instructions": count_words(task)},
                "objects": {**q, "depends_on": ["count"]}}
    raise ValueError(level)


def generate(body: dict, attempts: int = 4) -> tuple[dict, float]:
    """One call, retried only when no answer came (network, 429, 5xx, timeout); never a real answer."""
    key = os.environ["TYPELLM_API_KEY"]
    data = json.dumps(body).encode()
    last = None
    for i in range(attempts):
        req = urllib.request.Request(ENDPOINT, data=data, headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        t = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT + 30) as r:
                return json.loads(r.read()), time.perf_counter() - t
        except urllib.error.HTTPError as e:
            text = e.read().decode(errors="replace")[:500]
            last = f"HTTP {e.code}: {text}"
            if e.code == 429 or e.code >= 500:
                time.sleep(float(e.headers.get("Retry-After") or 5 * (i + 1)))
                continue
            raise RuntimeError(last) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = f"{type(e).__name__}: {e}"
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"no answer after {attempts} attempts: {last}")


def cost_usd(usage: dict) -> float:
    return (usage.get("input_tokens", 0) * PRICE["input"] + usage.get("thinking_tokens", 0) * PRICE["thinking"]) / 1e6


def ask(image_bytes: bytes, mime: str, version: str, task: T.Task, level: str) -> dict:
    """Ask one task; the record fields every runner and scorer reads (0..1000 grid for boxes)."""
    qs = questions(version, task, level)
    body = {"context": "", "images": [f"data:{mime};base64," + base64.b64encode(image_bytes).decode()],
            "questions": qs, "timeout": TIMEOUT}
    resp, wall = generate(body)
    items = resp["result"].get("objects") or []
    out = {"seconds": round(wall, 3), "elapsed_service": resp.get("elapsed"), "usage": resp.get("usage"),
           "cost_usd": cost_usd(resp.get("usage") or {}), "served_model": resp.get("model"), "id": resp.get("id"),
           "status": resp.get("status"), "thinking": resp.get("thinking"), "raw_result": resp["result"],
           "questions": qs, "count": resp["result"].get("count"), "items_returned": len(items)}
    if version in POINT_VERSIONS:
        objs = [{"x": p.get("x"), "y": p.get("y"), **({"label": p["label"]} if "label" in p else {})} for p in items]
        out["text"] = json.dumps({"objects": objs})
    else:
        boxes = []
        for b in items:
            if version == "typellm-box-free":
                g = [b["left"] * 1000, b["top"] * 1000, b["right"] * 1000, b["bottom"] * 1000]
            else:
                g = [b["x_min"], b["y_min"], b["x_max"], b["y_max"]]
            # corners written in the wrong order are the same rectangle (box benchmark's reading leniency)
            g = [min(g[0], g[2]), min(g[1], g[3]), max(g[0], g[2]), max(g[1], g[3])]
            boxes.append({"box": [round(float(v), 1) for v in g], **({"label": b["label"]} if "label" in b else {})})
        out["answer"] = {"items": boxes}
        out["text"] = json.dumps({"objects": boxes})
    return out
