"""Where does Qwen 3.8 Flash Next's time go? (2026-10-12, Maxime's question)

Benchmark calls took a median 39 s (photos) and 46 s (screens). Across them,
seconds ~= 8.5 + output tokens / 87 (r = 0.99). This probe splits the fixed
8.5 s and tries ways to shorten the thinking, on one photo (cows), outside the
benchmark: nothing here is scored or charted.

Every call goes straight to OpenRouter's chat endpoint (pinned to Alibaba, no
fallback, as in the benchmark), streamed, so we see:
  first_byte   the first byte of the reply (OpenRouter is answering)
  first_token  the first reasoning or answer token (the model is writing)
  total        the last byte
and OpenRouter's own record of the call (/api/v1/generation): its `latency`
(time to first token as OpenRouter saw it) and `generation_time`.

Run: uv run --with pillow python scripts/probe_qwen_latency.py   (a few cents)
Writes results/probe-qwen-latency-<time>.jsonl
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from segbench import tasks  # noqa: E402

URL = "https://openrouter.ai/api/v1/chat/completions"
PIN = {"order": ["Alibaba"], "allow_fallbacks": False}


def key() -> str:
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    text = Path("~/Projects/lm15-dev/.env").expanduser().read_text()
    # The file sets the key twice; a shell that sources it keeps the last one.
    return re.findall(r"OPENROUTER_API_KEY=['\"]?([^'\"\n]+)", text)[-1]


def jpeg(path: Path, max_side: int | None) -> bytes:
    if max_side is None:
        return path.read_bytes()
    im = Image.open(path)
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=92)
    return buf.getvalue()


def call(model: str, prompt: str, image: bytes | None, reasoning: dict | None, max_tokens=16000) -> dict:
    content: list = [{"type": "text", "text": prompt}]
    if image is not None:
        b64 = base64.b64encode(image).decode()
        content.insert(0, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    body = {"model": model, "messages": [{"role": "user", "content": content}], "stream": True,
            "provider": PIN, "max_tokens": max_tokens, "usage": {"include": True}}
    if reasoning is not None:
        body["reasoning"] = reasoning
    data = json.dumps(body).encode()
    req = urllib.request.Request(URL, data=data, headers={"Authorization": f"Bearer {key()}",
                                                         "Content-Type": "application/json", "User-Agent": "segbench-probe"})
    t0 = time.perf_counter()
    first_byte = first_token = None
    gen_id, usage, text, reasoning_text = None, None, [], []
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            now = time.perf_counter() - t0
            first_byte = first_byte or now
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            gen_id = gen_id or ev.get("id")
            usage = ev.get("usage") or usage
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                if d.get("content") or d.get("reasoning"):
                    first_token = first_token or now
                text.append(d.get("content") or "")
                reasoning_text.append(d.get("reasoning") or "")
    total = time.perf_counter() - t0
    return {"model": model, "reasoning_setting": reasoning, "upload_bytes": len(data),
            "first_byte": round(first_byte or total, 2), "first_token": round(first_token or total, 2),
            "total": round(total, 2), "usage": usage, "generation_id": gen_id,
            "answer": "".join(text)[:400], "reasoning_chars": len("".join(reasoning_text))}


def openrouter_record(gen_id: str) -> dict:
    for _ in range(6):
        time.sleep(2)
        try:
            req = urllib.request.Request(f"https://openrouter.ai/api/v1/generation?id={gen_id}",
                                         headers={"Authorization": f"Bearer {key()}", "User-Agent": "segbench-probe"})
            d = json.load(urllib.request.urlopen(req, timeout=30))["data"]
            return {k: d.get(k) for k in ("provider_name", "latency", "generation_time", "moderation_latency",
                                          "native_tokens_prompt", "native_tokens_completion",
                                          "native_tokens_reasoning", "total_cost")}
        except Exception as e:  # not ready yet
            last = str(e)
    return {"error": last}


def main() -> None:
    cows = next(t for t in tasks.TASKS if t.key == "cows")
    prompt = tasks.prompt_v2("find", cows)
    photo = ROOT / "images" / "field_with_cows.jpg"
    full, small = jpeg(photo, None), jpeg(photo, 1600)
    flash, q27 = "qwen/qwen3.8-flash", "qwen/qwen3.8-27b"
    plan = [
        # 1. fixed cost without a picture or thinking
        ("text, thinking off", flash, "Reply with the word OK.", None, {"enabled": False}),
        ("text, thinking off", flash, "Reply with the word OK.", None, {"enabled": False}),
        ("27B text, thinking off", q27, "Reply with the word OK.", None, {"enabled": False}),
        # 2. the picture's share: full file (2.5 MB, as in the benchmark) vs 1600 px
        ("full photo, thinking off", flash, prompt, full, {"enabled": False}),
        ("full photo, thinking off", flash, prompt, full, {"enabled": False}),
        ("1600px photo, thinking off", flash, prompt, small, {"enabled": False}),
        ("1600px photo, thinking off", flash, prompt, small, {"enabled": False}),
        # 3. thinking: benchmark setting, then smaller budgets
        ("full photo, medium (benchmark)", flash, prompt, full, {"effort": "medium"}),
        ("full photo, low", flash, prompt, full, {"effort": "low"}),
        ("full photo, budget 512", flash, prompt, full, {"max_tokens": 512}),
    ]
    out = ROOT / "results" / f"probe-qwen-latency-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    with out.open("w") as f:
        for name, model, p, img, r in plan:
            try:
                rec = call(model, p, img, r)
                rec["openrouter"] = openrouter_record(rec["generation_id"]) if rec["generation_id"] else {}
            except Exception as e:
                rec = {"model": model, "error": f"{type(e).__name__}: {e}"}
            rec["case"] = name
            f.write(json.dumps(rec) + "\n")
            u = rec.get("usage") or {}
            o = rec.get("openrouter") or {}
            print(f"{name:32} bytes={rec.get('upload_bytes', 0):>8} byte1={rec.get('first_byte')}s "
                  f"tok1={rec.get('first_token')}s total={rec.get('total')}s out={u.get('completion_tokens')} "
                  f"reason={(u.get('completion_tokens_details') or {}).get('reasoning_tokens')} "
                  f"OR latency={o.get('latency')} gen={o.get('generation_time')} {rec.get('error', '')}",
                  flush=True)
    print("saved", out)


if __name__ == "__main__":
    main()
