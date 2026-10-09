"""One model call, measured: the reply, the tokens, the time, and any error.

Every call returns a plain dict that can be written as one JSON line. A
failed call is a record too (its `error` is set), never an exception, so a
run over many models always finishes and the failures stay on record.
"""

from __future__ import annotations

import time
import traceback
from datetime import datetime, timezone

from lm15 import Config, LMRouter, Message, Reasoning, Request
from lm15.serde import response_to_dict, usage_to_dict
from lm15.types import image as image_part

from . import cost
from .models import THINKING, Model

SUBSCRIPTION_PROVIDERS = {"openai-codex", "claude-code"}

# Ceiling on the reply, thinking included. Generous on purpose: a model that
# thinks long before drawing 30 polygons must not be cut off. The Codex
# backend refuses any ceiling, so it gets none.
MAX_TOKENS = 32_000

_router = LMRouter()


def call(model: Model, image: bytes | None, media_type: str, prompt: str,
         effort: str | None = None, level: str | None = None) -> dict:
    """Send one picture and one prompt to `model`; return the measured record.

    `level` is our thinking level ("min", "medium", "max"; see
    segbench.models.THINKING), translated into the model's own settings.
    `effort` is lm15's raw dial (off, minimal, low, medium, high, xhigh, max),
    for probing. Neither: the provider's default. `image=None` sends text only.
    """
    extensions = dict(model.extensions or {})
    if level is not None:
        chosen = THINKING[model.key][level]
        effort = chosen.get("effort")
        extensions.update(chosen.get("extensions", {}))
    route = _router.resolve(model.route)
    record: dict = {
        "model": model.key,
        "route": model.route,
        "provider": route.provider,
        "wire_model": route.model,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "level": level,
        "effort": effort,
        "extensions": extensions or None,
    }
    if (model.billing == "subscription") != (route.provider in SUBSCRIPTION_PROVIDERS):
        record["error"] = f"route resolves to {route.provider!r}, which does not match billing {model.billing!r}"
        return record

    settings: dict = {} if route.provider == "openai-codex" else {"max_tokens": MAX_TOKENS}
    if extensions:
        settings["extensions"] = extensions
    if effort is not None:
        settings["reasoning"] = Reasoning(effort=effort)
    content = [prompt] if image is None else [image_part(data=image, media_type=media_type), prompt]
    start = time.perf_counter()
    try:  # building the request can refuse a setting too (an effort word the model lacks)
        request = Request(model=model.route, messages=(Message.user(content),), config=Config(**settings))
        response = _router.complete(request)
    except Exception as error:
        record["seconds"] = round(time.perf_counter() - start, 3)
        record["error"] = f"{type(error).__name__}: {error}"
        record["traceback"] = traceback.format_exc(limit=3)
        return record
    record["seconds"] = round(time.perf_counter() - start, 3)
    record["text"] = response.text or ""
    record["finish_reason"] = response.finish_reason
    record["usage"] = usage_to_dict(response.usage)
    record["response"] = response_to_dict(response, include_provider_data=True)
    record["cost_usd"] = cost.usd(record)
    record["prices_date"] = cost.PRICES_DATE
    provider_data = response.provider_data or {}
    if isinstance(provider_data, dict) and isinstance(provider_data.get("usage"), dict):
        # OpenRouter bills each call itself and says so; keep it to check ours.
        record["provider_cost_usd"] = provider_data["usage"].get("cost")
        record["served_by"] = provider_data.get("provider")
    return record
