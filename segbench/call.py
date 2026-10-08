"""One model call, measured: the reply, the tokens, the time, and any error.

Every call returns a plain dict that can be written as one JSON line. A
failed call is a record too (its `error` is set), never an exception, so a
run over many models always finishes and the failures stay on record.
"""

from __future__ import annotations

import time
import traceback
from datetime import datetime, timezone

from lm15 import Config, LMRouter, Message, Request
from lm15.serde import response_to_dict, usage_to_dict
from lm15.types import image as image_part

from .models import Model

SUBSCRIPTION_PROVIDERS = {"openai-codex", "claude-code"}

# Ceiling on the reply, thinking included. Generous on purpose: a model that
# thinks long before drawing 30 polygons must not be cut off. The Codex
# backend refuses any ceiling, so it gets none.
MAX_TOKENS = 32_000

_router = LMRouter()


def call(model: Model, image: bytes, media_type: str, prompt: str) -> dict:
    """Send one picture and one prompt to `model`; return the measured record."""
    route = _router.resolve(model.route)
    record: dict = {
        "model": model.key,
        "route": model.route,
        "provider": route.provider,
        "wire_model": route.model,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if (model.billing == "subscription") != (route.provider in SUBSCRIPTION_PROVIDERS):
        record["error"] = f"route resolves to {route.provider!r}, which does not match billing {model.billing!r}"
        return record

    config = Config() if route.provider == "openai-codex" else Config(max_tokens=MAX_TOKENS)
    request = Request(
        model=model.route,
        messages=(Message.user([image_part(data=image, media_type=media_type), prompt]),),
        config=config,
    )
    start = time.perf_counter()
    try:
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
    return record
