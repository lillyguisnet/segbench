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
from lm15.errors import RETRYABLE_ERRORS
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


def _gemini_media_resolution_patch() -> None:
    """Let a request ask Gemini for higher image detail.

    Gemini 3 takes the image's detail on the image part itself:
    {"inlineData": ..., "mediaResolution": {"level": "MEDIA_RESOLUTION_ULTRA_HIGH"}}.
    Measured 2026-10-09 on a 4000x2252 picture: default and HIGH 1,108 input
    tokens (HIGH is the default for images), ULTRA_HIGH 2,213; ULTRA_HIGH is
    refused in generationConfig. lm15 1.2.1 has no door for it (extensions
    land at the top level of the payload), so the extension key below is
    removed after lm15 builds the payload and set on every image part.
    """
    from lm15.providers.gemini import GeminiLM

    if getattr(GeminiLM._payload, "_segbench", False):
        return
    original = GeminiLM._payload

    def _payload(self, request):
        payload = original(self, request)
        level = payload.pop("segbench_media_resolution", None)
        if level:
            for content in payload.get("contents", []):
                for part in content.get("parts", []):
                    if "inlineData" in part or "fileData" in part:
                        part["mediaResolution"] = {"level": level}
        return payload

    _payload._segbench = True
    GeminiLM._payload = _payload


_gemini_media_resolution_patch()


def call(model: Model, image: bytes | None, media_type: str, prompt: str,
         effort: str | None = None, level: str | None = None, detail: str | None = None,
         media_resolution: str | None = None) -> dict:
    """Send one picture and one prompt to `model`; return the measured record.

    `level` is our thinking level ("min", "medium", "max"; see
    segbench.models.THINKING), translated into the model's own settings.
    `effort` is lm15's raw dial (off, minimal, low, medium, high, xhigh, max),
    for probing. Neither: the provider's default. `image=None` sends text only.
    """
    extensions = dict(model.extensions or {})
    if media_resolution:  # Gemini only; see _gemini_media_resolution_patch
        extensions["segbench_media_resolution"] = media_resolution
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
        "image_detail": detail,
    }
    if (model.billing == "subscription") != (route.provider in SUBSCRIPTION_PROVIDERS):
        record["error"] = f"route resolves to {route.provider!r}, which does not match billing {model.billing!r}"
        return record

    settings: dict = {} if route.provider == "openai-codex" else {"max_tokens": MAX_TOKENS}
    if extensions:
        settings["extensions"] = extensions
    if effort is not None:
        settings["reasoning"] = Reasoning(effort=effort)
    start = time.perf_counter()
    try:  # building the request can refuse a setting too (an effort word, an image detail)
        content = [prompt] if image is None else [image_part(data=image, media_type=media_type, detail=detail), prompt]
        request = Request(model=model.route, messages=(Message.user(content),), config=Config(**settings))
        response = _router.complete(request)
    except Exception as error:
        record["seconds"] = round(time.perf_counter() - start, 3)
        record["error"] = f"{type(error).__name__}: {error}"
        record["traceback"] = traceback.format_exc(limit=3)
        # Worth sending again: the provider never gave an answer (rate limit,
        # dropped connection, server busy). OpenAI's "overloaded" arrives as a
        # plain ProviderError, so it is matched by its words.
        record["retryable"] = isinstance(error, RETRYABLE_ERRORS) or "overloaded" in str(error).lower()
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
