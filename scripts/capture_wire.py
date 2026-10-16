"""What each model would be sent, captured before it leaves the machine (nothing is sent).

    uv run --no-project --with 'lm15==X' scripts/capture_wire.py OUT.json

For checking an lm15 upgrade (2026-10-16, 1.2.1 -> 1.5.0): run it under the
old and the new version and compare the two files. Every model of
segbench/models.py, at each of its thinking levels, with the image settings
the benchmark runs use (scripts/run.py image_settings), goes through
segbench.call.call exactly as a run does; the transport is replaced by one
that records the request (address, method, headers, body) and stops.

Secrets are not saved: authorization headers and anything that looks like a
key or token become "<redacted>", so two versions compare equal when they
send the same key. Requests a provider makes before the model call (a login
refresh) are recorded too, in order.
"""
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lm15  # noqa: E402
from lm15 import transports  # noqa: E402

SECRET_HEADERS = {"authorization", "x-api-key", "x-goog-api-key", "api-key", "cookie", "chatgpt-account-id",
                  "x-stainless-arch", "x-stainless-os", "x-stainless-runtime-version", "user-agent",
                  "x-stainless-package-version", "anthropic-client-version", "x-client-request-id",
                  "session_id", "x-request-id", "idempotency-key"}


class Captured(Exception):
    pass


CAPTURED: list = []


def _body(raw):
    if raw is None:
        return None
    if isinstance(raw, (bytes, bytearray)):
        try:
            return json.loads(raw)
        except ValueError:
            return {"<bytes sha/len>": len(raw), "head": base64.b64encode(bytes(raw[:64])).decode()}
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except ValueError:
            return raw
    return repr(raw)


def _shrink(x):
    """Long base64 (the picture) is replaced by its length and ends: the same picture compares equal."""
    if isinstance(x, dict):
        return {k: _shrink(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_shrink(v) for v in x]
    if isinstance(x, str) and len(x) > 500:
        return f"<{len(x)} chars {x[:24]}...{x[-24:]}>"
    return x


def _record(req):
    url = getattr(req, "url", None)
    if url and "key=" in url:
        url = url.split("key=")[0] + "key=<redacted>"
    headers = getattr(req, "headers", None) or []
    if isinstance(headers, dict):
        headers = list(headers.items())
    kept = sorted((k.lower(), "<redacted>" if k.lower() in SECRET_HEADERS else v) for k, v in headers)
    CAPTURED.append({"method": getattr(req, "method", None), "url": url, "headers": kept,
                     "body": _shrink(_body(getattr(req, "body", None) or getattr(req, "content", None)))})
    raise Captured()


def _patch():
    for name in dir(transports):
        cls = getattr(transports, name)
        if isinstance(cls, type):
            for meth in ("stream", "request", "send"):
                if callable(getattr(cls, meth, None)):
                    setattr(cls, meth, lambda self, req, *a, **k: _record(req))
    import urllib.request
    urllib.request.urlopen = lambda req, *a, **k: _record(req)


def main():
    out = Path(sys.argv[1])
    _patch()
    from run import image_settings  # noqa: E402
    from segbench.call import call
    from segbench.models import MODELS, THINKING, levels_for
    picture = b"\xff\xd8\xff\xe0" + bytes(range(256)) * 8  # a stand-in: same bytes for both versions
    result = {"lm15": lm15.__version__, "calls": {}}
    for model in MODELS:
        for level in (levels_for(model.key) if model.key in THINKING else (None,)):
            for version in ("v2", "box-a"):
                CAPTURED.clear()
                extra = image_settings(model, version)
                rec = call(model, picture, "image/jpeg", "Find each cow in this photo.", level=level, **extra)
                result["calls"][f"{model.key} {level} {version}"] = {
                    "requests": list(CAPTURED),
                    "error": None if "Captured" in (rec.get("error") or "Captured") else rec.get("error")}
    out.write_text(json.dumps(result, indent=1, sort_keys=True, default=repr))
    n = len(result["calls"])
    bad = [k for k, v in result["calls"].items() if not v["requests"]]
    print(f"lm15 {lm15.__version__}: {n} calls, {n - len(bad)} captured; not captured: {bad}")


if __name__ == "__main__":
    main()
