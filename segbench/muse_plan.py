"""Calls billed to our Muse Code plan (Meta), never to the pay-per-token key.

`muse login` (Meta's Muse Code CLI) saves the plan's own Model API key in
~/.config/muse/auth.json (providers.meta.api_key, with api_base_url and
subs_active). That key is not the META_API_KEY in .env: Meta bills it to
the plan while the plan is active (the plan's key mint returns the same key
on every call; lm15-contract receipts/2026-10-12-muse-code/REPORT.md).

segbench sends Muse models through their own lm15 router that holds only
that key and sees no environment, so the META_API_KEY loaded from .env can
never take over. If the login is missing or the plan is not active, the
call fails and says so; it never falls back to paying per token.

The file is read again for every call, so a fresh `muse login` is picked up
by a running benchmark.

Meta's Model API terms (2026-10-09, section 3.2) reserve this key for Muse
Code; using it here was Maxime's decision (2026-10-12, lm15-dev record
01a1320d: "Connect directly anyway").

Contributor mode: the plan also lists "-contributor" copies of each model
(Meta may train on what they receive). segbench names the plain ids
(muse-wonder-1.0, muse-wonder-full-1.0) on purpose, because the screenshots
show other people's names and messages.

When lm15 ships its own Muse Code connection (being added in lm15-dev on
2026-10-12, as a managed `meta` login), this module can be replaced by it.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.parse
from pathlib import Path

from lm15 import LMRouter
from lm15.router import RouterConfig

CREDENTIAL = "muse-code"  # Model.credential value that selects this module
AUTH_FILE = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "muse" / "auth.json"
DESCRIPTION = "Muse Code plan login (~/.config/muse/auth.json)"
KEY_MINT_URL = "https://api.meta.ai/muse-code/key"


class NotOnPlan(RuntimeError):
    """The Muse Code login is missing, unusable, or the plan is not active."""


def _fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:10]


def _login() -> dict:
    try:
        meta = json.loads(AUTH_FILE.read_text())["providers"]["meta"]
    except FileNotFoundError:
        raise NotOnPlan(f"no Muse Code login: {AUTH_FILE} is missing; run `muse login`") from None
    except (KeyError, TypeError, json.JSONDecodeError):
        raise NotOnPlan(f"no Muse Code login in {AUTH_FILE}; run `muse login`") from None
    key = meta.get("api_key")
    if not isinstance(key, str) or not key:
        raise NotOnPlan("the Muse Code login holds no API key; run `muse login`")
    if meta.get("subs_active") is not True:
        raise NotOnPlan("the Muse Code login says the plan is not active; calls would be billed per token")
    if meta.get("obtained_via") != "device_code":
        raise NotOnPlan("the Muse Code login is a pasted key, not the plan's browser sign-in")
    pay_per_token = os.environ.get("META_API_KEY")
    if pay_per_token and pay_per_token == key:
        raise NotOnPlan("the Muse Code login holds the same key as META_API_KEY (pay per token)")
    base = meta.get("api_base_url") or "https://api.meta.ai/v1"
    parsed = urllib.parse.urlsplit(base)
    host = parsed.hostname or ""
    if parsed.scheme != "https" or not (host == "meta.ai" or host.endswith(".meta.ai")):
        raise NotOnPlan(f"the Muse Code login names an API address outside meta.ai ({base!r}); not sending the key there")
    return {"key": key, "base_url": base.rstrip("/"), "access_token": meta.get("access_token")}


def router() -> tuple[LMRouter, dict]:
    """A router whose only Meta credential is the plan's key, and what to record about it."""
    login = _login()
    from .call import TIMEOUTS  # the same reply ceiling as every other model

    config = RouterConfig(env={}, api_keys={"meta": login["key"]}, base_urls={"meta": login["base_url"]},
                          timeouts=TIMEOUTS)
    return LMRouter(config), {"credential": DESCRIPTION, "credential_fingerprint": _fingerprint(login["key"]),
                              "base_url": login["base_url"]}


def usage() -> dict:
    """The plan's name and its 5-hour and weekly usage, as Meta reports them.

    Asks Meta's key mint, the request `muse` itself sends at sign-in (it
    returns the same key, so nothing changes). Personal and payment fields
    of the reply are dropped.
    """
    import urllib.request

    login = _login()
    if not login["access_token"]:
        raise NotOnPlan("the Muse Code login has no sign-in token; run `muse login`")
    request = urllib.request.Request(KEY_MINT_URL, data=b"{}", method="POST", headers={
        "Authorization": f"Bearer {login['access_token']}", "x-api-version": "1.0.0",
        "content-type": "application/json", "accept": "application/json", "user-agent": "segbench"})
    with urllib.request.urlopen(request, timeout=60) as reply:
        body = json.loads(reply.read())
    return {"plan": body.get("subs_tier_name"), "active": body.get("is_subs_active"),
            "usage": body.get("subs_usage"),
            "same_key_as_login": body.get("api_key") == login["key"]}
