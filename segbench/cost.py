"""Turn a call's token counts into US dollars, from the dated list prices in prices.json."""

from __future__ import annotations

import json
from pathlib import Path

PRICES_FILE = Path(__file__).resolve().parent.parent / "prices.json"
PRICES_DATE = "2026-10-08"

# Providers whose `output_tokens` leaves out the thinking tokens, which are
# still billed at the output price (lm15's Usage is provider-verbatim).
THINKING_NOT_IN_OUTPUT = {"gemini", "xai"}


def _table(date: str) -> dict:
    return json.loads(PRICES_FILE.read_text())["snapshots"][date]["models"]


def billed_output_tokens(provider: str, usage: dict) -> int | None:
    out = usage.get("output_tokens")
    if out is None:
        return None
    if provider in THINKING_NOT_IN_OUTPUT:
        out += usage.get("reasoning_tokens") or 0
    return out


def usd(record: dict, date: str = PRICES_DATE) -> float | None:
    """List-price cost of one call record from segbench.call, or None when unknown."""
    usage = record.get("usage")
    price = _table(date).get(record["model"])
    if not usage or price is None:
        return None
    tokens_in = usage.get("input_tokens")
    tokens_out = billed_output_tokens(record["provider"], usage)
    if tokens_in is None or tokens_out is None:
        return None
    return (tokens_in * price["input"] + tokens_out * price["output"]) / 1_000_000
