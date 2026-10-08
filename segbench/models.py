"""The remote models we benchmark, and the exact route each one is called by.

A route is an lm15 model string with an explicit provider prefix. Write the
prefix every time: without it lm15 may pick another provider for the same
name (for example the pay-per-token OpenAI API instead of our subscription).

`vision` is what we observed, not what a provider page claims: run
`uv run scripts/check_models.py` and update it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    key: str  # short name used in results files and on the chart
    label: str  # name shown on the chart
    maker: str
    route: str  # lm15 model string, provider prefix included
    billing: str  # "subscription" or "per-token"
    vision: bool | None  # None = not checked yet
    note: str = ""


MODELS: tuple[Model, ...] = (
    # Our ChatGPT and Claude plans (no per-token bill; see README for how cost is counted).
    Model("luna", "GPT-5.6 Luna", "OpenAI", "openai-codex:gpt-5.6-luna", "subscription", True),
    Model("terra", "GPT-5.6 Terra", "OpenAI", "openai-codex:gpt-5.6-terra", "subscription", True),
    Model("sonnet", "Claude Sonnet 5.5", "Anthropic", "claude-code:claude-sonnet-5-5", "subscription", True),
    # Google, direct.
    Model("gemini-pro", "Gemini 3.1 Pro", "Google", "gemini:gemini-3.1-pro-preview", "per-token", None,
          "newest Pro on 2026-10-08; still a preview"),
    Model("gemini-flash", "Gemini 3.8 Flash", "Google", "gemini:gemini-3.8-flash", "per-token", None),
    Model("gemini-flash-lite", "Gemini 3.5 Flash Lite", "Google", "gemini:gemini-3.5-flash-lite", "per-token", None),
    # Z.AI, direct.
    Model("glm-5.3", "GLM 5.3", "Z.AI", "zai:glm-5.3", "per-token", None,
          "OpenRouter lists it as text-only"),
    Model("glm-5.3-flash", "GLM 5.3 Flash", "Z.AI", "zai:glm-5.3-flash", "per-token", None),
    # DeepSeek, direct.
    Model("deepseek-pro", "DeepSeek V4 Pro", "DeepSeek", "deepseek:deepseek-v4-pro", "per-token", None,
          "OpenRouter lists it as text-only"),
    Model("deepseek-flash", "DeepSeek Flash", "DeepSeek", "deepseek:deepseek-flash", "per-token", None),
    # Moonshot, direct.
    Model("kimi-k3", "Kimi K3", "Moonshot", "moonshotai:kimi-k3", "per-token", None),
    # Through OpenRouter: no direct key for Alibaba's Qwen.
    Model("qwen-27b", "Qwen 3.8 27B", "Alibaba", "openrouter:qwen/qwen3.8-27b", "per-token", None,
          "newest 27B Qwen on OpenRouter on 2026-10-08"),
)

BY_KEY = {m.key: m for m in MODELS}
