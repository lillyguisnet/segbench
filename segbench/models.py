"""The remote models we benchmark, and the exact route each one is called by.

A route is an lm15 model string with an explicit provider prefix. Write the
prefix every time: without it lm15 may pick another provider for the same
name (for example the pay-per-token OpenAI API instead of our subscription).

`vision` is what we observed, not what a provider page claims: run
`uv run scripts/check_models.py` and update it. Checked 2026-10-08.

Open-weight models served by many hosts (through OpenRouter) are pinned to
one host. On 2026-10-08 the same Qwen 3.8 27B found all four test circles 3
times out of 3 on Alibaba's host and 0 times out of 3 on Cerebras's.
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
    extensions: dict | None = None  # extra request fields sent as-is (lm15 Config.extensions)
    # Three thinking levels, each as lm15 settings: {"effort": word} and/or
    # {"extensions": {...}}. See THINKING below for why each is what it is.
    thinking: dict | None = None


MODELS: tuple[Model, ...] = (
    # Our ChatGPT and Claude plans (no per-token bill; see README for how cost is counted).
    Model("luna", "GPT-6 Luna", "OpenAI", "openai-codex:gpt-6-luna", "subscription", True,
          "not in the plan's model list, but callable; checked on the plan 2026-10-09"),
    Model("terra", "GPT-5.6 Terra", "OpenAI", "openai-codex:gpt-5.6-terra", "subscription", True,
          "newest Terra on 2026-10-08: there is no GPT-6 Terra"),
    Model("sonnet", "Claude Sonnet 5.5", "Anthropic", "claude-code:claude-sonnet-5-5", "subscription", True),
    Model("opus", "Claude Opus 5.5", "Anthropic", "claude-code:claude-opus-5-5", "subscription", True),
    Model("astra", "GPT-6 Astra", "OpenAI", "openai-codex:gpt-6-astra", "subscription", True,
          "listed only to Codex clients >= 0.153 (lm15 claims 0.147), but callable"),
    Model("sol", "GPT-6.1 Sol", "OpenAI", "openai-codex:gpt-6.1-sol", "subscription", True,
          "not in the plan's model list on 2026-10-08, but callable"),
    # Google, direct.
    Model("gemini-pro", "Gemini 3.1 Pro", "Google", "gemini:gemini-3.1-pro-preview", "per-token", True,
          "newest Pro on 2026-10-08; still a preview"),
    Model("gemini-flash", "Gemini 3.8 Flash", "Google", "gemini:gemini-3.8-flash", "per-token", True,
          "answered HTTP 503 'high demand' once on 2026-10-08"),
    Model("gemini-flash-lite", "Gemini 3.5 Flash Lite", "Google", "gemini:gemini-3.5-flash-lite", "per-token", True,
          "writes points as [y, x] (Google's own habit) although the prompt asks for [x, y]"),
    # Z.AI, direct.
    Model("glm-5.3", "GLM 5.3", "Z.AI", "zai:glm-5.3", "per-token", False,
          "text only: Z.AI rejects the image (HTTP 400, 'allowed values: text')"),
    Model("glm-5.3-flash", "GLM 5.3 Flash", "Z.AI", "zai:glm-5.3-flash", "per-token", True,
          "thinks long: ~9,000 thinking tokens and 140 s for the 4-circle test"),
    # DeepSeek, direct.
    Model("deepseek-pro", "DeepSeek V4 Pro", "DeepSeek", "deepseek:deepseek-v4-pro", "per-token", False,
          "text only, and SILENT about it: DeepSeek swaps the picture for '[Unsupported Image]' and answers anyway"),
    Model("deepseek-flash", "DeepSeek Flash", "DeepSeek", "deepseek:deepseek-flash", "per-token", True,
          "the name points to DeepSeek V4.1 Flash (since 2026-09-10)"),
    # Moonshot, direct.
    Model("kimi-k3", "Kimi K3", "Moonshot", "moonshotai:kimi-k3", "per-token", True,
          "thinks long: ~6,000 thinking tokens and 140 s for the 4-circle test"),
    # Through OpenRouter: no direct key for Alibaba's Qwen.
    Model("qwen-27b", "Qwen 3.8 27B", "Alibaba", "openrouter:qwen/qwen3.8-27b", "per-token", True,
          "newest 27B Qwen on 2026-10-08; pinned to Alibaba's host (see above)",
          extensions={"provider": {"order": ["Alibaba"], "allow_fallbacks": False}}),
)

# Minimum, medium and maximum thinking, in each model's own levels (checked
# 2026-10-08 by sending each lm15 effort word and reading the request on the
# wire and the thinking tokens back; scripts/probe_thinking.py):
# - minimum = off where the model can stop thinking, else its lowest level;
#   maximum = its highest level that is really sent (lm15 turns Gemini's
#   xhigh/max into high, so high is Gemini's top); medium = the middle of
#   the levels the model really has.
# - Sonnet 5.5: lm15 1.2.1's effort="off" sends nothing, and Sonnet 5.5
#   thinks by default; Anthropic's own off for this model is
#   thinking={"type": "between_tools"}. Sent here directly.
# - GLM 5.3 Flash has only low, high, max and cannot stop thinking.
# - Opus 5.5, GPT-6 Astra and GPT-6.1 Sol cannot stop thinking: Anthropic
#   refuses both "disabled" and "between_tools" for Opus 5.5; OpenAI accepts
#   only low, medium, high, xhigh, max for Astra and Sol. Their min is low.
# - Kimi K3 has off, low, high, max (lm15 sends minimal/medium as low).
_OFF_SONNET = {"extensions": {"thinking": {"type": "between_tools"}}}
THINKING = {
    "luna":              {"min": {"effort": "off"}, "medium": {"effort": "medium"}, "max": {"effort": "max"}},
    "terra":             {"min": {"effort": "off"}, "medium": {"effort": "medium"}, "max": {"effort": "max"}},
    "sonnet":            {"min": _OFF_SONNET, "medium": {"effort": "medium"}, "max": {"effort": "max"}},
    "opus":              {"min": {"effort": "low"}, "medium": {"effort": "medium"}, "max": {"effort": "max"}},
    "astra":             {"min": {"effort": "low"}, "medium": {"effort": "medium"}},
    "sol":               {"min": {"effort": "low"}, "medium": {"effort": "medium"}, "max": {"effort": "max"}},
    "gemini-pro":        {"min": {"effort": "low"}, "medium": {"effort": "medium"}, "max": {"effort": "high"}},
    "gemini-flash":      {"min": {"effort": "low"}, "medium": {"effort": "medium"}, "max": {"effort": "high"}},
    "gemini-flash-lite": {"min": {"effort": "minimal"}, "medium": {"effort": "medium"}, "max": {"effort": "high"}},
    "glm-5.3-flash":     {"min": {"effort": "low"}, "medium": {"effort": "high"}, "max": {"effort": "max"}},
    "deepseek-flash":    {"min": {"effort": "off"}, "medium": {"effort": "medium"}, "max": {"effort": "max"}},
    "kimi-k3":           {"min": {"effort": "off"}, "medium": {"effort": "high"}},
    "qwen-27b":          {"min": {"effort": "off"}, "medium": {"effort": "medium"}, "max": {"effort": "max"}},
}
LEVELS = ("min", "medium", "max")
# Dropped 2026-10-09: "max" for Kimi K3 and GPT-6 Astra. Measured at up to
# $0.16 and $0.24 per call on one small picture (and up to 4 minutes for
# Kimi), too expensive for the benchmark's ~thousands of calls.


def levels_for(key: str) -> tuple[str, ...]:
    """The thinking levels a model is run at, in LEVELS order."""
    return tuple(level for level in LEVELS if level in THINKING[key])

BY_KEY = {m.key: m for m in MODELS}

# The models that can take part: they see the picture.
BENCHMARKED: tuple[Model, ...] = tuple(m for m in MODELS if m.vision)
