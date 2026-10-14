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
    # Where the key comes from when it is not lm15's usual one for the route.
    # "muse-code": our Muse Code plan's login (segbench/muse_plan.py).
    credential: str | None = None


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
    # Added 2026-10-10 (Maxime). OpenRouter's "Qwen3.8 Flash" links to the
    # Qwen/Qwen3.8-Flash-Next model page; Alibaba is its only host there.
    Model("qwen-flash", "Qwen 3.8 Flash Next", "Alibaba", "openrouter:qwen/qwen3.8-flash", "per-token", True,
          "OpenRouter name qwen/qwen3.8-flash (hugging_face_id Qwen/Qwen3.8-Flash-Next); pinned to Alibaba; circle test passed 2026-10-10",
          extensions={"provider": {"order": ["Alibaba"], "allow_fallbacks": False}}),
    # Added 2026-10-12 (Maxime). Closed weights: Perceptron's API, reached
    # through OpenRouter, whose only host for it is Perceptron itself.
    # OpenRouter does not pass Perceptron's own "vision_config" switch
    # (annotation_format): checked 2026-10-12, identical replies with and
    # without it. Its own answer format (tags) is tested as a prompt instead:
    # prompt version "tags" (segbench/tasks.py, segbench/perceptron.py).
    Model("mk1.5", "Perceptron Mk1.5", "Perceptron", "openrouter:perceptron/perceptron-mk1.5", "per-token", True,
          "OpenRouter perceptron/perceptron-mk1.5 (snapshot 20260925), pinned to Perceptron; circle test passed 2026-10-12",
          extensions={"provider": {"order": ["Perceptron"], "allow_fallbacks": False}}),
    # Added 2026-10-12 (Maxime): Meta's Muse Wonder and Muse Wonder Full,
    # released that day. Both are called on our Muse Code plan (Power Usage),
    # with the plan's own key from `muse login`, never the pay-per-token
    # META_API_KEY: segbench/muse_plan.py. Full exists only on the plan (the
    # pay-per-token key gets HTTP 404 for it). The plain ids, not the
    # "-contributor" copies Meta may train on (see muse_plan.py).
    Model("muse-wonder", "Muse Wonder", "Meta", "meta:muse-wonder-1.0", "subscription", True,
          "circle test passed 2026-10-12 (pay-per-token key, then on the plan); cannot stop thinking",
          credential="muse-code"),
    Model("muse-wonder-full", "Muse Wonder Full", "Meta", "meta:muse-wonder-full-1.0", "subscription", True,
          "Muse Code Power or Maximum Usage plan only; circle test passed on the plan 2026-10-12 (4 of 4, 32 s); "
          "no per-token price exists (see prices.json)",
          credential="muse-code"),
    # Added 2026-10-13 (Maxime). Mistral Large 4 (released 2026-10-06, "Public
    # Preview": Mistral says it keeps training it under the same name, so a
    # score holds for its run date). Weights promised for end of October 2026;
    # 1.05T parameters, too big for our GPUs anyway. Reached through OpenRouter,
    # whose only host for it is Mistral itself.
    Model("mistral-large-4", "Mistral Large 4", "Mistral", "openrouter:mistralai/mistral-large-4-0", "per-token", True,
          "OpenRouter mistralai/mistral-large-4-0 (snapshot 20261006), pinned to Mistral; circle test passed 3 of 3 on 2026-10-12",
          extensions={"provider": {"order": ["Mistral"], "allow_fallbacks": False}}),
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
    # Checked 2026-10-10 (scripts/probe_thinking.py): off 0 thinking tokens, medium accepted.
    "qwen-flash":        {"min": {"effort": "off"}, "medium": {"effort": "medium"}, "max": {"effort": "max"}},
    # OpenRouter lists none, minimal, low, medium, high (default high) for Mk1.5.
    "mk1.5":             {"min": {"effort": "off"}, "medium": {"effort": "medium"}, "max": {"effort": "high"}},
    # Checked 2026-10-12 (scripts/probe_thinking.py): Meta refuses "none" for
    # both and accepts minimal, low, medium, high, xhigh, max. Thinking tokens
    # on the test question, one try each: Wonder 178, 291, 254, 369, 350, 630
    # (default 415); Full on the plan 151, 289, 281, 323, 209, 209 (default
    # 367). Like GPT-6 Astra and Sol, they cannot stop thinking: min is their
    # lowest level, medium the middle.
    "muse-wonder":       {"min": {"effort": "minimal"}, "medium": {"effort": "medium"}, "max": {"effort": "max"}},
    "muse-wonder-full":  {"min": {"effort": "minimal"}, "medium": {"effort": "medium"}, "max": {"effort": "max"}},
    # Mistral's docs (docs.mistral.ai/studio/conversations/reasoning, read
    # 2026-10-13) give Large 4 two reasoning_effort values only: "none" and
    # "high". OpenRouter accepts the other words, but on 2026-10-12 they gave
    # no ordered effect (thinking tokens on the test question, one try each:
    # off 0, minimal 690, low 394, medium 1038, high 261, xhigh 574, max 396).
    # So, like Kimi K3: min = off, medium = its one thinking level (high), and
    # no max (it would repeat high). Maxime, 2026-10-13: run it with full thinking.
    "mistral-large-4":   {"min": {"effort": "off"}, "medium": {"effort": "high"}},
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
