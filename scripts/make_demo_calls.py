"""Write made-up call records, to work on the charts before real results exist.

uv run scripts/make_demo_calls.py [OUT.jsonl]     (default: charts/demo/calls.jsonl)

Every record is marked "synthetic": true, and every chart drawn from them
says FAKE DATA across it. The numbers are shaped like the real thing
(costs from token counts and our list prices, thinking tokens from the
2026-10-08 checks, specialist costs and speeds from our GPU measurements),
but the scores are invented. Same seed, same file.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from segbench.models import BENCHMARKED, levels_for  # noqa: E402

PRICES = json.loads((ROOT / "prices.json").read_text())["snapshots"]["2026-10-08"]["models"]

TASKS = {
    "find": ["logs", "cows", "fig", "dishes"],
    "outline": ["logs", "cows", "fig", "tree", "road", "dishes"],
    "whole": ["logs", "cows", "fig", "tree", "road", "dishes"],
}
DIFFICULTY = {"logs": -0.10, "cows": -0.04, "fig": 0.0, "tree": 0.03, "road": 0.06, "dishes": -0.08}

# Invented skill: (finding, outlining), 0..1, at medium thinking.
SKILL = {
    "luna": (0.55, 0.33), "terra": (0.63, 0.40), "sonnet": (0.60, 0.38), "opus": (0.70, 0.45),
    "astra": (0.75, 0.49), "sol": (0.71, 0.46), "gemini-pro": (0.79, 0.60), "gemini-flash": (0.71, 0.53),
    "gemini-flash-lite": (0.56, 0.42), "glm-5.3-flash": (0.45, 0.28), "deepseek-flash": (0.42, 0.26),
    "kimi-k3": (0.61, 0.37), "qwen-27b": (0.58, 0.45),
}
LEVEL_GAIN = {"min": -0.07, "medium": 0.0, "max": 0.025}
# Thinking tokens per call (rough middle of the 2026-10-08 circle test, x1.5 for real photos).
THINK = {
    "luna": (0, 240, 1200), "terra": (0, 300, 10000), "sonnet": (0, 0, 2700), "opus": (0, 450, 6200),
    "astra": (0, 0, 3000), "sol": (0, 150, 2500), "gemini-pro": (0, 1600, 5000), "gemini-flash": (0, 550, 2500),
    "gemini-flash-lite": (0, 470, 520), "glm-5.3-flash": (140, 150, 3000), "deepseek-flash": (0, 2700, 2400),
    "kimi-k3": (0, 1900, 8000), "qwen-27b": (0, 350, 1000),
}
# (seconds before the first token, output tokens per second): invented, plausible.
SPEED = {
    "luna": (1.5, 180), "terra": (3.0, 60), "sonnet": (2.0, 70), "opus": (3.0, 45), "astra": (4.0, 50),
    "sol": (3.0, 70), "gemini-pro": (4.0, 110), "gemini-flash": (1.5, 220), "gemini-flash-lite": (0.8, 300),
    "glm-5.3-flash": (2.5, 60), "deepseek-flash": (1.5, 90), "kimi-k3": (3.5, 40), "qwen-27b": (1.0, 120),
}
ANSWER_TOKENS = {"find": 600, "outline": 3000, "whole": 3500}
IMAGE_TOKENS = 1600

# Specialists: track -> {model: (skill, usd per image under load, seconds per image, tasks it can do or None)}
GPU = {
    "find": {
        "sam3": (0.66, 1.1e-5, 0.18, None),
        "yoloe-26x": (0.52, 8e-7, 0.02, None),
        "rfdetr-seg-2xl": (0.72, 1.0e-6, 0.02, ["cows", "dishes"]),  # only its 80 COCO categories
        "dinov3": (0.40, 3e-6, 0.05, None),
    },
    "outline": {
        "sam1-vit-h": (0.74, 1.5e-5, 0.9, None),
        "sam2.1-large": (0.82, 4.4e-6, 0.35, None),
        "sam3": (0.84, 1.1e-5, 0.6, None),
        "gen2seg-sd": (0.60, 1.4e-5, 0.25, None),
        "dinov3": (0.70, 3e-6, 0.06, ["tree"]),
    },
    "whole": {
        "sam3": (0.68, 1.1e-5, 0.2, None),
        "yoloe-26x": (0.54, 8e-7, 0.02, None),
    },
}
# Pairs for the whole task: (finder, finder level, outliner, skill).
PAIRS = [
    ("gemini-flash", "medium", "sam2.1-large", 0.75),
    ("gemini-pro", "medium", "sam3", 0.81),
    ("qwen-27b", "medium", "sam2.1-large", 0.63),
    ("gemini-flash-lite", "min", "sam2.1-large", 0.58),
    ("gemini-flash", "min", "sam2.1-large", 0.70),
    ("gemini-pro", "min", "sam3", 0.74),
]


def clip(x: float) -> float:
    return min(1.0, max(0.0, x))


def llm_call(rng: random.Random, model: str, level: str, track: str) -> tuple[float, float]:
    """(cost in USD, seconds) of one made-up call."""
    think = THINK[model][("min", "medium", "max").index(level)] * rng.lognormvariate(0, 0.35)
    out = ANSWER_TOKENS[track] * rng.lognormvariate(0, 0.2) + think
    price = PRICES[model]
    cost = (IMAGE_TOKENS * price["input"] + out * price["output"]) / 1e6
    ttft, tps = SPEED[model]
    seconds = (ttft + out / tps) * rng.lognormvariate(0, 0.25)
    return cost, seconds


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "charts" / "demo" / "calls.jsonl"
    rng = random.Random(20261009)
    rows = []

    def add(**r):
        rows.append({**r, "synthetic": True})

    for track, tasks in TASKS.items():
        for m in BENCHMARKED:
            find, outline = SKILL[m.key]
            base = {"find": find, "outline": outline, "whole": 0.5 * (find + 0.9 * outline)}[track]
            model_bias = rng.gauss(0, 0.02)
            for level in levels_for(m.key):
                for task in tasks:
                    for rep in range(3):
                        cost, secs = llm_call(rng, m.key, level, track)
                        readable = rng.random() > 0.03
                        score = clip(base + LEVEL_GAIN[level] + DIFFICULTY[task] + model_bias + rng.gauss(0, 0.06))
                        add(track=track, task=task, model=m.key, level=level, repeat=rep,
                            score=round(score, 4) if readable else None, readable=readable,
                            cost_usd=round(cost, 7), seconds=round(secs, 3))
        for model, (skill, usd, secs, can) in GPU[track].items():
            for task in tasks:
                if can is not None and task not in can:
                    continue
                score = clip(skill + DIFFICULTY[task] + rng.gauss(0, 0.05))
                for rep in range(3):  # deterministic: scored once, timed three times
                    add(track=track, task=task, model=model, level=None, repeat=rep,
                        score=round(score, 4) if rep == 0 else None, readable=True,
                        cost_usd=usd, seconds=round(secs * rng.lognormvariate(0, 0.05), 4))
        if track == "whole":
            for finder, level, outliner, skill in PAIRS:
                o_usd, o_secs = GPU["outline"][outliner][1:3]
                for task in tasks:
                    for rep in range(3):
                        cost, secs = llm_call(rng, finder, level, "find")
                        score = clip(skill + DIFFICULTY[task] + rng.gauss(0, 0.05))
                        add(track=track, task=task, parts=[{"model": finder, "level": level}, {"model": outliner}],
                            repeat=rep, score=round(score, 4), readable=True,
                            cost_usd=round(cost + o_usd, 7), seconds=round(secs + o_secs, 3))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} made-up calls -> {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")


if __name__ == "__main__":
    main()
