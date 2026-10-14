"""The standard treatment every new entrant gets, and the published results it joins.

Until 2026-10-13 this lived only in conversations: each new model's session
had to dig out which runs the others had (Mk1.5, Muse, Mistral). It is the
protocol of point benchmark v2 (docs/point-benchmark-v2.md) and screen
round 2 (segbench/screens.py), as run for every language model since
2026-10-09. scripts/add_model.py reads it; so do the site builders.

Changing anything here changes what "the same treatment as every model"
means: do it only with a new benchmark version, never for one model.
"""

from __future__ import annotations

from dataclasses import dataclass

# Every language model: its middle thinking level (segbench/models.py
# THINKING "medium"; for a model with only off and on, "on" sits there, as
# for Kimi K3, GLM 5.3 Flash and Mistral Large 4), 3 tries of each question.
LEVEL = "medium"
REPEATS = 3


@dataclass(frozen=True)
class PhotoRun:
    name: str  # results/run-<name>.jsonl, shared by every model
    prompt_version: str
    tracks: tuple[str, ...]


# Photos: both fixed prompts; each model keeps the better one over the four
# tasks (scripts/select_best_prompt.py, ties to v2). Trick questions (things
# not in the photo) are asked once, with prompt v2.
PHOTO_RUNS = (
    PhotoRun("points-v2", "v2", ("find", "trick")),
    PhotoRun("points-v1hd", "v1-hd", ("find",)),
)
PROMPT_TIE_GOES_TO = "v2"

# Screenshots: the harder round only (Maxime, 2026-10-11 and 2026-10-13),
# naive prompt, generous reading.
SCREEN_RUN = "screens-2"
SCREEN_ROUND = 2
SCREEN_KEY = "annotations/screens/key-round2.json"

# How photo answers are scored for the published chart (scripts/score_points.py).
PHOTO_SCORING = {"parser": "lenient", "dish_labels": "separate", "method": "docs/point-benchmark-v2.md"}
PHOTO_REFERENCE = "annotations/reviews/segbench-review-6187f623-approved.json"

# What is published now. A new entrant gets a new version (v2.7, ...); move
# these when it is published, and the sites follow (scripts/build_site.py).
PUBLISHED_PHOTOS = "point-benchmark-v2.6"  # results/<this>/ (scores) and charts/<this>/ (chart)
PUBLISHED_SCREENS = "screens-2"  # charts/<this>/

# Local models' answer files on the photos, as chosen into the published
# selection by scripts/select_best_prompt.py.
LOCAL_PHOTO_RUNS = ("run-specialists-2.jsonl", "run-finders-3.jsonl", "run-rfdetr-3.jsonl", "run-finders-4.jsonl",
                    "run-finders-5.jsonl")
