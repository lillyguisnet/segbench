"""The model registry is complete, and moving the local models into one table changed nothing.

Run: uv run python -m unittest discover -s tests -p 'test_registry.py'   (offline, no model calls)
"""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import add_model  # noqa: E402
from segbench import protocol  # noqa: E402
from segbench.chart_data import SPECIALISTS  # noqa: E402
from segbench.local_models import LOCAL_MODELS, photo_setups, screen_setups  # noqa: E402
from segbench.models import BY_KEY, MODELS  # noqa: E402

# The tables as they were written in the scripts at commit 9ab38c9 (2026-10-12),
# before segbench/local_models.py. Published costs and runs depend on them.
SPECIALISTS_9AB38C9 = {
    "sam1-vit-h": ("SAM 1", "Meta"), "sam2.1-large": ("SAM 2.1", "Meta"), "sam3": ("SAM 3", "Meta"),
    "dinov3": ("DINOv3", "Meta"), "yoloe-26x": ("YOLOE-26x", "Ultralytics"),
    "rfdetr-seg-2xl": ("RF-DETR Seg", "Roboflow"), "gen2seg-sd": ("gen2seg", "UC Davis"),
    "locateanything-3b": ("LocateAnything 3B", "NVIDIA"), "molmopoint-8b": ("MolmoPoint 8B", "Ai2"),
    "rexomni-3b": ("Rex-Omni 3B", "IDEA"), "florence2-large": ("Florence-2 Large", "Microsoft"),
    "florence2-large-grounding": ("Florence-2 Large", "Microsoft"),
    "florence2-large-ovd": ("Florence-2 · detection", "Microsoft"), "moondream-3.1": ("Moondream 3.1", "Moondream"),
}
TIME_FINDERS_SETUPS_9AB38C9 = {  # scripts/time_finders.py SETUPS
    "locateanything-3b": ("locateanything", "point", None, (2,), 3),
    "molmopoint-8b": ("molmopoint", "point", None, (), 2),
    "rexomni-3b": ("rexomni", "point", None, (2,), 3),
    "florence2-large-grounding": ("florence2", "box", "large", (2, 4), 3),
    "florence2-large-ovd": ("florence2", "ovd", "large", (2, 4), 3),
    "moondream-3.1": ("moondream", "point", None, (), 3),
}
SCREENS_LOCAL_MODELS_9AB38C9 = {  # scripts/run_screens_local.py MODELS
    "locateanything-3b": ("locateanything", "point", None, 3, 1.0),
    "molmopoint-8b": ("molmopoint", "point", None, 1, 1.0),
    "rexomni-3b": ("rexomni", "point", None, 1, 0.6254),
    "florence2-large-grounding": ("florence2", "box", "large", 1, 0.4041),
    "florence2-large-ovd": ("florence2", "ovd", "large", 1, 0.528),
    "sam3": ("sam3", "segment", None, 1, 1.022),
    "moondream-3.1": ("moondream", "point", None, 1, 1.0),
}
PLURAL_MODELS_9AB38C9 = {"locateanything-3b", "molmopoint-8b"}  # scripts/run_finders.py


def _earlier(table: dict, snapshot: dict) -> dict:
    """The entries of `table` for the models in `snapshot`: models added later are left out."""
    return {k: v for k, v in table.items() if k in snapshot}


class LocalRegistryUnchanged(unittest.TestCase):
    """Every model as it was at 9ab38c9; models added since then are free to be added."""

    def test_chart_names(self):
        self.assertEqual(_earlier(SPECIALISTS, SPECIALISTS_9AB38C9), SPECIALISTS_9AB38C9)

    def test_photo_timing_setups(self):
        self.assertEqual(_earlier(photo_setups(), TIME_FINDERS_SETUPS_9AB38C9), TIME_FINDERS_SETUPS_9AB38C9)

    def test_screen_setups_and_load_factors(self):
        self.assertEqual(_earlier(screen_setups(), SCREENS_LOCAL_MODELS_9AB38C9), SCREENS_LOCAL_MODELS_9AB38C9)

    def test_plural_wording(self):
        self.assertEqual({m.key for m in LOCAL_MODELS if m.plural and m.key in SPECIALISTS_9AB38C9},
                         PLURAL_MODELS_9AB38C9)


class RegistryComplete(unittest.TestCase):
    """Fails when a model is half-registered (no price, no thinking level, unpinned host...)."""

    def test_no_registry_errors(self):
        self.assertEqual(add_model.lint_global(), [])
        for key in [m.key for m in MODELS] + [m.key for m in LOCAL_MODELS]:
            with self.subTest(key=key):
                self.assertEqual(add_model.lint_one(key)[0], [])


class LintCatchesMistakes(unittest.TestCase):
    base = BY_KEY["qwen-flash"]  # a complete OpenRouter model

    def errors(self, model):
        return " | ".join(add_model.lint_remote(model)[0])

    def test_complete_model_passes(self):
        self.assertEqual(self.errors(self.base), "")

    def test_unpinned_openrouter_host(self):
        self.assertIn("pinned host", self.errors(replace(self.base, extensions=None)))
        fallback = {"provider": {"order": ["Alibaba"], "allow_fallbacks": True}}
        self.assertIn("pinned host", self.errors(replace(self.base, extensions=fallback)))

    def test_missing_price_and_thinking(self):
        new = replace(self.base, key="not-registered-yet")
        self.assertIn("no price", self.errors(new))
        self.assertIn("no THINKING entry", self.errors(new))

    def test_route_without_prefix(self):
        self.assertIn("no provider prefix", self.errors(replace(self.base, route="qwen/qwen3.8-flash")))

    def test_subscription_must_be_on_a_plan(self):
        self.assertIn("does not match", self.errors(replace(self.base, billing="subscription")))
        luna = BY_KEY["luna"]
        self.assertIn("does not match", self.errors(replace(luna, billing="per-token")))
        self.assertEqual(self.errors(BY_KEY["muse-wonder"]), "")  # plan reached by its own credential


class ProtocolMatchesWhatWasRun(unittest.TestCase):
    def test_standard_run_sizes(self):
        # What every language model since 2026-10-09 has in the shared run files.
        sizes = {label: n for label, n, *_ in add_model.expected_runs("qwen-flash")}
        self.assertEqual(sizes, {"photos v2": 30, "photos v1-hd": 12, "screens round 2": 15})

    def test_published_files_exist(self):
        self.assertTrue((ROOT / "results" / protocol.PUBLISHED_PHOTOS / "summary.csv").is_file())
        self.assertTrue((ROOT / "charts" / protocol.PUBLISHED_SCREENS / "with-specialists" / "points.csv").is_file())
        self.assertTrue((ROOT / protocol.SCREEN_KEY).is_file())
        self.assertTrue((ROOT / protocol.PHOTO_REFERENCE).is_file())
        for name in protocol.LOCAL_PHOTO_RUNS:
            self.assertTrue((ROOT / "results" / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
