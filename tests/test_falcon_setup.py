"""Falcon's photo timing setup, and the warm-up rule that is Falcon's alone."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import time_finders as TF  # noqa: E402
from segbench.local_models import BY_KEY, LOCAL_MODELS, photo_setups, screen_setups  # noqa: E402


class FalconSetup(unittest.TestCase):
    def test_photos_only(self):
        m = BY_KEY["falcon-perception-hd"]
        self.assertEqual(photo_setups()[m.key], ("falcon", "segment", "hd", (2,), 3))
        self.assertNotIn(m.key, screen_setups())
        self.assertFalse(m.plural)

    def test_warmup_photos_only_for_falcon(self):
        self.assertEqual([m.key for m in LOCAL_MODELS if m.warmup_photos], ["falcon-perception-hd"])

    def test_other_models_keep_their_single_cows_warmup(self):
        qs = TF.queries("rexomni-3b")
        warm = TF.warmups("rexomni-3b", "point", qs)
        self.assertEqual(len(warm), 1)
        self.assertEqual(warm[0]["id"], f"warmup|cows|{qs[1][1]}|99")

    def test_falcon_warms_every_photo(self):
        warm = TF.warmups("falcon-perception-hd", "segment", TF.queries("falcon-perception-hd"))
        self.assertEqual([w["id"].split("|")[1] for w in warm], list(TF.POINT_TASKS))


if __name__ == "__main__":
    unittest.main()
