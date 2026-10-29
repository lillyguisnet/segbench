"""CPU-only checks of coverage, scoring files and GPU safety; no models loaded."""
import argparse
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import numpy as np
    import pyarrow  # noqa: F401 -- test runner dependency availability
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("Run with scripts/pixmo_eval.py's numpy/scipy/pyarrow/pillow dependencies") from exc

PATH = Path(__file__).resolve().parents[1] / "scripts/pixmo_eval.py"
SPEC = importlib.util.spec_from_file_location("pixmo_eval_runner", PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class PixMoRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name) / "data"
        self.base.mkdir()
        mask = self.base / "masks.npz"
        np.savez_compressed(mask, masks=np.ones((1, 10, 20), dtype=bool))
        common = {"masks": "masks.npz", "masks_sha256": runner.digest(mask.read_bytes()),
                  "width": 40, "height": 20, "mask_width": 20, "mask_height": 10,
                  "status": "verified", "label": "target"}
        self.examples = [{**common, "id": "present", "reference_percent": [[50, 50]]},
                         {**common, "id": "absent", "reference_percent": []}]
        self.manifest = self.base / "examples.jsonl"
        self.manifest.write_text("".join(json.dumps(e) + "\n" for e in self.examples))
        self.dest = Path(self.temp.name) / "run"
        self.dest.mkdir()
        runner.save_json(self.dest / "config.json", {
            "data": str(self.base), "manifest_sha256": runner.digest(self.manifest.read_bytes()),
            "selected_ids": ["present", "absent"], "model": "fixture-not-a-model-run"})
        self.present = {"id": "present", "width": 40, "height": 20, "seconds": 0.1,
                        "objects": [{"point": [20, 10], "point_from": "model"}]}
        self.absent = {"id": "absent", "width": 40, "height": 20, "seconds": 0.1, "objects": []}

    def score(self, records):
        (self.dest / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        with contextlib.redirect_stdout(io.StringIO()):
            runner.score_run(argparse.Namespace(run=self.dest))
        return json.loads((self.dest / "summary.json").read_text())

    def test_rescale_and_absent_target_macro_score(self):
        summary = self.score([self.present, self.absent])
        self.assertAlmostEqual(summary["f1"], 1)
        self.assertTrue(summary["complete_selected_run"])
        self.assertTrue(summary["full_test_set"])

    def test_missing_predictions_not_silently_complete(self):
        summary = self.score([self.present])
        self.assertFalse(summary["complete_selected_run"])
        self.assertFalse(summary["full_test_set"])
        self.assertEqual(summary["missing_prediction_ids"], ["absent"])
        self.assertEqual(summary["scored_questions"], 1)

    def test_reject_modified_masks(self):
        (self.base / "masks.npz").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "mask checksum"):
            self.score([self.present])

    def test_reject_modified_manifest(self):
        self.manifest.write_text(self.manifest.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            self.score([self.present])

    def test_reject_duplicate_predictions(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.score([self.present, self.present])

    def test_busy_gpu_refused_before_launch(self):
        finders = Path(self.temp.name) / "finders"
        (finders / "models/molmopoint/.venv").mkdir(parents=True)
        out = Path(self.temp.name) / "new-run"
        args = argparse.Namespace(data=self.base, finders=finders, out=out, limit=1, seed=0,
                                  model="molmopoint-8b", gpu="0", resume=False,
                                  allow_partial=False, allow_shared_gpu=False)
        with patch.object(runner, "gpu_state", return_value={"processes": ["existing-job"], "utilization": 90}), \
                patch.object(runner.subprocess, "Popen") as launch, \
                self.assertRaisesRegex(ValueError, "GPU is busy"):
            runner.run(args)
        launch.assert_not_called()
        self.assertFalse(out.exists())

    def test_incomplete_dataset_requires_explicit_opt_in(self):
        self.examples[1]["status"] = "unavailable"
        self.manifest.write_text("".join(json.dumps(e) + "\n" for e in self.examples))
        args = argparse.Namespace(data=self.base, finders=Path("unused"), out=Path("unused"),
                                  limit=None, seed=0, allow_partial=False)
        with self.assertRaisesRegex(ValueError, "--allow-partial"):
            runner.run(args)


if __name__ == "__main__":
    unittest.main()
