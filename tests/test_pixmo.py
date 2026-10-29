"""CPU-only tests; numpy/scipy are the dependencies of scripts/pixmo_eval.py."""
import ast
import hashlib
import unittest
from collections import defaultdict
from pathlib import Path

try:
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    from segbench.pixmo import score_points
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("PixMo tests need the numpy/scipy environment documented in docs/pixmo-eval.md") from exc


class PixMoTests(unittest.TestCase):
    def test_percentage_reference_and_mask_membership(self):
        masks = np.zeros((1, 20, 40), dtype=bool)
        masks[0, 8:13, 18:23] = True
        s = score_points([[20, 10]], [[50, 50]], masks)
        self.assertEqual(s["hits"], 1)
        self.assertAlmostEqual(s["f1"], 1)
        # Valid points don't have to equal the reference dot.
        self.assertEqual(score_points([[18, 8]], [[50, 50]], masks)["hits"], 1)

    def test_duplicates_and_outside_points_are_not_repaired(self):
        masks = np.ones((1, 10, 10), dtype=bool)
        s = score_points([[5, 5], [5, 5], [-1, 5]], [[50, 50]], masks)
        self.assertEqual((s["hits"], s["predicted"], s["reference"]), (1, 3, 1))
        self.assertAlmostEqual(s["f1"], 0.5)
        self.assertEqual(score_points([[10, 5]], [[50, 50]], masks)["f1"], 0)

    def test_rounding_matches_official_python_round(self):
        masks = np.zeros((1, 4, 4), dtype=bool)
        masks[0, 2, 2] = True
        self.assertEqual(score_points([[1.5, 2.5]], [[50, 50]], masks)["hits"], 1)
        self.assertEqual(score_points([[2.51, 2]], [[50, 50]], masks)["hits"], 0)

    def test_empty_answers(self):
        self.assertEqual(score_points([], [[50, 50]], np.ones((1, 4, 4)))["f1"], 0)
        self.assertEqual(score_points([], [], [])["f1"], 1)
        self.assertEqual(score_points([[1, 1]], [], [])["f1"], 0)
        # The real dataset stores one dummy mask for some absent targets.
        self.assertEqual(score_points([], [], np.zeros((1, 4, 4)))["f1"], 1)
        self.assertEqual(score_points([[1, 1]], [], np.zeros((1, 4, 4)))["f1"], 0)

    def test_distance_assignment_not_maximum_mask_matching(self):
        masks = np.zeros((2, 10, 10), dtype=bool)
        # Nearest-dot assignment maps each prediction to a mask it misses.
        # A swapped, maximum-mask-hit assignment would incorrectly give 100%.
        masks[0, 5, 8] = True
        masks[1, 5, 2] = True
        s = score_points([[2, 5], [8, 5]], [[20, 50], [80, 50]], masks)
        self.assertEqual(s["hits"], 0)

    def test_reject_nonfinite_or_wrong_reference_shape(self):
        with self.assertRaises(ValueError):
            score_points([[float("nan"), 1]], [[50, 50]], np.ones((1, 4, 4)))
        with self.assertRaises(ValueError):
            score_points([], [[50, 50]], [])

    def test_against_pinned_authors_implementation(self):
        source = Path(__file__).resolve().parents[1] / "cache/pixmo-points-eval/source/evaluators.py"
        if not source.exists():
            self.skipTest("run pixmo_eval.py prepare to cache the pinned official evaluator")
        # Load only metric functions, not the upstream torch/model dependencies.
        names = {"is_point_in_region", "compute_precision", "compute_recall", "f1_score", "PointingEval"}
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),
                         "db774284f70f6e501c5d487b146dc2d92ba7df58f36c56b58d5a56fa57a5e081")
        tree = ast.parse(source.read_text())
        tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                     and node.name in names]
        ns = {"np": np, "Tuple": tuple, "List": list, "Evaluator": object,
              "defaultdict": defaultdict, "linear_sum_assignment": linear_sum_assignment, "cdist": cdist,
              "extract_points": lambda pred, w, h: pred, "mean_metric": lambda xs: float(np.mean(xs))}
        exec(compile(tree, str(source), "exec"), ns)  # noqa: S102 -- checksum-verified reference metric
        official = ns["PointingEval"]()
        rng = np.random.default_rng(42)
        for _ in range(150):
            n_ref, n_pred = rng.integers(0, 8, size=2)
            masks = rng.random((n_ref, 12, 20)) > 0.6
            refs = rng.uniform(0, 100, size=(n_ref, 2))
            pred = rng.uniform([-3, -3], [23, 15], size=(n_pred, 2)).tolist()
            meta = {"gt_points": refs * [0.2, 0.12], "masks": masks, "image_size": (20, 12)}
            expected = official([meta], {"predictions_text": [pred]}, None)
            actual = score_points(pred, refs, masks)
            for key in ("precision", "recall", "f1"):
                self.assertAlmostEqual(actual[key], expected[key])


if __name__ == "__main__":
    unittest.main()
