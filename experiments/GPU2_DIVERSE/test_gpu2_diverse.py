from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("gpu2_diverse_utils", HERE / "diverse_utils.py")
utils = importlib.util.module_from_spec(spec); assert spec and spec.loader
sys.modules[spec.name] = utils; spec.loader.exec_module(utils)


class DiverseUtilityTests(unittest.TestCase):
    def test_numeric_validation_rejects_infinity_and_object(self):
        utils.validate_numeric_matrix(pd.DataFrame([[1.0, 2.0]], columns=["a", "b"]), np.array([1]), ["a", "b"], "ok")
        with self.assertRaisesRegex(ValueError, "infinite"):
            utils.validate_numeric_matrix(pd.DataFrame([[1.0, np.inf]], columns=["a", "b"]), np.array([1]), ["a", "b"], "bad")
        with self.assertRaisesRegex(TypeError, "object"):
            utils.validate_numeric_matrix(pd.DataFrame([["x"]], columns=["a"]), np.array([1]), ["a"], "bad")

    def test_probability_rows_have_canonical_schema(self):
        rows = utils.probability_rows(["Q1"], ["S2-A"], ["IN"], np.array([1]), [0.9], "catboost_probability")
        self.assertEqual(["Q1", "S2-A", "S2", "IN", 1, 0.9], [rows[key][0] for key in ["query_id", "candidate_id", "candidate_source", "country", "label", "catboost_probability"]])

    def test_alignment_is_order_independent_but_strict(self):
        left = pd.DataFrame({"query_id": ["Q2", "Q1"], "candidate_id": ["S2-B", "S2-A"], "candidate_source": ["S2", "S2"], "p": [0.2, 0.8]})
        right = pd.DataFrame({"query_id": ["Q1", "Q2"], "candidate_id": ["S2-A", "S2-B"], "candidate_source": ["S2", "S2"], "p": [0.7, 0.3]})
        a, b = utils.assert_canonical_alignment(left, right, "a", "b")
        self.assertTrue(a[utils.KEY_COLUMNS].equals(b[utils.KEY_COLUMNS]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            utils.assert_canonical_alignment(pd.concat([left, left.iloc[[0]]]), right, "a", "b")

    def test_blend_grid_math(self):
        result = utils.blend_probabilities([1.0, 0.0], [0.0, 1.0], 0.2)
        np.testing.assert_allclose([0.8, 0.2], result)

    def test_diversity_counts(self):
        truth = {"Q1": {"S2-A"}, "Q2": set()}
        primary = {"Q1": set(), "Q2": {"S2-X"}}
        auxiliary = {"Q1": {"S2-A"}, "Q2": set()}
        result = utils.diversity_report(truth, primary, auxiliary, [0.1, 0.9], [0.9, 0.1])
        self.assertEqual(1, result["positives_primary_misses_auxiliary_gets"])
        self.assertEqual(1, result["false_positives_primary_makes_auxiliary_rejects"])


if __name__ == "__main__":
    unittest.main()
