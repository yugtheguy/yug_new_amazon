from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_b006a as audit  # noqa: E402


class B006AForensicTests(unittest.TestCase):
    def test_canonical_key_includes_source(self):
        self.assertEqual(("S1-1", "S2-9", "S2"), audit.canonical_key("S1-1", "S2-9"))
        self.assertEqual(("S1-1", "S3-9", "S3"), audit.canonical_key("S1-1", "S3-9"))

    def test_ablation_zero_is_exact_base_schema(self):
        self.assertEqual([], audit.ABLATIONS["B006A_00_REPRO"])
        self.assertEqual(32, len(audit.B006.BASE_FEATURES))

    def test_families_are_isolated(self):
        self.assertEqual(audit.FLAGS, audit.ABLATIONS["B006A_01_FLAGS"])
        self.assertEqual(audit.RECIPROCAL, audit.ABLATIONS["B006A_05_RECIPROCAL"])
        self.assertTrue(set(audit.GAPS).isdisjoint(audit.ABLATIONS["B006A_07_ALL_EXCEPT_GAPS"]))

    def test_gap_sign_correction_only_for_a06(self):
        matrix = np.zeros((1, len(audit.B006.ALL_FEATURES)), dtype=np.float32)
        gap = audit.GAPS[2]; matrix[0, audit.B006.ALL_FEATURES.index(gap)] = -0.08
        gap_values = np.zeros((1, 4), dtype=np.float32); gap_values[0, 2] = 0.08
        corrected, names = audit.matrix_for(matrix, [gap], gap_values)
        original, _ = audit.matrix_for(matrix, [gap], None)
        self.assertAlmostEqual(0.08, float(corrected[0, names.index(gap)]), places=6)
        self.assertAlmostEqual(-0.08, float(original[0, names.index(gap)]), places=6)

    def test_missing_rank_and_rrf_examples(self):
        self.assertGreater(1 / 61, 1 / 80)
        self.assertEqual(0.0, audit.B006.reciprocal(None))
        self.assertEqual(31.0, audit.B006.encode_rank(None, 31))

    def test_exact_positive_pattern(self):
        row = {"found_by_existing": True, "found_by_name_char": False, "found_by_address_word": True, "found_by_name_word": False}
        self.assertEqual("R0+R2", audit.exact_pattern(row))

    def test_source_specific_gap_does_not_mix_s2_and_s3(self):
        matrix = np.zeros((3, len(audit.B006.ALL_FEATURES)), dtype=np.float32)
        flag = audit.B006.ALL_FEATURES.index("found_R2")
        score = audit.B006.ALL_FEATURES.index("R2_address_word_score")
        matrix[:, flag] = 1
        matrix[:, score] = [0.8, 0.9, 0.7]
        gaps = audit.source_specific_gaps(
            matrix, ["S1-1", "S1-1", "S1-1"], ["S2-a", "S3-a", "S3-b"], ["US", "US", "US"]
        )
        self.assertAlmostEqual(0.0, float(gaps[0, 2]), places=6)
        self.assertAlmostEqual(0.0, float(gaps[1, 2]), places=6)
        self.assertAlmostEqual(0.2, float(gaps[2, 2]), places=6)


if __name__ == "__main__":
    unittest.main()
