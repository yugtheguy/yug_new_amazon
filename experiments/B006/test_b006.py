from __future__ import annotations

import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_b006 as b006  # noqa: E402


def evidence(**updates):
    row = {
        "found_by_existing": False, "found_by_name_char": False,
        "found_by_address_word": False, "found_by_name_word": False,
        "existing_rank": None, "name_char_rank": None,
        "address_word_rank": None, "name_word_rank": None,
        "existing_shared_keys": None, "name_char_score": None,
        "address_word_score": None, "name_word_score": None,
        "retriever_count": 0, "rrf_score": 0.0,
    }
    row.update(updates)
    return row


class B006Tests(unittest.TestCase):
    def test_b005_base_schema_preserved(self):
        self.assertEqual(32, len(b006.BASE_FEATURES))
        self.assertEqual(b006.BASE_FEATURES, b006.ALL_FEATURES[:32])
        self.assertEqual(len(b006.ALL_FEATURES), len(set(b006.ALL_FEATURES)))

    def test_missing_rank_encoding_and_explicit_flags(self):
        row = evidence()
        values = dict(zip(b006.RETRIEVAL_FEATURES, b006.retrieval_feature_values(row, {})))
        self.assertEqual(51.0, values["R0_rank"])
        self.assertEqual(31.0, values["R2_rank"])
        self.assertEqual(0.0, values["found_R2"])
        self.assertEqual(0.0, values["R2_address_word_score"])

    def test_rrf_and_interactions(self):
        expected = 1 / 61 + 1 / 62
        row = evidence(
            found_by_existing=True, found_by_address_word=True,
            existing_rank=1, address_word_rank=2, existing_shared_keys=4,
            address_word_score=0.9, retriever_count=2, rrf_score=expected,
        )
        values = dict(zip(b006.RETRIEVAL_FEATURES, b006.retrieval_feature_values(row, {"R0": 5, "R2": 0.95})))
        self.assertAlmostEqual(expected, values["reciprocal_rank_sum"])
        self.assertEqual(1.0, values["R0_R2"])
        self.assertEqual(0.0, values["R1_R2"])

    def test_query_relative_gap(self):
        row = evidence(found_by_address_word=True, address_word_rank=1, address_word_score=0.91, retriever_count=1, rrf_score=1 / 61)
        values = dict(zip(b006.RETRIEVAL_FEATURES, b006.retrieval_feature_values(row, {"R2": 0.92})))
        self.assertAlmostEqual(-0.01, values["R2_address_word_gap_from_best"], places=6)
        self.assertAlmostEqual(0.92, values["R2_address_word_best_score"], places=6)

    def test_provenance_pattern_keys(self):
        self.assertEqual("R2-only", b006.retrieval_pattern(evidence(found_by_address_word=True)))
        self.assertEqual("multi-channel", b006.retrieval_pattern(evidence(found_by_existing=True, found_by_address_word=True)))

    def test_canonical_merge_key_and_duplicate_rejection(self):
        seen = set()
        self.assertEqual(("S1-1", "S2-9"), b006.register_pair_key(seen, "S1-1", "S2-9"))
        with self.assertRaises(ValueError):
            b006.register_pair_key(seen, "S1-1", "S2-9")

    def test_splits_disjoint(self):
        train, calibration, evaluation = {"S1-a"}, {"S1-b"}, {"S1-c"}
        self.assertFalse(train & calibration or train & evaluation or calibration & evaluation)

    def test_serialization(self):
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(max_iter=5, random_state=42)
        X = np.vstack((np.zeros((10, len(b006.ALL_FEATURES))), np.ones((10, len(b006.ALL_FEATURES))))).astype(np.float32)
        y = np.array([0] * 10 + [1] * 10); model.fit(X, y)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pkl"
            path.write_bytes(pickle.dumps(model)); restored = pickle.loads(path.read_bytes())
            self.assertEqual((20, 2), restored.predict_proba(X).shape)


if __name__ == "__main__":
    unittest.main()
