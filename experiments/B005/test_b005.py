from __future__ import annotations

import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_b005 as b005  # noqa: E402


def row(candidate_id: str, **kwargs):
    value = {
        "candidate_id": candidate_id, "rrf_score": 0.01,
        "retriever_count": 1, "found_by_existing": False,
        "found_by_name_char": False, "found_by_address_word": False,
        "found_by_name_word": False, "existing_rank": None,
        "name_char_rank": None, "address_word_rank": None, "name_word_rank": None,
    }
    value.update(kwargs)
    return value


class B005Tests(unittest.TestCase):
    def test_split_is_disjoint_stratified_and_deterministic(self):
        ids = [f"S1-{i}" for i in range(40)]
        truth = {value: (set() if i % 2 else {f"S2-{i}"}) for i, value in enumerate(ids)}
        countries = {value: ("IN" if i % 3 else "US") for i, value in enumerate(ids)}
        first = b005.deterministic_calibration_split(ids, truth, countries, 0.1, 42)
        second = b005.deterministic_calibration_split(ids, truth, countries, 0.1, 42)
        self.assertEqual(first, second)
        self.assertFalse(set(first[0]) & set(first[1]))
        self.assertEqual(set(ids), set(first[0]) | set(first[1]))

    def test_hard_negative_budget_and_categories(self):
        negatives = [
            row("S2-r0", found_by_existing=True, existing_rank=1),
            row("S2-r1", found_by_name_char=True, name_char_rank=1),
            row("S2-r2", found_by_address_word=True, address_word_rank=1),
            row("S2-r3", found_by_name_word=True, name_word_rank=1),
            row("S2-multi", retriever_count=2, found_by_existing=True, found_by_address_word=True, existing_rank=2, address_word_rank=2),
        ]
        selected = b005.select_hard_negatives(negatives, 5)
        self.assertEqual(5, len(selected))
        ids = {value["candidate_id"] for value in selected}
        self.assertIn("S2-multi", ids)
        self.assertTrue({"S2-r0", "S2-r1", "S2-r2", "S2-r3"} <= ids)
        self.assertEqual(selected, b005.select_hard_negatives(negatives, 5))

    def test_missing_positive_is_not_force_added(self):
        candidates = [
            row("S2-positive", found_by_existing=True, existing_rank=1),
            row("S2-wrong", found_by_existing=True, existing_rank=2),
        ]
        positives, negatives, missed = b005.partition_candidates(
            candidates, {"S2-positive", "S2-missing-positive"}
        )
        self.assertEqual(["S2-positive"], [value["candidate_id"] for value in positives])
        self.assertEqual({"S2-missing-positive"}, missed)
        selected = b005.select_hard_negatives(negatives, 50)
        self.assertNotIn("S2-missing-positive", {value["candidate_id"] for value in selected})

    def test_threshold_tuner_applies_target_uniqueness(self):
        truth = {"S1-a": {"S2-x"}, "S1-b": set()}
        scores = {"S1-a": [("S2-x", 0.8)], "S1-b": [("S2-x", 0.9)]}
        calls = []
        def apply(values, t2, t3):
            calls.append((t2, t3))
            pairs = sorted(
                [(score, query_id, target_id) for query_id, candidates in values.items() for target_id, score in candidates if score >= t2],
                reverse=True,
            )
            result = {query_id: set() for query_id in values}; assigned = set()
            for _score, query_id, target_id in pairs:
                if target_id not in assigned:
                    assigned.add(target_id); result[query_id].add(target_id)
            return result
        def evaluate(expected, predicted):
            values = []
            for query_id, true_values in expected.items():
                actual = predicted.get(query_id, set())
                if not true_values:
                    values.append(float(not actual)); continue
                tp = len(true_values & actual)
                if not tp:
                    values.append(0.0); continue
                precision = tp / len(actual); recall = tp / len(true_values)
                values.append(1.25 * precision * recall / (0.25 * precision + recall))
            return {"macro_f05": float(np.mean(values))}
        t2, t3, metrics = b005.tune_thresholds(truth, scores, apply, evaluate)
        self.assertTrue(calls)
        self.assertGreaterEqual(t2, 0.0); self.assertGreaterEqual(t3, 0.0)
        self.assertIn("threshold_trials", metrics)

    def test_tiny_model_serialization(self):
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(max_iter=5, random_state=42)
        X = np.vstack([np.zeros((10, 32)), np.ones((10, 32))]).astype(np.float32)
        y = np.array([0] * 10 + [1] * 10)
        model.fit(X, y)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pkl"
            with path.open("wb") as handle:
                pickle.dump(model, handle)
            with path.open("rb") as handle:
                restored = pickle.load(handle)
            self.assertEqual((20, 2), restored.predict_proba(X).shape)


if __name__ == "__main__":
    unittest.main()
