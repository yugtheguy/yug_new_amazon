from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("gpu2_decision_rules", HERE / "decision_rules.py")
rules = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = rules
spec.loader.exec_module(rules)

runner_spec = importlib.util.spec_from_file_location("gpu2_runner", HERE / "run_gpu2_challenger.py")
runner = importlib.util.module_from_spec(runner_spec)
assert runner_spec and runner_spec.loader
sys.modules[runner_spec.name] = runner
runner_spec.loader.exec_module(runner)


class CompetitionFeatureTests(unittest.TestCase):
    def test_query_and_target_competition_values(self):
        scores = {
            "Q1": [("S2-A", 0.91), ("S2-B", 0.81), ("S2-C", 0.60), ("S3-Z", 0.72)],
            "Q2": [("S2-A", 0.88)],
        }
        query, target = rules.competition_features(scores)
        self.assertAlmostEqual(0.91, query[("Q1", "S2")]["p1"])
        self.assertAlmostEqual(0.10, query[("Q1", "S2")]["p1_minus_p2"])
        self.assertEqual(2.0, query[("Q1", "S2")]["above_0.80"])
        self.assertAlmostEqual(0.03, target["S2-A"]["ownership_margin"])
        self.assertEqual(2.0, target["S2-A"]["number_strong_s1_claimants"])

    def test_zero_one_or_many_and_target_uniqueness(self):
        scores = {
            "Q0": [("S2-X", 0.20)],
            "Q1": [("S2-A", 0.91), ("S3-B", 0.92)],
            "Q2": [("S2-A", 0.90)],
        }
        predicted = rules.apply_decision_rule(scores, 0.80, 0.80, {"name": "baseline"})
        self.assertEqual(set(), predicted["Q0"])
        self.assertEqual({"S2-A", "S3-B"}, predicted["Q1"])
        self.assertEqual(set(), predicted["Q2"])

    def test_ownership_margin_can_reject_without_forcing_top1(self):
        scores = {"Q1": [("S2-A", 0.91), ("S2-B", 0.90)], "Q2": [("S2-A", 0.90)]}
        rule = {"ownership_margin": 0.02, "confidence_override": 0.99}
        predicted = rules.apply_decision_rule(scores, 0.80, 0.80, rule)
        self.assertNotIn("S2-A", predicted["Q1"])
        self.assertIn("S2-B", predicted["Q1"])

    def test_calibration_tie_prefers_simpler_earlier_rule(self):
        scores = {"Q1": [("S2-A", 0.9)]}
        truth = {"Q1": {"S2-A"}}

        def evaluate(expected, actual):
            return {"macro_f05": float(actual == expected)}

        family = [{"name": "baseline"}, {"name": "also_equal", "ambiguity_bump": 0.01}]
        winner, trials = rules.select_rule_on_calibration(truth, scores, 0.8, 0.8, family, evaluate)
        self.assertEqual("baseline", winner["name"])
        self.assertEqual(2, len(trials))


class RunnerContractTests(unittest.TestCase):
    def test_safe_schema_is_exactly_32_plus_8(self):
        self.assertEqual(40, len(runner.SAFE_FEATURES))
        self.assertEqual(
            [
                "found_R0", "found_R1", "found_R2", "found_R3", "retriever_count",
                "R1_name_char_score", "R2_address_word_score", "R3_name_word_score",
            ],
            runner.SAFE_RETRIEVAL_FEATURES,
        )

    def test_c3_is_capped_at_six_interpretable_configurations(self):
        configurations = runner.c3_configurations({
            "num_leaves": 31, "min_child_samples": 20, "max_depth": 6, "learning_rate": 0.05,
        })
        self.assertEqual(6, len(configurations))
        self.assertEqual("current", configurations[0]["name"])
        self.assertEqual({"num_leaves": 63}, configurations[1]["changes"])
        self.assertEqual({"min_child_samples": 80}, configurations[4]["changes"])

    def test_frozen_evaluation_gate_rejects_a_second_call(self):
        gate = runner.FrozenEvaluationGate("test", lambda truth, predictions: {"macro_f05": 1.0})
        self.assertEqual(1.0, gate.evaluate({}, {})["macro_f05"])
        with self.assertRaisesRegex(RuntimeError, "more than once"):
            gate.evaluate({}, {})


if __name__ == "__main__":
    unittest.main()
