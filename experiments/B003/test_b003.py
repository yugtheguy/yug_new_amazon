"""Cheap synthetic checks for B003. No competition data or model training required."""

from __future__ import annotations

import json
import argparse
import sys
import tempfile
import types
import unittest
from pathlib import Path

import run_b003


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "code" / "business_entity_resolution" / "src"
sys.path.insert(0, str(SRC))

# The local bundled runtime does not include text-unidecode. ASCII synthetic inputs
# need only an identity implementation, while Kaggle uses the declared real package.
if "text_unidecode" not in sys.modules:
    sys.modules["text_unidecode"] = types.SimpleNamespace(unidecode=lambda value: value)

import blocking  # noqa: E402
from evaluation import compute_entity_f05, evaluate_predictions  # noqa: E402
from thresholding import apply_threshold_and_deduplication  # noqa: E402


class B003Tests(unittest.TestCase):
    def test_candidate_generation_matches_production_helper(self) -> None:
        targets = {
            "S2-1": ("Alpha Foods", "10 Main St", "US"),
            "S2-2": ("Alpha Tools", "11 Main St", "US"),
            "S3-1": ("Beta Foods", "10 Oak Rd", "US"),
        }
        index = blocking.build_inverted_index_for_country(targets)
        _, b003_index, pruned = run_b003.build_production_index(
            targets,
            blocking.norm.normalize_name,
            blocking.norm.normalize_address,
            blocking.get_blocking_keys,
            300,
            150,
        )
        self.assertEqual(index, b003_index)
        self.assertEqual(pruned, 0)
        expected = blocking.retrieve_candidates_for_s1(
            "Alpha Foods", "10 Main St", "US", index, top_k=50
        )
        actual = run_b003.retrieve_production_candidates(
            "Alpha Foods", "10 Main St", "US", b003_index,
            blocking.get_blocking_keys, 50,
        )
        self.assertEqual(expected, actual)

    def test_candidate_restricted_oracle(self) -> None:
        truth = {"S1-1": {"S2-1", "S3-1"}, "S1-2": set()}
        candidates = {"S1-1": ["S2-1", "S2-X"], "S1-2": ["S3-X"]}
        oracle = {sid: truth[sid] & set(candidates[sid]) for sid in truth}
        metrics = evaluate_predictions(truth, oracle)
        self.assertEqual(oracle, {"S1-1": {"S2-1"}, "S1-2": set()})
        self.assertAlmostEqual(metrics["macro_f05"], (5 / 6 + 1) / 2)

    def test_official_f05_singleton_and_partial_match(self) -> None:
        self.assertEqual(compute_entity_f05(set(), set()), 1.0)
        self.assertEqual(compute_entity_f05(set(), {"S2-1"}), 0.0)
        self.assertAlmostEqual(compute_entity_f05({"a", "b"}, {"a"}), 5 / 6)

    def test_production_module_import_contract(self) -> None:
        if "rapidfuzz" not in sys.modules:
            rapidfuzz = types.ModuleType("rapidfuzz")
            rapidfuzz.fuzz = types.SimpleNamespace(
                token_sort_ratio=lambda left, right: 100.0,
                token_set_ratio=lambda left, right: 100.0,
            )
            distance = types.ModuleType("rapidfuzz.distance")
            similarity = staticmethod(lambda left, right: 1.0 if left == right else 0.0)
            distance.Levenshtein = type("Levenshtein", (), {"normalized_similarity": similarity})
            distance.JaroWinkler = type("JaroWinkler", (), {"similarity": similarity})
            sys.modules["rapidfuzz"] = rapidfuzz
            sys.modules["rapidfuzz.distance"] = distance
        if "joblib" not in sys.modules:
            joblib = types.ModuleType("joblib")
            joblib.load = lambda path: None
            joblib.dump = lambda value, path: None
            sys.modules["joblib"] = joblib
        if "lightgbm" not in sys.modules:
            lightgbm = types.ModuleType("lightgbm")
            lightgbm.LGBMClassifier = type("LGBMClassifier", (), {})
            sys.modules["lightgbm"] = lightgbm
        if "sklearn" not in sys.modules:
            sklearn = types.ModuleType("sklearn")
            ensemble = types.ModuleType("sklearn.ensemble")
            ensemble.HistGradientBoostingClassifier = type("HistGradientBoostingClassifier", (), {})
            sys.modules["sklearn"] = sklearn
            sys.modules["sklearn.ensemble"] = ensemble
        production = run_b003.import_production(REPO_ROOT)
        self.assertEqual(production["norm"].normalize_name("Alpha LLC")[1], "alpha")
        self.assertEqual(len(run_b003.FEATURE_NAMES), 32)
        self.assertIn("apply_threshold_and_deduplication", production)

    def test_target_uniqueness(self) -> None:
        scores = {
            "S1-1": [("S2-1", 0.90), ("S3-1", 0.81)],
            "S1-2": [("S2-1", 0.95)],
        }
        result = apply_threshold_and_deduplication(scores, 0.85, 0.80)
        self.assertEqual(result["S1-1"], {"S3-1"})
        self.assertEqual(result["S1-2"], {"S2-1"})

    def test_manifest_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ids_path = root / "ids.txt"
            ids_path.write_text("S1-1\nS1-2\n", encoding="utf-8")
            ids, total = run_b003.load_selected_ids(ids_path, 2)
            manifest_path = root / "manifest.json"
            payload = {
                "experiment_id": "B003",
                "selected_validation_count": total,
                "selected_validation_ids_sha256": run_b003.sha256_ids(ids),
            }
            run_b003.atomic_json(manifest_path, payload)
            self.assertEqual(json.loads(manifest_path.read_text(encoding="utf-8")), payload)
            self.assertEqual(len(payload["selected_validation_ids_sha256"]), 64)

            dummy = root / "dummy.tsv"
            dummy.write_text("header\n", encoding="utf-8")
            args = argparse.Namespace(
                source1=dummy, source2=dummy, source3=dummy, ground_truth=dummy,
                val_ids=ids_path, model_a=dummy, model_b=dummy,
                output_dir=root / "outputs", hash_inputs=False, top_k=50,
                name_prune=300, addr_prune=150,
            )
            manifest = run_b003.base_manifest(args, REPO_ROOT, ids, total)
            self.assertEqual(manifest["experiment_id"], "B003")
            self.assertEqual(manifest["models"]["A"]["sha256"], run_b003.sha256_file(dummy))
            self.assertIn("experiments/B003/run_b003.py", manifest["semantic_code_sha256"])


if __name__ == "__main__":
    unittest.main()
