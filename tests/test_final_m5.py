import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.final_m5 import (
    META,
    compare_predictions,
    discover_runs,
    feature_groups,
    grouped_permutation_scores,
    paired_deltas,
)


ROOT = Path(__file__).resolve().parents[1]


class FinalM5ContractTests(unittest.TestCase):
    def setUp(self):
        self.definitions = json.loads((ROOT / "configs/final_m5_groups.json").read_text())

    def test_frozen_package_has_complete_run_and_predictor_matrix(self):
        input_root = ROOT / "results/final_m2_m4"
        _, runs = discover_runs(input_root)
        self.assertEqual([(rate, seed) for rate, seed, _ in runs], [
            (rate, seed) for rate in (1, 5, 10) for seed in (42, 43, 44)
        ])
        for _, _, run in runs:
            for feature_set, expected in (("H1000", 34), ("combined_H", 204)):
                columns = json.loads((run / f"{feature_set}_columns.json").read_text())
                self.assertEqual(len(columns), expected)
                self.assertFalse(set(columns) & META)
                groups = feature_groups(columns, self.definitions)
                self.assertEqual(set(sum(groups.values(), [])), set(columns))

    def test_discover_runs_rejects_missing_or_unknown_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "protocol.json").write_text(json.dumps({
                "rates_percent_of_original": [1],
                "seeds": [42],
            }))
            with self.assertRaises(FileNotFoundError):
                discover_runs(root)
            with self.assertRaises(ValueError):
                discover_runs(root, rates=[5])

    def test_groups_reject_metadata_duplicates_unknowns_and_empty_groups(self):
        columns = [f"H1000_{name}" for names in self.definitions.values() for name in names]
        groups = feature_groups(columns, self.definitions)
        self.assertEqual({name: len(values) for name, values in groups.items()}, {
            "endpoint_activity": 12,
            "pair_history": 3,
            "global_graph": 19,
        })
        for invalid in (
            columns + ["split"],
            columns + [columns[0]],
            columns + ["H1000_unknown"],
        ):
            with self.assertRaises(ValueError):
                feature_groups(invalid, self.definitions)
        with self.assertRaises(ValueError):
            feature_groups(columns, {**self.definitions, "empty": ["unused"]})

    def test_grouped_permutation_is_joint_deterministic_and_non_mutating(self):
        class Model:
            def predict_proba(self, X):
                np.testing.assert_array_equal(X.a, X.b)
                return np.column_stack([1 - X.a, X.a])

        test = pd.DataFrame({
            "a": [0., 0., 1., 1.],
            "b": [0., 0., 1., 1.],
            "H5_is_fraud": [0, 0, 1, 1],
        })
        before = test.copy()
        first = grouped_permutation_scores(Model(), test, ["a", "b"], {"pair": ["a", "b"]}, 3, 9)
        second = grouped_permutation_scores(Model(), test, ["a", "b"], {"pair": ["a", "b"]}, 3, 9)
        self.assertEqual(first, second)
        pd.testing.assert_frame_equal(test, before)

    def test_paired_deltas_use_within_seed_settings(self):
        rows = []
        values = {
            "random_to_random": 0.8,
            "random_to_activity_matched": 0.3,
            "activity_matched_to_activity_matched": 0.6,
        }
        for omitted, offset in (("none", 0.0), ("endpoint_activity", -0.1)):
            for setting, value in values.items():
                rows.append({
                    "rate": 5,
                    "seed": 42,
                    "feature_set": "H1000",
                    "setting": setting,
                    "omitted_group": omitted,
                    "pr_auc": value + offset,
                    "roc_auc": value + offset,
                    "recall_at_validation_1pct_fpr": value + offset,
                    "actual_test_fpr": value + offset,
                })
        result = paired_deltas(pd.DataFrame(rows)).set_index("omitted_group")
        self.assertAlmostEqual(result.loc["none", "pr_auc_cross_minus_random"], -0.5)
        self.assertAlmostEqual(result.loc["none", "pr_auc_retraining_recovery"], 0.3)
        self.assertAlmostEqual(
            result.loc["endpoint_activity", "pr_auc_full_minus_variant_cross_minus_random"],
            0.0,
        )

    def test_prediction_drift_requires_identical_rows_and_scores_similarity(self):
        frozen = pd.DataFrame({
            "event_id": [1, 2, 3],
            "t": [10, 20, 30],
            "label": [0, 1, 0],
            "score": [0.1, 0.9, 0.2],
            "prediction": [0, 1, 0],
        })
        current = frozen.copy()
        current["score"] += [0.001, -0.001, 0.002]
        result = compare_predictions(current, frozen)
        self.assertTrue(result["row_identity_match"])
        self.assertTrue(result["matches"])
        changed = current.copy()
        changed.loc[0, "event_id"] = 99
        with self.assertRaises(AssertionError):
            compare_predictions(changed, frozen)


if __name__ == "__main__":
    unittest.main()
