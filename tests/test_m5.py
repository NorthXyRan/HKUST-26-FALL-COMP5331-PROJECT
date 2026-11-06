import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.m5 import feature_groups, chronological_split, permutation_scores
from src.modeling import fit_and_score
from sklearn.ensemble import RandomForestClassifier


class M5Tests(unittest.TestCase):
    def test_complete_nonoverlapping_multihorizon_groups_exclude_label(self):
        definitions = json.loads(Path("configs/m5_groups.json").read_text())
        columns = [f"H{h}_{name}" for h in (5, 1000) for names in definitions.values() for name in names]
        groups = feature_groups(columns + ["H5_is_fraud"], definitions)
        self.assertEqual(sorted(sum(groups.values(), [])), sorted(columns))
        with self.assertRaises(ValueError):
            feature_groups(columns + ["H5_unknown"], definitions)

    def test_split_uses_m4_rounding_and_keeps_order(self):
        frame = pd.DataFrame({"x": range(15)})
        train, test = chronological_split(frame, .7)
        self.assertEqual(train.x.tolist(), list(range(10)))
        self.assertEqual(test.x.tolist(), list(range(10, 15)))

    def test_shared_modeling_preserves_original_m4_training(self):
        rng = np.random.default_rng(7)
        train = pd.DataFrame({"x": rng.normal(size=160), "H5_is_fraud": [0]*130+[1]*30})
        test = pd.DataFrame({"x": rng.normal(size=80), "H5_is_fraud": [0]*60+[1]*20})
        normal, anomaly = train[train.H5_is_fraud == 0], train[train.H5_is_fraud == 1]
        sample = pd.concat([normal.sample(n=len(anomaly), random_state=12), anomaly]).sample(frac=1, random_state=42)
        model = RandomForestClassifier(random_state=42).fit(sample[["x"]], sample.H5_is_fraud)
        expected = average_precision_score(test.H5_is_fraud, model.predict_proba(test[["x"]])[:, 1])
        self.assertEqual(fit_and_score(train, test, 42)["pr_auc"], expected)

    def test_group_permutation_keeps_duplicate_columns_paired_and_input_unchanged(self):
        class Model:
            def predict_proba(self, X):
                np.testing.assert_array_equal(X.a, X.b)
                return np.column_stack([1-X.a, X.a])
        X = pd.DataFrame({"a": [0., 0., 1., 1.], "b": [0., 0., 1., 1.]})
        before = X.copy()
        rows = permutation_scores(Model(), X, [0, 0, 1, 1], {"pair": ["a", "b"]}, 3, 9)
        self.assertEqual(rows, permutation_scores(Model(), X, [0, 0, 1, 1], {"pair": ["a", "b"]}, 3, 9))
        pd.testing.assert_frame_equal(X, before)


if __name__ == "__main__":
    unittest.main()
