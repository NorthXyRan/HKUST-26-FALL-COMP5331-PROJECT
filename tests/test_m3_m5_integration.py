"""Behavioral and artifact checks for strict-protocol M3--M5 integration."""

import json
from pathlib import Path
import unittest

import pandas as pd

from src.final_m3_m5 import (
    FEATURE_SETS,
    GROUPS,
    RATES,
    SEEDS,
    SETTINGS,
    build_joint_evidence,
    component_contract,
    verify,
)


ROOT = Path(__file__).resolve().parents[1]
M3 = ROOT / "results/m3"
M5 = ROOT / "results/final_m5"
JOINT = ROOT / "results/final_m3_m5"


class M3M5IntegrationTests(unittest.TestCase):
    def test_components_share_the_strict_frozen_protocol(self):
        contract = component_contract(M3, M5)
        self.assertEqual(contract["protocol"], "strict_m2_m4_v1")
        self.assertEqual(contract["rates"], list(RATES))
        self.assertEqual(contract["injection_seeds"], list(SEEDS))
        self.assertIn("not independently crossed", contract["classifier_seed_policy"])
        self.assertEqual(contract["permutation_repeats"], 5)

    def test_joint_matrix_is_complete_and_matching_improves_primary_diagnostics(self):
        evidence = build_joint_evidence(M3, M5)
        self.assertEqual(len(evidence), len(RATES) * len(SEEDS) * len(FEATURE_SETS))
        self.assertFalse(evidence.isna().any().any())
        self.assertEqual(
            set(map(tuple, evidence[["rate", "seed", "feature_set"]].to_numpy())),
            {(rate, seed, feature_set) for rate in RATES for seed in SEEDS for feature_set in FEATURE_SETS},
        )
        self.assertTrue(evidence.m3_activity_error_reduction_pct.gt(0).all())
        for metric in ("degree_mean", "common_neighbors", "jaccard"):
            self.assertTrue(
                evidence[f"m3_ks_activity_matched_{metric}"].lt(
                    evidence[f"m3_ks_random_{metric}"]
                ).all()
            )

    def test_saved_effects_equal_the_original_m5_ablations(self):
        evidence = build_joint_evidence(M3, M5).set_index(["rate", "seed", "feature_set"])
        ablations = pd.read_csv(M5 / "ablations.csv")
        for setting in SETTINGS:
            for group in GROUPS:
                expected = (
                    ablations[
                        (ablations.setting == setting) & (ablations.omitted_group == group)
                    ]
                    .set_index(["rate", "seed", "feature_set"])
                    .pr_auc_drop
                    .sort_index()
                )
                actual = evidence[f"m5_ap_drop_{setting}_{group}"].sort_index()
                pd.testing.assert_series_equal(actual, expected, check_names=False)

    def test_feature_semantics_and_randomness_are_not_overstated(self):
        manifest = json.loads((M5 / "manifest.json").read_text())
        self.assertEqual(set(manifest["groups"]), set(GROUPS))
        self.assertFalse(
            any(
                token in feature
                for feature in manifest["groups"]["global_graph"]
                for token in ("neighbor", "jaccard", "common")
            )
        )
        summary = json.loads((JOINT / "joint_summary.json").read_text())
        self.assertIn("not independently crossed", summary["randomness_accounting"]["m5_classifier"])
        self.assertIn("not independent datasets", summary["randomness_accounting"]["permutation"])

    def test_checked_in_joint_package(self):
        result = verify(JOINT, check_components=False)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["joint_rows"], 18)


if __name__ == "__main__":
    unittest.main()
