"""Audit saved M5 runs against source inputs and numerical result contracts."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .m5 import METRICS, SETTINGS, feature_groups


def verify(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    for path, expected in (manifest["input_sha256"] | manifest["source_sha256"]).items():
        actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"hash mismatch: {path}")
    a = pd.read_csv(directory / "ablations.csv")
    p = pd.read_csv(directory / "permutation.csv")
    inventory = pd.read_csv(directory / "inventory.csv")
    checks = pd.read_csv(directory / "baseline_check.csv")
    contexts = len(manifest["feature_sets"]) * len(SETTINGS)
    assert len(a) == contexts * (len(manifest["groups"]) + 1)
    assert len(p) == contexts * len(manifest["groups"]) * manifest["permutation_repeats"]
    assert len(checks) == contexts * len(METRICS)
    assert checks.matches.all()
    np.testing.assert_allclose(checks.observed, checks.expected, rtol=0, atol=1e-10)
    assert not inventory[["missing", "nonfinite"]].to_numpy().any()
    for fs in manifest["feature_sets"]:
        for condition in ("random", "activity_matched"):
            rows = inventory[(inventory.feature_set == fs) & (inventory.condition == condition)]
            assert not rows.feature.duplicated().any()
            groups = feature_groups(rows.feature, manifest["groups"])
            assert len(rows) == (34 if fs == "H1000" else 204)
            assert set(rows.group) == set(groups)
        for setting, _, _ in SETTINGS:
            rows = a[(a.feature_set == fs) & (a.setting == setting)]
            assert set(rows.omitted_group) == {"none", *manifest["groups"]}
            full = rows[rows.omitted_group == "none"].iloc[0]
            for metric in METRICS[:3]:
                np.testing.assert_allclose(rows[f"{metric}_drop"], full[metric] - rows[metric], atol=1e-12)
            for field in ("train_rows_before_balance", "train_anomalies", "test_rows", "test_anomalies"):
                assert rows[field].nunique() == 1
            for group, columns in groups.items():
                assert rows[rows.omitted_group == group].feature_count.iloc[0] == full.feature_count - len(columns)
    np.testing.assert_allclose(p.pr_auc_drop, p.baseline_pr_auc - p.permuted_pr_auc, atol=1e-12)
    return {"run": directory.name, "model_variants": len(a), "baseline_checks": len(checks),
            "permutation_measurements": len(p), "inventoried_columns": len(inventory), "status": "passed"}


if __name__ == "__main__":
    runs = sorted(Path("results/m5").glob("*p_seed_42"))
    if not runs:
        raise SystemExit("No saved M5 runs found")
    print(json.dumps([verify(d) for d in runs], indent=2))
