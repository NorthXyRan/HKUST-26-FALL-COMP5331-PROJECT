"""Verify Final M5 inputs, outputs, counts, and numerical relationships."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .final_m5 import EFFECT_METRICS, FINAL_METRICS, SETTINGS, ROOT, display_path


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_recorded(path):
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def verify(output: Path):
    manifest = json.loads((output / "manifest.json").read_text())
    for name, expected in manifest["input_sha256"].items():
        actual = file_digest(resolve_recorded(name))
        if actual != expected:
            raise AssertionError(f"input hash mismatch: {name}")
    for name, expected in manifest["source_sha256"].items():
        actual = file_digest(resolve_recorded(name))
        if actual != expected:
            raise AssertionError(f"source hash mismatch: {name}")
    actual_output_files = {
        str(path.relative_to(output))
        for path in output.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if actual_output_files != set(manifest["output_sha256"]):
        raise AssertionError("output file set differs from manifest")
    for name, expected in manifest["output_sha256"].items():
        if file_digest(output / name) != expected:
            raise AssertionError(f"output hash mismatch: {name}")

    tables = {
        name: pd.read_csv(output / f"{name}.csv")
        for name in manifest["counts"]
    }
    for name, expected in manifest["counts"].items():
        if len(tables[name]) != expected:
            raise AssertionError(f"row count mismatch: {name}")

    rates = len(manifest["rates"])
    seeds = len(manifest["seeds"])
    feature_sets = len(manifest["feature_sets"])
    settings = len(SETTINGS)
    variants = len(manifest["groups"]) + 1
    contexts = rates * seeds * feature_sets * settings
    ablations = tables["ablations"]
    if len(ablations) != contexts * variants:
        raise AssertionError("incomplete ablation matrix")
    if ablations[["rate", "seed", "feature_set", "setting", "omitted_group"]].duplicated().any():
        raise AssertionError("duplicate ablation context")
    expected_variants = {"none", *manifest["groups"]}
    grouped_variants = ablations.groupby(
        ["rate", "seed", "feature_set", "setting"]
    ).omitted_group.apply(set)
    if not grouped_variants.apply(lambda value: value == expected_variants).all():
        raise AssertionError("missing ablation variant")
    if (ablations.actual_validation_fpr > .01 + 1e-15).any():
        raise AssertionError("validation FPR exceeds budget")
    for metric in EFFECT_METRICS:
        full = ablations.loc[ablations.omitted_group == "none"].set_index(
            ["rate", "seed", "feature_set", "setting"]
        )[metric]
        expected = np.array([
            full.loc[(row.rate, row.seed, row.feature_set, row.setting)] - getattr(row, metric)
            for row in ablations.itertuples()
        ])
        np.testing.assert_allclose(ablations[f"{metric}_drop"], expected, rtol=0, atol=1e-12)

    checks = tables["baseline_check"]
    if len(checks) != contexts * len(FINAL_METRICS):
        raise AssertionError("incomplete M4 metric-drift checks")
    if not checks.matches.astype(bool).all():
        raise AssertionError("full-model metrics exceed drift guardrails")
    if (checks.difference.abs() > checks.tolerance).any():
        raise AssertionError("full-model metrics exceed declared drift guardrails")
    for metric, tolerance in manifest["metric_drift_guardrails"].items():
        recorded = checks.loc[checks.metric == metric, "tolerance"]
        np.testing.assert_allclose(recorded, tolerance, rtol=0, atol=0)

    prediction_checks = tables["baseline_prediction_check"]
    if len(prediction_checks) != contexts:
        raise AssertionError("incomplete frozen-prediction drift checks")
    if not prediction_checks.row_identity_match.astype(bool).all():
        raise AssertionError("frozen and current test rows differ")
    if not prediction_checks.matches.astype(bool).all():
        raise AssertionError("full-model predictions exceed drift guardrails")
    prediction_guardrails = manifest["prediction_drift_guardrails"]
    if (prediction_checks.score_correlation
            < prediction_guardrails["score_correlation_min"]).any():
        raise AssertionError("prediction score correlation is too low")
    if (prediction_checks.mean_absolute_score_difference
            > prediction_guardrails["mean_absolute_score_difference_max"]).any():
        raise AssertionError("mean absolute prediction-score drift is too high")
    if (prediction_checks.prediction_agreement
            < prediction_guardrails["prediction_agreement_min"]).any():
        raise AssertionError("prediction agreement is too low")

    permutation = tables["permutation"]
    expected_permutation = contexts * len(manifest["groups"]) * manifest["permutation_repeats"]
    if len(permutation) != expected_permutation:
        raise AssertionError("incomplete permutation matrix")
    np.testing.assert_allclose(
        permutation.pr_auc_drop,
        permutation.baseline_pr_auc - permutation.permuted_pr_auc,
        rtol=0,
        atol=1e-12,
    )

    inventory = tables["inventory"]
    if inventory[["missing", "nonfinite"]].to_numpy().any():
        raise AssertionError("invalid predictor values in inventory")
    expected_inventory = rates * seeds * 2 * sum(
        34 if feature_set == "H1000" else 204
        for feature_set in manifest["feature_sets"]
    )
    if len(inventory) != expected_inventory:
        raise AssertionError("incomplete feature inventory")

    paired = tables["paired_deltas"]
    if len(paired) != rates * seeds * feature_sets * variants:
        raise AssertionError("incomplete paired deltas")

    return {
        "status": "passed",
        "output": display_path(output),
        "ablations": len(ablations),
        "baseline_checks": len(checks),
        "baseline_prediction_checks": len(prediction_checks),
        "permutation_measurements": len(permutation),
        "inventory_rows": len(inventory),
        "mdi_rows": len(tables["mdi"]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=ROOT / "results/final_m5")
    args = parser.parse_args(argv)
    print(json.dumps(verify(args.output), indent=2))


if __name__ == "__main__":
    main()
