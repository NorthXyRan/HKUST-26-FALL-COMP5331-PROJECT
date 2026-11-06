"""Final M5 retrained ablations and feature importance on frozen M2/M4 inputs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

from .final_pipeline import LABEL, fit_score

ROOT = Path(__file__).resolve().parents[1]
META = {"t", "u", "v", "split", "event_id", LABEL}
SETTINGS = (
    ("random_to_random", "random", "random"),
    ("random_to_activity_matched", "random", "activity_matched"),
    ("activity_matched_to_activity_matched", "activity_matched", "activity_matched"),
)
FEATURE_SETS = ("H1000", "combined_H")
FINAL_METRICS = (
    "pr_auc",
    "roc_auc",
    "recall_at_validation_1pct_fpr",
    "actual_test_fpr",
    "actual_validation_fpr",
    "threshold",
)
EFFECT_METRICS = FINAL_METRICS[:4]
METRIC_DRIFT_GUARDRAILS = {
    "pr_auc": 0.02,
    "roc_auc": 0.01,
    "recall_at_validation_1pct_fpr": 0.15,
    "actual_test_fpr": 0.01,
    "actual_validation_fpr": 0.01,
    "threshold": 0.02,
}
PREDICTION_DRIFT_GUARDRAILS = {
    "score_correlation_min": 0.995,
    "mean_absolute_score_difference_max": 0.01,
    "prediction_agreement_min": 0.98,
}
RUN_PATTERN = re.compile(r"(?P<rate>\d+)p_seed_(?P<seed>\d+)")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def discover_runs(input_root: Path, rates=None, seeds=None):
    protocol_path = input_root / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"missing final protocol: {protocol_path}")
    protocol = json.loads(protocol_path.read_text())
    available_rates = tuple(int(x) for x in protocol["rates_percent_of_original"])
    available_seeds = tuple(int(x) for x in protocol["seeds"])
    chosen_rates = available_rates if rates is None else tuple(int(x) for x in rates)
    chosen_seeds = available_seeds if seeds is None else tuple(int(x) for x in seeds)
    if not set(chosen_rates) <= set(available_rates):
        raise ValueError("requested rate is not in the frozen protocol")
    if not set(chosen_seeds) <= set(available_seeds):
        raise ValueError("requested seed is not in the frozen protocol")
    runs = []
    for rate in chosen_rates:
        for seed in chosen_seeds:
            directory = input_root / f"{rate}p_seed_{seed}"
            if not directory.is_dir():
                raise FileNotFoundError(f"missing frozen run: {directory}")
            runs.append((rate, seed, directory))
    return protocol, runs


def feature_groups(columns, definitions):
    """Map every saved predictor to exactly one semantic family."""
    columns = list(columns)
    if len(columns) != len(set(columns)):
        raise ValueError("predictor list contains duplicate columns")
    forbidden = set(columns) & META
    if forbidden:
        raise ValueError(f"metadata/label cannot be predictors: {sorted(forbidden)}")
    groups = {name: [] for name in definitions}
    for column in columns:
        if not re.fullmatch(r"H\d+_.+", column):
            raise ValueError(f"invalid predictor name: {column}")
        suffix = column.split("_", 1)[1]
        matches = [name for name, suffixes in definitions.items() if suffix in suffixes]
        if len(matches) != 1:
            raise ValueError(f"predictor needs exactly one group: {column}")
        groups[matches[0]].append(column)
    empty = [name for name, members in groups.items() if not members]
    if empty:
        raise ValueError(f"empty feature groups: {empty}")
    return groups


def validate_frame(frame: pd.DataFrame, columns, name: str):
    required = META | set(columns)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    if set(frame["split"].unique()) != {"train", "validation", "test"}:
        raise ValueError(f"{name} must contain train/validation/test splits")
    if not set(frame[LABEL].unique()) <= {0, 1}:
        raise ValueError(f"{name} has non-binary labels")
    if frame[list(columns)].isna().any().any():
        raise ValueError(f"{name} has missing predictor values")
    values = frame[list(columns)].to_numpy()
    if not np.isfinite(values).all():
        raise ValueError(f"{name} has non-finite predictor values")


def compare_predictions(current: pd.DataFrame, frozen: pd.DataFrame):
    """Audit cross-platform RF drift after requiring identical test rows/labels."""
    required = ("event_id", "t", "label", "score", "prediction")
    for name, frame in (("current", current), ("frozen", frozen)):
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} predictions are missing columns: {sorted(missing)}")
    current = current.loc[:, required].reset_index(drop=True)
    frozen = frozen.loc[:, required].reset_index(drop=True)
    if len(current) != len(frozen):
        raise AssertionError("current and frozen predictions have different row counts")
    for column in ("event_id", "t", "label"):
        if not np.array_equal(current[column].to_numpy(), frozen[column].to_numpy()):
            raise AssertionError(f"current and frozen predictions differ in {column}")

    current_score = current.score.to_numpy(dtype=float)
    frozen_score = frozen.score.to_numpy(dtype=float)
    if not np.isfinite(current_score).all() or not np.isfinite(frozen_score).all():
        raise ValueError("prediction scores must be finite")
    if np.std(current_score) == 0 or np.std(frozen_score) == 0:
        correlation = 1.0 if np.array_equal(current_score, frozen_score) else 0.0
    else:
        correlation = float(np.corrcoef(current_score, frozen_score)[0, 1])
    difference = np.abs(current_score - frozen_score)
    agreement = float(np.mean(
        current.prediction.to_numpy() == frozen.prediction.to_numpy()
    ))
    result = {
        "rows": len(current),
        "row_identity_match": True,
        "score_correlation": correlation,
        "mean_absolute_score_difference": float(difference.mean()),
        "max_absolute_score_difference": float(difference.max()),
        "score_exact_fraction": float(np.mean(difference < 1e-12)),
        "prediction_agreement": agreement,
        "frozen_positive_predictions": int(frozen.prediction.sum()),
        "observed_positive_predictions": int(current.prediction.sum()),
    }
    result["matches"] = bool(
        correlation >= PREDICTION_DRIFT_GUARDRAILS["score_correlation_min"]
        and result["mean_absolute_score_difference"]
        <= PREDICTION_DRIFT_GUARDRAILS["mean_absolute_score_difference_max"]
        and agreement >= PREDICTION_DRIFT_GUARDRAILS["prediction_agreement_min"]
    )
    return result


def fit_full_model(train: pd.DataFrame, columns, seed: int):
    """Mirror final_pipeline.fit_score and expose the fitted model for importance."""
    normal = train.loc[train[LABEL] == 0]
    anomaly = train.loc[train[LABEL] == 1]
    if anomaly.empty or len(normal) < len(anomaly):
        raise ValueError("invalid training class counts")
    balanced = pd.concat([
        normal.sample(n=len(anomaly), random_state=12),
        anomaly,
    ]).sample(frac=1, random_state=42)
    model = RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(balanced[list(columns)], balanced[LABEL])
    return model


def grouped_permutation_scores(model, test, columns, groups, repeats, seed):
    """Jointly permute every column in a family with the same test-row order."""
    X = test[list(columns)].copy()
    y = test[LABEL].to_numpy()
    baseline = average_precision_score(y, model.predict_proba(X)[:, 1])
    rng = np.random.default_rng(seed)
    rows = []
    for repeat in range(repeats):
        order = rng.permutation(len(X))
        for group, members in groups.items():
            shuffled = X.copy()
            shuffled[members] = X.iloc[order][members].to_numpy()
            score = average_precision_score(y, model.predict_proba(shuffled)[:, 1])
            rows.append({
                "group": group,
                "repeat": repeat,
                "baseline_pr_auc": baseline,
                "permuted_pr_auc": score,
                "pr_auc_drop": baseline - score,
            })
    return rows


def add_metric_drops(ablations: pd.DataFrame):
    keys = ["rate", "seed", "feature_set", "setting"]
    full = ablations.loc[ablations.omitted_group == "none", keys + list(EFFECT_METRICS)]
    full = full.rename(columns={metric: f"full_{metric}" for metric in EFFECT_METRICS})
    result = ablations.merge(full, on=keys, validate="many_to_one")
    for metric in EFFECT_METRICS:
        result[f"{metric}_drop"] = result[f"full_{metric}"] - result[metric]
        result = result.drop(columns=f"full_{metric}")
    return result


def aggregate(frame: pd.DataFrame, keys, metrics):
    grouped = frame.groupby(list(keys), sort=True)[list(metrics)].agg(["mean", "std", "min", "max"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    return grouped.reset_index()


def paired_deltas(ablations: pd.DataFrame):
    index = ["rate", "seed", "feature_set", "omitted_group"]
    pivot = ablations.pivot(index=index, columns="setting", values=list(EFFECT_METRICS))
    rows = pivot.index.to_frame(index=False)
    for metric in EFFECT_METRICS:
        rows[f"{metric}_cross_minus_random"] = (
            pivot[(metric, "random_to_activity_matched")]
            - pivot[(metric, "random_to_random")]
        ).to_numpy()
        rows[f"{metric}_retraining_recovery"] = (
            pivot[(metric, "activity_matched_to_activity_matched")]
            - pivot[(metric, "random_to_activity_matched")]
        ).to_numpy()
    full = rows.loc[rows.omitted_group == "none"].set_index(["rate", "seed", "feature_set"])
    for metric in EFFECT_METRICS:
        for suffix in ("cross_minus_random", "retraining_recovery"):
            source = f"{metric}_{suffix}"
            contribution = f"{metric}_full_minus_variant_{suffix}"
            rows[contribution] = [
                full.loc[(r.rate, r.seed, r.feature_set), source] - getattr(r, source)
                for r in rows.itertuples()
            ]
    return rows


def table_hashes(output: Path):
    return {
        str(path.relative_to(output)): digest(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def git_revision():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def run_experiment(args):
    input_root = args.input
    output = args.output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    protocol, runs = discover_runs(input_root, args.rates, args.seeds)
    definitions = json.loads(args.groups.read_text())
    output.mkdir(parents=True)

    inventory_rows = []
    ablation_rows = []
    baseline_rows = []
    baseline_prediction_rows = []
    permutation_rows = []
    mdi_rows = []
    input_hashes = {
        display_path(input_root / "protocol.json"): digest(input_root / "protocol.json"),
        display_path(input_root / "manifest.json"): digest(input_root / "manifest.json"),
    }

    for rate, seed, run_dir in runs:
        frames = {}
        for condition in ("random", "activity_matched"):
            path = run_dir / f"{condition}_features.csv.gz"
            input_hashes[display_path(path)] = digest(path)
            frames[condition] = pd.read_csv(path)
        reference_path = run_dir / "metrics.csv"
        input_hashes[display_path(reference_path)] = digest(reference_path)
        reference = pd.read_csv(reference_path).set_index(["feature_set", "setting"])

        for feature_set in args.feature_sets:
            columns_path = run_dir / f"{feature_set}_columns.json"
            input_hashes[display_path(columns_path)] = digest(columns_path)
            columns = json.loads(columns_path.read_text())
            groups = feature_groups(columns, definitions)
            for condition, frame in frames.items():
                validate_frame(frame, columns, f"{run_dir.name}/{condition}/{feature_set}")
                for group, members in groups.items():
                    for column in members:
                        series = frame[column]
                        inventory_rows.append({
                            "rate": rate,
                            "seed": seed,
                            "feature_set": feature_set,
                            "condition": condition,
                            "feature": column,
                            "group": group,
                            "rows": len(frame),
                            "positive_rows": int(frame[LABEL].sum()),
                            "missing": int(series.isna().sum()),
                            "nonfinite": int((~np.isfinite(series.to_numpy())).sum()),
                            "unique_values": int(series.nunique()),
                            "constant": bool(series.nunique() == 1),
                        })

            for setting, train_condition, test_condition in SETTINGS:
                train_source = frames[train_condition]
                test_source = frames[test_condition]
                train = train_source.loc[train_source.split == "train"]
                validation = train_source.loc[train_source.split == "validation"]
                test = test_source.loc[test_source.split == "test"]
                context = {
                    "rate": rate,
                    "seed": seed,
                    "feature_set": feature_set,
                    "setting": setting,
                }
                for omitted in ("none", *groups):
                    removed = set() if omitted == "none" else set(groups[omitted])
                    keep = [column for column in columns if column not in removed]
                    if not keep:
                        raise ValueError(f"omitting {omitted} leaves no predictors")
                    metrics, predictions = fit_score(train, validation, test, keep, seed)
                    ablation_rows.append({**context, "omitted_group": omitted, **metrics})
                    if omitted == "none":
                        expected_row = reference.loc[(feature_set, setting)]
                        for metric in FINAL_METRICS:
                            expected = float(expected_row[metric])
                            observed = float(metrics[metric])
                            tolerance = METRIC_DRIFT_GUARDRAILS[metric]
                            baseline_rows.append({
                                **context,
                                "metric": metric,
                                "expected": expected,
                                "observed": observed,
                                "difference": observed - expected,
                                "tolerance": tolerance,
                                "exact_match": abs(observed - expected) < 1e-12,
                                "matches": abs(observed - expected) <= tolerance,
                            })
                        prediction_path = run_dir / f"{feature_set}_{setting}_predictions.csv.gz"
                        input_hashes[display_path(prediction_path)] = digest(prediction_path)
                        frozen_predictions = pd.read_csv(
                            prediction_path, float_precision="round_trip"
                        )
                        baseline_prediction_rows.append({
                            **context,
                            **compare_predictions(predictions, frozen_predictions),
                        })

                model = fit_full_model(train, columns, seed)
                permutation_rows.extend(
                    {**context, **row}
                    for row in grouped_permutation_scores(
                        model,
                        test,
                        columns,
                        groups,
                        args.permutation_repeats,
                        args.permutation_seed,
                    )
                )
                for column, importance in zip(model.feature_names_in_, model.feature_importances_):
                    group = next(name for name, members in groups.items() if column in members)
                    mdi_rows.append({
                        **context,
                        "feature": column,
                        "group": group,
                        "mdi": float(importance),
                    })
                print(f"completed {run_dir.name}/{feature_set}/{setting}", flush=True)

    ablations = add_metric_drops(pd.DataFrame(ablation_rows))
    effect_columns = [f"{metric}_drop" for metric in EFFECT_METRICS]
    ablation_summary = aggregate(
        ablations,
        ("rate", "feature_set", "setting", "omitted_group"),
        (*EFFECT_METRICS, *effect_columns),
    )
    paired = paired_deltas(ablations)
    paired_metrics = [column for column in paired if column not in {
        "rate", "seed", "feature_set", "omitted_group"
    }]
    paired_summary = aggregate(
        paired,
        ("rate", "feature_set", "omitted_group"),
        paired_metrics,
    )
    permutation = pd.DataFrame(permutation_rows)
    permutation_shuffle_summary = (
        permutation.groupby(
            ["rate", "seed", "feature_set", "setting", "group"], sort=True
        ).pr_auc_drop.agg(["mean", "std"]).reset_index()
        .rename(columns={"mean": "mean_pr_auc_drop", "std": "shuffle_sd"})
    )
    permutation_seed_summary = (
        permutation_shuffle_summary.groupby(
            ["rate", "feature_set", "setting", "group"], sort=True
        ).mean_pr_auc_drop.agg(["mean", "std", "min", "max"]).reset_index()
        .rename(columns={
            "mean": "pr_auc_drop_seed_mean",
            "std": "pr_auc_drop_seed_sd",
            "min": "pr_auc_drop_seed_min",
            "max": "pr_auc_drop_seed_max",
        })
    )
    tables = {
        "inventory": pd.DataFrame(inventory_rows),
        "ablations": ablations,
        "ablation_summary": ablation_summary,
        "baseline_check": pd.DataFrame(baseline_rows),
        "baseline_prediction_check": pd.DataFrame(baseline_prediction_rows),
        "paired_deltas": paired,
        "paired_deltas_summary": paired_summary,
        "permutation": permutation,
        "permutation_shuffle_summary": permutation_shuffle_summary,
        "permutation_seed_summary": permutation_seed_summary,
        "mdi": pd.DataFrame(mdi_rows),
    }
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)

    from .final_m5_report import write_report
    write_report(output, tables)

    source_paths = [
        ROOT / "src/final_m5.py",
        ROOT / "src/final_m5_report.py",
        ROOT / "src/final_m5_verify.py",
        args.groups,
    ]
    manifest = {
        "protocol_version": "final_m5_v2",
        "input_protocol_version": protocol["protocol_version"],
        "base_revision": git_revision(),
        "input_root": display_path(input_root),
        "input_sha256": input_hashes,
        "source_sha256": {display_path(path): digest(path) for path in source_paths},
        "output_sha256": table_hashes(output),
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "matplotlib", "scipy", "joblib", "threadpoolctl")
        },
        "rates": sorted({rate for rate, _, _ in runs}),
        "seeds": sorted({seed for _, seed, _ in runs}),
        "feature_sets": list(args.feature_sets),
        "settings": [name for name, _, _ in SETTINGS],
        "groups": definitions,
        "permutation_repeats": args.permutation_repeats,
        "permutation_seed": args.permutation_seed,
        "metric_drift_guardrails": METRIC_DRIFT_GUARDRAILS,
        "prediction_drift_guardrails": PREDICTION_DRIFT_GUARDRAILS,
        "counts": {name: int(len(table)) for name, table in tables.items()},
        "scope": "Frozen strict_m2_m4_v1 inputs; retrained group ablations and post-hoc importance only.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    if not tables["baseline_check"].matches.all():
        raise RuntimeError("full-model metrics exceed the cross-platform drift guardrails")
    if not tables["baseline_prediction_check"].matches.all():
        raise RuntimeError("full-model predictions exceed the cross-platform drift guardrails")
    return tables


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/final_m2_m4")
    parser.add_argument("--output", type=Path, default=ROOT / "results/final_m5")
    parser.add_argument("--groups", type=Path, default=ROOT / "configs/final_m5_groups.json")
    parser.add_argument("--rates", type=int, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--feature-sets", nargs="+", choices=FEATURE_SETS, default=list(FEATURE_SETS))
    parser.add_argument("--permutation-repeats", type=int, default=5)
    parser.add_argument("--permutation-seed", type=int, default=31415)
    args = parser.parse_args(argv)
    if args.permutation_repeats < 2:
        parser.error("permutation repeats must be at least 2")
    run_experiment(args)


if __name__ == "__main__":
    main()
