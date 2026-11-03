"""Local feature audit, retrained ablation, and grouped permutation importance."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .modeling import fit_and_score, fit_model

ROOT = Path(__file__).resolve().parents[1]
LABEL = "H5_is_fraud"
SETTINGS = (("random_to_random", "random", "random"),
            ("random_to_activity_matched", "random", "activity_matched"),
            ("activity_matched_to_activity_matched", "activity_matched", "activity_matched"))
METRICS = ("pr_auc", "roc_auc", "recall_at_1pct_fpr", "actual_fpr_at_threshold")


def feature_groups(columns, definitions):
    """Use the same semantic group across every H window; reject incomplete maps."""
    groups = {name: [] for name in definitions}
    for column in columns:
        if column == LABEL:
            continue
        suffix = column.split("_", 1)[1]
        matches = [name for name, names in definitions.items() if suffix in names]
        if len(matches) != 1:
            raise ValueError(f"feature needs exactly one group: {column}")
        groups[matches[0]].append(column)
    return groups


def chronological_split(frame, fraction):
    cut = int(round(len(frame) * fraction))
    return frame.iloc[:cut].copy(), frame.iloc[cut:].copy()


def permutation_scores(model, X, y, groups, repeats, seed):
    """Shuffle a whole family together, preserving its within-row relationships."""
    baseline = average_precision_score(y, model.predict_proba(X)[:, 1])
    rng = np.random.default_rng(seed)
    rows = []
    for repeat in range(repeats):
        order = rng.permutation(len(X))
        for group, columns in groups.items():
            shuffled = X.copy()
            shuffled[columns] = X.iloc[order][columns].to_numpy()
            score = average_precision_score(y, model.predict_proba(shuffled)[:, 1])
            rows.append({"group": group, "repeat": repeat, "baseline_pr_auc": baseline,
                         "permuted_pr_auc": score, "pr_auc_drop": baseline - score})
    return rows


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.run_dir / "run_config.json").read_text())
    definitions = json.loads(args.groups.read_text())
    fraction = config["chronological_train_fraction"]
    seed = config["seeds"]["random_forest"]
    audit, ablations, importance, mdi, checks = [], [], [], [], []
    hashes = {str(args.run_dir / "run_config.json"): digest(args.run_dir / "run_config.json")}
    for feature_set in args.feature_sets:
        frames = {}
        for condition in ("random", "activity_matched"):
            path = args.run_dir / f"{condition}_features_{feature_set}.csv.gz"
            hashes[str(path)] = digest(path)
            frame = pd.read_csv(path)
            frames[condition] = frame
            groups = feature_groups(frame.columns, definitions)
            for group, columns in groups.items():
                for column in columns:
                    audit.append({"feature_set": feature_set, "condition": condition,
                                  "feature": column, "group": group, "rows": len(frame),
                                  "positive_rows": int(frame[LABEL].sum()),
                                  "missing": int(frame[column].isna().sum()),
                                  "nonfinite": int((~np.isfinite(frame[column])).sum()),
                                  "unique_values": int(frame[column].nunique()),
                                  "constant": bool(frame[column].nunique() == 1)})
        if list(frames["random"].columns) != list(frames["activity_matched"].columns):
            raise ValueError("R and M feature schemas differ")
        splits = {name: chronological_split(frame, fraction) for name, frame in frames.items()}
        reference = pd.read_csv(args.run_dir / f"metrics_{feature_set}.csv").set_index("setting")
        # Full model and family omissions share exactly the same sampled train rows.
        for setting, train_name, test_name in SETTINGS:
            train, test = splits[train_name][0], splits[test_name][1]
            for omitted in ["none", *groups]:
                columns = [] if omitted == "none" else groups[omitted]
                scores = fit_and_score(train.drop(columns=columns), test.drop(columns=columns), seed)
                row = {"feature_set": feature_set, "setting": setting, "omitted_group": omitted, **scores}
                ablations.append(row)
                if omitted == "none":
                    for metric in METRICS:
                        expected = float(reference.loc[setting, metric])
                        delta = scores[metric] - expected
                        checks.append({"feature_set": feature_set, "setting": setting,
                                       "metric": metric, "expected": expected,
                                       "observed": scores[metric], "difference": delta,
                                       "matches": abs(delta) < 1e-10})
            model = fit_model(train, seed)
            X, y = test.drop(columns=LABEL), test[LABEL]
            context = {"feature_set": feature_set, "setting": setting}
            importance.extend({**context, **r} for r in permutation_scores(
                model, X, y, groups, args.permutation_repeats, args.permutation_seed))
            for column, value in zip(model.feature_names_in_, model.feature_importances_):
                group = next(g for g, cols in groups.items() if column in cols)
                mdi.append({**context, "feature": column, "group": group, "mdi": value})
            print(f"Completed {feature_set}/{setting}", flush=True)
    result = pd.DataFrame(ablations)
    for metric in METRICS[:3]:
        full = result[result.omitted_group == "none"].set_index(["feature_set", "setting"])[metric]
        result[f"{metric}_drop"] = [full.loc[(r.feature_set, r.setting)] - getattr(r, metric)
                                            for r in result.itertuples()]
    tables = {"inventory": pd.DataFrame(audit), "ablations": result,
              "permutation": pd.DataFrame(importance), "mdi": pd.DataFrame(mdi),
              "baseline_check": pd.DataFrame(checks)}
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    source_paths = [Path("src/m5.py"), Path("src/modeling.py"), Path("src/m5_report.py"), args.groups]
    manifest = {"base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "input_sha256": hashes, "source_sha256": {str(p): digest(p) for p in source_paths},
                "python": platform.python_version(), "packages": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn", "matplotlib", "scipy", "joblib", "threadpoolctl")},
                "train_fraction": fraction, "model_seed": seed, "injection_seed": config["seeds"],
                "permutation_repeats": args.permutation_repeats, "permutation_seed": args.permutation_seed,
                "feature_sets": args.feature_sets, "groups": definitions,
                "scope": "Fixed existing single-injection-seed data; no generator changes or parameter tuning."}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    from .m5_report import write_report
    write_report(out, tables)
    if not tables["baseline_check"].matches.all():
        raise RuntimeError("Full-model metrics differ from saved M4: inspect baseline_check.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("results/full_pipeline/5p_seed_42"))
    parser.add_argument("--output", type=Path, default=Path("results/m5/5p_seed_42"))
    parser.add_argument("--feature-sets", nargs="+", choices=("H1000", "combined_H"), default=["H1000"])
    parser.add_argument("--groups", type=Path, default=Path("configs/m5_groups.json"))
    parser.add_argument("--permutation-repeats", type=int, default=5)
    parser.add_argument("--permutation-seed", type=int, default=31415)
    args = parser.parse_args()
    if args.permutation_repeats < 2:
        parser.error("permutation repeats must be at least 2 to report shuffle SD")
    run(args)


if __name__ == "__main__":
    main()
