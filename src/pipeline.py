"""Staged CLI for injection, official TGF feature extraction, and evaluation."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd

from .features import DEFAULT_HORIZONS, OFFICIAL_COMMIT, PROJECT, official_features
from .injection import (
    make_activity_matched,
    make_random_tgf_style,
    make_shared_timestamps,
    read_events,
    stream_rows,
    write_stream,
)
from .modeling import fit_and_score

DEFAULT_DATA = PROJECT / "TGF/1-PreprocessingOfDatasets/UnipartiteNetworks/opsahl-ucsocial/ucimessages.txt"
FEATURE_CONFIGURATIONS = {
    "H1000": (1000,),
    "combined_H": DEFAULT_HORIZONS,
}


def read_config(run_dir: Path) -> dict:
    path = run_dir / "run_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}; run the inject stage first")
    return json.loads(path.read_text())


def inject(data: Path, run_dir: Path, seed: int, anomaly_percent: float,
           activity_window: int, activity_bins: int) -> None:
    events = read_events(data)
    n_anomalies = int(len(events) * anomaly_percent / 100)
    if n_anomalies < 1:
        raise ValueError("anomaly percentage yields zero injected events")
    timestamps = make_shared_timestamps(events, n_anomalies, seed)
    matched = make_activity_matched(
        events, timestamps, seed + 1,
        activity_window=activity_window, bins=activity_bins,
    )
    random_edges = make_random_tgf_style(events, timestamps, seed + 2)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_stream(run_dir / "random_injected.tsv.gz", stream_rows(events, random_edges))
    write_stream(run_dir / "activity_matched_injected.tsv.gz", stream_rows(events, matched))
    config = {
        "input_data": str(data),
        "input_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "official_tgf_commit": OFFICIAL_COMMIT,
        "original_events": len(events),
        "anomaly_percent_of_original": anomaly_percent,
        "injected_anomalies_per_condition": n_anomalies,
        "combined_stream_anomaly_share": n_anomalies / (len(events) + n_anomalies),
        "seeds": {
            "shared_timestamps": seed,
            "activity_matched_generator": seed + 1,
            "random_generator": seed + 2,
            "random_forest": seed,
            "training_normal_undersampling": 12,
            "training_shuffle": 42,
        },
        "activity_window_events": activity_window,
        "activity_quantile_bins": activity_bins,
        "shared_timestamps_between_conditions": True,
        "injection_history_policy": "strict timestamp < t; same-time rows enter history after generation",
        "label_horizon": 5,
        "chronological_train_fraction": 0.70,
        "feature_configurations": {},
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Injected {n_anomalies} anomalies per condition into {len(events)} normal events.")
    print(f"Saved injection streams and run config under {run_dir}")


def extract(run_dir: Path, feature_set: str) -> None:
    config = read_config(run_dir)
    selected = FEATURE_CONFIGURATIONS if feature_set == "all" else {
        feature_set: FEATURE_CONFIGURATIONS[feature_set]
    }
    for condition in ("random", "activity_matched"):
        stream_path = run_dir / f"{condition}_injected.tsv.gz"
        if not stream_path.is_file():
            raise FileNotFoundError(f"missing injection stream: {stream_path}")
        with gzip.open(stream_path, "rt", encoding="utf-8") as stream:
            rows = []
            for line_no, line in enumerate(stream, 1):
                fields = line.split()
                if len(fields) != 4:
                    raise ValueError(f"{stream_path}:{line_no}: expected t u v label")
                rows.append(tuple(map(int, fields)))
        for name, horizons in selected.items():
            features, warmup = official_features(rows, horizons)
            features.to_csv(run_dir / f"{condition}_features_{name}.csv.gz",
                            index=False, compression="gzip")
            config.setdefault("feature_configurations", {})[name] = list(horizons)
            config.setdefault("feature_rows", {})[f"{condition}_{name}"] = len(features)
            config.setdefault("warmup_rows_removed", {})[f"{condition}_{name}"] = warmup
            print(f"{condition}/{name}: {len(rows)} stream rows, {len(features)} feature rows, {warmup} warm-up rows removed")
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")


def evaluate(run_dir: Path, train_fraction: float) -> list[dict]:
    config = read_config(run_dir)
    feature_sets = config.get("feature_configurations", {})
    if not feature_sets:
        raise ValueError("no features found; run the extract stage first")
    seed = int(config["seeds"]["random_forest"])
    all_results = []
    for feature_set in feature_sets:
        frames = {}
        for condition in ("random", "activity_matched"):
            path = run_dir / f"{condition}_features_{feature_set}.csv.gz"
            if not path.is_file():
                raise FileNotFoundError(f"missing features: {path}; extract this feature set first")
            frames[condition] = pd.read_csv(path)
        splits = {}
        for condition, frame in frames.items():
            split = int(round(len(frame) * train_fraction))
            splits[condition] = (frame.iloc[:split].copy(), frame.iloc[split:].copy())
        random_train, random_test = splits["random"]
        matched_train, matched_test = splits["activity_matched"]
        settings = (
            ("random_to_random", random_train, random_test),
            ("random_to_activity_matched", random_train, matched_test),
            ("activity_matched_to_activity_matched", matched_train, matched_test),
        )
        results = [{"feature_set": feature_set, "setting": name,
                    **fit_and_score(train, test, seed)}
                   for name, train, test in settings]
        pd.DataFrame(results).drop(columns="feature_set").to_csv(
            run_dir / f"metrics_{feature_set}.csv", index=False
        )
        all_results.extend(results)

    metrics = pd.DataFrame(all_results)
    metrics.to_csv(run_dir / "metrics_all_configurations.csv", index=False)
    summary = {
        **config,
        "chronological_train_fraction": train_fraction,
        "classifier": "sklearn.ensemble.RandomForestClassifier(default parameters; random_state=seed)",
        "evaluation_metrics": ["PR-AUC", "ROC-AUC", "recall at 1% FPR"],
        "results": all_results,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_report(run_dir, summary, all_results)
    print(metrics.to_string(index=False))
    print(f"\nSaved run outputs under {run_dir}")
    return all_results


def _write_report(run_dir: Path, config: dict, rows: list[dict]) -> None:
    report = [
        "# TGF injection experiment",
        "",
        f"- Data: `{config['input_data']}` ({config['original_events']:,} original events)",
        f"- Injection: {config['anomaly_percent_of_original']:g}% of original events; "
        f"{config['injected_anomalies_per_condition']:,} anomalies per generator.",
        f"- Seed: {config['seeds']['random_forest']}; chronological train fraction: "
        f"{config['chronological_train_fraction']:.0%}.",
        f"- Official feature extractor commit: `{config['official_tgf_commit']}`.",
        "- Training: normal undersampling state 12, shuffle state 42, default Random Forest.",
        "- Evaluation: PR-AUC, ROC-AUC, recall at 1% FPR.",
        "",
    ]
    for feature_set in dict.fromkeys(row["feature_set"] for row in rows):
        report.extend([f"## {feature_set}", "", "| Train → test | PR-AUC | ROC-AUC | Recall at 1% FPR |", "|---|---:|---:|---:|"])
        for row in rows:
            if row["feature_set"] == feature_set:
                report.append(
                    f"| {row['setting']} | {row['pr_auc']:.6f} | {row['roc_auc']:.6f} | {row['recall_at_1pct_fpr']:.6f} |"
                )
        report.append("")
    report.extend([
        "## Reproduction commands",
        "",
        "```bash",
        f"python -m src.pipeline inject --data {config['input_data']} --seed {config['seeds']['random_forest']} --anomaly-percent {config['anomaly_percent_of_original']:g} --output {run_dir}",
        f"python -m src.pipeline extract --run-dir {run_dir} --feature-set all",
        f"python -m src.pipeline evaluate --run-dir {run_dir} --train-fraction {config['chronological_train_fraction']}",
        "```",
        "",
    ])
    (run_dir / "REPORT.md").write_text("\n".join(report))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    p = subparsers.add_parser("inject", help="generate random and activity-matched streams")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output", type=Path, default=PROJECT / "results/full_pipeline/seed_42")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--anomaly-percent", type=float, default=5.0)
    p.add_argument("--activity-window", type=int, default=100)
    p.add_argument("--activity-bins", type=int, default=10)
    p = subparsers.add_parser("extract", help="extract features with official TGF code")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--feature-set", choices=(*FEATURE_CONFIGURATIONS, "all"), default="all")
    p = subparsers.add_parser("evaluate", help="run the three train/test settings")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--train-fraction", type=float, default=0.70)
    args = parser.parse_args(argv)
    if args.stage == "inject":
        if args.anomaly_percent <= 0 or args.activity_window <= 0 or args.activity_bins <= 0:
            parser.error("anomaly percent, activity window, and activity bins must be positive")
        inject(args.data, args.output, args.seed, args.anomaly_percent,
               args.activity_window, args.activity_bins)
    elif args.stage == "extract":
        extract(args.run_dir, args.feature_set)
    else:
        if not 0 < args.train_fraction < 1:
            parser.error("--train-fraction must be between 0 and 1")
        evaluate(args.run_dir, args.train_fraction)


if __name__ == "__main__":
    main()
