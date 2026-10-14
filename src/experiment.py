"""End-to-end, seeded UCI Messages experiment using the official TGF extractor.

The random injector follows the candidate-generation procedure in TGF's
2-Random-Injection/build_injectable.sh + inject.sh, with an explicit Python RNG
instead of shell `shuf` and unseeded random_sample.py. Feature extraction always
executes the untouched official 3-FeatureGeneration/main.py.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import platform
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from .generators import HistoryIndex, NoEligibleAnomalyError, generate_activity_matched_anomaly


PROJECT = Path(__file__).resolve().parents[1]
OFFICIAL = PROJECT / "TGF"
DEFAULT_DATA = OFFICIAL / "1-PreprocessingOfDatasets/UnipartiteNetworks/opsahl-ucsocial/ucimessages.txt"
FEATURE_MAIN = OFFICIAL / "3-FeatureGeneration/main.py"
FEATURE_DIR = FEATURE_MAIN.parent
OFFICIAL_COMMIT = "af9d575e07c7bc3fd0b4a0b8d0249e7cdd2eb88b"
DEFAULT_HORIZONS = (5, 50, 100, 1000, 10000, 15000)


def read_events(path: Path) -> list[tuple[int, int, int]]:
    events = []
    previous = None
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            fields = line.split()
            if len(fields) != 3:
                raise ValueError(f"{path}:{line_no}: expected three columns t u v")
            t, u, v = map(int, fields)
            if previous is not None and t < previous:
                raise ValueError(f"{path}:{line_no}: input is not timestamp-sorted")
            if u == v:
                raise ValueError(f"{path}:{line_no}: self-loop in preprocessed input")
            events.append((t, u, v))
            previous = t
    if not events:
        raise ValueError(f"no events in {path}")
    return events


def _eligible_timestamps(events: list[tuple[int, int, int]]) -> list[int]:
    """TGF samples timestamps; omit the first two so matched history has two nodes."""
    seen: set[int] = set()
    eligible = []
    for t, _, _ in events:
        if t not in seen:
            seen.add(t)
            if len(seen) > 2:
                eligible.append(t)
    if not eligible:
        raise ValueError("need at least three distinct timestamps for injection")
    return eligible


def make_shared_timestamps(events, n_injected: int, seed: int) -> list[int]:
    """Sample TGF-style injection times once and share them across strategies."""
    rng = random.Random(seed)
    eligible = _eligible_timestamps(events)
    return sorted(rng.choices(eligible, k=n_injected))


def make_activity_matched(events, timestamps, seed, *, activity_window=100, bins=10):
    """Generate anomaly edges at shared timestamps using only t'<t history."""
    base_by_time: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for t, u, v in events:
        base_by_time[t].append((u, v))
    counts: dict[int, int] = defaultdict(int)
    for t in timestamps:
        counts[t] += 1

    rng = random.Random(seed)
    index = HistoryIndex()
    generated: list[tuple[int, int, int]] = []
    all_times = sorted(base_by_time)
    for t in all_times:
        current: list[tuple[int, int, int]] = []
        blocked = {frozenset((u, v)) for u, v in base_by_time[t]}
        for _ in range(counts.get(t, 0)):
            reference_u, reference_v = rng.choice(base_by_time[t])
            accepted = None
            # The public generator samples a valid historical non-edge. Retry only
            # when a candidate collides with another event at this same timestamp.
            for _attempt in range(256):
                try:
                    candidate = generate_activity_matched_anomaly(
                        t, reference_u, reference_v, index,
                        rng=rng, activity_window=activity_window, quantile_bins=bins,
                    )
                except NoEligibleAnomalyError:
                    break
                key = frozenset(candidate[1:])
                if key not in blocked:
                    accepted = candidate
                    blocked.add(key)
                    break
            if accepted is None:
                raise NoEligibleAnomalyError(
                    f"activity matching failed at timestamp {t}; lower bins or report coverage failure"
                )
            current.append(accepted)
            generated.append(accepted)
        # All base edges and generated anomalies at t enter history only after
        # every injection at t has been selected: strict timestamp visibility.
        index.add_batch([(t, u, v) for u, v in base_by_time[t]] + current)
    return generated


def make_random_tgf_style(events, timestamps, seed):
    """Seeded implementation of TGF's random timestamp/node candidate injection."""
    rng = random.Random(seed)
    nodes = sorted({node for _, u, v in events for node in (u, v)})
    original_at_time: dict[int, set[frozenset[int]]] = defaultdict(set)
    for t, u, v in events:
        original_at_time[t].add(frozenset((u, v)))
    used: dict[int, set[frozenset[int]]] = defaultdict(set)
    generated = []
    for t in timestamps:
        accepted = None
        for _attempt in range(10000):
            u, v = rng.sample(nodes, 2)
            key = frozenset((u, v))
            if key not in original_at_time[t] and key not in used[t]:
                accepted = (t, min(u, v), max(u, v))
                used[t].add(key)
                break
        if accepted is None:
            raise RuntimeError(f"could not find a TGF-style candidate for timestamp {t}")
        generated.append(accepted)
    return generated


def stream_rows(events, anomalies):
    rows = [(t, u, v, 0) for t, u, v in events]
    rows.extend((t, u, v, 1) for t, u, v in anomalies)
    # Matches the official injection's timestamp sort; lexical endpoint tie-break
    # gives deterministic ordering for simultaneous interactions.
    rows.sort(key=lambda r: (r[0], str(r[1]), str(r[2]), r[3]))
    return rows


def write_stream(path: Path, rows):
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        for row in rows:
            f.write("%d %d %d %d\n" % row)


def _flatten(record, prefix=""):
    flattened = {}
    for key, value in record.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten(value, name))
        else:
            flattened[name] = value
    return flattened


def official_features(rows, horizons):
    """Call official TGF main.py once per H and reproduce notebook concatenation."""
    stream = "".join(f"{t} {u} {v} {label}\n" for t, u, v, label in rows)
    feature_frames = []
    for h in horizons:
        proc = subprocess.run(
            [sys.executable, str(FEATURE_MAIN), "-H", str(h), "-int"],
            cwd=FEATURE_DIR, input=stream, text=True, capture_output=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        if proc.returncode:
            raise RuntimeError(f"official TGF feature extraction failed for H={h}:\n{proc.stderr[-4000:]}")
        records = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
        if len(records) != len(rows):
            raise RuntimeError(f"TGF returned {len(records)} rows for {len(rows)} input events at H={h}")
        frame = pd.DataFrame.from_records([_flatten(x) for x in records]).add_prefix(f"H{h}_")
        feature_frames.append(frame)
    combined = pd.concat(feature_frames, axis=1)
    combined = combined.dropna(axis=0, how="any")
    target = "H5_is_fraud"
    metadata = []
    for h in horizons:
        metadata.extend(f"H{h}_{suffix}" for suffix in ("t", "u", "v", "history_type", "bip", "cost"))
        if h != 5:
            metadata.append(f"H{h}_is_fraud")
    Xy = combined.drop(columns=metadata, errors="ignore").copy()
    if target not in Xy:
        raise RuntimeError("H=5 label was not returned by official feature extractor")
    return Xy, len(rows) - len(combined)


def _recall_at_fpr(y, score, max_fpr=0.01):
    y = np.asarray(y)
    negatives = score[y == 0]
    positives = y == 1
    threshold = np.quantile(negatives, 1.0 - max_fpr, method="higher")
    predicted = score >= threshold
    return float(predicted[positives].mean()), float(predicted[~positives].mean())


def fit_and_score(train, test, seed):
    y_col = "H5_is_fraud"
    Xtr = train.drop(columns=y_col)
    ytr = train[y_col].astype(int)
    Xte = test.drop(columns=y_col)
    yte = test[y_col].astype(int)
    normal = train.loc[ytr == 0]
    anomaly = train.loc[ytr == 1]
    if normal.empty or anomaly.empty or yte.nunique() < 2:
        raise ValueError("chronological split must contain both classes in train and test")
    # Match the official UCI Messages notebook: balance training normals by
    # undersampling with random_state=12, then shuffle with random_state=42.
    balanced = pd.concat([normal.sample(n=len(anomaly), random_state=12), anomaly])
    balanced = balanced.sample(frac=1, random_state=42)
    model = RandomForestClassifier(random_state=seed)
    model.fit(balanced.drop(columns=y_col), balanced[y_col].astype(int))
    scores = model.predict_proba(Xte)[:, 1]
    recall, actual_fpr = _recall_at_fpr(yte, scores)
    return {
        "pr_auc": float(average_precision_score(yte, scores)),
        "roc_auc": float(roc_auc_score(yte, scores)),
        "recall_at_1pct_fpr": recall,
        "actual_fpr_at_threshold": actual_fpr,
        "train_rows_before_balance": int(len(train)),
        "train_normals_before_balance": int((ytr == 0).sum()),
        "train_anomalies": int((ytr == 1).sum()),
        "test_rows": int(len(test)),
        "test_anomalies": int((yte == 1).sum()),
        "feature_count": int(Xtr.shape[1]),
    }


def _write_csv(path, records):
    if not records:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def run_one(data_path, output_dir, seed, anomaly_pct, horizons, activity_window, bins, train_fraction):
    events = read_events(data_path)
    n_injected = int(len(events) * anomaly_pct / 100)
    if n_injected <= 0:
        raise ValueError("anomaly percentage yields zero injected events")
    timestamps = make_shared_timestamps(events, n_injected, seed)
    matched = make_activity_matched(events, timestamps, seed + 1, activity_window=activity_window, bins=bins)
    random_edges = make_random_tgf_style(events, timestamps, seed + 2)
    output_dir.mkdir(parents=True, exist_ok=True)

    condition_frames = {}
    for name, anomalies in (("random", random_edges), ("activity_matched", matched)):
        rows = stream_rows(events, anomalies)
        write_stream(output_dir / f"{name}_injected.tsv.gz", rows)
        Xy, warmup_dropped = official_features(rows, horizons)
        split = int(round(len(Xy) * train_fraction))
        train, test = Xy.iloc[:split].copy(), Xy.iloc[split:].copy()
        condition_frames[name] = (train, test, warmup_dropped, len(rows))
        # Compact reproducibility output: merged feature table with labels.
        Xy.to_csv(output_dir / f"{name}_features.csv.gz", index=False, compression="gzip")
        print(f"{name}: {len(rows)} stream rows, {len(Xy)} feature rows, {warmup_dropped} warm-up rows removed", flush=True)

    random_train, random_test, _, _ = condition_frames["random"]
    matched_train, matched_test, _, _ = condition_frames["activity_matched"]
    results = []
    for setting, train, test in (
        ("random_to_random", random_train, random_test),
        ("random_to_matched", random_train, matched_test),
        ("matched_to_matched", matched_train, matched_test),
    ):
        results.append({"seed": seed, "setting": setting, **fit_and_score(train, test, seed)})

    summary = {
        "seed": seed,
        "seeds": {
            "shared_timestamp_schedule": seed,
            "activity_matched_generator": seed + 1,
            "random_tgf_style_generator": seed + 2,
            "random_forest": seed,
            "training_negative_undersample": 12,
            "training_shuffle": 42,
        },
        "dataset": str(data_path.relative_to(PROJECT)) if data_path.is_relative_to(PROJECT) else data_path.name,
        "data_sha256": __import__("hashlib").sha256(data_path.read_bytes()).hexdigest(),
        "official_tgf_commit": OFFICIAL_COMMIT,
        "official_feature_extractor_sha256": __import__("hashlib").sha256(FEATURE_MAIN.read_bytes()).hexdigest(),
        "feature_extractor": "TGF/3-FeatureGeneration/main.py (called in place)",
        "random_injection": "seeded reimplementation of TGF 2-Random-Injection candidate timestamp/node procedure",
        "events_original": len(events),
        "anomaly_percentage_of_original": anomaly_pct,
        "anomalies_injected_each_condition": n_injected,
        "anomaly_share_after_injection": n_injected / (len(events) + n_injected),
        "activity_window_events": activity_window,
        "activity_quantile_bins": bins,
        "history_visibility_for_injection": "strict timestamp < t; equal-time events are added after all injections at t are generated",
        "feature_extractor_history_policy": "official TGF event-count H; rows are processed sequentially in the deterministic sorted stream, so earlier rows at an equal timestamp can enter its graph state",
        "random_endpoint_population": "all distinct endpoints in the original UCI Messages stream (TGF candidate procedure)",
        "shared_injection_timestamp_schedule": "sampled with replacement from original timestamps after the first two distinct times; identical times used for both conditions",
        "horizons": list(horizons),
        "split": "chronological feature rows",
        "train_fraction": train_fraction,
        "rf": "sklearn.ensemble.RandomForestClassifier(random_state=seed; other defaults)",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": __import__("sklearn").__version__,
        "feature_rows_random": len(random_train) + len(random_test),
        "feature_rows_activity_matched": len(matched_train) + len(matched_test),
        "warmup_rows_dropped": condition_frames["random"][2],
        "train_rows": len(random_train),
        "test_rows": len(random_test),
        "train_last_timestamp": int(stream_rows(events, random_edges)[condition_frames["random"][2] + len(random_train) - 1][0]),
        "test_first_timestamp": int(stream_rows(events, random_edges)[condition_frames["random"][2] + len(random_train)][0]),
        "results": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_csv(output_dir / "metrics.csv", results)
    print("\n" + pd.DataFrame(results).to_string(index=False), flush=True)
    print(f"\nSaved run outputs to {output_dir}", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=PROJECT / "results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", help="comma-separated repeated-run seeds; overrides --seed")
    parser.add_argument("--anomaly-percent", type=float, default=5.0)
    parser.add_argument("--horizons", default=",".join(map(str, DEFAULT_HORIZONS)))
    parser.add_argument("--activity-window", type=int, default=100)
    parser.add_argument("--activity-bins", type=int, default=10)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    args = parser.parse_args(argv)
    horizons = tuple(int(x) for x in args.horizons.split(","))
    if 5 not in horizons:
        parser.error("horizons must include 5 because the official notebook uses H5_is_fraud as the label")
    if not 0 < args.train_fraction < 1:
        parser.error("--train-fraction must be between 0 and 1")
    seeds = [int(value) for value in args.seeds.split(",")] if args.seeds else [args.seed]
    if not seeds or len(seeds) != len(set(seeds)):
        parser.error("--seeds must contain unique integer values")
    all_metrics = []
    for seed in seeds:
        run_dir = args.output / f"seed_{seed}"
        summary = run_one(args.data, run_dir, seed, args.anomaly_percent, horizons, args.activity_window, args.activity_bins, args.train_fraction)
        all_metrics.extend(summary["results"])
    if len(seeds) > 1:
        metrics = pd.DataFrame(all_metrics)
        metrics.to_csv(args.output / "metrics_all_seeds.csv", index=False)
        grouped = metrics.groupby("setting").agg(
            pr_auc_mean=("pr_auc", "mean"), pr_auc_sd=("pr_auc", "std"),
            roc_auc_mean=("roc_auc", "mean"), roc_auc_sd=("roc_auc", "std"),
            recall_at_1pct_fpr_mean=("recall_at_1pct_fpr", "mean"),
            recall_at_1pct_fpr_sd=("recall_at_1pct_fpr", "std"),
        ).reset_index()
        grouped.to_csv(args.output / "metrics_mean_sd.csv", index=False)
        print("\nAcross-seed mean and sample SD:\n" + grouped.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
