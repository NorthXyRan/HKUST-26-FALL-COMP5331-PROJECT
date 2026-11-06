"""Frozen M2/M4 protocol, separate from the legacy paper reproduction.

Run: python -m src.final_pipeline --output results/final_m2_m4
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from .features import DEFAULT_HORIZONS, PROJECT, OFFICIAL_COMMIT
from .generators import HistoryIndex, NoEligibleAnomalyError, generate_anomaly
from .generators._common import edge_key
from .injection import read_events, stream_rows, write_stream
from .strict_features import strict_features

SEEDS = (42, 43, 44)
RATES = (1, 5, 10)
WARMUP = 15000
GRID = ((100, 2), (100, 5), (100, 10), (1000, 2), (1000, 5), (1000, 10))
LABEL = "H5_is_fraud"
META = {"t", "u", "v", "split", "event_id", LABEL}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(frame, path):
    # Deterministic gzip header, independent of destination filename and wall time.
    if str(path).endswith(".gz"):
        with path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as packed:
                packed.write(frame.to_csv(index=False).encode())
    else:
        frame.to_csv(path, index=False)


def split_boundaries(events):
    """Choose 60/20/20 benign row quantiles, moving ties wholly to later split."""
    train_end = events[int(len(events) * .60)][0]
    val_end = events[int(len(events) * .80)][0]
    if train_end >= val_end:
        raise ValueError("insufficient distinct timestamps for three splits")
    return train_end, val_end


def split_name(t, bounds):
    return "train" if t < bounds[0] else "validation" if t < bounds[1] else "test"


def make_schedule(events, bounds, rate, seed, warmup=WARMUP):
    by_split = defaultdict(list)
    for i, (t, _, _) in enumerate(events):
        by_split[split_name(t, bounds)].append((i, t))
    # Exact total floor(p E); largest remainders allocate rounding across splits.
    exact = {s: len(rows) * rate / 100 for s, rows in by_split.items()}
    quotas = {s: math.floor(n) for s, n in exact.items()}
    remainder = math.floor(len(events) * rate / 100) - sum(quotas.values())
    for s in sorted(exact, key=lambda s: (-(exact[s] - quotas[s]), s))[:remainder]:
        quotas[s] += 1
    rng = random.Random(seed)
    schedule = []
    first_index = {}
    for i, (t, _, _) in enumerate(events):
        first_index.setdefault(t, i)
    for s in ("train", "validation", "test"):
        times = sorted({t for _, t in by_split[s] if first_index[t] >= warmup})
        if quotas.get(s, 0) and not times:
            raise ValueError("split has no post-warm-up injection timestamps")
        schedule.extend(rng.choices(times, k=quotas.get(s, 0)))
    return sorted(schedule)


def activity_error(state, reference, candidate):
    u, v = reference
    a, b = candidate
    f = lambda n: math.log1p(state.activity.get(n, 0))
    return min(abs(f(u)-f(a)) + abs(f(v)-f(b)),
               abs(f(u)-f(b)) + abs(f(v)-f(a))) / 2


def generate_condition(events, timestamps, seed, strategy, window, bins, bounds):
    by_time = defaultdict(list)
    for t, u, v in events:
        by_time[t].append((u, v))
    counts = Counter(timestamps)
    reference_rng = random.Random(seed + 1000)
    rng = random.Random(seed + (2000 if strategy == "random" else 3000))
    history = HistoryIndex()
    generated, audit = [], []
    started = time.perf_counter()
    for t, normals in by_time.items():
        blocked = {edge_key(u, v) for u, v in normals}
        current = []
        for _ in range(counts[t]):
            u, v = reference_rng.choice(normals)
            state = history.snapshot(t, 1000)
            accepted = None
            for attempt in range(256):
                candidate = generate_anomaly(t, u, v, history, strategy,
                                            rng=rng, activity_window=window, quantile_bins=bins)
                _, a, b = candidate
                if edge_key(a, b) not in blocked:
                    accepted = (t, min(a, b), max(a, b))
                    break
            if accepted is None:
                raise NoEligibleAnomalyError(f"{strategy}: same-time collision budget exhausted at {t}")
            _, a, b = accepted
            # Fail closed on every promised constraint; retain per-anomaly evidence.
            if a == b or a not in state.nodes or b not in state.nodes or edge_key(a,b) in state.pairs:
                raise AssertionError("generator violated candidate constraints")
            audit.append({"t": t, "split": split_name(t, bounds), "reference_u": u, "reference_v": v,
                          "u": a, "v": b, "activity_error": activity_error(state, (u,v), (a,b)),
                          "reference_u_activity": state.activity.get(u, 0),
                          "reference_v_activity": state.activity.get(v, 0),
                          "u_activity": state.activity.get(a, 0), "v_activity": state.activity.get(b, 0),
                          "same_time_rejections": attempt})
            blocked.add(edge_key(a,b))
            generated.append(accepted)
            current.append(accepted)
        history.add_batch(sorted([(t,u,v) for u,v in normals] + current))
    if len(generated) != len(timestamps):
        raise AssertionError("incomplete injection")
    return generated, pd.DataFrame(audit), time.perf_counter() - started


def select_parameters(events, bounds, output):
    """Only prefix through validation is passed to candidate generators. No test scores."""
    prefix = [row for row in events if row[0] < bounds[1]]
    # Build and persist this training/validation-only schedule independently of test.
    timestamps = make_schedule(prefix, bounds, 5, SEEDS[0])
    pd.DataFrame({"t": timestamps}).to_csv(output / "selection_timestamps.csv", index=False)
    rows = []
    for window, bins in GRID:
        print(f"Selecting window={window}, bins={bins}", flush=True)
        started = time.perf_counter()
        try:
            _, audit, _ = generate_condition(prefix, timestamps, SEEDS[0], "activity_matched", window, bins, bounds)
            validation = audit.loc[audit.split == "validation", "activity_error"]
            if validation.empty:
                raise ValueError("no validation anomalies")
            rows.append({"window": window, "bins": bins, "feasible": True,
                         "validation_mean_error": validation.mean(), "validation_p90_error": validation.quantile(.9),
                         "generated": len(audit), "seconds": time.perf_counter()-started, "failure": ""})
        except NoEligibleAnomalyError as exc:
            rows.append({"window": window, "bins": bins, "feasible": False,
                         "validation_mean_error": None, "validation_p90_error": None,
                         "generated": None, "seconds": time.perf_counter()-started, "failure": str(exc)})
    table = pd.DataFrame(rows)
    table.to_csv(output / "parameter_selection.csv", index=False)
    feasible = table.loc[table.feasible].sort_values(["validation_mean_error", "validation_p90_error", "window", "bins"])
    if feasible.empty:
        raise RuntimeError("no feasible configuration; do not run final testing")
    best = feasible.iloc[0]
    return int(best.window), int(best.bins)


def validation_threshold(y, scores, max_fpr=.01):
    """Lowest score threshold with validation FPR <= budget; ties never exceed it."""
    negatives = np.asarray(scores)[np.asarray(y) == 0]
    if not len(negatives):
        raise ValueError("validation negatives required")
    allowed = math.floor(max_fpr * len(negatives))
    descending = np.sort(negatives)[::-1]
    return float(np.nextafter(descending[allowed], np.inf))


def fit_score(train, validation, test, columns, seed):
    normal, anomaly = train.loc[train[LABEL] == 0], train.loc[train[LABEL] == 1]
    if anomaly.empty or len(normal) < len(anomaly):
        raise ValueError("invalid training class counts")
    balanced = pd.concat([normal.sample(n=len(anomaly), random_state=12), anomaly]).sample(frac=1, random_state=42)
    model = RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=None,
                                   min_samples_split=2, min_samples_leaf=1, max_features="sqrt",
                                   bootstrap=True, random_state=seed, n_jobs=1)
    model.fit(balanced[columns], balanced[LABEL])
    val_scores = model.predict_proba(validation[columns])[:, 1]
    threshold = validation_threshold(validation[LABEL], val_scores)
    scores = model.predict_proba(test[columns])[:, 1]
    y = test[LABEL].to_numpy()
    prediction = scores >= threshold
    result = {"pr_auc": average_precision_score(y, scores), "roc_auc": roc_auc_score(y, scores),
              "recall_at_validation_1pct_fpr": prediction[y == 1].mean(),
              "actual_test_fpr": prediction[y == 0].mean(),
              "actual_validation_fpr": (val_scores[validation[LABEL].to_numpy() == 0] >= threshold).mean(),
              "threshold": threshold, "train_rows": len(train), "train_anomalies": len(anomaly),
              "validation_rows": len(validation), "test_rows": len(test), "test_anomalies": int(y.sum()),
              "feature_count": len(columns)}
    return result, pd.DataFrame({"event_id": test.event_id, "t": test.t, "label": y,
                                 "score": scores, "prediction": prediction.astype(int)})


def build_run(events, bounds, rate, seed, window, bins, output):
    run = output / f"{rate}p_seed_{seed}"
    run.mkdir(exist_ok=True)
    timestamps = make_schedule(events, bounds, rate, seed)
    pd.DataFrame({"t": timestamps, "split": [split_name(t, bounds) for t in timestamps]}).to_csv(run / "timestamps.csv", index=False)
    frames, generation = {}, {}
    for strategy in ("random", "activity_matched"):
        print(f"{run.name}: generating {strategy}", flush=True)
        anomalies, audit, seconds = generate_condition(events, timestamps, seed, strategy, window, bins, bounds)
        rows = stream_rows(events, anomalies)
        # Deterministic order and gzip make hashes reproducible on a rerun.
        write_csv(pd.DataFrame(rows, columns=["t", "u", "v", "label"]), run / f"{strategy}_stream.csv.gz")
        write_csv(audit, run / f"{strategy}_audit.csv.gz")
        print(f"{run.name}: extracting strict historical features for {strategy}", flush=True)
        started = time.perf_counter()
        frame = strict_features(rows)
        frame["split"] = [split_name(t, bounds) for t in frame.t]
        frame["event_id"] = range(len(frame))
        # Exclude the same benign warm-up timestamp range in both conditions.
        first_time = events[WARMUP][0]
        while sum(t < first_time for t, _, _ in events) < WARMUP:
            first_time = next(t for t, _, _ in events if t > first_time)
        frame = frame.loc[frame.t >= first_time].reset_index(drop=True)
        frames[strategy] = frame
        write_csv(frame, run / f"{strategy}_features.csv.gz")
        generation[strategy] = {"generated": len(anomalies), "generation_seconds": seconds,
                                "feature_seconds": time.perf_counter()-started,
                                "mean_activity_error": float(audit.activity_error.mean()),
                                "median_activity_error": float(audit.activity_error.median()),
                                "p90_activity_error": float(audit.activity_error.quantile(.9)),
                                "same_time_rejections": int(audit.same_time_rejections.sum()),
                                "constraint_violations": 0}
    results = []
    settings = (("random_to_random", "random", "random"),
                ("random_to_activity_matched", "random", "activity_matched"),
                ("activity_matched_to_activity_matched", "activity_matched", "activity_matched"))
    for feature_set, horizons in (("H1000", (1000,)), ("combined_H", DEFAULT_HORIZONS)):
        columns = [c for c in frames["random"] if c not in META and any(c.startswith(f"H{h}_") for h in horizons)]
        save_json(run / f"{feature_set}_columns.json", columns)
        for name, train_condition, test_condition in settings:
            tr, te = frames[train_condition], frames[test_condition]
            metrics, predictions = fit_score(tr.loc[tr.split == "train"], tr.loc[tr.split == "validation"],
                                             te.loc[te.split == "test"], columns, seed)
            results.append({"rate": rate, "seed": seed, "feature_set": feature_set, "setting": name, **metrics})
            write_csv(predictions, run / f"{feature_set}_{name}_predictions.csv.gz")
            print(f"{run.name} {feature_set} {name}: ROC={metrics['roc_auc']:.4f}", flush=True)
    pd.DataFrame(results).to_csv(run / "metrics.csv", index=False)
    save_json(run / "generation.json", generation)
    return results


def package_hashes(output):
    return {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(output.rglob("*")) if path.is_file() and path.name != "manifest.json"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "results/final_m2_m4")
    args = parser.parse_args(argv)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    events = read_events(PROJECT / "data/UnipartiteNetworks/opsahl-ucsocial/ucimessages.txt")
    bounds = split_boundaries(events)
    config_path = output / "protocol.json"
    if config_path.exists():
        raise FileExistsError("refusing to overwrite frozen package; choose a fresh output directory")
    window, bins = select_parameters(events, bounds, output)
    config = {"protocol_version": "strict_m2_m4_v1", "matched_generator": "activity_matched",
              "selected_activity_window": window, "selected_bins": bins, "selection_grid": GRID,
              "selection_seed": SEEDS[0], "selection_rate": 5, "diagnostic_activity_window": 1000,
              "selection_objective": "feasibility, validation mean error, validation p90, smaller window, fewer bins",
              "seeds": SEEDS, "rates_percent_of_original": RATES, "warmup_benign_events": WARMUP,
              "split_fractions": [.6, .2, .2], "train_end_exclusive": bounds[0], "validation_end_exclusive": bounds[1],
              "history": "all previous observed benign and injected events, strictly t_prime < t; carry across splits",
              "candidate_nodes": "previously observed nodes only, including currently inactive nodes",
              "candidate_pairs": "undirected; no self-loop, historical pair, same-time benign pair or duplicate",
              "coverage_policy": "4096 proposals then exact same-bin reservoir; 256 same-time collision retries; fail closed; no bin widening or random fallback",
              "timestamps": "uniform distinct eligible times with replacement within fixed splits; identical schedule and reference RNG for both conditions",
              "quota": "floor(rate * total benign events), allocated proportionally by split via largest remainders; initial warm-up has no anomalies",
              "features": "independent strict-history TGF-style implementation; absent counters zero; lower median; no current-event insertion",
              "feature_horizons": DEFAULT_HORIZONS, "label": "original injected row label, not inferred",
              "classifier": {"type": "RandomForestClassifier", "n_estimators": 100, "criterion": "gini", "max_depth": None,
                             "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt", "bootstrap": True, "n_jobs": 1},
              "training_balance": "normal undersampling to anomaly count with seed 12; shuffle seed 42",
              "threshold": "training-condition validation negatives only; lowest threshold satisfying validation FPR <= .01; fixed before test",
              "reproduction_baseline": "legacy src.pipeline and results/full_pipeline remain separate; final metrics are not exact upstream reproduction",
              "upstream_formula_reference_commit": OFFICIAL_COMMIT,
              "dataset_sha256": hashlib.sha256((PROJECT / "data/UnipartiteNetworks/opsahl-ucsocial/ucimessages.txt").read_bytes()).hexdigest()}
    save_json(config_path, config)
    all_results = []
    for rate in RATES:
        for seed in SEEDS:
            all_results.extend(build_run(events, bounds, rate, seed, window, bins, output))
    table = pd.DataFrame(all_results)
    table.to_csv(output / "metrics_all_seeds.csv", index=False)
    metrics = ["pr_auc", "roc_auc", "recall_at_validation_1pct_fpr", "actual_test_fpr"]
    summary = table.groupby(["rate", "feature_set", "setting"])[metrics].agg(["mean", "std", "min", "max"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(output / "metrics_summary.csv", index=False)
    paired = table.pivot(index=["rate", "seed", "feature_set"], columns="setting", values=metrics)
    deltas = pd.DataFrame(index=paired.index)
    for metric in metrics:
        deltas[metric + "_cross_minus_random"] = paired[metric]["random_to_activity_matched"] - paired[metric]["random_to_random"]
        deltas[metric + "_retraining_recovery"] = paired[metric]["activity_matched_to_activity_matched"] - paired[metric]["random_to_activity_matched"]
    deltas.reset_index().to_csv(output / "paired_deltas.csv", index=False)
    deltas.groupby(["rate", "feature_set"]).agg(["mean", "std"]).to_csv(output / "paired_deltas_summary.csv")
    source_hashes = {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted((PROJECT / "src").rglob("*.py"))}
    save_json(output / "manifest.json", {"base_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
                                       "source_sha256": source_hashes, "files_sha256": package_hashes(output)})
    print(f"Final package complete: {output}", flush=True)


if __name__ == "__main__":
    main()
