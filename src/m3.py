"""Reproducible M3 audit of frozen M2/M4 streams (never modifies inputs).

python -m src.m3 --output results/m3
A: activity errors, paired effects, window sensitivity.
B: structure/timing, common benign history, input validation.
"""

import argparse
from collections import Counter
import gzip
import hashlib
from importlib.metadata import version
from itertools import groupby
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from .m3_history import History, activity_error

ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    "activity_low",
    "activity_high",
    "degree_mean",
    "common_neighbors",
    "jaccard",
    "pair_count",
    "recent_pair_count",
    "recency_minutes",
    "concentration_1h_24h",
    "burstiness",
    "unseen_endpoint",
)
KEYS = ["rate", "seed", "strategy", "split", "history_policy", "window"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_csv(frame, path):
    if path.suffix == ".gz":
        with path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as packed:
                packed.write(frame.to_csv(index=False).encode())
    else:
        frame.to_csv(path, index=False)


def read_verified(path, package, manifest):
    require(
        digest(path) == manifest["files_sha256"][str(path.relative_to(package))],
        f"frozen input hash mismatch: {path}",
    )
    return pd.read_csv(path)


def replay(stream, audit, rate, seed, strategy, windows, bounds):
    """Scores the same edges under observed and counterfactual benign-only histories."""
    require(stream.t.is_monotonic_increasing, "stream must be timestamp sorted")
    require(set(stream.label) <= {0, 1}, "invalid stream labels")
    require(not audit.duplicated(["t", "u", "v"]).any(), "duplicate audit edge")
    audits = {int(t): list(g.itertuples(index=False)) for t, g in audit.groupby("t", sort=False)}
    histories = {name: History(windows) for name in ("observed", "benign_only")}
    records, checked = [], 0
    for t, group in groupby(stream.itertuples(index=False, name=None), key=lambda r: r[0]):
        batch = list(group)
        normals = {(min(u, v), max(u, v)) for _, u, v, label in batch if label == 0}
        expected = Counter((min(r.u, r.v), max(r.u, r.v)) for r in audits.get(t, []))
        actual = Counter((min(u, v), max(u, v)) for _, u, v, label in batch if label == 1)
        require(actual == expected, "audit/stream anomaly mismatch")
        for h in histories.values():
            h.advance(t)
        split = "train" if t < bounds[0] else "validation" if t < bounds[1] else "test"
        for r in audits.get(t, []):
            require(r.split == split, "audit split disagrees with frozen boundaries")
            require(
                tuple(sorted((r.reference_u, r.reference_v))) in normals,
                "reference not in current benign batch",
            )
            require(tuple(sorted((r.u, r.v))) not in normals, "same-time benign collision")
            require(r.u != r.v, "anomaly self-loop")
            for policy, h in histories.items():
                normal = h.score(t, r.reference_u, r.reference_v)
                anomaly = h.score(t, r.u, r.v)
                require(anomaly[1000]["pair_count"] == 0, "historical anomaly pair")
                require(anomaly[1000]["unseen_endpoint"] == 0, "unseen anomaly endpoint")
                if policy == "observed":
                    a = h.windows[1000].activity
                    for node, value in [
                        (r.reference_u, r.reference_u_activity),
                        (r.reference_v, r.reference_v_activity),
                        (r.u, r.u_activity),
                        (r.v, r.v_activity),
                    ]:
                        require(a[node] == value, "activity audit mismatch")
                    require(
                        np.isclose(
                            activity_error(normal[1000], anomaly[1000]),
                            r.activity_error,
                            rtol=0,
                            atol=1e-12,
                        ),
                        "activity error mismatch",
                    )
                for window in windows:
                    error = activity_error(normal[window], anomaly[window])
                    for kind, scores in [
                        ("normal_reference", normal[window]),
                        ("anomaly", anomaly[window]),
                    ]:
                        records.append(
                            dict(
                                rate=rate,
                                seed=seed,
                                strategy=strategy,
                                split=split,
                                history_policy=policy,
                                window=window,
                                reference_id=checked,
                                t=t,
                                kind=kind,
                                activity_error=error,
                                **scores,
                            )
                        )
            checked += 1
        for policy, h in histories.items():
            pairs = [(u, v) for _, u, v, label in batch if policy == "observed" or label == 0]
            h.add_batch(t, pairs)
    require(checked == len(audit), "audit timestamp absent from stream")
    return pd.DataFrame(records)


def summarize(frame, output):
    rows, distances = [], []
    for keys, g in frame.groupby(KEYS, sort=True):
        meta = dict(zip(KEYS, keys))
        for metric in METRICS:
            arrays = {}
            for kind in ("normal_reference", "anomaly"):
                values = g.loc[g.kind == kind, metric]
                finite = values.dropna().to_numpy()
                arrays[kind] = finite
                rows.append(
                    dict(
                        **meta,
                        kind=kind,
                        metric=metric,
                        n_total=len(values),
                        n_valid=len(finite),
                        missing_fraction=values.isna().mean(),
                        mean=np.mean(finite) if len(finite) else np.nan,
                        median=np.median(finite) if len(finite) else np.nan,
                        p90=np.quantile(finite, 0.9) if len(finite) else np.nan,
                        zero_fraction=np.mean(finite == 0) if len(finite) else np.nan,
                    )
                )
            x, y = arrays["normal_reference"], arrays["anomaly"]
            log_scale = metric not in (
                "jaccard",
                "concentration_1h_24h",
                "burstiness",
                "unseen_endpoint",
            )
            distances.append(
                dict(
                    **meta,
                    metric=metric,
                    n_reference=len(x),
                    n_anomaly=len(y),
                    ks=ks_2samp(x, y, method="asymp").statistic if len(x) and len(y) else np.nan,
                    wasserstein=(
                        wasserstein_distance(
                            np.log1p(x) if log_scale else x, np.log1p(y) if log_scale else y
                        )
                        if len(x) and len(y)
                        else np.nan
                    ),
                    scale="log1p" if log_scale else "raw",
                )
            )
    write_csv(pd.DataFrame(rows), output / "distributions.csv")
    write_csv(pd.DataFrame(distances), output / "distances.csv")
    errors = (
        frame[frame.kind == "anomaly"]
        .groupby(KEYS)
        .activity_error.agg(["count", "mean", "median", lambda x: x.quantile(0.9)])
        .reset_index()
        .rename(columns={"<lambda_0>": "p90"})
    )
    write_csv(errors, output / "activity_errors.csv")
    paired = errors.pivot(
        index=[k for k in KEYS if k != "strategy"], columns="strategy", values="mean"
    ).reset_index()
    paired["matched_minus_random"] = paired.activity_matched - paired.random
    paired["reduction_pct"] = 100 * (1 - paired.activity_matched / paired.random)
    write_csv(paired, output / "paired_effects.csv")
    aggregation = paired.groupby(["rate", "split", "history_policy", "window"])[
        ["random", "activity_matched", "matched_minus_random", "reduction_pct"]
    ].agg(["mean", "std", "min", "max"])
    aggregation.columns = ["_".join(c) for c in aggregation.columns]
    write_csv(aggregation.reset_index(), output / "seed_summary.csv")


def main(package, output, windows=(100, 1000, 10000)):
    require(
        1000 in windows and len(set(windows)) == len(windows),
        "unique windows must include audit window 1000",
    )
    require(not output.exists(), "output exists: choose a new directory to preserve prior runs")
    require(
        not output.resolve().is_relative_to(package.resolve()),
        "output cannot be inside frozen inputs",
    )
    upstream = json.loads((package / "manifest.json").read_text())
    require(
        digest(package / "protocol.json") == upstream["files_sha256"]["protocol.json"],
        "protocol hash mismatch",
    )
    protocol = json.loads((package / "protocol.json").read_text())
    require(protocol["protocol_version"] == "strict_m2_m4_v1", "unsupported input protocol")
    benign = ROOT / "data/UnipartiteNetworks/opsahl-ucsocial/ucimessages.txt"
    require(digest(benign) == protocol["dataset_sha256"], "benign dataset hash mismatch")
    original = pd.read_csv(benign, sep=r"\s+", names=["t", "u", "v"])
    output.mkdir(parents=True)
    (output / "rows").mkdir()
    frames = []
    checks = []
    inputs = {
        str(p.relative_to(ROOT)): digest(p)
        for p in (package / "manifest.json", package / "protocol.json", benign)
    }
    for rate in protocol["rates_percent_of_original"]:
        for seed in protocol["seeds"]:
            references = None
            for strategy in ("random", "activity_matched"):
                run = package / f"{rate}p_seed_{seed}"
                ap, sp = run / f"{strategy}_audit.csv.gz", run / f"{strategy}_stream.csv.gz"
                audit, stream = (read_verified(p, package, upstream) for p in (ap, sp))
                inputs.update({str(p.relative_to(ROOT)): digest(p) for p in (ap, sp)})
                require(len(audit) == len(original) * rate // 100, "wrong anomaly quota")
                pd.testing.assert_frame_equal(
                    stream.loc[stream.label == 0, ["t", "u", "v"]]
                    .sort_values(["t", "u", "v"])
                    .reset_index(drop=True),
                    original.sort_values(["t", "u", "v"]).reset_index(drop=True),
                )
                ref = audit[["t", "reference_u", "reference_v"]]
                if references is None:
                    references = ref
                else:
                    pd.testing.assert_frame_equal(references, ref)
                frame = replay(
                    stream,
                    audit,
                    rate,
                    seed,
                    strategy,
                    windows,
                    (protocol["train_end_exclusive"], protocol["validation_end_exclusive"]),
                )
                write_csv(frame, output / "rows" / f"{rate}p_seed_{seed}_{strategy}.csv.gz")
                frames.append(frame)
                checks.append(
                    dict(
                        rate=rate,
                        seed=seed,
                        strategy=strategy,
                        anomalies=len(audit),
                        diagnostic_rows=len(frame),
                        status="passed",
                    )
                )
                print(
                    f"{rate}% seed {seed} {strategy}: {len(audit)} anomalies validated", flush=True
                )
    all_rows = pd.concat(frames, ignore_index=True)
    # Common benign-history references must agree exactly across both conditions.
    shared = all_rows[
        (all_rows.kind == "normal_reference") & (all_rows.history_policy == "benign_only")
    ]
    a, b = (
        shared[shared.strategy == s]
        .drop(columns=["strategy", "activity_error"])
        .reset_index(drop=True)
        for s in ("random", "activity_matched")
    )
    pd.testing.assert_frame_equal(a, b)
    summarize(all_rows, output)
    write_csv(pd.DataFrame(checks), output / "checks.csv")
    from .m3_report import report

    report(output, all_rows)
    sources = [
        "src/m3.py",
        "src/m3_history.py",
        "src/m3_report.py",
        "src/m3_verify.py",
        "tests/test_m3.py",
    ]
    manifest = dict(
        protocol="m3_v2",
        base_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        input_sha256=inputs,
        source_sha256={p: digest(ROOT / p) for p in sources},
        environment={
            p: version(p) for p in ("numpy", "pandas", "scipy", "matplotlib", "scikit-learn")
        },
        python=platform.python_version(),
        configuration=dict(
            windows=list(windows),
            primary_window=1000,
            primary_split="test",
            history_policies=["observed", "benign_only"],
            burstiness_intervals=20,
            min_burstiness_intervals=5,
            interval_sd="population",
            seed_variation="injection only; no model training",
            reference="sampled benign reference edge at each anomaly timestamp, separately scored in each history",
            sensitivity="post-hoc diagnostics only; no parameter selection or reinjection",
        ),
        output_sha256={
            str(p.relative_to(output)): digest(p) for p in output.rglob("*") if p.is_file()
        },
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=ROOT / "results/final_m2_m4")
    parser.add_argument("--output", type=Path, default=ROOT / "results/m3")
    parser.add_argument("--windows", type=int, nargs="+", default=[100, 1000, 10000])
    args = parser.parse_args()
    main(args.package.resolve(), args.output.resolve(), tuple(args.windows))
