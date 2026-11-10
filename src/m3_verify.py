"""Verify M3 hashes, row counts, pairing and numerically recomputed summaries."""

import argparse
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from .m3 import ROOT, digest, require, summarize


def verify(output, check_sources=True):
    output = Path(output)
    m = json.loads((output / "manifest.json").read_text())
    for name, expected in m["input_sha256"].items():
        require(digest(ROOT / name) == expected, f"input hash mismatch: {name}")
    if check_sources:
        for name, expected in m["source_sha256"].items():
            require(digest(ROOT / name) == expected, f"source hash mismatch: {name}")
    for name, expected in m["output_sha256"].items():
        require(digest(output / name) == expected, f"output hash mismatch: {name}")
    paths = sorted((output / "rows").glob("*.csv.gz"))
    require(len(paths) == 18, "expected 18 streams")
    frames = [pd.read_csv(p) for p in paths]
    rows = pd.concat(frames, ignore_index=True)
    checks = pd.read_csv(output / "checks.csv")
    require(len(checks) == 18 and checks.status.eq("passed").all(), "run checks failed")
    require(len(rows) == checks.diagnostic_rows.sum(), "diagnostic row count mismatch")
    require(
        not rows.duplicated(
            ["rate", "seed", "strategy", "reference_id", "history_policy", "window", "kind"]
        ).any(),
        "duplicate rows",
    )
    anomalies = rows[rows.kind == "anomaly"]
    require(anomalies.pair_count.eq(0).all(), "non-novel anomaly")
    require(anomalies.unseen_endpoint.eq(0).all(), "unseen anomaly node")
    require(rows.burstiness.dropna().between(-1, 1).all(), "burstiness out of bounds")
    require(rows.concentration_1h_24h.between(0, 1).all(), "concentration out of bounds")
    shared = rows.query("kind=='normal_reference' and history_policy=='benign_only'")
    cols = ["rate", "seed", "reference_id", "window"]
    a, b = (
        shared[shared.strategy == s]
        .sort_values(cols)
        .drop(columns=["strategy", "activity_error"])
        .reset_index(drop=True)
        for s in ["random", "activity_matched"]
    )
    pd.testing.assert_frame_equal(a, b)
    with tempfile.TemporaryDirectory(prefix="m3-verify-") as tmp:
        summarize(rows, Path(tmp))
        for name in (
            "distributions.csv",
            "distances.csv",
            "activity_errors.csv",
            "paired_effects.csv",
            "seed_summary.csv",
        ):
            pd.testing.assert_frame_equal(
                pd.read_csv(output / name),
                pd.read_csv(Path(tmp) / name),
                check_exact=False,
                rtol=1e-10,
                atol=1e-12,
            )
    return dict(
        status="passed",
        streams=len(paths),
        anomalies=int(checks.anomalies.sum()),
        diagnostic_rows=len(rows),
        input_files=len(m["input_sha256"]),
        figures=len(list((output / "figures").glob("*.png"))),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=ROOT / "results/m3")
    args = parser.parse_args()
    print(json.dumps(verify(args.output), indent=2))
