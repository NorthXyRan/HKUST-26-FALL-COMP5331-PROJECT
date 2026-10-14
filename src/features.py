"""Feature extraction through the pinned, unchanged TGF extractor."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
OFFICIAL = PROJECT / "TGF"
FEATURE_MAIN = OFFICIAL / "3-FeatureGeneration/main.py"
OFFICIAL_COMMIT = "af9d575e07c7bc3fd0b4a0b8d0249e7cdd2eb88b"
DEFAULT_HORIZONS = (5, 50, 100, 1000, 10000, 15000)


def flatten(record, prefix=""):
    result = {}
    for key, value in record.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, name))
        else:
            result[name] = value
    return result


def official_features(rows, horizons):
    """Extract selected H predictors and the notebook's H5 fraud label."""
    horizons = tuple(dict.fromkeys(int(h) for h in horizons))
    if not horizons or any(h <= 0 for h in horizons):
        raise ValueError("at least one positive predictor horizon is required")
    if not FEATURE_MAIN.is_file():
        raise FileNotFoundError(f"missing pinned TGF extractor: {FEATURE_MAIN}")
    extraction_horizons = tuple(sorted(set(horizons) | {5}))
    stream = "".join(f"{t} {u} {v} {label}\n" for t, u, v, label in rows)
    frames = {}
    for horizon in extraction_horizons:
        proc = subprocess.run(
            [sys.executable, str(FEATURE_MAIN), "-H", str(horizon), "-int"],
            cwd=FEATURE_MAIN.parent, input=stream, text=True, capture_output=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        if proc.returncode:
            raise RuntimeError(
                f"official TGF feature extraction failed for H={horizon}:\n{proc.stderr[-4000:]}"
            )
        records = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
        if len(records) != len(rows):
            raise RuntimeError(
                f"TGF returned {len(records)} rows for {len(rows)} input events at H={horizon}"
            )
        frames[horizon] = pd.DataFrame.from_records(
            [flatten(row) for row in records]
        ).add_prefix(f"H{horizon}_")

    combined = pd.concat([frames[h] for h in horizons], axis=1).dropna(how="any")
    label = frames[5]["H5_is_fraud"].reindex(combined.index)
    metadata = []
    for horizon in horizons:
        metadata.extend(
            f"H{horizon}_{suffix}"
            for suffix in ("t", "u", "v", "history_type", "bip", "cost", "is_fraud")
        )
    result = combined.drop(columns=metadata, errors="ignore").copy()
    if label.isna().any():
        raise RuntimeError("H=5 labels do not align with selected horizon features")
    result["H5_is_fraud"] = label.astype(int)
    return result, len(rows) - len(result)
