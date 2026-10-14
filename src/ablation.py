"""Feature-family ablation using the same estimator/evaluation protocol."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .evaluation import score_binary_model


def _select_columns(X, columns):
    if hasattr(X, "loc"):
        return X.loc[:, columns]
    return np.asarray(X)[:, columns]


def run_feature_group_ablations(
    X_train, y_train, X_test, y_test, groups,
    *, seeds=(42,), max_fpr=0.01, n_estimators=100,
):
    """Train full and leave-one-feature-family-out models.

    `groups` maps family names to dataframe column names or integer column indices.
    Group definitions must be fixed using the training/validation data and documented.
    """
    if not groups:
        raise ValueError("groups must define at least one feature family")
    all_columns = list(X_train.columns) if hasattr(X_train, "columns") else list(range(np.asarray(X_train).shape[1]))
    group_columns = {name: list(columns) for name, columns in groups.items()}
    for name, cols in group_columns.items():
        missing = set(cols) - set(all_columns)
        if missing:
            raise ValueError(f"unknown columns in group {name!r}: {sorted(missing, key=repr)}")
    result = []
    for omitted in [None, *group_columns]:
        excluded = set() if omitted is None else set(group_columns[omitted])
        keep = [col for col in all_columns if col not in excluded]
        if not keep:
            raise ValueError(f"omitting {omitted!r} leaves no features")
        Xtr, Xte = _select_columns(X_train, keep), _select_columns(X_test, keep)
        for seed in seeds:
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
            scores = score_binary_model(model, Xtr, y_train, Xte, y_test, max_fpr=max_fpr)
            result.append({"omitted_group": omitted or "none", "n_features": len(keep), "seed": seed, **scores})
    return result
