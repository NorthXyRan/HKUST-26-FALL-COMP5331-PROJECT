"""Paired three-setting evaluation for random and activity-matched anomalies."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


def _recall_at_fpr(y_true, scores, max_fpr):
    negatives = scores[np.asarray(y_true) == 0]
    positives = np.asarray(y_true) == 1
    if len(negatives) == 0 or not positives.any():
        return float("nan")
    threshold = np.quantile(negatives, 1.0 - max_fpr, method="higher")
    return float(np.mean(scores[positives] >= threshold))


def score_binary_model(model, X_train, y_train, X_test, y_test, *, max_fpr=0.01):
    """Fit an estimator and return the project's primary and supporting metrics."""
    y_test = np.asarray(y_test)
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        raise ValueError("train and test labels must each contain both classes")
    model.fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]
    return {
        "pr_auc": float(average_precision_score(y_test, scores)),
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "recall_at_max_fpr": _recall_at_fpr(y_test, scores, max_fpr),
        "max_fpr": max_fpr,
    }


def evaluate_three_settings(
    X_random_train, y_random_train,
    X_matched_train, y_matched_train,
    X_random_test, y_random_test,
    X_matched_test, y_matched_test,
    *, seeds=(42,), max_fpr=0.01, n_estimators=100,
):
    """Run random->random, random->matched, and matched->matched settings.

    Inputs must be precomputed using identical feature columns and chronological
    train/validation/test splits. Feature/generator tuning must not use the test set.
    """
    result = []
    settings = (
        ("random_to_random", X_random_train, y_random_train, X_random_test, y_random_test),
        ("random_to_matched", X_random_train, y_random_train, X_matched_test, y_matched_test),
        ("matched_to_matched", X_matched_train, y_matched_train, X_matched_test, y_matched_test),
    )
    for setting, Xtr, ytr, Xte, yte in settings:
        for seed in seeds:
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
            scores = score_binary_model(model, Xtr, ytr, Xte, yte, max_fpr=max_fpr)
            result.append({"setting": setting, "seed": seed, **scores})
    return result
