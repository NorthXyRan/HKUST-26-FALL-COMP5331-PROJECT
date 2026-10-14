"""Fixed classifier protocol and evaluation metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


def recall_at_fpr(y_true, scores, max_fpr=0.01):
    y_true = np.asarray(y_true)
    negatives = np.asarray(scores)[y_true == 0]
    positives = y_true == 1
    if not len(negatives) or not positives.any():
        return float("nan"), float("nan")
    threshold = np.quantile(negatives, 1 - max_fpr, method="higher")
    predicted = np.asarray(scores) >= threshold
    return float(predicted[positives].mean()), float(predicted[~positives].mean())


def fit_and_score(train: pd.DataFrame, test: pd.DataFrame, seed: int):
    label = "H5_is_fraud"
    y_train = train[label].astype(int)
    y_test = test[label].astype(int)
    normal, anomaly = train.loc[y_train == 0], train.loc[y_train == 1]
    if normal.empty or anomaly.empty or y_test.nunique() < 2:
        raise ValueError("chronological split must contain both classes in train and test")
    balanced = pd.concat([normal.sample(n=len(anomaly), random_state=12), anomaly])
    balanced = balanced.sample(frac=1, random_state=42)
    model = RandomForestClassifier(random_state=seed)
    model.fit(balanced.drop(columns=label), balanced[label].astype(int))
    scores = model.predict_proba(test.drop(columns=label))[:, 1]
    recall, actual_fpr = recall_at_fpr(y_test, scores)
    return {
        "pr_auc": float(average_precision_score(y_test, scores)),
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "recall_at_1pct_fpr": recall,
        "actual_fpr_at_threshold": actual_fpr,
        "train_rows_before_balance": int(len(train)),
        "train_normals_before_balance": int((y_train == 0).sum()),
        "train_anomalies": int((y_train == 1).sum()),
        "test_rows": int(len(test)),
        "test_anomalies": int((y_test == 1).sum()),
        "feature_count": int(train.drop(columns=label).shape[1]),
    }
