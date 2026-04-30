# Evaluation metric utilities for anomaly detection experiments.
# This file converts model scores into binary predictions and computes
# the main classification metrics used throughout the project.
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# Phase A: Convert predicted probabilities into binary labels
def apply_probability_threshold(
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Convert predicted probabilities into binary predictions.
    """
    y_score = np.asarray(y_score, dtype=np.float64)

    if y_score.ndim != 1:
        raise ValueError("y_score must be a 1D array.")

    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be between 0 and 1.")

    return (y_score >= threshold).astype(int)


# Phase B: Compute classification metrics from labels and scores
def compute_binary_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Compute binary classification metrics for anomaly detection.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)

    if y_true.ndim != 1:
        raise ValueError("y_true must be a 1D array.")

    if y_score.ndim != 1:
        raise ValueError("y_score must be a 1D array.")

    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length.")

    y_pred = apply_probability_threshold(y_score=y_score, threshold=threshold)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        pr_auc = average_precision_score(y_true, y_score)
    except ValueError:
        pr_auc = np.nan

    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except ValueError:
        roc_auc = np.nan

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": threshold,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(pr_auc) if not np.isnan(pr_auc) else np.nan,
        "roc_auc": float(roc_auc) if not np.isnan(roc_auc) else np.nan,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "n_rows": int(len(y_true)),
        "n_anomalous": int(y_true.sum()),
    }
