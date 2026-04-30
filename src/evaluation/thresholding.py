# Threshold tuning utilities for anomaly detection scores.
# This file evaluates model outputs across multiple thresholds and selects
# the operating point that gives the best development-set performance.
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_binary_classification_metrics


# Phase A: Build a threshold grid for tuning
def build_threshold_grid(
    start: float = 0.05,
    stop: float = 0.95,
    step: float = 0.05,
) -> np.ndarray:
    """
    Create a grid of probability thresholds.

    Parameters
    ----------
    start : float
        Starting threshold.
    stop : float
        Ending threshold.
    step : float
        Threshold increment.

    Returns
    -------
    np.ndarray
        1D array of thresholds.
    """
    thresholds = np.arange(start, stop + 1e-12, step, dtype=np.float64)
    thresholds = np.round(thresholds, 10)
    thresholds = thresholds[(thresholds > 0.0) & (thresholds < 1.0)]

    if len(thresholds) == 0:
        raise ValueError("Threshold grid is empty. Check start/stop/step values.")

    return thresholds


# Phase B: Evaluate a full threshold grid on prediction scores
def evaluate_threshold_grid(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Evaluate classification metrics across many probability thresholds.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_score : np.ndarray
        Predicted probabilities or anomaly scores in [0, 1].
    thresholds : np.ndarray | None
        Threshold grid. If None, a default grid is used.

    Returns
    -------
    pd.DataFrame
        Metrics per threshold.
    """
    if thresholds is None:
        thresholds = build_threshold_grid()

    records = []

    for threshold in thresholds:
        metrics = compute_binary_classification_metrics(
            y_true=y_true,
            y_score=y_score,
            threshold=float(threshold),
        )
        records.append(metrics)

    threshold_metrics_df = (
        pd.DataFrame(records).sort_values("threshold").reset_index(drop=True)
    )

    return threshold_metrics_df


# Phase C: Select the best threshold using a target metric
def select_best_threshold(
    threshold_metrics_df: pd.DataFrame,
    metric: str = "f1",
) -> dict[str, Any]:
    """
    Select the best threshold based on a target metric.

    Tiebreak order:
    1. target metric descending
    2. precision descending
    3. recall descending
    4. threshold ascending

    Parameters
    ----------
    threshold_metrics_df : pd.DataFrame
        Output from evaluate_threshold_grid.
    metric : str
        Metric to maximize, e.g. "f1", "precision", "recall".

    Returns
    -------
    dict[str, Any]
        Best threshold row as a dictionary.
    """
    required_cols = {"threshold", "precision", "recall", "f1", metric}
    missing = required_cols - set(threshold_metrics_df.columns)
    if missing:
        raise ValueError(f"threshold_metrics_df is missing required columns: {missing}")

    ranked_df = threshold_metrics_df.sort_values(
        by=[metric, "precision", "recall", "threshold"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    return ranked_df.iloc[0].to_dict()


# Phase D: Evaluate one chosen threshold
def evaluate_scores_with_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """
    Evaluate one chosen threshold.
    """
    return compute_binary_classification_metrics(
        y_true=y_true,
        y_score=y_score,
        threshold=threshold,
    )


# Phase E: Build a tidy comparison table across tuned models
def build_model_comparison_table(
    comparison_records: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Build a tidy dataframe summarizing tuned model comparison results.

    Parameters
    ----------
    comparison_records : list[dict[str, Any]]
        One record per model.

    Returns
    -------
    pd.DataFrame
        Comparison table.
    """
    comparison_df = pd.DataFrame(comparison_records)

    preferred_order = [
        "model",
        "dev_selection_metric",
        "dev_best_threshold",
        "dev_precision",
        "dev_recall",
        "dev_f1",
        "dev_pr_auc",
        "dev_roc_auc",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_pr_auc",
        "test_roc_auc",
        "test_tp",
        "test_fp",
        "test_tn",
        "test_fn",
    ]

    ordered_cols = [col for col in preferred_order if col in comparison_df.columns]
    remaining_cols = [col for col in comparison_df.columns if col not in ordered_cols]

    return comparison_df[ordered_cols + remaining_cols]
