from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import RANDOM_STATE
from src.models.tuning import get_feature_columns


# Phase A: Build a baseline Isolation Forest model
def build_isolation_forest_model(
    random_state: int = RANDOM_STATE,
    n_estimators: int = 300,
    max_samples: str | int | float = "auto",
    max_features: float = 1.0,
    bootstrap: bool = False,
    contamination: str | float = "auto",
    n_jobs: int = -1,
) -> IsolationForest:
    """
    Create an Isolation Forest model for unsupervised anomaly detection.
    """
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        bootstrap=bootstrap,
        contamination=contamination,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    return model


# Phase B: Convert a feature dataframe into X using a fixed feature order
def dataframe_to_unsupervised_arrays(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    """
    Convert a feature dataframe into X using the supplied feature column list.
    """
    missing = set(feature_cols) - set(feature_df.columns)
    if missing:
        raise ValueError(f"feature_df is missing required feature columns: {missing}")

    X = feature_df[feature_cols].to_numpy(dtype=np.float64)
    return X


# Phase C: Score anomaly strength so that larger = more anomalous
def compute_isolation_forest_anomaly_scores(
    model: IsolationForest,
    X: np.ndarray,
) -> np.ndarray:
    """
    Compute anomaly scores where larger values indicate stronger anomaly evidence.

    sklearn's decision_function returns higher values for more normal points,
    so we negate it.
    """
    raw_scores = -model.decision_function(X)
    return raw_scores.astype(np.float64, copy=False)


# Phase D: Fit a score normalizer on development scores
def fit_score_normalizer(raw_scores: np.ndarray) -> dict[str, float]:
    """
    Fit a simple min-max normalizer using development raw anomaly scores.
    """
    raw_scores = np.asarray(raw_scores, dtype=np.float64)

    score_min = float(np.min(raw_scores))
    score_max = float(np.max(raw_scores))

    return {
        "score_min": score_min,
        "score_max": score_max,
    }


# Phase E: Apply the score normalizer so thresholds can be tuned on [0, 1]
def apply_score_normalizer(
    raw_scores: np.ndarray,
    score_min: float,
    score_max: float,
) -> np.ndarray:
    """
    Apply min-max score normalization and clip to [0, 1].
    """
    raw_scores = np.asarray(raw_scores, dtype=np.float64)

    if np.isclose(score_max, score_min):
        return np.zeros_like(raw_scores, dtype=np.float64)

    normalized = (raw_scores - score_min) / (score_max - score_min)
    normalized = np.clip(normalized, 0.0, 1.0)

    return normalized.astype(np.float64, copy=False)


# Phase F: Fit on one dataframe and score another
def fit_and_score_isolation_forest(
    train_feature_df: pd.DataFrame,
    eval_feature_df: pd.DataFrame,
    model_params: dict[str, Any] | None = None,
    feature_cols: list[str] | None = None,
) -> dict[str, Any]:
    """
    Fit Isolation Forest on train_feature_df and score eval_feature_df.
    """
    if model_params is None:
        model_params = {}

    if feature_cols is None:
        feature_cols = get_feature_columns(train_feature_df)

    X_train = dataframe_to_unsupervised_arrays(
        feature_df=train_feature_df,
        feature_cols=feature_cols,
    )

    X_eval = dataframe_to_unsupervised_arrays(
        feature_df=eval_feature_df,
        feature_cols=feature_cols,
    )

    model = build_isolation_forest_model(**model_params)
    model.fit(X_train)

    raw_scores = compute_isolation_forest_anomaly_scores(model=model, X=X_eval)

    metadata_cols = [col for col in eval_feature_df.columns if col not in feature_cols]
    score_df = eval_feature_df[metadata_cols].copy().reset_index(drop=True)
    score_df["raw_score"] = raw_scores

    return {
        "model": model,
        "feature_cols": feature_cols,
        "score_df": score_df,
    }


# Phase G: Fit on development train windows and score both dev-test and held-out test-test
def run_isolation_forest_dev_test_pipeline(
    dev_train_df: pd.DataFrame,
    dev_test_df: pd.DataFrame,
    heldout_test_df: pd.DataFrame,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Fit Isolation Forest on development train windows only.
    Then score:
    - development labeled test windows
    - held-out labeled test windows

    This keeps held-out test channels fully unseen.
    """
    if model_params is None:
        model_params = {}

    feature_cols = get_feature_columns(dev_train_df)

    dev_result = fit_and_score_isolation_forest(
        train_feature_df=dev_train_df,
        eval_feature_df=dev_test_df,
        model_params=model_params,
        feature_cols=feature_cols,
    )

    test_result = fit_and_score_isolation_forest(
        train_feature_df=dev_train_df,
        eval_feature_df=heldout_test_df,
        model_params=model_params,
        feature_cols=feature_cols,
    )

    normalizer = fit_score_normalizer(dev_result["score_df"]["raw_score"].to_numpy())

    dev_score_df = dev_result["score_df"].copy()
    test_score_df = test_result["score_df"].copy()

    dev_score_df["score"] = apply_score_normalizer(
        raw_scores=dev_score_df["raw_score"].to_numpy(),
        score_min=normalizer["score_min"],
        score_max=normalizer["score_max"],
    )

    test_score_df["score"] = apply_score_normalizer(
        raw_scores=test_score_df["raw_score"].to_numpy(),
        score_min=normalizer["score_min"],
        score_max=normalizer["score_max"],
    )

    return {
        "model": dev_result["model"],
        "feature_cols": feature_cols,
        "dev_score_df": dev_score_df,
        "test_score_df": test_score_df,
        "score_normalizer": normalizer,
    }
