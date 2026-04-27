from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.config import RANDOM_STATE
from src.evaluation.metrics import compute_binary_classification_metrics
from src.models.tuning import (
    create_group_kfold_splits,
    get_feature_columns,
)


# Phase A: Compute a safe positive-class weight from the training labels
def compute_scale_pos_weight(y: np.ndarray) -> float:
    """
    Compute scale_pos_weight = n_negative / n_positive.
    """
    y = np.asarray(y, dtype=np.int64)

    n_positive = int(np.sum(y == 1))
    n_negative = int(np.sum(y == 0))

    if n_positive == 0:
        return 1.0

    return float(n_negative / n_positive)


# Phase B: Build an XGBoost classifier with sensible baseline settings
def build_xgboost_model(
    random_state: int = RANDOM_STATE,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    min_child_weight: float = 1.0,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_lambda: float = 1.0,
    n_jobs: int = -1,
    scale_pos_weight: float = 1.0,
) -> XGBClassifier:
    """
    Create a baseline XGBoost classifier for binary anomaly detection.
    """
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        n_jobs=n_jobs,
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
    )
    return model


# Phase C: Convert a labeled dataframe into X and y using a fixed feature order
def dataframe_to_model_arrays(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "label",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a labeled dataframe into X and y using the supplied feature column list.
    """
    missing = set(feature_cols) - set(feature_df.columns)
    if missing:
        raise ValueError(f"feature_df is missing required feature columns: {missing}")

    if label_col not in feature_df.columns:
        raise ValueError(f"feature_df must contain '{label_col}'.")

    X = feature_df[feature_cols].to_numpy(dtype=np.float64)
    y = feature_df[label_col].to_numpy(dtype=np.int64)

    return X, y


# Phase D: Run grouped cross-validation on the development dataframe
def run_xgboost_grouped_cv(
    dev_df: pd.DataFrame,
    n_splits: int = 4,
    threshold: float = 0.5,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Train and evaluate XGBoost across GroupKFold splits.
    """
    if model_params is None:
        model_params = {}

    cv_data = create_group_kfold_splits(
        feature_df=dev_df,
        n_splits=n_splits,
    )

    X = cv_data["X"]
    y = cv_data["y"]
    groups = cv_data["groups"]
    feature_cols = cv_data["feature_cols"]
    fold_indices = cv_data["fold_indices"]
    metadata_df = cv_data["metadata_df"]

    fold_records = []
    oof_prediction_dfs = []

    for fold_id, (train_idx, val_idx) in enumerate(fold_indices):
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]

        train_groups = groups[train_idx]
        val_groups = groups[val_idx]

        fold_params = dict(model_params)

        if "scale_pos_weight" not in fold_params:
            fold_params["scale_pos_weight"] = compute_scale_pos_weight(y_train)

        model = build_xgboost_model(**fold_params)
        model.fit(X_train, y_train)

        y_val_score = model.predict_proba(X_val)[:, 1]
        metrics = compute_binary_classification_metrics(
            y_true=y_val,
            y_score=y_val_score,
            threshold=threshold,
        )

        metrics.update(
            {
                "fold_id": fold_id,
                "n_train_rows": int(len(train_idx)),
                "n_val_rows": int(len(val_idx)),
                "n_train_channels": int(len(set(train_groups))),
                "n_val_channels": int(len(set(val_groups))),
                "scale_pos_weight": float(fold_params["scale_pos_weight"]),
            }
        )

        fold_records.append(metrics)

        fold_oof_df = metadata_df.iloc[val_idx].copy()
        fold_oof_df["dev_row_idx"] = val_idx
        fold_oof_df["y_true"] = y_val
        fold_oof_df["y_score"] = y_val_score
        fold_oof_df["fold_id"] = fold_id
        oof_prediction_dfs.append(fold_oof_df)

    fold_metrics_df = pd.DataFrame(fold_records)
    oof_prediction_df = pd.concat(oof_prediction_dfs, axis=0, ignore_index=True)
    oof_prediction_df = oof_prediction_df.sort_values("dev_row_idx").reset_index(
        drop=True
    )

    mean_metrics = (
        fold_metrics_df[
            [
                "precision",
                "recall",
                "f1",
                "pr_auc",
                "roc_auc",
            ]
        ]
        .mean()
        .to_dict()
    )

    std_metrics = (
        fold_metrics_df[
            [
                "precision",
                "recall",
                "f1",
                "pr_auc",
                "roc_auc",
            ]
        ]
        .std()
        .to_dict()
    )

    return {
        "fold_metrics_df": fold_metrics_df,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "feature_cols": feature_cols,
        "oof_prediction_df": oof_prediction_df,
    }


# Phase E: Train on full dev and evaluate on held-out outer test
def train_xgboost_final_model(
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    threshold: float = 0.5,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Fit XGBoost on the full development set and evaluate on the held-out test set.
    """
    if model_params is None:
        model_params = {}

    feature_cols = get_feature_columns(dev_df)

    X_dev, y_dev = dataframe_to_model_arrays(
        feature_df=dev_df,
        feature_cols=feature_cols,
    )

    X_test, y_test = dataframe_to_model_arrays(
        feature_df=test_df,
        feature_cols=feature_cols,
    )

    final_params = dict(model_params)
    if "scale_pos_weight" not in final_params:
        final_params["scale_pos_weight"] = compute_scale_pos_weight(y_dev)

    model = build_xgboost_model(**final_params)
    model.fit(X_dev, y_dev)

    y_test_score = model.predict_proba(X_test)[:, 1]
    test_metrics = compute_binary_classification_metrics(
        y_true=y_test,
        y_score=y_test_score,
        threshold=threshold,
    )

    metadata_cols = [col for col in test_df.columns if col not in feature_cols]
    test_prediction_df = test_df[metadata_cols].copy().reset_index(drop=True)
    test_prediction_df["y_true"] = y_test
    test_prediction_df["y_score"] = y_test_score

    return {
        "model": model,
        "feature_cols": feature_cols,
        "test_metrics": test_metrics,
        "y_test_score": y_test_score,
        "test_prediction_df": test_prediction_df,
        "scale_pos_weight": float(final_params["scale_pos_weight"]),
    }


# Phase F: Create a tidy feature importance table
def get_xgboost_feature_importance(
    model: XGBClassifier,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Return XGBoost feature importances as a sorted dataframe.
    """
    importance_df = (
        pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return importance_df
