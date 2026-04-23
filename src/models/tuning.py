from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


# Phase A: Define which columns are metadata rather than model features
DEFAULT_NON_FEATURE_COLUMNS = {
    "chan_id",
    "spacecraft",
    "split",
    "window_id",
    "start_idx",
    "end_idx",
    "label",
    "has_overlap",
    "num_overlapping_intervals",
    "overlapping_intervals",
}


# Phase B: Get the feature column names from a dataframe
def get_feature_columns(
    feature_df: pd.DataFrame,
    non_feature_columns: set[str] | None = None,
) -> list[str]:
    """
    Return the columns that should be used as model features.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Input dataframe containing metadata, labels, and engineered features.
    non_feature_columns : set[str] | None
        Optional custom set of columns to exclude from the model feature matrix.

    Returns
    -------
    list[str]
        Ordered list of feature column names.
    """
    if non_feature_columns is None:
        non_feature_columns = DEFAULT_NON_FEATURE_COLUMNS

    feature_cols = [col for col in feature_df.columns if col not in non_feature_columns]

    if not feature_cols:
        raise ValueError("No feature columns were found in feature_df.")

    return feature_cols


# Phase C: Prepare X, y, groups, and feature column names for supervised modeling
def prepare_supervised_model_data(
    feature_df: pd.DataFrame,
    label_col: str = "label",
    group_col: str = "chan_id",
    non_feature_columns: set[str] | None = None,
) -> dict[str, Any]:
    """
    Convert a labeled feature dataframe into model-ready arrays and metadata.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Labeled feature dataframe.
    label_col : str
        Name of the binary label column.
    group_col : str
        Name of the grouping column, usually channel ID.
    non_feature_columns : set[str] | None
        Optional columns to exclude from the feature matrix.

    Returns
    -------
    dict[str, Any]
        X, y, groups, feature columns, and metadata dataframe.
    """
    if label_col not in feature_df.columns:
        raise ValueError(f"feature_df must contain '{label_col}'.")

    if group_col not in feature_df.columns:
        raise ValueError(f"feature_df must contain '{group_col}'.")

    df = feature_df.copy().reset_index(drop=True)
    feature_cols = get_feature_columns(df, non_feature_columns=non_feature_columns)

    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df[label_col].to_numpy(dtype=np.int64)
    groups = df[group_col].astype(str).to_numpy()

    metadata_cols = [col for col in df.columns if col not in feature_cols]
    metadata_df = df[metadata_cols].copy()

    return {
        "X": X,
        "y": y,
        "groups": groups,
        "feature_cols": feature_cols,
        "metadata_df": metadata_df,
    }


# Phase D: Build grouped K-fold splits so no channel appears in both train and validation
def create_group_kfold_splits(
    feature_df: pd.DataFrame,
    n_splits: int = 4,
    label_col: str = "label",
    group_col: str = "chan_id",
) -> dict[str, Any]:
    """
    Create grouped cross-validation folds using GroupKFold.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Development dataframe used for inner tuning.
    n_splits : int
        Number of folds.
    label_col : str
        Name of the binary label column.
    group_col : str
        Name of the grouping column.

    Returns
    -------
    dict[str, Any]
        Prepared model arrays plus fold index pairs.
    """
    prepared = prepare_supervised_model_data(
        feature_df=feature_df,
        label_col=label_col,
        group_col=group_col,
    )

    X = prepared["X"]
    y = prepared["y"]
    groups = prepared["groups"]

    unique_groups = np.unique(groups)
    if len(unique_groups) < n_splits:
        raise ValueError(
            f"Number of unique groups ({len(unique_groups)}) is smaller than n_splits ({n_splits})."
        )

    gkf = GroupKFold(n_splits=n_splits)
    fold_indices = list(gkf.split(X=X, y=y, groups=groups))

    return {
        **prepared,
        "fold_indices": fold_indices,
    }


# Phase E: Summarize the grouped folds for quick inspection
def summarize_group_kfold_splits(
    feature_df: pd.DataFrame,
    fold_indices: list[tuple[np.ndarray, np.ndarray]],
    label_col: str = "label",
    group_col: str = "chan_id",
) -> pd.DataFrame:
    """
    Summarize each grouped CV fold in terms of row counts, class balance,
    and unique channel counts.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Development dataframe used for inner tuning.
    fold_indices : list[tuple[np.ndarray, np.ndarray]]
        Output folds from GroupKFold.
    label_col : str
        Name of the binary label column.
    group_col : str
        Name of the grouping column.

    Returns
    -------
    pd.DataFrame
        Fold summary table.
    """
    records = []

    for fold_id, (train_idx, val_idx) in enumerate(fold_indices):
        train_df = feature_df.iloc[train_idx]
        val_df = feature_df.iloc[val_idx]

        train_groups = set(train_df[group_col].astype(str).unique())
        val_groups = set(val_df[group_col].astype(str).unique())

        records.append(
            {
                "fold_id": fold_id,
                "n_train_rows": len(train_df),
                "n_val_rows": len(val_df),
                "n_train_channels": len(train_groups),
                "n_val_channels": len(val_groups),
                "train_anomalous_rows": int(train_df[label_col].sum()),
                "val_anomalous_rows": int(val_df[label_col].sum()),
                "train_anomaly_ratio": float(train_df[label_col].mean()),
                "val_anomaly_ratio": float(val_df[label_col].mean()),
                "channel_overlap_count": len(train_groups & val_groups),
            }
        )

    return pd.DataFrame(records)


# Phase F: Extract one specific fold's train/validation data
def get_fold_data(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    fold_indices: list[tuple[np.ndarray, np.ndarray]],
    fold_id: int,
) -> dict[str, Any]:
    """
    Return train/validation arrays for one selected fold.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Label vector.
    groups : np.ndarray
        Group labels.
    fold_indices : list[tuple[np.ndarray, np.ndarray]]
        GroupKFold indices.
    fold_id : int
        Fold number to extract.

    Returns
    -------
    dict[str, Any]
        Arrays for one fold.
    """
    train_idx, val_idx = fold_indices[fold_id]

    return {
        "X_train": X[train_idx],
        "y_train": y[train_idx],
        "groups_train": groups[train_idx],
        "X_val": X[val_idx],
        "y_val": y[val_idx],
        "groups_val": groups[val_idx],
        "train_idx": train_idx,
        "val_idx": val_idx,
    }
