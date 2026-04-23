from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import SCALER_NAME, WINDOW_SIZE, WINDOW_STRIDE
from src.data.loader import load_labels_metadata
from src.data.pipeline import (
    build_channel_supervised_data,
    build_channel_unsupervised_data,
)


# Phase A: Get an ordered list of channel IDs, optionally filtered by spacecraft
def get_channel_ids(
    labels_df: pd.DataFrame | None = None,
    spacecraft: str | None = None,
) -> list[str]:
    """
    Return channel IDs, optionally filtered to one spacecraft.

    Parameters
    ----------
    labels_df : pd.DataFrame | None
        Optional preloaded metadata dataframe.
    spacecraft : str | None
        Optional spacecraft filter, e.g. "SMAP" or "MSL".

    Returns
    -------
    list[str]
        Sorted list of channel IDs.
    """
    if labels_df is None:
        labels_df = load_labels_metadata()

    df = labels_df.copy()

    if spacecraft is not None:
        spacecraft = spacecraft.strip().upper()
        df = df[df["spacecraft"].str.upper() == spacecraft]

    return sorted(df["chan_id"].tolist())


# Phase B: Add spacecraft as a metadata column to a feature dataframe
def attach_spacecraft_column(
    feature_df: pd.DataFrame,
    spacecraft: str,
) -> pd.DataFrame:
    """
    Add spacecraft as a column near the front of a feature dataframe.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Feature dataframe that already includes chan_id.
    spacecraft : str
        Spacecraft name.

    Returns
    -------
    pd.DataFrame
        Updated dataframe.
    """
    df = feature_df.copy()

    if "spacecraft" not in df.columns:
        insert_at = 1 if "chan_id" in df.columns else 0
        df.insert(insert_at, "spacecraft", spacecraft)

    return df


# Phase C: Build a combined supervised dataset across many channels
def build_supervised_dataset(
    chan_ids: list[str] | None = None,
    labels_df: pd.DataFrame | None = None,
    spacecraft: str | None = None,
    scaler_name: str = SCALER_NAME,
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
) -> dict[str, Any]:
    """
    Build one combined supervised dataset from labeled test windows
    across multiple channels.

    Parameters
    ----------
    chan_ids : list[str] | None
        Optional explicit list of channels to process.
    labels_df : pd.DataFrame | None
        Optional preloaded metadata dataframe.
    spacecraft : str | None
        Optional spacecraft filter.
    scaler_name : str
        Scaler name.
    window_size : int
        Sliding window size.
    stride : int
        Sliding window stride.

    Returns
    -------
    dict[str, Any]
        Combined supervised dataset and per-channel summary table.
    """
    if labels_df is None:
        labels_df = load_labels_metadata()

    if chan_ids is None:
        chan_ids = get_channel_ids(labels_df=labels_df, spacecraft=spacecraft)

    all_feature_dfs: list[pd.DataFrame] = []
    summary_records: list[dict[str, Any]] = []

    for chan_id in chan_ids:
        channel_data = build_channel_supervised_data(
            chan_id=chan_id,
            labels_df=labels_df,
            scaler_name=scaler_name,
            window_size=window_size,
            stride=stride,
        )

        feature_df = attach_spacecraft_column(
            feature_df=channel_data["supervised_feature_df"],
            spacecraft=channel_data["spacecraft"],
        )

        all_feature_dfs.append(feature_df)

        n_windows = len(feature_df)
        n_anomalous = int(feature_df["label"].sum())
        anomaly_ratio = n_anomalous / n_windows if n_windows > 0 else 0.0

        summary_records.append(
            {
                "chan_id": channel_data["chan_id"],
                "spacecraft": channel_data["spacecraft"],
                "n_windows": n_windows,
                "n_anomalous_windows": n_anomalous,
                "anomaly_ratio": anomaly_ratio,
            }
        )

    combined_feature_df = pd.concat(all_feature_dfs, axis=0, ignore_index=True)
    channel_summary_df = (
        pd.DataFrame(summary_records)
        .sort_values(["spacecraft", "chan_id"])
        .reset_index(drop=True)
    )

    return {
        "feature_df": combined_feature_df,
        "channel_summary_df": channel_summary_df,
        "chan_ids": chan_ids,
    }


# Phase D: Build combined unsupervised train/test datasets across many channels
def build_unsupervised_dataset(
    chan_ids: list[str] | None = None,
    labels_df: pd.DataFrame | None = None,
    spacecraft: str | None = None,
    scaler_name: str = SCALER_NAME,
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
) -> dict[str, Any]:
    """
    Build combined train/test feature datasets across multiple channels
    for the unsupervised pipeline.

    Parameters
    ----------
    chan_ids : list[str] | None
        Optional explicit list of channels to process.
    labels_df : pd.DataFrame | None
        Optional preloaded metadata dataframe.
    spacecraft : str | None
        Optional spacecraft filter.
    scaler_name : str
        Scaler name.
    window_size : int
        Sliding window size.
    stride : int
        Sliding window stride.

    Returns
    -------
    dict[str, Any]
        Combined unsupervised datasets and per-channel summary table.
    """
    if labels_df is None:
        labels_df = load_labels_metadata()

    if chan_ids is None:
        chan_ids = get_channel_ids(labels_df=labels_df, spacecraft=spacecraft)

    train_feature_dfs: list[pd.DataFrame] = []
    test_feature_dfs: list[pd.DataFrame] = []
    summary_records: list[dict[str, Any]] = []

    for chan_id in chan_ids:
        channel_data = build_channel_unsupervised_data(
            chan_id=chan_id,
            labels_df=labels_df,
            scaler_name=scaler_name,
            window_size=window_size,
            stride=stride,
        )

        train_feature_df = attach_spacecraft_column(
            feature_df=channel_data["train_feature_df"],
            spacecraft=channel_data["spacecraft"],
        )

        test_feature_df = attach_spacecraft_column(
            feature_df=channel_data["test_feature_df"],
            spacecraft=channel_data["spacecraft"],
        )

        train_feature_dfs.append(train_feature_df)
        test_feature_dfs.append(test_feature_df)

        n_train_windows = len(train_feature_df)
        n_test_windows = len(test_feature_df)
        n_test_anomalous = int(test_feature_df["label"].sum())
        anomaly_ratio = n_test_anomalous / n_test_windows if n_test_windows > 0 else 0.0

        summary_records.append(
            {
                "chan_id": channel_data["chan_id"],
                "spacecraft": channel_data["spacecraft"],
                "n_train_windows": n_train_windows,
                "n_test_windows": n_test_windows,
                "n_test_anomalous_windows": n_test_anomalous,
                "test_anomaly_ratio": anomaly_ratio,
            }
        )

    combined_train_feature_df = pd.concat(train_feature_dfs, axis=0, ignore_index=True)
    combined_test_feature_df = pd.concat(test_feature_dfs, axis=0, ignore_index=True)

    channel_summary_df = (
        pd.DataFrame(summary_records)
        .sort_values(["spacecraft", "chan_id"])
        .reset_index(drop=True)
    )

    return {
        "train_feature_df": combined_train_feature_df,
        "test_feature_df": combined_test_feature_df,
        "channel_summary_df": channel_summary_df,
        "chan_ids": chan_ids,
    }
