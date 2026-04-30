# This file builds full datasets across multiple telemetry channels.
# It loops through selected channel IDs and combines their processed outputs
# into unified supervised and unsupervised dataframes for model training and evaluation.

from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import SCALER_NAME, WINDOW_SIZE, WINDOW_STRIDE
from src.data.loader import load_labels_metadata
from src.data.pipeline import (
    build_channel_supervised_data,
    build_channel_unsupervised_data,
)


# Phase A: Restrict metadata to one spacecraft when requested
def filter_labels_df_by_spacecraft(
    labels_df: pd.DataFrame,
    spacecraft: str | None = None,
) -> pd.DataFrame:
    """
    Return a metadata dataframe optionally filtered to one spacecraft.
    """
    df = labels_df.copy()

    if spacecraft is not None:
        spacecraft = spacecraft.strip().upper()
        df = df[df["spacecraft"].str.upper() == spacecraft]

    return df.reset_index(drop=True)


# Phase B: Get an ordered list of channel IDs, optionally filtered by spacecraft
def get_channel_ids(
    labels_df: pd.DataFrame | None = None,
    spacecraft: str | None = None,
) -> list[str]:
    """
    Return unique channel IDs, optionally filtered to one spacecraft.
    """
    if labels_df is None:
        labels_df = load_labels_metadata()

    filtered_df = filter_labels_df_by_spacecraft(
        labels_df=labels_df,
        spacecraft=spacecraft,
    )

    return sorted(filtered_df["chan_id"].unique().tolist())


# Phase C: Add spacecraft as a metadata column to a feature dataframe
def attach_spacecraft_column(
    feature_df: pd.DataFrame,
    spacecraft: str,
) -> pd.DataFrame:
    """
    Add spacecraft as a column near the front of a feature dataframe.
    """
    df = feature_df.copy()

    if "spacecraft" not in df.columns:
        insert_at = 1 if "chan_id" in df.columns else 0
        df.insert(insert_at, "spacecraft", spacecraft)

    return df


# Phase D: Build a combined supervised dataset across many channels
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
    """
    if labels_df is None:
        labels_df = load_labels_metadata()

    working_labels_df = filter_labels_df_by_spacecraft(
        labels_df=labels_df,
        spacecraft=spacecraft,
    )

    if chan_ids is None:
        chan_ids = sorted(working_labels_df["chan_id"].unique().tolist())

    all_feature_dfs: list[pd.DataFrame] = []
    summary_records: list[dict[str, Any]] = []

    for chan_id in chan_ids:
        channel_data = build_channel_supervised_data(
            chan_id=chan_id,
            labels_df=working_labels_df,
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


# Phase E: Build combined unsupervised train/test datasets across many channels
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
    """
    if labels_df is None:
        labels_df = load_labels_metadata()

    working_labels_df = filter_labels_df_by_spacecraft(
        labels_df=labels_df,
        spacecraft=spacecraft,
    )

    if chan_ids is None:
        chan_ids = sorted(working_labels_df["chan_id"].unique().tolist())

    train_feature_dfs: list[pd.DataFrame] = []
    test_feature_dfs: list[pd.DataFrame] = []
    summary_records: list[dict[str, Any]] = []

    for chan_id in chan_ids:
        channel_data = build_channel_unsupervised_data(
            chan_id=chan_id,
            labels_df=working_labels_df,
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
