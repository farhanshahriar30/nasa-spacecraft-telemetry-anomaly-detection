from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import SCALER_NAME, WINDOW_SIZE, WINDOW_STRIDE
from src.data.labeling import label_windows
from src.data.loader import load_channel_pair, load_labels_metadata
from src.data.preprocess import preprocess_channel_pair
from src.data.windowing import window_channel_array
from src.features.feature_engineering import build_window_feature_dataframe


# Phase A: Get one metadata row for a specific channel
def get_channel_metadata_row(
    chan_id: str,
    labels_df: pd.DataFrame | None = None,
) -> pd.Series:
    """
    Fetch the metadata row for a given channel ID.

    Parameters
    ----------
    chan_id : str
        Channel identifier.
    labels_df : pd.DataFrame | None
        Optional preloaded metadata dataframe.

    Returns
    -------
    pd.Series
        Metadata row for the requested channel.
    """
    if labels_df is None:
        labels_df = load_labels_metadata()

    matches = labels_df[labels_df["chan_id"] == chan_id]

    if matches.empty:
        raise ValueError(f"No metadata row found for channel: {chan_id}")

    if len(matches) > 1:
        raise ValueError(f"Multiple metadata rows found for channel: {chan_id}")

    return matches.iloc[0]


# Phase B: Build the unsupervised feature tables
# Train windows come from the train sequence and stay unlabeled.
# Test windows come from the test sequence and receive anomaly labels.
def build_channel_unsupervised_data(
    chan_id: str,
    labels_df: pd.DataFrame | None = None,
    scaler_name: str = SCALER_NAME,
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
) -> dict[str, Any]:
    """
    Build train/test feature data for the unsupervised pipeline.

    Parameters
    ----------
    chan_id : str
        Channel identifier.
    labels_df : pd.DataFrame | None
        Optional preloaded metadata dataframe.
    scaler_name : str
        Scaler type.
    window_size : int
        Sliding window size.
    stride : int
        Sliding window stride.

    Returns
    -------
    dict[str, Any]
        Channel-level processed data for unsupervised experiments.
    """
    if labels_df is None:
        labels_df = load_labels_metadata()

    metadata_row = get_channel_metadata_row(chan_id=chan_id, labels_df=labels_df)
    channel_pair = load_channel_pair(chan_id)

    preprocessed = preprocess_channel_pair(
        train_array=channel_pair["train"],
        test_array=channel_pair["test"],
        scaler_name=scaler_name,
    )

    train_scaled = preprocessed["train_scaled"]
    test_scaled = preprocessed["test_scaled"]

    train_windows, train_window_df = window_channel_array(
        array=train_scaled,
        window_size=window_size,
        stride=stride,
        chan_id=chan_id,
        split="train",
    )

    test_windows, test_window_df = window_channel_array(
        array=test_scaled,
        window_size=window_size,
        stride=stride,
        chan_id=chan_id,
        split="test",
    )

    labeled_test_window_df = label_windows(
        window_df=test_window_df,
        anomaly_intervals=metadata_row["anomaly_sequences"],
    )

    train_feature_df = build_window_feature_dataframe(
        windows=train_windows,
        window_df=train_window_df,
    )

    test_feature_df = build_window_feature_dataframe(
        windows=test_windows,
        window_df=labeled_test_window_df,
    )

    return {
        "chan_id": chan_id,
        "spacecraft": metadata_row["spacecraft"],
        "metadata_row": metadata_row,
        "scaler": preprocessed["scaler"],
        "train_scaled": train_scaled,
        "test_scaled": test_scaled,
        "train_windows": train_windows,
        "test_windows": test_windows,
        "train_window_df": train_window_df,
        "test_window_df": labeled_test_window_df,
        "train_feature_df": train_feature_df,
        "test_feature_df": test_feature_df,
    }


# Phase C: Build the supervised labeled dataset for one channel
# For supervised learning, we use labeled windows from the TEST sequence.
def build_channel_supervised_data(
    chan_id: str,
    labels_df: pd.DataFrame | None = None,
    scaler_name: str = SCALER_NAME,
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
) -> dict[str, Any]:
    """
    Build a labeled feature table for one channel using the test sequence.

    Parameters
    ----------
    chan_id : str
        Channel identifier.
    labels_df : pd.DataFrame | None
        Optional preloaded metadata dataframe.
    scaler_name : str
        Scaler type.
    window_size : int
        Sliding window size.
    stride : int
        Sliding window stride.

    Returns
    -------
    dict[str, Any]
        Channel-level processed data for supervised experiments.
    """
    unsup_data = build_channel_unsupervised_data(
        chan_id=chan_id,
        labels_df=labels_df,
        scaler_name=scaler_name,
        window_size=window_size,
        stride=stride,
    )

    supervised_feature_df = unsup_data["test_feature_df"].copy()

    return {
        "chan_id": unsup_data["chan_id"],
        "spacecraft": unsup_data["spacecraft"],
        "metadata_row": unsup_data["metadata_row"],
        "scaler": unsup_data["scaler"],
        "test_scaled": unsup_data["test_scaled"],
        "test_windows": unsup_data["test_windows"],
        "test_window_df": unsup_data["test_window_df"],
        "supervised_feature_df": supervised_feature_df,
    }


# Phase D: Add a quick channel summary for debugging and inspection
def summarize_channel_processed_data(channel_data: dict[str, Any]) -> dict[str, Any]:
    """
    Produce a compact summary of processed channel data.

    Parameters
    ----------
    channel_data : dict[str, Any]
        Output dictionary from one of the build functions.

    Returns
    -------
    dict[str, Any]
        Summary information.
    """
    summary = {
        "chan_id": channel_data["chan_id"],
        "spacecraft": channel_data["spacecraft"],
    }

    if "train_feature_df" in channel_data:
        summary["n_train_windows"] = len(channel_data["train_feature_df"])

    if "test_feature_df" in channel_data:
        summary["n_test_windows"] = len(channel_data["test_feature_df"])
        if "label" in channel_data["test_feature_df"].columns:
            summary["n_test_anomalous_windows"] = int(
                channel_data["test_feature_df"]["label"].sum()
            )

    if "supervised_feature_df" in channel_data:
        summary["n_supervised_windows"] = len(channel_data["supervised_feature_df"])
        if "label" in channel_data["supervised_feature_df"].columns:
            summary["n_supervised_anomalous_windows"] = int(
                channel_data["supervised_feature_df"]["label"].sum()
            )

    return summary
