# Window-level labeling utilities for anomaly detection.
# This file converts anomaly intervals from the metadata into binary labels by checking whether each sliding window overlaps a labeled anomaly range.
from __future__ import annotations

from typing import Any

import pandas as pd


# Phase A: Clean and validate anomaly intervals
def normalize_anomaly_intervals(anomaly_intervals: Any) -> list[tuple[int, int]]:
    """
    Convert raw anomaly interval input into a clean list of (start, end) tuples.

    Parameters
    ----------
    anomaly_intervals : Any
        Raw anomaly interval object, usually from the metadata table.

    Returns
    -------
    list[tuple[int, int]]
        Cleaned list of valid anomaly intervals.
    """
    if anomaly_intervals is None:
        return []

    if not isinstance(anomaly_intervals, list):
        return []

    cleaned_intervals: list[tuple[int, int]] = []

    for interval in anomaly_intervals:
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            continue

        start, end = interval

        try:
            start = int(start)
            end = int(end)
        except (TypeError, ValueError):
            continue

        if start > end:
            start, end = end, start

        cleaned_intervals.append((start, end))

    return cleaned_intervals


# Phase B: Check whether one window overlaps one anomaly interval
def intervals_overlap(
    window_start: int,
    window_end: int,
    anomaly_start: int,
    anomaly_end: int,
) -> bool:
    """
    Return True if the window overlaps the anomaly interval at any point.
    """
    return not (window_end < anomaly_start or window_start > anomaly_end)


# Phase C: Find all anomaly intervals overlapped by a given window
def find_overlapping_intervals(
    window_start: int,
    window_end: int,
    anomaly_intervals: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """
    Return all anomaly intervals that overlap the given window.
    """
    overlaps = []

    for anomaly_start, anomaly_end in anomaly_intervals:
        if intervals_overlap(window_start, window_end, anomaly_start, anomaly_end):
            overlaps.append((anomaly_start, anomaly_end))

    return overlaps


# Phase D: Label one window using the any-overlap rule
def label_single_window(
    window_start: int,
    window_end: int,
    anomaly_intervals: list[tuple[int, int]],
) -> dict[str, Any]:
    """
    Label a single window based on overlap with anomaly intervals.

    Returns
    -------
    dict[str, Any]
        Dictionary with binary label and overlap details.
    """
    overlapping_intervals = find_overlapping_intervals(
        window_start=window_start,
        window_end=window_end,
        anomaly_intervals=anomaly_intervals,
    )

    return {
        "label": int(len(overlapping_intervals) > 0),
        "has_overlap": len(overlapping_intervals) > 0,
        "num_overlapping_intervals": len(overlapping_intervals),
        "overlapping_intervals": overlapping_intervals,
    }


# Phase E: Label all windows in a window index table
def label_windows(
    window_df: pd.DataFrame,
    anomaly_intervals: Any,
) -> pd.DataFrame:
    """
    Label all windows in the provided window table using anomaly interval overlap.

    Parameters
    ----------
    window_df : pd.DataFrame
        Must contain 'start_idx' and 'end_idx' columns.
    anomaly_intervals : Any
        Raw anomaly interval object from metadata.

    Returns
    -------
    pd.DataFrame
        Copy of window_df with added label columns.
    """
    required_cols = {"start_idx", "end_idx"}
    missing = required_cols - set(window_df.columns)
    if missing:
        raise ValueError(f"window_df is missing required columns: {missing}")

    anomaly_intervals = normalize_anomaly_intervals(anomaly_intervals)

    labeled_df = window_df.copy()

    label_records = labeled_df.apply(
        lambda row: label_single_window(
            window_start=int(row["start_idx"]),
            window_end=int(row["end_idx"]),
            anomaly_intervals=anomaly_intervals,
        ),
        axis=1,
    )

    label_details_df = pd.DataFrame(label_records.tolist(), index=labeled_df.index)
    labeled_df = pd.concat([labeled_df, label_details_df], axis=1)

    return labeled_df
