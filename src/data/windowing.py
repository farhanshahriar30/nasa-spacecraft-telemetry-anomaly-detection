# Sliding-window utilities for telemetry sequences.
# This file converts raw multivariate time-series arrays into overlapping windows
# and builds matching index tables so each window can be traced back to its source position.
from __future__ import annotations

import numpy as np
import pandas as pd


# Phase A: Validate the array before windowing
def validate_time_series_array(array: np.ndarray) -> np.ndarray:
    """
    Ensure the telemetry array is a 2D NumPy array shaped as (timesteps, variables).

    Parameters
    ----------
    array : np.ndarray
        Input telemetry array.

    Returns
    -------
    np.ndarray
        Validated 2D array.
    """
    if not isinstance(array, np.ndarray):
        raise TypeError("Input must be a NumPy array.")

    if array.ndim != 2:
        raise ValueError(
            f"Expected a 2D array shaped (timesteps, variables), got shape {array.shape}."
        )

    if array.shape[0] == 0:
        raise ValueError("Input array has zero timesteps.")

    if array.shape[1] == 0:
        raise ValueError("Input array has zero variables.")

    return array


# Phase B: Compute the start indices for sliding windows
def get_window_start_indices(
    n_timesteps: int,
    window_size: int,
    stride: int,
) -> np.ndarray:
    """
    Generate valid start indices for sliding windows.

    Parameters
    ----------
    n_timesteps : int
        Number of timesteps in the sequence.
    window_size : int
        Number of timesteps per window.
    stride : int
        Step size between consecutive windows.

    Returns
    -------
    np.ndarray
        Array of start indices.
    """
    if window_size <= 0:
        raise ValueError("window_size must be a positive integer.")

    if stride <= 0:
        raise ValueError("stride must be a positive integer.")

    if n_timesteps < window_size:
        return np.array([], dtype=int)

    return np.arange(0, n_timesteps - window_size + 1, stride, dtype=int)


# Phase C: Build the actual sliding windows
def create_sliding_windows(
    array: np.ndarray,
    window_size: int,
    stride: int,
) -> np.ndarray:
    """
    Convert a multivariate telemetry array into overlapping sliding windows.

    Parameters
    ----------
    array : np.ndarray
        2D telemetry array shaped (timesteps, variables).
    window_size : int
        Number of timesteps per window.
    stride : int
        Step size between consecutive windows.

    Returns
    -------
    np.ndarray
        Window tensor shaped (n_windows, window_size, n_variables).
    """
    array = validate_time_series_array(array)

    n_timesteps, n_variables = array.shape
    start_indices = get_window_start_indices(
        n_timesteps=n_timesteps,
        window_size=window_size,
        stride=stride,
    )

    if len(start_indices) == 0:
        return np.empty((0, window_size, n_variables), dtype=array.dtype)

    windows = np.stack(
        [array[start : start + window_size] for start in start_indices],
        axis=0,
    )

    return windows


# Phase D: Build a window index table so later stages know where each window came from
def create_window_index_table(
    n_timesteps: int,
    window_size: int,
    stride: int,
    chan_id: str | None = None,
    split: str | None = None,
) -> pd.DataFrame:
    """
    Create a table describing the start and end index of each sliding window.

    Parameters
    ----------
    n_timesteps : int
        Number of timesteps in the source sequence.
    window_size : int
        Number of timesteps per window.
    stride : int
        Step size between consecutive windows.
    chan_id : str | None
        Optional channel identifier.
    split : str | None
        Optional split label such as 'train' or 'test'.

    Returns
    -------
    pd.DataFrame
        Window metadata table.
    """
    start_indices = get_window_start_indices(
        n_timesteps=n_timesteps,
        window_size=window_size,
        stride=stride,
    )

    if len(start_indices) == 0:
        columns = ["window_id", "start_idx", "end_idx"]
        if chan_id is not None:
            columns.insert(0, "chan_id")
        if split is not None:
            insert_at = 1 if chan_id is not None else 0
            columns.insert(insert_at, "split")
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(
        {
            "window_id": np.arange(len(start_indices), dtype=int),
            "start_idx": start_indices,
            "end_idx": start_indices + window_size - 1,
        }
    )

    if chan_id is not None:
        df.insert(0, "chan_id", chan_id)

    if split is not None:
        insert_at = 1 if chan_id is not None else 0
        df.insert(insert_at, "split", split)

    return df


# Phase E: Produce both windows and their index table together for convenience
def window_channel_array(
    array: np.ndarray,
    window_size: int,
    stride: int,
    chan_id: str | None = None,
    split: str | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Create sliding windows and a matching window index table.

    Parameters
    ----------
    array : np.ndarray
        2D telemetry array shaped (timesteps, variables).
    window_size : int
        Number of timesteps per window.
    stride : int
        Step size between consecutive windows.
    chan_id : str | None
        Optional channel identifier.
    split : str | None
        Optional split label.

    Returns
    -------
    tuple[np.ndarray, pd.DataFrame]
        Window tensor and window metadata table.
    """
    array = validate_time_series_array(array)

    windows = create_sliding_windows(
        array=array,
        window_size=window_size,
        stride=stride,
    )

    window_df = create_window_index_table(
        n_timesteps=array.shape[0],
        window_size=window_size,
        stride=stride,
        chan_id=chan_id,
        split=split,
    )

    if len(windows) != len(window_df):
        raise RuntimeError(
            "Window tensor and window index table have inconsistent lengths."
        )

    return windows, window_df
