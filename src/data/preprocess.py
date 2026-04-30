# This file handles preprocessing for raw telemetry arrays.
# It contains helper functions for fitting and applying channel-wise scaling
# so train and test sequences are transformed consistently without data leakage.

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.data.windowing import validate_time_series_array


ScalerName = Literal["standard", "minmax", "robust"]


# Phase A: Build the requested scaler object
def build_scaler(scaler_name: ScalerName = "standard"):
    """
    Create a scaler instance based on the requested name.

    Parameters
    ----------
    scaler_name : {"standard", "minmax", "robust"}
        Name of the scaler to build.

    Returns
    -------
    sklearn scaler
        Unfitted scaler object.
    """
    scaler_name = scaler_name.lower().strip()

    if scaler_name == "standard":
        return StandardScaler()
    if scaler_name == "minmax":
        return MinMaxScaler()
    if scaler_name == "robust":
        return RobustScaler()

    raise ValueError(
        "Unsupported scaler_name. Choose from: 'standard', 'minmax', 'robust'."
    )


# Phase B: Fit a scaler using only the training array
def fit_channel_scaler(
    train_array: np.ndarray,
    scaler_name: ScalerName = "standard",
):
    """
    Fit a scaler on the training telemetry array only.

    Parameters
    ----------
    train_array : np.ndarray
        Training array shaped (timesteps, variables).
    scaler_name : {"standard", "minmax", "robust"}
        Type of scaler to use.

    Returns
    -------
    sklearn scaler
        Fitted scaler object.
    """
    train_array = validate_time_series_array(train_array)

    scaler = build_scaler(scaler_name=scaler_name)
    scaler.fit(train_array)

    return scaler


# Phase C: Apply an already fitted scaler to a telemetry array
def transform_channel_array(
    array: np.ndarray,
    scaler,
) -> np.ndarray:
    """
    Transform a telemetry array using a fitted scaler.

    Parameters
    ----------
    array : np.ndarray
        Array shaped (timesteps, variables).
    scaler : sklearn scaler
        Previously fitted scaler.

    Returns
    -------
    np.ndarray
        Scaled array with the same shape as input.
    """
    array = validate_time_series_array(array)

    transformed = scaler.transform(array)
    return transformed.astype(np.float64, copy=False)


# Phase D: Fit on train and transform both train and test
def preprocess_channel_pair(
    train_array: np.ndarray,
    test_array: np.ndarray,
    scaler_name: ScalerName = "standard",
) -> dict[str, np.ndarray]:
    """
    Fit a scaler on the training array and apply it to both train and test.

    Parameters
    ----------
    train_array : np.ndarray
        Training array shaped (timesteps, variables).
    test_array : np.ndarray
        Test array shaped (timesteps, variables).
    scaler_name : {"standard", "minmax", "robust"}
        Type of scaler to use.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing scaled train/test arrays and the fitted scaler.
    """
    train_array = validate_time_series_array(train_array)
    test_array = validate_time_series_array(test_array)

    if train_array.shape[1] != test_array.shape[1]:
        raise ValueError(
            "Train and test arrays must have the same number of variables."
        )

    scaler = fit_channel_scaler(
        train_array=train_array,
        scaler_name=scaler_name,
    )

    train_scaled = transform_channel_array(train_array, scaler)
    test_scaled = transform_channel_array(test_array, scaler)

    return {
        "train_scaled": train_scaled,
        "test_scaled": test_scaled,
        "scaler": scaler,
    }
