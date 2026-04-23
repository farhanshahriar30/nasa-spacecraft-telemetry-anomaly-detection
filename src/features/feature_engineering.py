from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


# Phase A: Validate that the input is a 3D window tensor
def validate_window_tensor(windows: np.ndarray) -> np.ndarray:
    """
    Ensure the input is a 3D NumPy array shaped as
    (n_windows, window_size, n_variables).

    Parameters
    ----------
    windows : np.ndarray
        Window tensor.

    Returns
    -------
    np.ndarray
        Validated window tensor.
    """
    if not isinstance(windows, np.ndarray):
        raise TypeError("windows must be a NumPy array.")

    if windows.ndim != 3:
        raise ValueError(
            f"Expected a 3D array shaped (n_windows, window_size, n_variables), "
            f"got shape {windows.shape}."
        )

    if windows.shape[0] == 0:
        raise ValueError("Window tensor contains zero windows.")

    if windows.shape[1] == 0:
        raise ValueError("Window tensor has window_size = 0.")

    if windows.shape[2] == 0:
        raise ValueError("Window tensor has zero variables.")

    return windows


# Phase B: Create consistent feature names for all variables
def generate_feature_names(n_variables: int) -> list[str]:
    """
    Generate flat feature names in the same order used during extraction.

    Parameters
    ----------
    n_variables : int
        Number of variables in each window.

    Returns
    -------
    list[str]
        Ordered list of feature names.
    """
    stat_names = [
        "mean",
        "std",
        "var",
        "min",
        "max",
        "median",
        "slope",
        "rms",
        "mean_abs_diff",
        "std_diff",
        "skewness",
        "kurtosis",
    ]

    feature_names = []
    for stat_name in stat_names:
        for var_idx in range(n_variables):
            feature_names.append(f"{stat_name}_v{var_idx}")

    return feature_names


# Phase C: Compute slope for each variable within one window
def compute_window_slopes(window: np.ndarray) -> np.ndarray:
    """
    Compute a simple linear slope per variable across time within one window.

    Parameters
    ----------
    window : np.ndarray
        2D array shaped (window_size, n_variables).

    Returns
    -------
    np.ndarray
        One slope value per variable.
    """
    n_timesteps = window.shape[0]

    x = np.arange(n_timesteps, dtype=np.float64)
    x_centered = x - x.mean()
    denom = np.sum(x_centered**2)

    if denom == 0:
        return np.zeros(window.shape[1], dtype=np.float64)

    y_centered = window - window.mean(axis=0, keepdims=True)
    slopes = (x_centered[:, None] * y_centered).sum(axis=0) / denom

    return slopes.astype(np.float64, copy=False)


# Phase D: Extract engineered features from one multivariate window
def extract_features_from_single_window(window: np.ndarray) -> np.ndarray:
    """
    Extract a flat feature vector from one window shaped
    (window_size, n_variables).

    Parameters
    ----------
    window : np.ndarray
        One multivariate window.

    Returns
    -------
    np.ndarray
        Flat feature vector.
    """
    if not isinstance(window, np.ndarray):
        raise TypeError("window must be a NumPy array.")

    if window.ndim != 2:
        raise ValueError(
            f"Expected window shape (window_size, n_variables), got {window.shape}."
        )

    window = window.astype(np.float64, copy=False)

    means = np.mean(window, axis=0)
    stds = np.std(window, axis=0)
    vars_ = np.var(window, axis=0)
    mins = np.min(window, axis=0)
    maxs = np.max(window, axis=0)
    medians = np.median(window, axis=0)
    slopes = compute_window_slopes(window)
    rms = np.sqrt(np.mean(window**2, axis=0))

    if window.shape[0] > 1:
        diffs = np.diff(window, axis=0)
        mean_abs_diff = np.mean(np.abs(diffs), axis=0)
        std_diff = np.std(diffs, axis=0)
    else:
        mean_abs_diff = np.zeros(window.shape[1], dtype=np.float64)
        std_diff = np.zeros(window.shape[1], dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        skewness = skew(window, axis=0, bias=False, nan_policy="omit")
        kurt = kurtosis(window, axis=0, fisher=True, bias=False, nan_policy="omit")

    skewness = np.nan_to_num(skewness, nan=0.0, posinf=0.0, neginf=0.0)
    kurt = np.nan_to_num(kurt, nan=0.0, posinf=0.0, neginf=0.0)

    feature_vector = np.concatenate(
        [
            means,
            stds,
            vars_,
            mins,
            maxs,
            medians,
            slopes,
            rms,
            mean_abs_diff,
            std_diff,
            skewness,
            kurt,
        ]
    )

    return feature_vector.astype(np.float64, copy=False)


# Phase E: Extract engineered features from all windows
def extract_features_from_windows(
    windows: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """
    Extract flat feature vectors from a full window tensor.

    Parameters
    ----------
    windows : np.ndarray
        3D tensor shaped (n_windows, window_size, n_variables).

    Returns
    -------
    tuple[np.ndarray, list[str]]
        Feature matrix and feature names.
    """
    windows = validate_window_tensor(windows)

    n_windows, _, n_variables = windows.shape
    feature_names = generate_feature_names(n_variables=n_variables)

    feature_matrix = np.vstack(
        [extract_features_from_single_window(window) for window in windows]
    )

    if feature_matrix.shape[0] != n_windows:
        raise RuntimeError("Feature matrix row count does not match number of windows.")

    if feature_matrix.shape[1] != len(feature_names):
        raise RuntimeError("Feature matrix column count does not match feature names.")

    return feature_matrix, feature_names


# Phase F: Build a feature dataframe and optionally attach window metadata
def build_window_feature_dataframe(
    windows: np.ndarray,
    window_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Convert a window tensor into a feature dataframe.

    Parameters
    ----------
    windows : np.ndarray
        3D tensor shaped (n_windows, window_size, n_variables).
    window_df : pd.DataFrame | None
        Optional metadata table to attach alongside features.

    Returns
    -------
    pd.DataFrame
        DataFrame containing engineered features, optionally merged with metadata.
    """
    feature_matrix, feature_names = extract_features_from_windows(windows)
    feature_df = pd.DataFrame(feature_matrix, columns=feature_names)

    if window_df is not None:
        if len(window_df) != len(feature_df):
            raise ValueError(
                "window_df length does not match the number of extracted windows."
            )

        window_df = window_df.reset_index(drop=True)
        feature_df = feature_df.reset_index(drop=True)

        return pd.concat([window_df, feature_df], axis=1)

    return feature_df
