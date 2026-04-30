# This file defines channel-level data splitting utilities.
# It is used to create development/test splits and grouped folds
# so that windows from the same channel do not leak across evaluation boundaries.

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE


# Phase A: Validate the channel summary table before splitting
def validate_channel_summary_df(channel_summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the per-channel summary dataframe used for channel-level splitting.

    Required columns
    ----------------
    chan_id, n_windows, n_anomalous_windows, anomaly_ratio
    """
    required_cols = {
        "chan_id",
        "n_windows",
        "n_anomalous_windows",
        "anomaly_ratio",
    }
    missing = required_cols - set(channel_summary_df.columns)
    if missing:
        raise ValueError(f"channel_summary_df is missing required columns: {missing}")

    df = channel_summary_df.copy()

    if df["chan_id"].duplicated().any():
        dupes = df.loc[df["chan_id"].duplicated(), "chan_id"].tolist()
        raise ValueError(f"Duplicate chan_id values found: {dupes}")

    return df.reset_index(drop=True)


# Phase B: Build simple stratification labels from anomaly ratios
# This gives us a better chance of balancing low/medium/high anomaly-burden channels
# across the outer dev/test split.
def make_channel_stratify_labels(
    channel_summary_df: pd.DataFrame,
    n_bins: int = 3,
) -> pd.Series | None:
    """
    Create approximate stratification labels from anomaly_ratio.

    Returns None if the data are too small or too degenerate for safe stratification.
    """
    df = validate_channel_summary_df(channel_summary_df)
    ratios = df["anomaly_ratio"].astype(float)

    if ratios.nunique() < 2:
        return None

    q = min(n_bins, int(ratios.nunique()))

    try:
        labels = pd.qcut(ratios, q=q, labels=False, duplicates="drop")
    except ValueError:
        return None

    if labels is None:
        return None

    labels = pd.Series(labels, index=df.index)

    if labels.nunique() < 2:
        return None

    counts = labels.value_counts()
    if (counts < 2).any():
        return None

    return labels.astype(str)


# Phase C: Create the outer channel-level development/test split
def create_outer_channel_split(
    channel_summary_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    use_stratification: bool = True,
) -> dict[str, Any]:
    """
    Split channels into development and held-out test sets.

    Parameters
    ----------
    channel_summary_df : pd.DataFrame
        Per-channel summary table.
    test_size : float
        Fraction of channels to place in the outer test split.
    random_state : int
        Random seed.
    use_stratification : bool
        If True, attempt approximate stratification using anomaly_ratio bins.

    Returns
    -------
    dict[str, Any]
        Development/test channel IDs and annotated split summary table.
    """
    df = validate_channel_summary_df(channel_summary_df)

    stratify_labels = None
    if use_stratification:
        stratify_labels = make_channel_stratify_labels(df)

    dev_chan_ids, test_chan_ids = train_test_split(
        df["chan_id"].tolist(),
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify_labels if stratify_labels is not None else None,
    )

    dev_chan_ids = sorted(dev_chan_ids)
    test_chan_ids = sorted(test_chan_ids)

    dev_set = set(dev_chan_ids)
    test_set = set(test_chan_ids)

    split_summary_df = df.copy()
    split_summary_df["outer_split"] = split_summary_df["chan_id"].map(
        lambda c: "test" if c in test_set else "dev"
    )

    return {
        "dev_chan_ids": dev_chan_ids,
        "test_chan_ids": test_chan_ids,
        "split_summary_df": split_summary_df,
        "used_stratification": stratify_labels is not None,
    }


# Phase D: Apply a channel-level split to the full window-level feature dataframe
def apply_channel_split_to_feature_df(
    feature_df: pd.DataFrame,
    dev_chan_ids: list[str],
    test_chan_ids: list[str],
) -> dict[str, pd.DataFrame]:
    """
    Split a window-level feature dataframe according to channel membership.
    """
    if "chan_id" not in feature_df.columns:
        raise ValueError("feature_df must contain a 'chan_id' column.")

    dev_set = set(dev_chan_ids)
    test_set = set(test_chan_ids)

    overlap = dev_set & test_set
    if overlap:
        raise ValueError(
            f"Development and test channel sets overlap: {sorted(overlap)}"
        )

    df = feature_df.copy()

    known_channels = dev_set | test_set
    observed_channels = set(df["chan_id"].unique())
    unknown = observed_channels - known_channels
    if unknown:
        raise ValueError(
            f"feature_df contains channels not present in the provided split: {sorted(unknown)}"
        )

    dev_df = df[df["chan_id"].isin(dev_set)].reset_index(drop=True)
    test_df = df[df["chan_id"].isin(test_set)].reset_index(drop=True)

    return {
        "dev_df": dev_df,
        "test_df": test_df,
    }


# Phase E: Summarize the resulting split for quick inspection
def summarize_outer_split(
    split_summary_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Build a compact summary of the outer split.
    """
    summary = {
        "n_dev_channels": int((split_summary_df["outer_split"] == "dev").sum()),
        "n_test_channels": int((split_summary_df["outer_split"] == "test").sum()),
        "mean_dev_anomaly_ratio": float(
            split_summary_df.loc[
                split_summary_df["outer_split"] == "dev", "anomaly_ratio"
            ].mean()
        ),
        "mean_test_anomaly_ratio": float(
            split_summary_df.loc[
                split_summary_df["outer_split"] == "test", "anomaly_ratio"
            ].mean()
        ),
        "n_dev_windows": len(dev_df),
        "n_test_windows": len(test_df),
    }

    if "label" in dev_df.columns:
        summary["n_dev_anomalous_windows"] = int(dev_df["label"].sum())

    if "label" in test_df.columns:
        summary["n_test_anomalous_windows"] = int(test_df["label"].sum())

    return summary
