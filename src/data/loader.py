# Data loading utilities for the spacecraft telemetry benchmark.
# This file reads the anomaly metadata and raw train/test channel arrays,
# and provides helper functions for quick channel-level dataset inspection.
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import LABELS_PATH, TEST_DIR, TRAIN_DIR


# Phase A: Parse list-like strings from the anomaly metadata safely
def parse_literal(value: Any) -> Any:
    """
    Safely parse Python-literal-style strings from the metadata file.
    Falls back to the original value if parsing fails.
    """
    if pd.isna(value):
        return value

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return value
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value

    return value


# Phase B: Parse anomaly class strings like:
# "[contextual, contextual, point]"
# because these are not valid Python literals
def parse_class_list(value: Any) -> list[str]:
    """
    Parse the anomaly class column into a clean list of strings.
    """
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return [str(x).strip() for x in value]

    if isinstance(value, str):
        value = value.strip()

        if not value:
            return []

        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            if not inner:
                return []
            return [item.strip().strip("'\"") for item in inner.split(",")]

        return [value.strip().strip("'\"")]

    return [str(value).strip()]


# Phase C: Load the anomaly metadata CSV and clean key columns
def load_labels_metadata(labels_path: Path = LABELS_PATH) -> pd.DataFrame:
    """
    Load and preprocess the labeled anomalies metadata.
    """
    if not labels_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {labels_path}")

    df = pd.read_csv(labels_path)

    required_cols = {
        "chan_id",
        "spacecraft",
        "anomaly_sequences",
        "class",
        "num_values",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Metadata file is missing required columns: {missing}")

    df = df.copy()
    df["anomaly_sequences"] = df["anomaly_sequences"].apply(parse_literal)
    df["class"] = df["class"].apply(parse_class_list)
    df["num_values"] = pd.to_numeric(df["num_values"], errors="coerce")

    return df


# Phase D: Load a single channel array from either train or test folder
def load_channel_array(chan_id: str, split: str) -> np.ndarray:
    """
    Load one telemetry channel array from the requested split.
    """
    split = split.lower().strip()

    if split == "train":
        base_dir = TRAIN_DIR
    elif split == "test":
        base_dir = TEST_DIR
    else:
        raise ValueError("split must be either 'train' or 'test'")

    file_path = base_dir / f"{chan_id}.npy"
    if not file_path.exists():
        raise FileNotFoundError(f"Channel file not found: {file_path}")

    arr = np.load(file_path)

    return arr


# Phase E: Load both train and test arrays for a given channel
def load_channel_pair(chan_id: str) -> dict[str, np.ndarray]:
    """
    Load both training and test arrays for a telemetry channel.
    """
    return {
        "train": load_channel_array(chan_id, split="train"),
        "test": load_channel_array(chan_id, split="test"),
    }


# Phase F: Build a channel summary table for quick inspection
def build_channel_inventory(
    labels_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build a summary table describing channels and file availability.
    """
    if labels_df is None:
        labels_df = load_labels_metadata()

    df = labels_df.copy()

    df["train_exists"] = df["chan_id"].apply(
        lambda c: (TRAIN_DIR / f"{c}.npy").exists()
    )
    df["test_exists"] = df["chan_id"].apply(lambda c: (TEST_DIR / f"{c}.npy").exists())

    def count_intervals(x: Any) -> int:
        if isinstance(x, list):
            return len(x)
        return 0

    def count_classes(x: Any) -> int:
        if isinstance(x, list):
            return len(x)
        return 0

    df["num_anomaly_intervals"] = df["anomaly_sequences"].apply(count_intervals)
    df["num_anomaly_classes"] = df["class"].apply(count_classes)

    return df.sort_values(["spacecraft", "chan_id"]).reset_index(drop=True)
