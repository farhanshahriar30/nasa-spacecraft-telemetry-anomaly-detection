# Reporting utilities for final tables and figures.
# This file contains helper functions for rounding results, saving tables,
# and generating compact plots used in the project analysis.
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


# Phase A: Round numeric columns for cleaner paper-ready tables
def round_numeric_columns(
    df: pd.DataFrame,
    decimals: int = 4,
) -> pd.DataFrame:
    """
    Return a copy of the dataframe with numeric columns rounded.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    decimals : int
        Number of decimal places.

    Returns
    -------
    pd.DataFrame
        Rounded dataframe copy.
    """
    rounded_df = df.copy()
    numeric_cols = rounded_df.select_dtypes(include="number").columns
    rounded_df[numeric_cols] = rounded_df[numeric_cols].round(decimals)
    return rounded_df


# Phase B: Save dataframe to CSV
def save_dataframe_csv(
    df: pd.DataFrame,
    output_path: str | Path,
    index: bool = False,
) -> Path:
    """
    Save a dataframe to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to save.
    output_path : str | Path
        Output CSV path.
    index : bool
        Whether to save index.

    Returns
    -------
    Path
        Saved file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=index)
    return output_path


# Phase C: Build a compact final comparison table for the paper
def build_paper_comparison_table(
    comparison_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select and reorder the most paper-relevant columns.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Full comparison dataframe.

    Returns
    -------
    pd.DataFrame
        Compact paper-ready comparison table.
    """
    desired_cols = [
        "model",
        "dev_best_threshold",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_pr_auc",
        "test_roc_auc",
        "test_tp",
        "test_fp",
        "test_tn",
        "test_fn",
    ]

    available_cols = [col for col in desired_cols if col in comparison_df.columns]
    return comparison_df[available_cols].copy()


# Phase D: Plot threshold vs one chosen metric
def plot_threshold_metric_curve(
    threshold_df: pd.DataFrame,
    metric: str,
    title: str,
    output_path: str | Path | None = None,
) -> None:
    """
    Plot threshold against a chosen metric.

    Parameters
    ----------
    threshold_df : pd.DataFrame
        Threshold evaluation dataframe.
    metric : str
        Metric column to plot, e.g. "f1", "precision", "recall".
    title : str
        Plot title.
    output_path : str | Path | None
        Optional path to save the figure.
    """
    if "threshold" not in threshold_df.columns:
        raise ValueError("threshold_df must contain a 'threshold' column.")

    if metric not in threshold_df.columns:
        raise ValueError(f"threshold_df must contain '{metric}'.")

    plt.figure(figsize=(8, 5))
    plt.plot(threshold_df["threshold"], threshold_df[metric], marker="o")
    plt.xlabel("Threshold")
    plt.ylabel(metric.upper())
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()


# Phase E: Plot final test metric comparison across models
def plot_model_metric_comparison(
    comparison_df: pd.DataFrame,
    metric: str,
    title: str,
    output_path: str | Path | None = None,
) -> None:
    """
    Plot one final test metric across models.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Model comparison dataframe.
    metric : str
        Metric column to plot, e.g. "test_f1", "test_pr_auc", "test_roc_auc".
    title : str
        Plot title.
    output_path : str | Path | None
        Optional path to save the figure.
    """
    if "model" not in comparison_df.columns:
        raise ValueError("comparison_df must contain a 'model' column.")

    if metric not in comparison_df.columns:
        raise ValueError(f"comparison_df must contain '{metric}'.")

    plt.figure(figsize=(8, 5))
    plt.bar(comparison_df["model"], comparison_df[metric])
    plt.xlabel("Model")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(title)
    plt.xticks(rotation=15)
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()


# Phase F: Build a compact threshold summary table
def build_threshold_summary_table(
    threshold_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build a compact table summarizing the best threshold per model.

    Parameters
    ----------
    threshold_tables : dict[str, pd.DataFrame]
        Mapping from model name to threshold sweep dataframe.

    Returns
    -------
    pd.DataFrame
        Summary table of best threshold rows.
    """
    records = []

    for model_name, df in threshold_tables.items():
        if df.empty:
            continue

        best_row = df.sort_values(
            ["f1", "precision", "recall", "threshold"],
            ascending=[False, False, False, True],
        ).iloc[0]

        records.append(
            {
                "model": model_name,
                "best_threshold": best_row["threshold"],
                "dev_precision": best_row["precision"],
                "dev_recall": best_row["recall"],
                "dev_f1": best_row["f1"],
                "dev_pr_auc": best_row["pr_auc"],
                "dev_roc_auc": best_row["roc_auc"],
            }
        )

    return pd.DataFrame(records)
