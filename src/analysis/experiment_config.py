"""Shared experiment configuration: feature sets for raw, zscores, rawzscore.

Provides unified feature resolution across linear_regression_rcv.py,
regression_optuna.py, and correlation_analysis.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.correlation_analysis import (
    _compute_targets,
    find_means_tables,
    load_feature_table,
)
from src.pipeline.speechgraph import load_metadata


EXPERIMENTS = {
    "raw": {
        "base": [
            "nodes", "edges", "re", "pe", "l1", "l2", "l3",
            "lcc", "lsc", "atd", "density", "diameter", "asp",
        ],
        "cc": "cc",
    },
    "zscores": {
        "base": [
            "z_re", "z_pe", "z_l1", "z_l2", "z_l3",
            "z_lcc", "z_lsc", "z_atd", "z_density", "z_diameter", "z_asp",
        ],
        "cc": "z_cc",
    },
    "rawzscore": {
        "base": [
            "nodes", "edges", "re", "pe", "l1", "l2", "l3",
            "lcc", "lsc", "atd", "density", "diameter", "asp",
            "z_re", "z_pe", "z_l1", "z_l2", "z_l3",
            "z_lcc", "z_lsc", "z_atd", "z_density", "z_diameter", "z_asp",
        ],
        "cc": ["cc", "z_cc"],
    },
}

WINDOW_CC_THRESHOLD = 100


def get_experiment_feature_names(experiment: str, window: int) -> list[str]:
    """Return the list of feature column names for a given experiment and window size."""
    exp = EXPERIMENTS[experiment]
    base = list(exp["base"])
    cc = exp["cc"]
    if window >= WINDOW_CC_THRESHOLD:
        if isinstance(cc, list):
            base.extend(cc)
        else:
            base.append(cc)
    return base


def drop_zero_variance_columns(
    X: pd.DataFrame,
    feature_names: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Remove columns with zero variance and return the filtered DataFrame + name list."""
    if feature_names is None:
        feature_names = list(X.columns)
    present = [c for c in feature_names if c in X.columns]
    keep = [c for c in present if X[c].std() > 0]
    dropped = set(present) - set(keep)
    if dropped:
        print(f"  Dropped zero-variance columns: {sorted(dropped)}")
    return X[keep], keep


def load_experiment_matrix(
    experiment: str,
    task: int,
    window: int,
    target: str,
    metrics_dir: str | Path = "data/processed/metrics",
    metadata_path: str | Path = "data/raw/metadata.xlsx",
    covar_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the feature matrix X and target vector y for a given experiment.

    Steps:
        1. Resolve feature names from experiment config (handles cc filter).
        2. Load raw and/or z means tables.
        3. Merge with metadata.
        4. Drop zero-variance columns.
        5. Return (X, y).
    """
    metrics_dir = Path(metrics_dir)
    full_window = f"T{task}W{window}"
    feature_names = get_experiment_feature_names(experiment, window)
    meta = _compute_targets(load_metadata(metadata_path))
    target_ser = meta.set_index("Cod")[target]

    if covar_cols:
        for col in covar_cols:
            if col not in meta.columns:
                raise KeyError(f"Covariate not found in metadata: {col}")
            if not pd.api.types.is_numeric_dtype(meta[col]):
                meta[col] = pd.factorize(meta[col])[0].astype(float)

    # Determine which table types to load
    if experiment == "raw":
        table_types = ["raw"]
    elif experiment == "zscores":
        table_types = ["z"]
    else:
        table_types = ["raw", "z"]

    tables = find_means_tables(metrics_dir, tasks=[task])
    tables = [t for t in tables if t["tag"] == full_window and t["type"] in table_types]

    if not tables:
        raise FileNotFoundError(
            f"No means tables found for task={task}, window={full_window}, "
            f"types={table_types}"
        )

    # Load each table and merge on file
    merged = None
    for entry in tables:
        feats = load_feature_table(entry["path"])
        if merged is None:
            merged = feats
        else:
            merged = merged.merge(feats, on="file", how="inner", suffixes=("_raw", "_z"))

    # Build X from feature columns that exist in the merged table
    X_list = []
    for feat in feature_names:
        if feat in merged.columns:
            X_list.append(merged.set_index("file")[feat].rename(feat))

    X = pd.concat(X_list, axis=1)

    # Build y from metadata (avoids target-column suffix issues when merging tables)
    y = merged["file"].map(target_ser)
    y.index = merged["file"]
    y = y.dropna()

    # Add covariates
    if covar_cols:
        covar_df = meta[covar_cols]
        X = X.join(covar_df, how="left")

    # Align X and y
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]

    # Drop NaN
    valid = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid]
    y = y[valid]

    if len(X) < 20:
        raise ValueError(f"Only {len(X)} complete subjects; at least 20 required")

    # Drop zero-variance columns
    X, _ = drop_zero_variance_columns(X)

    return X, y
