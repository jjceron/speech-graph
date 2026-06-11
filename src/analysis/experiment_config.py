"""Shared experiment configuration: feature sets for raw, zscores, rawzscore.

Provides unified feature resolution across linear_regression_rcv.py,
regression_optuna.py, and correlation_analysis.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.speechgraph import load_metadata


ITEM = {
    "MOT_V4": ["8.", "13.", "16.", "21.", "23."],
    "COG_V1": ["3.", "6."],
}


def _compute_targets(meta: pd.DataFrame) -> pd.DataFrame:
    """Add MOT_V4 and COG_V1 columns to metadata."""
    meta = meta.copy()
    meta["MOT_V4"] = meta[ITEM["MOT_V4"]].sum(axis=1)
    meta["COG_V1"] = meta[ITEM["COG_V1"]].sum(axis=1)
    return meta


def find_means_tables(metrics_dir: Path, tasks: list[int] | None = None) -> list[dict]:
    records = []
    for task_dir in sorted(metrics_dir.iterdir()):
        if not task_dir.is_dir() or not task_dir.name.startswith("Task"):
            continue
        task_num = int(task_dir.name.replace("Task", ""))
        if tasks is not None and task_num not in tasks:
            continue
        for fpath in sorted(task_dir.iterdir()):
            name = fpath.name
            if "means_params_table" in name:
                is_z = name.startswith("z_")
                tag = name.replace("z_means_params_table", "").replace("means_params_table", "").replace(".txt", "")
                records.append({
                    "path": fpath,
                    "task": task_num,
                    "tag": tag,
                    "type": "z" if is_z else "raw",
                })
    return records


def load_feature_table(
    path: Path,
    z_metrics: list[str] | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    mapping = {}
    for col in df.columns:
        if col == "file":
            continue
        if col.startswith("z_"):
            mapping[col] = col
        else:
            mapping[col] = col
    df = df.rename(columns=mapping)
    df["file"] = df["file"].apply(lambda x: Path(str(x)).stem)
    for suffix in ("_CorrEtiq", "-CorrEtiq", " CorrEtiq"):
        df["file"] = df["file"].str.replace(suffix, "", regex=False)
    df["file"] = df["file"].str.strip()
    return df


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


SRL_ZSCORE_FEATURES: dict[str, dict] = {
    "ap": {
        "base": [
            "z_re", "z_pe", "z_l1", "z_l2", "z_l3",
            "z_lcc", "z_lsc", "z_atd", "z_density", "z_diameter", "z_asp",
        ],
        "cc": "z_cc",
    },
    "pa": {
        "base": [
            "z_re", "z_pe", "z_l1", "z_l2", "z_l3",
            "z_lcc", "z_lsc", "z_atd", "z_density", "z_diameter", "z_asp",
        ],
        "cc": "z_cc",
    },
    "semantic": {
        "base": [
            "z_re", "z_pe", "z_l1", "z_l2", "z_l3",
            "z_lcc", "z_lsc", "z_atd", "z_density", "z_diameter", "z_asp",
        ],
        "cc": "z_cc",
    },
}


def find_srl_means_tables(
    metrics_dir: str | Path,
    tasks: list[int] | None = None,
    graph_types: list[str] | None = None,
) -> list[dict]:
    """Find SRL z-score means tables in the metrics directory.

    Searches for files matching ``z_means_params_table_T{task}W{window}_{graph_type}*.txt``
    and returns metadata for each.

    Args:
        metrics_dir: Base metrics directory (``data/processed/metrics``).
        tasks: Filter by task numbers (e.g. ``[2, 6, 7]``).
        graph_types: Filter by graph type (e.g. ``["ap", "pa"]``).

    Returns:
        List of dicts with keys ``path``, ``task``, ``window``, ``graph_type``.
    """
    records: list[dict] = []
    for task_dir in sorted(Path(metrics_dir).iterdir()):
        if not task_dir.is_dir() or not task_dir.name.startswith("Task"):
            continue
        try:
            task_num = int(task_dir.name.replace("Task", ""))
        except ValueError:
            continue
        if tasks is not None and task_num not in tasks:
            continue
        for fpath in sorted(task_dir.iterdir()):
            name = fpath.name
            if not name.startswith("z_means_params_table_"):
                continue
            body = name.replace("z_means_params_table_", "").replace(".txt", "")
            parts = body.split("_")
            if len(parts) < 2:
                continue
            tw_part = parts[0]
            gt_part = parts[1]
            if not tw_part.startswith("T") or "W" not in tw_part:
                continue
            try:
                t_part = tw_part.split("T")[1].split("W")[0]
                w_part = tw_part.split("W")[1]
                t = int(t_part)
                w = int(w_part)
            except (IndexError, ValueError):
                continue
            if graph_types is not None and gt_part not in graph_types:
                continue
            records.append({
                "path": fpath,
                "task": t,
                "window": w,
                "graph_type": gt_part,
            })
    return records


def load_srl_feature_table(path: str | Path) -> pd.DataFrame:
    """Load an SRL z-score means table and normalize subject codes."""
    df = pd.read_csv(path)
    df["file"] = df["file"].apply(lambda x: Path(str(x)).stem)
    for suffix in ("_CorrEtiq", "-CorrEtiq", " CorrEtiq"):
        df["file"] = df["file"].str.replace(suffix, "", regex=False)
    df["file"] = df["file"].str.strip()
    return df


def get_srl_experiment_feature_names(graph_type: str, window: int) -> list[str]:
    """Return the list of z-score features for an SRL graph type and window."""
    exp = SRL_ZSCORE_FEATURES[graph_type]
    base = list(exp["base"])
    cc = exp["cc"]
    if window >= WINDOW_CC_THRESHOLD:
        if isinstance(cc, list):
            base.extend(cc)
        else:
            base.append(cc)
    return base


def load_srl_experiment_matrix(
    graph_type: str,
    task: int,
    window: int,
    target: str,
    metrics_dir: str | Path = "data/processed/metrics",
    metadata_path: str | Path = "data/raw/metadata.xlsx",
    covar_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the feature matrix X and target vector y for an SRL z-score experiment.

    Args:
        graph_type: One of ``"ap"``, ``"pa"``, ``"semantic"``.
        task: Activity number (2, 6, or 7).
        window: Window size in sentences.
        target: BIS target column name.
        metrics_dir: Base metrics directory.
        metadata_path: Path to metadata Excel file.
        covar_cols: Optional covariate column names from metadata.

    Returns:
        ``(X, y)`` tuple where ``X`` is the feature DataFrame and ``y`` is
        the target Series, aligned on subject code index.

    Raises:
        FileNotFoundError: If no matching SRL z-score table is found.
        ValueError: If fewer than 20 complete subjects are available.
    """
    metrics_dir = Path(metrics_dir)
    feature_names = get_srl_experiment_feature_names(graph_type, window)
    meta = _compute_targets(load_metadata(metadata_path))
    target_ser = meta.set_index("Cod")[target]

    if covar_cols:
        for col in covar_cols:
            if col not in meta.columns:
                raise KeyError(f"Covariate not found in metadata: {col}")
            if not pd.api.types.is_numeric_dtype(meta[col]):
                meta[col] = pd.factorize(meta[col])[0].astype(float)

    tables = find_srl_means_tables(
        metrics_dir, tasks=[task], graph_types=[graph_type],
    )
    table = None
    for t in tables:
        if t["window"] == window:
            table = t
            break

    if table is None:
        raise FileNotFoundError(
            f"No SRL z-score means table found for graph_type={graph_type}, "
            f"task={task}, window={window}"
        )

    merged = load_srl_feature_table(table["path"])

    X_list = []
    for feat in feature_names:
        if feat in merged.columns:
            X_list.append(merged.set_index("file")[feat].rename(feat))

    if not X_list:
        raise ValueError(f"None of the {len(feature_names)} features found in table")

    X = pd.concat(X_list, axis=1)

    y = merged["file"].map(target_ser)
    y.index = merged["file"]
    y = y.dropna()

    if covar_cols:
        covar_df = meta[covar_cols]
        X = X.join(covar_df, how="left")

    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]

    valid = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid]
    y = y[valid]

    if len(X) < 20:
        raise ValueError(f"Only {len(X)} complete subjects; at least 20 required")

    X, _ = drop_zero_variance_columns(X)

    return X, y


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
