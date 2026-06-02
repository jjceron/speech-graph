from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.stats import demographic_columns


def contribution_frame(
    x_scaled: np.ndarray,
    coefficients: np.ndarray,
    feature_names: list[str],
    row_metadata: pd.DataFrame,
    target: str,
    fold: int,
) -> pd.DataFrame:
    values = x_scaled * coefficients.reshape(1, -1)
    meta_cols = [c for c in ["code", "age_group", "schooling_group"] if c in row_metadata.columns]
    rows: list[dict] = []
    for i in range(values.shape[0]):
        base = {c: row_metadata.iloc[i][c] for c in meta_cols}
        base.update({"target": target, "fold": fold})
        for j, feature in enumerate(feature_names):
            value = float(values[i, j])
            if np.isfinite(value):
                rows.append({**base, "feature": feature, "contribution": value, "abs_contribution": abs(value)})
    return pd.DataFrame(rows)


def summarize_population(contrib: pd.DataFrame) -> pd.DataFrame:
    if contrib.empty:
        return pd.DataFrame(columns=["target", "feature", "mean_contribution", "mean_abs_contribution", "std_contribution", "n"])
    out = (
        contrib.groupby(["target", "feature"], dropna=False)["contribution"]
        .agg(mean_contribution="mean", mean_abs_contribution=lambda s: s.abs().mean(), std_contribution="std", n="count")
        .reset_index()
    )
    return out.sort_values(["target", "mean_abs_contribution"], ascending=[True, False]).reset_index(drop=True)


def summarize_group(contrib: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if contrib.empty or group_col not in contrib.columns:
        return pd.DataFrame()
    out = (
        contrib.groupby(["target", group_col, "feature"], dropna=False)["contribution"]
        .agg(mean_contribution="mean", mean_abs_contribution=lambda s: s.abs().mean(), n="count")
        .reset_index()
    )
    return out.sort_values(["target", group_col, "mean_abs_contribution"], ascending=[True, True, False]).reset_index(drop=True)


def make_age_groups(df: pd.DataFrame) -> pd.Series:
    demo = demographic_columns(df)
    col = demo.get("age")
    if col is None:
        return pd.Series("unknown", index=df.index)
    age = pd.to_numeric(df[col], errors="coerce")
    bins = [-np.inf, 7, 9, 11, 13, 15, np.inf]
    labels = ["<=7", "8-9", "10-11", "12-13", "14-15", ">=16"]
    return pd.cut(age, bins=bins, labels=labels).astype(object).where(age.notna(), "unknown")


def make_schooling_groups(df: pd.DataFrame) -> pd.Series:
    demo = demographic_columns(df)
    col = demo.get("schooling")
    if col is None:
        return pd.Series("unknown", index=df.index)
    values = df[col]
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() >= max(5, int(0.5 * len(values))):
        bins = [-np.inf, 2, 4, 6, 8, 10, np.inf]
        labels = ["<=2", "3-4", "5-6", "7-8", "9-10", ">=11"]
        return pd.cut(numeric, bins=bins, labels=labels).astype(object).where(numeric.notna(), "unknown")
    return values.astype(str).replace({"nan": "unknown", "None": "unknown", "": "unknown"}).fillna("unknown")
