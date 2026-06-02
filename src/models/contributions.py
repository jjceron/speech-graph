from __future__ import annotations

import re

import numpy as np
import pandas as pd


def linear_contributions(
    x_test_scaled: np.ndarray,
    coefficients: np.ndarray,
    feature_names: list[str],
    row_metadata: pd.DataFrame,
    target: str,
    fold: int,
) -> pd.DataFrame:
    values = x_test_scaled * coefficients.reshape(1, -1)
    meta_cols = [col for col in row_metadata.columns if col not in feature_names]
    rows: list[dict] = []
    for row_idx in range(values.shape[0]):
        meta = {col: row_metadata.iloc[row_idx][col] for col in meta_cols if col in row_metadata.columns}
        meta.update({"target": target, "fold": int(fold)})
        for feature_idx, feature in enumerate(feature_names):
            value = float(values[row_idx, feature_idx])
            if np.isfinite(value):
                rows.append(meta | {"feature": feature, "contribution": value, "abs_contribution": abs(value)})
    return pd.DataFrame(rows)


def summarize_population(contrib: pd.DataFrame) -> pd.DataFrame:
    if contrib.empty:
        return pd.DataFrame(columns=["target", "feature", "mean_contribution", "mean_abs_contribution", "std_contribution", "n"])
    out = (
        contrib.groupby(["target", "feature"], dropna=False)["contribution"]
        .agg(mean_contribution="mean", mean_abs_contribution=lambda s: s.abs().mean(), std_contribution="std", n="count")
        .reset_index()
    )
    return out.sort_values(["target", "mean_abs_contribution"], ascending=[True, False])


def _norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {_norm(col): col for col in df.columns}
    for candidate in candidates:
        key = _norm(candidate)
        if key in lookup:
            return lookup[key]
    for col in df.columns:
        ncol = _norm(col)
        if any(_norm(candidate) in ncol for candidate in candidates):
            return col
    return None


def add_age_group(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    age_col = find_column(df, ["Age", "Edad"])
    if age_col is None:
        return df, None
    out = df.copy()
    age = pd.to_numeric(out[age_col], errors="coerce")
    bins = [0, 6, 8, 10, 12, 14, 18, 100]
    labels = ["<=6", "7-8", "9-10", "11-12", "13-14", "15-18", ">18"]
    out["age_group"] = pd.cut(age, bins=bins, labels=labels, right=True, include_lowest=True).astype("object")
    out.loc[age.notna() & out["age_group"].isna(), "age_group"] = age.astype("Int64").astype(str)
    return out, "age_group"


def summarize_by_column(contrib: pd.DataFrame, group_col: str | None) -> pd.DataFrame:
    if contrib.empty or group_col is None or group_col not in contrib.columns:
        return pd.DataFrame()
    out = (
        contrib.dropna(subset=[group_col])
        .groupby(["target", group_col, "feature"], dropna=False)["contribution"]
        .agg(mean_contribution="mean", mean_abs_contribution=lambda s: s.abs().mean(), n="count")
        .reset_index()
    )
    return out.sort_values(["target", group_col, "mean_abs_contribution"], ascending=[True, True, False])


def schooling_column(df: pd.DataFrame) -> str | None:
    return find_column(df, ["School year", "Educational level", "Escolaridad", "Nivel educativo", "Grado"])
