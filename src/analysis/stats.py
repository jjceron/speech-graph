from __future__ import annotations

from typing import Iterable, List

import pandas as pd
from scipy import stats


def numeric_columns(df: pd.DataFrame, exclude: Iterable[str] | None = None) -> list[str]:
    """Return columns that can be interpreted as numeric."""
    excluded = set(exclude or [])
    cols: list[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().sum() > 0:
            cols.append(col)
    return cols


def correlation_table(
    df: pd.DataFrame,
    metric_cols: Iterable[str],
    target_cols: Iterable[str],
    method: str = "spearman",
) -> pd.DataFrame:
    rows: List[dict] = []
    for metric in metric_cols:
        if metric not in df.columns:
            continue
        for target in target_cols:
            if target not in df.columns or metric == target:
                continue
            x = pd.to_numeric(df[metric], errors="coerce")
            y = pd.to_numeric(df[target], errors="coerce")
            mask = x.notna() & y.notna()
            if mask.sum() < 3:
                continue
            if x[mask].nunique(dropna=True) < 2 or y[mask].nunique(dropna=True) < 2:
                continue

            if method == "pearson":
                r, p = stats.pearsonr(x[mask], y[mask])
            else:
                r, p = stats.spearmanr(x[mask], y[mask])

            rows.append(
                {
                    "metric": metric,
                    "target": target,
                    "r": float(r),
                    "p": float(p),
                    "n": int(mask.sum()),
                }
            )

    return pd.DataFrame(rows)


def group_profile(
    df: pd.DataFrame,
    group_cols: Iterable[str],
    metric_cols: Iterable[str],
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for group in group_cols:
        if group not in df.columns:
            continue
        cols = [c for c in metric_cols if c in df.columns]
        if not cols:
            continue
        agg = df.groupby(group, dropna=False)[cols].agg(["mean", "std", "count"])
        agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
        agg = agg.reset_index()
        agg.insert(0, "group_col", group)
        frames.append(agg)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)
