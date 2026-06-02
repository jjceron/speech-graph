from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from src.graphs import CANONICAL_METRICS

BARRATT_TARGETS = ["TOTAL", "NPLAN", "MOT", "COG"]
ID_COLUMNS = {
    "code", "file", "level", "activity", "activity_number", "activity_index", "start_time", "end_time",
    "window_size", "window_step", "scheme_window_size", "random_times", "token_count", "segment_count",
    "window_count", "valid_window", "_merge", "_join_code", "Cod",
}
TARGET_ALIASES = {
    "TOTAL": ["TOTAL", "Total", "Barratt Total", "Barratt (pre)", "BIS Total", "BIS_TOTAL"],
    "NPLAN": ["NPLAN", "No plan", "No planificación", "No planeación", "Nonplanning", "NPLAN_zscore"],
    "MOT": ["MOT", "Motor", "MOT_zscore"],
    "COG": ["COG", "Cognitive", "Cognitiva", "COG_zscore"],
}
DEMOGRAPHIC_ALIASES = {
    "age": ["Age", "Edad", "EDAD"],
    "schooling": ["School year", "Escolaridad", "Educational level", "Nivel educativo", "Grado", "Curso"],
}


def _norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def find_column(df: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    lookup = {_norm(col): col for col in df.columns}
    for alias in aliases:
        key = _norm(alias)
        if key in lookup:
            return lookup[key]
    for alias in aliases:
        key = _norm(alias)
        if not key:
            continue
        for col in df.columns:
            if key in _norm(col):
                return col
    return None


def resolve_target_columns(df: pd.DataFrame, requested: str | Iterable[str] = BARRATT_TARGETS) -> dict[str, str]:
    names = [x.strip() for x in str(requested).split(",") if x.strip()] if isinstance(requested, str) else list(requested)
    resolved: dict[str, str] = {}
    for name in names:
        canonical = name.upper() if name.upper() in BARRATT_TARGETS else name
        aliases = TARGET_ALIASES.get(canonical, [name])
        col = find_column(df, aliases)
        if col is not None and col not in resolved.values():
            resolved[canonical] = col
    return resolved


def add_standard_target_columns(df: pd.DataFrame, requested: str | Iterable[str] = BARRATT_TARGETS) -> pd.DataFrame:
    out = df.copy()
    for canonical, col in resolve_target_columns(out, requested).items():
        out[canonical] = pd.to_numeric(out[col], errors="coerce")
    return out


def demographic_columns(df: pd.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, aliases in DEMOGRAPHIC_ALIASES.items():
        col = find_column(df, aliases)
        if col is not None:
            out[key] = col
    return out


def canonical_feature_columns(df: pd.DataFrame, include_global: bool = True, include_window_stats: bool = True) -> list[str]:
    allowed: set[str] = set()
    if include_window_stats:
        for prefix in ("mean", "std"):
            allowed.update(f"{prefix}_{metric}" for metric in CANONICAL_METRICS)
    if include_global:
        allowed.update(f"global_{metric}" for metric in CANONICAL_METRICS)
    cols: list[str] = []
    for col in df.columns:
        if col in allowed:
            values = pd.to_numeric(df[col], errors="coerce")
            if values.notna().sum() >= 3 and values.nunique(dropna=True) > 1:
                cols.append(col)
    return cols


def _safe_corr(x: pd.Series, y: pd.Series, method: str) -> tuple[float, float, int]:
    x_num = pd.to_numeric(x, errors="coerce")
    y_num = pd.to_numeric(y, errors="coerce")
    mask = x_num.notna() & y_num.notna()
    n = int(mask.sum())
    if n < 3 or x_num[mask].nunique(dropna=True) < 2 or y_num[mask].nunique(dropna=True) < 2:
        return np.nan, np.nan, n
    if method == "pearson":
        r, p = stats.pearsonr(x_num[mask], y_num[mask])
    else:
        r, p = stats.spearmanr(x_num[mask], y_num[mask])
    return float(r), float(p), n


def correlation_table(df: pd.DataFrame, metric_cols: Iterable[str], target_cols: dict[str, str] | Iterable[str], method: str = "spearman") -> pd.DataFrame:
    if isinstance(target_cols, dict):
        targets = target_cols.items()
    else:
        targets = [(str(col), str(col)) for col in target_cols]
    rows: list[dict] = []
    for metric in metric_cols:
        if metric not in df.columns:
            continue
        for target_name, target_col in targets:
            if target_col not in df.columns or metric == target_col:
                continue
            r, p, n = _safe_corr(df[metric], df[target_col], method)
            if n >= 3 and np.isfinite(r):
                rows.append({"metric": metric, "target": target_name, "target_column": target_col, "r": r, "p": p, "n": n})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["abs_r"] = out["r"].abs()
        out = out.sort_values(["abs_r", "metric", "target"], ascending=[False, True, True]).reset_index(drop=True)
    return out


def _fdr_bh(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
    q = np.full(len(p), np.nan)
    mask = np.isfinite(p)
    if mask.sum() == 0:
        return pd.Series(q, index=p_values.index)
    idx = np.where(mask)[0]
    order = idx[np.argsort(p[mask])]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.clip(ranked, 0, 1)
    return pd.Series(q, index=p_values.index)


def correlations_by_activity_window(df: pd.DataFrame, targets: dict[str, str], method: str = "spearman") -> pd.DataFrame:
    work = df.copy()
    if "_merge" in work.columns:
        work = work[work["_merge"].astype(str).eq("both")]
    if "valid_window" in work.columns:
        work = work[pd.to_numeric(work["valid_window"], errors="coerce").eq(1)]
    metrics = canonical_feature_columns(work, include_global=False, include_window_stats=True)
    rows: list[pd.DataFrame] = []
    group_cols = [c for c in ["scheme_window_size", "activity", "activity_number"] if c in work.columns]
    for key, sub in work.groupby(group_cols, dropna=False) if group_cols else [((), work)]:
        corr = correlation_table(sub, metrics, targets, method)
        if corr.empty:
            continue
        key_tuple = key if isinstance(key, tuple) else (key,)
        for col, value in zip(group_cols, key_tuple):
            corr.insert(0, col, value)
        rows.append(corr)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not out.empty:
        out["q_fdr"] = _fdr_bh(out["p"])
        out = out.sort_values(["abs_r", "q_fdr"], ascending=[False, True]).reset_index(drop=True)
    return out


def build_activity_window_features(df: pd.DataFrame, targets: dict[str, str]) -> pd.DataFrame:
    work = df.copy()
    if "_merge" in work.columns:
        work = work[work["_merge"].astype(str).eq("both")]
    if "valid_window" in work.columns:
        work = work[pd.to_numeric(work["valid_window"], errors="coerce").eq(1)]
    base_cols = [c for c in ["code", "file", "activity", "activity_number", "scheme_window_size", "token_count", "window_count"] if c in work.columns]
    feature_cols = canonical_feature_columns(work, include_global=True, include_window_stats=True)
    target_cols = list(targets.values())
    demo_cols = list(demographic_columns(work).values())
    cols = list(dict.fromkeys(base_cols + feature_cols + target_cols + demo_cols))
    return work[cols].copy()


def build_subject_level_features(df: pd.DataFrame, targets: dict[str, str]) -> pd.DataFrame:
    work = df.copy()
    if "_merge" in work.columns:
        work = work[work["_merge"].astype(str).eq("both")]
    if "valid_window" in work.columns:
        work = work[pd.to_numeric(work["valid_window"], errors="coerce").eq(1)]
    if "activity_number" not in work.columns or "scheme_window_size" not in work.columns:
        raise ValueError("Input must include activity_number and scheme_window_size to build subject-level features.")

    feature_cols = canonical_feature_columns(work, include_global=False, include_window_stats=True)
    pieces: list[pd.DataFrame] = []
    for _, row in work.iterrows():
        code = row["code"]
        act = int(row["activity_number"]) if pd.notna(row["activity_number"]) else None
        win = int(row["scheme_window_size"]) if pd.notna(row["scheme_window_size"]) else None
        if act is None or win is None:
            continue
        item = {"code": code}
        for col in feature_cols:
            item[f"w{win}_a{act}_{col}"] = row[col]
        pieces.append(pd.DataFrame([item]))
    if pieces:
        feature_matrix = pd.concat(pieces, ignore_index=True).groupby("code", as_index=False).first()
    else:
        feature_matrix = pd.DataFrame(columns=["code"])

    global_cols = canonical_feature_columns(work, include_global=True, include_window_stats=False)
    global_rows: list[dict] = []
    first_per_activity = work.sort_values(["code", "activity_number", "scheme_window_size"]).drop_duplicates(["code", "activity_number"])
    for _, row in first_per_activity.iterrows():
        act = int(row["activity_number"]) if pd.notna(row["activity_number"]) else None
        if act is None:
            continue
        item = {"code": row["code"]}
        for col in global_cols:
            item[f"a{act}_{col}"] = row[col]
        global_rows.append(item)
    if global_rows:
        global_matrix = pd.DataFrame(global_rows).groupby("code", as_index=False).first()
        feature_matrix = feature_matrix.merge(global_matrix, on="code", how="outer")

    metadata_cols = ["code"] + list(targets.values()) + list(demographic_columns(work).values())
    metadata = work[metadata_cols].drop_duplicates("code").copy()
    out = feature_matrix.merge(metadata, on="code", how="left")
    return out


def correlations_subject_level(subject_df: pd.DataFrame, targets: dict[str, str], method: str = "spearman") -> pd.DataFrame:
    excluded = {"code", *targets.values(), *demographic_columns(subject_df).values()}
    metrics = []
    for col in subject_df.columns:
        if col in excluded:
            continue
        values = pd.to_numeric(subject_df[col], errors="coerce")
        if values.notna().sum() >= 3 and values.nunique(dropna=True) > 1:
            metrics.append(col)
    out = correlation_table(subject_df, metrics, targets, method)
    if not out.empty:
        out["q_fdr"] = _fdr_bh(out["p"])
    return out


def profile_by_group(df: pd.DataFrame, group_cols: Iterable[str], value_cols: Iterable[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for group in group_cols:
        if group not in df.columns:
            continue
        cols = [c for c in value_cols if c in df.columns]
        if not cols:
            continue
        agg = df.groupby(group, dropna=False)[cols].agg(["mean", "std", "count"])
        agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
        agg = agg.reset_index()
        agg.insert(0, "group_col", group)
        rows.append(agg.copy())
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
