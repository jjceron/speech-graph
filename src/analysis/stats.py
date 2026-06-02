from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from src.graphs import MODEL_METRICS

ID_COLUMNS = {
    "code", "file", "level", "activity", "activity_number", "activity_index", "start_time", "end_time",
    "scheme_window_size", "window_size", "window_step", "random_times", "valid_window", "window_count",
    "_join_code", "_merge", "Cod",
}

TARGET_ALIASES = {
    "TOTAL": ["TOTAL", "Total", "Barratt Total", "BIS Total", "BIS_TOTAL", "Barratt (pre)"],
    "NPLAN": ["NPLAN", "No plan", "No planificación", "No planeación", "Nonplanning", "NPLAN_zscore"],
    "MOT": ["MOT", "Motor", "MOT_zscore"],
    "COG": ["COG", "Cognitive", "Cognitiva", "COG_zscore"],
}


def parse_csv_list(text: str | None) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def parse_int_set(text: str | None, default: Iterable[int] = range(1, 8)) -> set[int]:
    if not text:
        return set(default)
    return {int(item.strip()) for item in str(text).split(",") if item.strip()}


def _norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def resolve_targets(df: pd.DataFrame, requested: str = "Total,NPLAN,MOT,COG") -> dict[str, str]:
    lookup = {_norm(col): col for col in df.columns}
    out: dict[str, str] = {}
    for raw_name in parse_csv_list(requested):
        label = raw_name.upper() if raw_name.lower() == "total" else raw_name.upper()
        candidates = TARGET_ALIASES.get(label, [raw_name, label])
        column = None
        for candidate in candidates:
            if candidate in df.columns:
                column = candidate
                break
            normalized = _norm(candidate)
            if normalized in lookup:
                column = lookup[normalized]
                break
        if column is None:
            normalized_raw = _norm(raw_name)
            for col in df.columns:
                if normalized_raw and normalized_raw == _norm(col):
                    column = col
                    break
        if column is not None and label not in out:
            out[label] = column
    return out


def canonical_metric_columns(df: pd.DataFrame, prefixes: Iterable[str] = ("mean_", "std_", "global_")) -> list[str]:
    cols: list[str] = []
    for prefix in prefixes:
        for metric in MODEL_METRICS:
            col = f"{prefix}{metric}"
            if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().sum() > 0:
                cols.append(col)
    return cols


def safe_corr(x: pd.Series, y: pd.Series, method: str = "spearman") -> tuple[float, float, int]:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    n = int(mask.sum())
    if n < 3 or x[mask].nunique(dropna=True) < 2 or y[mask].nunique(dropna=True) < 2:
        return float("nan"), float("nan"), n
    if method == "pearson":
        r, p = stats.pearsonr(x[mask], y[mask])
    else:
        r, p = stats.spearmanr(x[mask], y[mask])
    return float(r), float(p), n


def correlations_by_activity_window(
    df: pd.DataFrame,
    targets: dict[str, str],
    method: str = "spearman",
    metrics: Iterable[str] = MODEL_METRICS,
    min_n: int = 3,
) -> pd.DataFrame:
    work = df.copy()
    if "valid_window" in work.columns:
        work = work[pd.to_numeric(work["valid_window"], errors="coerce").fillna(0).astype(int) == 1]
    if "_merge" in work.columns:
        work = work[work["_merge"].astype(str).eq("both")]
    if "activity_number" in work.columns:
        activity = pd.to_numeric(work["activity_number"], errors="coerce")
        work = work[activity.between(1, 7)]

    group_cols = [col for col in ["scheme_window_size", "activity_number", "activity"] if col in work.columns]
    grouped = work.groupby(group_cols, dropna=False) if group_cols else [((), work)]
    rows: list[dict] = []
    for keys, sub in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        if "code" in sub.columns:
            sub = sub.drop_duplicates(subset=["code"])
        for metric in metrics:
            col = f"mean_{metric}"
            if col not in sub.columns:
                continue
            for target_label, target_col in targets.items():
                r, p, n = safe_corr(sub[col], sub[target_col], method=method)
                if n < min_n:
                    continue
                rows.append(
                    base
                    | {
                        "metric": metric,
                        "metric_column": col,
                        "target": target_label,
                        "target_column": target_col,
                        "r": r,
                        "p": p,
                        "n": n,
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["abs_r"] = out["r"].abs()
        out = out.sort_values(["target", "scheme_window_size", "activity_number", "abs_r"], ascending=[True, True, True, False])
    return out


def profile_by_group(df: pd.DataFrame, group_cols: Iterable[str], metric_cols: Iterable[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for group in group_cols:
        if group not in df.columns:
            continue
        cols = [col for col in metric_cols if col in df.columns]
        if not cols:
            continue
        agg = df.groupby(group, dropna=False)[cols].agg(["mean", "std", "count"]).reset_index()
        agg.columns = ["_".join(str(part) for part in col if part) if isinstance(col, tuple) else str(col) for col in agg.columns]
        agg = pd.concat([pd.Series(group, index=agg.index, name="group_col"), agg], axis=1)
        frames.append(agg)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_analysis_outputs(
    combined: pd.DataFrame,
    output_dir: Path,
    targets_text: str = "Total,NPLAN,MOT,COG",
    method: str = "spearman",
    group_cols: str = "Gender,Educational level,School,School year,Age,Tipo",
) -> dict[str, Path]:
    analysis_dir = Path(output_dir) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    targets = resolve_targets(combined, targets_text)
    metric_cols = canonical_metric_columns(combined, prefixes=("mean_", "std_", "global_"))

    corr = correlations_by_activity_window(combined, targets=targets, method=method, min_n=3)
    corr_path = analysis_dir / "correlations_by_activity_window.csv"
    corr.to_csv(corr_path, index=False)
    corr_filtered = corr[pd.to_numeric(corr.get("n", pd.Series(dtype=float)), errors="coerce") >= 100].copy() if not corr.empty else corr
    corr_filtered_path = analysis_dir / "correlations_by_activity_window_min_n100.csv"
    corr_filtered.to_csv(corr_filtered_path, index=False)

    profile = profile_by_group(combined, parse_csv_list(group_cols), metric_cols)
    profile_path = analysis_dir / "profile_by_group.csv"
    profile.to_csv(profile_path, index=False)

    feature_summary = pd.DataFrame({"metric_column": metric_cols})
    feature_summary_path = analysis_dir / "canonical_metric_columns.csv"
    feature_summary.to_csv(feature_summary_path, index=False)

    return {"correlations": corr_path, "correlations_min_n100": corr_filtered_path, "profile": profile_path, "metric_columns": feature_summary_path}
