from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis import parse_csv_list, resolve_targets, safe_corr
from src.graphs import MODEL_METRICS


def _bh_fdr(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg FDR correction. NaN p-values remain NaN."""
    p = pd.to_numeric(p_values, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna()
    m = len(valid)
    if m == 0:
        return out
    order = valid.sort_values().index.to_list()
    ranked = valid.loc[order].to_numpy(dtype=float)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out.loc[order] = adjusted
    return out


def _numeric_or_dummies(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for col in columns:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() >= max(3, int(0.5 * len(df))):
            frames.append(pd.DataFrame({col: numeric}))
        else:
            dummies = pd.get_dummies(df[col].astype("object"), prefix=col, dummy_na=False, drop_first=True)
            if not dummies.empty:
                frames.append(dummies.astype(float))
    if not frames:
        return pd.DataFrame(index=df.index)
    return pd.concat(frames, axis=1)


def _rank_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda s: pd.Series(s).rank(method="average"), axis=0)


def _residualize(y: np.ndarray, controls: np.ndarray) -> np.ndarray:
    if controls.size == 0:
        return y - np.nanmean(y)
    x = np.column_stack([np.ones(len(y)), controls])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return y - x @ beta


def partial_corr(
    x: pd.Series,
    y: pd.Series,
    controls: pd.DataFrame,
    method: str = "spearman",
) -> tuple[float, float, int, int]:
    """Partial correlation between x and y after residualizing controls.

    For method='spearman', x, y, and controls are rank transformed before residualization.
    Returns r, p, n, number_of_control_columns.
    """
    frame = pd.concat(
        [
            pd.to_numeric(x, errors="coerce").rename("x"),
            pd.to_numeric(y, errors="coerce").rename("y"),
            controls,
        ],
        axis=1,
    ).dropna()
    n = int(len(frame))
    control_cols = [col for col in frame.columns if col not in {"x", "y"}]
    k = int(len(control_cols))
    if n < max(4, k + 3):
        return float("nan"), float("nan"), n, k
    if frame["x"].nunique(dropna=True) < 2 or frame["y"].nunique(dropna=True) < 2:
        return float("nan"), float("nan"), n, k
    work = frame.copy()
    if method == "spearman":
        work = _rank_frame(work)
    cx = work[control_cols].to_numpy(dtype=float) if control_cols else np.empty((n, 0))
    rx = _residualize(work["x"].to_numpy(dtype=float), cx)
    ry = _residualize(work["y"].to_numpy(dtype=float), cx)
    if np.nanstd(rx) == 0 or np.nanstd(ry) == 0:
        return float("nan"), float("nan"), n, k
    r = float(np.corrcoef(rx, ry)[0, 1])
    dfree = n - k - 2
    if dfree <= 0 or not np.isfinite(r) or abs(r) >= 1:
        p = float("nan")
    else:
        t = r * np.sqrt(dfree / max(1e-12, 1 - r**2))
        p = float(2 * stats.t.sf(abs(t), dfree))
    return r, p, n, k


def partial_correlations_by_activity_window(
    df: pd.DataFrame,
    targets: dict[str, str],
    control_cols: Iterable[str] = ("School year",),
    method: str = "spearman",
    metrics: Iterable[str] = MODEL_METRICS,
    min_n: int = 3,
) -> pd.DataFrame:
    work = df.copy()
    work = work.loc[:, ~work.columns.duplicated()].copy()
    if "valid_window" in work.columns:
        valid = pd.to_numeric(work["valid_window"], errors="coerce").fillna(0).astype(int)
        work = work[valid == 1]
    if "activity_number" in work.columns:
        activity = pd.to_numeric(work["activity_number"], errors="coerce")
        work = work[activity.between(1, 7)]
    if "scheme_window_size" not in work.columns and "window_size" in work.columns:
        work["scheme_window_size"] = work["window_size"]
    if "code" in work.columns:
        keys = [col for col in ["code", "activity_number", "scheme_window_size"] if col in work.columns]
        if len(keys) == 3:
            work = work.drop_duplicates(subset=keys, keep="first")

    controls_requested = [col for col in control_cols if col in work.columns]
    group_cols = [col for col in ["scheme_window_size", "activity_number", "activity"] if col in work.columns]
    grouped = work.groupby(group_cols, dropna=False) if group_cols else [((), work)]
    rows: list[dict] = []
    for keys, sub in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        controls = _numeric_or_dummies(sub, controls_requested)
        for metric in metrics:
            metric_col = metric if metric in sub.columns else f"mean_{metric}" if f"mean_{metric}" in sub.columns else None
            if metric_col is None:
                continue
            for target_label, target_col in targets.items():
                if target_col not in sub.columns:
                    continue
                simple_r, simple_p, simple_n = safe_corr(sub[metric_col], sub[target_col], method=method)
                r, p, n, k = partial_corr(sub[metric_col], sub[target_col], controls, method=method)
                if n < min_n:
                    continue
                rows.append(
                    base
                    | {
                        "metric": metric,
                        "metric_column": metric_col,
                        "target": target_label,
                        "target_column": target_col,
                        "control_columns": ",".join(controls_requested),
                        "control_design_columns": int(k),
                        "method": method,
                        "partial_r": r,
                        "partial_p": p,
                        "simple_r": simple_r,
                        "simple_p": simple_p,
                        "simple_n": simple_n,
                        "n": n,
                        "abs_partial_r": abs(r) if np.isfinite(r) else np.nan,
                        "abs_simple_r": abs(simple_r) if np.isfinite(simple_r) else np.nan,
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["partial_q_fdr_global"] = _bh_fdr(out["partial_p"])
    out["simple_q_fdr_global"] = _bh_fdr(out["simple_p"])
    out["partial_q_fdr_by_target"] = out.groupby("target", group_keys=False)["partial_p"].apply(_bh_fdr)
    out["simple_q_fdr_by_target"] = out.groupby("target", group_keys=False)["simple_p"].apply(_bh_fdr)
    return out.sort_values(["target", "scheme_window_size", "activity_number", "abs_partial_r"], ascending=[True, True, True, False])


def run_partial_correlations(
    input_csv: Path,
    output_dir: Path,
    targets_text: str = "Total,NPLAN,MOT,COG",
    control_cols_text: str = "School year",
    method: str = "spearman",
    min_n: int = 3,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv)
    targets = resolve_targets(df, targets_text)
    if not targets:
        raise ValueError(f"No target columns found for: {targets_text}")
    control_cols = parse_csv_list(control_cols_text)
    out = partial_correlations_by_activity_window(df, targets, control_cols=control_cols, method=method, min_n=min_n)
    main_path = output_dir / "partial_correlations_by_activity_window.csv"
    min_path = output_dir / f"partial_correlations_by_activity_window_min_n{min_n}.csv"
    top_path = output_dir / "partial_correlations_top.csv"
    out.to_csv(main_path, index=False)
    filtered = out[pd.to_numeric(out.get("n", pd.Series(dtype=float)), errors="coerce") >= min_n].copy() if not out.empty else out
    filtered.to_csv(min_path, index=False)
    top = filtered.sort_values("abs_partial_r", ascending=False).head(100) if not filtered.empty else filtered
    top.to_csv(top_path, index=False)
    print(f"Partial correlations saved: {main_path}")
    return {"partial": main_path, "partial_min_n": min_path, "partial_top": top_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Partial SpeechGraph-Barratt correlations controlling for metadata covariates")
    parser.add_argument("--input-csv", default="outputs/02_run/analysis/activity_window_features_02.csv")
    parser.add_argument("--output-dir", default="outputs/02_run/analysis")
    parser.add_argument("--targets", default="Total,NPLAN,MOT,COG")
    parser.add_argument("--control-cols", default="School year")
    parser.add_argument("--method", default="spearman", choices=["spearman", "pearson"])
    parser.add_argument("--min-n", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_partial_correlations(
        input_csv=Path(args.input_csv),
        output_dir=Path(args.output_dir),
        targets_text=args.targets,
        control_cols_text=args.control_cols,
        method=args.method,
        min_n=args.min_n,
    )


if __name__ == "__main__":
    main()
