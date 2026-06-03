from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import parse_csv_list, resolve_targets
from src.analysis.partial_correlations import run_partial_correlations
from src.graphs import MODEL_METRICS
from src.models.montecarlo_cv import parse_alphas, run_monte_carlo_cv


def _first_non_null(series: pd.Series):
    vals = series.dropna()
    return vals.iloc[0] if len(vals) else np.nan


def _weighted_average(group: pd.DataFrame, col: str, weight_col: str = "window_count") -> float:
    values = pd.to_numeric(group[col], errors="coerce")
    if values.notna().sum() == 0:
        return np.nan
    if weight_col in group.columns:
        weights = pd.to_numeric(group[weight_col], errors="coerce").fillna(0)
        valid = values.notna() & (weights > 0)
        if valid.any():
            return float(np.average(values[valid], weights=weights[valid]))
    return float(values.mean())


def prepare_activity_window_features(
    input_csv: Path,
    output_dir: Path,
    targets_text: str = "Total,NPLAN,MOT,COG",
    control_cols_text: str = "School year",
    group_col: str = "code",
) -> dict[str, Path]:
    """Create a clean 02_run modelling matrix without modifying the 01_run files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    if "scheme_window_size" not in df.columns and "window_size" in df.columns:
        df["scheme_window_size"] = df["window_size"]
    if "window_size" not in df.columns and "scheme_window_size" in df.columns:
        df["window_size"] = df["scheme_window_size"]
    if "valid_window" in df.columns:
        valid = pd.to_numeric(df["valid_window"], errors="coerce").fillna(0).astype(int)
        df = df[valid == 1].copy()
    if "activity_number" in df.columns:
        activity = pd.to_numeric(df["activity_number"], errors="coerce")
        df = df[activity.between(1, 7)].copy()
        df["activity_number"] = activity[activity.between(1, 7)].astype(int)
    if "window_size" in df.columns:
        df["window_size"] = pd.to_numeric(df["window_size"], errors="coerce").astype("Int64")
    if "scheme_window_size" in df.columns:
        df["scheme_window_size"] = pd.to_numeric(df["scheme_window_size"], errors="coerce").astype("Int64")

    target_map = resolve_targets(df, targets_text)
    control_cols = [col for col in parse_csv_list(control_cols_text) if col in df.columns]
    key_cols = [col for col in [group_col, "activity_number", "window_size"] if col in df.columns]
    if len(key_cols) < 3:
        raise ValueError("Input CSV must contain code, activity_number, and window_size/scheme_window_size")

    duplicate_mask = df.duplicated(key_cols, keep=False)
    duplicate_report = df.loc[duplicate_mask].sort_values(key_cols).copy()
    duplicate_report_path = output_dir / "activity_window_duplicate_rows_02.csv"
    duplicate_report.to_csv(duplicate_report_path, index=False)

    metric_cols = [m for m in MODEL_METRICS if m in df.columns]
    first_preferred = [
        group_col, "Cod", "activity", "activity_number", "window_size", "scheme_window_size",
        "Gender", "Age", "Edad", "School year", "Educational level", "Escolaridad", "School", "Tipo",
    ]
    first_cols = []
    for col in first_preferred + list(target_map.values()) + control_cols:
        if col in df.columns and col not in first_cols and col not in metric_cols:
            first_cols.append(col)

    rows: list[dict] = []
    for _, group in df.groupby(key_cols, dropna=False):
        row = {col: _first_non_null(group[col]) for col in first_cols if col in group.columns}
        for metric in metric_cols:
            row[metric] = _weighted_average(group, metric)
        if "window_count" in group.columns:
            row["window_count"] = int(pd.to_numeric(group["window_count"], errors="coerce").fillna(0).sum())
        if "valid_window" in group.columns:
            row["valid_window"] = int(pd.to_numeric(group["valid_window"], errors="coerce").fillna(0).max())
        rows.append(row)
    out = pd.DataFrame(rows)
    for col in ["activity_number", "window_size", "scheme_window_size"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    first = [col for col in [group_col, "Cod", "activity", "activity_number", "window_size", "scheme_window_size"] if col in out.columns]
    rest = [col for col in out.columns if col not in first]
    out = out[first + rest]
    output_csv = output_dir / "activity_window_features_02.csv"
    out.to_csv(output_csv, index=False)

    summary = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "rows_input_after_filtering": int(len(df)),
        "rows_output": int(len(out)),
        "duplicate_rows_reported": int(len(duplicate_report)),
        "duplicated_keys_after_prepare": int(out.duplicated(key_cols).sum()) if set(key_cols).issubset(out.columns) else None,
        "targets_resolved": target_map,
        "control_cols_found": control_cols,
        "metric_cols": metric_cols,
    }
    summary_path = output_dir / "activity_window_features_02_manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"02_run modelling matrix saved: {output_csv}")
    print(f"Duplicate report saved: {duplicate_report_path} ({len(duplicate_report)} rows)")
    return {"features": output_csv, "duplicates": duplicate_report_path, "manifest": summary_path}


def run_02_from_args(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = Path(args.output_dir)
    analysis_dir = output_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare_activity_window_features(
        input_csv=Path(args.input_csv),
        output_dir=analysis_dir,
        targets_text=args.targets,
        control_cols_text=args.control_cols,
        group_col=args.group_col,
    )

    partial_paths = run_partial_correlations(
        input_csv=prepared["features"],
        output_dir=analysis_dir,
        targets_text=args.targets,
        control_cols_text=args.control_cols,
        method=args.correlation_method,
        min_n=args.min_n,
    )

    mc_paths = run_monte_carlo_cv(
        input_csv=prepared["features"],
        output_dir=output_dir,
        targets_text=args.targets,
        control_cols_text=args.control_cols,
        model_sets_text=args.model_sets,
        n_repeats=args.n_repeats,
        test_size=args.test_size,
        alphas=parse_alphas(args.alphas),
        random_state=args.random_state,
        group_col=args.group_col,
        min_n=args.min_model_n,
        save_predictions=args.save_predictions,
    )

    plot_paths = {}
    if not args.skip_plots:
        try:
            from src.visualization.plotting_02 import generate_figures_02

            count = generate_figures_02(output_dir)
            plot_paths = {"figures_dir": output_dir / "figures", "figure_count": count}
        except Exception as exc:
            print(f"WARNING: 02_run figures were not generated: {exc}")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_csv": str(args.input_csv),
        "output_dir": str(output_dir),
        "targets": args.targets,
        "control_cols": args.control_cols,
        "correlation_method": args.correlation_method,
        "n_repeats": int(args.n_repeats),
        "test_size": float(args.test_size),
        "model_sets": args.model_sets,
        "alphas": args.alphas,
        "random_state": int(args.random_state),
        "prepared": {k: str(v) for k, v in prepared.items()},
        "partial_correlations": {k: str(v) for k, v in partial_paths.items()},
        "monte_carlo_cv": {k: str(v) for k, v in mc_paths.items()},
        "plots": {k: str(v) for k, v in plot_paths.items()},
    }
    manifest_path = output_dir / "run_02_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"02_run completed. Manifest: {manifest_path}")
    return {"manifest": manifest_path, **prepared, **partial_paths, **mc_paths}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 02 analysis: partial correlations + Monte Carlo CV + exact linear SHAP")
    parser.add_argument("--input-csv", default="outputs/01_run/analysis/activity_window_features.csv")
    parser.add_argument("--output-dir", default="outputs/02_run")
    parser.add_argument("--targets", default="Total,NPLAN,MOT,COG")
    parser.add_argument("--control-cols", default="School year")
    parser.add_argument("--correlation-method", default="spearman", choices=["spearman", "pearson"])
    parser.add_argument("--model-sets", default="nlp,school_year,school_year_nlp")
    parser.add_argument("--n-repeats", type=int, default=400)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--alphas", default="0.1,1,10,100,1000")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--group-col", default="code")
    parser.add_argument("--min-n", type=int, default=100, help="Minimum n for partial-correlation output filtering")
    parser.add_argument("--min-model-n", type=int, default=30, help="Minimum n to fit one Monte Carlo model")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_02_from_args(parse_args())


if __name__ == "__main__":
    main()
