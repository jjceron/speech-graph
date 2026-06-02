from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import parse_csv_list, parse_int_set, resolve_targets, write_analysis_outputs
from src.graphs import MODEL_METRICS
from src.io.processed_tasks import save_processed_tasks
from src.pipeline.extract_graph_metrics import DEFAULT_RANDOM_METRICS, extract_graph_metrics, parse_activity_window_sizes
from src.pipeline.merge_metadata import merge_metadata

DEFAULT_GROUP_COLS = "Gender,Educational level,School,School year,Age,Tipo"
DEFAULT_TARGETS = "Total,NPLAN,MOT,COG"


def parse_window_sizes(text: str | None) -> list[int]:
    values = [int(part.strip()) for part in str(text or "").split(",") if part.strip()]
    if not values:
        raise ValueError("At least one window size is required")
    return values


def add_scheme_columns(df: pd.DataFrame, window_size: int, step: int, random_times: int) -> pd.DataFrame:
    out = df.reset_index(drop=True).copy()
    out = out.loc[:, ~out.columns.duplicated()].copy()
    out["scheme_window_size"] = int(window_size)
    out["window_step"] = int(step)
    out["random_times"] = int(random_times)
    first = ["scheme_window_size", "window_size", "window_step", "random_times"]
    ordered = [col for col in first if col in out.columns]
    ordered += [col for col in out.columns if col not in ordered]
    return out[ordered]


def _metadata_columns(df: pd.DataFrame, target_cols: list[str]) -> list[str]:
    preferred = [
        "code", "Cod", "Gender", "Age", "Edad", "School year", "Educational level",
        "Escolaridad", "School", "Tipo", "Grupo", "Group",
    ]
    cols = [col for col in preferred if col in df.columns]
    for col in target_cols:
        if col in df.columns and col not in cols:
            cols.append(col)
    return cols


def _base_model_rows(df: pd.DataFrame, require_valid_window: bool) -> pd.DataFrame:
    work = df.copy()
    if "_merge" in work.columns:
        work = work[work["_merge"].astype(str).eq("both")]
    if "activity_number" in work.columns:
        activity = pd.to_numeric(work["activity_number"], errors="coerce")
        work = work[activity.between(1, 7)]
    if require_valid_window and "valid_window" in work.columns:
        valid = pd.to_numeric(work["valid_window"], errors="coerce").fillna(0).astype(int)
        work = work[valid == 1]
    return work.copy()


def _filtered_model_rows(df: pd.DataFrame) -> pd.DataFrame:
    return _base_model_rows(df, require_valid_window=True)


def _first_metadata_by_subject(work: pd.DataFrame, targets_text: str) -> pd.DataFrame:
    target_map = resolve_targets(work, targets_text)
    meta_cols = _metadata_columns(work, list(target_map.values()))
    if "code" not in meta_cols:
        meta_cols.insert(0, "code")
    return work.sort_values("code").drop_duplicates("code")[meta_cols].copy()


def _metadata_for_rows(work: pd.DataFrame, targets_text: str) -> list[str]:
    target_map = resolve_targets(work, targets_text)
    meta_cols = _metadata_columns(work, list(target_map.values()))
    for col in ["code", "activity", "activity_number", "scheme_window_size", "window_size", "valid_window", "window_count"]:
        if col in work.columns and col not in meta_cols:
            meta_cols.append(col)
    return meta_cols


def _numeric_value(row: pd.Series, col: str) -> float:
    value = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else np.nan


def build_subject_level_by_activity(combined: pd.DataFrame, targets_text: str = DEFAULT_TARGETS) -> pd.DataFrame:
    """One row per subject with global graph metrics per activity."""
    work = _base_model_rows(combined, require_valid_window=False)
    if work.empty:
        return pd.DataFrame()
    meta = _first_metadata_by_subject(work, targets_text)
    rows: list[dict] = []
    ordered = work.sort_values([col for col in ["code", "activity_number", "scheme_window_size"] if col in work.columns])
    for code, sub_code in ordered.groupby("code", dropna=False):
        row: dict[str, float | str] = {"code": code}
        for activity_number, sub_activity in sub_code.groupby("activity_number", dropna=False):
            try:
                activity = int(activity_number)
            except Exception:
                continue
            item = sub_activity.iloc[0]
            for metric in MODEL_METRICS:
                source = f"global_{metric}" if f"global_{metric}" in sub_activity.columns else f"mean_{metric}"
                values = pd.to_numeric(sub_activity[source], errors="coerce") if source in sub_activity.columns else pd.Series(dtype=float)
                row[f"a{activity}_{metric}"] = float(values.dropna().iloc[0]) if values.notna().any() else np.nan
        rows.append(row)
    features = pd.DataFrame(rows)
    out = meta.merge(features, on="code", how="left")
    feature_cols = [col for col in out.columns if col.startswith("a") and "_" in col]
    keep_features = [col for col in feature_cols if pd.to_numeric(out[col], errors="coerce").nunique(dropna=True) > 1]
    return out[[col for col in out.columns if col not in feature_cols] + keep_features]


def build_activity_window_features(combined: pd.DataFrame, targets_text: str = DEFAULT_TARGETS) -> pd.DataFrame:
    """Long matrix: one row per subject, activity and window size using canonical window metrics."""
    work = _filtered_model_rows(combined)
    if work.empty:
        return pd.DataFrame()
    meta_cols = _metadata_for_rows(work, targets_text)
    rows: list[dict] = []
    for _, item in work.iterrows():
        row = {col: item.get(col) for col in meta_cols if col in work.columns}
        row["window_size"] = int(item.get("scheme_window_size", item.get("window_size")))
        row["scheme_window_size"] = row["window_size"]
        row["activity_number"] = int(item.get("activity_number"))
        row["activity"] = item.get("activity", f"Actividad{row['activity_number']}")
        for metric in MODEL_METRICS:
            source = f"mean_{metric}"
            row[metric] = _numeric_value(item, source) if source in work.columns else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    first = [col for col in ["code", "Cod", "activity", "activity_number", "window_size", "scheme_window_size"] if col in out.columns]
    rest = [col for col in out.columns if col not in first]
    return out[first + rest]


def build_subject_level_windowed(combined: pd.DataFrame, targets_text: str = DEFAULT_TARGETS) -> pd.DataFrame:
    """Wide exploratory matrix: w{window}_a{activity}_{metric}."""
    work = _filtered_model_rows(combined)
    if work.empty:
        return pd.DataFrame()
    meta = _first_metadata_by_subject(work, targets_text)
    rows: list[dict] = []
    for code, sub_code in work.groupby("code", dropna=False):
        row: dict[str, float | str] = {"code": code}
        for _, item in sub_code.iterrows():
            window = int(item.get("scheme_window_size", item.get("window_size")))
            activity = int(item["activity_number"])
            prefix = f"w{window}_a{activity}"
            for metric in MODEL_METRICS:
                col = f"mean_{metric}"
                row[f"{prefix}_{metric}"] = _numeric_value(item, col) if col in sub_code.columns else np.nan
        rows.append(row)
    features = pd.DataFrame(rows)
    out = meta.merge(features, on="code", how="left")
    feature_cols = [col for col in out.columns if col.startswith("w") and "_a" in col]
    keep_features = [col for col in feature_cols if pd.to_numeric(out[col], errors="coerce").nunique(dropna=True) > 1]
    return out[[col for col in out.columns if col not in feature_cols] + keep_features]


def build_subject_level_full(combined: pd.DataFrame, targets_text: str = DEFAULT_TARGETS) -> pd.DataFrame:
    work = _filtered_model_rows(combined)
    if work.empty:
        return pd.DataFrame()
    meta = _first_metadata_by_subject(work, targets_text)
    metric_cols = [f"{prefix}{metric}" for prefix in ("mean_", "std_", "global_") for metric in MODEL_METRICS if f"{prefix}{metric}" in work.columns]
    rows: list[dict] = []
    for code, sub_code in work.groupby("code", dropna=False):
        row: dict[str, float | str] = {"code": code}
        for _, item in sub_code.iterrows():
            window = int(item.get("scheme_window_size", item.get("window_size")))
            activity = int(item["activity_number"])
            prefix = f"w{window}_a{activity}"
            for col in metric_cols:
                row[f"{prefix}_{col}"] = _numeric_value(item, col)
        rows.append(row)
    features = pd.DataFrame(rows)
    out = meta.merge(features, on="code", how="left")
    feature_cols = [col for col in out.columns if col.startswith("w") and "_a" in col]
    keep_features = [col for col in feature_cols if pd.to_numeric(out[col], errors="coerce").nunique(dropna=True) > 1]
    return out[[col for col in out.columns if col not in feature_cols] + keep_features]


def save_subject_matrices(combined: pd.DataFrame, output_dir: Path, targets_text: str) -> dict[str, Path]:
    analysis_dir = Path(output_dir) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    by_activity = build_subject_level_by_activity(combined, targets_text=targets_text)
    activity_window = build_activity_window_features(combined, targets_text=targets_text)
    windowed = build_subject_level_windowed(combined, targets_text=targets_text)
    full = build_subject_level_full(combined, targets_text=targets_text)

    by_activity_path = analysis_dir / "subject_level_features.csv"
    activity_window_path = analysis_dir / "activity_window_features.csv"
    windowed_path = analysis_dir / "subject_level_features_windowed.csv"
    full_path = analysis_dir / "subject_level_features_full.csv"

    by_activity.to_csv(by_activity_path, index=False)
    activity_window.to_csv(activity_window_path, index=False)
    windowed.to_csv(windowed_path, index=False)
    full.to_csv(full_path, index=False)

    manifest = pd.DataFrame(
        [
            {"matrix": "subject_level_features", "purpose": "global metrics by activity", "path": str(by_activity_path), "rows": len(by_activity), "columns": len(by_activity.columns)},
            {"matrix": "activity_window_features", "purpose": "main activity x window modelling matrix", "path": str(activity_window_path), "rows": len(activity_window), "columns": len(activity_window.columns)},
            {"matrix": "subject_level_features_windowed", "purpose": "wide exploratory activity x window matrix", "path": str(windowed_path), "rows": len(windowed), "columns": len(windowed.columns)},
            {"matrix": "subject_level_features_full", "purpose": "wide exploratory mean/std/global matrix", "path": str(full_path), "rows": len(full), "columns": len(full.columns)},
        ]
    )
    manifest_path = analysis_dir / "feature_matrices_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return {"by_activity": by_activity_path, "activity_window": activity_window_path, "windowed": windowed_path, "full": full_path, "manifest": manifest_path}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run activity-level speech graph extraction across window schemes")
    parser.add_argument("--transcripts-dir", default="data/processed/Transcripciones")
    parser.add_argument("--metadata-xlsx", default="data/processed/df_dataset.xlsx")
    parser.add_argument("--output-dir", default="outputs/01_run")
    parser.add_argument("--window-sizes", default="10,20,30")
    parser.add_argument("--activity-window-sizes", default="")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--lowercase", dest="lowercase", action="store_true", default=True)
    parser.add_argument("--preserve-case", dest="lowercase", action="store_false")
    parser.add_argument("--include-speakers", default="spk_1")
    parser.add_argument("--valid-activities", default="1,2,3,4,5,6,7")
    parser.add_argument("--by-activity", action="store_true", default=True)
    parser.add_argument("--random-times", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--random-metrics", default=DEFAULT_RANDOM_METRICS)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--method", default="spearman", choices=["spearman", "pearson"])
    parser.add_argument("--group-cols", default=DEFAULT_GROUP_COLS)
    parser.add_argument("--target-cols", default=DEFAULT_TARGETS)
    parser.add_argument("--skip-metadata", action="store_true")
    parser.add_argument("--save-task-ansi", action="store_true")
    parser.add_argument("--save-task-sw", action="store_true")
    return parser.parse_args(argv)


def run_from_args(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    window_sizes = parse_window_sizes(args.window_sizes)
    include_speakers = parse_csv_list(args.include_speakers)
    valid_activities = parse_int_set(args.valid_activities)
    random_metrics = parse_csv_list(args.random_metrics)
    activity_windows = parse_activity_window_sizes(args.activity_window_sizes)

    if args.save_task_ansi or args.save_task_sw:
        manifest = save_processed_tasks(
            transcripts_dir=Path(args.transcripts_dir),
            output_dir=output_dir,
            include_speakers=include_speakers,
            valid_activities=valid_activities,
            lowercase=args.lowercase,
            save_ansi=args.save_task_ansi,
            save_sw=args.save_task_sw,
            max_files=args.max_files,
        )
        print(f"Processed task texts saved: {output_dir / 'activities_processed'} ({len(manifest)} rows)")

    frames: list[pd.DataFrame] = []
    for window_size in window_sizes:
        metrics_csv = output_dir / f"graph_metrics_w{window_size}_s{args.step}.csv"
        windows_csv = output_dir / f"graph_metrics_w{window_size}_s{args.step}_windows.csv"
        print(f"[window={window_size}] extracting metrics -> {metrics_csv}")
        df = extract_graph_metrics(
            transcripts_dir=Path(args.transcripts_dir),
            output_csv=metrics_csv,
            window_rows_csv=windows_csv,
            window_size=window_size,
            step=args.step,
            allow_short=args.allow_short,
            lowercase=args.lowercase,
            include_speakers=include_speakers,
            valid_activities=valid_activities,
            random_times=args.random_times,
            random_seed=args.random_seed,
            random_metrics=random_metrics,
            max_files=args.max_files,
            progress_every=args.progress_every,
            activity_window_sizes=activity_windows,
        )
        frames.append(add_scheme_columns(df, window_size, args.step, args.random_times))

    all_windows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    all_windows_csv = output_dir / "graph_metrics_all_windows.csv"
    all_windows.to_csv(all_windows_csv, index=False)

    combined = all_windows
    combined_csv = all_windows_csv
    if not args.skip_metadata and Path(args.metadata_xlsx).exists():
        combined_csv = output_dir / "graph_metrics_all_windows_with_meta.csv"
        combined = merge_metadata(all_windows_csv, Path(args.metadata_xlsx), combined_csv)

    write_analysis_outputs(combined, output_dir, targets_text=args.target_cols, method=args.method, group_cols=args.group_cols)
    matrix_paths = save_subject_matrices(combined, output_dir, targets_text=args.target_cols)

    run_manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "transcripts_dir": str(args.transcripts_dir),
        "metadata_xlsx": str(args.metadata_xlsx),
        "output_dir": str(output_dir),
        "window_sizes": window_sizes,
        "step": int(args.step),
        "allow_short": bool(args.allow_short),
        "include_speakers": include_speakers,
        "valid_activities": sorted(valid_activities),
        "save_task_ansi": bool(args.save_task_ansi),
        "save_task_sw": bool(args.save_task_sw),
        "main_output": str(combined_csv),
        "subject_level_features": str(matrix_paths["by_activity"]),
        "activity_window_features": str(matrix_paths["activity_window"]),
        "subject_level_features_windowed": str(matrix_paths["windowed"]),
        "subject_level_features_full": str(matrix_paths["full"]),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Done. Main output: {combined_csv}")
    print(f"Subject-level features: {matrix_paths['by_activity']}")
    print(f"Activity-window features: {matrix_paths['activity_window']}")
    return combined_csv


def main(argv: list[str] | None = None) -> None:
    run_from_args(parse_args(argv))


if __name__ == "__main__":
    main()
