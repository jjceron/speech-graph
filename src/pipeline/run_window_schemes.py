from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.analysis import (
    BARRATT_TARGETS,
    build_activity_window_features,
    build_subject_level_features,
    canonical_feature_columns,
    correlation_table,
    profile_by_group,
    resolve_target_columns,
)
from src.analysis.stats import correlations_by_activity_window, correlations_subject_level, demographic_columns
from src.graphs import CANONICAL_METRICS
from src.pipeline.extract_graph_metrics import extract_graph_metrics, parse_csv_list, parse_int_set
from src.pipeline.merge_metadata import merge_metadata


def parse_window_sizes(text: str) -> list[int]:
    return [int(value.strip()) for value in str(text or "").split(",") if value.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run speech-graph metrics by activity and window size")
    parser.add_argument("--transcripts-dir", default="data/processed/Transcripciones")
    parser.add_argument("--metadata-xlsx", default="data/processed/df_dataset.xlsx")
    parser.add_argument("--output-dir", default="outputs/01_run")
    parser.add_argument("--window-sizes", default="10,20,30")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--lowercase", dest="lowercase", action="store_true", default=True)
    parser.add_argument("--preserve-case", dest="lowercase", action="store_false")
    parser.add_argument("--include-speakers", default="spk_1")
    parser.add_argument("--valid-activities", default="1,2,3,4,5,6,7")
    parser.add_argument("--targets", default="Total,NPLAN,MOT,COG")
    parser.add_argument("--group-cols", default="Gender,Educational level,School,School year,Tipo,Age")
    parser.add_argument("--method", default="spearman", choices=["spearman", "pearson"])
    parser.add_argument("--skip-metadata", action="store_true")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args(argv)


def _window_stability(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [f"mean_{m}" for m in CANONICAL_METRICS] + [f"std_{m}" for m in CANONICAL_METRICS]
    rows: list[dict] = []
    group_cols = [c for c in ["scheme_window_size", "activity", "activity_number"] if c in df.columns]
    for metric in metrics:
        if metric not in df.columns:
            continue
        tmp = df[group_cols].copy()
        tmp[metric] = pd.to_numeric(df[metric], errors="coerce")
        agg = tmp.groupby(group_cols, dropna=False)[metric].agg(["mean", "std", "count"]).reset_index()
        for row in agg.to_dict("records"):
            rows.append({**{c: row[c] for c in group_cols}, "metric": metric, "mean": row["mean"], "std": row["std"], "count": int(row["count"])})
    return pd.DataFrame(rows)


def _write_analysis_tables(df: pd.DataFrame, output_dir: Path, targets_text: str, group_cols_text: str, method: str) -> None:
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    targets = resolve_target_columns(df, targets_text)
    if not targets:
        print("Warning: no Barratt targets found in merged data. Correlation and model-ready tables will be limited.")

    valid = df.copy()
    if "_merge" in valid.columns:
        valid = valid[valid["_merge"].astype(str).eq("both")]
    if "valid_window" in valid.columns:
        valid = valid[pd.to_numeric(valid["valid_window"], errors="coerce").eq(1)]

    correlations_by_activity_window(valid, targets, method).to_csv(analysis_dir / "correlations_by_activity_window.csv", index=False)
    _window_stability(valid).to_csv(analysis_dir / "window_metric_stability.csv", index=False)

    metrics = canonical_feature_columns(valid, include_global=True, include_window_stats=True)
    groups = [col.strip() for col in str(group_cols_text or "").split(",") if col.strip()]
    profile_by_group(valid, groups, metrics).to_csv(analysis_dir / "profile_by_group.csv", index=False)

    activity_features = build_activity_window_features(df, targets)
    activity_features.to_csv(analysis_dir / "activity_window_features.csv", index=False)

    if targets:
        subject_features = build_subject_level_features(df, targets)
        subject_features.to_csv(analysis_dir / "subject_level_features.csv", index=False)
        correlations_subject_level(subject_features, targets, method).to_csv(analysis_dir / "correlations_subject_level.csv", index=False)
    else:
        pd.DataFrame().to_csv(analysis_dir / "subject_level_features.csv", index=False)
        pd.DataFrame().to_csv(analysis_dir / "correlations_subject_level.csv", index=False)

    demo = demographic_columns(df)
    summary = {
        "rows_total": int(len(df)),
        "rows_matched_metadata": int(df["_merge"].astype(str).eq("both").sum()) if "_merge" in df.columns else int(len(df)),
        "unique_subjects_total": int(df["code"].nunique()) if "code" in df.columns else 0,
        "unique_subjects_matched": int(df.loc[df["_merge"].astype(str).eq("both"), "code"].nunique()) if "_merge" in df.columns and "code" in df.columns else int(df["code"].nunique()) if "code" in df.columns else 0,
        "valid_rows": int(len(valid)),
        "targets_found": targets,
        "demographic_columns": demo,
        "canonical_metrics": CANONICAL_METRICS,
    }
    (analysis_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def run_from_args(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis").mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    (output_dir / "interpretability").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    window_sizes = parse_window_sizes(args.window_sizes)
    include_speakers = parse_csv_list(args.include_speakers)
    valid_activities = parse_int_set(args.valid_activities)
    frames: list[pd.DataFrame] = []

    for window_size in window_sizes:
        metrics_csv = output_dir / f"graph_metrics_w{window_size}_s{args.step}.csv"
        windows_csv = output_dir / f"graph_metrics_w{window_size}_s{args.step}_windows.csv"
        print(f"[window={window_size}] extracting metrics -> {metrics_csv}")
        frame = extract_graph_metrics(
            transcripts_dir=Path(args.transcripts_dir),
            output_csv=metrics_csv,
            window_rows_csv=windows_csv,
            invalid_activities_csv=output_dir / "analysis" / f"invalid_activities_w{window_size}.csv",
            window_size=window_size,
            step=args.step,
            allow_short=args.allow_short,
            lowercase=args.lowercase,
            include_speakers=include_speakers,
            valid_activities=valid_activities,
            max_files=args.max_files,
            progress_every=args.progress_every,
        )
        frames.append(frame)

    all_windows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    all_windows_csv = output_dir / "graph_metrics_all_windows.csv"
    all_windows.to_csv(all_windows_csv, index=False)

    combined = all_windows
    combined_csv = all_windows_csv
    if not args.skip_metadata and Path(args.metadata_xlsx).exists():
        combined_csv = output_dir / "graph_metrics_all_windows_with_meta.csv"
        combined = merge_metadata(all_windows_csv, Path(args.metadata_xlsx), combined_csv)
    elif not args.skip_metadata:
        print(f"Warning: metadata file not found: {args.metadata_xlsx}")

    _write_analysis_tables(combined, output_dir, args.targets, args.group_cols, args.method)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "transcripts_dir": str(args.transcripts_dir),
        "metadata_xlsx": str(args.metadata_xlsx),
        "output_dir": str(output_dir),
        "window_sizes": window_sizes,
        "step": args.step,
        "allow_short": bool(args.allow_short),
        "include_speakers": include_speakers,
        "valid_activities": sorted(valid_activities),
        "main_output": str(combined_csv),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done. Main output: {combined_csv}")
    print(f"Model-ready subject-level table: {output_dir / 'analysis' / 'subject_level_features.csv'}")
    return combined_csv


def main(argv: list[str] | None = None) -> None:
    run_from_args(parse_args(argv))


if __name__ == "__main__":
    main()
