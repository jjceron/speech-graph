from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.analysis import correlation_table, group_profile, numeric_columns
from src.pipeline.extract_graph_metrics import extract_graph_metrics
from src.pipeline.merge_metadata import merge_metadata
from src.pipeline.profile_and_correlations import DEFAULT_GROUP_COLS, DEFAULT_TARGET_COLS, metric_columns, target_columns


def parse_window_sizes(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 10/20/30-word window schemes and consolidate outputs."
    )
    parser.add_argument("--transcripts-dir", default="data/processed/Transcripciones")
    parser.add_argument("--metadata-xlsx", default="data/processed/df_dataset.xlsx")
    parser.add_argument("--output-dir", default="outputs/window_schemes")
    parser.add_argument("--window-sizes", default="10,20,30")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--no-allow-short", action="store_true")
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--include-speakers", default="spk_1")
    parser.add_argument("--by-activity", action="store_true")
    parser.add_argument("--lexicon-path", default=None)
    parser.add_argument("--random-times", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--random-metrics", default="lcc,lsc,edges,repeated_edges,density,asp,l1,l2,l3")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--method", default="spearman", choices=["spearman", "pearson"])
    parser.add_argument("--group-cols", default=DEFAULT_GROUP_COLS)
    parser.add_argument(
        "--target-cols",
        default=DEFAULT_TARGET_COLS + ",auto",
        help="Comma-separated target columns. Default includes Barratt scores/questions via auto.",
    )
    parser.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Do not merge metadata, useful when df_dataset.xlsx is unavailable.",
    )
    return parser.parse_args(argv)


def add_window_columns(df: pd.DataFrame, window_size: int, step: int, random_times: int) -> pd.DataFrame:
    df = df.copy()
    df.insert(0, "window_size", window_size)
    df.insert(1, "window_step", step)
    df.insert(2, "random_times", random_times)
    return df


def compare_window_stability(df: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    """Summarise metric values per window scheme and level/activity."""
    group_cols = ["window_size"]
    for optional in ["level", "activity"]:
        if optional in df.columns:
            group_cols.append(optional)

    rows: list[dict] = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        values = pd.to_numeric(df[metric], errors="coerce")
        if values.notna().sum() == 0:
            continue
        tmp = df[group_cols].copy()
        tmp[metric] = values
        agg = tmp.groupby(group_cols, dropna=False)[metric].agg(["mean", "std", "count"]).reset_index()
        for _, row in agg.iterrows():
            out = {col: row[col] for col in group_cols}
            out.update({"metric": metric, "mean": row["mean"], "std": row["std"], "count": int(row["count"])})
            rows.append(out)
    return pd.DataFrame(rows)


def analyze_combined(
    df: pd.DataFrame,
    output_dir: Path,
    method: str,
    group_cols: str,
    target_cols_text: str,
) -> None:
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    metrics = metric_columns(df)
    targets = target_columns(df, target_cols_text, metrics)
    groups = parse_csv_list(group_cols)

    corr_frames: list[pd.DataFrame] = []
    for window_size, sub in df.groupby("window_size", dropna=False):
        corr = correlation_table(sub, metrics, targets, method=method)
        if not corr.empty:
            corr.insert(0, "window_size", window_size)
            corr["abs_r"] = corr["r"].abs()
            corr_frames.append(corr)
    corr_by_window = (
        pd.concat(corr_frames, ignore_index=True)
        if corr_frames
        else pd.DataFrame(columns=["window_size", "metric", "target", "r", "p", "n", "abs_r"])
    )
    if not corr_by_window.empty:
        corr_by_window = corr_by_window.sort_values(
            ["abs_r", "window_size", "metric", "target"], ascending=[False, True, True, True]
        )
    corr_by_window.to_csv(analysis_dir / "correlations_by_window.csv", index=False)

    profile_frames: list[pd.DataFrame] = []
    for window_size, sub in df.groupby("window_size", dropna=False):
        prof = group_profile(sub, groups, metrics)
        if not prof.empty:
            prof.insert(0, "window_size", window_size)
            profile_frames.append(prof)
    profile = (
        pd.concat(profile_frames, ignore_index=True)
        if profile_frames
        else pd.DataFrame(columns=["window_size", "group_col"])
    )
    profile.to_csv(analysis_dir / "profile_by_group_and_window.csv", index=False)

    stability = compare_window_stability(df, metrics)
    stability.to_csv(analysis_dir / "window_metric_stability.csv", index=False)


def run_from_args(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    window_sizes = parse_window_sizes(args.window_sizes)
    include_speakers = parse_csv_list(args.include_speakers)
    random_metrics = parse_csv_list(args.random_metrics)
    allow_short = not args.no_allow_short
    lexicon_path = Path(args.lexicon_path) if args.lexicon_path else None

    per_window_frames: list[pd.DataFrame] = []

    for window_size in window_sizes:
        metrics_csv = output_dir / f"graph_metrics_w{window_size}_s{args.step}.csv"
        print(f"[window={window_size}] extracting metrics -> {metrics_csv}")
        df = extract_graph_metrics(
            transcripts_dir=Path(args.transcripts_dir),
            output_csv=metrics_csv,
            window_size=window_size,
            step=args.step,
            allow_short=allow_short,
            lowercase=args.lowercase,
            include_speakers=include_speakers,
            by_activity=args.by_activity,
            lexicon_path=lexicon_path,
            random_times=args.random_times,
            random_seed=args.random_seed,
            random_metrics=random_metrics,
            max_files=args.max_files,
            progress_every=args.progress_every,
        )
        per_window_frames.append(add_window_columns(df, window_size, args.step, args.random_times))

    all_windows = pd.concat(per_window_frames, ignore_index=True) if per_window_frames else pd.DataFrame()
    all_windows_csv = output_dir / "graph_metrics_all_windows.csv"
    all_windows.to_csv(all_windows_csv, index=False)

    combined = all_windows
    if not args.skip_metadata and Path(args.metadata_xlsx).exists():
        combined_csv = output_dir / "graph_metrics_all_windows_with_meta.csv"
        combined = merge_metadata(all_windows_csv, Path(args.metadata_xlsx), combined_csv)
    else:
        combined_csv = all_windows_csv

    analyze_combined(
        combined,
        output_dir=output_dir,
        method=args.method,
        group_cols=args.group_cols,
        target_cols_text=args.target_cols,
    )
    print(f"Done. Main output: {combined_csv}")
    return combined_csv


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_from_args(args)


if __name__ == "__main__":
    main()
