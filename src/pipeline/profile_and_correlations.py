from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis import correlation_table, group_profile, numeric_columns

DEFAULT_GROUP_COLS = "Gender,Educational level,School,Tipo,level,activity"
DEFAULT_TARGET_COLS = "Age,School year,TOTAL,NPLAN,MOT,COG,Barratt (pre),TOTAL_zscore,COG_zscore,MOT_zscore"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile and correlation analysis")
    parser.add_argument(
        "--input-csv",
        default="outputs/graph_metrics_with_meta.csv",
        help="Merged CSV with metrics and metadata",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/analysis",
        help="Output directory",
    )
    parser.add_argument(
        "--method",
        default="spearman",
        choices=["spearman", "pearson"],
    )
    parser.add_argument(
        "--group-cols",
        default=DEFAULT_GROUP_COLS,
        help="Comma-separated group columns",
    )
    parser.add_argument(
        "--target-cols",
        default=DEFAULT_TARGET_COLS,
        help="Comma-separated target columns. Add 'auto' to include all numeric metadata columns.",
    )
    return parser.parse_args()


def metric_columns(df: pd.DataFrame) -> list[str]:
    prefixes = ("mean_", "std_", "emotion_", "z_")
    return [c for c in df.columns if c.startswith(prefixes)]


def target_columns(df: pd.DataFrame, requested: str, metrics: list[str]) -> list[str]:
    cols = [c.strip() for c in requested.split(",") if c.strip()]
    include_auto = "auto" in {c.lower() for c in cols}
    cols = [c for c in cols if c.lower() != "auto"]
    if include_auto:
        excluded = set(metrics) | {
            "code", "file", "level", "activity", "activity_index", "start_time", "end_time", "Cod", "_merge"
        }
        for c in numeric_columns(df, exclude=excluded):
            if c not in cols and c not in metrics:
                cols.append(c)
    return [c for c in cols if c in df.columns]


def profile_and_correlations(
    input_csv: Path,
    output_dir: Path,
    method: str = "spearman",
    group_cols: str = DEFAULT_GROUP_COLS,
    target_cols: str = DEFAULT_TARGET_COLS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv)

    metrics = metric_columns(df)
    groups = [c.strip() for c in group_cols.split(",") if c.strip()]
    targets = target_columns(df, target_cols, metrics)

    corr = correlation_table(df, metrics, targets, method=method)
    if not corr.empty:
        corr["abs_r"] = corr["r"].abs()
        corr = corr.sort_values(["abs_r", "metric", "target"], ascending=[False, True, True])
    corr.to_csv(output_dir / "correlations.csv", index=False)

    profile = group_profile(df, groups, metrics)
    profile.to_csv(output_dir / "profile_by_group.csv", index=False)
    return corr, profile


def main() -> None:
    args = parse_args()
    profile_and_correlations(
        input_csv=Path(args.input_csv),
        output_dir=Path(args.output_dir),
        method=args.method,
        group_cols=args.group_cols,
        target_cols=args.target_cols,
    )


if __name__ == "__main__":
    main()
