from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.profile_preprocessing import (
    DEFAULT_SPEECHGRAPH_METRICS,
    first_existing_column,
    parse_csv_list,
)


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


def _standardize_activity_table(
    df: pd.DataFrame,
    activity_code_col: str | None = None,
) -> tuple[pd.DataFrame, str]:
    df = df.loc[:, ~df.columns.duplicated()].copy()
    code_col = activity_code_col or first_existing_column(df, ["code", "Cod", "COD", "subject", "id"])
    if code_col is None:
        raise ValueError("Could not find subject code column in activity-window CSV.")
    if code_col != "code":
        df = df.rename(columns={code_col: "code"})
        code_col = "code"
    if "scheme_window_size" not in df.columns and "window_size" in df.columns:
        df["scheme_window_size"] = df["window_size"]
    if "window_size" not in df.columns and "scheme_window_size" in df.columns:
        df["window_size"] = df["scheme_window_size"]
    required = ["code", "activity_number", "window_size"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Activity-window CSV is missing required columns: {missing}")
    if "valid_window" in df.columns:
        valid = pd.to_numeric(df["valid_window"], errors="coerce").fillna(0).astype(int)
        df = df[valid == 1].copy()
    df["activity_number"] = pd.to_numeric(df["activity_number"], errors="coerce")
    df["window_size"] = pd.to_numeric(df["window_size"], errors="coerce")
    df = df[df["activity_number"].notna() & df["window_size"].notna()].copy()
    df["activity_number"] = df["activity_number"].astype(int)
    df["window_size"] = df["window_size"].astype(int)
    return df, code_col


def _speechgraph_wide(activity_df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows: dict[str, dict[str, float]] = {}
    key_cols = ["code", "activity_number", "window_size"]
    for (code, activity, window), group in activity_df.groupby(key_cols, dropna=False):
        code = str(code)
        rows.setdefault(code, {"code": code})
        for metric in metrics:
            if metric not in group.columns:
                continue
            col = f"A{int(activity)}_W{int(window)}_{metric}"
            rows[code][col] = _weighted_average(group, metric)
    out = pd.DataFrame(rows.values())
    if out.empty:
        return pd.DataFrame(columns=["code"])
    first = ["code"]
    rest = sorted([c for c in out.columns if c != "code"], key=lambda x: (int(x.split("_")[0][1:]), int(x.split("_")[1][1:]), x))
    return out[first + rest]


def build_subject_profile_features(
    activity_window_csv: str | Path,
    metadata_xlsx: str | Path,
    output_dir: str | Path,
    activity_code_col: str | None = None,
    metadata_code_col: str | None = None,
    metrics_text: str | None = None,
) -> dict[str, Path]:
    """Build one-row-per-subject multimodal feature table for profile analysis."""
    activity_window_csv = Path(activity_window_csv)
    metadata_xlsx = Path(metadata_xlsx)
    output_dir = Path(output_dir)
    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    activity_df = pd.read_csv(activity_window_csv)
    activity_df, _ = _standardize_activity_table(activity_df, activity_code_col=activity_code_col)
    metrics = parse_csv_list(metrics_text) if metrics_text else [m for m in DEFAULT_SPEECHGRAPH_METRICS if m in activity_df.columns]
    if not metrics:
        raise ValueError("No SpeechGraph metric columns were found in activity-window CSV.")
    wide_graph = _speechgraph_wide(activity_df, metrics)

    metadata = pd.read_excel(metadata_xlsx)
    metadata = metadata.loc[:, ~metadata.columns.duplicated()].copy()
    meta_code = metadata_code_col or first_existing_column(metadata, ["code", "Cod", "COD", "subject", "id"])
    if meta_code is None:
        raise ValueError("Could not find subject code column in metadata XLSX.")
    if meta_code != "code":
        metadata = metadata.rename(columns={meta_code: "code"})
    metadata["code"] = metadata["code"].astype(str)
    wide_graph["code"] = wide_graph["code"].astype(str)

    # Preserve all metadata columns because Barratt outcomes are needed for external validation.
    subject = metadata.merge(wide_graph, on="code", how="left", validate="one_to_one")
    subject = subject.loc[:, ~subject.columns.duplicated()].copy()

    out_csv = features_dir / "subject_level_multimodal_features.csv"
    graph_csv = features_dir / "subject_level_speechgraph_wide.csv"
    manifest_path = features_dir / "subject_profile_feature_manifest.json"
    subject.to_csv(out_csv, index=False)
    wide_graph.to_csv(graph_csv, index=False)

    graph_cols = [c for c in subject.columns if str(c).startswith("A") and "_W" in str(c)]
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "activity_window_csv": str(activity_window_csv),
        "metadata_xlsx": str(metadata_xlsx),
        "output_csv": str(out_csv),
        "n_subjects_metadata": int(len(metadata)),
        "n_subjects_with_any_speechgraph": int(subject[graph_cols].notna().any(axis=1).sum()) if graph_cols else 0,
        "n_speechgraph_features": int(len(graph_cols)),
        "metrics_used": metrics,
        "metadata_columns": [c for c in metadata.columns if c != "code"],
        "speechgraph_columns": graph_cols,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Subject-level multimodal features saved: {out_csv}")
    print(f"SpeechGraph wide table saved: {graph_csv}")
    return {"subject_features": out_csv, "speechgraph_wide": graph_csv, "manifest": manifest_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build subject-level multimodal profile features for 03_run.")
    parser.add_argument("--activity-window-csv", required=True)
    parser.add_argument("--metadata-xlsx", required=True)
    parser.add_argument("--output-dir", default="outputs/03_run")
    parser.add_argument("--activity-code-col", default=None)
    parser.add_argument("--metadata-code-col", default=None)
    parser.add_argument("--metrics", default=None, help="Optional comma-separated SpeechGraph metrics to pivot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_subject_profile_features(
        activity_window_csv=args.activity_window_csv,
        metadata_xlsx=args.metadata_xlsx,
        output_dir=args.output_dir,
        activity_code_col=args.activity_code_col,
        metadata_code_col=args.metadata_code_col,
        metrics_text=args.metrics,
    )


if __name__ == "__main__":
    main()
