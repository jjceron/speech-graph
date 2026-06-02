from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from src.analysis.stats import add_standard_target_columns
from src.io import normalize_code


def _norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def find_metadata_code_column(meta: pd.DataFrame) -> str:
    preferred = ["Cod", "Código", "Codigo", "Code", "ID", "Subject", "Sujeto"]
    lookup = {_norm(col): col for col in meta.columns}
    for name in preferred:
        if _norm(name) in lookup:
            return lookup[_norm(name)]
    for col in meta.columns:
        key = _norm(col)
        if key in {"cod", "codigo", "code"} or "codigo" in key:
            return col
    raise ValueError("Metadata Excel must include a subject code column such as Cod, Código, Codigo, or Code.")


def _parse_sheet_name(value: str) -> str | int:
    return int(value) if str(value).isdigit() else value


def merge_metadata(metrics_csv: Path, metadata_xlsx: Path, output_csv: Path, sheet_name: str | int = 0) -> pd.DataFrame:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(metrics_csv).copy()
    meta = pd.read_excel(metadata_xlsx, sheet_name=sheet_name, engine="openpyxl", dtype=object).copy()
    meta.columns = [str(c).strip() for c in meta.columns]

    if "code" not in metrics.columns:
        raise ValueError("Metrics CSV must include a 'code' column.")
    code_col = find_metadata_code_column(meta)
    metrics["code"] = metrics["code"].map(normalize_code)
    metrics["_join_code"] = metrics["code"].map(normalize_code)
    meta[code_col] = meta[code_col].map(normalize_code)
    meta["_join_code"] = meta[code_col].map(normalize_code)

    merged = metrics.merge(meta, on="_join_code", how="left", indicator=True, suffixes=("", "_meta"))
    if "Cod" not in merged.columns:
        merged["Cod"] = merged.get(code_col, merged["_join_code"])
    merged = add_standard_target_columns(merged)
    merged.to_csv(output_csv, index=False)

    analysis_dir = output_csv.parent / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    missing_cols = [c for c in ["code", "file", "activity", "activity_number", "scheme_window_size", "_join_code", "_merge"] if c in merged.columns]
    missing = merged.loc[merged["_merge"].ne("both"), missing_cols].drop_duplicates()
    missing.to_csv(analysis_dir / "metadata_unmatched_transcripts.csv", index=False)

    matched_codes = set(merged.loc[merged["_merge"].eq("both"), "_join_code"].dropna().astype(str))
    meta_only = meta.loc[~meta["_join_code"].astype(str).isin(matched_codes), [code_col, "_join_code"]].drop_duplicates()
    meta_only.to_csv(analysis_dir / "metadata_without_transcript.csv", index=False)

    if not missing.empty:
        print(f"Rows without metadata: {len(missing)}. See {analysis_dir / 'metadata_unmatched_transcripts.csv'}")
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge graph metrics with metadata")
    parser.add_argument("--metrics-csv", default="outputs/01_run/graph_metrics_all_windows.csv")
    parser.add_argument("--metadata-xlsx", default="data/processed/df_dataset.xlsx")
    parser.add_argument("--output-csv", default="outputs/01_run/graph_metrics_all_windows_with_meta.csv")
    parser.add_argument("--sheet-name", default=0, type=_parse_sheet_name)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_metadata(Path(args.metrics_csv), Path(args.metadata_xlsx), Path(args.output_csv), args.sheet_name)


if __name__ == "__main__":
    main()
