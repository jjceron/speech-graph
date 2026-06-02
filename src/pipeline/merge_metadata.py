from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

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
        normalized = _norm(col)
        if normalized in {"cod", "codigo", "code"} or "codigo" in normalized:
            return col
    raise ValueError("Metadata Excel must include a code column such as Cod, Código or Code")


def _parse_sheet_name(value: str) -> str | int:
    return int(value) if str(value).isdigit() else value


def _write_merge_qc(merged: pd.DataFrame, meta: pd.DataFrame, output_csv: Path, code_col: str) -> None:
    qc_dir = output_csv.parent / "analysis"
    qc_dir.mkdir(parents=True, exist_ok=True)
    if "_merge" in merged.columns:
        cols = [col for col in ["code", "file", "level", "activity", "activity_number", "_join_code", "_merge"] if col in merged.columns]
        missing = merged.loc[merged["_merge"] != "both", cols]
        if not missing.empty:
            missing.drop_duplicates().to_csv(qc_dir / "metadata_unmatched_transcripts.csv", index=False)
    if "_join_code" in meta.columns and "_join_code" in merged.columns:
        matched = set(merged.loc[merged.get("_merge", "") == "both", "_join_code"].dropna().astype(str))
        meta_only = meta.loc[~meta["_join_code"].astype(str).isin(matched)].copy()
        if not meta_only.empty:
            cols = [col for col in [code_col, "_join_code"] if col in meta_only.columns]
            meta_only[cols].drop_duplicates().to_csv(qc_dir / "metadata_without_transcript.csv", index=False)


def merge_metadata(metrics_csv: Path, metadata_xlsx: Path, output_csv: Path, sheet_name: str | int = 0) -> pd.DataFrame:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(metrics_csv).copy()
    meta = pd.read_excel(metadata_xlsx, sheet_name=sheet_name, engine="openpyxl", dtype=object).copy()
    meta.columns = [str(col).strip() for col in meta.columns]
    if "code" not in metrics.columns:
        raise ValueError("Metrics CSV must include a 'code' column")
    code_col = find_metadata_code_column(meta)
    metrics["code"] = metrics["code"].map(normalize_code)
    metrics["_join_code"] = metrics["code"].map(normalize_code)
    meta[code_col] = meta[code_col].map(normalize_code)
    meta["_join_code"] = meta[code_col].map(normalize_code)
    meta = meta.drop_duplicates(subset=["_join_code"], keep="first")
    merged = metrics.merge(meta, on="_join_code", how="left", indicator=True, suffixes=("", "_meta"))
    if "Cod" not in merged.columns:
        merged["Cod"] = merged.get(code_col, merged["_join_code"])
    merged.to_csv(output_csv, index=False)
    missing = int((merged["_merge"] != "both").sum())
    if missing:
        print(f"Rows without metadata: {missing}. See {output_csv.parent / 'analysis' / 'metadata_unmatched_transcripts.csv'}")
    _write_merge_qc(merged, meta, output_csv, code_col)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge graph metrics with metadata")
    parser.add_argument("--metrics-csv", default="outputs/graph_metrics.csv")
    parser.add_argument("--metadata-xlsx", default="data/processed/df_dataset.xlsx")
    parser.add_argument("--output-csv", default="outputs/graph_metrics_with_meta.csv")
    parser.add_argument("--sheet-name", default=0, type=_parse_sheet_name)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_metadata(Path(args.metrics_csv), Path(args.metadata_xlsx), Path(args.output_csv), args.sheet_name)


if __name__ == "__main__":
    main()
