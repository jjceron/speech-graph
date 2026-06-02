from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.io import normalize_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge graph metrics with metadata")
    parser.add_argument(
        "--metrics-csv",
        default="outputs/graph_metrics.csv",
        help="Metrics CSV path",
    )
    parser.add_argument(
        "--metadata-xlsx",
        default="data/processed/df_dataset.xlsx",
        help="Metadata Excel path",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/graph_metrics_with_meta.csv",
        help="Merged output CSV",
    )
    parser.add_argument("--sheet-name", default="Sheet1", help="Excel sheet name to read")
    return parser.parse_args()


def _write_merge_qc(merged: pd.DataFrame, meta: pd.DataFrame, output_csv: Path) -> None:
    qc_dir = output_csv.parent / "analysis"
    qc_dir.mkdir(parents=True, exist_ok=True)

    if "_merge" in merged.columns:
        missing = merged.loc[merged["_merge"] != "both", [c for c in ["code", "file", "level", "activity", "_join_code", "_merge"] if c in merged.columns]]
        if not missing.empty:
            missing.drop_duplicates().to_csv(qc_dir / "metadata_unmatched_transcripts.csv", index=False)

    if "_join_code" in meta.columns and "_join_code" in merged.columns:
        matched_codes = set(merged.loc[merged.get("_merge", "") == "both", "_join_code"].dropna().astype(str))
        meta_only = meta.loc[~meta["_join_code"].astype(str).isin(matched_codes)].copy()
        if not meta_only.empty:
            cols = [c for c in ["Cod", "_join_code"] if c in meta_only.columns]
            if cols:
                meta_only[cols].drop_duplicates().to_csv(qc_dir / "metadata_without_transcript.csv", index=False)


def merge_metadata(
    metrics_csv: Path,
    metadata_xlsx: Path,
    output_csv: Path,
    sheet_name: str = "Sheet1",
) -> pd.DataFrame:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # copy() defragments wide metric tables, avoiding pandas PerformanceWarning
    # after many label feature columns have been created.
    metrics = pd.read_csv(metrics_csv).copy()
    meta = pd.read_excel(metadata_xlsx, sheet_name=sheet_name, engine="openpyxl", dtype=object).copy()

    meta.columns = [str(c).strip() for c in meta.columns]
    if "code" not in metrics.columns:
        raise ValueError("Metrics CSV must include a 'code' column")
    if "Cod" not in meta.columns:
        raise ValueError("Metadata Excel must include a 'Cod' column")

    metrics = metrics.assign(code=metrics["code"].map(normalize_code))
    meta = meta.assign(Cod=meta["Cod"].map(normalize_code))
    metrics = metrics.assign(_join_code=metrics["code"].map(normalize_code))
    meta = meta.assign(_join_code=meta["Cod"].map(normalize_code))

    merged = metrics.merge(meta, on="_join_code", how="left", indicator=True, suffixes=("", "_meta"))
    if "Cod" not in merged.columns and "Cod_meta" in merged.columns:
        merged["Cod"] = merged["Cod_meta"]
    merged.to_csv(output_csv, index=False)

    missing = int((merged["_merge"] != "both").sum())
    if missing:
        print(f"Rows without metadata: {missing}. See {output_csv.parent / 'analysis' / 'metadata_unmatched_transcripts.csv'}")
    _write_merge_qc(merged, meta, output_csv)
    return merged


def main() -> None:
    args = parse_args()
    merge_metadata(
        Path(args.metrics_csv),
        Path(args.metadata_xlsx),
        Path(args.output_csv),
        sheet_name=args.sheet_name,
    )


if __name__ == "__main__":
    main()
