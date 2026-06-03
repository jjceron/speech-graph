"""Run 04 SpeechGraph high/low group contrast analysis.

This pipeline keeps the graph construction fixed from 01_run by consuming the
already extracted activity-window SpeechGraph table. It does not rebuild graphs;
it tests whether graph metrics differ between high_imp and low_imp groups after
adjusting for demographic covariates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.analysis.group_contrast_stats import (
    ContrastConfig,
    infer_code_column,
    resolve_column,
    resolve_columns,
    run_group_contrasts,
    summarize_contrasts,
)
from src.visualization.group_contrast_plots import make_all_group_contrast_plots


DEFAULT_METRICS = (
    "nodes,re,pe,l1,l2,l3,lcc,lsc,atd,density,diameter,asp,cc"
)
DEFAULT_COVARIATES = "Age,Gender,School year"


def _split_csv_arg(value: str | None) -> list[str]:
    if value is None or str(value).strip() == "":
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def build_analysis_table(
    activity_window_csv: str | Path,
    metadata_xlsx: str | Path,
    label_col: str,
    covariates: Sequence[str],
    activity_code_col: str | None = None,
    metadata_code_col: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load and merge SpeechGraph activity-window features with metadata.

    The 01_run activity-window table already contains several metadata fields
    such as Tipo, Age, Gender and School year. Earlier versions of this module
    always pulled those fields again from the Excel file, which created pandas
    suffixes such as Tipo_x/Tipo_y after merging. This version resolves each
    requested label/covariate preferentially from the activity-window table and
    only imports it from the Excel metadata file when it is missing there.
    """
    aw = pd.read_csv(activity_window_csv)
    md = pd.read_excel(metadata_xlsx)
    aw_code = infer_code_column(aw.columns, activity_code_col)
    md_code = infer_code_column(md.columns, metadata_code_col)

    if aw_code != "code":
        aw = aw.rename(columns={aw_code: "code"})
        aw_code = "code"

    # Resolve label and covariates. Prefer the already merged 01_run table;
    # fall back to df_dataset.xlsx only for fields that are absent from it.
    label_aw = resolve_column(aw.columns, label_col, required=False)
    label_md = None if label_aw is not None else resolve_column(md.columns, label_col, required=True)
    label_resolved = label_aw if label_aw is not None else label_md

    cov_resolved: list[str] = []
    md_needed: list[str] = []
    cov_sources: dict[str, str] = {}
    for cov in covariates:
        cov_aw = resolve_column(aw.columns, cov, required=False)
        if cov_aw is not None:
            if cov_aw not in cov_resolved:
                cov_resolved.append(cov_aw)
                cov_sources[cov_aw] = "activity_window_csv"
            continue
        cov_md = resolve_column(md.columns, cov, required=False)
        if cov_md is not None:
            if cov_md not in cov_resolved:
                cov_resolved.append(cov_md)
                cov_sources[cov_md] = "metadata_xlsx"
            if cov_md not in md_needed:
                md_needed.append(cov_md)

    if label_md is not None:
        md_needed.append(label_md)

    # Preserve one metadata row per subject and merge only missing fields.
    md_cols = [md_code] + [c for c in md_needed if c != md_code]
    md_small = md[md_cols].drop_duplicates(subset=[md_code]).copy()
    if md_code != "code":
        md_small = md_small.rename(columns={md_code: "code"})
        md_code = "code"

    if len(md_cols) > 1:
        merged = aw.merge(md_small, on="code", how="inner", suffixes=("", "_metadata"))
    else:
        # Still restrict to subjects available in df_dataset.xlsx, but avoid
        # importing duplicate label/covariate columns.
        merged = aw.merge(md_small[["code"]], on="code", how="inner")

    # If a label/covariate was imported from metadata and got renamed by the
    # merge, resolve its final name from the merged table.
    label_resolved_final = resolve_column(merged.columns, label_resolved, required=True)
    cov_resolved_final = resolve_columns(merged.columns, cov_resolved, required=False)

    manifest = {
        "activity_window_rows": int(len(aw)),
        "metadata_rows": int(len(md)),
        "merged_rows": int(len(merged)),
        "unique_subjects_activity_window": int(aw[aw_code].nunique()),
        "unique_subjects_metadata": int(md_small[md_code].nunique()),
        "unique_subjects_merged": int(merged["code"].nunique()),
        "label_col": label_resolved_final,
        "label_source": "activity_window_csv" if label_aw is not None else "metadata_xlsx",
        "covariates": cov_resolved_final,
        "covariate_sources": cov_sources,
        "activity_code_col": aw_code,
        "metadata_code_col": md_code,
        "metadata_columns_imported": [c for c in md_needed],
    }
    return merged, manifest


def run_04_from_args(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_requested = _split_csv_arg(args.metrics)
    covariates_requested = _split_csv_arg(args.covariates)
    analysis_df, build_manifest = build_analysis_table(
        activity_window_csv=args.activity_window_csv,
        metadata_xlsx=args.metadata_xlsx,
        label_col=args.label_col,
        covariates=covariates_requested,
        activity_code_col=args.activity_code_col,
        metadata_code_col=args.metadata_code_col,
    )
    analysis_table_path = tables_dir / "group_contrast_analysis_table.csv"
    analysis_df.to_csv(analysis_table_path, index=False)

    label_col = build_manifest["label_col"]
    config = ContrastConfig(
        label_col=label_col,
        positive_label=args.positive_label,
        negative_label=args.negative_label,
        covariates=tuple(build_manifest["covariates"]),
        min_n=args.min_n,
        min_group_n=args.min_group_n,
        n_bootstrap=args.n_bootstrap,
        random_state=args.random_state,
    )
    results = run_group_contrasts(
        data=analysis_df,
        metrics=metrics_requested,
        config=config,
        activity_col=args.activity_col,
        activity_name_col=args.activity_name_col,
        window_col=args.window_col,
    )
    results_path = tables_dir / "speechgraph_group_contrasts.csv"
    results.to_csv(results_path, index=False)

    ok = results[results.get("status", "").eq("ok")].copy()
    top_all = ok.reindex(ok["adj_std_beta_high"].abs().sort_values(ascending=False).index) if not ok.empty else ok
    top_all_path = tables_dir / "top_group_contrasts_by_abs_effect.csv"
    top_all.to_csv(top_all_path, index=False)

    significant = ok[ok.get("significant_global_fdr_05", False).fillna(False)].copy() if not ok.empty else ok
    significant_path = tables_dir / "significant_group_contrasts_global_fdr05.csv"
    significant.to_csv(significant_path, index=False)

    candidates = ok[ok.get("candidate_marker", False).fillna(False)].copy() if not ok.empty else ok
    candidates_path = tables_dir / "candidate_group_markers_q10_ci.csv"
    candidates.to_csv(candidates_path, index=False)

    summary = summarize_contrasts(results)
    summary_path = tables_dir / "group_contrast_summary_by_scheme.csv"
    summary.to_csv(summary_path, index=False)

    plot_paths = make_all_group_contrast_plots(
        analysis_df=analysis_df,
        results=results,
        summary=summary,
        output_dir=figures_dir,
        label_col=label_col,
        top_n=args.top_n,
    )

    manifest = {
        "run": "04_group_contrast_speechgraph",
        "objective": "Adjusted high/low group contrasts for SpeechGraph metrics by activity and window.",
        "input": {
            "activity_window_csv": str(args.activity_window_csv),
            "metadata_xlsx": str(args.metadata_xlsx),
        },
        "output_dir": str(output_dir),
        "build": build_manifest,
        "config": {
            "label_col": label_col,
            "positive_label": args.positive_label,
            "negative_label": args.negative_label,
            "covariates": build_manifest["covariates"],
            "metrics_requested": metrics_requested,
            "min_n": args.min_n,
            "min_group_n": args.min_group_n,
            "n_bootstrap": args.n_bootstrap,
            "random_state": args.random_state,
        },
        "outputs": {
            "analysis_table": str(analysis_table_path),
            "results": str(results_path),
            "top_all": str(top_all_path),
            "significant_global_fdr05": str(significant_path),
            "candidates_q10_ci": str(candidates_path),
            "summary_by_scheme": str(summary_path),
            "figures": plot_paths,
        },
        "result_counts": {
            "tested_ok": int(len(ok)),
            "significant_global_fdr05": int(len(significant)),
            "candidate_q10_ci": int(len(candidates)),
        },
    }
    manifest_path = output_dir / "run_04_group_contrast_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Analysis table saved: {analysis_table_path}")
    print(f"Group contrasts saved: {results_path}")
    print(f"Summary saved: {summary_path}")
    print(f"Figures generated: {len(plot_paths)}")
    print(json.dumps(manifest["result_counts"], indent=2))
    print(f"04 group contrast analysis completed. Manifest: {manifest_path}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run adjusted SpeechGraph high/low group contrasts.")
    parser.add_argument("--activity-window-csv", required=True)
    parser.add_argument("--metadata-xlsx", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-col", default="Tipo")
    parser.add_argument("--positive-label", default="high_imp")
    parser.add_argument("--negative-label", default="low_imp")
    parser.add_argument("--covariates", default=DEFAULT_COVARIATES)
    parser.add_argument("--metrics", default=DEFAULT_METRICS)
    parser.add_argument("--activity-code-col", default=None)
    parser.add_argument("--metadata-code-col", default=None)
    parser.add_argument("--activity-col", default="activity_number")
    parser.add_argument("--activity-name-col", default="activity")
    parser.add_argument("--window-col", default="window_size")
    parser.add_argument("--min-n", type=int, default=80)
    parser.add_argument("--min-group-n", type=int, default=20)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    run_04_from_args(parse_args())


if __name__ == "__main__":
    main()
