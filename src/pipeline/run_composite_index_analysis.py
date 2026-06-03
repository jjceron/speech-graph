"""Run 05 composite SpeechGraph index analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.analysis.composite_index_stats import (
    build_subject_table,
    compute_indices,
    default_index_specs,
    default_model_specs,
    run_index_contrasts,
)
from src.visualization.composite_index_plots import (
    plot_index_effects,
    plot_index_boxplots,
    plot_index_component_heatmap,
    plot_sensitivity,
)


def _parse_csv_list(value: str | None) -> list[str]:
    if value is None or value == "":
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="05_run focused SpeechGraph composite index analysis")
    parser.add_argument("--activity-window-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-col", default="Tipo")
    parser.add_argument("--positive-label", default="high_imp")
    parser.add_argument("--negative-label", default="low_imp")
    parser.add_argument("--id-col", default="code")
    parser.add_argument(
        "--metrics",
        default="nodes,re,pe,l1,l2,l3,lcc,lsc,atd,density,diameter,asp,cc",
        help="SpeechGraph metrics to pivot to subject-level wide format.",
    )
    parser.add_argument(
        "--primary-covariates",
        default="School year,Gender",
        help="Primary covariates. Recommended: School year,Gender because Age and School year are highly collinear.",
    )
    parser.add_argument("--sensitivity-age-covariates", default="Age,Gender")
    parser.add_argument("--sensitivity-full-covariates", default="Age,School year,Gender")
    parser.add_argument("--primary-model", default="primary_school_year_gender")
    parser.add_argument("--transform", default="rank_normal", choices=["rank_normal", "winsor_z", "zscore"])
    parser.add_argument("--min-component-fraction", type=float, default=0.80)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--n-permutations", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--include-secondary-global", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def run_05_from_args(args: argparse.Namespace) -> dict:
    out_dir = Path(args.output_dir)
    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    activity_df = pd.read_csv(args.activity_window_csv)
    metrics = _parse_csv_list(args.metrics)

    covariates_all = sorted(
        set(_parse_csv_list(args.primary_covariates))
        | set(_parse_csv_list(args.sensitivity_age_covariates))
        | set(_parse_csv_list(args.sensitivity_full_covariates))
    )
    subject_df, manifest = build_subject_table(
        activity_df,
        metrics=metrics,
        label_col=args.label_col,
        covariates=covariates_all,
        id_col=args.id_col,
    )
    subject_path = table_dir / "subject_level_speechgraph_wide_for_indices.csv"
    subject_df.to_csv(subject_path, index=False)

    index_specs = default_index_specs(include_secondary_global=args.include_secondary_global)
    index_df, component_table = compute_indices(
        subject_df,
        specs=index_specs,
        transform=args.transform,
        min_component_fraction=args.min_component_fraction,
    )
    index_df_path = table_dir / "composite_indices_subject_level.csv"
    component_path = table_dir / "composite_index_components.csv"
    index_df.to_csv(index_df_path, index=False)
    component_table.to_csv(component_path, index=False)

    model_specs = default_model_specs(
        primary_covariates=_parse_csv_list(args.primary_covariates),
        sensitivity_age_covariates=_parse_csv_list(args.sensitivity_age_covariates),
        sensitivity_full_covariates=_parse_csv_list(args.sensitivity_full_covariates),
    )
    results, sensitivity = run_index_contrasts(
        index_df,
        index_specs=index_specs,
        model_specs=model_specs,
        label_col=args.label_col,
        positive_label=args.positive_label,
        negative_label=args.negative_label,
        primary_model_name=args.primary_model,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        random_state=args.random_state,
        alternative="greater",
    )
    results_path = table_dir / "composite_index_contrast_results.csv"
    sensitivity_path = table_dir / "composite_index_sensitivity_summary.csv"
    results.to_csv(results_path, index=False)
    sensitivity.to_csv(sensitivity_path, index=False)

    primary_sig = results[
        (results["primary_family_primary_model"])
        & (results["perm_q_primary_family"] < 0.05)
        & (results["bootstrap_ci_low"] > 0)
    ].copy()
    primary_sig.to_csv(table_dir / "significant_composite_indices_q05.csv", index=False)
    primary_candidates = results[
        (results["primary_family_primary_model"])
        & (results["perm_q_primary_family"] < 0.10)
        & (results["bootstrap_ci_low"] > 0)
    ].copy()
    primary_candidates.to_csv(table_dir / "candidate_composite_indices_q10.csv", index=False)

    if not args.skip_plots:
        plot_index_effects(results, fig_dir, primary_model=args.primary_model)
        plot_index_boxplots(index_df, results, args.label_col, fig_dir, primary_model=args.primary_model)
        plot_index_component_heatmap(index_df, component_table, fig_dir)
        plot_sensitivity(results, fig_dir)

    run_manifest = {
        "activity_window_csv": args.activity_window_csv,
        "output_dir": str(out_dir),
        "label_col": args.label_col,
        "positive_label": args.positive_label,
        "negative_label": args.negative_label,
        "metrics": metrics,
        "primary_covariates": _parse_csv_list(args.primary_covariates),
        "sensitivity_age_covariates": _parse_csv_list(args.sensitivity_age_covariates),
        "sensitivity_full_covariates": _parse_csv_list(args.sensitivity_full_covariates),
        "transform": args.transform,
        "min_component_fraction": args.min_component_fraction,
        "n_bootstrap": args.n_bootstrap,
        "n_permutations": args.n_permutations,
        "random_state": args.random_state,
        "subject_manifest": manifest,
        "outputs": {
            "subject_table": str(subject_path),
            "index_subject_table": str(index_df_path),
            "component_table": str(component_path),
            "results": str(results_path),
            "sensitivity": str(sensitivity_path),
        },
    }
    manifest_path = out_dir / "run_05_composite_index_manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    print(f"Subject-level SpeechGraph table saved: {subject_path}")
    print(f"Composite index table saved: {index_df_path}")
    print(f"Composite index components saved: {component_path}")
    print(f"Contrast results saved: {results_path}")
    print(f"Sensitivity summary saved: {sensitivity_path}")
    print("Primary results:")
    primary = results[results["primary_family_primary_model"]].copy()
    cols = [
        "index",
        "n",
        "n_high",
        "n_low",
        "adj_beta_high",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "perm_p_one_sided_greater",
        "perm_q_primary_family",
        "ci_excludes_zero",
    ]
    print(primary[cols].to_string(index=False))
    return run_manifest


def main() -> None:
    run_05_from_args(parse_args())


if __name__ == "__main__":
    main()
