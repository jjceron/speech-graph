from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.analysis.profile_preprocessing import make_feature_blocks, parse_csv_list
from src.analysis.validate_profiles import run_profile_validation
from src.models.profile_gmm import run_profile_gmm
from src.pipeline.build_subject_profiles import build_subject_profile_features
from src.visualization.profile_plots import generate_profile_figures


def run_03_from_args(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    built = build_subject_profile_features(
        activity_window_csv=args.activity_window_csv,
        metadata_xlsx=args.metadata_xlsx,
        output_dir=output_dir,
        activity_code_col=args.activity_code_col,
        metadata_code_col=args.metadata_code_col,
        metrics_text=args.metrics,
    )
    subject_csv = built["subject_features"]
    subject_df = pd.read_csv(subject_csv)
    code_col = "code"

    blocks = make_feature_blocks(
        subject_df,
        code_col=code_col,
        targets_text=args.targets,
        label_col=args.label_col,
        demographic_cols_text=args.demographic_cols,
        cognitive_cols_text=args.cognitive_cols,
        covariate_cols_text=args.covariate_cols,
    )

    feature_manifest = {
        "targets": blocks.targets,
        "label_col": blocks.label_col,
        "demographics": blocks.demographics,
        "cognitive": blocks.cognitive,
        "speechgraph_n": len(blocks.speechgraph),
        "speechgraph_first_20": blocks.speechgraph[:20],
        "covariates": blocks.covariates,
        "metadata_feature_n": len(blocks.metadata_features),
        "multimodal_feature_n": len(blocks.multimodal_features),
    }
    manifest_dir = output_dir / "features"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "profile_feature_blocks_manifest.json").write_text(
        json.dumps(feature_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Feature blocks resolved:")
    print(json.dumps(feature_manifest, indent=2, ensure_ascii=False))

    k_values = [int(k) for k in parse_csv_list(args.k_values)]
    min_profile_size = args.min_profile_size
    if min_profile_size <= 0:
        min_profile_size = max(15, int(round(0.08 * len(subject_df))))

    profile_sets = [x.strip() for x in parse_csv_list(args.profile_sets) if x.strip()]
    if "metadata" not in profile_sets:
        profile_sets = ["metadata", *profile_sets]

    profile_paths = {}
    for profile_set in profile_sets:
        profile_paths[profile_set] = run_profile_gmm(
            subject_features_csv=subject_csv,
            output_dir=output_dir,
            blocks=blocks,
            profile_set=profile_set,
            k_values=k_values,
            min_profile_size=min_profile_size,
            n_bootstrap=args.n_bootstrap,
            random_state=args.random_state,
            variance_threshold=args.pca_variance,
            max_cognitive_components=args.max_cognitive_components,
            max_speechgraph_components=args.max_speechgraph_components,
            covariance_type=args.covariance_type,
            final_standardize=not args.no_final_standardize,
        )

    validation_paths = run_profile_validation(
        subject_features_csv=subject_csv,
        output_dir=output_dir,
        blocks=blocks,
        assignment_paths={name: paths["assignments"] for name, paths in profile_paths.items()},
        n_permutations=args.n_permutations,
        random_state=args.random_state,
    )

    figure_count = 0
    if not args.skip_plots:
        figure_count = generate_profile_figures(output_dir, subject_features_csv=subject_csv, targets_text=args.targets)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "analysis": "03_run_multimodal_impulsivity_profiles",
        "activity_window_csv": args.activity_window_csv,
        "metadata_xlsx": args.metadata_xlsx,
        "output_dir": str(output_dir),
        "targets": args.targets,
        "label_col": args.label_col,
        "profile_sets": profile_sets,
        "k_values": k_values,
        "min_profile_size": min_profile_size,
        "n_bootstrap": args.n_bootstrap,
        "n_permutations": args.n_permutations,
        "pca_variance": args.pca_variance,
        "covariance_type": args.covariance_type,
        "random_state": args.random_state,
        "feature_blocks": feature_manifest,
        "built": {k: str(v) for k, v in built.items()},
        "profiles": {ps: {k: str(v) for k, v in paths.items()} for ps, paths in profile_paths.items()},
        "validation": {k: str(v) for k, v in validation_paths.items()},
        "figure_count": figure_count,
    }
    manifest_path = output_dir / "run_03_profile_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"03 profile analysis completed. Manifest: {manifest_path}")
    return {"manifest": manifest_path, **built, **validation_paths}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 03 analysis: multimodal impulsivity profiling with external Barratt validation.")
    parser.add_argument("--activity-window-csv", default="outputs/01_run/analysis/activity_window_features.csv")
    parser.add_argument("--metadata-xlsx", default="df_dataset.xlsx")
    parser.add_argument("--output-dir", default="outputs/03_run")
    parser.add_argument("--activity-code-col", default=None)
    parser.add_argument("--metadata-code-col", default=None)
    parser.add_argument("--targets", default="TOTAL,NPLAN,MOT,COG")
    parser.add_argument("--label-col", default="Tipo")
    parser.add_argument("--demographic-cols", default="Age,Gender,School year,School,Educational level")
    parser.add_argument("--cognitive-cols", default=None, help="Optional comma-separated cognitive/language metadata columns. If omitted, inferred automatically.")
    parser.add_argument("--covariate-cols", default="Age,Gender,School year")
    parser.add_argument("--metrics", default=None, help="Optional comma-separated SpeechGraph metrics to pivot from activity-window CSV.")
    parser.add_argument("--profile-sets", default="metadata,speechgraph,multimodal_balanced", help="Comma-separated profile solutions to fit. Supported: metadata,speechgraph,multimodal,multimodal_balanced.")
    parser.add_argument("--k-values", default="2,3,4,5")
    parser.add_argument("--min-profile-size", type=int, default=0, help="Minimum subjects per profile. Default 0 uses max(15, 8% of n).")
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--pca-variance", type=float, default=0.80)
    parser.add_argument("--max-cognitive-components", type=int, default=6)
    parser.add_argument("--max-speechgraph-components", type=int, default=12)
    parser.add_argument("--covariance-type", default="diag", choices=["diag", "full", "tied", "spherical"])
    parser.add_argument("--no-final-standardize", action="store_true", help="Disable final standardization of encoded profile dimensions.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_03_from_args(parse_args())


if __name__ == "__main__":
    main()
