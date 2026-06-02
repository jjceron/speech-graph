from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.activity_focus import generate_activity_focus_outputs
from src.pipeline.run_window_schemes import main as run_window_schemes_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run activity/class-level Mota-style speech-graph analysis. "
            "Default: window 30, spk_1, by activity, metadata merge, and two-file focused report."
        )
    )
    parser.add_argument("--transcripts-dir", default="data/processed/Transcripciones")
    parser.add_argument("--metadata-xlsx", default="data/processed/df_dataset.xlsx")
    parser.add_argument("--output-dir", default="outputs/02_w30_by_activity_no_random_all")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--include-speakers", default="spk_1")
    parser.add_argument("--random-times", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--random-metrics", default="lcc,lsc,edges,repeated_edges,density,asp,l1,l2,l3")
    parser.add_argument("--method", default="spearman", choices=["spearman", "pearson"])
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--no-allow-short", action="store_true")
    parser.add_argument("--lexicon-path", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--skip-metadata", action="store_true")
    parser.add_argument("--group-cols", default="Gender,Educational level,School,Tipo,level,activity")
    parser.add_argument(
        "--target-cols",
        default="Age,School year,TOTAL,NPLAN,MOT,COG,Barratt (pre),TOTAL_zscore,COG_zscore,MOT_zscore,auto",
    )
    parser.add_argument("--target-activities", default="2,6,7", help="Main activities/classes to interpret.")
    parser.add_argument("--secondary-activities", default="1,4,5", help="Secondary/control activities/classes.")
    parser.add_argument("--min-n", type=int, default=30)
    parser.add_argument("--min-abs-r", type=float, default=0.15)
    parser.add_argument("--label-min-nonzero", type=int, default=20)
    parser.add_argument("--group-col", default="Tipo")
    parser.add_argument("--min-group-n", type=int, default=10)
    parser.add_argument("--include-barratt-items", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    argv = [
        "--transcripts-dir", args.transcripts_dir,
        "--metadata-xlsx", args.metadata_xlsx,
        "--output-dir", str(output_dir),
        "--window-sizes", str(args.window_size),
        "--step", str(args.step),
        "--include-speakers", args.include_speakers,
        "--random-times", str(args.random_times),
        "--random-seed", str(args.random_seed),
        "--random-metrics", args.random_metrics,
        "--method", args.method,
        "--progress-every", str(args.progress_every),
        "--group-cols", args.group_cols,
        "--target-cols", args.target_cols,
        "--by-activity",
    ]
    if args.lowercase:
        argv.append("--lowercase")
    if args.no_allow_short:
        argv.append("--no-allow-short")
    if args.lexicon_path:
        argv.extend(["--lexicon-path", args.lexicon_path])
    if args.max_files is not None:
        argv.extend(["--max-files", str(args.max_files)])
    if args.skip_metadata:
        argv.append("--skip-metadata")

    run_window_schemes_main(argv)

    input_csv = output_dir / "graph_metrics_all_windows_with_meta.csv"
    if args.skip_metadata or not input_csv.exists():
        input_csv = output_dir / "graph_metrics_all_windows.csv"
    generate_activity_focus_outputs(
        input_csv=input_csv,
        output_dir=output_dir,
        window_size=args.window_size,
        target_activities=args.target_activities,
        secondary_activities=args.secondary_activities,
        method=args.method,
        min_n=args.min_n,
        min_abs_r=args.min_abs_r,
        label_min_nonzero=args.label_min_nonzero,
        group_col=args.group_col,
        min_group_n=args.min_group_n,
        include_barratt_items=args.include_barratt_items,
    )
    print(f"Activity-focused outputs:")
    print(f"  {output_dir / 'analysis' / 'activity_focus_results.csv'}")
    print(f"  {output_dir / 'analysis' / 'activity_focus_report.md'}")


if __name__ == "__main__":
    main()
