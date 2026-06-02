from __future__ import annotations

import argparse
from pathlib import Path

from .run_window_schemes import main as run_window_schemes_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Mota-style speech-graph window analysis for this project: "
            "10/20/30-word windows, random baseline, metadata merge and summary tables."
        )
    )
    parser.add_argument("--transcripts-dir", default="data/processed/Transcripciones")
    parser.add_argument("--metadata-xlsx", default="data/processed/df_dataset.xlsx")
    parser.add_argument("--output-dir", default="outputs/04_windows_random1000")
    parser.add_argument("--window-sizes", default="10,20,30")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--include-speakers", default="spk_1")
    parser.add_argument("--random-times", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--random-metrics",
        default="lcc,lsc,edges,repeated_edges,density,asp,l1,l2,l3",
        help="Metrics compared against shuffled-token baselines.",
    )
    parser.add_argument("--method", default="spearman", choices=["spearman", "pearson"])
    parser.add_argument("--by-activity", action="store_true", help="Also emit activity/block-level rows.")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    argv = [
        "--transcripts-dir", args.transcripts_dir,
        "--metadata-xlsx", args.metadata_xlsx,
        "--output-dir", args.output_dir,
        "--window-sizes", args.window_sizes,
        "--step", str(args.step),
        "--include-speakers", args.include_speakers,
        "--random-times", str(args.random_times),
        "--random-seed", str(args.random_seed),
        "--random-metrics", args.random_metrics,
        "--method", args.method,
        "--progress-every", str(args.progress_every),
        "--group-cols", args.group_cols,
        "--target-cols", args.target_cols,
    ]
    if args.by_activity:
        argv.append("--by-activity")
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

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    run_window_schemes_main(argv)


if __name__ == "__main__":
    main()
