from __future__ import annotations

import argparse
from pathlib import Path

from .common import discover_run_dirs, parse_csv_list
from .compare_runs import generate_run_comparison
from .core import generate_core_figures
from .groups_focused import generate_group_profile_figures
from .labels import generate_label_figures
from .sensitivity import generate_sensitivity_figures
from .subjects import generate_subject_figures
from .activities import generate_activity_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate focused NLP/speech-graph figures from existing outputs. "
            "Default is optimized for 03_windows_random100 and 04_windows_random1000; "
            "it does not recompute graph metrics."
        )
    )
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument(
        "--runs",
        default="03_windows_random100,04_windows_random1000",
        help="Comma-separated run directory names under outputs/. Empty = auto-discover random runs.",
    )
    parser.add_argument("--run-dir", default="", help="Analyze one explicit output directory.")
    parser.add_argument("--window-size", type=int, default=30, help="Main window for Mota-style analysis; default w30.")
    parser.add_argument("--level", default="file", choices=["file", "activity", "all"])
    parser.add_argument("--method", default="spearman", choices=["spearman", "pearson"])
    parser.add_argument("--min-n", type=int, default=20)
    parser.add_argument("--min-abs-r", type=float, default=0.20)
    parser.add_argument("--label-min-n", type=int, default=12)
    parser.add_argument("--label-min-nonzero", type=int, default=8)
    parser.add_argument("--label-min-abs-r", type=float, default=0.25)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--scatter-top-n", type=int, default=6)
    parser.add_argument("--subject-top-n", type=int, default=30)
    parser.add_argument("--group-col", default="Tipo")
    parser.add_argument("--target-activities", default="2,6,7")
    parser.add_argument("--secondary-activities", default="1,4,5")
    parser.add_argument(
        "--only",
        default="all",
        choices=["all", "core", "labels", "subjects", "sensitivity", "groups", "activities", "compare-runs"],
        help="Which focused figure family to create.",
    )
    parser.add_argument(
        "--compare-random-runs",
        action="store_true",
        help="Also compare outputs/03_windows_random100 vs outputs/04_windows_random1000 convergence.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_dir:
        run_dirs = [Path(args.run_dir)]
    else:
        runs = parse_csv_list(args.runs) or None
        run_dirs = discover_run_dirs(Path(args.outputs_root), runs=runs, random_only=True)

    if not run_dirs:
        print("No output runs found. Expected outputs/03_windows_random100 or outputs/04_windows_random1000.")
        return

    total = 0
    for run_dir in run_dirs:
        if not run_dir.exists():
            print(f"Skipping missing run: {run_dir}")
            continue
        print(f"\n=== Focused NLP figures for {run_dir} ===")
        if args.only in {"all", "core"}:
            total += len(generate_core_figures(
                run_dir,
                window_size=args.window_size,
                level=args.level,
                method=args.method,
                min_n=args.min_n,
                min_abs_r=args.min_abs_r,
                top_n=args.top_n,
                scatter_top_n=args.scatter_top_n,
                group_col=args.group_col,
            ))
        if args.only in {"all", "labels"}:
            total += len(generate_label_figures(
                run_dir,
                window_size=args.window_size,
                level=args.level,
                method=args.method,
                min_nonzero=args.label_min_nonzero,
                min_n=args.label_min_n,
                min_abs_r=args.label_min_abs_r,
                scatter_top_n=args.scatter_top_n,
                group_col=args.group_col,
            ))
        if args.only in {"all", "subjects"}:
            total += len(generate_subject_figures(
                run_dir,
                window_size=args.window_size,
                level=args.level,
                top_n=args.subject_top_n,
                group_col=args.group_col,
                min_n=args.min_n,
                min_abs_r=args.min_abs_r,
            ))

        if args.only in {"all", "activities"} and args.level in {"activity", "all"}:
            total += len(generate_activity_figures(
                run_dir,
                window_size=args.window_size,
                target_activities=args.target_activities,
                secondary_activities=args.secondary_activities,
                method=args.method,
                min_n=args.min_n,
                min_abs_r=args.min_abs_r,
                label_min_nonzero=args.label_min_nonzero,
                group_col=args.group_col,
                scatter_top_n=args.scatter_top_n,
            ))
        if args.only in {"all", "sensitivity"}:
            total += len(generate_sensitivity_figures(
                run_dir,
                level=args.level,
                method=args.method,
                min_n=args.min_n,
                min_abs_r=args.min_abs_r,
                group_col=args.group_col,
            ))
        if args.only in {"all", "groups"}:
            total += len(generate_group_profile_figures(
                run_dir,
                window_size=args.window_size,
                level=args.level,
            ))

    if args.only == "compare-runs" or args.compare_random_runs:
        total += len(generate_run_comparison(
            Path(args.outputs_root) / "03_windows_random100",
            Path(args.outputs_root) / "04_windows_random1000",
            window_size=args.window_size,
            level=args.level,
        ))

    print(f"\nDone. Focused figures/tables generated: {total}")


if __name__ == "__main__":
    main()
