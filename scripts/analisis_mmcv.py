"""Analyse how MAE sampling error and computation time per trial
vary with the number of MMCV splits (100 → 1000, step 100),
using a single regressor with rfe_fixed on task6/W30/raw/MOT_V4.

Usage:
    py scripts/analisis_mmcv.py --regressor ExtraTreesRegressor --n-trials 300 --n-iter-start 100 --n-iter-end 1000 --n-iter-step 100
    py scripts/analisis_mmcv.py --regressor LinearRegression  --n-trials 300 --n-iter-start 100 --n-iter-end 1000 --n-iter-step 100
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.regression_optuna import ALL_REGRESSORS, load_per_window_matrix, run_one_target

TASK = 6
WINDOW = 30
EXPERIMENT = "raw"
TARGET = "MOT_V4"
RFE_MODE = "fixed"
CHOICES = ["ExtraTreesRegressor", "LinearRegression"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MMCV split-size analysis for a single regressor",
    )
    parser.add_argument("--regressor", required=True, choices=CHOICES,
                        help="Regressor to use")
    parser.add_argument("--n-trials", type=int, default=300)
    parser.add_argument("--n-iter-start", type=int, default=100)
    parser.add_argument("--n-iter-end", type=int, default=1000)
    parser.add_argument("--n-iter-step", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/regression_optuna")
    parser.add_argument("--timeout-sec", type=int, default=480)
    parser.add_argument("--max-rfe-features", type=int, default=100)
    parser.add_argument("--pruner-startup-trials", type=int, default=100)
    parser.add_argument("--pruner-warmup-steps", type=int, default=15)
    args = parser.parse_args()

    regressors = [args.regressor]

    print(f"Loading X, y for task={TASK}, window={WINDOW}, experiment={EXPERIMENT}, target={TARGET}...")
    X, y = load_per_window_matrix(
        task=TASK,
        window=WINDOW,
        experiment=EXPERIMENT,
        target=TARGET,
        metrics_dir="data/processed/metrics",
        metadata_path="data/raw/metadata.xlsx",
    )
    print(f"  Subjects={len(y)}, Features={X.shape[1]}")

    for n_iter in range(args.n_iter_start, args.n_iter_end + 1, args.n_iter_step):
        experiment_name = (
            f"n{n_iter}_task{TASK}_W{WINDOW}_{EXPERIMENT}_{TARGET}"
            f"_rfe{RFE_MODE}_mae_{args.regressor}"
        )
        output_dir = Path(args.output) / "analisis_mmcv" / f"n{n_iter}"

        print(f"\n{'='*60}")
        print(f"Regressor={args.regressor} | n_iter={n_iter} → {output_dir}")
        print(f"{'='*60}")

        run_one_target(
            X=X,
            y=y,
            output_dir=output_dir,
            experiment_name=experiment_name,
            optimize_metric="mae",
            regressors=regressors,
            n_trials=args.n_trials,
            n_iter=n_iter,
            seed=args.seed,
            timeout_sec=args.timeout_sec,
            max_rfe_features=args.max_rfe_features,
            pruner_startup_trials=args.pruner_startup_trials,
            pruner_warmup_steps=args.pruner_warmup_steps,
            rfe_mode=RFE_MODE,
        )

    print(f"\nDone. All results in {Path(args.output) / 'analisis_mmcv'}/")


if __name__ == "__main__":
    main()
