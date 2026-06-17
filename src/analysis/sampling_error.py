"""Sampling error analysis for different MMCV split ratios.

Usage:
    py -m src.analysis.sampling_error --task 6 --window 30 --experiment raw --target MOT_V4 --splitting "70-30-10" "80-15-5" "90-8-2" --n-iter 400 --output outputs/sampling_error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.regression_optuna import generate_mmcv_splits, load_per_window_matrix


RATIOS = {
    "70-30-10": (0.7, 0.2, 0.1),
    "80-15-5":  (0.8, 0.1, 0.05),
    "90-8-2":   (0.82, 0.1, 0.08),
}


def run_one_ratio(X: pd.DataFrame, y: pd.Series, label: str, n_iter: int) -> dict:
    tr, va, te = RATIOS[label]
    splits, _ = generate_mmcv_splits(X, y, n_splits=n_iter, train_size=tr, val_size=va, test_size=te)

    mae_vals, r2_vals = [], []
    for tri, vai, tei in splits:
        imp = SimpleImputer(strategy="mean")
        X_tr = imp.fit_transform(X.iloc[tri])
        X_va = imp.transform(X.iloc[vai])
        X_te = imp.transform(X.iloc[tei])

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)
        X_te = scaler.transform(X_te)

        model = LinearRegression().fit(X_tr, y.iloc[tri])
        mae_vals.append(mean_absolute_error(y.iloc[tei], model.predict(X_te)))
        r2_vals.append(r2_score(y.iloc[tei], model.predict(X_te)))

    n = len(mae_vals)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    sem_mae = stats.sem(mae_vals)
    sem_r2 = stats.sem(r2_vals)

    return {
        "ratio": label,
        "train_val_test": f"{tr*100:.0f}-{va*100:.0f}-{te*100:.0f}",
        "n_splits": n,
        "MAE_mean": float(np.mean(mae_vals)),
        "MAE_std": float(np.std(mae_vals, ddof=1)),
        "MAE_sem": float(sem_mae),
        "MAE_sampling_error": float(t_crit * sem_mae),
        "MAE_ci_lower": float(np.mean(mae_vals) - t_crit * sem_mae),
        "MAE_ci_upper": float(np.mean(mae_vals) + t_crit * sem_mae),
        "R2_mean": float(np.mean(r2_vals)),
        "R2_std": float(np.std(r2_vals, ddof=1)),
        "R2_sem": float(sem_r2),
        "R2_sampling_error": float(t_crit * sem_r2),
    }


def plot_results(df: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    colors = ["#4C72B0", "#DD8452", "#55A868"]

    ax1.bar(df["ratio"], df["MAE_mean"], yerr=df["MAE_sampling_error"],
            capsize=5, color=colors, edgecolor="black", linewidth=0.8)
    ax1.set_xlabel("Split ratio (train-val-test)")
    ax1.set_ylabel("MAE")
    ax1.set_title("Mean Absolute Error ± 95% CI")

    ax2.bar(df["ratio"], df["R2_mean"], yerr=df["R2_sampling_error"],
            capsize=5, color=colors, edgecolor="black", linewidth=0.8)
    ax2.axhline(y=0, color="gray", linestyle="--", linewidth=0.7)
    ax2.set_xlabel("Split ratio (train-val-test)")
    ax2.set_ylabel("R²")
    ax2.set_title("R² ± 95% CI")

    for ax in (ax1, ax2):
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "sampling_error.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="py -m src.analysis.sampling_error",
        description="Sampling error analysis for MMCV split ratios using LinearRegression.",
    )
    parser.add_argument("--task", type=int, required=True, choices=[2, 6, 7])
    parser.add_argument("--window", required=True, help="e.g. 30 for W30")
    parser.add_argument("--experiment", default="raw", choices=["raw", "zscores", "rawzscore"])
    parser.add_argument("--target", default="MOT_V4")
    parser.add_argument("--splitting", nargs="+", required=True,
                        choices=list(RATIOS.keys()),
                        help="Split ratio(s) to test, e.g. 70-30-10 80-15-5 90-8-2")
    parser.add_argument("--n-iter", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics-dir", default="data/processed/metrics")
    parser.add_argument("--metadata", default="data/raw/metadata.xlsx")
    parser.add_argument("--output", default="outputs/sampling_error")
    args = parser.parse_args()

    np.random.seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data: T{args.task} W{args.window} {args.experiment} → {args.target}")
    X, y = load_per_window_matrix(
        task=args.task,
        window=args.window,
        experiment=args.experiment,
        target=args.target,
        metrics_dir=args.metrics_dir,
        metadata_path=args.metadata,
    )
    print(f"  Subjects: {len(y)}, Features: {X.shape[1]}")

    results = []
    for label in args.splitting:
        print(f"  Running {label} ({args.n_iter} splits)...")
        r = run_one_ratio(X, y, label, args.n_iter)
        results.append(r)
        print(f"    MAE={r['MAE_mean']:.4f} ±{r['MAE_sampling_error']:.4f}  "
              f"R²={r['R2_mean']:.4f} ±{r['R2_sampling_error']:.4f}")

    df = pd.DataFrame(results)
    csv_path = output_dir / "sampling_error.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved: {csv_path}")

    json_path = output_dir / "sampling_error.json"
    json_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"JSON saved: {json_path}")

    plot_results(df, output_dir)

    print("\nSummary:")
    print(df[["ratio", "MAE_mean", "MAE_sampling_error", "R2_mean", "R2_sampling_error"]].to_string(index=False))


if __name__ == "__main__":
    main()
