"""Sampling error analysis for MMCV split ratios — validation and test.

Computes MAE and R² on both validation and test sets across MMCV iterations,
reporting mean, std, and 95% CI sampling error for each split ratio.

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


def _ci(values: list[float]) -> dict[str, float]:
    arr = np.array(values)
    n = len(arr)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    sem = stats.sem(arr)
    return {
        "n": n,
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)),
        "sem": float(sem),
        "sampling_error": float(t_crit * sem),
        "ci_lower": float(np.mean(arr) - t_crit * sem),
        "ci_upper": float(np.mean(arr) + t_crit * sem),
    }


def run_one_ratio(X: pd.DataFrame, y: pd.Series, label: str, n_iter: int) -> dict:
    tr, va, te = RATIOS[label]
    splits, _ = generate_mmcv_splits(X, y, n_splits=n_iter, train_size=tr, val_size=va, test_size=te)

    mae_val, r2_val = [], []
    mae_test, r2_test = [], []
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
        mae_val.append(mean_absolute_error(y.iloc[vai], model.predict(X_va)))
        r2_val.append(r2_score(y.iloc[vai], model.predict(X_va)))
        mae_test.append(mean_absolute_error(y.iloc[tei], model.predict(X_te)))
        r2_test.append(r2_score(y.iloc[tei], model.predict(X_te)))

    out = {"ratio": label, "train_val_test": f"{tr*100:.0f}-{va*100:.0f}-{te*100:.0f}"}

    for prefix, values in [("MAE_val", mae_val), ("R2_val", r2_val),
                           ("MAE_test", mae_test), ("R2_test", r2_test)]:
        ci = _ci(values)
        out[f"{prefix}_mean"] = ci["mean"]
        out[f"{prefix}_std"] = ci["std"]
        out[f"{prefix}_sem"] = ci["sem"]
        out[f"{prefix}_sampling_error"] = ci["sampling_error"]
        out[f"{prefix}_ci_lower"] = ci["ci_lower"]
        out[f"{prefix}_ci_upper"] = ci["ci_upper"]

    # diff test - val (sesgo optimismo validación)
    out["MAE_diff_test_minus_val"] = out["MAE_test_mean"] - out["MAE_val_mean"]
    out["R2_diff_test_minus_val"] = out["R2_test_mean"] - out["R2_val_mean"]

    return out


def plot_results(df: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot")
        return

    colors = ["#4C72B0", "#DD8452", "#55A868"]
    x = np.arange(len(df))
    width = 0.3

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8))

    # MAE validation
    ax1.bar(x - width / 2, df["MAE_val_mean"], width,
            yerr=df["MAE_val_sampling_error"], capsize=4,
            color=colors, edgecolor="black", linewidth=0.8, label="Val")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["ratio"])
    ax1.set_ylabel("MAE")
    ax1.set_title("MAE — Validation set")
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_axisbelow(True)

    # MAE test
    ax2.bar(x - width / 2, df["MAE_test_mean"], width,
            yerr=df["MAE_test_sampling_error"], capsize=4,
            color=colors, edgecolor="black", linewidth=0.8, label="Test")
    ax2.set_xticks(x)
    ax2.set_xticklabels(df["ratio"])
    ax2.set_ylabel("MAE")
    ax2.set_title("MAE — Test set")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_axisbelow(True)

    # R² validation + test side-by-side
    ax3.bar(x - width / 2, df["R2_val_mean"], width,
            yerr=df["R2_val_sampling_error"], capsize=4,
            color=colors, edgecolor="black", linewidth=0.8, label="Val")
    ax3.bar(x + width / 2, df["R2_test_mean"], width,
            yerr=df["R2_test_sampling_error"], capsize=4,
            color=[c + "80" for c in colors], edgecolor="black", linewidth=0.8, label="Test")
    ax3.axhline(y=0, color="gray", linestyle="--", linewidth=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(df["ratio"])
    ax3.set_ylabel("R²")
    ax3.set_title("R² — Validation vs Test")
    ax3.legend(fontsize=8)
    ax3.grid(axis="y", alpha=0.3)
    ax3.set_axisbelow(True)

    # Sampling error comparison
    ax4.bar(x - width / 2, df["MAE_val_sampling_error"], width,
            color=colors, edgecolor="black", linewidth=0.8, label="Val")
    ax4.bar(x + width / 2, df["MAE_test_sampling_error"], width,
            color=[c + "80" for c in colors], edgecolor="black", linewidth=0.8, label="Test")
    ax4.set_xticks(x)
    ax4.set_xticklabels(df["ratio"])
    ax4.set_ylabel("95% sampling error")
    ax4.set_title("Sampling error (MAE) — Val vs Test")
    ax4.legend(fontsize=8)
    ax4.grid(axis="y", alpha=0.3)
    ax4.set_axisbelow(True)

    plt.tight_layout()
    path = output_dir / "sampling_error.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="py -m src.analysis.sampling_error",
        description="Sampling error (val + test) for MMCV split ratios using LinearRegression.",
    )
    parser.add_argument("--task", type=int, required=True, choices=[2, 6, 7])
    parser.add_argument("--window", required=True, help="e.g. 30 for W30")
    parser.add_argument("--experiment", default="raw", choices=["raw", "zscores", "rawzscore"])
    parser.add_argument("--target", default="MOT_V4")
    parser.add_argument("--splitting", nargs="+", required=True,
                        choices=list(RATIOS.keys()),
                        help="Split ratio(s) to test")
    parser.add_argument("--n-iter", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics-dir", default="data/processed/metrics")
    parser.add_argument("--metadata", default="data/raw/metadata.xlsx")
    parser.add_argument("--output", default="outputs/sampling_error")
    args = parser.parse_args()

    np.random.seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data: T{args.task} W{args.window} {args.experiment} -> {args.target}")
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
        print(f"    MAE_val={r['MAE_val_mean']:.4f} ±{r['MAE_val_sampling_error']:.4f}")
        print(f"    MAE_test={r['MAE_test_mean']:.4f} ±{r['MAE_test_sampling_error']:.4f}")
        print(f"    R²_val={r['R2_val_mean']:.4f} ±{r['R2_val_sampling_error']:.4f}")
        print(f"    R²_test={r['R2_test_mean']:.4f} ±{r['R2_test_sampling_error']:.4f}")
        print(f"    MAE_test-val diff={r['MAE_diff_test_minus_val']:.4f}")

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

    print("\n========== MAE Validation ==========")
    print(df[["ratio", "MAE_val_mean", "MAE_val_sampling_error", "R2_val_mean", "R2_val_sampling_error"]].to_string(index=False))
    print("\n========== MAE Test ==========")
    print(df[["ratio", "MAE_test_mean", "MAE_test_sampling_error", "R2_test_mean", "R2_test_sampling_error"]].to_string(index=False))
    print("\n========== Diff Test - Val ==========")
    print(df[["ratio", "MAE_diff_test_minus_val", "R2_diff_test_minus_val"]].to_string(index=False))


if __name__ == "__main__":
    main()
