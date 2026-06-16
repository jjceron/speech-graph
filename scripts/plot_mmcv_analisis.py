"""Plot sampling error and trial time vs. number of MMCV splits
with 95 % confidence intervals across completed trials.

Generates one interactive HTML per regressor found.

Usage:
    py scripts/plot_mmcv_analisis.py
    py scripts/plot_mmcv_analisis.py --results-dir outputs/regression_optuna/analisis_mmcv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CSV_PATTERN = re.compile(
    r"optuna_trials_n(\d+)_task\d+_W\d+_raw_MOT_V4_rfefixed_mae_(.+)\.csv"
)


def find_regressors(results_dir: Path) -> list[str]:
    regressors: set[str] = set()
    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir() or not subdir.name.startswith("n"):
            continue
        for f in subdir.iterdir():
            if f.suffix != ".csv" or not f.name.startswith("optuna_trials"):
                continue
            m = CSV_PATTERN.match(f.name)
            if m:
                regressors.add(m.group(2))
    return sorted(regressors)


def load_trials_df(results_dir: Path, regressor: str) -> pd.DataFrame:
    rows = []
    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir() or not subdir.name.startswith("n"):
            continue
        n_iter_match = re.match(r"n(\d+)", subdir.name)
        if not n_iter_match:
            continue
        n_iter = int(n_iter_match.group(1))

        for f in subdir.iterdir():
            if not f.name.startswith("optuna_trials") or f.suffix != ".csv":
                continue
            m = CSV_PATTERN.match(f.name)
            if m and m.group(2) == regressor:
                df = pd.read_csv(f)
                df["n_iter"] = n_iter
                rows.append(df)
                break

    if not rows:
        raise FileNotFoundError(f"No CSV files found for regressor '{regressor}'")

    return pd.concat(rows, ignore_index=True)


def compute_ci(values: pd.Series) -> dict:
    vals = values.dropna().astype(float).values
    n = len(vals)
    if n < 2:
        return {"mean": float(vals[0]) if n == 1 else np.nan, "ci": np.nan, "n": n}
    mean = float(np.mean(vals))
    sem = float(sp_stats.sem(vals))
    t_crit = float(sp_stats.t.ppf(0.975, df=n - 1))
    ci = t_crit * sem
    return {"mean": mean, "ci": ci, "n": n}


def compute_overlap_counts(complete: pd.DataFrame, n_iters: list[int]) -> list[dict]:
    records = []
    for n_iter in n_iters:
        subset = complete[complete["n_iter"] == n_iter].copy()
        subset = subset.drop_duplicates(subset=["user_attrs_mae_mean_val"])
        if len(subset) < 2:
            records.append({"n_iter": n_iter, "n_overlap": 0, "n_unique": len(subset)})
            continue

        best_idx = subset["user_attrs_mae_mean_val"].idxmin()
        ci_lower_best = subset.loc[best_idx, "user_attrs_mae_ci_lower_val"]
        ci_upper_best = subset.loc[best_idx, "user_attrs_mae_ci_upper_val"]

        count = 0
        for i in subset.index:
            if i == best_idx:
                continue
            ci_lower_i = subset.loc[i, "user_attrs_mae_ci_lower_val"]
            ci_upper_i = subset.loc[i, "user_attrs_mae_ci_upper_val"]
            if ci_lower_i <= ci_upper_best and ci_lower_best <= ci_upper_i:
                count += 1

        records.append({"n_iter": n_iter, "n_overlap": count, "n_unique": len(subset)})

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot sampling error and trial time vs. MMCV splits",
    )
    parser.add_argument("--results-dir", default="outputs/regression_optuna/analisis_mmcv")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--n-iter-start", type=int, default=100)
    parser.add_argument("--n-iter-end", type=int, default=1000)
    parser.add_argument("--n-iter-step", type=int, default=100)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    regressors = find_regressors(results_dir)
    if not regressors:
        print(f"No regressor CSV files found in {results_dir}")
        sys.exit(1)

    n_iters = list(range(args.n_iter_start, args.n_iter_end + 1, args.n_iter_step))

    for regressor in regressors:
        print(f"Processing regressor: {regressor}")
        full = load_trials_df(results_dir, regressor)

        complete = full[full["state"] == "COMPLETE"].copy()

        # --- Figura 1: violín (sampling error, eje y1) + scatter (tiempo con IC, eje y2) ---
        fig = go.Figure()

        time_records = []
        for n_iter in n_iters:
            subset = complete[complete["n_iter"] == n_iter]

            # Violin trace for sampling error (eje y1)
            samperror_vals = subset["user_attrs_mae_samperror_val"].dropna().astype(float)
            if len(samperror_vals) > 0:
                bw = max(samperror_vals.std() * 0.3, 0.001)
                fig.add_trace(go.Violin(
                    x=[n_iter] * len(samperror_vals),
                    y=samperror_vals,
                    name=f"n={n_iter}",
                    legendgroup=f"n{n_iter}",
                    showlegend=False,
                    scalemode="count",
                    bandwidth=bw,
                    points=False,
                    box_visible=True,
                    line=dict(color="rgba(0,100,200,0.6)"),
                    fillcolor="rgba(0,100,200,0.15)",
                    meanline_visible=True,
                ))

            # CI for trial time
            trial_time = compute_ci(subset["user_attrs_trial_time_seconds"])
            time_records.append({
                "n_iter": n_iter,
                "n_trials": trial_time["n"],
                "time_mean": trial_time["mean"],
                "time_ci": trial_time["ci"],
            })

        time_summary = pd.DataFrame(time_records)
        time_summary = time_summary.dropna(subset=["time_mean"])
        time_summary["time_pct_change"] = time_summary["time_mean"].pct_change() * 100
        time_summary["time_slope"] = time_summary["time_mean"].diff() / time_summary["n_iter"].diff()
        time_summary["prev_n_iter"] = time_summary["n_iter"].shift()

        fig.add_trace(go.Scatter(
            x=time_summary["n_iter"],
            y=time_summary["time_mean"],
            error_y=dict(
                type="data",
                array=time_summary["time_ci"],
                visible=True,
                color="rgba(200,50,50,0.5)",
            ),
            mode="markers+lines",
            showlegend=False,
            line=dict(color="rgba(200,50,50,1)", width=2, dash="dash"),
            marker=dict(color="rgba(200,50,50,1)", size=8),
            yaxis="y2",
        ))

        fig.update_layout(
            title=dict(
                text=f"Sampling error y tiempo por trial vs. n_iter — {regressor}",
                x=0.5,
            ),
            xaxis=dict(
                title="Número de splits MMCV (n_iter)",
                dtick=args.n_iter_step,
            ),
            yaxis=dict(
                title=dict(
                    text="Sampling error MAE val",
                    font=dict(color="rgba(0,100,200,1)"),
                ),
                tickfont=dict(color="rgba(0,100,200,1)"),
                zeroline=False,
            ),
            yaxis2=dict(
                title=dict(
                    text="Tiempo por trial (s)",
                    font=dict(color="rgba(200,50,50,1)"),
                ),
                tickfont=dict(color="rgba(200,50,50,1)"),
                overlaying="y",
                side="right",
                zeroline=False,
            ),
            hovermode="x unified",
            template="plotly_white",
            legend=dict(x=0.02, y=0.98),
            margin=dict(l=80, r=80, t=60, b=60),
            height=800,
            width=1200,
        )

        # Hover text for time trace
        hover_texts = []
        for _, row in time_summary.iterrows():
            extra_str = ""
            if pd.notna(row["time_slope"]):
                extra_str += (
                    f"Pendiente: {row['time_slope']:+.4f} s/split<br>"
                    f"Cambio vs n_{int(row['prev_n_iter'])}: {row['time_pct_change']:+.2f}%"
                )
            hover_texts.append(
                f"n_iter={int(row['n_iter'])}<br>"
                f"Trials completados: {int(row['n_trials'])}<br>"
                f"Tiempo: {row['time_mean']:.2f} ± {row['time_ci']:.2f} s<br>"
                f"{extra_str}"
            )
        fig.data[-1].hovertext = hover_texts

        out_path = output_dir / f"{regressor}.html"
        fig.write_html(str(out_path))
        print(f"  Saved → {out_path}")

        # --- Figura 2: bar chart de superposición de IC ---
        overlap_records = compute_overlap_counts(complete, n_iters)
        overlap_df = pd.DataFrame(overlap_records)
        overlap_df = overlap_df[overlap_df["n_unique"] > 0]

        fig2 = go.Figure()

        fig2.add_trace(go.Bar(
            x=overlap_df["n_iter"],
            y=overlap_df["n_overlap"],
            marker_color="rgba(0,150,100,0.8)",
            marker_line_color="rgba(0,150,100,1)",
            marker_line_width=1.5,
            name="Trials con IC superpuesto",
        ))

        n_overlap_bars = overlap_df["n_iter"].nunique()
        bar_width = max(800, n_overlap_bars * 150)

        max_y = overlap_df["n_overlap"].max()
        y_max_padded = max_y + max(1, int(max_y * 0.3))
        y_dtick = 1 if max_y <= 10 else max(1, round(max_y / 8))

        fig2.update_layout(
            title=dict(
                text=f"Trials con IC de MAE val superpuesto al mejor trial — {regressor}",
                x=0.5,
            ),
            xaxis=dict(
                title="Número de splits MMCV (n_iter)",
                dtick=args.n_iter_step,
                showline=True,
                linewidth=1,
                linecolor="gray",
            ),
            yaxis=dict(
                title="N° de trials con IC superpuesto",
                dtick=y_dtick,
                tick0=0,
                range=[-0.5, y_max_padded],
                zeroline=True,
                showline=True,
                linewidth=1,
                linecolor="gray",
            ),
            hovermode="x",
            template="plotly_white",
            margin=dict(l=80, r=80, t=60, b=60),
            width=bar_width,
            height=600,
        )

        # Add hover with details
        hover_texts2 = []
        for _, row in overlap_df.iterrows():
            hover_texts2.append(
                f"n_iter={int(row['n_iter'])}<br>"
                f"Trials únicos (sin duplicados MAE val): {int(row['n_unique'])}<br>"
                f"Trials con IC superpuesto al mejor: {int(row['n_overlap'])}"
            )
        fig2.data[0].hovertext = hover_texts2

        out_path2 = output_dir / f"{regressor}_overlap.html"
        fig2.write_html(str(out_path2))
        print(f"  Saved → {out_path2}")


if __name__ == "__main__":
    main()
