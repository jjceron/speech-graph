from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import spearmanr

TARGET_COLORS = {"MOT": "#1f77b4", "COG": "#ff7f0e", "MOT_V4": "#2ca02c", "COG_V1": "#d62728"}
EXPERIMENT_LABELS = {"raw": "Raw (13)", "zscores": "Z-scores (9)", "rawzscore": "Raw+Z (22)"}


def bar_r2_comparison(
    df, metric: str = "r2", title: str = "R² test [IC 95%]", suffix: str = ""
) -> go.Figure:
    fig = go.Figure()
    for target in df["target"].unique():
        tdf = df[df["target"] == target]
        mean_col = f"{metric}{suffix}_mean"
        upper_col = f"{metric}{suffix}_upper"
        lower_col = f"{metric}{suffix}_lower"
        ci_upper = tdf[upper_col] - tdf[mean_col]
        ci_lower = tdf[mean_col] - tdf[lower_col]
        fig.add_trace(
            go.Scatter(
                name=target,
                x=tdf["label"],
                y=tdf[mean_col],
                mode="markers+text",
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=ci_upper,
                    arrayminus=ci_lower,
                    visible=True,
                    thickness=1.5,
                    width=5,
                ),
                marker=dict(size=8, color=TARGET_COLORS.get(target, "#333")),
                text=tdf[mean_col].round(3).astype(str),
                textposition="top center",
                textfont=dict(size=9),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Window — Experiment",
        yaxis_title=title.split(" ")[0],
        template="plotly_white",
        height=450,
        legend_title="Target",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    return fig


def hist_metric(values: np.ndarray, label: str, color: str, bins: int = 40) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=values,
            nbinsx=bins,
            marker_color=color,
            opacity=0.75,
            name=label,
        )
    )
    mean_val = float(np.mean(values))
    fig.add_vline(x=mean_val, line_dash="dash", line_color="red", line_width=2)
    fig.add_annotation(
        x=mean_val,
        y=0.95,
        yref="paper",
        text=f"Mean = {mean_val:.4f}",
        showarrow=False,
        font=dict(size=11, color="red"),
    )
    fig.update_layout(
        title=f"{label} distribution",
        xaxis_title=label,
        yaxis_title="Splits",
        template="plotly_white",
        height=300,
        bargap=0.05,
    )
    return fig


def scatter_obs_vs_pred(pred_df: pd.DataFrame, set_name: str = "TEST") -> go.Figure:
    subj_stats = pred_df.groupby("subject")[["y_true", "y_pred"]].agg(["mean", "std"])
    subj_stats.columns = ["y_true_mean", "y_true_std", "y_pred_mean", "y_pred_std"]
    subj_stats = subj_stats.reset_index()

    y_true_mean = subj_stats["y_true_mean"].values
    y_pred_mean = subj_stats["y_pred_mean"].values

    hover_text = []
    for _, row in subj_stats.iterrows():
        hover_text.append(
            f"<b>{row['subject']}</b><br>"
            f"y_true: {row['y_true_mean']:.5f} ± {row['y_true_std']:.5f}<br>"
            f"y_pred: {row['y_pred_mean']:.5f} ± {row['y_pred_std']:.5f}"
        )

    rho, _ = spearmanr(y_true_mean, y_pred_mean)
    margin = (y_true_mean.max() - y_true_mean.min()) * 0.05 or 1
    lim_min = min(y_true_mean.min(), y_pred_mean.min()) - margin
    lim_max = max(y_true_mean.max(), y_pred_mean.max()) + margin
    lims = [lim_min, lim_max]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_true_mean,
            y=y_pred_mean,
            mode="markers",
            marker=dict(size=6, opacity=0.7, color="steelblue"),
            name=f"{set_name} (per subject)",
            text=hover_text,
            hoverinfo="text",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=lims,
            y=lims,
            mode="lines",
            line=dict(dash="dash", color="red", width=1.5),
            name="Identity",
        )
    )
    fig.update_layout(
        title=f"Observed vs Predicted — {set_name} (Spearman ρ = {rho:.4f})",
        xaxis_title="Observed (mean per subject)",
        yaxis_title="Predicted (mean per subject)",
        template="plotly_white",
        height=500,
        width=500,
        xaxis=dict(scaleanchor="y", scaleratio=1),
    )
    return fig


def scatter_target_vs_pred_raw(pred_df: pd.DataFrame, set_name: str = "TEST") -> go.Figure:
    df = pred_df[pred_df["set"] == set_name].copy()
    if len(df) == 0:
        return go.Figure()
    df["error"] = df["y_pred"] - df["y_true"]
    df["abs_error"] = df["error"].abs()
    lim_min = min(df["y_true"].min(), df["y_pred"].min())
    lim_max = max(df["y_true"].max(), df["y_pred"].max())
    margin = (lim_max - lim_min) * 0.05 if lim_max > lim_min else 1
    lims = [lim_min - margin, lim_max + margin]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["y_pred"], y=df["y_true"],
        mode="markers",
        marker=dict(size=5, opacity=0.55),
        name=set_name,
        customdata=df[["subject", "split", "error", "abs_error"]],
        hovertemplate=(
            "<b>Subject:</b> %{customdata[0]}<br>"
            "<b>Split:</b> %{customdata[1]}<br>"
            "<b>Predicted:</b> %{x:.3f}<br>"
            "<b>Observed:</b> %{y:.3f}<br>"
            "<b>Error:</b> %{customdata[2]:.3f}<br>"
            "<b>Absolute error:</b> %{customdata[3]:.3f}"
            "<extra></extra>"
        ),
    ))
    fig.add_trace(go.Scatter(
        x=lims, y=lims,
        mode="lines",
        line=dict(dash="dash", color="black", width=1.5),
        name="Perfect prediction",
    ))
    fig.update_layout(
        title=f"Target vs Predicted — {set_name}",
        xaxis_title="Predicted",
        yaxis_title="Target / Observed",
        template="plotly_white",
        height=550,
        xaxis=dict(scaleanchor="y", scaleratio=1),
    )
    return fig


def scatter_r2_val_vs_test(r2_val: np.ndarray, r2_test: np.ndarray) -> go.Figure:
    lim = min(r2_val.min(), r2_test.min()), max(r2_val.max(), r2_test.max())
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=r2_val,
            y=r2_test,
            mode="markers",
            marker=dict(size=4, opacity=0.4, color="steelblue"),
            name="Splits",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=lim,
            y=lim,
            mode="lines",
            line=dict(dash="dash", color="red", width=1.5),
            name="Identity",
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4)
    fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.4)
    fig.update_layout(
        title="R² validation vs test (per split)",
        xaxis_title="R² validation",
        yaxis_title="R² test",
        template="plotly_white",
        height=500,
        width=500,
        xaxis=dict(scaleanchor="y", scaleratio=1),
    )
    return fig


def rfe_ranking_chart(df: pd.DataFrame) -> go.Figure:
    df = df.copy()
    df["feature"] = df["feature"].str.replace(r"_[A-Z]\d+[A-Z]\w+$", "", regex=True)

    df["_sort"] = (~df["selected"]).astype(int)
    df = df.sort_values(["_sort", "ranking", "feature"])

    colors = ["#2ca02c" if s else "#aaaaaa" for s in df["selected"]]
    n_sel = df["selected"].sum()
    max_name_len = df["feature"].str.len().max()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["ranking"], y=df["feature"],
        orientation="h", marker_color=colors,
        text=df["ranking"], textposition="outside",
    ))
    fig.update_layout(
        title="RFE Feature Ranking (1 = best)",
        xaxis_title="Ranking",
        yaxis_title="Feature",
        template="plotly_white",
        height=max(300, 25 * len(df)),
        margin=dict(l=max(100, 9 * max_name_len + 10)),
        shapes=[{
            "type": "line", "x0": 0.5, "y0": n_sel - 0.5,
            "x1": df["ranking"].max() + 1, "y1": n_sel - 0.5,
            "line": {"color": "gray", "width": 1, "dash": "dot"},
        }] if n_sel < len(df) else [],
    )
    return fig


def optimization_history(df: pd.DataFrame) -> go.Figure:
    df = df.dropna(subset=["value"]).sort_values("number")
    best = df["value"].cummin()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["number"],
            y=df["value"],
            mode="markers",
            marker=dict(size=4, opacity=0.4, color="steelblue"),
            name="Trial",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["number"],
            y=best,
            mode="lines",
            line=dict(color="red", width=2),
            name="Best so far",
        )
    )
    fig.update_layout(
        title="Optuna Optimization History",
        xaxis_title="Trial",
        yaxis_title="Objective (MAE validation)",
        template="plotly_white",
        height=400,
    )
    return fig


def model_selection_bar(df: pd.DataFrame) -> go.Figure:
    counts = df["params_regressor"].value_counts()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=counts.index,
            y=counts.values,
            marker_color=px.colors.qualitative.Plotly[: len(counts)],
            text=counts.values,
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Regressor Selection Frequency (300 trials)",
        xaxis_title="Regressor",
        yaxis_title="Trials",
        template="plotly_white",
        height=400,
    )
    return fig


def optuna_parallel_coords(df: pd.DataFrame, params: list[str], top_k: int = 100) -> go.Figure:
    dfp = df.dropna(subset=["value"]).copy()
    if len(dfp) == 0:
        return go.Figure()

    if top_k < len(dfp):
        dfp = dfp.nsmallest(top_k, "value")

    dims = []
    for p in params:
        if p in dfp.columns:
            col = dfp[p]
            label = p.replace("params_", "")
            if col.dtype == "object":
                codes, labels = pd.factorize(col)
                dims.append(
                    dict(
                        label=label,
                        values=codes,
                        tickvals=list(range(len(labels))),
                        ticktext=[str(l) for l in labels],
                    )
                )
            else:
                sorted_col = col.dropna()
                if len(sorted_col) == 0:
                    continue
                try:
                    cmin, cmax = float(sorted_col.min()), float(sorted_col.max())
                except (ValueError, TypeError):
                    continue
                if not (np.isfinite(cmin) and np.isfinite(cmax)):
                    continue
                dims.append(
                    dict(
                        label=label,
                        values=col,
                        range=[cmin, cmax],
                    )
                )

    if not dims:
        return go.Figure()

    fig = go.Figure(
        go.Parcoords(
            dimensions=dims,
            line=dict(
                color=dfp["value"],
                colorscale="Viridis_r",
                showscale=True,
                colorbar=dict(title="MAE val"),
            ),
        )
    )
    fig.update_layout(
        title=f"Parallel Coordinates — Top {top_k} of {len(df.dropna(subset=['value']))} Trials",
        height=600,
    )
    return fig


def residual_plot(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    residuals = y_pred - y_true
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=residuals,
            mode="markers",
            marker=dict(size=3, opacity=0.3, color="steelblue"),
            name="Residuals",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", line_width=1.5)
    fig.update_layout(
        title="Residuals (y_pred - y_true) vs Observed",
        xaxis_title="Observed",
        yaxis_title="Residual",
        template="plotly_white",
        height=400,
    )
    return fig


def forest_plot(all_data: list[dict]) -> go.Figure:
    fig = go.Figure()
    for row in all_data:
        ci_low = row["r2_mean"] - row["r2_lower"]
        ci_high = row["r2_upper"] - row["r2_mean"]
        fig.add_trace(
            go.Scatter(
                x=[row["r2_mean"]],
                y=[row["label"]],
                mode="markers",
                marker=dict(size=8, color=row.get("color", "#1f77b4")),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[ci_high],
                    arrayminus=[ci_low],
                    visible=True,
                    thickness=1.5,
                    width=5,
                ),
                showlegend=False,
            )
        )
    fig.add_vline(x=0, line_dash="dash", line_color="red", line_width=1.5, opacity=0.7)
    fig.add_annotation(x=0, y=1.02, yref="paper", text="Null (R²=0)", showarrow=False, font=dict(color="red", size=10))
    fig.update_layout(
        title="All Scenarios — R² test [IC 95%]",
        xaxis_title="R² test",
        template="plotly_white",
        height=max(400, 30 * len(all_data)),
        margin=dict(l=250),
    )
    return fig


def target_distribution_plot(y_true: np.ndarray, mae: float, rmse: float) -> go.Figure:
    mean_val = float(y_true.mean())
    std_val = float(y_true.std())
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=y_true,
            nbinsx=40,
            marker_color="#1f77b4",
            opacity=0.75,
            name="Target",
        )
    )
    fig.add_vline(x=mean_val, line_dash="dash", line_color="red", line_width=2)
    fig.add_vline(x=mean_val - mae, line_dash="dot", line_color="green", line_width=1.5)
    fig.add_vline(x=mean_val + mae, line_dash="dot", line_color="green", line_width=1.5)
    fig.update_layout(
        title=f"Target Distribution (μ={mean_val:.3f}, σ={std_val:.3f})",
        xaxis_title="Target value",
        yaxis_title="Frequency",
        template="plotly_white",
        height=350,
    )
    return fig


EXPERIMENT_COLORS = {"raw": "#1f77b4", "zscores": "#ff7f0e", "rawzscore": "#2ca02c"}


def metric_comparison_chart(
    df: pd.DataFrame,
    metric: str,
    metric_label: str,
    title: str,
    show_hline_zero: bool = False,
    color_col: str = "experiment",
) -> go.Figure:
    fig = go.Figure()
    for exp in df[color_col].unique():
        edf = df[df[color_col] == exp].sort_values("label")
        fig.add_trace(
            go.Bar(
                name=EXPERIMENT_LABELS.get(exp, exp),
                x=edf["label"],
                y=edf[f"{metric}_mean"],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=edf[f"{metric}_upper"] - edf[f"{metric}_mean"],
                    arrayminus=edf[f"{metric}_mean"] - edf[f"{metric}_lower"],
                    visible=True,
                    thickness=1.2,
                    width=3,
                ),
                marker_color=EXPERIMENT_COLORS.get(exp, "#333"),
                opacity=0.85,
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Scenario",
        yaxis_title=metric_label,
        barmode="group",
        template="plotly_white",
        height=400,
        legend_title="Experiment",
    )
    if show_hline_zero:
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    return fig


def compute_subject_metrics(pred_df: pd.DataFrame, set_name: str = "TEST",
                             standardize: bool = False) -> pd.DataFrame:
    df = pred_df[pred_df["set"] == set_name].copy()
    if len(df) == 0:
        return pd.DataFrame()

    if standardize:
        mu = float(df["y_true"].mean())
        sigma = float(df["y_true"].std()) or 1.0
        df["y_true"] = (df["y_true"] - mu) / sigma
        df["y_pred"] = (df["y_pred"] - mu) / sigma

    def _metrics(g):
        abs_err = (g["y_pred"] - g["y_true"]).abs()
        err = g["y_pred"] - g["y_true"]
        n = len(g)
        z = 1.96

        mae = float(abs_err.mean())
        mae_std = float(abs_err.std()) if n > 1 else 0.0
        mae_ci = z * mae_std / np.sqrt(n) if n > 1 else 0.0

        y_t_mean = float(g["y_true"].mean())
        y_t_std = float(g["y_true"].std()) if n > 1 else 0.0
        y_t_ci = z * y_t_std / np.sqrt(n) if n > 1 else 0.0

        y_p_mean = float(g["y_pred"].mean())
        y_p_std = float(g["y_pred"].std()) if n > 1 else 0.0
        y_p_ci = z * y_p_std / np.sqrt(n) if n > 1 else 0.0

        ss_res = ((g["y_true"] - g["y_pred"]) ** 2).sum()
        ss_tot = ((g["y_true"] - y_t_mean) ** 2).sum()
        r2 = 0.0 if ss_tot == 0 else 1 - float(ss_res / ss_tot)

        return pd.Series({
            "mae": mae,
            "mae_std": mae_std,
            "mae_lower": max(0.0, mae - mae_ci),
            "mae_upper": mae + mae_ci,
            "n_predictions": n,
            "y_true_mean": y_t_mean,
            "y_true_std": y_t_std,
            "y_true_lower": y_t_mean - y_t_ci,
            "y_true_upper": y_t_mean + y_t_ci,
            "y_pred_mean": y_p_mean,
            "y_pred_std": y_p_std,
            "y_pred_lower": y_p_mean - y_p_ci,
            "y_pred_upper": y_p_mean + y_p_ci,
            "bias_mean": float(err.mean()),
            "r2": r2,
        })

    subject_df = df.groupby("subject", sort=False).apply(_metrics).reset_index()
    subject_df["subject_short"] = (
        subject_df["subject"].astype(str).str.split("-").str[-1].str[:8]
    )
    return subject_df.sort_values("mae", ascending=True)


def plot_subject_mae(subject_df: pd.DataFrame, title: str = "MAE by subject") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=subject_df["subject_short"],
            y=subject_df["mae"],
            mode="markers+text",
            text=subject_df["mae"].round(3).astype(str),
            textposition="top center",
            customdata=subject_df[
                ["subject", "n_predictions", "y_true_mean", "y_pred_mean", "bias_mean"]
            ],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "MAE: %{y:.3f}<br>"
                "N predictions: %{customdata[1]}<br>"
                "y_true mean: %{customdata[2]:.3f}<br>"
                "y_pred mean: %{customdata[3]:.3f}<br>"
                "Bias mean: %{customdata[4]:.3f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Subject",
        yaxis_title="MAE",
        template="plotly_white",
        height=500,
    )
    return fig


def plot_subject_metric(
    subject_df: pd.DataFrame,
    metric: str = "MAE",
    title: str = "",
) -> go.Figure:
    is_mae = metric.upper() == "MAE"
    if is_mae:
        subject_df = subject_df.sort_values("mae", ascending=True).reset_index(drop=True)
        y_col, y_label = "mae", "MAE"
    else:
        subject_df = subject_df.sort_values("r2", ascending=False).reset_index(drop=True)
        y_col, y_label = "r2", "R²"

    x = subject_df["subject_short"]
    y = subject_df[y_col]
    has_ci = is_mae and "mae_lower" in subject_df.columns and "mae_upper" in subject_df.columns

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers",
        marker=dict(size=8, color="#1f77b4" if is_mae else "#d62728"),
        error_y=dict(
            type="data",
            symmetric=False,
            array=(subject_df["mae_upper"] - y) if has_ci else [],
            arrayminus=(y - subject_df["mae_lower"]) if has_ci else [],
            visible=has_ci,
            thickness=1.5,
            width=5,
        ) if has_ci else None,
        customdata=subject_df[["subject", "n_predictions", "y_true_mean", "y_pred_mean", "bias_mean",
                               "mae" if not is_mae else "r2"]],
        hovertemplate=(
            f"<b>{{customdata[0]}}</b><br>"
            f"{y_label}: {{y:.4f}}<br>"
            f"{'MAE: %{customdata[5]:.4f}<br>' if not is_mae else 'R²: %{customdata[5]:.4f}<br>'}"
            f"N predictions: {{customdata[1]}}<br>"
            f"y_true mean: {{customdata[2]:.3f}}<br>"
            f"y_pred mean: {{customdata[3]:.3f}}<br>"
            f"Bias mean: {{customdata[4]:.3f}}<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=title or f"{y_label} by subject",
        xaxis_title="Subject",
        yaxis_title=y_label,
        template="plotly_white",
        height=500,
    )
    if not is_mae:
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    return fig


def plot_subject_obs_pred_paired(
    subject_df: pd.DataFrame,
    set_name: str = "TEST",
    scenario_label: str = "",
) -> go.Figure:
    df = subject_df.sort_values("y_true_mean", ascending=True).reset_index(drop=True)
    x = df["subject_short"]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x, y=df["y_true_mean"],
        mode="markers",
        name="Observed / y_true",
        marker=dict(size=8, color="#1f77b4"),
        error_y=dict(
            type="data", symmetric=False,
            array=df["y_true_upper"] - df["y_true_mean"],
            arrayminus=df["y_true_mean"] - df["y_true_lower"],
            visible=True, thickness=1.5, width=5,
        ),
        customdata=df[["subject", "n_predictions",
                        "y_true_lower", "y_true_upper",
                        "y_pred_mean", "y_pred_lower", "y_pred_upper",
                        "mae"]],
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Observed: %{y:.3f} [%{customdata[2]:.3f}, %{customdata[3]:.3f}]<br>"
            "Predicted: %{customdata[4]:.3f} [%{customdata[5]:.3f}, %{customdata[6]:.3f}]<br>"
            "MAE: %{customdata[7]:.4f}<br>"
            "N predictions: %{customdata[1]}<extra></extra>"
        ),
    ))

    fig.add_trace(go.Scatter(
        x=x, y=df["y_pred_mean"],
        mode="markers",
        name="Predicted / y_pred",
        marker=dict(size=8, color="#ff7f0e"),
        error_y=dict(
            type="data", symmetric=False,
            array=df["y_pred_upper"] - df["y_pred_mean"],
            arrayminus=df["y_pred_mean"] - df["y_pred_lower"],
            visible=True, thickness=1.5, width=5,
        ),
        customdata=df[["subject", "n_predictions",
                        "y_true_mean", "y_true_lower", "y_true_upper",
                        "mae"]],
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Predicted: %{y:.3f}<br>"
            "Observed: %{customdata[2]:.3f} [%{customdata[3]:.3f}, %{customdata[4]:.3f}]<br>"
            "MAE: %{customdata[5]:.4f}<br>"
            "N predictions: %{customdata[1]}<extra></extra>"
        ),
    ))

    title = f"Observed and Predicted by subject — {set_name} [95% CI]"
    if scenario_label:
        title += f" — {scenario_label}"

    fig.update_layout(
        title=title,
        xaxis_title="Subject",
        yaxis_title="Target value",
        template="plotly_white",
        height=550,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_subject_obs_pred_scatter(
    subject_df: pd.DataFrame,
    set_name: str = "TEST",
    scenario_label: str = "",
) -> go.Figure:
    df = subject_df.copy()

    lims_data = list(df["y_pred_lower"]) + list(df["y_pred_upper"]) \
              + list(df["y_true_lower"]) + list(df["y_true_upper"])
    lim_min = min(lims_data)
    lim_max = max(lims_data)
    margin = (lim_max - lim_min) * 0.05 if lim_max > lim_min else 1
    lims = [lim_min - margin, lim_max + margin]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["y_pred_mean"], y=df["y_true_mean"],
        mode="markers",
        marker=dict(size=8, color="#1f77b4"),
        name="Subjects",
        error_x=dict(
            type="data", symmetric=False,
            array=df["y_pred_upper"] - df["y_pred_mean"],
            arrayminus=df["y_pred_mean"] - df["y_pred_lower"],
            visible=True, thickness=1.5, width=5,
        ),
        error_y=dict(
            type="data", symmetric=False,
            array=df["y_true_upper"] - df["y_true_mean"],
            arrayminus=df["y_true_mean"] - df["y_true_lower"],
            visible=True, thickness=1.5, width=5,
        ),
        customdata=df[["subject", "n_predictions",
                        "y_pred_lower", "y_pred_upper",
                        "y_true_lower", "y_true_upper",
                        "mae"]],
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Predicted: %{x:.3f} [%{customdata[2]:.3f}, %{customdata[3]:.3f}]<br>"
            "Observed: %{y:.3f} [%{customdata[4]:.3f}, %{customdata[5]:.3f}]<br>"
            "MAE: %{customdata[6]:.4f}<br>"
            "N predictions: %{customdata[1]}<extra></extra>"
        ),
    ))

    fig.add_trace(go.Scatter(
        x=lims, y=lims,
        mode="lines",
        line=dict(dash="dash", color="black", width=1.5),
        name="Perfect prediction",
    ))

    title = f"Observed vs Predicted by subject — {set_name} [95% CI]"
    if scenario_label:
        title += f" — {scenario_label}"

    fig.update_layout(
        title=title,
        xaxis_title="Predicted mean per subject",
        yaxis_title="Target / Observed mean per subject",
        template="plotly_white",
        height=550,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_target_vs_predicted(
    subject_df: pd.DataFrame,
    set_name: str = "TEST",
    scenario_label: str = "",
    marker_color: str = "#1f77b4",
) -> go.Figure:
    df = subject_df.copy()

    all_vals = list(df["y_pred_lower"]) + list(df["y_pred_upper"]) + list(df["y_true_mean"])
    lo, hi = min(all_vals), max(all_vals)
    margin = (hi - lo) * 0.05 if hi > lo else 1
    lims = [lo - margin, hi + margin]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["y_pred_mean"], y=df["y_true_mean"],
        mode="markers",
        marker=dict(size=8, color=marker_color),
        name="Subjects",
        error_x=dict(
            type="data", symmetric=False,
            array=df["y_pred_upper"] - df["y_pred_mean"],
            arrayminus=df["y_pred_mean"] - df["y_pred_lower"],
            visible=True, thickness=1.5, width=5,
        ),
        customdata=df[["subject", "y_pred_lower", "y_pred_upper",
                        "n_predictions", "mae"]],
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Predicted: %{x:.3f} [%{customdata[1]:.3f}, %{customdata[2]:.3f}]<br>"
            "Observed: %{y:.3f}<br>"
            "MAE: %{customdata[4]:.4f}<br>"
            "N test predictions: %{customdata[3]}<extra></extra>"
        ),
    ))

    fig.add_trace(go.Scatter(
        x=lims, y=lims,
        mode="lines",
        line=dict(dash="dash", color="black", width=1.5),
        name="Perfect prediction",
    ))

    title = f"Target vs Predicted — {set_name}"
    if scenario_label:
        title += f" — {scenario_label}"

    fig.update_layout(
        title=title,
        xaxis_title="Predicted mean [95% CI]",
        yaxis_title="Target / Observed",
        template="plotly_white",
        height=500,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
