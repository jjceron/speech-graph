from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import spearmanr

TARGET_COLORS = {"MOT": "#1f77b4", "COG": "#ff7f0e", "MOT_V4": "#2ca02c", "COG_V1": "#d62728"}
EXPERIMENT_LABELS = {"raw": "Raw (13)", "zscores": "Z-scores (9)", "rawzscore": "Raw+Z (22)"}


def bar_r2_comparison(
    df, metric: str = "r2", title: str = "R² test [IC 95%]"
) -> go.Figure:
    fig = go.Figure()
    for target in df["target"].unique():
        tdf = df[df["target"] == target]
        fig.add_trace(
            go.Bar(
                name=target,
                x=tdf["label"],
                y=tdf[f"{metric}_mean"],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=tdf[f"{metric}_upper"] - tdf[f"{metric}_mean"],
                    arrayminus=tdf[f"{metric}_mean"] - tdf[f"{metric}_lower"],
                    visible=True,
                    thickness=1,
                    width=3,
                ),
                marker_color=TARGET_COLORS.get(target, "#333"),
                text=tdf[f"{metric}_mean"].round(3).astype(str),
                textposition="outside",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Window — Experiment",
        yaxis_title=metric.upper(),
        barmode="group",
        template="plotly_white",
        height=400,
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


def scatter_obs_vs_pred(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    rho, _ = spearmanr(y_true, y_pred)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_pred,
            mode="markers",
            marker=dict(size=3, opacity=0.3, color="steelblue"),
            name="Predictions",
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
        title=f"Observed vs Predicted (Spearman ρ = {rho:.4f})",
        xaxis_title="Observed",
        yaxis_title="Predicted",
        template="plotly_white",
        height=500,
        width=500,
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
    df = df.sort_values("ranking")
    colors = ["#2ca02c" if s else "#d62728" for s in df["selected"]]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["ranking"],
            y=df["feature"],
            orientation="h",
            marker_color=colors,
            text=df["ranking"],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="RFE Feature Ranking (1 = best)",
        xaxis_title="Ranking",
        yaxis_title="Feature",
        template="plotly_white",
        height=max(300, 25 * len(df)),
        margin=dict(l=200),
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


def optuna_parallel_coords(df: pd.DataFrame, params: list[str]) -> go.Figure:
    dfp = df.dropna(subset=["value"]).copy()
    dims = []
    for p in params:
        if p in dfp.columns:
            col = dfp[p].copy()
            if col.dtype == "object":
                codes, _ = pd.factorize(col)
                dims.append(
                    go.parcats.Dimension(
                        label=p.replace("params_", ""),
                        values=col,
                        tickvals=col.unique().tolist(),
                    )
                )
            else:
                dims.append(
                    go.parcats.Dimension(
                        label=p.replace("params_", ""),
                        values=col,
                    )
                )
    if not dims:
        return go.Figure()
    dims.append(
        go.parcats.Dimension(
            label="Objective (MAE)",
            values=dfp["value"],
        )
    )
    fig = go.Figure(
        go.Parcats(
            dimensions=dims,
            line=dict(color=dfp["value"], colorscale="Viridis_r", showscale=True),
        )
    )
    fig.update_layout(title="Parallel Coordinates — Hyperparameters vs Objective", height=600)
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
