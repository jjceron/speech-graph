from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess


def histogram_simple(df_op: pd.DataFrame, name_op: str, n_bins: int = 30) -> go.Figure:
    values = df_op[name_op].dropna()

    q1 = values.quantile(0.25)
    q2 = values.quantile(0.50)
    q3 = values.quantile(0.75)
    mean_val = values.mean()

    fig = px.histogram(
        df_op,
        x=name_op,
        nbins=n_bins,
        title="Distribution %s" % name_op,
        color_discrete_sequence=["#C8A2C8"],
        marginal="box",
        opacity=0.75,
    )

    fig.update_traces(marker_line_color="purple", marker_line_width=1)

    fig.add_vline(x=q1, line_dash="dash", line_color="blue", line_width=2)
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="lines", line=dict(color="blue", dash="dash", width=2), name="Q1"
        )
    )

    fig.add_vline(x=q2, line_dash="dash", line_color="green", line_width=2)
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="lines", line=dict(color="green", dash="dash", width=2), name="Q2 (Median)"
        )
    )

    fig.add_vline(x=q3, line_dash="dash", line_color="orange", line_width=2)
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="lines", line=dict(color="orange", dash="dash", width=2), name="Q3"
        )
    )

    fig.add_vline(x=mean_val, line_dash="dash", line_color="red", line_width=2)
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="lines", line=dict(color="red", dash="dash", width=2), name="Mean"
        )
    )

    fig.update_layout(template="plotly_white", legend_title="Statistics")
    return fig


def compute_stats_text(series: pd.Series) -> str:
    s = series.dropna()
    if len(s) == 0:
        return ""
    mean = s.mean()
    median = s.median()
    std = s.std()
    rng = s.max() - s.min()
    iqr = s.quantile(0.75) - s.quantile(0.25)
    return f"Media: {mean:.2f}, Median: {median:.2f}, Std: {std:.2f}, Range: {rng:.2f}, IQR: {iqr:.2f}"


def join_plot_loess(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    n_bins: int = 30,
) -> go.Figure:
    valid = df[[x_col, y_col]].dropna()
    x = valid[x_col].values
    y = valid[y_col].values

    if len(x) < 5:
        fig = go.Figure()
        fig.update_layout(title=title, template="plotly_white")
        return fig

    rho, pval = spearmanr(x, y)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(size=5, opacity=0.6, color="#1f77b4"),
            name="Data",
        )
    )

    try:
        loess_result = lowess(y, x, frac=0.3, it=3, return_sorted=True)
        sorted_idx = np.argsort(x)
        fig.add_trace(
            go.Scatter(
                x=loess_result[:, 0],
                y=loess_result[:, 1],
                mode="lines",
                line=dict(color="red", width=2),
                name="LOESS",
            )
        )
    except Exception:
        pass

    rho_text = f"ρ = {rho:.4f}"
    p_text = f"p = {pval:.4e}" if pval < 0.001 else f"p = {pval:.4f}"

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.95,
        y=0.95,
        text=f"<b>{rho_text}</b><br>{p_text}",
        showarrow=False,
        font=dict(size=12),
        align="right",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1,
    )

    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        template="plotly_white",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
