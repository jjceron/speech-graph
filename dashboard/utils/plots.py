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

    colors = ["#2ca02c" if s else "#d9534f" for s in df["selected"]]
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
        height=max(350, 28 * len(df) + 30),
        margin=dict(l=max(130, 10 * max_name_len + 40), r=20, t=40, b=40),
        yaxis=dict(tickfont=dict(size=11)),
        shapes=[{
            "type": "line", "x0": 0.5, "y0": n_sel - 0.5,
            "x1": df["ranking"].max() + 1, "y1": n_sel - 0.5,
            "line": {"color": "gray", "width": 1, "dash": "dot"},
        }] if n_sel < len(df) else [],
    )
    return fig


def optimization_history(df: pd.DataFrame) -> go.Figure:
    df = df.dropna(subset=["value"]).copy()
    if "state" in df.columns:
        df = df[df["state"] == "COMPLETE"]
    df = df.sort_values("number")
    best = df["value"].cummin()

    y_lower = best.iloc[-1] * 0.90
    first_val = df["value"].iloc[0]
    q90 = df["value"].quantile(0.90)
    y_upper = min(first_val, q90) * 1.05

    fig = go.Figure()
    for reg in sorted(df["params_regressor"].unique()):
        mask = df["params_regressor"] == reg
        fig.add_trace(go.Scatter(
            x=df.loc[mask, "number"],
            y=df.loc[mask, "value"],
            mode="markers",
            marker=dict(size=4, opacity=0.5),
            name=reg,
            hovertemplate=f"<b>{reg}</b><br>Trial: %{{x}}<br>MAE val: %{{y:.4f}}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=df["number"], y=best, mode="lines",
        line=dict(color="red", width=2), name="Best so far",
    ))
    fig.update_layout(
        title="Optuna Optimization History",
        xaxis_title="Trial",
        yaxis_title="Objective (MAE validation)",
        template="plotly_white",
        height=450,
        hovermode="x unified",
    )
    fig.update_yaxes(range=[y_lower, y_upper])
    return fig


def plot_optimization_ecdf(df: pd.DataFrame) -> go.Figure:
    dfp = df.dropna(subset=["value", "params_regressor"]).copy()
    if "state" in dfp.columns:
        dfp = dfp[dfp["state"] == "COMPLETE"]
    if len(dfp) == 0:
        return go.Figure()

    best_val = dfp["value"].min()

    fig = go.Figure()
    for i, reg in enumerate(sorted(dfp["params_regressor"].unique())):
        vals = np.sort(dfp[dfp["params_regressor"] == reg]["value"].values)
        ecdf = np.arange(1, len(vals) + 1) / len(vals)
        c = px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
        fig.add_trace(go.Scatter(
            x=vals, y=ecdf, mode="lines",
            name=reg, legendgroup=reg, line=dict(width=2.5, color=c),
            fill="tozeroy",
            fillcolor=f"rgba({int(c[1:3], 16)},{int(c[3:5], 16)},{int(c[5:7], 16)},0.15)",
            hovertemplate=(
                f"<b>{reg}</b><br>"
                "MAE: %{x:.4f}<br>"
                "Frac ≤ threshold: %{y:.0%}<extra></extra>"
            ),
        ))
        med = float(np.median(vals))
        fig.add_trace(go.Scatter(
            x=[med], y=[0.5], mode="markers",
            marker=dict(size=11, symbol="diamond", color=c,
                        line=dict(width=1.5, color="black")),
            legendgroup=reg,
            showlegend=False,
            hovertemplate=(
                f"<b>{reg}</b><br>"
                f"Median MAE: {med:.4f}<extra></extra>"
            ),
        ))

    data_range = dfp["value"].max() - best_val
    x_range = [dfp["value"].max() + data_range * 0.05, best_val - data_range * 0.18]

    fig.add_vline(x=best_val, line_dash="dash", line_color="red", line_width=1.5)
    fig.add_annotation(
        x=best_val, y=0.97,
        text=f"Best: {best_val:.4f}",
        showarrow=True, arrowhead=1, arrowsize=1.2,
        ax=40, ay=-30,
        font=dict(size=13, color="red"),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="red",
        borderwidth=1.5,
    )

    fig.update_xaxes(range=x_range, autorange=False)
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(
        title="Trials Achieving Threshold (ECDF per Regressor)",
        xaxis_title="Objective (MAE val) — worse → better",
        yaxis_title="Trials ≤ threshold (%)",
        margin=dict(t=95),
        template="plotly_white", height=450,
        legend=dict(orientation="h", yanchor="bottom", y=0.925, xanchor="right", x=1),
    )
    return fig


def model_selection_bar(df: pd.DataFrame) -> go.Figure:
    dfp = df.dropna(subset=["value", "params_regressor"]).copy()
    if "state" in dfp.columns:
        dfp = dfp[dfp["state"] == "COMPLETE"]
    if len(dfp) == 0:
        return go.Figure()

    grp = dfp.groupby("params_regressor").agg(
        trials=("value", "count"),
        best_mae=("value", "min"),
        mean_mae=("value", "mean"),
        mean_nf=("params_rfe_n_features", "mean"),
    ).reset_index().sort_values("trials", ascending=True)

    global_best = dfp["value"].min()
    best_reg = grp.loc[(grp["best_mae"] - global_best).abs().idxmin(), "params_regressor"]

    bar_colors = grp["best_mae"]
    line_widths = [3 if r == best_reg else 0.8 for r in grp["params_regressor"]]
    line_colors = ["#222" if r == best_reg else "rgba(0,0,0,0.3)" for r in grp["params_regressor"]]

    text_template = [
        f"<b>{r}</b><br>Trials: {int(t)}<br>Best MAE: {b:.4f}<br>Mean MAE: {m:.4f}<br>Avg N Features: {n:.1f}{'<br><b>◀ BEST</b>' if r == best_reg else ''}<extra></extra>"
        for r, t, b, m, n in zip(grp["params_regressor"], grp["trials"], grp["best_mae"], grp["mean_mae"], grp["mean_nf"])
    ]

    fig = go.Figure(go.Bar(
        y=grp["params_regressor"],
        x=grp["trials"],
        orientation="h",
        marker=dict(
            color=bar_colors,
            colorscale="RdYlGn_r",
            showscale=True,
            colorbar=dict(title="MAE Val"),
            line=dict(width=line_widths, color=line_colors),
        ),
        text=grp["trials"],
        textposition="outside",
        textfont=dict(
            size=12,
            color=["#222" if r == best_reg else "#333" for r in grp["params_regressor"]],
            weight=["bold" if r == best_reg else "normal" for r in grp["params_regressor"]],
        ),
        hovertemplate=text_template,
        customdata=grp[["best_mae", "mean_mae", "mean_nf"]].to_numpy(),
    ))
    ticktexts = [
        f"<b>{r}</b>" if r == best_reg else r
        for r in grp["params_regressor"]
    ]
    fig.update_yaxes(ticktext=ticktexts, tickvals=grp["params_regressor"])
    fig.update_layout(
        title="Completed Trials per Regressor",
        xaxis_title="Completed Trials",
        yaxis_title="",
        template="plotly_white",
        height=400,
        margin=dict(l=160),
    )
    return fig


def plot_objective_by_regressor(df: pd.DataFrame) -> go.Figure:
    dfp = df.dropna(subset=["value", "params_regressor"]).copy()
    if "state" in dfp.columns:
        dfp = dfp[dfp["state"] == "COMPLETE"]
    reg_order = dfp.groupby("params_regressor")["value"].median().sort_values().index.tolist()
    fig = go.Figure()
    for i, reg in enumerate(reg_order):
        vals = dfp[dfp["params_regressor"] == reg]["value"]
        fig.add_trace(go.Box(
            y=vals, name=reg, legendgroup=reg, boxmean="sd",
            boxpoints="outliers",
            marker_color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)],
            width=0.5,
        ))
    fig.update_layout(
        title="Objective Distribution per Regressor",
        yaxis_title="Objective (MAE val)",
        yaxis=dict(autorange=True),
        template="plotly_white", height=400,
    )
    return fig


def plot_parameter_importance(df: pd.DataFrame, height: int | None = None, param_cols: list | None = None) -> go.Figure:
    dfp = df.dropna(subset=["value"])
    if param_cols is None:
        param_cols = [c for c in dfp.columns
                      if c.startswith("params_") and c not in ("params_regressor", "params_use_scaler")]
    importances = []
    for col in param_cols:
        sub = dfp[[col, "value"]].dropna()
        if len(sub) < 3 or sub[col].nunique() < 3:
            continue
        try:
            corr = sub["value"].corr(sub[col].astype(float), method="spearman")
        except (ValueError, TypeError):
            continue
        if not np.isfinite(corr):
            continue
        importances.append((col.replace("params_", ""), corr))

    importances.sort(key=lambda x: abs(x[1]), reverse=True)
    labels = [x[0] for x in importances]
    corrs = [x[1] for x in importances]
    colors = ["#d62728" if c > 0 else "#2ca02c" for c in corrs]

    fig = go.Figure(go.Bar(
        x=corrs, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{c:.3f}" for c in corrs],
        textposition="outside",
    ))
    fig.update_layout(
        title="Parameter Importance (Spearman ρ with objective)",
        xaxis_title="Spearman ρ with MAE val  (← better | worse →)",
        template="plotly_white",
        height=height if height is not None else max(250, len(labels) * 35),
        margin=dict(l=160),
    )
    fig.add_vline(x=0, line_dash="dot", line_color="gray")
    return fig


def plot_regressor_nfeatures_heatmap(df: pd.DataFrame, metric: str = "mae") -> go.Figure:
    if metric == "mae":
        value_col = "value"
        aggfunc = "min"
        ascending = True
        colorscale = "RdYlGn_r"
        label = "Best MAE"
        title = "Best MAE by Regressor × N Features"
    else:
        value_col = "user_attrs_r2_mean_val"
        aggfunc = "max"
        ascending = False
        colorscale = "RdYlGn"
        label = "Best R²"
        title = "Best R² by Regressor × N Features"

    dfp = df.dropna(subset=[value_col, "params_regressor", "params_rfe_n_features"]).copy()
    if "state" in dfp.columns:
        dfp = dfp[dfp["state"] == "COMPLETE"]
    if len(dfp) == 0:
        return go.Figure()
    dfp["params_rfe_n_features"] = dfp["params_rfe_n_features"].astype(int)

    pivot_best = dfp.pivot_table(
        index="params_regressor", columns="params_rfe_n_features",
        values=value_col, aggfunc=aggfunc,
    )
    pivot_cnt = dfp.pivot_table(
        index="params_regressor", columns="params_rfe_n_features",
        values=value_col, aggfunc="count",
    )

    row_order = dfp.groupby("params_regressor")[value_col].agg(aggfunc).sort_values(ascending=ascending).index.tolist()
    pivot_best = pivot_best.reindex(row_order)
    pivot_cnt = pivot_cnt.reindex(row_order)

    zmin = pivot_best.min().min()
    zmax = pivot_best.max().max()
    best_val = pivot_best.min().min() if ascending else pivot_best.max().max()
    best_pos = None

    hover = []
    texts = []
    for i in range(len(pivot_best.index)):
        row_h = []
        row_t = []
        for j in range(len(pivot_best.columns)):
            v = pivot_best.iloc[i, j]
            c = pivot_cnt.iloc[i, j]
            if pd.isna(v):
                row_h.append(f"<b>{pivot_best.index[i]}</b><br>N={int(pivot_best.columns[j])}<br>No trials")
                row_t.append("")
            else:
                is_best = (ascending and v == best_val) or (not ascending and v == best_val)
                row_t.append(f"{v:.4f}")
                row_h.append(
                    f"<b>{pivot_best.index[i]}</b><br>"
                    f"N={int(pivot_best.columns[j])}<br>"
                    f"{label}: {v:.4f}<br>"
                    f"Trials: {int(c)}"
                    f"{'<br><b>◀ BEST</b>' if is_best else ''}"
                )
                if is_best:
                    best_pos = (i, j)
        hover.append(row_h)
        texts.append(row_t)

    fig = go.Figure(go.Heatmap(
        z=pivot_best.values,
        x=[str(int(c)) for c in pivot_best.columns],
        y=pivot_best.index,
        colorscale=colorscale,
        zmin=zmin, zmax=zmax,
        text=texts, texttemplate="%{text}", textfont=dict(size=11),
        hovertext=hover, hovertemplate="%{hovertext}<extra></extra>",
        colorbar=dict(title=label, thickness=15),
        xgap=2, ygap=2,
    ))

    if best_pos:
        fig.add_shape(
            type="rect",
            x0=best_pos[1] - 0.5, x1=best_pos[1] + 0.5,
            y0=best_pos[0] - 0.5, y1=best_pos[0] + 0.5,
            xref="x", yref="y",
            line=dict(color="#222", width=3.5),
            fillcolor="rgba(0,0,0,0)",
            layer="above",
        )

    fig.update_layout(
        title=title,
        xaxis_title="RFE N Features",
        yaxis_title="",
        template="plotly_white",
        height=max(250, len(pivot_best.index) * 45 + 60),
        margin=dict(l=160),
    )
    return fig


def plot_regressor_performance_summary(df: pd.DataFrame) -> go.Figure:
    grp = df.dropna(subset=["value", "params_regressor"]).groupby("params_regressor").agg(
        trials=("value", "count"),
        best_mae=("value", "min"),
        mean_nfeatures=("params_rfe_n_features", "mean"),
    ).reset_index()

    fig = go.Figure(go.Scatter(
        x=grp["trials"],
        y=grp["best_mae"],
        mode="markers+text",
        marker=dict(
            size=grp["mean_nfeatures"].fillna(5) * 3,
            sizemode="area",
            sizeref=2 * max(grp["mean_nfeatures"].fillna(5)) / (40 ** 2),
            color=grp["mean_nfeatures"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Avg N Features"),
            line=dict(width=1, color="black"),
        ),
        text=grp["params_regressor"],
        textposition="top center",
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Trials: %{x}<br>"
            "Best MAE: %{y:.4f}<br>"
            "Avg N Features: %{marker.color:.1f}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title="Regressor Performance Summary",
        xaxis_title="Trials (frequency)",
        yaxis_title="Best MAE val (lower is better)",
        template="plotly_white",
        height=450,
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
    metric: str = "MAE",
) -> go.Figure:
    df = subject_df.copy()

    all_vals = list(df["y_pred_lower"]) + list(df["y_pred_upper"]) + list(df["y_true_mean"])
    lo, hi = min(all_vals), max(all_vals)
    margin = (hi - lo) * 0.05 if hi > lo else 1
    lims = [lo - margin, hi + margin]

    is_mae = metric == "MAE"
    color_col = "mae" if is_mae else "r2"
    metric_label = "MAE" if is_mae else "R²"
    colorscale = "RdYlGn_r" if is_mae else "RdYlBu"
    hover_line = f"{metric_label}: %{{customdata[4]:.4f}}" if is_mae else f"{metric_label}: %{{customdata[5]:.4f}}"

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["y_pred_mean"], y=df["y_true_mean"],
        mode="markers",
        marker=dict(
            size=8,
            color=df[color_col],
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(title=metric_label),
        ),
        name="Subjects",
        error_x=dict(
            type="data", symmetric=False,
            array=df["y_pred_upper"] - df["y_pred_mean"],
            arrayminus=df["y_pred_mean"] - df["y_pred_lower"],
            visible=True, thickness=1.5, width=5,
        ),
        customdata=df[["subject", "y_pred_lower", "y_pred_upper",
                        "n_predictions", "mae", "r2"]],
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Predicted: %{x:.3f} [%{customdata[1]:.3f}, %{customdata[2]:.3f}]<br>"
            "Observed: %{y:.3f}<br>"
            + hover_line + "<br>"
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
