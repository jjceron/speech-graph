import streamlit as st
import numpy as np
import plotly.graph_objects as go
from utils.loader import list_completed, get_targets, get_experiments, get_windows, load_best_report
from utils.loader import load_test_iterations, load_val_iterations, load_predictions
from utils.plots import (
    scatter_obs_vs_pred, scatter_target_vs_pred_raw,
    residual_plot,
    target_distribution_plot, compute_subject_metrics,
    plot_target_vs_predicted,
)
from utils.sidebar import render_sidebar

st.set_page_config(page_title="Distributions", page_icon="📈", layout="wide")
render_sidebar()
st.title("📈 Split Distributions")

completed = list_completed()
if not completed:
    st.warning("No completed experiments.")
    st.stop()

col_w, col_e, col_t = st.columns(3)
with col_w:
    windows = sorted(set(w for w, _ in completed))
    window = st.selectbox("Window", windows, index=0)
with col_e:
    exps = [e for w, e in completed if w == window]
    experiment = st.selectbox("Experiment", exps, index=0)
with col_t:
    target = st.selectbox("Target", get_targets(), index=0)

test_df = load_test_iterations(window, experiment, target)
val_df = load_val_iterations(window, experiment, target)
pred_df = load_predictions(window, experiment, target)

if test_df is None:
    st.warning("No test data found.")
    st.stop()

report = load_best_report(window, experiment, target)
if report:
    bp = report.get("best_params", {})
    feat = report.get("selected_features", [])
    ts = report["test_summary"]
    st.info(
        f"**Best model:** {bp.get('regressor', '?')} | "
        f"**Features:** {bp.get('rfe_n_features', '?')} selected | "
        f"**MAE test:** {ts['mae_mean_test']:.4f} [{ts['mae_ci_lower_test']:.4f}, {ts['mae_ci_upper_test']:.4f}] | "
        f"**R² test:** {ts['r2_mean_test']:.4f} [{ts['r2_ci_lower_test']:.4f}, {ts['r2_ci_upper_test']:.4f}]"
    )

tab1, tab3, tab4, tab5 = st.tabs(["Test/Val Distributions", "Obs vs Pred", "Residuals", "Target Distribution"])

with tab1:
    for label, col_name, val_color, test_color in [
        ("MAE", "mae", "#ff7f0e", "#2ca02c"),
        ("R²", "r2", "#d62728", "#1f77b4"),
        ("ρ (Spearman)", "rho", "#8c564b", "#9467bd"),
    ]:
        v = val_df[col_name].dropna().values if val_df is not None else np.array([])
        t = test_df[col_name].dropna().values
        if col_name == "rho":
            v = v[np.isfinite(v)]
            t = t[np.isfinite(t)]

        if len(t) == 0:
            st.info(f"{label} not computable for this target.")
            continue

        fig = go.Figure()
        if len(v) > 0:
            fig.add_trace(go.Histogram(
                x=v, nbinsx=30,
                name=f"Val (μ={v.mean():.4f})",
                marker_color=val_color, opacity=0.6,
            ))
        fig.add_trace(go.Histogram(
            x=t, nbinsx=30,
            name=f"Test (μ={t.mean():.4f})",
            marker_color=test_color, opacity=0.6,
        ))

        t_mean = float(t.mean())
        t_lo = float(np.percentile(t, 2.5))
        t_hi = float(np.percentile(t, 97.5))

        fig.add_vline(x=t_mean, line_dash="dash", line_color="red", line_width=2,
                       annotation_text=f"μ={t_mean:.4f}")
        fig.add_vline(x=t_lo, line_dash="dot", line_color="red", line_width=1, opacity=0.4)
        fig.add_vline(x=t_hi, line_dash="dot", line_color="red", line_width=1, opacity=0.4)

        fig.update_layout(
            title=f"{label} — Val vs Test",
            barmode="overlay",
            template="plotly_white",
            height=300,
        )
        if col_name == "r2":
            fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.4)

        st.plotly_chart(fig, use_container_width=True)

with tab3:
    if pred_df is not None and len(pred_df) > 0:
        set_options = sorted(pred_df["set"].dropna().unique())
        default_idx = set_options.index("TEST") if "TEST" in set_options else 0
        set_choice = st.selectbox("Set", set_options, index=default_idx)

        subset = pred_df[pred_df["set"] == set_choice]
        n_pred_unique = subset["y_pred"].nunique()
        if n_pred_unique == 1:
            const_val = float(subset["y_pred"].iloc[0])
            st.warning(
                f"Predictions are constant ({const_val:.4f}) for this configuration. "
                "Correlation is not computable and the model may be collapsing to the "
                "mean/median target."
            )

        subject_df = compute_subject_metrics(pred_df, set_name=set_choice)
        if len(subject_df) == 0:
            st.info(f"No {set_choice} predictions available.")
        else:
            scenario_label = f"W{window} {experiment}  {target}"

            fig_scatter = plot_target_vs_predicted(
                subject_df, set_name=set_choice,
                scenario_label=scenario_label,
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("No predictions data available.")

with tab4:
    if pred_df is not None and len(pred_df) > 0:
        test_preds = pred_df[pred_df["set"] == "TEST"]
        if len(test_preds) > 0:
            fig = residual_plot(test_preds["y_true"].values, test_preds["y_pred"].values)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No test predictions available.")
    else:
        st.info("No predictions data available.")

with tab5:
    if pred_df is not None and len(pred_df) > 0:
        test_preds = pred_df[pred_df["set"] == "TEST"]
        if len(test_preds) > 0:
            y_true = test_preds["y_true"].values
            mae_val = float(test_df["mae"].mean()) if test_df is not None else 0
            rmse_val = float(test_df["rmse"].mean()) if test_df is not None else 0
            fig = target_distribution_plot(y_true, mae_val, rmse_val)
            st.plotly_chart(fig, use_container_width=True)
            tmin, tmax = y_true.min(), y_true.max()
            tmean, tstd = y_true.mean(), y_true.std()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Target range", f"{tmin:.3f} – {tmax:.3f}")
            col2.metric("Target mean ± std", f"{tmean:.3f} ± {tstd:.3f}")
            col3.metric("MAE (% of range)", f"{mae_val:.3f} ({mae_val/(tmax-tmin)*100:.1f}%)")
            col4.metric("RMSE (% of range)", f"{rmse_val:.3f} ({rmse_val/(tmax-tmin)*100:.1f}%)")
        else:
            st.info("No test predictions available.")
    else:
        st.info("No predictions data available.")
