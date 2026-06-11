import streamlit as st
import numpy as np
import plotly.graph_objects as go
from utils.loader import list_completed, ALL_TARGETS, EXPERIMENTS, WINDOWS, load_best_report
from utils.plots import scatter_obs_vs_pred, scatter_r2_val_vs_test, hist_metric, residual_plot
from utils.loader import load_test_iterations, load_val_iterations, load_predictions

st.set_page_config(page_title="Distributions", page_icon="📈", layout="wide")
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
    target = st.selectbox("Target", ALL_TARGETS, index=0)

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
    st.info(
        f"**Best model:** {bp.get('regressor', '?')} | "
        f"**Features:** {bp.get('rfe_n_features', '?')} selected | "
        f"**R² test:** {report['test_summary']['r2_mean_test']:.4f} [{report['test_summary']['r2_ci_lower_test']:.4f}, {report['test_summary']['r2_ci_upper_test']:.4f}]"
    )

tab1, tab2, tab3, tab4 = st.tabs(["Test Distributions", "Val vs Test", "Obs vs Pred", "Residuals"])

with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        r2 = test_df["r2"].dropna().values
        fig = hist_metric(r2, "R²", "#1f77b4")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        mae = test_df["mae"].dropna().values
        fig = hist_metric(mae, "MAE", "#2ca02c")
        st.plotly_chart(fig, use_container_width=True)
    with col3:
        rho = test_df["rho"].dropna()
        rho = rho[np.isfinite(rho)].values
        if len(rho) > 0:
            fig = hist_metric(rho, "ρ", "#9467bd")
        else:
            fig = None
            st.info("ρ not computable for this target.")
        if fig:
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    if val_df is not None:
        r2_v = val_df["r2"].dropna().values
        r2_t = test_df["r2"].dropna().values
        min_len = min(len(r2_v), len(r2_t))
        if min_len >= 10:
            r2_v, r2_t = r2_v[:min_len], r2_t[:min_len]
            fig_val = scatter_r2_val_vs_test(r2_v, r2_t)
            st.plotly_chart(fig_val, use_container_width=True)
        else:
            st.info("Too few splits for scatter plot.")

        col1, col2, col3 = st.columns(3)
        figs_val = []
        for label, col_name, color in [
            ("R²", "r2", "#1f77b4"),
            ("MAE", "mae", "#2ca02c"),
            ("ρ", "rho", "#9467bd"),
        ]:
            v = val_df[col_name].dropna().values
            if col_name == "rho":
                v = v[np.isfinite(v)]
            t = test_df[col_name].dropna().values
            if col_name == "rho":
                t = t[np.isfinite(t)]
            if len(v) > 0 and len(t) > 0:
                lo = min(v.min(), t.min())
                hi = max(v.max(), t.max())
                bins = np.linspace(lo, hi, 31)
            fig = go.Figure()
            if len(v) > 0:
                fig.add_trace(go.Histogram(x=v, bins=bins if len(v) > 0 else 30, name=f"Val ({v.mean():.4f})", marker_color=color, opacity=0.6))
            if len(t) > 0:
                fig.add_trace(go.Histogram(x=t, bins=bins if len(t) > 0 else 30, name=f"Test ({t.mean():.4f})", marker_color=color, opacity=0.6))
            fig.update_layout(title=f"{label} — val vs test", barmode="overlay", template="plotly_white", height=300)
            if label in ("R²",):
                fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.4)
            figs_val.append(fig)

        if figs_val[0]:
            with col1:
                st.plotly_chart(figs_val[0], use_container_width=True)
            with col2:
                st.plotly_chart(figs_val[1], use_container_width=True)
            with col3:
                st.plotly_chart(figs_val[2], use_container_width=True)
    else:
        st.info("No validation data available.")

with tab3:
    if pred_df is not None and len(pred_df) > 0:
        test_preds = pred_df[pred_df["set"] == "TEST"]
        if len(test_preds) > 0:
            fig = scatter_obs_vs_pred(test_preds["y_true"].values, test_preds["y_pred"].values)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No test predictions available.")
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
