import streamlit as st
import numpy as np
import plotly.graph_objects as go
from utils.loader import list_completed, get_targets, get_experiments, get_windows, load_best_report
from utils.loader import load_test_iterations, load_val_iterations
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
    window = st.selectbox("Window", get_windows(), index=0)
with col_e:
    experiment = st.selectbox("Experiment", [e for e in get_experiments() if get_targets(window=window, experiment=e)], index=0)
with col_t:
    target = st.selectbox("Target", get_targets(), index=0)

test_df = load_test_iterations(window, experiment, target)
val_df = load_val_iterations(window, experiment, target)

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

st.subheader("Test/Val Distributions")

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

        fig.add_vline(x=t_mean, line_dash="dash", line_color=test_color, line_width=2,
                       annotation_text=f"μ<sub>t</sub>={t_mean:.4f}",
                       annotation_position="top right")
        fig.add_vline(x=t_lo, line_dash="dot", line_color=test_color, line_width=1, opacity=0.4)
        fig.add_vline(x=t_hi, line_dash="dot", line_color=test_color, line_width=1, opacity=0.4)

        if len(v) > 0:
            v_mean = float(v.mean())
            v_lo = float(np.percentile(v, 2.5))
            v_hi = float(np.percentile(v, 97.5))
            fig.add_vline(x=v_mean, line_dash="dash", line_color=val_color, line_width=2,
                           annotation_text=f"μ<sub>v</sub>={v_mean:.4f}",
                           annotation_position="top left")
            fig.add_vline(x=v_lo, line_dash="dot", line_color=val_color, line_width=1, opacity=0.4)
            fig.add_vline(x=v_hi, line_dash="dot", line_color=val_color, line_width=1, opacity=0.4)

        fig.update_layout(
            title=f"{label} — Val vs Test",
            barmode="overlay",
            template="plotly_white",
            height=450,
        )
        if col_name == "r2":
            fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.4)

        st.plotly_chart(fig, use_container_width=True)
