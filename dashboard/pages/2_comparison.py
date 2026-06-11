import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.loader import (
    list_completed,
    ALL_TARGETS,
    load_best_report,
    EXPERIMENTS,
    WINDOWS,
)
from utils.plots import bar_r2_comparison

st.set_page_config(page_title="Cross-Experiment", page_icon="📊", layout="wide")
st.title("📊 Cross-Experiment Comparison")

completed = list_completed()
target = st.selectbox("Target", ALL_TARGETS, index=0)

rows = []
for w, e in completed:
    r = load_best_report(w, e, target)
    if r:
        ts = r.get("test_summary", {})
        rows.append(
            {
                "label": f"W{w} — {e}",
                "window": w,
                "experiment": e,
                "target": target,
                "r2_mean": ts.get("r2_mean_test", 0),
                "r2_lower": ts.get("r2_ci_lower_test", 0),
                "r2_upper": ts.get("r2_ci_upper_test", 0),
                "mae_mean": ts.get("mae_mean_test", 0),
                "mae_lower": ts.get("mae_ci_lower_test", 0),
                "mae_upper": ts.get("mae_ci_upper_test", 0),
                "rho_mean": ts.get("rho_mean_test", 0) or 0,
                "rho_lower": ts.get("rho_ci_lower_test", 0) or 0,
                "rho_upper": ts.get("rho_ci_upper_test", 0) or 0,
                "pct_below": ts.get("r2_below_zero_test", 0) * 100,
            }
        )

df = pd.DataFrame(rows)

if df.empty:
    st.warning("No data for this target.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    fig_r2 = bar_r2_comparison(df, "r2", f"R² test — {target}")
    st.plotly_chart(fig_r2, use_container_width=True)

with col2:
    fig_mae = bar_r2_comparison(df, "mae", f"MAE test — {target}")
    st.plotly_chart(fig_mae, use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    fig_rho = bar_r2_comparison(df, "rho", f"Spearman ρ test — {target}")
    st.plotly_chart(fig_rho, use_container_width=True)

with col4:
    fig_below = go.Figure()
    fig_below.add_trace(
        go.Bar(
            x=df["label"],
            y=df["pct_below"],
            marker_color="#d62728",
            text=df["pct_below"].round(1).astype(str) + "%",
            textposition="outside",
        )
    )
    fig_below.update_layout(
        title=f"% R² < 0 — {target}",
        xaxis_title="Window — Experiment",
        yaxis_title="% Splits",
        template="plotly_white",
        height=400,
    )
    st.plotly_chart(fig_below, use_container_width=True)

if target == "COG_V1":
    st.markdown("""
    **Insight:** COG_V1 is the only target with positive R² — only in **raw** and **rawzscore** experiments.
    Z-scores alone lose the signal entirely. W20 raw shows the best performance (R² ≈ 0.041, MAE ≈ 0.970).
    """)
elif target == "MOT":
    st.markdown("""
    **Insight:** MOT shows weak, inconsistent signal. W10 zscores (R² ≈ 0.023) and W20 zscores (R² ≈ 0.031)
    are the best performers. Raw features turn negative at W20.
    """)
