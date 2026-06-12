import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.loader import list_completed, get_targets, load_predictions, load_best_report
from utils.sidebar import render_sidebar

st.set_page_config(page_title="Subjects", page_icon="👤", layout="wide")
render_sidebar()
st.title("👤 Per-Subject Analysis")

completed = list_completed()
if not completed:
    st.warning("No completed experiments.")
    st.stop()

col_w, col_e, col_t = st.columns(3)
with col_w:
    windows = sorted(set(w for w, _ in completed))
    window = st.selectbox("Window", windows, index=0, key="sub_w")
with col_e:
    exps = [e for w, e in completed if w == window]
    experiment = st.selectbox("Experiment", exps, index=0, key="sub_e")
with col_t:
    target = st.selectbox("Target", get_targets(), index=0, key="sub_t")

pred_df = load_predictions(window, experiment, target)
if pred_df is None or len(pred_df) == 0:
    st.warning("No predictions data available.")
    st.stop()

test_preds = pred_df[pred_df["set"] == "TEST"].copy()

if len(test_preds) == 0:
    st.warning("No test set predictions found.")
    st.stop()

test_preds["abs_error"] = (test_preds["y_pred"] - test_preds["y_true"]).abs()

st.subheader("Subject-Level Error (Test Set)")

col1, col2 = st.columns(2)

with col1:
    subject_errors = test_preds.groupby("subject")["abs_error"].agg(["mean", "std", "count"]).reset_index()
    subject_errors.columns = ["Subject", "Mean AE", "Std AE", "Splits"]
    subject_errors = subject_errors.sort_values("Mean AE", ascending=False).head(20)
    st.markdown("**Top 20 subjects by mean absolute error**")
    st.dataframe(subject_errors, use_container_width=True, hide_index=True)

with col2:
    subject_r2 = (
        test_preds.groupby("subject")
        .apply(lambda g: 1 - ((g["y_true"] - g["y_pred"]) ** 2).sum() / ((g["y_true"] - g["y_true"].mean()) ** 2).sum())
        .reset_index()
    )
    subject_r2.columns = ["Subject", "R²"]
    subject_r2 = subject_r2.sort_values("R²").head(20)
    st.markdown("**Bottom 20 subjects by R² (worst predicted)**")
    st.dataframe(subject_r2, use_container_width=True, hide_index=True)

st.subheader("Per-Subject Error Distribution")
test_preds["error"] = test_preds["y_pred"] - test_preds["y_true"]
box_data = test_preds[["subject", "error"]].copy()
subjects_sorted = (
    box_data.groupby("subject")["error"].mean().sort_values().index.tolist()
)
fig = go.Figure()
for subj in subjects_sorted[:30]:
    sdata = box_data[box_data["subject"] == subj]["error"].values
    fig.add_trace(go.Box(y=sdata, name=subj.split("-")[-1][:8], boxmean=True, marker_size=2))
fig.update_layout(
    title="Error Distribution by Subject (first 30)",
    xaxis_title="Subject",
    yaxis_title="Error (y_pred - y_true)",
    template="plotly_white",
    height=500,
    showlegend=False,
)
st.plotly_chart(fig, use_container_width=True)

report = load_best_report(window, experiment, target)
if report:
    ts = report.get("test_summary", {})
    st.info(
        f"**{target}** @ W{window} {experiment} — "
        f"MAE test: {ts.get('mae_mean_test', 0):.3f} [{ts.get('mae_ci_lower_test', 0):.3f}, {ts.get('mae_ci_upper_test', 0):.3f}] — "
        f"R²: {ts.get('r2_mean_test', 0):.4f} — "
        f"Subjects: {test_preds['subject'].nunique()}"
    )
