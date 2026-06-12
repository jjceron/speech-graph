import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.loader import list_completed, get_targets, load_predictions, load_best_report
from utils.plots import compute_subject_metrics, plot_subject_mae
from utils.sidebar import render_sidebar

st.set_page_config(page_title="Subjects", page_icon="👤", layout="wide")
render_sidebar()
st.title("👤 Per-Subject Analysis")

completed = list_completed()
if not completed:
    st.warning("No completed experiments.")
    st.stop()

col_w, col_e, col_t, col_s = st.columns(4)
with col_w:
    windows = sorted(set(w for w, _ in completed))
    window = st.selectbox("Window", windows, index=0, key="sub_w")
with col_e:
    exps = [e for w, e in completed if w == window]
    experiment = st.selectbox("Experiment", exps, index=0, key="sub_e")
with col_t:
    target = st.selectbox("Target", get_targets(), index=0, key="sub_t")
with col_s:
    subset = st.selectbox("Set", ["TEST", "VALIDATION"], index=0, key="sub_set")

pred_df = load_predictions(window, experiment, target)
if pred_df is None or len(pred_df) == 0:
    st.warning("No predictions data available.")
    st.stop()

subject_df = compute_subject_metrics(pred_df, set_name=subset)

if len(subject_df) == 0:
    st.warning(f"No {subset} set predictions found.")
    st.stop()

st.subheader(f"Subjects ordered by MAE — {subset}")
st.caption("Lower MAE = better prediction")

top_n = st.slider("Show top N subjects", min_value=5, max_value=len(subject_df),
                  value=min(30, len(subject_df)), key="sub_topn")

display_df = subject_df.head(top_n)[
    ["subject", "subject_short", "mae", "mae_std", "n_predictions",
     "y_true_mean", "y_pred_mean", "bias_mean"]
].copy()
display_df.columns = [
    "Subject", "Short", "MAE", "MAE Std", "N Splits",
    "y_true Mean", "y_pred Mean", "Bias Mean",
]
display_df["MAE"] = display_df["MAE"].round(4)
display_df["MAE Std"] = display_df["MAE Std"].round(4)
display_df["y_true Mean"] = display_df["y_true Mean"].round(3)
display_df["y_pred Mean"] = display_df["y_pred Mean"].round(3)
display_df["Bias Mean"] = display_df["Bias Mean"].round(4)

col_tab, col_chart = st.columns([1, 2])
with col_tab:
    st.dataframe(display_df, use_container_width=True, hide_index=True)
with col_chart:
    fig_mae = plot_subject_mae(subject_df.head(top_n),
                                title=f"MAE by subject — {subset}")
    st.plotly_chart(fig_mae, use_container_width=True)

with st.expander("Bottom subjects by R² (worst predicted)"):
    subject_r2 = (
        pred_df[pred_df["set"] == subset]
        .groupby("subject")
        .apply(lambda g: 1 - ((g["y_true"] - g["y_pred"]) ** 2).sum()
               / ((g["y_true"] - g["y_true"].mean()) ** 2).sum())
        .reset_index()
    )
    subject_r2.columns = ["Subject", "R²"]
    subject_r2 = subject_r2.sort_values("R²").head(20)
    st.dataframe(subject_r2, use_container_width=True, hide_index=True)

st.subheader("Target Variable Distribution by Subject")

subj_df = pred_df[pred_df["set"] == subset].copy()
subj_true = subj_df.groupby("subject")["y_true"].mean()

fig_hist = go.Figure()
fig_hist.add_trace(go.Histogram(
    x=subj_true.values,
    nbinsx=40,
    marker_color="#1f77b4",
    opacity=0.75,
    name="Subjects",
    showlegend=False,
))
fig_hist.add_trace(go.Scatter(
    x=subj_true.values,
    y=[0] * len(subj_true),
    mode="markers",
    marker=dict(symbol="line-ns-open", size=10, color="gray", opacity=0.5),
    name="Each subject",
    showlegend=False,
    hovertemplate="%{x:.2f}<extra></extra>",
))

mean_val = float(subj_true.mean())
fig_hist.add_vline(
    x=mean_val, line_dash="dash", line_color="red", line_width=2,
    annotation_text=f"Mean = {mean_val:.2f}",
    annotation_position="top right",
)

subj_list = sorted(subj_true.index)
highlight_subj = st.selectbox(
    "Highlight subject on histogram",
    [f"{s} (μ={subj_true[s]:.2f})" for s in subj_list],
    key="subj_highlight",
)
if highlight_subj:
    subj_id = highlight_subj.split(" (")[0]
    subj_val = float(subj_true[subj_id])
    fig_hist.add_vline(
        x=subj_val, line_dash="dash", line_color="green", line_width=2,
        annotation_text=subj_id.split("-")[-1][:8],
        annotation_position="top left",
    )

fig_hist.update_layout(
    title=f"{target} distribution across subjects ({subset})",
    xaxis_title=target,
    yaxis_title="Number of subjects",
    template="plotly_white",
    height=400,
    bargap=0.05,
)
st.plotly_chart(fig_hist, use_container_width=True)

st.subheader("Per-Subject Error Distribution")
subj_df["error"] = subj_df["y_pred"] - subj_df["y_true"]
box_data = subj_df[["subject", "error"]].copy()
subjects_sorted = (
    box_data.groupby("subject")["error"].mean().sort_values().index.tolist()
)
fig_box = go.Figure()
for subj in subjects_sorted[:30]:
    sdata = box_data[box_data["subject"] == subj]["error"].values
    fig_box.add_trace(go.Box(y=sdata, name=subj.split("-")[-1][:8], boxmean=True, marker_size=2))
fig_box.update_layout(
    title=f"Error Distribution by Subject (first 30) — {subset}",
    xaxis_title="Subject",
    yaxis_title="Error (y_pred - y_true)",
    template="plotly_white",
    height=500,
    showlegend=False,
)
st.plotly_chart(fig_box, use_container_width=True)

report = load_best_report(window, experiment, target)
if report:
    ts = report.get("test_summary", {})
    st.info(
        f"**{target}** @ W{window} {experiment} — "
        f"MAE test: {ts.get('mae_mean_test', 0):.3f} [{ts.get('mae_ci_lower_test', 0):.3f}, {ts.get('mae_ci_upper_test', 0):.3f}] — "
        f"R²: {ts.get('r2_mean_test', 0):.4f} — "
        f"Subjects ({subset}): {subj_df['subject'].nunique()}"
    )
