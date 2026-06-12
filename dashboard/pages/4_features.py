import streamlit as st
import pandas as pd
from utils.loader import list_completed, get_targets, load_rfe_ranking, load_selected_features, load_best_report
from utils.plots import rfe_ranking_chart
from utils.sidebar import render_sidebar

st.set_page_config(page_title="Features", page_icon="🔬", layout="wide")
render_sidebar()
st.title("🔬 Features Analysis")

completed = list_completed()
if not completed:
    st.warning("No completed experiments.")
    st.stop()

col_w, col_e, col_t = st.columns(3)
with col_w:
    windows = sorted(set(w for w, _ in completed))
    window = st.selectbox("Window", windows, index=0, key="feat_w")
with col_e:
    exps = [e for w, e in completed if w == window]
    experiment = st.selectbox("Experiment", exps, index=0, key="feat_e")
with col_t:
    target = st.selectbox("Target", get_targets(), index=0, key="feat_t")

st.subheader(f"W{window} — {experiment} — {target}")

col1, col2 = st.columns(2)

with col1:
    ranking = load_rfe_ranking(window, experiment, target)
    if ranking is not None:
        fig = rfe_ranking_chart(ranking)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No RFE ranking data.")

with col2:
    report = load_best_report(window, experiment, target)
    if report:
        feat = report.get("selected_features", [])
        bp = report.get("best_params", {})
        st.subheader("Selected Features")
        st.write(f"**{len(feat)}** features selected by RFE")
        st.write(f"Model: **{bp.get('regressor', '?')}**")
        st.dataframe(pd.DataFrame({"Feature": feat}), use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader(f"Feature Comparison Across Experiments — Target: {target}")

all_rows = []
for w, e in completed:
    report = load_best_report(w, e, target)
    if report:
        feat = report.get("selected_features", [])
        bp = report.get("best_params", {})
        vs = report.get("validation_summary", {})
        ts = report.get("test_summary", {})
        all_rows.append({
            "Window": f"W{w}",
            "Experiment": e,
            "Target": target,
            "Model": bp.get("regressor", "?"),
            "MAE val": vs.get("mae_mean_val"),
            "MAE test": ts.get("mae_mean_test"),
            "N Features": len(feat),
            "Features": ", ".join(feat),
        })

df_all = pd.DataFrame(all_rows)
df_all = df_all.sort_values("MAE val")
st.dataframe(df_all, use_container_width=True, hide_index=True)
