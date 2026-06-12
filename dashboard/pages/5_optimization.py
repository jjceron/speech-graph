import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.loader import list_completed, get_windows, get_experiments, get_targets, load_optuna_trials, load_best_report
from utils.plots import plot_optimization_ecdf, model_selection_bar, plot_objective_by_regressor, plot_parameter_importance, plot_rfe_vs_objective
from utils.sidebar import render_sidebar

st.set_page_config(page_title="Optimization", page_icon="⚙️", layout="wide")
render_sidebar()
st.title("⚙️ Optuna Optimization Analysis")

completed = list_completed()
if not completed:
    st.warning("No completed experiments.")
    st.stop()

col_w, col_e, col_t = st.columns(3)
with col_w:
    window = st.selectbox("Window", get_windows(), index=0, key="opt_w")
with col_e:
    experiment = st.selectbox("Experiment", [e for e in get_experiments() if get_targets(window=window, experiment=e)], index=0, key="opt_e")
with col_t:
    target = st.selectbox("Target", get_targets(), index=0, key="opt_t")

trials = load_optuna_trials(window, experiment, target)
if trials is None:
    st.warning("No Optuna trial data available.")
    st.stop()

report = load_best_report(window, experiment, target)
if report:
    bp = report.get("best_params", {})
    st.success(
        f"**Best trial:** #{report['best_trial_number']} — "
        f"**Model:** {bp.get('regressor', '?')} — "
        f"**Objective (MAE val):** {report['best_value_internal_minimized']:.4f}"
    )

tab1, tab2, tab3 = st.tabs(["Optimization Trace", "Model Comparison", "Parameter Analysis"])

with tab1:
    fig_ecdf = plot_optimization_ecdf(trials)
    st.plotly_chart(fig_ecdf, use_container_width=True, key="ecdf_t1")

    fig_box = plot_objective_by_regressor(trials)
    st.plotly_chart(fig_box, use_container_width=True, key="box_t1")

    fig_rfe = plot_rfe_vs_objective(trials)
    st.plotly_chart(fig_rfe, use_container_width=True, key="rfe_t1")

with tab2:
    col_a, col_b = st.columns([3, 2])
    with col_a:
        fig_models = model_selection_bar(trials)
        st.plotly_chart(fig_models, use_container_width=True, key="bar_t2")
    with col_b:
        df = trials.dropna(subset=["value", "params_regressor"]).copy()
        if len(df) > 0:
            stats = df.groupby("params_regressor")["value"].agg(["count", "mean", "std", "min"])
            stats = stats.rename(columns={
                "count": "Trials",
                "mean": "Mean MAE val",
                "std": "Std MAE val",
                "min": "Best MAE val",
            })
            stats["Mean MAE val"] = stats["Mean MAE val"].map("{:.4f}".format)
            stats["Std MAE val"] = stats["Std MAE val"].map("{:.4f}".format)
            stats["Best MAE val"] = stats["Best MAE val"].map("{:.4f}".format)
            stats = stats.sort_values("Best MAE val")
            st.subheader("Per-Regressor Stats")
            st.dataframe(stats, use_container_width=True)

with tab3:
    col_a, col_b = st.columns([3, 2])
    with col_a:
        fig_imp = plot_parameter_importance(trials)
        st.plotly_chart(fig_imp, use_container_width=True)
    with col_b:
        if report:
            params = report.get("best_params", {})
            feat = report.get("selected_features", [])
            st.subheader("Best Trial Parameters")
            rows = []
            for k, v in params.items():
                rows.append({"Parameter": k, "Value": str(v)})
            rows.append({"Parameter": "n_features", "Value": str(len(feat))})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.subheader("Selected Features")
            st.dataframe(pd.DataFrame({"Feature": feat}), use_container_width=True, hide_index=True)
