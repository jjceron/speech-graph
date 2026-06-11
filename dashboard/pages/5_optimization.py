import streamlit as st
import numpy as np
import plotly.graph_objects as go
from utils.loader import list_completed, get_targets, load_optuna_trials, load_best_report
from utils.plots import optimization_history, model_selection_bar, optuna_parallel_coords
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
    windows = sorted(set(w for w, _ in completed))
    window = st.selectbox("Window", windows, index=0, key="opt_w")
with col_e:
    exps = [e for w, e in completed if w == window]
    experiment = st.selectbox("Experiment", exps, index=0, key="opt_e")
with col_t:
    target = st.selectbox("Target", get_targets(), index=0, key="opt_t")

trials = load_optuna_trials(window, experiment, target)
if trials is None:
    st.warning("No Optuna trial data available.")
    st.stop()

report = load_best_report(window, experiment, target)
if report:
    st.success(
        f"**Best trial:** #{report['best_trial_number']} — "
        f"**Model:** {report['best_params'].get('regressor', '?')} — "
        f"**Objective (MAE val):** {report['best_value_internal_minimized']:.4f}"
    )

tab1, tab2, tab3 = st.tabs(["History", "Model Selection", "Parallel Coordinates"])

with tab1:
    fig_hist = optimization_history(trials)
    st.plotly_chart(fig_hist, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if "params_rfe_n_features" in trials.columns:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=trials["params_rfe_n_features"],
                    y=trials["value"],
                    mode="markers",
                    marker=dict(size=5, opacity=0.5, color="steelblue"),
                )
            )
            fig.update_layout(
                title="RFE N Features vs Objective",
                xaxis_title="RFE N Features",
                yaxis_title="Objective (MAE val)",
                template="plotly_white",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "params_quantile_alpha" in trials.columns:
            alpha_col = trials["params_quantile_alpha"].dropna()
            if len(alpha_col) > 0:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=alpha_col,
                        y=trials.loc[alpha_col.index, "value"],
                        mode="markers",
                        marker=dict(size=5, opacity=0.5, color="steelblue"),
                    )
                )
                fig.update_layout(
                    title="Quantile Alpha vs Objective",
                    xaxis_title="Alpha",
                    yaxis_title="Objective (MAE val)",
                    template="plotly_white",
                    height=350,
                    xaxis_type="log",
                )
                st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig_models = model_selection_bar(trials)
    st.plotly_chart(fig_models, use_container_width=True)

    if "params_rfe_n_features" in trials.columns:
        st.subheader("RFE N Features Distribution")
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=trials["params_rfe_n_features"].dropna(),
                nbinsx=22,
                marker_color="#2ca02c",
                opacity=0.75,
            )
        )
        fig.update_layout(
            title="Distribution of RFE N Features (300 trials)",
            xaxis_title="N Features",
            yaxis_title="Trials",
            template="plotly_white",
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    param_options = {
        "params_regressor": "Regressor",
        "params_rfe_n_features": "RFE N",
        "params_quantile_alpha": "Alpha",
        "params_elastic_alpha": "Elastic α",
        "params_ridge_alpha": "Ridge α",
        "params_knn_n_neighbors": "KNN k",
        "params_rf_n_estimators": "RF n_est",
        "params_et_max_depth": "ET depth",
        "params_dt_max_depth": "DT depth",
        "params_use_scaler": "Scaler",
    }
    available = [k for k in param_options if k in trials.columns and trials[k].nunique() > 1]
    col_pc1, col_pc2 = st.columns([3, 1])
    with col_pc1:
        selected_params = st.multiselect(
            "Select hyperparameters for parallel coordinates",
            options=available,
            default=[k for k in ["params_regressor", "params_rfe_n_features", "params_use_scaler"] if k in available],
            format_func=lambda x: param_options.get(x, x),
        )
    with col_pc2:
        n_trials = len(trials.dropna(subset=["value"]))
        top_k = st.slider(
            "Top K trials",
            min_value=min(30, n_trials),
            max_value=n_trials,
            value=min(100, n_trials),
            step=10,
        )
    if selected_params:
        fig_pc = optuna_parallel_coords(trials, selected_params, top_k=top_k)
        st.plotly_chart(fig_pc, use_container_width=True)
    else:
        st.info("Select at least one parameter to display.")
