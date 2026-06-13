import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.loader import list_completed, get_windows, get_experiments, get_targets, load_optuna_trials, load_best_report
from utils.plots import plot_optimization_ecdf, model_selection_bar, plot_objective_by_regressor, plot_parameter_importance, plot_regressor_nfeatures_heatmap
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

with tab2:
    fig_models = model_selection_bar(trials)
    st.plotly_chart(fig_models, use_container_width=True, key="bar_t2")

    df = trials.dropna(subset=["value", "params_regressor"]).copy()
    if len(df) > 0:
        global_best_val = df["value"].min()
        best_reg = df.loc[df["value"].idxmin(), "params_regressor"]

        def best_trial_ci(grp):
            idx = grp["value"].idxmin()
            return pd.Series({
                "ci_lower": grp.loc[idx, "user_attrs_mae_ci_lower_val"],
                "ci_upper": grp.loc[idx, "user_attrs_mae_ci_upper_val"],
            })

        stats = df.groupby("params_regressor").agg(
            Trials=("value", "count"),
            best_val=("value", "min"),
            mean_val=("value", "mean"),
            std_val=("value", "std"),
        ).reset_index()

        ci_df = df.groupby("params_regressor").apply(best_trial_ci).reset_index()
        stats = stats.merge(ci_df, on="params_regressor")

        stats["Regressor"] = stats["params_regressor"]
        stats["Best MAE Val"] = stats["best_val"].map("{:.4f}".format)
        stats["Mean MAE Val"] = stats["mean_val"].map("{:.4f}".format)
        stats["Std MAE Val"] = stats["std_val"].map("{:.4f}".format)
        stats["CI 95% Val"] = stats.apply(
            lambda r: f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]", axis=1
        )

        display_cols = ["Regressor", "Trials", "Best MAE Val", "Mean MAE Val", "Std MAE Val", "CI 95% Val"]
        stats_display = stats[display_cols].sort_values("Best MAE Val")

        def highlight_best(row):
            if row["Regressor"] == best_reg:
                return ["font-weight: bold; color: #222"] * len(row)
            return [""] * len(row)

        st.subheader("Per-Regressor Stats (Validation)")
        st.dataframe(
            stats_display.style.apply(highlight_best, axis=1),
            use_container_width=True,
            hide_index=True,
        )

        if report:
            ts = report.get("test_summary", {})
            if ts:
                st.subheader("Test Results (Best Trial)")
                test_rows = [{
                    "Metric": metric,
                    "Mean": f"{ts.get(f'{key}_mean_test', '?'):.4f}",
                    "CI 95%": f"[{ts.get(f'{key}_ci_lower_test', '?'):.4f}, {ts.get(f'{key}_ci_upper_test', '?'):.4f}]",
                } for metric, key in [("MAE", "mae"), ("RMSE", "rmse"), ("R²", "r2"), ("rho", "rho")]]
                st.dataframe(pd.DataFrame(test_rows), use_container_width=True, hide_index=True)

with tab3:
    param_cols = [c for c in trials.columns if c.startswith("params_") and c not in ("params_regressor", "params_use_scaler")]
    n_imp = sum(1 for c in param_cols if trials[c].dropna().nunique() >= 3 and trials[c].notna().sum() >= 10)

    plot_h = max(250, n_imp * 35)
    n_table_rows = (len(report.get("best_params", {})) + 1 + len(report.get("selected_features", []))) if report else 0
    table_h = n_table_rows * 36 + 80
    height = min(max(plot_h, table_h, 250), 520)

    col_a, col_b = st.columns([2, 3])
    with col_a:
        if report:
            with st.container(height=height):
                st.subheader("Best Trial Parameters")
                rows = [{"Parameter": k, "Value": str(v)} for k, v in report.get("best_params", {}).items()]
                rows.append({"Parameter": "n_features", "Value": str(len(report.get("selected_features", [])))})
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                st.subheader("Selected Features")
                st.dataframe(pd.DataFrame({"Feature": report.get("selected_features", [])}), use_container_width=True, hide_index=True)
    with col_b:
        fig_imp = plot_parameter_importance(trials, height=height)
        st.plotly_chart(fig_imp, use_container_width=True, key="imp_t3")

    heat_metric = st.radio("Heatmap metric", ["MAE (val)", "R² (val)"], horizontal=True, key="heat_metric")
    metric_key = "mae" if heat_metric == "MAE (val)" else "r2"
    fig_heat = plot_regressor_nfeatures_heatmap(trials, metric=metric_key)
    st.plotly_chart(fig_heat, use_container_width=True, key="heat_t3")
