import streamlit as st
from utils.loader import list_completed, load_all_reports, load_best_report, list_tasks, get_task, set_task, get_windows, get_experiments, get_targets
from utils.plots import TARGET_COLORS
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="SpeechGraph — Regression Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("📊 SpeechGraph")

tasks = list_tasks()
current_task = get_task()
avail_tasks = [t for t in tasks] if tasks else [current_task]
selected_task = st.sidebar.selectbox(
    "Task", avail_tasks, index=avail_tasks.index(current_task) if current_task in avail_tasks else 0
)
if selected_task != current_task:
    set_task(selected_task)
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown(f"### Regression Optuna — Task {get_task()}")
st.sidebar.markdown("---")

completed = list_completed()
tot_possible = len(get_windows()) * len(get_experiments()) if get_windows() else 0
st.sidebar.success(f"**{len(completed)} / {tot_possible}** experiments complete")
st.sidebar.markdown("---")

st.sidebar.page_link("app.py", label="Overview", icon="🏠")
st.sidebar.page_link("pages/2_comparison.py", label="Comparison", icon="📊")
st.sidebar.page_link("pages/3_distributions.py", label="Distributions", icon="📈")
st.sidebar.page_link("pages/4_features.py", label="Features", icon="🔬")
st.sidebar.page_link("pages/5_optimization.py", label="Optimization", icon="⚙️")
st.sidebar.page_link("pages/6_subjects.py", label="Subjects", icon="👤")

st.title("📊 SpeechGraph Regression Dashboard")
st.markdown(f"#### Task {get_task()} — Optuna Regression Results (W10–W40)")

reports = load_all_reports()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Experiments complete", f"{len(completed)}/{len(get_windows()) * len(get_experiments())}")

best_r2 = -999
best_label = ""
for (w, e, t), r in reports.items():
    ts = r.get("test_summary", {})
    r2 = ts.get("r2_mean_test", -999)
    if r2 is not None and r2 > best_r2:
        best_r2 = r2
        best_label = f"W{w} {e} {t}"
col2.metric("Best R²", f"{best_r2:.4f}", delta=best_label)

r2_vals = [r["test_summary"].get("r2_mean_test", -999) for r in reports.values()]
r2_pos = sum(1 for v in r2_vals if v > 0)
col3.metric("R² > 0", f"{r2_pos}/{len(r2_vals)}")
col4.metric("Total scenarios", f"{len(reports)}")

st.markdown("---")

st.subheader("🏆 Top 5 by R² test")

rows = []
for (w, e, t), r in reports.items():
    ts = r.get("test_summary", {})
    bp = r.get("best_params", {})
    feat = r.get("selected_features", [])
    rows.append({
        "R²": ts.get("r2_mean_test", 0),
        "Window": f"W{w}",
        "Experiment": e,
        "Target": t,
        "R² [IC 95%]": f"{ts.get('r2_mean_test', 0):.4f} [{ts.get('r2_ci_lower_test', 0):.4f}, {ts.get('r2_ci_upper_test', 0):.4f}]",
        "MAE [IC 95%]": f"{ts.get('mae_mean_test', 0):.3f} [{ts.get('mae_ci_lower_test', 0):.3f}, {ts.get('mae_ci_upper_test', 0):.3f}]",
        "% R²<0": f"{ts.get('r2_below_zero_test', 0) * 100:.1f}%",
        "Model": bp.get("regressor", "?"),
        "Features": ", ".join(feat) if feat else "-",
    })

if rows:
    df = pd.DataFrame(rows).sort_values("R²", ascending=False).head(5).drop(columns=["R²"])
    st.dataframe(df, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption(f"Data from `outputs/regression_optuna/task{get_task()}/` — Last updated: see git log")
