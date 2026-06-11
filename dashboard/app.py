import streamlit as st
from utils.loader import list_completed, ALL_TARGETS, load_all_reports, load_best_report
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="SpeechGraph — Regression Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("📊 SpeechGraph")
st.sidebar.markdown("### Regression Optuna — Task 2")
st.sidebar.markdown("---")

completed = list_completed()
st.sidebar.success(f"**{len(completed)} / 12** experiments complete")
st.sidebar.markdown("---")

st.sidebar.markdown(
    """
**Navigation**
- 🏠 [Overview](#overview)
- 📊 [Cross-Experiment](/Cross_Experiment)
- 📈 [Distributions](/Distributions)
- 🔬 [Features](/Features)
- ⚙️ [Optimization](/Optimization)
- 👤 [Subjects](/Subjects)
"""
)

st.title("📊 SpeechGraph Regression Dashboard")
st.markdown("#### Task 2 — Optuna Regression Results (W10–W40)")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Experiments complete", f"{len(completed)}/12")
reports = load_all_reports()

best_r2 = -999
best_target = ""
best_exp = ""
for (w, e, t), r in reports.items():
    ts = r.get("test_summary", {})
    r2 = ts.get("r2_mean_test", -999)
    if r2 is not None and r2 > best_r2:
        best_r2 = r2
        best_target = t
        best_exp = f"W{w} {e}"

col2.metric("Best R²", f"{best_r2:.4f}", delta=f"{best_target} ({best_exp})" if best_target else "")

target_counts = {t: sum(1 for (_, _, tt) in reports if tt == t) for t in ALL_TARGETS}
col3.metric("MOT experiments", f"{target_counts.get('MOT', 0)}/12")
col4.metric("COG_V1 experiments", f"{target_counts.get('COG_V1', 0)}/12")

st.markdown("---")

st.subheader("📋 Results Summary")

rows = []
for (w, e, t), r in reports.items():
    ts = r.get("test_summary", {})
    vs = r.get("validation_summary", {})
    bp = r.get("best_params", {})
    feat = r.get("selected_features", [])
    reg = bp.get("regressor", "?")
    row = {
        "Window": f"W{w}",
        "Experiment": e,
        "Target": t,
        "R²_test": f"{ts.get('r2_mean_test', 0):.4f}",
        "R²_CI": f"[{ts.get('r2_ci_lower_test', 0):.4f}, {ts.get('r2_ci_upper_test', 0):.4f}]",
        "MAE_test": f"{ts.get('mae_mean_test', 0):.3f}",
        "MAE_CI": f"[{ts.get('mae_ci_lower_test', 0):.3f}, {ts.get('mae_ci_upper_test', 0):.3f}]",
        "% R²<0": f"{ts.get('r2_below_zero_test', 0) * 100:.1f}%",
        "ρ_test": f"{ts.get('rho_mean_test', '—')}",
        "Model": reg,
        "Features": len(feat),
    }
    rows.append(row)

if rows:
    df_summary = pd.DataFrame(rows)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Data from `outputs/regression_optuna/task2/` — Last updated: see git log")
