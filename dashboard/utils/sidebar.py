from __future__ import annotations

import streamlit as st
from utils.loader import (
    list_completed,
    list_tasks,
    get_task,
    set_task,
    get_windows,
    get_experiments,
)


def render_sidebar() -> None:
    st.markdown(
        """
<style>
[data-testid="stSidebarNav"] { display: none; }
</style>
""",
        unsafe_allow_html=True,
    )
    st.sidebar.title("📊 SpeechGraph")

    tasks = list_tasks()
    current_task = get_task()
    avail_tasks = [t for t in tasks] if tasks else [current_task]
    selected_task = st.sidebar.selectbox(
        "Task",
        avail_tasks,
        index=avail_tasks.index(current_task) if current_task in avail_tasks else 0,
    )
    if selected_task != current_task:
        set_task(selected_task)
        st.cache_data.clear()
        st.rerun()

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
