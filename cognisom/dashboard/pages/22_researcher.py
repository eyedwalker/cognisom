"""Researcher Workflow — Scenario → Run → Evaluate → Compare → Publish.

Four tabs:
    1. Scenario Builder — create/clone simulation scenarios
    2. Run Manager — execute and track simulation runs
    3. Results Browser — view and compare past runs
    4. Projects — group runs for analysis and publication
"""

import sys
import time
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Researcher | Cognisom", page_icon="🔬", layout="wide")

from cognisom.auth.middleware import streamlit_page_gate
user = streamlit_page_gate("22_researcher")

from cognisom.library.store import EntityStore
from cognisom.library.models import (
    EntityType, SimulationScenario, ParameterSet,
    SimulationRun, ResearchProject,
)
from cognisom.workflow.artifact_store import ArtifactStore
from cognisom.workflow.run_manager import RunManager
from cognisom.workflow.project_manager import ProjectManager
from cognisom.dashboard.engine_runner import SCENARIOS


# ── Shared state ──────────────────────────────────────────────────────

@st.cache_resource
def get_store():
    return EntityStore()

@st.cache_resource
def get_artifact_store():
    return ArtifactStore()

@st.cache_resource
def get_run_manager():
    return RunManager(get_store(), get_artifact_store())

@st.cache_resource
def get_project_manager():
    return ProjectManager(get_store())


store = get_store()
artifacts = get_artifact_store()
run_mgr = get_run_manager()
proj_mgr = get_project_manager()


# ── Page header ───────────────────────────────────────────────────────

st.title("Researcher Workflow")
st.markdown(
    "Build simulation scenarios, execute runs with persistent results, "
    "compare across conditions, and organize into research projects."
)

tab_scenario, tab_runs, tab_results, tab_projects = st.tabs([
    "Scenario Builder", "Run Manager", "Results Browser", "Projects"
])


# ═══════════════════════════════════════════════════════════════════════
# TAB 1: Scenario Builder
# ═══════════════════════════════════════════════════════════════════════

with tab_scenario:
    st.subheader("Create or Clone a Simulation Scenario")

    col_new, col_existing = st.columns(2)

    with col_new:
        st.markdown("**Create from Preset**")
        preset = st.selectbox("Preset", list(SCENARIOS.keys()), key="scenario_preset")
        st.caption(SCENARIOS[preset]["desc"])

        scenario_name = st.text_input("Scenario Name", value=f"{preset} Scenario", key="new_scenario_name")
        duration = st.slider("Duration (hours)", 1.0, 48.0, 6.0, step=0.5, key="new_duration")
        dt = st.select_slider("Time step (hours)", [0.01, 0.02, 0.05, 0.1], value=0.05, key="new_dt")

        st.markdown("**Modules**")
        module_names = ["cellular", "immune", "vascular", "lymphatic", "molecular",
                        "spatial", "epigenetic", "circadian", "morphogen"]
        modules_on = {}
        cols = st.columns(3)
        for i, mod in enumerate(module_names):
            with cols[i % 3]:
                modules_on[mod] = st.checkbox(mod.capitalize(), value=True, key=f"mod_{mod}")

        if st.button("Create Scenario", type="primary", key="create_scenario_btn"):
            import uuid
            scenario = SimulationScenario(
                entity_id=str(uuid.uuid4()),
                name=scenario_name,
                entity_type=EntityType.SIMULATION_SCENARIO,
                scenario_type=preset.lower().replace(" ", "_"),
                duration_hours=duration,
                time_step_hours=dt,
                initial_cell_counts=SCENARIOS[preset].get("cellular", {}),
                created_by=user.username if user else "researcher",
            )
            store.add_entity(scenario)
            st.success(f"Created scenario: **{scenario_name}** ({scenario.entity_id[:8]}...)")
            st.session_state["last_scenario_id"] = scenario.entity_id

    with col_existing:
        st.markdown("**Existing Scenarios**")
        scenarios_list, _ = store.search(
            query="", entity_type=EntityType.SIMULATION_SCENARIO.value, limit=20,
        )
        if scenarios_list:
            for s in scenarios_list:
                with st.expander(f"{s.name} ({s.entity_id[:8]}...)"):
                    st.write(f"**Type:** {s.scenario_type}")
                    st.write(f"**Duration:** {s.duration_hours}h, dt={s.time_step_hours}h")
                    if s.initial_cell_counts:
                        st.write(f"**Initial cells:** {s.initial_cell_counts}")
                    if st.button("Use for New Run", key=f"use_{s.entity_id}"):
                        st.session_state["last_scenario_id"] = s.entity_id
                        st.rerun()
        else:
            st.info("No scenarios yet. Create one above.")


# ═══════════════════════════════════════════════════════════════════════
# TAB 2: Run Manager
# ═══════════════════════════════════════════════════════════════════════

with tab_runs:
    st.subheader("Execute Simulation Runs")

    # Select scenario
    scenarios_list, _ = store.search(
        query="", entity_type=EntityType.SIMULATION_SCENARIO.value, limit=50,
    )
    scenario_options = {f"{s.name} ({s.entity_id[:8]})": s.entity_id for s in scenarios_list}

    if not scenario_options:
        st.warning("Create a scenario first in the Scenario Builder tab.")
    else:
        selected_scenario_label = st.selectbox(
            "Scenario", list(scenario_options.keys()), key="run_scenario_select"
        )
        selected_scenario_id = scenario_options[selected_scenario_label]

        col_run_cfg, col_run_btn = st.columns([3, 1])
        with col_run_cfg:
            run_name = st.text_input("Run Name", value=f"Run {time.strftime('%H:%M')}", key="run_name_input")
            seed = st.number_input("Random Seed", value=42, min_value=0, max_value=2**31-1, key="run_seed")
        with col_run_btn:
            st.write("")
            st.write("")
            start_run = st.button("Start Run", type="primary", key="start_run_btn")

        if start_run:
            with st.spinner("Creating run..."):
                run = run_mgr.create_run(
                    scenario_id=selected_scenario_id,
                    random_seed=seed,
                    run_name=run_name,
                )
            st.info(f"Run **{run.name}** created. Executing...")

            progress_bar = st.progress(0)
            status_text = st.empty()

            def on_progress(current, total):
                progress_bar.progress(current / total)
                status_text.text(f"Step {current}/{total}")

            run = run_mgr.execute_run(
                run.entity_id,
                modules_enabled=modules_on if "modules_on" in dir() else None,
                progress_callback=on_progress,
            )

            progress_bar.progress(1.0)
            if run.run_status == "completed":
                st.success(
                    f"Run completed in {run.elapsed_seconds:.1f}s — "
                    f"Accuracy: **{run.accuracy_grade or 'N/A'}**, "
                    f"Fidelity: **{run.fidelity_grade or 'N/A'}**"
                )
            else:
                st.error(f"Run failed: {run.final_metrics.get('error', 'Unknown error')}")

    # Run history
    st.markdown("---")
    st.subheader("Run History")

    all_runs = run_mgr.list_runs(limit=20)
    if all_runs:
        for run in all_runs:
            status_icon = {
                "completed": "✅", "failed": "❌", "running": "⏳", "pending": "⏸️"
            }.get(run.run_status, "❓")

            with st.expander(
                f"{status_icon} {run.name} — {run.run_status} "
                f"({run.elapsed_seconds:.0f}s, Acc={run.accuracy_grade or '—'}, "
                f"Fid={run.fidelity_grade or '—'})"
            ):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**ID:** {run.entity_id[:12]}...")
                    st.write(f"**Seed:** {run.random_seed}")
                with col2:
                    st.write(f"**dt:** {run.dt}h")
                    st.write(f"**Duration:** {run.duration_hours}h")
                with col3:
                    if run.final_metrics:
                        st.write(f"**Final cancer:** {run.final_metrics.get('final_cancer', '—')}")
                        st.write(f"**Total kills:** {run.final_metrics.get('total_kills', '—')}")

                if st.button("View Results", key=f"view_{run.entity_id}"):
                    st.session_state["view_run_id"] = run.entity_id
                    # Will be shown in Results Browser tab

                if st.button("Delete", key=f"del_{run.entity_id}"):
                    run_mgr.delete_run(run.entity_id)
                    st.rerun()
    else:
        st.info("No runs yet. Execute one above.")


# ═══════════════════════════════════════════════════════════════════════
# TAB 3: Results Browser
# ═══════════════════════════════════════════════════════════════════════

with tab_results:
    st.subheader("Browse & Compare Results")

    all_runs = run_mgr.list_runs(status="completed", limit=50)

    if not all_runs:
        st.info("No completed runs to display.")
    else:
        # Run selector
        run_options = {f"{r.name} ({r.entity_id[:8]})": r.entity_id for r in all_runs}
        selected_runs = st.multiselect(
            "Select runs to view (multi-select for comparison)",
            list(run_options.keys()),
            default=list(run_options.keys())[:1],
            key="results_run_select",
        )
        selected_ids = [run_options[label] for label in selected_runs]

        if len(selected_ids) == 1:
            # ── Single run view ──────────────────────────────────────
            run_id = selected_ids[0]
            run = run_mgr.get_run(run_id)
            ts = artifacts.load_time_series(run_id)

            if ts and "time" in ts:
                st.markdown(f"### {run.name}")

                # Metrics row
                m = run.final_metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Cancer Cells", m.get("final_cancer", "—"))
                col2.metric("Immune Kills", m.get("total_kills", "—"))
                col3.metric("Metastases", m.get("total_metastases", "—"))
                col4.metric("Accuracy", run.accuracy_grade or "—")
                col5.metric("Fidelity", run.fidelity_grade or "—")

                # Time series charts
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=[
                        "Cell Populations", "Immune Dynamics",
                        "O₂ & Glucose", "Cumulative Events"
                    ],
                )

                t = ts["time"]
                fig.add_trace(go.Scatter(x=t, y=ts.get("n_cancer", []), name="Cancer", line=dict(color="#D55E00")), row=1, col=1)
                fig.add_trace(go.Scatter(x=t, y=ts.get("n_normal", []), name="Normal", line=dict(color="#0072B2")), row=1, col=1)

                fig.add_trace(go.Scatter(x=t, y=ts.get("n_t_cells", []), name="T cells", line=dict(color="#0072B2")), row=1, col=2)
                fig.add_trace(go.Scatter(x=t, y=ts.get("n_nk_cells", []), name="NK cells", line=dict(color="#009E73")), row=1, col=2)
                fig.add_trace(go.Scatter(x=t, y=ts.get("total_kills", []), name="Kills", line=dict(color="#D55E00", dash="dot")), row=1, col=2)

                fig.add_trace(go.Scatter(x=t, y=ts.get("avg_cell_O2", []), name="O₂", line=dict(color="#56B4E9")), row=2, col=1)
                fig.add_trace(go.Scatter(x=t, y=ts.get("avg_cell_glucose", []), name="Glucose", line=dict(color="#E69F00")), row=2, col=1)

                fig.add_trace(go.Scatter(x=t, y=ts.get("total_divisions", []), name="Divisions", line=dict(color="#0072B2")), row=2, col=2)
                fig.add_trace(go.Scatter(x=t, y=ts.get("total_deaths", []), name="Deaths", line=dict(color="#D55E00")), row=2, col=2)
                fig.add_trace(go.Scatter(x=t, y=ts.get("total_metastases", []), name="Metastases", line=dict(color="#CC79A7")), row=2, col=2)

                fig.update_layout(height=600, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)

                # Event log
                events = artifacts.load_events(run_id)
                if events:
                    with st.expander(f"Event Log ({len(events)} events)"):
                        for ev in events[-20:]:
                            st.text(f"  t={ev.get('time', '?'):.2f}  {ev.get('event', '')}  {ev.get('details', '')}")

                # Export
                col_exp1, col_exp2 = st.columns(2)
                with col_exp1:
                    import csv as csv_mod
                    import io
                    buf = io.StringIO()
                    writer = csv_mod.writer(buf)
                    cols = list(ts.keys())
                    writer.writerow(cols)
                    n_rows = len(ts[cols[0]])
                    for i in range(n_rows):
                        writer.writerow([ts[c][i] for c in cols])
                    st.download_button("Download CSV", buf.getvalue(), f"{run.name}.csv", "text/csv")

                with col_exp2:
                    import json
                    st.download_button(
                        "Download JSON",
                        json.dumps({"metrics": run.final_metrics, "events": run.event_summary}, indent=2),
                        f"{run.name}.json",
                        "application/json",
                    )
            else:
                st.warning("No time-series data found for this run.")

        elif len(selected_ids) >= 2:
            # ── Multi-run comparison ─────────────────────────────────
            comparison = run_mgr.compare_runs(selected_ids)

            if comparison["runs"]:
                st.markdown("### Run Comparison")

                # Metrics table
                import pandas as pd
                metrics_df = pd.DataFrame(comparison["metrics_table"])
                metrics_df.index = [r["name"] for r in comparison["runs"]]
                st.dataframe(metrics_df.T, use_container_width=True)

                # Overlay charts
                metric_to_plot = st.selectbox(
                    "Metric to overlay",
                    ["n_cancer", "n_normal", "total_kills", "avg_cell_O2", "total_metastases"],
                    key="comparison_metric",
                )

                fig = go.Figure()
                colors = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]
                for i, run_id in enumerate(selected_ids):
                    ts = artifacts.load_time_series(run_id)
                    if ts and "time" in ts:
                        run = run_mgr.get_run(run_id)
                        fig.add_trace(go.Scatter(
                            x=ts["time"],
                            y=ts.get(metric_to_plot, []),
                            name=run.name if run else run_id[:8],
                            line=dict(color=colors[i % len(colors)]),
                        ))

                fig.update_layout(
                    title=f"{metric_to_plot} — Comparison",
                    xaxis_title="Time (hours)",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# TAB 4: Projects
# ═══════════════════════════════════════════════════════════════════════

with tab_projects:
    st.subheader("Research Projects")

    col_create, col_list = st.columns([1, 2])

    with col_create:
        st.markdown("**Create New Project**")
        proj_title = st.text_input("Title", key="proj_title")
        proj_hypothesis = st.text_area("Hypothesis", key="proj_hypothesis", height=80)
        proj_methodology = st.text_area("Methodology", key="proj_methodology", height=80)

        if st.button("Create Project", type="primary", key="create_proj_btn"):
            if proj_title:
                project = proj_mgr.create_project(
                    title=proj_title,
                    hypothesis=proj_hypothesis,
                    methodology=proj_methodology,
                )
                st.success(f"Created project: **{proj_title}**")
            else:
                st.error("Project title is required.")

    with col_list:
        st.markdown("**Existing Projects**")
        projects = proj_mgr.list_projects()

        if projects:
            for proj in projects:
                summary = proj_mgr.get_project_summary(proj.entity_id)
                with st.expander(
                    f"{proj.title} — {summary.get('n_completed', 0)}/{summary.get('n_runs', 0)} runs"
                ):
                    st.write(f"**Hypothesis:** {proj.hypothesis or '(not set)'}")
                    st.write(f"**Status:** {proj.paper_status}")

                    # Add runs to project
                    all_runs = run_mgr.list_runs(status="completed", limit=50)
                    available = [r for r in all_runs if r.entity_id not in proj.run_ids]
                    if available:
                        run_to_add = st.selectbox(
                            "Add run to project",
                            ["(select)"] + [f"{r.name} ({r.entity_id[:8]})" for r in available],
                            key=f"add_run_{proj.entity_id}",
                        )
                        if run_to_add != "(select)" and st.button("Add", key=f"add_btn_{proj.entity_id}"):
                            # Parse run_id from selection
                            for r in available:
                                if f"{r.name} ({r.entity_id[:8]})" == run_to_add:
                                    proj_mgr.add_run(proj.entity_id, r.entity_id)
                                    st.rerun()

                    # Show linked runs
                    if summary.get("runs"):
                        st.markdown("**Linked Runs:**")
                        for r in summary["runs"]:
                            baseline_marker = " ⭐ baseline" if r["run_id"] == proj.baseline_run_id else ""
                            st.write(
                                f"- {r['name']} ({r['run_status']}, "
                                f"Acc={r.get('accuracy_grade') or '—'}, "
                                f"Fid={r.get('fidelity_grade') or '—'})"
                                f"{baseline_marker}"
                            )

                        # Set baseline
                        baseline_options = [r["run_id"] for r in summary["runs"]]
                        baseline_labels = [r["name"] for r in summary["runs"]]
                        if baseline_labels:
                            bl = st.selectbox(
                                "Set baseline run",
                                baseline_labels,
                                key=f"baseline_{proj.entity_id}",
                            )
                            idx = baseline_labels.index(bl)
                            if st.button("Set Baseline", key=f"bl_btn_{proj.entity_id}"):
                                proj_mgr.set_baseline(proj.entity_id, baseline_options[idx])
                                st.rerun()

                    # Parameter sweep
                    with st.expander("Parameter Sweep"):
                        scenarios_list, _ = store.search(
                            query="", entity_type=EntityType.SIMULATION_SCENARIO.value, limit=20
                        )
                        if scenarios_list:
                            sweep_scenario = st.selectbox(
                                "Scenario",
                                [f"{s.name} ({s.entity_id[:8]})" for s in scenarios_list],
                                key=f"sweep_scenario_{proj.entity_id}",
                            )
                            sweep_module = st.selectbox(
                                "Module", ["cellular", "immune", "vascular", "lymphatic"],
                                key=f"sweep_module_{proj.entity_id}",
                            )
                            sweep_param = st.text_input(
                                "Parameter name", value="division_time_cancer",
                                key=f"sweep_param_{proj.entity_id}",
                            )
                            sweep_values = st.text_input(
                                "Values (comma-separated)", value="8, 12, 16, 24",
                                key=f"sweep_values_{proj.entity_id}",
                            )

                            if st.button("Create Sweep Runs", key=f"sweep_btn_{proj.entity_id}"):
                                try:
                                    values = [float(v.strip()) for v in sweep_values.split(",")]
                                    # Find scenario ID
                                    for s in scenarios_list:
                                        if f"{s.name} ({s.entity_id[:8]})" == sweep_scenario:
                                            runs = proj_mgr.create_sweep_runs(
                                                proj.entity_id, s.entity_id,
                                                sweep_module, sweep_param, values, run_mgr,
                                            )
                                            st.success(f"Created {len(runs)} sweep runs.")
                                            break
                                except ValueError:
                                    st.error("Invalid values. Use comma-separated numbers.")

                    if st.button("Delete Project", key=f"del_proj_{proj.entity_id}"):
                        proj_mgr.delete_project(proj.entity_id)
                        st.rerun()
        else:
            st.info("No projects yet.")


# ── Footer ────────────────────────────────────────────────────────────

from cognisom.dashboard.footer import render_footer
render_footer()
