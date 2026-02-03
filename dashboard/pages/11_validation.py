"""
Page 11: Validation & Benchmarking
===================================

Dashboard for running simulations against published experimental
benchmarks, calibrating parameters, and profiling performance.
"""

import streamlit as st
import json
import time

import numpy as np

st.set_page_config(page_title="Validation", page_icon="V", layout="wide")

from cognisom.auth.middleware import streamlit_page_gate
user = streamlit_page_gate("11_validation")

st.title("Validation & Benchmarking")
st.caption("Compare simulation outputs against published experimental data")

# ── Tabs ─────────────────────────────────────────────────────────

tab_bench, tab_calib, tab_profile, tab_synth = st.tabs([
    "Benchmarks", "Parameter Calibration", "Performance Profiler", "Synthetic Data",
])


# ────────────────────────────────────────────────────────────────
# TAB 1: Benchmarks
# ────────────────────────────────────────────────────────────────

with tab_bench:
    st.subheader("Published Experimental Benchmarks")

    try:
        from cognisom.validation.benchmarks import get_all_benchmarks, BenchmarkDataset
        from cognisom.validation.validator import ValidationRunner
    except ImportError:
        st.error("Validation module not found. Ensure cognisom.validation is installed.")
        st.stop()

    all_benchmarks = get_all_benchmarks()

    # Summary metrics
    total = sum(len(bms) for bms in all_benchmarks.values())
    cols = st.columns(len(all_benchmarks) + 1)
    cols[0].metric("Total Benchmarks", total)
    for i, (cat, bms) in enumerate(all_benchmarks.items()):
        cols[i + 1].metric(cat.replace("_", " ").title(), len(bms))

    st.divider()

    # Category filter
    categories = st.multiselect(
        "Filter by category",
        options=list(all_benchmarks.keys()),
        default=list(all_benchmarks.keys()),
        format_func=lambda x: x.replace("_", " ").title(),
    )

    # Display benchmarks
    for cat in categories:
        st.markdown(f"### {cat.replace('_', ' ').title()}")
        for bm in all_benchmarks[cat]:
            with st.expander(f"{bm.name} — {bm.source}"):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"**Description:** {bm.description}")
                    st.markdown(f"**DOI:** `{bm.doi}`")
                    st.markdown(
                        f"**Tolerance:** {bm.tolerance_pct}% | "
                        f"**Module:** `{bm.module}` | "
                        f"**Metric:** `{bm.metric}`"
                    )
                with col2:
                    # Mini chart of expected data
                    import plotly.graph_objects as go

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=bm.x_values, y=bm.y_values,
                        mode="lines+markers",
                        name="Expected",
                        line=dict(color="#2196F3"),
                    ))
                    fig.update_layout(
                        height=200, margin=dict(l=20, r=20, t=20, b=20),
                        xaxis_title=f"{bm.x_label} ({bm.x_units})",
                        yaxis_title=f"{bm.y_label} ({bm.y_units})",
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Run validation
    st.subheader("Run Validation Suite")
    run_cats = st.multiselect(
        "Categories to validate",
        options=list(all_benchmarks.keys()),
        default=list(all_benchmarks.keys()),
        format_func=lambda x: x.replace("_", " ").title(),
        key="run_cats",
    )

    btn_col1, btn_col2 = st.columns(2)
    run_selected = btn_col1.button("Run Selected", type="primary")
    run_all = btn_col2.button("Run All 9 Benchmarks")

    if run_selected or run_all:
        if run_all:
            run_cats = list(all_benchmarks.keys())
        runner = ValidationRunner()
        progress = st.progress(0, text="Starting...")

        def update_progress(step, total, msg):
            progress.progress(step / max(total, 1), text=msg)

        results = runner.run_all(categories=run_cats, progress_cb=update_progress)
        report = runner.generate_report(results)

        progress.progress(1.0, text="Complete")

        # Display results
        st.session_state["validation_results"] = results
        st.session_state["validation_report"] = report

    # Show results if available
    if "validation_report" in st.session_state:
        report = st.session_state["validation_report"]
        results = st.session_state["validation_results"]

        st.markdown("### Results")
        r_cols = st.columns(4)
        status_color = {"pass": "green", "partial": "orange", "fail": "red"}.get(
            report["status"], "gray"
        )
        r_cols[0].metric("Status", report["status"].upper())
        r_cols[1].metric("Passed", f"{report['passed']}/{report['benchmarks_run']}")
        r_cols[2].metric("Pass Rate", f"{report['pass_rate_pct']}%")
        r_cols[3].metric("Avg Score", f"{report['avg_score']}/100")

        for r in results:
            icon = "PASS" if r.passed else "FAIL"
            color = "green" if r.passed else "red"
            st.markdown(
                f":{color}[**{icon}**] **{r.benchmark_name}** — "
                f"Score: {r.score:.0f}/100, "
                f"Mean Error: {r.mean_error_pct:.1f}%, "
                f"Time: {r.elapsed_sec:.2f}s"
            )

            if r.sim_values and r.exp_values:
                import plotly.graph_objects as go

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=r.x_values, y=r.exp_values,
                    mode="lines+markers", name="Expected",
                    line=dict(color="#2196F3"),
                ))
                fig.add_trace(go.Scatter(
                    x=r.x_values, y=r.sim_values,
                    mode="lines+markers", name="Simulated",
                    line=dict(color="#FF5722", dash="dash"),
                ))
                fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)

        # Export
        st.download_button(
            "Export Report (JSON)",
            data=json.dumps(report, indent=2),
            file_name="validation_report.json",
            mime="application/json",
        )


# ────────────────────────────────────────────────────────────────
# TAB 2: Parameter Calibration
# ────────────────────────────────────────────────────────────────

with tab_calib:
    st.subheader("Parameter Calibration")
    st.markdown(
        "Fit simulation parameters to match experimental data using "
        "optimization algorithms."
    )

    try:
        from cognisom.validation.calibrator import ParameterCalibrator, DEFAULT_PARAMETERS
        from cognisom.validation.benchmarks import get_all_benchmarks
    except ImportError:
        st.error("Calibrator module not found.")
        st.stop()

    all_bm = get_all_benchmarks()

    # Benchmark selector
    bm_options = {}
    for cat, bms in all_bm.items():
        for bm in bms:
            bm_options[bm.name] = bm

    selected_bm_name = st.selectbox(
        "Select benchmark to calibrate against",
        options=list(bm_options.keys()),
    )
    selected_bm = bm_options[selected_bm_name]

    # Method
    method = st.radio(
        "Optimization method",
        options=["nelder-mead", "differential_evolution"],
        format_func=lambda x: {
            "nelder-mead": "Nelder-Mead (fast, local)",
            "differential_evolution": "Differential Evolution (thorough, global)",
        }[x],
        horizontal=True,
    )

    max_iter = st.slider("Max iterations", 10, 500, 50)

    # Show tunable parameters
    relevant_params = [p for p in DEFAULT_PARAMETERS if p.module == selected_bm.module]
    if relevant_params:
        st.markdown("**Tunable Parameters:**")
        for p in relevant_params:
            st.markdown(
                f"- `{p.name}`: {p.description} "
                f"(range: {p.min_val}-{p.max_val} {p.units}, "
                f"initial: {p.initial})"
            )
    else:
        st.info(f"No tunable parameters defined for module '{selected_bm.module}'.")

    if st.button("Run Calibration", type="primary") and relevant_params:
        calibrator = ParameterCalibrator()
        progress = st.progress(0, text="Calibrating...")

        def cal_progress(iteration, max_i, error):
            progress.progress(
                min(iteration / max(max_i, 1), 1.0),
                text=f"Iteration {iteration}: error = {error:.6f}",
            )

        result = calibrator.calibrate(
            benchmark=selected_bm,
            parameters=relevant_params,
            method=method,
            max_iterations=max_iter,
            progress_cb=cal_progress,
        )
        progress.progress(1.0, text="Complete")

        st.session_state["cal_result"] = result

    if "cal_result" in st.session_state:
        result = st.session_state["cal_result"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Initial Error", f"{result.initial_error:.4f}")
        c2.metric("Final Error", f"{result.final_error:.4f}")
        c3.metric(
            "Improvement",
            f"{result.improvement_pct:.1f}%",
            delta=f"{result.improvement_pct:.1f}%",
        )

        st.markdown("**Optimized Parameters:**")
        for name, val in result.parameters.items():
            st.code(f"{name} = {val:.6f}")

        if result.history:
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[h["iteration"] for h in result.history],
                y=[h["error"] for h in result.history],
                mode="lines",
                name="Error",
                line=dict(color="#4CAF50"),
            ))
            fig.update_layout(
                title="Convergence",
                xaxis_title="Iteration",
                yaxis_title="Error",
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "Export Calibration (JSON)",
            data=json.dumps(result.to_dict(), indent=2),
            file_name="calibration_result.json",
            mime="application/json",
        )


# ────────────────────────────────────────────────────────────────
# TAB 3: Performance Profiler
# ────────────────────────────────────────────────────────────────

with tab_profile:
    st.subheader("Simulation Performance Profiler")
    st.markdown("Profile per-module execution time and identify bottlenecks.")

    try:
        from cognisom.validation.profiler import SimulationProfiler
    except ImportError:
        st.error("Profiler module not found.")
        st.stop()

    p_col1, p_col2, p_col3 = st.columns(3)
    duration = p_col1.number_input("Duration (hours)", 1.0, 168.0, 24.0)
    dt = p_col2.number_input("Time step (hours)", 0.01, 1.0, 0.1)
    n_cells = p_col3.number_input("Initial cells", 10, 10000, 100)

    available_modules = [
        "cellular", "immune", "molecular", "vascular",
        "lymphatic", "spatial", "epigenetic", "circadian", "morphogen",
    ]
    selected_modules = st.multiselect(
        "Modules to profile",
        options=available_modules,
        default=["cellular", "immune", "molecular"],
    )

    if st.button("Run Profiler", type="primary"):
        profiler = SimulationProfiler()
        progress = st.progress(0, text="Profiling...")

        total_steps = int(duration / dt)

        def prof_progress(step, total, elapsed):
            progress.progress(
                step / max(total, 1),
                text=f"Step {step}/{total} ({elapsed:.1f}s elapsed)",
            )

        report = profiler.profile(
            duration=duration,
            dt=dt,
            modules=selected_modules,
            n_cells=n_cells,
            progress_cb=prof_progress,
        )
        progress.progress(1.0, text="Complete")

        st.session_state["profile_report"] = report

    if "profile_report" in st.session_state:
        report = st.session_state["profile_report"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Time", f"{report.total_time_sec:.2f}s")
        m2.metric("Steps/Second", f"{report.steps_per_second:.0f}")
        m3.metric("Memory", f"{report.memory_current_mb:.0f} MB")
        m4.metric("Bottleneck", report.bottleneck_module)

        # Bar chart of module times
        import plotly.graph_objects as go

        sorted_mods = sorted(
            report.modules.values(),
            key=lambda m: m.total_time_sec, reverse=True,
        )
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[m.name for m in sorted_mods],
            y=[m.total_time_sec for m in sorted_mods],
            marker_color=["#FF5722" if m.name == report.bottleneck_module
                          else "#2196F3" for m in sorted_mods],
        ))
        fig.update_layout(
            title="Module Execution Time",
            xaxis_title="Module",
            yaxis_title="Total Time (s)",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Detailed table
        st.markdown("**Per-Module Statistics:**")
        rows = []
        for m in sorted_mods:
            pct = m.total_time_sec / max(report.total_time_sec, 0.001) * 100
            rows.append({
                "Module": m.name,
                "Total (s)": f"{m.total_time_sec:.3f}",
                "Avg (ms/step)": f"{m.avg_step_ms:.3f}",
                "P95 (ms)": f"{m.p95_step_ms:.3f}",
                "Max (ms)": f"{m.max_step_ms:.3f}",
                "% of Total": f"{pct:.1f}%",
            })
        st.dataframe(rows, use_container_width=True)

        # Text summary
        with st.expander("Full Profile Summary"):
            st.code(report.summary())

        st.download_button(
            "Export Profile (JSON)",
            data=json.dumps(report.to_dict(), indent=2),
            file_name="profile_report.json",
            mime="application/json",
        )


# ────────────────────────────────────────────────────────────────
# TAB 4: Synthetic Reference Data
# ────────────────────────────────────────────────────────────────

with tab_synth:
    st.subheader("Synthetic Reference Data Generator")
    st.markdown(
        "Generate synthetic datasets that follow published growth curves, "
        "immune kinetics, and metabolic profiles. Useful for testing the "
        "validation pipeline before real experimental data is available."
    )

    try:
        from cognisom.validation.data_sources import SyntheticDataGenerator
    except ImportError:
        st.error("Data sources module not found.")
        st.stop()

    gen = SyntheticDataGenerator(seed=42)

    if st.button("Generate All Synthetic Datasets", type="primary"):
        with st.spinner("Generating..."):
            paths = gen.generate_all()
        st.success(f"Generated {len(paths)} datasets")
        for name, path in paths.items():
            st.markdown(f"- `{name}`: {path}")
        st.session_state["synth_generated"] = True

    st.divider()

    # Preview each dataset type
    st.markdown("### Previews")

    # Tumor growth
    with st.expander("Tumor Spheroid Growth (Gompertzian)"):
        s_col1, s_col2 = st.columns(2)
        n_init = s_col1.number_input("Initial cells", 10, 500, 50, key="sg_n")
        dt_h = s_col2.number_input("Doubling time (h)", 8.0, 120.0, 18.0, key="sg_dt")

        data = gen.tumor_spheroid_growth(n_initial=n_init, doubling_time_h=dt_h)

        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.array(data["time_hours"]) / 24,
            y=data["cell_counts"],
            name="Cell Count",
        ))
        fig.update_layout(xaxis_title="Days", yaxis_title="Cells", height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Immune infiltration
    with st.expander("Immune Infiltration Time Course"):
        data = gen.immune_infiltration()
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data["time_days"], y=data["cd8_density_per_mm2"],
            name="CD8+ T cells",
        ))
        fig.add_trace(go.Scatter(
            x=data["time_days"], y=data["nk_density_per_mm2"],
            name="NK cells",
        ))
        fig.update_layout(
            xaxis_title="Days", yaxis_title="Cells/mm2", height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Metabolic
    with st.expander("Metabolic Profile (Normal vs Cancer)"):
        data = gen.metabolic_profile()
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data["time_hours"], y=data["normal"]["glucose_mM"],
            name="Normal Glucose", line=dict(color="#2196F3"),
        ))
        fig.add_trace(go.Scatter(
            x=data["time_hours"], y=data["cancer"]["glucose_mM"],
            name="Cancer Glucose", line=dict(color="#FF5722"),
        ))
        fig.add_trace(go.Scatter(
            x=data["time_hours"], y=data["cancer"]["lactate_mM"],
            name="Cancer Lactate", line=dict(color="#FF9800", dash="dash"),
        ))
        fig.update_layout(
            xaxis_title="Hours", yaxis_title="Concentration (mM)", height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Cell type proportions
    with st.expander("Cell Type Proportions (Prostate Tissue)"):
        data = gen.cell_type_proportions()
        import plotly.graph_objects as go
        fig = go.Figure()
        for tissue_type in ["normal_tissue", "tumor_tissue"]:
            props = data[tissue_type]
            fig.add_trace(go.Bar(
                x=list(props.keys()),
                y=list(props.values()),
                name=tissue_type.replace("_", " ").title(),
            ))
        fig.update_layout(
            barmode="group", height=350,
            xaxis_title="Cell Type", yaxis_title="Proportion",
        )
        st.plotly_chart(fig, use_container_width=True)
