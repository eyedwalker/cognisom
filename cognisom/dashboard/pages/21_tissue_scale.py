"""
Tissue-Scale Simulation — Multi-GPU Digital Twin
=================================================

Dashboard page for configuring and running tissue-scale simulations
(100K-5M cells) across multiple GPUs. Supports:

1. Local CPU simulation (free, for development/testing)
2. AWS single-GPU validation (g6e.2xlarge, ~$2.56/hr)
3. Lambda Labs 8x B200 full simulation (~$39.92/hr)

Sections:
- Instance Control: Launch/terminate Lambda Labs, status, cost tracker
- Configuration: Cell count, grid, archetypes, module toggles
- Live Simulation: 3D scatter, concentration heatmaps, population charts
"""

import streamlit as st
import time
import numpy as np

st.set_page_config(
    page_title="Tissue-Scale Simulation | Cognisom",
    page_icon="🧬",
    layout="wide",
)

# Auth gate
try:
    from cognisom.auth.middleware import streamlit_page_gate
    user = streamlit_page_gate(required_tier="researcher")
except Exception:
    user = None

st.title("🧬 Tissue-Scale Simulation")
st.markdown(
    "Multi-GPU tissue simulation: 100K–5M cells with diffusion, "
    "mechanics, and intracellular signaling"
)

# ═══════════════════════════════════════════════════════════════════
# Session State
# ═══════════════════════════════════════════════════════════════════

if "tissue_engine" not in st.session_state:
    st.session_state.tissue_engine = None
if "tissue_running" not in st.session_state:
    st.session_state.tissue_running = False
if "tissue_snapshots" not in st.session_state:
    st.session_state.tissue_snapshots = []
if "tissue_metrics" not in st.session_state:
    st.session_state.tissue_metrics = []
if "lambda_manager" not in st.session_state:
    st.session_state.lambda_manager = None
if "lambda_instance_id" not in st.session_state:
    st.session_state.lambda_instance_id = None


# ═══════════════════════════════════════════════════════════════════
# Tab Layout
# ═══════════════════════════════════════════════════════════════════

tab_config, tab_sim, tab_instance, tab_results = st.tabs([
    "Configuration", "Live Simulation", "Instance Control", "Results"
])


# ═══════════════════════════════════════════════════════════════════
# TAB 1: Configuration
# ═══════════════════════════════════════════════════════════════════

with tab_config:
    st.subheader("Simulation Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Scale**")
        preset = st.selectbox(
            "Preset",
            ["Small Test (10K cells)", "Medium (100K cells)", "Full Tissue (1M cells)", "Custom"],
            index=0,
        )

        if preset == "Small Test (10K cells)":
            default_cells, default_grid, default_gpus = 10_000, 100, 8
            default_duration = 1.0
        elif preset == "Medium (100K cells)":
            default_cells, default_grid, default_gpus = 100_000, 200, 8
            default_duration = 4.0
        elif preset == "Full Tissue (1M cells)":
            default_cells, default_grid, default_gpus = 1_000_000, 500, 8
            default_duration = 24.0
        else:
            default_cells, default_grid, default_gpus = 50_000, 150, 8
            default_duration = 2.0

        n_cells = st.number_input(
            "Number of cells",
            min_value=1000, max_value=5_000_000, value=default_cells, step=10_000,
        )
        grid_size = st.number_input(
            "Grid size (voxels per axis)",
            min_value=50, max_value=1000, value=default_grid, step=50,
        )
        n_gpus = st.slider("GPU partitions", 1, 16, default_gpus)
        resolution = st.number_input(
            "Resolution (um/voxel)", min_value=1.0, max_value=50.0, value=10.0,
        )

    with col2:
        st.markdown("**Time Stepping**")
        duration = st.number_input(
            "Duration (hours)", min_value=0.1, max_value=48.0,
            value=default_duration, step=0.5,
        )
        dt = st.number_input(
            "dt (hours/step)", min_value=0.001, max_value=0.1,
            value=0.01, step=0.005, format="%.3f",
        )
        diff_substeps = st.slider("Diffusion substeps", 1, 20, 5)
        mech_substeps = st.slider("Mechanics substeps", 1, 20, 10)
        snapshot_interval = st.number_input(
            "Snapshot interval (steps)", min_value=1, max_value=1000, value=100,
        )

    st.markdown("---")
    st.markdown("**Modules**")
    mcol1, mcol2, mcol3 = st.columns(3)
    with mcol1:
        enable_diffusion = st.checkbox("3D Diffusion", value=True)
        enable_ode = st.checkbox("Intracellular ODE (AR signaling)", value=True)
    with mcol2:
        enable_mechanics = st.checkbox("Cell Mechanics", value=True)
        enable_cell_cycle = st.checkbox("Cell Cycle (division/death)", value=True)
    with mcol3:
        enable_ssa = st.checkbox("Stochastic Gene Expression", value=False)
        enable_immune = st.checkbox("Immune Response", value=False)

    st.markdown("---")
    st.markdown("**Cell Composition**")
    composition = {}
    comp_cols = st.columns(5)
    default_comp = {
        "luminal_secretory": 0.30, "basal": 0.15, "fibroblast": 0.15,
        "smooth_muscle": 0.10, "endothelial": 0.08,
        "cancer_epithelial": 0.10, "t_cell": 0.05,
        "macrophage": 0.04, "nk_cell": 0.02, "neuroendocrine": 0.01,
    }
    for i, (name, frac) in enumerate(default_comp.items()):
        with comp_cols[i % 5]:
            composition[name] = st.number_input(
                name.replace("_", " ").title(),
                min_value=0.0, max_value=1.0, value=frac,
                step=0.01, format="%.2f", key=f"comp_{name}",
            )

    st.markdown("---")
    st.markdown("**Execution Mode**")
    exec_mode = st.radio(
        "Run on",
        ["Local CPU (free)", "Lambda Labs 8x B200 (~$40/hr)"],
        horizontal=True,
    )

    # Safety settings for remote
    if exec_mode != "Local CPU (free)":
        scol1, scol2 = st.columns(2)
        with scol1:
            max_runtime = st.number_input(
                "Max runtime (hours)", min_value=0.5, max_value=24.0,
                value=2.0, step=0.5,
            )
        with scol2:
            budget = st.number_input(
                "Budget cap (USD)", min_value=10.0, max_value=5000.0,
                value=80.0, step=10.0,
            )

    # Build config button
    if st.button("Build Configuration", type="primary"):
        try:
            from cognisom.core.tissue_config import TissueScaleConfig
            tissue_size = (
                grid_size * resolution,
                grid_size * resolution,
                grid_size * resolution,
            )
            config = TissueScaleConfig(
                n_cells=n_cells,
                grid_shape=(grid_size, grid_size, grid_size),
                resolution_um=resolution,
                tissue_size_um=tissue_size,
                n_gpus=n_gpus,
                dt=dt,
                duration=duration,
                diffusion_substeps=diff_substeps,
                mechanics_substeps=mech_substeps,
                initial_composition=composition,
                enable_ode=enable_ode,
                enable_ssa=enable_ssa,
                enable_mechanics=enable_mechanics,
                enable_diffusion=enable_diffusion,
                enable_cell_cycle=enable_cell_cycle,
                enable_immune=enable_immune,
                snapshot_interval_steps=snapshot_interval,
                execution_mode="local" if exec_mode == "Local CPU (free)" else "remote",
                max_runtime_hours=max_runtime if exec_mode != "Local CPU (free)" else 24.0,
                budget_usd=budget if exec_mode != "Local CPU (free)" else 0.0,
            )
            st.session_state.tissue_config = config
            st.success(
                f"Configuration built: {n_cells:,} cells, "
                f"{grid_size}^3 grid, {n_gpus} GPUs, "
                f"{duration}h duration ({config.total_steps} steps)"
            )
        except Exception as e:
            st.error(f"Configuration error: {e}")


# ═══════════════════════════════════════════════════════════════════
# TAB 2: Live Simulation
# ═══════════════════════════════════════════════════════════════════

with tab_sim:
    st.subheader("Live Simulation")

    config = st.session_state.get("tissue_config")
    engine = st.session_state.get("tissue_engine")

    if config is None:
        st.info("Configure the simulation in the Configuration tab first.")
    else:
        # Control buttons
        bcol1, bcol2, bcol3, bcol4 = st.columns(4)

        with bcol1:
            if st.button("Initialize Engine"):
                with st.spinner("Initializing tissue engine..."):
                    try:
                        from cognisom.core.tissue_engine import TissueSimulationEngine
                        engine = TissueSimulationEngine(config)
                        engine.initialize()
                        st.session_state.tissue_engine = engine
                        st.session_state.tissue_snapshots = []
                        st.session_state.tissue_metrics = []
                        st.success(
                            f"Engine initialized: {config.n_cells:,} cells "
                            f"across {config.n_gpus} partitions "
                            f"(mode: {engine._backend.mode})"
                        )
                    except Exception as e:
                        st.error(f"Initialization failed: {e}")

        with bcol2:
            run_steps = st.number_input(
                "Steps to run", min_value=1, max_value=10000, value=100,
                key="run_steps",
            )

        with bcol3:
            if st.button("Run Steps") and engine is not None:
                st.session_state.tissue_running = True

        with bcol4:
            if st.button("Stop"):
                st.session_state.tissue_running = False

        # Run simulation steps
        if st.session_state.tissue_running and engine is not None:
            progress_bar = st.progress(0)
            status_text = st.empty()
            step_count = st.session_state.get("run_steps", 100)

            for i in range(step_count):
                if not st.session_state.tissue_running:
                    break

                snapshot = engine.step()
                progress_bar.progress((i + 1) / step_count)

                if snapshot is not None:
                    st.session_state.tissue_snapshots.append(snapshot)
                    metrics = snapshot.get("metrics", {})
                    st.session_state.tissue_metrics.append(metrics)

                status_text.text(
                    f"Step {engine.step_count} | "
                    f"t = {engine.sim_time:.3f} hr | "
                    f"Cells: {sum(ca.n_real for ca in engine.cell_arrays):,}"
                )

            st.session_state.tissue_running = False
            st.success(f"Completed {step_count} steps")

        # Visualization
        if engine is not None and engine.is_initialized:
            st.markdown("---")
            viz_data = engine.get_visualization_data()

            # Metrics
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            summary = viz_data.get("summary", {})
            with mcol1:
                st.metric("Total Cells", f"{summary.get('total_cells', 0):,}")
            with mcol2:
                st.metric("Alive Cells", f"{summary.get('alive_cells', 0):,}")
            with mcol3:
                st.metric("Sim Time", f"{viz_data.get('sim_time', 0):.3f} hr")
            with mcol4:
                st.metric("Balance", f"{summary.get('balance_pct', 0):.0f}%")

            # 3D Scatter plot
            positions = viz_data.get("positions")
            if positions is not None and len(positions) > 0:
                st.markdown("### Cell Positions (3D)")
                # Subsample for display
                max_display = min(50_000, len(positions))
                if len(positions) > max_display:
                    idx = np.random.choice(len(positions), max_display, replace=False)
                    display_pos = positions[idx]
                else:
                    display_pos = positions

                import pandas as pd
                df = pd.DataFrame(display_pos, columns=["x", "y", "z"])
                st.scatter_chart(df, x="x", y="y", height=400)

            # Field slices
            field_slices = viz_data.get("field_slices", {})
            if field_slices:
                st.markdown("### Concentration Fields (center plane)")
                fcols = st.columns(len(field_slices))
                for i, (name, slice_data) in enumerate(field_slices.items()):
                    with fcols[i]:
                        st.markdown(f"**{name}**")
                        # Normalize for display
                        if slice_data.max() > slice_data.min():
                            norm = (slice_data - slice_data.min()) / (
                                slice_data.max() - slice_data.min()
                            )
                        else:
                            norm = np.zeros_like(slice_data)
                        st.image(
                            norm,
                            caption=f"{name}: [{slice_data.min():.3f}, {slice_data.max():.3f}]",
                            clamp=True,
                        )

            # Per-GPU load balance
            per_gpu = summary.get("per_gpu", [])
            if per_gpu:
                st.markdown("### GPU Load Balance")
                import pandas as pd
                gpu_df = pd.DataFrame(per_gpu)
                st.bar_chart(gpu_df.set_index("gpu_id")["n_real"])

            # Metrics history
            metrics_list = st.session_state.get("tissue_metrics", [])
            if len(metrics_list) > 1:
                st.markdown("### Population Over Time")
                import pandas as pd
                mdf = pd.DataFrame(metrics_list)
                if "sim_time_hr" in mdf.columns and "alive_cells" in mdf.columns:
                    st.line_chart(mdf.set_index("sim_time_hr")["alive_cells"])


# ═══════════════════════════════════════════════════════════════════
# TAB 3: Instance Control (Lambda Labs)
# ═══════════════════════════════════════════════════════════════════

with tab_instance:
    st.subheader("Lambda Labs Instance Control")

    st.markdown(
        "Launch GPU instances on Lambda Labs for tissue-scale simulation. "
        "**Safety limits protect against runaway costs.**"
    )

    # Auto-load API key from Secrets Manager
    if "lambda_api_key" not in st.session_state or not st.session_state.lambda_api_key:
        try:
            from cognisom.infrastructure.lambda_lifecycle import _fetch_lambda_key_from_secrets
            auto_key = _fetch_lambda_key_from_secrets()
            if auto_key:
                st.session_state.lambda_api_key = auto_key
        except Exception:
            pass

    api_key = st.session_state.get("lambda_api_key", "")
    if api_key:
        st.success("API key loaded from AWS Secrets Manager")
    else:
        api_key = st.text_input(
            "Lambda API Key",
            type="password",
            help="Auto-loaded from Secrets Manager, or enter manually",
        )
        if api_key:
            st.session_state.lambda_api_key = api_key

    # Fetch live availability
    from cognisom.infrastructure.lambda_lifecycle import LambdaLifecycleManager
    available_types = []
    if api_key:
        _probe = LambdaLifecycleManager(api_key=api_key)
        available_types = _probe.list_available_types()

    icol1, icol2 = st.columns(2)

    with icol1:
        st.markdown("**Instance Settings**")
        # Build instance type list from live API + known types
        type_options = [t["type"] for t in available_types] if available_types else []
        # Always include B200 options (will appear when credits land)
        for fallback in ["gpu_8x_b200", "gpu_1x_b200", "gpu_8x_h100_sxm5"]:
            if fallback not in type_options:
                type_options.append(fallback)
        instance_type = st.selectbox("Instance Type", type_options)

        # Show availability status
        if available_types:
            avail_names = {t["type"] for t in available_types}
            if instance_type in avail_names:
                match = next(t for t in available_types if t["type"] == instance_type)
                st.success(
                    f"Available: {match['gpus']} GPUs, "
                    f"${match['price_per_hour']:.2f}/hr, "
                    f"regions: {', '.join(match['regions'])}"
                )
            else:
                st.warning(f"{instance_type} not currently available (waiting for credits?)")

        ssh_key = st.text_input("SSH Key Name", value="cognisom")
        region = st.text_input("Region (blank = auto)", value="")

    with icol2:
        st.markdown("**Safety Limits**")
        inst_max_runtime = st.number_input(
            "Max runtime (hours)", 0.5, 24.0, 2.0, step=0.5,
            key="inst_max_runtime",
        )
        inst_budget = st.number_input(
            "Budget cap (USD)", 10.0, 5000.0, 80.0, step=10.0,
            key="inst_budget",
        )

        # Cost estimate
        from cognisom.infrastructure.lambda_lifecycle import LambdaLifecycleManager
        rate = LambdaLifecycleManager.HOURLY_COST.get(instance_type, 40.0)
        est_cost = rate * inst_max_runtime
        st.info(f"Estimated max cost: **${est_cost:.2f}** ({inst_max_runtime}h x ${rate:.2f}/hr)")

    st.markdown("---")

    # Launch / Terminate buttons
    lcol1, lcol2, lcol3, lcol4 = st.columns(4)

    with lcol1:
        if st.button("Launch Instance", type="primary"):
            if not api_key:
                st.error("Enter Lambda API key first")
            else:
                manager = LambdaLifecycleManager(
                    api_key=api_key,
                    instance_type=instance_type,
                    region=region or None,
                    ssh_key_name=ssh_key or None,
                    max_runtime_hours=inst_max_runtime,
                    budget_usd=inst_budget,
                )
                with st.spinner(
                    f"Launching {instance_type}... "
                    f"(max ${est_cost:.0f}, {inst_max_runtime}h limit)"
                ):
                    ok, msg, iid = manager.launch_instance()
                if ok:
                    st.session_state.lambda_manager = manager
                    st.session_state.lambda_instance_id = iid
                    st.success(f"Launched: {msg}")
                else:
                    st.error(msg)

    with lcol2:
        if st.button("Check Status"):
            manager = st.session_state.get("lambda_manager")
            if manager:
                status = manager.get_status()
                st.json(status)
            else:
                st.warning("No instance tracked")

    with lcol3:
        if st.button("Terminate", type="secondary"):
            manager = st.session_state.get("lambda_manager")
            if manager:
                ok, msg = manager.terminate_instance()
                if ok:
                    st.success(msg)
                    st.session_state.lambda_instance_id = None
                else:
                    st.error(msg)

    with lcol4:
        if st.button("List Available"):
            key = api_key or st.session_state.get("lambda_api_key", "")
            if key:
                mgr = LambdaLifecycleManager(api_key=key)
                available = mgr.list_available_types()
                if available:
                    import pandas as pd
                    st.dataframe(pd.DataFrame(available))
                else:
                    st.info("No instance types available (or API error)")

    # Live cost tracker
    manager = st.session_state.get("lambda_manager")
    if manager and manager.is_launched:
        st.markdown("---")
        st.markdown("### Live Cost Tracker")
        cost_info = manager.cost_summary()
        ccol1, ccol2, ccol3, ccol4 = st.columns(4)
        with ccol1:
            st.metric("Elapsed", f"{cost_info['elapsed_hours']:.1f} hr")
        with ccol2:
            st.metric("Cost", f"${cost_info['estimated_cost']:.2f}")
        with ccol3:
            st.metric("Budget Left", f"${cost_info['budget_remaining']:.2f}")
        with ccol4:
            st.metric("Max Runtime", f"{cost_info['max_runtime_hours']:.1f} hr")

        # Safety check
        safe, reason = manager.check_safety_limits()
        if not safe:
            st.error(f"SAFETY LIMIT: {reason}")


# ═══════════════════════════════════════════════════════════════════
# TAB 4: Results
# ═══════════════════════════════════════════════════════════════════

with tab_results:
    st.subheader("Simulation Results")

    engine = st.session_state.get("tissue_engine")
    metrics_list = st.session_state.get("tissue_metrics", [])

    if engine is None:
        st.info("Run a simulation first to see results.")
    elif not metrics_list:
        st.info("No snapshots collected yet. Run some steps in the Live Simulation tab.")
    else:
        # Summary
        import pandas as pd
        mdf = pd.DataFrame(metrics_list)

        st.markdown("### Simulation Summary")
        scol1, scol2, scol3 = st.columns(3)
        with scol1:
            st.metric("Steps", engine.step_count)
        with scol2:
            st.metric("Sim Time", f"{engine.sim_time:.3f} hr")
        with scol3:
            total = sum(ca.n_real for ca in engine.cell_arrays)
            st.metric("Final Cells", f"{total:,}")

        # Population chart
        if "alive_cells" in mdf.columns:
            st.markdown("### Cell Population")
            st.line_chart(mdf.set_index("step")["alive_cells"])

        # Balance chart
        if "balance_pct" in mdf.columns:
            st.markdown("### GPU Balance")
            st.line_chart(mdf.set_index("step")["balance_pct"])

        # Step timing
        if "wall_time_s" in mdf.columns:
            st.markdown("### Step Timing")
            st.line_chart(mdf.set_index("step")["wall_time_s"])

        # Export
        st.markdown("---")
        if st.button("Export Metrics CSV"):
            csv = mdf.to_csv(index=False)
            st.download_button(
                "Download CSV",
                data=csv,
                file_name="tissue_metrics.csv",
                mime="text/csv",
            )
