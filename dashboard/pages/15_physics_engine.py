"""
Physics Engine Dashboard
========================

Configure and visualize GPU-accelerated cell mechanics simulation.

Features:
- Force parameter tuning (repulsion, adhesion, chemotaxis, Brownian)
- GPU backend status and selection
- Real-time force field visualization
- Cell mechanics simulation sandbox
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Physics Engine | Cognisom",
    page_icon="⚛️",
    layout="wide",
)

# Auth gate
try:
    from cognisom.auth.middleware import streamlit_page_gate
    user = streamlit_page_gate(required_tier="researcher")
except Exception:
    user = None

st.title("⚛️ Physics Engine")
st.markdown("GPU-accelerated cell mechanics using NVIDIA Warp")

# ═══════════════════════════════════════════════════════════════════════
# Sidebar: Configuration
# ═══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Configuration")

    # GPU Status
    st.subheader("GPU Backend")
    try:
        from cognisom.physics.warp_backend import check_warp_available, get_warp_backend
        warp_available = check_warp_available()
        if warp_available:
            backend = get_warp_backend()
            st.success(f"✓ Warp available on {backend.device}")
            gpu_info = backend.info() if hasattr(backend, 'info') else {}
            if gpu_info:
                st.json(gpu_info)
        else:
            st.warning("⚠ Warp not available - using NumPy fallback")
    except ImportError:
        warp_available = False
        st.warning("⚠ Physics module not installed")

    st.divider()

    # Force Parameters
    st.subheader("Force Parameters")

    k_repulsion = st.slider(
        "Repulsion Stiffness (pN/μm)",
        min_value=10.0,
        max_value=500.0,
        value=100.0,
        step=10.0,
        help="Soft-sphere collision avoidance strength"
    )

    k_adhesion = st.slider(
        "Adhesion Stiffness (pN/μm)",
        min_value=1.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
        help="E-cadherin mediated cell-cell adhesion"
    )

    adhesion_range = st.slider(
        "Adhesion Range (μm)",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.5,
        help="Distance beyond contact for adhesion"
    )

    noise_strength = st.slider(
        "Brownian Noise",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.01,
        help="Thermal fluctuation strength"
    )

    chemotaxis = st.slider(
        "Chemotaxis Strength",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help="Gradient-following sensitivity"
    )

    st.divider()

    # Dynamics
    st.subheader("Dynamics")
    overdamped = st.checkbox("Overdamped (Langevin)", value=True)
    damping = st.slider("Damping γ (pN·s/μm)", 0.1, 10.0, 1.0, 0.1)
    if not overdamped:
        mass = st.slider("Cell Mass", 0.1, 10.0, 1.0, 0.1)
    else:
        mass = 1.0

    dt = st.number_input("Time Step (hours)", 0.0001, 0.01, 0.001, format="%.4f")

# ═══════════════════════════════════════════════════════════════════════
# Main Content
# ═══════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "🧪 Sandbox", "📊 Force Fields", "📈 Benchmarks", "📖 Documentation"
])

# ─────────────────────────────────────────────────────────────────────────
# Tab 1: Sandbox Simulation
# ─────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Cell Mechanics Sandbox")
    st.markdown("Run a test simulation with the configured parameters.")

    col1, col2 = st.columns([1, 2])

    with col1:
        n_cells = st.number_input("Number of Cells", 10, 5000, 100, 10)
        region_size = st.slider("Region Size (μm)", 50, 500, 200, 10)
        cell_radius = st.slider("Cell Radius (μm)", 2.0, 15.0, 5.0, 0.5)
        n_steps = st.number_input("Simulation Steps", 10, 10000, 500, 10)

        # Force toggles
        st.markdown("**Active Forces:**")
        use_repulsion = st.checkbox("Repulsion", value=True)
        use_adhesion = st.checkbox("Adhesion", value=True)
        use_brownian = st.checkbox("Brownian Motion", value=True)
        use_chemotaxis = st.checkbox("Chemotaxis", value=False)

        run_sim = st.button("▶ Run Simulation", type="primary", use_container_width=True)

    with col2:
        if run_sim:
            with st.spinner("Running cell mechanics simulation..."):
                try:
                    from cognisom.physics.cell_mechanics import (
                        CellMechanics, CellMechanicsConfig, ForceType
                    )

                    # Build force type flag
                    force_types = ForceType.NONE
                    if use_repulsion:
                        force_types |= ForceType.REPULSION
                    if use_adhesion:
                        force_types |= ForceType.ADHESION
                    if use_brownian:
                        force_types |= ForceType.BROWNIAN
                    if use_chemotaxis:
                        force_types |= ForceType.CHEMOTAXIS

                    config = CellMechanicsConfig(
                        k_repulsion=k_repulsion,
                        k_adhesion=k_adhesion,
                        adhesion_range=adhesion_range,
                        noise_strength=noise_strength,
                        chemotaxis_strength=chemotaxis,
                        damping=damping,
                        mass=mass,
                        overdamped=overdamped,
                        dt=dt,
                        force_types=force_types,
                    )

                    mechanics = CellMechanics(n_cells=n_cells, config=config)

                    # Seed cells
                    mechanics.seed_random(
                        n_cells=n_cells,
                        region=(0, 0, 0, region_size, region_size, region_size),
                        radii=cell_radius,
                    )

                    # Run simulation and collect trajectory
                    positions_history = [mechanics.get_positions().copy()]
                    progress = st.progress(0)

                    for i in range(n_steps):
                        mechanics.step(dt)
                        if i % (n_steps // 10) == 0:
                            positions_history.append(mechanics.get_positions().copy())
                            progress.progress((i + 1) / n_steps)

                    positions_history.append(mechanics.get_positions().copy())
                    progress.progress(1.0)

                    # Visualize final state
                    final_pos = mechanics.get_positions()

                    fig = go.Figure(data=[go.Scatter3d(
                        x=final_pos[:, 0],
                        y=final_pos[:, 1],
                        z=final_pos[:, 2],
                        mode='markers',
                        marker=dict(
                            size=cell_radius,
                            color=np.linalg.norm(mechanics.get_forces(), axis=1),
                            colorscale='Viridis',
                            colorbar=dict(title="Force (pN)"),
                            opacity=0.8,
                        ),
                        text=[f"Cell {i}" for i in range(n_cells)],
                        hovertemplate="<b>%{text}</b><br>x: %{x:.1f}<br>y: %{y:.1f}<br>z: %{z:.1f}<extra></extra>"
                    )])

                    fig.update_layout(
                        title=f"Final Cell Positions (t={mechanics.time:.3f}h)",
                        scene=dict(
                            xaxis_title="X (μm)",
                            yaxis_title="Y (μm)",
                            zaxis_title="Z (μm)",
                            aspectmode='cube',
                        ),
                        height=500,
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Statistics
                    st.success(f"Simulation complete: {n_steps} steps, t={mechanics.time:.4f}h")

                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    forces = mechanics.get_forces()
                    force_mags = np.linalg.norm(forces, axis=1)

                    with col_stat1:
                        st.metric("Mean Force", f"{force_mags.mean():.2f} pN")
                    with col_stat2:
                        st.metric("Max Force", f"{force_mags.max():.2f} pN")
                    with col_stat3:
                        # Estimate packing density
                        vol_cells = n_cells * (4/3) * np.pi * cell_radius**3
                        vol_region = region_size**3
                        st.metric("Packing Density", f"{100*vol_cells/vol_region:.1f}%")

                except ImportError as e:
                    st.error(f"Physics module not available: {e}")
                except Exception as e:
                    st.error(f"Simulation error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        else:
            # Show placeholder
            st.info("Configure parameters and click **Run Simulation** to start.")

            # Show force equations
            st.markdown("""
            **Force Model:**

            ```
            F_total = F_repulsion + F_adhesion + F_chemotaxis + F_random

            F_repulsion = k_rep × overlap × n̂     (soft-sphere)
            F_adhesion  = -k_adh × stretch × n̂   (E-cadherin)
            F_chemotaxis = μ × ∇[concentration]  (gradient following)
            F_random    = √(2kT/γ) × ξ(t)        (Brownian)

            Overdamped: dx/dt = F/γ
            Inertial:   m×dv/dt = F - γv
            ```
            """)

# ─────────────────────────────────────────────────────────────────────────
# Tab 2: Force Field Visualization
# ─────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Force Field Visualization")
    st.markdown("Visualize how forces vary with distance and parameters.")

    col1, col2 = st.columns(2)

    with col1:
        # Repulsion + Adhesion force profile
        distances = np.linspace(0.1, 30, 200)
        contact_dist = 2 * cell_radius  # Two cells touching

        forces_rep = np.zeros_like(distances)
        forces_adh = np.zeros_like(distances)

        for i, d in enumerate(distances):
            if d < contact_dist:
                overlap = contact_dist - d
                forces_rep[i] = k_repulsion * overlap
            if contact_dist <= d < contact_dist + adhesion_range:
                stretch = d - contact_dist
                forces_adh[i] = -k_adhesion * stretch

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=distances, y=forces_rep,
            name="Repulsion",
            line=dict(color="red", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=distances, y=forces_adh,
            name="Adhesion",
            line=dict(color="blue", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=distances, y=forces_rep + forces_adh,
            name="Net Force",
            line=dict(color="green", width=3, dash="dash"),
        ))

        fig.add_vline(x=contact_dist, line_dash="dot",
                      annotation_text="Contact", annotation_position="top right")
        fig.add_vline(x=contact_dist + adhesion_range, line_dash="dot",
                      annotation_text="Adhesion limit", annotation_position="top right")

        fig.update_layout(
            title="Pairwise Force Profile",
            xaxis_title="Distance (μm)",
            yaxis_title="Force (pN)",
            height=400,
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Chemotaxis gradient response
        gradient_mag = np.linspace(0, 10, 100)
        chemotaxis_force = chemotaxis * gradient_mag

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=gradient_mag, y=chemotaxis_force,
            fill='tozeroy',
            line=dict(color="purple", width=2),
        ))

        fig2.update_layout(
            title="Chemotaxis Response",
            xaxis_title="Gradient Magnitude (μM/μm)",
            yaxis_title="Chemotactic Force (pN)",
            height=400,
        )

        st.plotly_chart(fig2, use_container_width=True)

    # Brownian motion distribution
    st.markdown("### Brownian Force Distribution")
    st.markdown(f"Noise strength: **{noise_strength}** pN")

    # Sample Brownian forces
    n_samples = 10000
    brownian_forces = noise_strength * np.random.randn(n_samples, 3)
    force_magnitudes = np.linalg.norm(brownian_forces, axis=1)

    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(
        x=force_magnitudes,
        nbinsx=50,
        name="Brownian Force Magnitude",
        marker_color="orange",
    ))

    fig3.update_layout(
        title="Brownian Force Magnitude Distribution",
        xaxis_title="Force Magnitude (pN)",
        yaxis_title="Count",
        height=300,
    )

    st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────
# Tab 3: Benchmarks
# ─────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Performance Benchmarks")
    st.markdown("Compare GPU vs CPU performance for cell mechanics.")

    if st.button("Run Benchmark", type="primary"):
        import time

        benchmark_sizes = [100, 500, 1000, 2000]
        cpu_times = []
        gpu_times = []

        progress = st.progress(0)

        for idx, n in enumerate(benchmark_sizes):
            st.write(f"Testing {n} cells...")

            # NumPy timing
            try:
                from cognisom.physics.cell_mechanics import CellMechanics, CellMechanicsConfig

                config = CellMechanicsConfig(
                    k_repulsion=k_repulsion,
                    k_adhesion=k_adhesion,
                    adhesion_range=adhesion_range,
                )

                mechanics = CellMechanics(n_cells=n, config=config)
                mechanics.seed_random(n, (0, 0, 0, 200, 200, 200), radii=5.0)

                # Warmup
                for _ in range(10):
                    mechanics.step(0.001)

                # Benchmark
                start = time.perf_counter()
                for _ in range(100):
                    mechanics.step(0.001)
                elapsed = time.perf_counter() - start

                if warp_available:
                    gpu_times.append(elapsed)
                else:
                    cpu_times.append(elapsed)

            except Exception as e:
                st.warning(f"Benchmark error for n={n}: {e}")
                if warp_available:
                    gpu_times.append(None)
                else:
                    cpu_times.append(None)

            progress.progress((idx + 1) / len(benchmark_sizes))

        # Plot results
        fig = go.Figure()

        if cpu_times:
            fig.add_trace(go.Bar(
                x=[str(n) for n in benchmark_sizes],
                y=cpu_times,
                name="CPU (NumPy)",
                marker_color="blue",
            ))

        if gpu_times:
            fig.add_trace(go.Bar(
                x=[str(n) for n in benchmark_sizes],
                y=gpu_times,
                name="GPU (Warp)",
                marker_color="green",
            ))

        fig.update_layout(
            title="100 Steps Execution Time",
            xaxis_title="Number of Cells",
            yaxis_title="Time (seconds)",
            barmode='group',
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Speedup calculation
        if cpu_times and gpu_times and all(cpu_times) and all(gpu_times):
            speedups = [c/g for c, g in zip(cpu_times, gpu_times)]
            st.metric("Average Speedup", f"{np.mean(speedups):.1f}x")
    else:
        st.info("Click **Run Benchmark** to compare CPU vs GPU performance.")

        # Show expected scaling
        st.markdown("""
        **Expected Complexity:**

        | Algorithm | Complexity | Notes |
        |-----------|------------|-------|
        | Pairwise forces | O(N²) | All-pairs interaction |
        | Spatial hashing | O(N) | For large N (>5000) |
        | Integration | O(N) | Per-cell position update |

        GPU acceleration provides ~10-100x speedup for N > 1000 cells.
        """)

# ─────────────────────────────────────────────────────────────────────────
# Tab 4: Documentation
# ─────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Physics Engine Documentation")

    st.markdown("""
    ## Overview

    The Cognisom physics engine provides GPU-accelerated cell mechanics simulation
    using **NVIDIA Warp**. This enables real-time simulation of thousands of cells
    with physically accurate force interactions.

    ## Force Model

    ### 1. Repulsion (Soft-Sphere)

    Prevents cell overlap using a linear spring force:

    ```
    F_rep = k_rep × max(0, contact_distance - distance) × n̂
    ```

    - `k_rep`: Repulsion stiffness (pN/μm)
    - `contact_distance`: Sum of cell radii
    - `n̂`: Unit vector from cell j to cell i

    ### 2. Adhesion (E-Cadherin)

    Models cell-cell adhesion via cadherin molecules:

    ```
    F_adh = -k_adh × stretch × n̂   (if contact < d < contact + range)
    ```

    - `k_adh`: Adhesion stiffness
    - `stretch`: Distance beyond contact
    - `range`: Maximum adhesion distance

    ### 3. Chemotaxis

    Cells follow chemical gradients (e.g., CXCL12 for bone metastasis):

    ```
    F_chem = μ × ∇C
    ```

    - `μ`: Chemotaxis sensitivity
    - `∇C`: Concentration gradient

    ### 4. Brownian Motion

    Thermal fluctuations via Langevin dynamics:

    ```
    F_random = √(2kT/γ) × ξ(t)
    ```

    - `ξ(t)`: Gaussian white noise
    - `γ`: Damping coefficient

    ## Integration Methods

    ### Overdamped (Default)

    For systems where inertia is negligible:

    ```
    dx/dt = F/γ
    ```

    ### Velocity Verlet

    For inertial dynamics:

    ```
    a = (F - γv) / m
    v += a × dt
    x += v × dt
    ```

    ## API Usage

    ```python
    from cognisom.physics.cell_mechanics import CellMechanics, CellMechanicsConfig

    config = CellMechanicsConfig(
        k_repulsion=100.0,
        k_adhesion=10.0,
        adhesion_range=2.0,
        noise_strength=0.1,
        overdamped=True,
    )

    mechanics = CellMechanics(n_cells=1000, config=config)
    mechanics.seed_random(1000, (0, 0, 0, 200, 200, 200), radii=5.0)

    for _ in range(10000):
        mechanics.step(dt=0.001)

    positions = mechanics.get_positions()
    ```

    ## References

    - Drasdo & Höhme (2005) - Individual-based models of tumor growth
    - Van Liedekerke et al. (2015) - Computational models for cell mechanics
    - Newman (2005) - Mathematical modeling of tissue mechanics
    """)

# Footer
from cognisom.dashboard.footer import render_footer
render_footer()
