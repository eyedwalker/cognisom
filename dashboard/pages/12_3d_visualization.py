"""
Page 12: 3D Visualization & Scientific Inspector
==================================================

Interactive 3D viewers for cell populations, spatial concentration
fields, and cell interaction networks. Includes scientific inspection
tools: time scrubber, cell picker, and lineage tree. Supports export
to PDB, glTF, VTK, and CSV.
"""

import streamlit as st
import numpy as np
import tempfile
import json
from pathlib import Path

st.set_page_config(page_title="3D Visualization", page_icon="3", layout="wide")

from cognisom.auth.middleware import streamlit_page_gate
user = streamlit_page_gate("12_3d_visualization")

st.title("3D Visualization & Scientific Inspector")
st.caption("Interactive 3D viewers and scientific inspection tools")

# ── Tabs ─────────────────────────────────────────────────────────

tab_cells, tab_fields, tab_network, tab_inspect, tab_lineage, tab_omniverse, tab_export = st.tabs([
    "Cell Population", "Spatial Fields", "Interaction Network",
    "Scientific Inspector", "Lineage Tree", "Omniverse/USD", "Export",
])


# ────────────────────────────────────────────────────────────────
# TAB 1: Cell Population 3D
# ────────────────────────────────────────────────────────────────

with tab_cells:
    st.subheader("3D Cell Population Viewer")

    try:
        from cognisom.visualization.cell_renderer import (
            CellPopulationRenderer, CELL_TYPE_COLORS,
        )
    except ImportError:
        st.error("Visualization module not available.")
        st.stop()

    # Data source selection
    source = st.radio(
        "Data Source",
        ["Demo Data", "From Simulation State"],
        horizontal=True,
        key="cell_source",
    )

    if source == "Demo Data":
        col1, col2, col3 = st.columns(3)
        n_normal = col1.slider("Normal Cells", 10, 200, 60, key="n_normal")
        n_cancer = col2.slider("Cancer Cells", 5, 100, 20, key="n_cancer")
        n_immune = col3.slider("Immune Cells", 5, 50, 10, key="n_immune")

        cells = CellPopulationRenderer.generate_demo_cells(
            n_normal=n_normal, n_cancer=n_cancer, n_immune=n_immune,
        )
        st.session_state["viz_cells"] = cells
    else:
        if "sim_state" in st.session_state:
            renderer = CellPopulationRenderer()
            fig = renderer.render_from_engine(st.session_state["sim_state"])
            st.plotly_chart(fig, use_container_width=True)
            st.stop()
        else:
            st.info("No simulation state available. Run a simulation on the Simulation page first, or use Demo Data.")
            cells = CellPopulationRenderer.generate_demo_cells()
            st.session_state["viz_cells"] = cells

    # Render options
    col_a, col_b, col_c = st.columns(3)
    color_by = col_a.selectbox(
        "Color By",
        ["type", "phase", "oxygen", "glucose", "atp", "mhc1"],
        key="cell_color",
    )
    size_by = col_b.selectbox(
        "Size By",
        ["fixed", "volume", "atp"],
        key="cell_size",
    )
    show_dead = col_c.checkbox("Show Dead Cells", value=False, key="show_dead")
    highlight_mut = col_c.checkbox("Highlight Mutations", value=True, key="highlight_mut")

    renderer = CellPopulationRenderer()
    fig = renderer.render(
        cells,
        color_by=color_by,
        size_by=size_by,
        show_dead=show_dead,
        highlight_mutations=highlight_mut,
        title=f"Cell Population ({len(cells)} cells)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Stats
    with st.expander("Population Statistics"):
        types = {}
        for c in cells:
            ct = c.get("cell_type", "unknown")
            types[ct] = types.get(ct, 0) + 1

        cols = st.columns(min(len(types), 5))
        for i, (ct, count) in enumerate(sorted(types.items(), key=lambda x: -x[1])):
            color = CELL_TYPE_COLORS.get(ct, "#999")
            cols[i % len(cols)].metric(
                ct.replace("_", " ").title(),
                count,
            )

        alive = sum(1 for c in cells if c.get("alive", True))
        st.caption(f"Alive: {alive} / {len(cells)}")

        if any(c.get("mutations") for c in cells):
            mut_counts = {}
            for c in cells:
                for m in c.get("mutations", []):
                    mut_counts[m] = mut_counts.get(m, 0) + 1
            st.write("**Mutation Frequency:**")
            for m, cnt in sorted(mut_counts.items(), key=lambda x: -x[1]):
                st.write(f"- {m}: {cnt} cells ({cnt/len(cells)*100:.1f}%)")


# ────────────────────────────────────────────────────────────────
# TAB 2: Spatial Fields
# ────────────────────────────────────────────────────────────────

with tab_fields:
    st.subheader("3D Concentration Fields")

    try:
        from cognisom.visualization.field_renderer import SpatialFieldRenderer
    except ImportError:
        st.error("Field renderer not available.")
        st.stop()

    # Generate demo fields
    if "viz_fields" not in st.session_state:
        st.session_state["viz_fields"] = SpatialFieldRenderer.generate_demo_fields()

    fields = st.session_state["viz_fields"]

    field_name = st.selectbox("Field", list(fields.keys()), key="field_select")
    field_data = fields[field_name]

    view_mode = st.radio(
        "View Mode",
        ["Isosurface (3D)", "Cross-Section Slices", "Multi-Field Comparison"],
        horizontal=True,
        key="field_view",
    )

    renderer = SpatialFieldRenderer()

    if view_mode == "Isosurface (3D)":
        col1, col2 = st.columns(2)
        opacity = col1.slider("Opacity", 0.1, 1.0, 0.3, key="iso_opacity")
        colorscale = col2.selectbox(
            "Colorscale",
            ["Viridis", "Inferno", "Plasma", "Cividis", "Turbo", "RdBu"],
            key="iso_colorscale",
        )

        fig = renderer.render_isosurface(
            field_data, name=field_name,
            opacity=opacity, colorscale=colorscale,
        )
        st.plotly_chart(fig, use_container_width=True)

    elif view_mode == "Cross-Section Slices":
        col1, col2 = st.columns(2)
        axis = col1.selectbox("Slice Axis", ["z", "y", "x"], key="slice_axis")
        axis_map = {"x": 0, "y": 1, "z": 2}
        max_idx = field_data.shape[axis_map[axis]] - 1
        slice_idx = col2.slider("Slice Position", 0, max_idx, max_idx // 2, key="slice_pos")

        fig = renderer.render_slices(
            field_data, name=field_name,
            slice_axis=axis, slice_indices=[slice_idx],
        )
        st.plotly_chart(fig, use_container_width=True)

        # Field statistics for this slice
        slice_2d = np.take(field_data, slice_idx, axis=axis_map[axis])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Min", f"{slice_2d.min():.4f}")
        c2.metric("Max", f"{slice_2d.max():.4f}")
        c3.metric("Mean", f"{slice_2d.mean():.4f}")
        c4.metric("Std", f"{slice_2d.std():.4f}")

    else:  # Multi-Field
        selected = st.multiselect(
            "Fields to Compare",
            list(fields.keys()),
            default=list(fields.keys())[:3],
            key="multi_fields",
        )
        if selected:
            sub_fields = {k: fields[k] for k in selected if k in fields}
            fig = renderer.render_multi_field(sub_fields)
            st.plotly_chart(fig, use_container_width=True)

    # Volume stats
    with st.expander("Volume Statistics"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"{field_name} Min", f"{field_data.min():.4f}")
        c2.metric(f"{field_name} Max", f"{field_data.max():.4f}")
        c3.metric(f"{field_name} Mean", f"{field_data.mean():.4f}")
        c4.metric("Grid Shape", f"{field_data.shape}")


# ────────────────────────────────────────────────────────────────
# TAB 3: Interaction Network
# ────────────────────────────────────────────────────────────────

with tab_network:
    st.subheader("Cell Interaction Network")

    try:
        from cognisom.visualization.network_renderer import InteractionNetworkRenderer
        from cognisom.visualization.cell_renderer import CellPopulationRenderer
    except ImportError:
        st.error("Network renderer not available.")
        st.stop()

    # Use cells from Tab 1 or generate
    cells = st.session_state.get("viz_cells")
    if not cells:
        cells = CellPopulationRenderer.generate_demo_cells()
        st.session_state["viz_cells"] = cells

    net_renderer = InteractionNetworkRenderer()

    # Generate interactions
    col1, col2 = st.columns(2)
    n_interactions = col1.slider("Number of Interactions", 5, 100, 30, key="n_interact")
    layout_mode = col2.selectbox("Layout", ["spatial", "force"], key="net_layout")

    interactions = net_renderer.generate_demo_interactions(cells, n_interactions=n_interactions)
    st.session_state["viz_interactions"] = interactions

    # Network graph
    fig = net_renderer.render_network(
        cells, interactions,
        layout=layout_mode,
        title=f"Interaction Network ({len(interactions)} edges)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Interaction type breakdown
    with st.expander("Interaction Statistics"):
        type_counts = {}
        for inter in interactions:
            itype = inter.get("type", "unknown")
            type_counts[itype] = type_counts.get(itype, 0) + 1

        cols = st.columns(min(len(type_counts), 6))
        for i, (itype, cnt) in enumerate(sorted(type_counts.items(), key=lambda x: -x[1])):
            color = InteractionNetworkRenderer.EDGE_COLORS.get(itype, "#999")
            cols[i % len(cols)].metric(
                itype.replace("_", " ").title(),
                cnt,
            )

        strengths = [i.get("strength", 0.5) for i in interactions]
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean Strength", f"{np.mean(strengths):.3f}")
        c2.metric("Max Strength", f"{np.max(strengths):.3f}")
        c3.metric("Min Strength", f"{np.min(strengths):.3f}")

    # Contact map
    st.divider()
    st.subheader("Cell Contact Map")

    dist_threshold = st.slider(
        "Contact Distance (um)", 10.0, 100.0, 30.0, key="contact_dist",
    )

    fig_contact = net_renderer.render_contact_map(
        cells, distance_threshold=dist_threshold,
    )
    st.plotly_chart(fig_contact, use_container_width=True)


# ────────────────────────────────────────────────────────────────
# TAB 4: Scientific Inspector (time scrubber + cell picker)
# ────────────────────────────────────────────────────────────────

with tab_inspect:
    st.subheader("Scientific Inspector")
    st.caption("Time-resolved playback and per-cell deep inspection")

    # ── Time Scrubber ─────────────────────────────────────────
    st.markdown("#### Time Scrubber")
    st.write("Replay simulation history frame-by-frame to inspect "
             "spatial and metabolic dynamics over time.")

    # Generate or load time series data
    if "sim_history" not in st.session_state:
        # Generate synthetic time-series for demo
        n_frames = 48  # 48 hours
        n_cells_demo = 80
        rng = np.random.RandomState(42)

        history = []
        for t in range(n_frames):
            frame_cells = []
            n_alive = n_cells_demo + t * 2  # growing population
            for i in range(n_alive):
                ct = "cancer" if i < 20 + t else ("immune" if i < 30 + t else "normal")
                frame_cells.append({
                    "cell_id": i,
                    "position": (
                        float(rng.uniform(0, 200)),
                        float(rng.uniform(0, 200)),
                        float(rng.uniform(0, 100)),
                    ),
                    "cell_type": ct,
                    "phase": rng.choice(["G1", "S", "G2", "M"]),
                    "alive": bool(rng.random() > 0.05),
                    "oxygen": float(max(0, 0.21 - 0.002 * t + rng.normal(0, 0.01))),
                    "glucose": float(max(0, 5.0 - 0.05 * t + rng.normal(0, 0.2))),
                    "atp": float(max(0, 1000 - 10 * t + rng.normal(0, 50))),
                    "age": float(t * 0.5 + rng.uniform(0, 5)),
                    "volume": float(1.0 + rng.uniform(-0.2, 0.3)),
                    "parent_id": max(0, i - n_cells_demo) if i >= n_cells_demo else -1,
                })
            history.append({
                "time": float(t),
                "cells": frame_cells,
                "n_alive": sum(1 for c in frame_cells if c.get("alive", True)),
                "n_cancer": sum(1 for c in frame_cells if c["cell_type"] == "cancer"),
                "n_immune": sum(1 for c in frame_cells if c["cell_type"] == "immune"),
                "n_normal": sum(1 for c in frame_cells if c["cell_type"] == "normal"),
                "mean_oxygen": float(np.mean([c["oxygen"] for c in frame_cells])),
                "mean_glucose": float(np.mean([c["glucose"] for c in frame_cells])),
                "mean_atp": float(np.mean([c["atp"] for c in frame_cells])),
            })
        st.session_state["sim_history"] = history

    history = st.session_state["sim_history"]
    n_frames = len(history)

    # Time controls
    col_play, col_slider, col_speed = st.columns([1, 6, 2])
    with col_play:
        st.write("")  # spacer
        is_playing = st.checkbox("Auto-play", value=False, key="time_play")
    with col_slider:
        frame_idx = st.slider(
            "Time (hours)",
            0, n_frames - 1,
            value=0,
            key="time_frame",
            format="t=%d h",
        )
    with col_speed:
        playback_speed = st.selectbox("Speed", [1, 2, 5, 10], index=1, key="play_speed")

    frame = history[frame_idx]

    # Population curves
    st.markdown("##### Population Dynamics")
    col_chart, col_metrics = st.columns([3, 1])

    with col_chart:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig_pop = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=("Cell Counts", "Metabolic Means"),
            vertical_spacing=0.12,
        )

        times = [h["time"] for h in history]
        fig_pop.add_trace(go.Scatter(
            x=times, y=[h["n_cancer"] for h in history],
            name="Cancer", line=dict(color="#e74c3c"),
        ), row=1, col=1)
        fig_pop.add_trace(go.Scatter(
            x=times, y=[h["n_immune"] for h in history],
            name="Immune", line=dict(color="#2ecc71"),
        ), row=1, col=1)
        fig_pop.add_trace(go.Scatter(
            x=times, y=[h["n_normal"] for h in history],
            name="Normal", line=dict(color="#3498db"),
        ), row=1, col=1)

        # Metabolic
        fig_pop.add_trace(go.Scatter(
            x=times, y=[h["mean_oxygen"] for h in history],
            name="O2", line=dict(color="#e67e22"),
        ), row=2, col=1)
        fig_pop.add_trace(go.Scatter(
            x=times, y=[h["mean_glucose"] for h in history],
            name="Glucose", line=dict(color="#9b59b6"),
        ), row=2, col=1)

        # Current time marker
        fig_pop.add_vline(x=frame["time"], line_dash="dash", line_color="red")

        fig_pop.update_layout(height=400, showlegend=True, margin=dict(t=40, b=20))
        fig_pop.update_xaxes(title_text="Time (hours)", row=2, col=1)
        fig_pop.update_yaxes(title_text="Count", row=1, col=1)
        fig_pop.update_yaxes(title_text="Concentration", row=2, col=1)
        st.plotly_chart(fig_pop, use_container_width=True)

    with col_metrics:
        st.metric("Time", f"{frame['time']:.0f} h")
        st.metric("Alive", frame["n_alive"])
        st.metric("Cancer", frame["n_cancer"])
        st.metric("Immune", frame["n_immune"])
        st.metric("Mean O2", f"{frame['mean_oxygen']:.4f}")
        st.metric("Mean ATP", f"{frame['mean_atp']:.0f}")

    # ── Cell Picker ───────────────────────────────────────────
    st.divider()
    st.markdown("#### Cell Picker")
    st.write("Select a cell to inspect its full state at the current time point.")

    frame_cells = frame["cells"]
    cell_ids = [c["cell_id"] for c in frame_cells]

    col_pick, col_filter = st.columns([2, 2])
    with col_filter:
        type_filter = st.selectbox(
            "Filter by Type",
            ["All"] + sorted(set(c["cell_type"] for c in frame_cells)),
            key="picker_filter",
        )
    with col_pick:
        if type_filter != "All":
            filtered = [c for c in frame_cells if c["cell_type"] == type_filter]
        else:
            filtered = frame_cells
        filtered_ids = [c["cell_id"] for c in filtered]
        picked_id = st.selectbox(
            "Cell ID",
            filtered_ids,
            key="picked_cell",
        )

    # Find picked cell
    picked = next((c for c in frame_cells if c["cell_id"] == picked_id), None)

    if picked:
        col_state, col_pos, col_meta = st.columns(3)

        with col_state:
            st.markdown("**Identity & Phase**")
            st.write(f"- **ID**: {picked['cell_id']}")
            st.write(f"- **Type**: {picked['cell_type']}")
            st.write(f"- **Phase**: {picked['phase']}")
            st.write(f"- **Alive**: {picked['alive']}")
            st.write(f"- **Age**: {picked['age']:.1f} h")
            if picked.get("parent_id", -1) >= 0:
                st.write(f"- **Parent**: Cell {picked['parent_id']}")

        with col_pos:
            st.markdown("**Position & Morphology**")
            pos = picked["position"]
            st.write(f"- **X**: {pos[0]:.1f} um")
            st.write(f"- **Y**: {pos[1]:.1f} um")
            st.write(f"- **Z**: {pos[2]:.1f} um")
            st.write(f"- **Volume**: {picked.get('volume', 1.0):.2f}")

        with col_meta:
            st.markdown("**Metabolic State**")
            o2 = picked.get("oxygen", 0)
            gluc = picked.get("glucose", 0)
            atp = picked.get("atp", 0)
            st.write(f"- **O2**: {o2:.4f}")
            st.write(f"- **Glucose**: {gluc:.2f} mM")
            st.write(f"- **ATP**: {atp:.0f}")

            # Health indicators
            if o2 < 0.02:
                st.warning("Severely hypoxic")
            elif o2 < 0.05:
                st.warning("Hypoxic")
            if atp < 100:
                st.warning("ATP critically low")

        # Cell history across time
        st.markdown("##### Cell History Across Time")
        cell_trace_o2 = []
        cell_trace_atp = []
        cell_trace_times = []
        for h in history:
            match = next((c for c in h["cells"] if c["cell_id"] == picked_id), None)
            if match:
                cell_trace_times.append(h["time"])
                cell_trace_o2.append(match.get("oxygen", 0))
                cell_trace_atp.append(match.get("atp", 0))

        if cell_trace_times:
            fig_cell = make_subplots(
                rows=1, cols=2,
                subplot_titles=(f"Cell {picked_id} O2", f"Cell {picked_id} ATP"),
            )
            fig_cell.add_trace(go.Scatter(
                x=cell_trace_times, y=cell_trace_o2,
                mode="lines", name="O2", line=dict(color="#e67e22"),
            ), row=1, col=1)
            fig_cell.add_trace(go.Scatter(
                x=cell_trace_times, y=cell_trace_atp,
                mode="lines", name="ATP", line=dict(color="#2ecc71"),
            ), row=1, col=2)
            fig_cell.add_vline(x=frame["time"], line_dash="dash", line_color="red")
            fig_cell.update_layout(height=250, showlegend=False, margin=dict(t=30, b=20))
            st.plotly_chart(fig_cell, use_container_width=True)


# ────────────────────────────────────────────────────────────────
# TAB 5: Lineage Tree
# ────────────────────────────────────────────────────────────────

with tab_lineage:
    st.subheader("Cell Lineage Tree")
    st.caption("Track cell division history and clonal evolution")

    import plotly.graph_objects as go

    # Generate demo lineage data if not available
    if "lineage_tree" not in st.session_state:
        rng = np.random.RandomState(123)

        # Build a division tree: each node is a cell
        nodes = []
        edges = []

        # Root cells (generation 0)
        for i in range(5):
            ct = "cancer" if i < 2 else "normal"
            nodes.append({
                "cell_id": i,
                "parent_id": -1,
                "generation": 0,
                "birth_time": 0.0,
                "death_time": None,
                "cell_type": ct,
                "n_divisions": 0,
                "mutations": [],
            })

        next_id = 5
        # Simulate divisions over 48 hours
        for t in range(1, 49):
            new_nodes = []
            for node in nodes:
                if node["death_time"] is not None:
                    continue
                # Division probability depends on type
                p_div = 0.08 if node["cell_type"] == "cancer" else 0.03
                if rng.random() < p_div and node["n_divisions"] < 5:
                    # Create daughter cell
                    daughter = {
                        "cell_id": next_id,
                        "parent_id": node["cell_id"],
                        "generation": node["generation"] + 1,
                        "birth_time": float(t),
                        "death_time": None,
                        "cell_type": node["cell_type"],
                        "n_divisions": 0,
                        "mutations": list(node["mutations"]),
                    }
                    # Occasionally acquire mutation
                    if rng.random() < 0.15 and node["cell_type"] == "cancer":
                        muts = ["TP53_R175H", "PTEN_loss", "AR_V7", "MYC_amp",
                                "BRCA2_del", "ERG_fusion", "SPOP_F133L"]
                        new_mut = rng.choice(muts)
                        if new_mut not in daughter["mutations"]:
                            daughter["mutations"].append(new_mut)

                    edges.append((node["cell_id"], next_id))
                    new_nodes.append(daughter)
                    node["n_divisions"] += 1
                    next_id += 1

                # Death probability
                p_death = 0.01 if node["cell_type"] == "cancer" else 0.02
                if rng.random() < p_death:
                    node["death_time"] = float(t)

            nodes.extend(new_nodes)

        st.session_state["lineage_tree"] = {"nodes": nodes, "edges": edges}

    tree_data = st.session_state["lineage_tree"]
    nodes = tree_data["nodes"]
    edges = tree_data["edges"]

    # Stats
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Cells", len(nodes))
    c2.metric("Divisions", len(edges))
    max_gen = max(n["generation"] for n in nodes)
    c3.metric("Max Generation", max_gen)
    alive = sum(1 for n in nodes if n["death_time"] is None)
    c4.metric("Alive", alive)
    with_muts = sum(1 for n in nodes if n["mutations"])
    c5.metric("With Mutations", with_muts)

    # ── Tree visualization ────────────────────────────────────
    st.markdown("#### Division Tree")

    col_filt, col_gen = st.columns(2)
    with col_filt:
        lineage_type = st.selectbox(
            "Filter Cell Type",
            ["All", "cancer", "normal"],
            key="lineage_type",
        )
    with col_gen:
        max_gen_show = st.slider(
            "Max Generation to Show",
            0, max_gen, max_gen,
            key="lineage_max_gen",
        )

    # Filter nodes
    show_nodes = [n for n in nodes if n["generation"] <= max_gen_show]
    if lineage_type != "All":
        show_nodes = [n for n in show_nodes if n["cell_type"] == lineage_type]
    show_ids = {n["cell_id"] for n in show_nodes}
    show_edges = [(p, c) for p, c in edges if p in show_ids and c in show_ids]

    # Layout: x = birth_time, y = generation (inverted for tree)
    node_map = {n["cell_id"]: n for n in show_nodes}

    # Assign y positions to avoid overlap within same generation
    gen_counts = {}
    for n in show_nodes:
        g = n["generation"]
        gen_counts[g] = gen_counts.get(g, 0)
        n["_y_pos"] = gen_counts[g]
        gen_counts[g] += 1

    # Normalize y positions per generation
    for n in show_nodes:
        g = n["generation"]
        total = gen_counts.get(g, 1)
        n["_y_norm"] = (n["_y_pos"] - total / 2) * 2

    # Draw tree
    fig_tree = go.Figure()

    # Edges
    for parent_id, child_id in show_edges:
        if parent_id in node_map and child_id in node_map:
            p = node_map[parent_id]
            c = node_map[child_id]
            fig_tree.add_trace(go.Scatter(
                x=[p["birth_time"], c["birth_time"]],
                y=[p["_y_norm"], c["_y_norm"]],
                mode="lines",
                line=dict(color="#ccc", width=0.5),
                showlegend=False,
                hoverinfo="skip",
            ))

    # Nodes by type
    type_colors = {"cancer": "#e74c3c", "normal": "#3498db", "immune": "#2ecc71"}
    for ct, color in type_colors.items():
        ct_nodes = [n for n in show_nodes if n["cell_type"] == ct]
        if not ct_nodes:
            continue

        marker_colors = []
        symbols = []
        for n in ct_nodes:
            if n["death_time"] is not None:
                marker_colors.append("#999")
                symbols.append("x")
            elif n["mutations"]:
                marker_colors.append("#f39c12")  # mutation = orange
                symbols.append("diamond")
            else:
                marker_colors.append(color)
                symbols.append("circle")

        fig_tree.add_trace(go.Scatter(
            x=[n["birth_time"] for n in ct_nodes],
            y=[n["_y_norm"] for n in ct_nodes],
            mode="markers",
            marker=dict(
                size=8,
                color=marker_colors,
                symbol=symbols,
                line=dict(width=1, color="#333"),
            ),
            name=ct.title(),
            text=[
                f"Cell {n['cell_id']}<br>"
                f"Gen {n['generation']}<br>"
                f"Type: {n['cell_type']}<br>"
                f"Born: {n['birth_time']:.0f}h<br>"
                f"{'Dead: ' + str(n['death_time']) + 'h' if n['death_time'] else 'Alive'}<br>"
                f"Mutations: {', '.join(n['mutations']) if n['mutations'] else 'none'}"
                for n in ct_nodes
            ],
            hovertemplate="%{text}<extra></extra>",
        ))

    fig_tree.update_layout(
        height=500,
        xaxis_title="Time (hours)",
        yaxis_title="Lineage Position",
        title="Cell Division Lineage",
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig_tree, use_container_width=True)

    # ── Mutation tracker ──────────────────────────────────────
    st.divider()
    st.markdown("#### Clonal Evolution — Mutation Tracker")

    all_mutations = set()
    for n in nodes:
        all_mutations.update(n.get("mutations", []))

    if all_mutations:
        # Mutation frequency over time
        mut_time = {}
        for n in nodes:
            for m in n.get("mutations", []):
                t = n["birth_time"]
                if m not in mut_time:
                    mut_time[m] = []
                mut_time[m].append(t)

        fig_mut = go.Figure()
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22"]
        for i, (mut, times) in enumerate(sorted(mut_time.items())):
            # Cumulative count over time
            times_sorted = sorted(times)
            cumulative = list(range(1, len(times_sorted) + 1))
            fig_mut.add_trace(go.Scatter(
                x=times_sorted, y=cumulative,
                mode="lines+markers",
                name=mut,
                line=dict(color=colors[i % len(colors)]),
                marker=dict(size=4),
            ))

        fig_mut.update_layout(
            height=350,
            xaxis_title="Time (hours)",
            yaxis_title="Cumulative Cells with Mutation",
            title="Mutation Expansion Over Time",
            margin=dict(t=40, b=30),
        )
        st.plotly_chart(fig_mut, use_container_width=True)

        # Mutation table
        with st.expander("Mutation Details"):
            for mut in sorted(all_mutations):
                carriers = [n for n in nodes if mut in n.get("mutations", [])]
                alive_carriers = [n for n in carriers if n["death_time"] is None]
                st.write(
                    f"- **{mut}**: {len(carriers)} total cells, "
                    f"{len(alive_carriers)} alive, "
                    f"first appeared at t={min(n['birth_time'] for n in carriers):.0f}h"
                )
    else:
        st.info("No mutations recorded in this lineage.")


# ────────────────────────────────────────────────────────────────
# TAB 6: Omniverse/USD (REAL 3D)
# ────────────────────────────────────────────────────────────────

with tab_omniverse:
    st.subheader("NVIDIA Omniverse / OpenUSD")
    st.caption("Real 3D scene generation using OpenUSD - No mocks, actual USD files")

    # Check USD availability
    try:
        from pxr import Usd, UsdGeom, Gf
        USD_AVAILABLE = True
    except ImportError:
        USD_AVAILABLE = False

    if not USD_AVAILABLE:
        st.error("OpenUSD not installed. Install with: `pip install usd-core`")
        st.code("pip install usd-core", language="bash")
        st.stop()

    st.success("OpenUSD (pxr) is available - Real USD operations enabled")

    # Import real connector
    try:
        from cognisom.omniverse.real_connector import (
            RealOmniverseConnector, CellVisualization, SimulationFrame
        )
        CONNECTOR_AVAILABLE = True
    except ImportError as e:
        st.error(f"Real connector not available: {e}")
        CONNECTOR_AVAILABLE = False

    if CONNECTOR_AVAILABLE:
        st.divider()

        # ── Scene Generation ────────────────────────────────────────
        st.markdown("### Generate USD Scene")

        col_cfg, col_gen = st.columns([2, 1])

        with col_cfg:
            scene_name = st.text_input("Scene Name", value="cognisom_simulation", key="usd_scene_name")

            st.markdown("**Cell Configuration**")
            col_a, col_b, col_c = st.columns(3)
            n_stem = col_a.slider("Stem Cells", 5, 50, 15, key="usd_n_stem")
            n_prog = col_b.slider("Progenitor", 10, 100, 30, key="usd_n_prog")
            n_diff = col_c.slider("Differentiated", 10, 100, 40, key="usd_n_diff")
            n_div = col_a.slider("Dividing", 2, 30, 8, key="usd_n_div")

            cluster_radius = col_b.slider("Cluster Radius", 10, 50, 25, key="usd_radius")

        with col_gen:
            st.markdown("**Output Directory**")
            output_dir = st.text_input("Path", value="data/simulation/usd", key="usd_output")

            generate_btn = st.button("Generate USD Scene", type="primary", key="gen_usd")

        if generate_btn:
            import math
            import random

            with st.spinner("Creating real USD stage..."):
                # Initialize connector
                connector = RealOmniverseConnector(output_dir)

                if connector.create_stage(scene_name):
                    # Cell type configurations
                    cell_types = {
                        "stem": {"color": (0.2, 0.9, 0.3), "count": n_stem},
                        "progenitor": {"color": (0.3, 0.6, 0.9), "count": n_prog},
                        "differentiated": {"color": (0.9, 0.5, 0.2), "count": n_diff},
                        "dividing": {"color": (0.9, 0.2, 0.9), "count": n_div},
                    }

                    random.seed(42)
                    total_cells = 0

                    for cell_type, cfg in cell_types.items():
                        for i in range(cfg["count"]):
                            # Spherical distribution
                            theta = random.uniform(0, 2 * math.pi)
                            phi = random.uniform(0, math.pi)
                            r = random.uniform(3, cluster_radius)

                            x = r * math.sin(phi) * math.cos(theta)
                            y = r * math.sin(phi) * math.sin(theta) + 10
                            z = r * math.cos(phi)

                            cell = CellVisualization(
                                cell_id=f"{cell_type}_{i:03d}",
                                position=(x, y, z),
                                radius=random.uniform(0.8, 2.0),
                                color=cfg["color"],
                                cell_type=cell_type,
                                metabolic_state=random.uniform(0.5, 1.0),
                            )
                            connector.add_cell(cell)
                            total_cells += 1

                    connector.save()

                    st.success(f"Created USD scene with {total_cells} cells")

                    # Store connector info
                    st.session_state["usd_connector"] = connector.get_info()
                    st.session_state["usd_stage_path"] = str(connector.stage_path)

                else:
                    st.error("Failed to create USD stage")

        # ── Display Current Scene ───────────────────────────────────
        if "usd_stage_path" in st.session_state:
            st.divider()
            st.markdown("### Current USD Scene")

            stage_path = st.session_state["usd_stage_path"]
            connector_info = st.session_state.get("usd_connector", {})

            col_info, col_actions = st.columns([2, 1])

            with col_info:
                st.write(f"**File:** `{stage_path}`")
                st.write(f"**Cells:** {connector_info.get('cell_count', 'N/A')}")
                st.write(f"**Frames:** {connector_info.get('frame_count', 'N/A')}")
                st.write(f"**Real USD:** {connector_info.get('is_real', False)}")

            with col_actions:
                # Download USD file
                from pathlib import Path
                usd_path = Path(stage_path)
                if usd_path.exists():
                    with open(usd_path, 'r') as f:
                        usd_content = f.read()
                    st.download_button(
                        "Download .usda",
                        usd_content,
                        file_name=usd_path.name,
                        mime="text/plain",
                        key="dl_usda",
                    )

            # Preview USD content
            with st.expander("Preview USD File (first 100 lines)"):
                from pathlib import Path
                usd_path = Path(stage_path)
                if usd_path.exists():
                    with open(usd_path, 'r') as f:
                        lines = f.readlines()[:100]
                    st.code("".join(lines), language="python")
                else:
                    st.warning("USD file not found")

        # ── Viewer Options ──────────────────────────────────────────
        st.divider()
        st.markdown("### View USD Files")

        st.markdown("""
        Your generated USD files can be viewed in:

        | Application | Platform | Notes |
        |-------------|----------|-------|
        | **NVIDIA Omniverse** | Windows/Linux | Full Omniverse experience |
        | **usdview** | All | `pip install usd-core` then `usdview file.usda` |
        | **Blender** | All | File > Import > USD |
        | **Houdini** | All | Native USD support |
        | **Maya** | All | With USD plugin |
        | **Three.js** | Web | Using USD loader |

        **Quick View Command:**
        ```bash
        # If you have usd-core installed
        python -m pxr.Usdviewq data/simulation/usd/cognisom_simulation.usda
        ```
        """)

        # ── Animation Generation ────────────────────────────────────
        st.divider()
        st.markdown("### Generate Animation Sequence")

        n_frames = st.slider("Number of Frames", 10, 100, 24, key="usd_n_frames")

        if st.button("Generate Animated Sequence", key="gen_anim"):
            import math
            import random

            with st.spinner(f"Generating {n_frames} frame animation..."):
                connector = RealOmniverseConnector(output_dir)

                if connector.create_stage(f"{scene_name}_animated"):
                    random.seed(42)

                    # Create initial cells
                    cells_data = []
                    for i in range(50):
                        cell_type = random.choice(["stem", "progenitor", "differentiated"])
                        colors = {
                            "stem": (0.2, 0.9, 0.3),
                            "progenitor": (0.3, 0.6, 0.9),
                            "differentiated": (0.9, 0.5, 0.2),
                        }

                        theta = random.uniform(0, 2 * math.pi)
                        phi = random.uniform(0, math.pi)
                        r = random.uniform(5, 20)

                        cells_data.append({
                            "id": f"cell_{i:03d}",
                            "base_pos": (
                                r * math.sin(phi) * math.cos(theta),
                                r * math.sin(phi) * math.sin(theta) + 10,
                                r * math.cos(phi)
                            ),
                            "color": colors[cell_type],
                            "cell_type": cell_type,
                            "phase": random.uniform(0, 2 * math.pi),
                        })

                    # Generate frames
                    progress = st.progress(0)
                    for frame in range(n_frames):
                        cells = []
                        for cd in cells_data:
                            # Animate position (oscillation)
                            t = frame / n_frames * 2 * math.pi
                            offset = math.sin(t + cd["phase"]) * 2

                            pos = (
                                cd["base_pos"][0] + offset * 0.5,
                                cd["base_pos"][1] + offset,
                                cd["base_pos"][2] + offset * 0.3,
                            )

                            cell = CellVisualization(
                                cell_id=cd["id"],
                                position=pos,
                                radius=1.0 + math.sin(t + cd["phase"]) * 0.2,
                                color=cd["color"],
                                cell_type=cd["cell_type"],
                                metabolic_state=0.5 + 0.5 * math.sin(t + cd["phase"]),
                            )
                            cells.append(cell)

                        sim_frame = SimulationFrame(
                            timestamp=frame / 24.0,  # 24 fps
                            cells=cells,
                        )
                        connector.render_frame(sim_frame)
                        progress.progress((frame + 1) / n_frames)

                    connector.save()
                    st.success(f"Created animated USD with {n_frames} frames at {connector.stage_path}")
                    st.session_state["usd_anim_path"] = str(connector.stage_path)

        if "usd_anim_path" in st.session_state:
            st.write(f"**Animated Scene:** `{st.session_state['usd_anim_path']}`")
            st.info("Open in Omniverse or usdview to play the animation timeline")


# ────────────────────────────────────────────────────────────────
# TAB 7: Export
# ────────────────────────────────────────────────────────────────

with tab_export:
    st.subheader("Export 3D Data")
    st.write("Export cell populations and spatial fields to standard formats for use in external tools.")

    try:
        from cognisom.visualization.exporters import SceneExporter
    except ImportError:
        st.error("Exporter not available.")
        st.stop()

    exporter = SceneExporter()

    cells = st.session_state.get("viz_cells")
    fields = st.session_state.get("viz_fields")
    interactions = st.session_state.get("viz_interactions")

    if not cells:
        st.info("Generate data in the other tabs first.")
        st.stop()

    st.write(f"**Available data:** {len(cells)} cells"
             + (f", {len(fields)} fields" if fields else "")
             + (f", {len(interactions)} interactions" if interactions else ""))

    st.divider()

    # ── Cell Export ──────────────────────────────────────────────
    st.subheader("Cell Population Export")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.write("**PDB Format**")
        st.caption("Open in PyMOL, UCSF Chimera")
        if st.button("Export PDB", key="exp_pdb"):
            path = Path(tempfile.mkdtemp()) / "cells.pdb"
            exporter.cells_to_pdb(cells, str(path), scale=10.0)
            with open(path) as f:
                st.download_button(
                    "Download .pdb",
                    f.read(),
                    file_name="cognisom_cells.pdb",
                    mime="chemical/x-pdb",
                    key="dl_pdb",
                )

    with col2:
        st.write("**glTF Format**")
        st.caption("Open in Blender, three.js")
        if st.button("Export glTF", key="exp_gltf"):
            path = Path(tempfile.mkdtemp()) / "cells.gltf"
            exporter.cells_to_gltf(cells, str(path))
            with open(path) as f:
                st.download_button(
                    "Download .gltf",
                    f.read(),
                    file_name="cognisom_cells.gltf",
                    mime="model/gltf+json",
                    key="dl_gltf",
                )

    with col3:
        st.write("**CSV Format**")
        st.caption("Open in Excel, pandas")
        if st.button("Export CSV", key="exp_csv"):
            path = Path(tempfile.mkdtemp()) / "cells.csv"
            exporter.cells_to_csv(cells, str(path))
            with open(path) as f:
                st.download_button(
                    "Download .csv",
                    f.read(),
                    file_name="cognisom_cells.csv",
                    mime="text/csv",
                    key="dl_csv",
                )

    with col4:
        st.write("**JSON Scene**")
        st.caption("Full scene with all data")
        if st.button("Export JSON", key="exp_json"):
            path = Path(tempfile.mkdtemp()) / "scene.json"
            exporter.scene_to_json(
                cells=cells,
                fields=fields,
                interactions=interactions,
                metadata={"source": "Cognisom Dashboard"},
                output_path=str(path),
            )
            with open(path) as f:
                st.download_button(
                    "Download .json",
                    f.read(),
                    file_name="cognisom_scene.json",
                    mime="application/json",
                    key="dl_json",
                )

    # ── Field Export ─────────────────────────────────────────────
    if fields:
        st.divider()
        st.subheader("Spatial Field Export")

        col1, col2 = st.columns(2)

        with col1:
            field_to_export = st.selectbox(
                "Field", list(fields.keys()), key="export_field",
            )
            st.write("**VTK Format** — Open in ParaView, VisIt")
            if st.button("Export Single Field VTK", key="exp_vtk_single"):
                path = Path(tempfile.mkdtemp()) / f"{field_to_export}.vtk"
                exporter.field_to_vtk(
                    fields[field_to_export], str(path),
                    field_name=field_to_export,
                )
                with open(path) as f:
                    st.download_button(
                        "Download .vtk",
                        f.read(),
                        file_name=f"cognisom_{field_to_export}.vtk",
                        mime="application/octet-stream",
                        key="dl_vtk_single",
                    )

        with col2:
            st.write("**Multi-Field VTK** — All fields in one file")
            if st.button("Export All Fields VTK", key="exp_vtk_multi"):
                path = Path(tempfile.mkdtemp()) / "all_fields.vtk"
                exporter.fields_to_vtk(fields, str(path))
                with open(path) as f:
                    st.download_button(
                        "Download .vtk",
                        f.read(),
                        file_name="cognisom_all_fields.vtk",
                        mime="application/octet-stream",
                        key="dl_vtk_multi",
                    )

# Footer
from cognisom.dashboard.footer import render_footer
render_footer()
