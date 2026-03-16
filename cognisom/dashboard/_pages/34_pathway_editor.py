"""
Page 34: Interactive Pathway Editor
=====================================

Visual graph editor for biological signaling pathways. Select entities,
wire them together via interactions, and watch the ODE simulation update
in real-time. A visual systems biology workbench.

Features:
- Drag-and-drop entity selection from library
- Visual pathway graph (Plotly network)
- Real-time ODE simulation from entity interactions
- Parameter adjustment with live feedback
"""

import streamlit as st
from cognisom.dashboard.page_config import safe_set_page_config

safe_set_page_config(page_title="Pathway Editor | Cognisom", page_icon="\U0001f310", layout="wide")

from cognisom.auth.middleware import streamlit_page_gate
user = streamlit_page_gate("34_pathway_editor")

import numpy as np
import plotly.graph_objects as go
import logging

logger = logging.getLogger(__name__)

st.title("Interactive Pathway Editor")
st.caption("Build signaling pathways from entity library and simulate their dynamics.")

# ── Entity Selection ─────────────────────────────────────────────

from cognisom.library.store import EntityStore

@st.cache_resource
def _get_store():
    return EntityStore()

store = _get_store()

st.subheader("1. Select Entities")

# Preset pathways
preset = st.selectbox("Load Preset Pathway", [
    "Custom (select entities below)",
    "AR Signaling (AR → DHT → PSA)",
    "PI3K/AKT/mTOR (PTEN → PIK3CA → AKT1 → mTOR)",
    "p53 Pathway (TP53 → MDM2 → BAX)",
    "Immune Checkpoint (PD-1 → CD8+ T cell)",
    "Apoptosis (BCL-2 → BAX → Caspase-3 → Cytochrome c)",
])

# Map presets to entity names
_PRESETS = {
    "AR Signaling (AR → DHT → PSA)": ["AR", "DHT", "PSA", "FOXA1"],
    "PI3K/AKT/mTOR (PTEN → PIK3CA → AKT1 → mTOR)": ["PTEN", "PIK3CA", "AKT1", "mTOR"],
    "p53 Pathway (TP53 → MDM2 → BAX)": ["TP53", "MDM2", "BAX", "Caspase-3"],
    "Immune Checkpoint (PD-1 → CD8+ T cell)": ["PD-1", "CTLA-4 (CD152)", "Pembrolizumab"],
    "Apoptosis (BCL-2 → BAX → Caspase-3 → Cytochrome c)": ["BCL-2", "BAX", "Caspase-3", "Cytochrome c"],
}

if preset != "Custom (select entities below)" and preset in _PRESETS:
    selected_names = _PRESETS[preset]
else:
    # Manual entity selection
    all_entities, _ = store.search(limit=300)
    entity_names = sorted(set(e.name for e in all_entities))
    selected_names = st.multiselect(
        "Select entities to include in pathway",
        entity_names,
        default=["AR", "DHT", "PSA"],
        max_selections=10,
    )

if len(selected_names) < 2:
    st.info("Select at least 2 entities to build a pathway.")
    st.stop()

st.divider()

# ── Build Pathway Graph ──────────────────────────────────────────

st.subheader("2. Pathway Graph")

# Collect entities and their interactions
entities = []
for name in selected_names:
    e = store.find_entity_by_name(name, None)
    if e:
        entities.append(e)

if not entities:
    st.error("No entities found.")
    st.stop()

# Build nodes and edges
nodes = {}
edges = []
for i, e in enumerate(entities):
    nodes[e.name] = {
        "index": i,
        "type": e.entity_type.value,
        "color": e.color_rgb if hasattr(e, "color_rgb") and e.color_rgb else [0.5, 0.5, 0.5],
    }

    # Find interactions with other selected entities
    interactions = e.interacts_with if hasattr(e, "interacts_with") and e.interacts_with else []
    for inter in interactions:
        target = inter.get("target", "")
        if target in nodes or target in selected_names:
            edges.append({
                "source": e.name,
                "target": target,
                "type": inter.get("type", "unknown"),
                "kd": inter.get("kd_nm", inter.get("kd_um", "")),
                "note": inter.get("note", ""),
            })

# Layout: circular
n = len(entities)
angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
pos_x = np.cos(angles) * 2
pos_y = np.sin(angles) * 2
positions = {e.name: (pos_x[i], pos_y[i]) for i, e in enumerate(entities)}

# Draw graph
fig = go.Figure()

# Edges
for edge in edges:
    if edge["source"] in positions and edge["target"] in positions:
        x0, y0 = positions[edge["source"]]
        x1, y1 = positions[edge["target"]]

        edge_color = {
            "activates": "rgb(50,200,50)",
            "inhibits": "rgb(200,50,50)",
            "binds_to": "rgb(100,100,200)",
            "phosphorylates": "rgb(200,200,50)",
            "catalyzes": "rgb(50,200,200)",
        }.get(edge["type"], "rgb(150,150,150)")

        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode="lines",
            line=dict(color=edge_color, width=2),
            hoverinfo="text",
            text=f"{edge['source']} → {edge['type']} → {edge['target']}"
                 + (f" (Kd: {edge['kd']} nM)" if edge['kd'] else ""),
            showlegend=False,
        ))
        # Arrow head
        mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
        fig.add_trace(go.Scatter(
            x=[mid_x], y=[mid_y],
            mode="markers+text",
            marker=dict(size=8, color=edge_color, symbol="arrow-right"),
            text=edge["type"],
            textposition="top center",
            textfont=dict(size=8, color=edge_color),
            showlegend=False,
        ))

# Nodes
for name, info in nodes.items():
    x, y = positions.get(name, (0, 0))
    r, g, b = [int(c * 255) for c in info["color"][:3]]

    fig.add_trace(go.Scatter(
        x=[x], y=[y],
        mode="markers+text",
        marker=dict(
            size=30,
            color=f"rgb({r},{g},{b})",
            line=dict(width=2, color="white"),
        ),
        text=name,
        textposition="bottom center",
        textfont=dict(size=11, color="white"),
        hoverinfo="text",
        hovertext=f"{name} ({info['type']})",
        showlegend=False,
    ))

fig.update_layout(
    height=400,
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    plot_bgcolor="#0a0a1a",
    paper_bgcolor="#0a0a1a",
    margin=dict(l=20, r=20, t=20, b=20),
)
st.plotly_chart(fig, use_container_width=True)

# Edge legend
edge_types = set(e["type"] for e in edges)
if edge_types:
    legend_items = []
    for etype in sorted(edge_types):
        color = {"activates": "green", "inhibits": "red", "binds_to": "blue",
                 "phosphorylates": "yellow", "catalyzes": "cyan"}.get(etype, "gray")
        legend_items.append(f'<span style="color:{color};">\u2501\u2501</span> {etype}')
    st.markdown(" &nbsp;&nbsp; ".join(legend_items), unsafe_allow_html=True)

st.divider()

# ── ODE Simulation ───────────────────────────────────────────────

st.subheader("3. Dynamic Simulation")

try:
    from cognisom.genomics.reaction_builder import ReactionNetworkBuilder

    builder = ReactionNetworkBuilder(store)

    # Check if this is the AR signaling preset
    if set(selected_names) & {"AR", "DHT", "PSA"}:
        system = builder.build_ar_signaling()
        st.caption("Using AR signaling ODE system (6 species, entity-driven parameters)")
    else:
        system = builder.build_from_entity_names(selected_names)
        st.caption(f"Custom ODE: {system.n_species} species, {len(system.reactions)} reactions")

    if system.n_species > 0:
        # Parameter adjustment
        with st.expander("Adjust Parameters", expanded=False):
            adjusted_params = dict(system.parameters)
            cols = st.columns(min(3, len(system.parameters)))
            for i, (key, val) in enumerate(sorted(system.parameters.items())):
                if key.startswith("_"):
                    continue
                with cols[i % len(cols)]:
                    new_val = st.number_input(
                        key, value=float(val), format="%.4f",
                        key=f"param_{key}",
                    )
                    adjusted_params[key] = new_val

        # Run simulation
        duration_hours = st.slider("Simulation Duration (hours)", 1, 48, 12)
        dt = 0.001
        n_steps = int(duration_hours / dt)
        n_steps = min(n_steps, 50000)  # Cap for performance

        rhs = system.get_rhs_func()
        y = np.array(system.species_initial, dtype=float)
        if np.all(y == 0):
            # Set reasonable initial conditions
            y = np.ones(system.n_species) * 0.1
            if "DHT" in system.species_names:
                y[system._species_index["DHT"]] = 1.0

        # Euler integration
        trajectory = [y.copy()]
        times = [0.0]
        t = 0.0
        for step in range(n_steps):
            dy = rhs(t, y, adjusted_params)
            y = y + dt * dy
            y = np.maximum(y, 0)  # Non-negative
            t += dt
            if step % max(1, n_steps // 500) == 0:
                trajectory.append(y.copy())
                times.append(t)

        trajectory = np.array(trajectory)

        # Plot
        sim_fig = go.Figure()
        for i, name in enumerate(system.species_names):
            sim_fig.add_trace(go.Scatter(
                x=times, y=trajectory[:, i],
                mode="lines",
                name=name,
                line=dict(width=2),
            ))

        sim_fig.update_layout(
            title="Species Concentrations Over Time",
            xaxis_title="Time (hours)",
            yaxis_title="Concentration (arbitrary units)",
            height=400,
            legend=dict(font=dict(size=10)),
        )
        st.plotly_chart(sim_fig, use_container_width=True)

        # Show reactions
        with st.expander(f"Reactions ({len(system.reactions)})"):
            for rxn in system.reactions:
                reactants = " + ".join(rxn.reactants) if rxn.reactants else "\u2205"
                products = " + ".join(rxn.products) if rxn.products else "\u2205"
                st.markdown(
                    f"- **{rxn.name}**: {reactants} \u2192 {products} "
                    f"({rxn.reaction_type})"
                )

        # Show final concentrations
        with st.expander("Final State"):
            final = trajectory[-1]
            for i, name in enumerate(system.species_names):
                st.markdown(f"- **{name}**: {final[i]:.4f}")

    else:
        st.info("No reactions found between the selected entities. Try a preset pathway.")

except Exception as e:
    st.error(f"Simulation error: {e}")
    logger.exception("Pathway simulation failed")

st.divider()

# ── SABIO-RK Integration ─────────────────────────────────────────

st.subheader("4. Enrich with Real Kinetics (SABIO-RK)")
st.caption("Look up experimentally measured rate constants from SABIO-RK database.")

enrich_entity = st.selectbox("Select entity to enrich", selected_names)

if st.button("Query SABIO-RK", type="secondary"):
    with st.spinner(f"Querying SABIO-RK for {enrich_entity}..."):
        try:
            from cognisom.library.kinetics_client import SABIORKClient, enrich_entity_kinetics

            result = enrich_entity_kinetics(store, enrich_entity)
            if result.get("parameters_found", 0) > 0:
                st.success(
                    f"Found {result['parameters_found']} kinetic parameters for "
                    f"{enrich_entity} from SABIO-RK"
                )
                # Show what was added
                entity = store.find_entity_by_name(enrich_entity, None)
                if entity and entity.physics_params:
                    sabio_params = {k: v for k, v in entity.physics_params.items()
                                    if "sabio" in str(entity.physics_params.get(f"_{k}_source", "")).lower()
                                    or "SABIO" in str(entity.physics_params.get(f"_{k}_source", ""))}
                    if sabio_params:
                        st.json(sabio_params)
            else:
                st.info(f"No kinetic data found for {enrich_entity} in SABIO-RK. "
                        f"Try a well-characterized enzyme name.")
        except Exception as e:
            st.error(f"SABIO-RK query failed: {e}")
