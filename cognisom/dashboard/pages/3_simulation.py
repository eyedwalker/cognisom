"""Simulation page - Real 9-module engine with interactive visualization."""

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Simulation | Cognisom", page_icon="🧬", layout="wide")

from cognisom.auth.middleware import streamlit_page_gate
user = streamlit_page_gate("3_simulation")

from cognisom.dashboard.engine_runner import EngineRunner, SCENARIOS

# ── Page header ──────────────────────────────────────────────────────

st.title("9-Module Simulation Engine")
st.markdown(
    "Run the **real** multi-scale cellular simulation — cellular dynamics, "
    "immune surveillance, vascular O2 delivery, lymphatic metastasis, "
    "epigenetic silencing, circadian gating, morphogen gradients, and more."
)

# ── Sidebar: Scenario & Parameters ──────────────────────────────────

st.sidebar.header("Simulation Setup")

scenario = st.sidebar.selectbox(
    "Scenario",
    list(SCENARIOS.keys()),
    help="Preset parameter combinations for common experimental conditions",
)
st.sidebar.caption(SCENARIOS[scenario]["desc"])

st.sidebar.markdown("---")
st.sidebar.subheader("Time")
duration = st.sidebar.slider("Duration (hours)", 1.0, 24.0, 4.0, step=0.5)
dt = st.sidebar.select_slider("dt (hours)", options=[0.01, 0.02, 0.05, 0.1], value=0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("Modules")
mod_cellular = st.sidebar.checkbox("Cellular", value=True, help="Cell division, metabolism, death")
mod_immune = st.sidebar.checkbox("Immune", value=True, help="T cells, NK cells, macrophages")
mod_vascular = st.sidebar.checkbox("Vascular", value=True, help="Blood vessels, O2/glucose delivery")
mod_lymphatic = st.sidebar.checkbox("Lymphatic", value=True, help="Lymph drainage, metastasis")
mod_molecular = st.sidebar.checkbox("Molecular", value=True, help="Gene expression, exosomes")
mod_spatial = st.sidebar.checkbox("Spatial", value=True, help="3D diffusion fields")
mod_epigenetic = st.sidebar.checkbox("Epigenetic", value=True, help="Methylation, gene silencing")
mod_circadian = st.sidebar.checkbox("Circadian", value=True, help="24h oscillators")
mod_morphogen = st.sidebar.checkbox("Morphogen", value=True, help="BMP/Wnt gradients, cell fate")

modules_enabled = {
    "cellular": mod_cellular, "immune": mod_immune,
    "vascular": mod_vascular, "lymphatic": mod_lymphatic,
    "molecular": mod_molecular, "spatial": mod_spatial,
    "epigenetic": mod_epigenetic, "circadian": mod_circadian,
    "morphogen": mod_morphogen,
}
n_enabled = sum(modules_enabled.values())
st.sidebar.caption(f"{n_enabled} / 9 modules enabled")

# ── Run button ───────────────────────────────────────────────────────

st.divider()
run_btn = st.button("Run Simulation", type="primary", use_container_width=True)

if run_btn or st.session_state.get("sim_engine_ran"):
    if run_btn:
        runner = EngineRunner(
            dt=dt, duration=duration, scenario=scenario,
            modules_enabled=modules_enabled,
            record_interval=max(1, int(0.1 / dt)),  # ~every 0.1h
        )

        progress_bar = st.progress(0, text="Building engine...")
        runner.build()

        def _progress(current, total):
            frac = current / total
            progress_bar.progress(frac, text=f"Simulating... {frac*100:.0f}%  (t={current*dt:.2f}h)")

        runner.run(progress_callback=_progress)
        progress_bar.progress(1.0, text="Simulation complete.")

        # Cache in session
        st.session_state["sim_ts"] = runner.get_time_series()
        st.session_state["sim_events"] = runner.event_log
        st.session_state["sim_event_summary"] = runner.get_event_summary()
        st.session_state["sim_key_events"] = runner.get_key_events()
        st.session_state["sim_cell_snaps"] = runner.cell_snapshots
        st.session_state["sim_final"] = runner.get_final_state()
        st.session_state["sim_scenario"] = scenario
        st.session_state["sim_duration"] = duration
        st.session_state["sim_dt"] = dt
        st.session_state["sim_modules"] = modules_enabled
        st.session_state["sim_engine_ran"] = True

    # ── Retrieve cached data ─────────────────────────────────────
    ts = st.session_state["sim_ts"]
    events = st.session_state["sim_events"]
    event_summary = st.session_state["sim_event_summary"]
    key_events = st.session_state["sim_key_events"]
    cell_snaps = st.session_state["sim_cell_snaps"]
    final = st.session_state["sim_final"]
    sim_scenario = st.session_state["sim_scenario"]

    time_arr = ts["time"]

    # ══════════════════════════════════════════════════════════════
    # HEADLINE METRICS
    # ══════════════════════════════════════════════════════════════

    st.subheader(f"Results — {sim_scenario}")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Cancer Cells", f"{ts['n_cancer'][-1]}",
              delta=f"{ts['n_cancer'][-1] - ts['n_cancer'][0]:+d}")
    m2.metric("Normal Cells", f"{ts['n_normal'][-1]}")
    m3.metric("Immune Kills", f"{ts['total_kills'][-1]}")
    m4.metric("Metastases", f"{ts['total_metastases'][-1]}")
    m5.metric("Hypoxic Cells", f"{ts['hypoxic_regions'][-1]}")
    m6.metric("Events", f"{len(events)}")

    st.divider()

    # ══════════════════════════════════════════════════════════════
    # TIME-SERIES PLOTS
    # ══════════════════════════════════════════════════════════════

    st.subheader("Simulation Dynamics (Real Engine)")

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Cell Populations", "Immune System",
            "Oxygen & Metabolism", "Metastasis & Lymphatic",
            "Epigenetic Changes", "Event Accumulation",
        ),
        vertical_spacing=0.08, horizontal_spacing=0.08,
    )

    # 1. Cell populations
    fig.add_trace(go.Scatter(x=time_arr, y=ts["n_cancer"], name="Cancer",
                             line=dict(color="#e74c3c", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=time_arr, y=ts["n_normal"], name="Normal",
                             line=dict(color="#2ecc71", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=time_arr, y=ts["n_cells"], name="Total",
                             line=dict(color="#95a5a6", width=1, dash="dot")), row=1, col=1)

    # 2. Immune
    fig.add_trace(go.Scatter(x=time_arr, y=ts["n_immune"], name="Immune Total",
                             line=dict(color="#3498db", width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=time_arr, y=ts["n_activated"], name="Activated",
                             line=dict(color="#e67e22", width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=time_arr, y=ts["total_kills"], name="Cumulative Kills",
                             line=dict(color="#9b59b6", width=2), fill="tozeroy",
                             fillcolor="rgba(155,89,182,0.15)"), row=1, col=2)

    # 3. Oxygen
    fig.add_trace(go.Scatter(x=time_arr, y=ts["avg_cell_O2"], name="Avg Cell O2",
                             line=dict(color="#2980b9", width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=time_arr, y=ts["avg_cell_glucose"], name="Avg Glucose",
                             line=dict(color="#27ae60", width=2)), row=2, col=1)
    fig.add_hline(y=0.05, line_dash="dash", line_color="red",
                  annotation_text="Hypoxia threshold", row=2, col=1)

    # 4. Metastasis
    fig.add_trace(go.Scatter(x=time_arr, y=ts["total_metastases"], name="Metastases",
                             line=dict(color="#c0392b", width=3)), row=2, col=2)
    fig.add_trace(go.Scatter(x=time_arr, y=ts["cancer_in_vessels"], name="Cancer in Lymph",
                             line=dict(color="#e74c3c", width=1, dash="dot")), row=2, col=2)
    fig.add_trace(go.Scatter(x=time_arr, y=ts["immune_in_vessels"], name="Immune in Lymph",
                             line=dict(color="#3498db", width=1, dash="dot")), row=2, col=2)

    # 5. Epigenetic
    fig.add_trace(go.Scatter(x=time_arr, y=ts["avg_methylation"], name="Avg Methylation",
                             line=dict(color="#8e44ad", width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=time_arr, y=ts["silenced_genes"], name="Silenced Genes",
                             line=dict(color="#c0392b", width=2)), row=3, col=1)

    # 6. Events accumulation
    div_arr = ts["total_divisions"]
    death_arr = ts["total_deaths"]
    trans_arr = ts["total_transformations"]
    fig.add_trace(go.Scatter(x=time_arr, y=div_arr, name="Divisions",
                             line=dict(color="#2ecc71", width=2)), row=3, col=2)
    fig.add_trace(go.Scatter(x=time_arr, y=death_arr, name="Deaths",
                             line=dict(color="#e74c3c", width=2)), row=3, col=2)
    fig.add_trace(go.Scatter(x=time_arr, y=trans_arr, name="Transformations",
                             line=dict(color="#f39c12", width=2)), row=3, col=2)

    fig.update_xaxes(title_text="Time (hours)", row=3)
    fig.update_layout(height=900, showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.08))
    st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════
    # 3D TISSUE VIEW
    # ══════════════════════════════════════════════════════════════

    st.divider()
    st.subheader("3D Tissue View (Real Cell Positions)")

    if cell_snaps:
        # Time slider
        snap_idx = st.slider(
            "Time step", 0, len(cell_snaps) - 1, len(cell_snaps) - 1,
            help="Scrub through simulation snapshots",
        )
        snap = cell_snaps[snap_idx]

        # Build 3D figure
        fig3d = go.Figure()

        # Cells
        if snap.cell_positions is not None and len(snap.cell_positions) > 0:
            pos = snap.cell_positions
            colors = ["#e74c3c" if t == "cancer" else "#2ecc71" for t in snap.cell_types]
            sizes = [8 if t == "cancer" else 5 for t in snap.cell_types]
            hover = [
                f"Cell {cid}<br>Type: {ct}<br>Phase: {cp}<br>"
                f"O2: {co:.3f}<br>MHC-I: {cm:.2f}"
                for cid, ct, cp, co, cm in zip(
                    snap.cell_ids, snap.cell_types, snap.cell_phases,
                    snap.cell_oxygen, snap.cell_mhc1
                )
            ]
            fig3d.add_trace(go.Scatter3d(
                x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
                mode="markers",
                marker=dict(size=sizes, color=colors, opacity=0.8,
                            line=dict(width=0.5, color="white")),
                text=hover, hoverinfo="text", name="Cells",
            ))

        # Immune cells
        if snap.immune_positions is not None and len(snap.immune_positions) > 0:
            ipos = snap.immune_positions
            icolors = {
                "T_cell": "#00bcd4", "NK_cell": "#ff4081", "macrophage": "#ff9800"
            }
            ic = [icolors.get(t, "#999") for t in snap.immune_types]
            isymbols = ["diamond" if a else "circle" for a in snap.immune_activated]
            ihover = [
                f"{it} ({'ACTIVE' if ia else 'patrolling'})"
                for it, ia in zip(snap.immune_types, snap.immune_activated)
            ]
            fig3d.add_trace(go.Scatter3d(
                x=ipos[:, 0], y=ipos[:, 1], z=ipos[:, 2],
                mode="markers",
                marker=dict(size=6, color=ic, opacity=0.9, symbol=isymbols),
                text=ihover, hoverinfo="text", name="Immune",
            ))

        # Capillaries (red lines)
        for start, end in zip(snap.capillary_starts, snap.capillary_ends):
            fig3d.add_trace(go.Scatter3d(
                x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]],
                mode="lines", line=dict(color="#e74c3c", width=3),
                showlegend=False, hoverinfo="skip",
            ))

        # Lymphatics (blue lines)
        for start, end in zip(snap.lymph_starts, snap.lymph_ends):
            fig3d.add_trace(go.Scatter3d(
                x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]],
                mode="lines", line=dict(color="#2196f3", width=4),
                showlegend=False, hoverinfo="skip",
            ))

        fig3d.update_layout(
            height=600,
            scene=dict(
                xaxis_title="X (μm)", yaxis_title="Y (μm)", zaxis_title="Z (μm)",
                aspectmode="cube",
                bgcolor="#0e1117",
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            title=f"Tissue at t={time_arr[min(snap_idx, len(time_arr)-1)]:.2f}h",
        )
        st.plotly_chart(fig3d, use_container_width=True)

        # Legend
        lc1, lc2, lc3, lc4 = st.columns(4)
        lc1.markdown("**Cells**: :green[Normal] / :red[Cancer]")
        lc2.markdown("**Immune**: :blue[T cell] / :red[NK] / :orange[Macrophage]")
        lc3.markdown("**Red lines**: Capillaries")
        lc4.markdown("**Blue lines**: Lymphatic vessels")

    # ══════════════════════════════════════════════════════════════
    # EVENT LOG & TIMELINE
    # ══════════════════════════════════════════════════════════════

    st.divider()
    st.subheader("Event Log")

    ev1, ev2 = st.columns([1, 2])

    with ev1:
        st.markdown("**Event Counts**")
        if event_summary:
            for etype, count in sorted(event_summary.items(), key=lambda x: -x[1]):
                label = etype.replace("_", " ").title()
                st.markdown(f"- **{label}**: {count}")
        else:
            st.info("No events recorded.")

    with ev2:
        st.markdown("**Key Events Timeline**")
        if key_events:
            event_icons = {
                "cancer_killed": "KILL",
                "metastasis_occurred": "METASTASIS",
                "cell_transformed": "TRANSFORM",
                "hypoxia_detected": "HYPOXIA",
                "immune_activated": "IMMUNE",
            }
            for ev in key_events[-30:]:  # Show last 30
                t = ev["time"]
                etype = ev["event"]
                label = event_icons.get(etype, etype)
                data = ev.get("data", {})

                detail = ""
                if etype == "cancer_killed":
                    detail = f"Cell {data.get('cell_id','')} killed by {data.get('killer_type','')}"
                elif etype == "metastasis_occurred":
                    detail = f"Cell {data.get('cell_id','')} entered lymphatic vessel {data.get('vessel_id','')}"
                elif etype == "cell_transformed":
                    detail = f"Cell {data.get('cell_id','')} transformed (mutations: {data.get('mutations',[])})"
                elif etype == "hypoxia_detected":
                    detail = f"{data.get('n_hypoxic_cells',0)} hypoxic cells ({data.get('severity','')})"
                elif etype == "immune_activated":
                    detail = f"{data.get('immune_type','')} targeting cell {data.get('target_id','')}"

                st.markdown(f"`t={t:.2f}h` **[{label}]** {detail}")
        else:
            st.info("No significant events.")

    # ══════════════════════════════════════════════════════════════
    # MODULE DETAIL PANELS
    # ══════════════════════════════════════════════════════════════

    st.divider()
    st.subheader("Module Details")

    for mod_name in ["cellular", "immune", "vascular", "lymphatic",
                     "molecular", "epigenetic", "circadian", "morphogen", "spatial"]:
        mod_state = final.get(mod_name, {})
        if not mod_state:
            continue
        with st.expander(f"**{mod_name.title()}** Module"):
            cols = st.columns(min(len(mod_state), 4))
            for i, (k, v) in enumerate(mod_state.items()):
                if isinstance(v, (dict, list)):
                    continue
                with cols[i % len(cols)]:
                    if isinstance(v, float):
                        st.metric(k.replace("_", " ").title(), f"{v:.4f}")
                    else:
                        st.metric(k.replace("_", " ").title(), str(v))

    # ══════════════════════════════════════════════════════════════
    # EXPORT
    # ══════════════════════════════════════════════════════════════

    st.divider()
    st.subheader("Export Results")

    ex1, ex2 = st.columns(2)
    import json, io

    with ex1:
        summary = {
            "scenario": sim_scenario,
            "duration_hours": time_arr[-1] if time_arr else 0,
            "final_cancer": ts["n_cancer"][-1],
            "final_normal": ts["n_normal"][-1],
            "total_kills": ts["total_kills"][-1],
            "total_metastases": ts["total_metastases"][-1],
            "total_events": len(events),
            "event_summary": event_summary,
        }
        st.download_button("Download Summary (JSON)",
                           json.dumps(summary, indent=2, default=str),
                           "sim_summary.json", "application/json",
                           use_container_width=True)

    with ex2:
        buf = io.StringIO()
        keys = list(ts.keys())
        buf.write(",".join(keys) + "\n")
        n_rows = len(ts[keys[0]])
        for i in range(n_rows):
            row = ",".join(str(ts[k][i]) for k in keys)
            buf.write(row + "\n")
        st.download_button("Download Time Series (CSV)", buf.getvalue(),
                           "sim_timeseries.csv", "text/csv",
                           use_container_width=True)

else:
    st.info(
        "Configure scenario and parameters in the sidebar, then click "
        "**Run Simulation** to start the real 9-module engine."
    )

    # Show scenario descriptions
    st.subheader("Available Scenarios")
    for name, cfg in SCENARIOS.items():
        with st.expander(f"**{name}**"):
            st.markdown(cfg["desc"])
            for mod, params in cfg.items():
                if mod == "desc":
                    continue
                st.markdown(f"**{mod}**: `{params}`")

# Footer
from cognisom.dashboard.footer import render_footer
render_footer()
