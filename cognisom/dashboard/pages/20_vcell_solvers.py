"""
Page 20 — VCell Parity: GPU-Accelerated Solvers
===============================================

Interactive demos and configuration for VCell-compatible simulation solvers:
1. ODE Solver - Batched deterministic integration
2. Smoldyn Spatial - Particle-based Brownian dynamics
3. Hybrid ODE/SSA - Automatic fast/slow partitioning
4. BNGL Rule-Based - Combinatorial complexity
5. Imaging Pipeline - Image-to-geometry conversion

Each solver achieves VCell-level capabilities with GPU acceleration.
"""

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
import numpy as np
import time

st.set_page_config(
    page_title="VCell Solvers | Cognisom",
    page_icon="⚡",
    layout="wide"
)

from cognisom.auth.middleware import streamlit_page_gate
user = streamlit_page_gate("20_vcell_solvers")

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Custom CSS ───────────────────────────────────────────────────────

st.markdown("""
<style>
.solver-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.1) 0%, rgba(168,85,247,0.1) 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}
.solver-title {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.solver-desc {
    font-size: 0.85rem;
    opacity: 0.8;
}
.status-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 600;
}
.status-ready { background: rgba(34,197,94,0.2); color: #4ade80; }
.status-beta { background: rgba(251,191,36,0.2); color: #fbbf24; }
.metric-box {
    background: rgba(255,255,255,0.05);
    border-radius: 8px;
    padding: 0.8rem;
    text-align: center;
}
.doc-section {
    background: rgba(0,0,0,0.2);
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
    font-family: monospace;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────

st.title("⚡ VCell Parity: GPU-Accelerated Solvers")
st.markdown("""
Five solver types matching [VCell](https://vcell.org) capabilities with **GPU acceleration** for 10-100x speedup.
Configure, run, and visualize simulations directly in the browser.
""")

# ── Solver Status Overview ───────────────────────────────────────────

st.subheader("Solver Status")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown("""
    <div class="solver-card">
        <div class="solver-title">🔢 ODE Solver</div>
        <span class="status-badge status-ready">READY</span>
        <div class="solver-desc">BDF/Adams batched integration for 10K+ cells</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="solver-card">
        <div class="solver-title">🔬 Smoldyn Spatial</div>
        <span class="status-badge status-ready">READY</span>
        <div class="solver-desc">Particle Brownian dynamics with reactions</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="solver-card">
        <div class="solver-title">🔀 Hybrid ODE/SSA</div>
        <span class="status-badge status-ready">READY</span>
        <div class="solver-desc">Auto-partitioned deterministic + stochastic</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="solver-card">
        <div class="solver-title">📐 BNGL Rules</div>
        <span class="status-badge status-ready">READY</span>
        <div class="solver-desc">Rule-based modeling for combinatorial systems</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class="solver-card">
        <div class="solver-title">🖼️ Imaging</div>
        <span class="status-badge status-ready">READY</span>
        <div class="solver-desc">Microscopy to simulation geometry</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Tabs for each solver ─────────────────────────────────────────────

tab_mol3d, tab_ode, tab_smoldyn, tab_hybrid, tab_bngl, tab_imaging, tab_docs = st.tabs([
    "🔬 Molecular 3D", "ODE Solver", "Smoldyn Spatial", "Hybrid ODE/SSA", "BNGL Rules", "Imaging", "Documentation"
])

# ══════════════════════════════════════════════════════════════════════
# TAB 0: Molecular 3D Visualization (NEW)
# ══════════════════════════════════════════════════════════════════════

with tab_mol3d:
    st.header("🔬 Molecular-Level Cell Visualization")

    st.markdown("""
    **See inside a cell at the molecular level** — receptors, ligands, signaling molecules,
    and reactions happening in real-time. This is what VCell simulations actually represent.
    """)

    import streamlit.components.v1 as components
    import json
    import math
    import random

    col_config3d, col_view3d = st.columns([1, 2])

    with col_config3d:
        st.subheader("Cell Configuration")

        cell_scenario = st.selectbox(
            "Scenario",
            ["Receptor-Ligand Binding", "EGFR Signaling Cascade", "T Cell Activation", "Apoptosis Pathway"],
            key="mol3d_scenario"
        )

        st.subheader("Molecule Counts")
        n_receptors = st.slider("Receptors (membrane)", 20, 200, 80, key="n_receptors")
        n_ligands = st.slider("Ligands (extracellular)", 10, 100, 40, key="n_ligands")
        n_signaling = st.slider("Signaling molecules (cytoplasm)", 50, 500, 200, key="n_signaling")

        st.subheader("Dynamics")
        diffusion_speed = st.slider("Diffusion speed", 0.1, 2.0, 0.5, key="diff_speed")
        reaction_rate = st.slider("Reaction rate", 0.1, 2.0, 1.0, key="rxn_rate")
        show_trails = st.checkbox("Show molecule trails", False, key="show_trails")
        show_bonds = st.checkbox("Show molecular bonds", True, key="show_bonds")

        run_mol3d = st.button("▶ Run Visualization", type="primary", key="run_mol3d")

    with col_view3d:
        # Always show the visualization
        if True:  # Always render
            # Generate molecule data based on SPECIFIC scenario
            molecules = []
            extra_structures = []  # For scenario-specific structures

            # Cell parameters
            cell_radius = 10.0
            nucleus_radius = 3.0

            random.seed(42)

            # ════════════════════════════════════════════════════════════════
            # SCENARIO-SPECIFIC MOLECULE GENERATION
            # ════════════════════════════════════════════════════════════════

            if cell_scenario == "Receptor-Ligand Binding":
                # Simple L + R <-> LR binding
                # Receptors clustered in patches on membrane
                for patch in range(5):
                    patch_theta = patch * 2 * math.pi / 5
                    patch_phi = math.pi / 2 + random.uniform(-0.3, 0.3)
                    for i in range(n_receptors // 5):
                        theta = patch_theta + random.gauss(0, 0.3)
                        phi = patch_phi + random.gauss(0, 0.3)
                        molecules.append({
                            "type": "receptor",
                            "x": cell_radius * math.sin(phi) * math.cos(theta),
                            "y": cell_radius * math.sin(phi) * math.sin(theta),
                            "z": cell_radius * math.cos(phi),
                            "color": "#ff4444",
                            "size": 0.5,
                            "bound": random.random() < 0.2,
                        })

                # Ligands approaching from one direction (gradient)
                for i in range(n_ligands):
                    r = random.uniform(cell_radius + 0.5, cell_radius + 8)
                    theta = random.gauss(0, 0.8)  # Concentrated in one direction
                    phi = random.gauss(math.pi/2, 0.5)
                    molecules.append({
                        "type": "ligand",
                        "x": r * math.sin(phi) * math.cos(theta),
                        "y": r * math.sin(phi) * math.sin(theta),
                        "z": r * math.cos(phi),
                        "color": "#44ff44",
                        "size": 0.3,
                        "bound": False,
                    })

                legend_items = [
                    ("Receptors (R)", "#ff4444"),
                    ("Ligands (L)", "#44ff44"),
                    ("Complex (LR)", "#ffff00"),
                ]

            elif cell_scenario == "EGFR Signaling Cascade":
                # EGFR dimerization and downstream signaling
                # EGF receptors that can dimerize
                for i in range(n_receptors):
                    theta = random.uniform(0, 2 * math.pi)
                    phi = math.acos(2 * random.uniform(0, 1) - 1)
                    is_dimer = random.random() < 0.3
                    molecules.append({
                        "type": "EGFR",
                        "x": cell_radius * math.sin(phi) * math.cos(theta),
                        "y": cell_radius * math.sin(phi) * math.sin(theta),
                        "z": cell_radius * math.cos(phi),
                        "color": "#ff6600" if is_dimer else "#ff4444",
                        "size": 0.6 if is_dimer else 0.4,
                        "bound": is_dimer,
                        "phosphorylated": is_dimer and random.random() < 0.5,
                    })

                # EGF ligands
                for i in range(n_ligands):
                    r = random.uniform(cell_radius + 1, cell_radius + 4)
                    theta = random.uniform(0, 2 * math.pi)
                    phi = math.acos(2 * random.uniform(0, 1) - 1)
                    molecules.append({
                        "type": "EGF",
                        "x": r * math.sin(phi) * math.cos(theta),
                        "y": r * math.sin(phi) * math.sin(theta),
                        "z": r * math.cos(phi),
                        "color": "#00ff88",
                        "size": 0.25,
                        "bound": False,
                    })

                # MAPK cascade inside: Ras -> Raf -> MEK -> ERK
                cascade_molecules = [
                    ("Ras", "#ff0000", 0.25),
                    ("Raf", "#ff8800", 0.3),
                    ("MEK", "#ffff00", 0.25),
                    ("ERK", "#88ff00", 0.2),
                ]
                for mol_type, color, size in cascade_molecules:
                    for i in range(n_signaling // 4):
                        r = random.uniform(nucleus_radius + 1, cell_radius - 1)
                        theta = random.uniform(0, 2 * math.pi)
                        phi = math.acos(2 * random.uniform(0, 1) - 1)
                        molecules.append({
                            "type": mol_type,
                            "x": r * math.sin(phi) * math.cos(theta),
                            "y": r * math.sin(phi) * math.sin(theta),
                            "z": r * math.cos(phi),
                            "color": color,
                            "size": size,
                            "active": random.random() < 0.15,
                        })

                legend_items = [
                    ("EGFR", "#ff4444"),
                    ("EGFR dimer", "#ff6600"),
                    ("EGF", "#00ff88"),
                    ("Ras", "#ff0000"),
                    ("Raf", "#ff8800"),
                    ("MEK", "#ffff00"),
                    ("ERK", "#88ff00"),
                ]

            elif cell_scenario == "T Cell Activation":
                # TCR, CD4/CD8, MHC interaction + calcium signaling
                # TCR complexes on membrane
                for i in range(n_receptors):
                    theta = random.uniform(0, 2 * math.pi)
                    phi = math.acos(2 * random.uniform(0, 1) - 1)
                    molecules.append({
                        "type": "TCR",
                        "x": cell_radius * math.sin(phi) * math.cos(theta),
                        "y": cell_radius * math.sin(phi) * math.sin(theta),
                        "z": cell_radius * math.cos(phi),
                        "color": "#4444ff",
                        "size": 0.5,
                        "bound": random.random() < 0.25,
                    })

                # CD4/CD8 co-receptors
                for i in range(n_receptors // 3):
                    theta = random.uniform(0, 2 * math.pi)
                    phi = math.acos(2 * random.uniform(0, 1) - 1)
                    molecules.append({
                        "type": "CD4",
                        "x": cell_radius * math.sin(phi) * math.cos(theta),
                        "y": cell_radius * math.sin(phi) * math.sin(theta),
                        "z": cell_radius * math.cos(phi),
                        "color": "#00ffff",
                        "size": 0.35,
                        "bound": False,
                    })

                # Antigen-presenting cell nearby (MHC + peptide)
                apc_center = (cell_radius + 6, 0, 0)
                for i in range(n_ligands):
                    theta = random.uniform(0, 2 * math.pi)
                    phi = math.acos(2 * random.uniform(0, 1) - 1)
                    r = 4  # APC radius
                    molecules.append({
                        "type": "MHC_peptide",
                        "x": apc_center[0] + r * math.sin(phi) * math.cos(theta) * 0.5,
                        "y": apc_center[1] + r * math.sin(phi) * math.sin(theta),
                        "z": apc_center[2] + r * math.cos(phi),
                        "color": "#ff00ff",
                        "size": 0.3,
                        "bound": False,
                    })

                extra_structures.append({
                    "type": "APC",
                    "x": apc_center[0], "y": apc_center[1], "z": apc_center[2],
                    "radius": 5,
                    "color": "#884488",
                })

                # Calcium ions and signaling
                for i in range(n_signaling):
                    r = random.uniform(nucleus_radius + 0.5, cell_radius - 1)
                    theta = random.uniform(0, 2 * math.pi)
                    phi = math.acos(2 * random.uniform(0, 1) - 1)
                    mol_type = random.choice(["Ca2+", "NFAT", "ZAP70", "LAT"])
                    colors = {"Ca2+": "#ffff00", "NFAT": "#ff00ff", "ZAP70": "#ff8800", "LAT": "#00ff00"}
                    molecules.append({
                        "type": mol_type,
                        "x": r * math.sin(phi) * math.cos(theta),
                        "y": r * math.sin(phi) * math.sin(theta),
                        "z": r * math.cos(phi),
                        "color": colors[mol_type],
                        "size": 0.15 if mol_type == "Ca2+" else 0.25,
                        "active": random.random() < 0.2,
                    })

                legend_items = [
                    ("TCR", "#4444ff"),
                    ("CD4", "#00ffff"),
                    ("MHC-peptide", "#ff00ff"),
                    ("Ca²⁺", "#ffff00"),
                    ("NFAT", "#ff00ff"),
                    ("ZAP70", "#ff8800"),
                ]

            else:  # Apoptosis Pathway
                # Death receptor, caspase cascade, mitochondrial pathway
                # Death receptors (Fas, TRAIL-R)
                for i in range(n_receptors):
                    theta = random.uniform(0, 2 * math.pi)
                    phi = math.acos(2 * random.uniform(0, 1) - 1)
                    molecules.append({
                        "type": "Fas",
                        "x": cell_radius * math.sin(phi) * math.cos(theta),
                        "y": cell_radius * math.sin(phi) * math.sin(theta),
                        "z": cell_radius * math.cos(phi),
                        "color": "#880000",
                        "size": 0.45,
                        "bound": random.random() < 0.3,
                    })

                # FasL (death ligand)
                for i in range(n_ligands):
                    r = random.uniform(cell_radius + 1, cell_radius + 5)
                    theta = random.uniform(0, 2 * math.pi)
                    phi = math.acos(2 * random.uniform(0, 1) - 1)
                    molecules.append({
                        "type": "FasL",
                        "x": r * math.sin(phi) * math.cos(theta),
                        "y": r * math.sin(phi) * math.sin(theta),
                        "z": r * math.cos(phi),
                        "color": "#ff0000",
                        "size": 0.3,
                        "bound": False,
                    })

                # Mitochondria (release cytochrome c)
                mito_positions = []
                for i in range(8):
                    r = random.uniform(4, 7)
                    theta = random.uniform(0, 2 * math.pi)
                    phi = math.acos(2 * random.uniform(0, 1) - 1)
                    pos = (r * math.sin(phi) * math.cos(theta),
                           r * math.sin(phi) * math.sin(theta),
                           r * math.cos(phi))
                    mito_positions.append(pos)
                    extra_structures.append({
                        "type": "mitochondria",
                        "x": pos[0], "y": pos[1], "z": pos[2],
                        "radius": 0.8,
                        "color": "#ff6600",
                    })

                # Caspase cascade
                caspases = [
                    ("Caspase8", "#ff4400", 0.25),   # Initiator
                    ("Caspase9", "#ff8800", 0.25),   # Initiator (mito)
                    ("Caspase3", "#ffcc00", 0.3),    # Executioner
                    ("CytC", "#00ffff", 0.15),       # Cytochrome C
                    ("Bcl2", "#00ff00", 0.2),        # Anti-apoptotic
                    ("Bax", "#ff0088", 0.2),         # Pro-apoptotic
                ]
                for mol_type, color, size in caspases:
                    for i in range(n_signaling // 6):
                        r = random.uniform(nucleus_radius + 0.5, cell_radius - 1)
                        theta = random.uniform(0, 2 * math.pi)
                        phi = math.acos(2 * random.uniform(0, 1) - 1)
                        molecules.append({
                            "type": mol_type,
                            "x": r * math.sin(phi) * math.cos(theta),
                            "y": r * math.sin(phi) * math.sin(theta),
                            "z": r * math.cos(phi),
                            "color": color,
                            "size": size,
                            "active": random.random() < 0.1,
                        })

                legend_items = [
                    ("Fas receptor", "#880000"),
                    ("FasL", "#ff0000"),
                    ("Caspase-8", "#ff4400"),
                    ("Caspase-3", "#ffcc00"),
                    ("Cytochrome C", "#00ffff"),
                    ("Bcl-2", "#00ff00"),
                    ("Bax", "#ff0088"),
                ]

            molecules_json = json.dumps(molecules)
            extra_json = json.dumps(extra_structures)
            show_trails_js = "true" if show_trails else "false"
            show_bonds_js = "true" if show_bonds else "false"
            legend_json = json.dumps(legend_items)

            # Three.js molecular visualization
            mol3d_html = f'''
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{ margin: 0; overflow: hidden; background: #050510; }}
                    canvas {{ display: block; }}
                    #info {{
                        position: absolute;
                        top: 10px;
                        left: 10px;
                        color: #fff;
                        font-family: 'Segoe UI', sans-serif;
                        font-size: 13px;
                        background: rgba(0,0,0,0.85);
                        padding: 12px 16px;
                        border-radius: 8px;
                        border: 1px solid rgba(100,100,255,0.3);
                        max-width: 220px;
                    }}
                    #info h3 {{ margin: 0 0 8px 0; font-size: 15px; color: #88aaff; }}
                    #legend {{
                        position: absolute;
                        bottom: 10px;
                        right: 10px;
                        color: #fff;
                        font-family: monospace;
                        font-size: 11px;
                        background: rgba(0,0,0,0.85);
                        padding: 12px 16px;
                        border-radius: 8px;
                        border: 1px solid rgba(255,255,255,0.1);
                    }}
                    .leg-title {{ font-weight: bold; margin-bottom: 8px; color: #88aaff; }}
                    .leg-item {{ display: flex; align-items: center; margin: 4px 0; }}
                    .leg-dot {{ width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; }}
                    #stats {{
                        position: absolute;
                        bottom: 10px;
                        left: 10px;
                        color: #888;
                        font-family: monospace;
                        font-size: 10px;
                        background: rgba(0,0,0,0.7);
                        padding: 8px 12px;
                        border-radius: 6px;
                    }}
                </style>
            </head>
            <body>
                <div id="info">
                    <h3>🔬 {cell_scenario}</h3>
                    <div>Cell diameter: {cell_radius * 2:.0f} μm</div>
                    <div>Receptors: {n_receptors}</div>
                    <div>Ligands: {n_ligands}</div>
                    <div>Signaling: {n_signaling}</div>
                    <div style="margin-top:8px;font-size:11px;color:#666">
                        Drag to rotate | Scroll to zoom
                    </div>
                </div>
                <div id="legend"></div>
                <div id="stats">Initializing...</div>

                <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
                <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
                <script>
                    const molecules = {molecules_json};
                    const extraStructures = {extra_json};
                    const legendItems = {legend_json};
                    const CELL_RADIUS = {cell_radius};
                    const NUCLEUS_RADIUS = {nucleus_radius};
                    const DIFFUSION_SPEED = {diffusion_speed};
                    const REACTION_RATE = {reaction_rate};
                    const SHOW_TRAILS = {show_trails_js};
                    const SHOW_BONDS = {show_bonds_js};
                    const SCENARIO = "{cell_scenario}";

                    // Build dynamic legend
                    let legendHtml = '<div class="leg-title">Molecules</div>';
                    legendItems.forEach(item => {{
                        legendHtml += `<div class="leg-item"><div class="leg-dot" style="background:${{item[1]}}"></div>${{item[0]}}</div>`;
                    }});
                    document.getElementById('legend').innerHTML = legendHtml;

                    // Scene setup
                    const scene = new THREE.Scene();
                    scene.background = new THREE.Color(0x050510);

                    const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 500);
                    camera.position.set(25, 18, 25);

                    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
                    renderer.setSize(window.innerWidth, window.innerHeight);
                    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
                    document.body.appendChild(renderer.domElement);

                    const controls = new THREE.OrbitControls(camera, renderer.domElement);
                    controls.enableDamping = true;
                    controls.dampingFactor = 0.05;
                    controls.autoRotate = true;
                    controls.autoRotateSpeed = 0.3;

                    // Lighting
                    scene.add(new THREE.AmbientLight(0x404060, 0.6));
                    const keyLight = new THREE.DirectionalLight(0xffffff, 1.0);
                    keyLight.position.set(20, 30, 20);
                    scene.add(keyLight);
                    scene.add(new THREE.PointLight(0x4488ff, 0.5, 50));

                    // ═══════════════════════════════════════════════════════
                    // MOLECULAR GEOMETRY FACTORY
                    // Creates realistic shapes for different molecule types
                    // ═══════════════════════════════════════════════════════

                    // Helper: Create capsule-like shape (cylinder + sphere caps) - r128 compatible
                    function createCapsule(radius, height, radialSegs, heightSegs) {{
                        const group = new THREE.Group();
                        // Cylinder body
                        const cylGeom = new THREE.CylinderGeometry(radius, radius, height, radialSegs, heightSegs);
                        const cylMesh = new THREE.Mesh(cylGeom);
                        group.add(cylMesh);
                        // Top hemisphere
                        const topGeom = new THREE.SphereGeometry(radius, radialSegs, heightSegs / 2, 0, Math.PI * 2, 0, Math.PI / 2);
                        const topMesh = new THREE.Mesh(topGeom);
                        topMesh.position.y = height / 2;
                        group.add(topMesh);
                        // Bottom hemisphere
                        const botGeom = new THREE.SphereGeometry(radius, radialSegs, heightSegs / 2, 0, Math.PI * 2, Math.PI / 2, Math.PI / 2);
                        const botMesh = new THREE.Mesh(botGeom);
                        botMesh.position.y = -height / 2;
                        group.add(botMesh);
                        return group;
                    }}

                    // Apply material to capsule group
                    function applyCapsuleMaterial(capsule, material) {{
                        capsule.children.forEach(child => {{ child.material = material; }});
                        return capsule;
                    }}

                    function createReceptorGeometry(size) {{
                        // Transmembrane receptor: extracellular domain + transmembrane + intracellular
                        const group = new THREE.Group();

                        // Extracellular domain (binding site) - lobed structure
                        const extraGeom = new THREE.DodecahedronGeometry(size * 1.2, 1);
                        const extraMat = new THREE.MeshStandardMaterial({{ color: 0xff4444, roughness: 0.4 }});
                        const extra = new THREE.Mesh(extraGeom, extraMat);
                        extra.position.y = size * 1.5;
                        group.add(extra);

                        // Transmembrane helix
                        const tmGeom = new THREE.CylinderGeometry(size * 0.25, size * 0.25, size * 2, 8);
                        const tmMat = new THREE.MeshStandardMaterial({{ color: 0xcc3333, roughness: 0.5 }});
                        const tm = new THREE.Mesh(tmGeom, tmMat);
                        group.add(tm);

                        // Intracellular domain (kinase domain)
                        const intraGeom = new THREE.IcosahedronGeometry(size * 0.8, 0);
                        const intraMat = new THREE.MeshStandardMaterial({{ color: 0xaa2222, roughness: 0.4 }});
                        const intra = new THREE.Mesh(intraGeom, intraMat);
                        intra.position.y = -size * 1.2;
                        group.add(intra);

                        return group;
                    }}

                    function createLigandGeometry(size) {{
                        // Small growth factor - compact globular protein
                        const group = new THREE.Group();
                        const mainGeom = new THREE.IcosahedronGeometry(size, 1);
                        const mainMat = new THREE.MeshStandardMaterial({{ color: 0x44ff44, roughness: 0.3 }});
                        group.add(new THREE.Mesh(mainGeom, mainMat));

                        // Binding loop
                        const loopGeom = new THREE.TorusGeometry(size * 0.6, size * 0.15, 8, 12);
                        const loopMat = new THREE.MeshStandardMaterial({{ color: 0x66ff66, roughness: 0.4 }});
                        const loop = new THREE.Mesh(loopGeom, loopMat);
                        loop.rotation.x = Math.PI / 2;
                        loop.position.y = size * 0.3;
                        group.add(loop);

                        return group;
                    }}

                    function createKinaseGeometry(size, color) {{
                        // Kinase: bilobed structure (N-lobe + C-lobe)
                        const group = new THREE.Group();

                        // N-terminal lobe (smaller, beta-sheet rich)
                        const nLobeGeom = new THREE.DodecahedronGeometry(size * 0.7, 0);
                        const nLobeMat = new THREE.MeshStandardMaterial({{ color: color, roughness: 0.35 }});
                        const nLobe = new THREE.Mesh(nLobeGeom, nLobeMat);
                        nLobe.position.set(size * 0.4, size * 0.3, 0);
                        group.add(nLobe);

                        // C-terminal lobe (larger, alpha-helix rich)
                        const cLobeGeom = new THREE.IcosahedronGeometry(size * 0.9, 1);
                        const cLobeMat = new THREE.MeshStandardMaterial({{ color: color, roughness: 0.35 }});
                        const cLobe = new THREE.Mesh(cLobeGeom, cLobeMat);
                        cLobe.position.set(-size * 0.3, -size * 0.2, 0);
                        group.add(cLobe);

                        // Active site cleft connector
                        const cleftGeom = new THREE.CylinderGeometry(size * 0.2, size * 0.3, size * 0.8, 6);
                        const cleftMat = new THREE.MeshStandardMaterial({{ color: color, roughness: 0.4 }});
                        const cleft = new THREE.Mesh(cleftGeom, cleftMat);
                        cleft.rotation.z = Math.PI / 4;
                        group.add(cleft);

                        return group;
                    }}

                    function createSmallMoleculeGeometry(size, color) {{
                        // Small molecules (Ca2+, ATP, etc.) - simple but distinct
                        const group = new THREE.Group();
                        const geom = new THREE.OctahedronGeometry(size, 0);
                        const mat = new THREE.MeshStandardMaterial({{
                            color: color,
                            emissive: color,
                            emissiveIntensity: 0.3,
                            roughness: 0.2
                        }});
                        group.add(new THREE.Mesh(geom, mat));
                        return group;
                    }}

                    function createGProteinGeometry(size, color) {{
                        // G-protein (Ras-like): small globular with switch regions
                        const group = new THREE.Group();

                        // Main body
                        const bodyGeom = new THREE.SphereGeometry(size, 16, 16);
                        const bodyMat = new THREE.MeshStandardMaterial({{ color: color, roughness: 0.35 }});
                        group.add(new THREE.Mesh(bodyGeom, bodyMat));

                        // Switch I region
                        const sw1Mat = new THREE.MeshStandardMaterial({{ color: 0xffffff, roughness: 0.4 }});
                        const sw1 = applyCapsuleMaterial(createCapsule(size * 0.2, size * 0.4, 8, 8), sw1Mat);
                        sw1.position.set(size * 0.7, size * 0.3, 0);
                        sw1.rotation.z = Math.PI / 3;
                        group.add(sw1);

                        // Switch II region
                        const sw2Mat = new THREE.MeshStandardMaterial({{ color: 0xdddddd, roughness: 0.4 }});
                        const sw2 = applyCapsuleMaterial(createCapsule(size * 0.15, size * 0.3, 8, 8), sw2Mat);
                        sw2.position.set(size * 0.5, -size * 0.5, size * 0.3);
                        group.add(sw2);

                        return group;
                    }}

                    function createTranscriptionFactorGeometry(size, color) {{
                        // Transcription factor with DNA-binding domain
                        const group = new THREE.Group();

                        // DNA-binding domain (helix-turn-helix) - using cylinders for r128 compat
                        const helixMat = new THREE.MeshStandardMaterial({{ color: color, roughness: 0.3 }});
                        const helix1 = applyCapsuleMaterial(createCapsule(size * 0.3, size * 0.8, 8, 8), helixMat);
                        helix1.rotation.z = Math.PI / 6;
                        helix1.position.set(-size * 0.3, 0, 0);
                        group.add(helix1);

                        const helixMat2 = new THREE.MeshStandardMaterial({{ color: color, roughness: 0.3 }});
                        const helix2 = applyCapsuleMaterial(createCapsule(size * 0.3, size * 0.8, 8, 8), helixMat2);
                        helix2.rotation.z = -Math.PI / 6;
                        helix2.position.set(size * 0.3, 0, 0);
                        group.add(helix2);

                        // Dimerization domain
                        const dimerGeom = new THREE.SphereGeometry(size * 0.5, 12, 12);
                        const dimerMat = new THREE.MeshStandardMaterial({{ color: color, roughness: 0.35 }});
                        const dimer = new THREE.Mesh(dimerGeom, dimerMat);
                        dimer.position.y = -size * 0.6;
                        group.add(dimer);

                        return group;
                    }}

                    function createCaspaseGeometry(size, color) {{
                        // Caspase: heterodimer with active site
                        const group = new THREE.Group();

                        // Large subunit
                        const largeGeom = new THREE.BoxGeometry(size * 1.2, size * 0.8, size * 0.6);
                        const largeMat = new THREE.MeshStandardMaterial({{ color: color, roughness: 0.4 }});
                        const large = new THREE.Mesh(largeGeom, largeMat);
                        large.position.y = size * 0.3;
                        group.add(large);

                        // Small subunit
                        const smallGeom = new THREE.BoxGeometry(size * 0.8, size * 0.5, size * 0.5);
                        const smallMat = new THREE.MeshStandardMaterial({{ color: color, roughness: 0.4 }});
                        const small = new THREE.Mesh(smallGeom, smallMat);
                        small.position.y = -size * 0.3;
                        group.add(small);

                        // Active site cysteine (bright spot)
                        const activeGeom = new THREE.SphereGeometry(size * 0.15, 8, 8);
                        const activeMat = new THREE.MeshStandardMaterial({{
                            color: 0xffff00,
                            emissive: 0xffff00,
                            emissiveIntensity: 0.5
                        }});
                        const active = new THREE.Mesh(activeGeom, activeMat);
                        active.position.set(size * 0.4, 0, size * 0.3);
                        group.add(active);

                        return group;
                    }}

                    function createCytochromeCGeometry(size) {{
                        // Cytochrome C: small heme protein
                        const group = new THREE.Group();

                        // Protein shell
                        const shellGeom = new THREE.SphereGeometry(size, 12, 12);
                        const shellMat = new THREE.MeshStandardMaterial({{
                            color: 0x00ffff,
                            transparent: true,
                            opacity: 0.7,
                            roughness: 0.3
                        }});
                        group.add(new THREE.Mesh(shellGeom, shellMat));

                        // Heme group (flat disk)
                        const hemeGeom = new THREE.CylinderGeometry(size * 0.5, size * 0.5, size * 0.1, 16);
                        const hemeMat = new THREE.MeshStandardMaterial({{
                            color: 0xff0000,
                            emissive: 0x880000,
                            emissiveIntensity: 0.3
                        }});
                        const heme = new THREE.Mesh(hemeGeom, hemeMat);
                        group.add(heme);

                        // Iron center
                        const feGeom = new THREE.SphereGeometry(size * 0.1, 8, 8);
                        const feMat = new THREE.MeshStandardMaterial({{ color: 0xff4400 }});
                        group.add(new THREE.Mesh(feGeom, feMat));

                        return group;
                    }}

                    // ═══════════════════════════════════════════════════════
                    // CELL MEMBRANE (lipid bilayer representation)
                    // ═══════════════════════════════════════════════════════
                    const membraneGeom = new THREE.SphereGeometry(CELL_RADIUS, 64, 64);
                    const membraneMat = new THREE.MeshStandardMaterial({{
                        color: 0x4488ff,
                        transparent: true,
                        opacity: 0.12,
                        roughness: 0.3,
                        metalness: 0.1,
                        side: THREE.DoubleSide,
                    }});
                    scene.add(new THREE.Mesh(membraneGeom, membraneMat));

                    // Lipid bilayer particles (phospholipid heads)
                    const lipidHeadGeom = new THREE.SphereGeometry(0.08, 6, 6);
                    const lipidHeadMat = new THREE.MeshStandardMaterial({{
                        color: 0x6699cc,
                        roughness: 0.5
                    }});
                    for (let i = 0; i < 800; i++) {{
                        const phi = Math.acos(2 * Math.random() - 1);
                        const theta = Math.random() * Math.PI * 2;

                        // Outer leaflet
                        const lipidOuter = new THREE.Mesh(lipidHeadGeom, lipidHeadMat);
                        lipidOuter.position.set(
                            (CELL_RADIUS + 0.05) * Math.sin(phi) * Math.cos(theta),
                            (CELL_RADIUS + 0.05) * Math.sin(phi) * Math.sin(theta),
                            (CELL_RADIUS + 0.05) * Math.cos(phi)
                        );
                        scene.add(lipidOuter);

                        // Inner leaflet
                        const lipidInner = new THREE.Mesh(lipidHeadGeom, lipidHeadMat);
                        lipidInner.position.set(
                            (CELL_RADIUS - 0.05) * Math.sin(phi) * Math.cos(theta),
                            (CELL_RADIUS - 0.05) * Math.sin(phi) * Math.sin(theta),
                            (CELL_RADIUS - 0.05) * Math.cos(phi)
                        );
                        scene.add(lipidInner);
                    }}

                    // ═══════════════════════════════════════════════════════
                    // NUCLEUS with DNA
                    // ═══════════════════════════════════════════════════════
                    const nucleusGeom = new THREE.SphereGeometry(NUCLEUS_RADIUS, 32, 32);
                    const nucleusMat = new THREE.MeshStandardMaterial({{
                        color: 0x220044,
                        transparent: true,
                        opacity: 0.75,
                        roughness: 0.5,
                    }});
                    scene.add(new THREE.Mesh(nucleusGeom, nucleusMat));

                    // Nuclear envelope pores
                    const poreGeom = new THREE.TorusGeometry(0.15, 0.05, 8, 16);
                    const poreMat = new THREE.MeshStandardMaterial({{ color: 0x443388 }});
                    for (let i = 0; i < 30; i++) {{
                        const phi = Math.acos(2 * Math.random() - 1);
                        const theta = Math.random() * Math.PI * 2;
                        const pore = new THREE.Mesh(poreGeom, poreMat);
                        pore.position.set(
                            NUCLEUS_RADIUS * Math.sin(phi) * Math.cos(theta),
                            NUCLEUS_RADIUS * Math.sin(phi) * Math.sin(theta),
                            NUCLEUS_RADIUS * Math.cos(phi)
                        );
                        pore.lookAt(0, 0, 0);
                        scene.add(pore);
                    }}

                    // DNA double helix inside nucleus
                    function createDNAHelix(startPos, length, turns) {{
                        const group = new THREE.Group();
                        const radius = 0.3;
                        const points1 = [], points2 = [];

                        for (let t = 0; t < turns * Math.PI * 2; t += 0.2) {{
                            const y = (t / (turns * Math.PI * 2)) * length - length/2;
                            points1.push(new THREE.Vector3(
                                radius * Math.cos(t),
                                y,
                                radius * Math.sin(t)
                            ));
                            points2.push(new THREE.Vector3(
                                radius * Math.cos(t + Math.PI),
                                y,
                                radius * Math.sin(t + Math.PI)
                            ));
                        }}

                        // Backbone strands
                        const curve1 = new THREE.CatmullRomCurve3(points1);
                        const curve2 = new THREE.CatmullRomCurve3(points2);
                        const tubeGeom1 = new THREE.TubeGeometry(curve1, 64, 0.04, 8, false);
                        const tubeGeom2 = new THREE.TubeGeometry(curve2, 64, 0.04, 8, false);
                        const backboneMat = new THREE.MeshStandardMaterial({{ color: 0x4444ff }});
                        group.add(new THREE.Mesh(tubeGeom1, backboneMat));
                        group.add(new THREE.Mesh(tubeGeom2, backboneMat));

                        // Base pairs (rungs)
                        const basePairColors = [0xff4444, 0x44ff44, 0xffff44, 0x44ffff]; // A, T, G, C
                        for (let i = 0; i < points1.length - 1; i += 3) {{
                            const p1 = points1[i];
                            const p2 = points2[i];
                            const midpoint = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);
                            const length = p1.distanceTo(p2);

                            const bpGeom = new THREE.CylinderGeometry(0.02, 0.02, length, 6);
                            const bpMat = new THREE.MeshStandardMaterial({{
                                color: basePairColors[i % 4],
                                emissive: basePairColors[i % 4],
                                emissiveIntensity: 0.2
                            }});
                            const bp = new THREE.Mesh(bpGeom, bpMat);
                            bp.position.copy(midpoint);
                            bp.lookAt(p1);
                            bp.rotateX(Math.PI / 2);
                            group.add(bp);
                        }}

                        group.position.copy(startPos);
                        return group;
                    }}

                    // Add several DNA segments in nucleus
                    for (let i = 0; i < 5; i++) {{
                        const r = Math.random() * (NUCLEUS_RADIUS - 1);
                        const theta = Math.random() * Math.PI * 2;
                        const phi = Math.acos(2 * Math.random() - 1);
                        const dna = createDNAHelix(
                            new THREE.Vector3(
                                r * Math.sin(phi) * Math.cos(theta),
                                r * Math.sin(phi) * Math.sin(theta),
                                r * Math.cos(phi)
                            ),
                            1.5 + Math.random(),
                            2 + Math.random() * 2
                        );
                        dna.rotation.set(Math.random() * Math.PI, Math.random() * Math.PI, Math.random() * Math.PI);
                        scene.add(dna);
                    }}

                    // Nucleolus
                    const nucleolusGeom = new THREE.SphereGeometry(0.8, 16, 16);
                    const nucleolusMat = new THREE.MeshStandardMaterial({{ color: 0x550088 }});
                    const nucleolus = new THREE.Mesh(nucleolusGeom, nucleolusMat);
                    nucleolus.position.set(0.5, 0.5, 0);
                    scene.add(nucleolus);

                    // ═══════════════════════════════════════════════════════
                    // EXTRA STRUCTURES (APC, mitochondria, etc.)
                    // ═══════════════════════════════════════════════════════
                    extraStructures.forEach(struct => {{
                        if (struct.type === 'APC') {{
                            // Antigen-presenting cell
                            const apcGeom = new THREE.SphereGeometry(struct.radius, 32, 32);
                            const apcMat = new THREE.MeshStandardMaterial({{
                                color: parseInt(struct.color.slice(1), 16),
                                transparent: true,
                                opacity: 0.4,
                                roughness: 0.5,
                            }});
                            const apc = new THREE.Mesh(apcGeom, apcMat);
                            apc.position.set(struct.x, struct.y, struct.z);
                            scene.add(apc);

                            // APC wireframe
                            const apcWire = new THREE.Mesh(
                                new THREE.SphereGeometry(struct.radius * 1.01, 16, 16),
                                new THREE.MeshBasicMaterial({{ color: 0x664466, wireframe: true, transparent: true, opacity: 0.3 }})
                            );
                            apcWire.position.copy(apc.position);
                            scene.add(apcWire);
                        }} else if (struct.type === 'mitochondria') {{
                            // Mitochondria with outer membrane, inner membrane + cristae (r128 compatible)
                            const mitoGroup = new THREE.Group();

                            // Outer membrane (elongated ellipsoid using scaled sphere)
                            const outerGeom = new THREE.SphereGeometry(struct.radius * 0.6, 16, 16);
                            const outerMat = new THREE.MeshStandardMaterial({{
                                color: parseInt(struct.color.slice(1), 16),
                                transparent: true,
                                opacity: 0.4,
                                roughness: 0.3,
                                side: THREE.DoubleSide
                            }});
                            const outer = new THREE.Mesh(outerGeom, outerMat);
                            outer.scale.y = 2.5;  // Elongate to capsule-like shape
                            mitoGroup.add(outer);

                            // Inner membrane (smaller, more opaque)
                            const innerGeom = new THREE.SphereGeometry(struct.radius * 0.45, 12, 12);
                            const innerMat = new THREE.MeshStandardMaterial({{
                                color: 0xcc5500,
                                transparent: true,
                                opacity: 0.6,
                                roughness: 0.4
                            }});
                            const inner = new THREE.Mesh(innerGeom, innerMat);
                            inner.scale.y = 2.2;
                            mitoGroup.add(inner);

                            // Cristae (folded inner membrane)
                            for (let c = 0; c < 5; c++) {{
                                const cristaGeom = new THREE.PlaneGeometry(struct.radius * 0.6, struct.radius * 0.3);
                                const cristaMat = new THREE.MeshStandardMaterial({{
                                    color: 0xff6600,
                                    transparent: true,
                                    opacity: 0.7,
                                    side: THREE.DoubleSide
                                }});
                                const crista = new THREE.Mesh(cristaGeom, cristaMat);
                                crista.position.y = (c - 2) * struct.radius * 0.35;
                                crista.rotation.x = Math.PI / 2 + (Math.random() - 0.5) * 0.3;
                                crista.rotation.z = Math.random() * 0.5;
                                mitoGroup.add(crista);
                            }}

                            // Matrix (inner space, very faint)
                            const matrixGeom = new THREE.SphereGeometry(struct.radius * 0.35, 8, 8);
                            const matrixMat = new THREE.MeshStandardMaterial({{
                                color: 0xffaa00,
                                transparent: true,
                                opacity: 0.2
                            }});
                            const matrix = new THREE.Mesh(matrixGeom, matrixMat);
                            matrix.scale.y = 2.0;
                            mitoGroup.add(matrix);

                            mitoGroup.position.set(struct.x, struct.y, struct.z);
                            mitoGroup.rotation.set(Math.random() * Math.PI, Math.random() * Math.PI, 0);
                            scene.add(mitoGroup);
                        }}
                    }});

                    // ═══════════════════════════════════════════════════════
                    // MOLECULES - Using realistic geometries based on type
                    // ═══════════════════════════════════════════════════════
                    const moleculeMeshes = [];
                    const bonds = [];

                    // Map molecule types to geometry creators
                    const receptorTypes = ['receptor', 'EGFR', 'TCR', 'Fas', 'CD4'];
                    const ligandTypes = ['ligand', 'EGF', 'FasL', 'MHC_peptide'];
                    const kinaseTypes = ['Raf', 'MEK', 'ERK', 'ZAP70', 'LAT'];
                    const gProteinTypes = ['Ras'];
                    const smallMolTypes = ['Ca2+', 'second_messenger'];
                    const tfTypes = ['transcription_factor', 'NFAT'];
                    const caspaseTypes = ['Caspase8', 'Caspase9', 'Caspase3'];
                    const cytoCTypes = ['CytC'];
                    const bcl2Types = ['Bcl2', 'Bax'];

                    molecules.forEach((mol, i) => {{
                        let mesh;
                        const color = parseInt(mol.color.slice(1), 16);

                        if (receptorTypes.includes(mol.type)) {{
                            // Transmembrane receptor with domains
                            mesh = createReceptorGeometry(mol.size);
                            // Orient outward from cell center
                            const pos = new THREE.Vector3(mol.x, mol.y, mol.z);
                            mesh.position.copy(pos);
                            mesh.lookAt(pos.clone().multiplyScalar(2));
                            mesh.rotateX(Math.PI / 2);
                            // Color adjustment
                            mesh.children.forEach(child => {{
                                if (child.material) {{
                                    child.material.color.setHex(color);
                                    if (mol.bound) {{
                                        child.material.emissive = new THREE.Color(color);
                                        child.material.emissiveIntensity = 0.4;
                                    }}
                                }}
                            }});
                        }} else if (ligandTypes.includes(mol.type)) {{
                            // Growth factor / ligand
                            mesh = createLigandGeometry(mol.size);
                            mesh.position.set(mol.x, mol.y, mol.z);
                            mesh.children.forEach(child => {{
                                if (child.material) child.material.color.setHex(color);
                            }});
                        }} else if (kinaseTypes.includes(mol.type)) {{
                            // Bilobed kinase structure
                            mesh = createKinaseGeometry(mol.size, color);
                            mesh.position.set(mol.x, mol.y, mol.z);
                            mesh.rotation.set(Math.random() * Math.PI, Math.random() * Math.PI, 0);
                        }} else if (gProteinTypes.includes(mol.type)) {{
                            // G-protein (Ras-like)
                            mesh = createGProteinGeometry(mol.size, color);
                            mesh.position.set(mol.x, mol.y, mol.z);
                        }} else if (smallMolTypes.includes(mol.type)) {{
                            // Small signaling molecules
                            mesh = createSmallMoleculeGeometry(mol.size, color);
                            mesh.position.set(mol.x, mol.y, mol.z);
                        }} else if (tfTypes.includes(mol.type)) {{
                            // Transcription factors
                            mesh = createTranscriptionFactorGeometry(mol.size, color);
                            mesh.position.set(mol.x, mol.y, mol.z);
                        }} else if (caspaseTypes.includes(mol.type)) {{
                            // Caspases (apoptosis)
                            mesh = createCaspaseGeometry(mol.size, color);
                            mesh.position.set(mol.x, mol.y, mol.z);
                            mesh.rotation.set(Math.random() * Math.PI, Math.random() * Math.PI, 0);
                        }} else if (cytoCTypes.includes(mol.type)) {{
                            // Cytochrome C
                            mesh = createCytochromeCGeometry(mol.size);
                            mesh.position.set(mol.x, mol.y, mol.z);
                        }} else if (bcl2Types.includes(mol.type)) {{
                            // Bcl-2 family (use kinase-like structure)
                            mesh = createKinaseGeometry(mol.size * 0.8, color);
                            mesh.position.set(mol.x, mol.y, mol.z);
                        }} else {{
                            // Default: simple sphere for unrecognized types
                            const geom = new THREE.SphereGeometry(mol.size, 12, 12);
                            const mat = new THREE.MeshStandardMaterial({{
                                color: color,
                                emissive: color,
                                emissiveIntensity: mol.active ? 0.5 : 0.1,
                                roughness: 0.4,
                            }});
                            mesh = new THREE.Mesh(geom, mat);
                            mesh.position.set(mol.x, mol.y, mol.z);
                        }}

                        mesh.userData = {{
                            type: mol.type,
                            bound: mol.bound || false,
                            active: mol.active || false,
                            velocity: {{
                                x: (Math.random() - 0.5) * 0.02,
                                y: (Math.random() - 0.5) * 0.02,
                                z: (Math.random() - 0.5) * 0.02
                            }},
                            basePos: {{ x: mol.x, y: mol.y, z: mol.z }},
                            isGroup: mesh.isGroup || false
                        }};
                        scene.add(mesh);
                        moleculeMeshes.push(mesh);

                        // If receptor is bound, create a bond line to show ligand interaction
                        if (mol.bound && SHOW_BONDS) {{
                            const bondMat = new THREE.LineBasicMaterial({{ color: 0xffff00, transparent: true, opacity: 0.7, linewidth: 2 }});
                            const bondGeom = new THREE.BufferGeometry();
                            const outward = new THREE.Vector3(mol.x, mol.y, mol.z).normalize();
                            const bondEnd = outward.clone().multiplyScalar(CELL_RADIUS + 1.5);
                            const positions = new Float32Array([mol.x, mol.y, mol.z, bondEnd.x, bondEnd.y, bondEnd.z]);
                            bondGeom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                            const bond = new THREE.Line(bondGeom, bondMat);
                            scene.add(bond);
                            bonds.push({{ line: bond, receptor: mesh }});
                        }}
                    }});

                    // ═══════════════════════════════════════════════════════
                    // ANIMATION - handles both Mesh and Group objects
                    // ═══════════════════════════════════════════════════════
                    let time = 0;
                    let bindingEvents = 0;
                    let signalingEvents = 0;

                    // Helper to set emissive on mesh or group
                    function setEmissive(obj, intensity) {{
                        if (obj.isGroup) {{
                            obj.children.forEach(child => {{
                                if (child.material && child.material.emissive) {{
                                    child.material.emissiveIntensity = intensity;
                                }}
                            }});
                        }} else if (obj.material && obj.material.emissive) {{
                            obj.material.emissiveIntensity = intensity;
                        }}
                    }}

                    function animate() {{
                        requestAnimationFrame(animate);
                        time += 0.016;

                        // Animate molecules
                        moleculeMeshes.forEach((mesh, i) => {{
                            const data = mesh.userData;

                            // Ligands (EGF, FasL, etc.) diffuse toward cell
                            if (ligandTypes.includes(data.type) && !data.bound) {{
                                const pos = mesh.position;
                                const dist = pos.length();

                                // Brownian motion + drift toward membrane
                                pos.x += data.velocity.x * DIFFUSION_SPEED + (Math.random() - 0.5) * 0.05;
                                pos.y += data.velocity.y * DIFFUSION_SPEED + (Math.random() - 0.5) * 0.05;
                                pos.z += data.velocity.z * DIFFUSION_SPEED + (Math.random() - 0.5) * 0.05;

                                // Tumble rotation for realistic diffusion
                                if (mesh.isGroup) {{
                                    mesh.rotation.x += (Math.random() - 0.5) * 0.05;
                                    mesh.rotation.y += (Math.random() - 0.5) * 0.05;
                                }}

                                // Drift toward cell
                                if (dist > CELL_RADIUS + 0.5) {{
                                    const drift = 0.01 * REACTION_RATE;
                                    pos.x -= pos.x / dist * drift;
                                    pos.y -= pos.y / dist * drift;
                                    pos.z -= pos.z / dist * drift;
                                }}

                                // Check for binding
                                if (dist < CELL_RADIUS + 0.8 && Math.random() < 0.001 * REACTION_RATE) {{
                                    data.bound = true;
                                    bindingEvents++;
                                    // Snap to membrane
                                    const norm = pos.clone().normalize();
                                    pos.copy(norm.multiplyScalar(CELL_RADIUS + 0.3));
                                    setEmissive(mesh, 0.8);
                                }}
                            }}
                            // Kinases, G-proteins, and signaling molecules diffuse inside
                            else if (kinaseTypes.includes(data.type) || gProteinTypes.includes(data.type) ||
                                     smallMolTypes.includes(data.type) || caspaseTypes.includes(data.type) ||
                                     cytoCTypes.includes(data.type) || bcl2Types.includes(data.type)) {{
                                const pos = mesh.position;

                                // Brownian motion inside cell
                                pos.x += (Math.random() - 0.5) * 0.06 * DIFFUSION_SPEED;
                                pos.y += (Math.random() - 0.5) * 0.06 * DIFFUSION_SPEED;
                                pos.z += (Math.random() - 0.5) * 0.06 * DIFFUSION_SPEED;

                                // Tumble for realism
                                if (mesh.isGroup) {{
                                    mesh.rotation.x += (Math.random() - 0.5) * 0.02;
                                    mesh.rotation.y += (Math.random() - 0.5) * 0.02;
                                }}

                                // Keep inside cell, outside nucleus
                                const dist = pos.length();
                                if (dist > CELL_RADIUS - 0.8) {{
                                    pos.multiplyScalar((CELL_RADIUS - 1) / dist);
                                }}
                                if (dist < NUCLEUS_RADIUS + 0.5) {{
                                    pos.multiplyScalar((NUCLEUS_RADIUS + 0.6) / dist);
                                }}

                                // Random activation/deactivation (phosphorylation)
                                if (Math.random() < 0.002 * REACTION_RATE) {{
                                    data.active = !data.active;
                                    setEmissive(mesh, data.active ? 0.6 : 0.15);
                                    if (data.active) signalingEvents++;
                                }}
                            }}
                            // Transcription factors can enter nucleus when active
                            else if (tfTypes.includes(data.type)) {{
                                const pos = mesh.position;

                                // Brownian motion
                                pos.x += (Math.random() - 0.5) * 0.05 * DIFFUSION_SPEED;
                                pos.y += (Math.random() - 0.5) * 0.05 * DIFFUSION_SPEED;
                                pos.z += (Math.random() - 0.5) * 0.05 * DIFFUSION_SPEED;

                                const dist = pos.length();

                                // Random activation
                                if (Math.random() < 0.003 * REACTION_RATE) {{
                                    data.active = !data.active;
                                    setEmissive(mesh, data.active ? 0.7 : 0.15);
                                    if (data.active) signalingEvents++;
                                }}

                                // Active TFs translocate to nucleus
                                if (data.active && dist > NUCLEUS_RADIUS - 0.5) {{
                                    const toNucleus = new THREE.Vector3(-pos.x, -pos.y, -pos.z).normalize();
                                    pos.add(toNucleus.multiplyScalar(0.04));
                                }}

                                // Keep within bounds
                                if (dist > CELL_RADIUS - 0.5) {{
                                    pos.multiplyScalar((CELL_RADIUS - 0.6) / dist);
                                }}
                            }}
                            // Receptors stay on membrane but wiggle and rotate
                            else if (receptorTypes.includes(data.type)) {{
                                const base = data.basePos;
                                mesh.position.x = base.x + Math.sin(time * 2 + i) * 0.08;
                                mesh.position.y = base.y + Math.sin(time * 2.5 + i * 0.5) * 0.08;
                                mesh.position.z = base.z + Math.sin(time * 3 + i * 0.3) * 0.08;

                                // Keep on membrane
                                const pos = mesh.position;
                                const dist = pos.length();
                                pos.multiplyScalar(CELL_RADIUS / dist);

                                // Re-orient outward
                                mesh.lookAt(pos.clone().multiplyScalar(2));
                                mesh.rotateX(Math.PI / 2);
                            }}
                        }});

                        // Update stats
                        document.getElementById('stats').textContent =
                            `Bindings: ${{bindingEvents}} | Signaling: ${{signalingEvents}} | t=${{time.toFixed(1)}}s`;

                        controls.update();
                        renderer.render(scene, camera);
                    }}

                    animate();

                    window.addEventListener('resize', () => {{
                        camera.aspect = window.innerWidth / window.innerHeight;
                        camera.updateProjectionMatrix();
                        renderer.setSize(window.innerWidth, window.innerHeight);
                    }});
                </script>
            </body>
            </html>
            '''

            components.html(mol3d_html, height=600)

            st.caption("**This is what the simulation actually represents** — molecules diffusing, binding to receptors, and triggering signaling cascades inside the cell.")

    st.divider()

    # Explanation of what the visualization shows
    st.markdown("""
    ### What You're Seeing — Realistic Molecular Structures

    | Element | Shape | Description |
    |---------|-------|-------------|
    | **Cell membrane** | Sphere with lipid bilayer | Phospholipid heads visible as small spheres on both leaflets |
    | **Nuclear envelope** | Purple sphere with pores | Toroidal nuclear pore complexes for transport |
    | **DNA** | Double helix | Base pairs (A-T-G-C) as colored rungs, sugar-phosphate backbone |
    | **Receptors** | Y-shaped transmembrane | Extracellular binding domain + transmembrane helix + intracellular kinase domain |
    | **Ligands** | Compact globular + loop | Growth factors with binding loops (EGF, FasL) |
    | **Kinases** | Bilobed structure | N-lobe (ATP binding) + C-lobe (substrate binding) + active site cleft |
    | **G-proteins** | Sphere + switch regions | Ras-like with switch I/II conformational regions |
    | **Transcription factors** | Helix-turn-helix | DNA-binding domain + dimerization domain |
    | **Caspases** | Heterodimer + active site | Large + small subunit with yellow cysteine active site |
    | **Cytochrome C** | Sphere + heme disk | Protein shell with red heme group and iron center |
    | **Small molecules** | Octahedron (glowing) | Ca²⁺, ATP, second messengers |

    ### Molecular Dynamics

    - **Diffusion**: Molecules undergo Brownian motion with tumbling rotation
    - **Binding**: Ligands snap to membrane receptors, glow when bound
    - **Phosphorylation**: Kinases light up when active (phosphorylated)
    - **Nuclear translocation**: Active transcription factors migrate to nucleus

    ### How This Connects to VCell Solvers

    - **ODE Solver**: Tracks concentrations of all these molecule types over time
    - **Smoldyn**: Simulates the actual positions and movements shown here
    - **Hybrid**: Uses ODEs for abundant molecules, stochastic for rare ones
    - **BNGL**: Defines the rules for how molecules bind and react
    """)

# ══════════════════════════════════════════════════════════════════════
# TAB 1: ODE Solver
# ══════════════════════════════════════════════════════════════════════

with tab_ode:
    st.header("ODE Solver: Batched Deterministic Integration")

    st.markdown("""
    GPU-accelerated ODE integration for simulating **thousands of cells in parallel**.
    Each cell can have different parameters while sharing the same equation structure.

    **Use cases:**
    - Gene regulatory networks across cell populations
    - Drug response heterogeneity
    - Parameter sensitivity analysis
    """)

    col_config, col_viz = st.columns([1, 2])

    with col_config:
        st.subheader("Configuration")

        ode_system = st.selectbox(
            "System",
            ["Gene Expression (2 species)", "Michaelis-Menten Enzyme", "Toggle Switch"],
            key="ode_system"
        )

        n_cells = st.slider("Number of Cells", 100, 50000, 10000, step=1000, key="ode_n_cells")
        t_end = st.slider("Duration", 1.0, 50.0, 10.0, key="ode_t_end")
        method = st.selectbox("Integration Method", ["rk45", "bdf", "adams"], key="ode_method")

        heterogeneity = st.slider(
            "Parameter Heterogeneity (CV)",
            0.0, 0.5, 0.1,
            help="Coefficient of variation for per-cell parameter randomization"
        )

        run_ode = st.button("Run ODE Simulation", type="primary", key="run_ode")

    with col_viz:
        if run_ode:
            with st.spinner("Running ODE simulation..."):
                try:
                    sys.path.insert(0, str(Path(_project_root) / "cognisom"))
                    from gpu.ode_solver import BatchedODEIntegrator, ODESystem

                    # Create system
                    system = ODESystem.gene_expression_2species()
                    solver = BatchedODEIntegrator(system, n_cells=n_cells, method=method)

                    # Initial conditions with heterogeneity
                    y0 = np.zeros((n_cells, 2), dtype=np.float32)
                    y0[:, 0] = 10.0 * np.random.lognormal(0, heterogeneity, n_cells)
                    y0[:, 1] = 100.0 * np.random.lognormal(0, heterogeneity, n_cells)

                    # Run
                    t_eval = np.linspace(0, t_end, 101)
                    start = time.time()
                    solution = solver.integrate(t_span=(0, t_end), y0=y0, t_eval=t_eval)
                    elapsed = time.time() - start

                    # Display metrics
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Cells Simulated", f"{n_cells:,}")
                    m2.metric("Time Points", len(t_eval))
                    m3.metric("Runtime", f"{elapsed:.2f}s")

                    # Plot
                    fig = make_subplots(rows=1, cols=2, subplot_titles=("mRNA Distribution", "Protein Distribution"))

                    # Sample 100 cells for visualization
                    sample_idx = np.random.choice(n_cells, min(100, n_cells), replace=False)

                    for i in sample_idx:
                        fig.add_trace(go.Scatter(
                            x=t_eval, y=solution.y[:, i, 0],
                            mode='lines', line=dict(width=0.5, color='rgba(99,102,241,0.3)'),
                            showlegend=False
                        ), row=1, col=1)
                        fig.add_trace(go.Scatter(
                            x=t_eval, y=solution.y[:, i, 1],
                            mode='lines', line=dict(width=0.5, color='rgba(34,197,94,0.3)'),
                            showlegend=False
                        ), row=1, col=2)

                    # Mean trajectory
                    fig.add_trace(go.Scatter(
                        x=t_eval, y=solution.y[:, :, 0].mean(axis=1),
                        mode='lines', line=dict(width=3, color='#6366f1'),
                        name='Mean mRNA'
                    ), row=1, col=1)
                    fig.add_trace(go.Scatter(
                        x=t_eval, y=solution.y[:, :, 1].mean(axis=1),
                        mode='lines', line=dict(width=3, color='#22c55e'),
                        name='Mean Protein'
                    ), row=1, col=2)

                    fig.update_layout(height=400, showlegend=True)
                    fig.update_xaxes(title_text="Time")
                    fig.update_yaxes(title_text="Concentration")
                    st.plotly_chart(fig, use_container_width=True)

                    # Statistics
                    st.caption(f"Final mRNA: {solution.y[-1, :, 0].mean():.2f} ± {solution.y[-1, :, 0].std():.2f}")
                    st.caption(f"Final Protein: {solution.y[-1, :, 1].mean():.2f} ± {solution.y[-1, :, 1].std():.2f}")

                except Exception as e:
                    st.error(f"ODE solver error: {e}")
                    st.info("Make sure you're running on a GPU-enabled server (Brev L40S)")
        else:
            st.info("Configure parameters and click 'Run ODE Simulation' to start.")

# ══════════════════════════════════════════════════════════════════════
# TAB 2: Smoldyn Spatial
# ══════════════════════════════════════════════════════════════════════

with tab_smoldyn:
    st.header("Smoldyn: Particle-Based Spatial Stochastic")

    st.markdown("""
    Simulate **individual molecules** diffusing in 3D space with bimolecular reactions.
    GPU-accelerated for 100K+ particles.

    **Use cases:**
    - Single-molecule tracking
    - Receptor-ligand binding kinetics
    - Spatial pattern formation
    """)

    col_config, col_viz = st.columns([1, 2])

    with col_config:
        st.subheader("Configuration")

        n_particles_a = st.slider("Particles A", 1000, 50000, 10000, key="smol_a")
        n_particles_b = st.slider("Particles B", 1000, 50000, 10000, key="smol_b")
        diffusion_a = st.slider("Diffusion A (μm²/s)", 0.1, 10.0, 1.0, key="smol_diff_a")
        diffusion_b = st.slider("Diffusion B (μm²/s)", 0.1, 10.0, 0.5, key="smol_diff_b")
        domain_size = st.slider("Domain Size (μm)", 5.0, 50.0, 10.0, key="smol_domain")
        n_steps = st.slider("Simulation Steps", 10, 500, 100, key="smol_steps")

        run_smoldyn = st.button("Run Smoldyn Simulation", type="primary", key="run_smoldyn")

    with col_viz:
        if run_smoldyn:
            with st.spinner("Running Smoldyn simulation..."):
                try:
                    sys.path.insert(0, str(Path(_project_root) / "cognisom"))
                    from gpu.smoldyn_solver import (
                        SmoldynSolver, SmoldynSystem, SmoldynSpecies,
                        SmoldynCompartment, BoundaryType
                    )

                    # Create system
                    species = [
                        SmoldynSpecies(name='A', diffusion_coeff=diffusion_a, color='red'),
                        SmoldynSpecies(name='B', diffusion_coeff=diffusion_b, color='blue'),
                    ]
                    compartment = SmoldynCompartment(
                        name='box',
                        bounds=(0, domain_size, 0, domain_size, 0, domain_size),
                        boundary_type=BoundaryType.REFLECT
                    )
                    system = SmoldynSystem(species=species, reactions=[], compartment=compartment)

                    solver = SmoldynSolver(system, n_max_particles=n_particles_a + n_particles_b + 10000)

                    # Add particles
                    pos_a = np.random.uniform(0, domain_size, (n_particles_a, 3)).astype(np.float32)
                    pos_b = np.random.uniform(0, domain_size, (n_particles_b, 3)).astype(np.float32)
                    solver.add_particles('A', pos_a)
                    solver.add_particles('B', pos_b)

                    # Run simulation
                    start = time.time()
                    for _ in range(n_steps):
                        solver.step(0.001)
                    elapsed = time.time() - start

                    # Get final positions
                    final_pos_a = solver.get_positions('A')
                    final_pos_b = solver.get_positions('B')

                    # Display metrics
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Particles", f"{n_particles_a + n_particles_b:,}")
                    m2.metric("Steps", n_steps)
                    m3.metric("Runtime", f"{elapsed:.3f}s")

                    # 3D scatter plot
                    fig = go.Figure()

                    # Sample for visualization
                    max_show = 5000
                    if len(final_pos_a) > max_show:
                        idx_a = np.random.choice(len(final_pos_a), max_show, replace=False)
                        show_a = final_pos_a[idx_a]
                    else:
                        show_a = final_pos_a

                    if len(final_pos_b) > max_show:
                        idx_b = np.random.choice(len(final_pos_b), max_show, replace=False)
                        show_b = final_pos_b[idx_b]
                    else:
                        show_b = final_pos_b

                    fig.add_trace(go.Scatter3d(
                        x=show_a[:, 0], y=show_a[:, 1], z=show_a[:, 2],
                        mode='markers', marker=dict(size=2, color='#ef4444', opacity=0.6),
                        name=f'A ({solver.count_species("A")})'
                    ))
                    fig.add_trace(go.Scatter3d(
                        x=show_b[:, 0], y=show_b[:, 1], z=show_b[:, 2],
                        mode='markers', marker=dict(size=2, color='#3b82f6', opacity=0.6),
                        name=f'B ({solver.count_species("B")})'
                    ))

                    fig.update_layout(
                        height=500,
                        scene=dict(
                            xaxis_title='X (μm)',
                            yaxis_title='Y (μm)',
                            zaxis_title='Z (μm)',
                            aspectmode='cube'
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Smoldyn solver error: {e}")
        else:
            st.info("Configure parameters and click 'Run Smoldyn Simulation' to start.")

# ══════════════════════════════════════════════════════════════════════
# TAB 3: Hybrid ODE/SSA
# ══════════════════════════════════════════════════════════════════════

with tab_hybrid:
    st.header("Hybrid ODE/SSA: Automatic Partitioning")

    st.markdown("""
    Combines **deterministic ODE** for high-copy species with **stochastic SSA** for low-copy species.
    Automatic partitioning based on copy number threshold.

    **Use cases:**
    - Gene regulatory networks with bursty transcription
    - Systems with widely varying species abundances
    - Accurate noise modeling with computational efficiency
    """)

    col_config, col_viz = st.columns([1, 2])

    with col_config:
        st.subheader("Configuration")

        hybrid_system = st.selectbox(
            "System",
            ["Gene Regulatory Network", "Toggle Switch", "Enzyme-Substrate (MM)"],
            key="hybrid_system"
        )

        hybrid_n_cells = st.slider("Number of Cells", 100, 10000, 1000, key="hybrid_n_cells")
        hybrid_threshold = st.slider(
            "Copy Number Threshold",
            10, 500, 100,
            help="Species above this threshold use ODE, below use SSA"
        )
        hybrid_steps = st.slider("Simulation Steps", 10, 200, 50, key="hybrid_steps")

        run_hybrid = st.button("Run Hybrid Simulation", type="primary", key="run_hybrid")

    with col_viz:
        if run_hybrid:
            with st.spinner("Running Hybrid ODE/SSA simulation..."):
                try:
                    sys.path.insert(0, str(Path(_project_root) / "cognisom"))
                    from gpu.hybrid_solver import HybridSolver, HybridSystem

                    # Create system
                    if hybrid_system == "Toggle Switch":
                        system = HybridSystem.toggle_switch()
                    elif hybrid_system == "Enzyme-Substrate (MM)":
                        system = HybridSystem.enzyme_substrate_mm()
                    else:
                        system = HybridSystem.gene_regulatory_network()

                    solver = HybridSolver(system, n_cells=hybrid_n_cells, threshold=hybrid_threshold)
                    solver.initialize()

                    # Run simulation and record
                    start = time.time()
                    history = []
                    for i in range(hybrid_steps):
                        solver.step(0.1)
                        stats = solver.get_statistics()
                        history.append({
                            't': (i + 1) * 0.1,
                            **{f'{name}_mean': stats.get(f'{name}_mean', 0) for name in system.species_names},
                            **{f'{name}_std': stats.get(f'{name}_std', 0) for name in system.species_names},
                        })
                    elapsed = time.time() - start

                    # Display metrics
                    partition = solver.get_partition()
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Cells", f"{hybrid_n_cells:,}")
                    m2.metric("Fast Species (ODE)", partition.n_fast)
                    m3.metric("Slow Species (SSA)", partition.n_slow)
                    m4.metric("Runtime", f"{elapsed:.2f}s")

                    # Plot time series
                    fig = go.Figure()
                    times = [h['t'] for h in history]

                    colors = ['#6366f1', '#22c55e', '#f59e0b', '#ef4444']
                    for i, name in enumerate(system.species_names[:4]):
                        means = [h[f'{name}_mean'] for h in history]
                        stds = [h[f'{name}_std'] for h in history]

                        # Mean line
                        fig.add_trace(go.Scatter(
                            x=times, y=means,
                            mode='lines', name=name,
                            line=dict(color=colors[i % len(colors)], width=2)
                        ))

                        # Std band
                        upper = [m + s for m, s in zip(means, stds)]
                        lower = [m - s for m, s in zip(means, stds)]
                        fig.add_trace(go.Scatter(
                            x=times + times[::-1],
                            y=upper + lower[::-1],
                            fill='toself',
                            fillcolor=colors[i % len(colors)].replace(')', ', 0.2)').replace('rgb', 'rgba'),
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo='skip'
                        ))

                    fig.update_layout(
                        height=400,
                        xaxis_title='Time',
                        yaxis_title='Copy Number',
                        legend=dict(orientation='h', yanchor='bottom', y=1.02)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Partition info
                    st.caption(f"**Partition:** Fast={list(system.species_names[i] for i in partition.fast_species)}, "
                              f"Slow={list(system.species_names[i] for i in partition.slow_species)}")

                except Exception as e:
                    st.error(f"Hybrid solver error: {e}")
        else:
            st.info("Configure parameters and click 'Run Hybrid Simulation' to start.")

# ══════════════════════════════════════════════════════════════════════
# TAB 4: BNGL Rules
# ══════════════════════════════════════════════════════════════════════

with tab_bngl:
    st.header("BNGL: Rule-Based Modeling")

    st.markdown("""
    Handle **combinatorial complexity** in signaling pathways using reaction rules.
    Automatic network generation from compact specifications.

    **Use cases:**
    - Receptor signaling with multiple phosphorylation sites
    - Protein-protein interaction networks
    - Systems with many possible states
    """)

    col_config, col_viz = st.columns([1, 2])

    with col_config:
        st.subheader("Model Selection")

        bngl_model = st.selectbox(
            "Pre-defined Model",
            ["Simple Receptor (L + R ↔ LR)", "EGFR Signaling (dimerization + phosphorylation)"],
            key="bngl_model"
        )

        st.subheader("Or Enter BNGL Code")

        default_bngl = """begin parameters
    L_tot 1000
    R_tot 1000
    kon 1e-3
    koff 0.1
end parameters

begin molecule types
    L(r)
    R(l)
end molecule types

begin seed species
    L(r) L_tot
    R(l) R_tot
end seed species

begin observables
    Molecules L_free L(r)
    Species LR_complex L(r!1).R(l!1)
end observables

begin reaction rules
    L(r) + R(l) <-> L(r!1).R(l!1) kon, koff
end reaction rules"""

        bngl_code = st.text_area("BNGL Model", value=default_bngl, height=300, key="bngl_code")

        load_bngl = st.button("Load & Analyze Model", type="primary", key="load_bngl")

    with col_viz:
        if load_bngl:
            try:
                sys.path.insert(0, str(Path(_project_root) / "cognisom"))
                from bngl import BNGLModel, BNGLParser

                if "Simple Receptor" in bngl_model:
                    model = BNGLModel.simple_receptor()
                else:
                    model = BNGLModel.egfr_signaling()

                st.success(f"Model loaded: **{model.name}**")

                # Display model info
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Parameters", model.n_parameters)
                col_b.metric("Molecule Types", model.n_molecule_types)
                col_c.metric("Rules", model.n_rules)

                # Parameters table
                st.subheader("Parameters")
                param_data = [{"Name": k, "Value": v} for k, v in model.parameters.items()]
                st.dataframe(param_data, use_container_width=True)

                # Molecule types
                st.subheader("Molecule Types")
                for name, mol_type in model.molecule_types.items():
                    components = ", ".join([c.name + (f"~{'~'.join(s.name for s in c.states)}" if c.states else "")
                                           for c in mol_type.components])
                    st.code(f"{name}({components})")

                # Rules
                st.subheader("Reaction Rules")
                for rule in model.rules:
                    st.code(str(rule))

                # Observables
                st.subheader("Observables")
                for obs in model.observables.observables:
                    st.code(f"{obs.obs_type.name} {obs.name}: {' + '.join(str(p) for p in obs.patterns)}")

            except Exception as e:
                st.error(f"BNGL parsing error: {e}")
        else:
            st.info("Select a model or enter BNGL code and click 'Load & Analyze Model'.")

# ══════════════════════════════════════════════════════════════════════
# TAB 5: Imaging Pipeline
# ══════════════════════════════════════════════════════════════════════

with tab_imaging:
    st.header("Imaging Pipeline: Image to Geometry")

    st.markdown("""
    Convert **microscopy images** into simulation-ready geometries.
    GPU-accelerated preprocessing and segmentation.

    **Capabilities:**
    - Cell segmentation (Otsu, watershed, Cellpose, StarDist)
    - 3D mesh generation (marching cubes)
    - Multi-format import (TIFF, CZI, ND2, OME-TIFF)
    """)

    col_config, col_viz = st.columns([1, 2])

    with col_config:
        st.subheader("Demo: GPU Image Processing")

        image_size = st.slider("Image Size", 128, 1024, 256, step=128, key="img_size")
        sigma = st.slider("Gaussian Blur σ", 0.5, 5.0, 2.0, key="img_sigma")

        st.subheader("Segmentation Method")
        seg_method = st.selectbox(
            "Method",
            ["otsu", "watershed"],
            help="Cellpose and StarDist require additional dependencies"
        )

        run_imaging = st.button("Run Imaging Pipeline", type="primary", key="run_imaging")

    with col_viz:
        if run_imaging:
            with st.spinner("Running imaging pipeline..."):
                try:
                    sys.path.insert(0, str(Path(_project_root) / "cognisom"))
                    from imaging import CellSegmenter, GPUImageProcessor

                    # Create test image with cell-like objects
                    np.random.seed(42)
                    img = np.zeros((image_size, image_size), dtype=np.float32)

                    # Add random cell-like blobs
                    n_cells = 20
                    for _ in range(n_cells):
                        cx, cy = np.random.randint(20, image_size-20, 2)
                        r = np.random.randint(10, 30)
                        y, x = np.ogrid[:image_size, :image_size]
                        mask = ((x - cx)**2 + (y - cy)**2) < r**2
                        img[mask] = np.random.uniform(0.6, 1.0)

                    # Add noise
                    img += np.random.normal(0, 0.1, img.shape).astype(np.float32)
                    img = np.clip(img, 0, 1)

                    # GPU preprocessing
                    proc = GPUImageProcessor()
                    start = time.time()
                    blurred = proc.gaussian_blur(img, sigma=sigma)
                    binary = proc.threshold_otsu(blurred)
                    preproc_time = time.time() - start

                    # Segmentation
                    segmenter = CellSegmenter(method=seg_method)
                    start = time.time()
                    result = segmenter.segment(blurred)
                    seg_time = time.time() - start

                    # Metrics
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Cells Detected", result.n_cells)
                    m2.metric("Preprocessing", f"{preproc_time*1000:.1f}ms")
                    m3.metric("Segmentation", f"{seg_time*1000:.1f}ms")

                    # Visualization
                    fig = make_subplots(
                        rows=1, cols=3,
                        subplot_titles=("Original", "Preprocessed", "Segmentation")
                    )

                    fig.add_trace(go.Heatmap(z=img, colorscale='gray', showscale=False), row=1, col=1)
                    fig.add_trace(go.Heatmap(z=blurred, colorscale='gray', showscale=False), row=1, col=2)
                    fig.add_trace(go.Heatmap(z=result.labels, colorscale='viridis', showscale=False), row=1, col=3)

                    fig.update_layout(height=350)
                    fig.update_xaxes(showticklabels=False)
                    fig.update_yaxes(showticklabels=False)
                    st.plotly_chart(fig, use_container_width=True)

                    # Cell properties
                    if result.cell_properties:
                        st.subheader("Detected Cells")
                        cell_data = [
                            {
                                "Cell": i + 1,
                                "Area": f"{p.area:.0f}",
                                "Centroid": f"({p.centroid[0]:.0f}, {p.centroid[1]:.0f})"
                            }
                            for i, p in enumerate(result.cell_properties[:10])
                        ]
                        st.dataframe(cell_data, use_container_width=True)

                except Exception as e:
                    st.error(f"Imaging pipeline error: {e}")
        else:
            st.info("Configure parameters and click 'Run Imaging Pipeline' to start.")

# ══════════════════════════════════════════════════════════════════════
# TAB 6: Documentation
# ══════════════════════════════════════════════════════════════════════

with tab_docs:
    st.header("Documentation")

    st.markdown("""
    ## VCell Parity Overview

    These solvers bring [VCell](https://vcell.org) capabilities to Cognisom with GPU acceleration:

    | Solver | VCell Equivalent | GPU Speedup | Key Files |
    |--------|------------------|-------------|-----------|
    | ODE Solver | CVODE | 10-50x | `gpu/ode_solver.py` |
    | Smoldyn | Smoldyn | 20-100x | `gpu/smoldyn_solver.py` |
    | Hybrid | Hybrid Solvers | 5-20x | `gpu/hybrid_solver.py` |
    | BNGL | BioNetGen | 1x (CPU) | `bngl/` |
    | Imaging | Image-based | 10-50x | `imaging/` |

    ## Integration with Entity Model

    VCell solvers integrate with Cognisom's entity model:

    - **`ParameterSet`** entities store kinetic parameters
    - **`SimulationScenario`** entities define complete simulation setups
    - **`PhysicsModelEntity`** references specific solver configurations

    ### Example: Creating a Simulation Scenario

    ```python
    from cognisom.library.models import SimulationScenario, ParameterSet

    # Define parameters
    params = ParameterSet(
        name="GRN_baseline",
        context="gene_regulatory_network",
        parameters={
            "k_transcription": 1.0,
            "k_translation": 10.0,
            "gamma_mrna": 0.1,
            "gamma_protein": 0.01,
        }
    )

    # Define scenario
    scenario = SimulationScenario(
        name="GRN_1000_cells",
        scenario_type="baseline",
        duration_hours=24.0,
        parameter_set_ids=[params.entity_id],
    )
    ```

    ## API Reference

    ### ODE Solver

    ```python
    from gpu.ode_solver import BatchedODEIntegrator, ODESystem

    system = ODESystem.gene_expression_2species()
    solver = BatchedODEIntegrator(system, n_cells=10000, method='rk45')
    solution = solver.integrate(t_span=(0, 10), y0=y0)
    ```

    ### Smoldyn Solver

    ```python
    from gpu.smoldyn_solver import SmoldynSolver, SmoldynSystem, SmoldynSpecies

    species = [SmoldynSpecies(name='A', diffusion_coeff=1.0)]
    system = SmoldynSystem(species=species, reactions=[], compartment=compartment)
    solver = SmoldynSolver(system, n_max_particles=100000)
    solver.add_particles('A', positions)
    solver.step(dt)
    ```

    ### Hybrid Solver

    ```python
    from gpu.hybrid_solver import HybridSolver, HybridSystem

    system = HybridSystem.gene_regulatory_network()
    solver = HybridSolver(system, n_cells=5000, threshold=100)
    solver.initialize()
    solver.step(dt)
    ```

    ### BNGL Parser

    ```python
    from bngl import BNGLModel, BNGLParser

    model = BNGLModel.egfr_signaling()
    # or
    parser = BNGLParser()
    model = parser.parse_file("model.bngl")
    ```

    ### Imaging Pipeline

    ```python
    from imaging import CellSegmenter, MeshGenerator, GPUImageProcessor

    proc = GPUImageProcessor()
    blurred = proc.gaussian_blur(image, sigma=2.0)
    binary = proc.threshold_otsu(blurred)

    segmenter = CellSegmenter(method='otsu')
    result = segmenter.segment(image)
    ```
    """)

# ── Footer ───────────────────────────────────────────────────────────

st.divider()
st.caption("VCell Parity Solvers — GPU-accelerated computational biology matching VCell capabilities")
