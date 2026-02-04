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

tab_live3d, tab_cells, tab_fields, tab_network, tab_inspect, tab_lineage, tab_omniverse, tab_export = st.tabs([
    "🎬 Live 3D", "Cell Population", "Spatial Fields", "Interaction Network",
    "Scientific Inspector", "Lineage Tree", "Omniverse/USD", "Export",
])

# ────────────────────────────────────────────────────────────────
# TAB 0: Live 3D Viewer (In-Browser, No Installation)
# ────────────────────────────────────────────────────────────────

with tab_live3d:
    st.subheader("Live 3D Cell Simulation")
    st.success("**In-Browser 3D** — No software installation required. Works on any device.")

    # Configuration
    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        n_cells_3d = st.slider("Number of Cells", 20, 300, 100, key="live3d_n")
    with col_cfg2:
        animation_speed = st.slider("Animation Speed", 0.1, 2.0, 0.5, key="live3d_speed")
    with col_cfg3:
        auto_rotate = st.checkbox("Auto-Rotate Camera", value=True, key="live3d_rotate")

    col_ct1, col_ct2, col_ct3, col_ct4 = st.columns(4)
    with col_ct1:
        show_cancer = st.checkbox("Cancer Cells", value=True, key="live3d_cancer")
    with col_ct2:
        show_normal = st.checkbox("Normal Cells", value=True, key="live3d_normal")
    with col_ct3:
        show_tcells = st.checkbox("T Cells", value=True, key="live3d_tcells")
    with col_ct4:
        show_dividing = st.checkbox("Dividing", value=True, key="live3d_dividing")

    # Build cell configuration for Three.js
    import random
    import math

    random.seed(42)
    cells_js = []

    cell_configs = []
    if show_cancer:
        cell_configs.append({"type": "cancer", "color": "0xff3333", "count": int(n_cells_3d * 0.3)})
    if show_normal:
        cell_configs.append({"type": "normal", "color": "0x3399ff", "count": int(n_cells_3d * 0.4)})
    if show_tcells:
        cell_configs.append({"type": "tcell", "color": "0x33ff66", "count": int(n_cells_3d * 0.2)})
    if show_dividing:
        cell_configs.append({"type": "dividing", "color": "0xff33ff", "count": int(n_cells_3d * 0.1)})

    for cfg in cell_configs:
        for i in range(cfg["count"]):
            theta = random.uniform(0, 2 * math.pi)
            phi = random.uniform(0, math.pi)
            r = random.uniform(5, 30)
            cells_js.append({
                "x": r * math.sin(phi) * math.cos(theta),
                "y": r * math.sin(phi) * math.sin(theta),
                "z": r * math.cos(phi),
                "radius": random.uniform(0.8, 2.0),
                "color": cfg["color"],
                "type": cfg["type"],
                "phase": random.uniform(0, 2 * math.pi),
            })

    cells_json = json.dumps(cells_js)
    auto_rotate_js = "true" if auto_rotate else "false"

    # Three.js HTML component
    threejs_html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ margin: 0; overflow: hidden; background: #0a0a15; }}
            canvas {{ display: block; }}
            #info {{
                position: absolute;
                top: 10px;
                left: 10px;
                color: #fff;
                font-family: monospace;
                font-size: 12px;
                background: rgba(0,0,0,0.5);
                padding: 8px 12px;
                border-radius: 4px;
            }}
            #legend {{
                position: absolute;
                bottom: 10px;
                left: 10px;
                color: #fff;
                font-family: monospace;
                font-size: 11px;
                background: rgba(0,0,0,0.5);
                padding: 8px 12px;
                border-radius: 4px;
            }}
            .legend-item {{ display: flex; align-items: center; margin: 4px 0; }}
            .legend-color {{ width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }}
        </style>
    </head>
    <body>
        <div id="info">Cells: {len(cells_js)} | Drag to rotate | Scroll to zoom</div>
        <div id="legend">
            <div class="legend-item"><div class="legend-color" style="background:#ff3333"></div>Cancer</div>
            <div class="legend-item"><div class="legend-color" style="background:#3399ff"></div>Normal</div>
            <div class="legend-item"><div class="legend-color" style="background:#33ff66"></div>T Cell</div>
            <div class="legend-item"><div class="legend-color" style="background:#ff33ff"></div>Dividing</div>
        </div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
        <script>
            const cells = {cells_json};
            const animSpeed = {animation_speed};
            const autoRotate = {auto_rotate_js};

            // Scene setup
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x0a0a15);

            const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.set(50, 30, 50);

            const renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            document.body.appendChild(renderer.domElement);

            // Controls
            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.autoRotate = autoRotate;
            controls.autoRotateSpeed = 0.5;

            // Lighting
            const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
            scene.add(ambientLight);

            const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
            directionalLight.position.set(50, 50, 50);
            scene.add(directionalLight);

            const pointLight = new THREE.PointLight(0x4488ff, 0.5, 100);
            pointLight.position.set(-20, 20, -20);
            scene.add(pointLight);

            // Cell meshes
            const cellMeshes = [];
            const geometry = new THREE.SphereGeometry(1, 16, 16);

            cells.forEach((cell, i) => {{
                const material = new THREE.MeshPhongMaterial({{
                    color: parseInt(cell.color),
                    shininess: 80,
                    transparent: true,
                    opacity: 0.85,
                }});

                const mesh = new THREE.Mesh(geometry, material);
                mesh.position.set(cell.x, cell.y, cell.z);
                mesh.scale.setScalar(cell.radius);
                mesh.userData = {{ basePos: {{ x: cell.x, y: cell.y, z: cell.z }}, phase: cell.phase, type: cell.type }};
                scene.add(mesh);
                cellMeshes.push(mesh);
            }});

            // Grid helper
            const gridHelper = new THREE.GridHelper(80, 20, 0x333355, 0x222233);
            gridHelper.position.y = -20;
            scene.add(gridHelper);

            // Axes helper (subtle)
            const axesHelper = new THREE.AxesHelper(10);
            axesHelper.position.set(-35, -19, -35);
            scene.add(axesHelper);

            // Animation loop
            let time = 0;
            function animate() {{
                requestAnimationFrame(animate);
                time += 0.016 * animSpeed;

                // Animate cells
                cellMeshes.forEach((mesh, i) => {{
                    const base = mesh.userData.basePos;
                    const phase = mesh.userData.phase;

                    // Gentle oscillation
                    const offset = Math.sin(time * 2 + phase) * 0.5;
                    mesh.position.x = base.x + offset * 0.3;
                    mesh.position.y = base.y + offset;
                    mesh.position.z = base.z + offset * 0.2;

                    // Pulsing for dividing cells
                    if (mesh.userData.type === 'dividing') {{
                        const pulse = 1 + Math.sin(time * 4 + phase) * 0.15;
                        mesh.scale.setScalar(mesh.scale.x * 0.95 + pulse * 0.05);
                    }}
                }});

                controls.update();
                renderer.render(scene, camera);
            }}

            animate();

            // Resize handler
            window.addEventListener('resize', () => {{
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            }});
        </script>
    </body>
    </html>
    '''

    import streamlit.components.v1 as components
    components.html(threejs_html, height=600)

    st.caption("**Controls:** Click and drag to rotate • Scroll to zoom • Right-click to pan")

    st.divider()

    # ── Detailed Cell View with Prim Editor ───────────────────────────
    st.markdown("### 🔬 Detailed Cell View — Prim Editor")
    st.info("""
    **How to use this editor:**
    1. Select a cell type to set the base colors
    2. Toggle organelles on/off in the visibility panel
    3. Adjust size, position, and color of each organelle type
    4. The 3D view updates in real-time with your changes
    5. Use mouse to rotate, scroll to zoom, right-click to pan
    """)

    # ── Row 1: Cell Type and Membrane Settings ──
    st.markdown("#### Cell Membrane")
    col_type, col_mem_size, col_mem_opacity, col_mem_color = st.columns(4)

    with col_type:
        cell_type_detail = st.selectbox(
            "Cell Type",
            ["Cancer Cell", "Normal Cell", "T Cell", "Dividing Cell"],
            key="detail_cell_type"
        )

    # Default colors based on cell type
    cell_presets = {
        "Cancer Cell": {"membrane": "#ff4444", "nucleus": "#990000"},
        "Normal Cell": {"membrane": "#4488ff", "nucleus": "#002266"},
        "T Cell": {"membrane": "#44ff66", "nucleus": "#006622"},
        "Dividing Cell": {"membrane": "#ff44ff", "nucleus": "#660066"},
    }
    preset = cell_presets[cell_type_detail]

    with col_mem_size:
        membrane_radius = st.slider("Membrane Radius", 5.0, 15.0, 10.0, 0.5, key="mem_radius")
    with col_mem_opacity:
        membrane_opacity = st.slider("Membrane Opacity", 0.1, 0.8, 0.3, 0.05, key="mem_opacity")
    with col_mem_color:
        membrane_color = st.color_picker("Membrane Color", preset["membrane"], key="mem_color")

    # ── Row 2: Organelle Visibility ──
    st.markdown("#### Organelle Visibility")
    col_v1, col_v2, col_v3, col_v4, col_v5 = st.columns(5)
    show_nucleus = col_v1.checkbox("Nucleus", True, key="show_nucleus")
    show_mitochondria = col_v2.checkbox("Mitochondria", True, key="show_mito")
    show_er = col_v3.checkbox("ER", True, key="show_er")
    show_golgi = col_v4.checkbox("Golgi", True, key="show_golgi")
    show_ribosomes = col_v5.checkbox("Ribosomes", False, key="show_ribo")

    # ── Row 3: Organelle Properties Editor ──
    st.markdown("#### Organelle Properties")

    # Initialize organelle config
    organelle_config = []

    # Nucleus settings
    if show_nucleus:
        with st.expander("Nucleus Settings", expanded=False):
            col_n1, col_n2, col_n3, col_n4 = st.columns(4)
            nuc_radius = col_n1.slider("Radius", 1.0, 5.0, 3.0, 0.1, key="nuc_radius")
            nuc_x = col_n2.slider("X Position", -5.0, 5.0, 0.0, 0.1, key="nuc_x")
            nuc_y = col_n3.slider("Y Position", -5.0, 5.0, 0.0, 0.1, key="nuc_y")
            nuc_color = col_n4.color_picker("Color", preset["nucleus"], key="nuc_color")
        organelle_config.append({
            "type": "nucleus", "color": nuc_color, "radius": nuc_radius,
            "x": nuc_x, "y": nuc_y, "z": 0
        })

    # Mitochondria settings
    if show_mitochondria:
        with st.expander("Mitochondria Settings", expanded=False):
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            mito_count = col_m1.slider("Count", 4, 16, 8, key="mito_count")
            mito_size = col_m2.slider("Size", 0.3, 1.5, 0.6, 0.1, key="mito_size")
            mito_distance = col_m3.slider("Distance from center", 3.0, 8.0, 5.0, 0.5, key="mito_dist")
            mito_color = col_m4.color_picker("Color", "#ff8800", key="mito_color")

        import math as m
        for i in range(mito_count):
            angle = i * 2 * m.pi / mito_count
            organelle_config.append({
                "type": "mitochondria", "color": mito_color,
                "radius": mito_size, "length": mito_size * 2.5,
                "x": m.cos(angle) * mito_distance,
                "y": m.sin(angle) * mito_distance,
                "z": (i % 3 - 1) * 2
            })

    # ER settings
    if show_er:
        with st.expander("Endoplasmic Reticulum Settings", expanded=False):
            col_e1, col_e2, col_e3, col_e4 = st.columns(4)
            er_x = col_e1.slider("X Position", -6.0, 6.0, -3.0, 0.5, key="er_x")
            er_y = col_e2.slider("Y Position", -6.0, 6.0, 2.0, 0.5, key="er_y")
            er_scale = col_e3.slider("Scale", 0.5, 2.0, 1.0, 0.1, key="er_scale")
            er_color = col_e4.color_picker("Color", "#8844ff", key="er_color")
        organelle_config.append({
            "type": "er", "color": er_color, "x": er_x, "y": er_y, "z": 0, "scale": er_scale
        })

    # Golgi settings
    if show_golgi:
        with st.expander("Golgi Apparatus Settings", expanded=False):
            col_g1, col_g2, col_g3, col_g4 = st.columns(4)
            golgi_x = col_g1.slider("X Position", -6.0, 6.0, 4.0, 0.5, key="golgi_x")
            golgi_y = col_g2.slider("Y Position", -6.0, 6.0, -2.0, 0.5, key="golgi_y")
            golgi_stacks = col_g3.slider("Stack Count", 3, 8, 5, key="golgi_stacks")
            golgi_color = col_g4.color_picker("Color", "#ffcc00", key="golgi_color")
        organelle_config.append({
            "type": "golgi", "color": golgi_color,
            "x": golgi_x, "y": golgi_y, "z": 1, "stacks": golgi_stacks
        })

    # Ribosome settings
    if show_ribosomes:
        with st.expander("Ribosome Settings", expanded=False):
            col_r1, col_r2, col_r3 = st.columns(3)
            ribo_count = col_r1.slider("Count", 10, 50, 20, key="ribo_count")
            ribo_size = col_r2.slider("Size", 0.1, 0.5, 0.2, 0.05, key="ribo_size")
            ribo_color = col_r3.color_picker("Color", "#00ffff", key="ribo_color")

        import random
        random.seed(42)  # Consistent positions
        for i in range(ribo_count):
            organelle_config.append({
                "type": "ribosome", "color": ribo_color, "radius": ribo_size,
                "x": random.uniform(-6, 6), "y": random.uniform(-6, 6), "z": random.uniform(-4, 4)
            })

    # ── Prim Hierarchy Display ──
    with st.expander("Prim Hierarchy (USD-style)", expanded=False):
        st.markdown("```")
        st.text(f"""/Cell_{cell_type_detail.replace(' ', '')}
  /Membrane (Sphere, r={membrane_radius:.1f}, opacity={membrane_opacity:.2f})
  /InnerMembrane (Sphere, r={membrane_radius - 0.5:.1f})
  /Cytoplasm (Points, count=500)""")
        if show_nucleus:
            st.text(f"  /Nucleus (Sphere, r={nuc_radius:.1f})")
            st.text("    /Nucleolus (Sphere, r=1.0)")
            st.text("    /NuclearPores (Points, count=30)")
        if show_mitochondria:
            st.text(f"  /Mitochondria (Group, count={mito_count})")
            for i in range(min(3, mito_count)):
                st.text(f"    /Mito_{i:02d} (Capsule)")
        if show_er:
            st.text("  /EndoplasmicReticulum (TubeGroup)")
        if show_golgi:
            st.text(f"  /GolgiApparatus (TorusStack, count={golgi_stacks})")
        if show_ribosomes:
            st.text(f"  /Ribosomes (SphereGroup, count={ribo_count})")
        st.markdown("```")

    organelles_json = json.dumps(organelle_config)

    # Convert hex colors to int for JS
    def hex_to_js(hex_color):
        """Convert #RRGGBB to 0xRRGGBB"""
        return "0x" + hex_color.lstrip("#")

    membrane_color_js = hex_to_js(membrane_color)

    detail_html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ margin: 0; overflow: hidden; background: #0a0a20; }}
            canvas {{ display: block; }}
            #cell-info {{
                position: absolute;
                top: 10px;
                left: 10px;
                color: #fff;
                font-family: 'Segoe UI', sans-serif;
                font-size: 14px;
                background: rgba(0,0,0,0.8);
                padding: 12px 16px;
                border-radius: 8px;
                border: 1px solid rgba(100,100,255,0.3);
                box-shadow: 0 4px 12px rgba(0,0,0,0.5);
            }}
            #cell-info h3 {{ margin: 0 0 8px 0; font-size: 16px; color: #88aaff; }}
            #stats {{
                position: absolute;
                bottom: 10px;
                left: 10px;
                color: #aaa;
                font-family: monospace;
                font-size: 11px;
                background: rgba(0,0,0,0.7);
                padding: 8px 12px;
                border-radius: 6px;
            }}
            #legend {{
                position: absolute;
                bottom: 10px;
                right: 10px;
                color: #fff;
                font-family: monospace;
                font-size: 11px;
                background: rgba(0,0,0,0.8);
                padding: 10px 14px;
                border-radius: 6px;
                border: 1px solid rgba(255,255,255,0.1);
            }}
            .leg-item {{ display: flex; align-items: center; margin: 4px 0; }}
            .leg-dot {{ width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; }}
        </style>
    </head>
    <body>
        <div id="cell-info">
            <h3>{cell_type_detail}</h3>
            <div>Diameter: {membrane_radius * 2:.0f} μm</div>
            <div>Organelles: {len(organelle_config)}</div>
            <div style="margin-top:8px;font-size:11px;color:#888">
                Drag to rotate | Scroll to zoom | Right-click to pan
            </div>
        </div>
        <div id="stats">Loading...</div>
        <div id="legend">
            <div style="font-weight:bold;margin-bottom:6px;color:#88aaff">Organelles</div>
            <div class="leg-item"><div class="leg-dot" style="background:{membrane_color}"></div>Membrane</div>
            <div class="leg-item"><div class="leg-dot" style="background:#ff8800"></div>Mitochondria</div>
            <div class="leg-item"><div class="leg-dot" style="background:#8844ff"></div>ER</div>
            <div class="leg-item"><div class="leg-dot" style="background:#ffcc00"></div>Golgi</div>
            <div class="leg-item"><div class="leg-dot" style="background:#00ffff"></div>Ribosomes</div>
        </div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
        <script>
            const MEMBRANE_COLOR = {membrane_color_js};
            const MEMBRANE_RADIUS = {membrane_radius};
            const MEMBRANE_OPACITY = {membrane_opacity};
            const organelles = {organelles_json};

            // Helper to parse color (handles both hex string and 0x format)
            function parseColor(c) {{
                if (typeof c === 'string' && c.startsWith('#')) {{
                    return parseInt(c.slice(1), 16);
                }}
                if (typeof c === 'string' && c.startsWith('0x')) {{
                    return parseInt(c, 16);
                }}
                return parseInt(c) || 0x888888;
            }}

            // Scene setup
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x0a0a20);

            const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 500);
            camera.position.set(25, 18, 25);
            camera.lookAt(0, 0, 0);

            const renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            document.body.appendChild(renderer.domElement);

            // Controls
            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.autoRotate = true;
            controls.autoRotateSpeed = 0.5;
            controls.target.set(0, 0, 0);

            // Lighting - enhanced for better visibility
            const ambientLight = new THREE.AmbientLight(0x606080, 0.8);
            scene.add(ambientLight);

            const keyLight = new THREE.DirectionalLight(0xffffff, 1.2);
            keyLight.position.set(15, 25, 20);
            scene.add(keyLight);

            const fillLight = new THREE.DirectionalLight(0x4488ff, 0.6);
            fillLight.position.set(-15, 10, -15);
            scene.add(fillLight);

            const backLight = new THREE.DirectionalLight(0xff8844, 0.4);
            backLight.position.set(0, -15, -20);
            scene.add(backLight);

            const pointLight = new THREE.PointLight(0xffffff, 0.5, 50);
            pointLight.position.set(0, 0, 0);
            scene.add(pointLight);

            // ============ CELL MEMBRANE ============
            // Outer membrane (translucent, visible from outside)
            const membraneGeom = new THREE.SphereGeometry(MEMBRANE_RADIUS, 64, 64);
            const membraneMat = new THREE.MeshStandardMaterial({{
                color: MEMBRANE_COLOR,
                transparent: true,
                opacity: MEMBRANE_OPACITY,
                roughness: 0.3,
                metalness: 0.1,
                side: THREE.FrontSide,
            }});
            const membrane = new THREE.Mesh(membraneGeom, membraneMat);
            scene.add(membrane);

            // Inner membrane glow
            const innerGeom = new THREE.SphereGeometry(MEMBRANE_RADIUS * 0.98, 48, 48);
            const innerMat = new THREE.MeshBasicMaterial({{
                color: MEMBRANE_COLOR,
                transparent: true,
                opacity: 0.08,
                side: THREE.BackSide,
            }});
            scene.add(new THREE.Mesh(innerGeom, innerMat));

            // ============ CYTOPLASM PARTICLES ============
            const cytoGeom = new THREE.BufferGeometry();
            const cytoCount = 600;
            const cytoPositions = new Float32Array(cytoCount * 3);
            const maxR = MEMBRANE_RADIUS * 0.85;
            for (let i = 0; i < cytoCount; i++) {{
                const r = Math.random() * maxR;
                const theta = Math.random() * Math.PI * 2;
                const phi = Math.acos(2 * Math.random() - 1);
                cytoPositions[i*3] = r * Math.sin(phi) * Math.cos(theta);
                cytoPositions[i*3+1] = r * Math.sin(phi) * Math.sin(theta);
                cytoPositions[i*3+2] = r * Math.cos(phi);
            }}
            cytoGeom.setAttribute('position', new THREE.BufferAttribute(cytoPositions, 3));
            const cytoMat = new THREE.PointsMaterial({{
                color: 0xaaccee,
                size: 0.12,
                transparent: true,
                opacity: 0.5
            }});
            scene.add(new THREE.Points(cytoGeom, cytoMat));

            // ============ ORGANELLES ============
            const organelleMeshes = [];

            organelles.forEach((org, idx) => {{
                let mesh;
                const color = parseColor(org.color);

                if (org.type === 'nucleus') {{
                    // Main nucleus sphere
                    const nucGeom = new THREE.SphereGeometry(org.radius, 32, 32);
                    const nucMat = new THREE.MeshStandardMaterial({{
                        color: color,
                        roughness: 0.4,
                        metalness: 0.1,
                    }});
                    mesh = new THREE.Mesh(nucGeom, nucMat);

                    // Nucleolus (bright spot inside)
                    const nuclGeom = new THREE.SphereGeometry(org.radius * 0.35, 16, 16);
                    const nuclMat = new THREE.MeshStandardMaterial({{
                        color: 0xffcc88,
                        roughness: 0.3,
                        emissive: 0x332200,
                        emissiveIntensity: 0.3,
                    }});
                    const nucleolus = new THREE.Mesh(nuclGeom, nuclMat);
                    nucleolus.position.set(org.radius * 0.3, org.radius * 0.2, 0);
                    mesh.add(nucleolus);

                    // Nuclear pores
                    for (let i = 0; i < 25; i++) {{
                        const poreGeom = new THREE.SphereGeometry(0.12, 6, 6);
                        const poreMat = new THREE.MeshBasicMaterial({{ color: 0x333344 }});
                        const pore = new THREE.Mesh(poreGeom, poreMat);
                        const t = Math.random() * Math.PI * 2;
                        const p = Math.acos(2 * Math.random() - 1);
                        const pr = org.radius * 1.01;
                        pore.position.set(
                            pr * Math.sin(p) * Math.cos(t),
                            pr * Math.sin(p) * Math.sin(t),
                            pr * Math.cos(p)
                        );
                        mesh.add(pore);
                    }}

                }} else if (org.type === 'mitochondria') {{
                    // Use SphereGeometry stretched as capsule (CapsuleGeometry may not exist in r128)
                    const mitoGeom = new THREE.SphereGeometry(org.radius, 12, 12);
                    mitoGeom.scale(1, 2.5, 1);  // Elongate
                    const mitoMat = new THREE.MeshStandardMaterial({{
                        color: color,
                        roughness: 0.5,
                        metalness: 0.1,
                    }});
                    mesh = new THREE.Mesh(mitoGeom, mitoMat);

                    // Random rotation
                    mesh.rotation.set(
                        Math.random() * Math.PI,
                        Math.random() * Math.PI,
                        Math.random() * Math.PI
                    );

                    // Inner cristae
                    for (let i = 0; i < 2; i++) {{
                        const cristaGeom = new THREE.PlaneGeometry(org.radius * 1.5, org.radius * 0.8);
                        const cristaMat = new THREE.MeshBasicMaterial({{
                            color: 0xcc6600,
                            side: THREE.DoubleSide,
                            transparent: true,
                            opacity: 0.6
                        }});
                        const crista = new THREE.Mesh(cristaGeom, cristaMat);
                        crista.rotation.y = Math.PI / 2;
                        crista.position.x = (i - 0.5) * org.radius * 0.6;
                        mesh.add(crista);
                    }}

                }} else if (org.type === 'er') {{
                    // ER as curved tubes
                    const group = new THREE.Group();
                    const scale = org.scale || 1.0;

                    for (let i = 0; i < 6; i++) {{
                        const points = [];
                        for (let j = 0; j <= 12; j++) {{
                            const t = j / 12;
                            points.push(new THREE.Vector3(
                                (i - 2.5) * 0.4 * scale,
                                (t - 0.5) * 4 * scale,
                                Math.sin(t * Math.PI * 2 + i * 0.5) * 0.8 * scale
                            ));
                        }}
                        const curve = new THREE.CatmullRomCurve3(points);
                        const tubeGeom = new THREE.TubeGeometry(curve, 16, 0.15 * scale, 8, false);
                        const tubeMat = new THREE.MeshStandardMaterial({{
                            color: color,
                            roughness: 0.4,
                            transparent: true,
                            opacity: 0.8
                        }});
                        group.add(new THREE.Mesh(tubeGeom, tubeMat));
                    }}
                    mesh = group;

                }} else if (org.type === 'golgi') {{
                    // Stacked curved discs
                    const group = new THREE.Group();
                    const stacks = org.stacks || 5;

                    for (let i = 0; i < stacks; i++) {{
                        const discGeom = new THREE.TorusGeometry(1.4 - i * 0.12, 0.25, 8, 20);
                        const discMat = new THREE.MeshStandardMaterial({{
                            color: color,
                            roughness: 0.4,
                        }});
                        const disc = new THREE.Mesh(discGeom, discMat);
                        disc.position.z = i * 0.35;
                        disc.rotation.x = Math.PI / 2;
                        // Slight curve
                        disc.position.y = Math.sin(i * 0.3) * 0.3;
                        group.add(disc);
                    }}
                    mesh = group;

                }} else if (org.type === 'ribosome') {{
                    const riboGeom = new THREE.SphereGeometry(org.radius, 8, 8);
                    const riboMat = new THREE.MeshStandardMaterial({{
                        color: color,
                        roughness: 0.3,
                        emissive: color,
                        emissiveIntensity: 0.2,
                    }});
                    mesh = new THREE.Mesh(riboGeom, riboMat);
                }}

                if (mesh) {{
                    mesh.position.set(org.x || 0, org.y || 0, org.z || 0);
                    mesh.userData = {{ type: org.type, index: idx }};
                    scene.add(mesh);
                    organelleMeshes.push(mesh);
                }}
            }});

            // Stats update
            const statsEl = document.getElementById('stats');
            let frameCount = 0;
            let lastTime = performance.now();
            let fps = 60;

            // Animation loop
            let time = 0;
            function animate() {{
                requestAnimationFrame(animate);
                time += 0.016;

                // Gentle membrane breathing
                const breathe = 1 + Math.sin(time * 0.5) * 0.008;
                membrane.scale.setScalar(breathe);

                // Subtle organelle movement
                organelleMeshes.forEach((m, i) => {{
                    if (m.userData.type === 'mitochondria') {{
                        m.rotation.z += 0.002;
                    }}
                }});

                controls.update();
                renderer.render(scene, camera);

                // FPS counter
                frameCount++;
                const now = performance.now();
                if (now - lastTime >= 1000) {{
                    fps = Math.round(frameCount * 1000 / (now - lastTime));
                    frameCount = 0;
                    lastTime = now;
                    statsEl.textContent = `FPS: ${{fps}} | Meshes: ${{scene.children.length}}`;
                }}
            }}

            animate();

            // Resize handler
            window.addEventListener('resize', () => {{
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            }});
        </script>
    </body>
    </html>
    '''

    components.html(detail_html, height=550)

    # Export configuration button
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        if st.button("Export Cell Configuration as JSON", key="export_cell_config"):
            config_data = {
                "cell_type": cell_type_detail,
                "membrane": {
                    "radius": membrane_radius,
                    "opacity": membrane_opacity,
                    "color": membrane_color,
                },
                "organelles": organelle_config,
            }
            st.download_button(
                "Download cell_config.json",
                json.dumps(config_data, indent=2),
                file_name="cell_config.json",
                mime="application/json",
                key="dl_cell_config"
            )
    with col_exp2:
        st.caption("Export your customized cell configuration for use in simulations or external 3D tools.")

    st.caption("**Detailed cell view** — Interactive 3D visualization with customizable organelles.")

    st.divider()

    # ══════════════════════════════════════════════════════════════════
    # BIOLOGICAL PROCESSES SIMULATIONS
    # ══════════════════════════════════════════════════════════════════
    st.markdown("## 🧬 Biological Process Simulations")
    st.info("Select a biological process to visualize realistic cellular dynamics")

    process_type = st.selectbox(
        "Select Process",
        [
            "Tissue Microenvironment (500+ cells)",
            "Central Dogma: DNA → RNA → Protein",
            "Viral Infection Cycle",
            "Apoptosis (Programmed Cell Death)",
            "Phagocytosis (Immune Response)",
        ],
        key="bio_process_select"
    )

    # ── Process-specific controls ──
    if process_type == "Tissue Microenvironment (500+ cells)":
        st.markdown("### Tumor Microenvironment")
        col_t1, col_t2, col_t3, col_t4 = st.columns(4)
        tissue_cells = col_t1.slider("Total Cells", 200, 1000, 500, 50, key="tissue_n")
        cancer_pct = col_t2.slider("Cancer %", 10, 80, 40, key="cancer_pct")
        immune_pct = col_t3.slider("Immune %", 5, 40, 20, key="immune_pct")
        hypoxic_core = col_t4.checkbox("Hypoxic Core", True, key="hypoxic")

    elif process_type == "Central Dogma: DNA → RNA → Protein":
        st.markdown("### Gene Expression: Transcription & Translation")
        col_d1, col_d2, col_d3 = st.columns(3)
        gene_name = col_d1.selectbox("Gene", ["TP53", "MYC", "BRCA1", "HER2", "EGFR"], key="gene_sel")
        show_ribosomes = col_d2.checkbox("Show Ribosomes", True, key="show_ribo_cd")
        animation_phase = col_d3.slider("Animation Phase", 0.0, 1.0, 0.5, 0.01, key="anim_phase")

    elif process_type == "Viral Infection Cycle":
        st.markdown("### Viral Entry, Replication & Lysis")
        col_v1, col_v2, col_v3 = st.columns(3)
        virus_type = col_v1.selectbox("Virus Type", ["SARS-CoV-2", "HIV", "Influenza", "Bacteriophage"], key="virus_type")
        infection_stage = col_v2.selectbox("Stage", ["Entry", "Replication", "Assembly", "Lysis"], key="inf_stage")
        virus_count = col_v3.slider("Viral Particles", 5, 50, 20, key="virus_n")

    elif process_type == "Apoptosis (Programmed Cell Death)":
        st.markdown("### Programmed Cell Death Cascade")
        col_a1, col_a2, col_a3 = st.columns(3)
        apoptosis_stage = col_a1.selectbox(
            "Stage",
            ["Healthy", "Initiation", "Chromatin Condensation", "Membrane Blebbing", "Apoptotic Bodies"],
            key="apop_stage"
        )
        trigger = col_a2.selectbox("Trigger", ["DNA Damage", "Death Receptor", "Mitochondrial"], key="apop_trigger")
        show_caspases = col_a3.checkbox("Show Caspase Cascade", True, key="show_casp")

    elif process_type == "Phagocytosis (Immune Response)":
        st.markdown("### Macrophage Engulfing Pathogen")
        col_p1, col_p2, col_p3 = st.columns(3)
        phago_stage = col_p1.selectbox(
            "Stage",
            ["Recognition", "Attachment", "Engulfment", "Phagosome Formation", "Digestion"],
            key="phago_stage"
        )
        target_type = col_p2.selectbox("Target", ["Bacteria", "Apoptotic Cell", "Cancer Cell"], key="phago_target")
        show_lysosomes = col_p3.checkbox("Show Lysosomes", True, key="show_lyso")

    # ── Build the Three.js visualization ──
    import random
    import math

    if process_type == "Tissue Microenvironment (500+ cells)":
        random.seed(42)
        tissue_data = []
        normal_count = int(tissue_cells * (100 - cancer_pct - immune_pct) / 100)
        cancer_count = int(tissue_cells * cancer_pct / 100)
        immune_count = tissue_cells - normal_count - cancer_count

        # Generate cells in tumor pattern
        for i in range(cancer_count):
            # Cancer cells cluster in center
            r = random.gauss(0, 15)
            theta = random.uniform(0, 2 * math.pi)
            phi = random.uniform(0, math.pi)
            tissue_data.append({
                "x": r * math.sin(phi) * math.cos(theta),
                "y": r * math.sin(phi) * math.sin(theta),
                "z": r * math.cos(phi),
                "type": "cancer",
                "color": "0xff3333",
                "radius": random.uniform(1.5, 2.5),
                "hypoxic": abs(r) < 8 if hypoxic_core else False,
            })

        for i in range(normal_count):
            # Normal cells form outer tissue
            r = random.uniform(20, 40)
            theta = random.uniform(0, 2 * math.pi)
            phi = random.uniform(0, math.pi)
            tissue_data.append({
                "x": r * math.sin(phi) * math.cos(theta),
                "y": r * math.sin(phi) * math.sin(theta),
                "z": r * math.cos(phi),
                "type": "normal",
                "color": "0x4488ff",
                "radius": random.uniform(1.0, 1.8),
                "hypoxic": False,
            })

        for i in range(immune_count):
            # Immune cells infiltrating
            r = random.uniform(5, 35)
            theta = random.uniform(0, 2 * math.pi)
            phi = random.uniform(0, math.pi)
            tissue_data.append({
                "x": r * math.sin(phi) * math.cos(theta),
                "y": r * math.sin(phi) * math.sin(theta),
                "z": r * math.cos(phi),
                "type": random.choice(["t_cell", "macrophage", "nk_cell"]),
                "color": random.choice(["0x33ff66", "0xffaa33", "0xffff33"]),
                "radius": random.uniform(0.8, 1.5),
                "hypoxic": False,
            })

        tissue_json = json.dumps(tissue_data)

        process_html = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ margin: 0; overflow: hidden; background: #050510; }}
                #info {{ position: absolute; top: 10px; left: 10px; color: #fff; font-family: monospace; font-size: 12px; background: rgba(0,0,0,0.8); padding: 12px; border-radius: 8px; }}
                #legend {{ position: absolute; bottom: 10px; right: 10px; color: #fff; font-family: monospace; font-size: 11px; background: rgba(0,0,0,0.8); padding: 10px; border-radius: 6px; }}
                .leg-item {{ display: flex; align-items: center; margin: 3px 0; }}
                .leg-dot {{ width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; }}
            </style>
        </head>
        <body>
            <div id="info">
                <div style="font-size:14px;font-weight:bold;margin-bottom:8px;color:#ff6666">Tumor Microenvironment</div>
                <div>Cells: {tissue_cells}</div>
                <div>Cancer: {cancer_count} ({cancer_pct}%)</div>
                <div>Immune: {immune_count}</div>
                <div>Normal: {normal_count}</div>
                {"<div style='color:#8844ff'>⚠ Hypoxic Core Active</div>" if hypoxic_core else ""}
            </div>
            <div id="legend">
                <div style="font-weight:bold;margin-bottom:6px">Cell Types</div>
                <div class="leg-item"><div class="leg-dot" style="background:#ff3333"></div>Cancer</div>
                <div class="leg-item"><div class="leg-dot" style="background:#4488ff"></div>Normal</div>
                <div class="leg-item"><div class="leg-dot" style="background:#33ff66"></div>T Cell</div>
                <div class="leg-item"><div class="leg-dot" style="background:#ffaa33"></div>Macrophage</div>
                <div class="leg-item"><div class="leg-dot" style="background:#ffff33"></div>NK Cell</div>
                {"<div class='leg-item'><div class='leg-dot' style='background:#660066'></div>Hypoxic</div>" if hypoxic_core else ""}
            </div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
            <script>
                const cells = {tissue_json};
                const hypoxicCore = {'true' if hypoxic_core else 'false'};

                const scene = new THREE.Scene();
                scene.background = new THREE.Color(0x050510);
                scene.fog = new THREE.Fog(0x050510, 60, 120);

                const camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.1, 500);
                camera.position.set(60, 45, 60);

                const renderer = new THREE.WebGLRenderer({{ antialias: true }});
                renderer.setSize(window.innerWidth, window.innerHeight);
                renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
                document.body.appendChild(renderer.domElement);

                const controls = new THREE.OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;
                controls.autoRotate = true;
                controls.autoRotateSpeed = 0.3;

                // Lighting
                scene.add(new THREE.AmbientLight(0x404060, 0.6));
                const sun = new THREE.DirectionalLight(0xffffff, 1.0);
                sun.position.set(30, 50, 30);
                scene.add(sun);
                scene.add(new THREE.PointLight(0xff4444, 0.5, 80));

                // Hypoxic core glow
                if (hypoxicCore) {{
                    const coreGlow = new THREE.PointLight(0x660066, 1.5, 20);
                    coreGlow.position.set(0, 0, 0);
                    scene.add(coreGlow);
                }}

                // Create cells
                const cellMeshes = [];
                const sphereGeo = new THREE.SphereGeometry(1, 12, 12);

                cells.forEach((c, i) => {{
                    let color = parseInt(c.color);
                    if (c.hypoxic) color = 0x660066;

                    const mat = new THREE.MeshStandardMaterial({{
                        color: color,
                        roughness: 0.5,
                        metalness: 0.1,
                        transparent: c.type === 'normal',
                        opacity: c.type === 'normal' ? 0.7 : 1.0,
                    }});

                    const mesh = new THREE.Mesh(sphereGeo, mat);
                    mesh.position.set(c.x, c.y, c.z);
                    mesh.scale.setScalar(c.radius);
                    mesh.userData = {{ type: c.type, phase: Math.random() * Math.PI * 2 }};
                    scene.add(mesh);
                    cellMeshes.push(mesh);
                }});

                // Blood vessels (tubes)
                for (let v = 0; v < 5; v++) {{
                    const points = [];
                    const startAngle = v * Math.PI * 2 / 5;
                    for (let t = 0; t <= 20; t++) {{
                        const tt = t / 20;
                        const r = 15 + tt * 25;
                        points.push(new THREE.Vector3(
                            Math.cos(startAngle + tt * 2) * r,
                            (tt - 0.5) * 40 + Math.sin(tt * 4) * 5,
                            Math.sin(startAngle + tt * 2) * r
                        ));
                    }}
                    const curve = new THREE.CatmullRomCurve3(points);
                    const tubeGeo = new THREE.TubeGeometry(curve, 30, 0.8, 8, false);
                    const tubeMat = new THREE.MeshStandardMaterial({{ color: 0x880022, roughness: 0.3 }});
                    scene.add(new THREE.Mesh(tubeGeo, tubeMat));
                }}

                let time = 0;
                function animate() {{
                    requestAnimationFrame(animate);
                    time += 0.016;

                    cellMeshes.forEach((m, i) => {{
                        const phase = m.userData.phase;
                        // Gentle movement
                        m.position.x += Math.sin(time + phase) * 0.01;
                        m.position.y += Math.cos(time * 0.7 + phase) * 0.01;

                        // Cancer cells divide (pulse)
                        if (m.userData.type === 'cancer') {{
                            const pulse = 1 + Math.sin(time * 2 + phase) * 0.05;
                            m.scale.setScalar(m.scale.x * 0.99 + pulse * 0.01 * cells[i].radius);
                        }}
                        // Immune cells move more
                        if (m.userData.type === 't_cell' || m.userData.type === 'macrophage') {{
                            m.position.x += Math.sin(time * 3 + phase) * 0.03;
                            m.position.z += Math.cos(time * 3 + phase) * 0.03;
                        }}
                    }});

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

    elif process_type == "Central Dogma: DNA → RNA → Protein":
        phase = animation_phase
        process_html = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ margin: 0; overflow: hidden; background: #0a0a20; }}
                #info {{ position: absolute; top: 10px; left: 10px; color: #fff; font-family: 'Segoe UI', sans-serif; font-size: 13px; background: rgba(0,0,0,0.85); padding: 15px; border-radius: 10px; max-width: 280px; }}
                #stage {{ position: absolute; top: 10px; right: 10px; color: #fff; font-family: monospace; font-size: 18px; background: linear-gradient(135deg, #1a1a4a, #2a2a6a); padding: 15px 25px; border-radius: 10px; border: 2px solid #4488ff; }}
            </style>
        </head>
        <body>
            <div id="info">
                <div style="font-size:16px;font-weight:bold;color:#88aaff;margin-bottom:10px">🧬 Gene: {gene_name}</div>
                <div style="margin-bottom:8px"><span style="color:#ff8844">DNA</span> → <span style="color:#44ff88">mRNA</span> → <span style="color:#ff44aa">Protein</span></div>
                <div style="font-size:11px;color:#888;margin-top:10px">
                    {"📖 Transcription: DNA → mRNA in nucleus" if phase < 0.4 else ""}
                    {"🔄 mRNA Export: Through nuclear pore" if 0.4 <= phase < 0.6 else ""}
                    {"🔧 Translation: Ribosome reads mRNA" if phase >= 0.6 else ""}
                </div>
            </div>
            <div id="stage">
                {"TRANSCRIPTION" if phase < 0.4 else "EXPORT" if phase < 0.6 else "TRANSLATION"}
            </div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
            <script>
                const phase = {phase};
                const showRibosomes = {'true' if show_ribosomes else 'false'};

                const scene = new THREE.Scene();
                scene.background = new THREE.Color(0x0a0a20);

                const camera = new THREE.PerspectiveCamera(50, window.innerWidth/window.innerHeight, 0.1, 200);
                camera.position.set(0, 15, 35);

                const renderer = new THREE.WebGLRenderer({{ antialias: true }});
                renderer.setSize(window.innerWidth, window.innerHeight);
                document.body.appendChild(renderer.domElement);

                const controls = new THREE.OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;

                // Lighting
                scene.add(new THREE.AmbientLight(0x404060, 0.8));
                scene.add(new THREE.DirectionalLight(0xffffff, 1.0)).position.set(10, 20, 15);

                // ═══════ NUCLEUS ═══════
                const nucleusGeo = new THREE.SphereGeometry(8, 48, 48);
                const nucleusMat = new THREE.MeshStandardMaterial({{
                    color: 0x334488,
                    transparent: true,
                    opacity: 0.3,
                    side: THREE.DoubleSide,
                }});
                const nucleus = new THREE.Mesh(nucleusGeo, nucleusMat);
                scene.add(nucleus);

                // Nuclear envelope (inner)
                const envGeo = new THREE.SphereGeometry(7.8, 32, 32);
                const envMat = new THREE.MeshBasicMaterial({{ color: 0x223366, transparent: true, opacity: 0.1, side: THREE.BackSide }});
                scene.add(new THREE.Mesh(envGeo, envMat));

                // ═══════ DNA DOUBLE HELIX ═══════
                const dnaGroup = new THREE.Group();
                const helixTurns = 4;
                const helixRadius = 1.5;
                const helixHeight = 10;

                for (let strand = 0; strand < 2; strand++) {{
                    const points = [];
                    for (let i = 0; i <= 100; i++) {{
                        const t = i / 100;
                        const angle = t * helixTurns * Math.PI * 2 + strand * Math.PI;
                        points.push(new THREE.Vector3(
                            Math.cos(angle) * helixRadius,
                            (t - 0.5) * helixHeight,
                            Math.sin(angle) * helixRadius
                        ));
                    }}
                    const curve = new THREE.CatmullRomCurve3(points);
                    const tubeGeo = new THREE.TubeGeometry(curve, 80, 0.15, 8, false);
                    const tubeMat = new THREE.MeshStandardMaterial({{ color: strand === 0 ? 0xff6644 : 0xff8866, roughness: 0.4 }});
                    dnaGroup.add(new THREE.Mesh(tubeGeo, tubeMat));
                }}

                // Base pairs
                for (let i = 0; i < 40; i++) {{
                    const t = i / 40;
                    const angle = t * helixTurns * Math.PI * 2;
                    const y = (t - 0.5) * helixHeight;

                    const pairGeo = new THREE.CylinderGeometry(0.08, 0.08, helixRadius * 1.8, 6);
                    const pairMat = new THREE.MeshStandardMaterial({{
                        color: i % 4 === 0 ? 0x44ff44 : i % 4 === 1 ? 0xff4444 : i % 4 === 2 ? 0x4444ff : 0xffff44
                    }});
                    const pair = new THREE.Mesh(pairGeo, pairMat);
                    pair.position.set(0, y, 0);
                    pair.rotation.z = Math.PI / 2;
                    pair.rotation.y = angle;
                    dnaGroup.add(pair);
                }}

                dnaGroup.position.set(-2, 0, 0);
                scene.add(dnaGroup);

                // ═══════ RNA POLYMERASE (transcription machine) ═══════
                const rnaPolGeo = new THREE.SphereGeometry(1.2, 16, 16);
                rnaPolGeo.scale(1.5, 1, 1);
                const rnaPolMat = new THREE.MeshStandardMaterial({{ color: 0x44aaff, roughness: 0.3 }});
                const rnaPol = new THREE.Mesh(rnaPolGeo, rnaPolMat);
                rnaPol.position.set(-2, -4 + phase * 8, 0);
                if (phase < 0.4) scene.add(rnaPol);

                // ═══════ mRNA STRAND ═══════
                const mrnaGroup = new THREE.Group();
                const mrnaLength = phase < 0.4 ? phase * 2.5 : 1.0;
                const mrnaPoints = [];
                for (let i = 0; i <= 50 * mrnaLength; i++) {{
                    const t = i / 50;
                    let x = 3 + t * 8;
                    let y = -4 + phase * 8 + Math.sin(t * 6) * 0.5;
                    let z = Math.cos(t * 6) * 0.5;

                    // Export through nuclear pore
                    if (phase >= 0.4 && phase < 0.6) {{
                        const exportT = (phase - 0.4) / 0.2;
                        x = 3 + t * 8 + exportT * 5;
                    }}
                    // In cytoplasm
                    if (phase >= 0.6) {{
                        x = 8 + t * 10;
                        y = -2 + Math.sin(t * 4) * 1;
                    }}

                    mrnaPoints.push(new THREE.Vector3(x, y, z));
                }}
                if (mrnaPoints.length > 1) {{
                    const mrnaCurve = new THREE.CatmullRomCurve3(mrnaPoints);
                    const mrnaGeo = new THREE.TubeGeometry(mrnaCurve, 40, 0.2, 8, false);
                    const mrnaMat = new THREE.MeshStandardMaterial({{ color: 0x44ff88, roughness: 0.4, emissive: 0x113322, emissiveIntensity: 0.3 }});
                    mrnaGroup.add(new THREE.Mesh(mrnaGeo, mrnaMat));
                }}
                scene.add(mrnaGroup);

                // ═══════ RIBOSOMES (translation) ═══════
                if (showRibosomes && phase >= 0.5) {{
                    for (let r = 0; r < 4; r++) {{
                        const riboGeo = new THREE.SphereGeometry(0.8, 12, 12);
                        riboGeo.scale(1.2, 0.8, 1);
                        const riboMat = new THREE.MeshStandardMaterial({{ color: 0xaa66ff, roughness: 0.5 }});
                        const ribo = new THREE.Mesh(riboGeo, riboMat);
                        ribo.position.set(10 + r * 3, -2 + Math.sin(r) * 0.5, 0);
                        if (phase >= 0.6) scene.add(ribo);

                        // Nascent protein chain
                        if (phase >= 0.7 && r < 2) {{
                            const protPoints = [];
                            for (let p = 0; p <= 10; p++) {{
                                protPoints.push(new THREE.Vector3(
                                    10 + r * 3,
                                    -3 - p * 0.5,
                                    Math.sin(p * 0.8) * 0.5
                                ));
                            }}
                            const protCurve = new THREE.CatmullRomCurve3(protPoints);
                            const protGeo = new THREE.TubeGeometry(protCurve, 15, 0.25, 8, false);
                            const protMat = new THREE.MeshStandardMaterial({{ color: 0xff44aa, roughness: 0.4 }});
                            scene.add(new THREE.Mesh(protGeo, protMat));
                        }}
                    }}
                }}

                // Nuclear pore
                const poreGeo = new THREE.TorusGeometry(1.5, 0.3, 8, 16);
                const poreMat = new THREE.MeshStandardMaterial({{ color: 0x556688 }});
                const pore = new THREE.Mesh(poreGeo, poreMat);
                pore.position.set(8, 0, 0);
                pore.rotation.y = Math.PI / 2;
                scene.add(pore);

                // Labels
                function addLabel(text, pos, color) {{
                    const canvas = document.createElement('canvas');
                    canvas.width = 256; canvas.height = 64;
                    const ctx = canvas.getContext('2d');
                    ctx.fillStyle = color;
                    ctx.font = 'bold 24px Arial';
                    ctx.fillText(text, 10, 40);
                    const tex = new THREE.CanvasTexture(canvas);
                    const spriteMat = new THREE.SpriteMaterial({{ map: tex, transparent: true }});
                    const sprite = new THREE.Sprite(spriteMat);
                    sprite.position.set(pos.x, pos.y, pos.z);
                    sprite.scale.set(6, 1.5, 1);
                    scene.add(sprite);
                }}

                addLabel('DNA', {{ x: -6, y: 6, z: 0 }}, '#ff8844');
                addLabel('Nucleus', {{ x: 0, y: 10, z: 0 }}, '#6688aa');
                if (phase >= 0.3) addLabel('mRNA', {{ x: 5, y: 5, z: 0 }}, '#44ff88');
                if (phase >= 0.6) addLabel('Cytoplasm', {{ x: 15, y: 8, z: 0 }}, '#888888');
                if (phase >= 0.7) addLabel('Protein', {{ x: 12, y: -8, z: 0 }}, '#ff44aa');

                let time = 0;
                function animate() {{
                    requestAnimationFrame(animate);
                    time += 0.016;

                    dnaGroup.rotation.y = time * 0.2;

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

    elif process_type == "Viral Infection Cycle":
        stage_map = {"Entry": 0, "Replication": 1, "Assembly": 2, "Lysis": 3}
        stage_idx = stage_map[infection_stage]

        virus_colors = {
            "SARS-CoV-2": "0xff4444",
            "HIV": "0x44ff44",
            "Influenza": "0x4488ff",
            "Bacteriophage": "0xaa44ff",
        }
        v_color = virus_colors[virus_type]

        process_html = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ margin: 0; overflow: hidden; background: #0a0a15; }}
                #info {{ position: absolute; top: 10px; left: 10px; color: #fff; font-family: 'Segoe UI', sans-serif; font-size: 13px; background: rgba(0,0,0,0.85); padding: 15px; border-radius: 10px; }}
                #stage {{ position: absolute; top: 10px; right: 10px; color: #ff4444; font-family: monospace; font-size: 20px; background: rgba(0,0,0,0.8); padding: 15px 25px; border-radius: 10px; border: 2px solid #ff4444; }}
            </style>
        </head>
        <body>
            <div id="info">
                <div style="font-size:16px;font-weight:bold;color:#ff6666;margin-bottom:10px">🦠 {virus_type} Infection</div>
                <div style="margin-bottom:5px">Stage: <span style="color:#ffaa44">{infection_stage}</span></div>
                <div style="font-size:11px;color:#888;margin-top:8px">
                    {"Viral attachment to cell receptor" if stage_idx == 0 else ""}
                    {"Viral RNA/DNA replication in nucleus" if stage_idx == 1 else ""}
                    {"New virions assembling in cytoplasm" if stage_idx == 2 else ""}
                    {"Cell bursting, releasing virions" if stage_idx == 3 else ""}
                </div>
            </div>
            <div id="stage">⚠ {infection_stage.upper()}</div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
            <script>
                const stageIdx = {stage_idx};
                const virusColor = {v_color};
                const virusCount = {virus_count};

                const scene = new THREE.Scene();
                scene.background = new THREE.Color(0x0a0a15);

                const camera = new THREE.PerspectiveCamera(50, window.innerWidth/window.innerHeight, 0.1, 200);
                camera.position.set(25, 20, 30);

                const renderer = new THREE.WebGLRenderer({{ antialias: true }});
                renderer.setSize(window.innerWidth, window.innerHeight);
                document.body.appendChild(renderer.domElement);

                const controls = new THREE.OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;

                scene.add(new THREE.AmbientLight(0x404050, 0.7));
                const sun = new THREE.DirectionalLight(0xffffff, 1.0);
                sun.position.set(20, 30, 20);
                scene.add(sun);

                // ═══════ HOST CELL ═══════
                const cellGeo = new THREE.SphereGeometry(10, 48, 48);
                let cellOpacity = stageIdx === 3 ? 0.15 : 0.35;
                const cellMat = new THREE.MeshStandardMaterial({{
                    color: 0x4488aa,
                    transparent: true,
                    opacity: cellOpacity,
                    side: stageIdx === 3 ? THREE.DoubleSide : THREE.FrontSide,
                }});
                const cell = new THREE.Mesh(cellGeo, cellMat);

                // Lysis effect - cell membrane breaking
                if (stageIdx === 3) {{
                    cell.geometry.vertices?.forEach?.(v => {{
                        v.x += (Math.random() - 0.5) * 2;
                        v.y += (Math.random() - 0.5) * 2;
                    }});
                }}
                scene.add(cell);

                // Nucleus
                const nucGeo = new THREE.SphereGeometry(3, 32, 32);
                const nucMat = new THREE.MeshStandardMaterial({{ color: 0x224466, roughness: 0.4 }});
                const nuc = new THREE.Mesh(nucGeo, nucMat);
                if (stageIdx < 3) scene.add(nuc);

                // Viral RNA in nucleus (replication stage)
                if (stageIdx >= 1) {{
                    for (let i = 0; i < 8; i++) {{
                        const rnaGeo = new THREE.TorusKnotGeometry(0.5, 0.1, 30, 8);
                        const rnaMat = new THREE.MeshStandardMaterial({{ color: virusColor, emissive: virusColor, emissiveIntensity: 0.3 }});
                        const rna = new THREE.Mesh(rnaGeo, rnaMat);
                        rna.position.set(
                            (Math.random() - 0.5) * 4,
                            (Math.random() - 0.5) * 4,
                            (Math.random() - 0.5) * 4
                        );
                        rna.scale.setScalar(0.6);
                        scene.add(rna);
                    }}
                }}

                // ═══════ VIRIONS ═══════
                const virions = [];

                function createVirion(pos, isSpike) {{
                    const group = new THREE.Group();

                    // Core
                    const coreGeo = new THREE.IcosahedronGeometry(0.6, 1);
                    const coreMat = new THREE.MeshStandardMaterial({{ color: virusColor, roughness: 0.3 }});
                    group.add(new THREE.Mesh(coreGeo, coreMat));

                    // Spike proteins (corona)
                    if (isSpike) {{
                        for (let s = 0; s < 20; s++) {{
                            const spikeGeo = new THREE.ConeGeometry(0.08, 0.4, 4);
                            const spikeMat = new THREE.MeshStandardMaterial({{ color: 0xffcc44 }});
                            const spike = new THREE.Mesh(spikeGeo, spikeMat);

                            const phi = Math.acos(2 * Math.random() - 1);
                            const theta = Math.random() * Math.PI * 2;
                            spike.position.set(
                                0.7 * Math.sin(phi) * Math.cos(theta),
                                0.7 * Math.sin(phi) * Math.sin(theta),
                                0.7 * Math.cos(phi)
                            );
                            spike.lookAt(spike.position.clone().multiplyScalar(2));
                            group.add(spike);
                        }}
                    }}

                    group.position.copy(pos);
                    return group;
                }}

                // Position virions based on stage
                for (let v = 0; v < virusCount; v++) {{
                    let pos;
                    if (stageIdx === 0) {{
                        // Outside cell, approaching
                        const angle = v / virusCount * Math.PI * 2;
                        pos = new THREE.Vector3(
                            Math.cos(angle) * 14,
                            Math.sin(angle) * 14,
                            (Math.random() - 0.5) * 10
                        );
                    }} else if (stageIdx === 1) {{
                        // Some inside
                        pos = new THREE.Vector3(
                            (Math.random() - 0.5) * 6,
                            (Math.random() - 0.5) * 6,
                            (Math.random() - 0.5) * 6
                        );
                    }} else if (stageIdx === 2) {{
                        // Assembling near membrane
                        const r = 7 + Math.random() * 2;
                        const phi = Math.random() * Math.PI;
                        const theta = Math.random() * Math.PI * 2;
                        pos = new THREE.Vector3(
                            r * Math.sin(phi) * Math.cos(theta),
                            r * Math.sin(phi) * Math.sin(theta),
                            r * Math.cos(phi)
                        );
                    }} else {{
                        // Bursting out
                        const r = 10 + Math.random() * 8;
                        const phi = Math.random() * Math.PI;
                        const theta = Math.random() * Math.PI * 2;
                        pos = new THREE.Vector3(
                            r * Math.sin(phi) * Math.cos(theta),
                            r * Math.sin(phi) * Math.sin(theta),
                            r * Math.cos(phi)
                        );
                    }}

                    const virion = createVirion(pos, true);
                    virion.userData = {{ basePos: pos.clone(), phase: Math.random() * Math.PI * 2 }};
                    scene.add(virion);
                    virions.push(virion);
                }}

                // ACE2 receptors on cell surface
                if (stageIdx === 0) {{
                    for (let r = 0; r < 30; r++) {{
                        const recGeo = new THREE.CylinderGeometry(0.15, 0.1, 0.8, 6);
                        const recMat = new THREE.MeshStandardMaterial({{ color: 0x44ff88 }});
                        const rec = new THREE.Mesh(recGeo, recMat);

                        const phi = Math.random() * Math.PI;
                        const theta = Math.random() * Math.PI * 2;
                        rec.position.set(
                            10.2 * Math.sin(phi) * Math.cos(theta),
                            10.2 * Math.sin(phi) * Math.sin(theta),
                            10.2 * Math.cos(phi)
                        );
                        rec.lookAt(0, 0, 0);
                        scene.add(rec);
                    }}
                }}

                let time = 0;
                function animate() {{
                    requestAnimationFrame(animate);
                    time += 0.016;

                    virions.forEach((v, i) => {{
                        const phase = v.userData.phase;
                        if (stageIdx === 0) {{
                            // Move toward cell
                            v.position.lerp(new THREE.Vector3(0, 0, 0), 0.0005);
                        }} else if (stageIdx === 3) {{
                            // Explode outward
                            v.position.add(v.position.clone().normalize().multiplyScalar(0.02));
                        }}
                        v.rotation.x += 0.01;
                        v.rotation.y += 0.01;
                    }});

                    // Cell lysis animation
                    if (stageIdx === 3) {{
                        cell.scale.setScalar(1 + Math.sin(time * 3) * 0.05);
                    }}

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

    elif process_type == "Apoptosis (Programmed Cell Death)":
        stage_map = {"Healthy": 0, "Initiation": 1, "Chromatin Condensation": 2, "Membrane Blebbing": 3, "Apoptotic Bodies": 4}
        stage_idx = stage_map[apoptosis_stage]

        process_html = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ margin: 0; overflow: hidden; background: #0a0510; }}
                #info {{ position: absolute; top: 10px; left: 10px; color: #fff; font-family: 'Segoe UI', sans-serif; font-size: 13px; background: rgba(0,0,0,0.85); padding: 15px; border-radius: 10px; }}
                #stage {{ position: absolute; top: 10px; right: 10px; color: #aa44ff; font-family: monospace; font-size: 18px; background: rgba(0,0,0,0.8); padding: 15px 25px; border-radius: 10px; border: 2px solid #aa44ff; }}
            </style>
        </head>
        <body>
            <div id="info">
                <div style="font-size:16px;font-weight:bold;color:#aa66ff;margin-bottom:10px">💀 Apoptosis</div>
                <div>Trigger: <span style="color:#ffaa44">{trigger}</span></div>
                <div>Stage: <span style="color:#ff66aa">{apoptosis_stage}</span></div>
                <div style="font-size:11px;color:#888;margin-top:8px">
                    {"Normal cell function" if stage_idx == 0 else ""}
                    {"Caspase cascade activated" if stage_idx == 1 else ""}
                    {"DNA fragmentation, nuclear condensation" if stage_idx == 2 else ""}
                    {"Cell membrane blebbing" if stage_idx == 3 else ""}
                    {"Cell fragmenting into apoptotic bodies" if stage_idx == 4 else ""}
                </div>
            </div>
            <div id="stage">{apoptosis_stage.upper()}</div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
            <script>
                const stageIdx = {stage_idx};
                const showCaspases = {'true' if show_caspases else 'false'};

                const scene = new THREE.Scene();
                scene.background = new THREE.Color(0x0a0510);

                const camera = new THREE.PerspectiveCamera(50, window.innerWidth/window.innerHeight, 0.1, 200);
                camera.position.set(20, 15, 25);

                const renderer = new THREE.WebGLRenderer({{ antialias: true }});
                renderer.setSize(window.innerWidth, window.innerHeight);
                document.body.appendChild(renderer.domElement);

                const controls = new THREE.OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;
                controls.autoRotate = true;
                controls.autoRotateSpeed = 0.3;

                scene.add(new THREE.AmbientLight(0x404050, 0.6));
                scene.add(new THREE.DirectionalLight(0xffffff, 1.0)).position.set(15, 25, 20);

                // Cell state based on stage
                const cellRadius = stageIdx <= 1 ? 8 : stageIdx === 2 ? 7 : stageIdx === 3 ? 6 : 0;
                const cellColor = stageIdx === 0 ? 0x4488ff : stageIdx <= 2 ? 0x8866aa : 0x664488;

                if (stageIdx < 4) {{
                    // Main cell membrane
                    const cellGeo = new THREE.SphereGeometry(cellRadius, 48, 48);
                    const cellMat = new THREE.MeshStandardMaterial({{
                        color: cellColor,
                        transparent: true,
                        opacity: 0.35,
                        side: THREE.DoubleSide,
                    }});
                    const cell = new THREE.Mesh(cellGeo, cellMat);
                    scene.add(cell);

                    // Membrane blebs (stage 3)
                    if (stageIdx === 3) {{
                        for (let b = 0; b < 15; b++) {{
                            const blebGeo = new THREE.SphereGeometry(0.8 + Math.random() * 0.8, 12, 12);
                            const blebMat = new THREE.MeshStandardMaterial({{
                                color: 0x664488,
                                transparent: true,
                                opacity: 0.5,
                            }});
                            const bleb = new THREE.Mesh(blebGeo, blebMat);

                            const phi = Math.random() * Math.PI;
                            const theta = Math.random() * Math.PI * 2;
                            const r = cellRadius + 0.5;
                            bleb.position.set(
                                r * Math.sin(phi) * Math.cos(theta),
                                r * Math.sin(phi) * Math.sin(theta),
                                r * Math.cos(phi)
                            );
                            scene.add(bleb);
                        }}
                    }}
                }}

                // Nucleus (condensing in later stages)
                const nucRadius = stageIdx === 0 ? 3 : stageIdx === 1 ? 2.8 : stageIdx === 2 ? 2 : stageIdx === 3 ? 1.5 : 0;
                if (nucRadius > 0) {{
                    const nucGeo = new THREE.SphereGeometry(nucRadius, 32, 32);
                    const nucMat = new THREE.MeshStandardMaterial({{
                        color: stageIdx <= 1 ? 0x224488 : 0x442266,
                        roughness: 0.4,
                    }});
                    const nuc = new THREE.Mesh(nucGeo, nucMat);
                    scene.add(nuc);

                    // Chromatin condensation (stage 2+)
                    if (stageIdx >= 2) {{
                        for (let c = 0; c < 8; c++) {{
                            const chromGeo = new THREE.SphereGeometry(0.4, 8, 8);
                            const chromMat = new THREE.MeshStandardMaterial({{ color: 0x220044 }});
                            const chrom = new THREE.Mesh(chromGeo, chromMat);
                            chrom.position.set(
                                (Math.random() - 0.5) * nucRadius,
                                (Math.random() - 0.5) * nucRadius,
                                (Math.random() - 0.5) * nucRadius
                            );
                            scene.add(chrom);
                        }}
                    }}
                }}

                // Caspase cascade visualization
                if (showCaspases && stageIdx >= 1) {{
                    for (let c = 0; c < 20; c++) {{
                        const caspGeo = new THREE.TetrahedronGeometry(0.3);
                        const caspMat = new THREE.MeshStandardMaterial({{
                            color: 0xff4444,
                            emissive: 0xff2222,
                            emissiveIntensity: 0.5,
                        }});
                        const casp = new THREE.Mesh(caspGeo, caspMat);
                        casp.position.set(
                            (Math.random() - 0.5) * 12,
                            (Math.random() - 0.5) * 12,
                            (Math.random() - 0.5) * 12
                        );
                        casp.userData = {{ speed: 0.02 + Math.random() * 0.02 }};
                        scene.add(casp);
                    }}
                }}

                // Apoptotic bodies (stage 4)
                if (stageIdx === 4) {{
                    for (let a = 0; a < 12; a++) {{
                        const bodyGeo = new THREE.SphereGeometry(1 + Math.random() * 1.5, 16, 16);
                        const bodyMat = new THREE.MeshStandardMaterial({{
                            color: 0x664488,
                            transparent: true,
                            opacity: 0.7,
                        }});
                        const body = new THREE.Mesh(bodyGeo, bodyMat);

                        const angle = a / 12 * Math.PI * 2;
                        const r = 4 + Math.random() * 6;
                        body.position.set(
                            Math.cos(angle) * r,
                            (Math.random() - 0.5) * 8,
                            Math.sin(angle) * r
                        );
                        body.userData = {{ drift: new THREE.Vector3(Math.random()-0.5, Math.random()-0.5, Math.random()-0.5).multiplyScalar(0.02) }};
                        scene.add(body);

                        // DNA fragments inside
                        const fragGeo = new THREE.IcosahedronGeometry(0.3, 0);
                        const fragMat = new THREE.MeshStandardMaterial({{ color: 0x220044 }});
                        const frag = new THREE.Mesh(fragGeo, fragMat);
                        frag.position.copy(body.position);
                        scene.add(frag);
                    }}

                    // Phosphatidylserine "eat me" signals
                    for (let p = 0; p < 30; p++) {{
                        const psGeo = new THREE.SphereGeometry(0.15, 6, 6);
                        const psMat = new THREE.MeshStandardMaterial({{ color: 0x44ff44, emissive: 0x22aa22, emissiveIntensity: 0.5 }});
                        const ps = new THREE.Mesh(psGeo, psMat);
                        ps.position.set(
                            (Math.random() - 0.5) * 16,
                            (Math.random() - 0.5) * 12,
                            (Math.random() - 0.5) * 16
                        );
                        scene.add(ps);
                    }}
                }}

                // Mitochondria (release cytochrome c in initiation)
                if (stageIdx >= 1 && stageIdx < 4) {{
                    for (let m = 0; m < 6; m++) {{
                        const mitoGeo = new THREE.SphereGeometry(0.5, 10, 10);
                        mitoGeo.scale(1, 2, 1);
                        const mitoMat = new THREE.MeshStandardMaterial({{ color: 0xff8844 }});
                        const mito = new THREE.Mesh(mitoGeo, mitoMat);
                        mito.position.set(
                            (Math.random() - 0.5) * 8,
                            (Math.random() - 0.5) * 8,
                            (Math.random() - 0.5) * 8
                        );
                        mito.rotation.set(Math.random() * Math.PI, Math.random() * Math.PI, 0);
                        scene.add(mito);

                        // Cytochrome c leaking
                        if (stageIdx >= 1) {{
                            for (let cy = 0; cy < 3; cy++) {{
                                const cytoGeo = new THREE.SphereGeometry(0.12, 6, 6);
                                const cytoMat = new THREE.MeshStandardMaterial({{ color: 0xff2222, emissive: 0xff0000, emissiveIntensity: 0.3 }});
                                const cyto = new THREE.Mesh(cytoGeo, cytoMat);
                                cyto.position.copy(mito.position).add(new THREE.Vector3(Math.random()-0.5, Math.random()-0.5, Math.random()-0.5));
                                scene.add(cyto);
                            }}
                        }}
                    }}
                }}

                let time = 0;
                function animate() {{
                    requestAnimationFrame(animate);
                    time += 0.016;

                    scene.children.forEach(obj => {{
                        if (obj.userData?.drift) {{
                            obj.position.add(obj.userData.drift);
                        }}
                        if (obj.userData?.speed) {{
                            obj.rotation.x += obj.userData.speed;
                            obj.rotation.y += obj.userData.speed * 0.7;
                        }}
                    }});

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

    else:  # Phagocytosis
        stage_map = {"Recognition": 0, "Attachment": 1, "Engulfment": 2, "Phagosome Formation": 3, "Digestion": 4}
        stage_idx = stage_map[phago_stage]

        target_colors = {"Bacteria": "0x44ff44", "Apoptotic Cell": "0x8866aa", "Cancer Cell": "0xff4444"}
        t_color = target_colors[target_type]

        process_html = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ margin: 0; overflow: hidden; background: #050a10; }}
                #info {{ position: absolute; top: 10px; left: 10px; color: #fff; font-family: 'Segoe UI', sans-serif; font-size: 13px; background: rgba(0,0,0,0.85); padding: 15px; border-radius: 10px; }}
                #stage {{ position: absolute; top: 10px; right: 10px; color: #44aaff; font-family: monospace; font-size: 18px; background: rgba(0,0,0,0.8); padding: 15px 25px; border-radius: 10px; border: 2px solid #44aaff; }}
            </style>
        </head>
        <body>
            <div id="info">
                <div style="font-size:16px;font-weight:bold;color:#66aaff;margin-bottom:10px">🦠 Phagocytosis</div>
                <div>Target: <span style="color:#ffaa44">{target_type}</span></div>
                <div>Stage: <span style="color:#44ff88">{phago_stage}</span></div>
                <div style="font-size:11px;color:#888;margin-top:8px">
                    {"Pattern recognition receptors detect pathogen" if stage_idx == 0 else ""}
                    {"Pseudopods extending around target" if stage_idx == 1 else ""}
                    {"Membrane wrapping around pathogen" if stage_idx == 2 else ""}
                    {"Phagosome formed, fusing with lysosome" if stage_idx == 3 else ""}
                    {"Lysosomal enzymes destroying pathogen" if stage_idx == 4 else ""}
                </div>
            </div>
            <div id="stage">{phago_stage.upper()}</div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
            <script>
                const stageIdx = {stage_idx};
                const targetColor = {t_color};
                const showLysosomes = {'true' if show_lysosomes else 'false'};

                const scene = new THREE.Scene();
                scene.background = new THREE.Color(0x050a10);

                const camera = new THREE.PerspectiveCamera(50, window.innerWidth/window.innerHeight, 0.1, 200);
                camera.position.set(22, 18, 28);

                const renderer = new THREE.WebGLRenderer({{ antialias: true }});
                renderer.setSize(window.innerWidth, window.innerHeight);
                document.body.appendChild(renderer.domElement);

                const controls = new THREE.OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;
                controls.autoRotate = true;
                controls.autoRotateSpeed = 0.2;

                scene.add(new THREE.AmbientLight(0x405060, 0.7));
                scene.add(new THREE.DirectionalLight(0xffffff, 1.0)).position.set(15, 25, 20);

                // ═══════ MACROPHAGE ═══════
                const macroGeo = new THREE.SphereGeometry(8, 48, 48);
                const macroMat = new THREE.MeshStandardMaterial({{
                    color: 0x4488aa,
                    transparent: true,
                    opacity: 0.4,
                    side: THREE.DoubleSide,
                }});
                const macro = new THREE.Mesh(macroGeo, macroMat);
                scene.add(macro);

                // Macrophage nucleus
                const macNucGeo = new THREE.SphereGeometry(2.5, 24, 24);
                const macNucMat = new THREE.MeshStandardMaterial({{ color: 0x224466 }});
                const macNuc = new THREE.Mesh(macNucGeo, macNucMat);
                macNuc.position.set(-3, 0, 0);
                scene.add(macNuc);

                // ═══════ PSEUDOPODS (stages 1-2) ═══════
                if (stageIdx >= 1 && stageIdx <= 2) {{
                    const pseudopods = [];
                    for (let p = 0; p < 4; p++) {{
                        const angle = (p / 4) * Math.PI + Math.PI * 0.25;
                        const points = [];
                        const extension = stageIdx === 1 ? 0.5 : 1.0;

                        for (let t = 0; t <= 10; t++) {{
                            const tt = t / 10;
                            const r = 8 + tt * 6 * extension;
                            const spread = tt * 0.3;
                            points.push(new THREE.Vector3(
                                r * Math.cos(angle + spread * Math.sin(p)),
                                tt * 3 - 1.5,
                                r * Math.sin(angle + spread * Math.sin(p))
                            ));
                        }}

                        const curve = new THREE.CatmullRomCurve3(points);
                        const tubeGeo = new THREE.TubeGeometry(curve, 20, 1.2 - t * 0.08, 12, false);
                        const tubeMat = new THREE.MeshStandardMaterial({{
                            color: 0x4488aa,
                            transparent: true,
                            opacity: 0.5,
                        }});
                        scene.add(new THREE.Mesh(tubeGeo, tubeMat));
                    }}
                }}

                // ═══════ TARGET (bacteria/cell) ═══════
                const targetGroup = new THREE.Group();

                // Position based on stage
                let targetX = stageIdx === 0 ? 16 : stageIdx === 1 ? 12 : stageIdx === 2 ? 8 : 2;
                let targetScale = stageIdx >= 3 ? 0.7 : 1.0;

                const targetGeo = new THREE.SphereGeometry(2, 24, 24);
                const targetMat = new THREE.MeshStandardMaterial({{
                    color: targetColor,
                    roughness: 0.4,
                    transparent: stageIdx >= 4,
                    opacity: stageIdx >= 4 ? 0.3 : 1.0,
                }});
                const target = new THREE.Mesh(targetGeo, targetMat);
                targetGroup.add(target);

                // Bacteria flagella
                if ("{target_type}" === "Bacteria") {{
                    for (let f = 0; f < 3; f++) {{
                        const flagPoints = [];
                        for (let t = 0; t <= 20; t++) {{
                            flagPoints.push(new THREE.Vector3(
                                -2 - t * 0.3,
                                Math.sin(t * 0.8) * 0.5,
                                Math.cos(t * 0.8 + f * 2) * 0.5
                            ));
                        }}
                        const flagCurve = new THREE.CatmullRomCurve3(flagPoints);
                        const flagGeo = new THREE.TubeGeometry(flagCurve, 15, 0.08, 6, false);
                        const flagMat = new THREE.MeshStandardMaterial({{ color: 0x88ff88 }});
                        targetGroup.add(new THREE.Mesh(flagGeo, flagMat));
                    }}
                }}

                targetGroup.position.set(targetX, 0, 0);
                targetGroup.scale.setScalar(targetScale);
                scene.add(targetGroup);

                // ═══════ PHAGOSOME (stages 3-4) ═══════
                if (stageIdx >= 3) {{
                    const phagoGeo = new THREE.SphereGeometry(3.5, 24, 24);
                    const phagoMat = new THREE.MeshStandardMaterial({{
                        color: 0x336688,
                        transparent: true,
                        opacity: 0.4,
                    }});
                    const phago = new THREE.Mesh(phagoGeo, phagoMat);
                    phago.position.set(2, 0, 0);
                    scene.add(phago);
                }}

                // ═══════ LYSOSOMES ═══════
                if (showLysosomes) {{
                    const lysoCount = stageIdx >= 3 ? 8 : 4;
                    for (let l = 0; l < lysoCount; l++) {{
                        const lysoGeo = new THREE.SphereGeometry(0.6, 12, 12);
                        const lysoMat = new THREE.MeshStandardMaterial({{
                            color: 0xffaa22,
                            emissive: 0xff8800,
                            emissiveIntensity: stageIdx >= 4 ? 0.5 : 0.2,
                        }});
                        const lyso = new THREE.Mesh(lysoGeo, lysoMat);

                        if (stageIdx < 3) {{
                            // Distributed in cytoplasm
                            lyso.position.set(
                                (Math.random() - 0.5) * 10,
                                (Math.random() - 0.5) * 10,
                                (Math.random() - 0.5) * 10
                            );
                        }} else if (stageIdx === 3) {{
                            // Moving toward phagosome
                            const angle = l / lysoCount * Math.PI * 2;
                            lyso.position.set(
                                2 + Math.cos(angle) * 4,
                                Math.sin(angle) * 4,
                                0
                            );
                        }} else {{
                            // Fused with phagosome
                            const angle = l / lysoCount * Math.PI * 2;
                            lyso.position.set(
                                2 + Math.cos(angle) * 2,
                                Math.sin(angle) * 2,
                                0
                            );
                        }}
                        lyso.userData = {{ phase: l * 0.5 }};
                        scene.add(lyso);
                    }}

                    // Digestive enzymes (stage 4)
                    if (stageIdx >= 4) {{
                        for (let e = 0; e < 20; e++) {{
                            const enzGeo = new THREE.TetrahedronGeometry(0.15);
                            const enzMat = new THREE.MeshStandardMaterial({{
                                color: 0xff4444,
                                emissive: 0xff2222,
                                emissiveIntensity: 0.4
                            }});
                            const enz = new THREE.Mesh(enzGeo, enzMat);
                            enz.position.set(
                                2 + (Math.random() - 0.5) * 4,
                                (Math.random() - 0.5) * 4,
                                (Math.random() - 0.5) * 4
                            );
                            enz.userData = {{ speed: 0.03 + Math.random() * 0.02 }};
                            scene.add(enz);
                        }}
                    }}
                }}

                // Pattern recognition receptors (stage 0)
                if (stageIdx === 0) {{
                    for (let r = 0; r < 12; r++) {{
                        const recGeo = new THREE.CylinderGeometry(0.2, 0.15, 1, 6);
                        const recMat = new THREE.MeshStandardMaterial({{ color: 0x88ffaa }});
                        const rec = new THREE.Mesh(recGeo, recMat);

                        const phi = Math.random() * Math.PI * 0.5 + Math.PI * 0.25;
                        const theta = Math.random() * Math.PI - Math.PI * 0.5;
                        rec.position.set(
                            8.5 * Math.cos(theta),
                            8.5 * Math.sin(theta) * Math.sin(phi),
                            8.5 * Math.sin(theta) * Math.cos(phi)
                        );
                        rec.lookAt(16, 0, 0);
                        scene.add(rec);
                    }}
                }}

                let time = 0;
                function animate() {{
                    requestAnimationFrame(animate);
                    time += 0.016;

                    // Animate lysosomes
                    scene.children.forEach(obj => {{
                        if (obj.userData?.phase !== undefined) {{
                            obj.position.y += Math.sin(time * 2 + obj.userData.phase) * 0.01;
                        }}
                        if (obj.userData?.speed) {{
                            obj.rotation.x += obj.userData.speed;
                            obj.rotation.y += obj.userData.speed * 0.7;
                        }}
                    }});

                    // Target movement
                    if (stageIdx === 0) {{
                        targetGroup.position.x = 16 + Math.sin(time) * 0.5;
                    }}

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

    components.html(process_html, height=600)

    st.caption(f"**{process_type}** — Interactive 3D biological process simulation")


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
