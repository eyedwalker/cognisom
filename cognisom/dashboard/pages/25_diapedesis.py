"""
Leukocyte Diapedesis Simulator
==============================

Interactive 3D visualization of the 6-step leukocyte extravasation cascade
through postcapillary venule endothelium (Abbas Fig 3.3):

1. Cytokine production → E-selectin expression
2. Selectin-mediated rolling
3. Chemokine-mediated integrin activation
4. Integrin-mediated firm adhesion
5. Transmigration through endothelial junctions
6. Chemotaxis to infection site

Features:
- 3D vessel cutaway with Three.js (half-cylinder, see inside)
- Leukocytes color-coded by diapedesis state
- RBCs flowing with Poiseuille profile
- Endothelial wall tiles colored by inflammation
- Real-time metrics panel
- 6 preset scenarios including LAD diseases
- Export simulation data
"""

import json
import math
import time

import streamlit as st

st.set_page_config(
    page_title="Diapedesis Simulator | Cognisom",
    page_icon="🩸",
    layout="wide",
)

# Auth gate
try:
    from cognisom.auth.middleware import streamlit_page_gate
    user = streamlit_page_gate(required_tier="researcher")
except Exception:
    user = None


# ═══════════════════════════════════════════════════════════════════════
# Simulation Import
# ═══════════════════════════════════════════════════════════════════════

try:
    from cognisom.simulations.diapedesis import (
        DiapedesisSim, DiapedesisConfig, LeukocyteState,
    )
    SIM_AVAILABLE = True
except ImportError as e:
    SIM_AVAILABLE = False
    _sim_error = str(e)


# ═══════════════════════════════════════════════════════════════════════
# Session State
# ═══════════════════════════════════════════════════════════════════════

if "diap_frames" not in st.session_state:
    st.session_state.diap_frames = []
if "diap_running" not in st.session_state:
    st.session_state.diap_running = False
if "diap_preset" not in st.session_state:
    st.session_state.diap_preset = "Severe inflammation"


# ═══════════════════════════════════════════════════════════════════════
# Page Header
# ═══════════════════════════════════════════════════════════════════════

st.title("🩸 Leukocyte Diapedesis Simulator")
st.markdown(
    "Visualize the **6-step leukocyte extravasation cascade** through "
    "postcapillary venule endothelium. Based on Abbas *Cellular and "
    "Molecular Immunology* 10th Ed, Fig 3.3."
)

if not SIM_AVAILABLE:
    st.error(f"Simulation engine not available: {_sim_error}")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════
# Sidebar Controls
# ═══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("Simulation Parameters")

    # Preset scenarios
    preset = st.selectbox(
        "Preset Scenario",
        [
            "Severe inflammation",
            "Mild inflammation",
            "Healthy vessel",
            "LAD-1 (no LFA-1 / β2 integrin)",
            "LAD-2 (no sialyl Lewis-x)",
            "LAD-3 (no kindlin-3)",
            "Custom",
        ],
        index=0,
        key="diap_preset_select",
    )

    st.divider()

    # Get config from preset or custom sliders
    if preset == "Custom":
        inflammation = st.slider("Inflammation level", 0.0, 1.0, 0.7, 0.05)
        flow_vel = st.slider("Flow velocity (μm/s)", 100, 2000, 500, 50)
        n_leuko = st.slider("Leukocytes", 5, 50, 20)
        n_rbc = st.slider("RBCs", 50, 500, 200, 50)
        selectin_on = st.slider("Selectin on-rate (s⁻¹)", 0.0, 20.0, 10.0, 0.5)
        integrin_time = st.slider("Integrin activation time (s)", 0.5, 10.0, 2.0, 0.5)
        chemokine_thresh = st.slider("Chemokine threshold", 0.05, 1.0, 0.3, 0.05)

        cfg = DiapedesisConfig(
            inflammation_level=inflammation,
            flow_velocity_max=flow_vel,
            n_leukocytes=n_leuko,
            n_rbc=n_rbc,
            selectin_on_rate=selectin_on,
            integrin_activation_time=integrin_time,
            chemokine_activation_threshold=chemokine_thresh,
        )
    else:
        # Show preset description
        preset_info = {
            "Severe inflammation": (
                "High inflammation (0.9), rapid leukocyte recruitment. "
                "Most leukocytes should transmigrate within 2 minutes."
            ),
            "Mild inflammation": (
                "Moderate inflammation (0.5). Some rolling and arrest, "
                "fewer transmigrations."
            ),
            "Healthy vessel": (
                "Minimal inflammation (0.1). Very few leukocytes interact "
                "with the endothelium."
            ),
            "LAD-1 (no LFA-1 / β2 integrin)": (
                "**Leukocyte Adhesion Deficiency type 1**: Normal selectin-mediated "
                "rolling, but integrins cannot bind → no firm adhesion, no "
                "transmigration. Recurrent bacterial infections."
            ),
            "LAD-2 (no sialyl Lewis-x)": (
                "**Leukocyte Adhesion Deficiency type 2**: Selectin ligands absent → "
                "no rolling at all → no recruitment. Rare, ~30 cases worldwide."
            ),
            "LAD-3 (no kindlin-3)": (
                "**Leukocyte Adhesion Deficiency type 3**: Rolling occurs normally, "
                "but integrin inside-out activation fails (kindlin-3 mutation) → "
                "no firm adhesion."
            ),
        }
        st.info(preset_info.get(preset, ""))
        cfg = None  # Will use factory method

    st.divider()

    duration = st.slider("Duration (seconds)", 30, 300, 120, 10)
    fps = st.selectbox("Output FPS", [5, 10, 15, 30], index=1)
    playback_speed = st.select_slider(
        "Playback speed", options=[0.5, 1.0, 2.0, 4.0], value=2.0
    )

    st.divider()

    run_btn = st.button("▶ Run Simulation", type="primary", use_container_width=True)

    if st.session_state.diap_frames:
        st.download_button(
            "📥 Export frames (JSON)",
            data=json.dumps(st.session_state.diap_frames[-1], indent=2),
            file_name="diapedesis_last_frame.json",
            mime="application/json",
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════════════
# Run Simulation
# ═══════════════════════════════════════════════════════════════════════

if run_btn:
    with st.spinner(f"Running diapedesis simulation ({duration}s at dt=0.01)..."):
        t0 = time.time()

        # Create sim from preset or custom config
        if cfg is not None:
            sim = DiapedesisSim(cfg)
        elif preset == "Severe inflammation":
            sim = DiapedesisSim.severe_inflammation()
        elif preset == "Mild inflammation":
            sim = DiapedesisSim.mild_inflammation()
        elif preset == "Healthy vessel":
            sim = DiapedesisSim.healthy_vessel()
        elif "LAD-1" in preset:
            sim = DiapedesisSim.lad1_no_lfa1()
        elif "LAD-2" in preset:
            sim = DiapedesisSim.lad2_no_selectin_ligand()
        elif "LAD-3" in preset:
            sim = DiapedesisSim.lad3_no_kindlin3()
        else:
            sim = DiapedesisSim.severe_inflammation()

        sim.initialize()
        frames = sim.run(duration=float(duration), fps=fps)
        elapsed = time.time() - t0

        st.session_state.diap_frames = frames
        st.success(
            f"Simulation complete: {len(frames)} frames in {elapsed:.1f}s "
            f"({sim.config.n_leukocytes} leukocytes, {sim.config.n_rbc} RBCs)"
        )


# ═══════════════════════════════════════════════════════════════════════
# Three.js Vessel Cutaway Viewer
# ═══════════════════════════════════════════════════════════════════════

def _build_vessel_viewer(frames, playback_speed=2.0):
    """Build Three.js 3D vessel cutaway with animated diapedesis."""
    if not frames:
        return ""

    frames_json = json.dumps(frames)
    n_frames = len(frames)
    first = frames[0]
    R = first["vessel_radius"]
    L = first["vessel_length"]

    return f'''
<div id="vesselViewer" style="width:100%;height:600px;border-radius:8px;overflow:hidden;background:#050510;position:relative;"></div>
<div id="vesselControls" style="width:100%;padding:8px 0;display:flex;align-items:center;gap:10px;font:13px monospace;color:#ccc;">
    <button id="vPlayBtn" style="padding:4px 12px;cursor:pointer;background:#2a2a4a;color:#ccc;border:1px solid #555;border-radius:4px;">Play</button>
    <input id="vSlider" type="range" min="0" max="{n_frames - 1}" value="0" style="flex:1;">
    <span id="vFrameLabel">0 / {n_frames - 1}</span>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
(function() {{
    const container = document.getElementById('vesselViewer');
    if (!container) return;
    const W = container.clientWidth, H = container.clientHeight;
    const R = {R}, L = {L};

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x050510);

    const camera = new THREE.PerspectiveCamera(50, W/H, 0.1, 2000);
    camera.position.set(L*0.5, R*2.5, R*3.0);
    camera.lookAt(L*0.5, 0, 0);

    const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
    renderer.setSize(W, H);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.target.set(L*0.5, 0, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.update();

    // Lighting
    scene.add(new THREE.AmbientLight(0x404060, 0.5));
    const key = new THREE.DirectionalLight(0xfff5e8, 1.0);
    key.position.set(L, R*3, R*2);
    scene.add(key);
    scene.add(new THREE.DirectionalLight(0x6688cc, 0.3).position.set(-L, R, -R));

    // ── Vessel wall (half-cylinder cutaway) ──
    const vesselGeo = new THREE.CylinderGeometry(R, R, L, 48, 1, true, 0, Math.PI);
    vesselGeo.rotateZ(Math.PI / 2);  // length along X
    vesselGeo.translate(L/2, 0, 0);
    const vesselMat = new THREE.MeshPhongMaterial({{
        color: 0xe8b0b0,
        transparent: true,
        opacity: 0.25,
        side: THREE.DoubleSide,
        depthWrite: false,
    }});
    const vesselMesh = new THREE.Mesh(vesselGeo, vesselMat);
    scene.add(vesselMesh);

    // Vessel wireframe
    const wireGeo = new THREE.CylinderGeometry(R, R, L, 24, 4, true, 0, Math.PI);
    wireGeo.rotateZ(Math.PI / 2);
    wireGeo.translate(L/2, 0, 0);
    const wireMat = new THREE.MeshBasicMaterial({{ color: 0xcc8888, wireframe: true, transparent: true, opacity: 0.15 }});
    scene.add(new THREE.Mesh(wireGeo, wireMat));

    // Vessel axis line
    const axisGeo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 0, 0), new THREE.Vector3(L, 0, 0)
    ]);
    scene.add(new THREE.Line(axisGeo, new THREE.LineBasicMaterial({{ color: 0x334466, transparent: true, opacity: 0.3 }})));

    // ── Infection site glow (below vessel) ──
    const infGeo = new THREE.SphereGeometry(R*0.6, 16, 12);
    const infMat = new THREE.MeshBasicMaterial({{ color: 0xff4422, transparent: true, opacity: 0.15 }});
    const infMesh = new THREE.Mesh(infGeo, infMat);
    infMesh.position.set(L*0.5, -R*1.8, 0);
    scene.add(infMesh);

    // Infection label
    const infLight = new THREE.PointLight(0xff4422, 0.5, R*4);
    infLight.position.copy(infMesh.position);
    scene.add(infLight);

    // ── Load frames data ──
    const frames = {frames_json};

    // ── InstancedMesh: leukocytes ──
    const nLeuko = frames[0].leukocyte_positions.length;
    const leukoGeo = new THREE.SphereGeometry(1, 16, 12);
    const leukoMat = new THREE.MeshPhongMaterial({{ vertexColors: false }});
    const leukoMesh = new THREE.InstancedMesh(leukoGeo, leukoMat, nLeuko);
    leukoMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    // Per-instance color
    const leukoColors = new Float32Array(nLeuko * 3);
    leukoMesh.instanceColor = new THREE.InstancedBufferAttribute(leukoColors, 3);
    leukoMat.vertexColors = false;
    scene.add(leukoMesh);

    // ── InstancedMesh: RBCs ──
    const nRBC = frames[0].rbc_positions.length;
    const rbcGeo = new THREE.SphereGeometry(1, 8, 6);
    rbcGeo.scale(1, 1, 0.35);  // oblate disc
    const rbcMat = new THREE.MeshPhongMaterial({{ color: 0xb01010, transparent: true, opacity: 0.7 }});
    const rbcMesh = new THREE.InstancedMesh(rbcGeo, rbcMat, nRBC);
    rbcMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    scene.add(rbcMesh);

    // ── InstancedMesh: endothelial cells ──
    const nEndo = frames[0].endo_positions.length;
    const endoGeo = new THREE.SphereGeometry(1, 8, 6);
    endoGeo.scale(1, 1, 0.15);  // very flat (squamous)
    const endoMat = new THREE.MeshPhongMaterial({{ vertexColors: false }});
    const endoMesh = new THREE.InstancedMesh(endoGeo, endoMat, nEndo);
    const endoColors = new Float32Array(nEndo * 3);
    endoMesh.instanceColor = new THREE.InstancedBufferAttribute(endoColors, 3);
    endoMat.vertexColors = false;
    scene.add(endoMesh);

    // ── State color map ──
    const stateColors = [
        [0.9, 0.9, 0.9],       // 0 FLOWING: white
        [1.0, 1.0, 0.7],       // 1 MARGINATING: light yellow
        [1.0, 0.85, 0.0],      // 2 ROLLING: yellow
        [1.0, 0.55, 0.0],      // 3 ACTIVATING: orange
        [1.0, 0.3, 0.0],       // 4 ARRESTED: red-orange
        [0.9, 0.1, 0.1],       // 5 CRAWLING: red
        [0.8, 0.0, 0.5],       // 6 TRANSMIGRATING: magenta
        [0.2, 0.8, 0.2],       // 7 MIGRATED: green
    ];
    const stateNames = ['Flowing','Marginating','Rolling','Activating',
                        'Arrested','Crawling','Transmigrating','Migrated'];

    const dummy = new THREE.Object3D();

    function applyFrame(fi) {{
        const f = frames[fi];
        if (!f) return;

        // Leukocytes
        const lp = f.leukocyte_positions;
        const ls = f.leukocyte_states;
        const lr = f.leukocyte_radii;
        for (let i = 0; i < nLeuko; i++) {{
            const p = lp[i];
            dummy.position.set(p[0], p[1], p[2]);
            const rad = lr[i] || 6;
            dummy.scale.set(rad, rad, rad);
            dummy.updateMatrix();
            leukoMesh.setMatrixAt(i, dummy.matrix);
            const sc = stateColors[ls[i]] || stateColors[0];
            leukoColors[i*3] = sc[0];
            leukoColors[i*3+1] = sc[1];
            leukoColors[i*3+2] = sc[2];
        }}
        leukoMesh.instanceMatrix.needsUpdate = true;
        leukoMesh.instanceColor.needsUpdate = true;

        // RBCs
        const rp = f.rbc_positions;
        for (let i = 0; i < nRBC; i++) {{
            const p = rp[i];
            dummy.position.set(p[0], p[1], p[2]);
            dummy.scale.set(3.75, 3.75, 3.75);
            // Random rotation for visual variety
            dummy.rotation.set(p[0]*0.1, p[1]*0.1, p[2]*0.1);
            dummy.updateMatrix();
            rbcMesh.setMatrixAt(i, dummy.matrix);
        }}
        rbcMesh.instanceMatrix.needsUpdate = true;

        // Endothelial cells
        const ep = f.endo_positions;
        const ec = f.endo_colors;
        for (let i = 0; i < nEndo; i++) {{
            const p = ep[i];
            dummy.position.set(p[0], p[1], p[2]);
            dummy.scale.set(10, 10, 10);
            // Orient flat face toward vessel center
            dummy.lookAt(p[0], 0, 0);
            dummy.updateMatrix();
            endoMesh.setMatrixAt(i, dummy.matrix);
            endoColors[i*3] = ec[i][0];
            endoColors[i*3+1] = ec[i][1];
            endoColors[i*3+2] = ec[i][2];
        }}
        endoMesh.instanceMatrix.needsUpdate = true;
        endoMesh.instanceColor.needsUpdate = true;

        // Update metrics overlay
        const m = f.metrics;
        const sc = m.state_counts;
        let metricsHTML = '<b>Diapedesis State</b><br>';
        stateNames.forEach((name, si) => {{
            const count = sc[name.toLowerCase()] || 0;
            const c = stateColors[si];
            const hex = '#' + new THREE.Color(c[0],c[1],c[2]).getHexString();
            if (count > 0) {{
                metricsHTML += '<span style="color:'+hex+'">\\u25CF</span> '+name+': '+count+'<br>';
            }}
        }});
        metricsHTML += '<br>t = ' + f.time.toFixed(1) + 's';
        metricsHTML += '<br>Rolling v: ' + m.avg_rolling_velocity.toFixed(0) + ' \\u03BCm/s';
        metricsHTML += '<br>Junction: ' + (m.avg_junction_integrity * 100).toFixed(0) + '%';
        if (metricsOverlay) metricsOverlay.innerHTML = metricsHTML;

        // Step annotations
        let stepsHTML = '<b>Diapedesis Steps</b><br>';
        const steps = [
            ['1. Cytokine \\u2192 E-selectin', sc.rolling > 0 || sc.activating > 0 || sc.arrested > 0],
            ['2. Selectin rolling', sc.rolling > 0],
            ['3. Integrin activation', sc.activating > 0 || sc.arrested > 0],
            ['4. Firm adhesion', sc.arrested > 0 || sc.crawling > 0],
            ['5. Transmigration', sc.transmigrating > 0],
            ['6. Tissue migration', sc.migrated > 0],
        ];
        steps.forEach(([label, active]) => {{
            const icon = active ? '\\u2705' : '\\u2B1C';
            stepsHTML += icon + ' ' + label + '<br>';
        }});
        if (stepsOverlay) stepsOverlay.innerHTML = stepsHTML;
    }}

    // Metrics overlay
    const metricsOverlay = document.createElement('div');
    metricsOverlay.style.cssText = 'position:absolute;top:10px;left:10px;color:#ccc;font:12px monospace;background:rgba(5,5,16,0.8);padding:10px 14px;border-radius:8px;pointer-events:none;line-height:1.6;';
    container.appendChild(metricsOverlay);

    // Steps overlay
    const stepsOverlay = document.createElement('div');
    stepsOverlay.style.cssText = 'position:absolute;top:10px;right:10px;color:#ccc;font:12px monospace;background:rgba(5,5,16,0.8);padding:10px 14px;border-radius:8px;pointer-events:none;line-height:1.8;';
    container.appendChild(stepsOverlay);

    // Apply first frame
    applyFrame(0);

    // Playback controls
    let currentFrame = 0;
    let playing = false;
    let lastTime = 0;
    const playbackSpeed = {playback_speed};
    const frameDt = 1.0 / {fps};

    const playBtn = document.getElementById('vPlayBtn');
    const slider = document.getElementById('vSlider');
    const label = document.getElementById('vFrameLabel');

    if (playBtn) playBtn.addEventListener('click', () => {{
        playing = !playing;
        playBtn.textContent = playing ? 'Pause' : 'Play';
    }});
    if (slider) slider.addEventListener('input', () => {{
        currentFrame = parseInt(slider.value);
        applyFrame(currentFrame);
    }});

    // Animation loop
    function animate(t) {{
        requestAnimationFrame(animate);

        if (playing) {{
            const elapsed = (t - lastTime) / 1000;
            if (elapsed > frameDt / playbackSpeed) {{
                currentFrame = (currentFrame + 1) % {n_frames};
                applyFrame(currentFrame);
                if (slider) slider.value = currentFrame;
                if (label) label.textContent = currentFrame + ' / ' + ({n_frames} - 1);
                lastTime = t;
            }}
        }}

        // Pulse infection site
        const pulse = 0.12 + 0.05 * Math.sin(t * 0.003);
        infMat.opacity = pulse;

        controls.update();
        renderer.render(scene, camera);
    }}
    requestAnimationFrame(animate);
}})();
</script>
'''


# ═══════════════════════════════════════════════════════════════════════
# Main Content
# ═══════════════════════════════════════════════════════════════════════

frames = st.session_state.diap_frames

if frames:
    # Render 3D viewer
    viewer_html = _build_vessel_viewer(frames, playback_speed)
    st.components.v1.html(viewer_html, height=700, scrolling=False)

    # Metrics summary
    st.subheader("Simulation Summary")
    first_frame = frames[0]
    last_frame = frames[-1]
    m = last_frame["metrics"]
    sc = m["state_counts"]

    cols = st.columns(8)
    state_labels = ["Flowing", "Marginating", "Rolling", "Activating",
                    "Arrested", "Crawling", "Transmigrating", "Migrated"]
    state_emojis = ["🔵", "🟡", "🟨", "🟧", "🟥", "🔴", "🟣", "🟢"]
    for i, (label, emoji) in enumerate(zip(state_labels, state_emojis)):
        count = sc.get(label.lower(), 0)
        cols[i].metric(f"{emoji} {label}", count)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Simulation time", f"{m['time']:.1f} s")
    col2.metric("Avg rolling velocity", f"{m['avg_rolling_velocity']:.0f} μm/s")
    col3.metric("Junction integrity", f"{m['avg_junction_integrity']*100:.0f}%")
    col4.metric("Total frames", len(frames))

    # State distribution over time
    st.subheader("State Distribution Over Time")
    import_ok = True
    try:
        import pandas as pd
    except ImportError:
        import_ok = False

    if import_ok:
        time_series = []
        for f in frames:
            row = {"time": f["time"]}
            for s in state_labels:
                row[s] = f["metrics"]["state_counts"].get(s.lower(), 0)
            time_series.append(row)
        df = pd.DataFrame(time_series)
        st.area_chart(df.set_index("time"), height=300)

else:
    # No simulation yet — show instructions
    st.info(
        "Select a **preset scenario** in the sidebar and click **Run Simulation** "
        "to start the diapedesis visualization."
    )

    # Show overview diagram
    st.subheader("The 6 Steps of Diapedesis")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
**Step 1: Cytokine Production**
Macrophages at infection site release TNF-α and IL-1β.
Endothelial cells upregulate E-selectin and P-selectin.

**Step 2: Selectin-Mediated Rolling**
PSGL-1 on leukocytes binds E/P-selectin with low affinity.
Fast on/off kinetics → characteristic rolling motion (~30 μm/s).

**Step 3: Chemokine-Mediated Integrin Activation**
Endothelial chemokines (CXCL8, CCL2) bind leukocyte receptors.
Inside-out signaling: RAP1 → talin → kindlin-3 → integrin extension.
""")

    with col2:
        st.markdown("""
**Step 4: Integrin-Mediated Firm Adhesion**
LFA-1 binds ICAM-1 (high affinity, extended conformation).
VLA-4 binds VCAM-1. Complete arrest of leukocyte.

**Step 5: Transmigration**
Paracellular: squeeze between endothelial cells.
Requires PECAM-1 (CD31), JAM-A, CD99, VE-cadherin disruption.

**Step 6: Migration to Infection**
Chemotaxis through ECM following chemokine gradient.
Neutrophils arrive in minutes; monocytes in hours.
""")

    # LAD diseases reference
    with st.expander("Leukocyte Adhesion Deficiency (LAD) Diseases"):
        st.markdown("""
| Disease | Defect | Molecular Basis | Phenotype |
|---------|--------|----------------|-----------|
| **LAD-1** | No β2 integrins (CD18) | *ITGB2* mutation | No firm adhesion → recurrent infections, leukocytosis |
| **LAD-2** | No selectin ligands | *SLC35C1* (fucose transporter) | No rolling → no recruitment at all |
| **LAD-3** | No integrin activation | *FERMT3* (kindlin-3) mutation | Rolling but no arrest → bleeding + infections |
""")

    # Key molecules reference
    with st.expander("Key Adhesion Molecules"):
        st.markdown("""
| Molecule | Family | On | Ligand | Step |
|----------|--------|-----|--------|------|
| E-selectin (CD62E) | Selectin | Endothelium | PSGL-1, CD44 | Rolling |
| P-selectin (CD62P) | Selectin | Endothelium | PSGL-1 | Rolling |
| L-selectin (CD62L) | Selectin | Leukocyte | GlyCAM-1, CD34 | Rolling |
| LFA-1 (CD11a/CD18) | β2 integrin | Leukocyte | ICAM-1, ICAM-2 | Arrest |
| VLA-4 (α4β1) | β1 integrin | Leukocyte | VCAM-1 | Arrest |
| MAC-1 (CD11b/CD18) | β2 integrin | Neutrophil | ICAM-1, iC3b | Arrest |
| ICAM-1 (CD54) | Ig-SF | Endothelium | LFA-1, MAC-1 | Arrest |
| VCAM-1 (CD106) | Ig-SF | Endothelium | VLA-4 | Arrest |
| PECAM-1 (CD31) | Ig-SF | Both | PECAM-1 (homo) | Transmigration |
| VE-cadherin (CD144) | Cadherin | Endothelium | VE-cadherin (homo) | Transmigration |
| JAM-A | JAM | Endothelium | LFA-1, JAM-A | Transmigration |
""")
