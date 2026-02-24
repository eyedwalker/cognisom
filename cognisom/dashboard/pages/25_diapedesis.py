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
import os
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

    # ── Rendering Mode ───────────────────────────────────────────────
    st.subheader("Rendering")
    render_mode = st.radio(
        "Viewer",
        ["Three.js (Browser)", "Omniverse RTX (HD)"],
        index=0,
        help=(
            "**Three.js**: Runs in browser, works everywhere.\n\n"
            "**Omniverse RTX**: Subsurface scattering, volumetrics, "
            "PBR materials. Requires Kit running on GPU server."
        ),
    )
    use_omniverse = render_mode == "Omniverse RTX (HD)"

    if use_omniverse:
        omni_host = st.text_input(
            "Kit Browser URL",
            value="/kit",
            help="Browser-accessible Kit endpoint (proxied through nginx HTTPS)",
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

    gpu_tag = "GPU (Warp/CUDA)" if sim.gpu_enabled else "CPU (NumPy)"
    with st.spinner(f"Running diapedesis simulation ({duration}s at dt=0.01, {gpu_tag})..."):
        t0 = time.time()
        sim.initialize()
        frames = sim.run(duration=float(duration), fps=fps)
        elapsed = time.time() - t0

        st.session_state.diap_frames = frames
        st.success(
            f"Simulation complete: {len(frames)} frames in {elapsed:.1f}s "
            f"[{gpu_tag}] "
            f"({sim.config.n_leukocytes} leukocytes, {sim.config.n_rbc} RBCs)"
        )


# ═══════════════════════════════════════════════════════════════════════
# Omniverse RTX Viewer (HTTP MJPEG streaming from Kit)
# ═══════════════════════════════════════════════════════════════════════

def _render_omniverse_viewer(frames, streaming_url: str):
    """Render the Omniverse Kit viewport via HTTP MJPEG streaming.

    Sends simulation frames to Kit via REST API, then displays the
    RTX-rendered viewport as an MJPEG stream (no WebRTC needed).
    Falls back to Three.js if Kit is unreachable.

    Args:
        streaming_url: Browser-accessible URL for Kit (e.g. "/kit" for
            nginx proxy, or "http://host:8211" for direct access).
            Server-side Python calls always use localhost:8211 directly.
    """
    import urllib.request

    st.markdown("### Omniverse RTX Viewer")

    # Server-side URL (Python → Kit)
    # Inside Docker, localhost is the container itself. Use KIT_SERVER_URL env var
    # or fall back to host.docker.internal (requires --add-host flag on docker run).
    kit_server = os.environ.get(
        "KIT_SERVER_URL", "http://host.docker.internal:8211"
    )
    # Browser-side URL (JS/HTML → Kit, through nginx HTTPS proxy)
    kit_browser = streaming_url.rstrip("/")

    # Check if Kit is reachable (server-side)
    kit_status = None
    try:
        req = urllib.request.Request(f"{kit_server}/status", method="GET")
        req.add_header("Accept", "application/json")
        with urllib.request.urlopen(req, timeout=3) as resp:
            kit_status = json.loads(resp.read())
    except Exception:
        pass

    if not kit_status:
        st.warning(
            "Omniverse Kit not detected. "
            "Make sure the Kit container is running with the cognisom.sim extension."
        )
        with st.expander("Setup Instructions"):
            st.markdown("""
**To enable Omniverse RTX rendering:**

1. **Launch Kit container on the GPU server** (L4/L40S):
   ```bash
   docker run -d --name cognisom-kit --gpus all \\
       --restart unless-stopped \\
       -p 8211:8211 \\
       -v /path/to/cognisom.sim:/exts/cognisom.sim:ro \\
       nvcr.io/nvidia/omniverse/kit:105.1.2 \\
       --ext-folder /exts --enable cognisom.sim \\
       --/renderer/active="rtx"
   ```

2. **Ensure nginx proxies** `/kit/` to `localhost:8211`

The extension provides:
- PBR materials with subsurface scattering
- RTX viewport rendered at 1920x1080
- HTTP frame streaming (MJPEG) on port 8211
- REST API for playback control
""")
        st.info("Falling back to Three.js viewer.")
        viewer_html = _build_vessel_viewer(frames, 2.0)
        st.components.v1.html(viewer_html, height=700, scrolling=False)
        return

    # Kit is available
    st.success(
        f"Connected to Kit — {kit_status.get('extension', 'cognisom.sim')} "
        f"({'playing' if kit_status.get('playing') else 'idle'})"
    )

    # Send simulation frames to Kit (server-side, via localhost)
    if frames:
        try:
            payload = json.dumps({"frames": frames}).encode()
            req = urllib.request.Request(
                f"{kit_server}/cognisom/diapedesis",
                data=payload,
                method="POST",
            )
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read())
                st.caption(f"Scene: {result.get('message', 'OK')}")
        except Exception as e:
            st.error(f"Failed to load frames into Kit: {e}")

    # Embed MJPEG viewport stream (browser-side, through HTTPS proxy)
    # The viewer tries WebRTC first (if library is loaded), falls back to MJPEG
    viewer_html = f"""
    <div id="omniViewer" style="width:100%;height:650px;border-radius:8px;overflow:hidden;
         background:#040412;position:relative;">
        <img id="rtxStream" src="{kit_browser}/stream"
             style="width:100%;height:100%;object-fit:contain;"
             onerror="this.style.display='none';document.getElementById('rtxFallback').style.display='flex';" />
        <div id="rtxFallback" style="display:none;width:100%;height:100%;align-items:center;
             justify-content:center;color:#556;font:18px monospace;flex-direction:column;">
            <div style="font-size:28px;margin-bottom:12px;">Omniverse Kit RTX</div>
            <div>Waiting for viewport render...</div>
        </div>
        <div id="rtxBadge" style="position:absolute;top:8px;right:8px;background:rgba(180,140,0,0.85);
             color:white;padding:4px 12px;border-radius:4px;font:13px monospace;z-index:2;">
            MJPEG
        </div>
        <div id="rtxInfo" style="position:absolute;top:8px;left:8px;background:rgba(4,4,18,0.85);
             color:#aab;padding:8px 14px;border-radius:6px;font:11px monospace;z-index:2;
             line-height:1.6;border:1px solid rgba(100,100,150,0.15);">
            <b style="color:#00a0ff">Cognisom Diapedesis</b><br>
            <span id="rtxSimInfo">Connected to Kit</span>
        </div>
        <div id="rtxStatus" style="position:absolute;bottom:8px;left:8px;color:#8899aa;
             font:12px monospace;z-index:2;"></div>
    </div>
    <script>
        setInterval(async () => {{
            try {{
                const r = await fetch("{kit_browser}/status");
                const d = await r.json();
                const el = document.getElementById('rtxStatus');
                const info = document.getElementById('rtxSimInfo');
                if (el) {{
                    const parts = [];
                    if (d.diapedesis_loaded) parts.push(`Frame ${{d.current_frame}}/${{d.frames}}`);
                    parts.push(d.playing ? '\\u25B6 Playing' : '\\u23F8 Idle');
                    parts.push(`Renderer: ${{d.renderer}}`);
                    el.textContent = parts.join(' | ');
                }}
                if (info) {{
                    info.textContent = d.diapedesis_loaded
                        ? `${{d.frames}} frames | ${{d.playing ? 'Playing' : 'Idle'}}`
                        : 'No simulation loaded';
                }}
            }} catch(e) {{}}
        }}, 800);
    </script>
    """
    st.components.v1.html(viewer_html, height=680, scrolling=False)

    # Links to standalone viewers
    col_link1, col_link2 = st.columns(2)
    with col_link1:
        st.caption(
            f"[Open MJPEG viewer ↗]({kit_browser}/streaming/client)"
        )
    with col_link2:
        st.caption(
            f"[Open WebRTC viewer ↗](/rtx-viewer/?server={kit_browser.split('//')[1].split(':')[0] if '://' in kit_browser else 'localhost'})"
        )

    # Playback controls (server-side via localhost)
    col1, col2, col3, col4 = st.columns([1, 1, 6, 1])
    with col1:
        if st.button("▶ Play", key="omni_play"):
            try:
                urllib.request.urlopen(urllib.request.Request(
                    f"{kit_server}/cognisom/diapedesis/play", method="POST"), timeout=3)
            except Exception:
                pass
    with col2:
        if st.button("⏸ Pause", key="omni_pause"):
            try:
                urllib.request.urlopen(urllib.request.Request(
                    f"{kit_server}/cognisom/diapedesis/pause", method="POST"), timeout=3)
            except Exception:
                pass
    with col3:
        n_frames = max(len(frames) - 1, 1)
        frame_idx = st.slider("Frame", 0, n_frames, 0, key="omni_frame_slider")
        try:
            urllib.request.urlopen(urllib.request.Request(
                f"{kit_server}/cognisom/diapedesis/seek?frame={frame_idx}",
                method="POST"), timeout=3)
        except Exception:
            pass
    with col4:
        if st.button("⏹ Stop", key="omni_stop"):
            try:
                urllib.request.urlopen(urllib.request.Request(
                    f"{kit_server}/cognisom/diapedesis/stop", method="POST"), timeout=3)
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════
# Three.js Vessel Cutaway Viewer
# ═══════════════════════════════════════════════════════════════════════

def _build_vessel_viewer(frames, playback_speed=2.0):
    """Build Three.js 3D vessel cutaway with realistic molecular geometries.

    Features:
    - Selectin lollipops (C-type lectin head + SCR stalk) on endothelium
    - Integrin bent/extended conformational change on leukocytes
    - ICAM-1 bead-rods on endothelium
    - PECAM-1 dimers at endothelial junctions
    - Biconcave RBCs (Evans-Fung parametric)
    - Neutrophils with microvilli + lobulated nucleus
    - Tissue scene: complement-opsonized bacteria, macrophages, fibrin, ECM
    - Chemokine gradient cloud
    """
    if not frames:
        return ""

    frames_json = json.dumps(frames)
    n_frames = len(frames)
    first = frames[0]
    R = first["vessel_radius"]
    L = first["vessel_length"]

    return f'''
<div id="vesselViewer" style="width:100%;height:650px;border-radius:8px;overflow:hidden;background:#040412;position:relative;"></div>
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

    /* ═══════════ Scene Setup ═══════════ */
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x040412);
    scene.fog = new THREE.FogExp2(0x040412, 0.003);

    const camera = new THREE.PerspectiveCamera(50, W/H, 0.1, 2000);
    camera.position.set(L*0.5, R*2.2, R*3.5);
    camera.lookAt(L*0.5, -R*0.3, 0);

    const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
    renderer.setSize(W, H);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.target.set(L*0.5, -R*0.3, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.update();

    /* ═══════════ Lighting ═══════════ */
    scene.add(new THREE.AmbientLight(0x404060, 0.6));
    const keyLight = new THREE.DirectionalLight(0xfff5e8, 1.0);
    keyLight.position.set(L, R*3, R*2);
    keyLight.castShadow = true;
    scene.add(keyLight);
    const fillLight = new THREE.DirectionalLight(0x6688cc, 0.4);
    fillLight.position.set(-L*0.5, R, -R*2);
    scene.add(fillLight);
    const rimLight = new THREE.DirectionalLight(0xff8866, 0.2);
    rimLight.position.set(L*0.5, -R*3, 0);
    scene.add(rimLight);

    /* ═══════════ Geometry Builders ═══════════ */

    // --- Biconcave RBC (Evans-Fung equation) ---
    function buildBiconcaveRBC() {{
        const D = 1.0; // unit disc, scaled later
        const segs = 32, rings = 16;
        const verts = [], normals = [], indices = [];
        for (let j = 0; j <= rings; j++) {{
            const v = j / rings;
            const r = v * D * 0.5;
            const rn = r / (D * 0.5);
            const rn2 = rn*rn, rn4 = rn2*rn2;
            const sq = 1.0 - 4.0*rn2;
            const h = sq > 0 ? D*0.5*Math.sqrt(sq)*(0.0518 + 2.0026*rn2 - 4.491*rn4) : 0;
            for (let i = 0; i <= segs; i++) {{
                const u = i / segs;
                const theta = u * Math.PI * 2;
                const x = r * Math.cos(theta);
                const z = r * Math.sin(theta);
                // Top half
                verts.push(x, h, z);
                // Approximate normals
                const dx = 0.001;
                const rp = Math.min(0.499, (r+dx)/(D*0.5));
                const sq2 = 1.0 - 4.0*rp*rp;
                const hp = sq2 > 0 ? D*0.5*Math.sqrt(sq2)*(0.0518 + 2.0026*rp*rp - 4.491*rp*rp*rp*rp) : 0;
                const dh = (hp - h) / dx;
                const nx = -dh * Math.cos(theta);
                const nz = -dh * Math.sin(theta);
                const ny = 1.0;
                const nl = Math.sqrt(nx*nx + ny*ny + nz*nz) || 1;
                normals.push(nx/nl, ny/nl, nz/nl);
            }}
        }}
        // Bottom half (mirror)
        const topCount = verts.length / 3;
        for (let j = 0; j <= rings; j++) {{
            for (let i = 0; i <= segs; i++) {{
                const idx = (j * (segs+1) + i) * 3;
                verts.push(verts[idx], -verts[idx+1], verts[idx+2]);
                normals.push(normals[idx], -normals[idx+1], normals[idx+2]);
            }}
        }}
        // Indices for top
        for (let j = 0; j < rings; j++) {{
            for (let i = 0; i < segs; i++) {{
                const a = j*(segs+1)+i, b = a+segs+1, c = a+1, d = b+1;
                indices.push(a, b, c, c, b, d);
            }}
        }}
        // Indices for bottom
        for (let j = 0; j < rings; j++) {{
            for (let i = 0; i < segs; i++) {{
                const a = topCount+j*(segs+1)+i, b = a+segs+1, c = a+1, d = b+1;
                indices.push(a, c, b, c, d, b);
            }}
        }}
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
        geo.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
        geo.setIndex(indices);
        return geo;
    }}

    // --- Selectin lollipop (C-type lectin head + EGF + SCR stalk) ---
    function buildSelectin() {{
        const g = new THREE.Group();
        // SCR stalk: 5 small beads
        const bead = new THREE.SphereGeometry(0.25, 6, 4);
        const stalkMat = new THREE.MeshPhongMaterial({{ color: 0xccaa22 }});
        for (let i = 0; i < 5; i++) {{
            const m = new THREE.Mesh(bead, stalkMat);
            m.position.y = i * 0.45 + 0.2;
            g.add(m);
        }}
        // EGF domain
        const egf = new THREE.Mesh(new THREE.SphereGeometry(0.3, 6, 4),
            new THREE.MeshPhongMaterial({{ color: 0xddbb33 }}));
        egf.position.y = 2.5;
        g.add(egf);
        // Lectin head (oblate spheroid)
        const head = new THREE.Mesh(new THREE.SphereGeometry(0.55, 8, 6),
            new THREE.MeshPhongMaterial({{ color: 0xffaa00, emissive: 0x332200 }}));
        head.scale.set(1, 0.7, 1);
        head.position.y = 3.1;
        g.add(head);
        return g;
    }}

    // --- ICAM-1 (5 Ig-domain bead rod with kink) ---
    function buildICAM1() {{
        const g = new THREE.Group();
        const domainGeo = new THREE.SphereGeometry(0.22, 6, 4);
        for (let i = 0; i < 5; i++) {{
            const isD1 = (i === 4);
            const mat = new THREE.MeshPhongMaterial({{
                color: isD1 ? 0x4488ff : 0x3366aa,
                emissive: isD1 ? 0x112244 : 0x000000,
            }});
            const m = new THREE.Mesh(domainGeo, mat);
            // Kink between D2 and D3 (index 2 and 3)
            let y = i * 0.5;
            let x = 0;
            if (i >= 3) {{
                const kinkAngle = 0.4; // ~23 degrees
                x = (i - 2) * 0.5 * Math.sin(kinkAngle);
                y = 1.0 + (i - 2) * 0.5 * Math.cos(kinkAngle);
            }}
            m.position.set(x, y, 0);
            m.scale.set(1, 1.4, 1); // elongated Ig domains
            g.add(m);
        }}
        return g;
    }}

    // --- PECAM-1 dimer pair at junction ---
    function buildPECAM1Pair() {{
        const g = new THREE.Group();
        const domainGeo = new THREE.SphereGeometry(0.18, 6, 4);
        const mat = new THREE.MeshPhongMaterial({{ color: 0x33bb88, transparent: true, opacity: 0.9 }});
        const matTip = new THREE.MeshPhongMaterial({{ color: 0x55ddaa, emissive: 0x112211, transparent: true, opacity: 0.9 }});
        // Two rods pointing at each other
        for (let side = -1; side <= 1; side += 2) {{
            for (let i = 0; i < 6; i++) {{
                const isBinding = (i >= 4);
                const m = new THREE.Mesh(domainGeo, isBinding ? matTip : mat);
                m.position.set(0, side * (0.5 + i * 0.4), 0);
                m.scale.set(1, 1.3, 1);
                g.add(m);
            }}
        }}
        return g;
    }}

    // --- Integrin (bent vs extended, driven by activation 0-1) ---
    function buildIntegrinBent() {{
        const g = new THREE.Group();
        const legMat = new THREE.MeshPhongMaterial({{ color: 0x448899 }});
        const headMat = new THREE.MeshPhongMaterial({{ color: 0x55aabb, emissive: 0x112233 }});
        // Compact bent shape: headpiece near membrane
        const headDisc = new THREE.Mesh(new THREE.CylinderGeometry(0.4, 0.4, 0.15, 8),
            headMat);
        headDisc.position.y = 0.5;
        g.add(headDisc);
        const alphaI = new THREE.Mesh(new THREE.SphereGeometry(0.2, 6, 4), headMat);
        alphaI.position.y = 0.7;
        g.add(alphaI);
        // Short bent legs
        const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.08, 0.6, 4), legMat);
        leg.position.set(0.15, 0.15, 0);
        leg.rotation.z = 0.5;
        g.add(leg);
        const leg2 = leg.clone();
        leg2.position.set(-0.15, 0.15, 0);
        leg2.rotation.z = -0.5;
        g.add(leg2);
        return g;
    }}

    function buildIntegrinExtended() {{
        const g = new THREE.Group();
        const legMat = new THREE.MeshPhongMaterial({{ color: 0x00ccdd }});
        const headMat = new THREE.MeshPhongMaterial({{ color: 0x00ffff, emissive: 0x003344 }});
        // Tall extended: two straight legs, headpiece at top
        const leg1 = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 1.8, 4), legMat);
        leg1.position.set(0.12, 0.9, 0);
        leg1.rotation.z = 0.06;
        g.add(leg1);
        const leg2 = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 1.8, 4), legMat);
        leg2.position.set(-0.12, 0.9, 0);
        leg2.rotation.z = -0.06;
        g.add(leg2);
        // Headpiece (beta-propeller disc + alphaI knob)
        const disc = new THREE.Mesh(new THREE.CylinderGeometry(0.35, 0.35, 0.12, 8), headMat);
        disc.position.y = 1.85;
        g.add(disc);
        const knob = new THREE.Mesh(new THREE.SphereGeometry(0.18, 6, 4), headMat);
        knob.position.y = 2.05;
        g.add(knob);
        // Hybrid domain swung out
        const hybrid = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.3, 0.15),
            new THREE.MeshPhongMaterial({{ color: 0x00eeff }}));
        hybrid.position.set(0.45, 1.7, 0);
        g.add(hybrid);
        return g;
    }}

    // --- Neutrophil (lumpy sphere + microvilli + lobulated nucleus) ---
    function buildNeutrophil(radius) {{
        const g = new THREE.Group();
        // Body: sphere with vertex noise
        const bodyGeo = new THREE.SphereGeometry(1, 20, 16);
        const posArr = bodyGeo.attributes.position.array;
        for (let i = 0; i < posArr.length; i += 3) {{
            const len = Math.sqrt(posArr[i]*posArr[i]+posArr[i+1]*posArr[i+1]+posArr[i+2]*posArr[i+2]);
            const noise = 1.0 + 0.06 * (Math.sin(posArr[i]*11)*Math.cos(posArr[i+1]*13)*Math.sin(posArr[i+2]*17));
            const s = noise / len;
            posArr[i] *= s; posArr[i+1] *= s; posArr[i+2] *= s;
        }}
        bodyGeo.computeVertexNormals();
        const bodyMat = new THREE.MeshPhongMaterial({{
            color: 0xddddee, transparent: true, opacity: 0.75,
            side: THREE.DoubleSide,
        }});
        const body = new THREE.Mesh(bodyGeo, bodyMat);
        g.add(body);

        // Microvilli: small spikes on surface
        const mvGeo = new THREE.CylinderGeometry(0.02, 0.03, 0.15, 3);
        const mvMat = new THREE.MeshPhongMaterial({{ color: 0xccccdd }});
        const tipMat = new THREE.MeshPhongMaterial({{ color: 0xffcc00 }}); // PSGL-1 at tips
        for (let k = 0; k < 50; k++) {{
            const phi = Math.acos(2*Math.random()-1);
            const theta = Math.random()*Math.PI*2;
            const nx = Math.sin(phi)*Math.cos(theta);
            const ny = Math.sin(phi)*Math.sin(theta);
            const nz = Math.cos(phi);
            const mv = new THREE.Mesh(mvGeo, mvMat);
            mv.position.set(nx*1.0, ny*1.0, nz*1.0);
            mv.lookAt(nx*2, ny*2, nz*2);
            mv.rotateX(Math.PI/2);
            g.add(mv);
            // PSGL-1 tip (yellow dot) every 5th microvillus
            if (k % 5 === 0) {{
                const tip = new THREE.Mesh(new THREE.SphereGeometry(0.03, 4, 3), tipMat);
                tip.position.set(nx*1.15, ny*1.15, nz*1.15);
                g.add(tip);
            }}
        }}

        // Nucleus: 3-4 lobes (multilobed = neutrophil signature)
        const lobeMat = new THREE.MeshPhongMaterial({{ color: 0x5544aa, transparent: true, opacity: 0.6 }});
        const lobeGeo = new THREE.SphereGeometry(0.3, 8, 6);
        const lobePositions = [[0.15,0.1,0], [-0.15,-0.05,0.1], [0,-0.1,-0.15], [0.1,0.15,-0.05]];
        lobePositions.forEach(p => {{
            const lobe = new THREE.Mesh(lobeGeo, lobeMat);
            lobe.position.set(p[0], p[1], p[2]);
            lobe.scale.set(1, 0.9, 0.8);
            g.add(lobe);
        }});
        // Thin connections between lobes
        const connMat = new THREE.MeshPhongMaterial({{ color: 0x443388, transparent: true, opacity: 0.4 }});
        for (let i = 0; i < lobePositions.length - 1; i++) {{
            const conn = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 0.3, 3), connMat);
            const p1 = lobePositions[i], p2 = lobePositions[i+1];
            conn.position.set((p1[0]+p2[0])/2, (p1[1]+p2[1])/2, (p1[2]+p2[2])/2);
            g.add(conn);
        }}

        g.scale.set(radius, radius, radius);
        return g;
    }}

    // --- Bacterium (rod-shaped + complement coating) ---
    function buildBacterium() {{
        const g = new THREE.Group();
        // Rod body (capsule = cylinder + hemisphere caps)
        const bodyMat = new THREE.MeshPhongMaterial({{ color: 0x336633, emissive: 0x112211 }});
        const cyl = new THREE.Mesh(new THREE.CylinderGeometry(0.8, 0.8, 3, 8, 1), bodyMat);
        g.add(cyl);
        const capGeo = new THREE.SphereGeometry(0.8, 8, 6, 0, Math.PI*2, 0, Math.PI/2);
        const cap1 = new THREE.Mesh(capGeo, bodyMat);
        cap1.position.y = 1.5;
        g.add(cap1);
        const cap2 = new THREE.Mesh(capGeo, bodyMat);
        cap2.position.y = -1.5;
        cap2.rotation.x = Math.PI;
        g.add(cap2);
        // Complement coating (C3b/iC3b): scattered gold spheres on surface
        const compMat = new THREE.MeshPhongMaterial({{
            color: 0xcc9933, emissive: 0x332200,
            transparent: true, opacity: 0.8,
        }});
        const compGeo = new THREE.SphereGeometry(0.15, 4, 3);
        for (let i = 0; i < 25; i++) {{
            const theta = Math.random() * Math.PI * 2;
            const y = (Math.random() - 0.5) * 3.5;
            const cr = 0.85 + Math.random() * 0.1;
            const comp = new THREE.Mesh(compGeo, compMat);
            comp.position.set(cr*Math.cos(theta), y, cr*Math.sin(theta));
            g.add(comp);
        }}
        // Flagellum (thin helix)
        const flagPts = [];
        for (let t = 0; t < 4; t += 0.1) {{
            flagPts.push(new THREE.Vector3(
                0.3*Math.sin(t*4), -1.8 - t*1.2, 0.3*Math.cos(t*4)));
        }}
        const flagCurve = new THREE.CatmullRomCurve3(flagPts);
        const flagGeo = new THREE.TubeGeometry(flagCurve, 20, 0.04, 4, false);
        const flagMat = new THREE.MeshPhongMaterial({{ color: 0x558855 }});
        g.add(new THREE.Mesh(flagGeo, flagMat));
        return g;
    }}

    // --- Macrophage (large amoeboid + pseudopods) ---
    function buildMacrophage() {{
        const g = new THREE.Group();
        const bodyGeo = new THREE.SphereGeometry(1, 16, 12);
        const posArr = bodyGeo.attributes.position.array;
        // Heavy vertex displacement for ruffled membrane
        for (let i = 0; i < posArr.length; i += 3) {{
            const len = Math.sqrt(posArr[i]*posArr[i]+posArr[i+1]*posArr[i+1]+posArr[i+2]*posArr[i+2]);
            const noise = 1.0 + 0.15 * (Math.sin(posArr[i]*7)*Math.cos(posArr[i+1]*9)*Math.sin(posArr[i+2]*11));
            const s = noise / len;
            posArr[i] *= s; posArr[i+1] *= s; posArr[i+2] *= s;
        }}
        bodyGeo.computeVertexNormals();
        const bodyMat = new THREE.MeshPhongMaterial({{
            color: 0xdd5522, emissive: 0x221100,
            transparent: true, opacity: 0.8,
        }});
        g.add(new THREE.Mesh(bodyGeo, bodyMat));
        // Pseudopods (3 extending arms)
        const podMat = new THREE.MeshPhongMaterial({{ color: 0xcc4411, transparent: true, opacity: 0.7 }});
        const angles = [0, 2.1, 4.2];
        angles.forEach(a => {{
            const pod = new THREE.Mesh(new THREE.SphereGeometry(0.5, 8, 6), podMat);
            pod.position.set(Math.cos(a)*1.3, Math.sin(a)*0.3, Math.sin(a)*1.1);
            pod.scale.set(1.5, 0.5, 0.7);
            g.add(pod);
        }});
        // Kidney-shaped nucleus
        const nucMat = new THREE.MeshPhongMaterial({{ color: 0x663322, transparent: true, opacity: 0.5 }});
        const nuc = new THREE.Mesh(new THREE.TorusGeometry(0.35, 0.2, 6, 8, Math.PI*1.3), nucMat);
        g.add(nuc);
        return g;
    }}

    /* ═══════════ Vessel Wall ═══════════ */
    const vesselGeo = new THREE.CylinderGeometry(R, R, L, 48, 1, true, 0, Math.PI);
    vesselGeo.rotateZ(Math.PI / 2);
    vesselGeo.translate(L/2, 0, 0);
    const vesselMat = new THREE.MeshPhongMaterial({{
        color: 0xe8b0b0, transparent: true, opacity: 0.18,
        side: THREE.DoubleSide, depthWrite: false,
    }});
    scene.add(new THREE.Mesh(vesselGeo, vesselMat));

    // Wireframe
    const wireGeo = new THREE.CylinderGeometry(R, R, L, 24, 4, true, 0, Math.PI);
    wireGeo.rotateZ(Math.PI / 2);
    wireGeo.translate(L/2, 0, 0);
    scene.add(new THREE.Mesh(wireGeo, new THREE.MeshBasicMaterial({{
        color: 0xcc8888, wireframe: true, transparent: true, opacity: 0.1
    }})));

    // Basement membrane (thin layer outside vessel)
    const bmGeo = new THREE.CylinderGeometry(R*1.04, R*1.04, L*0.95, 32, 1, true, 0, Math.PI);
    bmGeo.rotateZ(Math.PI/2); bmGeo.translate(L/2, 0, 0);
    scene.add(new THREE.Mesh(bmGeo, new THREE.MeshPhongMaterial({{
        color: 0x886644, transparent: true, opacity: 0.12,
        side: THREE.DoubleSide, depthWrite: false, wireframe: true,
    }})));

    /* ═══════════ Load Frame Data ═══════════ */
    const frames = {frames_json};
    const nLeuko = frames[0].leukocyte_positions.length;
    const nRBC = frames[0].rbc_positions.length;
    const nEndo = frames[0].endo_positions.length;

    /* ═══════════ Biconcave RBCs (InstancedMesh) ═══════════ */
    const rbcGeo = buildBiconcaveRBC();
    const rbcMat = new THREE.MeshPhongMaterial({{
        color: 0xcc2222, transparent: true, opacity: 0.75,
        shininess: 40,
    }});
    const rbcMesh = new THREE.InstancedMesh(rbcGeo, rbcMat, nRBC);
    rbcMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    scene.add(rbcMesh);

    /* ═══════════ Neutrophil Leukocytes (Individual Groups) ═══════════ */
    const leukoGroups = [];
    const integrinBentProto = buildIntegrinBent();
    const integrinExtProto = buildIntegrinExtended();
    for (let i = 0; i < nLeuko; i++) {{
        const rad = frames[0].leukocyte_radii[i] || 6;
        const ng = buildNeutrophil(rad);
        // Add integrins (4 per leukocyte, on lower hemisphere)
        const integrins = [];
        for (let j = 0; j < 4; j++) {{
            const angle = (j / 4) * Math.PI * 2;
            const ig = {{ bent: integrinBentProto.clone(), ext: integrinExtProto.clone(), angle: angle }};
            ig.bent.position.set(Math.cos(angle)*rad*0.9, -rad*0.7, Math.sin(angle)*rad*0.9);
            ig.bent.scale.set(rad*0.4, rad*0.4, rad*0.4);
            ig.ext.position.copy(ig.bent.position);
            ig.ext.scale.copy(ig.bent.scale);
            ig.ext.visible = false;
            ng.add(ig.bent);
            ng.add(ig.ext);
            integrins.push(ig);
        }}
        scene.add(ng);
        leukoGroups.push({{ group: ng, integrins: integrins, radius: rad }});
    }}

    /* ═══════════ Endothelial Cells (InstancedMesh, colored) ═══════════ */
    const endoGeo = new THREE.SphereGeometry(1, 6, 4);
    endoGeo.scale(1, 1, 0.12);
    const endoMat = new THREE.MeshPhongMaterial({{ vertexColors: false }});
    const endoMesh = new THREE.InstancedMesh(endoGeo, endoMat, nEndo);
    const endoColors = new Float32Array(nEndo * 3);
    endoMesh.instanceColor = new THREE.InstancedBufferAttribute(endoColors, 3);
    scene.add(endoMesh);

    /* ═══════════ Surface Molecules on Endothelium ═══════════ */
    // Selectin lollipops (placed at endothelial positions, scaled by selectin expr)
    const selectinGroup = new THREE.Group();
    const selectinInstances = [];
    const ep0 = frames[0].endo_positions;
    const se0 = frames[0].endo_selectin_expr;
    for (let i = 0; i < nEndo; i++) {{
        if (se0[i] < 0.1) continue;
        const n = Math.floor(se0[i] * 3) + 1; // 1-3 selectins per endo cell
        const p = ep0[i];
        const yz = Math.sqrt(p[1]*p[1] + p[2]*p[2]) || 1;
        for (let j = 0; j < n; j++) {{
            const sel = buildSelectin();
            const offset = (j - n/2) * 2;
            sel.position.set(p[0] + offset, p[1], p[2]);
            // Orient toward vessel center (inward)
            sel.lookAt(p[0] + offset, 0, 0);
            sel.rotateX(-Math.PI/2);
            sel.scale.set(1.2, 1.2, 1.2);
            selectinGroup.add(sel);
            selectinInstances.push({{ mesh: sel, endoIdx: i }});
        }}
    }}
    scene.add(selectinGroup);

    // ICAM-1 on endothelium (blue bead-rods)
    const icamGroup = new THREE.Group();
    for (let i = 0; i < nEndo; i++) {{
        if (se0[i] < 0.2) continue; // ICAM-1 upregulated with inflammation
        const p = ep0[i];
        const icam = buildICAM1();
        icam.position.set(p[0] + 3, p[1], p[2]);
        icam.lookAt(p[0] + 3, 0, 0);
        icam.rotateX(-Math.PI/2);
        icam.scale.set(1.0, 1.0, 1.0);
        icamGroup.add(icam);
    }}
    scene.add(icamGroup);

    // PECAM-1 at junctions (between adjacent endothelial cells)
    const pecamGroup = new THREE.Group();
    const pecamInstances = [];
    for (let i = 0; i < nEndo - 1; i++) {{
        const p1 = ep0[i], p2 = ep0[i+1];
        const dx = p2[0]-p1[0], dy = p2[1]-p1[1], dz = p2[2]-p1[2];
        const dist = Math.sqrt(dx*dx+dy*dy+dz*dz);
        if (dist < 20) {{ // Adjacent cells
            const pecam = buildPECAM1Pair();
            pecam.position.set((p1[0]+p2[0])/2, (p1[1]+p2[1])/2, (p1[2]+p2[2])/2);
            pecam.lookAt(pecam.position.x, 0, 0);
            pecam.scale.set(0.8, 0.8, 0.8);
            pecamGroup.add(pecam);
            pecamInstances.push({{ mesh: pecam, endoIdx: i }});
        }}
    }}
    scene.add(pecamGroup);

    /* ═══════════ Tissue Scene (Static) ═══════════ */
    const tissueGroup = new THREE.Group();

    // --- Bacteria (positions from simulation, complement-opsonized) ---
    const bacteriaMeshes = [];
    const bp0 = frames[0].bacteria_positions || [];
    const nBacteria = bp0.length;
    const bacteriaPositions = bp0;
    for (let i = 0; i < nBacteria; i++) {{
        const bact = buildBacterium();
        const bp = bp0[i];
        bact.position.set(bp[0], bp[1], bp[2]);
        bact.rotation.set(Math.random()*0.5, Math.random()*Math.PI*2, Math.random()*0.5);
        bact.scale.set(1.5, 1.5, 1.5);
        bact.userData.baseScale = 1.5;
        tissueGroup.add(bact);
        bacteriaMeshes.push(bact);
    }}

    // --- Macrophages (2-3, near bacteria) ---
    for (let i = 0; i < 2; i++) {{
        const mac = buildMacrophage();
        const bp = bacteriaPositions[i];
        mac.position.set(bp[0] + (i===0?-5:5), bp[1] - 3, bp[2] + (i===0?3:-3));
        mac.scale.set(4, 4, 4);
        tissueGroup.add(mac);
    }}

    // --- Fibrin mesh (pale yellow fibers) ---
    const fibrinMat = new THREE.MeshPhongMaterial({{
        color: 0xeee8cc, transparent: true, opacity: 0.35,
    }});
    for (let i = 0; i < 40; i++) {{
        const len = 5 + Math.random()*12;
        const fib = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.12, len, 3), fibrinMat);
        fib.position.set(
            L*0.2 + Math.random()*L*0.6,
            -R*1.5 - Math.random()*R*1.5,
            (Math.random()-0.5)*R*1.0
        );
        fib.rotation.set(Math.random()*Math.PI, Math.random()*Math.PI, Math.random()*Math.PI);
        tissueGroup.add(fib);
    }}

    // --- ECM collagen fibers (thicker, beige, wavy) ---
    const colMat = new THREE.MeshPhongMaterial({{ color: 0xccbb99, transparent: true, opacity: 0.3 }});
    for (let i = 0; i < 20; i++) {{
        const pts = [];
        const sx = L*0.1 + Math.random()*L*0.8;
        const sy = -R*1.3 - Math.random()*R*1.5;
        const sz = (Math.random()-0.5)*R*1.2;
        for (let t = 0; t < 5; t++) {{
            pts.push(new THREE.Vector3(
                sx + t*5 + Math.sin(t*2)*1.5,
                sy + Math.cos(t*1.7)*0.8,
                sz + Math.sin(t*2.3)*0.6
            ));
        }}
        const curve = new THREE.CatmullRomCurve3(pts);
        const tubeGeo = new THREE.TubeGeometry(curve, 12, 0.25 + Math.random()*0.15, 4, false);
        tissueGroup.add(new THREE.Mesh(tubeGeo, colMat));
    }}

    // --- Chemokine gradient cloud ---
    const chemGeo = new THREE.SphereGeometry(0.4, 4, 3);
    const chemMat = new THREE.MeshBasicMaterial({{
        color: 0x55bbff, transparent: true, opacity: 0.2,
    }});
    const chemParticles = [];
    for (let i = 0; i < 150; i++) {{
        // Dense near infection, sparse near vessel
        const t = Math.pow(Math.random(), 0.7); // bias toward infection
        const cx = L*0.25 + Math.random()*L*0.5;
        const cy = -R*0.5 - t*R*2.2;
        const cz = (Math.random()-0.5)*R*0.8;
        const cm = new THREE.Mesh(chemGeo, chemMat.clone());
        cm.position.set(cx, cy, cz);
        const s = 0.3 + Math.random()*0.5;
        cm.scale.set(s, s, s);
        cm.material.opacity = 0.08 + t*0.2;
        chemParticles.push(cm);
        tissueGroup.add(cm);
    }}
    scene.add(tissueGroup);

    // --- Infection site glow ---
    const infGeo = new THREE.SphereGeometry(R*0.8, 16, 12);
    const infMat = new THREE.MeshBasicMaterial({{ color: 0xff4422, transparent: true, opacity: 0.08 }});
    const infMesh = new THREE.Mesh(infGeo, infMat);
    infMesh.position.set(L*0.5, -R*1.75, 0);
    scene.add(infMesh);
    const infLight = new THREE.PointLight(0xff4422, 0.6, R*5);
    infLight.position.copy(infMesh.position);
    scene.add(infLight);

    /* ═══════════ State Colors ═══════════ */
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

    /* ═══════════ Frame Application ═══════════ */
    function applyFrame(fi) {{
        const f = frames[fi];
        if (!f) return;

        // --- Leukocytes (individual groups) ---
        const lp = f.leukocyte_positions;
        const ls = f.leukocyte_states;
        const lr = f.leukocyte_radii;
        const ia = f.integrin_activation;
        const tp = f.transmigration_progress;
        for (let i = 0; i < nLeuko; i++) {{
            const lg = leukoGroups[i];
            const p = lp[i];
            lg.group.position.set(p[0], p[1], p[2]);

            // State-dependent coloring (tint the body)
            const sc = stateColors[ls[i]] || stateColors[0];
            const bodyMesh = lg.group.children[0]; // first child is body
            if (bodyMesh && bodyMesh.material) {{
                bodyMesh.material.color.setRGB(sc[0], sc[1], sc[2]);
                bodyMesh.material.opacity = ls[i] === 6 ? 0.5 : 0.75; // more transparent when transmigrating
            }}

            // State-dependent deformation
            const rad = lr[i] || 6;
            if (ls[i] === 6) {{ // TRANSMIGRATING: elongate
                const prog = tp[i] || 0;
                lg.group.scale.set(rad*(1-prog*0.3), rad*(1+prog*0.5), rad*(1-prog*0.3));
            }} else if (ls[i] >= 2 && ls[i] <= 5) {{ // ROLLING-CRAWLING: slightly flattened
                lg.group.scale.set(rad*1.1, rad*0.9, rad*1.1);
            }} else {{
                lg.group.scale.set(rad, rad, rad);
            }}

            // Integrin conformation: bent (low) vs extended (high)
            const activation = ia[i] || 0;
            lg.integrins.forEach(ig => {{
                const showExtended = activation > 0.5;
                ig.bent.visible = !showExtended;
                ig.ext.visible = showExtended;
            }});
        }}

        // --- RBCs (biconcave InstancedMesh) ---
        const rp = f.rbc_positions;
        for (let i = 0; i < nRBC; i++) {{
            const p = rp[i];
            dummy.position.set(p[0], p[1], p[2]);
            dummy.scale.set(3.75, 3.75, 3.75);
            dummy.rotation.set(p[0]*0.1, p[1]*0.1, p[2]*0.1);
            dummy.updateMatrix();
            rbcMesh.setMatrixAt(i, dummy.matrix);
        }}
        rbcMesh.instanceMatrix.needsUpdate = true;

        // --- Endothelial cells ---
        const ep = f.endo_positions;
        const ec = f.endo_colors;
        for (let i = 0; i < nEndo; i++) {{
            const p = ep[i];
            dummy.position.set(p[0], p[1], p[2]);
            dummy.scale.set(10, 10, 10);
            dummy.lookAt(p[0], 0, 0);
            dummy.updateMatrix();
            endoMesh.setMatrixAt(i, dummy.matrix);
            endoColors[i*3] = ec[i][0];
            endoColors[i*3+1] = ec[i][1];
            endoColors[i*3+2] = ec[i][2];
        }}
        endoMesh.instanceMatrix.needsUpdate = true;
        endoMesh.instanceColor.needsUpdate = true;

        // --- PECAM-1 opacity (junction integrity) ---
        const ji = f.endo_junction_integrity;
        pecamInstances.forEach(pi => {{
            const integrity = ji[pi.endoIdx] || 1.0;
            pi.mesh.traverse(child => {{
                if (child.material) child.material.opacity = integrity * 0.9;
            }});
        }});

        // --- Bacteria alive / phagocytosis animation ---
        const ba = f.bacteria_alive || [];
        const bph = f.bacteria_phagocytosis || [];
        for (let bi = 0; bi < nBacteria; bi++) {{
            const bm = bacteriaMeshes[bi];
            if (!bm) continue;
            if (!ba[bi]) {{
                bm.visible = false;
            }} else {{
                bm.visible = true;
                const prog = bph[bi] || 0;
                const s = bm.userData.baseScale * (1.0 - prog * 0.8);
                bm.scale.set(s, s, s);
                // Flash red when being engulfed
                if (prog > 0.05) {{
                    const flash = 0.5 + 0.5 * Math.sin(Date.now() * 0.008);
                    bm.traverse(child => {{
                        if (child.material && child.material.emissive) {{
                            child.material.emissive.setRGB(prog * flash * 0.4, 0, 0);
                        }}
                    }});
                }}
            }}
        }}

        // --- Metrics overlay ---
        const m = f.metrics;
        const sc2 = m.state_counts;
        let metricsHTML = '<b style="color:#aabbcc">Diapedesis State</b><br>';
        stateNames.forEach((name, si) => {{
            const count = sc2[name.toLowerCase()] || 0;
            const c = stateColors[si];
            const hex = '#' + new THREE.Color(c[0],c[1],c[2]).getHexString();
            if (count > 0) {{
                metricsHTML += '<span style="color:'+hex+'">\\u25CF</span> '+name+': '+count+'<br>';
            }}
        }});
        metricsHTML += '<br>t = ' + f.time.toFixed(1) + 's';
        metricsHTML += '<br>Rolling v: ' + m.avg_rolling_velocity.toFixed(0) + ' \\u03BCm/s';
        metricsHTML += '<br>Junction: ' + (m.avg_junction_integrity * 100).toFixed(0) + '%';
        metricsHTML += '<br>Integrin: ' + (m.avg_integrin_activation * 100).toFixed(0) + '%';
        const bAlive = m.bacteria_alive !== undefined ? m.bacteria_alive : '?';
        const bTotal = m.bacteria_total !== undefined ? m.bacteria_total : '?';
        metricsHTML += '<br><span style="color:#ff6644">\\u2620</span> Bacteria: ' + bAlive + '/' + bTotal + ' alive';
        if (metricsOverlay) metricsOverlay.innerHTML = metricsHTML;

        // Step annotations
        let stepsHTML = '<b style="color:#aabbcc">Diapedesis Steps</b><br>';
        const bKilled = (m.bacteria_total || 0) - (m.bacteria_alive || 0);
        const steps = [
            ['1. Cytokine \\u2192 E-selectin', sc2.rolling > 0 || sc2.activating > 0 || sc2.arrested > 0],
            ['2. Selectin rolling', sc2.rolling > 0],
            ['3. Integrin activation', sc2.activating > 0 || sc2.arrested > 0],
            ['4. Firm adhesion', sc2.arrested > 0 || sc2.crawling > 0],
            ['5. Transmigration', sc2.transmigrating > 0],
            ['6. Tissue migration', sc2.migrated > 0],
            ['7. Phagocytosis (' + bKilled + ' killed)', bKilled > 0],
        ];
        steps.forEach(([label, active]) => {{
            const icon = active ? '\\u2705' : '\\u2B1C';
            stepsHTML += icon + ' ' + label + '<br>';
        }});
        if (stepsOverlay) stepsOverlay.innerHTML = stepsHTML;
    }}

    /* ═══════════ Overlays ═══════════ */
    const metricsOverlay = document.createElement('div');
    metricsOverlay.style.cssText = 'position:absolute;top:10px;left:10px;color:#ccc;font:11px monospace;background:rgba(4,4,18,0.85);padding:10px 14px;border-radius:8px;pointer-events:none;line-height:1.6;border:1px solid rgba(100,100,150,0.2);';
    container.appendChild(metricsOverlay);

    const stepsOverlay = document.createElement('div');
    stepsOverlay.style.cssText = 'position:absolute;top:10px;right:10px;color:#ccc;font:11px monospace;background:rgba(4,4,18,0.85);padding:10px 14px;border-radius:8px;pointer-events:none;line-height:1.8;border:1px solid rgba(100,100,150,0.2);';
    container.appendChild(stepsOverlay);

    // Legend overlay (bottom-left)
    const legendOverlay = document.createElement('div');
    legendOverlay.style.cssText = 'position:absolute;bottom:10px;left:10px;color:#999;font:10px monospace;background:rgba(4,4,18,0.85);padding:8px 12px;border-radius:6px;pointer-events:none;line-height:1.5;border:1px solid rgba(100,100,150,0.15);';
    legendOverlay.innerHTML = '<b style="color:#aab">Molecules</b><br>'
        + '<span style="color:#ffaa00">\\u25CF</span> Selectin (E-sel)<br>'
        + '<span style="color:#4488ff">\\u25CF</span> ICAM-1<br>'
        + '<span style="color:#33bb88">\\u25CF</span> PECAM-1 (junction)<br>'
        + '<span style="color:#00ccdd">\\u25CF</span> Integrin (LFA-1)<br>'
        + '<span style="color:#cc9933">\\u25CF</span> Complement (C3b)<br>'
        + '<b style="color:#aab">Tissue</b><br>'
        + '<span style="color:#336633">\\u25CF</span> Bacteria<br>'
        + '<span style="color:#dd5522">\\u25CF</span> Macrophage<br>'
        + '<span style="color:#eee8cc">\\u25CF</span> Fibrin<br>'
        + '<span style="color:#55bbff">\\u25CF</span> Chemokines';
    container.appendChild(legendOverlay);

    // Apply first frame
    applyFrame(0);

    /* ═══════════ Playback ═══════════ */
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

        // Animate: infection pulse
        const pulse = 0.06 + 0.04 * Math.sin(t * 0.003);
        infMat.opacity = pulse;

        // Animate: chemokine drift
        chemParticles.forEach((cm, i) => {{
            cm.position.y += Math.sin(t*0.001 + i*0.5) * 0.01;
            cm.position.x += Math.cos(t*0.0008 + i*0.3) * 0.005;
        }});

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
    # Render 3D viewer (Three.js or Omniverse RTX)
    if use_omniverse:
        _render_omniverse_viewer(frames, omni_host)
    else:
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

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Simulation time", f"{m['time']:.1f} s")
    col2.metric("Avg rolling velocity", f"{m['avg_rolling_velocity']:.0f} μm/s")
    col3.metric("Junction integrity", f"{m['avg_junction_integrity']*100:.0f}%")
    b_killed = m.get('bacteria_total', 0) - m.get('bacteria_alive', 0)
    col4.metric("Bacteria killed", f"{b_killed}/{m.get('bacteria_total', 0)}")
    col5.metric("Total frames", len(frames))

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
