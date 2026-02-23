"""
USD & Omniverse — Bio-Digital Twin Workbench
=============================================

Interactive workbench for composing biological scenes from the prototype library,
running physics simulations, and exporting to OpenUSD.

Tabs:
1. Schema Browser — Browse Bio-USD prim types and API schemas
2. Prototype Library — Browse 35 biological prototypes across 5 categories
3. Scene Composer — Place entities, configure counts/placement, live 3D preview
4. Physics Workbench — Configure physics, force fields, run simulation, animated viewer
5. Export — Download composed scenes as .usda
6. Documentation — Bio-USD specification reference
"""

import streamlit as st
import json
import math
import os
from dataclasses import asdict, fields, is_dataclass
from typing import get_type_hints

st.set_page_config(
    page_title="USD & Omniverse | Cognisom",
    page_icon="🎬",
    layout="wide",
)

# Auth gate
try:
    from cognisom.auth.middleware import streamlit_page_gate
    user = streamlit_page_gate(required_tier="researcher")
except Exception:
    user = None

st.title("🎬 Bio-Digital Twin Workbench")
st.markdown("Compose biological scenes, simulate physics, and export to OpenUSD")


# ═══════════════════════════════════════════════════════════════════════
# Session State Initialization
# ═══════════════════════════════════════════════════════════════════════

if "wb_entities" not in st.session_state:
    st.session_state.wb_entities = []  # [{proto_name, count, placement, size_scale}]
if "wb_scene_instances" not in st.session_state:
    st.session_state.wb_scene_instances = []  # [{id, proto_name, x, y, z, r, cr, cg, cb}]
if "wb_force_fields" not in st.session_state:
    st.session_state.wb_force_fields = []  # [{type, center_x/y/z, strength, radius, falloff}]
if "wb_sim_frames" not in st.session_state:
    st.session_state.wb_sim_frames = []  # [[{id, x, y, z}, ...], ...] per frame
if "wb_scene_size" not in st.session_state:
    st.session_state.wb_scene_size = 300

# Legacy keys from old USD generation tab
if "real_connector" not in st.session_state:
    st.session_state.real_connector = None
if "usd_stage_path" not in st.session_state:
    st.session_state.usd_stage_path = None
if "usd_cell_data" not in st.session_state:
    st.session_state.usd_cell_data = None


# ═══════════════════════════════════════════════════════════════════════
# Import Prototype Library (safe — no pxr dependency for data)
# ═══════════════════════════════════════════════════════════════════════

try:
    from cognisom.omniverse.prototype_library import (
        PrototypeSpec, PrototypeCategory,
        ALL_PROTOTYPES, MOLECULE_PROTOTYPES, ORGANELLE_PROTOTYPES,
        CELL_PROTOTYPES, PARTICLE_PROTOTYPES, STRUCTURE_PROTOTYPES,
    )
    PROTOTYPES_AVAILABLE = True
except ImportError:
    PROTOTYPES_AVAILABLE = False
    ALL_PROTOTYPES = {}

CATEGORY_DICTS = {
    "All": ALL_PROTOTYPES,
    "Molecules": MOLECULE_PROTOTYPES if PROTOTYPES_AVAILABLE else {},
    "Organelles": ORGANELLE_PROTOTYPES if PROTOTYPES_AVAILABLE else {},
    "Cells": CELL_PROTOTYPES if PROTOTYPES_AVAILABLE else {},
    "Particles": PARTICLE_PROTOTYPES if PROTOTYPES_AVAILABLE else {},
    "Structures": STRUCTURE_PROTOTYPES if PROTOTYPES_AVAILABLE else {},
}


# ═══════════════════════════════════════════════════════════════════════
# Helper: Text-based USDA generator
# ═══════════════════════════════════════════════════════════════════════

def _generate_usda_from_instances(instances, scene_size):
    """Generate valid USDA text from scene instances."""
    lines = [
        '#usda 1.0',
        '(',
        '    defaultPrim = "World"',
        '    metersPerUnit = 0.000001',
        '    upAxis = "Y"',
        '    doc = "Cognisom Bio-Digital Twin Scene"',
        ')',
        '',
        'def Xform "World"',
        '{',
        '    def Xform "Entities"',
        '    {',
    ]

    for inst in instances:
        eid = inst["id"]
        x, y, z, r = inst["x"], inst["y"], inst["z"], inst["r"]
        cr, cg, cb = inst["cr"], inst["cg"], inst["cb"]
        proto = inst["proto_name"]

        lines.append(f'        def Sphere "{eid}"')
        lines.append('        {')
        lines.append(f'            double radius = {r:.4f}')
        lines.append(f'            double3 xformOp:translate = ({x:.4f}, {y:.4f}, {z:.4f})')
        lines.append('            uniform token[] xformOpOrder = ["xformOp:translate"]')
        lines.append(f'            custom string cognisom:prototypeType = "{proto}"')
        lines.append('')
        lines.append('            def Material "Material"')
        lines.append('            {')
        lines.append('                def Shader "Shader"')
        lines.append('                {')
        lines.append('                    uniform token info:id = "UsdPreviewSurface"')
        lines.append(f'                    color3f inputs:diffuseColor = ({cr:.3f}, {cg:.3f}, {cb:.3f})')
        lines.append('                    float inputs:opacity = 0.9')
        lines.append('                    float inputs:roughness = 0.6')
        lines.append('                    token outputs:surface')
        lines.append('                }')
        lines.append(f'                token outputs:surface.connect = </World/Entities/{eid}/Material/Shader.outputs:surface>')
        lines.append('            }')
        lines.append('        }')
        lines.append('')

    lines.append('    }')
    lines.append('')
    lines.append('    def Xform "Lights"')
    lines.append('    {')
    lines.append('        def DomeLight "DomeLight"')
    lines.append('        {')
    lines.append('            float inputs:intensity = 500')
    lines.append('        }')
    lines.append('        def DistantLight "KeyLight"')
    lines.append('        {')
    lines.append('            float inputs:intensity = 1000')
    lines.append('            float3 xformOp:rotateXYZ = (-45, 30, 0)')
    lines.append('            uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]')
    lines.append('        }')
    lines.append('    }')
    lines.append('')
    lines.append('    def Camera "MainCamera"')
    lines.append('    {')
    lines.append('        float focalLength = 50')
    lines.append(f'        double3 xformOp:translate = ({scene_size/2:.1f}, {scene_size*0.7:.1f}, {scene_size*1.2:.1f})')
    lines.append('        float3 xformOp:rotateXYZ = (-25, 0, 0)')
    lines.append('        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]')
    lines.append('    }')
    lines.append('}')
    lines.append('')
    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Helper: Static Three.js Viewer
# ═══════════════════════════════════════════════════════════════════════

def _build_static_viewer(instances, scene_size):
    """Build a static Three.js 3D viewer for the scene."""
    cells_json = json.dumps(instances)
    return f'''
<div id="viewer3d" style="width:100%;height:550px;border-radius:8px;overflow:hidden;background:#0a0a1a;"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
(function() {{
    const container = document.getElementById('viewer3d');
    const W = container.clientWidth, H = container.clientHeight;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a1a);
    scene.fog = new THREE.FogExp2(0x0a0a1a, 0.0006);

    const camera = new THREE.PerspectiveCamera(50, W / H, 0.1, 10000);
    camera.position.set({scene_size*0.6}, {scene_size*0.5}, {scene_size*1.1});
    camera.lookAt({scene_size/2}, {scene_size*0.15}, {scene_size/2});

    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setSize(W, H);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.target.set({scene_size/2}, {scene_size*0.15}, {scene_size/2});
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.update();

    scene.add(new THREE.AmbientLight(0x404060, 0.6));
    const key = new THREE.DirectionalLight(0xfff5e8, 1.2);
    key.position.set({scene_size}, {scene_size*1.5}, {scene_size*0.8});
    scene.add(key);
    scene.add(new THREE.DirectionalLight(0x8888ff, 0.3));

    const grid = new THREE.GridHelper({scene_size}, 20, 0x333355, 0x222244);
    grid.position.set({scene_size/2}, -0.5, {scene_size/2});
    scene.add(grid);

    const cells = {cells_json};
    const typeGroups = {{}};
    cells.forEach(c => {{
        const k = c.proto_name || c.type || 'unknown';
        if (!typeGroups[k]) typeGroups[k] = [];
        typeGroups[k].push(c);
    }});

    const sphereGeo = new THREE.SphereGeometry(1, 24, 16);
    const dummy = new THREE.Object3D();
    const legendItems = [];

    Object.entries(typeGroups).forEach(([type, group]) => {{
        const c0 = group[0];
        const mat = new THREE.MeshStandardMaterial({{
            color: new THREE.Color(c0.cr, c0.cg, c0.cb),
            metalness: 0.15, roughness: 0.55,
            transparent: true, opacity: 0.88,
        }});
        const mesh = new THREE.InstancedMesh(sphereGeo, mat, group.length);
        mesh.castShadow = true;
        group.forEach((c, i) => {{
            dummy.position.set(c.x, c.y, c.z);
            dummy.scale.set(c.r, c.r, c.r);
            dummy.updateMatrix();
            mesh.setMatrixAt(i, dummy.matrix);
        }});
        mesh.instanceMatrix.needsUpdate = true;
        scene.add(mesh);
        const hex = '#' + new THREE.Color(c0.cr, c0.cg, c0.cb).getHexString();
        legendItems.push('<span style="color:'+hex+'">\\u25CF</span> ' + type + ' ('+group.length+')');
    }});

    container.style.position = 'relative';
    const legend = document.createElement('div');
    legend.style.cssText = 'position:absolute;top:10px;left:10px;color:#ccc;font:12px monospace;background:rgba(0,0,0,0.6);padding:8px 12px;border-radius:6px;pointer-events:none;';
    legend.innerHTML = '<b>Scene Preview</b><br>' + legendItems.join(' &nbsp; ') + '<br><span style="color:#666">Drag to orbit \\u2022 Scroll to zoom</span>';
    container.appendChild(legend);

    const stats = document.createElement('div');
    stats.style.cssText = 'position:absolute;top:10px;right:10px;color:#888;font:11px monospace;background:rgba(0,0,0,0.5);padding:6px 10px;border-radius:6px;pointer-events:none;';
    stats.textContent = cells.length + ' entities | ' + {scene_size} + ' \\u03BCm';
    container.appendChild(stats);

    function animate() {{ requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }}
    animate();

    new ResizeObserver(() => {{
        const w = container.clientWidth, h = container.clientHeight;
        camera.aspect = w / h; camera.updateProjectionMatrix(); renderer.setSize(w, h);
    }}).observe(container);
}})();
</script>
'''


# ═══════════════════════════════════════════════════════════════════════
# Helper: Animated Three.js Viewer (for physics playback)
# ═══════════════════════════════════════════════════════════════════════

def _build_animated_viewer(instances, frames, scene_size):
    """Build a Three.js viewer with frame-by-frame animation playback."""
    instances_json = json.dumps(instances)
    frames_json = json.dumps(frames)
    return f'''
<div id="animViewer" style="width:100%;height:550px;border-radius:8px;overflow:hidden;background:#0a0a1a;position:relative;"></div>
<div id="animControls" style="width:100%;padding:8px 0;display:flex;align-items:center;gap:10px;font:13px monospace;color:#ccc;">
    <button id="playBtn" style="padding:4px 12px;cursor:pointer;background:#2a2a4a;color:#ccc;border:1px solid #555;border-radius:4px;">Play</button>
    <input id="frameSlider" type="range" min="0" max="{len(frames)-1}" value="0" style="flex:1;">
    <span id="frameLabel">0 / {len(frames)-1}</span>
    <select id="speedSelect" style="background:#2a2a4a;color:#ccc;border:1px solid #555;border-radius:4px;padding:2px 6px;">
        <option value="0.5">0.5x</option>
        <option value="1" selected>1x</option>
        <option value="2">2x</option>
        <option value="4">4x</option>
    </select>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
(function() {{
    const container = document.getElementById('animViewer');
    const W = container.clientWidth, H = container.clientHeight;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a1a);
    scene.fog = new THREE.FogExp2(0x0a0a1a, 0.0006);

    const camera = new THREE.PerspectiveCamera(50, W / H, 0.1, 10000);
    camera.position.set({scene_size*0.6}, {scene_size*0.5}, {scene_size*1.1});
    camera.lookAt({scene_size/2}, {scene_size*0.15}, {scene_size/2});

    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setSize(W, H);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.target.set({scene_size/2}, {scene_size*0.15}, {scene_size/2});
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.update();

    scene.add(new THREE.AmbientLight(0x404060, 0.6));
    const keyLight = new THREE.DirectionalLight(0xfff5e8, 1.2);
    keyLight.position.set({scene_size}, {scene_size*1.5}, {scene_size*0.8});
    scene.add(keyLight);
    scene.add(new THREE.DirectionalLight(0x8888ff, 0.3));

    const grid = new THREE.GridHelper({scene_size}, 20, 0x333355, 0x222244);
    grid.position.set({scene_size/2}, -0.5, {scene_size/2});
    scene.add(grid);

    const instances = {instances_json};
    const frames = {frames_json};

    // Group by prototype type
    const typeGroups = {{}};
    instances.forEach((c, idx) => {{
        const k = c.proto_name || 'unknown';
        if (!typeGroups[k]) typeGroups[k] = {{ indices: [], color: [c.cr, c.cg, c.cb] }};
        typeGroups[k].indices.push(idx);
    }});

    // Build id->index map for quick frame lookup
    const idToIdx = {{}};
    instances.forEach((c, i) => {{ idToIdx[c.id] = i; }});

    const sphereGeo = new THREE.SphereGeometry(1, 20, 14);
    const dummy = new THREE.Object3D();
    const meshes = {{}};  // type -> {{ mesh, indices }}
    const legendItems = [];

    Object.entries(typeGroups).forEach(([type, data]) => {{
        const mat = new THREE.MeshStandardMaterial({{
            color: new THREE.Color(data.color[0], data.color[1], data.color[2]),
            metalness: 0.15, roughness: 0.55, transparent: true, opacity: 0.88,
        }});
        const mesh = new THREE.InstancedMesh(sphereGeo, mat, data.indices.length);
        data.indices.forEach((gi, li) => {{
            const c = instances[gi];
            dummy.position.set(c.x, c.y, c.z);
            dummy.scale.set(c.r, c.r, c.r);
            dummy.updateMatrix();
            mesh.setMatrixAt(li, dummy.matrix);
        }});
        mesh.instanceMatrix.needsUpdate = true;
        scene.add(mesh);
        meshes[type] = {{ mesh, indices: data.indices }};
        const hex = '#' + new THREE.Color(data.color[0], data.color[1], data.color[2]).getHexString();
        legendItems.push('<span style="color:'+hex+'">\\u25CF</span> ' + type + ' ('+data.indices.length+')');
    }});

    // Legend
    const legend = document.createElement('div');
    legend.style.cssText = 'position:absolute;top:10px;left:10px;color:#ccc;font:12px monospace;background:rgba(0,0,0,0.6);padding:8px 12px;border-radius:6px;pointer-events:none;';
    legend.innerHTML = '<b>Physics Simulation</b><br>' + legendItems.join(' &nbsp; ');
    container.appendChild(legend);

    const frameInfo = document.createElement('div');
    frameInfo.style.cssText = 'position:absolute;top:10px;right:10px;color:#888;font:11px monospace;background:rgba(0,0,0,0.5);padding:6px 10px;border-radius:6px;pointer-events:none;';
    container.appendChild(frameInfo);

    // Animation state
    let currentFrame = 0;
    let playing = false;
    let speed = 1;
    let lastFrameTime = 0;

    const playBtn = document.getElementById('playBtn');
    const slider = document.getElementById('frameSlider');
    const label = document.getElementById('frameLabel');
    const speedSel = document.getElementById('speedSelect');

    playBtn.addEventListener('click', () => {{
        playing = !playing;
        playBtn.textContent = playing ? 'Pause' : 'Play';
    }});
    slider.addEventListener('input', () => {{
        currentFrame = parseInt(slider.value);
        applyFrame(currentFrame);
    }});
    speedSel.addEventListener('change', () => {{ speed = parseFloat(speedSel.value); }});

    function applyFrame(fi) {{
        if (fi < 0 || fi >= frames.length) return;
        const frame = frames[fi];

        // Build position map for this frame
        const posMap = {{}};
        frame.forEach(p => {{ posMap[p.id] = p; }});

        Object.entries(meshes).forEach(([type, data]) => {{
            data.indices.forEach((gi, li) => {{
                const c = instances[gi];
                const pos = posMap[c.id];
                if (pos) {{
                    dummy.position.set(pos.x, pos.y, pos.z);
                }} else {{
                    dummy.position.set(c.x, c.y, c.z);
                }}
                dummy.scale.set(c.r, c.r, c.r);
                dummy.updateMatrix();
                data.mesh.setMatrixAt(li, dummy.matrix);
            }});
            data.mesh.instanceMatrix.needsUpdate = true;
        }});

        slider.value = fi;
        label.textContent = fi + ' / ' + (frames.length - 1);
        frameInfo.textContent = 'Frame ' + fi + ' / ' + (frames.length - 1) + ' | ' + instances.length + ' entities';
    }}

    applyFrame(0);

    const targetInterval = 1000 / 30;  // 30 FPS base

    function animate(time) {{
        requestAnimationFrame(animate);
        controls.update();

        if (playing && frames.length > 1) {{
            const elapsed = time - lastFrameTime;
            if (elapsed >= targetInterval / speed) {{
                currentFrame = (currentFrame + 1) % frames.length;
                applyFrame(currentFrame);
                lastFrameTime = time;
            }}
        }}

        renderer.render(scene, camera);
    }}
    animate(0);

    new ResizeObserver(() => {{
        const w = container.clientWidth, h = container.clientHeight;
        camera.aspect = w / h; camera.updateProjectionMatrix(); renderer.setSize(w, h);
    }}).observe(container);
}})();
</script>
'''


# ═══════════════════════════════════════════════════════════════════════
# Helper: Placement Generators
# ═══════════════════════════════════════════════════════════════════════

def _generate_placements(count, placement, scene_size, offset_y=0.0):
    """Generate (x, y, z) positions for entities."""
    import numpy as np
    positions = []

    if placement == "random":
        xs = np.random.uniform(scene_size * 0.1, scene_size * 0.9, count)
        ys = np.random.uniform(scene_size * 0.05, scene_size * 0.4, count) + offset_y
        zs = np.random.uniform(scene_size * 0.1, scene_size * 0.9, count)
        for i in range(count):
            positions.append((float(xs[i]), float(ys[i]), float(zs[i])))

    elif placement == "grid":
        side = max(1, int(math.ceil(count ** (1/3))))
        spacing = scene_size * 0.8 / max(side, 1)
        start = scene_size * 0.1
        placed = 0
        for ix in range(side):
            for iy in range(side):
                for iz in range(side):
                    if placed >= count:
                        break
                    positions.append((
                        start + ix * spacing,
                        start * 0.5 + iy * spacing + offset_y,
                        start + iz * spacing,
                    ))
                    placed += 1

    elif placement == "ring":
        cx, cz = scene_size / 2, scene_size / 2
        radius = scene_size * 0.35
        for i in range(count):
            theta = 2 * math.pi * i / max(count, 1)
            positions.append((
                cx + radius * math.cos(theta),
                scene_size * 0.15 + offset_y,
                cz + radius * math.sin(theta),
            ))

    return positions


# ═══════════════════════════════════════════════════════════════════════
# Tab Layout
# ═══════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📐 Schema Browser",
    "🧬 Prototype Library",
    "🎨 Scene Composer",
    "⚛️ Physics Workbench",
    "📤 Export",
    "📖 Documentation",
])


# ─────────────────────────────────────────────────────────────────────────
# Tab 1: Schema Browser (unchanged)
# ─────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Bio-USD Schema Browser")
    st.markdown("Browse registered prim types and API schemas from the Bio-USD specification.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Prim Types")
        try:
            from cognisom.biousd.schema import (
                prim_registry, api_registry, list_prim_types, list_api_schemas,
                get_prim_class, get_api_class,
                CellType, CellPhase, ImmuneCellType, SpatialFieldType, GeneType
            )

            prim_types = list_prim_types()
            selected_prim = st.selectbox(
                "Select Prim Type",
                prim_types,
                format_func=lambda x: x.replace("bio_", "Bio").replace("_", " ").title()
            )

            st.divider()

            st.markdown("### API Schemas")
            api_schemas = list_api_schemas()
            selected_api = st.selectbox(
                "Select API Schema",
                api_schemas,
                format_func=lambda x: x.replace("bio_", "Bio").replace("_api", "").replace("_", " ").title()
            )

            st.divider()

            st.markdown("### Enums")
            enum_options = {
                "CellType": CellType,
                "CellPhase": CellPhase,
                "ImmuneCellType": ImmuneCellType,
                "SpatialFieldType": SpatialFieldType,
                "GeneType": GeneType,
            }
            selected_enum = st.selectbox("Select Enum", list(enum_options.keys()))

        except ImportError as e:
            st.error(f"Bio-USD module not available: {e}")
            prim_types = []
            selected_prim = None
            selected_api = None
            selected_enum = None

    with col2:
        if selected_prim:
            st.markdown(f"### `{selected_prim}`")

            try:
                prim_class = get_prim_class(selected_prim)

                if prim_class.__doc__:
                    st.markdown(f"**Description:**")
                    st.markdown(prim_class.__doc__.strip())

                st.markdown("**Fields:**")

                if is_dataclass(prim_class):
                    field_data = []
                    for f in fields(prim_class):
                        field_type = str(f.type).replace("typing.", "").replace("<class '", "").replace("'>", "")
                        default_val = f.default if f.default is not f.default_factory else "(factory)"
                        if f.default is f.default_factory and f.default_factory is not None:
                            try:
                                default_val = str(f.default_factory())
                            except Exception:
                                default_val = "(factory)"
                        field_data.append({
                            "Field": f.name,
                            "Type": field_type,
                            "Default": str(default_val)[:50],
                        })

                    st.table(field_data)

                bases = [b.__name__ for b in prim_class.__bases__ if b.__name__ != "object"]
                if bases:
                    st.markdown(f"**Inherits from:** `{', '.join(bases)}`")

                st.markdown("**Example:**")
                st.code(f"""from cognisom.biousd.schema import create_prim

prim = create_prim("{selected_prim}",
    prim_path="/World/Cells/cell_001",
    display_name="My Cell"
)""", language="python")

            except Exception as e:
                st.error(f"Error loading prim class: {e}")

        if selected_api:
            st.divider()
            st.markdown(f"### `{selected_api}`")

            try:
                api_class = get_api_class(selected_api)

                if api_class.__doc__:
                    st.markdown(f"**Description:**")
                    st.markdown(api_class.__doc__.strip())

                if is_dataclass(api_class):
                    st.markdown("**Fields:**")
                    field_data = []
                    for f in fields(api_class):
                        field_type = str(f.type).replace("typing.", "")
                        default_val = f.default if f.default is not f.default_factory else "(factory)"
                        field_data.append({
                            "Field": f.name,
                            "Type": field_type,
                            "Default": str(default_val)[:30],
                        })
                    st.table(field_data)

            except Exception as e:
                st.error(f"Error loading API schema: {e}")

        if selected_enum:
            st.divider()
            st.markdown(f"### `{selected_enum}`")

            enum_class = enum_options.get(selected_enum)
            if enum_class:
                values = [{"Name": e.name, "Value": e.value} for e in enum_class]
                st.table(values)


# ─────────────────────────────────────────────────────────────────────────
# Tab 2: Prototype Library
# ─────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Biological Prototype Library")
    st.markdown("Browse 35 biological prototypes across 5 categories. Add entities to your scene.")

    if not PROTOTYPES_AVAILABLE:
        st.error("Prototype library not available. Check cognisom.omniverse.prototype_library.")
    else:
        col_filter, col_detail = st.columns([1, 2])

        with col_filter:
            category = st.radio(
                "Category",
                list(CATEGORY_DICTS.keys()),
                horizontal=False,
            )
            protos = CATEGORY_DICTS[category]

            st.markdown(f"**{len(protos)} prototypes**")
            st.divider()

            # List prototypes
            proto_names = list(protos.keys())
            selected_proto_name = st.selectbox(
                "Select Prototype",
                proto_names,
                format_func=lambda x: x.replace("_", " ").title(),
            )

        with col_detail:
            if selected_proto_name and selected_proto_name in ALL_PROTOTYPES:
                spec = ALL_PROTOTYPES[selected_proto_name]

                # Header with color swatch
                r_hex = f"#{int(spec.color[0]*255):02x}{int(spec.color[1]*255):02x}{int(spec.color[2]*255):02x}"
                st.markdown(f"### {spec.name.replace('_', ' ').title()} <span style='display:inline-block;width:16px;height:16px;background:{r_hex};border-radius:50%;vertical-align:middle;margin-left:8px;'></span>", unsafe_allow_html=True)

                st.markdown(f"**{spec.description}**")

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Category", spec.category.value.title())
                with col_b:
                    if spec.default_size >= 1:
                        st.metric("Size", f"{spec.default_size:.1f} um")
                    else:
                        st.metric("Size", f"{spec.default_size * 1000:.1f} nm")
                with col_c:
                    st.metric("Geometry", spec.geometry_type.title())

                col_d, col_e = st.columns(2)
                with col_d:
                    st.metric("Opacity", f"{spec.opacity:.1f}")
                with col_e:
                    st.metric("Max Instances", f"{spec.max_instances:,.0f}")

                if spec.geometry_params:
                    st.markdown("**Geometry Parameters:**")
                    st.json(spec.geometry_params)

                st.divider()

                # Add to scene controls
                st.markdown("### Add to Scene")
                add_col1, add_col2 = st.columns(2)
                with add_col1:
                    add_count = st.number_input("Count", min_value=1, max_value=500, value=10, key=f"add_count_{selected_proto_name}")
                with add_col2:
                    add_placement = st.selectbox("Placement", ["random", "grid", "ring"], key=f"add_place_{selected_proto_name}")

                add_scale = st.slider("Size Scale", 0.1, 5.0, 1.0, 0.1, key=f"add_scale_{selected_proto_name}")

                if st.button(f"Add {spec.name.replace('_', ' ').title()} to Scene", type="primary", use_container_width=True):
                    st.session_state.wb_entities.append({
                        "proto_name": spec.name,
                        "count": add_count,
                        "placement": add_placement,
                        "size_scale": add_scale,
                    })
                    st.success(f"Added {add_count}x {spec.name.replace('_', ' ').title()} to scene")
                    st.rerun()

        # Show current scene composition at bottom
        if st.session_state.wb_entities:
            st.divider()
            st.markdown(f"### Scene Composition ({len(st.session_state.wb_entities)} entity types)")
            for i, ent in enumerate(st.session_state.wb_entities):
                spec = ALL_PROTOTYPES.get(ent["proto_name"])
                name = ent["proto_name"].replace("_", " ").title()
                st.markdown(f"- **{name}** x{ent['count']} ({ent['placement']}, scale {ent['size_scale']:.1f}x)")
            st.caption("Go to **Scene Composer** tab to generate the 3D scene.")


# ─────────────────────────────────────────────────────────────────────────
# Tab 3: Scene Composer
# ─────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Scene Composer")
    st.markdown("Configure entity placement and generate a 3D scene.")

    if not st.session_state.wb_entities:
        st.info("No entities added yet. Go to **Prototype Library** tab to add biological entities.")
    else:
        col_cfg, col_preview = st.columns([1, 2])

        with col_cfg:
            st.markdown("### Entity List")

            # Editable entity list
            entities_to_remove = []
            for i, ent in enumerate(st.session_state.wb_entities):
                spec = ALL_PROTOTYPES.get(ent["proto_name"])
                if not spec:
                    continue
                name = ent["proto_name"].replace("_", " ").title()
                r_hex = f"#{int(spec.color[0]*255):02x}{int(spec.color[1]*255):02x}{int(spec.color[2]*255):02x}"

                with st.expander(f"{name} x{ent['count']}", expanded=False):
                    new_count = st.number_input("Count", 1, 500, ent["count"], key=f"sc_count_{i}")
                    new_placement = st.selectbox("Placement", ["random", "grid", "ring"], index=["random", "grid", "ring"].index(ent["placement"]), key=f"sc_place_{i}")
                    new_scale = st.slider("Size Scale", 0.1, 5.0, ent["size_scale"], 0.1, key=f"sc_scale_{i}")

                    st.session_state.wb_entities[i]["count"] = new_count
                    st.session_state.wb_entities[i]["placement"] = new_placement
                    st.session_state.wb_entities[i]["size_scale"] = new_scale

                    if st.button("Remove", key=f"sc_remove_{i}", type="secondary"):
                        entities_to_remove.append(i)

            # Process removals
            if entities_to_remove:
                for idx in sorted(entities_to_remove, reverse=True):
                    st.session_state.wb_entities.pop(idx)
                st.rerun()

            st.divider()
            st.markdown("### Scene Settings")
            scene_size = st.slider("Scene Size (um)", 100, 1000, st.session_state.wb_scene_size, key="sc_scene_size")
            st.session_state.wb_scene_size = scene_size

            total_entities = sum(e["count"] for e in st.session_state.wb_entities)
            st.metric("Total Entities", total_entities)

            if st.button("Generate Scene", type="primary", use_container_width=True):
                import numpy as np
                np.random.seed(42)

                all_instances = []
                for ent in st.session_state.wb_entities:
                    spec = ALL_PROTOTYPES.get(ent["proto_name"])
                    if not spec:
                        continue

                    positions = _generate_placements(ent["count"], ent["placement"], scene_size)
                    radius = spec.default_size * ent["size_scale"]
                    # Clamp radius to be visible at scene scale
                    vis_radius = max(radius, scene_size * 0.005)
                    vis_radius = min(vis_radius, scene_size * 0.05)

                    for j, (px, py, pz) in enumerate(positions):
                        all_instances.append({
                            "id": f"{spec.name}_{j:04d}",
                            "proto_name": spec.name,
                            "x": px, "y": py, "z": pz,
                            "r": vis_radius,
                            "cr": spec.color[0],
                            "cg": spec.color[1],
                            "cb": spec.color[2],
                        })

                st.session_state.wb_scene_instances = all_instances
                st.session_state.wb_sim_frames = []  # Clear old sim
                st.success(f"Generated {len(all_instances)} entity instances")
                st.rerun()

        with col_preview:
            if st.session_state.wb_scene_instances:
                st.markdown("### 3D Scene Preview")
                import streamlit.components.v1 as components
                viewer_html = _build_static_viewer(
                    st.session_state.wb_scene_instances,
                    st.session_state.wb_scene_size,
                )
                components.html(viewer_html, height=580)
            else:
                st.info("Click **Generate Scene** to see the 3D preview.")
                st.markdown("""
                **Workflow:**
                1. Add entities from the **Prototype Library** tab
                2. Adjust counts and placement modes here
                3. Click **Generate Scene** to create the 3D layout
                4. Go to **Physics Workbench** to simulate interactions
                """)


# ─────────────────────────────────────────────────────────────────────────
# Tab 4: Physics Workbench
# ─────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Physics Workbench")
    st.markdown("Configure physics parameters, add force fields, and run a simulation.")

    if not st.session_state.wb_scene_instances:
        st.info("No scene generated yet. Go to **Scene Composer** tab to generate a scene first.")
    else:
        col_phys, col_viewer = st.columns([1, 2])

        with col_phys:
            st.markdown("### Physics Config")
            phys_dt = st.slider("Time Step", 0.005, 0.1, 0.02, 0.005, key="phys_dt")
            phys_substeps = st.slider("Substeps", 1, 8, 4, key="phys_substeps")
            phys_damping = st.slider("Damping", 0.0, 1.0, 0.3, 0.05, key="phys_damping")
            phys_restitution = st.slider("Restitution (bounciness)", 0.0, 1.0, 0.3, 0.05, key="phys_restitution")
            phys_collisions = st.checkbox("Enable Collisions", value=True, key="phys_collisions")

            st.divider()
            st.markdown("### Simulation")
            sim_duration = st.slider("Duration (seconds)", 1.0, 20.0, 5.0, 0.5, key="sim_duration")
            sim_fps = st.slider("Frames per second", 10, 60, 30, key="sim_fps")
            n_frames = int(sim_duration * sim_fps)
            st.caption(f"{n_frames} frames total")

            st.divider()
            st.markdown("### Force Fields")

            # Add force field
            with st.expander("Add Force Field", expanded=len(st.session_state.wb_force_fields) == 0):
                ff_type = st.selectbox("Type", ["attraction", "repulsion", "vortex", "flow"], key="ff_type")
                scene_size = st.session_state.wb_scene_size
                ff_cx = st.number_input("Center X", 0.0, float(scene_size), float(scene_size / 2), key="ff_cx")
                ff_cy = st.number_input("Center Y", 0.0, float(scene_size), float(scene_size * 0.15), key="ff_cy")
                ff_cz = st.number_input("Center Z", 0.0, float(scene_size), float(scene_size / 2), key="ff_cz")
                ff_strength = st.slider("Strength", 1.0, 500.0, 50.0, 5.0, key="ff_strength")
                ff_radius = st.slider("Radius", 10.0, float(scene_size), float(scene_size * 0.5), 10.0, key="ff_radius")
                ff_falloff = st.selectbox("Falloff", ["linear", "quadratic", "constant"], key="ff_falloff")

                if st.button("Add Force Field", type="secondary"):
                    st.session_state.wb_force_fields.append({
                        "type": ff_type,
                        "center_x": ff_cx, "center_y": ff_cy, "center_z": ff_cz,
                        "strength": ff_strength, "radius": ff_radius, "falloff": ff_falloff,
                    })
                    st.rerun()

            # List existing force fields
            if st.session_state.wb_force_fields:
                ff_to_remove = []
                for i, ff in enumerate(st.session_state.wb_force_fields):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(f"**{ff['type'].title()}** @ ({ff['center_x']:.0f}, {ff['center_y']:.0f}, {ff['center_z']:.0f}) str={ff['strength']:.0f}")
                    with c2:
                        if st.button("X", key=f"ff_rm_{i}"):
                            ff_to_remove.append(i)
                if ff_to_remove:
                    for idx in sorted(ff_to_remove, reverse=True):
                        st.session_state.wb_force_fields.pop(idx)
                    st.rerun()

            st.divider()

            if st.button("Run Simulation", type="primary", use_container_width=True):
                try:
                    from cognisom.omniverse.physics_bridge import (
                        PhysicsBridge, PhysicsConfig, PhysicsMode,
                    )

                    config = PhysicsConfig(
                        mode=PhysicsMode.BASIC,
                        time_step=phys_dt,
                        substeps=phys_substeps,
                        cell_damping=phys_damping,
                        collision_enabled=phys_collisions,
                        restitution=phys_restitution,
                    )

                    bridge = PhysicsBridge(config=config)
                    bridge.initialize()

                    # Add bodies
                    for inst in st.session_state.wb_scene_instances:
                        mass = max(inst["r"] ** 3 * 0.001, 0.1)
                        bridge.add_body(
                            entity_id=inst["id"],
                            position=(inst["x"], inst["y"], inst["z"]),
                            radius=inst["r"],
                            mass=mass,
                        )

                    # Add force fields
                    for ff in st.session_state.wb_force_fields:
                        bridge.add_force_field(
                            field_type=ff["type"],
                            center=(ff["center_x"], ff["center_y"], ff["center_z"]),
                            strength=ff["strength"],
                            radius=ff["radius"],
                            falloff=ff["falloff"],
                        )

                    # Run simulation
                    sim_frames = []
                    progress = st.progress(0, text="Simulating...")
                    for fi in range(n_frames):
                        bridge.step(phys_dt)
                        frame = []
                        for body in bridge._bodies.values():
                            frame.append({
                                "id": body.entity_id,
                                "x": round(body.position[0], 4),
                                "y": round(body.position[1], 4),
                                "z": round(body.position[2], 4),
                            })
                        sim_frames.append(frame)
                        if fi % 10 == 0:
                            progress.progress(fi / n_frames, text=f"Frame {fi}/{n_frames}")

                    progress.progress(1.0, text="Simulation complete!")
                    st.session_state.wb_sim_frames = sim_frames

                    stats = bridge.get_stats()
                    st.success(f"Simulated {n_frames} frames, {stats['collision_count']} collisions")
                    st.rerun()

                except Exception as e:
                    st.error(f"Simulation error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        with col_viewer:
            if st.session_state.wb_sim_frames:
                st.markdown("### Simulation Playback")
                import streamlit.components.v1 as components
                viewer_html = _build_animated_viewer(
                    st.session_state.wb_scene_instances,
                    st.session_state.wb_sim_frames,
                    st.session_state.wb_scene_size,
                )
                components.html(viewer_html, height=620)
            elif st.session_state.wb_scene_instances:
                st.markdown("### Static Scene Preview")
                import streamlit.components.v1 as components
                viewer_html = _build_static_viewer(
                    st.session_state.wb_scene_instances,
                    st.session_state.wb_scene_size,
                )
                components.html(viewer_html, height=580)
                st.caption("Configure physics and click **Run Simulation** to see animated playback.")
            else:
                st.info("Generate a scene in the **Scene Composer** tab first.")


# ─────────────────────────────────────────────────────────────────────────
# Tab 5: Export
# ─────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Export to USD")

    col_exp1, col_exp2 = st.columns([1, 1])

    with col_exp1:
        st.markdown("### Export Composed Scene")

        if st.session_state.wb_scene_instances:
            scene_size = st.session_state.wb_scene_size
            n_inst = len(st.session_state.wb_scene_instances)
            st.metric("Entities in Scene", n_inst)

            export_name = st.text_input("Scene Name", value="cognisom_scene", key="exp_name")
            export_dir = st.text_input("Output Directory", value="exports/usd", key="exp_dir")

            if st.button("Export as USDA", type="primary", use_container_width=True):
                try:
                    os.makedirs(export_dir, exist_ok=True)
                    usda_path = os.path.join(export_dir, f"{export_name}.usda")
                    usda = _generate_usda_from_instances(
                        st.session_state.wb_scene_instances,
                        scene_size,
                    )
                    with open(usda_path, "w") as f:
                        f.write(usda)

                    st.session_state.usd_stage_path = usda_path
                    st.success(f"Exported {n_inst} entities to `{usda_path}`")
                except Exception as e:
                    st.error(f"Export error: {e}")
        else:
            st.info("No scene to export. Compose a scene first in the **Scene Composer** tab.")

        st.divider()

        # Show download if file exists
        stage_path = st.session_state.usd_stage_path
        if stage_path and os.path.exists(str(stage_path)):
            with open(str(stage_path), 'r') as f:
                usd_content = f.read()

            st.metric("File Size", f"{os.path.getsize(str(stage_path)):,} bytes")

            st.download_button(
                label="Download .usda",
                data=usd_content,
                file_name=os.path.basename(str(stage_path)),
                mime="text/plain",
            )

            with st.expander("View USD Source"):
                preview_lines = usd_content.split('\n')[:60]
                st.code('\n'.join(preview_lines), language="python")
                remaining = len(usd_content.split('\n')) - 60
                if remaining > 0:
                    st.caption(f"... and {remaining} more lines")

    with col_exp2:
        st.markdown("### USD Scene Hierarchy")
        if st.session_state.wb_scene_instances:
            # Build dynamic hierarchy from actual instances
            proto_counts = {}
            for inst in st.session_state.wb_scene_instances:
                pn = inst["proto_name"]
                proto_counts[pn] = proto_counts.get(pn, 0) + 1

            hier_lines = ["/World"]
            hier_lines.append("  /Entities")
            for pn, cnt in proto_counts.items():
                hier_lines.append(f"    /{pn}_* ({cnt} instances)")
            hier_lines.append("  /Lights")
            hier_lines.append("    /DomeLight")
            hier_lines.append("    /KeyLight")
            hier_lines.append("  /MainCamera")

            st.code('\n'.join(hier_lines))
        else:
            st.markdown("""
            **Scene Hierarchy:**
            ```
            /World
            ├── /Entities
            │   ├── entity_0001 (PrototypeSpec)
            │   └── ...
            ├── /Lights
            │   ├── DomeLight
            │   └── KeyLight
            └── /MainCamera
            ```
            """)

        st.divider()
        st.markdown("### Viewing USDA Files")
        st.markdown("""
        Open exported `.usda` files with:
        - **NVIDIA Omniverse** — Full rendering pipeline
        - **Blender** (3.5+) — Import via USD add-on
        - **usdview** — Official Pixar viewer (`pip install usd-core`)
        - **Apple Quick Look** — Convert with `usdzconvert`
        """)


# ─────────────────────────────────────────────────────────────────────────
# Tab 6: Documentation
# ─────────────────────────────────────────────────────────────────────────
with tab6:
    st.subheader("Documentation")

    st.markdown("""
    ## Bio-USD Specification

    Bio-USD extends OpenUSD with schemas for biological simulation.

    ### Prim Type Hierarchy

    ```
    BioUnit (abstract base)
    ├── BioCell              — Single biological cell
    │   └── BioImmuneCell    — T cells, NK cells, macrophages
    ├── BioGene              — Gene with expression state
    ├── BioProtein           — Protein with 3D structure
    ├── BioMolecule          — Small molecule (drug, metabolite)
    ├── BioTissue            — Cell collection / tissue
    ├── BioCapillary         — Blood vessel segment
    ├── BioSpatialField      — 3D concentration field
    └── BioExosome           — Extracellular vesicle
    ```

    ### Applied API Schemas

    Composable metadata that can be applied to any prim:

    | Schema | Purpose |
    |--------|---------|
    | `BioMetabolicAPI` | O2, glucose, ATP, lactate |
    | `BioGeneExpressionAPI` | Expression levels, mutations |
    | `BioEpigeneticAPI` | Methylation, histone marks |
    | `BioImmuneAPI` | MHC-I, activation state |
    | `BioInteractionAPI` | Binding relationships |

    ### Extensibility

    Register custom prim types at runtime:

    ```python
    from cognisom.biousd.schema import register_prim, BioUnit

    @register_prim("bio_virus", version="1.0.0")
    class BioVirusParticle(BioUnit):
        virus_type: str = ""
        capsid_proteins: List[str] = field(default_factory=list)
        genome_rna: str = ""
    ```

    ## Prototype Library

    The prototype library provides 35 pre-defined biological entities:

    | Category | Count | Examples |
    |----------|-------|---------|
    | Molecules | 5 | ATP, glucose, oxygen, CO2, lipid |
    | Organelles | 7 | Mitochondrion, nucleus, ribosome, Golgi, lysosome, vesicle, ER |
    | Cells | 9 | Generic cell, RBC, WBC, platelet, neuron, hepatocyte, tumor, immune, bacteria |
    | Particles | 5 | Water, Na+, K+, Ca2+, drug particle |
    | Structures | 4 | Microtubule, actin, collagen, blood vessel |

    ## Physics Bridge

    The Physics Workbench uses `PhysicsBridge` from `cognisom.omniverse.physics_bridge`:

    - **Collision detection** — O(N^2) sphere-sphere with impulse resolution
    - **Force fields** — Attraction, repulsion, vortex, flow with linear/quadratic/constant falloff
    - **Integration** — Euler with substeps, velocity damping, and max velocity clamping
    - **Collision types** — Bounce, slide, absorb (cell-drug), fuse (cell division)

    ## Omniverse Integration

    ### Connection Architecture

    ```
    Cognisom Engine
         ↓
    Bio-USD Converter
         ↓
    Omniverse Connector
         ↓
    Nucleus Server (omni://localhost)
         ↓
    Omniverse Kit / Isaac Sim
    ```

    ### Requirements

    - NVIDIA Omniverse Kit 2024.x+
    - Nucleus server (local or remote)
    - `omni.client` Python package
    - RTX GPU for real-time rendering
    """)

# Footer
from cognisom.dashboard.footer import render_footer
render_footer()
