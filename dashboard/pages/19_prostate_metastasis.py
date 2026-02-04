"""
Page 19: Prostate Cancer — Full Multi-Scale Metastasis Journey
===============================================================

From organ anatomy to molecular mechanisms:
- Organ level: Prostate zones, vasculature, lymphatics, nerves
- Tissue level: Glandular structure, tumor microenvironment
- Cellular level: Cancer cells, immune evasion, invasion
- Metastasis: Local invasion → Intravasation → CTCs → cfDNA → Bone tropism

For Harvard immunology demo: Educational visualization of cancer progression.
"""

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import json

st.set_page_config(page_title="Prostate Metastasis", page_icon="🔬", layout="wide")

from cognisom.auth.middleware import streamlit_page_gate
user = streamlit_page_gate("19_prostate_metastasis")

st.title("🔬 Prostate Cancer: Multi-Scale Metastasis Journey")
st.caption("From organ anatomy to molecular mechanisms — interactive educational visualization")

# ─────────────────────────────────────────────────────────────────────────
# BIOLOGICAL DATA MODELS
# ─────────────────────────────────────────────────────────────────────────

PROSTATE_ANATOMY = {
    "zones": {
        "peripheral": {
            "volume_pct": 70,
            "color": "#4a9eff",
            "description": "70% of gland — where most cancers arise",
            "cancer_risk": "HIGH — 70-80% of prostate cancers",
        },
        "central": {
            "volume_pct": 25,
            "color": "#7eb3ff",
            "description": "25% — surrounds ejaculatory ducts",
            "cancer_risk": "LOW — 5-10% of cancers",
        },
        "transition": {
            "volume_pct": 5,
            "color": "#a8c8ff",
            "description": "5% — surrounds urethra, site of BPH",
            "cancer_risk": "MODERATE — 15-20% of cancers",
        },
    },
    "structures": {
        "urethra": {"color": "#ffcc00", "description": "Carries urine/semen"},
        "ejaculatory_ducts": {"color": "#ff9900", "description": "Sperm pathway"},
        "seminal_vesicles": {"color": "#cc6600", "description": "Fluid production"},
        "neurovascular_bundles": {"color": "#ff6666", "description": "Nerves + blood supply"},
    },
}

METASTASIS_STAGES = {
    "1_local_invasion": {
        "name": "Local Invasion",
        "description": "Cancer breaks through basement membrane",
        "mechanisms": [
            "E-cadherin loss (epithelial-to-mesenchymal transition)",
            "Matrix metalloproteinases (MMPs) digest ECM",
            "Invadopodia formation — actin protrusions",
        ],
        "molecules": ["E-cadherin ↓", "MMP-2 ↑", "MMP-9 ↑", "Vimentin ↑"],
    },
    "2_intravasation": {
        "name": "Intravasation (Entering Blood)",
        "description": "Cancer cells squeeze into blood/lymph vessels",
        "mechanisms": [
            "VEGF-induced vessel permeability",
            "Tumor-associated macrophages (TAMs) assist entry",
            "Cancer cells squeeze between endothelial cells",
        ],
        "molecules": ["VEGF ↑", "Angiopoietin-2 ↑", "CCL2 ↑"],
    },
    "3_circulation": {
        "name": "Circulating Tumor Cells (CTCs)",
        "description": "Cancer cells survive in bloodstream",
        "mechanisms": [
            "Platelet cloaking — hides from immune system",
            "Anoikis resistance — survives without attachment",
            "Clusters more lethal than single cells",
        ],
        "molecules": ["Tissue Factor ↑", "Bcl-2 ↑", "TGF-β ↑"],
    },
    "4_extravasation": {
        "name": "Extravasation (Exiting Blood)",
        "description": "Cancer cells exit blood vessel at distant site",
        "mechanisms": [
            "Arrest at narrow capillaries",
            "Adhesion to endothelium",
            "Transmigration through vessel wall",
        ],
        "molecules": ["E-selectin", "VCAM-1", "ICAM-1"],
    },
    "5_bone_colonization": {
        "name": "Bone Colonization",
        "description": "Why prostate cancer loves bone (osteotropism)",
        "mechanisms": [
            "CXCR4/CXCL12 axis — bone marrow homing signal",
            "RANKL/OPG — stimulates osteoclasts → bone destruction",
            "ET-1 — stimulates osteoblasts → bone formation",
            "Vicious cycle: bone releases TGF-β → cancer grows",
        ],
        "molecules": ["CXCR4 ↑", "RANKL ↑", "ET-1 ↑", "PTHrP ↑"],
    },
}

EXOSOME_CARGO = {
    "description": "Exosomes: 30-150nm vesicles carrying molecular cargo",
    "contents": {
        "mRNA": ["AR-V7", "PCA3", "TMPRSS2-ERG"],
        "miRNA": ["miR-141", "miR-375", "miR-21"],
        "proteins": ["PSA", "PSMA", "Survivin", "HSP70"],
        "DNA": ["cfDNA fragments", "Cell-free tumor DNA (ctDNA)"],
    },
    "functions": [
        "Prepare pre-metastatic niche",
        "Educate bone marrow cells",
        "Suppress immune response",
        "Transfer drug resistance",
    ],
}

CELL_DEATH_FRAGMENTS = {
    "apoptosis": {
        "name": "Apoptosis (Programmed Death)",
        "products": [
            "Apoptotic bodies (1-5 μm)",
            "cfDNA fragments (150-200 bp, nucleosome-sized)",
            "Phosphatidylserine on outer membrane (eat-me signal)",
        ],
        "fate": "Macrophages phagocytose — clean removal",
    },
    "necrosis": {
        "name": "Necrosis (Uncontrolled Death)",
        "products": [
            "Cellular debris (random sizes)",
            "DAMPs — damage-associated molecular patterns",
            "Large DNA fragments (>10kb)",
            "Inflammatory cytokines",
        ],
        "fate": "Inflammation, immune activation",
    },
    "pyroptosis": {
        "name": "Pyroptosis (Inflammatory Death)",
        "products": [
            "Gasdermin D pores → cell lysis",
            "IL-1β, IL-18 release",
            "cfDNA in inflammasome complexes",
        ],
        "fate": "Strong inflammation, immune activation",
    },
}

# ─────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────

tab_organ, tab_tissue, tab_metastasis, tab_exosomes, tab_bone, tab_simulation = st.tabs([
    "🫀 Organ Anatomy",
    "🔬 Tissue Structure",
    "🚀 Metastasis Cascade",
    "📦 Exosomes & cfDNA",
    "🦴 Bone Tropism",
    "▶️ Live Simulation",
])

# ─────────────────────────────────────────────────────────────────────────
# TAB 1: ORGAN ANATOMY
# ─────────────────────────────────────────────────────────────────────────

with tab_organ:
    st.subheader("Prostate Organ Anatomy")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Interactive 3D prostate
        organ_html = '''
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { margin: 0; overflow: hidden; font-family: system-ui; }
                #info { position: absolute; top: 10px; left: 10px; color: white;
                        background: rgba(0,0,0,0.7); padding: 10px; border-radius: 5px; }
                #labels { position: absolute; top: 10px; right: 10px; color: white;
                         background: rgba(0,0,0,0.7); padding: 10px; border-radius: 5px; font-size: 12px; }
            </style>
        </head>
        <body>
            <div id="info">🖱 Click zones to learn more</div>
            <div id="labels">
                <div style="margin-bottom: 5px;"><span style="color: #4a9eff;">●</span> Peripheral Zone (70%)</div>
                <div style="margin-bottom: 5px;"><span style="color: #7eb3ff;">●</span> Central Zone (25%)</div>
                <div style="margin-bottom: 5px;"><span style="color: #a8c8ff;">●</span> Transition Zone (5%)</div>
                <div style="margin-bottom: 5px;"><span style="color: #ffcc00;">━</span> Urethra</div>
                <div style="margin-bottom: 5px;"><span style="color: #ff6666;">━</span> Neurovascular Bundles</div>
                <div style="margin-bottom: 5px;"><span style="color: #0066ff;">━</span> Blood Vessels</div>
                <div><span style="color: #00ccff;">━</span> Lymphatics</div>
            </div>

            <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
            <script>
                const scene = new THREE.Scene();
                scene.background = new THREE.Color(0x1a1a2e);

                const camera = new THREE.PerspectiveCamera(60, window.innerWidth / 500, 0.1, 1000);
                camera.position.set(0, 0, 80);

                const renderer = new THREE.WebGLRenderer({ antialias: true });
                renderer.setSize(window.innerWidth, 500);
                document.body.appendChild(renderer.domElement);

                // Lighting
                const ambient = new THREE.AmbientLight(0xffffff, 0.4);
                scene.add(ambient);
                const directional = new THREE.DirectionalLight(0xffffff, 0.8);
                directional.position.set(50, 50, 50);
                scene.add(directional);

                // Prostate zones (nested ellipsoids)
                // Peripheral zone (outer, largest)
                const peripheralGeom = new THREE.SphereGeometry(25, 32, 32);
                peripheralGeom.scale(1.2, 0.8, 1);
                const peripheralMat = new THREE.MeshStandardMaterial({
                    color: 0x4a9eff, transparent: true, opacity: 0.4
                });
                const peripheral = new THREE.Mesh(peripheralGeom, peripheralMat);
                peripheral.name = 'peripheral';
                scene.add(peripheral);

                // Central zone
                const centralGeom = new THREE.SphereGeometry(18, 32, 32);
                centralGeom.scale(1.1, 0.7, 0.9);
                const centralMat = new THREE.MeshStandardMaterial({
                    color: 0x7eb3ff, transparent: true, opacity: 0.5
                });
                const central = new THREE.Mesh(centralGeom, centralMat);
                central.name = 'central';
                scene.add(central);

                // Transition zone (around urethra)
                const transitionGeom = new THREE.SphereGeometry(10, 32, 32);
                transitionGeom.scale(1.0, 0.6, 0.8);
                const transitionMat = new THREE.MeshStandardMaterial({
                    color: 0xa8c8ff, transparent: true, opacity: 0.6
                });
                const transition = new THREE.Mesh(transitionGeom, transitionMat);
                transition.name = 'transition';
                scene.add(transition);

                // Urethra (tube through center)
                const urethraGeom = new THREE.CylinderGeometry(2, 2, 50, 16);
                const urethraMat = new THREE.MeshStandardMaterial({ color: 0xffcc00 });
                const urethra = new THREE.Mesh(urethraGeom, urethraMat);
                urethra.name = 'urethra';
                scene.add(urethra);

                // Neurovascular bundles (two cylinders on sides)
                for (let side of [-1, 1]) {
                    const nvbGeom = new THREE.CylinderGeometry(3, 3, 35, 16);
                    const nvbMat = new THREE.MeshStandardMaterial({ color: 0xff6666 });
                    const nvb = new THREE.Mesh(nvbGeom, nvbMat);
                    nvb.position.set(side * 28, 0, 0);
                    nvb.name = 'nvb';
                    scene.add(nvb);
                }

                // Blood vessels (arteries/veins)
                function createVessel(start, end, color, radius) {
                    const dir = new THREE.Vector3().subVectors(end, start);
                    const length = dir.length();
                    const geom = new THREE.CylinderGeometry(radius, radius, length, 8);
                    const mat = new THREE.MeshStandardMaterial({ color: color });
                    const vessel = new THREE.Mesh(geom, mat);

                    const midpoint = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
                    vessel.position.copy(midpoint);
                    vessel.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
                    return vessel;
                }

                // Arterial supply (inferior vesical, internal pudendal)
                const vessels = [
                    [new THREE.Vector3(-30, 25, 0), new THREE.Vector3(-20, 0, 0), 0x0066ff, 1.5],
                    [new THREE.Vector3(30, 25, 0), new THREE.Vector3(20, 0, 0), 0x0066ff, 1.5],
                    [new THREE.Vector3(-20, 0, 0), new THREE.Vector3(-10, -15, 0), 0x0066ff, 1],
                    [new THREE.Vector3(20, 0, 0), new THREE.Vector3(10, -15, 0), 0x0066ff, 1],
                ];
                vessels.forEach(v => scene.add(createVessel(v[0], v[1], v[2], v[3])));

                // Lymphatics (larger, blue-green)
                const lymphatics = [
                    [new THREE.Vector3(-25, -20, 5), new THREE.Vector3(-35, -30, 10), 0x00ccff, 2],
                    [new THREE.Vector3(25, -20, 5), new THREE.Vector3(35, -30, 10), 0x00ccff, 2],
                    [new THREE.Vector3(0, -20, 10), new THREE.Vector3(0, -35, 15), 0x00ccff, 2.5],
                ];
                lymphatics.forEach(v => scene.add(createVessel(v[0], v[1], v[2], v[3])));

                // Tumor (red mass in peripheral zone)
                const tumorGeom = new THREE.SphereGeometry(6, 16, 16);
                const tumorMat = new THREE.MeshStandardMaterial({ color: 0xff3333 });
                const tumor = new THREE.Mesh(tumorGeom, tumorMat);
                tumor.position.set(15, -8, 10);
                tumor.name = 'tumor';
                scene.add(tumor);

                // Rotation
                let rotationSpeed = 0.002;
                function animate() {
                    requestAnimationFrame(animate);
                    scene.rotation.y += rotationSpeed;
                    renderer.render(scene, camera);
                }
                animate();

                // Mouse interaction
                document.addEventListener('click', () => {
                    rotationSpeed = rotationSpeed === 0 ? 0.002 : 0;
                });
            </script>
        </body>
        </html>
        '''
        components.html(organ_html, height=520)

    with col2:
        st.markdown("### Prostate Zones")
        for zone, data in PROSTATE_ANATOMY["zones"].items():
            with st.expander(f"**{zone.title()} Zone** ({data['volume_pct']}%)"):
                st.markdown(data["description"])
                st.warning(f"Cancer Risk: {data['cancer_risk']}")

        st.divider()
        st.markdown("### Key Structures")
        for struct, data in PROSTATE_ANATOMY["structures"].items():
            st.markdown(f"**{struct.replace('_', ' ').title()}**: {data['description']}")

        st.divider()
        st.info("""
        **Metastasis Routes:**
        1. **Lymphatic** → Obturator, internal iliac, presacral nodes
        2. **Hematogenous** → Batson's plexus (valveless) → Spine, pelvis, femur
        """)

# ─────────────────────────────────────────────────────────────────────────
# TAB 2: TISSUE STRUCTURE
# ─────────────────────────────────────────────────────────────────────────

with tab_tissue:
    st.subheader("Prostate Tissue Architecture")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Tissue cross-section view
        tissue_html = '''
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { margin: 0; overflow: hidden; font-family: system-ui; }
                #info { position: absolute; top: 10px; left: 10px; color: white;
                        background: rgba(0,0,0,0.7); padding: 10px; border-radius: 5px;
                        max-width: 300px; }
            </style>
        </head>
        <body>
            <div id="info">
                <strong>Glandular Acinus</strong><br>
                Secretory unit of prostate<br>
                <span style="color: #66ff66;">● Luminal cells</span> (secrete PSA)<br>
                <span style="color: #ffcc00;">● Basal cells</span> (stem-like)<br>
                <span style="color: #ff6666;">● Cancer cells</span> (invading)
            </div>

            <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
            <script>
                const scene = new THREE.Scene();
                scene.background = new THREE.Color(0x1a1a2e);

                const camera = new THREE.PerspectiveCamera(60, window.innerWidth / 500, 0.1, 1000);
                camera.position.set(0, 50, 100);
                camera.lookAt(0, 0, 0);

                const renderer = new THREE.WebGLRenderer({ antialias: true });
                renderer.setSize(window.innerWidth, 500);
                document.body.appendChild(renderer.domElement);

                // Lighting
                scene.add(new THREE.AmbientLight(0xffffff, 0.5));
                const dir = new THREE.DirectionalLight(0xffffff, 0.8);
                dir.position.set(50, 100, 50);
                scene.add(dir);

                // Create multiple glandular acini
                function createAcinus(x, y, z, hasCancer) {
                    const group = new THREE.Group();

                    // Lumen (hollow center)
                    const lumenGeom = new THREE.SphereGeometry(8, 16, 16);
                    const lumenMat = new THREE.MeshStandardMaterial({
                        color: 0x332244, transparent: true, opacity: 0.3
                    });
                    const lumen = new THREE.Mesh(lumenGeom, lumenMat);
                    group.add(lumen);

                    // Luminal epithelial cells (ring around lumen)
                    for (let i = 0; i < 12; i++) {
                        const angle = (i / 12) * Math.PI * 2;
                        const cellGeom = new THREE.SphereGeometry(3, 8, 8);
                        const isCancer = hasCancer && i >= 8;
                        const cellMat = new THREE.MeshStandardMaterial({
                            color: isCancer ? 0xff4444 : 0x66ff66
                        });
                        const cell = new THREE.Mesh(cellGeom, cellMat);
                        cell.position.set(Math.cos(angle) * 12, 0, Math.sin(angle) * 12);
                        group.add(cell);
                    }

                    // Basal cells (outer ring)
                    for (let i = 0; i < 16; i++) {
                        const angle = (i / 16) * Math.PI * 2;
                        const cellGeom = new THREE.SphereGeometry(2, 8, 8);
                        const cellMat = new THREE.MeshStandardMaterial({ color: 0xffcc00 });
                        const cell = new THREE.Mesh(cellGeom, cellMat);
                        cell.position.set(Math.cos(angle) * 17, 0, Math.sin(angle) * 17);
                        group.add(cell);
                    }

                    // Basement membrane (thin ring)
                    const bmGeom = new THREE.TorusGeometry(18, 0.5, 8, 32);
                    const bmMat = new THREE.MeshStandardMaterial({ color: 0x8866aa });
                    const bm = new THREE.Mesh(bmGeom, bmMat);
                    bm.rotation.x = Math.PI / 2;
                    group.add(bm);

                    group.position.set(x, y, z);
                    return group;
                }

                // Create acini grid
                const positions = [
                    [-40, 0, -30], [0, 0, -30], [40, 0, -30],
                    [-40, 0, 30], [0, 0, 30], [40, 0, 30],
                ];
                positions.forEach((pos, i) => {
                    const hasCancer = i === 4 || i === 5; // Cancer in right-lower acini
                    scene.add(createAcinus(pos[0], pos[1], pos[2], hasCancer));
                });

                // Stroma (fibroblasts between acini)
                for (let i = 0; i < 50; i++) {
                    const geom = new THREE.SphereGeometry(1.5, 4, 4);
                    const mat = new THREE.MeshStandardMaterial({ color: 0x886644 });
                    const fb = new THREE.Mesh(geom, mat);
                    fb.position.set(
                        (Math.random() - 0.5) * 120,
                        (Math.random() - 0.5) * 20,
                        (Math.random() - 0.5) * 100
                    );
                    scene.add(fb);
                }

                // Blood capillaries (thin red tubes)
                function createCapillary(points) {
                    const curve = new THREE.CatmullRomCurve3(points);
                    const geom = new THREE.TubeGeometry(curve, 20, 1, 8, false);
                    const mat = new THREE.MeshStandardMaterial({ color: 0xff3333 });
                    return new THREE.Mesh(geom, mat);
                }

                const cap1 = createCapillary([
                    new THREE.Vector3(-60, 10, -20),
                    new THREE.Vector3(-20, 8, 0),
                    new THREE.Vector3(20, 12, 10),
                    new THREE.Vector3(60, 10, 0),
                ]);
                scene.add(cap1);

                const cap2 = createCapillary([
                    new THREE.Vector3(-60, -10, 20),
                    new THREE.Vector3(-20, -8, 10),
                    new THREE.Vector3(20, -12, 0),
                    new THREE.Vector3(60, -10, 10),
                ]);
                scene.add(cap2);

                // Immune cells patrolling
                const immuneCells = [];
                for (let i = 0; i < 8; i++) {
                    const geom = new THREE.SphereGeometry(2.5, 8, 8);
                    const type = i < 4 ? 0x00ffff : 0xff00ff; // T cells cyan, NK cells magenta
                    const mat = new THREE.MeshStandardMaterial({ color: type });
                    const immune = new THREE.Mesh(geom, mat);
                    immune.position.set(
                        (Math.random() - 0.5) * 100,
                        (Math.random() - 0.5) * 30,
                        (Math.random() - 0.5) * 80
                    );
                    immune.userData.velocity = new THREE.Vector3(
                        (Math.random() - 0.5) * 0.5,
                        (Math.random() - 0.5) * 0.2,
                        (Math.random() - 0.5) * 0.5
                    );
                    scene.add(immune);
                    immuneCells.push(immune);
                }

                // Animation
                function animate() {
                    requestAnimationFrame(animate);

                    // Move immune cells
                    immuneCells.forEach(cell => {
                        cell.position.add(cell.userData.velocity);
                        // Bounce off boundaries
                        if (Math.abs(cell.position.x) > 60) cell.userData.velocity.x *= -1;
                        if (Math.abs(cell.position.y) > 20) cell.userData.velocity.y *= -1;
                        if (Math.abs(cell.position.z) > 50) cell.userData.velocity.z *= -1;
                    });

                    // Slow rotation
                    scene.rotation.y += 0.001;

                    renderer.render(scene, camera);
                }
                animate();
            </script>
        </body>
        </html>
        '''
        components.html(tissue_html, height=520)

    with col2:
        st.markdown("### Tissue Components")

        st.markdown("""
        #### Glandular Acinus
        - **Lumen**: Hollow center for secretions
        - **Luminal cells**: PSA-secreting, AR-dependent
        - **Basal cells**: Stem-like, express p63
        - **Basement membrane**: Barrier cancer must cross

        #### Stroma
        - Fibroblasts (ECM production)
        - Smooth muscle (contraction)
        - Blood vessels (nutrient supply)

        #### Immune Cells
        - **T cells** (cyan): MHC-I recognition
        - **NK cells** (magenta): Missing-self detection
        - **Macrophages**: Phagocytosis
        """)

        st.divider()
        st.error("""
        **Cancer Invasion Starts Here**

        When cancer breaks through the basement membrane,
        it gains access to:
        1. Blood vessels → Hematogenous spread
        2. Lymphatics → Lymph node metastasis
        3. Nerves → Perineural invasion
        """)

# ─────────────────────────────────────────────────────────────────────────
# TAB 3: METASTASIS CASCADE
# ─────────────────────────────────────────────────────────────────────────

with tab_metastasis:
    st.subheader("The Metastatic Cascade — Step by Step")

    # Progress through stages
    stage_names = list(METASTASIS_STAGES.keys())
    selected_stage = st.select_slider(
        "Metastasis Stage",
        options=stage_names,
        format_func=lambda x: METASTASIS_STAGES[x]["name"],
    )

    stage_data = METASTASIS_STAGES[selected_stage]

    col1, col2 = st.columns([2, 1])

    with col1:
        # Dynamic visualization based on stage
        stage_num = stage_names.index(selected_stage)

        metastasis_html = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ margin: 0; overflow: hidden; font-family: system-ui; }}
                #stage {{ position: absolute; top: 10px; left: 10px; color: white;
                         background: rgba(255,50,50,0.8); padding: 15px; border-radius: 8px;
                         font-size: 16px; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div id="stage">Stage {stage_num + 1}/5: {stage_data["name"]}</div>

            <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
            <script>
                const scene = new THREE.Scene();
                scene.background = new THREE.Color(0x1a1a2e);
                const stageNum = {stage_num};

                const camera = new THREE.PerspectiveCamera(60, window.innerWidth / 450, 0.1, 1000);
                camera.position.set(0, 20, 80);

                const renderer = new THREE.WebGLRenderer({{ antialias: true }});
                renderer.setSize(window.innerWidth, 450);
                document.body.appendChild(renderer.domElement);

                scene.add(new THREE.AmbientLight(0xffffff, 0.4));
                const dir = new THREE.DirectionalLight(0xffffff, 0.8);
                dir.position.set(50, 50, 50);
                scene.add(dir);

                // Stage 0: Local invasion - cancer breaking through basement membrane
                if (stageNum === 0) {{
                    // Basement membrane (purple ring)
                    const bmGeom = new THREE.TorusGeometry(25, 1, 8, 64);
                    const bmMat = new THREE.MeshStandardMaterial({{ color: 0x8866aa }});
                    const bm = new THREE.Mesh(bmGeom, bmMat);
                    bm.rotation.x = Math.PI / 2;
                    scene.add(bm);

                    // Normal cells inside
                    for (let i = 0; i < 15; i++) {{
                        const angle = (i / 15) * Math.PI * 2;
                        const r = 15 + Math.random() * 5;
                        const geom = new THREE.SphereGeometry(3, 8, 8);
                        const mat = new THREE.MeshStandardMaterial({{ color: 0x66ff66 }});
                        const cell = new THREE.Mesh(geom, mat);
                        cell.position.set(Math.cos(angle) * r, (Math.random() - 0.5) * 5, Math.sin(angle) * r);
                        scene.add(cell);
                    }}

                    // Cancer cells breaking through
                    for (let i = 0; i < 5; i++) {{
                        const geom = new THREE.SphereGeometry(4, 8, 8);
                        const mat = new THREE.MeshStandardMaterial({{ color: 0xff3333 }});
                        const cell = new THREE.Mesh(geom, mat);
                        // Position at membrane edge, pushing through
                        const angle = -0.3 + (i * 0.15);
                        const r = 25 + i * 4;
                        cell.position.set(Math.cos(angle) * r, (Math.random() - 0.5) * 3, Math.sin(angle) * r);
                        scene.add(cell);
                    }}

                    // ECM fibers being degraded
                    for (let i = 0; i < 20; i++) {{
                        const geom = new THREE.CylinderGeometry(0.2, 0.2, 8, 4);
                        const mat = new THREE.MeshStandardMaterial({{ color: 0xccaa88 }});
                        const fiber = new THREE.Mesh(geom, mat);
                        fiber.position.set(25 + Math.random() * 15, (Math.random() - 0.5) * 10, (Math.random() - 0.5) * 20);
                        fiber.rotation.set(Math.random() * Math.PI, Math.random() * Math.PI, Math.random() * Math.PI);
                        scene.add(fiber);
                    }}
                }}

                // Stage 1: Intravasation - entering blood vessel
                if (stageNum === 1) {{
                    // Blood vessel (red tube)
                    const vesselGeom = new THREE.CylinderGeometry(10, 10, 80, 32, 1, true);
                    const vesselMat = new THREE.MeshStandardMaterial({{
                        color: 0xff3333, side: THREE.DoubleSide, transparent: true, opacity: 0.5
                    }});
                    const vessel = new THREE.Mesh(vesselGeom, vesselMat);
                    vessel.rotation.z = Math.PI / 2;
                    scene.add(vessel);

                    // Endothelial cells lining vessel
                    for (let i = 0; i < 40; i++) {{
                        const angle = (i / 40) * Math.PI * 2;
                        const x = (Math.random() - 0.5) * 60;
                        const geom = new THREE.BoxGeometry(4, 1, 2);
                        const mat = new THREE.MeshStandardMaterial({{ color: 0xaa2222 }});
                        const endo = new THREE.Mesh(geom, mat);
                        endo.position.set(x, Math.sin(angle) * 9.5, Math.cos(angle) * 9.5);
                        endo.lookAt(x, 0, 0);
                        scene.add(endo);
                    }}

                    // Cancer cells squeezing through
                    const cancerGeom = new THREE.SphereGeometry(5, 8, 8);
                    cancerGeom.scale(0.6, 1.5, 0.6); // Elongated, squeezing
                    const cancerMat = new THREE.MeshStandardMaterial({{ color: 0xff3333 }});
                    const cancer = new THREE.Mesh(cancerGeom, cancerMat);
                    cancer.position.set(0, 10, 0);
                    scene.add(cancer);

                    // TAM (tumor-associated macrophage) helping
                    const tamGeom = new THREE.SphereGeometry(4, 8, 8);
                    const tamMat = new THREE.MeshStandardMaterial({{ color: 0xff9900 }});
                    const tam = new THREE.Mesh(tamGeom, tamMat);
                    tam.position.set(-5, 12, 3);
                    scene.add(tam);
                }}

                // Stage 2: Circulation - CTCs in bloodstream
                if (stageNum === 2) {{
                    // Blood vessel (tube)
                    const vesselGeom = new THREE.CylinderGeometry(15, 15, 100, 32, 1, true);
                    const vesselMat = new THREE.MeshStandardMaterial({{
                        color: 0x990000, side: THREE.DoubleSide, transparent: true, opacity: 0.3
                    }});
                    const vessel = new THREE.Mesh(vesselGeom, vesselMat);
                    vessel.rotation.z = Math.PI / 2;
                    scene.add(vessel);

                    // Red blood cells flowing
                    for (let i = 0; i < 100; i++) {{
                        const geom = new THREE.TorusGeometry(1.5, 0.5, 4, 8);
                        const mat = new THREE.MeshStandardMaterial({{ color: 0xcc0000 }});
                        const rbc = new THREE.Mesh(geom, mat);
                        rbc.position.set(
                            (Math.random() - 0.5) * 80,
                            (Math.random() - 0.5) * 20,
                            (Math.random() - 0.5) * 20
                        );
                        rbc.rotation.set(Math.random() * Math.PI, Math.random() * Math.PI, 0);
                        scene.add(rbc);
                    }}

                    // CTC (single circulating tumor cell)
                    const ctcGeom = new THREE.SphereGeometry(4, 16, 16);
                    const ctcMat = new THREE.MeshStandardMaterial({{ color: 0xff3333 }});
                    const ctc = new THREE.Mesh(ctcGeom, ctcMat);
                    ctc.position.set(-20, 0, 0);
                    scene.add(ctc);

                    // CTC cluster (more dangerous)
                    const clusterGroup = new THREE.Group();
                    for (let i = 0; i < 4; i++) {{
                        const geom = new THREE.SphereGeometry(3, 8, 8);
                        const mat = new THREE.MeshStandardMaterial({{ color: 0xff3333 }});
                        const cell = new THREE.Mesh(geom, mat);
                        cell.position.set(i * 5, (i % 2) * 3, 0);
                        clusterGroup.add(cell);
                    }}
                    clusterGroup.position.set(20, 5, 0);
                    scene.add(clusterGroup);

                    // Platelets coating CTC (immune evasion)
                    for (let i = 0; i < 20; i++) {{
                        const geom = new THREE.SphereGeometry(0.8, 4, 4);
                        const mat = new THREE.MeshStandardMaterial({{ color: 0xffcc00 }});
                        const plt = new THREE.Mesh(geom, mat);
                        const angle = (i / 20) * Math.PI * 2;
                        const angle2 = (i * 0.5);
                        plt.position.set(
                            -20 + Math.cos(angle) * 5 * Math.cos(angle2),
                            Math.sin(angle) * 5,
                            Math.cos(angle) * 5 * Math.sin(angle2)
                        );
                        scene.add(plt);
                    }}
                }}

                // Stage 3: Extravasation - exiting at distant site
                if (stageNum === 3) {{
                    // Narrow capillary
                    const vesselGeom = new THREE.CylinderGeometry(8, 8, 60, 32, 1, true);
                    const vesselMat = new THREE.MeshStandardMaterial({{
                        color: 0xff3333, side: THREE.DoubleSide, transparent: true, opacity: 0.4
                    }});
                    const vessel = new THREE.Mesh(vesselGeom, vesselMat);
                    vessel.rotation.z = Math.PI / 2;
                    scene.add(vessel);

                    // CTC arrested at narrow point
                    const ctcGeom = new THREE.SphereGeometry(6, 16, 16);
                    const ctcMat = new THREE.MeshStandardMaterial({{ color: 0xff3333 }});
                    const ctc = new THREE.Mesh(ctcGeom, ctcMat);
                    ctc.position.set(0, 0, 0);
                    scene.add(ctc);

                    // Adhesion molecules (selectins, integrins)
                    for (let i = 0; i < 12; i++) {{
                        const angle = (i / 12) * Math.PI * 2;
                        const geom = new THREE.CylinderGeometry(0.3, 0.3, 3, 4);
                        const mat = new THREE.MeshStandardMaterial({{ color: 0x00ff00 }});
                        const mol = new THREE.Mesh(geom, mat);
                        mol.position.set(0, Math.sin(angle) * 7, Math.cos(angle) * 7);
                        mol.lookAt(0, 0, 0);
                        scene.add(mol);
                    }}

                    // Tissue outside vessel
                    for (let i = 0; i < 30; i++) {{
                        const geom = new THREE.SphereGeometry(2, 4, 4);
                        const mat = new THREE.MeshStandardMaterial({{ color: 0x886644 }});
                        const cell = new THREE.Mesh(geom, mat);
                        const angle = Math.random() * Math.PI * 2;
                        const r = 15 + Math.random() * 20;
                        cell.position.set(
                            (Math.random() - 0.5) * 40,
                            Math.sin(angle) * r,
                            Math.cos(angle) * r
                        );
                        scene.add(cell);
                    }}
                }}

                // Stage 4: Bone colonization
                if (stageNum === 4) {{
                    // Bone matrix (lattice structure)
                    for (let i = 0; i < 80; i++) {{
                        const geom = new THREE.CylinderGeometry(0.5, 0.5, 15, 4);
                        const mat = new THREE.MeshStandardMaterial({{ color: 0xeeeecc }});
                        const trabecula = new THREE.Mesh(geom, mat);
                        trabecula.position.set(
                            (Math.random() - 0.5) * 60,
                            (Math.random() - 0.5) * 40,
                            (Math.random() - 0.5) * 40
                        );
                        trabecula.rotation.set(
                            Math.random() * Math.PI,
                            Math.random() * Math.PI,
                            Math.random() * Math.PI
                        );
                        scene.add(trabecula);
                    }}

                    // Bone marrow cells (hematopoietic niche)
                    for (let i = 0; i < 50; i++) {{
                        const geom = new THREE.SphereGeometry(1.5, 4, 4);
                        const mat = new THREE.MeshStandardMaterial({{ color: 0xcc6699 }});
                        const cell = new THREE.Mesh(geom, mat);
                        cell.position.set(
                            (Math.random() - 0.5) * 50,
                            (Math.random() - 0.5) * 30,
                            (Math.random() - 0.5) * 30
                        );
                        scene.add(cell);
                    }}

                    // Cancer cells colonizing
                    const cancerPositions = [
                        [0, 0, 0], [8, 5, 3], [-5, -3, 6], [3, 8, -4], [-6, 4, -5]
                    ];
                    cancerPositions.forEach((pos, i) => {{
                        const geom = new THREE.SphereGeometry(4 - i * 0.3, 16, 16);
                        const mat = new THREE.MeshStandardMaterial({{ color: 0xff3333 }});
                        const cancer = new THREE.Mesh(geom, mat);
                        cancer.position.set(pos[0], pos[1], pos[2]);
                        scene.add(cancer);
                    }});

                    // Osteoclasts (bone destruction)
                    for (let i = 0; i < 5; i++) {{
                        const geom = new THREE.SphereGeometry(3, 8, 8);
                        const mat = new THREE.MeshStandardMaterial({{ color: 0x9933ff }});
                        const oc = new THREE.Mesh(geom, mat);
                        oc.position.set(
                            10 + Math.random() * 10,
                            (Math.random() - 0.5) * 15,
                            (Math.random() - 0.5) * 15
                        );
                        scene.add(oc);
                    }}

                    // Osteoblasts (new bone around tumor)
                    for (let i = 0; i < 8; i++) {{
                        const geom = new THREE.BoxGeometry(2, 2, 2);
                        const mat = new THREE.MeshStandardMaterial({{ color: 0x66ff66 }});
                        const ob = new THREE.Mesh(geom, mat);
                        ob.position.set(
                            -10 - Math.random() * 10,
                            (Math.random() - 0.5) * 15,
                            (Math.random() - 0.5) * 15
                        );
                        scene.add(ob);
                    }}
                }}

                // Animation
                function animate() {{
                    requestAnimationFrame(animate);
                    scene.rotation.y += 0.002;
                    renderer.render(scene, camera);
                }}
                animate();
            </script>
        </body>
        </html>
        '''
        components.html(metastasis_html, height=470)

    with col2:
        st.markdown(f"### {stage_data['name']}")
        st.info(stage_data["description"])

        st.markdown("**Mechanisms:**")
        for mech in stage_data["mechanisms"]:
            st.markdown(f"- {mech}")

        st.divider()
        st.markdown("**Key Molecules:**")
        for mol in stage_data["molecules"]:
            if "↑" in mol:
                st.success(mol)
            else:
                st.error(mol)

# ─────────────────────────────────────────────────────────────────────────
# TAB 4: EXOSOMES & cfDNA
# ─────────────────────────────────────────────────────────────────────────

with tab_exosomes:
    st.subheader("Exosomes & Cell-Free DNA — Liquid Biopsy Targets")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Exosome and cfDNA visualization
        exosome_html = '''
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { margin: 0; overflow: hidden; font-family: system-ui; }
                #info { position: absolute; top: 10px; left: 10px; color: white;
                        background: rgba(0,0,0,0.8); padding: 15px; border-radius: 8px; }
            </style>
        </head>
        <body>
            <div id="info">
                <strong>Cell Death → cfDNA Release</strong><br>
                <span style="color: #66ff66;">● Apoptotic bodies</span><br>
                <span style="color: #ff6666;">● Necrotic debris</span><br>
                <span style="color: #ffcc00;">○ Exosomes (30-150nm)</span><br>
                <span style="color: #00ffff;">~ cfDNA fragments</span>
            </div>

            <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
            <script>
                const scene = new THREE.Scene();
                scene.background = new THREE.Color(0x1a1a2e);

                const camera = new THREE.PerspectiveCamera(60, window.innerWidth / 450, 0.1, 1000);
                camera.position.set(0, 30, 80);

                const renderer = new THREE.WebGLRenderer({ antialias: true });
                renderer.setSize(window.innerWidth, 450);
                document.body.appendChild(renderer.domElement);

                scene.add(new THREE.AmbientLight(0xffffff, 0.4));
                const dir = new THREE.DirectionalLight(0xffffff, 0.8);
                dir.position.set(50, 50, 50);
                scene.add(dir);

                // Dying cancer cell (blebbing)
                const dyingCellGeom = new THREE.SphereGeometry(15, 16, 16);
                const dyingCellMat = new THREE.MeshStandardMaterial({
                    color: 0xff3333, transparent: true, opacity: 0.6
                });
                const dyingCell = new THREE.Mesh(dyingCellGeom, dyingCellMat);
                dyingCell.position.set(-30, 0, 0);
                scene.add(dyingCell);

                // Blebs (apoptotic blebbing)
                const blebs = [];
                for (let i = 0; i < 8; i++) {
                    const size = 2 + Math.random() * 3;
                    const geom = new THREE.SphereGeometry(size, 8, 8);
                    const mat = new THREE.MeshStandardMaterial({ color: 0x66ff66 });
                    const bleb = new THREE.Mesh(geom, mat);
                    const angle = (i / 8) * Math.PI * 2;
                    bleb.position.set(
                        -30 + Math.cos(angle) * 18,
                        Math.sin(angle) * 10,
                        (Math.random() - 0.5) * 15
                    );
                    bleb.userData.velocity = new THREE.Vector3(
                        Math.cos(angle) * 0.2,
                        Math.sin(angle) * 0.1,
                        (Math.random() - 0.5) * 0.1
                    );
                    scene.add(bleb);
                    blebs.push(bleb);
                }

                // Exosomes (tiny vesicles)
                const exosomes = [];
                for (let i = 0; i < 30; i++) {
                    const geom = new THREE.SphereGeometry(1, 8, 8);
                    const mat = new THREE.MeshStandardMaterial({ color: 0xffcc00 });
                    const exo = new THREE.Mesh(geom, mat);
                    exo.position.set(
                        (Math.random() - 0.5) * 80,
                        (Math.random() - 0.5) * 40,
                        (Math.random() - 0.5) * 40
                    );
                    exo.userData.velocity = new THREE.Vector3(
                        (Math.random() - 0.5) * 0.3,
                        (Math.random() - 0.5) * 0.2,
                        (Math.random() - 0.5) * 0.2
                    );
                    scene.add(exo);
                    exosomes.push(exo);
                }

                // cfDNA fragments (helical curves)
                const cfDNAs = [];
                for (let i = 0; i < 15; i++) {
                    const points = [];
                    const baseX = (Math.random() - 0.5) * 60;
                    const baseY = (Math.random() - 0.5) * 30;
                    const baseZ = (Math.random() - 0.5) * 30;
                    for (let t = 0; t < 20; t++) {
                        points.push(new THREE.Vector3(
                            baseX + t * 0.3,
                            baseY + Math.sin(t * 0.5) * 2,
                            baseZ + Math.cos(t * 0.5) * 2
                        ));
                    }
                    const curve = new THREE.CatmullRomCurve3(points);
                    const geom = new THREE.TubeGeometry(curve, 20, 0.3, 4, false);
                    const mat = new THREE.MeshStandardMaterial({ color: 0x00ffff });
                    const dna = new THREE.Mesh(geom, mat);
                    dna.userData.velocity = new THREE.Vector3(
                        (Math.random() - 0.5) * 0.1,
                        (Math.random() - 0.5) * 0.1,
                        (Math.random() - 0.5) * 0.1
                    );
                    scene.add(dna);
                    cfDNAs.push(dna);
                }

                // Blood vessel wall (semi-transparent)
                const vesselGeom = new THREE.CylinderGeometry(35, 35, 80, 32, 1, true);
                const vesselMat = new THREE.MeshStandardMaterial({
                    color: 0xff3333, side: THREE.DoubleSide, transparent: true, opacity: 0.15
                });
                const vessel = new THREE.Mesh(vesselGeom, vesselMat);
                vessel.rotation.z = Math.PI / 2;
                scene.add(vessel);

                // Macrophage engulfing debris
                const macGeom = new THREE.SphereGeometry(8, 16, 16);
                const macMat = new THREE.MeshStandardMaterial({ color: 0xff9900 });
                const macrophage = new THREE.Mesh(macGeom, macMat);
                macrophage.position.set(30, -10, 0);
                scene.add(macrophage);

                // Pseudopods extending
                for (let i = 0; i < 4; i++) {
                    const angle = (i / 4) * Math.PI * 2;
                    const geom = new THREE.CylinderGeometry(1, 2, 8, 8);
                    const mat = new THREE.MeshStandardMaterial({ color: 0xff9900 });
                    const pod = new THREE.Mesh(geom, mat);
                    pod.position.set(
                        30 + Math.cos(angle) * 10,
                        -10 + Math.sin(angle) * 5,
                        Math.sin(angle) * 5
                    );
                    pod.lookAt(30, -10, 0);
                    scene.add(pod);
                }

                // Animation
                function animate() {
                    requestAnimationFrame(animate);

                    // Move blebs outward
                    blebs.forEach(bleb => {
                        bleb.position.add(bleb.userData.velocity);
                    });

                    // Move exosomes
                    exosomes.forEach(exo => {
                        exo.position.add(exo.userData.velocity);
                        if (Math.abs(exo.position.x) > 50) exo.userData.velocity.x *= -1;
                        if (Math.abs(exo.position.y) > 25) exo.userData.velocity.y *= -1;
                        if (Math.abs(exo.position.z) > 25) exo.userData.velocity.z *= -1;
                    });

                    // Move cfDNA
                    cfDNAs.forEach(dna => {
                        dna.position.add(dna.userData.velocity);
                        if (Math.abs(dna.position.x) > 40) dna.userData.velocity.x *= -1;
                        if (Math.abs(dna.position.y) > 20) dna.userData.velocity.y *= -1;
                    });

                    scene.rotation.y += 0.001;
                    renderer.render(scene, camera);
                }
                animate();
            </script>
        </body>
        </html>
        '''
        components.html(exosome_html, height=470)

    with col2:
        st.markdown("### Exosome Cargo")
        st.info(EXOSOME_CARGO["description"])

        for cargo_type, items in EXOSOME_CARGO["contents"].items():
            with st.expander(f"**{cargo_type}**"):
                for item in items:
                    st.markdown(f"- {item}")

        st.divider()
        st.markdown("### Functions in Metastasis")
        for func in EXOSOME_CARGO["functions"]:
            st.success(func)

        st.divider()
        st.markdown("### Cell Death Products")
        for death_type, data in CELL_DEATH_FRAGMENTS.items():
            with st.expander(data["name"]):
                st.markdown("**Products:**")
                for prod in data["products"]:
                    st.markdown(f"- {prod}")
                st.info(f"**Fate:** {data['fate']}")

# ─────────────────────────────────────────────────────────────────────────
# TAB 5: BONE TROPISM
# ─────────────────────────────────────────────────────────────────────────

with tab_bone:
    st.subheader("Why Prostate Cancer Goes to Bone — Osteotropism")

    col1, col2 = st.columns([2, 1])

    with col1:
        bone_html = '''
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { margin: 0; overflow: hidden; font-family: system-ui; }
                #info { position: absolute; top: 10px; left: 10px; color: white;
                        background: rgba(0,0,0,0.8); padding: 15px; border-radius: 8px; }
            </style>
        </head>
        <body>
            <div id="info">
                <strong>Bone Metastasis: The Vicious Cycle</strong><br>
                <span style="color: #ff3333;">● Cancer cells</span><br>
                <span style="color: #9933ff;">● Osteoclasts</span> (bone destruction)<br>
                <span style="color: #66ff66;">■ Osteoblasts</span> (new bone)<br>
                <span style="color: #ffcc00;">● TGF-β</span> (released from bone)
            </div>

            <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
            <script>
                const scene = new THREE.Scene();
                scene.background = new THREE.Color(0x1a1a2e);

                const camera = new THREE.PerspectiveCamera(60, window.innerWidth / 450, 0.1, 1000);
                camera.position.set(0, 30, 80);

                const renderer = new THREE.WebGLRenderer({ antialias: true });
                renderer.setSize(window.innerWidth, 450);
                document.body.appendChild(renderer.domElement);

                scene.add(new THREE.AmbientLight(0xffffff, 0.4));
                const dir = new THREE.DirectionalLight(0xffffff, 0.8);
                dir.position.set(50, 50, 50);
                scene.add(dir);

                // Trabecular bone (lattice)
                for (let i = 0; i < 100; i++) {
                    const geom = new THREE.CylinderGeometry(0.8, 0.8, 20, 4);
                    const mat = new THREE.MeshStandardMaterial({ color: 0xeeeecc });
                    const trabecula = new THREE.Mesh(geom, mat);
                    trabecula.position.set(
                        (Math.random() - 0.5) * 80,
                        (Math.random() - 0.5) * 50,
                        (Math.random() - 0.5) * 50
                    );
                    trabecula.rotation.set(
                        Math.random() * Math.PI,
                        Math.random() * Math.PI,
                        Math.random() * Math.PI
                    );
                    scene.add(trabecula);
                }

                // Bone marrow (hematopoietic niche)
                for (let i = 0; i < 80; i++) {
                    const geom = new THREE.SphereGeometry(1.5, 4, 4);
                    const colors = [0xcc6699, 0xff9999, 0xffcc99];
                    const mat = new THREE.MeshStandardMaterial({
                        color: colors[Math.floor(Math.random() * colors.length)]
                    });
                    const cell = new THREE.Mesh(geom, mat);
                    cell.position.set(
                        (Math.random() - 0.5) * 70,
                        (Math.random() - 0.5) * 40,
                        (Math.random() - 0.5) * 40
                    );
                    scene.add(cell);
                }

                // Cancer cells (growing colony)
                const cancerCells = [];
                const cancerCenter = new THREE.Vector3(0, 0, 0);
                for (let i = 0; i < 15; i++) {
                    const size = 4 - i * 0.15;
                    const geom = new THREE.SphereGeometry(size, 16, 16);
                    const mat = new THREE.MeshStandardMaterial({ color: 0xff3333 });
                    const cancer = new THREE.Mesh(geom, mat);
                    const angle = (i / 15) * Math.PI * 2;
                    const r = 5 + i * 0.8;
                    cancer.position.set(
                        cancerCenter.x + Math.cos(angle) * r * 0.8,
                        cancerCenter.y + Math.sin(angle) * r * 0.6,
                        cancerCenter.z + (Math.random() - 0.5) * 10
                    );
                    scene.add(cancer);
                    cancerCells.push(cancer);
                }

                // Osteoclasts (bone-destroying cells)
                const osteoclasts = [];
                for (let i = 0; i < 8; i++) {
                    const geom = new THREE.SphereGeometry(3, 8, 8);
                    geom.scale(1.3, 1, 1); // Multinucleated, large
                    const mat = new THREE.MeshStandardMaterial({ color: 0x9933ff });
                    const oc = new THREE.Mesh(geom, mat);
                    const angle = (i / 8) * Math.PI * 2;
                    oc.position.set(
                        Math.cos(angle) * 25,
                        Math.sin(angle) * 15,
                        (Math.random() - 0.5) * 20
                    );
                    scene.add(oc);
                    osteoclasts.push(oc);
                }

                // Osteoblasts (bone-forming, around tumor)
                for (let i = 0; i < 12; i++) {
                    const geom = new THREE.BoxGeometry(2.5, 2.5, 2.5);
                    const mat = new THREE.MeshStandardMaterial({ color: 0x66ff66 });
                    const ob = new THREE.Mesh(geom, mat);
                    const angle = (i / 12) * Math.PI * 2;
                    ob.position.set(
                        Math.cos(angle) * 30,
                        Math.sin(angle) * 20,
                        (Math.random() - 0.5) * 15
                    );
                    scene.add(ob);
                }

                // TGF-β particles (released from bone)
                const tgfbs = [];
                for (let i = 0; i < 20; i++) {
                    const geom = new THREE.OctahedronGeometry(1, 0);
                    const mat = new THREE.MeshStandardMaterial({ color: 0xffcc00 });
                    const tgfb = new THREE.Mesh(geom, mat);
                    tgfb.position.set(
                        (Math.random() - 0.5) * 50,
                        (Math.random() - 0.5) * 30,
                        (Math.random() - 0.5) * 30
                    );
                    tgfb.userData.target = cancerCenter.clone();
                    scene.add(tgfb);
                    tgfbs.push(tgfb);
                }

                // Animation
                function animate() {
                    requestAnimationFrame(animate);

                    // TGF-β moves toward cancer (feedback loop)
                    tgfbs.forEach(tgfb => {
                        const dir = tgfb.userData.target.clone().sub(tgfb.position).normalize();
                        tgfb.position.add(dir.multiplyScalar(0.1));

                        // Reset when reaches target
                        if (tgfb.position.distanceTo(tgfb.userData.target) < 3) {
                            const angle = Math.random() * Math.PI * 2;
                            tgfb.position.set(
                                Math.cos(angle) * 35,
                                Math.sin(angle) * 25,
                                (Math.random() - 0.5) * 25
                            );
                        }

                        tgfb.rotation.x += 0.05;
                        tgfb.rotation.y += 0.05;
                    });

                    // Cancer cells pulse (growth)
                    const time = Date.now() * 0.001;
                    cancerCells.forEach((cell, i) => {
                        cell.scale.setScalar(1 + Math.sin(time * 2 + i) * 0.1);
                    });

                    scene.rotation.y += 0.002;
                    renderer.render(scene, camera);
                }
                animate();
            </script>
        </body>
        </html>
        '''
        components.html(bone_html, height=470)

    with col2:
        st.markdown("### The Vicious Cycle")

        st.markdown("""
        **1. Cancer arrives at bone**
        - CXCR4/CXCL12 axis — bone marrow homing signal
        - Bone is rich in CXCL12, cancer has CXCR4 receptors

        **2. Cancer stimulates osteoclasts**
        - RANKL secretion → activates osteoclasts
        - Bone is destroyed, releasing stored growth factors

        **3. TGF-β released from bone matrix**
        - Bone stores TGF-β during formation
        - Destruction releases TGF-β → feeds cancer

        **4. Cancer stimulates osteoblasts**
        - Prostate cancer is often "osteoblastic" (new bone)
        - ET-1 (endothelin-1) stimulates bone formation
        - Results in mixed lytic/blastic lesions

        **5. Cycle continues**
        - More cancer → more bone destruction
        - More destruction → more TGF-β → more cancer
        """)

        st.divider()
        st.error("""
        **Key Molecules in Bone Tropism:**
        - **CXCR4/CXCL12**: Homing signal
        - **RANKL/OPG**: Bone destruction
        - **TGF-β**: Growth factor
        - **ET-1**: Bone formation
        - **PTHrP**: Calcium release
        """)

# ─────────────────────────────────────────────────────────────────────────
# TAB 6: LIVE SIMULATION
# ─────────────────────────────────────────────────────────────────────────

with tab_simulation:
    st.subheader("Live Multi-Scale Simulation")
    st.info("Run the full Cognisom physics engine with prostate cancer metastasis scenario")

    col1, col2, col3 = st.columns(3)

    with col1:
        sim_duration = st.slider("Duration (hours)", 1.0, 48.0, 12.0, key="prostate_dur")
        n_cancer_cells = st.slider("Initial cancer cells", 5, 50, 20, key="prostate_cancer")

    with col2:
        include_immune = st.checkbox("Include immune response", value=True)
        include_exosomes = st.checkbox("Include exosome signaling", value=True)

    with col3:
        show_vasculature = st.checkbox("Show blood vessels", value=True)
        show_lymphatics = st.checkbox("Show lymphatics", value=True)

    if st.button("▶ Run Prostate Metastasis Simulation", type="primary"):
        with st.spinner("Running simulation..."):
            try:
                from cognisom.dashboard.engine_runner import EngineRunner

                # Configure for prostate scenario
                runner = EngineRunner(
                    dt=0.05,
                    duration=sim_duration,
                    scenario="Aggressive Tumor",
                    modules_enabled={
                        "cellular": True,
                        "immune": include_immune,
                        "vascular": show_vasculature,
                        "spatial": True,
                        "lymphatic": show_lymphatics,
                        "molecular": include_exosomes,
                    },
                )

                progress = st.progress(0)
                status = st.empty()

                def update(current, total):
                    progress.progress(current / total)
                    status.text(f"Step {current}/{total}")

                runner.run(progress_callback=update)
                status.text("Simulation complete!")

                st.session_state["prostate_runner"] = runner
                st.session_state["prostate_ts"] = runner.get_time_series()

            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())

    # Show results
    if "prostate_runner" in st.session_state:
        runner = st.session_state["prostate_runner"]
        ts = st.session_state["prostate_ts"]

        st.success(f"Simulation complete — {len(runner.history)} time points")

        # Metrics
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Final Cells", ts["n_cells"][-1] if ts["n_cells"] else 0)
        col_m2.metric("Cancer Cells", ts["n_cancer"][-1] if ts["n_cancer"] else 0)
        col_m3.metric("Immune Kills", ts["total_kills"][-1] if ts["total_kills"] else 0)
        col_m4.metric("Cell Deaths", ts["total_deaths"][-1] if ts["total_deaths"] else 0)

        # Time series plot
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts["time"], y=ts["n_cancer"], name="Cancer", line=dict(color="red")))
        fig.add_trace(go.Scatter(x=ts["time"], y=ts["n_immune"], name="Immune", line=dict(color="cyan")))
        fig.add_trace(go.Scatter(x=ts["time"], y=ts["n_normal"], name="Normal", line=dict(color="green")))

        fig.update_layout(
            title="Population Dynamics",
            xaxis_title="Time (hours)",
            yaxis_title="Cell Count",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer
from cognisom.dashboard.footer import render_footer
render_footer()
