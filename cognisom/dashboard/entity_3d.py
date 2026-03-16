"""
Entity 3D Preview Renderer
============================

Generates interactive 3D visualizations for biological entities in the
entity library. Uses existing platform tools:

- Proteins: 3Dmol.js viewer with PDB data from RCSB or AlphaFold
- Drugs/Metabolites: RDKit 2D structure or 3Dmol.js 3D conformer
- Cells: Procedural Plotly 3D mesh (sphere + surface features)
- Viruses: Icosahedral mesh from Plotly
- Everything else: Colored type icon with scale indicator

All rendering is on-the-fly in the browser — no stored assets.
"""

import hashlib
import logging
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from typing import Optional

log = logging.getLogger(__name__)


def render_entity_3d(entity, height: int = 250) -> bool:
    """Render a 3D preview for an entity in its library card.

    Returns True if a visualization was rendered, False if fallback only.
    """
    etype = entity.entity_type.value
    props = entity.to_dict().get("properties", {})

    # Route to the appropriate renderer
    if etype == "gene":
        return _render_protein_structure(entity, props, height)
    elif etype == "protein":
        return _render_protein_structure(entity, props, height)
    elif etype == "drug":
        return _render_drug_molecule(entity, props, height)
    elif etype == "metabolite":
        return _render_drug_molecule(entity, props, height)
    elif etype in ("cell_type", "immune_cell"):
        return _render_cell_3d(entity, height)
    elif etype == "virus":
        return _render_virus_3d(entity, height)
    elif etype == "bacterium":
        return _render_bacterium_3d(entity, height)
    elif etype == "antibody":
        return _render_antibody_3d(entity, height)
    elif etype in ("receptor", "complement", "prr", "adhesion_molecule", "cytokine"):
        return _render_protein_structure(entity, props, height)
    else:
        return _render_type_icon(entity, height)


def _render_protein_structure(entity, props: dict, height: int) -> bool:
    """Render protein structure from PDB or AlphaFold."""
    # Try to find a PDB ID
    pdb_id = ""
    ext_ids = entity.external_ids or {}

    # Check properties for PDB IDs
    pdb_ids = props.get("pdb_ids", [])
    if pdb_ids and isinstance(pdb_ids, list):
        pdb_id = pdb_ids[0]

    # Check external IDs
    if not pdb_id:
        pdb_id = ext_ids.get("pdb", "")
    if not pdb_id:
        uniprot_id = ext_ids.get("uniprot", "")
        if uniprot_id:
            # Use AlphaFold predicted structure
            return _render_alphafold(uniprot_id, entity.name, height)

    # Try well-known PDB IDs for common proteins
    if not pdb_id:
        pdb_id = _get_known_pdb_id(entity.name)

    if pdb_id:
        return _render_pdb_viewer(pdb_id, entity.name, height)

    # Fallback: show type icon
    return _render_type_icon(entity, height)


# Known PDB IDs for entities in our library
_KNOWN_PDB_IDS = {
    # Cancer driver proteins
    "AR": "2AM9",           # Androgen receptor LBD
    "TP53": "1TUP",         # p53 DNA-binding domain tetramer
    "PTEN": "1D5R",         # PTEN phosphatase domain
    "BRCA2": "1MIU",        # BRCA2 DNA-binding domain
    "RB1": "1N4M",          # Retinoblastoma pocket domain
    "EZH2": "4MI5",         # EZH2 SET domain
    "MDM2": "1YCR",         # MDM2-p53 complex
    "AKT1": "3O96",         # AKT1 kinase domain
    "PIK3CA": "4OVV",       # PI3K catalytic subunit
    # Immune checkpoints
    "CTLA-4 (CD152)": "3OSK",   # CTLA-4/B7-1 complex
    "PD-1": "4ZQK",              # PD-1/PD-L1 complex
    "LAG-3 (CD223)": "7TZG",    # LAG-3 structure
    "TIM-3 (HAVCR2)": "5F71",   # TIM-3 IgV domain
    "CD47": "5TZU",              # CD47/SIRPalpha complex
    # Signaling kinases
    "JAK1": "6BBU",         # JAK1 kinase domain
    "JAK2": "4AQC",         # JAK2 kinase domain
    "mTOR": "4DRH",         # mTOR FRB domain with rapamycin
    "CDK4": "2W96",         # CDK4/cyclin D
    "BTK": "3GEN",          # BTK kinase with ibrutinib
    "SRC": "2SRC",          # SRC kinase domain
    "MEK1 (MAP2K1)": "4AN2",  # MEK1 with trametinib
    # Transcription factors
    "NF-kappaB (p65/RELA)": "1VKX",  # NF-kappaB p65/p50
    "STAT3": "6NUQ",                   # STAT3 dimer
    "HIF-1alpha": "4H6J",             # HIF-1alpha/ARNT
    # Apoptosis
    "BCL-2": "4MAN",        # BCL-2 with venetoclax
    "BAX": "1F16",          # BAX structure
    "Caspase-3": "1PAU",    # Caspase-3 active form
    "Cytochrome c": "1HRC", # Cytochrome c
    "XIAP (BIRC4)": "1I3O", # XIAP BIR3/caspase-9
    # Drugs (target structures)
    "Pembrolizumab": "5DK3", # Pembrolizumab Fab
    "Ipatasertib": "3O96",   # AKT1
    # Growth factors
    "EGF": "1NQL",           # EGF/EGFR complex
    "VEGF-A": "1VPF",        # VEGF-A
    "HGF": "1GMO",           # HGF NK1 domain
    "IGF-1": "1GZR",         # IGF-1
    # ECM
    "Collagen I": "1CAG",    # Collagen triple helix
    "MMP-9 (Gelatinase B)": "1L6J",  # MMP-9 catalytic domain
    "MMP-2 (Gelatinase A)": "1CK7",  # MMP-2
}


def _get_known_pdb_id(name: str) -> str:
    """Look up a known PDB ID for common entities."""
    return _KNOWN_PDB_IDS.get(name, "")


def _render_pdb_viewer(pdb_id: str, name: str, height: int) -> bool:
    """Render a 3Dmol.js viewer for a PDB structure."""
    viewer_id = f"pdb-{hashlib.md5(pdb_id.encode()).hexdigest()[:8]}"

    html = f"""
    <div id="{viewer_id}" style="width:100%; height:{height}px; position:relative;
         border-radius: 8px; overflow: hidden; background: #0a0a1a;">
    </div>
    <div style="text-align:center; font-size:0.7rem; opacity:0.5; margin-top:4px;">
        PDB: {pdb_id} | 3Dmol.js
    </div>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <script>
    (function() {{
        var viewer = $3Dmol.createViewer("{viewer_id}", {{
            backgroundColor: "#0a0a1a"
        }});
        jQuery.ajax("{_pdb_url(pdb_id)}", {{
            success: function(data) {{
                viewer.addModel(data, "pdb");
                viewer.setStyle({{}}, {{
                    cartoon: {{
                        color: "spectrum",
                        opacity: 0.9
                    }}
                }});
                viewer.addSurface($3Dmol.SurfaceType.VDW, {{
                    opacity: 0.1,
                    color: "white"
                }});
                viewer.zoomTo();
                viewer.spin("y", 0.5);
                viewer.render();
            }},
            error: function() {{
                document.getElementById("{viewer_id}").innerHTML =
                    '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#666;">' +
                    'Structure loading...<br>PDB: {pdb_id}</div>';
            }}
        }});
    }})();
    </script>
    """
    st.components.v1.html(html, height=height + 30)
    return True


def _pdb_url(pdb_id: str) -> str:
    """Get the RCSB PDB download URL."""
    return f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"


def _render_alphafold(uniprot_id: str, name: str, height: int) -> bool:
    """Render AlphaFold predicted structure."""
    viewer_id = f"af-{hashlib.md5(uniprot_id.encode()).hexdigest()[:8]}"
    af_url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"

    html = f"""
    <div id="{viewer_id}" style="width:100%; height:{height}px; position:relative;
         border-radius: 8px; overflow: hidden; background: #0a0a1a;">
    </div>
    <div style="text-align:center; font-size:0.7rem; opacity:0.5; margin-top:4px;">
        AlphaFold: {uniprot_id} | Predicted Structure
    </div>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <script>
    (function() {{
        var viewer = $3Dmol.createViewer("{viewer_id}", {{
            backgroundColor: "#0a0a1a"
        }});
        jQuery.ajax("{af_url}", {{
            success: function(data) {{
                viewer.addModel(data, "pdb");
                viewer.setStyle({{}}, {{
                    cartoon: {{color: "spectrum", opacity: 0.9}}
                }});
                viewer.zoomTo();
                viewer.spin("y", 0.5);
                viewer.render();
            }},
            error: function() {{
                document.getElementById("{viewer_id}").innerHTML =
                    '<div style="display:flex;align-items:center;justify-content:center;' +
                    'height:100%;color:#666;">No AlphaFold structure available</div>';
            }}
        }});
    }})();
    </script>
    """
    st.components.v1.html(html, height=height + 30)
    return True


def _render_drug_molecule(entity, props: dict, height: int) -> bool:
    """Render drug/metabolite 2D structure from SMILES."""
    smiles = props.get("smiles", "")
    if not smiles:
        return _render_type_icon(entity, height)

    try:
        from cognisom.dashboard.mol_viz import smiles_to_image
        img = smiles_to_image(smiles, size=(300, int(height * 0.8)))
        if img:
            st.image(img, caption=f"SMILES: {smiles[:40]}...", use_container_width=True)
            return True
    except Exception as e:
        log.debug("Drug 2D render failed for %s: %s", entity.name, e)

    return _render_type_icon(entity, height)


def _render_cell_3d(entity, height: int) -> bool:
    """Render anatomically accurate 3D cell with internal structures.

    Each cell type is a composite scene with:
    - Cell membrane (semi-transparent surface with type-specific shape)
    - Nucleus (correct shape: round, kidney, multilobed, eccentric)
    - Organelles (mitochondria, ER, Golgi, granules as appropriate)
    - Surface receptors (type-specific markers as small dots on membrane)
    """
    color = entity.color_rgb if hasattr(entity, "color_rgb") and entity.color_rgb else [0.8, 0.3, 0.3]
    r, g, b = [int(c * 255) for c in color[:3]]
    rng = np.random.RandomState(hash(entity.name) % 2**31)
    name_lower = entity.name.lower()
    morphology = _classify_cell_morphology(name_lower)

    fig = go.Figure()

    # Base sphere parameters
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)

    def sphere(radius=1.0, cx=0, cy=0, cz=0, stretch_z=1.0, noise_level=0.0):
        x = np.outer(np.cos(u), np.sin(v)) * radius + cx
        y = np.outer(np.sin(u), np.sin(v)) * radius + cy
        z = np.outer(np.ones_like(u), np.cos(v)) * radius * stretch_z + cz
        if noise_level > 0:
            x += rng.normal(0, noise_level, x.shape)
            y += rng.normal(0, noise_level, y.shape)
            z += rng.normal(0, noise_level, z.shape)
        return x, y, z

    def add_surface(x, y, z, col, opacity=0.8, lighting=None):
        if isinstance(col, str):
            c1, c2 = col, col
        else:
            c1, c2 = col
        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            colorscale=[[0, c1], [1, c2]],
            showscale=False, opacity=opacity,
            lighting=lighting or dict(ambient=0.35, diffuse=0.65, specular=0.3, roughness=0.5),
        ))

    def add_dots(positions, color, size=3, opacity=0.8):
        if positions:
            px, py, pz = zip(*positions)
            fig.add_trace(go.Scatter3d(
                x=list(px), y=list(py), z=list(pz), mode="markers",
                marker=dict(size=size, color=color, opacity=opacity),
                showlegend=False))

    def add_line(points, color, width=4):
        px, py, pz = zip(*points)
        fig.add_trace(go.Scatter3d(
            x=list(px), y=list(py), z=list(pz), mode="lines",
            line=dict(color=color, width=width), showlegend=False))

    def random_surface_points(n, radius, min_r=0.0):
        """Generate random points on/inside a sphere."""
        pts = []
        for _ in range(n * 3):
            p = rng.normal(0, radius * 0.6, 3)
            dist = np.sqrt(sum(p**2))
            if min_r * radius < dist < radius * 0.95:
                pts.append(tuple(p))
                if len(pts) >= n:
                    break
        return pts

    def surface_receptors(n, radius, color, size=2):
        """Place receptor dots on the cell surface."""
        pts = []
        for _ in range(n):
            th = rng.uniform(0, 2*np.pi)
            ph = rng.uniform(0, np.pi)
            pts.append((
                radius * np.sin(ph) * np.cos(th),
                radius * np.sin(ph) * np.sin(th),
                radius * np.cos(ph),
            ))
        add_dots(pts, color, size=size, opacity=0.9)

    # Colors
    membrane_col = f"rgb({r},{g},{b})"
    membrane_light = f"rgb({min(255,r+40)},{min(255,g+40)},{min(255,b+40)})"
    nuc_dark = f"rgb({max(0,r//3)},{max(0,g//3)},{min(255,b//2+80)})"
    nuc_light = f"rgb({max(0,r//3+30)},{max(0,g//3+30)},{min(255,b//2+100)})"
    mito_col = "rgb(200,80,80)"
    er_col = "rgb(80,130,200)"
    golgi_col = "rgb(220,180,60)"

    # ══════════════════════════════════════════════════════════════
    if morphology == "t_cell":
        # CD8/CD4 T cell: small, smooth, very large nucleus, thin cytoplasm
        # ~30,000 TCR/CD3 complexes on surface, CD8 or CD4 co-receptors
        R = 0.7  # cell radius
        x, y, z = sphere(R, noise_level=0.008)
        add_surface(x, y, z, (membrane_col, membrane_light), opacity=0.5)
        # Large round nucleus (N:C ratio ~0.85)
        nx, ny, nz = sphere(R * 0.6, noise_level=0.005)
        add_surface(nx, ny, nz, (nuc_dark, nuc_light), opacity=0.75)
        # Chromatin texture inside nucleus
        add_dots(random_surface_points(8, R * 0.4), "rgb(40,40,120)", size=2, opacity=0.5)
        # TCR/CD3 on surface (green dots)
        surface_receptors(40, R * 1.02, "rgb(50,220,50)", size=2)
        # A few mitochondria
        add_dots(random_surface_points(5, R * 0.55, min_r=0.6), mito_col, size=3, opacity=0.6)

    elif morphology == "macrophage_m1":
        # Large, irregular, ruffled membrane, pseudopods, kidney-shaped nucleus
        # FcgammaR, TLR4, CD14, MHC-II, CD80/86 on surface
        # Phagocytic vacuoles and phagolysosomes inside
        R = 1.0
        x, y, z = sphere(R, noise_level=0.06)  # Heavy ruffling
        add_surface(x, y, z, (membrane_col, membrane_light), opacity=0.45)
        # Kidney-shaped nucleus (squashed and indented)
        nx, ny, nz = sphere(0.35, cx=0.1, noise_level=0.02, stretch_z=0.6)
        add_surface(nx, ny, nz, (nuc_dark, nuc_light), opacity=0.7)
        # Pseudopods (5-7 extending lamellipodia)
        for _ in range(6):
            th, ph = rng.uniform(0, 2*np.pi), rng.uniform(0.3, 2.8)
            dx = np.sin(ph) * np.cos(th)
            dy = np.sin(ph) * np.sin(th)
            dz = np.cos(ph)
            length = rng.uniform(0.4, 0.8)
            pts = [(dx*(R + t*length) + rng.normal(0, 0.04),
                    dy*(R + t*length) + rng.normal(0, 0.04),
                    dz*(R + t*length) + rng.normal(0, 0.04))
                   for t in np.linspace(0, 1, 8)]
            add_line(pts, membrane_light, width=5)
        # Phagocytic vacuoles (larger spheres inside)
        for _ in range(3):
            vx, vy, vz = rng.normal(0, 0.3, 3)
            if vx**2 + vy**2 + vz**2 < 0.6:
                vsx, vsy, vsz = sphere(0.12, cx=vx, cy=vy, cz=vz)
                add_surface(vsx, vsy, vsz, "rgb(180,200,180)", opacity=0.4)
        # Surface receptors: TLR4 (yellow), MHC-II (green), FcgammaR (blue)
        surface_receptors(20, R * 1.03, "rgb(230,200,50)", size=2)  # TLR4
        surface_receptors(15, R * 1.04, "rgb(50,200,50)", size=2)    # MHC-II
        surface_receptors(10, R * 1.05, "rgb(80,80,230)", size=2)    # FcgammaR
        # Mitochondria scattered
        add_dots(random_surface_points(8, R * 0.7, min_r=0.3), mito_col, size=3, opacity=0.5)

    elif morphology == "macrophage_m2":
        # Elongated spindle, smoother, fewer pseudopods
        R = 0.9
        x, y, z = sphere(0.5, stretch_z=1.8, noise_level=0.02)
        add_surface(x, y, z, (membrane_col, membrane_light), opacity=0.5)
        # Elongated nucleus
        nx, ny, nz = sphere(0.2, stretch_z=1.2, noise_level=0.01)
        add_surface(nx, ny, nz, (nuc_dark, nuc_light), opacity=0.7)
        # CD163 (scavenger receptor, pink), CD206 (mannose receptor, green)
        surface_receptors(15, 0.52, "rgb(220,120,150)", size=2)  # CD163
        surface_receptors(12, 0.53, "rgb(100,200,100)", size=2)  # CD206

    elif morphology == "dendritic":
        # Stellate with 8-12 long thin dendrite projections
        # Large MIIC compartments (MHC-II loading vesicles)
        R = 0.5
        x, y, z = sphere(R, noise_level=0.03)
        add_surface(x, y, z, (membrane_col, membrane_light), opacity=0.45)
        # Round nucleus
        nx, ny, nz = sphere(R * 0.5, noise_level=0.01)
        add_surface(nx, ny, nz, (nuc_dark, nuc_light), opacity=0.7)
        # Long dendrite projections (10-12 branching arms)
        for _ in range(11):
            th = rng.uniform(0, 2*np.pi)
            ph = rng.uniform(0.2, 2.9)
            dx = np.sin(ph) * np.cos(th)
            dy = np.sin(ph) * np.sin(th)
            dz = np.cos(ph)
            length = rng.uniform(0.8, 1.4)
            n_pts = 15
            pts = []
            for t in np.linspace(0, 1, n_pts):
                pts.append((
                    dx * (R + t * length) + rng.normal(0, 0.06),
                    dy * (R + t * length) + rng.normal(0, 0.06),
                    dz * (R + t * length) + rng.normal(0, 0.06),
                ))
            add_line(pts, membrane_light, width=3)
            # Veil-like flattening at tips
            add_dots([pts[-1]], membrane_light, size=4, opacity=0.7)
            # Branch point
            if length > 1.0:
                bp = pts[n_pts // 2]
                branch_dir = (rng.normal(0, 0.3), rng.normal(0, 0.3), rng.normal(0, 0.3))
                branch_pts = [(bp[0] + branch_dir[0]*t, bp[1] + branch_dir[1]*t,
                               bp[2] + branch_dir[2]*t) for t in np.linspace(0, 0.4, 5)]
                add_line(branch_pts, membrane_light, width=2)
        # MIIC vesicles (MHC-II loading compartments)
        add_dots(random_surface_points(6, R * 0.4, min_r=0.1), "rgb(50,180,50)", size=4, opacity=0.6)
        # MHC-II on surface
        surface_receptors(25, R * 1.03, "rgb(50,220,50)", size=2)

    elif morphology == "neutrophil":
        # Round cell, MULTILOBED NUCLEUS (3-5 connected lobes, THE defining feature)
        # Three granule types visible
        R = 0.85
        x, y, z = sphere(R, noise_level=0.015)
        add_surface(x, y, z, (membrane_col, membrane_light), opacity=0.5)
        # Multilobed nucleus — 4 lobes connected by thin chromatin bridges
        lobe_positions = [(-0.2, 0.15, 0.1), (0.05, 0.15, -0.05),
                          (0.2, -0.1, 0.1), (-0.05, -0.15, -0.1)]
        for lx, ly, lz in lobe_positions:
            lsx, lsy, lsz = sphere(0.17, cx=lx, cy=ly, cz=lz, noise_level=0.01)
            add_surface(lsx, lsy, lsz, (nuc_dark, nuc_light), opacity=0.75)
        # Chromatin bridges between lobes
        for i in range(len(lobe_positions) - 1):
            add_line([lobe_positions[i], lobe_positions[i+1]], nuc_dark, width=3)
        # Primary (azurophilic) granules — large, dark purple
        add_dots(random_surface_points(12, R * 0.65, min_r=0.25),
                 "rgb(120,60,140)", size=4, opacity=0.7)
        # Secondary (specific) granules — smaller, lighter
        add_dots(random_surface_points(20, R * 0.7, min_r=0.2),
                 "rgb(180,160,200)", size=2, opacity=0.5)
        # Tertiary (gelatinase) granules — smallest
        add_dots(random_surface_points(15, R * 0.7, min_r=0.2),
                 "rgb(200,190,210)", size=1.5, opacity=0.4)
        # Surface: PSGL-1, LFA-1, Mac-1
        surface_receptors(20, R * 1.02, "rgb(200,200,50)", size=1.5)

    elif morphology == "nk_cell":
        # Larger than T cell, "large granular lymphocyte"
        # Distinctive perforin/granzyme granules (azurophilic)
        R = 0.8
        x, y, z = sphere(R, noise_level=0.02)
        add_surface(x, y, z, (membrane_col, membrane_light), opacity=0.5)
        # Round nucleus (less dominant than T cell, more cytoplasm)
        nx, ny, nz = sphere(R * 0.45, noise_level=0.008)
        add_surface(nx, ny, nz, (nuc_dark, nuc_light), opacity=0.7)
        # Large perforin/granzyme granules — THE distinguishing feature
        granule_pts = random_surface_points(18, R * 0.65, min_r=0.3)
        add_dots(granule_pts, "rgb(200,50,50)", size=5, opacity=0.8)
        # KIR receptors (inhibitory, orange), NKG2D (activating, green)
        surface_receptors(15, R * 1.02, "rgb(220,160,40)", size=2)  # KIR
        surface_receptors(12, R * 1.03, "rgb(50,200,100)", size=2)  # NKG2D
        # Mitochondria
        add_dots(random_surface_points(6, R * 0.55, min_r=0.3), mito_col, size=3, opacity=0.5)

    elif morphology == "mast_cell":
        # Round, PACKED with metachromatic granules (100-200 per cell)
        # FcepsilonRI on surface
        R = 0.85
        x, y, z = sphere(R, noise_level=0.01)
        add_surface(x, y, z, (membrane_col, membrane_light), opacity=0.4)  # More transparent to show granules
        # Small central nucleus (hard to see through granules)
        nx, ny, nz = sphere(R * 0.25, noise_level=0.005)
        add_surface(nx, ny, nz, (nuc_dark, nuc_light), opacity=0.5)
        # DENSE metachromatic granules (purple/violet) — fill the cytoplasm
        granule_pts = random_surface_points(60, R * 0.75, min_r=0.15)
        add_dots(granule_pts, "rgb(140,40,160)", size=5, opacity=0.85)
        # Smaller granules
        add_dots(random_surface_points(30, R * 0.7, min_r=0.1),
                 "rgb(160,60,180)", size=3, opacity=0.7)
        # FcepsilonRI (IgE receptor, yellow)
        surface_receptors(25, R * 1.02, "rgb(230,200,50)", size=2)

    elif morphology == "plasma_cell":
        # Oval, ECCENTRIC nucleus with clock-face chromatin
        # Massive expanded rough ER filling cytoplasm
        # Clear perinuclear halo (Golgi)
        R = 0.8
        x, y, z = sphere(R, stretch_z=1.15, noise_level=0.01)
        add_surface(x, y, z, (membrane_col, membrane_light), opacity=0.5)
        # Eccentric nucleus (pushed to one side)
        nx, ny, nz = sphere(0.28, cx=0.3, cy=0, cz=0.1, noise_level=0.008)
        add_surface(nx, ny, nz, (nuc_dark, nuc_light), opacity=0.75)
        # Clock-face chromatin pattern (radial dark spots in nucleus)
        for angle in np.linspace(0, 2*np.pi, 6, endpoint=False):
            add_dots([(0.3 + 0.15*np.cos(angle), 0.15*np.sin(angle), 0.1)],
                     "rgb(20,20,80)", size=3, opacity=0.6)
        # Perinuclear Golgi halo (gold arc near nucleus)
        golgi_t = np.linspace(-1, 1, 15)
        golgi_pts = [(0.3 + 0.35*np.cos(t), 0.35*np.sin(t), 0.1) for t in golgi_t]
        add_line(golgi_pts, golgi_col, width=4)
        # Massive rough ER — parallel lamellae filling opposite side from nucleus
        for layer in range(6):
            er_t = np.linspace(-1.2, 1.2, 20)
            er_x = np.full_like(er_t, -0.15 - layer * 0.08)
            er_y = 0.4 * np.sin(er_t + layer * 0.3) + rng.normal(0, 0.02, 20)
            er_z = er_t * 0.5
            er_pts = list(zip(er_x, er_y, er_z))
            add_line(er_pts, er_col, width=2)
            # Ribosomes on ER (tiny dots along ER)
            for j in range(0, 20, 3):
                add_dots([(er_x[j], er_y[j] + 0.03, er_z[j])],
                         "rgb(60,60,60)", size=1, opacity=0.4)

    elif morphology == "eosinophil":
        # BILOBED nucleus (always exactly 2 lobes, connected)
        # Large eosinophilic (bright pink/red) granules with crystalline core
        R = 0.85
        x, y, z = sphere(R, noise_level=0.012)
        add_surface(x, y, z, (membrane_col, membrane_light), opacity=0.5)
        # Bilobed nucleus — TWO distinct lobes with thin bridge
        for offset in [-0.18, 0.18]:
            lx, ly, lz = sphere(0.2, cx=offset, noise_level=0.008)
            add_surface(lx, ly, lz, (nuc_dark, nuc_light), opacity=0.75)
        add_line([(-0.18, 0, 0), (0.18, 0, 0)], nuc_dark, width=3)  # Bridge
        # Large eosinophilic granules (bright salmon-red with crystalloid core)
        granule_pts = random_surface_points(30, R * 0.65, min_r=0.2)
        add_dots(granule_pts, "rgb(230,100,80)", size=5, opacity=0.85)
        # Crystalline cores inside larger granules (darker centers)
        add_dots(granule_pts[:15], "rgb(180,60,40)", size=2, opacity=0.9)

    else:
        # Default cell
        R = 0.8
        x, y, z = sphere(R, noise_level=0.02)
        add_surface(x, y, z, (membrane_col, membrane_light), opacity=0.5)
        nx, ny, nz = sphere(R * 0.4, noise_level=0.01)
        add_surface(nx, ny, nz, (nuc_dark, nuc_light), opacity=0.65)
        add_dots(random_surface_points(5, R * 0.55, min_r=0.3), mito_col, size=3, opacity=0.5)

    # Caption
    scale = entity.scale_um if hasattr(entity, "scale_um") and entity.scale_um else ""
    mesh = entity.mesh_type if hasattr(entity, "mesh_type") and entity.mesh_type else morphology
    caption = mesh.replace("_", " ").title()
    if scale:
        caption += f" | ~{scale} um"

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            bgcolor="#0a0a1a",
            camera=dict(eye=dict(x=2.0, y=2.0, z=1.2)),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=0, b=25),
        height=height,
        paper_bgcolor="#0a0a1a",
        annotations=[dict(
            text=caption, x=0.5, y=0, xref="paper", yref="paper",
            showarrow=False, font=dict(size=10, color="rgba(255,255,255,0.4)"),
        )],
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    return True


def _classify_cell_morphology(name_lower: str) -> str:
    """Classify cell morphology from entity name."""
    if "macrophage m1" in name_lower or "macrophage_m1" in name_lower:
        return "macrophage_m1"
    if "macrophage m2" in name_lower or "macrophage_m2" in name_lower:
        return "macrophage_m2"
    if "macrophage" in name_lower:
        return "macrophage_m1"
    if "dendritic" in name_lower or "cdc" in name_lower or "pdc" in name_lower:
        return "dendritic"
    if "neutrophil" in name_lower:
        return "neutrophil"
    if "nk cell" in name_lower or "natural killer" in name_lower:
        return "nk_cell"
    if "nkt" in name_lower:
        return "nk_cell"
    if "mast cell" in name_lower:
        return "mast_cell"
    if "eosinophil" in name_lower:
        return "eosinophil"
    if "basophil" in name_lower:
        return "mast_cell"
    if "plasma cell" in name_lower or "plasma_cell" in name_lower:
        return "plasma_cell"
    if "t cell" in name_lower or "th1" in name_lower or "th2" in name_lower or \
       "th17" in name_lower or "treg" in name_lower or "tfh" in name_lower or \
       "cd8" in name_lower or "cd4" in name_lower or "gamma-delta" in name_lower:
        return "t_cell"
    if "b cell" in name_lower or "naive b" in name_lower or "memory b" in name_lower:
        return "t_cell"
    if "ilc" in name_lower:
        return "t_cell"
    if "cancer" in name_lower or "stem cell" in name_lower:
        return "macrophage_m1"
    if "fibroblast" in name_lower:
        return "macrophage_m2"
    if "endothelial" in name_lower:
        return "macrophage_m2"
    if "epithelial" in name_lower or "luminal" in name_lower or "basal" in name_lower:
        return "plasma_cell"
    if "neuroendocrine" in name_lower:
        return "mast_cell"
    return "default"


def _render_virus_3d(entity, height: int) -> bool:
    """Render virus with morphology-specific features."""
    import plotly.graph_objects as go
    import numpy as np

    color = entity.color_rgb if hasattr(entity, "color_rgb") and entity.color_rgb else [0.8, 0.1, 0.8]
    r, g, b = [int(c * 255) for c in color[:3]]
    rng = np.random.RandomState(hash(entity.name) % 2**31)
    name_lower = entity.name.lower()

    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    fig = go.Figure()

    # Determine if enveloped (corona-like spikes) or non-enveloped (faceted)
    is_enveloped = any(k in name_lower for k in
                       ["corona", "sars", "influenza", "hiv", "ebv", "herpes",
                        "hepatitis b", "zika", "measles", "rabies"])

    if is_enveloped:
        # Lipid bilayer envelope (smooth, slightly noisy)
        env_noise = rng.normal(0, 0.02, x.shape)
        ex, ey, ez = x + env_noise, y + env_noise, z + env_noise
        fig.add_trace(go.Surface(
            x=ex, y=ey, z=ez,
            colorscale=[[0, f"rgb({r},{g},{b})"],
                         [1, f"rgb({min(255,r+40)},{min(255,g+40)},{min(255,b+40)})"]],
            showscale=False, opacity=0.6,
            lighting=dict(ambient=0.3, diffuse=0.7, specular=0.4),
        ))
        # Inner capsid
        fig.add_trace(go.Surface(
            x=x * 0.7, y=y * 0.7, z=z * 0.7,
            colorscale=[[0, f"rgb({max(0,r-40)},{max(0,g-40)},{max(0,b-40)})"],
                         [1, f"rgb({r},{g},{b})"]],
            showscale=False, opacity=0.4,
        ))
        # Spike glycoproteins
        n_spikes = 40 if "corona" in name_lower or "sars" in name_lower else 20
        spike_len = 0.35 if "corona" in name_lower else 0.2
        for _ in range(n_spikes):
            theta = rng.uniform(0, 2 * np.pi)
            phi_s = rng.uniform(0, np.pi)
            sx = np.sin(phi_s) * np.cos(theta)
            sy = np.sin(phi_s) * np.sin(theta)
            sz = np.cos(phi_s)
            # Spike stalk
            fig.add_trace(go.Scatter3d(
                x=[sx, sx * (1 + spike_len)],
                y=[sy, sy * (1 + spike_len)],
                z=[sz, sz * (1 + spike_len)],
                mode="lines",
                line=dict(color=f"rgb({min(255,r+80)},{min(255,g+80)},{min(255,b+80)})", width=3),
                showlegend=False))
            # Spike tip (bulb for corona)
            if "corona" in name_lower or "sars" in name_lower:
                fig.add_trace(go.Scatter3d(
                    x=[sx * (1 + spike_len)], y=[sy * (1 + spike_len)], z=[sz * (1 + spike_len)],
                    mode="markers",
                    marker=dict(size=3, color=f"rgb({min(255,r+100)},{min(255,g+100)},{min(255,b+100)})"),
                    showlegend=False))
    else:
        # Non-enveloped: faceted icosahedral capsid
        facets = rng.normal(0, 0.06, x.shape)
        fig.add_trace(go.Surface(
            x=x + facets, y=y + facets, z=z + facets,
            colorscale=[[0, f"rgb({r},{g},{b})"],
                         [1, f"rgb({min(255,r+60)},{min(255,g+60)},{min(255,b+60)})"]],
            showscale=False, opacity=0.9,
            lighting=dict(ambient=0.3, diffuse=0.7, specular=0.5),
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            bgcolor="#0a0a1a",
            camera=dict(eye=dict(x=2.2, y=2.2, z=1.5)),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=height,
        paper_bgcolor="#0a0a1a",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    return True


def _render_bacterium_3d(entity, height: int) -> bool:
    """Render species-specific bacterium with anatomical features."""
    color = entity.color_rgb if hasattr(entity, "color_rgb") and entity.color_rgb else [0.6, 0.8, 0.2]
    r, g, b = [int(c * 255) for c in color[:3]]
    col = f"rgb({r},{g},{b})"
    light = f"rgb({min(255,r+40)},{min(255,g+40)},{min(255,b+40)})"
    rng = np.random.RandomState(hash(entity.name) % 2**31)
    name_lower = entity.name.lower()
    props = entity.to_dict().get("properties", {})
    shape = props.get("shape", "rod")

    fig = go.Figure()
    u = np.linspace(0, 2 * np.pi, 35)
    v = np.linspace(0, np.pi, 18)

    def capsule(rx=0.4, rz=1.2, cx=0, cy=0, cz=0):
        x = np.outer(np.cos(u), np.sin(v)) * rx + cx
        y = np.outer(np.sin(u), np.sin(v)) * rx + cy
        z = np.outer(np.ones_like(u), np.cos(v)) * rz + cz
        return x, y, z

    def coccus(radius=0.5, cx=0, cy=0, cz=0):
        x = np.outer(np.cos(u), np.sin(v)) * radius + cx
        y = np.outer(np.sin(u), np.sin(v)) * radius + cy
        z = np.outer(np.ones_like(u), np.cos(v)) * radius + cz
        return x, y, z

    def add_surf(x, y, z, opacity=0.85):
        fig.add_trace(go.Surface(
            x=x, y=y, z=z, colorscale=[[0, col], [1, light]],
            showscale=False, opacity=opacity,
            lighting=dict(ambient=0.4, diffuse=0.6, specular=0.3)))

    def add_flagella(base_x, base_y, base_z, n=1):
        for _ in range(n):
            t = np.linspace(0, 4*np.pi, 40)
            fx = base_x + t * 0.02 + rng.normal(0, 0.01, 40)
            fy = base_y + np.sin(t) * 0.15
            fz = base_z - t * 0.08
            fig.add_trace(go.Scatter3d(
                x=fx, y=fy, z=fz, mode="lines",
                line=dict(color="rgb(180,180,120)", width=2), showlegend=False))

    if "staphylococcus" in name_lower or "aureus" in name_lower:
        # Grape-like CLUSTERS of cocci (staphylo = grape)
        positions = [(0, 0, 0), (0.55, 0.3, 0), (-0.3, 0.5, 0.2),
                     (0.2, -0.4, 0.3), (-0.4, -0.2, -0.1), (0.1, 0.2, 0.5),
                     (-0.2, 0.1, -0.4), (0.4, -0.1, -0.3)]
        for px, py, pz in positions:
            x, y, z = coccus(0.28, px, py, pz)
            noise = rng.normal(0, 0.008, x.shape)
            add_surf(x + noise, y + noise, z + noise, 0.9)

    elif "streptococcus" in name_lower or "pyogenes" in name_lower:
        # CHAINS of cocci (strepto = chain)
        for i in range(7):
            x, y, z = coccus(0.22, cx=0, cy=0, cz=i * 0.48 - 1.4)
            add_surf(x, y, z, 0.9)
        # M-protein fibrils extending from surface
        for i in range(7):
            for _ in range(4):
                th = rng.uniform(0, 2*np.pi)
                bx, by, bz = 0.22*np.cos(th), 0.22*np.sin(th), i*0.48 - 1.4
                fig.add_trace(go.Scatter3d(
                    x=[bx, bx + 0.15*np.cos(th)], y=[by, by + 0.15*np.sin(th)], z=[bz, bz],
                    mode="lines", line=dict(color="rgb(255,200,100)", width=1.5), showlegend=False))

    elif "neisseria" in name_lower or "meningitidis" in name_lower:
        # DIPLOCOCCI (paired kidney-bean shaped cocci)
        for offset in [-0.25, 0.25]:
            x, y, z = coccus(0.35, cx=offset)
            # Slightly flatten adjacent sides
            mask = (x - offset) * np.sign(offset) < 0
            x[mask] *= 0.85
            add_surf(x, y, z, 0.9)
        # Polysaccharide capsule (transparent outer shell)
        cx, cy, cz = coccus(0.7)
        fig.add_trace(go.Surface(
            x=cx, y=cy, z=cz, colorscale=[[0, "rgb(200,200,255)"], [1, "rgb(220,220,255)"]],
            showscale=False, opacity=0.15))

    elif "clostridioides" in name_lower or "difficile" in name_lower:
        # Rod with TERMINAL SPORE (bulging end)
        x, y, z = capsule(0.35, 1.0)
        add_surf(x, y, z, 0.85)
        # Terminal endospore (bright, refractive)
        sx, sy, sz = coccus(0.3, cz=1.2)
        fig.add_trace(go.Surface(
            x=sx, y=sy, z=sz,
            colorscale=[[0, "rgb(240,240,200)"], [1, "rgb(255,255,220)"]],
            showscale=False, opacity=0.95,
            lighting=dict(ambient=0.5, specular=0.8)))

    elif "mycobacterium" in name_lower or "tuberculosis" in name_lower:
        # Slightly curved rod, WAXY CELL WALL (thick, lipid-rich)
        x, y, z = capsule(0.3, 1.0)
        # Slight curvature
        x += np.outer(np.ones_like(u), np.cos(v)) * 0.1
        add_surf(x, y, z, 0.9)
        # Thick waxy cell wall layer (outer translucent)
        wx, wy, wz = capsule(0.38, 1.08)
        wx += np.outer(np.ones_like(u), np.cos(v)) * 0.1
        fig.add_trace(go.Surface(
            x=wx, y=wy, z=wz,
            colorscale=[[0, "rgb(180,130,100)"], [1, "rgb(200,160,120)"]],
            showscale=False, opacity=0.2))

    elif "pseudomonas" in name_lower:
        # Rod with SINGLE POLAR FLAGELLUM + pili
        x, y, z = capsule(0.35, 1.0)
        add_surf(x, y, z, 0.85)
        add_flagella(0, 0, -1.0, n=1)
        # Type IV pili (thin, short, all around)
        for _ in range(8):
            th = rng.uniform(0, 2*np.pi)
            pz = rng.uniform(-0.8, 0.8)
            bx, by = 0.35*np.cos(th), 0.35*np.sin(th)
            fig.add_trace(go.Scatter3d(
                x=[bx, bx + 0.2*np.cos(th)], y=[by, by + 0.2*np.sin(th)], z=[pz, pz],
                mode="lines", line=dict(color="rgb(150,200,150)", width=1), showlegend=False))

    elif "escherichia" in name_lower or "salmonella" in name_lower:
        # Rod with PERITRICHOUS FLAGELLA (multiple, all around)
        x, y, z = capsule(0.35, 1.0)
        noise = rng.normal(0, 0.008, x.shape)
        add_surf(x + noise, y + noise, z + noise, 0.85)
        # Multiple flagella from all over
        for _ in range(6):
            th = rng.uniform(0, 2*np.pi)
            fz = rng.uniform(-0.5, 0.5)
            bx, by = 0.35*np.cos(th), 0.35*np.sin(th)
            add_flagella(bx, by, fz)
        # Fimbriae (short, hair-like)
        for _ in range(15):
            th = rng.uniform(0, 2*np.pi)
            fz = rng.uniform(-0.8, 0.8)
            bx, by = 0.35*np.cos(th), 0.35*np.sin(th)
            fig.add_trace(go.Scatter3d(
                x=[bx, bx + 0.12*np.cos(th)], y=[by, by + 0.12*np.sin(th)], z=[fz, fz],
                mode="lines", line=dict(color=light, width=1), showlegend=False))

    else:
        # Generic rod
        x, y, z = capsule(0.35, 1.0)
        add_surf(x, y, z, 0.85)

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            bgcolor="#0a0a1a", camera=dict(eye=dict(x=2.5, y=2.5, z=1.5)),
            aspectmode="data"),
        margin=dict(l=0, r=0, t=0, b=0), height=height, paper_bgcolor="#0a0a1a")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    return True


def _render_antibody_3d(entity, height: int) -> bool:
    """Render isotype-specific antibody structure."""
    color = entity.color_rgb if hasattr(entity, "color_rgb") and entity.color_rgb else [0.2, 0.4, 0.9]
    r, g, b = [int(c * 255) for c in color[:3]]
    col = f"rgb({r},{g},{b})"
    light = f"rgb({min(255,r+60)},{min(255,g+60)},{min(255,b+60)})"
    name_lower = entity.name.lower()

    fig = go.Figure()

    def add_ig_monomer(cx=0, cz=0, hinge_flex=0.3):
        """Draw one IgG-like monomer: 2 Fab arms + Fc stem + hinge."""
        # Fc stem (2 CH2-CH3 domains)
        for dx in [-0.12, 0.12]:
            fig.add_trace(go.Scatter3d(
                x=[cx+dx, cx+dx], y=[0, 0], z=[cz-0.8, cz],
                mode="lines", line=dict(color=col, width=8), showlegend=False))
        # Hinge region
        fig.add_trace(go.Scatter3d(
            x=[cx], y=[0], z=[cz],
            mode="markers", marker=dict(size=4, color="gold"), showlegend=False))
        # Left Fab (VL-CL + VH-CH1)
        fig.add_trace(go.Scatter3d(
            x=[cx, cx-0.5], y=[0, 0], z=[cz, cz+0.6],
            mode="lines", line=dict(color=col, width=7), showlegend=False))
        fig.add_trace(go.Scatter3d(
            x=[cx-0.5], y=[0], z=[cz+0.6],
            mode="markers", marker=dict(size=6, color=light), showlegend=False))
        # Right Fab
        fig.add_trace(go.Scatter3d(
            x=[cx, cx+0.5], y=[0, 0], z=[cz, cz+0.6],
            mode="lines", line=dict(color=col, width=7), showlegend=False))
        fig.add_trace(go.Scatter3d(
            x=[cx+0.5], y=[0], z=[cz+0.6],
            mode="markers", marker=dict(size=6, color=light), showlegend=False))
        # Fc glycosylation (small orange dots at CH2)
        fig.add_trace(go.Scatter3d(
            x=[cx-0.08, cx+0.08], y=[0, 0], z=[cz-0.3, cz-0.3],
            mode="markers", marker=dict(size=3, color="orange"), showlegend=False))

    if "igm" in name_lower:
        # PENTAMERIC IgM — 5 monomers arranged in a star around J-chain
        for angle in np.linspace(0, 2*np.pi, 5, endpoint=False):
            cx = 1.2 * np.cos(angle)
            cy = 1.2 * np.sin(angle)
            # Simplified: Fc points inward, Fab outward
            fig.add_trace(go.Scatter3d(
                x=[0, cx*0.5], y=[0, cy*0.5], z=[0, 0],
                mode="lines", line=dict(color=col, width=6), showlegend=False))
            fig.add_trace(go.Scatter3d(
                x=[cx*0.5, cx*0.8, cx], y=[cy*0.5, cy*0.8-0.15, cy-0.15],
                z=[0, 0.3, 0.3],
                mode="lines", line=dict(color=col, width=5), showlegend=False))
            fig.add_trace(go.Scatter3d(
                x=[cx*0.5, cx*0.8, cx], y=[cy*0.5, cy*0.8+0.15, cy+0.15],
                z=[0, 0.3, 0.3],
                mode="lines", line=dict(color=col, width=5), showlegend=False))
            # Antigen-binding tips
            fig.add_trace(go.Scatter3d(
                x=[cx, cx], y=[cy-0.15, cy+0.15], z=[0.3, 0.3],
                mode="markers", marker=dict(size=4, color=light), showlegend=False))
        # J-chain at center
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode="markers", marker=dict(size=6, color="gold"), showlegend=False))

    elif "ige" in name_lower:
        # IgE: no hinge, extra CH domain, bound to FcepsilonRI
        add_ig_monomer()
        # Extra CH4 domain (IgE has 4 CH domains, no hinge)
        fig.add_trace(go.Scatter3d(
            x=[-0.12, 0.12], y=[0, 0], z=[-1.0, -1.0],
            mode="markers", marker=dict(size=5, color=col), showlegend=False))
        # FcepsilonRI receptor (orange Y below Fc)
        fig.add_trace(go.Scatter3d(
            x=[0, -0.3, 0, 0.3], y=[0, 0, 0, 0], z=[-1.0, -1.4, -1.0, -1.4],
            mode="lines+markers",
            line=dict(color="rgb(220,140,40)", width=5),
            marker=dict(size=4, color="rgb(220,140,40)"),
            showlegend=False))

    elif "igg3" in name_lower:
        # IgG3: VERY LONG HINGE (62 amino acids, 11 disulfide bonds)
        # Fc
        for dx in [-0.12, 0.12]:
            fig.add_trace(go.Scatter3d(
                x=[dx, dx], y=[0, 0], z=[-0.8, -0.2],
                mode="lines", line=dict(color=col, width=8), showlegend=False))
        # Extended hinge (the distinguishing feature)
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[0, 0], z=[-0.2, 0.5],
            mode="lines", line=dict(color="gold", width=3), showlegend=False))
        # Multiple disulfide bonds along hinge
        for hz in np.linspace(-0.1, 0.4, 6):
            fig.add_trace(go.Scatter3d(
                x=[0], y=[0], z=[hz],
                mode="markers", marker=dict(size=2, color="gold"), showlegend=False))
        # Fab arms (more spread due to long hinge)
        for dx, sign in [(-0.7, -1), (0.7, 1)]:
            fig.add_trace(go.Scatter3d(
                x=[0, dx], y=[0, 0], z=[0.5, 1.0],
                mode="lines", line=dict(color=col, width=7), showlegend=False))
            fig.add_trace(go.Scatter3d(
                x=[dx], y=[0], z=[1.0],
                mode="markers", marker=dict(size=6, color=light), showlegend=False))

    elif "iga" in name_lower:
        # Secretory IgA: DIMER with J-chain + secretory component
        add_ig_monomer(cx=-0.6)
        add_ig_monomer(cx=0.6)
        # J-chain connecting Fc regions
        fig.add_trace(go.Scatter3d(
            x=[-0.6, 0.6], y=[0, 0], z=[-0.8, -0.8],
            mode="lines", line=dict(color="gold", width=4), showlegend=False))
        # Secretory component (wrapping around)
        sc_t = np.linspace(-0.8, 0.8, 20)
        sc_x = sc_t
        sc_y = np.sin(sc_t * 3) * 0.15
        sc_z = np.full_like(sc_t, -0.9)
        fig.add_trace(go.Scatter3d(
            x=sc_x, y=sc_y, z=sc_z, mode="lines",
            line=dict(color="rgb(180,220,180)", width=3), showlegend=False))

    else:
        # Standard IgG1/IgG2/IgG4/IgD monomer
        add_ig_monomer()

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            bgcolor="#0a0a1a", camera=dict(eye=dict(x=0, y=3, z=0.5)),
            aspectmode="data"),
        margin=dict(l=0, r=0, t=0, b=0), height=height, paper_bgcolor="#0a0a1a")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    return True


def _render_type_icon(entity, height: int) -> bool:
    """Render a colored indicator for entity types without specific 3D renderer."""
    color = entity.color_rgb if hasattr(entity, "color_rgb") and entity.color_rgb else [0.5, 0.5, 0.5]
    r, g, b = [int(c * 255) for c in color[:3]]
    scale = entity.scale_um if hasattr(entity, "scale_um") and entity.scale_um else 0
    etype = entity.entity_type.value

    # Type-specific icons
    icons = {
        "pathway": "\U0001f310", "mutation": "\U0001f9ec", "organ": "\U0001fac0",
        "tissue_type": "\U0001f9eb", "mhc": "\U0001f3af", "complement": "\u2b50",
    }
    icon = icons.get(etype, "\U0001f52c")

    st.markdown(
        f'<div style="background: rgba({r},{g},{b},0.1); border: 1px solid rgba({r},{g},{b},0.25); '
        f'border-radius: 12px; padding: 1.2rem; text-align: center; height: {height}px; '
        f'display: flex; flex-direction: column; align-items: center; justify-content: center;">'
        f'<div style="font-size: 3rem; margin-bottom: 0.3rem;">{icon}</div>'
        f'<div style="width: 40px; height: 40px; border-radius: 50%; '
        f'background: linear-gradient(135deg, rgb({r},{g},{b}), rgb({min(255,r+60)},{min(255,g+60)},{min(255,b+60)})); '
        f'margin-bottom: 0.3rem; box-shadow: 0 0 15px rgba({r},{g},{b},0.3);"></div>'
        f'<div style="font-size: 0.75rem; opacity: 0.6;">{etype.replace("_", " ").title()}</div>'
        + (f'<div style="font-size: 0.65rem; opacity: 0.4;">~{scale} um</div>' if scale else '')
        + '</div>',
        unsafe_allow_html=True,
    )
    return False
