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
    """Render morphology-specific 3D cell using Plotly.

    Each cell type gets a distinct shape based on real morphology:
    - T/B cells: smooth small spheres with large nucleus
    - Macrophage M1: ruffled surface with pseudopods
    - Macrophage M2: elongated spindle shape
    - Dendritic cell: stellate with dendrite extensions
    - Neutrophil: multilobed nucleus
    - NK cell: slightly irregular with granules
    - Mast cell: round packed with granules
    - Eosinophil: bilobed nucleus, granular
    - Plasma cell: eccentric nucleus, expanded ER
    """
    import plotly.graph_objects as go
    import numpy as np

    color = entity.color_rgb if hasattr(entity, "color_rgb") and entity.color_rgb else [0.8, 0.3, 0.3]
    r, g, b = [int(c * 255) for c in color[:3]]
    cell_color = f"rgb({r},{g},{b})"
    light_color = f"rgb({min(255,r+50)},{min(255,g+50)},{min(255,b+50)})"
    dark_color = f"rgb({max(0,r-80)},{max(0,g-80)},{max(0,b-80)})"
    nuc_color = f"rgb({max(0,r-60)},{max(0,g-60)},{min(255,b+40)})"

    rng = np.random.RandomState(hash(entity.name) % 2**31)
    name_lower = entity.name.lower()
    fig = go.Figure()

    # Determine morphology from name
    morphology = _classify_cell_morphology(name_lower)

    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    cos_u, sin_u = np.cos(u), np.sin(u)
    cos_v, sin_v = np.cos(v), np.sin(v)

    if morphology == "t_cell":
        # Small smooth sphere (7-8 um), large nucleus ratio
        x = np.outer(cos_u, sin_v) * 0.7
        y = np.outer(sin_u, sin_v) * 0.7
        z = np.outer(np.ones_like(u), cos_v) * 0.7
        noise = rng.normal(0, 0.01, x.shape)
        x += noise; y += noise; z += noise
        _add_cell_body(fig, x, y, z, cell_color, light_color, 0.85)
        _add_nucleus(fig, x * 0.55, y * 0.55, z * 0.55, nuc_color, 0.7)

    elif morphology == "macrophage_m1":
        # Large ruffled cell with pseudopods extending outward
        x = np.outer(cos_u, sin_v)
        y = np.outer(sin_u, sin_v)
        z = np.outer(np.ones_like(u), cos_v)
        # Heavy surface ruffling
        ruffle = rng.normal(0, 0.08, x.shape)
        x += ruffle; y += ruffle; z += ruffle
        _add_cell_body(fig, x, y, z, cell_color, light_color, 0.8)
        _add_nucleus(fig, x * 0.35, y * 0.35, z * 0.35, nuc_color, 0.6)
        # Pseudopods (extending protrusions)
        for _ in range(5):
            theta, phi = rng.uniform(0, 2*np.pi), rng.uniform(0.3, 2.8)
            dx, dy, dz = np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)
            length = rng.uniform(0.3, 0.6)
            t = np.linspace(0, 1, 8)
            px = dx * (1 + t * length) + rng.normal(0, 0.05, 8)
            py = dy * (1 + t * length) + rng.normal(0, 0.05, 8)
            pz = dz * (1 + t * length) + rng.normal(0, 0.05, 8)
            fig.add_trace(go.Scatter3d(
                x=px, y=py, z=pz, mode="lines",
                line=dict(color=cell_color, width=6), showlegend=False))

    elif morphology == "macrophage_m2":
        # Elongated spindle shape
        x = np.outer(cos_u, sin_v) * 0.6
        y = np.outer(sin_u, sin_v) * 0.6
        z = np.outer(np.ones_like(u), cos_v) * 1.4  # stretched along z
        noise = rng.normal(0, 0.02, x.shape)
        x += noise; y += noise
        _add_cell_body(fig, x, y, z, cell_color, light_color, 0.85)
        _add_nucleus(fig, x * 0.4, y * 0.4, z * 0.3, nuc_color, 0.6)

    elif morphology == "dendritic":
        # Stellate shape with long dendrite extensions
        x = np.outer(cos_u, sin_v) * 0.6
        y = np.outer(sin_u, sin_v) * 0.6
        z = np.outer(np.ones_like(u), cos_v) * 0.6
        noise = rng.normal(0, 0.04, x.shape)
        x += noise; y += noise; z += noise
        _add_cell_body(fig, x, y, z, cell_color, light_color, 0.75)
        _add_nucleus(fig, x * 0.45, y * 0.45, z * 0.45, nuc_color, 0.6)
        # Long thin dendrites (8-12 extending branches)
        for _ in range(10):
            theta = rng.uniform(0, 2*np.pi)
            phi = rng.uniform(0.2, 2.9)
            dx, dy, dz = np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)
            length = rng.uniform(0.6, 1.2)
            n_pts = 12
            t = np.linspace(0, 1, n_pts)
            # Branching wiggle
            px = dx * (0.6 + t * length) + rng.normal(0, 0.08, n_pts)
            py = dy * (0.6 + t * length) + rng.normal(0, 0.08, n_pts)
            pz = dz * (0.6 + t * length) + rng.normal(0, 0.08, n_pts)
            fig.add_trace(go.Scatter3d(
                x=px, y=py, z=pz, mode="lines",
                line=dict(color=light_color, width=3), showlegend=False))
            # Tip bulge
            fig.add_trace(go.Scatter3d(
                x=[px[-1]], y=[py[-1]], z=[pz[-1]], mode="markers",
                marker=dict(size=3, color=light_color), showlegend=False))

    elif morphology == "neutrophil":
        # Medium cell with multilobed nucleus (3-5 lobes)
        x = np.outer(cos_u, sin_v) * 0.85
        y = np.outer(sin_u, sin_v) * 0.85
        z = np.outer(np.ones_like(u), cos_v) * 0.85
        noise = rng.normal(0, 0.02, x.shape)
        x += noise; y += noise; z += noise
        _add_cell_body(fig, x, y, z, cell_color, light_color, 0.8)
        # Multilobed nucleus — 4 connected small spheres
        nuc_scale = 0.2
        offsets = [(-0.15, 0, 0.1), (0.05, 0.1, 0), (0.15, -0.05, -0.1), (-0.05, -0.1, 0.05)]
        for ox, oy, oz in offsets:
            nx = x * nuc_scale + ox
            ny = y * nuc_scale + oy
            nz = z * nuc_scale + oz
            fig.add_trace(go.Surface(
                x=nx, y=ny, z=nz,
                colorscale=[[0, nuc_color], [1, dark_color]],
                showscale=False, opacity=0.7))
        # Granules (small dots in cytoplasm)
        for _ in range(20):
            gx, gy, gz = rng.normal(0, 0.3, 3)
            if gx**2 + gy**2 + gz**2 < 0.5:
                fig.add_trace(go.Scatter3d(
                    x=[gx], y=[gy], z=[gz], mode="markers",
                    marker=dict(size=2, color="rgb(200,180,220)", opacity=0.6),
                    showlegend=False))

    elif morphology == "nk_cell":
        # Medium, slightly irregular with visible granules
        x = np.outer(cos_u, sin_v) * 0.8
        y = np.outer(sin_u, sin_v) * 0.8
        z = np.outer(np.ones_like(u), cos_v) * 0.8
        noise = rng.normal(0, 0.03, x.shape)
        x += noise; y += noise; z += noise
        _add_cell_body(fig, x, y, z, cell_color, light_color, 0.8)
        _add_nucleus(fig, x * 0.4, y * 0.4, z * 0.4, nuc_color, 0.65)
        # Azurophilic granules
        for _ in range(15):
            gx, gy, gz = rng.normal(0, 0.35, 3)
            if 0.1 < gx**2 + gy**2 + gz**2 < 0.5:
                fig.add_trace(go.Scatter3d(
                    x=[gx], y=[gy], z=[gz], mode="markers",
                    marker=dict(size=3, color="rgb(180,50,50)", opacity=0.7),
                    showlegend=False))

    elif morphology == "mast_cell":
        # Round, densely packed with large metachromatic granules
        x = np.outer(cos_u, sin_v) * 0.85
        y = np.outer(sin_u, sin_v) * 0.85
        z = np.outer(np.ones_like(u), cos_v) * 0.85
        _add_cell_body(fig, x, y, z, cell_color, light_color, 0.7)
        _add_nucleus(fig, x * 0.3, y * 0.3, z * 0.3, nuc_color, 0.5)
        # Dense granules filling cytoplasm
        for _ in range(40):
            gx, gy, gz = rng.normal(0, 0.4, 3)
            if 0.05 < gx**2 + gy**2 + gz**2 < 0.6:
                fig.add_trace(go.Scatter3d(
                    x=[gx], y=[gy], z=[gz], mode="markers",
                    marker=dict(size=4, color="rgb(160,50,160)", opacity=0.8),
                    showlegend=False))

    elif morphology == "plasma_cell":
        # Eccentric nucleus, expanded rough ER visible
        x = np.outer(cos_u, sin_v) * 0.8
        y = np.outer(sin_u, sin_v) * 0.8
        z = np.outer(np.ones_like(u), cos_v) * 0.8
        _add_cell_body(fig, x, y, z, cell_color, light_color, 0.8)
        # Eccentric nucleus (offset to one side)
        _add_nucleus(fig, x * 0.35 + 0.25, y * 0.35, z * 0.35, nuc_color, 0.7)
        # ER ribbons (concentric arcs on opposite side of nucleus)
        for i in range(4):
            er_t = np.linspace(-1.5, 1.5, 20)
            er_x = np.cos(er_t) * (0.15 + i * 0.08) - 0.2
            er_y = np.sin(er_t) * (0.15 + i * 0.08)
            er_z = np.zeros_like(er_t) + rng.normal(0, 0.02, 20)
            fig.add_trace(go.Scatter3d(
                x=er_x, y=er_y, z=er_z, mode="lines",
                line=dict(color="rgb(100,150,200)", width=2), showlegend=False))

    elif morphology == "eosinophil":
        # Bilobed nucleus, large pink-red granules
        x = np.outer(cos_u, sin_v) * 0.85
        y = np.outer(sin_u, sin_v) * 0.85
        z = np.outer(np.ones_like(u), cos_v) * 0.85
        noise = rng.normal(0, 0.015, x.shape)
        x += noise; y += noise; z += noise
        _add_cell_body(fig, x, y, z, cell_color, light_color, 0.8)
        # Bilobed nucleus (two connected lobes)
        for offset in [-0.15, 0.15]:
            nx = x * 0.22 + offset
            ny = y * 0.22
            nz = z * 0.22
            fig.add_trace(go.Surface(
                x=nx, y=ny, z=nz,
                colorscale=[[0, nuc_color], [1, dark_color]],
                showscale=False, opacity=0.7))
        # Large eosinophilic (pink-red) granules
        for _ in range(25):
            gx, gy, gz = rng.normal(0, 0.35, 3)
            if 0.08 < gx**2 + gy**2 + gz**2 < 0.55:
                fig.add_trace(go.Scatter3d(
                    x=[gx], y=[gy], z=[gz], mode="markers",
                    marker=dict(size=4, color="rgb(230,120,100)", opacity=0.8),
                    showlegend=False))

    else:
        # Default: generic cell with slight noise
        x = np.outer(cos_u, sin_v) * 0.8
        y = np.outer(sin_u, sin_v) * 0.8
        z = np.outer(np.ones_like(u), cos_v) * 0.8
        noise = rng.normal(0, 0.02, x.shape)
        x += noise; y += noise; z += noise
        _add_cell_body(fig, x, y, z, cell_color, light_color, 0.85)
        _add_nucleus(fig, x * 0.4, y * 0.4, z * 0.4, nuc_color, 0.6)

    # Add caption with morphology info
    scale = entity.scale_um if hasattr(entity, "scale_um") and entity.scale_um else ""
    caption = f"{morphology.replace('_', ' ').title()}"
    if scale:
        caption += f" | ~{scale} um"

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            bgcolor="#0a0a1a",
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.2)),
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
        return "mast_cell"  # similar morphology
    if "plasma cell" in name_lower or "plasma_cell" in name_lower:
        return "plasma_cell"
    if "t cell" in name_lower or "th1" in name_lower or "th2" in name_lower or \
       "th17" in name_lower or "treg" in name_lower or "tfh" in name_lower or \
       "cd8" in name_lower or "cd4" in name_lower or "gamma-delta" in name_lower:
        return "t_cell"
    if "b cell" in name_lower or "naive b" in name_lower or "memory b" in name_lower:
        return "t_cell"  # similar morphology (small round lymphocyte)
    if "ilc" in name_lower:
        return "t_cell"  # ILCs look like lymphocytes
    if "cancer" in name_lower or "stem cell" in name_lower:
        return "macrophage_m1"  # irregular
    if "fibroblast" in name_lower:
        return "macrophage_m2"  # spindle shaped
    if "endothelial" in name_lower:
        return "macrophage_m2"  # flat/elongated
    if "epithelial" in name_lower or "luminal" in name_lower or "basal" in name_lower:
        return "plasma_cell"  # polarized with ER
    if "neuroendocrine" in name_lower:
        return "mast_cell"  # granular
    return "default"


def _add_cell_body(fig, x, y, z, color, light_color, opacity):
    """Add a cell body surface trace."""
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, color], [1, light_color]],
        showscale=False, opacity=opacity,
        lighting=dict(ambient=0.35, diffuse=0.65, specular=0.3, roughness=0.5),
    ))


def _add_nucleus(fig, x, y, z, color, opacity):
    """Add a nucleus surface trace."""
    import plotly.graph_objects as go
    dark = color.replace("rgb(", "").replace(")", "")
    r, g, b = [int(c) for c in dark.split(",")]
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, color], [1, f"rgb({max(0,r-30)},{max(0,g-30)},{max(0,b-30)})"]],
        showscale=False, opacity=opacity,
    ))


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
    """Render bacterium as rod or coccus shape."""
    import plotly.graph_objects as go
    import numpy as np

    color = entity.color_rgb if hasattr(entity, "color_rgb") and entity.color_rgb else [0.6, 0.8, 0.2]
    r, g, b = [int(c * 255) for c in color[:3]]

    props = entity.to_dict().get("properties", {})
    shape = props.get("shape", "rod")

    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)

    if shape == "rod":
        # Capsule shape (elongated sphere)
        x = np.outer(np.cos(u), np.sin(v)) * 0.4
        y = np.outer(np.sin(u), np.sin(v)) * 0.4
        z = np.outer(np.ones(np.size(u)), np.cos(v)) * 1.2
    else:
        # Coccus (sphere)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, f"rgb({r},{g},{b})"],
                     [1, f"rgb({min(255,r+30)},{min(255,g+30)},{min(255,b+30)})"]],
        showscale=False, opacity=0.9,
        lighting=dict(ambient=0.4, diffuse=0.6, specular=0.3),
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            bgcolor="#0a0a1a",
            camera=dict(eye=dict(x=2, y=2, z=1.5)),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=height,
        paper_bgcolor="#0a0a1a",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    return True


def _render_antibody_3d(entity, height: int) -> bool:
    """Render antibody Y-shape."""
    import plotly.graph_objects as go
    import numpy as np

    color = entity.color_rgb if hasattr(entity, "color_rgb") and entity.color_rgb else [0.2, 0.4, 0.9]
    r, g, b = [int(c * 255) for c in color[:3]]
    col = f"rgb({r},{g},{b})"

    # Y-shape from line segments (simplified)
    fig = go.Figure()

    # Fc stem
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[-1, 0],
        mode="lines", line=dict(color=col, width=12), showlegend=False,
    ))
    # Left Fab arm
    fig.add_trace(go.Scatter3d(
        x=[0, -0.8], y=[0, 0], z=[0, 0.8],
        mode="lines", line=dict(color=col, width=10), showlegend=False,
    ))
    # Right Fab arm
    fig.add_trace(go.Scatter3d(
        x=[0, 0.8], y=[0, 0], z=[0, 0.8],
        mode="lines", line=dict(color=col, width=10), showlegend=False,
    ))
    # Antigen-binding tips
    for xp in [-0.8, 0.8]:
        fig.add_trace(go.Scatter3d(
            x=[xp], y=[0], z=[0.8],
            mode="markers",
            marker=dict(size=8, color=f"rgb({min(255,r+60)},{min(255,g+60)},{min(255,b+60)})"),
            showlegend=False,
        ))
    # Fc base
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[-1],
        mode="markers",
        marker=dict(size=10, color=col),
        showlegend=False,
    ))
    # Hinge
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers",
        marker=dict(size=6, color="gold"),
        showlegend=False,
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            bgcolor="#0a0a1a",
            camera=dict(eye=dict(x=0, y=3, z=0.5)),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=height,
        paper_bgcolor="#0a0a1a",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    return True


def _render_type_icon(entity, height: int) -> bool:
    """Render a simple colored indicator for entity types without 3D."""
    color = entity.color_rgb if hasattr(entity, "color_rgb") and entity.color_rgb else [0.5, 0.5, 0.5]
    r, g, b = [int(c * 255) for c in color[:3]]
    scale = entity.scale_um if hasattr(entity, "scale_um") and entity.scale_um else 0

    st.markdown(
        f'<div style="background: rgba({r},{g},{b},0.15); border: 1px solid rgba({r},{g},{b},0.3); '
        f'border-radius: 12px; padding: 1.5rem; text-align: center; height: {height}px; '
        f'display: flex; flex-direction: column; align-items: center; justify-content: center;">'
        f'<div style="width: 60px; height: 60px; border-radius: 50%; '
        f'background: linear-gradient(135deg, rgb({r},{g},{b}), rgb({min(255,r+60)},{min(255,g+60)},{min(255,b+60)})); '
        f'margin-bottom: 0.5rem; box-shadow: 0 0 20px rgba({r},{g},{b},0.3);"></div>'
        f'<div style="font-size: 0.75rem; opacity: 0.6;">{entity.entity_type.value}</div>'
        + (f'<div style="font-size: 0.65rem; opacity: 0.4;">Scale: {scale} um</div>' if scale else '')
        + '</div>',
        unsafe_allow_html=True,
    )
    return False
