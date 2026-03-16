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
    """Render a procedural 3D cell using Plotly."""
    import plotly.graph_objects as go
    import numpy as np

    color = entity.color_rgb if hasattr(entity, "color_rgb") and entity.color_rgb else [0.8, 0.3, 0.3]
    r, g, b = [int(c * 255) for c in color[:3]]
    cell_color = f"rgb({r},{g},{b})"

    # Generate sphere
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    # Add surface noise for realism (microvilli-like bumps)
    noise = np.random.RandomState(hash(entity.name) % 2**31).normal(0, 0.03, x.shape)
    x += noise
    y += noise
    z += noise

    fig = go.Figure()

    # Cell body
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, cell_color], [1, f"rgb({min(255,r+40)},{min(255,g+40)},{min(255,b+40)})"]],
        showscale=False,
        opacity=0.85,
        lighting=dict(ambient=0.4, diffuse=0.6, specular=0.3, roughness=0.5),
    ))

    # Nucleus (smaller, darker sphere inside)
    scale = 0.4
    fig.add_trace(go.Surface(
        x=x * scale, y=y * scale, z=z * scale,
        colorscale=[[0, f"rgb({max(0,r-80)},{max(0,g-80)},{max(0,b-80)})"],
                     [1, f"rgb({max(0,r-40)},{max(0,g-40)},{max(0,b-40)})"]],
        showscale=False,
        opacity=0.6,
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="#0a0a1a",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=height,
        paper_bgcolor="#0a0a1a",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    return True


def _render_virus_3d(entity, height: int) -> bool:
    """Render virus as icosahedral particle."""
    import plotly.graph_objects as go
    import numpy as np

    color = entity.color_rgb if hasattr(entity, "color_rgb") and entity.color_rgb else [0.8, 0.1, 0.8]
    r, g, b = [int(c * 255) for c in color[:3]]

    # Generate icosphere-like surface
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    # Add icosahedral facet pattern
    rng = np.random.RandomState(hash(entity.name) % 2**31)
    facets = rng.normal(0, 0.05, x.shape)
    x += facets

    # Spike proteins (small protrusions)
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, f"rgb({r},{g},{b})"],
                     [1, f"rgb({min(255,r+60)},{min(255,g+60)},{min(255,b+60)})"]],
        showscale=False,
        opacity=0.9,
        lighting=dict(ambient=0.3, diffuse=0.7, specular=0.5),
    ))

    # Add spike-like protrusions
    n_spikes = 30
    for i in range(n_spikes):
        theta = rng.uniform(0, 2 * np.pi)
        phi_s = rng.uniform(0, np.pi)
        sx = np.sin(phi_s) * np.cos(theta)
        sy = np.sin(phi_s) * np.sin(theta)
        sz = np.cos(phi_s)
        fig.add_trace(go.Scatter3d(
            x=[sx, sx * 1.25], y=[sy, sy * 1.25], z=[sz, sz * 1.25],
            mode="lines",
            line=dict(color=f"rgb({min(255,r+80)},{min(255,g+80)},{min(255,b+80)})", width=3),
            showlegend=False,
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
