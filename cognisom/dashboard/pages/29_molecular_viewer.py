"""
Page 29: Molecular Viewer
=========================

Interactive 3D protein structure visualization.

Default renderer: 3Dmol.js (WebGL) — works everywhere, no GPU server needed.
Optional upgrade: Kit RTX (MJPEG stream) — when Isaac Sim container is running.

Loads protein structures from:
- Patient profile (Page 26) — mutant proteins from VCF data
- BioNeMo NIM predictions (AlphaFold2, OpenFold3)
- Direct PDB upload or RCSB PDB fetch
- DiffDock docking results

Phase 4 of the Molecular Digital Twin pipeline.
"""

import streamlit as st
import json
import logging
import io

st.set_page_config(page_title="Molecular Viewer", page_icon="🔬", layout="wide")

from cognisom.auth.middleware import streamlit_page_gate
user = streamlit_page_gate("29_molecular_viewer")

logger = logging.getLogger(__name__)

st.title("🔬 Molecular Viewer")
st.caption(
    "Visualize protein structures, mutation impacts, and drug docking "
    "in RTX-rendered 3D. Powered by OpenUSD + Kit."
)

# ─────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────

KIT_BASE_URL = "http://host.docker.internal:8600"  # Server-side (container→container)
KIT_PUBLIC_URL = "/kit"  # Client-side (browser→nginx→Kit, HTTPS)

# Sample PDB for demo (small beta-hairpin)
DEMO_PDB = """HEADER    DEMO PROTEIN
ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00 85.00           N
ATOM      2  CA  ALA A   1      11.458  10.000  10.000  1.00 90.00           C
ATOM      3  C   ALA A   1      12.009  11.420  10.000  1.00 88.00           C
ATOM      4  O   ALA A   1      11.251  12.393  10.000  1.00 85.00           O
ATOM      5  N   ARG A   2      13.323  11.506  10.000  1.00 92.00           N
ATOM      6  CA  ARG A   2      14.019  12.791  10.000  1.00 95.00           C
ATOM      7  C   ARG A   2      15.537  12.665  10.000  1.00 93.00           C
ATOM      8  O   ARG A   2      16.098  11.570  10.000  1.00 90.00           O
ATOM      9  N   GLY A   3      16.194  13.819  10.000  1.00 88.00           N
ATOM     10  CA  GLY A   3      17.648  13.851  10.000  1.00 85.00           C
ATOM     11  C   GLY A   3      18.282  12.472  10.000  1.00 82.00           C
ATOM     12  O   GLY A   3      17.591  11.458  10.000  1.00 80.00           O
ATOM     13  N   ASP A   4      19.606  12.442   9.900  1.00 78.00           N
ATOM     14  CA  ASP A   4      20.355  11.198   9.800  1.00 75.00           C
ATOM     15  C   ASP A   4      21.855  11.421   9.800  1.00 72.00           C
ATOM     16  O   ASP A   4      22.306  12.566   9.900  1.00 70.00           O
ATOM     17  N   LEU A   5      22.615  10.333   9.700  1.00 68.00           N
ATOM     18  CA  LEU A   5      24.069  10.407   9.700  1.00 65.00           C
ATOM     19  C   LEU A   5      24.668  11.792   9.800  1.00 62.00           C
ATOM     20  O   LEU A   5      24.005  12.832   9.900  1.00 60.00           O
ATOM     21  N   PHE A   6      25.992  11.803   9.750  1.00 58.00           N
ATOM     22  CA  PHE A   6      26.722  13.063   9.800  1.00 55.00           C
ATOM     23  C   PHE A   6      26.050  14.194  10.580  1.00 52.00           C
ATOM     24  O   PHE A   6      26.660  15.253  10.780  1.00 50.00           O
ATOM     25  N   TRP A   7      24.800  13.984  11.020  1.00 48.00           N
ATOM     26  CA  TRP A   7      24.025  15.005  11.720  1.00 45.00           C
ATOM     27  C   TRP A   7      22.558  14.610  11.650  1.00 42.00           C
ATOM     28  O   TRP A   7      22.226  13.426  11.600  1.00 40.00           O
ATOM     29  N   LYS A   8      21.678  15.614  11.650  1.00 82.00           N
ATOM     30  CA  LYS A   8      20.237  15.389  11.600  1.00 85.00           C
ATOM     31  C   LYS A   8      19.507  16.679  11.260  1.00 88.00           C
ATOM     32  O   LYS A   8      20.098  17.760  11.200  1.00 90.00           O
END
"""


def send_to_kit(endpoint: str, data: dict) -> dict:
    """Send a request to the Kit streaming server."""
    import requests
    try:
        resp = requests.post(
            f"{KIT_BASE_URL}/cognisom/molecular/{endpoint}",
            json=data,
            timeout=10,
        )
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def get_kit_status() -> dict:
    """Get Kit server status."""
    import requests
    try:
        resp = requests.get(f"{KIT_BASE_URL}/status", timeout=5)
        return resp.json()
    except Exception:
        return None


def _fetch_rcsb(pdb_id: str) -> str:
    """Fetch PDB file from RCSB."""
    import requests
    pdb_id = pdb_id.strip().upper()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            st.success(f"Fetched {pdb_id} from RCSB ({len(resp.text)} bytes)")
            return resp.text
        else:
            st.error(f"Could not fetch {pdb_id}: HTTP {resp.status_code}")
            return None
    except Exception as e:
        st.error(f"RCSB fetch failed: {e}")
        return None


def _render_3d_viewer(pdb_text: str, mutations: list, mode: str = "ribbon",
                      color_mode: str = "bfactor"):
    """Render interactive 3D protein structure using 3Dmol.js.

    Embeds a full WebGL molecular viewer in the browser — no Kit required.
    Supports ribbon, ball-and-stick, and surface modes with pLDDT coloring
    and mutation highlighting.
    """
    import streamlit.components.v1 as components

    # Escape PDB text for safe embedding in JavaScript
    pdb_escaped = json.dumps(pdb_text)
    mutations_js = json.dumps(mutations)

    # Map mode names to 3Dmol.js style functions
    style_map = {
        "ribbon": "cartoon",
        "ball_and_stick": "stick",
        "surface": "cartoon",  # show cartoon + surface overlay
    }
    style_name = style_map.get(mode, "cartoon")

    html = f"""<!DOCTYPE html>
<html>
<head>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
  * {{ margin: 0; padding: 0; }}
  body {{ background: #0a0a1e; overflow: hidden; }}
  #viewer {{ width: 100%; height: 580px; position: relative; }}
  #mode-badge {{
    position: absolute; top: 8px; right: 8px; z-index: 10;
    background: rgba(0,160,255,0.85); color: white;
    padding: 3px 10px; border-radius: 4px; font: 11px monospace;
  }}
</style>
</head>
<body>
<div id="viewer">
  <div id="mode-badge">3Dmol.js WebGL</div>
</div>
<script>
(function() {{
  var viewer = $3Dmol.createViewer("viewer", {{
    backgroundColor: "0x0a0a1e",
    antialias: true,
  }});

  var pdbData = {pdb_escaped};
  var mutations = {mutations_js};
  var styleName = "{style_name}";
  var colorMode = "{color_mode}";
  var showSurface = {"true" if mode == "surface" else "false"};

  viewer.addModel(pdbData, "pdb");

  // --- Color scheme based on mode ---
  if (colorMode === "bfactor") {{
    // pLDDT confidence coloring (AlphaFold convention)
    viewer.setStyle({{}}, {{
      {style_name}: {{
        colorfunc: function(atom) {{
          var b = atom.b;
          if (b >= 90) return 0x3333e6;  // blue: very high
          if (b >= 70) return 0x33cccc;  // cyan: high
          if (b >= 50) return 0xe6cc33;  // yellow: low
          return 0xe63333;               // red: very low
        }}
      }}
    }});
  }} else if (colorMode === "element") {{
    viewer.setStyle({{}}, {{ {style_name}: {{ colorscheme: "Jmol" }} }});
  }} else if (colorMode === "chain") {{
    viewer.setStyle({{}}, {{ {style_name}: {{ colorscheme: "chain" }} }});
  }} else {{
    viewer.setStyle({{}}, {{ {style_name}: {{ color: "spectrum" }} }});
  }}

  // --- Surface overlay (transparent) ---
  if (showSurface) {{
    var surfOpts = {{ opacity: 0.35 }};
    if (colorMode === "bfactor") {{
      surfOpts.colorfunc = function(atom) {{
        var b = atom.b;
        if (b >= 90) return 0x3333e6;
        if (b >= 70) return 0x33cccc;
        if (b >= 50) return 0xe6cc33;
        return 0xe63333;
      }};
    }} else if (colorMode === "element") {{
      surfOpts.colorscheme = "Jmol";
    }} else {{
      surfOpts.colorscheme = "chain";
    }}
    viewer.addSurface($3Dmol.SurfaceType.VDW, surfOpts);
  }}

  // --- Highlight mutations (red spheres + labels) ---
  if (mutations.length > 0) {{
    for (var i = 0; i < mutations.length; i++) {{
      var resi = mutations[i];
      viewer.addStyle({{ resi: resi }}, {{
        sphere: {{ radius: 0.8, color: "red", opacity: 0.7 }}
      }});
      viewer.addResLabels({{ resi: resi }}, {{
        font: "Arial", fontSize: 12, fontColor: "white",
        backgroundColor: "rgba(200,0,0,0.7)", backgroundOpacity: 0.7,
        showBackground: true,
      }});
    }}
  }}

  viewer.zoomTo();
  viewer.spin("y", 0.5);  // Slow auto-rotation
  viewer.render();
}})();
</script>
</body>
</html>"""

    components.html(html, height=600)


# ─────────────────────────────────────────────────────────────────────────
# SIDEBAR: SOURCE SELECTION
# ─────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Structure Source")

    source = st.radio(
        "Load protein from:",
        ["Patient Profile", "Upload PDB", "RCSB PDB ID", "Demo"],
        index=3,
    )

    st.divider()
    st.header("Visualization")

    viz_mode = st.selectbox(
        "Rendering mode",
        ["ribbon", "ball_and_stick", "surface"],
        format_func=lambda x: x.replace("_", " ").title(),
    )

    color_mode = st.selectbox(
        "Color by",
        ["bfactor", "element", "chain"],
        format_func=lambda x: {
            "bfactor": "Confidence (pLDDT)",
            "element": "Element (CPK)",
            "chain": "Chain",
        }.get(x, x),
    )

    highlight_mutations = st.checkbox("Highlight mutations", value=True)

    st.divider()

    # Kit connection status
    kit_status = get_kit_status()
    if kit_status and kit_status.get("status") == "ok":
        renderer = kit_status.get("renderer", "unknown")
        st.success(f"Kit RTX connected ({renderer})")
        mol_loaded = kit_status.get("molecular_loaded", False)
        mol_built = kit_status.get("molecular_scene_built", False)
        if mol_loaded:
            st.info(f"Scene: {'Built' if mol_built else 'Building...'} "
                    f"| Mode: {kit_status.get('molecular_mode', '?')}")
    else:
        st.caption("Kit RTX: offline (3Dmol.js active)")


# ─────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────

pdb_text = None
mutations = []
title = ""

# ── Source: Patient Profile (from Page 26) ──
if source == "Patient Profile":
    profile = st.session_state.get("patient_profile")
    if profile is None:
        st.info(
            "No patient profile loaded. Go to **Page 26: Genomic Twin** first, "
            "or switch to Demo mode."
        )
    else:
        st.subheader("Patient Proteins")
        st.caption(f"Patient: {profile.patient_id} | Cancer: {profile.cancer_type}")

        # List affected proteins with structure prediction options
        drivers = profile.cancer_driver_mutations
        if drivers:
            gene_options = list({v.gene for v in drivers if v.gene})
            selected_gene = st.selectbox("Select gene", gene_options)

            # Get mutations for selected gene
            gene_mutations = [v for v in drivers if v.gene == selected_gene]
            for v in gene_mutations:
                if v.protein_change:
                    # Parse residue number from protein change (e.g. "T877A" → 877)
                    try:
                        res_num = int("".join(c for c in v.protein_change[1:-1] if c.isdigit()))
                        mutations.append(res_num)
                    except ValueError:
                        pass

            st.write(f"**{selected_gene}** mutations: "
                     f"{', '.join(v.protein_change or '?' for v in gene_mutations)}")

            # Check if we have predicted structure in session state
            pred_key = f"predicted_pdb_{selected_gene}"
            if pred_key in st.session_state:
                pdb_text = st.session_state[pred_key]
                title = f"{selected_gene} (predicted)"
                st.success(f"Using predicted structure for {selected_gene}")
            else:
                st.info(
                    f"No predicted structure for {selected_gene}. "
                    f"Use Page 26 to predict via AlphaFold2 NIM, or upload PDB below."
                )

                # Offer to fetch from RCSB
                col1, col2 = st.columns(2)
                with col1:
                    pdb_id = st.text_input(
                        "RCSB PDB ID (if known)",
                        placeholder="e.g. 1XOW for AR",
                    )
                    if pdb_id and st.button("Fetch from RCSB"):
                        pdb_text = _fetch_rcsb(pdb_id)
                        if pdb_text:
                            title = f"{selected_gene} ({pdb_id})"
                with col2:
                    uploaded = st.file_uploader(
                        "Or upload PDB file",
                        type=["pdb", "ent"],
                        key="patient_pdb_upload",
                    )
                    if uploaded:
                        pdb_text = uploaded.getvalue().decode("utf-8")
                        title = f"{selected_gene} (uploaded)"
        else:
            st.warning("No cancer driver mutations found in profile")

# ── Source: Upload PDB ──
elif source == "Upload PDB":
    uploaded = st.file_uploader("Upload PDB file", type=["pdb", "ent"])
    if uploaded:
        pdb_text = uploaded.getvalue().decode("utf-8")
        title = uploaded.name

        # Parse mutation residues from user input
        mut_input = st.text_input(
            "Highlight mutation residues (comma-separated numbers)",
            placeholder="e.g. 877, 702",
        )
        if mut_input:
            mutations = [int(x.strip()) for x in mut_input.split(",") if x.strip().isdigit()]

# ── Source: RCSB PDB ──
elif source == "RCSB PDB ID":
    pdb_id = st.text_input("PDB ID", placeholder="e.g. 1XOW, 3ZOV, 2AXA")
    if pdb_id:
        pdb_text = _fetch_rcsb(pdb_id) if 'fetch' not in st.session_state else None
        # Button to trigger fetch
        if st.button("Fetch Structure"):
            pdb_text = _fetch_rcsb(pdb_id)
            if pdb_text:
                st.session_state["fetched_pdb"] = pdb_text
                title = f"RCSB: {pdb_id.upper()}"

        if "fetched_pdb" in st.session_state:
            pdb_text = st.session_state["fetched_pdb"]

        mut_input = st.text_input(
            "Highlight mutation residues",
            placeholder="e.g. 877, 702",
        )
        if mut_input:
            mutations = [int(x.strip()) for x in mut_input.split(",") if x.strip().isdigit()]

# ── Source: Demo ──
elif source == "Demo":
    st.info("Using demo 8-residue peptide with simulated pLDDT scores")
    pdb_text = DEMO_PDB
    title = "Demo Peptide"
    mutations = []


# ─────────────────────────────────────────────────────────────────────────
# STRUCTURE INFO & SEND TO KIT
# ─────────────────────────────────────────────────────────────────────────

if pdb_text:
    # Parse basic stats
    atom_count = sum(1 for line in pdb_text.splitlines()
                     if line.startswith(("ATOM", "HETATM")))
    chain_set = set()
    residue_set = set()
    for line in pdb_text.splitlines():
        if line.startswith("ATOM"):
            chain_set.add(line[21:22].strip())
            residue_set.add((line[21:22].strip(), line[22:26].strip()))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Atoms", f"{atom_count:,}")
    col2.metric("Residues", f"{len(residue_set):,}")
    col3.metric("Chains", len(chain_set))
    col4.metric("Mutations", len(mutations))

    st.divider()

    # ── Visualization Area ──
    view_col, info_col = st.columns([3, 1])

    kit_available = kit_status and kit_status.get("status") == "ok"

    with view_col:
        st.subheader("3D Structure")

        # Viewer mode toggle: 3Dmol.js (always available) vs Kit RTX (when available)
        viewer_options = ["3Dmol.js (WebGL)"]
        if kit_available:
            viewer_options.append("Kit RTX (MJPEG)")
        active_viewer = st.radio(
            "Renderer",
            viewer_options,
            horizontal=True,
            label_visibility="collapsed",
        )

        if active_viewer == "Kit RTX (MJPEG)" and kit_available:
            # ── Kit RTX mode ──
            if st.button("Send to Kit RTX", type="primary"):
                result = send_to_kit("load", {
                    "pdb_text": pdb_text,
                    "mode": viz_mode,
                    "color_mode": color_mode,
                    "mutations": mutations if highlight_mutations else [],
                    "title": title,
                })
                if "error" in result:
                    st.error(f"Kit error: {result['error']}")
                else:
                    st.success(result.get("message", "Sent to Kit"))

            # MJPEG viewer
            st.markdown(
                f'<iframe src="{KIT_PUBLIC_URL}/stream" '
                f'width="100%" height="600" '
                f'style="border:1px solid #333; border-radius:8px;" '
                f'></iframe>',
                unsafe_allow_html=True,
            )

            # Camera controls
            cam_col1, cam_col2, cam_col3 = st.columns(3)
            with cam_col1:
                if st.button("Reset Camera"):
                    import requests as _req
                    try:
                        _req.post(f"{KIT_BASE_URL}/cognisom/camera/reset",
                                  timeout=5)
                    except Exception:
                        pass
            with cam_col2:
                if st.button("Orbit Left"):
                    import requests as _req
                    try:
                        _req.post(
                            f"{KIT_BASE_URL}/cognisom/camera/orbit",
                            json={"yaw": -15, "pitch": 0}, timeout=5,
                        )
                    except Exception:
                        pass
            with cam_col3:
                if st.button("Orbit Right"):
                    import requests as _req
                    try:
                        _req.post(
                            f"{KIT_BASE_URL}/cognisom/camera/orbit",
                            json={"yaw": 15, "pitch": 0}, timeout=5,
                        )
                    except Exception:
                        pass
        else:
            # ── 3Dmol.js WebGL mode (default, always works) ──
            _render_3d_viewer(
                pdb_text,
                mutations if highlight_mutations else [],
                mode=viz_mode,
                color_mode=color_mode,
            )

    with info_col:
        st.subheader("Structure Info")
        st.write(f"**Title:** {title}")
        st.write(f"**Mode:** {viz_mode.replace('_', ' ').title()}")
        st.write(f"**Color:** {color_mode}")

        if kit_available:
            renderer = kit_status.get("renderer", "unknown")
            st.caption(f"Kit RTX available ({renderer})")

        if mutations:
            st.write("**Highlighted residues:**")
            for m in mutations:
                st.write(f"  - Residue {m}")

        # Color legend
        st.divider()
        st.caption("**pLDDT Color Scale** (if color=Confidence)")
        st.markdown(
            '<div style="display:flex;gap:4px;align-items:center">'
            '<span style="background:#3333e6;width:16px;height:16px;display:inline-block;border-radius:2px"></span> 90-100 Very high'
            '</div>'
            '<div style="display:flex;gap:4px;align-items:center">'
            '<span style="background:#33cccc;width:16px;height:16px;display:inline-block;border-radius:2px"></span> 70-90 High'
            '</div>'
            '<div style="display:flex;gap:4px;align-items:center">'
            '<span style="background:#e6cc33;width:16px;height:16px;display:inline-block;border-radius:2px"></span> 50-70 Low'
            '</div>'
            '<div style="display:flex;gap:4px;align-items:center">'
            '<span style="background:#e63333;width:16px;height:16px;display:inline-block;border-radius:2px"></span> <50 Very low'
            '</div>',
            unsafe_allow_html=True,
        )

        if highlight_mutations and mutations:
            st.markdown(
                '<div style="display:flex;gap:4px;align-items:center;margin-top:8px">'
                '<span style="background:#ff0000;width:16px;height:16px;display:inline-block;border-radius:2px"></span> Mutation site'
                '</div>',
                unsafe_allow_html=True,
            )

    # ── PDB Text Preview ──
    with st.expander("PDB Text (first 50 lines)"):
        lines = pdb_text.splitlines()[:50]
        st.code("\n".join(lines), language="text")

else:
    st.info("Select a structure source from the sidebar to get started.")


