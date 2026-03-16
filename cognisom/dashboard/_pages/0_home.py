"""
Home Dashboard
==============

Platform overview with key metrics, component status, and architecture.
This is the default landing page after login.
"""

import os
from pathlib import Path

import streamlit as st
from cognisom.dashboard.page_config import safe_set_page_config

safe_set_page_config(page_title="Cognisom HDT Platform", page_icon="\U0001f9ec", layout="wide")

from cognisom.auth.middleware import streamlit_page_gate
user = streamlit_page_gate("0_home")

# ── Beta Disclaimer ──────────────────────────────────────────
st.markdown("""
<div style="background: linear-gradient(135deg, rgba(234,88,12,0.15), rgba(234,88,12,0.05));
            border: 1px solid rgba(234,88,12,0.35); border-radius: 8px;
            padding: 0.6rem 1rem; margin-bottom: 1rem;">
    <span style="color: #f97316; font-weight: 600; font-size: 0.85rem;">BETA</span>
    <span style="font-size: 0.8rem; opacity: 0.7;">
        &mdash; This platform is in beta testing. Results are for research purposes only
        and must not be used for clinical decision-making. Features may change without notice.
    </span>
</div>
""", unsafe_allow_html=True)

st.title("Cognisom HDT Platform")
st.markdown("**Personalized Molecular Digital Twin Platform for Precision Oncology**")


def _check_api_key(name, env_var):
    val = os.environ.get(env_var, "")
    if not val:
        env_path = Path(__file__).resolve().parent.parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith(f"{env_var}="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
    return bool(val) and not val.startswith("your-")


def _check_module(name, import_path):
    try:
        __import__(import_path)
        return True
    except ImportError:
        return False


# Key metrics
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Simulation Modules", "9",
            help="Cellular, Immune, Vascular, Lymphatic, Epigenetic, Circadian, Morphogen, Molecular, Receptor")
col2.metric("NIM Endpoints", "11",
            help="MolMIM, GenMol, RFdiffusion, ProteinMPNN, DiffDock, ESM2, OpenFold3, Boltz-2, Evo2, AlphaFold2-Multimer, MSA-Search")
col3.metric("Bio-USD Types", "16",
            help="9 IsA prim schemas + 5 applied API schemas + BioScene + BioExosome")

# Entity library stats
try:
    from cognisom.library.store import EntityStore
    _lib_store = EntityStore()
    _lib_stats = _lib_store.stats()
    col4.metric("Entity Library", _lib_stats["total_entities"],
                help="Genes, proteins, drugs, pathways, cell types, metabolites, mutations, receptors, tissues, organs")
    col5.metric("Relationships", _lib_stats["total_relationships"],
                help="Typed edges: binds_to, activates, inhibits, encodes, regulates, etc.")
except Exception:
    col4.metric("Entity Library", "0", help="Seed the entity library from the Entity Library page")
    col5.metric("Relationships", "0")

st.divider()

# Quick start guide
st.subheader("Getting Started")
st.markdown("""
**Digital Twin Pipeline** — Upload patient DNA to get personalized treatment predictions:

1. **Genomic Profile** — Upload a VCF file or load the synthetic demo patient
2. **Immune Landscape** — Analyze the tumor microenvironment at single-cell resolution
3. **Treatment Simulator** — Compare 9 therapy regimens including neoantigen vaccine
4. **Neoantigen Vaccine** — Design a personalized mRNA vaccine from predicted neoantigens
""")

st.divider()

# Platform components
st.subheader("Platform Components")
comp1, comp2 = st.columns(2)

with comp1:
    st.markdown("##### Core Engine")
    for name, pkg in [("NumPy", "numpy"), ("SciPy", "scipy"), ("Scanpy", "scanpy"),
                      ("AnnData", "anndata"), ("Flask", "flask"), ("Plotly", "plotly")]:
        ok = _check_module(name, pkg)
        st.markdown(f"- {'`OK`' if ok else '`MISSING`'} {name}")

    st.markdown("##### GPU Acceleration")
    for name, pkg in [("RAPIDS-singlecell", "rapids_singlecell"),
                      ("CuPy", "cupy"), ("PyTorch", "torch")]:
        ok = _check_module(name, pkg)
        st.markdown(f"- {'`OK`' if ok else '`N/A`'} {name}")

with comp2:
    st.markdown("##### API Keys")
    for name, env in [("NVIDIA NIM API", "NVIDIA_API_KEY"),
                      ("NGC Container Registry", "NGC_API_KEY")]:
        ok = _check_api_key(name, env)
        st.markdown(f"- {'`CONFIGURED`' if ok else '`NOT SET`'} {name}")

    st.markdown("##### NIM Microservices")
    for n in ["MolMIM (Molecule Generation)", "GenMol (Fragment-Based Generation)",
              "RFdiffusion (Protein Design)", "ProteinMPNN (Sequence Design)",
              "DiffDock (Molecular Docking)", "ESM2-650M (Protein Embeddings)",
              "OpenFold3 (Structure Prediction)", "Boltz-2 (Complex Prediction)",
              "Evo2 (Genomic Foundation Model)", "AlphaFold2-Multimer (Multi-chain)",
              "MSA-Search (Sequence Alignment)"]:
        st.markdown(f"- `READY` {n}")

st.divider()

# Architecture
st.subheader("Architecture")
st.code("""
PATIENT DATA
  VCF (Tumor DNA) --> Variant Annotator --> Cancer Driver ID (14 genes) --+
  scRNA-seq ---------> Cell Archetypes --> Immune Profiling ---------------+
  Spatial Transcriptomics --> Tissue Map --> TME Characterization ----------+
                                                                           |
DIGITAL TWIN                                                               v
  Genomic Profile + Immune Landscape --> Personalized Digital Twin
    --> Treatment Simulation (9 regimens) --> RECIST + Survival + irAE

AI LAYER (11 NVIDIA BioNeMo NIMs)
  MolMIM / GenMol / RFdiffusion / ProteinMPNN / DiffDock / ESM2
  OpenFold3 / Boltz-2 / AlphaFold2 / Evo2 / MSA-Search

SIMULATION (9 Modules) --> 3D Visualization --> Dashboard (you are here)
KNOWLEDGE: Bio-USD (16 types) + Entity Library (99 entities)
RESEARCH: 22 sources + AI Agent <-- PubMed / bioRxiv / arXiv
""", language=None)

st.divider()

# Data directory
st.subheader("Data Directory")
data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "scrna"
if data_dir.exists():
    h5ad_files = list(data_dir.glob("*.h5ad"))
    if h5ad_files:
        for f in h5ad_files:
            st.markdown(f"- `{f.name}` ({f.stat().st_size / 1e6:.1f} MB)")
    else:
        st.info("No .h5ad datasets yet. Use the **Data Ingestion** page to run the pipeline.")
else:
    st.info("No data directory yet. Use the **Data Ingestion** page to generate synthetic data.")
