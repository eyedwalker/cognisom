"""Admin page - Platform configuration, API status, system health."""

import sys
import os
import platform
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st

st.set_page_config(page_title="Admin | Cognisom", page_icon="🧬", layout="wide")

from cognisom.auth.middleware import streamlit_page_gate
user = streamlit_page_gate("4_admin")


def _load_env():
    env_path = Path(__file__).resolve().parent.parent.parent.parent / ".env"
    env_vars = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                env_vars[key.strip()] = val.strip().strip('"').strip("'")
    return env_vars, env_path


st.title("Platform Administration")
st.markdown("System health, API configuration, and platform management.")

# System info
st.subheader("System Information")
s1, s2 = st.columns(2)

with s1:
    st.markdown(f"- **Platform**: {platform.system()} {platform.release()}")
    st.markdown(f"- **Python**: {sys.version.split()[0]}")
    st.markdown(f"- **Architecture**: {platform.machine()}")
    gpu = "Not detected"
    try:
        import torch
        if torch.cuda.is_available():
            gpu = f"{torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            gpu = "Apple Silicon (MPS)"
    except ImportError:
        pass
    st.markdown(f"- **GPU**: {gpu}")

with s2:
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    st.markdown(f"- **Project root**: `{project_root}`")
    data_dir = project_root / "data" / "scrna"
    if data_dir.exists():
        h5 = list(data_dir.glob("*.h5ad"))
        total_mb = sum(f.stat().st_size for f in h5) / 1e6
        st.markdown(f"- **Datasets**: {len(h5)} files ({total_mb:.1f} MB)")
    else:
        st.markdown("- **Datasets**: None")
    for pkg in ["numpy", "scipy", "scanpy", "streamlit", "plotly"]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            st.markdown(f"- **{pkg}**: {ver}")
        except ImportError:
            st.markdown(f"- **{pkg}**: Not installed")

st.divider()

# API keys
st.subheader("API Key Configuration")
env_vars, env_path = _load_env()

with st.expander("Manage API Keys", expanded=True):
    st.markdown(f"Configuration file: `{env_path}`")
    for key_name, env_name in [("NVIDIA NIM API", "NVIDIA_API_KEY"),
                               ("NGC Container Registry", "NGC_API_KEY")]:
        val = env_vars.get(env_name, "")
        if len(val) > 14:
            masked = val[:10] + "..." + val[-4:]
        else:
            masked = val or "NOT SET"
        st.markdown(f"- **{key_name}** (`{env_name}`): `{masked}`")
    st.code("# Edit .env in project root:\nNVIDIA_API_KEY=nvapi-xxxx\nNGC_API_KEY=xxxx", language="bash")

st.divider()

# NIM health
st.subheader("NIM Endpoint Health")
if st.button("Test All Endpoints", use_container_width=True):
    api_key = env_vars.get("NVIDIA_API_KEY", "")
    if not api_key or api_key.startswith("your-"):
        st.warning("Set NVIDIA_API_KEY in .env first.")
    else:
        import requests
        from datetime import datetime
        endpoints = {
            "MolMIM": "https://health.api.nvidia.com/v1/biology/nvidia/molmim/generate",
            "GenMol": "https://health.api.nvidia.com/v1/biology/nvidia/genmol/generate",
            "RFdiffusion": "https://health.api.nvidia.com/v1/biology/ipd/rfdiffusion/generate",
            "ProteinMPNN": "https://health.api.nvidia.com/v1/biology/ipd/proteinmpnn/predict",
            "DiffDock": "https://health.api.nvidia.com/v1/biology/mit/diffdock",
            "ESM2-650M": "https://health.api.nvidia.com/v1/biology/meta/esm2-650m",
            "OpenFold3": "https://health.api.nvidia.com/v1/biology/openfold/openfold3/predict",
            "Boltz-2": "https://health.api.nvidia.com/v1/biology/mit/boltz2/predict",
            "Evo2": "https://health.api.nvidia.com/v1/biology/arc/evo2/generate",
            "AlphaFold2-Multimer": "https://health.api.nvidia.com/v1/biology/deepmind/alphafold2-multimer/predict",
            "MSA-Search": "https://health.api.nvidia.com/v1/biology/colabfold/msa-search",
        }
        results = []
        progress = st.progress(0)
        for i, (name, url) in enumerate(endpoints.items()):
            try:
                t0 = datetime.now()
                r = requests.options(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
                ms = (datetime.now() - t0).total_seconds() * 1000
                results.append({"Endpoint": name, "Status": f"{r.status_code}",
                                "Latency": f"{ms:.0f} ms"})
            except Exception as e:
                results.append({"Endpoint": name, "Status": f"Error: {str(e)[:40]}",
                                "Latency": "N/A"})
            progress.progress((i + 1) / len(endpoints))
        st.dataframe(results, use_container_width=True, hide_index=True)

st.divider()

# Module inventory
st.subheader("Platform Modules")
project_root = Path(__file__).resolve().parent.parent.parent.parent

module_info = [
    ("Core Engine", "cognisom/core/", "Event-driven simulation engine (243K+ events/sec)"),
    ("Simulation Modules", "cognisom/modules/", "9 biological simulation modules"),
    ("NIM Clients", "cognisom/nims/", "11 NVIDIA BioNeMo NIM API wrappers"),
    ("Drug Bridge", "cognisom/bridge/", "NIM-to-simulation translation layer + structure bridge"),
    ("Research Feed", "cognisom/research/", "PubMed / bioRxiv / arXiv innovation feed"),
    ("Research Agent", "cognisom/agent/", "Tool-based agent for gene / mutation / drug target exploration"),
    ("Ingestion Pipeline", "cognisom/ingestion/", "scRNA-seq data loading and archetype extraction"),
    ("REST API", "cognisom/api/", "Flask REST API and report generation"),
    ("Dashboard", "cognisom/dashboard/", "Streamlit admin dashboard (this page)"),
]

for mod_name, mod_path, desc in module_info:
    full = project_root / mod_path
    exists = full.exists()
    n_py = len(list(full.glob("*.py"))) if exists else 0
    with st.expander(f"{'[OK]' if exists else '[--]'} {mod_name} ({n_py} files) - {desc}"):
        if exists:
            for f in sorted(full.glob("*.py")):
                st.markdown(f"- `{f.name}` ({f.stat().st_size / 1024:.1f} KB)")

st.divider()

# Quick commands
st.subheader("Quick Commands")
cmds = {
    "Run ingestion test": "python -m cognisom.ingestion.test_ingestion",
    "Run NIM pipeline test": "python -m cognisom.nims.test_pipeline",
    "Start Flask API": "python -m cognisom.api.rest_server",
    "Launch dashboard": "streamlit run cognisom/dashboard/app.py",
    "Build Docker (GPU)": "docker build -f Dockerfile.gpu -t cognisom:gpu .",
}
for desc, cmd in cmds.items():
    st.markdown(f"**{desc}**")
    st.code(cmd, language="bash")
