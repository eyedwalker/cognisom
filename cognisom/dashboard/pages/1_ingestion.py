"""Ingestion page - scRNA-seq pipeline visualization."""

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Ingestion | Cognisom", page_icon="🧬", layout="wide")

from cognisom.auth.middleware import streamlit_page_gate
user = streamlit_page_gate("1_ingestion")


@st.cache_data(show_spinner="Running ingestion pipeline...")
def run_synthetic_pipeline():
    """Run the full ingestion pipeline on synthetic data (cached)."""
    import anndata as ad
    from scipy.sparse import csr_matrix
    import pandas as pd

    np.random.seed(42)

    cell_types = {
        "luminal_epithelial": 600, "basal_epithelial": 200,
        "stromal_fibroblast": 400, "endothelial": 100,
        "t_cell": 150, "macrophage": 100,
        "smooth_muscle": 150, "nk_cell": 50, "neuroendocrine": 50,
    }
    total_cells = sum(cell_types.values())

    marker_genes = {
        "luminal_epithelial": ["KLK3", "AR", "NKX3-1", "KRT8", "KRT18"],
        "basal_epithelial": ["TP63", "KRT5", "KRT14", "KRT15", "ITGA6"],
        "stromal_fibroblast": ["VIM", "COL1A1", "COL3A1", "DCN", "LUM"],
        "endothelial": ["PECAM1", "VWF", "CDH5", "ERG", "FLT1"],
        "t_cell": ["CD3D", "CD3E", "CD8A", "CD4", "TRAC"],
        "macrophage": ["CD68", "CD163", "CSF1R", "MARCO", "MSR1"],
        "smooth_muscle": ["ACTA2", "MYH11", "TAGLN", "CNN1", "DES"],
        "nk_cell": ["NKG7", "GNLY", "KLRD1", "NCAM1", "FCGR3A"],
        "neuroendocrine": ["CHGA", "SYP", "ENO2", "ASCL1"],
    }

    all_markers = []
    for markers in marker_genes.values():
        all_markers.extend(markers)
    all_markers = list(dict.fromkeys(all_markers))
    noise_genes = [f"GENE_{i}" for i in range(500)]
    all_genes = all_markers + noise_genes
    n_genes = len(all_genes)

    X = np.zeros((total_cells, n_genes), dtype=np.float32)
    cell_labels = []
    row = 0
    for cell_type, n_cells in cell_types.items():
        markers = marker_genes[cell_type]
        for i in range(n_cells):
            X[row, :] = np.random.exponential(0.1, n_genes)
            for gene in markers:
                gene_idx = all_genes.index(gene)
                X[row, gene_idx] = np.random.exponential(3.0) + 2.0
            for gene in all_markers:
                gene_idx = all_genes.index(gene)
                if gene not in markers:
                    X[row, gene_idx] += np.random.exponential(0.2)
            cell_labels.append(cell_type)
            row += 1

    mt_genes = ["MT-CO1", "MT-CO2", "MT-ND1", "MT-ND2", "MT-ATP6"]
    for g in mt_genes:
        all_genes.append(g)
        mt_col = np.random.exponential(0.5, total_cells).reshape(-1, 1)
        X = np.hstack([X, mt_col.astype(np.float32)])

    X = np.round(X).astype(np.float32)
    X = np.clip(X, 0, None)

    obs = pd.DataFrame({"true_cell_type": cell_labels},
                        index=[f"cell_{i}" for i in range(total_cells)])
    var = pd.DataFrame(index=all_genes)
    adata = ad.AnnData(X=csr_matrix(X), obs=obs, var=var)

    from cognisom.ingestion.preprocessor import ScRNAPreprocessor
    preprocessor = ScRNAPreprocessor(use_gpu=False)
    adata = preprocessor.run(adata, min_genes=5, min_cells=3,
                              max_pct_mito=50.0, n_top_genes=200, resolution=0.8)

    from cognisom.ingestion.archetypes import ArchetypeExtractor
    extractor = ArchetypeExtractor()
    archetypes = extractor.extract(adata)

    from cognisom.ingestion.single_cell_bridge import SingleCellBridge
    bridge = SingleCellBridge()
    config = bridge.create_config(archetypes, total_cells=200)

    # Serialize what we need (cache-friendly)
    umap = adata.obsm["X_umap"].tolist()
    leiden = adata.obs["leiden"].tolist()
    true_types = adata.obs["true_cell_type"].tolist()
    var_names = list(adata.var_names)
    n_obs = adata.n_obs
    n_vars = adata.n_vars

    arch_rows = []
    for a in archetypes:
        top = list(a.marker_scores.items())[:3]
        arch_rows.append({
            "cluster_id": a.cluster_id, "name": a.name,
            "cell_count": a.cell_count,
            "proportion": a.proportion,
            "simulation_type": a.simulation_type,
            "top_markers": ", ".join(f"{g}={s:.2f}" for g, s in top),
        })

    cells = [c for c in config.cells]
    immune = [c for c in config.immune_cells]

    # Marker heatmap data
    from cognisom.ingestion.archetypes import PROSTATE_MARKERS
    marker_names = []
    for an in [a.name for a in archetypes]:
        if an in PROSTATE_MARKERS:
            for g in PROSTATE_MARKERS[an]["markers"][:3]:
                if g not in marker_names:
                    marker_names.append(g)
    gene_set = set(var_names)
    present_markers = [g for g in marker_names if g in gene_set]
    heatmap = np.zeros((len(archetypes), len(present_markers)))
    for i, arch in enumerate(archetypes):
        mask = np.array(leiden) == arch.cluster_id
        for j, gene in enumerate(present_markers):
            if gene in gene_set:
                gene_idx = var_names.index(gene)
                cluster_x = adata[mask]
                if hasattr(cluster_x.X, "toarray"):
                    vals = cluster_x.X.toarray()[:, gene_idx]
                else:
                    vals = cluster_x.X[:, gene_idx]
                heatmap[i, j] = float(np.mean(vals))

    heat_labels = [f"{a.cluster_id}: {a.name}" for a in archetypes]

    return {
        "umap": umap, "leiden": leiden, "true_types": true_types,
        "n_obs": n_obs, "n_vars": n_vars, "n_clusters": len(set(leiden)),
        "archetypes": arch_rows, "n_archetypes": len(archetypes),
        "cells": cells, "immune_cells": immune,
        "spatial_fields": len(config.spatial_config.get("fields", {})),
        "heatmap": heatmap.tolist(), "heatmap_x": present_markers,
        "heatmap_y": heat_labels,
    }


# ── Real dataset loader ──────────────────────────────────────────────

@st.cache_data(show_spinner="Downloading real dataset from CellxGene...")
def run_real_pipeline(tissue="prostate gland", disease="prostate adenocarcinoma", max_cells=5000):
    """Load real scRNA-seq data from CellxGene Human Cell Atlas."""
    import anndata as ad
    import pandas as pd

    from cognisom.ingestion.loader import ScRNALoader
    loader = ScRNALoader()

    # Try CellxGene download
    adata = loader.from_cellxgene(tissue=tissue, organism="Homo sapiens",
                                   disease=disease, max_cells=max_cells)
    if adata is None:
        return None

    from cognisom.ingestion.preprocessor import ScRNAPreprocessor
    preprocessor = ScRNAPreprocessor(use_gpu=False)
    adata = preprocessor.run(adata, min_genes=5, min_cells=3,
                              max_pct_mito=50.0, n_top_genes=200, resolution=0.8)

    from cognisom.ingestion.archetypes import ArchetypeExtractor
    extractor = ArchetypeExtractor()
    archetypes = extractor.extract(adata)

    from cognisom.ingestion.single_cell_bridge import SingleCellBridge
    bridge = SingleCellBridge()
    config = bridge.create_config(archetypes, total_cells=min(500, adata.n_obs))

    umap = adata.obsm["X_umap"].tolist()
    leiden = adata.obs["leiden"].tolist()
    true_types = adata.obs.get("cell_type", adata.obs.get("true_cell_type",
                    pd.Series(["unknown"] * adata.n_obs))).tolist()
    var_names = list(adata.var_names)
    n_obs = adata.n_obs
    n_vars = adata.n_vars

    arch_rows = []
    for a in archetypes:
        top = list(a.marker_scores.items())[:3]
        arch_rows.append({
            "cluster_id": a.cluster_id, "name": a.name,
            "cell_count": a.cell_count,
            "proportion": a.proportion,
            "simulation_type": a.simulation_type,
            "top_markers": ", ".join(f"{g}={s:.2f}" for g, s in top),
        })

    cells = [c for c in config.cells]
    immune = [c for c in config.immune_cells]

    from cognisom.ingestion.archetypes import PROSTATE_MARKERS
    marker_names = []
    for an in [a.name for a in archetypes]:
        if an in PROSTATE_MARKERS:
            for g in PROSTATE_MARKERS[an]["markers"][:3]:
                if g not in marker_names:
                    marker_names.append(g)
    gene_set = set(var_names)
    present_markers = [g for g in marker_names if g in gene_set]
    heatmap = np.zeros((len(archetypes), len(present_markers)))
    for i, arch in enumerate(archetypes):
        mask = np.array(leiden) == arch.cluster_id
        for j, gene in enumerate(present_markers):
            if gene in gene_set:
                gene_idx = var_names.index(gene)
                cluster_x = adata[mask]
                if hasattr(cluster_x.X, "toarray"):
                    vals = cluster_x.X.toarray()[:, gene_idx]
                else:
                    vals = cluster_x.X[:, gene_idx]
                heatmap[i, j] = float(np.mean(vals))

    heat_labels = [f"{a.cluster_id}: {a.name}" for a in archetypes]

    return {
        "umap": umap, "leiden": leiden, "true_types": true_types,
        "n_obs": n_obs, "n_vars": n_vars, "n_clusters": len(set(leiden)),
        "archetypes": arch_rows, "n_archetypes": len(archetypes),
        "cells": cells, "immune_cells": immune,
        "spatial_fields": len(config.spatial_config.get("fields", {})),
        "heatmap": heatmap.tolist(), "heatmap_x": present_markers,
        "heatmap_y": heat_labels,
        "source": f"CellxGene: {tissue} / {disease}",
    }


# ── Page content ──

st.title("scRNA-seq Ingestion Pipeline")
st.markdown("Load, preprocess, cluster, and extract cell archetypes from single-cell RNA-seq data.")

# Data source selector
st.sidebar.header("Data Source")
data_source = st.sidebar.radio(
    "Choose data source",
    ["Synthetic (demo)", "CellxGene (real data)"],
    help="Synthetic: 1,800 cells, instant. CellxGene: real human tissue data from Human Cell Atlas.",
)

if data_source == "CellxGene (real data)":
    st.sidebar.markdown("---")
    tissue = st.sidebar.selectbox("Tissue", [
        "prostate gland", "lung", "breast", "colon", "liver", "kidney",
    ])
    disease = st.sidebar.selectbox("Disease", [
        "prostate adenocarcinoma", "normal", "lung adenocarcinoma",
        "breast carcinoma", "colorectal cancer",
    ])
    max_cells = st.sidebar.slider("Max cells", 1000, 20000, 5000, 1000)

run_col1, run_col2 = st.columns([3, 1])
with run_col1:
    if data_source == "Synthetic (demo)":
        st.markdown("**Synthetic prostate tissue dataset** (1,800 cells, 9 cell types)")
    else:
        st.markdown(f"**CellxGene Human Cell Atlas** ({tissue} / {disease}, up to {max_cells:,} cells)")
with run_col2:
    run_btn = st.button("Run Pipeline", type="primary", use_container_width=True)

if run_btn or st.session_state.get("ingestion_ran"):
    st.session_state["ingestion_ran"] = True
    if data_source == "CellxGene (real data)":
        data = run_real_pipeline(tissue=tissue, disease=disease, max_cells=max_cells)
        if data is None:
            st.error("Failed to download dataset from CellxGene. Check your internet connection or try synthetic data.")
            st.stop()
    else:
        data = run_synthetic_pipeline()

    # Metrics
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cells", f"{data['n_obs']:,}")
    m2.metric("Genes", f"{data['n_vars']:,}")
    m3.metric("Clusters", data["n_clusters"])
    m4.metric("Archetypes", data["n_archetypes"])

    st.divider()

    # UMAP
    st.subheader("UMAP Embedding")
    umap = np.array(data["umap"])
    tab1, tab2 = st.tabs(["By Leiden Cluster", "By True Cell Type"])

    with tab1:
        fig = px.scatter(x=umap[:, 0], y=umap[:, 1], color=data["leiden"],
                         labels={"x": "UMAP 1", "y": "UMAP 2", "color": "Leiden"},
                         title="UMAP colored by Leiden cluster",
                         color_discrete_sequence=px.colors.qualitative.Set3, opacity=0.7)
        fig.update_layout(height=550)
        fig.update_traces(marker_size=4)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig2 = px.scatter(x=umap[:, 0], y=umap[:, 1], color=data["true_types"],
                          labels={"x": "UMAP 1", "y": "UMAP 2", "color": "Cell Type"},
                          title="UMAP colored by true cell type",
                          color_discrete_sequence=px.colors.qualitative.D3, opacity=0.7)
        fig2.update_layout(height=550)
        fig2.update_traces(marker_size=4)
        st.plotly_chart(fig2, use_container_width=True)

    # Cluster composition
    st.subheader("Cluster Composition")
    cc1, cc2 = st.columns(2)

    with cc1:
        from collections import Counter
        counts = Counter(data["leiden"])
        labels = sorted(counts.keys(), key=lambda x: int(x))
        vals = [counts[l] for l in labels]
        fig_bar = px.bar(x=labels, y=vals,
                         labels={"x": "Leiden Cluster", "y": "Cell Count"},
                         title="Cells per cluster",
                         color=labels, color_discrete_sequence=px.colors.qualitative.Set3)
        fig_bar.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_bar, use_container_width=True)

    with cc2:
        type_counts = Counter(data["true_types"])
        fig_pie = px.pie(values=list(type_counts.values()), names=list(type_counts.keys()),
                         title="Cell type proportions",
                         color_discrete_sequence=px.colors.qualitative.D3)
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Archetypes table
    st.subheader("Extracted Archetypes")
    st.dataframe(data["archetypes"], use_container_width=True, hide_index=True)

    # Marker heatmap
    heatmap = np.array(data["heatmap"])
    if heatmap.size > 0:
        st.subheader("Marker Gene Expression")
        fig_heat = px.imshow(heatmap, x=data["heatmap_x"], y=data["heatmap_y"],
                             color_continuous_scale="Viridis",
                             title="Mean marker gene expression by archetype",
                             labels={"color": "Expression"}, aspect="auto")
        fig_heat.update_layout(height=400)
        st.plotly_chart(fig_heat, use_container_width=True)

    # 3D spatial preview
    st.subheader("Simulation Configuration Preview")
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Tissue Cells", len(data["cells"]))
    sc2.metric("Immune Cells", len(data["immune_cells"]))
    sc3.metric("Spatial Fields", data["spatial_fields"])

    all_cells = data["cells"] + data["immune_cells"]
    if all_cells:
        positions = np.array([c["position"] for c in all_cells])
        types = [c.get("archetype", c.get("cell_type", "unknown")) for c in all_cells]
        category = (["Tissue"] * len(data["cells"])) + (["Immune"] * len(data["immune_cells"]))

        fig3d = px.scatter_3d(x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
                              color=types, symbol=category,
                              labels={"x": "X (um)", "y": "Y (um)", "z": "Z (um)",
                                      "color": "Archetype", "symbol": "Category"},
                              title="Spatial cell placement (simulation initial conditions)",
                              opacity=0.8)
        fig3d.update_layout(height=600)
        fig3d.update_traces(marker_size=4)
        st.plotly_chart(fig3d, use_container_width=True)

else:
    st.info("Click **Run Pipeline** to process synthetic prostate tissue data and visualize results.")

# Footer
from cognisom.dashboard.footer import render_footer
render_footer()
