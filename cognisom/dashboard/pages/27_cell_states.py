"""
Page 27: Cell State Analysis — Cell2Sentence Integration
=========================================================

Analyze single-cell gene expression to predict cell states,
T-cell exhaustion scores, and macrophage polarization.

Uses Cell2Sentence-Scale 27B (or marker-based fallback) to
classify the tumor immune microenvironment.

Phase 2 of the Molecular Digital Twin pipeline.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import logging

st.set_page_config(page_title="Cell States", page_icon="🔬", layout="wide")

from cognisom.auth.middleware import streamlit_page_gate
user = streamlit_page_gate("27_cell_states")

logger = logging.getLogger(__name__)

st.title("🔬 Cell State Analysis")
st.caption(
    "Analyze single-cell gene expression to predict cell types, T-cell exhaustion, "
    "and macrophage polarization using Cell2Sentence AI."
)

# ─────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Configuration")

    data_source = st.radio(
        "Data Source:",
        ["Synthetic Demo", "CellXGene Census", "Upload h5ad"],
        index=0,
    )

    model_mode = st.radio(
        "Cell2Sentence Mode:",
        ["Marker Heuristic (fast)", "Full Model (27B, GPU required)"],
        index=0,
    )

    n_cells = st.slider("Number of cells to analyze", 50, 2000, 200, step=50)

    st.divider()
    st.header("Filters")
    tissue_filter = st.selectbox(
        "Tissue", ["prostate gland", "All"], index=0
    )
    cell_type_filter = st.multiselect(
        "Cell types to include",
        ["All", "T cells", "Macrophages", "NK cells", "Epithelial", "Cancer"],
        default=["All"],
    )

# ─────────────────────────────────────────────────────────────────────────
# LOAD / GENERATE DATA
# ─────────────────────────────────────────────────────────────────────────


@st.cache_data(ttl=600)
def generate_synthetic_expression(n_cells: int = 200):
    """Generate synthetic scRNA-seq data mimicking prostate tumor microenvironment."""
    np.random.seed(42)

    # Define cell type proportions (prostate tumor TIME)
    cell_types = {
        "CD8+ T cell": 0.15,
        "CD4+ T cell": 0.10,
        "regulatory T cell": 0.05,
        "macrophage": 0.12,
        "NK cell": 0.05,
        "dendritic cell": 0.03,
        "B cell": 0.05,
        "cancer epithelial": 0.25,
        "epithelial": 0.10,
        "fibroblast": 0.08,
        "endothelial": 0.02,
    }

    # Marker genes for each cell type
    cell_markers = {
        "CD8+ T cell": ["CD8A", "CD8B", "CD3E", "CD3D", "GZMB", "PRF1", "IFNG"],
        "CD4+ T cell": ["CD4", "CD3E", "CD3D", "IL7R", "IL2"],
        "regulatory T cell": ["FOXP3", "IL2RA", "CTLA4", "CD4", "TIGIT"],
        "macrophage": ["CD68", "CD14", "CSF1R", "FCGR3A"],
        "NK cell": ["NKG7", "GNLY", "KLRD1", "NCAM1"],
        "dendritic cell": ["ITGAX", "HLA-DRA", "CD1C"],
        "B cell": ["CD19", "MS4A1", "CD79A"],
        "cancer epithelial": ["EPCAM", "KRT18", "MKI67", "TOP2A", "PCNA"],
        "epithelial": ["EPCAM", "KRT18", "KRT8", "CDH1"],
        "fibroblast": ["COL1A1", "DCN", "FAP", "VIM"],
        "endothelial": ["PECAM1", "VWF", "CDH5"],
    }

    # Exhaustion-related genes
    exhaustion_genes = [
        "PDCD1", "HAVCR2", "LAG3", "TIGIT", "CTLA4", "CD244",
        "TOX", "TOX2", "NR4A1", "NR4A2",
        "TCF7", "LEF1", "MYB", "IL7R",
    ]

    # Polarization genes
    polarization_genes = [
        "TNF", "IL1B", "IL6", "NOS2", "CD80", "CD86", "CXCL10",
        "IL10", "MRC1", "CD163", "ARG1", "CCL22", "TGFB1", "VEGFA",
    ]

    # Housekeeping
    housekeeping = ["GAPDH", "ACTB", "TUBB", "RPL13A", "RPS18"]

    # Collect all genes
    all_genes = set(housekeeping + exhaustion_genes + polarization_genes)
    for markers in cell_markers.values():
        all_genes.update(markers)
    all_genes = sorted(all_genes)

    # Generate expression matrix
    n_genes = len(all_genes)
    gene_idx = {g: i for i, g in enumerate(all_genes)}
    X = np.random.exponential(0.3, (n_cells, n_genes))

    cell_type_labels = []
    exhaustion_true = []

    for i in range(n_cells):
        # Assign cell type
        r = np.random.random()
        cumsum = 0
        ct = "unknown"
        for ctype, frac in cell_types.items():
            cumsum += frac
            if r < cumsum:
                ct = ctype
                break
        cell_type_labels.append(ct)

        # Boost marker genes
        markers = cell_markers.get(ct, [])
        for g in markers:
            if g in gene_idx:
                X[i, gene_idx[g]] += np.random.uniform(3, 8)

        # Housekeeping always high
        for g in housekeeping:
            if g in gene_idx:
                X[i, gene_idx[g]] += np.random.uniform(5, 10)

        # Add exhaustion signature to some T cells
        exh_score = 0.0
        if "T cell" in ct and ct != "regulatory T cell":
            # 40% of T cells are exhausted in this tumor
            if np.random.random() < 0.4:
                exh_score = np.random.uniform(0.5, 0.95)
                for g in ["PDCD1", "HAVCR2", "LAG3", "TIGIT", "TOX"]:
                    if g in gene_idx:
                        X[i, gene_idx[g]] += exh_score * np.random.uniform(3, 7)
                # Reduce effector genes
                for g in ["GZMB", "PRF1", "IFNG", "IL2"]:
                    if g in gene_idx:
                        X[i, gene_idx[g]] *= (1 - exh_score * 0.7)
            else:
                exh_score = np.random.uniform(0.0, 0.3)
        exhaustion_true.append(exh_score)

        # Macrophage polarization
        if ct == "macrophage":
            if np.random.random() < 0.6:  # 60% M2 in tumor
                for g in ["IL10", "MRC1", "CD163", "ARG1", "TGFB1"]:
                    if g in gene_idx:
                        X[i, gene_idx[g]] += np.random.uniform(3, 6)
            else:
                for g in ["TNF", "IL1B", "IL6", "NOS2", "CD80"]:
                    if g in gene_idx:
                        X[i, gene_idx[g]] += np.random.uniform(3, 6)

    return {
        "gene_names": all_genes,
        "expression": X,
        "cell_types": cell_type_labels,
        "exhaustion_true": exhaustion_true,
        "n_cells": n_cells,
    }


# ─────────────────────────────────────────────────────────────────────────
# MAIN ANALYSIS
# ─────────────────────────────────────────────────────────────────────────

if data_source == "Synthetic Demo":
    with st.spinner("Generating synthetic prostate tumor microenvironment..."):
        data = generate_synthetic_expression(n_cells=n_cells)

elif data_source == "Upload h5ad":
    uploaded = st.file_uploader("Upload .h5ad file", type=["h5ad", "h5"])
    if not uploaded:
        st.info("Upload an h5ad file to begin analysis.")
        st.stop()
    st.warning("h5ad loading not yet implemented — using synthetic demo")
    data = generate_synthetic_expression(n_cells=n_cells)

else:
    st.info("CellXGene Census integration — select tissue and download")
    st.warning("Live CellXGene download not yet implemented — using synthetic demo")
    data = generate_synthetic_expression(n_cells=n_cells)

# Run Cell2Sentence analysis
with st.spinner("Running Cell2Sentence analysis..."):
    from cognisom.genomics.expression_ranker import ExpressionRanker
    from cognisom.genomics.cell2sentence import Cell2SentenceModel
    from cognisom.genomics.cell_state_classifier import CellStateClassifier

    ranker = ExpressionRanker(max_genes=200)
    model = Cell2SentenceModel()

    use_full_model = model_mode.startswith("Full Model")
    if use_full_model:
        loaded = model.load()
        if not loaded or model.is_fallback:
            st.warning("Full model not available — using marker heuristic fallback")
    else:
        model._loaded = True
        model._fallback_mode = True

    # Rank cells into sentences
    sentences = []
    for i in range(data["n_cells"]):
        sentence = ranker.rank_cell(data["gene_names"], data["expression"][i])
        sentences.append(sentence)

    # Predict cell states
    predictions = model.batch_predict(sentences)

    # Classify immune microenvironment
    tmb = st.session_state.get("patient_profile", {})
    tmb_val = tmb.tumor_mutational_burden if hasattr(tmb, "tumor_mutational_burden") else 0.0
    classifier = CellStateClassifier()
    classification = classifier.classify(predictions, tmb=tmb_val)

# Store in session state for Digital Twin page
st.session_state["cell_classification"] = classification
st.session_state["cell_predictions"] = predictions

# ─────────────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────────────

# Overview metrics
st.header("Immune Microenvironment")

score_colors = {"hot": "🔴", "cold": "🔵", "excluded": "🟡", "suppressed": "🟣"}
score_emoji = score_colors.get(classification.immune_score, "⚪")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Immune Score", f"{score_emoji} {classification.immune_score.upper()}")
with col2:
    st.metric("Immune Fraction", f"{classification.composition.immune_fraction:.1%}")
with col3:
    st.metric("Mean T-cell Exhaustion", f"{classification.mean_exhaustion:.2f}")
with col4:
    st.metric("Exhausted T-cells", f"{classification.exhausted_fraction:.1%}")
with col5:
    st.metric("M1:M2 Ratio", f"{classification.m1_m2_ratio:.1f}")

# Therapy insights
col_a, col_b, col_c = st.columns(3)
with col_a:
    if classification.immunotherapy_responsive:
        st.success("Immunotherapy likely responsive")
    else:
        st.warning("Immunotherapy unlikely to be effective alone")
with col_b:
    if classification.checkpoint_blockade_likely_effective:
        st.success("Checkpoint blockade may reactivate T-cells")
    else:
        st.info("Checkpoint blockade alone may be insufficient")
with col_c:
    if classification.combination_therapy_needed:
        st.warning("Combination therapy recommended")
    else:
        st.success("Monotherapy may be sufficient")

# ─────────────────────────────────────────────────────────────────────────
# CELL TYPE COMPOSITION
# ─────────────────────────────────────────────────────────────────────────

st.header("Cell Type Composition")

comp = classification.composition
comp_data = {
    "Cell Type": [
        "CD8+ T cells", "CD4+ T cells", "Regulatory T cells",
        "NK cells", "Macrophages", "Dendritic cells", "B cells",
        "Cancer epithelial", "Epithelial", "Stromal",
    ],
    "Count": [
        comp.cd8_t_cells, comp.cd4_t_cells, comp.regulatory_t_cells,
        comp.nk_cells, comp.macrophages, comp.dendritic_cells, comp.b_cells,
        comp.cancer_cells, comp.epithelial, comp.stromal,
    ],
}
comp_df = pd.DataFrame(comp_data)
comp_df = comp_df[comp_df["Count"] > 0]

col_pie, col_bar = st.columns(2)

with col_pie:
    fig_pie = px.pie(
        comp_df, names="Cell Type", values="Count",
        title="Cell Type Distribution",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_pie.update_layout(height=400)
    st.plotly_chart(fig_pie, use_container_width=True)

with col_bar:
    fig_bar = px.bar(
        comp_df, x="Cell Type", y="Count",
        title="Cell Counts",
        color="Cell Type",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_bar.update_layout(height=400, showlegend=False, xaxis_tickangle=-45)
    st.plotly_chart(fig_bar, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────
# T-CELL EXHAUSTION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────

st.header("T-Cell Exhaustion Profile")

t_cell_preds = [
    p for p in predictions
    if "t cell" in p.predicted_cell_type.lower() and
    "regulatory" not in p.predicted_cell_type.lower()
]

if t_cell_preds:
    exh_scores = [p.exhaustion_score for p in t_cell_preds if p.exhaustion_score is not None]

    col_hist, col_scatter = st.columns(2)

    with col_hist:
        fig_exh = go.Figure()
        fig_exh.add_trace(go.Histogram(
            x=exh_scores,
            nbinsx=20,
            marker_color="indianred",
            name="T-cell exhaustion",
        ))
        fig_exh.add_vline(x=0.6, line_dash="dash", line_color="red",
                         annotation_text="Exhaustion threshold")
        fig_exh.update_layout(
            title="T-Cell Exhaustion Score Distribution",
            xaxis_title="Exhaustion Score (0=effector, 1=exhausted)",
            yaxis_title="Count",
            height=400,
        )
        st.plotly_chart(fig_exh, use_container_width=True)

    with col_scatter:
        # Categorize T-cells
        categories = []
        for p in t_cell_preds:
            s = p.exhaustion_score or 0
            if s > 0.6:
                categories.append("Exhausted")
            elif s > 0.3:
                categories.append("Pre-exhausted")
            else:
                categories.append("Effector/Memory")

        cat_counts = pd.Series(categories).value_counts()
        fig_cat = px.pie(
            names=cat_counts.index, values=cat_counts.values,
            title="T-Cell State Categories",
            color=cat_counts.index,
            color_discrete_map={
                "Exhausted": "#ff4444",
                "Pre-exhausted": "#ffaa44",
                "Effector/Memory": "#44aa44",
            },
        )
        fig_cat.update_layout(height=400)
        st.plotly_chart(fig_cat, use_container_width=True)
else:
    st.info("No T cells detected in the sample.")

# ─────────────────────────────────────────────────────────────────────────
# MACROPHAGE POLARIZATION
# ─────────────────────────────────────────────────────────────────────────

st.header("Macrophage Polarization")

mac_preds = [p for p in predictions if "macrophage" in p.predicted_cell_type.lower()]

if mac_preds:
    pol_data = []
    for p in mac_preds:
        pol_data.append({
            "Cell": f"Mac_{p.cell_index}",
            "Polarization": p.polarization or "unknown",
            "Score": p.polarization_score or 0.0,
        })
    pol_df = pd.DataFrame(pol_data)

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        pol_counts = pol_df["Polarization"].value_counts()
        fig_pol = px.pie(
            names=pol_counts.index, values=pol_counts.values,
            title="Macrophage Polarization",
            color=pol_counts.index,
            color_discrete_map={"M1": "#4488ff", "M2": "#ff4488", "mixed": "#888888"},
        )
        fig_pol.update_layout(height=350)
        st.plotly_chart(fig_pol, use_container_width=True)

    with col_m2:
        st.markdown(
            f"**M1 (pro-inflammatory):** {classification.m1_fraction:.0%}\n\n"
            f"**M2 (immunosuppressive):** {classification.m2_fraction:.0%}\n\n"
            f"**M1:M2 ratio:** {classification.m1_m2_ratio:.1f}\n\n"
        )
        if classification.m1_m2_ratio < 1.0:
            st.warning(
                "M2-dominant macrophage profile — immunosuppressive. "
                "Consider CSF1R inhibitors or CD47/SIRPα blockade."
            )
        else:
            st.success("M1-dominant — pro-inflammatory environment.")
else:
    st.info("No macrophages detected in the sample.")

# ─────────────────────────────────────────────────────────────────────────
# ALL PREDICTIONS TABLE
# ─────────────────────────────────────────────────────────────────────────

with st.expander("All Cell Predictions", expanded=False):
    pred_data = []
    for p in predictions:
        pred_data.append({
            "Index": p.cell_index,
            "Predicted Type": p.predicted_cell_type,
            "State": p.predicted_state,
            "Exhaustion": f"{p.exhaustion_score:.2f}" if p.exhaustion_score is not None else "-",
            "Polarization": p.polarization or "-",
            "Confidence": f"{p.confidence:.2f}",
            "True Type": data["cell_types"][p.cell_index] if p.cell_index < len(data["cell_types"]) else "-",
        })
    pred_df = pd.DataFrame(pred_data)
    st.dataframe(pred_df, use_container_width=True, hide_index=True, height=400)

# ─────────────────────────────────────────────────────────────────────────
# LINK TO DIGITAL TWIN
# ─────────────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    "**Next step:** Go to **Page 28: Digital Twin** to combine this immune landscape "
    "analysis with the patient's genomic profile and run personalized treatment simulations."
)
