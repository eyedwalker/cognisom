"""
Page 30: Spatial Transcriptomics Viewer
========================================

Visualize gene expression data mapped onto tissue spatial coordinates.
Overlay immune infiltration, cell state predictions, and identify
spatial patterns in the tumor microenvironment.

Phase 5 of the Molecular Digital Twin pipeline.
"""

import streamlit as st
import numpy as np
import logging

st.set_page_config(page_title="Spatial Tissue", page_icon="🔬", layout="wide")

from cognisom.auth.middleware import streamlit_page_gate
user = streamlit_page_gate("30_spatial_tissue")

from cognisom.genomics.spatial_transcriptomics import SpatialData, SpatialStats

logger = logging.getLogger(__name__)

st.title("🔬 Spatial Transcriptomics")
st.caption(
    "Map gene expression onto tissue coordinates. Identify immune-hot zones, "
    "exclusion barriers, and exhausted T-cell clusters in the tumor microenvironment."
)

# ─────────────────────────────────────────────────────────────────────────
# SIDEBAR: DATA SOURCE
# ─────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Data Source")

    data_source = st.radio(
        "Load spatial data from:",
        ["Synthetic Prostate TME", "Upload h5ad"],
        index=0,
    )

    n_spots = st.slider("Number of spots", 50, 500, 200, 50,
                         help="For synthetic data only")

    st.divider()
    st.header("Display")

    color_by = st.selectbox(
        "Color spots by",
        ["immune_score", "cell_type", "exhaustion", "gene_expression"],
    )

    if color_by == "gene_expression":
        gene_to_show = st.text_input("Gene name", value="CD8A")
    else:
        gene_to_show = None

    show_regions = st.checkbox("Show classified regions", value=True)
    show_stats = st.checkbox("Show spatial statistics", value=True)
    spot_size = st.slider("Spot size", 2, 20, 8)


# ─────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_synthetic_data(n: int) -> SpatialData:
    data = SpatialData()
    data.load_synthetic_prostate(n_spots=n)
    data.classify_regions()
    return data


spatial_data = None

if data_source == "Synthetic Prostate TME":
    spatial_data = load_synthetic_data(n_spots)

elif data_source == "Upload h5ad":
    uploaded = st.file_uploader("Upload AnnData (.h5ad)", type=["h5ad"])
    if uploaded:
        try:
            import scanpy as sc
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
                f.write(uploaded.getvalue())
                tmp_path = f.name

            adata = sc.read_h5ad(tmp_path)
            os.unlink(tmp_path)

            spatial_data = SpatialData()
            spatial_data.load_from_anndata(adata)
            spatial_data.classify_regions()
            st.success(f"Loaded {spatial_data.n_spots} spots, {spatial_data.n_genes} genes")
        except ImportError:
            st.error("Install scanpy to load h5ad files: `pip install scanpy`")
        except Exception as e:
            st.error(f"Failed to load h5ad: {e}")


if spatial_data is None:
    st.info("Select a data source from the sidebar")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────
# OVERVIEW METRICS
# ─────────────────────────────────────────────────────────────────────────

st.divider()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Spots", f"{spatial_data.n_spots:,}")
col2.metric("Genes", f"{spatial_data.n_genes:,}")
col3.metric("Regions", len(spatial_data.regions))

# Count immune scores
immune_counts = {}
for spot in spatial_data.spots:
    immune_counts[spot.immune_score] = immune_counts.get(spot.immune_score, 0) + 1

hot_count = immune_counts.get("hot", 0)
cold_count = immune_counts.get("cold", 0)
col4.metric("Hot Spots", hot_count)
col5.metric("Cold Spots", cold_count)

# ─────────────────────────────────────────────────────────────────────────
# SPATIAL PLOT
# ─────────────────────────────────────────────────────────────────────────

st.subheader("Tissue Map")

try:
    import plotly.graph_objects as go
    import plotly.express as px

    coords = spatial_data.get_coordinates()
    x_vals = coords[:, 0]
    y_vals = coords[:, 1]

    # Determine colors and labels
    if color_by == "immune_score":
        color_map = {
            "hot": "#ff4444",
            "cold": "#4444ff",
            "suppressed": "#ff8800",
            "excluded": "#888888",
            "normal": "#44cc44",
        }
        colors = [color_map.get(s.immune_score, "#aaaaaa") for s in spatial_data.spots]
        hover_text = [
            f"{s.barcode}<br>Type: {s.cell_type}<br>Immune: {s.immune_score}"
            for s in spatial_data.spots
        ]

    elif color_by == "cell_type":
        type_map = {
            "tumor_cell": "#cc3333",
            "fibroblast": "#8888bb",
            "t_cell_exhausted": "#ff8800",
            "epithelial": "#44aa44",
        }
        colors = [type_map.get(s.cell_type, "#aaaaaa") for s in spatial_data.spots]
        hover_text = [
            f"{s.barcode}<br>Type: {s.cell_type}"
            for s in spatial_data.spots
        ]

    elif color_by == "exhaustion":
        exhaustion_vals = [s.exhaustion_score for s in spatial_data.spots]
        colors = exhaustion_vals
        hover_text = [
            f"{s.barcode}<br>Exhaustion: {s.exhaustion_score:.2f}<br>Type: {s.cell_type}"
            for s in spatial_data.spots
        ]

    elif color_by == "gene_expression" and gene_to_show:
        expr_vals = spatial_data.get_expression(gene_to_show)
        colors = expr_vals.tolist()
        hover_text = [
            f"{s.barcode}<br>{gene_to_show}: {expr_vals[i]:.2f}<br>Type: {s.cell_type}"
            for i, s in enumerate(spatial_data.spots)
        ]

    else:
        colors = ["#aaaaaa"] * len(spatial_data.spots)
        hover_text = [s.barcode for s in spatial_data.spots]

    fig = go.Figure()

    # Use continuous colorscale for numeric values
    if color_by in ("exhaustion", "gene_expression") and isinstance(colors[0], (int, float)):
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode="markers",
            marker=dict(
                size=spot_size,
                color=colors,
                colorscale="RdYlBu_r",
                showscale=True,
                colorbar=dict(
                    title=color_by.replace("_", " ").title() if color_by == "exhaustion"
                          else gene_to_show,
                ),
            ),
            text=hover_text,
            hoverinfo="text",
        ))
    else:
        # Categorical colors
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode="markers",
            marker=dict(
                size=spot_size,
                color=colors,
            ),
            text=hover_text,
            hoverinfo="text",
        ))

    # Draw region boundaries
    if show_regions and spatial_data.regions:
        for region in spatial_data.regions:
            if not region.spots:
                continue
            # Draw region center marker
            fig.add_trace(go.Scatter(
                x=[region.center_x],
                y=[region.center_y],
                mode="markers+text",
                marker=dict(size=3, color="white", symbol="x"),
                text=[region.name],
                textposition="top center",
                textfont=dict(size=9, color="white"),
                showlegend=False,
                hoverinfo="text",
            ))

    fig.update_layout(
        xaxis=dict(title="X (tissue)", scaleanchor="y", scaleratio=1),
        yaxis=dict(title="Y (tissue)", autorange="reversed"),
        plot_bgcolor="rgb(10,10,30)",
        paper_bgcolor="rgb(10,10,30)",
        font=dict(color="white"),
        height=600,
        margin=dict(l=40, r=40, t=20, b=40),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Legend for categorical colors
    if color_by == "immune_score":
        legend_cols = st.columns(5)
        for (label, clr), col in zip(color_map.items(), legend_cols):
            count = immune_counts.get(label, 0)
            col.markdown(
                f'<span style="color:{clr}">&#9679;</span> {label} ({count})',
                unsafe_allow_html=True,
            )

except ImportError:
    st.warning("Install plotly for visualization: `pip install plotly`")


# ─────────────────────────────────────────────────────────────────────────
# IMMUNE INFILTRATION HEATMAPS
# ─────────────────────────────────────────────────────────────────────────

st.divider()
st.subheader("Immune Infiltration Maps")

infiltration = spatial_data.get_immune_infiltration_map()

# Show 3 key maps side by side
map_cols = st.columns(3)
map_choices = [
    ("cd8_t_cells", "CD8+ T Cells"),
    ("exhaustion", "Exhaustion Markers"),
    ("m2_macrophages", "M2 Macrophages"),
]

try:
    import plotly.graph_objects as go

    for (key, label), col in zip(map_choices, map_cols):
        with col:
            st.caption(label)
            vals = infiltration[key]
            coords = spatial_data.get_coordinates()

            mini_fig = go.Figure()
            mini_fig.add_trace(go.Scatter(
                x=coords[:, 0], y=coords[:, 1],
                mode="markers",
                marker=dict(
                    size=6,
                    color=vals,
                    colorscale="Inferno",
                    showscale=False,
                ),
                hoverinfo="skip",
            ))
            mini_fig.update_layout(
                xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
                yaxis=dict(visible=False, autorange="reversed"),
                plot_bgcolor="rgb(10,10,30)",
                paper_bgcolor="rgb(10,10,30)",
                height=250,
                margin=dict(l=5, r=5, t=5, b=5),
            )
            st.plotly_chart(mini_fig, use_container_width=True)

except ImportError:
    pass


# ─────────────────────────────────────────────────────────────────────────
# SPATIAL STATISTICS
# ─────────────────────────────────────────────────────────────────────────

if show_stats:
    st.divider()
    st.subheader("Spatial Autocorrelation (Moran's I)")
    st.caption(
        "Moran's I measures spatial clustering of gene expression. "
        "Values near +1 indicate strong spatial clustering (genes expressed "
        "in neighboring spots), values near -1 indicate dispersion."
    )

    # Compute stats for key immune genes
    genes_to_test = ["CD8A", "PDCD1", "CD163", "COL1A1", "KRT8", "GZMB"]
    available_genes = [g for g in genes_to_test if g in spatial_data.gene_names]

    if available_genes:
        import pandas as pd

        stats_rows = []
        for gene in available_genes:
            stats = spatial_data.compute_spatial_stats(gene)
            stats_rows.append({
                "Gene": stats.feature,
                "Moran's I": round(stats.morans_i, 3),
                "p-value": f"{stats.p_value:.4f}",
                "Hotspots": stats.hotspot_count,
                "Coldspots": stats.coldspot_count,
                "Clustered?": "Yes" if stats.morans_i > 0.1 and stats.p_value < 0.05 else "No",
            })

        stats_df = pd.DataFrame(stats_rows)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # User-specified gene
    user_gene = st.text_input("Compute for gene:", value="", placeholder="e.g. TOX")
    if user_gene and user_gene in spatial_data.gene_names:
        stats = spatial_data.compute_spatial_stats(user_gene)
        st.write(
            f"**{user_gene}**: Moran's I = {stats.morans_i:.3f}, "
            f"p = {stats.p_value:.4f}, "
            f"Hotspots: {stats.hotspot_count}, Coldspots: {stats.coldspot_count}"
        )
    elif user_gene:
        st.warning(f"Gene '{user_gene}' not found in dataset")


# ─────────────────────────────────────────────────────────────────────────
# REGION DETAILS
# ─────────────────────────────────────────────────────────────────────────

if show_regions and spatial_data.regions:
    st.divider()
    st.subheader("Classified Regions")

    import pandas as pd

    region_rows = []
    for region in spatial_data.regions:
        region_rows.append({
            "Region": region.name,
            "Type": region.region_type,
            "Spots": len(region.spots),
            "Center X": round(region.center_x, 1),
            "Center Y": round(region.center_y, 1),
        })

    region_df = pd.DataFrame(region_rows)
    st.dataframe(region_df, use_container_width=True, hide_index=True)

    # Clinical interpretation
    st.subheader("Clinical Interpretation")

    has_excluded = any(r.region_type == "stroma" for r in spatial_data.regions)
    has_hot = any(r.region_type == "immune_hot" for r in spatial_data.regions)
    has_exhausted = any(r.region_type in ("immune_exhausted", "immune_suppressed")
                        for r in spatial_data.regions)
    has_cold = any(r.region_type == "immune_cold" for r in spatial_data.regions)

    if has_hot:
        st.success(
            "**Immune-hot regions detected** — T cells are actively infiltrating "
            "the tumor. This pattern suggests potential response to checkpoint "
            "inhibitor immunotherapy (anti-PD-1/PD-L1)."
        )
    if has_exhausted:
        st.warning(
            "**Exhausted T-cell zones** — CD8+ T cells are present but express "
            "high levels of inhibitory receptors (PD-1, TIM-3, LAG-3). "
            "Combination checkpoint blockade may be needed."
        )
    if has_excluded:
        st.warning(
            "**Stromal barrier detected** — Dense collagen/fibroblast region "
            "separating immune cells from tumor. This 'immune-excluded' pattern "
            "may require stromal targeting (e.g., anti-FAP, TGF-beta inhibition) "
            "to enable T-cell infiltration."
        )
    if has_cold:
        st.error(
            "**Immune-cold tumor core** — Minimal immune infiltration in the "
            "tumor center. Monotherapy immunotherapy is unlikely to be effective. "
            "Consider combination approaches: CTLA-4 blockade, tumor vaccines, "
            "or oncolytic viruses to recruit immune cells."
        )
