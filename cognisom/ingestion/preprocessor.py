"""
scRNA-seq Preprocessor
======================

Standard preprocessing pipeline for single-cell RNA-seq data.
Uses Scanpy (CPU) or RAPIDS-singlecell (GPU) depending on availability.

Pipeline:
    1. Quality control (filter cells/genes)
    2. Normalization (library size + log)
    3. Feature selection (highly variable genes)
    4. Dimensionality reduction (PCA)
    5. Neighborhood graph + clustering (Leiden)
    6. UMAP embedding
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _has_rapids():
    """Check if RAPIDS-singlecell is available for GPU acceleration."""
    try:
        import rapids_singlecell
        return True
    except ImportError:
        return False


class ScRNAPreprocessor:
    """Preprocess scRNA-seq data for archetype extraction.

    Automatically uses GPU (RAPIDS-singlecell) if available,
    otherwise falls back to CPU (Scanpy).

    Example:
        preprocessor = ScRNAPreprocessor()
        adata = preprocessor.run(adata)
        # adata now has .obs['leiden'] clusters and .obsm['X_umap']
    """

    def __init__(self, use_gpu: Optional[bool] = None):
        if use_gpu is None:
            self.use_gpu = _has_rapids()
        else:
            self.use_gpu = use_gpu

        if self.use_gpu:
            logger.info("Using RAPIDS-singlecell (GPU-accelerated)")
        else:
            logger.info("Using Scanpy (CPU) - install rapids-singlecell for GPU")

    def run(self, adata, min_genes: int = 200, min_cells: int = 3,
            max_pct_mito: float = 20.0, n_top_genes: int = 2000,
            n_pcs: int = 50, resolution: float = 1.0):
        """Run the full preprocessing pipeline.

        Args:
            adata: AnnData object (raw counts).
            min_genes: Minimum genes per cell.
            min_cells: Minimum cells per gene.
            max_pct_mito: Maximum mitochondrial gene percentage.
            n_top_genes: Number of highly variable genes to select.
            n_pcs: Number of principal components.
            resolution: Leiden clustering resolution.

        Returns:
            Preprocessed AnnData with clusters in .obs['leiden'].
        """
        import scanpy as sc

        logger.info(f"Input: {adata.n_obs} cells x {adata.n_vars} genes")

        if self.use_gpu:
            return self._run_gpu(
                adata, min_genes, min_cells, max_pct_mito,
                n_top_genes, n_pcs, resolution
            )

        # --- CPU pipeline (Scanpy) ---

        # Make a copy to avoid modifying the original
        adata = adata.copy()

        # Step 1: QC
        logger.info("Step 1: Quality control")
        adata.var["mt"] = adata.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(
            adata, qc_vars=["mt"], percent_top=None, inplace=True
        )
        n_before = adata.n_obs
        sc.pp.filter_cells(adata, min_genes=min_genes)
        sc.pp.filter_genes(adata, min_cells=min_cells)
        adata = adata[adata.obs["pct_counts_mt"] < max_pct_mito].copy()
        logger.info(f"  Filtered: {n_before} -> {adata.n_obs} cells")

        # Step 2: Normalize
        logger.info("Step 2: Normalization")
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # Step 3: Feature selection
        logger.info("Step 3: Highly variable genes")
        try:
            sc.pp.highly_variable_genes(
                adata, n_top_genes=n_top_genes, flavor="seurat_v3",
                layer="counts"
            )
        except (ValueError, Exception):
            # Fallback for datasets where seurat_v3 fails
            logger.info("  seurat_v3 failed, falling back to cell_ranger flavor")
            sc.pp.highly_variable_genes(
                adata, n_top_genes=min(n_top_genes, adata.n_vars),
                flavor="cell_ranger"
            )
        n_hvg = adata.var["highly_variable"].sum()
        logger.info(f"  Selected {n_hvg} highly variable genes")

        # Step 4: PCA
        logger.info("Step 4: PCA")
        sc.tl.pca(adata, n_comps=n_pcs, use_highly_variable=True)

        # Step 5: Neighbors + Leiden
        logger.info("Step 5: Clustering")
        sc.pp.neighbors(adata, n_pcs=n_pcs)
        sc.tl.leiden(adata, resolution=resolution)
        n_clusters = adata.obs["leiden"].nunique()
        logger.info(f"  Found {n_clusters} clusters")

        # Step 6: UMAP
        logger.info("Step 6: UMAP embedding")
        sc.tl.umap(adata)

        logger.info(f"Done: {adata.n_obs} cells, {n_clusters} clusters")
        return adata

    def _run_gpu(self, adata, min_genes, min_cells, max_pct_mito,
                 n_top_genes, n_pcs, resolution):
        """GPU-accelerated pipeline using RAPIDS-singlecell.

        Up to 938x faster than CPU Scanpy on large datasets.
        Requires NVIDIA GPU and rapids-singlecell package.
        """
        import rapids_singlecell as rsc
        import scanpy as sc

        adata = adata.copy()

        # Move to GPU
        rsc.get.anndata_to_GPU(adata)

        # QC
        logger.info("Step 1: Quality control (GPU)")
        adata.var["mt"] = adata.var_names.str.startswith("MT-")
        rsc.pp.calculate_qc_metrics(adata, qc_vars=["mt"])
        n_before = adata.n_obs
        rsc.pp.filter_cells(adata, min_genes=min_genes)
        rsc.pp.filter_genes(adata, min_cells=min_cells)
        adata = adata[adata.obs["pct_counts_mt"] < max_pct_mito].copy()
        logger.info(f"  Filtered: {n_before} -> {adata.n_obs} cells")

        # Normalize
        logger.info("Step 2: Normalization (GPU)")
        adata.layers["counts"] = adata.X.copy()
        rsc.pp.normalize_total(adata, target_sum=1e4)
        rsc.pp.log1p(adata)

        # HVG
        logger.info("Step 3: Highly variable genes (GPU)")
        rsc.pp.highly_variable_genes(
            adata, n_top_genes=n_top_genes, flavor="seurat_v3",
            layer="counts"
        )

        # PCA
        logger.info("Step 4: PCA (GPU)")
        rsc.pp.pca(adata, n_comps=n_pcs, use_highly_variable=True)

        # Neighbors + Leiden
        logger.info("Step 5: Clustering (GPU)")
        rsc.pp.neighbors(adata, n_pcs=n_pcs)
        rsc.tl.leiden(adata, resolution=resolution)
        n_clusters = adata.obs["leiden"].nunique()
        logger.info(f"  Found {n_clusters} clusters")

        # UMAP
        logger.info("Step 6: UMAP (GPU)")
        rsc.tl.umap(adata)

        # Move back to CPU for compatibility
        rsc.get.anndata_to_CPU(adata)

        logger.info(f"Done (GPU): {adata.n_obs} cells, {n_clusters} clusters")
        return adata
