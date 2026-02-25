"""
Spatial Transcriptomics
========================

Load and analyze spatial transcriptomics data (10x Visium, MERFISH,
Open-ST) for tissue-level gene expression mapping.

Maps gene expression onto tissue spatial coordinates, computes
spatial statistics, and integrates with Cell2Sentence cell state
predictions.

Phase 5 of the Molecular Digital Twin pipeline.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SpatialSpot:
    """A spot/cell in spatial transcriptomics data."""
    barcode: str
    x: float  # Tissue x coordinate
    y: float  # Tissue y coordinate
    row: int = 0
    col: int = 0
    in_tissue: bool = True
    gene_expression: Dict[str, float] = field(default_factory=dict)
    # Cell state annotations (from Cell2Sentence or other classifiers)
    cell_type: str = "unknown"
    exhaustion_score: float = 0.0
    polarization: str = "unknown"
    immune_score: str = "unknown"


@dataclass
class SpatialRegion:
    """A classified region in the tissue."""
    name: str
    spots: List[int]  # Indices into SpatialData.spots
    region_type: str  # "immune_hot", "immune_cold", "excluded", "stroma"
    center_x: float = 0.0
    center_y: float = 0.0
    mean_immune_fraction: float = 0.0


@dataclass
class SpatialStats:
    """Spatial statistics for a gene or feature."""
    feature: str
    morans_i: float  # Spatial autocorrelation (-1 to 1)
    p_value: float
    hotspot_count: int  # Number of significant hotspots (Gi*)
    coldspot_count: int  # Number of significant coldspots


class SpatialData:
    """Container for spatial transcriptomics data.

    Loads from various formats and provides analysis methods.

    Example:
        data = SpatialData()
        data.load_visium("filtered_feature_bc_matrix.h5",
                         "tissue_positions.csv")
        data.classify_regions()
        stats = data.compute_spatial_stats("CD8A")
    """

    def __init__(self):
        self.spots: List[SpatialSpot] = []
        self.gene_names: List[str] = []
        self.regions: List[SpatialRegion] = []
        self.tissue_image: Optional[np.ndarray] = None  # H&E image
        self.scale_factor: float = 1.0
        self._expression_matrix: Optional[np.ndarray] = None

    @property
    def n_spots(self) -> int:
        return len(self.spots)

    @property
    def n_genes(self) -> int:
        return len(self.gene_names)

    def load_from_anndata(self, adata) -> None:
        """Load from AnnData object (scanpy format).

        Expects adata.obsm["spatial"] for coordinates.
        """
        import pandas as pd

        if "spatial" not in adata.obsm:
            raise ValueError("AnnData must have 'spatial' in obsm")

        coords = adata.obsm["spatial"]
        self.gene_names = list(adata.var_names)

        # Get expression matrix (dense)
        try:
            expr = adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)
        except Exception:
            expr = np.array(adata.X)

        self._expression_matrix = expr

        self.spots = []
        for i in range(adata.n_obs):
            barcode = str(adata.obs_names[i])
            spot = SpatialSpot(
                barcode=barcode,
                x=float(coords[i, 0]),
                y=float(coords[i, 1]),
                in_tissue=True,
            )

            # Copy cell type annotations if present
            if "cell_type" in adata.obs.columns:
                spot.cell_type = str(adata.obs["cell_type"].iloc[i])

            self.spots.append(spot)

        logger.info(f"Loaded {self.n_spots} spots, {self.n_genes} genes from AnnData")

    def load_synthetic_prostate(self, n_spots: int = 200,
                                 seed: int = 42) -> None:
        """Generate synthetic spatial data for prostate tumor microenvironment.

        Creates a tissue layout with:
        - Central tumor core (cold, low immune infiltration)
        - Immune-excluded border (T cells at periphery)
        - Stroma barrier (collagen-rich, blocking infiltration)
        - Hot spots (some areas with immune infiltration)
        """
        rng = np.random.RandomState(seed)

        # Tissue dimensions (simulate a ~3mm tissue section)
        tissue_w, tissue_h = 300.0, 300.0

        # Define immune-related genes
        immune_genes = [
            "CD8A", "CD8B", "CD4", "FOXP3", "PDCD1", "HAVCR2", "LAG3",
            "TIGIT", "TOX", "GZMB", "PRF1", "IFNG", "TNF", "IL2",
            "CD68", "CD163", "CD14", "CSF1R", "ARG1", "NOS2",
            "COL1A1", "COL3A1", "FAP", "ACTA2",
            "KRT8", "KRT18", "AR", "TP53", "PTEN",
            "CD19", "MS4A1",
        ]
        self.gene_names = immune_genes

        self.spots = []
        expr_list = []

        for i in range(n_spots):
            x = rng.uniform(10, tissue_w - 10)
            y = rng.uniform(10, tissue_h - 10)

            # Determine tissue zone based on position
            cx, cy = tissue_w / 2, tissue_h / 2
            dist_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            max_dist = np.sqrt(cx ** 2 + cy ** 2)
            rel_dist = dist_from_center / max_dist

            # Zone assignment
            if rel_dist < 0.3:
                zone = "tumor_core"
            elif rel_dist < 0.5:
                zone = "stroma_barrier"
            elif rel_dist < 0.7:
                zone = "immune_border"
            else:
                zone = "normal_tissue"

            # Generate expression based on zone
            expr = np.zeros(len(immune_genes))

            if zone == "tumor_core":
                # High tumor markers, low immune
                expr[immune_genes.index("KRT8")] = rng.exponential(5.0)
                expr[immune_genes.index("KRT18")] = rng.exponential(4.0)
                expr[immune_genes.index("AR")] = rng.exponential(3.0)
                # Some exhausted T cells
                expr[immune_genes.index("CD8A")] = rng.exponential(0.3)
                expr[immune_genes.index("PDCD1")] = rng.exponential(0.8)
                expr[immune_genes.index("TOX")] = rng.exponential(0.6)
                cell_type = "tumor_cell"
                exhaustion = 0.0
                immune = "cold"

            elif zone == "stroma_barrier":
                # High collagen/fibroblast markers
                expr[immune_genes.index("COL1A1")] = rng.exponential(6.0)
                expr[immune_genes.index("COL3A1")] = rng.exponential(5.0)
                expr[immune_genes.index("FAP")] = rng.exponential(4.0)
                expr[immune_genes.index("ACTA2")] = rng.exponential(3.0)
                # M2 macrophages in stroma
                expr[immune_genes.index("CD68")] = rng.exponential(2.0)
                expr[immune_genes.index("CD163")] = rng.exponential(3.0)
                expr[immune_genes.index("ARG1")] = rng.exponential(2.0)
                cell_type = "fibroblast"
                exhaustion = 0.0
                immune = "excluded"

            elif zone == "immune_border":
                # High T cell markers — exhausted
                expr[immune_genes.index("CD8A")] = rng.exponential(4.0)
                expr[immune_genes.index("CD8B")] = rng.exponential(3.0)
                expr[immune_genes.index("PDCD1")] = rng.exponential(3.0)
                expr[immune_genes.index("HAVCR2")] = rng.exponential(2.0)
                expr[immune_genes.index("LAG3")] = rng.exponential(1.5)
                expr[immune_genes.index("TOX")] = rng.exponential(2.5)
                # Some effector function remaining
                expr[immune_genes.index("GZMB")] = rng.exponential(1.0)
                expr[immune_genes.index("IFNG")] = rng.exponential(0.5)
                # M1 macrophages
                expr[immune_genes.index("CD68")] = rng.exponential(2.0)
                expr[immune_genes.index("NOS2")] = rng.exponential(1.5)
                cell_type = "t_cell_exhausted"
                exhaustion = rng.uniform(0.5, 0.9)
                immune = "suppressed"

                # Some hot spots (random)
                if rng.random() < 0.3:
                    expr[immune_genes.index("GZMB")] += rng.exponential(3.0)
                    expr[immune_genes.index("PRF1")] += rng.exponential(2.0)
                    expr[immune_genes.index("IFNG")] += rng.exponential(2.0)
                    exhaustion = rng.uniform(0.1, 0.4)
                    immune = "hot"

            else:  # normal_tissue
                # Normal prostate epithelium
                expr[immune_genes.index("KRT8")] = rng.exponential(3.0)
                expr[immune_genes.index("AR")] = rng.exponential(2.0)
                # Some resident immune cells
                expr[immune_genes.index("CD4")] = rng.exponential(0.5)
                expr[immune_genes.index("CD68")] = rng.exponential(0.5)
                cell_type = "epithelial"
                exhaustion = 0.0
                immune = "normal"

            spot = SpatialSpot(
                barcode=f"spot_{i:04d}",
                x=x, y=y,
                row=int(y // 10),
                col=int(x // 10),
                cell_type=cell_type,
                exhaustion_score=exhaustion,
                immune_score=immune,
            )
            self.spots.append(spot)
            expr_list.append(expr)

        self._expression_matrix = np.array(expr_list)
        logger.info(f"Generated {n_spots} synthetic prostate spatial spots")

    def get_expression(self, gene: str) -> np.ndarray:
        """Get expression values for a gene across all spots."""
        if gene not in self.gene_names:
            return np.zeros(self.n_spots)
        idx = self.gene_names.index(gene)
        if self._expression_matrix is not None:
            return self._expression_matrix[:, idx]
        return np.array([s.gene_expression.get(gene, 0.0) for s in self.spots])

    def get_coordinates(self) -> np.ndarray:
        """Get spot coordinates as (N, 2) array."""
        return np.array([(s.x, s.y) for s in self.spots])

    def classify_regions(self, n_clusters: int = 4) -> List[SpatialRegion]:
        """Classify tissue into spatial regions using expression clustering.

        Uses k-means on expression profiles, then labels clusters
        based on marker gene enrichment.
        """
        if self._expression_matrix is None or self.n_spots == 0:
            return []

        # Simple k-means (no sklearn dependency)
        coords = self.get_coordinates()
        expr = self._expression_matrix

        # Combine spatial + expression features (normalized)
        from numpy.linalg import norm
        coords_norm = coords / (coords.max(axis=0) + 1e-10)
        expr_norm = expr / (expr.max(axis=0, keepdims=True) + 1e-10)

        # Weight spatial less than expression
        features = np.hstack([coords_norm * 0.3, expr_norm])

        # K-means
        labels = self._simple_kmeans(features, n_clusters)

        # Classify each cluster
        self.regions = []
        for k in range(n_clusters):
            mask = labels == k
            cluster_spots = np.where(mask)[0].tolist()
            if not cluster_spots:
                continue

            cluster_expr = expr[mask].mean(axis=0)
            cluster_coords = coords[mask]

            # Determine region type by marker enrichment
            region_type = self._classify_cluster(cluster_expr)

            region = SpatialRegion(
                name=f"Region {k + 1} ({region_type})",
                spots=cluster_spots,
                region_type=region_type,
                center_x=float(cluster_coords[:, 0].mean()),
                center_y=float(cluster_coords[:, 1].mean()),
            )
            self.regions.append(region)

        logger.info(f"Classified {len(self.regions)} spatial regions")
        return self.regions

    def compute_spatial_stats(self, gene: str) -> SpatialStats:
        """Compute spatial autocorrelation (Moran's I) for a gene."""
        expr = self.get_expression(gene)
        coords = self.get_coordinates()

        if len(expr) < 10:
            return SpatialStats(gene, 0.0, 1.0, 0, 0)

        # Compute Moran's I
        n = len(expr)
        mean_expr = expr.mean()
        expr_dev = expr - mean_expr

        # Distance-based weights (inverse distance, k-nearest)
        k = min(6, n - 1)  # 6 nearest neighbors
        W = np.zeros((n, n))

        for i in range(n):
            dists = np.sqrt(((coords - coords[i]) ** 2).sum(axis=1))
            dists[i] = np.inf
            nearest = np.argsort(dists)[:k]
            for j in nearest:
                W[i, j] = 1.0

        # Row-normalize
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        W = W / row_sums

        # Moran's I
        numerator = n * (expr_dev @ W @ expr_dev)
        denominator = W.sum() * (expr_dev ** 2).sum()

        if denominator == 0:
            morans_i = 0.0
        else:
            morans_i = float(numerator / denominator)

        # Approximate p-value (normal approximation)
        expected_i = -1.0 / (n - 1)
        # Variance under null (simplified)
        var_i = 1.0 / n if n > 10 else 0.5
        z_score = (morans_i - expected_i) / max(np.sqrt(var_i), 1e-10)
        # Two-tailed p-value approximation
        p_value = float(2 * (1 - min(1.0, 0.5 * (1 + np.tanh(0.7 * abs(z_score))))))

        # Count hotspots (spots with high local correlation)
        local_scores = np.zeros(n)
        for i in range(n):
            neighbors = np.where(W[i] > 0)[0]
            if len(neighbors) > 0:
                local_scores[i] = expr_dev[i] * expr_dev[neighbors].mean()

        hotspot_threshold = np.percentile(local_scores, 90)
        coldspot_threshold = np.percentile(local_scores, 10)

        return SpatialStats(
            feature=gene,
            morans_i=morans_i,
            p_value=p_value,
            hotspot_count=int((local_scores > hotspot_threshold).sum()),
            coldspot_count=int((local_scores < coldspot_threshold).sum()),
        )

    def get_immune_infiltration_map(self) -> Dict[str, np.ndarray]:
        """Compute immune infiltration scores per spot."""
        cd8_expr = self.get_expression("CD8A") + self.get_expression("CD8B")
        cd4_expr = self.get_expression("CD4")
        treg_expr = self.get_expression("FOXP3")
        macro_expr = self.get_expression("CD68")
        m2_expr = self.get_expression("CD163")

        exhaustion_markers = (
            self.get_expression("PDCD1") +
            self.get_expression("HAVCR2") +
            self.get_expression("LAG3") +
            self.get_expression("TIGIT")
        )

        # Normalize to 0-1 range
        def normalize(arr):
            mx = arr.max()
            return arr / mx if mx > 0 else arr

        return {
            "cd8_t_cells": normalize(cd8_expr),
            "cd4_t_cells": normalize(cd4_expr),
            "tregs": normalize(treg_expr),
            "macrophages": normalize(macro_expr),
            "m2_macrophages": normalize(m2_expr),
            "exhaustion": normalize(exhaustion_markers),
            "total_immune": normalize(cd8_expr + cd4_expr + macro_expr),
        }

    def _simple_kmeans(self, features: np.ndarray, k: int,
                        max_iter: int = 50) -> np.ndarray:
        """Simple k-means clustering (no sklearn dependency)."""
        n = features.shape[0]
        rng = np.random.RandomState(42)

        # Initialize centroids randomly
        idx = rng.choice(n, k, replace=False)
        centroids = features[idx].copy()

        labels = np.zeros(n, dtype=int)

        for _ in range(max_iter):
            # Assign
            for i in range(n):
                dists = np.sqrt(((centroids - features[i]) ** 2).sum(axis=1))
                labels[i] = np.argmin(dists)

            # Update
            new_centroids = np.zeros_like(centroids)
            for c in range(k):
                mask = labels == c
                if mask.sum() > 0:
                    new_centroids[c] = features[mask].mean(axis=0)
                else:
                    new_centroids[c] = centroids[c]

            if np.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        return labels

    def _classify_cluster(self, mean_expr: np.ndarray) -> str:
        """Classify a cluster by its mean expression profile."""
        genes = self.gene_names

        def gene_val(name):
            if name in genes:
                return mean_expr[genes.index(name)]
            return 0.0

        # Immune markers
        cd8 = gene_val("CD8A") + gene_val("CD8B")
        cd4 = gene_val("CD4")
        macro = gene_val("CD68")
        collagen = gene_val("COL1A1") + gene_val("COL3A1")
        tumor = gene_val("KRT8") + gene_val("KRT18")
        exhaustion = gene_val("PDCD1") + gene_val("TOX")

        immune_total = cd8 + cd4 + macro

        if tumor > immune_total and tumor > collagen:
            if cd8 > 1.0 and exhaustion > 1.0:
                return "immune_suppressed"
            return "immune_cold"
        elif collagen > immune_total:
            return "stroma"
        elif cd8 > 2.0 and exhaustion < cd8 * 0.5:
            return "immune_hot"
        elif cd8 > 1.0 and exhaustion > 1.0:
            return "immune_exhausted"
        elif immune_total > 1.0:
            return "immune_mixed"
        else:
            return "normal"
