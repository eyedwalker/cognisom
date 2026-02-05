"""
Cell Archetype Extractor
========================

Extracts biologically meaningful cell archetypes from clustered scRNA-seq data.
Maps Leiden clusters to known prostate tissue cell types using marker genes.

Prostate Tissue Archetypes:
    - Luminal epithelial (KLK3, AR, NKX3-1, KRT8, KRT18)
    - Basal epithelial (TP63, KRT5, KRT14)
    - Neuroendocrine (CHGA, SYP, ENO2)
    - Stromal fibroblast (VIM, COL1A1, DCN, LUM)
    - Smooth muscle (ACTA2, MYH11, TAGLN)
    - Endothelial (PECAM1, VWF, CDH5)
    - T cell (CD3D, CD3E, CD8A, CD4)
    - Macrophage (CD68, CD163, CSF1R)
    - NK cell (NKG7, GNLY, KLRD1)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Canonical marker genes for prostate tissue cell types
PROSTATE_MARKERS = {
    "luminal_epithelial": {
        "markers": ["KLK3", "AR", "NKX3-1", "KRT8", "KRT18", "ACPP", "MSMB"],
        "simulation_type": "normal",  # maps to CellState.cell_type
        "metabolism": "oxidative",
        "oxygen_consumption": 0.15,
        "glucose_consumption": 0.2,
        "division_time": 48.0,  # hours (slow-cycling)
    },
    "basal_epithelial": {
        "markers": ["TP63", "KRT5", "KRT14", "KRT15", "ITGA6", "ITGB4"],
        "simulation_type": "normal",
        "metabolism": "oxidative",
        "oxygen_consumption": 0.12,
        "glucose_consumption": 0.18,
        "division_time": 72.0,  # hours (stem-like, slow)
    },
    "neuroendocrine": {
        "markers": ["CHGA", "SYP", "ENO2", "NCAM1", "ASCL1"],
        "simulation_type": "normal",
        "metabolism": "glycolytic",
        "oxygen_consumption": 0.08,
        "glucose_consumption": 0.3,
        "division_time": 120.0,  # hours (rare, slow)
    },
    "stromal_fibroblast": {
        "markers": ["VIM", "COL1A1", "COL3A1", "DCN", "LUM", "PDGFRA"],
        "simulation_type": "normal",
        "metabolism": "oxidative",
        "oxygen_consumption": 0.10,
        "glucose_consumption": 0.15,
        "division_time": 96.0,
    },
    "smooth_muscle": {
        "markers": ["ACTA2", "MYH11", "TAGLN", "CNN1", "DES"],
        "simulation_type": "normal",
        "metabolism": "oxidative",
        "oxygen_consumption": 0.12,
        "glucose_consumption": 0.15,
        "division_time": 168.0,  # rarely divides
    },
    "endothelial": {
        "markers": ["PECAM1", "VWF", "CDH5", "ERG", "FLT1", "KDR"],
        "simulation_type": "normal",
        "metabolism": "glycolytic",
        "oxygen_consumption": 0.08,
        "glucose_consumption": 0.25,
        "division_time": 120.0,
    },
    "t_cell": {
        "markers": ["CD3D", "CD3E", "CD8A", "CD4", "TRAC", "IL7R"],
        "simulation_type": "immune",
        "immune_subtype": "T_cell",
        "speed": 10.0,
        "detection_radius": 10.0,
    },
    "macrophage": {
        "markers": ["CD68", "CD163", "CSF1R", "MARCO", "MSR1", "MRC1"],
        "simulation_type": "immune",
        "immune_subtype": "macrophage",
        "speed": 5.0,
        "detection_radius": 15.0,
    },
    "nk_cell": {
        "markers": ["NKG7", "GNLY", "KLRD1", "NCAM1", "FCGR3A"],
        "simulation_type": "immune",
        "immune_subtype": "NK_cell",
        "speed": 12.0,
        "detection_radius": 8.0,
    },
    "cancer_epithelial": {
        "markers": ["MKI67", "TOP2A", "PCNA", "MCM2", "CDK1"],
        "simulation_type": "cancer",
        "metabolism": "glycolytic",  # Warburg effect
        "oxygen_consumption": 0.10,
        "glucose_consumption": 0.50,
        "division_time": 12.0,
        "mutations": ["KRAS_G12D"],
    },
}


@dataclass
class CellArchetype:
    """A cell archetype extracted from scRNA-seq clustering."""
    name: str
    cluster_id: str  # Leiden cluster label
    cell_count: int
    proportion: float  # Fraction of total cells
    simulation_type: str  # 'normal', 'cancer', 'immune'

    # Top marker genes and their mean expression
    marker_scores: Dict[str, float] = field(default_factory=dict)
    top_genes: List[str] = field(default_factory=list)

    # Expression centroid (mean expression profile for this archetype)
    centroid: Optional[np.ndarray] = None

    # Simulation parameters
    params: Dict = field(default_factory=dict)


class ArchetypeExtractor:
    """Extract cell archetypes from preprocessed scRNA-seq data.

    Maps Leiden clusters to known cell types using marker gene scoring,
    then packages each archetype with simulation-ready parameters.

    Example:
        extractor = ArchetypeExtractor()
        archetypes = extractor.extract(adata)  # adata must have .obs['leiden']
        for a in archetypes:
            print(f"{a.name}: {a.cell_count} cells ({a.proportion:.1%})")
    """

    def __init__(self, markers: Optional[Dict] = None):
        self.markers = markers or PROSTATE_MARKERS

    def extract(self, adata, cluster_key: str = "leiden") -> List[CellArchetype]:
        """Extract archetypes from clustered AnnData.

        Args:
            adata: Preprocessed AnnData with clusters in .obs[cluster_key].
            cluster_key: Column in .obs with cluster labels.

        Returns:
            List of CellArchetype, one per identified cell type.
        """
        import scanpy as sc

        if cluster_key not in adata.obs.columns:
            raise ValueError(
                f"Cluster column '{cluster_key}' not found. "
                f"Run ScRNAPreprocessor.run() first."
            )

        clusters = adata.obs[cluster_key].unique()
        total_cells = adata.n_obs
        archetypes = []

        logger.info(f"Scoring {len(clusters)} clusters against {len(self.markers)} archetypes")

        for cluster_id in sorted(clusters, key=lambda x: int(x)):
            cluster_mask = adata.obs[cluster_key] == cluster_id
            cluster_adata = adata[cluster_mask]
            n_cells = cluster_adata.n_obs

            # Score this cluster against all known archetypes
            best_name, best_score, marker_scores = self._score_cluster(
                cluster_adata, adata.var_names
            )

            # Get archetype parameters
            archetype_info = self.markers.get(best_name, {})

            archetype = CellArchetype(
                name=best_name,
                cluster_id=str(cluster_id),
                cell_count=n_cells,
                proportion=n_cells / total_cells,
                simulation_type=archetype_info.get("simulation_type", "normal"),
                marker_scores=marker_scores,
                top_genes=self._top_expressed_genes(cluster_adata, n=10),
                params=self._archetype_params(best_name),
            )

            # Compute expression centroid
            if hasattr(cluster_adata.X, "toarray"):
                archetype.centroid = np.array(
                    cluster_adata.X.toarray().mean(axis=0)
                ).flatten()
            else:
                archetype.centroid = np.array(
                    cluster_adata.X.mean(axis=0)
                ).flatten()

            archetypes.append(archetype)
            logger.info(
                f"  Cluster {cluster_id} -> {best_name} "
                f"({n_cells} cells, {n_cells/total_cells:.1%}, score={best_score:.3f})"
            )

        return archetypes

    def _score_cluster(self, cluster_adata, all_genes) -> Tuple[str, float, Dict]:
        """Score a cluster against all marker gene sets.

        Returns (best_archetype_name, best_score, {marker: score}).
        """
        best_name = "unknown"
        best_score = -1.0
        best_markers = {}

        gene_set = set(all_genes)

        for arch_name, arch_info in self.markers.items():
            markers = arch_info["markers"]
            present = [g for g in markers if g in gene_set]

            if not present:
                continue

            # Mean expression of marker genes in this cluster
            scores = {}
            total_score = 0.0
            for gene in present:
                gene_idx = list(all_genes).index(gene)
                if hasattr(cluster_adata.X, "toarray"):
                    expr = cluster_adata.X.toarray()[:, gene_idx]
                else:
                    expr = cluster_adata.X[:, gene_idx]
                mean_expr = float(np.mean(expr))
                pct_expressing = float(np.mean(expr > 0))
                # Combined score: expression level * fraction of cells expressing
                score = mean_expr * pct_expressing
                scores[gene] = score
                total_score += score

            # Normalize by number of markers checked
            avg_score = total_score / len(present)

            if avg_score > best_score:
                best_score = avg_score
                best_name = arch_name
                best_markers = scores

        return best_name, best_score, best_markers

    def _top_expressed_genes(self, cluster_adata, n: int = 10) -> List[str]:
        """Get the top N most expressed genes in this cluster."""
        if hasattr(cluster_adata.X, "toarray"):
            mean_expr = np.array(cluster_adata.X.toarray().mean(axis=0)).flatten()
        else:
            mean_expr = np.array(cluster_adata.X.mean(axis=0)).flatten()

        top_idx = np.argsort(mean_expr)[-n:][::-1]
        return [cluster_adata.var_names[i] for i in top_idx]

    def _archetype_params(self, archetype_name: str) -> Dict:
        """Get simulation parameters for an archetype."""
        info = self.markers.get(archetype_name, {})
        params = {}

        # Copy relevant simulation parameters
        for key in ["metabolism", "oxygen_consumption", "glucose_consumption",
                     "division_time", "mutations", "immune_subtype",
                     "speed", "detection_radius"]:
            if key in info:
                params[key] = info[key]

        return params

    def summary(self, archetypes: List[CellArchetype]) -> Dict:
        """Generate a summary of extracted archetypes.

        Returns:
            Dict with counts by type, proportions, etc.
        """
        total = sum(a.cell_count for a in archetypes)
        by_type = {}
        for a in archetypes:
            sim_type = a.simulation_type
            if sim_type not in by_type:
                by_type[sim_type] = {"count": 0, "archetypes": []}
            by_type[sim_type]["count"] += a.cell_count
            by_type[sim_type]["archetypes"].append(a.name)

        return {
            "total_cells": total,
            "n_archetypes": len(archetypes),
            "by_simulation_type": {
                k: {
                    "count": v["count"],
                    "proportion": v["count"] / total,
                    "archetypes": v["archetypes"],
                }
                for k, v in by_type.items()
            },
            "archetypes": [
                {
                    "name": a.name,
                    "cluster": a.cluster_id,
                    "cells": a.cell_count,
                    "proportion": a.proportion,
                    "type": a.simulation_type,
                }
                for a in archetypes
            ],
        }
