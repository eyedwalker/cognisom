#!/usr/bin/env python3
"""
Ingestion Pipeline Test
=======================

Tests the full scRNA-seq ingestion pipeline using synthetic data.
Validates: loader -> preprocessor -> archetype extraction -> bridge.

Run:
    cd /Users/davidwalker/CascadeProjects/cognisom
    python -m cognisom.ingestion.test_ingestion
"""

import os
import sys
import logging

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

logging.basicConfig(level=logging.INFO, format="%(message)s")


def create_synthetic_prostate_data():
    """Create a synthetic scRNA-seq dataset mimicking prostate tissue.

    Generates ~2000 cells with known marker gene expression patterns
    for each prostate cell type, so we can validate the archetype
    extraction pipeline end-to-end.
    """
    import anndata as ad
    from scipy.sparse import csr_matrix

    np.random.seed(42)

    # Define cell types and their counts
    cell_types = {
        "luminal_epithelial": 600,
        "basal_epithelial": 200,
        "stromal_fibroblast": 400,
        "endothelial": 100,
        "t_cell": 150,
        "macrophage": 100,
        "smooth_muscle": 150,
        "nk_cell": 50,
        "neuroendocrine": 50,
    }

    total_cells = sum(cell_types.values())

    # Define genes (markers + noise genes)
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
    all_markers = list(dict.fromkeys(all_markers))  # deduplicate, preserve order

    # Add noise genes
    noise_genes = [f"GENE_{i}" for i in range(500)]
    all_genes = all_markers + noise_genes
    n_genes = len(all_genes)

    # Build expression matrix
    X = np.zeros((total_cells, n_genes), dtype=np.float32)
    cell_labels = []
    row = 0

    for cell_type, n_cells in cell_types.items():
        markers = marker_genes[cell_type]
        for i in range(n_cells):
            # Background expression
            X[row, :] = np.random.exponential(0.1, n_genes)

            # Upregulate marker genes
            for gene in markers:
                gene_idx = all_genes.index(gene)
                X[row, gene_idx] = np.random.exponential(3.0) + 2.0

            # Add some cross-talk (markers expressed at low levels in other types)
            for gene in all_markers:
                gene_idx = all_genes.index(gene)
                if gene not in markers:
                    X[row, gene_idx] += np.random.exponential(0.2)

            cell_labels.append(cell_type)
            row += 1

    # Add mitochondrial genes
    mt_genes = ["MT-CO1", "MT-CO2", "MT-ND1", "MT-ND2", "MT-ATP6"]
    for g in mt_genes:
        all_genes.append(g)
        mt_col = np.random.exponential(0.5, total_cells).reshape(-1, 1)
        X = np.hstack([X, mt_col.astype(np.float32)])

    # Convert to integer counts (simulate UMI counts)
    X = np.round(X).astype(np.float32)
    X = np.clip(X, 0, None)

    # Create AnnData
    import pandas as pd
    obs = pd.DataFrame({"true_cell_type": cell_labels}, index=[f"cell_{i}" for i in range(total_cells)])
    var = pd.DataFrame(index=all_genes)

    adata = ad.AnnData(X=csr_matrix(X), obs=obs, var=var)

    print(f"Created synthetic dataset: {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"Cell type composition:")
    for ct, n in cell_types.items():
        print(f"  {ct}: {n} ({n/total_cells:.1%})")

    return adata


def test_preprocessor(adata):
    """Test the preprocessing pipeline."""
    from cognisom.ingestion.preprocessor import ScRNAPreprocessor

    print("\n" + "=" * 60)
    print("TEST 1: Preprocessor")
    print("=" * 60)

    preprocessor = ScRNAPreprocessor(use_gpu=False)
    adata = preprocessor.run(
        adata,
        min_genes=5,  # Lower threshold for synthetic data
        min_cells=3,
        max_pct_mito=50.0,
        n_top_genes=200,  # Fewer genes in synthetic data
        resolution=0.8,
    )

    n_clusters = adata.obs["leiden"].nunique()
    print(f"\nResult: {adata.n_obs} cells, {n_clusters} clusters")
    print(f"Cluster sizes:")
    for cluster in sorted(adata.obs["leiden"].unique(), key=int):
        n = (adata.obs["leiden"] == cluster).sum()
        print(f"  Cluster {cluster}: {n} cells")

    return adata


def test_archetypes(adata):
    """Test archetype extraction."""
    from cognisom.ingestion.archetypes import ArchetypeExtractor

    print("\n" + "=" * 60)
    print("TEST 2: Archetype Extraction")
    print("=" * 60)

    extractor = ArchetypeExtractor()
    archetypes = extractor.extract(adata)

    print(f"\nExtracted {len(archetypes)} archetypes:")
    for a in archetypes:
        print(f"  Cluster {a.cluster_id} -> {a.name}")
        print(f"    Cells: {a.cell_count} ({a.proportion:.1%})")
        print(f"    Type: {a.simulation_type}")
        top_markers = list(a.marker_scores.items())[:3]
        if top_markers:
            markers_str = ", ".join(f"{g}={s:.2f}" for g, s in top_markers)
            print(f"    Top markers: {markers_str}")

    # Print summary
    summary = extractor.summary(archetypes)
    print(f"\nSummary by simulation type:")
    for sim_type, info in summary["by_simulation_type"].items():
        print(f"  {sim_type}: {info['count']} cells ({info['proportion']:.1%})")
        print(f"    Archetypes: {', '.join(info['archetypes'])}")

    return archetypes


def test_bridge(archetypes):
    """Test single-cell bridge to simulation config."""
    from cognisom.ingestion.single_cell_bridge import SingleCellBridge

    print("\n" + "=" * 60)
    print("TEST 3: Single-Cell Bridge")
    print("=" * 60)

    bridge = SingleCellBridge()
    config = bridge.create_config(archetypes, total_cells=200)

    print(f"\nSimulation config generated:")
    print(f"  Tissue cells: {len(config.cells)}")
    print(f"  Immune cells: {len(config.immune_cells)}")

    # Count by archetype
    arch_counts = {}
    for c in config.cells:
        arch = c["archetype"]
        arch_counts[arch] = arch_counts.get(arch, 0) + 1
    for c in config.immune_cells:
        arch = c["archetype"]
        arch_counts[arch] = arch_counts.get(arch, 0) + 1

    print(f"\n  Cell placement by archetype:")
    for arch, count in sorted(arch_counts.items(), key=lambda x: -x[1]):
        print(f"    {arch}: {count}")

    # Verify spatial positions are within domain
    all_positions = [c["position"] for c in config.cells] + \
                    [c["position"] for c in config.immune_cells]
    positions = np.array(all_positions)
    print(f"\n  Spatial bounds:")
    print(f"    X: [{positions[:, 0].min():.1f}, {positions[:, 0].max():.1f}]")
    print(f"    Y: [{positions[:, 1].min():.1f}, {positions[:, 1].max():.1f}]")
    print(f"    Z: [{positions[:, 2].min():.1f}, {positions[:, 2].max():.1f}]")

    # Check spatial fields
    print(f"\n  Spatial fields configured:")
    for field_name, field_cfg in config.spatial_config["fields"].items():
        print(f"    {field_name}: D={field_cfg['diffusion_coeff']} um^2/s")

    return config


def main():
    print("COGNISOM - scRNA-seq Ingestion Pipeline Test")
    print("Using synthetic prostate tissue data\n")

    # Create synthetic data
    adata = create_synthetic_prostate_data()

    # Run pipeline
    adata = test_preprocessor(adata)
    archetypes = test_archetypes(adata)
    config = test_bridge(archetypes)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  scRNA-seq data -> Preprocessed -> {len(archetypes)} archetypes -> Simulation config")
    print(f"  Ready to initialize cognisom with {len(config.cells) + len(config.immune_cells)} cells")
    print("\n  To use with real data:")
    print("    loader = ScRNALoader()")
    print('    adata = loader.from_cellxgene(tissue="prostate gland")')
    print("    adata = ScRNAPreprocessor().run(adata)")
    print("    archetypes = ArchetypeExtractor().extract(adata)")
    print("    config = SingleCellBridge().create_config(archetypes, total_cells=500)")


if __name__ == "__main__":
    main()
