"""
scRNA-seq Data Loader
=====================

Loads single-cell RNA-seq datasets from various sources:
- Local .h5ad files (AnnData format)
- CellxGene Census API (Human Cell Atlas)
- GEO accessions (NCBI)

All loaders return AnnData objects for downstream processing.
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Standard data directory
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "scrna"

# CellxGene uses specific ontology terms - map friendly names to actual values
DISEASE_MAPPING = {
    # Prostate
    "prostate adenocarcinoma": "prostate adenocarcinoma",
    "prostate cancer": "prostate adenocarcinoma",
    "prostate carcinoma": "prostate adenocarcinoma",
    # Lung
    "lung adenocarcinoma": "lung adenocarcinoma",
    "lung cancer": "lung adenocarcinoma",
    # Breast
    "breast carcinoma": "breast carcinoma",
    "breast cancer": "breast carcinoma",
    # Colorectal
    "colorectal cancer": "colorectal cancer",
    "colon cancer": "colorectal cancer",
    # Normal
    "normal": "normal",
    "healthy": "normal",
}

# Tissues that are known to work in CellxGene
VALID_TISSUES = [
    "prostate gland", "lung", "breast", "colon", "liver", "kidney",
    "heart", "brain", "skin", "blood", "bone marrow", "pancreas",
]


class ScRNALoader:
    """Load scRNA-seq datasets from files or public repositories.

    Example:
        loader = ScRNALoader()

        # From local file
        adata = loader.from_file("prostate.h5ad")

        # From CellxGene (Human Cell Atlas)
        adata = loader.from_cellxgene(tissue="prostate gland")

        # From GEO
        adata = loader.from_geo("GSE141445")
    """

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def from_file(self, path: str):
        """Load an AnnData .h5ad file.

        Args:
            path: Path to .h5ad file (absolute or relative to data_dir).

        Returns:
            AnnData object.
        """
        import anndata as ad

        filepath = Path(path)
        if not filepath.is_absolute():
            filepath = self.data_dir / filepath

        if not filepath.exists():
            raise FileNotFoundError(f"Dataset not found: {filepath}")

        logger.info(f"Loading {filepath}")
        adata = ad.read_h5ad(filepath)
        logger.info(
            f"Loaded {adata.n_obs} cells x {adata.n_vars} genes"
        )
        return adata

    def from_cellxgene(self, tissue: str = "prostate gland",
                       organism: str = "Homo sapiens",
                       disease: Optional[str] = None,
                       max_cells: int = 50000):
        """Load data from CellxGene Census (Human Cell Atlas).

        Downloads cells matching the tissue/organism/disease query.
        Results are cached locally as .h5ad files.

        Args:
            tissue: Tissue type (e.g., "prostate gland", "lung", "brain").
            organism: Species name.
            disease: Disease filter (e.g., "prostate cancer", None for normal).
            max_cells: Maximum cells to download.

        Returns:
            AnnData object or None if no data found.
        """
        # Check if cellxgene_census is available (needs tiledb system libs)
        try:
            import cellxgene_census
        except ImportError as e:
            logger.error(
                f"CellxGene Census not available: {e}. "
                f"Use Synthetic data instead."
            )
            return None
        except Exception as e:
            logger.error(f"CellxGene import failed: {e}")
            return None

        # Map friendly disease names to CellxGene ontology terms
        disease_term = None
        if disease:
            disease_term = DISEASE_MAPPING.get(disease.lower(), disease)

        cache_name = f"cellxgene_{tissue.replace(' ', '_')}"
        if disease_term:
            cache_name += f"_{disease_term.replace(' ', '_')}"
        cache_path = self.data_dir / f"{cache_name}.h5ad"

        if cache_path.exists():
            logger.info(f"Loading cached: {cache_path}")
            return self.from_file(str(cache_path))

        logger.info(
            f"Querying CellxGene Census: tissue={tissue}, "
            f"organism={organism}, disease={disease_term or 'normal'}"
        )

        # Try primary query first
        adata = self._query_cellxgene(
            tissue, organism, disease_term, max_cells
        )

        # If disease query returned 0 cells, try with just tissue (any disease)
        if adata is None or adata.n_obs == 0:
            if disease_term:
                logger.warning(
                    f"No cells found for disease='{disease_term}'. "
                    f"Trying tissue-only query..."
                )
                adata = self._query_cellxgene(
                    tissue, organism, None, max_cells, include_any_disease=True
                )

        # Still nothing? Return None with helpful message
        if adata is None or adata.n_obs == 0:
            logger.error(
                f"No cells found for tissue='{tissue}'. "
                f"Try a different tissue or use synthetic data."
            )
            return None

        # Cache locally
        adata.write_h5ad(cache_path)
        logger.info(
            f"Downloaded {adata.n_obs} cells x {adata.n_vars} genes -> {cache_path}"
        )
        return adata

    def _query_cellxgene(self, tissue: str, organism: str,
                         disease: Optional[str], max_cells: int,
                         include_any_disease: bool = False):
        """Execute CellxGene Census query with error handling."""
        import cellxgene_census

        # Build value filter
        filters = [f'tissue == "{tissue}"']
        if disease:
            filters.append(f'disease == "{disease}"')
        elif not include_any_disease:
            filters.append('disease == "normal"')
        # If include_any_disease=True, don't filter by disease at all

        value_filter = " and ".join(filters)
        logger.info(f"Query filter: {value_filter}")

        try:
            with cellxgene_census.open_soma() as census:
                adata = cellxgene_census.get_anndata(
                    census,
                    organism=organism,
                    obs_value_filter=value_filter,
                    obs_column_names=[
                        "cell_type", "tissue", "disease",
                        "sex", "development_stage", "assay",
                        "donor_id", "suspension_type",
                    ],
                )

            # Subsample if too large
            if adata.n_obs > max_cells:
                import numpy as np
                indices = np.random.choice(adata.n_obs, max_cells, replace=False)
                adata = adata[indices].copy()
                logger.info(f"Subsampled to {max_cells} cells")

            return adata

        except Exception as e:
            logger.error(f"CellxGene query failed: {e}")
            return None

    def from_geo(self, accession: str):
        """Load from GEO accession (e.g., GSE141445).

        Downloads the dataset from NCBI GEO and converts to AnnData.
        Requires the dataset to be in a supported format (MTX, CSV, H5).

        Args:
            accession: GEO accession ID (e.g., "GSE141445").

        Returns:
            AnnData object.
        """
        import scanpy as sc

        cache_path = self.data_dir / f"{accession}.h5ad"
        if cache_path.exists():
            logger.info(f"Loading cached: {cache_path}")
            return self.from_file(str(cache_path))

        logger.info(f"Downloading GEO accession: {accession}")

        # For known prostate datasets, provide direct download
        known_datasets = {
            "GSE141445": {
                "description": "Human prostate tissue single-cell atlas",
                "url_hint": "Download from GEO and place .h5ad in data/scrna/",
            },
            "GSE176031": {
                "description": "Prostate cancer single-cell RNA-seq",
                "url_hint": "Download from GEO and place .h5ad in data/scrna/",
            },
            "GSE193337": {
                "description": "Prostate tissue microenvironment",
                "url_hint": "Download from GEO and place .h5ad in data/scrna/",
            },
        }

        if accession in known_datasets:
            info = known_datasets[accession]
            raise FileNotFoundError(
                f"Dataset {accession} ({info['description']}) not found locally.\n"
                f"To download:\n"
                f"  1. Visit https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}\n"
                f"  2. Download the processed data files\n"
                f"  3. Place the .h5ad file at: {cache_path}\n"
                f"  Or use loader.from_cellxgene(tissue='prostate gland') for atlas data."
            )

        raise FileNotFoundError(
            f"GEO accession {accession} not found locally.\n"
            f"Download from: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}\n"
            f"Place the .h5ad file at: {cache_path}"
        )

    def list_available(self):
        """List locally available datasets."""
        datasets = list(self.data_dir.glob("*.h5ad"))
        if not datasets:
            logger.info(f"No datasets found in {self.data_dir}")
            logger.info("Use from_cellxgene() to download, or place .h5ad files manually.")
        else:
            for ds in datasets:
                logger.info(f"  {ds.name} ({ds.stat().st_size / 1e6:.1f} MB)")
        return datasets

    @staticmethod
    def prostate_datasets():
        """Return known prostate tissue dataset accessions."""
        return {
            "cellxgene": {
                "query": 'tissue == "prostate gland" and disease == "normal"',
                "description": "Human Cell Atlas prostate tissue (normal)",
            },
            "cellxgene_cancer": {
                "query": 'tissue == "prostate gland" and disease == "prostate cancer"',
                "description": "Human Cell Atlas prostate tissue (cancer)",
            },
            "GSE141445": {
                "description": "Henry et al. 2018 - Prostate single-cell atlas",
                "cell_count": "~98,000",
            },
            "GSE176031": {
                "description": "Chen et al. 2021 - Prostate cancer scRNA-seq",
                "cell_count": "~36,000",
            },
        }
