"""
scRNA-seq Data Ingestion
========================

Load, process, and convert single-cell RNA-seq datasets
into cognisom simulation initial conditions.

Pipeline:
    1. Load raw data (AnnData .h5ad files)
    2. Preprocess (QC, normalize, cluster)
    3. Extract cell archetypes (marker genes, proportions)
    4. Bridge to simulation (archetypes -> CellState objects)
"""

from .loader import ScRNALoader
from .preprocessor import ScRNAPreprocessor
from .archetypes import ArchetypeExtractor
from .single_cell_bridge import SingleCellBridge

__all__ = [
    'ScRNALoader',
    'ScRNAPreprocessor',
    'ArchetypeExtractor',
    'SingleCellBridge',
]
