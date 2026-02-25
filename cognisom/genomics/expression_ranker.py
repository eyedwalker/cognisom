"""
Expression Ranker
=================

Convert scRNA-seq gene expression vectors into rank-ordered
"gene sentences" — the input format for Cell2Sentence models.

The Cell2Sentence encoding represents each cell as a space-separated
list of gene names ordered by descending expression level.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ExpressionRanker:
    """Convert gene expression data to Cell2Sentence format.

    Cell2Sentence (Levine et al., Yale + Google) encodes single-cell
    gene expression as "cell sentences": gene symbols ranked by
    expression level, space-separated.

    Example:
        ranker = ExpressionRanker()

        # From raw expression vector
        gene_names = ["GAPDH", "CD3E", "PDCD1", "HAVCR2", "TOX"]
        expression = np.array([8.5, 6.2, 4.1, 3.8, 5.0])
        sentence = ranker.rank_cell(gene_names, expression)
        # → "GAPDH CD3E TOX PDCD1 HAVCR2"

        # From AnnData object
        sentences = ranker.rank_adata(adata, max_genes=200)
    """

    def __init__(self, max_genes: int = 200, min_expression: float = 0.0):
        """
        Args:
            max_genes: Maximum number of genes to include in sentence.
            min_expression: Minimum expression threshold (genes below are excluded).
        """
        self.max_genes = max_genes
        self.min_expression = min_expression

    def rank_cell(self, gene_names: List[str],
                  expression: np.ndarray) -> str:
        """Convert a single cell's expression to a gene sentence.

        Args:
            gene_names: List of gene symbols (same order as expression).
            expression: Expression values (raw counts, log-normalized, etc.).

        Returns:
            Space-separated gene sentence, descending by expression.
        """
        if len(gene_names) != len(expression):
            raise ValueError(
                f"gene_names ({len(gene_names)}) and expression "
                f"({len(expression)}) must have same length"
            )

        # Filter by minimum expression
        mask = expression > self.min_expression
        filtered_genes = [g for g, m in zip(gene_names, mask) if m]
        filtered_expr = expression[mask]

        if len(filtered_genes) == 0:
            return ""

        # Sort by descending expression
        order = np.argsort(-filtered_expr)
        ranked = [filtered_genes[i] for i in order[:self.max_genes]]

        return " ".join(ranked)

    def rank_adata(self, adata, max_genes: Optional[int] = None,
                   cell_indices: Optional[List[int]] = None,
                   layer: Optional[str] = None) -> List[str]:
        """Convert cells in an AnnData object to gene sentences.

        Args:
            adata: AnnData object with .X expression matrix.
            max_genes: Override max genes per sentence.
            cell_indices: Specific cell indices to convert (None = all).
            layer: Use a specific layer instead of .X.

        Returns:
            List of gene sentences (one per cell).
        """
        gene_names = list(adata.var_names)
        mg = max_genes or self.max_genes

        # Get expression matrix
        if layer and layer in adata.layers:
            X = adata.layers[layer]
        else:
            X = adata.X

        # Handle sparse matrices
        import scipy.sparse as sp
        if sp.issparse(X):
            X = X.toarray()

        indices = cell_indices if cell_indices is not None else range(X.shape[0])
        sentences = []

        for i in indices:
            expr = np.asarray(X[i]).flatten()
            sentence = self.rank_cell(gene_names, expr)
            sentences.append(sentence)

        logger.info(
            f"Ranked {len(sentences)} cells into gene sentences "
            f"(max {mg} genes each)"
        )
        return sentences

    def rank_with_metadata(self, adata,
                           cell_indices: Optional[List[int]] = None,
                           ) -> List[Dict]:
        """Rank cells and include metadata (cell type, etc.).

        Returns list of dicts with 'sentence', 'cell_type', 'index'.
        """
        sentences = self.rank_adata(adata, cell_indices=cell_indices)
        indices = cell_indices if cell_indices is not None else list(range(len(sentences)))

        results = []
        for i, (idx, sent) in enumerate(zip(indices, sentences)):
            meta = {"sentence": sent, "index": idx}

            # Extract common metadata from .obs
            if "cell_type" in adata.obs.columns:
                meta["cell_type"] = str(adata.obs.iloc[idx]["cell_type"])
            if "leiden" in adata.obs.columns:
                meta["cluster"] = str(adata.obs.iloc[idx]["leiden"])

            results.append(meta)

        return results
