"""
NCBI Integration Module
=======================

Centralized client for NCBI E-utilities, BLAST, ClinVar, and PubChem APIs.
Uses NCBI_API_KEY from environment for 10 req/sec rate limit (vs 3 without).
"""

from .client import NCBIClient
from .sequences import fetch_sequence, search_gene_sequences
from .blast import submit_blast, check_blast_status, get_blast_results
from .clinvar import search_clinvar
from .pubchem import search_compound, get_compound_properties, similarity_search

__all__ = [
    "NCBIClient",
    "fetch_sequence",
    "search_gene_sequences",
    "submit_blast",
    "check_blast_status",
    "get_blast_results",
    "search_clinvar",
    "search_compound",
    "get_compound_properties",
    "similarity_search",
]
