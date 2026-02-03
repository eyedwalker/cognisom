"""
MSA-Search NIM Client
=====================

GPU-accelerated multiple sequence alignment from a query protein sequence.
Searches sequence databases for homologs and builds an alignment (a3m format)
that feeds into structure prediction NIMs (OpenFold3, AlphaFold2).
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from .client import NIMClient

logger = logging.getLogger(__name__)

ENDPOINT = "colabfold/msa-search"


@dataclass
class MSAResult:
    """Multiple sequence alignment result."""
    query_sequence: str
    alignment: str          # a3m-formatted alignment text
    num_sequences: int      # Number of homologs found
    databases_searched: List[str]


class MSASearchClient(NIMClient):
    """Client for MSA-Search NIM.

    Runs GPU-accelerated multiple sequence alignment search,
    finding evolutionary homologs for a query protein sequence.
    The resulting MSA provides co-evolutionary signals that dramatically
    improve protein structure prediction accuracy.

    Example:
        client = MSASearchClient()
        msa = client.search("MKFLILLFNILCLFPVLAADNHGVS...")
        print(f"Found {msa.num_sequences} homologs")

        # Feed into structure prediction
        from cognisom.nims import OpenFold3Client
        of3 = OpenFold3Client()
        structure = of3.predict_structure(["MKFL..."], msa_data=msa.alignment)
    """

    def search(self, sequence: str,
               databases: Optional[List[str]] = None) -> MSAResult:
        """Run MSA search for a protein sequence.

        Args:
            sequence: Protein amino acid sequence.
            databases: Optional list of databases to search
                       (default: UniRef, environmental sequences).

        Returns:
            MSAResult with a3m alignment and metadata.
        """
        payload = {"sequence": sequence}
        if databases:
            payload["databases"] = databases

        result = self._post(ENDPOINT, payload, timeout=300)

        alignment = result.get("alignment", result.get("a3m", ""))
        num_seqs = result.get("num_sequences", alignment.count(">"))
        dbs = result.get("databases_searched",
                         databases or ["UniRef30", "environmental"])

        msa = MSAResult(
            query_sequence=sequence,
            alignment=alignment,
            num_sequences=num_seqs,
            databases_searched=dbs,
        )
        logger.info(f"MSA search found {msa.num_sequences} homologs "
                     f"for {len(sequence)}-residue query")
        return msa

    def search_for_structure_prediction(self, sequence: str) -> str:
        """Search and return raw a3m for feeding into OpenFold3/AlphaFold2.

        Convenience method that returns just the alignment string.
        """
        msa = self.search(sequence)
        return msa.alignment
