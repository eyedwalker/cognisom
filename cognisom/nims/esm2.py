"""
ESM2 NIM Client
===============

Protein sequence embeddings using Meta's ESM2-650m language model.
Converts amino acid sequences into dense vector representations
for downstream analysis: mutation impact scoring, similarity search,
functional clustering.
"""

import logging
import math
from dataclasses import dataclass
from typing import List, Optional

from .client import NIMClient

logger = logging.getLogger(__name__)

ENDPOINT = "meta/esm2-650m"


@dataclass
class ProteinEmbedding:
    """Embedding from ESM2-650m."""
    sequence: str
    embedding: List[float]
    dimension: int


class ESM2Client(NIMClient):
    """Client for ESM2-650m protein language model NIM.

    ESM2 generates dense vector embeddings for protein sequences.
    These embeddings encode structural and functional information,
    enabling mutation impact scoring and protein similarity analysis.

    Example:
        client = ESM2Client()
        emb = client.embed("MKFLILLFNILCLFPVLAADNHGVS")
        print(f"Embedding dimension: {emb.dimension}")

        # Compare two sequences
        sim = client.similarity("MKFLILLFNILCLFPVLAADNHGVS",
                                "MKFLILLFNILCLFPVLAADNHGVD")
        print(f"Cosine similarity: {sim:.4f}")
    """

    def embed(self, sequence: str) -> ProteinEmbedding:
        """Get embedding for a protein sequence.

        Args:
            sequence: Amino acid sequence (standard single-letter codes,
                      max 1024 characters).

        Returns:
            ProteinEmbedding with sequence, embedding vector, and dimension.
        """
        if len(sequence) > 1024:
            raise ValueError(
                f"ESM2-650m supports sequences up to 1024 chars, "
                f"got {len(sequence)}"
            )

        result = self._post(ENDPOINT, {"sequence": sequence})

        embedding = result.get("embedding", [])
        return ProteinEmbedding(
            sequence=sequence,
            embedding=embedding,
            dimension=len(embedding),
        )

    def embed_batch(self, sequences: List[str]) -> List[ProteinEmbedding]:
        """Embed multiple protein sequences.

        Args:
            sequences: List of amino acid sequences.

        Returns:
            List of ProteinEmbedding objects (same order as input).
        """
        return [self.embed(seq) for seq in sequences]

    def similarity(self, seq_a: str, seq_b: str) -> float:
        """Cosine similarity between two protein sequence embeddings.

        Args:
            seq_a: First protein sequence.
            seq_b: Second protein sequence.

        Returns:
            Cosine similarity in [-1, 1]. Higher = more similar.
        """
        emb_a = self.embed(seq_a)
        emb_b = self.embed(seq_b)
        return self._cosine_similarity(emb_a.embedding, emb_b.embedding)

    def mutation_impact(self, wild_type: str, mutant: str) -> dict:
        """Score the impact of a mutation by comparing embeddings.

        Args:
            wild_type: Wild-type protein sequence.
            mutant: Mutant protein sequence.

        Returns:
            Dict with cosine_similarity, euclidean_distance, and
            impact_score (0-1, higher = more disruptive).
        """
        emb_wt = self.embed(wild_type)
        emb_mut = self.embed(mutant)

        cos_sim = self._cosine_similarity(emb_wt.embedding, emb_mut.embedding)
        euc_dist = self._euclidean_distance(emb_wt.embedding, emb_mut.embedding)

        # Impact: 1 - cosine_similarity, clamped to [0, 1]
        impact = max(0.0, min(1.0, 1.0 - cos_sim))

        return {
            "cosine_similarity": cos_sim,
            "euclidean_distance": euc_dist,
            "impact_score": impact,
        }

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _euclidean_distance(a: List[float], b: List[float]) -> float:
        if len(a) != len(b):
            return float("inf")
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
