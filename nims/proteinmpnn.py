"""
ProteinMPNN NIM Client
======================

Design amino acid sequences that fold into a desired 3D structure.
Given a PDB backbone, predicts optimal sequences.
"""

import json
import logging
from dataclasses import dataclass
from typing import List, Optional

from .client import NIMClient

logger = logging.getLogger(__name__)

ENDPOINT = "ipd/proteinmpnn/predict"


@dataclass
class DesignedSequence:
    """A protein sequence designed by ProteinMPNN."""
    sequence: str
    score: float  # Lower is better (negative log-likelihood)
    recovery: float  # Fraction of original sequence recovered


class ProteinMPNNClient(NIMClient):
    """Client for ProteinMPNN protein sequence design NIM.

    ProteinMPNN designs amino acid sequences that are predicted to fold
    into a given 3D backbone structure. Used after RFdiffusion to find
    sequences for designed binder structures.

    Example:
        client = ProteinMPNNClient()
        pdb_data = open("binder.pdb").read()
        sequences = client.design_sequences(pdb_data, num_sequences=5)
        for seq in sequences:
            print(f"{seq.sequence}  (score: {seq.score:.3f})")
    """

    def design_sequences(self, pdb_data: str, num_sequences: int = 3,
                         sampling_temp: float = 0.1,
                         use_soluble_model: bool = False) -> List[DesignedSequence]:
        """Design sequences for a protein backbone.

        Args:
            pdb_data: PDB file content (ATOM lines).
            num_sequences: Number of sequence variants to generate.
            sampling_temp: Temperature for sampling (lower = more conservative).
            use_soluble_model: Use soluble-protein-specific model.

        Returns:
            List of DesignedSequence with sequence, score, and recovery.
        """
        result = self._post(ENDPOINT, {
            "input_pdb": pdb_data,
            "ca_only": False,
            "use_soluble_model": use_soluble_model,
            "sampling_temp": [sampling_temp],
        }, timeout=120)

        sequences = []
        mfasta = result.get("mfasta", "")
        scores = result.get("scores", [])

        # Parse FASTA output
        entries = mfasta.strip().split(">")
        for entry in entries:
            if not entry.strip():
                continue
            lines = entry.strip().split("\n")
            header = lines[0]
            seq = "".join(lines[1:])

            # Parse score and recovery from header
            score = 0.0
            recovery = 0.0
            for part in header.split(","):
                part = part.strip()
                if part.startswith("score="):
                    score = float(part.split("=")[1])
                elif part.startswith("seq_recovery="):
                    recovery = float(part.split("=")[1])

            sequences.append(DesignedSequence(
                sequence=seq,
                score=score,
                recovery=recovery,
            ))

        logger.info(f"Designed {len(sequences)} sequences")
        return sequences

    def design_for_binder(self, binder_pdb: str,
                          num_sequences: int = 5) -> List[DesignedSequence]:
        """Design sequences for an RFdiffusion-generated binder.

        Uses low temperature for high-confidence designs.
        """
        return self.design_sequences(
            binder_pdb,
            num_sequences=num_sequences,
            sampling_temp=0.1,
            use_soluble_model=True,
        )
