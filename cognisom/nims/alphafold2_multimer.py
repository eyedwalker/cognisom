"""
AlphaFold2-Multimer NIM Client
==============================

Predict multi-chain protein complex structures. Given multiple amino acid
sequences, predicts how the chains fold and assemble together.
Ideal for immune checkpoint complexes (PD-1/PD-L1, TCR/MHC-I).
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .client import NIMClient

logger = logging.getLogger(__name__)

ENDPOINT = "deepmind/alphafold2-multimer/predict"


@dataclass
class MultimerPrediction:
    """Predicted multi-chain protein complex from AlphaFold2-Multimer."""
    structure_data: str             # PDB or mmCIF format
    format: str = "pdb"             # "pdb" or "cif"
    plddt_scores: Optional[List[float]] = None   # Per-residue confidence (0-100)
    pae_matrix: Optional[List[List[float]]] = None  # Predicted Aligned Error
    num_chains: int = 0
    total_residues: int = 0


class AlphaFold2MultimerClient(NIMClient):
    """Client for AlphaFold2-Multimer complex structure prediction NIM.

    Predicts how multiple protein chains fold and assemble together.
    The Predicted Aligned Error (PAE) matrix indicates confidence
    in the relative positions of residue pairs, useful for assessing
    interface quality.

    Example:
        client = AlphaFold2MultimerClient()

        # Predict a heterodimer
        pred = client.predict_complex([
            "MKFLILLFNILCLFPVLAADNHGVS...",  # Chain A
            "AGHTKDPNRSQELQEALGTVASQ...",     # Chain B
        ])
        print(f"Predicted {pred.num_chains}-chain complex")

        # Predict a homodimer
        pred = client.predict_homodimer("MKFLILLFNILCLFPVLAADNHGVS...")
    """

    def predict_complex(self, sequences: List[str]) -> MultimerPrediction:
        """Predict multi-chain protein complex structure.

        Args:
            sequences: List of amino acid sequences (one per chain).

        Returns:
            MultimerPrediction with structure, pLDDT, and PAE data.
        """
        if len(sequences) < 2:
            logger.warning("AlphaFold2-Multimer is for multi-chain complexes; "
                          "use AlphaFold2 or OpenFold3 for single chains")

        result = self._post(ENDPOINT, {
            "sequences": sequences,
        }, timeout=600)

        structure = result.get("structure", result.get("output", ""))
        fmt = "cif" if "data_" in structure[:50] else "pdb"
        plddt = result.get("plddt", result.get("confidence", None))
        pae = result.get("pae", result.get("predicted_aligned_error", None))

        pred = MultimerPrediction(
            structure_data=structure,
            format=fmt,
            plddt_scores=plddt,
            pae_matrix=pae,
            num_chains=len(sequences),
            total_residues=sum(len(s) for s in sequences),
        )
        logger.info(f"AlphaFold2-Multimer predicted {pred.num_chains}-chain "
                     f"complex ({pred.total_residues} residues)")
        return pred

    def predict_homodimer(self, sequence: str) -> MultimerPrediction:
        """Predict homodimer (two copies of the same protein).

        Args:
            sequence: Amino acid sequence of the monomer.

        Returns:
            MultimerPrediction for the homodimer.
        """
        return self.predict_complex([sequence, sequence])

    def predict_immune_complex(self, receptor_seq: str,
                                ligand_seq: str) -> MultimerPrediction:
        """Convenience: predict immune receptor-ligand complex.

        Useful for PD-1/PD-L1, TCR/MHC, KIR/HLA complexes.

        Args:
            receptor_seq: Immune receptor sequence (e.g., PD-1, TCR).
            ligand_seq: Ligand sequence (e.g., PD-L1, MHC-I).

        Returns:
            MultimerPrediction for the immune complex.
        """
        return self.predict_complex([receptor_seq, ligand_seq])
