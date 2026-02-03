"""
Boltz-2 NIM Client
==================

Predict complex biomolecular structures: protein-ligand, protein-DNA,
protein-RNA, and multi-protein complexes. Supports up to 12 polymers
and 20 ligands per prediction with bond/pocket constraints.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .client import NIMClient

logger = logging.getLogger(__name__)

ENDPOINT = "mit/boltz2/predict"


@dataclass
class ComplexPrediction:
    """Predicted complex structure from Boltz-2."""
    structure_mmcif: str        # mmCIF format structure
    confidence: float = 0.0     # Overall model confidence
    polymer_chains: List[str] = field(default_factory=list)
    ligand_ids: List[str] = field(default_factory=list)
    num_polymers: int = 0
    num_ligands: int = 0


class Boltz2Client(NIMClient):
    """Client for Boltz-2 complex structure prediction NIM.

    Boltz-2 predicts 3D structures of biomolecular complexes containing
    proteins, DNA, RNA, and small molecule ligands. Supports up to
    12 polymers and 20 ligands per prediction.

    Example:
        client = Boltz2Client()

        # Protein-ligand complex
        pred = client.predict_protein_ligand(
            protein_sequence="MKFLILLFNILCLFPVLAADNHGVS...",
            ligand_smiles="CC(=O)Oc1ccccc1C(=O)O",
        )
        with open("complex.cif", "w") as f:
            f.write(pred.structure_mmcif)

        # Multi-protein complex (e.g., TCR + MHC-I)
        pred = client.predict_complex(polymers=[
            {"molecule_type": "protein", "sequence": "MKFL..."},
            {"molecule_type": "protein", "sequence": "AGHT..."},
        ])
    """

    def predict_complex(self, polymers: List[Dict],
                        ligands: Optional[List[Dict]] = None) -> ComplexPrediction:
        """Predict complex structure from polymers and ligands.

        Args:
            polymers: List of polymer dicts, each with:
                - molecule_type: "protein", "dna", or "rna"
                - sequence: amino acid or nucleotide sequence
            ligands: Optional list of ligand dicts, each with:
                - smiles: SMILES string
                - (optional) ccd_id: Chemical Component Dictionary ID

        Returns:
            ComplexPrediction with mmCIF structure.
        """
        if len(polymers) > 12:
            raise ValueError(f"Boltz-2 supports max 12 polymers, got {len(polymers)}")
        if ligands and len(ligands) > 20:
            raise ValueError(f"Boltz-2 supports max 20 ligands, got {len(ligands)}")

        payload = {"polymers": polymers}
        if ligands:
            payload["ligands"] = ligands

        result = self._post(ENDPOINT, payload, timeout=600)

        structure = result.get("structure", result.get("output", ""))
        confidence = float(result.get("confidence", 0.0))

        pred = ComplexPrediction(
            structure_mmcif=structure,
            confidence=confidence,
            polymer_chains=[p.get("molecule_type", "unknown") for p in polymers],
            ligand_ids=[l.get("smiles", "")[:20] for l in (ligands or [])],
            num_polymers=len(polymers),
            num_ligands=len(ligands or []),
        )
        logger.info(f"Boltz-2 predicted complex: {pred.num_polymers} polymers, "
                     f"{pred.num_ligands} ligands, confidence={confidence:.3f}")
        return pred

    def predict_protein_ligand(self, protein_sequence: str,
                                ligand_smiles: str) -> ComplexPrediction:
        """Convenience: single protein + single ligand complex."""
        return self.predict_complex(
            polymers=[{"molecule_type": "protein", "sequence": protein_sequence}],
            ligands=[{"smiles": ligand_smiles}],
        )

    def predict_protein_dna(self, protein_sequence: str,
                             dna_sequence: str) -> ComplexPrediction:
        """Convenience: protein-DNA complex (e.g., transcription factor binding)."""
        return self.predict_complex(polymers=[
            {"molecule_type": "protein", "sequence": protein_sequence},
            {"molecule_type": "dna", "sequence": dna_sequence},
        ])

    def predict_multimer(self, sequences: List[str]) -> ComplexPrediction:
        """Convenience: multi-chain protein complex."""
        polymers = [
            {"molecule_type": "protein", "sequence": seq}
            for seq in sequences
        ]
        return self.predict_complex(polymers)
