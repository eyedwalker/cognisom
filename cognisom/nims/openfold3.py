"""
OpenFold3 NIM Client
====================

All-atom biomolecular complex structure prediction.
Predicts 3D structures of proteins, DNA, RNA, and small molecule ligands
together in a single complex. PyTorch reimplementation of AlphaFold3,
up to 1.7x faster with TensorRT backend.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .client import NIMClient

logger = logging.getLogger(__name__)

ENDPOINT = "openfold/openfold3/predict"


@dataclass
class StructurePrediction:
    """Predicted 3D structure from OpenFold3."""
    structure_data: str             # CIF or PDB format content
    format: str = "cif"             # "cif" or "pdb"
    confidence_scores: Optional[Dict[str, float]] = None
    plddt: Optional[List[float]] = None   # Per-residue confidence (0-100)
    num_residues: int = 0
    num_chains: int = 0


class OpenFold3Client(NIMClient):
    """Client for OpenFold3 biomolecular structure prediction NIM.

    OpenFold3 predicts all-atom 3D structures of biomolecular complexes:
    proteins, DNA, RNA, and small molecule ligands — all together in
    a single prediction. Uses the TRT backend for 1.7x faster inference.

    Example:
        client = OpenFold3Client()

        # Single protein
        pred = client.predict_structure(["MKFLILLFNILCLFPVLAADNHGVS..."])
        with open("structure.cif", "w") as f:
            f.write(pred.structure_data)

        # Protein + ligand complex
        pred = client.predict_complex(
            sequences=["MKFLILLFNILCLFPVLAADNHGVS..."],
            ligand_smiles=["CC(=O)Oc1ccccc1C(=O)O"],
        )
    """

    def predict_structure(self, sequences: List[str],
                          msa_data: Optional[str] = None,
                          output_format: str = "cif") -> StructurePrediction:
        """Predict structure from protein sequence(s).

        Args:
            sequences: List of amino acid sequences (one per chain).
            msa_data: Optional MSA alignment (a3m format) from MSA-Search.
            output_format: Output format ("cif" or "pdb").

        Returns:
            StructurePrediction with atomic coordinates and confidence.
        """
        payload = {
            "sequences": sequences,
            "output_format": output_format,
        }
        if msa_data:
            payload["msa"] = msa_data

        result = self._post(ENDPOINT, payload, timeout=600)
        return self._parse_result(result, sequences, output_format)

    def predict_complex(self, sequences: List[str],
                        ligand_smiles: Optional[List[str]] = None,
                        dna_sequences: Optional[List[str]] = None,
                        rna_sequences: Optional[List[str]] = None,
                        msa_data: Optional[str] = None) -> StructurePrediction:
        """Predict biomolecular complex structure.

        Args:
            sequences: Protein amino acid sequences.
            ligand_smiles: Small molecule SMILES strings.
            dna_sequences: DNA sequences.
            rna_sequences: RNA sequences.
            msa_data: Optional MSA from MSA-Search.

        Returns:
            StructurePrediction with the full complex.
        """
        payload = {"sequences": sequences}
        if ligand_smiles:
            payload["ligands"] = [{"smiles": s} for s in ligand_smiles]
        if dna_sequences:
            payload["dna_sequences"] = dna_sequences
        if rna_sequences:
            payload["rna_sequences"] = rna_sequences
        if msa_data:
            payload["msa"] = msa_data

        result = self._post(ENDPOINT, payload, timeout=600)
        return self._parse_result(result, sequences, "cif")

    def predict_protein_ligand(self, protein_sequence: str,
                                ligand_smiles: str) -> StructurePrediction:
        """Convenience: predict a single protein + single ligand complex."""
        return self.predict_complex(
            sequences=[protein_sequence],
            ligand_smiles=[ligand_smiles],
        )

    def _parse_result(self, result: dict, sequences: List[str],
                      output_format: str) -> StructurePrediction:
        structure_data = result.get("structure", result.get("output", ""))
        plddt = result.get("plddt", result.get("confidence", None))
        confidence_scores = result.get("confidence_scores", None)

        pred = StructurePrediction(
            structure_data=structure_data,
            format=output_format,
            confidence_scores=confidence_scores,
            plddt=plddt,
            num_residues=sum(len(s) for s in sequences),
            num_chains=len(sequences),
        )
        logger.info(f"OpenFold3 predicted structure: {pred.num_chains} chains, "
                     f"{pred.num_residues} residues")
        return pred
