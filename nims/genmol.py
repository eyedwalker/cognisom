"""
GenMol NIM Client
=================

Fragment-based molecular generation using masked diffusion.
Replaces MolMIM with more controlled scaffold-constrained generation.
"""

import json
import logging
from dataclasses import dataclass
from typing import List, Optional

from .client import NIMClient
from .molmim import GeneratedMolecule  # Reuse for interchangeability

logger = logging.getLogger(__name__)

ENDPOINT = "nvidia/genmol/generate"


class GenMolClient(NIMClient):
    """Client for GenMol fragment-based molecule generation NIM.

    GenMol uses masked diffusion to generate drug-like molecules.
    Unlike MolMIM's CMA-ES, GenMol supports scaffold-constrained generation:
    mask parts of the molecule and let the model fill them in while keeping
    desired fragments (motifs, scaffolds) intact.

    Example:
        client = GenMolClient()
        molecules = client.generate("CC(=O)Oc1ccccc1C(=O)O", num_molecules=10)
        for mol in molecules:
            print(f"{mol.smiles}  (QED: {mol.score:.3f})")
    """

    def generate(self, smiles: str, num_molecules: int = 10,
                 temperature: float = 1.0,
                 scoring: str = "QED") -> List[GeneratedMolecule]:
        """Generate molecules from a seed or masked SMILES.

        Args:
            smiles: Seed SMILES (or masked SMILES with [MASK] tokens for
                    scaffold-constrained generation).
            num_molecules: Number of molecules to generate.
            temperature: Sampling temperature (higher = more diverse).
            scoring: Scoring function ("QED" or "LogP").

        Returns:
            List of GeneratedMolecule with SMILES and scores.
        """
        result = self._post(ENDPOINT, {
            "smi": smiles,
            "num_molecules": num_molecules,
            "temperature": temperature,
            "scoring": scoring,
        })

        raw = result.get("molecules", "[]")
        if isinstance(raw, str):
            raw = json.loads(raw)

        molecules = [
            GeneratedMolecule(
                smiles=m.get("sample", m.get("smiles", "")),
                score=float(m.get("score", 0.0)),
            )
            for m in raw
        ]
        logger.info(f"GenMol generated {len(molecules)} molecules from {smiles[:40]}")
        return molecules

    def generate_for_target(self, seed_smiles: str, num_molecules: int = 20,
                            min_score: float = 0.5) -> List[GeneratedMolecule]:
        """Generate and filter molecules by score threshold.

        Args:
            seed_smiles: Starting molecule SMILES.
            num_molecules: Number to generate (before filtering).
            min_score: Minimum score to keep (0-1).

        Returns:
            Filtered list sorted by score (highest first).
        """
        molecules = self.generate(seed_smiles, num_molecules)
        filtered = [m for m in molecules if m.score >= min_score]
        filtered.sort(key=lambda m: m.score, reverse=True)
        logger.info(f"Kept {len(filtered)}/{len(molecules)} with score >= {min_score}")
        return filtered

    def scaffold_constrained_generate(self, scaffold_smiles: str,
                                       num_molecules: int = 10,
                                       temperature: float = 0.8) -> List[GeneratedMolecule]:
        """Generate molecules while preserving a scaffold core.

        The scaffold is kept intact; GenMol optimizes the variable regions.
        """
        return self.generate(scaffold_smiles, num_molecules, temperature)
