"""
DiffDock NIM Client
===================

Predict molecular docking poses -- how a small molecule binds to a protein.
Requires uploading protein (PDB) and ligand (SDF) as NVCF assets first.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from .client import NIMClient

logger = logging.getLogger(__name__)

ENDPOINT = "mit/diffdock"


@dataclass
class DockingPose:
    """A predicted docking pose from DiffDock."""
    ligand_sdf: str  # SDF content with predicted pose
    confidence: float  # Model confidence score


class DiffDockClient(NIMClient):
    """Client for DiffDock molecular docking NIM.

    DiffDock predicts how a small molecule (ligand) binds to a protein.
    Unlike other NIMs, DiffDock requires a two-step process:
    1. Upload protein PDB and ligand SDF as NVCF assets
    2. Call the docking endpoint with asset IDs

    Example:
        client = DiffDockClient()
        result = client.dock(
            protein_pdb=open("protein.pdb").read(),
            ligand_sdf=open("ligand.sdf").read(),
            num_poses=5,
        )
        print(f"Got {len(result)} docking poses")
    """

    def dock(self, protein_pdb: str, ligand_sdf: str,
             num_poses: int = 5, steps: int = 18,
             time_divisor: int = 2) -> str:
        """Dock a ligand to a protein.

        Args:
            protein_pdb: Protein structure in PDB format.
            ligand_sdf: Ligand molecule in SDF format.
            num_poses: Number of docking poses to generate.
            steps: Number of diffusion steps.
            time_divisor: Time division parameter.

        Returns:
            Raw response from DiffDock (SDF with docked poses).
        """
        # Upload assets
        protein_id = self.upload_asset(
            protein_pdb, "text/plain", "protein PDB"
        )
        ligand_id = self.upload_asset(
            ligand_sdf, "chemical/x-mdl-sdfile", "ligand SDF"
        )
        logger.info(f"Uploaded protein={protein_id}, ligand={ligand_id}")

        # Call DiffDock with asset references
        url = f"{self._base_url()}/{ENDPOINT}"
        r = self._session.post(
            url,
            json={
                "protein": protein_id,
                "ligand": ligand_id,
                "ligand_file_type": "sdf",
                "num_poses": num_poses,
                "time_divisor": time_divisor,
                "steps": steps,
                "is_staged": False,
            },
            headers={
                **self._session.headers,
                "NVCF-INPUT-ASSET-REFERENCES": f"{protein_id},{ligand_id}",
            },
            timeout=300,
        )

        if r.status_code != 200:
            logger.error(f"DiffDock error {r.status_code}: {r.text[:500]}")
            r.raise_for_status()

        return r.text

    def dock_smiles(self, protein_pdb: str, ligand_smiles: str,
                    num_poses: int = 5) -> str:
        """Dock a ligand specified as SMILES string.

        Note: DiffDock may require SDF format. This method uploads
        the SMILES as-is -- for production use, convert SMILES to SDF
        first using RDKit.
        """
        protein_id = self.upload_asset(
            protein_pdb, "text/plain", "protein PDB"
        )
        ligand_id = self.upload_asset(
            ligand_smiles, "text/plain", "ligand SMILES"
        )

        url = f"{self._base_url()}/{ENDPOINT}"
        r = self._session.post(
            url,
            json={
                "protein": protein_id,
                "ligand": ligand_id,
                "ligand_file_type": "smi",
                "num_poses": num_poses,
                "time_divisor": 2,
                "steps": 18,
                "is_staged": False,
            },
            headers={
                **self._session.headers,
                "NVCF-INPUT-ASSET-REFERENCES": f"{protein_id},{ligand_id}",
            },
            timeout=300,
        )

        if r.status_code != 200:
            logger.error(f"DiffDock error {r.status_code}: {r.text[:500]}")
            r.raise_for_status()

        return r.text

    @staticmethod
    def _base_url():
        from .client import HEALTH_API_BASE
        return HEALTH_API_BASE
