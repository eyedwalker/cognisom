"""
Structure Bridge
================

Converts NIM structure predictions (CIF/PDB) into simulation-usable data.
Extracts binding sites, contact maps, and structural comparisons.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BindingSite:
    """Extracted binding site from a structure."""
    residue_ids: List[int]
    residue_names: List[str]
    center: Tuple[float, float, float]
    radius: float
    pocket_volume_estimate: float  # cubic angstroms


@dataclass
class StructureComparison:
    """Comparison between two structures."""
    rmsd: float
    aligned_residues: int
    total_residues_a: int
    total_residues_b: int
    per_residue_deviation: List[float]


class StructureBridge:
    """Bridge between NIM structure predictions and the simulation engine.

    Converts CIF/PDB structure data into numerical representations
    usable by the Cognisom simulation modules.
    """

    @staticmethod
    def extract_ca_coords(pdb_text: str) -> np.ndarray:
        """Extract C-alpha coordinates from PDB-format text.

        Returns array of shape (N, 3) with x, y, z for each residue.
        """
        coords = []
        for line in pdb_text.splitlines():
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
        return np.array(coords) if coords else np.empty((0, 3))

    @staticmethod
    def extract_all_atom_coords(pdb_text: str) -> np.ndarray:
        """Extract all atom coordinates from PDB-format text."""
        coords = []
        for line in pdb_text.splitlines():
            if line.startswith(("ATOM", "HETATM")):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                except (ValueError, IndexError):
                    continue
        return np.array(coords) if coords else np.empty((0, 3))

    def extract_binding_site(
        self,
        pdb_text: str,
        ligand_center: Tuple[float, float, float],
        radius: float = 8.0,
    ) -> BindingSite:
        """Extract residues within `radius` angstroms of a ligand center.

        Parameters
        ----------
        pdb_text : str
            PDB-format structure text.
        ligand_center : tuple
            (x, y, z) of the ligand center of mass.
        radius : float
            Distance cutoff in angstroms (default 8.0).
        """
        center = np.array(ligand_center)
        residue_ids = []
        residue_names = []
        seen = set()

        for line in pdb_text.splitlines():
            if not line.startswith("ATOM"):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                dist = np.linalg.norm(np.array([x, y, z]) - center)
                if dist <= radius:
                    res_id = int(line[22:26].strip())
                    res_name = line[17:20].strip()
                    if res_id not in seen:
                        seen.add(res_id)
                        residue_ids.append(res_id)
                        residue_names.append(res_name)
            except (ValueError, IndexError):
                continue

        # Estimate pocket volume as sphere volume
        pocket_vol = (4 / 3) * np.pi * radius**3

        return BindingSite(
            residue_ids=residue_ids,
            residue_names=residue_names,
            center=tuple(ligand_center),
            radius=radius,
            pocket_volume_estimate=round(pocket_vol, 1),
        )

    def structure_to_contact_map(
        self, pdb_text: str, threshold: float = 8.0
    ) -> np.ndarray:
        """Build a residue-level contact map from C-alpha distances.

        Returns a binary NxN matrix where 1 = residues within `threshold` angstroms.
        """
        ca = self.extract_ca_coords(pdb_text)
        if len(ca) == 0:
            return np.empty((0, 0))

        n = len(ca)
        # Pairwise distances
        diff = ca[:, None, :] - ca[None, :, :]
        dists = np.sqrt((diff**2).sum(axis=2))
        contact = (dists <= threshold).astype(int)
        np.fill_diagonal(contact, 0)
        return contact

    def compare_structures(
        self, pdb_a: str, pdb_b: str
    ) -> StructureComparison:
        """Compare two structures by C-alpha RMSD after alignment.

        Uses simple centroid alignment (no rotation — for quick comparison).
        For publication-quality alignment, use a proper structural alignment tool.
        """
        ca_a = self.extract_ca_coords(pdb_a)
        ca_b = self.extract_ca_coords(pdb_b)

        # Align on the shorter length
        n = min(len(ca_a), len(ca_b))
        if n == 0:
            return StructureComparison(
                rmsd=float("inf"), aligned_residues=0,
                total_residues_a=len(ca_a), total_residues_b=len(ca_b),
                per_residue_deviation=[],
            )

        a = ca_a[:n]
        b = ca_b[:n]

        # Center both
        a_centered = a - a.mean(axis=0)
        b_centered = b - b.mean(axis=0)

        # Per-residue deviation
        deviations = np.sqrt(((a_centered - b_centered) ** 2).sum(axis=1))
        rmsd = float(np.sqrt((deviations**2).mean()))

        return StructureComparison(
            rmsd=round(rmsd, 3),
            aligned_residues=n,
            total_residues_a=len(ca_a),
            total_residues_b=len(ca_b),
            per_residue_deviation=[round(d, 3) for d in deviations.tolist()],
        )

    def structure_to_simulation_params(
        self, pdb_text: str
    ) -> Dict:
        """Convert a structure into parameters usable by simulation modules.

        Returns dict with:
        - n_residues: residue count
        - radius_of_gyration: compactness measure
        - contact_density: fraction of possible contacts
        - surface_area_estimate: rough surface area
        """
        ca = self.extract_ca_coords(pdb_text)
        if len(ca) == 0:
            return {"n_residues": 0, "radius_of_gyration": 0, "contact_density": 0, "surface_area_estimate": 0}

        centroid = ca.mean(axis=0)
        dists_from_center = np.sqrt(((ca - centroid) ** 2).sum(axis=1))
        rg = float(np.sqrt((dists_from_center**2).mean()))

        # Contact density
        contact_map = self.structure_to_contact_map(pdb_text)
        n = len(ca)
        max_contacts = n * (n - 1) / 2
        actual_contacts = contact_map.sum() / 2
        density = actual_contacts / max_contacts if max_contacts > 0 else 0

        # Rough surface area estimate (using convex hull proxy)
        sa_estimate = 4 * np.pi * rg**2  # sphere approximation

        return {
            "n_residues": n,
            "radius_of_gyration": round(rg, 2),
            "contact_density": round(float(density), 4),
            "surface_area_estimate": round(float(sa_estimate), 1),
        }
