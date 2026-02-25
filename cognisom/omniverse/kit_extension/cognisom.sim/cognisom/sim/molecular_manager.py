"""
Molecular Visualization Manager
================================

Manages molecular visualization state in Kit, analogous to
DiapedesisManager but for protein structures.

Handles:
- Loading PDB data (from text or file path)
- Scene building via MolecularSceneBuilder
- Visualization mode switching
- Thread-safe scene build requests from HTTP handlers

Usage from extension.py::

    mgr = MolecularManager()
    mgr.load_pdb(pdb_text, mode="ribbon", mutations=[877])
    mgr.request_build_scene()
    # Per-frame: mgr.process_pending()  (on Kit main thread)
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import carb
except ImportError:
    carb = None

try:
    import omni.usd
    from pxr import Usd
    USD_AVAILABLE = True
except ImportError:
    USD_AVAILABLE = False


class MolecularManager:
    """Manage molecular visualization state in Kit."""

    def __init__(self):
        self._stage = None
        self._lock = threading.Lock()

        # Current state
        self._pdb_text: Optional[str] = None
        self._ligand_sdf: Optional[str] = None
        self._mode: str = "ribbon"  # ribbon, ball_and_stick, surface
        self._color_mode: str = "bfactor"
        self._mutations: List[int] = []
        self._binding_residues: List[int] = []
        self._scene_type: str = "protein"  # protein, docking, comparison
        self._wt_pdb: Optional[str] = None
        self._title: str = ""

        # Build tracking
        self._scene_built = False
        self._scene_build_count = 0
        self._pending_build = False

        # Scene builder (lazy import to avoid issues in non-Kit environments)
        self._builder = None

    @property
    def is_loaded(self) -> bool:
        return self._pdb_text is not None

    @property
    def scene_built(self) -> bool:
        return self._scene_built

    @property
    def current_mode(self) -> str:
        return self._mode

    @property
    def scene_type(self) -> str:
        return self._scene_type

    def load_pdb(self, pdb_text: str, mode: str = "ribbon",
                 color_mode: str = "bfactor",
                 mutations: Optional[List[int]] = None,
                 title: str = ""):
        """Load PDB data for protein visualization.

        Args:
            pdb_text: PDB-format text.
            mode: "ribbon", "ball_and_stick", or "surface".
            color_mode: "element", "bfactor", "chain".
            mutations: Residue IDs to highlight.
            title: Structure title/label.
        """
        with self._lock:
            self._pdb_text = pdb_text
            self._mode = mode
            self._color_mode = color_mode
            self._mutations = mutations or []
            self._scene_type = "protein"
            self._title = title
            self._scene_built = False
            self._ligand_sdf = None
            self._wt_pdb = None

        if carb:
            carb.log_info(f"[molecular] PDB loaded: mode={mode}, "
                          f"mutations={len(self._mutations)}")

    def load_docking(self, protein_pdb: str, ligand_sdf: str,
                     binding_residues: Optional[List[int]] = None,
                     protein_mode: str = "surface"):
        """Load protein-ligand docking data.

        Args:
            protein_pdb: PDB text for protein.
            ligand_sdf: SDF/MOL text for ligand.
            binding_residues: Binding site residues to highlight.
            protein_mode: "surface" or "ribbon".
        """
        with self._lock:
            self._pdb_text = protein_pdb
            self._ligand_sdf = ligand_sdf
            self._binding_residues = binding_residues or []
            self._mode = protein_mode
            self._scene_type = "docking"
            self._scene_built = False
            self._wt_pdb = None

        if carb:
            carb.log_info(f"[molecular] Docking data loaded")

    def load_comparison(self, wt_pdb: str, mut_pdb: str,
                        mutations: Optional[List[int]] = None,
                        mode: str = "ribbon"):
        """Load wild-type vs mutant comparison data.

        Args:
            wt_pdb: PDB text for wild-type.
            mut_pdb: PDB text for mutant.
            mutations: Mutation residue IDs.
            mode: Visualization mode.
        """
        with self._lock:
            self._wt_pdb = wt_pdb
            self._pdb_text = mut_pdb
            self._mutations = mutations or []
            self._mode = mode
            self._scene_type = "comparison"
            self._scene_built = False
            self._ligand_sdf = None

        if carb:
            carb.log_info(f"[molecular] Comparison data loaded")

    def request_build_scene(self):
        """Request scene build on the next main thread tick."""
        self._pending_build = True

    def process_pending(self):
        """Process pending scene build. Must run on Kit main thread."""
        if self._pending_build:
            self._pending_build = False
            self._build_scene()

    def set_mode(self, mode: str):
        """Change visualization mode and rebuild."""
        with self._lock:
            self._mode = mode
        self.request_build_scene()

    def clear(self):
        """Clear molecular visualization."""
        with self._lock:
            self._pdb_text = None
            self._ligand_sdf = None
            self._wt_pdb = None
            self._scene_built = False
            self._mutations = []
            self._binding_residues = []

        if self._stage and USD_AVAILABLE:
            try:
                from .molecular_scene import MOLECULAR_ROOT, CAMERA_PATH
                for path in [MOLECULAR_ROOT, CAMERA_PATH]:
                    prim = self._stage.GetPrimAtPath(path)
                    if prim and prim.IsValid():
                        self._stage.RemovePrim(path)
            except Exception as e:
                if carb:
                    carb.log_warn(f"[molecular] Clear failed: {e}")

    def get_status(self) -> Dict:
        """Get current visualization status."""
        return {
            "loaded": self.is_loaded,
            "scene_built": self._scene_built,
            "scene_type": self._scene_type,
            "mode": self._mode,
            "color_mode": self._color_mode,
            "mutations": self._mutations,
            "title": self._title,
            "build_count": self._scene_build_count,
        }

    def _build_scene(self):
        """Build the molecular scene on the Kit main thread."""
        if not self._pdb_text:
            if carb:
                carb.log_warn("[molecular] No PDB data to build scene from")
            return

        # Get stage
        stage = self._stage
        if not stage and USD_AVAILABLE:
            try:
                stage = omni.usd.get_context().get_stage()
            except Exception:
                pass

        if not stage:
            if carb:
                carb.log_warn("[molecular] No USD stage available")
            return

        self._stage = stage

        # Lazy-init builder
        if self._builder is None:
            from .molecular_scene import MolecularSceneBuilder
            self._builder = MolecularSceneBuilder()

        try:
            if self._scene_type == "docking" and self._ligand_sdf:
                self._builder.build_docking_scene(
                    stage, self._pdb_text, self._ligand_sdf,
                    binding_site_residues=self._binding_residues,
                    protein_mode=self._mode,
                )
            elif self._scene_type == "comparison" and self._wt_pdb:
                self._builder.build_comparison_scene(
                    stage, self._wt_pdb, self._pdb_text,
                    mutation_residues=self._mutations,
                    mode=self._mode,
                )
            else:
                self._builder.build_protein_scene(
                    stage, self._pdb_text,
                    mode=self._mode,
                    color_mode=self._color_mode,
                    mutations=self._mutations,
                    title=self._title,
                )

            self._scene_built = True
            self._scene_build_count += 1

            if carb:
                carb.log_warn(
                    f"[molecular] Scene built #{self._scene_build_count}: "
                    f"type={self._scene_type}, mode={self._mode}"
                )
        except Exception as e:
            if carb:
                carb.log_error(f"[molecular] Scene build failed: {e}")
                import traceback
                carb.log_error(f"[molecular] {traceback.format_exc()}")
