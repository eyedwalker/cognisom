"""
Molecular USD Scene Builder
============================

Build complete USD scenes for molecular visualization:
- Single protein structures (from PDB text)
- Protein-ligand docking views (from DiffDock output)
- Wild-type vs mutant comparison (side-by-side)

Uses PDBtoUSD for geometry conversion and adds lighting, camera,
and materials for RTX rendering in Kit.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

import numpy as np

from .pdb_to_usd import PDBtoUSD, PDBAtom, parse_pdb, get_ca_trace

logger = logging.getLogger(__name__)

try:
    from pxr import Usd, UsdGeom, UsdShade, UsdLux, Gf, Sdf
    USD_AVAILABLE = True
except ImportError:
    USD_AVAILABLE = False

# Scene hierarchy paths
MOLECULAR_ROOT = "/World/Molecular"
PROTEIN_PATH = f"{MOLECULAR_ROOT}/Protein"
LIGAND_PATH = f"{MOLECULAR_ROOT}/Ligand"
COMPARISON_PATH = f"{MOLECULAR_ROOT}/Comparison"
LIGHTS_PATH = f"{MOLECULAR_ROOT}/Lights"
CAMERA_PATH = "/World/MolecularCam"


class MolecularSceneBuilder:
    """Build USD scenes for molecular visualization.

    Orchestrates PDBtoUSD to create complete scenes with lighting
    and camera placement.

    Example:
        builder = MolecularSceneBuilder()
        builder.build_protein_scene(stage, pdb_text,
                                    mode="ribbon", mutations=[877, 702])
    """

    def __init__(self):
        self._converter = PDBtoUSD(atom_scale=0.4, bond_radius=0.15)

    def build_protein_scene(self, stage, pdb_text: str,
                            mode: str = "ribbon",
                            color_mode: str = "bfactor",
                            mutations: Optional[List[int]] = None,
                            title: str = ""):
        """Build a complete protein visualization scene.

        Args:
            stage: USD stage.
            pdb_text: PDB-format text data.
            mode: "ball_and_stick", "ribbon", or "surface".
            color_mode: "element", "bfactor" (pLDDT), "chain".
            mutations: Residue IDs to highlight.
            title: Optional label for the structure.
        """
        if not USD_AVAILABLE:
            logger.error("USD (pxr) not available")
            return

        atoms = parse_pdb(pdb_text)
        if not atoms:
            logger.error("No atoms parsed from PDB text")
            return

        logger.info(f"Building protein scene: {len(atoms)} atoms, mode={mode}")

        # Clear previous molecular scene
        self._clear_scene(stage)

        # Create root
        UsdGeom.Xform.Define(stage, MOLECULAR_ROOT)

        # Build protein geometry
        if mode == "ball_and_stick":
            self._converter.build_ball_and_stick(
                atoms, stage, PROTEIN_PATH,
                color_mode=color_mode,
                mutation_residues=mutations,
            )
        elif mode == "surface":
            self._converter.build_surface(
                atoms, stage, PROTEIN_PATH,
                mutation_residues=mutations,
            )
        else:  # ribbon (default)
            self._converter.build_ribbon(
                atoms, stage, PROTEIN_PATH,
                color_mode=color_mode,
                mutation_residues=mutations,
            )

        # Add lighting
        self._setup_molecular_lighting(stage)

        # Position camera to frame the protein
        extent = self._compute_extent(atoms)
        self._setup_molecular_camera(stage, extent)

        logger.info(f"Protein scene built at {PROTEIN_PATH}")

    def build_docking_scene(self, stage, protein_pdb: str, ligand_sdf: str,
                            binding_site_residues: Optional[List[int]] = None,
                            protein_mode: str = "surface"):
        """Build protein-ligand docking visualization.

        Shows protein surface with binding pocket highlighted and
        ligand rendered as ball-and-stick inside.

        Args:
            stage: USD stage.
            protein_pdb: PDB text for protein.
            ligand_sdf: SDF/MOL text for ligand.
            binding_site_residues: Residues to highlight as binding pocket.
            protein_mode: "surface" or "ribbon" for protein representation.
        """
        if not USD_AVAILABLE:
            logger.error("USD (pxr) not available")
            return

        protein_atoms = parse_pdb(protein_pdb)
        if not protein_atoms:
            logger.error("No protein atoms parsed")
            return

        self._clear_scene(stage)
        UsdGeom.Xform.Define(stage, MOLECULAR_ROOT)

        # Build protein
        if protein_mode == "surface":
            self._converter.build_surface(
                protein_atoms, stage, PROTEIN_PATH,
                mutation_residues=binding_site_residues,
            )
        else:
            self._converter.build_ribbon(
                protein_atoms, stage, PROTEIN_PATH,
                mutation_residues=binding_site_residues,
            )

        # Parse and build ligand from SDF
        ligand_atoms = self._parse_sdf(ligand_sdf)
        if ligand_atoms:
            # Center ligand relative to protein centroid
            protein_coords = np.array([(a.x, a.y, a.z) for a in protein_atoms])
            protein_centroid = protein_coords.mean(axis=0)

            # Offset ligand atoms by protein centroid
            for a in ligand_atoms:
                a.x -= protein_centroid[0]
                a.y -= protein_centroid[1]
                a.z -= protein_centroid[2]

            # Build ligand as ball-and-stick (always — shows atom detail)
            ligand_converter = PDBtoUSD(atom_scale=0.6, bond_radius=0.2)
            UsdGeom.Xform.Define(stage, LIGAND_PATH)

            # Create ligand atoms with green highlight
            self._build_ligand_geometry(
                ligand_atoms, stage, LIGAND_PATH
            )

        # Lighting and camera
        self._setup_molecular_lighting(stage)
        extent = self._compute_extent(protein_atoms)
        self._setup_molecular_camera(stage, extent, zoom_factor=0.8)

        logger.info(f"Docking scene built: protein + ligand")

    def build_comparison_scene(self, stage, wt_pdb: str, mut_pdb: str,
                               mutation_residues: Optional[List[int]] = None,
                               mode: str = "ribbon"):
        """Build side-by-side wild-type vs mutant comparison.

        Places two structures offset along the X axis for comparison.

        Args:
            stage: USD stage.
            wt_pdb: PDB text for wild-type structure.
            mut_pdb: PDB text for mutant structure.
            mutation_residues: Residues to highlight on mutant.
            mode: Visualization mode.
        """
        if not USD_AVAILABLE:
            logger.error("USD (pxr) not available")
            return

        wt_atoms = parse_pdb(wt_pdb)
        mut_atoms = parse_pdb(mut_pdb)

        if not wt_atoms or not mut_atoms:
            logger.error("Need both WT and mutant PDB data")
            return

        self._clear_scene(stage)
        UsdGeom.Xform.Define(stage, MOLECULAR_ROOT)
        UsdGeom.Xform.Define(stage, COMPARISON_PATH)

        # Compute offset based on protein extent
        wt_extent = self._compute_extent(wt_atoms)
        offset_x = wt_extent["range_x"] * 1.5

        # Build wild-type (left)
        wt_path = f"{COMPARISON_PATH}/WildType"
        if mode == "ribbon":
            self._converter.build_ribbon(wt_atoms, stage, wt_path, color_mode="bfactor")
        elif mode == "surface":
            self._converter.build_surface(wt_atoms, stage, wt_path)
        else:
            self._converter.build_ball_and_stick(wt_atoms, stage, wt_path)

        # Offset wild-type to the left
        wt_xf = UsdGeom.Xformable(stage.GetPrimAtPath(wt_path))
        wt_xf.AddTranslateOp().Set(Gf.Vec3d(-offset_x / 2, 0, 0))

        # Build mutant (right, with mutations highlighted)
        mut_path = f"{COMPARISON_PATH}/Mutant"
        if mode == "ribbon":
            self._converter.build_ribbon(
                mut_atoms, stage, mut_path,
                color_mode="bfactor",
                mutation_residues=mutation_residues,
            )
        elif mode == "surface":
            self._converter.build_surface(
                mut_atoms, stage, mut_path,
                mutation_residues=mutation_residues,
            )
        else:
            self._converter.build_ball_and_stick(
                mut_atoms, stage, mut_path,
                mutation_residues=mutation_residues,
            )

        # Offset mutant to the right
        mut_xf = UsdGeom.Xformable(stage.GetPrimAtPath(mut_path))
        mut_xf.AddTranslateOp().Set(Gf.Vec3d(offset_x / 2, 0, 0))

        # Add labels (text prims aren't well-supported, use small spheres as markers)
        self._add_label_marker(stage, f"{COMPARISON_PATH}/wt_marker",
                               (-offset_x / 2, wt_extent["range_y"] / 2 + 5, 0),
                               (0.2, 0.5, 0.9))  # Blue for WT
        self._add_label_marker(stage, f"{COMPARISON_PATH}/mut_marker",
                               (offset_x / 2, wt_extent["range_y"] / 2 + 5, 0),
                               (0.9, 0.2, 0.2))  # Red for mutant

        # Lighting and camera
        self._setup_molecular_lighting(stage)
        # Wider camera to show both structures
        combined_extent = {
            "center": (0, 0, 0),
            "range_x": offset_x * 2,
            "range_y": wt_extent["range_y"],
            "range_z": wt_extent["range_z"],
            "max_dim": max(offset_x * 2, wt_extent["range_y"], wt_extent["range_z"]),
        }
        self._setup_molecular_camera(stage, combined_extent, zoom_factor=1.3)

        logger.info(f"Comparison scene built: WT vs Mutant")

    # ── Helpers ──────────────────────────────────────────────────────────

    def _clear_scene(self, stage):
        """Remove previous molecular scene prims."""
        for path in [MOLECULAR_ROOT, CAMERA_PATH]:
            prim = stage.GetPrimAtPath(path)
            if prim and prim.IsValid():
                stage.RemovePrim(path)

    def _compute_extent(self, atoms: List[PDBAtom]) -> Dict:
        """Compute bounding box extent of atoms."""
        if not atoms:
            return {"center": (0, 0, 0), "range_x": 10, "range_y": 10,
                    "range_z": 10, "max_dim": 10}

        coords = np.array([(a.x, a.y, a.z) for a in atoms])
        centroid = coords.mean(axis=0)
        mins = coords.min(axis=0) - centroid
        maxs = coords.max(axis=0) - centroid
        ranges = maxs - mins

        return {
            "center": tuple(centroid),
            "range_x": float(ranges[0]),
            "range_y": float(ranges[1]),
            "range_z": float(ranges[2]),
            "max_dim": float(ranges.max()),
        }

    def _setup_molecular_lighting(self, stage):
        """Create lighting suitable for molecular visualization."""
        UsdGeom.Xform.Define(stage, LIGHTS_PATH)

        # Key light — dome for soft ambient
        dome = UsdLux.DomeLight.Define(stage, f"{LIGHTS_PATH}/Dome")
        dome.GetIntensityAttr().Set(200.0)
        dome.GetColorAttr().Set(Gf.Vec3f(0.9, 0.92, 1.0))

        # Fill light from the side
        fill = UsdLux.DistantLight.Define(stage, f"{LIGHTS_PATH}/Fill")
        fill.GetIntensityAttr().Set(500.0)
        fill.GetAngleAttr().Set(2.0)
        fill.GetColorAttr().Set(Gf.Vec3f(0.85, 0.9, 1.0))
        fill_xf = UsdGeom.Xformable(fill.GetPrim())
        fill_xf.AddRotateXYZOp().Set(Gf.Vec3f(-30, 45, 0))

        # Rim light from behind for depth
        rim = UsdLux.DistantLight.Define(stage, f"{LIGHTS_PATH}/Rim")
        rim.GetIntensityAttr().Set(300.0)
        rim.GetAngleAttr().Set(1.5)
        rim.GetColorAttr().Set(Gf.Vec3f(0.8, 0.85, 1.0))
        rim_xf = UsdGeom.Xformable(rim.GetPrim())
        rim_xf.AddRotateXYZOp().Set(Gf.Vec3f(-20, -150, 0))

    def _setup_molecular_camera(self, stage, extent: Dict,
                                zoom_factor: float = 1.0):
        """Position camera to frame the molecular structure."""
        # Remove existing camera
        old = stage.GetPrimAtPath(CAMERA_PATH)
        if old and old.IsValid():
            stage.RemovePrim(CAMERA_PATH)

        camera_prim = stage.DefinePrim(CAMERA_PATH, "Camera")
        camera = UsdGeom.Camera(camera_prim)
        camera.GetFocalLengthAttr().Set(35.0)
        camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 5000.0))
        camera.GetHorizontalApertureAttr().Set(36.0)

        # Distance based on protein size
        max_dim = extent.get("max_dim", 50)
        cam_distance = max(30, max_dim * 1.8 * zoom_factor)

        # Position: elevated 3/4 view
        cam_pos = Gf.Vec3d(
            cam_distance * 0.5,
            cam_distance * 0.4,
            cam_distance * 0.7,
        )
        target = Gf.Vec3d(0, 0, 0)  # Protein is centered at origin

        xformable = UsdGeom.Xformable(camera_prim)
        xformable.AddTranslateOp().Set(cam_pos)

        # Compute rotation to look at origin
        dx = target[0] - cam_pos[0]
        dy = target[1] - cam_pos[1]
        dz = target[2] - cam_pos[2]
        dist_xz = math.sqrt(dx * dx + dz * dz)
        pitch = math.degrees(math.atan2(-dy, dist_xz))
        yaw = math.degrees(math.atan2(dx, -dz))

        xformable.AddRotateXYZOp().Set(Gf.Vec3f(-pitch, yaw, 0.0))

        logger.info(f"Camera at {CAMERA_PATH}: distance={cam_distance:.1f}")

    def _parse_sdf(self, sdf_text: str) -> List[PDBAtom]:
        """Parse SDF/MOL format into PDBAtom list (minimal parser)."""
        atoms = []
        lines = sdf_text.strip().splitlines()

        if len(lines) < 4:
            return atoms

        # Counts line is at index 3 in V2000 MOL format
        try:
            counts_line = lines[3].strip()
            parts = counts_line.split()
            n_atoms = int(parts[0])
            n_bonds = int(parts[1])
        except (IndexError, ValueError):
            return atoms

        # Parse atom block (starts at line 4)
        for i in range(4, min(4 + n_atoms, len(lines))):
            parts = lines[i].split()
            if len(parts) < 4:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
                z = float(parts[2])
                element = parts[3].strip()
                atoms.append(PDBAtom(
                    serial=i - 3,
                    name=element,
                    res_name="LIG",
                    chain="L",
                    res_seq=1,
                    x=x, y=y, z=z,
                    element=element,
                    bfactor=0.0,
                ))
            except (ValueError, IndexError):
                continue

        return atoms

    def _build_ligand_geometry(self, atoms: List[PDBAtom], stage,
                               root_path: str):
        """Build ligand as green-tinted ball-and-stick."""
        from .pdb_to_usd import estimate_bonds, ATOM_COLORS

        atoms_path = f"{root_path}/atoms"
        UsdGeom.Xform.Define(stage, atoms_path)

        for i, atom in enumerate(atoms):
            if atom.element.upper() == "H":
                continue

            sphere_path = f"{atoms_path}/atom_{i}"
            sphere = UsdGeom.Sphere.Define(stage, sphere_path)
            sphere.GetRadiusAttr().Set(atom.radius * 0.6)

            xf = UsdGeom.Xformable(sphere.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(atom.x, atom.y, atom.z))

            # Ligand atoms get a green tint
            base_color = ATOM_COLORS.get(atom.element.upper(), (0.3, 0.8, 0.3))
            # Mix with green
            color = (
                base_color[0] * 0.6,
                min(1.0, base_color[1] * 0.6 + 0.4),
                base_color[2] * 0.6,
            )
            sphere.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])

        # Bonds
        non_h = [a for a in atoms if a.element.upper() != "H"]
        bonds = estimate_bonds(non_h, cutoff=1.9)
        if bonds:
            bonds_path = f"{root_path}/bonds"
            UsdGeom.Xform.Define(stage, bonds_path)
            for b_idx, (i, j) in enumerate(bonds):
                a1, a2 = non_h[i], non_h[j]
                self._converter._create_bond_cylinder(
                    stage, f"{bonds_path}/bond_{b_idx}",
                    (a1.x, a1.y, a1.z),
                    (a2.x, a2.y, a2.z),
                )

    def _add_label_marker(self, stage, path: str, position: tuple,
                          color: tuple):
        """Add a small colored sphere as a position marker/label."""
        sphere = UsdGeom.Sphere.Define(stage, path)
        sphere.GetRadiusAttr().Set(1.5)
        xf = UsdGeom.Xformable(sphere.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(*position))
        sphere.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
