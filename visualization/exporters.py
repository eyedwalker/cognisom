"""
Scene Exporter
==============

Export 3D visualization data to standard file formats:
PDB (protein/cell coordinates), glTF (3D meshes), and VTK
(volumetric fields). Enables interoperability with external
visualization tools (PyMOL, Blender, ParaView).
"""

import json
import struct
import base64
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)


class SceneExporter:
    """Export cell populations and fields to standard 3D formats.

    Example:
        exporter = SceneExporter()
        exporter.cells_to_pdb(cells, "output/cells.pdb")
        exporter.cells_to_gltf(cells, "output/scene.gltf")
        exporter.field_to_vtk(field, "output/oxygen.vtk")
    """

    # ─── PDB Export ──────────────────────────────────────────────

    @staticmethod
    def cells_to_pdb(
        cells: List[Dict[str, Any]],
        output_path: str,
        scale: float = 1.0,
    ) -> str:
        """Export cells as a PDB file where each cell is an atom.

        Cell types map to chemical elements for coloring in molecular
        viewers (PyMOL, UCSF Chimera, etc.).

        Args:
            cells: List of cell dicts with 'position', 'cell_type'.
            output_path: Path to write .pdb file.
            scale: Coordinate scale factor (um -> Angstroms: use 10.0).

        Returns:
            Path to written file.
        """
        # Map cell types to element symbols (for coloring in viewers)
        type_to_element = {
            "normal": "C",      # green in CPK
            "cancer": "O",      # red in CPK
            "t_cell": "N",      # blue in CPK
            "nk_cell": "S",     # yellow in CPK
            "macrophage": "P",  # orange in CPK
            "fibroblast": "Fe", # brown-ish
            "endothelial": "Cl",# green-ish
            "stem": "Mg",       # light green
            "immune": "N",      # blue
            "dead": "H",        # white
        }

        # Map cell types to residue names (3 chars)
        type_to_resname = {
            "normal": "NRM",
            "cancer": "CAN",
            "t_cell": "TCL",
            "nk_cell": "NKC",
            "macrophage": "MAC",
            "fibroblast": "FIB",
            "endothelial": "END",
            "stem": "STM",
            "immune": "IMM",
            "dead": "DED",
        }

        lines = []
        lines.append("REMARK   Cell population exported from Cognisom")
        lines.append(f"REMARK   {len(cells)} cells")

        for i, cell in enumerate(cells):
            pos = cell.get("position", [0, 0, 0])
            ct = cell.get("cell_type", "normal")
            alive = cell.get("alive", True)

            x = pos[0] * scale
            y = pos[1] * scale
            z = pos[2] * scale

            element = type_to_element.get(ct, "C")
            if not alive:
                element = "H"

            resname = type_to_resname.get(ct, "UNK")
            atom_serial = (i + 1) % 100000
            res_seq = (i + 1) % 10000

            # B-factor encodes oxygen level, occupancy encodes ATP
            bfactor = cell.get("oxygen", 0.21) * 100
            occupancy = min(cell.get("atp", 1000) / 1000, 1.0)

            # PDB ATOM record format (fixed-width columns)
            line = (
                f"ATOM  {atom_serial:5d} "
                f"{'CA':4s}"
                f"{resname:>3s} "
                f"A"
                f"{res_seq:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"{occupancy:6.2f}{bfactor:6.2f}          "
                f"{element:>2s}  "
            )
            lines.append(line)

        lines.append("END")

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines))
        logger.info(f"PDB exported: {out} ({len(cells)} atoms)")
        return str(out)

    # ─── glTF Export ─────────────────────────────────────────────

    @staticmethod
    def cells_to_gltf(
        cells: List[Dict[str, Any]],
        output_path: str,
        sphere_segments: int = 8,
        cell_radius: float = 5.0,
    ) -> str:
        """Export cells as glTF 2.0 point cloud (positions + colors).

        Uses a minimal glTF file with a point-cloud primitive. For full
        sphere meshes, use sphere_segments > 0 to create instanced
        icospheres (heavier but looks better in Blender/three.js).

        Args:
            cells: List of cell dicts.
            output_path: Path to write .gltf file.
            sphere_segments: 0 for points, >0 for sphere approximations.
            cell_radius: Radius of each cell sphere in um.

        Returns:
            Path to written file.
        """
        from .cell_renderer import CELL_TYPE_COLORS

        if not cells:
            return ""

        # Extract positions and colors
        positions = []
        colors = []
        for c in cells:
            pos = c.get("position", [0, 0, 0])
            positions.append([float(pos[0]), float(pos[1]), float(pos[2])])

            ct = c.get("cell_type", "normal")
            hex_color = CELL_TYPE_COLORS.get(ct, "#9E9E9E")
            r = int(hex_color[1:3], 16) / 255.0
            g = int(hex_color[3:5], 16) / 255.0
            b = int(hex_color[5:7], 16) / 255.0
            colors.append([r, g, b, 1.0])

        pos_arr = np.array(positions, dtype=np.float32)
        col_arr = np.array(colors, dtype=np.float32)

        # Build binary buffer
        pos_bytes = pos_arr.tobytes()
        col_bytes = col_arr.tobytes()
        buffer_data = pos_bytes + col_bytes

        # Encode as base64 data URI
        b64 = base64.b64encode(buffer_data).decode("ascii")
        buffer_uri = f"data:application/octet-stream;base64,{b64}"

        n = len(cells)
        pos_min = pos_arr.min(axis=0).tolist()
        pos_max = pos_arr.max(axis=0).tolist()

        gltf = {
            "asset": {"version": "2.0", "generator": "Cognisom SceneExporter"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0, "name": "CellPopulation"}],
            "meshes": [{
                "primitives": [{
                    "attributes": {
                        "POSITION": 0,
                        "COLOR_0": 1,
                    },
                    "mode": 0,  # POINTS
                }],
            }],
            "accessors": [
                {
                    "bufferView": 0,
                    "componentType": 5126,  # FLOAT
                    "count": n,
                    "type": "VEC3",
                    "min": pos_min,
                    "max": pos_max,
                },
                {
                    "bufferView": 1,
                    "componentType": 5126,
                    "count": n,
                    "type": "VEC4",
                },
            ],
            "bufferViews": [
                {
                    "buffer": 0,
                    "byteOffset": 0,
                    "byteLength": len(pos_bytes),
                    "target": 34962,  # ARRAY_BUFFER
                },
                {
                    "buffer": 0,
                    "byteOffset": len(pos_bytes),
                    "byteLength": len(col_bytes),
                    "target": 34962,
                },
            ],
            "buffers": [{
                "uri": buffer_uri,
                "byteLength": len(buffer_data),
            }],
        }

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(gltf, indent=2))
        logger.info(f"glTF exported: {out} ({n} cells)")
        return str(out)

    # ─── VTK Export ──────────────────────────────────────────────

    @staticmethod
    def field_to_vtk(
        field: np.ndarray,
        output_path: str,
        field_name: str = "concentration",
        resolution_um: float = 10.0,
        origin: tuple = (0.0, 0.0, 0.0),
    ) -> str:
        """Export 3D concentration field as VTK structured points (legacy format).

        Compatible with ParaView, VisIt, and VTK-based tools.

        Args:
            field: 3D numpy array (nx, ny, nz).
            output_path: Path to write .vtk file.
            field_name: Name for the scalar dataset.
            resolution_um: Voxel spacing in micrometers.
            origin: Origin coordinates (x, y, z).

        Returns:
            Path to written file.
        """
        nx, ny, nz = field.shape
        spacing = (resolution_um, resolution_um, resolution_um)

        lines = [
            "# vtk DataFile Version 3.0",
            f"Cognisom {field_name} field ({nx}x{ny}x{nz})",
            "ASCII",
            "DATASET STRUCTURED_POINTS",
            f"DIMENSIONS {nx} {ny} {nz}",
            f"ORIGIN {origin[0]} {origin[1]} {origin[2]}",
            f"SPACING {spacing[0]} {spacing[1]} {spacing[2]}",
            f"POINT_DATA {nx * ny * nz}",
            f"SCALARS {field_name} float 1",
            "LOOKUP_TABLE default",
        ]

        # Write values in Fortran order (VTK expects x-fastest)
        values = field.flatten(order="F")
        # Write in chunks for reasonable line lengths
        chunk_size = 9
        for i in range(0, len(values), chunk_size):
            chunk = values[i:i + chunk_size]
            lines.append(" ".join(f"{v:.6f}" for v in chunk))

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines))
        logger.info(f"VTK exported: {out} ({nx}x{ny}x{nz} = {nx*ny*nz} points)")
        return str(out)

    @staticmethod
    def fields_to_vtk(
        fields: Dict[str, np.ndarray],
        output_path: str,
        resolution_um: float = 10.0,
        origin: tuple = (0.0, 0.0, 0.0),
    ) -> str:
        """Export multiple 3D fields as a single multi-scalar VTK file.

        All fields must have the same shape.

        Args:
            fields: Dict of name -> 3D array.
            output_path: Path to write .vtk file.
            resolution_um: Voxel spacing.
            origin: Origin coordinates.

        Returns:
            Path to written file.
        """
        if not fields:
            return ""

        first_field = next(iter(fields.values()))
        nx, ny, nz = first_field.shape
        spacing = (resolution_um, resolution_um, resolution_um)
        n_points = nx * ny * nz

        lines = [
            "# vtk DataFile Version 3.0",
            f"Cognisom multi-field ({nx}x{ny}x{nz}, {len(fields)} fields)",
            "ASCII",
            "DATASET STRUCTURED_POINTS",
            f"DIMENSIONS {nx} {ny} {nz}",
            f"ORIGIN {origin[0]} {origin[1]} {origin[2]}",
            f"SPACING {spacing[0]} {spacing[1]} {spacing[2]}",
            f"POINT_DATA {n_points}",
        ]

        for fname, field in fields.items():
            lines.append(f"SCALARS {fname} float 1")
            lines.append("LOOKUP_TABLE default")
            values = field.flatten(order="F")
            chunk_size = 9
            for i in range(0, len(values), chunk_size):
                chunk = values[i:i + chunk_size]
                lines.append(" ".join(f"{v:.6f}" for v in chunk))

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines))
        logger.info(f"VTK multi-field exported: {out} ({len(fields)} fields)")
        return str(out)

    # ─── CSV Export ──────────────────────────────────────────────

    @staticmethod
    def cells_to_csv(
        cells: List[Dict[str, Any]],
        output_path: str,
    ) -> str:
        """Export cells as CSV for spreadsheet / pandas analysis.

        Args:
            cells: List of cell dicts.
            output_path: Path to write .csv file.

        Returns:
            Path to written file.
        """
        if not cells:
            return ""

        header = [
            "cell_id", "cell_type", "phase", "alive",
            "x", "y", "z",
            "oxygen", "glucose", "atp",
            "mhc1_expression", "mutations",
        ]

        rows = [",".join(header)]
        for c in cells:
            pos = c.get("position", [0, 0, 0])
            mutations = c.get("mutations", [])
            mut_str = "|".join(mutations) if mutations else ""
            row = [
                str(c.get("cell_id", "")),
                c.get("cell_type", ""),
                c.get("phase", ""),
                str(c.get("alive", True)),
                f"{pos[0]:.3f}",
                f"{pos[1]:.3f}",
                f"{pos[2]:.3f}",
                f"{c.get('oxygen', 0):.4f}",
                f"{c.get('glucose', 0):.3f}",
                f"{c.get('atp', 0):.1f}",
                f"{c.get('mhc1_expression', 0):.3f}",
                mut_str,
            ]
            rows.append(",".join(row))

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(rows))
        logger.info(f"CSV exported: {out} ({len(cells)} cells)")
        return str(out)

    # ─── JSON Export ─────────────────────────────────────────────

    @staticmethod
    def scene_to_json(
        cells: List[Dict[str, Any]],
        fields: Optional[Dict[str, np.ndarray]] = None,
        interactions: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        output_path: str = "scene.json",
    ) -> str:
        """Export full scene as JSON for web-based viewers.

        Args:
            cells: Cell data.
            fields: Optional concentration fields (downsampled for size).
            interactions: Optional interaction edges.
            metadata: Optional metadata dict.
            output_path: Path to write .json file.

        Returns:
            Path to written file.
        """
        scene = {
            "version": "1.0",
            "generator": "Cognisom SceneExporter",
            "cells": cells,
        }

        if interactions:
            scene["interactions"] = interactions

        if metadata:
            scene["metadata"] = metadata

        if fields:
            field_data = {}
            for name, arr in fields.items():
                # Downsample large fields for JSON size
                if arr.size > 50000:
                    step = max(2, int(np.cbrt(arr.size / 50000)))
                    arr = arr[::step, ::step, ::step]
                field_data[name] = {
                    "shape": list(arr.shape),
                    "values": arr.flatten().tolist(),
                }
            scene["fields"] = field_data

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(scene, indent=2, default=str))
        logger.info(f"JSON scene exported: {out}")
        return str(out)

    # ─── USD Export ─────────────────────────────────────────────

    @staticmethod
    def cells_to_usd(
        cells: List[Dict[str, Any]],
        output_path: str,
        scene_name: str = "cognisom_cells",
    ) -> str:
        """Export cells as a real USD file using OpenUSD (pxr).

        Creates actual USD geometry viewable in:
        - NVIDIA Omniverse
        - usdview (pip install usd-core)
        - Blender, Houdini, Maya

        Args:
            cells: List of cell dicts with 'position', 'cell_type'.
            output_path: Path to write .usda file.
            scene_name: Name for the USD stage.

        Returns:
            Path to written file.
        """
        try:
            from pxr import Usd, UsdGeom, UsdShade, Gf, Sdf, UsdLux
        except ImportError:
            logger.error("OpenUSD not available. Install with: pip install usd-core")
            return ""

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Create stage
        stage = Usd.Stage.CreateNew(str(out))
        stage.SetMetadata("documentation", f"Cognisom cell export: {len(cells)} cells")
        stage.SetStartTimeCode(0)
        stage.SetEndTimeCode(1)

        # Create scene hierarchy
        UsdGeom.Xform.Define(stage, "/World")
        UsdGeom.Xform.Define(stage, "/World/Cells")
        UsdGeom.Xform.Define(stage, "/World/Lights")

        # Add dome light
        dome = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
        dome.GetIntensityAttr().Set(500.0)

        # Cell type colors
        type_colors = {
            "normal": (0.3, 0.6, 0.9),      # blue
            "cancer": (0.9, 0.2, 0.2),      # red
            "t_cell": (0.2, 0.9, 0.3),      # green
            "nk_cell": (0.9, 0.9, 0.2),     # yellow
            "macrophage": (0.9, 0.5, 0.2),  # orange
            "stem": (0.4, 0.9, 0.5),        # light green
            "progenitor": (0.5, 0.7, 0.9),  # light blue
            "differentiated": (0.9, 0.6, 0.3), # orange
            "dividing": (0.9, 0.3, 0.9),    # magenta
            "immune": (0.2, 0.9, 0.3),      # green
            "dead": (0.5, 0.5, 0.5),        # gray
        }

        # Create material for each cell type used
        used_types = set(c.get("cell_type", "normal") for c in cells)
        materials = {}

        for ct in used_types:
            mat_path = f"/World/Materials/{ct}_material"
            material = UsdShade.Material.Define(stage, mat_path)
            shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")

            color = type_colors.get(ct, (0.5, 0.5, 0.5))
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.1)

            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            materials[ct] = material

        # Create cells
        for i, cell in enumerate(cells):
            pos = cell.get("position", [0, 0, 0])
            ct = cell.get("cell_type", "normal")
            alive = cell.get("alive", True)
            radius = cell.get("volume", 1.0) if "volume" in cell else 1.5

            cell_path = f"/World/Cells/Cell_{i:05d}"
            sphere = UsdGeom.Sphere.Define(stage, cell_path)
            sphere.GetRadiusAttr().Set(radius)

            # Position
            xform = UsdGeom.Xformable(sphere)
            xform.AddTranslateOp().Set(Gf.Vec3d(pos[0], pos[1], pos[2]))

            # Material
            mat_to_use = ct if alive else "dead"
            if mat_to_use in materials:
                UsdShade.MaterialBindingAPI(sphere).Bind(materials[mat_to_use])

        stage.Save()
        logger.info(f"USD exported: {out} ({len(cells)} cells)")
        return str(out)
