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


# ── Cell colour mapping for the USD export ───────────────────────────
#
# Hue carries the cell TYPE; lightness carries the cell STATE.  Both are
# authored as a per-cell/per-instance `primvars:displayColor`, which a
# single shared UsdPreviewSurface reads through a UsdPrimvarReader_float3.
# Per-instance *materials* would break instance sharing (see
# `cognisom/omniverse/instancing.py`), so state never gets its own
# material.
#
# The legend is written into the exported stage as well (stage `comment`
# metadata plus `customLayerData["cognisom:colorLegend"]`), so a render
# can always be checked against the mapping that produced it.

#: Base colour per `cell_type`, linear RGB.  Lookup is case-insensitive,
#: so the immune module's 'T_cell' / 'NK_cell' land on the same entries as
#: the dashboard's lower-case names.  'dividing', 'stem', 'progenitor' and
#: 'differentiated' are *types* in the dashboard's synthetic generator,
#: which is why they appear here as well as in the state table.
CELL_TYPE_USD_COLORS = {
    "normal": (0.26, 0.55, 0.86),          # blue
    "cancer": (0.88, 0.16, 0.16),          # red
    "t_cell": (0.20, 0.78, 0.33),          # green
    "nk_cell": (0.66, 0.31, 0.86),         # violet
    "macrophage": (0.96, 0.60, 0.16),      # orange
    "immune": (0.20, 0.78, 0.33),          # green (generic infiltrate)
    "fibroblast": (0.55, 0.40, 0.28),      # brown
    "endothelial": (0.20, 0.72, 0.74),     # teal
    "stem": (0.93, 0.33, 0.60),            # pink
    "progenitor": (0.56, 0.72, 0.93),      # pale blue
    "differentiated": (0.85, 0.66, 0.36),  # tan
    "dividing": (0.90, 0.35, 0.90),        # magenta
    "dead": (0.32, 0.32, 0.34),            # grey
}

#: Fallback for a type the table does not know.  Deliberately a neutral
#: grey: an unknown type must not borrow a hue that means something.
UNKNOWN_CELL_USD_COLOR = (0.60, 0.60, 0.62)

#: State tint applied over the type colour as `lerp(base, target, weight)`.
#: Ordered by precedence -- the first matching state wins.
CELL_STATE_USD_TINTS = {
    "dead": ((0.32, 0.32, 0.34), 1.00),       # flat grey, type erased
    "dying": ((0.38, 0.05, 0.05), 0.70),      # dark maroon
    "hypoxic": ((0.07, 0.07, 0.42), 0.60),    # dark blue-violet
    "dividing": ((1.00, 1.00, 1.00), 0.55),   # pale, bright
    "activated": ((1.00, 0.93, 0.30), 0.50),  # hot yellow
    "alive": ((0.0, 0.0, 0.0), 0.0),          # untinted
}

#: Integer code per state, written as `primvars:cognisom:cellState` so a
#: downstream tool can re-colour without parsing RGB back.
CELL_STATE_CODES = {
    "alive": 0, "dividing": 1, "activated": 2,
    "hypoxic": 3, "dying": 4, "dead": 5,
}

#: Oxygen below this flags a cell hypoxic.  This is the engine's own
#: threshold -- `VascularModule._check_hypoxia` (modules/vascular_module.py)
#: counts `cell.oxygen < 0.05` as a hypoxic region.
HYPOXIA_O2_THRESHOLD = 0.05

#: The engine's lethal thresholds: `CellularModule.update` starts killing
#: a cell when `cell.oxygen < ANOXIC_O2_LEVEL` (0.01% O2, Papandreou 2005)
#: or `cell.glucose < STARVATION_GLUCOSE_MM`.  A still-alive cell past
#: either one is drawn as 'dying'.  Copied, not imported, so this module
#: does not load the engine; tests/test_kill_calibration.py holds them equal.
DYING_O2_THRESHOLD = 0.21 * (0.0001 * 713.0) / 100.0
DYING_GLUCOSE_THRESHOLD = 0.5


def _cell_field(cell, name, default=None):
    """Read `name` off a cell that may be a dict or an engine object."""
    if isinstance(cell, dict):
        return cell.get(name, default)
    return getattr(cell, name, default)


def _cell_xyz(cell):
    """Cell position as three plain Python floats.

    `CellState.position` is a `numpy.ndarray` of float32, and
    `Gf.Vec3d(numpy.float32, ...)` raises `ArgumentError` -- which is what
    made this exporter work on the test fixtures and crash on every real
    cell.  Accepts numpy arrays, lists, tuples and anything indexable.
    """
    pos = _cell_field(cell, "position")
    if pos is None:
        return (0.0, 0.0, 0.0)
    try:
        return (float(pos[0]), float(pos[1]), float(pos[2]))
    except (TypeError, IndexError, ValueError):
        try:
            flat = list(np.asarray(pos, dtype=float).reshape(-1))
        except Exception:
            return (0.0, 0.0, 0.0)
        flat = (flat + [0.0, 0.0, 0.0])[:3]
        return (float(flat[0]), float(flat[1]), float(flat[2]))


def cell_state_name(cell) -> str:
    """Classify a cell into one of CELL_STATE_USD_TINTS.

    Precedence: dead > dying > hypoxic > dividing > activated > alive.

    A caller that already knows the state can pass it directly as a
    `state` field, or set the booleans `alive` / `dying` / `dividing` /
    `activated`.  Otherwise the state is derived from the same quantities
    the engine itself tests -- oxygen, glucose and cell-cycle phase.

    Note on 'dividing': `CellularModule` never advances `CellState.phase`
    past 'G1' (it divides on age, not on phase), so a snapshot taken from
    a live run will contain no cells in M and none will be drawn as
    dividing.  Demo cells and any module that does set phase will show it.
    """
    explicit = _cell_field(cell, "state")
    if isinstance(explicit, str) and explicit.lower() in CELL_STATE_USD_TINTS:
        return explicit.lower()

    if not bool(_cell_field(cell, "alive", True)):
        return "dead"

    if bool(_cell_field(cell, "dying", False)) or bool(
        _cell_field(cell, "apoptotic", False)
    ):
        return "dying"

    oxygen = _cell_field(cell, "oxygen")
    glucose = _cell_field(cell, "glucose")
    try:
        oxygen = None if oxygen is None else float(oxygen)
        glucose = None if glucose is None else float(glucose)
    except (TypeError, ValueError):
        oxygen = glucose = None

    if (oxygen is not None and oxygen < DYING_O2_THRESHOLD) or (
        glucose is not None and glucose < DYING_GLUCOSE_THRESHOLD
    ):
        return "dying"
    if oxygen is not None and oxygen < HYPOXIA_O2_THRESHOLD:
        return "hypoxic"

    if bool(_cell_field(cell, "dividing", False)):
        return "dividing"
    phase = _cell_field(cell, "phase")
    if isinstance(phase, str) and phase.upper() == "M":
        return "dividing"

    if bool(_cell_field(cell, "activated", False)):
        return "activated"

    return "alive"


#: The exporter's cell_type vocabulary is not the library's. Five names
#: overlap outright; the alias table maps the rest that have a real shape
#: behind them. The remainder -- normal, dead, dividing, differentiated,
#: progenitor -- are STATES wearing a type's clothing, and a state has no
#: morphology, so they get a sphere and are counted as unresolved rather
#: than drawn as whatever cell sorts first. The tables and the resolver
#: now live in :mod:`.lineage_appearance`, which is also what the run
#: exporter and the Bio-USD writer draw from; they are re-exported here
#: because this is where callers and tests have always found them.
from .lineage_appearance import (  # noqa: E402,F401
    CELL_TYPE_MESH_ALIASES, STATE_TOKENS, UNRESOLVED_PROTOTYPE,
    mesh_type_for_cell_token)


def cell_mesh_type(cell) -> Optional[str]:
    """The library mesh_type for a cell, or ``None`` if it has no shape.

    Delegates to :func:`~.lineage_appearance.mesh_type_for_cell_token`,
    which goes through ``ARCHETYPE_ENTITY`` -- the same route
    ``cell_scene`` takes to draw a library card -- so a macrophage in a
    population export and a macrophage on its card are one object. A
    state token (``normal`` among them, here: it is also this exporter's
    default for an untyped cell) resolves to ``None``.
    """
    return mesh_type_for_cell_token(_cell_field(cell, "cell_type", None))


def cell_display_color(cell):
    """RGB for one cell: type hue, tinted by state.  Returns a 3-tuple."""
    ctype = _cell_field(cell, "cell_type", "normal")
    key = str(ctype).lower() if ctype is not None else "normal"
    base = CELL_TYPE_USD_COLORS.get(key, UNKNOWN_CELL_USD_COLOR)

    state = cell_state_name(cell)
    target, weight = CELL_STATE_USD_TINTS.get(state, ((0.0, 0.0, 0.0), 0.0))
    if weight <= 0.0:
        return tuple(float(c) for c in base)
    return tuple(
        float(base[i] * (1.0 - weight) + target[i] * weight) for i in range(3)
    )


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

    #: Above this many cells the export switches from one `UsdGeomSphere`
    #: per cell to a single `UsdGeomPointInstancer`.  Below it, individual
    #: prims are easier to select and inspect; above it they are what makes
    #: a 20,000-cell stage unopenable.
    USD_INSTANCER_THRESHOLD = 256

    @staticmethod
    def cells_to_usd(
        cells: List[Any],
        output_path: str,
        scene_name: str = "cognisom_cells",
        instancer_threshold: Optional[int] = None,
        radius: Optional[float] = None,
        add_camera: bool = True,
        cutaway: str = "none",
        stroma_only: bool = False,
    ) -> str:
        """Export cells as a real USD file using OpenUSD (pxr).

        Writes a stage that renders on its own -- explicit camera, key /
        fill / dome lights, and a colour per cell -- so
        `usdrecord scene.usda out.png` produces a deterministic image
        rather than relying on the viewer's auto-framing and headlight.
        Also opens in Omniverse, Blender, Houdini and Maya.

        Args:
            cells: Cells to export.  Each may be a dict or an engine
                object (`CellularModule.CellState`, `ImmuneCell`, ...).
                Read: position, cell_type, alive, phase, oxygen, glucose,
                activated, volume/radius.  A numpy position is fine.
            output_path: Path to write .usda file.
            scene_name: Name recorded in the stage's documentation.
            instancer_threshold: Cell count at or above which a single
                PointInstancer replaces one Sphere prim per cell.
                Defaults to `SceneExporter.USD_INSTANCER_THRESHOLD`.
            radius: Cell radius in simulation units.  None (default)
                derives it per cell from `volume` (femtolitres, 1 fL =
                1 um^3) when present, otherwise from the median
                nearest-neighbour spacing of the population, so cells
                touch without interpenetrating at any scale.
            add_camera: Author a framed camera and light rig.

        Returns:
            Path to written file, or "" if OpenUSD is not installed.

        Colour: hue is the cell type, lightness is the state --
        see `CELL_TYPE_USD_COLORS` / `CELL_STATE_USD_TINTS`.  The legend
        travels with the stage in its `comment` metadata.
        """
        try:
            from pxr import Usd, UsdGeom, Gf
        except ImportError:
            logger.error("OpenUSD not available. Install with: pip install usd-core")
            return ""

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        cells = [] if cells is None else list(cells)
        n = len(cells)
        threshold = (
            SceneExporter.USD_INSTANCER_THRESHOLD
            if instancer_threshold is None else int(instancer_threshold)
        )

        positions = np.array(
            [_cell_xyz(c) for c in cells], dtype=np.float64
        ).reshape(-1, 3)
        colors = np.array(
            [cell_display_color(c) for c in cells], dtype=np.float32
        ).reshape(-1, 3)
        states = [cell_state_name(c) for c in cells]
        radii = SceneExporter._usd_cell_radii(cells, positions, radius)

        stage = Usd.Stage.CreateNew(str(out))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        # The simulation works in microns; say so rather than leaving the
        # 1-metre default to be read as a cell the size of a room.
        UsdGeom.SetStageMetersPerUnit(stage, 1e-6)
        stage.SetMetadata(
            "documentation",
            f"Cognisom cell export '{scene_name}': {n} cells, "
            f"{'point-instanced' if n >= threshold else 'one prim per cell'}",
        )
        stage.SetMetadata("comment", SceneExporter.usd_color_legend_text())
        stage.SetStartTimeCode(0)
        stage.SetEndTimeCode(1)

        UsdGeom.Xform.Define(stage, "/World")

        legend = {
            f"type:{name}": Gf.Vec3f(*rgb)
            for name, rgb in CELL_TYPE_USD_COLORS.items()
        }
        legend.update({
            f"state:{name}": Gf.Vec3f(*target)
            for name, (target, weight) in CELL_STATE_USD_TINTS.items()
            if weight > 0.0
        })
        stage.SetMetadata("customLayerData", {
            "cognisom:colorLegend": legend,
            "cognisom:cellCount": n,
            "cognisom:stateCounts": {
                s: states.count(s) for s in sorted(set(states))
            },
        })

        material = SceneExporter._usd_cell_material(stage, "/World/Looks/CellSurface")

        if n == 0:
            UsdGeom.Scope.Define(stage, "/World/Cells")
        elif n >= threshold:
            SceneExporter._usd_point_instanced_cells(
                stage, positions, colors, states, radii, material,
                mesh_types=[cell_mesh_type(c) for c in cells],
                cell_types=[str(_cell_field(c, "cell_type", "normal") or "normal")
                            for c in cells],
                cutaway=cutaway, stroma_only=stroma_only,
            )
        else:
            SceneExporter._usd_individual_cells(
                stage, positions, colors, states, radii, material
            )

        if add_camera:
            SceneExporter._usd_camera_and_lights(stage, positions, radii)

        stage.Save()
        logger.info(
            "USD exported: %s (%d cells, %s)",
            out, n, "point instancer" if n >= threshold else "individual prims",
        )
        return str(out)

    # ── USD helpers ──────────────────────────────────────────────

    @staticmethod
    def usd_color_legend_text() -> str:
        """The colour mapping, as the text embedded in every exported stage."""
        lines = [
            "Cognisom cell colours: hue = cell type, lightness = cell state.",
            "",
            "  type:",
        ]
        for name, rgb in CELL_TYPE_USD_COLORS.items():
            lines.append(
                f"    {name:<16} ({rgb[0]:.2f}, {rgb[1]:.2f}, {rgb[2]:.2f})"
            )
        lines += [
            "",
            "  state (blended over the type colour, first match wins):",
            "    dead       flat grey, type erased          (alive is False)",
            "    dying      dark maroon, 70%                (anoxic O2 < 0.00015 or glucose < 0.5)",
            "    hypoxic    dark blue-violet, 60%           (O2 < 0.05)",
            "    dividing   pale/bright, 55%                (phase == 'M')",
            "    activated  hot yellow, 50%                 (activated is True)",
            "    alive      untinted",
            "",
            "  CellularModule never advances phase past 'G1', so a live run",
            "  contains no dividing cells and none will be drawn as such.",
        ]
        return "\n".join(lines)

    @staticmethod
    def _usd_cell_radii(cells, positions, radius):
        """Per-cell radius array.

        An explicit `radius` wins.  Otherwise a cell's own `radius`, or a
        `volume` in femtolitres converted as a sphere (1 fL = 1 um^3), or
        -- when the cells carry no size at all, which is the case for
        `CellularModule.CellState` -- 0.45x the median nearest-neighbour
        spacing, so the population reads as tissue at any density instead
        of as dust at one scale and a solid blob at another.
        """
        n = len(cells)
        if n == 0:
            return np.zeros(0, dtype=np.float64)

        if radius is not None:
            return np.full(n, float(radius), dtype=np.float64)

        default = SceneExporter._usd_spacing_radius(positions)
        out = np.full(n, default, dtype=np.float64)
        for i, cell in enumerate(cells):
            r = _cell_field(cell, "radius")
            if r is None:
                vol = _cell_field(cell, "volume")
                if vol is not None:
                    try:
                        r = (3.0 * float(vol) / (4.0 * np.pi)) ** (1.0 / 3.0)
                    except (TypeError, ValueError):
                        r = None
            if r is not None:
                try:
                    r = float(r)
                except (TypeError, ValueError):
                    continue
                if r > 0:
                    out[i] = r
        return out

    @staticmethod
    def _usd_spacing_radius(positions) -> float:
        """0.45x the median nearest-neighbour distance (1.5 if undefined).

        Measured exactly, on every cell.  Subsampling is not an option
        here even though it looks cheap: nearest-neighbour distance scales
        as density^(-1/3), so a sample of a tenth of the cells reports a
        spacing about twice the real one and the cells come out twice too
        big.  scipy is a declared dependency; without it, fall back to the
        O(n^2) measurement while it is affordable and to a fixed radius
        above that.
        """
        n = len(positions)
        if n < 2:
            return 1.5
        try:
            from scipy.spatial import cKDTree
            nearest = cKDTree(positions).query(positions, k=2)[0][:, 1]
        except Exception:
            if n > 3000:
                return 1.5
            d = np.linalg.norm(
                positions[:, None, :] - positions[None, :, :], axis=-1
            )
            np.fill_diagonal(d, np.inf)
            nearest = d.min(axis=1)
        nearest = np.asarray(nearest)
        nearest = nearest[np.isfinite(nearest) & (nearest > 0)]
        if nearest.size == 0:
            return 1.5
        return float(np.median(nearest) * 0.45)

    @staticmethod
    def _usd_cell_material(stage, path):
        """One UsdPreviewSurface whose diffuse colour reads displayColor.

        One material for every cell, of every type, in every state: the
        variation rides on the primvar.  A material per instance would
        defeat instance sharing -- the point the instancing module makes
        at length -- and a material per type could not express state.
        """
        from pxr import UsdShade, Sdf, Gf

        material = UsdShade.Material.Define(stage, path)

        reader = UsdShade.Shader.Define(stage, f"{path}/ColorReader")
        reader.CreateIdAttr("UsdPrimvarReader_float3")
        reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("displayColor")
        reader.CreateInput("fallback", Sdf.ValueTypeNames.Float3).Set(
            Gf.Vec3f(*UNKNOWN_CELL_USD_COLOR)
        )
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float3)

        shader = UsdShade.Shader.Define(stage, f"{path}/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            reader.ConnectableAPI(), "result"
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )
        return material

    @staticmethod
    def _usd_individual_cells(stage, positions, colors, states, radii, material):
        """One Sphere prim per cell -- selectable, inspectable, small N."""
        from pxr import UsdGeom, UsdShade, Gf, Sdf

        UsdGeom.Xform.Define(stage, "/World/Cells")
        for i in range(len(positions)):
            r = float(radii[i])
            path = f"/World/Cells/Cell_{i:05d}"
            sphere = UsdGeom.Sphere.Define(stage, path)
            sphere.GetRadiusAttr().Set(r)
            # Authoring extent matters: the fallback is the unit sphere,
            # which makes every bounding-box query wrong for r != 1.
            sphere.GetExtentAttr().Set([(-r, -r, -r), (r, r, r)])

            UsdGeom.Xformable(sphere).AddTranslateOp().Set(
                Gf.Vec3d(float(positions[i][0]), float(positions[i][1]),
                         float(positions[i][2]))
            )

            pv = UsdGeom.PrimvarsAPI(sphere.GetPrim()).CreatePrimvar(
                "displayColor", Sdf.ValueTypeNames.Color3fArray,
                UsdGeom.Tokens.constant,
            )
            pv.Set([Gf.Vec3f(*(float(c) for c in colors[i]))])
            UsdGeom.PrimvarsAPI(sphere.GetPrim()).CreatePrimvar(
                "cognisom:cellState", Sdf.ValueTypeNames.Int,
                UsdGeom.Tokens.constant,
            ).Set(CELL_STATE_CODES.get(states[i], 0))

            UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim())
            UsdShade.MaterialBindingAPI(sphere).Bind(material)

    @staticmethod
    def _usd_point_instanced_cells(stage, positions, colors, states, radii,
                                   material, mesh_types=None, cell_types=None,
                                   cutaway="none", stroma_only=False):
        """One PointInstancer for the whole population.

        The prototype used to be a single `UsdGeom.Sphere` of radius 1,
        which made a 500,000-cell export half a million copies of a
        sphere while the library held 32 real morphologies the library
        card already drew.  `mesh_types` carries one library mesh_type
        per cell, resolved through the same `ARCHETYPE_ENTITY` route
        `cell_scene` takes, and each distinct one becomes a prototype.

        A cell whose type resolves to no shape instances an explicitly
        named sphere rather than borrowing another cell's morphology.
        That matters: several of the exporter's "types" -- dead,
        dividing, progenitor -- are states, and a state has no shape.

        `NestedInstancingManager` is no longer used here.  It authors one
        instancer per prototype, and the export's own tests read
        `_instancers(stage)[0]`, so several prototypes have to share one
        instancer with `protoIndices`.
        """
        from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

        from cognisom.biousd.usd_writer import author_prototypes

        n = len(positions)
        names = list(mesh_types or [None] * n)
        resolved = sorted({m for m in names if m})
        order = resolved + [UNRESOLVED_PROTOTYPE]
        index_of = {name: i for i, name in enumerate(order)}

        proto_paths, _faces = author_prototypes(stage, resolved)
        fallback = UsdGeom.Sphere.Define(
            stage, f"/Prototypes/{UNRESOLVED_PROTOTYPE}")
        fallback.GetRadiusAttr().Set(1.0)
        fallback.GetExtentAttr().Set([(-1, -1, -1), (1, 1, 1)])
        proto_paths = list(proto_paths) + [fallback.GetPath()]
        for path in proto_paths:
            prim = stage.GetPrimAtPath(path)
            UsdShade.MaterialBindingAPI.Apply(prim)
            UsdShade.MaterialBindingAPI(prim).Bind(material)

        instancer = UsdGeom.PointInstancer.Define(
            stage, "/World/Cells/CellBody_Instancer")
        instancer.GetPrototypesRel().SetTargets(proto_paths)
        instancer.GetPositionsAttr().Set(
            Vt.Vec3fArray.FromNumpy(np.asarray(positions, dtype="float32")))
        instancer.GetProtoIndicesAttr().Set(Vt.IntArray.FromNumpy(np.array(
            [index_of.get(m or UNRESOLVED_PROTOTYPE,
                          index_of[UNRESOLVED_PROTOTYPE]) for m in names],
            dtype="int32")))
        scale = np.asarray(radii, dtype="float32")[:, None]
        instancer.GetScalesAttr().Set(Vt.Vec3fArray.FromNumpy(
            np.repeat(scale, 3, axis=1).astype("float32")))

        api = UsdGeom.PrimvarsAPI(instancer.GetPrim())
        api.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray,
                          UsdGeom.Tokens.varying).Set(
            Vt.Vec3fArray.FromNumpy(np.asarray(colors, dtype="float32")))
        api.CreatePrimvar("cognisom:cellState", Sdf.ValueTypeNames.IntArray,
                          UsdGeom.Tokens.varying).Set(
            Vt.IntArray.FromNumpy(np.array(
                [CELL_STATE_CODES.get(s, 0) for s in states], dtype="int32")))

        # A dense population renders as a solid brick: 500,000 cells come
        # back in 0.8 s and the frame shows nothing but its own surface.
        # Speed was never the constraint here, occlusion is, and
        # `cutaway` already worked this out -- of 20,000 cells, 15,200
        # stroma hide the 2,378 that carry the biology. It has been
        # sitting unwired to this path.
        #
        # invisibleIds is the PointInstancer's own mechanism, so a cut
        # costs an int array rather than a second export. The cut plane
        # needs only the direction to the camera, which is why this does
        # not have to wait for _usd_camera_and_lights to fit an eye
        # point further down.
        #
        # The two knobs answer different questions. `section` alone is an
        # anatomical slice: it cuts the lesion at the same rate as the
        # stroma (measured 50.7% against 50.0%) and you read the cut
        # face. `stroma_only` is a reveal, keeping every lesion cell
        # whichever side of the plane it sits on -- which is the view
        # this platform usually wants, because the lesion is the subject
        # and the stroma is what is in the way.
        if cutaway and cutaway != "none" and cell_types is not None:
            from cognisom.visualization.cutaway import (
                hidden_for_camera_direction)

            hidden = hidden_for_camera_direction(
                positions, cell_types, SceneExporter.USD_CAMERA_DIRECTION,
                mode=cutaway, stroma_only=stroma_only)
            if hidden:
                instancer.CreateInvisibleIdsAttr().Set(
                    Vt.Int64Array([int(i) for i in hidden]))

        if instancer:
            lo = (positions - radii[:, None]).min(axis=0)
            hi = (positions + radii[:, None]).max(axis=0)
            instancer.CreateExtentAttr().Set([
                Gf.Vec3f(*(float(v) for v in lo)),
                Gf.Vec3f(*(float(v) for v in hi)),
            ])

    #: Where the camera sits, as a direction from the cells (Z up).
    USD_CAMERA_DIRECTION = (0.62, -1.0, 0.45)

    @staticmethod
    def _usd_camera_and_lights(stage, positions, radii):
        """A camera framed on the cells, plus a key / fill / dome rig.

        Without this the image depends on the viewer: `usdrecord`
        auto-frames and lights with a headlight, which is neither
        reproducible nor readable for a dense population.

        The frame is fitted, not guessed: every cell is projected onto the
        camera's own right/up axes and the distance is the smallest one
        that still contains the furthest of them.  A bounding sphere would
        be far looser -- a flat tissue slab seen from a corner would sit
        in the middle third of the image.
        """
        from pxr import UsdGeom, UsdLux, Gf

        focal, aperture = 35.0, 24.0   # 37.8 deg FOV, square -> square image
        tan_half_fov = (aperture * 0.5) / focal
        up_world = np.array([0.0, 0.0, 1.0])

        direction = np.asarray(SceneExporter.USD_CAMERA_DIRECTION, dtype=float)
        direction = direction / np.linalg.norm(direction)

        forward = -direction
        right = np.cross(forward, up_world)
        if np.linalg.norm(right) < 1e-9:          # looking straight down
            right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
        right = right / np.linalg.norm(right)
        up_cam = np.cross(right, forward)

        if len(positions) == 0:
            center = np.zeros(3)
            distance = 10.0
            near, far = 0.1, 100.0
        else:
            lo = (positions - radii[:, None]).min(axis=0)
            hi = (positions + radii[:, None]).max(axis=0)
            center = (lo + hi) * 0.5

            rel = positions - center
            a, b = rel @ right, rel @ up_cam
            # Re-centre on what is actually visible, so the population
            # sits in the middle of the frame rather than at the midpoint
            # of a bounding box a few outliers stretched.
            center = center                 + 0.5 * ((a + radii).max() + (a - radii).min()) * right                 + 0.5 * ((b + radii).max() + (b - radii).min()) * up_cam

            rel = positions - center
            a, b = rel @ right, rel @ up_cam
            depth = rel @ forward
            need = np.maximum(np.abs(a), np.abs(b)) + radii
            distance = float((need / tan_half_fov - depth).max()) * 1.03
            distance = max(distance, float(radii.max()) * 4.0, 1e-3)

            near = max(distance + float(depth.min()) - float(radii.max()),
                       distance * 1e-3)
            far = distance + float(depth.max()) + float(radii.max()) * 2.0 + 1.0

        ctr = Gf.Vec3d(*(float(v) for v in center))
        eye = center + direction * distance

        def _look_at(eye_v, up_hint):
            """Transform for a prim at `eye_v` aimed at the cells.

            `Gf.Matrix4d.SetLookAt` degenerates to garbage -- a camera
            that renders an empty frame -- when the up vector is parallel
            to the view direction, which is exactly what a straight-down
            view asks for.  Pick another up in that case.
            """
            f = np.asarray(ctr, dtype=float) - np.asarray(eye_v, dtype=float)
            u = np.asarray(up_hint, dtype=float)
            for candidate in (u, np.array([0.0, 1.0, 0.0]),
                              np.array([1.0, 0.0, 0.0])):
                if np.linalg.norm(np.cross(f, candidate)) > 1e-9:
                    u = candidate
                    break
            m = Gf.Matrix4d()
            m.SetLookAt(Gf.Vec3d(*(float(v) for v in eye_v)), ctr,
                        Gf.Vec3d(*(float(v) for v in u)))
            return m.GetInverse()

        camera = UsdGeom.Camera.Define(stage, "/World/Camera")
        camera.CreateFocalLengthAttr(focal)
        camera.CreateHorizontalApertureAttr(aperture)
        camera.CreateVerticalApertureAttr(aperture)
        camera.CreateClippingRangeAttr(Gf.Vec2f(float(near), float(far)))
        UsdGeom.Xformable(camera).AddTransformOp().Set(_look_at(eye, up_cam))

        UsdGeom.Xform.Define(stage, "/World/Lights")

        def _distant(name, from_dir, intensity):
            """A DistantLight aimed at the centre from `from_dir`."""
            light = UsdLux.DistantLight.Define(stage, f"/World/Lights/{name}")
            light.CreateIntensityAttr(intensity)
            light.CreateAngleAttr(1.5)
            d = np.asarray(from_dir, dtype=float)
            d = d / np.linalg.norm(d)
            pos = center + d * (distance * 2.0)
            UsdGeom.Xformable(light).AddTransformOp().Set(_look_at(pos, up_world))
            return light

        _distant("Key", (0.35, -0.75, 0.9), 1.6)
        _distant("Fill", (-0.9, -0.3, 0.1), 0.45)

        # Kept from the original rig for ambient fill.  usdrecord warns
        # "Dome light has no texture asset path" -- harmless, and silencing
        # it would mean shipping an HDRI next to every .usda.
        dome = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
        dome.CreateIntensityAttr(0.35)
