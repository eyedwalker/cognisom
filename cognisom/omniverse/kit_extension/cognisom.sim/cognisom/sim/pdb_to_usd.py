"""
PDB to USD Converter
====================

Convert PDB/mmCIF protein structure data into OpenUSD scene geometry.
Supports ball-and-stick, ribbon, and molecular surface representations.

Designed for headless Kit RTX rendering on L40S.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# CPK atom colors (RGB 0-1)
ATOM_COLORS = {
    "C": (0.40, 0.40, 0.40),
    "N": (0.15, 0.15, 0.90),
    "O": (0.90, 0.10, 0.10),
    "S": (0.90, 0.80, 0.15),
    "H": (0.85, 0.85, 0.85),
    "P": (0.80, 0.50, 0.10),
    "FE": (0.70, 0.40, 0.10),
    "ZN": (0.50, 0.50, 0.70),
    "CA": (0.15, 0.70, 0.15),
    "MG": (0.15, 0.60, 0.15),
}
DEFAULT_ATOM_COLOR = (0.60, 0.30, 0.60)

# Van der Waals radii (angstroms)
ATOM_RADII = {
    "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "H": 1.20,
    "P": 1.80, "FE": 1.47, "ZN": 1.39, "CA": 1.97, "MG": 1.73,
}
DEFAULT_ATOM_RADIUS = 1.50

# pLDDT confidence color scale (AlphaFold)
# Very high (90-100): blue, High (70-90): cyan, Low (50-70): yellow, Very low (<50): red
PLDDT_COLORS = [
    (0.90, 0.20, 0.20),  # <50: red
    (0.90, 0.80, 0.20),  # 50-70: yellow
    (0.20, 0.80, 0.80),  # 70-90: cyan
    (0.20, 0.20, 0.90),  # 90-100: blue
]

# Amino acid one-letter to three-letter mapping
AA_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# Secondary structure color
SS_COLORS = {
    "helix": (0.90, 0.20, 0.30),   # Red for alpha-helices
    "sheet": (0.20, 0.60, 0.90),   # Blue for beta-sheets
    "coil": (0.70, 0.70, 0.70),    # Gray for coils/loops
}


def _plddt_color(plddt: float) -> Tuple[float, float, float]:
    """Map pLDDT score to color."""
    if plddt >= 90:
        return PLDDT_COLORS[3]
    elif plddt >= 70:
        return PLDDT_COLORS[2]
    elif plddt >= 50:
        return PLDDT_COLORS[1]
    return PLDDT_COLORS[0]


class PDBAtom:
    """Parsed PDB atom record."""
    __slots__ = [
        "serial", "name", "res_name", "chain", "res_seq",
        "x", "y", "z", "element", "bfactor",
    ]

    def __init__(self, serial, name, res_name, chain, res_seq, x, y, z, element, bfactor):
        self.serial = serial
        self.name = name
        self.res_name = res_name
        self.chain = chain
        self.res_seq = res_seq
        self.x = x
        self.y = y
        self.z = z
        self.element = element
        self.bfactor = bfactor

    @property
    def position(self):
        return (self.x, self.y, self.z)

    @property
    def is_ca(self):
        return self.name.strip() == "CA"

    @property
    def is_backbone(self):
        return self.name.strip() in ("N", "CA", "C", "O")

    @property
    def color(self):
        return ATOM_COLORS.get(self.element.upper(), DEFAULT_ATOM_COLOR)

    @property
    def radius(self):
        return ATOM_RADII.get(self.element.upper(), DEFAULT_ATOM_RADIUS)


def parse_pdb(pdb_text: str) -> List[PDBAtom]:
    """Parse PDB-format text into atom records."""
    atoms = []
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            serial = int(line[6:11].strip())
            name = line[12:16].strip()
            res_name = line[17:20].strip()
            chain = line[21:22].strip() or "A"
            res_seq = int(line[22:26].strip())
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            bfactor = float(line[60:66].strip()) if len(line) >= 66 and line[60:66].strip() else 0.0
            element = line[76:78].strip() if len(line) >= 78 else name[0]
            atoms.append(PDBAtom(serial, name, res_name, chain, res_seq, x, y, z, element, bfactor))
        except (ValueError, IndexError):
            continue
    return atoms


def get_ca_trace(atoms: List[PDBAtom]) -> List[PDBAtom]:
    """Extract C-alpha trace (one atom per residue)."""
    return [a for a in atoms if a.is_ca]


def estimate_bonds(atoms: List[PDBAtom], cutoff: float = 1.9) -> List[Tuple[int, int]]:
    """Estimate covalent bonds by distance (within same residue or sequential residues)."""
    bonds = []
    n = len(atoms)
    # Build spatial index for efficiency
    coords = np.array([(a.x, a.y, a.z) for a in atoms])

    for i in range(n):
        for j in range(i + 1, min(i + 30, n)):  # Only check nearby atoms
            # Only bond within same residue or adjacent residues
            if abs(atoms[i].res_seq - atoms[j].res_seq) > 1:
                continue
            if atoms[i].chain != atoms[j].chain:
                continue
            dist = math.sqrt(
                (coords[i][0] - coords[j][0]) ** 2 +
                (coords[i][1] - coords[j][1]) ** 2 +
                (coords[i][2] - coords[j][2]) ** 2
            )
            if dist < cutoff:
                bonds.append((i, j))
    return bonds


class PDBtoUSD:
    """Convert PDB structure data to USD scene geometry.

    Creates USD prims (spheres for atoms, cylinders for bonds)
    that can be rendered by Kit's RTX renderer.

    Example:
        converter = PDBtoUSD()
        atoms = parse_pdb(pdb_text)
        converter.build_ball_and_stick(atoms, stage, "/World/Protein")
    """

    def __init__(self, atom_scale: float = 0.4, bond_radius: float = 0.15):
        """
        Args:
            atom_scale: Scale factor for atom spheres (relative to VdW radius).
            bond_radius: Radius of bond cylinders in angstroms.
        """
        self.atom_scale = atom_scale
        self.bond_radius = bond_radius

    def build_ball_and_stick(self, atoms: List[PDBAtom], stage, root_path: str,
                              color_mode: str = "element",
                              mutation_residues: Optional[List[int]] = None):
        """Build ball-and-stick model.

        Args:
            atoms: Parsed PDB atoms.
            stage: USD stage.
            root_path: Root prim path (e.g. "/World/Protein").
            color_mode: "element" (CPK), "bfactor" (pLDDT), "chain".
            mutation_residues: Residue IDs to highlight in red.
        """
        try:
            from pxr import UsdGeom, Gf, Sdf, UsdShade
        except ImportError:
            logger.error("USD (pxr) not available — cannot build molecular scene")
            return

        mutation_set = set(mutation_residues or [])

        # Create root xform
        root = UsdGeom.Xform.Define(stage, root_path)

        # Center the protein
        coords = np.array([(a.x, a.y, a.z) for a in atoms])
        centroid = coords.mean(axis=0) if len(coords) > 0 else np.zeros(3)

        # Create atoms as spheres
        atoms_path = f"{root_path}/atoms"
        UsdGeom.Xform.Define(stage, atoms_path)

        for i, atom in enumerate(atoms):
            # Skip hydrogens for cleaner visualization
            if atom.element.upper() == "H":
                continue

            sphere_path = f"{atoms_path}/atom_{i}"
            sphere = UsdGeom.Sphere.Define(stage, sphere_path)
            sphere.GetRadiusAttr().Set(atom.radius * self.atom_scale)

            # Position (centered)
            xf = UsdGeom.Xformable(sphere.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(
                atom.x - centroid[0],
                atom.y - centroid[1],
                atom.z - centroid[2],
            ))

            # Color
            if atom.res_seq in mutation_set:
                color = (1.0, 0.0, 0.0)  # Red for mutations
            elif color_mode == "bfactor":
                color = _plddt_color(atom.bfactor)
            else:
                color = atom.color

            sphere.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])

        # Create bonds as cylinders
        non_h_atoms = [a for a in atoms if a.element.upper() != "H"]
        bonds = estimate_bonds(non_h_atoms)

        if bonds:
            bonds_path = f"{root_path}/bonds"
            UsdGeom.Xform.Define(stage, bonds_path)

            for b_idx, (i, j) in enumerate(bonds):
                a1, a2 = non_h_atoms[i], non_h_atoms[j]
                self._create_bond_cylinder(
                    stage, f"{bonds_path}/bond_{b_idx}",
                    (a1.x - centroid[0], a1.y - centroid[1], a1.z - centroid[2]),
                    (a2.x - centroid[0], a2.y - centroid[1], a2.z - centroid[2]),
                )

        logger.info(
            f"Built ball-and-stick: {len(atoms)} atoms, {len(bonds)} bonds "
            f"at {root_path}"
        )

    def build_ribbon(self, atoms: List[PDBAtom], stage, root_path: str,
                     color_mode: str = "bfactor",
                     mutation_residues: Optional[List[int]] = None):
        """Build ribbon diagram from C-alpha trace.

        Uses tube geometry along the C-alpha backbone with B-spline smoothing.
        """
        try:
            from pxr import UsdGeom, Gf, Sdf
        except ImportError:
            logger.error("USD (pxr) not available")
            return

        ca_atoms = get_ca_trace(atoms)
        if len(ca_atoms) < 4:
            logger.warning("Not enough CA atoms for ribbon — falling back to ball-and-stick")
            self.build_ball_and_stick(atoms, stage, root_path, color_mode, mutation_residues)
            return

        mutation_set = set(mutation_residues or [])

        # Center
        coords = np.array([(a.x, a.y, a.z) for a in ca_atoms])
        centroid = coords.mean(axis=0)
        centered = coords - centroid

        # Smooth with Catmull-Rom spline
        smooth_points = self._catmull_rom_spline(centered, subdivisions=4)

        root = UsdGeom.Xform.Define(stage, root_path)
        ribbon_path = f"{root_path}/ribbon"

        # Create tube as a BasisCurves prim
        curves = UsdGeom.BasisCurves.Define(stage, ribbon_path)
        points = [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in smooth_points]
        curves.GetPointsAttr().Set(points)
        curves.GetCurveVertexCountsAttr().Set([len(points)])
        curves.GetTypeAttr().Set("cubic")
        curves.GetBasisAttr().Set("catmullRom")

        # Varying width (thicker at helices — simplified)
        widths = [0.8] * len(points)
        curves.GetWidthsAttr().Set(widths)

        # Colors per vertex
        colors = []
        n_ca = len(ca_atoms)
        pts_per_ca = len(smooth_points) // n_ca if n_ca > 0 else 1

        for i, pt_idx in enumerate(range(0, len(smooth_points))):
            ca_idx = min(pt_idx // max(1, pts_per_ca), n_ca - 1)
            ca = ca_atoms[ca_idx]

            if ca.res_seq in mutation_set:
                colors.append(Gf.Vec3f(1.0, 0.0, 0.0))
            elif color_mode == "bfactor":
                c = _plddt_color(ca.bfactor)
                colors.append(Gf.Vec3f(*c))
            else:
                colors.append(Gf.Vec3f(0.2, 0.5, 0.9))

        curves.GetDisplayColorAttr().Set(colors)

        # Also add spheres at mutation sites
        if mutation_set:
            mut_path = f"{root_path}/mutations"
            UsdGeom.Xform.Define(stage, mut_path)
            for ca in ca_atoms:
                if ca.res_seq in mutation_set:
                    sphere = UsdGeom.Sphere.Define(
                        stage, f"{mut_path}/mut_{ca.res_seq}"
                    )
                    sphere.GetRadiusAttr().Set(2.0)
                    xf = UsdGeom.Xformable(sphere.GetPrim())
                    xf.AddTranslateOp().Set(Gf.Vec3d(
                        ca.x - centroid[0],
                        ca.y - centroid[1],
                        ca.z - centroid[2],
                    ))
                    sphere.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])

        logger.info(
            f"Built ribbon: {len(ca_atoms)} residues, {len(smooth_points)} spline points "
            f"at {root_path}"
        )

    def build_surface(self, atoms: List[PDBAtom], stage, root_path: str,
                      probe_radius: float = 1.4, grid_spacing: float = 1.0,
                      mutation_residues: Optional[List[int]] = None):
        """Build molecular surface mesh (Gaussian surface approximation).

        Creates a mesh by evaluating a sum-of-Gaussians on a 3D grid
        and extracting an isosurface via marching cubes.
        """
        try:
            from pxr import UsdGeom, Gf, Vt
        except ImportError:
            logger.error("USD (pxr) not available")
            return

        non_h = [a for a in atoms if a.element.upper() != "H"]
        if not non_h:
            return

        coords = np.array([(a.x, a.y, a.z) for a in non_h])
        radii = np.array([a.radius + probe_radius for a in non_h])
        centroid = coords.mean(axis=0)
        centered = coords - centroid

        # Bounding box
        margin = 4.0
        mins = centered.min(axis=0) - margin
        maxs = centered.max(axis=0) + margin
        shape = np.ceil((maxs - mins) / grid_spacing).astype(int)
        shape = np.clip(shape, 5, 80)  # Limit resolution for performance

        # Evaluate Gaussian density
        grid = np.zeros(tuple(shape), dtype=np.float32)
        for i in range(len(non_h)):
            # Convert atom position to grid coordinates
            gx = int((centered[i, 0] - mins[0]) / grid_spacing)
            gy = int((centered[i, 1] - mins[1]) / grid_spacing)
            gz = int((centered[i, 2] - mins[2]) / grid_spacing)

            r = int(radii[i] / grid_spacing) + 2
            sigma = radii[i] * 0.5

            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    for dz in range(-r, r + 1):
                        ix, iy, iz = gx + dx, gy + dy, gz + dz
                        if 0 <= ix < shape[0] and 0 <= iy < shape[1] and 0 <= iz < shape[2]:
                            dist_sq = (dx * grid_spacing) ** 2 + (dy * grid_spacing) ** 2 + (dz * grid_spacing) ** 2
                            grid[ix, iy, iz] += math.exp(-dist_sq / (2 * sigma ** 2))

        # Extract isosurface via simple marching cubes
        verts, faces = self._marching_cubes_simple(grid, threshold=0.5, spacing=grid_spacing, origin=mins)

        if len(verts) == 0:
            logger.warning("No surface generated — falling back to ribbon")
            self.build_ribbon(atoms, stage, root_path, mutation_residues=mutation_residues)
            return

        # Create USD mesh
        root = UsdGeom.Xform.Define(stage, root_path)
        mesh_path = f"{root_path}/surface"
        mesh = UsdGeom.Mesh.Define(stage, mesh_path)

        points = [Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in verts]
        mesh.GetPointsAttr().Set(points)

        face_counts = [3] * len(faces)
        face_indices = []
        for f in faces:
            face_indices.extend(f)

        mesh.GetFaceVertexCountsAttr().Set(face_counts)
        mesh.GetFaceVertexIndicesAttr().Set(face_indices)
        mesh.GetSubdivisionSchemeAttr().Set("none")

        # Color by proximity to mutation sites
        mutation_set = set(mutation_residues or [])
        if mutation_set:
            mut_coords = np.array([
                (a.x - centroid[0], a.y - centroid[1], a.z - centroid[2])
                for a in non_h if a.res_seq in mutation_set
            ])
            colors = []
            for v in verts:
                if len(mut_coords) > 0:
                    dists = np.sqrt(((mut_coords - np.array(v)) ** 2).sum(axis=1))
                    min_dist = dists.min()
                    if min_dist < 5.0:
                        t = min_dist / 5.0
                        c = (1.0 - t, t * 0.5, t * 0.5)
                    else:
                        c = (0.7, 0.7, 0.8)
                else:
                    c = (0.7, 0.7, 0.8)
                colors.append(Gf.Vec3f(*c))
            mesh.GetDisplayColorAttr().Set(colors)
        else:
            mesh.GetDisplayColorAttr().Set([Gf.Vec3f(0.7, 0.7, 0.8)])

        logger.info(
            f"Built surface: {len(verts)} vertices, {len(faces)} triangles "
            f"at {root_path}"
        )

    def _create_bond_cylinder(self, stage, path: str,
                               p1: tuple, p2: tuple):
        """Create a cylinder between two points."""
        from pxr import UsdGeom, Gf

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dz = p2[2] - p1[2]
        length = math.sqrt(dx * dx + dy * dy + dz * dz)

        if length < 0.01:
            return

        mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, (p1[2] + p2[2]) / 2)

        cyl = UsdGeom.Cylinder.Define(stage, path)
        cyl.GetRadiusAttr().Set(self.bond_radius)
        cyl.GetHeightAttr().Set(length)

        xf = UsdGeom.Xformable(cyl.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(*mid))

        # Orient cylinder along bond direction
        # Default cylinder is along Y axis
        bond_dir = np.array([dx, dy, dz]) / length
        up = np.array([0, 1, 0])

        if abs(np.dot(bond_dir, up)) > 0.99:
            up = np.array([1, 0, 0])

        right = np.cross(up, bond_dir)
        right_len = np.linalg.norm(right)
        if right_len > 1e-6:
            right /= right_len
            new_up = np.cross(bond_dir, right)

            # Rotation matrix → quaternion
            mat = np.eye(3)
            mat[:, 0] = right
            mat[:, 1] = bond_dir
            mat[:, 2] = new_up
            quat = self._rotation_matrix_to_quaternion(mat)
            xf.AddOrientOp().Set(Gf.Quatf(quat[0], quat[1], quat[2], quat[3]))

        cyl.GetDisplayColorAttr().Set([Gf.Vec3f(0.6, 0.6, 0.6)])

    @staticmethod
    def _catmull_rom_spline(points: np.ndarray, subdivisions: int = 4) -> np.ndarray:
        """Catmull-Rom spline interpolation through points."""
        if len(points) < 4:
            return points

        result = []
        for i in range(len(points) - 1):
            p0 = points[max(0, i - 1)]
            p1 = points[i]
            p2 = points[min(len(points) - 1, i + 1)]
            p3 = points[min(len(points) - 1, i + 2)]

            for t_step in range(subdivisions):
                t = t_step / subdivisions
                t2 = t * t
                t3 = t2 * t

                pt = 0.5 * (
                    (2 * p1) +
                    (-p0 + p2) * t +
                    (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
                    (-p0 + 3 * p1 - 3 * p2 + p3) * t3
                )
                result.append(pt)

        result.append(points[-1])
        return np.array(result)

    @staticmethod
    def _rotation_matrix_to_quaternion(m: np.ndarray) -> tuple:
        """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
        trace = m[0, 0] + m[1, 1] + m[2, 2]
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (m[2, 1] - m[1, 2]) * s
            y = (m[0, 2] - m[2, 0]) * s
            z = (m[1, 0] - m[0, 1]) * s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
        return (w, x, y, z)

    @staticmethod
    def _marching_cubes_simple(grid: np.ndarray, threshold: float = 0.5,
                                spacing: float = 1.0,
                                origin: np.ndarray = None) -> tuple:
        """Simplified marching cubes — extracts vertices and triangular faces."""
        if origin is None:
            origin = np.zeros(3)

        verts = []
        faces = []
        vert_map = {}

        sx, sy, sz = grid.shape

        def interp_edge(p1, v1, p2, v2):
            """Interpolate vertex position on edge."""
            if abs(v1 - v2) < 1e-10:
                t = 0.5
            else:
                t = (threshold - v1) / (v2 - v1)
            return (
                origin[0] + (p1[0] + t * (p2[0] - p1[0])) * spacing,
                origin[1] + (p1[1] + t * (p2[1] - p1[1])) * spacing,
                origin[2] + (p1[2] + t * (p2[2] - p1[2])) * spacing,
            )

        def get_vert(key, p1, v1, p2, v2):
            if key in vert_map:
                return vert_map[key]
            v = interp_edge(p1, v1, p2, v2)
            idx = len(verts)
            verts.append(v)
            vert_map[key] = idx
            return idx

        # Process each cube
        for x in range(sx - 1):
            for y in range(sy - 1):
                for z in range(sz - 1):
                    # 8 corner values
                    vals = [
                        grid[x, y, z], grid[x + 1, y, z],
                        grid[x + 1, y + 1, z], grid[x, y + 1, z],
                        grid[x, y, z + 1], grid[x + 1, y, z + 1],
                        grid[x + 1, y + 1, z + 1], grid[x, y + 1, z + 1],
                    ]

                    # Cube index
                    idx = 0
                    for i, v in enumerate(vals):
                        if v >= threshold:
                            idx |= (1 << i)

                    if idx == 0 or idx == 255:
                        continue

                    # Simplified: create faces for edges that cross threshold
                    # Using a subset of the full marching cubes table
                    corners = [
                        (x, y, z), (x + 1, y, z),
                        (x + 1, y + 1, z), (x, y + 1, z),
                        (x, y, z + 1), (x + 1, y, z + 1),
                        (x + 1, y + 1, z + 1), (x, y + 1, z + 1),
                    ]

                    edges = [
                        (0, 1), (1, 2), (2, 3), (3, 0),
                        (4, 5), (5, 6), (6, 7), (7, 4),
                        (0, 4), (1, 5), (2, 6), (3, 7),
                    ]

                    crossing_verts = []
                    for e_idx, (c1, c2) in enumerate(edges):
                        if (vals[c1] >= threshold) != (vals[c2] >= threshold):
                            key = (
                                min(corners[c1], corners[c2]),
                                max(corners[c1], corners[c2]),
                            )
                            vi = get_vert(key, corners[c1], vals[c1], corners[c2], vals[c2])
                            crossing_verts.append(vi)

                    # Triangulate crossing vertices (fan)
                    if len(crossing_verts) >= 3:
                        for i in range(1, len(crossing_verts) - 1):
                            faces.append((
                                crossing_verts[0],
                                crossing_verts[i],
                                crossing_verts[i + 1],
                            ))

        return verts, faces
