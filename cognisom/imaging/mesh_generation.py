"""
Mesh Generation
===============

Generate simulation meshes from segmentation labels.

Converts image segmentation results into meshes suitable for
spatial simulation in Cognisom.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Mesh:
    """
    A triangular surface mesh.

    Attributes
    ----------
    vertices : np.ndarray
        (n_vertices, 3) vertex positions
    triangles : np.ndarray
        (n_triangles, 3) triangle vertex indices
    normals : Optional[np.ndarray]
        (n_triangles, 3) face normals
    """
    vertices: np.ndarray
    triangles: np.ndarray
    normals: Optional[np.ndarray] = None

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_triangles(self) -> int:
        return len(self.triangles)

    def compute_normals(self):
        """Compute face normals."""
        v0 = self.vertices[self.triangles[:, 0]]
        v1 = self.vertices[self.triangles[:, 1]]
        v2 = self.vertices[self.triangles[:, 2]]

        normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.normals = normals / norms

    def scale(self, factor: float):
        """Scale mesh by factor."""
        self.vertices *= factor

    def translate(self, offset: np.ndarray):
        """Translate mesh by offset."""
        self.vertices += offset


@dataclass
class Compartment:
    """
    A simulation compartment (cell or organelle).

    Attributes
    ----------
    label : int
        Unique compartment identifier
    name : str
        Compartment name (e.g., 'cell_1', 'nucleus')
    mesh : Optional[Mesh]
        Surface mesh
    volume : float
        Volume in physical units
    centroid : Tuple[float, float, float]
        Center of mass
    parent : Optional[int]
        Parent compartment label (for nested compartments)
    """
    label: int
    name: str
    mesh: Optional[Mesh] = None
    volume: float = 0.0
    centroid: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    parent: Optional[int] = None


@dataclass
class SimulationMesh:
    """
    Complete simulation mesh with multiple compartments.

    Attributes
    ----------
    compartments : List[Compartment]
        List of compartments
    resolution : float
        Spatial resolution (um per pixel/voxel)
    origin : Tuple[float, float, float]
        Origin in physical coordinates
    """
    compartments: List[Compartment] = field(default_factory=list)
    resolution: float = 1.0
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def n_compartments(self) -> int:
        return len(self.compartments)

    def get_compartment(self, label: int) -> Optional[Compartment]:
        """Get compartment by label."""
        for comp in self.compartments:
            if comp.label == label:
                return comp
        return None

    def get_total_volume(self) -> float:
        """Get total volume of all compartments."""
        return sum(c.volume for c in self.compartments)

    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounding box of all compartments."""
        if not self.compartments:
            return np.zeros(3), np.zeros(3)

        all_vertices = []
        for comp in self.compartments:
            if comp.mesh is not None:
                all_vertices.append(comp.mesh.vertices)

        if not all_vertices:
            return np.zeros(3), np.zeros(3)

        vertices = np.vstack(all_vertices)
        return vertices.min(axis=0), vertices.max(axis=0)


class MeshGenerator:
    """
    Generate simulation meshes from segmentation labels.

    Parameters
    ----------
    resolution : float
        Spatial resolution in um per pixel (default: 1.0)
    smoothing : float
        Mesh smoothing factor (0-1, default: 0.5)
    simplify : bool
        Whether to simplify mesh (default: True)
    target_faces : int
        Target number of faces after simplification (default: 5000)

    Example
    -------
    >>> generator = MeshGenerator(resolution=0.5)
    >>> mesh = generator.labels_to_mesh(labels)
    >>> print(f"{mesh.n_compartments} compartments")
    """

    def __init__(
        self,
        resolution: float = 1.0,
        smoothing: float = 0.5,
        simplify: bool = True,
        target_faces: int = 5000,
    ):
        self.resolution = resolution
        self.smoothing = smoothing
        self.simplify = simplify
        self.target_faces = target_faces

    def labels_to_mesh(
        self,
        labels: np.ndarray,
        level: float = 0.5,
    ) -> SimulationMesh:
        """
        Convert label image to simulation mesh.

        Parameters
        ----------
        labels : np.ndarray
            Label image (H, W) or (D, H, W)
        level : float
            Isosurface level for marching cubes

        Returns
        -------
        SimulationMesh
            Mesh with compartments for each label
        """
        result = SimulationMesh(resolution=self.resolution)

        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels > 0]

        log.info(f"Generating meshes for {len(unique_labels)} labels")

        for label in unique_labels:
            # Create binary mask for this label
            mask = (labels == label).astype(np.float32)

            # Generate mesh
            if labels.ndim == 3:
                mesh = self._marching_cubes(mask, level)
            else:
                mesh = self._contour_to_mesh(mask)

            if mesh is None:
                continue

            # Scale by resolution
            mesh.scale(self.resolution)

            # Compute volume
            volume = self._compute_volume(mask) * (self.resolution ** labels.ndim)

            # Compute centroid
            coords = np.where(mask > 0)
            centroid = tuple(
                float(np.mean(c)) * self.resolution
                for c in coords
            )

            # Create compartment
            compartment = Compartment(
                label=int(label),
                name=f"cell_{label}",
                mesh=mesh,
                volume=volume,
                centroid=centroid,
            )
            result.compartments.append(compartment)

        log.info(f"Generated {len(result.compartments)} compartment meshes")
        return result

    def _marching_cubes(
        self,
        volume: np.ndarray,
        level: float = 0.5,
    ) -> Optional[Mesh]:
        """
        Extract isosurface using marching cubes.

        Parameters
        ----------
        volume : np.ndarray
            3D volume
        level : float
            Isosurface level

        Returns
        -------
        Mesh or None
            Surface mesh
        """
        try:
            from skimage.measure import marching_cubes

            verts, faces, normals, _ = marching_cubes(
                volume,
                level=level,
                allow_degenerate=False,
            )

            if len(verts) == 0:
                return None

            return Mesh(
                vertices=verts.astype(np.float32),
                triangles=faces.astype(np.int32),
                normals=normals.astype(np.float32),
            )

        except ImportError:
            log.warning("skimage.measure.marching_cubes not available")
            return self._simple_marching_cubes(volume, level)

    def _simple_marching_cubes(
        self,
        volume: np.ndarray,
        level: float = 0.5,
    ) -> Optional[Mesh]:
        """
        Simplified marching cubes implementation.

        This is a basic implementation that extracts surface voxels
        as mesh vertices.
        """
        # Find surface voxels (boundary of object)
        from scipy import ndimage

        try:
            # Erode to find interior, subtract to get boundary
            eroded = ndimage.binary_erosion(volume > level)
            boundary = (volume > level) & ~eroded

        except ImportError:
            # Manual boundary detection
            boundary = np.zeros_like(volume, dtype=bool)
            mask = volume > level

            for d in range(volume.ndim):
                slices_forward = [slice(None)] * volume.ndim
                slices_backward = [slice(None)] * volume.ndim
                slices_forward[d] = slice(1, None)
                slices_backward[d] = slice(None, -1)

                boundary_d = mask.copy()
                boundary_d[tuple(slices_forward)] &= ~mask[tuple(slices_backward)]
                boundary |= boundary_d

        coords = np.array(np.where(boundary)).T.astype(np.float32)

        if len(coords) < 3:
            return None

        # Create simple triangulation
        # For simplicity, we'll create a point cloud as vertices
        # and attempt basic triangulation
        vertices = coords

        # Simple triangulation: connect nearby vertices
        triangles = self._triangulate_points(vertices)

        if len(triangles) == 0:
            return None

        mesh = Mesh(
            vertices=vertices,
            triangles=triangles,
        )
        mesh.compute_normals()

        return mesh

    def _triangulate_points(self, points: np.ndarray) -> np.ndarray:
        """
        Simple triangulation of point cloud.

        Creates triangles from nearest-neighbor connections.
        """
        try:
            from scipy.spatial import Delaunay

            # 2D projection for triangulation
            if points.shape[1] == 3:
                # Project to XY plane for triangulation
                tri = Delaunay(points[:, :2])
                return tri.simplices.astype(np.int32)
            else:
                tri = Delaunay(points)
                return tri.simplices.astype(np.int32)

        except ImportError:
            # Return empty if no scipy
            return np.zeros((0, 3), dtype=np.int32)

    def _contour_to_mesh(self, mask: np.ndarray) -> Optional[Mesh]:
        """
        Convert 2D mask to 3D mesh by extrusion.

        Parameters
        ----------
        mask : np.ndarray
            2D binary mask

        Returns
        -------
        Mesh or None
            Extruded mesh
        """
        try:
            from skimage.measure import find_contours

            contours = find_contours(mask, 0.5)

            if not contours:
                return None

            # Take largest contour
            contour = max(contours, key=len)

            # Create 3D mesh by extruding contour
            n_points = len(contour)
            thickness = 1.0

            # Vertices: bottom and top of extrusion
            vertices_bottom = np.column_stack([
                contour,
                np.zeros(n_points)
            ])
            vertices_top = np.column_stack([
                contour,
                np.full(n_points, thickness)
            ])
            vertices = np.vstack([vertices_bottom, vertices_top]).astype(np.float32)

            # Triangles: side faces
            triangles = []
            for i in range(n_points):
                j = (i + 1) % n_points
                # Two triangles per quad
                triangles.append([i, j, i + n_points])
                triangles.append([j, j + n_points, i + n_points])

            return Mesh(
                vertices=vertices,
                triangles=np.array(triangles, dtype=np.int32),
            )

        except ImportError:
            log.warning("skimage.measure.find_contours not available")
            return None

    def _compute_volume(self, mask: np.ndarray) -> float:
        """Compute volume (area for 2D) of binary mask."""
        return float(np.sum(mask > 0))

    def simplify_mesh(self, mesh: Mesh, target_faces: int = None) -> Mesh:
        """
        Simplify mesh by reducing face count.

        Parameters
        ----------
        mesh : Mesh
            Input mesh
        target_faces : int, optional
            Target number of faces

        Returns
        -------
        Mesh
            Simplified mesh
        """
        if target_faces is None:
            target_faces = self.target_faces

        if mesh.n_triangles <= target_faces:
            return mesh

        try:
            import trimesh

            tm = trimesh.Trimesh(
                vertices=mesh.vertices,
                faces=mesh.triangles,
            )
            tm = tm.simplify_quadric_decimation(target_faces)

            return Mesh(
                vertices=tm.vertices.astype(np.float32),
                triangles=tm.faces.astype(np.int32),
            )

        except ImportError:
            log.warning("trimesh not available for mesh simplification")
            return mesh

    def smooth_mesh(self, mesh: Mesh, iterations: int = 10) -> Mesh:
        """
        Smooth mesh using Laplacian smoothing.

        Parameters
        ----------
        mesh : Mesh
            Input mesh
        iterations : int
            Number of smoothing iterations

        Returns
        -------
        Mesh
            Smoothed mesh
        """
        try:
            import trimesh

            tm = trimesh.Trimesh(
                vertices=mesh.vertices,
                faces=mesh.triangles,
            )
            trimesh.smoothing.filter_laplacian(tm, iterations=iterations)

            return Mesh(
                vertices=tm.vertices.astype(np.float32),
                triangles=tm.faces.astype(np.int32),
            )

        except ImportError:
            log.warning("trimesh not available for mesh smoothing")
            return mesh


def labels_to_mesh(
    labels: np.ndarray,
    resolution: float = 1.0,
) -> SimulationMesh:
    """
    Convenience function to generate mesh from labels.

    Parameters
    ----------
    labels : np.ndarray
        Label image
    resolution : float
        Spatial resolution

    Returns
    -------
    SimulationMesh
        Generated mesh
    """
    generator = MeshGenerator(resolution=resolution)
    return generator.labels_to_mesh(labels)
