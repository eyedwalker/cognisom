"""Multi-GPU spatial-hash cell mechanics for tissue-scale simulation.

Replaces O(N²) pairwise force computation with O(N·k) spatial hash lookup.
Each GPU builds a local spatial hash over its cells (real + ghost), then
computes JKR adhesion + Hertz repulsion forces using 27-bucket neighbor search.

Supports both real multi-GPU (CuPy) and CPU simulation mode (NumPy).

Usage::

    mechanics = MultiGPUCellMechanics(multi_gpu, decomposer, ghost_exchanger)
    mechanics.build_spatial_hash(cell_arrays)
    forces = mechanics.compute_forces(cell_arrays)
    mechanics.integrate(cell_arrays, forces, dt)

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ── Physics constants ─────────────────────────────────────────────

@dataclass
class MechanicsConfig:
    """Parameters for cell-cell mechanical interactions."""

    # JKR adhesion (Hertz-like repulsion + adhesion)
    repulsion_stiffness: float = 2000.0    # pN/um
    adhesion_strength: float = 400.0       # pN/um
    damping_coeff: float = 10.0            # pN·s/um (overdamped regime)

    # Migration
    migration_force: float = 100.0         # pN (chemotaxis base force)
    random_force: float = 20.0             # pN (Brownian-like fluctuations)

    # Spatial hash
    bucket_size_um: float = 15.0           # Should be >= 2 * max_cell_radius
    max_neighbors: int = 64                # Max contacts per cell

    # Integration
    max_displacement_um: float = 1.0       # Clamp per step (stability)

    @staticmethod
    def default() -> MechanicsConfig:
        return MechanicsConfig()


# ── Spatial Hash ──────────────────────────────────────────────────

class SpatialHash:
    """3D spatial hash grid for fast neighbor lookups on one partition.

    Cells are hashed into buckets of size bucket_size_um.
    Neighbor queries check the 27 surrounding buckets (3x3x3 stencil).
    """

    def __init__(self, bucket_size: float = 15.0):
        self.bucket_size = bucket_size
        self._inv_bucket = 1.0 / bucket_size
        self._table: Dict[Tuple[int, int, int], List[int]] = {}
        self._positions: Optional[np.ndarray] = None
        self._n_cells: int = 0

    def build(self, positions: np.ndarray) -> None:
        """Build hash from (N, 3) positions array.

        Args:
            positions: (N, 3) float64 cell positions in physical coords.
        """
        self._positions = positions
        self._n_cells = len(positions)
        self._table.clear()

        if self._n_cells == 0:
            return

        # Hash all cells
        keys = np.floor(positions * self._inv_bucket).astype(np.int64)

        for i in range(self._n_cells):
            k = (int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2]))
            if k not in self._table:
                self._table[k] = []
            self._table[k].append(i)

    def query_neighbors(
        self,
        idx: int,
        max_dist: float,
    ) -> List[int]:
        """Find all neighbors of cell idx within max_dist.

        Uses 27-bucket stencil search.

        Returns:
            List of neighbor indices (excluding idx itself).
        """
        if self._positions is None or self._n_cells == 0:
            return []

        pos = self._positions[idx]
        key = (
            int(np.floor(pos[0] * self._inv_bucket)),
            int(np.floor(pos[1] * self._inv_bucket)),
            int(np.floor(pos[2] * self._inv_bucket)),
        )

        max_dist_sq = max_dist * max_dist
        neighbors = []

        # 27-bucket stencil
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    bk = (key[0] + dx, key[1] + dy, key[2] + dz)
                    bucket = self._table.get(bk)
                    if bucket is None:
                        continue
                    for j in bucket:
                        if j == idx:
                            continue
                        diff = self._positions[j] - pos
                        dist_sq = diff[0] ** 2 + diff[1] ** 2 + diff[2] ** 2
                        if dist_sq < max_dist_sq:
                            neighbors.append(j)

        return neighbors

    def query_neighbors_batch(
        self,
        indices: np.ndarray,
        max_dist: float,
        max_neighbors: int = 64,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Batch neighbor query for multiple cells.

        Args:
            indices: (M,) int array of cell indices to query.
            max_dist: Maximum interaction distance.
            max_neighbors: Max neighbors per cell.

        Returns:
            neighbor_ids: (M, max_neighbors) int array, -1 for empty slots.
            neighbor_counts: (M,) int array of actual neighbor counts.
        """
        M = len(indices)
        neighbor_ids = np.full((M, max_neighbors), -1, dtype=np.int32)
        neighbor_counts = np.zeros(M, dtype=np.int32)

        for qi, idx in enumerate(indices):
            nbrs = self.query_neighbors(idx, max_dist)
            n = min(len(nbrs), max_neighbors)
            neighbor_ids[qi, :n] = nbrs[:n]
            neighbor_counts[qi] = n

        return neighbor_ids, neighbor_counts


# ── Multi-GPU Cell Mechanics ──────────────────────────────────────

class MultiGPUCellMechanics:
    """Distributed cell mechanics across multiple GPU partitions.

    Each GPU:
    1. Builds a spatial hash over its cells (real + ghost)
    2. Computes pairwise JKR forces using neighbor lookup
    3. Integrates positions with overdamped Langevin dynamics
    """

    def __init__(
        self,
        multi_gpu: "MultiGPUBackend",
        decomposer: "DomainDecomposer",
        ghost_exchanger: "GhostExchanger",
        config: Optional[MechanicsConfig] = None,
    ):
        self._mgpu = multi_gpu
        self._decomp = decomposer
        self._ghost_ex = ghost_exchanger
        self._config = config or MechanicsConfig.default()

        # Per-GPU spatial hashes
        self._hashes: List[Optional[SpatialHash]] = [
            None for _ in range(multi_gpu.n_gpus)
        ]

    @property
    def config(self) -> MechanicsConfig:
        return self._config

    def build_spatial_hash(
        self,
        cell_arrays: List["TissueCellArrays"],
    ) -> None:
        """Build spatial hash on each GPU partition.

        Uses all cells (real + ghost) so boundary forces are correct.

        Args:
            cell_arrays: Per-GPU cell data (from TissueDistributor).
        """
        for gpu_id, ca in enumerate(cell_arrays):
            n_total = ca.n_total
            if n_total == 0:
                self._hashes[gpu_id] = SpatialHash(self._config.bucket_size_um)
                continue

            # Get positions on CPU for spatial hash building
            pos_np = self._mgpu.to_numpy(ca.positions[:n_total])

            sh = SpatialHash(self._config.bucket_size_um)
            sh.build(pos_np)
            self._hashes[gpu_id] = sh

        total_cells = sum(ca.n_total for ca in cell_arrays)
        total_buckets = sum(
            len(h._table) for h in self._hashes if h is not None
        )
        log.debug(
            "Spatial hashes built: %d cells in %d buckets across %d GPUs",
            total_cells, total_buckets, len(cell_arrays),
        )

    def compute_forces(
        self,
        cell_arrays: List["TissueCellArrays"],
        gradients: Optional[List[Any]] = None,
    ) -> List[Any]:
        """Compute cell-cell interaction forces on each GPU.

        Force model:
        - Hertz repulsion when overlap > 0
        - JKR adhesion when close but not overlapping
        - Optional chemotaxis from concentration gradients

        Args:
            cell_arrays: Per-GPU cell data.
            gradients: Optional per-GPU (N, 3) concentration gradients
                      for chemotaxis forces.

        Returns:
            Per-GPU (n_real, 3) float32 force arrays.
        """
        forces_list = []

        for gpu_id, ca in enumerate(cell_arrays):
            n_real = ca.n_real
            if n_real == 0:
                forces_list.append(
                    self._mgpu.allocate_on_device((0, 3), np.float32, gpu_id)
                )
                continue

            sh = self._hashes[gpu_id]
            if sh is None:
                forces_list.append(
                    self._mgpu.allocate_on_device((n_real, 3), np.float32, gpu_id)
                )
                continue

            # Get data on CPU
            pos_np = self._mgpu.to_numpy(ca.positions[:ca.n_total])
            radii_np = self._mgpu.to_numpy(ca.radii[:ca.n_total])
            alive_np = self._mgpu.to_numpy(ca.alive[:ca.n_total])

            forces = self._compute_forces_cpu(
                pos_np, radii_np, alive_np, n_real, sh,
            )

            # Add chemotaxis forces if gradients provided
            if gradients is not None and gradients[gpu_id] is not None:
                grad_np = self._mgpu.to_numpy(gradients[gpu_id])
                if len(grad_np) >= n_real:
                    # Chemotaxis: cells migrate up concentration gradient
                    migration = grad_np[:n_real] * self._config.migration_force
                    forces += migration

            # Add random fluctuations (Brownian-like)
            if self._config.random_force > 0:
                noise = np.random.randn(n_real, 3).astype(np.float32)
                forces += noise * self._config.random_force

            forces_list.append(
                self._mgpu.to_device(forces.astype(np.float32), gpu_id)
            )

        return forces_list

    def integrate(
        self,
        cell_arrays: List["TissueCellArrays"],
        forces: List[Any],
        dt: float,
    ) -> None:
        """Integrate cell positions using overdamped Langevin dynamics.

        dx/dt = F / gamma  (overdamped: inertia negligible)

        Only updates REAL cells (not ghosts).

        Args:
            cell_arrays: Per-GPU cell data (modified in-place).
            forces: Per-GPU (n_real, 3) force arrays.
            dt: Time step in hours.
        """
        gamma = self._config.damping_coeff
        max_disp = self._config.max_displacement_um

        for gpu_id, ca in enumerate(cell_arrays):
            n_real = ca.n_real
            if n_real == 0:
                continue

            # Get force and position on CPU
            f_np = self._mgpu.to_numpy(forces[gpu_id])
            pos_np = self._mgpu.to_numpy(ca.positions[:n_real])
            alive_np = self._mgpu.to_numpy(ca.alive[:n_real])

            # Overdamped: displacement = F * dt / gamma
            # Convert dt from hours to seconds for force balance
            dt_seconds = dt * 3600.0
            displacement = f_np[:n_real] * dt_seconds / gamma

            # Clamp displacement for stability
            disp_mag = np.linalg.norm(displacement, axis=1, keepdims=True)
            scale = np.minimum(1.0, max_disp / np.maximum(disp_mag, 1e-10))
            displacement *= scale

            # Only move alive cells
            alive_mask = alive_np[:, np.newaxis] if alive_np.ndim == 1 else alive_np
            if alive_mask.ndim == 1:
                alive_mask = alive_mask[:, np.newaxis]
            displacement *= alive_mask

            # Update positions
            new_pos = pos_np + displacement.astype(np.float64)

            # Write back to device
            full_pos = self._mgpu.to_numpy(ca.positions)
            full_pos[:n_real] = new_pos
            ca.positions = self._mgpu.to_device(full_pos, gpu_id)

    def compute_pressure(
        self,
        cell_arrays: List["TissueCellArrays"],
    ) -> List[Any]:
        """Compute local mechanical pressure at each cell.

        Pressure = sum of repulsive force magnitudes on each cell.
        Used for contact inhibition of proliferation.

        Returns:
            Per-GPU (n_real,) float32 pressure values.
        """
        pressure_list = []

        for gpu_id, ca in enumerate(cell_arrays):
            n_real = ca.n_real
            if n_real == 0:
                pressure_list.append(
                    self._mgpu.allocate_on_device((0,), np.float32, gpu_id)
                )
                continue

            sh = self._hashes[gpu_id]
            if sh is None:
                pressure_list.append(
                    self._mgpu.allocate_on_device((n_real,), np.float32, gpu_id)
                )
                continue

            pos_np = self._mgpu.to_numpy(ca.positions[:ca.n_total])
            radii_np = self._mgpu.to_numpy(ca.radii[:ca.n_total])

            pressure = np.zeros(n_real, dtype=np.float32)
            max_dist = self._config.bucket_size_um * 2

            for i in range(n_real):
                nbrs = sh.query_neighbors(i, max_dist)
                for j in nbrs:
                    diff = pos_np[j] - pos_np[i]
                    dist = np.sqrt(diff @ diff)
                    overlap = (radii_np[i] + radii_np[j]) - dist
                    if overlap > 0:
                        pressure[i] += overlap * self._config.repulsion_stiffness

            pressure_list.append(
                self._mgpu.to_device(pressure, gpu_id)
            )

        return pressure_list

    # ── Internal ───────────────────────────────────────────────────

    def _compute_forces_cpu(
        self,
        positions: np.ndarray,
        radii: np.ndarray,
        alive: np.ndarray,
        n_real: int,
        spatial_hash: SpatialHash,
    ) -> np.ndarray:
        """Compute pairwise forces on CPU using spatial hash.

        JKR-like model:
        - Overlap > 0: repulsive force = k_rep * overlap * normal
        - Overlap in [-adhesion_range, 0]: adhesive force = -k_adh * |overlap| * normal
        - Overlap < -adhesion_range: no force

        Args:
            positions: (N_total, 3) all cell positions (real + ghost).
            radii: (N_total,) cell radii.
            alive: (N_total,) alive flags.
            n_real: Number of real cells (forces only for these).
            spatial_hash: Built spatial hash over all cells.

        Returns:
            (n_real, 3) float32 force array.
        """
        k_rep = self._config.repulsion_stiffness
        k_adh = self._config.adhesion_strength
        max_dist = self._config.bucket_size_um * 2
        adhesion_range = 2.0  # um — max distance for adhesion beyond contact

        forces = np.zeros((n_real, 3), dtype=np.float32)

        for i in range(n_real):
            if not alive[i]:
                continue

            nbrs = spatial_hash.query_neighbors(i, max_dist)

            for j in nbrs:
                if not alive[j]:
                    continue

                diff = positions[j] - positions[i]
                dist = np.sqrt(diff @ diff)
                if dist < 1e-8:
                    continue  # Same position, skip

                normal = diff / dist
                contact_dist = radii[i] + radii[j]
                overlap = contact_dist - dist  # positive = overlapping

                if overlap > 0:
                    # Hertz repulsion
                    force_mag = k_rep * overlap
                    forces[i] -= force_mag * normal  # Push apart

                elif overlap > -adhesion_range:
                    # JKR adhesion (attractive)
                    force_mag = k_adh * (adhesion_range + overlap) / adhesion_range
                    forces[i] += force_mag * normal  # Pull together

        return forces

    def gather_force_stats(
        self,
        forces: List[Any],
    ) -> Dict:
        """Gather force statistics across all GPUs.

        Returns:
            Dict with max_force, mean_force, per_gpu stats.
        """
        max_forces = []
        mean_forces = []

        for gpu_id, f in enumerate(forces):
            f_np = self._mgpu.to_numpy(f)
            if len(f_np) == 0:
                max_forces.append(0.0)
                mean_forces.append(0.0)
                continue
            magnitudes = np.linalg.norm(f_np, axis=1)
            max_forces.append(float(magnitudes.max()))
            mean_forces.append(float(magnitudes.mean()))

        return {
            "max_force_pN": max(max_forces) if max_forces else 0.0,
            "mean_force_pN": (
                sum(mean_forces) / len(mean_forces) if mean_forces else 0.0
            ),
            "per_gpu": [
                {"gpu_id": i, "max": m, "mean": a}
                for i, (m, a) in enumerate(zip(max_forces, mean_forces))
            ],
        }
