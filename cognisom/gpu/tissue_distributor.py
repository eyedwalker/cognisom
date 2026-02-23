"""Tissue-scale cell distribution and management across GPU partitions.

Manages the lifecycle of 1M+ cells distributed across 8 GPUs:
- Initial spatial partitioning based on cell position
- Structure-of-Arrays (SoA) cell state on each GPU
- Ghost cell management for boundary force calculations
- Dynamic re-balancing when cells migrate between partitions

Usage::

    distributor = TissueDistributor(multi_gpu, decomposer)
    cell_arrays = distributor.initialize_cells(positions, types, radii)
    # ... simulation step ...
    distributor.update_ghosts(cell_arrays)
    # ... periodically ...
    cell_arrays = distributor.rebalance(cell_arrays)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


# Number of float32 state values per cell (metabolic + cycle)
N_CELL_STATE = 12  # oxygen, glucose, atp, lactate, age, phase, mhc1,
                    # division_timer, damage, apoptotic_signal, migration_speed, reserved


@dataclass
class TissueCellArrays:
    """Structure-of-Arrays cell state for one GPU partition.

    All arrays are on the same device (GPU or numpy for CPU mode).
    Ghost cells are appended after real cells: real cells = [0, n_real),
    ghost cells = [n_real, n_real + n_ghost).
    """
    device_id: int
    n_real: int = 0
    n_ghost: int = 0

    # Core arrays (all on same device)
    positions: Any = None       # (N, 3) float64
    velocities: Any = None      # (N, 3) float32
    radii: Any = None           # (N,) float32
    cell_types: Any = None      # (N,) int32 — archetype index
    alive: Any = None           # (N,) bool
    state: Any = None           # (N, N_CELL_STATE) float32

    # Intracellular ODE state (optional)
    ode_state: Any = None       # (N, n_ode_species) float32

    @property
    def n_total(self) -> int:
        return self.n_real + self.n_ghost

    @property
    def real_slice(self) -> slice:
        return slice(0, self.n_real)

    @property
    def ghost_slice(self) -> slice:
        return slice(self.n_real, self.n_real + self.n_ghost)


class TissueDistributor:
    """Distributes and manages cells across GPU partitions."""

    def __init__(
        self,
        multi_gpu: "MultiGPUBackend",
        decomposer: "DomainDecomposer",
        ghost_exchanger: "GhostExchanger",
        ghost_width_um: float = 30.0,
        resolution_um: float = 10.0,
    ):
        self._mgpu = multi_gpu
        self._decomp = decomposer
        self._ghost_ex = ghost_exchanger
        self._ghost_width_um = ghost_width_um
        self._resolution_um = resolution_um
        self._partition_bounds: List[Tuple[float, float]] = []

        # Compute physical bounds per partition
        for part in decomposer.partitions:
            low = part.start * resolution_um
            high = part.end * resolution_um
            self._partition_bounds.append((low, high))

    @property
    def partition_bounds(self) -> List[Tuple[float, float]]:
        return self._partition_bounds

    def initialize_cells(
        self,
        positions: np.ndarray,
        cell_types: np.ndarray,
        radii: Optional[np.ndarray] = None,
        n_ode_species: int = 6,
    ) -> List[TissueCellArrays]:
        """Scatter cells to GPU partitions based on spatial position.

        Args:
            positions: (N, 3) float64 — cell positions in physical coords (um).
            cell_types: (N,) int32 — archetype index per cell.
            radii: (N,) float32 — cell radii. Default 5.0 um.
            n_ode_species: Number of ODE state variables per cell.

        Returns:
            List of TissueCellArrays, one per GPU partition.
        """
        n_cells = len(positions)
        if radii is None:
            radii = np.full(n_cells, 5.0, dtype=np.float32)

        # Convert positions to voxel coords for partitioning
        axis = self._decomp.split_axis
        voxel_coords = positions / self._resolution_um
        assignments = self._decomp.assign_cells(voxel_coords)

        cell_arrays = []

        for gpu_id, indices in enumerate(assignments):
            n_local = len(indices)

            # Extract this partition's cells
            local_pos = positions[indices].astype(np.float64)
            local_types = cell_types[indices].astype(np.int32)
            local_radii = radii[indices].astype(np.float32)

            # Initialize state arrays
            local_vel = np.zeros((n_local, 3), dtype=np.float32)
            local_alive = np.ones(n_local, dtype=bool)
            local_state = np.zeros((n_local, N_CELL_STATE), dtype=np.float32)

            # Initialize metabolic state: oxygen=0.21, glucose=5.0, atp=1.0
            local_state[:, 0] = 0.21   # oxygen
            local_state[:, 1] = 5.0    # glucose
            local_state[:, 2] = 1.0    # atp

            # ODE state
            local_ode = np.zeros((n_local, n_ode_species), dtype=np.float32)

            # Transfer to device
            ca = TissueCellArrays(
                device_id=gpu_id,
                n_real=n_local,
                n_ghost=0,
                positions=self._mgpu.to_device(local_pos, gpu_id),
                velocities=self._mgpu.to_device(local_vel, gpu_id),
                radii=self._mgpu.to_device(local_radii, gpu_id),
                cell_types=self._mgpu.to_device(local_types, gpu_id),
                alive=self._mgpu.to_device(local_alive, gpu_id),
                state=self._mgpu.to_device(local_state, gpu_id),
                ode_state=self._mgpu.to_device(local_ode, gpu_id),
            )
            cell_arrays.append(ca)

            log.debug(
                "GPU %d: %d cells in [%.0f, %.0f] um",
                gpu_id, n_local,
                self._partition_bounds[gpu_id][0],
                self._partition_bounds[gpu_id][1],
            )

        total = sum(ca.n_real for ca in cell_arrays)
        log.info(
            "Distributed %d cells across %d partitions (%.1f%% balance)",
            total, len(cell_arrays),
            100 * min(ca.n_real for ca in cell_arrays) /
            max(1, max(ca.n_real for ca in cell_arrays)),
        )

        return cell_arrays

    def update_ghosts(self, cell_arrays: List[TissueCellArrays]) -> None:
        """Update ghost cells from neighbor partitions.

        Strips old ghosts, finds new ghost candidates, and exchanges.
        Called once per main simulation step.
        """
        # Strip existing ghosts (keep only real cells)
        for ca in cell_arrays:
            if ca.n_ghost > 0:
                s = ca.real_slice
                ca.positions = ca.positions[s]
                ca.velocities = ca.velocities[s]
                ca.radii = ca.radii[s]
                ca.cell_types = ca.cell_types[s]
                ca.alive = ca.alive[s]
                ca.state = ca.state[s]
                if ca.ode_state is not None:
                    ca.ode_state = ca.ode_state[s]
                ca.n_ghost = 0

        # Get real positions and state for ghost exchange
        real_positions = [ca.positions for ca in cell_arrays]
        real_states = [ca.state for ca in cell_arrays]

        # Exchange ghost cells
        ghost_pos, ghost_states = self._ghost_ex.exchange_cell_ghosts(
            real_positions, real_states,
            self._partition_bounds, self._ghost_width_um,
        )

        # Append ghost cells to each partition
        xp = np  # Will be cupy on GPU
        for i, ca in enumerate(cell_arrays):
            n_ghost = len(ghost_pos[i]) if ghost_pos[i] is not None else 0
            if n_ghost == 0:
                continue

            if hasattr(ca.positions, '__cuda_array_interface__'):
                try:
                    import cupy as cp
                    xp = cp
                except ImportError:
                    xp = np

            ca.positions = xp.concatenate([ca.positions, ghost_pos[i]], axis=0)
            ca.state = xp.concatenate([ca.state, ghost_states[i]], axis=0)

            # Pad other arrays for ghost cells
            ghost_vel = xp.zeros((n_ghost, 3), dtype=np.float32)
            ghost_radii = xp.full(n_ghost, 5.0, dtype=np.float32)
            ghost_types = xp.zeros(n_ghost, dtype=np.int32)
            ghost_alive = xp.ones(n_ghost, dtype=bool)

            ca.velocities = xp.concatenate([ca.velocities, ghost_vel], axis=0)
            ca.radii = xp.concatenate([ca.radii, ghost_radii], axis=0)
            ca.cell_types = xp.concatenate([ca.cell_types, ghost_types], axis=0)
            ca.alive = xp.concatenate([ca.alive, ghost_alive], axis=0)

            if ca.ode_state is not None:
                ghost_ode = xp.zeros(
                    (n_ghost, ca.ode_state.shape[1]), dtype=np.float32,
                )
                ca.ode_state = xp.concatenate([ca.ode_state, ghost_ode], axis=0)

            ca.n_ghost = n_ghost

    def rebalance(
        self, cell_arrays: List[TissueCellArrays],
    ) -> List[TissueCellArrays]:
        """Re-partition cells that migrated across partition boundaries.

        Checks each cell's position against its partition bounds.
        Cells outside bounds are moved to the correct partition.

        Returns updated cell_arrays (may have different sizes).
        """
        axis = self._decomp.split_axis
        n = self._mgpu.n_gpus

        # Count misplaced cells
        total_misplaced = 0
        for i, ca in enumerate(cell_arrays):
            low, high = self._partition_bounds[i]
            pos = self._mgpu.to_numpy(ca.positions[:ca.n_real])
            axis_coords = pos[:, axis]
            misplaced = np.sum((axis_coords < low) | (axis_coords >= high))
            total_misplaced += misplaced

        total_cells = sum(ca.n_real for ca in cell_arrays)
        if total_cells == 0 or total_misplaced / total_cells < 0.01:
            return cell_arrays  # Less than 1% misplaced, skip

        log.info(
            "Rebalancing: %d/%d cells (%.1f%%) misplaced",
            total_misplaced, total_cells,
            100 * total_misplaced / total_cells,
        )

        # Gather all real cells to CPU
        all_positions = []
        all_types = []
        all_radii = []
        all_states = []
        all_ode_states = []

        for ca in cell_arrays:
            s = ca.real_slice
            all_positions.append(self._mgpu.to_numpy(ca.positions[s]))
            all_types.append(self._mgpu.to_numpy(ca.cell_types[s]))
            all_radii.append(self._mgpu.to_numpy(ca.radii[s]))
            all_states.append(self._mgpu.to_numpy(ca.state[s]))
            if ca.ode_state is not None:
                all_ode_states.append(self._mgpu.to_numpy(ca.ode_state[s]))

        positions = np.concatenate(all_positions, axis=0)
        types = np.concatenate(all_types, axis=0)
        radii = np.concatenate(all_radii, axis=0)
        states = np.concatenate(all_states, axis=0)
        ode_states = np.concatenate(all_ode_states, axis=0) if all_ode_states else None

        # Reassign
        voxel_coords = positions / self._resolution_um
        assignments = self._decomp.assign_cells(voxel_coords)

        new_arrays = []
        for gpu_id, indices in enumerate(assignments):
            n_local = len(indices)
            n_ode = ode_states.shape[1] if ode_states is not None else 6

            ca = TissueCellArrays(
                device_id=gpu_id,
                n_real=n_local,
                n_ghost=0,
                positions=self._mgpu.to_device(positions[indices].astype(np.float64), gpu_id),
                velocities=self._mgpu.allocate_on_device((n_local, 3), np.float32, gpu_id),
                radii=self._mgpu.to_device(radii[indices].astype(np.float32), gpu_id),
                cell_types=self._mgpu.to_device(types[indices].astype(np.int32), gpu_id),
                alive=self._mgpu.to_device(np.ones(n_local, dtype=bool), gpu_id),
                state=self._mgpu.to_device(states[indices].astype(np.float32), gpu_id),
                ode_state=self._mgpu.to_device(
                    ode_states[indices].astype(np.float32) if ode_states is not None
                    else np.zeros((n_local, n_ode), dtype=np.float32),
                    gpu_id,
                ),
            )
            new_arrays.append(ca)

        return new_arrays

    def gather_positions(
        self, cell_arrays: List[TissueCellArrays],
        max_cells: int = 0,
    ) -> np.ndarray:
        """Gather all cell positions to CPU (for visualization).

        Args:
            cell_arrays: Per-GPU cell data.
            max_cells: If > 0, subsample to this many cells.

        Returns:
            (total_real_cells, 3) float64 array.
        """
        parts = []
        for ca in cell_arrays:
            pos = self._mgpu.to_numpy(ca.positions[:ca.n_real])
            parts.append(pos)

        all_pos = np.concatenate(parts, axis=0) if parts else np.empty((0, 3))

        if max_cells > 0 and len(all_pos) > max_cells:
            indices = np.random.choice(len(all_pos), max_cells, replace=False)
            indices.sort()
            return all_pos[indices]

        return all_pos

    def gather_types(
        self, cell_arrays: List[TissueCellArrays],
    ) -> np.ndarray:
        """Gather all cell types to CPU."""
        parts = []
        for ca in cell_arrays:
            types = self._mgpu.to_numpy(ca.cell_types[:ca.n_real])
            parts.append(types)
        return np.concatenate(parts, axis=0) if parts else np.empty(0, dtype=np.int32)

    def gather_summary(
        self, cell_arrays: List[TissueCellArrays],
    ) -> Dict:
        """Gather summary statistics without full data transfer."""
        total_cells = 0
        per_gpu = []
        alive_count = 0

        for ca in cell_arrays:
            n = ca.n_real
            total_cells += n
            alive = int(self._mgpu.to_numpy(ca.alive[:n]).sum()) if n > 0 else 0
            alive_count += alive

            per_gpu.append({
                "gpu_id": ca.device_id,
                "n_real": n,
                "n_ghost": ca.n_ghost,
                "n_alive": alive,
            })

        return {
            "total_cells": total_cells,
            "alive_cells": alive_count,
            "per_gpu": per_gpu,
            "balance_pct": (
                100 * min(g["n_real"] for g in per_gpu) /
                max(1, max(g["n_real"] for g in per_gpu))
                if per_gpu else 100
            ),
        }
