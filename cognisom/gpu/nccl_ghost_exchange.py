"""Ghost layer exchange for multi-GPU domain decomposition.

Exchanges boundary data between neighboring GPU partitions for:
- 3D diffusion fields (2-voxel ghost layers)
- Cell positions/states (ghost cells near partition boundaries)

On multi-GPU: uses MultiGPUBackend peer copies (NCCL when available).
On CPU: direct numpy array copies (same logic, just slower).

Usage::

    exchanger = GhostExchanger(multi_gpu, decomposer)
    exchanger.exchange_field_ghosts(per_gpu_fields)
    ghost_cells = exchanger.find_ghost_cells(cell_arrays, ghost_width_um=30.0)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


class GhostExchanger:
    """Exchanges ghost layers between neighboring GPU partitions.

    Uses the DomainDecomposer's partition info to know which slabs
    to send/receive, and the MultiGPUBackend for actual data transfer.
    """

    def __init__(
        self,
        multi_gpu: "MultiGPUBackend",
        decomposer: "DomainDecomposer",
    ):
        self._mgpu = multi_gpu
        self._decomp = decomposer
        self._axis = decomposer.split_axis
        self._gw = decomposer._ghost_width

    def exchange_field_ghosts(self, device_fields: List[Any]) -> None:
        """Exchange ghost layers for 3D diffusion fields in-place.

        Each GPU sends its boundary interior slabs to neighbors' ghost regions.

        Args:
            device_fields: Per-GPU arrays of shape (local_nx, ny, nz).
                           Indexed by gpu_id.
        """
        gw = self._gw
        axis = self._axis
        if gw == 0:
            return

        partitions = self._decomp.partitions
        n = len(partitions)

        # Build send buffers: extract boundary slabs from interior
        send_right = [None] * n  # Right edge of GPU i -> left ghost of GPU i+1
        send_left = [None] * n   # Left edge of GPU i -> right ghost of GPU i-1

        for part in partitions:
            i = part.gpu_id
            local = device_fields[i]
            interior_size = part.n_interior

            # Right edge: last gw voxels of interior
            if part.right_neighbor >= 0:
                slices = [slice(None)] * 3
                slices[axis] = slice(gw + interior_size - gw, gw + interior_size)
                send_right[i] = self._extract_slab(local, slices)

            # Left edge: first gw voxels of interior
            if part.left_neighbor >= 0:
                slices = [slice(None)] * 3
                slices[axis] = slice(gw, gw + gw)
                send_left[i] = self._extract_slab(local, slices)

        # Exchange via MultiGPUBackend
        recv_left, recv_right = self._mgpu.neighbor_exchange(send_left, send_right)

        # Write received data into ghost regions
        for part in partitions:
            i = part.gpu_id
            local = device_fields[i]

            # Received from left neighbor -> fill left ghost
            if recv_left[i] is not None:
                slices = [slice(None)] * 3
                slices[axis] = slice(0, gw)
                self._write_slab(local, slices, recv_left[i])

            # Received from right neighbor -> fill right ghost
            if recv_right[i] is not None:
                interior_size = part.n_interior
                slices = [slice(None)] * 3
                slices[axis] = slice(gw + interior_size, gw + interior_size + gw)
                self._write_slab(local, slices, recv_right[i])

    def find_ghost_cells(
        self,
        positions_per_gpu: List[Any],
        partition_bounds: List[Tuple[float, float]],
        ghost_width_um: float = 30.0,
    ) -> List[Dict]:
        """Identify cells that should be ghost-copied to neighbor partitions.

        A cell is a ghost candidate if it's within ghost_width_um of a
        partition boundary.

        Args:
            positions_per_gpu: Per-GPU (N_cells, 3) position arrays.
            partition_bounds: Per-GPU (low, high) physical bounds along split axis.
            ghost_width_um: Physical distance for ghost region.

        Returns:
            Per-GPU dict with:
              'send_left_mask': bool mask of cells to ghost-copy left
              'send_right_mask': bool mask of cells to ghost-copy right
        """
        axis = self._axis
        n = self._mgpu.n_gpus
        results = []

        for i in range(n):
            positions = positions_per_gpu[i]
            low, high = partition_bounds[i]

            xp = np if isinstance(positions, np.ndarray) else self._get_xp()
            axis_coords = positions[:, axis]

            send_left_mask = xp.zeros(len(positions), dtype=bool)
            send_right_mask = xp.zeros(len(positions), dtype=bool)

            # Cells near left boundary
            if i > 0:
                send_left_mask = axis_coords < (low + ghost_width_um)

            # Cells near right boundary
            if i < n - 1:
                send_right_mask = axis_coords > (high - ghost_width_um)

            results.append({
                "send_left_mask": send_left_mask,
                "send_right_mask": send_right_mask,
            })

        return results

    def exchange_cell_ghosts(
        self,
        positions_per_gpu: List[Any],
        states_per_gpu: List[Any],
        partition_bounds: List[Tuple[float, float]],
        ghost_width_um: float = 30.0,
    ) -> Tuple[List[Any], List[Any]]:
        """Exchange ghost cells between neighboring partitions.

        Returns additional ghost positions and states to append to each GPU.

        Args:
            positions_per_gpu: Per-GPU (N, 3) float64 arrays.
            states_per_gpu: Per-GPU (N, n_state) float32 arrays.
            partition_bounds: Per-GPU (low, high) bounds.
            ghost_width_um: Ghost region width.

        Returns:
            (ghost_positions, ghost_states) — per-GPU arrays of ghost cells
            received from neighbors.
        """
        masks = self.find_ghost_cells(
            positions_per_gpu, partition_bounds, ghost_width_um,
        )
        n = self._mgpu.n_gpus

        # Gather send buffers
        send_left_pos = [None] * n
        send_left_state = [None] * n
        send_right_pos = [None] * n
        send_right_state = [None] * n

        for i in range(n):
            m = masks[i]
            xp = np if isinstance(positions_per_gpu[i], np.ndarray) else self._get_xp()

            if m["send_left_mask"].any():
                idx = xp.where(m["send_left_mask"])[0]
                send_left_pos[i] = positions_per_gpu[i][idx]
                send_left_state[i] = states_per_gpu[i][idx]

            if m["send_right_mask"].any():
                idx = xp.where(m["send_right_mask"])[0]
                send_right_pos[i] = positions_per_gpu[i][idx]
                send_right_state[i] = states_per_gpu[i][idx]

        # Exchange positions
        recv_left_pos, recv_right_pos = self._mgpu.neighbor_exchange(
            send_left_pos, send_right_pos,
        )
        recv_left_state, recv_right_state = self._mgpu.neighbor_exchange(
            send_left_state, send_right_state,
        )

        # Combine received ghosts per GPU
        ghost_positions = []
        ghost_states = []

        for i in range(n):
            parts_pos = []
            parts_state = []

            if recv_left_pos[i] is not None:
                parts_pos.append(recv_left_pos[i])
                parts_state.append(recv_left_state[i])
            if recv_right_pos[i] is not None:
                parts_pos.append(recv_right_pos[i])
                parts_state.append(recv_right_state[i])

            xp = np if not parts_pos else (
                np if isinstance(parts_pos[0], np.ndarray) else self._get_xp()
            )

            if parts_pos:
                ghost_positions.append(xp.concatenate(parts_pos, axis=0))
                ghost_states.append(xp.concatenate(parts_state, axis=0))
            else:
                ghost_positions.append(xp.empty((0, 3), dtype=np.float64))
                ghost_states.append(xp.empty((0, states_per_gpu[0].shape[1]),
                                              dtype=np.float32))

        return ghost_positions, ghost_states

    # ── Internal helpers ───────────────────────────────────────

    def _extract_slab(self, arr: Any, slices: list) -> Any:
        """Extract a slab from an array (handles both numpy and cupy)."""
        slab = arr[tuple(slices)]
        if hasattr(slab, 'copy'):
            return slab.copy()
        return np.array(slab, copy=True)

    def _write_slab(self, arr: Any, slices: list, data: Any) -> None:
        """Write data into a slab of an array."""
        arr[tuple(slices)] = data

    def _get_xp(self):
        """Get the array module (cupy or numpy)."""
        if self._mgpu.mode in ("multi_gpu", "single_gpu"):
            try:
                import cupy as cp
                return cp
            except ImportError:
                pass
        return np
