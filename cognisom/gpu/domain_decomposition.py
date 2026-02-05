"""
Multi-GPU Domain Decomposition (Phase 3)
=========================================

Partitions the 3D simulation grid across multiple GPUs using slab
decomposition along the longest axis. Each GPU owns a contiguous
sub-volume plus ghost layers for boundary exchange.

Design goals:
    - Overlap communication with computation
    - Minimize surface-to-volume ratio (slab along longest axis)
    - Support 1-8 GPUs with automatic partitioning
    - Ghost width configurable (default 2 voxels for 7-point stencil)

Architecture::

    GPU 0: [ghost | slab_0 | ghost]
    GPU 1: [ghost | slab_1 | ghost]
    GPU 2: [ghost | slab_2 | ghost]
    ...

    After each diffusion step, ghost layers are exchanged between
    neighbors via peer-to-peer (P2P) copies or host staging.

Usage::

    from cognisom.gpu.domain_decomposition import DomainDecomposer

    decomposer = DomainDecomposer(
        grid_shape=(200, 200, 100),
        n_gpus=4,
        ghost_width=2,
    )

    # Scatter global field to per-GPU sub-volumes
    sub_fields = decomposer.scatter(global_oxygen_field)

    # Each GPU runs diffusion on its sub-volume ...

    # Exchange ghost layers between neighbors
    decomposer.exchange_ghosts(sub_fields)

    # Gather back to global field
    global_field = decomposer.gather(sub_fields)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .backend import get_backend

log = logging.getLogger(__name__)


@dataclass
class Partition:
    """A single GPU's domain partition."""
    gpu_id: int
    # Global index range along split axis (exclusive end)
    start: int
    end: int
    # Shape of local array including ghost layers
    local_shape: Tuple[int, int, int]
    # Ghost layer width
    ghost_width: int
    # Neighbor GPU IDs (-1 = boundary)
    left_neighbor: int = -1
    right_neighbor: int = -1

    @property
    def interior_slices(self) -> Tuple[slice, slice, slice]:
        """Slices to extract interior (non-ghost) data from local array."""
        gw = self.ghost_width
        return (slice(gw, -gw if gw > 0 else None), slice(None), slice(None))

    @property
    def n_interior(self) -> int:
        """Number of voxels along split axis (excluding ghosts)."""
        return self.end - self.start


class DomainDecomposer:
    """Partition 3D grid across multiple GPUs.

    Split along the longest axis to minimize ghost surface area.
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int, int] = (200, 200, 100),
        n_gpus: int = 1,
        ghost_width: int = 2,
    ):
        self._grid_shape = grid_shape
        self._n_gpus = max(1, n_gpus)
        self._ghost_width = ghost_width

        # Detect available GPUs
        backend = get_backend()
        if backend.has_gpu:
            try:
                import cupy as cp
                self._available_gpus = cp.cuda.runtime.getDeviceCount()
            except Exception:
                self._available_gpus = 0
        else:
            self._available_gpus = 0

        if self._n_gpus > 1 and self._available_gpus < self._n_gpus:
            log.warning(
                "Requested %d GPUs but only %d available. "
                "Using CPU domain decomposition (for testing/planning).",
                self._n_gpus, self._available_gpus,
            )

        # Choose split axis (longest dimension)
        self._split_axis = int(np.argmax(grid_shape))
        self._split_length = grid_shape[self._split_axis]

        # Create partitions
        self._partitions = self._create_partitions()

        log.info(
            "Domain decomposition: %s -> %d partitions along axis %d "
            "(ghost_width=%d)",
            grid_shape, len(self._partitions), self._split_axis,
            ghost_width,
        )

    @property
    def partitions(self) -> List[Partition]:
        return self._partitions

    @property
    def n_gpus(self) -> int:
        return self._n_gpus

    @property
    def split_axis(self) -> int:
        return self._split_axis

    def _create_partitions(self) -> List[Partition]:
        """Create balanced partitions along split axis."""
        n = self._n_gpus
        length = self._split_length
        gw = self._ghost_width

        # Even split with remainder distributed to first partitions
        base_size = length // n
        remainder = length % n

        partitions = []
        offset = 0

        for i in range(n):
            size = base_size + (1 if i < remainder else 0)
            start = offset
            end = offset + size
            offset = end

            # Local shape: interior + 2 * ghost_width along split axis
            local_dims = list(self._grid_shape)
            local_dims[self._split_axis] = size + 2 * gw

            partitions.append(Partition(
                gpu_id=i,
                start=start,
                end=end,
                local_shape=tuple(local_dims),
                ghost_width=gw,
                left_neighbor=i - 1 if i > 0 else -1,
                right_neighbor=i + 1 if i < n - 1 else -1,
            ))

        return partitions

    # ── Scatter / Gather ──────────────────────────────────────────

    def scatter(self, global_field: np.ndarray) -> List[np.ndarray]:
        """Distribute global field to per-partition local arrays.

        Each local array includes ghost layers filled from neighbors.
        Returns list of (local_shape) arrays, one per partition.
        """
        assert global_field.shape == self._grid_shape, (
            f"Expected shape {self._grid_shape}, got {global_field.shape}"
        )

        gw = self._ghost_width
        axis = self._split_axis
        local_arrays = []

        for part in self._partitions:
            # Determine global index range including ghosts
            ghost_start = max(0, part.start - gw)
            ghost_end = min(self._split_length, part.end + gw)

            # Extract slab from global field
            slices = [slice(None)] * 3
            slices[axis] = slice(ghost_start, ghost_end)
            slab = global_field[tuple(slices)].copy()

            # Pad if at boundary (zero-flux / Neumann)
            pad_left = (part.start - gw) - ghost_start  # how many extra needed on left
            pad_right = (part.end + gw) - ghost_end      # how many extra needed on right

            # Ghost start might be 0 when we needed gw cells before part.start
            actual_pad_left = gw - (part.start - ghost_start)
            actual_pad_right = gw - (ghost_end - part.end)

            if actual_pad_left > 0 or actual_pad_right > 0:
                pad_widths = [(0, 0)] * 3
                pad_widths[axis] = (max(0, actual_pad_left), max(0, actual_pad_right))
                slab = np.pad(slab, pad_widths, mode='edge')

            local_arrays.append(slab)

        return local_arrays

    def gather(self, local_arrays: List[np.ndarray]) -> np.ndarray:
        """Reassemble global field from per-partition local arrays.

        Only interior (non-ghost) data is used.
        """
        global_field = np.empty(self._grid_shape, dtype=local_arrays[0].dtype)
        axis = self._split_axis
        gw = self._ghost_width

        for part, local in zip(self._partitions, local_arrays):
            # Extract interior from local array
            interior_slices = [slice(None)] * 3
            end_gw = -gw if gw > 0 else None
            interior_slices[axis] = slice(gw, end_gw)
            interior = local[tuple(interior_slices)]

            # Place into global field
            global_slices = [slice(None)] * 3
            global_slices[axis] = slice(part.start, part.end)
            global_field[tuple(global_slices)] = interior

        return global_field

    def exchange_ghosts(self, local_arrays: List[np.ndarray]) -> None:
        """Exchange ghost layers between neighboring partitions.

        On multi-GPU: uses peer-to-peer copies.
        On CPU: direct NumPy array copies.
        """
        gw = self._ghost_width
        axis = self._split_axis
        if gw == 0:
            return

        for part in self._partitions:
            local = local_arrays[part.gpu_id]
            interior_size = part.n_interior

            # Send right edge -> right neighbor's left ghost
            if part.right_neighbor >= 0:
                right_local = local_arrays[part.right_neighbor]

                # Source: last gw interior slices of current partition
                src_slices = [slice(None)] * 3
                src_slices[axis] = slice(gw + interior_size - gw, gw + interior_size)
                src_data = local[tuple(src_slices)]

                # Dest: left ghost of right neighbor
                dst_slices = [slice(None)] * 3
                dst_slices[axis] = slice(0, gw)
                right_local[tuple(dst_slices)] = src_data

            # Send left edge -> left neighbor's right ghost
            if part.left_neighbor >= 0:
                left_local = local_arrays[part.left_neighbor]
                left_part = self._partitions[part.left_neighbor]
                left_interior = left_part.n_interior

                # Source: first gw interior slices
                src_slices = [slice(None)] * 3
                src_slices[axis] = slice(gw, gw + gw)
                src_data = local[tuple(src_slices)]

                # Dest: right ghost of left neighbor
                dst_slices = [slice(None)] * 3
                dst_slices[axis] = slice(gw + left_interior, gw + left_interior + gw)
                left_local[tuple(dst_slices)] = src_data

    # ── Cell partitioning ─────────────────────────────────────────

    def assign_cells(
        self, positions: np.ndarray
    ) -> List[np.ndarray]:
        """Assign cells to partitions based on position along split axis.

        Args:
            positions: (n_cells, 3) array of cell positions (physical coords)

        Returns:
            List of (n_local_cells,) index arrays — one per partition,
            containing indices into the original positions array.
        """
        axis_coords = positions[:, self._split_axis]
        assignments = [[] for _ in self._partitions]

        for part in self._partitions:
            # Map partition boundaries to physical coordinates
            # (assumes positions are in voxel indices for simplicity)
            mask = (axis_coords >= part.start) & (axis_coords < part.end)
            assignments[part.gpu_id] = np.where(mask)[0]

        return assignments

    # ── Diagnostics ───────────────────────────────────────────────

    def summary(self) -> Dict:
        """Return decomposition summary."""
        return {
            "grid_shape": self._grid_shape,
            "n_gpus": self._n_gpus,
            "available_gpus": self._available_gpus,
            "split_axis": self._split_axis,
            "ghost_width": self._ghost_width,
            "partitions": [
                {
                    "gpu_id": p.gpu_id,
                    "range": (p.start, p.end),
                    "local_shape": p.local_shape,
                    "interior_voxels": p.n_interior,
                    "left": p.left_neighbor,
                    "right": p.right_neighbor,
                }
                for p in self._partitions
            ],
        }
