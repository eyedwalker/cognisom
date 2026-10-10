"""Multi-GPU distributed diffusion solver.

Wraps the existing diffusion.py Laplacian kernel to operate on
domain-decomposed sub-volumes across multiple GPUs, with ghost
layer exchange between partitions.

Supports both real multi-GPU (CuPy/NCCL) and CPU simulation mode.

Usage::

    solver = MultiGPUDiffusionSolver(multi_gpu, decomposer, ghost_exchanger)
    solver.initialize_fields({"oxygen": oxygen_field}, {"oxygen": 2000.0})
    solver.step(dt=0.01, resolution=10.0)
    values = solver.sample_at_positions(cell_positions, "oxygen", 10.0)

Performance
-----------
The explicit-Euler substep is the hot loop of a tissue-scale step: the 1M-cell
preset is a 500^3 grid over 8 partitions with three diffusible fields and
``diffusion_substeps = 5``, which is ~1.9e9 voxel updates per engine step.
Two things dominate at that size, and neither of them is arithmetic:

* **Memory traffic.** Writing the stencil as one expression over six shifted
  3-D views allocates a zeroed Laplacian plus four whole-field temporaries per
  substep, and every pass runs over a trailing-axis slice (``[..., 1:-1]``),
  which costs about twice as much per element as a contiguous pass because the
  innermost run is shorter than the row. ``_diffuse_local`` instead walks the
  sub-volume as contiguous 1-D views: the array is C-contiguous, so the
  neighbour in z is a flat offset of 1, in y of ``nz``, and in x of ``ny*nz``.
  The six neighbour terms accumulate into one preallocated buffer and the
  result is written back over the field in place, so a substep allocates
  nothing.
* **Single-core ufunc throughput.** NumPy ufuncs are single threaded, so one
  partition can never use more than one core. Partitions are independent once
  ghost layers have been exchanged, so the per-partition updates of every
  field are handed to a small thread pool; NumPy releases the GIL inside the
  ufunc loops, so they genuinely overlap. Threading is CPU-mode only: on GPU
  the kernels are already parallel and each partition's work is queued on its
  own device. ``COGNISOM_DIFFUSION_THREADS`` overrides the worker count (1
  disables).

The arithmetic is unchanged operation for operation, so the result is
bit-identical to the original expression — see
``tests/test_diffusion_characterization.py``, which checks the rewrite against
a frozen copy of it. The cost is scratch memory: one accumulator the size of a
sub-volume interior per worker thread (per device on multi-GPU), plus a small
chunk buffer.

Units
-----
The engine calls :meth:`step` with ``dt`` in **hours** (``cfg.dt`` is hours)
while the field diffusion coefficients are in **um^2/s**, so the Euler update
advances each field by ``dt`` *seconds* of diffusion rather than ``dt`` hours.
Oxygen (D = 2000 um^2/s) therefore spreads ~60x less far per step than it
should. That is pre-existing behaviour and is deliberately left alone here —
this module was changed for speed only — but it is the first thing to fix
before any diffusion result is read as biology. Note also that correcting it
makes ``D*dt/dx^2`` = 144 at the shipped configuration, far outside the 1/6
stability limit of explicit Euler in 3-D, so the fix has to come with either
a smaller ``dt`` or an implicit scheme.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

#: Sampling used to hardcode this instead of reading ``cfg.resolution_um``.
#: It is kept as the default so behaviour at the shipped 10 um resolution is
#: unchanged for callers that do not pass a resolution.
_DEFAULT_RESOLUTION_UM = 10.0

#: Voxels per chunk in the local (non-stencil) half of the substep. Big
#: enough that each NumPy call is long compared with GIL hand-off, small
#: enough that the chunk and its scratch stay in cache.
_PHASE2_CHUNK_VOXELS = 2 << 20

#: Below this sub-volume size, handing partitions to threads costs more than
#: it saves. Measured on the 10k preset (170k-voxel sub-volumes, which still
#: gain ~1.2x) down to the test grids, which do not.
_THREAD_MIN_VOXELS = 1 << 16


def _xp_for(arr: Any):
    """Return the array module that owns ``arr`` (numpy or cupy)."""
    if arr is not None and hasattr(arr, "__cuda_array_interface__"):
        try:
            import cupy
            return cupy
        except ImportError:  # pragma: no cover - device array without cupy
            pass
    return np


def _default_thread_count(multi_gpu: Any) -> int:
    """Workers for the per-partition update loop.

    GPU modes stay serial: the device kernels are already parallel and the
    Python loop only queues them.
    """
    env = os.environ.get("COGNISOM_DIFFUSION_THREADS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            log.warning("Ignoring non-integer COGNISOM_DIFFUSION_THREADS=%r", env)

    if getattr(multi_gpu, "mode", "cpu") != "cpu":
        return 1

    n_partitions = max(1, int(getattr(multi_gpu, "n_gpus", 1) or 1))
    return max(1, min(n_partitions, os.cpu_count() or 1, 8))


class MultiGPUDiffusionSolver:
    """Distributed 3D diffusion across multiple GPU partitions."""

    def __init__(
        self,
        multi_gpu: "MultiGPUBackend",
        decomposer: "DomainDecomposer",
        ghost_exchanger: "GhostExchanger",
    ):
        self._mgpu = multi_gpu
        self._decomp = decomposer
        self._ghost_ex = ghost_exchanger

        # Per-field data: name -> list of per-GPU arrays
        self._fields: Dict[str, List[Any]] = {}
        self._diff_coeffs: Dict[str, float] = {}

        # Reused stencil scratch, keyed by (device, thread).
        self._scratch_pool: Dict[Tuple[int, int], Tuple[Any, Any]] = {}
        self._n_threads = _default_thread_count(multi_gpu)
        self._thread_min_voxels = _THREAD_MIN_VOXELS
        self._pool: Optional[ThreadPoolExecutor] = None
        self._warned_sources = False

    @property
    def field_names(self) -> List[str]:
        return list(self._fields.keys())

    def initialize_fields(
        self,
        global_fields: Dict[str, np.ndarray],
        diffusion_coeffs: Dict[str, float],
    ) -> None:
        """Scatter global 3D fields to per-GPU sub-volumes.

        Args:
            global_fields: name -> (nx, ny, nz) float32 arrays.
            diffusion_coeffs: name -> diffusion coefficient (um^2/s).
        """
        for name, field_data in global_fields.items():
            # Use DomainDecomposer to scatter
            local_arrays_np = self._decomp.scatter(
                field_data.astype(np.float32)
            )

            # Transfer to GPU devices
            device_arrays = []
            for gpu_id, arr in enumerate(local_arrays_np):
                device_arrays.append(
                    self._mgpu.to_device(arr, gpu_id)
                )

            self._fields[name] = device_arrays
            self._diff_coeffs[name] = diffusion_coeffs.get(name, 0.0)

            log.debug(
                "Initialized field '%s': %d partitions, D=%.1f um^2/s",
                name, len(device_arrays), self._diff_coeffs[name],
            )

    def step(
        self,
        dt: float,
        resolution: float,
        sources: Optional[Dict[str, List]] = None,
        sinks: Optional[Dict[str, List]] = None,
    ) -> None:
        """Advance all diffusion fields by dt on all GPUs.

        Steps:
        1. Exchange ghost layers between neighbors
        2. Compute Laplacian on each GPU
        3. Apply Euler update: C += D * lap * dt / dx^2
        4. Non-negative clamp

        ``sources`` and ``sinks`` are **not implemented**, and never were:
        nothing in the engine injects or consumes field mass, so the fields
        stay at their initial uniform values and the Laplacian is identically
        zero at every step. No caller passes them; they are kept so the
        signature does not change, and a caller that does pass them now gets
        a warning instead of silently losing the term.
        """
        if (sources or sinks) and not self._warned_sources:
            self._warned_sources = True
            log.warning(
                "MultiGPUDiffusionSolver.step() ignores sources/sinks; "
                "field mass is neither injected nor consumed.",
            )

        # Ghost exchange first, for every field: fields are independent, so
        # hoisting the exchanges lets all the partition updates run together.
        tasks: List[Tuple[str, int, float]] = []
        for name, device_fields in self._fields.items():
            D = self._diff_coeffs[name]
            if D <= 0:
                continue
            self._ghost_ex.exchange_field_ghosts(device_fields)
            tasks.extend(
                (name, gpu_id, D) for gpu_id in range(len(device_fields))
            )

        if not tasks:
            return

        if self._should_thread(tasks):
            pool = self._thread_pool()
            futures = [
                pool.submit(self._update_partition, name, gpu_id, D, dt, resolution)
                for name, gpu_id, D in tasks
            ]
            for future in futures:
                future.result()   # re-raise worker exceptions here
        else:
            for name, gpu_id, D in tasks:
                self._update_partition(name, gpu_id, D, dt, resolution)

    def sample_at_positions(
        self,
        cell_positions_per_gpu: List[Any],
        field_name: str,
        resolution: float = _DEFAULT_RESOLUTION_UM,
    ) -> List[Any]:
        """Sample concentration at cell positions (nearest voxel).

        Indexes the voxels the cells are actually in. It does not move the
        sub-volume anywhere: on GPU that was a device-to-host copy of the
        whole slab per field per step, to read a few thousand voxels.

        Args:
            cell_positions_per_gpu: Per-GPU (N, 3) position arrays in physical coords.
            field_name: Which field to sample.
            resolution: Microns per voxel. Defaults to 10 um, which is what
                this method used to assume unconditionally; pass
                ``cfg.resolution_um`` so a non-default grid samples the right
                voxels.

        Returns:
            Per-GPU (N,) float32 arrays of sampled values.
        """
        if field_name not in self._fields:
            raise KeyError(f"Unknown field: {field_name}")

        device_fields = self._fields[field_name]
        results = []

        for gpu_id, (positions, local_field) in enumerate(
            zip(cell_positions_per_gpu, device_fields)
        ):
            n_cells = len(positions)
            if n_cells == 0:
                results.append(self._mgpu.allocate_on_device((0,), np.float32, gpu_id))
                continue

            with self._device_scope(gpu_id):
                idx = self._voxel_indices(
                    positions, local_field, gpu_id, resolution, margin=0,
                )
                values = local_field[idx[0], idx[1], idx[2]]
                results.append(values.astype(np.float32))

        return results

    def compute_gradient_at_positions(
        self,
        cell_positions_per_gpu: List[Any],
        field_name: str,
        resolution: float = _DEFAULT_RESOLUTION_UM,
    ) -> List[Any]:
        """Compute concentration gradient at cell positions.

        Uses central differences for interior cells. Like
        :meth:`sample_at_positions`, this gathers the few voxels each cell
        needs instead of copying the sub-volume to the host.

        Returns:
            Per-GPU (N, 3) float32 gradient vectors.
        """
        if field_name not in self._fields:
            raise KeyError(f"Unknown field: {field_name}")

        device_fields = self._fields[field_name]
        results = []

        for gpu_id, (positions, local_field) in enumerate(
            zip(cell_positions_per_gpu, device_fields)
        ):
            n_cells = len(positions)
            if n_cells == 0:
                results.append(self._mgpu.allocate_on_device((0, 3), np.float32, gpu_id))
                continue

            with self._device_scope(gpu_id):
                xp = _xp_for(local_field)
                shape = local_field.shape
                # Central differences need a voxel on each side, so the base
                # index is clamped one voxel inside the sub-volume.
                idx = self._voxel_indices(
                    positions, local_field, gpu_id, resolution, margin=1,
                )

                grad = xp.zeros((n_cells, 3), dtype=xp.float32)
                for d in range(3):
                    plus = list(idx)
                    minus = list(idx)
                    plus[d] = xp.clip(idx[d] + 1, 0, shape[d] - 1)
                    minus[d] = xp.clip(idx[d] - 1, 0, shape[d] - 1)

                    c_plus = local_field[plus[0], plus[1], plus[2]]
                    c_minus = local_field[minus[0], minus[1], minus[2]]
                    grad[:, d] = (c_plus - c_minus) / (2 * resolution)

                results.append(grad)

        return results

    def gather_field(self, field_name: str) -> np.ndarray:
        """Gather a full field to CPU for visualization."""
        if field_name not in self._fields:
            raise KeyError(f"Unknown field: {field_name}")

        local_np = [
            self._mgpu.to_numpy(f) for f in self._fields[field_name]
        ]
        return self._decomp.gather(local_np)

    def gather_field_slice(
        self, field_name: str, axis: int, index: int,
    ) -> np.ndarray:
        """Gather a single 2D slice of a field.

        Only the voxels in the slice are moved: along the split axis that is
        one partition's plane, otherwise one plane per partition. Gathering
        the whole field first cost ~500 MB of transfer per snapshot at the
        1M-cell preset to produce a 1 MB image.

        Args:
            field_name: Field to slice.
            axis: Axis to slice along (0, 1, or 2).
            index: Global index along that axis (negatives count from the end).

        Returns:
            2D float32 array.
        """
        if field_name not in self._fields:
            raise KeyError(f"Unknown field: {field_name}")
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0, 1 or 2, got {axis}")

        device_fields = self._fields[field_name]
        partitions = self._decomp.partitions
        gw = self._decomp._ghost_width
        split = self._decomp.split_axis

        extent = self._decomp._grid_shape[axis]
        if index < 0:
            index += extent
        if not 0 <= index < extent:
            raise IndexError(
                f"index {index} out of range for axis {axis} of length {extent}"
            )

        if axis == split:
            for part, local in zip(partitions, device_fields):
                if part.start <= index < part.end:
                    slices = [slice(None)] * 3
                    slices[axis] = gw + (index - part.start)
                    # Copy: the caller gets a snapshot, not a live view.
                    return np.array(self._mgpu.to_numpy(local[tuple(slices)]))
            raise IndexError(f"no partition owns index {index}")  # pragma: no cover

        pieces = []
        for part, local in zip(partitions, device_fields):
            slices = [slice(None)] * 3
            slices[split] = slice(gw, gw + part.n_interior)
            slices[axis] = index
            pieces.append(self._mgpu.to_numpy(local[tuple(slices)]))

        # Dropping `axis` shifts the split axis down by one if it came after.
        return np.concatenate(pieces, axis=split - (1 if axis < split else 0))

    # ── Internal ───────────────────────────────────────────────

    def _device_scope(self, gpu_id: int):
        """Make partition ``gpu_id``'s device current for ``xp`` work.

        CPU mode is a no-op. On GPU the kernels below operate on the
        partition's own arrays, so the matching device has to be current.
        """
        mode = getattr(self._mgpu, "mode", "cpu")
        if mode == "multi_gpu":
            import cupy as cp
            return cp.cuda.Device(gpu_id)
        if mode == "single_gpu":
            import cupy as cp
            return cp.cuda.Device(0)
        return contextlib.nullcontext()

    def _should_thread(self, tasks: List[Tuple[str, int, float]]) -> bool:
        if self._n_threads <= 1 or len(tasks) < 2:
            return False
        name, gpu_id, _ = tasks[0]
        return self._fields[name][gpu_id].size >= self._thread_min_voxels

    def _thread_pool(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(
                max_workers=self._n_threads,
                thread_name_prefix="diffusion",
            )
            log.debug("Diffusion thread pool: %d workers", self._n_threads)
        return self._pool

    def _update_partition(
        self, name: str, gpu_id: int, D: float, dt: float, resolution: float,
    ) -> None:
        device_fields = self._fields[name]
        with self._device_scope(gpu_id):
            device_fields[gpu_id] = self._diffuse_local(
                device_fields[gpu_id], D, dt, resolution, slot=gpu_id,
            )

    def _voxel_indices(
        self,
        positions: Any,
        local_field: Any,
        gpu_id: int,
        resolution: float,
        margin: int,
    ) -> Tuple[Any, Any, Any]:
        """Per-axis voxel indices for cell positions, clamped into the slab.

        ``margin`` is how far inside the sub-volume the indices are kept: 0
        for sampling, 1 for central differences.
        """
        partition = self._decomp.partitions[gpu_id]
        gw = self._decomp._ghost_width
        axis = self._decomp.split_axis
        xp = _xp_for(local_field)

        if _xp_for(positions) is not xp:
            positions = xp.asarray(self._mgpu.to_numpy(positions))

        voxel = positions / resolution
        # Physical coords are global; the slab's interior starts at ghost_width.
        voxel[:, axis] -= partition.start
        voxel[:, axis] += gw

        idx = voxel.astype(xp.int64)
        shape = local_field.shape
        # Same bounds the original clipped to, including the degenerate case
        # where the margin leaves no room (clip then returns the upper bound).
        return tuple(
            xp.clip(idx[:, d], margin, shape[d] - 1 - margin)
            for d in range(3)
        )

    def _scratch(
        self, xp: Any, n_voxels: int, dtype: Any, slot: int,
    ) -> Tuple[Any, Any]:
        """Stencil buffers for this worker: (accumulator, chunk buffer).

        Keyed by thread, and by device as well when partitions really are on
        separate devices, so a buffer is never shared across either.
        """
        mode = getattr(self._mgpu, "mode", "cpu")
        key = (slot if mode == "multi_gpu" else 0, threading.get_ident())
        entry = self._scratch_pool.get(key)
        if (
            entry is None
            or entry[0].size < n_voxels
            or entry[0].dtype != dtype
        ):
            entry = (
                xp.empty(n_voxels, dtype=dtype),
                xp.empty(min(n_voxels, _PHASE2_CHUNK_VOXELS), dtype=dtype),
            )
            self._scratch_pool[key] = entry
        return entry

    def _diffuse_local(
        self, field: Any, D: float, dt: float, resolution: float,
        slot: int = 0,
    ) -> Any:
        """Run one diffusion substep on a single GPU's sub-volume, in place.

        The returned array is the one that was passed in. The arithmetic is
        the original 7-point stencil, term for term and in the same order, so
        the result is bit-identical; see the module docstring for why it is
        written this way.
        """
        xp = _xp_for(field)
        nx, ny, nz = field.shape

        if min(nx, ny, nz) < 3:
            # The stencil's interior is empty along at least one axis, so the
            # Laplacian is zero everywhere and only the clamp applies.
            return xp.maximum(field, 0.0, out=field)

        if not getattr(field.flags, "c_contiguous", False):
            return self._diffuse_local_strided(field, D, dt, resolution)

        N = ny * nz                 # flat stride of one x-plane
        M = (nx - 2) * N            # the x-interior, as a flat run
        flat = field.reshape(-1)    # a view: the sub-volume is contiguous
        acc, chunk_buf = self._scratch(xp, M, field.dtype, slot)

        # The flat run spans whole x-planes, so it also covers the y and z
        # faces, where a flat neighbour offset walks into the adjacent row
        # instead of the adjacent voxel. The original left those faces
        # untouched apart from the clamp, so save them now and put them back
        # afterwards; that keeps the write-back one contiguous block.
        face_y0 = field[1:-1, 0, :].copy()
        face_y1 = field[1:-1, -1, :].copy()
        face_z0 = field[1:-1, :, 0].copy()
        face_z1 = field[1:-1, :, -1].copy()

        # The six neighbour terms, in the original's summation order.
        lap = acc[:M]
        xp.add(flat[2 * N:2 * N + M], flat[:M], out=lap)    # x+1, x-1
        lap += flat[N + nz:N + nz + M]                      # y+1
        lap += flat[N - nz:N - nz + M]                      # y-1
        lap += flat[N + 1:N + 1 + M]                        # z+1
        lap += flat[N - 1:N - 1 + M]                        # z-1

        # Everything from here is voxel-local, so it runs in cache-sized
        # chunks: -6*centre, the Euler update, the clamp, the write-back.
        chunk = min(M, chunk_buf.size)
        for lo in range(0, M, chunk):
            m = min(chunk, M - lo)
            block = lap[lo:lo + m]
            centre = flat[N + lo:N + lo + m]
            six_centre = chunk_buf[:m]

            xp.multiply(centre, 6, out=six_centre)
            block -= six_centre
            block *= D
            block *= dt
            block /= resolution ** 2
            block += centre
            xp.maximum(block, 0.0, out=block)
            centre[...] = block

        # Restore the frozen faces, clamped as the original clamped them.
        xp.maximum(face_y0, 0.0, out=field[1:-1, 0, :])
        xp.maximum(face_y1, 0.0, out=field[1:-1, -1, :])
        xp.maximum(face_z0, 0.0, out=field[1:-1, :, 0])
        xp.maximum(face_z1, 0.0, out=field[1:-1, :, -1])
        xp.maximum(field[0], 0.0, out=field[0])
        xp.maximum(field[-1], 0.0, out=field[-1])

        return field

    def _diffuse_local_strided(
        self, field: Any, D: float, dt: float, resolution: float,
    ) -> Any:
        """Fallback for a sub-volume that is not C-contiguous.

        Nothing in this solver produces one — ``DomainDecomposer.scatter``
        and ``MultiGPUBackend.to_device`` both return contiguous arrays — but
        the flat-view kernel would be wrong on a strided array rather than
        slow, so it is not left to assumption.
        """
        xp = _xp_for(field)
        lap = xp.zeros_like(field)
        lap[1:-1, 1:-1, 1:-1] = (
            field[2:, 1:-1, 1:-1] + field[:-2, 1:-1, 1:-1] +
            field[1:-1, 2:, 1:-1] + field[1:-1, :-2, 1:-1] +
            field[1:-1, 1:-1, 2:] + field[1:-1, 1:-1, :-2] -
            6 * field[1:-1, 1:-1, 1:-1]
        )
        lap *= D
        lap *= dt
        lap /= resolution ** 2
        lap += field
        return xp.maximum(lap, 0.0, out=lap)
