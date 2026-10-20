"""Multi-GPU spatial-hash cell mechanics for tissue-scale simulation.

Replaces O(N²) pairwise force computation with O(N·k) binned neighbour
search. Each partition bins its cells (real + ghost) on a uniform lattice,
then computes JKR adhesion + Hertz repulsion forces over the pairs that the
3x3x3 bin stencil turns up.

Neighbour search, force kernel and integration are whole-array operations
dispatched through ``xp`` (numpy or cupy), so the same code runs on CPU and
on GPU and never round-trips cell state to the host.

``MechanicsConfig.max_neighbors`` bounds the memory of a pair-expansion
chunk, not the number of contacts a cell may have: the scalar loop this
replaced never capped contacts either.

Usage::

    mechanics = MultiGPUCellMechanics(multi_gpu, decomposer, ghost_exchanger)
    mechanics.build_spatial_hash(cell_arrays)
    forces = mechanics.compute_forces(cell_arrays)
    mechanics.integrate(cell_arrays, forces, dt)

"""

from __future__ import annotations

import contextlib
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

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


#: Distance beyond contact over which JKR adhesion still acts (um).
#: Was an inline literal in the original force loop.
ADHESION_RANGE_UM = 2.0

#: Bin-table size target for :class:`NeighborGrid`, in bins per cell. The
#: cost curve is a broad minimum: measured on 8 CPU partitions, 10k and
#: 100k cells, 8 is 30% off the optimum and 128 is 10% off, with 32 best at
#: both scales. See the class docstring for the trade-off it balances.
BINS_PER_CELL = float(os.environ.get("COGNISOM_BINS_PER_CELL", "32"))


# ── Spatial Hash (legacy, scalar) ─────────────────────────────────

class SpatialHash:
    """3D spatial hash grid for fast neighbor lookups on one partition.

    Cells are hashed into buckets of size bucket_size_um.
    Neighbor queries check the 27 surrounding buckets (3x3x3 stencil).

    Retained as public API and as a readable statement of the neighbour
    relation. The simulation itself uses :class:`NeighborGrid`, which
    computes the same relation with array operations instead of a Python
    dict; this class walks Python lists and is far too slow for a step loop.
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


# ── Vectorized neighbour search ───────────────────────────────────

# The 27-bin stencil only needs nine lookups: bins adjacent along z are
# contiguous in the linear bin index, so one (dx, dy) pair covers its three
# dz neighbours as a single contiguous run of sorted cells.
_STENCIL_XY: Tuple[Tuple[int, int], ...] = tuple(
    (dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
)
_N_STENCIL = len(_STENCIL_XY)


def _xp_for(arr: Any):
    """Return the array module that owns ``arr`` (numpy or cupy)."""
    if arr is not None and hasattr(arr, "__cuda_array_interface__"):
        try:
            import cupy
            return cupy
        except ImportError:  # pragma: no cover - device array without cupy
            pass
    return np


class NeighborGrid:
    """Uniform-bin neighbour search expressed as array operations.

    Cells are binned on a uniform lattice, sorted by bin index, and queried
    by gathering the contiguous runs of sorted cells that the 3x3x3 bin
    stencil covers. Every step is a whole-array operation dispatched through
    ``xp``, so the same code runs on NumPy today and on CuPy unchanged.

    The bin edge is at least ``search_radius``, which guarantees the stencil
    contains every cell within ``search_radius`` of a query cell. Beyond
    that the edge trades two costs against each other: a smaller edge means
    more bins to count and prefix-sum (work and memory proportional to
    ``volume / edge**3``), a larger edge means more candidate pairs to test
    (work proportional to ``n_cells * density * edge**3``). Balancing them
    puts the optimum at a fixed number of bins per cell, which is what
    ``bins_per_cell`` expresses; tissue densities here are sparse enough
    that the bin table, not the pair list, is the thing worth capping.

    Candidate pairs are produced in chunks (``pair_budget`` at a time) to
    bound peak memory in dense configurations.
    """

    def __init__(
        self,
        positions: Any,
        search_radius: float,
        bins_per_cell: float = BINS_PER_CELL,
        pair_budget: int = 1 << 20,
        cell_chunk: int = 8192,
    ):
        self._xp = xp = _xp_for(positions)
        self._positions = positions
        self.n_cells = n = int(positions.shape[0])
        self.search_radius = float(search_radius)
        self._pair_budget = int(pair_budget)
        self._cell_chunk = int(cell_chunk)

        self.bin_size = max(float(search_radius), 1e-9)
        self.n_bins = 0
        self._lin = None
        self._starts = None
        self._order = None
        self._offsets = None

        if n == 0:
            return

        # One host round-trip for the bounding box: three scalars, used to
        # size the bin table. Everything else stays on the device.
        lo = positions.min(axis=0)
        hi = positions.max(axis=0)
        extent = [float(v) for v in (hi - lo)]

        budget = max(4096.0, bins_per_cell * n)
        for _ in range(96):
            est = 1.0
            for e in extent:
                est *= int(e / self.bin_size) + 4
                if est > budget:
                    break
            if est <= budget:
                break
            self.bin_size *= 1.5

        inv = 1.0 / self.bin_size
        keys = xp.floor(positions * inv).astype(xp.int64)
        rel = keys - keys.min(axis=0)[None, :]
        span = rel.max(axis=0) + 3          # one empty pad bin at each end
        nx, ny, nz = (int(span[0]), int(span[1]), int(span[2]))

        # Pad by one bin per axis so the +-1 stencil can never wrap from the
        # top of one row into the bottom of the next.
        lin = ((rel[:, 0] + 1) * ny + (rel[:, 1] + 1)) * nz + (rel[:, 2] + 1)
        self.n_bins = nx * ny * nz
        self._lin = lin
        self._order = xp.argsort(lin)
        counts = xp.bincount(lin, minlength=self.n_bins)
        self._starts = xp.concatenate([
            xp.zeros(1, dtype=counts.dtype), xp.cumsum(counts),
        ])
        # dz = -1 is the start of each contiguous three-bin run.
        self._offsets = xp.asarray(
            [(dx * ny + dy) * nz - 1 for dx, dy in _STENCIL_XY],
            dtype=lin.dtype,
        )

    def _runs(self, start: int, end: int):
        """(lo, count) of the sorted-cell runs for query cells [start, end)."""
        lo_bin = self._lin[start:end, None] + self._offsets[None, :]
        lo = self._starts[lo_bin].ravel()
        cnt = self._starts[lo_bin + 3].ravel() - lo
        return lo, cnt

    def iter_candidate_pairs(self, n_query: int) -> Iterator[Tuple[Any, Any]]:
        """Yield ``(i_idx, j_idx)`` candidate pair chunks.

        Every cell pair closer than ``search_radius`` appears exactly once
        (``i`` restricted to the first ``n_query`` cells). Self-pairs and
        pairs beyond the radius may also appear; callers filter them.
        """
        if self._lin is None or n_query <= 0:
            return
        xp = self._xp
        start = 0
        chunk = max(1, self._cell_chunk)

        while start < n_query:
            end = min(start + chunk, n_query)
            lo, cnt = self._runs(start, end)
            total = int(cnt.sum())

            # Too many candidates to expand at once: retry a smaller slice.
            if total > self._pair_budget and (end - start) > 1:
                chunk = max(1, int(chunk * self._pair_budget / max(total, 1)))
                continue

            if total > 0:
                ends = xp.cumsum(cnt)
                slot = xp.repeat(xp.arange(cnt.size, dtype=cnt.dtype), cnt)
                within = xp.arange(total, dtype=cnt.dtype) - (ends - cnt)[slot]
                j_idx = self._order[lo[slot] + within]
                i_idx = slot // _N_STENCIL + start
                yield i_idx, j_idx

            start = end
            if chunk < self._cell_chunk:
                chunk = min(self._cell_chunk, chunk * 2)


# ── Multi-GPU Cell Mechanics ──────────────────────────────────────

class _Partition:
    """Built neighbour state for one partition."""

    __slots__ = ("grid", "cutoff_sq", "stencil_keys")

    def __init__(self, grid: NeighborGrid, cutoff_sq: float, stencil_keys):
        self.grid = grid
        self.cutoff_sq = cutoff_sq
        # Only set when the configured bucket is too small for the cells in
        # it; then the original 27-bucket stencil is reproduced exactly.
        self.stencil_keys = stencil_keys


class MultiGPUCellMechanics:
    """Distributed cell mechanics across multiple GPU partitions.

    Each GPU:
    1. Bins its cells (real + ghost) into a :class:`NeighborGrid`
    2. Computes pairwise JKR forces over the stencil's candidate pairs
    3. Integrates positions with overdamped Langevin dynamics
    """

    def __init__(
        self,
        multi_gpu: "MultiGPUBackend",
        decomposer: "DomainDecomposer",
        ghost_exchanger: "GhostExchanger",
        config: Optional[MechanicsConfig] = None,
        seed: Optional[Any] = None,
    ):
        self._mgpu = multi_gpu
        self._decomp = decomposer
        self._ghost_ex = ghost_exchanger
        self._config = config or MechanicsConfig.default()

        # Random-force generators, one per partition, or None for the
        # global stream. The tissue engine passes a seed, so a run's
        # trajectory is a function of its configuration alone: drawn from
        # the global stream, it changed with anything else in the process
        # that drew a number -- including the visualisation subsample.
        # Standalone use without a seed keeps the global stream.
        if seed is not None and not isinstance(seed, np.random.SeedSequence):
            seed = np.random.SeedSequence(int(seed))
        self._seed: Optional[np.random.SeedSequence] = seed
        self._noise_rngs: Dict[int, Any] = {}

        # Per-GPU neighbour state
        self._hashes: List[Optional[_Partition]] = [
            None for _ in range(multi_gpu.n_gpus)
        ]
        self._warned_bucket_too_small = False

    @property
    def config(self) -> MechanicsConfig:
        return self._config

    def _partition_noise(self, xp, gpu_id: int, n_real: int):
        """(n_real, 3) float32 standard normals for partition ``gpu_id``.

        Each partition has its own stream, addressed by its id rather than
        by the order partitions are visited, so the draw a partition gets
        does not depend on what the others drew.
        """
        if self._seed is None:
            return _normal_noise(xp, n_real)
        rng = self._noise_rngs.get(gpu_id)
        if rng is None:
            child = np.random.SeedSequence(
                self._seed.entropy,
                spawn_key=tuple(self._seed.spawn_key) + (gpu_id,),
            )
            if xp is np:
                rng = np.random.default_rng(child)
            else:                                           # pragma: no cover
                seed_int = int(child.generate_state(1, np.uint64)[0])
                try:
                    rng = xp.random.default_rng(seed_int)
                except AttributeError:
                    rng = xp.random.RandomState(seed_int)
            self._noise_rngs[gpu_id] = rng
        return rng.standard_normal((n_real, 3), dtype=xp.float32)

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

    def build_spatial_hash(
        self,
        cell_arrays: List["TissueCellArrays"],
    ) -> None:
        """Build the neighbour grid on each GPU partition.

        Uses all cells (real + ghost) so boundary forces are correct. The
        arrays stay on their device: positions are binned with ``xp`` ops.

        Args:
            cell_arrays: Per-GPU cell data (from TissueDistributor).
        """
        for gpu_id, ca in enumerate(cell_arrays):
            with self._device_scope(gpu_id):
                self._hashes[gpu_id] = self._build_partition(ca)

        if log.isEnabledFor(logging.DEBUG):
            total_cells = sum(ca.n_total for ca in cell_arrays)
            total_bins = sum(
                p.grid.n_bins for p in self._hashes if p is not None
            )
            log.debug(
                "Neighbour grids built: %d cells in %d bins across %d GPUs",
                total_cells, total_bins, len(cell_arrays),
            )

    def _build_partition(self, ca: "TissueCellArrays") -> _Partition:
        """Bin one partition's cells (real + ghost) for neighbour queries."""
        bucket = self._config.bucket_size_um
        n_total = ca.n_total
        if n_total == 0:
            return _Partition(
                NeighborGrid(_empty_positions(ca.positions), bucket),
                (bucket * 2.0) ** 2, None,
            )

        positions = ca.positions[:n_total]
        radii = ca.radii[:n_total]
        r_max = float(radii.max())

        # Farthest separation at which the force law is still non-zero.
        contact_range = 2.0 * r_max + ADHESION_RANGE_UM
        stencil_keys = None

        if contact_range <= bucket:
            # The original 27-bucket stencil saw every contact, so any
            # search that finds all pairs within contact_range gives
            # identical forces — and the bin edge is free to be tuned.
            search_radius = contact_range
            cutoff_sq = contact_range * contact_range
        else:
            # The configured bucket is smaller than the cells in it, so
            # the original stencil missed contacts depending on how
            # cells aligned to buckets. Reproduce it exactly: search
            # out to max_dist and keep only bucket-adjacent pairs.
            max_dist = bucket * 2.0
            search_radius = max_dist
            cutoff_sq = max_dist * max_dist
            stencil_keys = _xp_for(positions).floor(
                positions * (1.0 / bucket)
            ).astype(np.int64)
            if not self._warned_bucket_too_small:
                self._warned_bucket_too_small = True
                log.warning(
                    "bucket_size_um=%.1f is below the %.1f um contact "
                    "range of cells with radius %.1f um: the 27-bucket "
                    "stencil cannot see every contact. Reproducing the "
                    "original behaviour, misses included.",
                    bucket, contact_range, r_max,
                )

        # max_neighbors sizes the pair-expansion chunk rather than
        # truncating contacts: the scalar force loop called
        # SpatialHash.query_neighbors, which never applied the cap (only
        # the unused query_neighbors_batch did), so capping here would
        # change the physics in dense tissue. Read as "max_neighbors
        # contacts for each of 16k cells at a time".
        return _Partition(
            NeighborGrid(
                positions, search_radius,
                pair_budget=max(1 << 16, self._config.max_neighbors << 14),
            ),
            cutoff_sq, stencil_keys,
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

            part = self._hashes[gpu_id]
            if part is None:
                forces_list.append(
                    self._mgpu.allocate_on_device((n_real, 3), np.float32, gpu_id)
                )
                continue

            with self._device_scope(gpu_id):
                xp = _xp_for(ca.positions)
                forces = self._pair_forces(xp, ca, part, n_real)

                # Add chemotaxis forces if gradients provided
                if gradients is not None and gradients[gpu_id] is not None:
                    grad = gradients[gpu_id]
                    if len(grad) >= n_real:
                        # Chemotaxis: cells migrate up the gradient
                        forces += grad[:n_real] * self._config.migration_force

                # Add random fluctuations (Brownian-like)
                if self._config.random_force > 0:
                    forces += (
                        self._partition_noise(xp, gpu_id, n_real)
                        * self._config.random_force
                    )

            forces_list.append(forces)

        return forces_list

    def integrate(
        self,
        cell_arrays: List["TissueCellArrays"],
        forces: List[Any],
        dt: float,
    ) -> None:
        """Integrate cell positions using overdamped Langevin dynamics.

        dx/dt = F / gamma  (overdamped: inertia negligible)

        Only updates REAL cells (not ghosts), in place on the device.

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

            with self._device_scope(gpu_id):
                xp = _xp_for(ca.positions)
                f = forces[gpu_id]
                alive = ca.alive[:n_real]

                # Overdamped: displacement = F * dt / gamma
                # Convert dt from hours to seconds for force balance
                dt_seconds = dt * 3600.0
                displacement = f[:n_real] * dt_seconds / gamma

                # Clamp displacement for stability
                disp_mag = xp.linalg.norm(displacement, axis=1, keepdims=True)
                scale = xp.minimum(1.0, max_disp / xp.maximum(disp_mag, 1e-10))
                displacement = displacement * scale

                # Only move alive cells
                alive_mask = alive if alive.ndim > 1 else alive[:, None]
                displacement = displacement * alive_mask

                # Update positions in place (ghosts untouched)
                ca.positions[:n_real] += displacement

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
        k_rep = self._config.repulsion_stiffness
        pressure_list = []

        for gpu_id, ca in enumerate(cell_arrays):
            n_real = ca.n_real
            if n_real == 0:
                pressure_list.append(
                    self._mgpu.allocate_on_device((0,), np.float32, gpu_id)
                )
                continue

            part = self._hashes[gpu_id]
            if part is None:
                pressure_list.append(
                    self._mgpu.allocate_on_device((n_real,), np.float32, gpu_id)
                )
                continue

            with self._device_scope(gpu_id):
                xp = _xp_for(ca.positions)
                positions = ca.positions[:ca.n_total]
                radii = ca.radii[:ca.n_total]
                pressure = xp.zeros(n_real, dtype=xp.float64)

                for i_idx, j_idx in part.grid.iter_candidate_pairs(n_real):
                    sel, dist = _select_pairs(
                        xp, positions, i_idx, j_idx, part,
                    )
                    if sel is None:
                        continue
                    ii = i_idx[sel]
                    overlap = (radii[ii] + radii[j_idx[sel]]) - dist
                    weight = xp.where(overlap > 0, overlap * k_rep, 0.0)
                    pressure += xp.bincount(
                        ii, weights=weight, minlength=n_real,
                    )

                pressure_list.append(pressure.astype(xp.float32))

        return pressure_list

    # ── Internal ───────────────────────────────────────────────────

    def _pair_forces(
        self,
        xp: Any,
        ca: "TissueCellArrays",
        part: _Partition,
        n_real: int,
    ) -> Any:
        """Vectorized JKR/Hertz pair forces for one partition.

        JKR-like model (unchanged from the original scalar loop):
        - Overlap > 0: repulsive force = k_rep * overlap * normal
        - Overlap in [-adhesion_range, 0]: adhesive force = -k_adh * |overlap| * normal
        - Overlap < -adhesion_range: no force

        Returns:
            (n_real, 3) float32 force array on the cells' own device.
        """
        k_rep = self._config.repulsion_stiffness
        k_adh = self._config.adhesion_strength
        adh = ADHESION_RANGE_UM

        n_total = ca.n_total
        positions = ca.positions[:n_total]
        radii = ca.radii[:n_total]
        alive = ca.alive[:n_total]
        if alive.dtype != bool:
            alive = alive.astype(bool)

        acc = xp.zeros((n_real, 3), dtype=xp.float64)

        for i_idx, j_idx in part.grid.iter_candidate_pairs(n_real):
            sel, dist = _select_pairs(
                xp, positions, i_idx, j_idx, part,
            )
            if sel is None:
                continue

            ii = i_idx[sel]
            jj = j_idx[sel]
            diff = positions[jj] - positions[ii]

            contact_dist = radii[ii] + radii[jj]
            overlap = contact_dist - dist          # positive = overlapping

            # Coincident cells are skipped, exactly as the scalar loop
            # did, and dead cells neither feel nor exert a force.
            acting = (dist >= 1e-8) & alive[ii] & alive[jj]
            repulsive = overlap > 0
            adhesive = (~repulsive) & (overlap > -adh)

            force_mag = xp.where(repulsive, -(k_rep * overlap), 0.0)
            force_mag = xp.where(
                adhesive, k_adh * (adh + overlap) / adh, force_mag,
            )
            force_mag = xp.where(acting, force_mag, 0.0)

            # normal = diff / dist, then force_mag * normal (same order as
            # the scalar loop, so each pair's contribution is bit-identical).
            safe_dist = xp.where(acting, dist, 1.0)
            for axis in range(3):
                contrib = force_mag * (diff[:, axis] / safe_dist)
                acc[:, axis] += xp.bincount(
                    ii, weights=contrib, minlength=n_real,
                )

        return acc.astype(xp.float32)

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


# ── Pair helpers ──────────────────────────────────────────────────

def _select_pairs(xp, positions, i_idx, j_idx, part: _Partition):
    """Filter candidate pairs down to the ones close enough to interact.

    Returns ``(sel, dist)`` where ``sel`` indexes the surviving candidates
    and ``dist`` is their separation, or ``(None, None)`` if none survive.
    The alive mask is applied afterwards, on the far smaller surviving set.
    """
    diff = positions[j_idx] - positions[i_idx]
    # Summed in the same order as the scalar loop's dot product.
    dist_sq = (
        diff[:, 0] * diff[:, 0]
        + diff[:, 1] * diff[:, 1]
        + diff[:, 2] * diff[:, 2]
    )

    keep = (dist_sq < part.cutoff_sq) & (j_idx != i_idx)
    if part.stencil_keys is not None:
        keep &= _bucket_adjacent(xp, part.stencil_keys, i_idx, j_idx)

    sel = xp.nonzero(keep)[0]
    if sel.size == 0:
        return None, None
    return sel, xp.sqrt(dist_sq[sel])


def _bucket_adjacent(xp, keys, i_idx, j_idx):
    """True where two cells sit in the same or an adjoining hash bucket."""
    delta = xp.abs(keys[j_idx] - keys[i_idx])
    return delta.max(axis=1) <= 1


def _normal_noise(xp, n_real: int):
    """(n_real, 3) float32 standard normals.

    The NumPy path deliberately keeps drawing from the global ``np.random``
    stream so seeded runs reproduce exactly as before.
    """
    if xp is np:
        return np.random.randn(n_real, 3).astype(np.float32)
    return xp.random.standard_normal((n_real, 3), dtype=xp.float32)


def _empty_positions(like: Any):
    """A (0, 3) float64 array on the same device as ``like``."""
    xp = _xp_for(like)
    return xp.zeros((0, 3), dtype=xp.float64)
