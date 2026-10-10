"""Characterization tests for cell-cell mechanics.

The vectorized neighbour search and force kernel in
``cognisom.gpu.multi_gpu_mechanics`` replaced a pure-Python double loop.
These tests pin the *output* of the original loop so the replacement can be
checked against it rather than against a re-derivation of the physics.

``_ReferenceSpatialHash`` and ``_reference_pair_forces`` below are a frozen,
verbatim copy of the implementation as of commit 38116bc (the Python-dict
spatial hash and the per-cell force loop). They are deliberately *not*
imported from the module under test — that is the whole point. The
``GOLDEN`` table was captured from the live pre-change module, so the frozen
copy is itself pinned and cannot silently drift away from what shipped.
"""

from __future__ import annotations

import numpy as np
import pytest

from cognisom.gpu.multi_gpu_backend import MultiGPUBackend
from cognisom.gpu.multi_gpu_mechanics import MechanicsConfig, MultiGPUCellMechanics
from cognisom.gpu.tissue_distributor import TissueCellArrays


# ── Frozen reference implementation (pre-vectorization) ───────────

class _ReferenceSpatialHash:
    """Verbatim copy of the original Python-dict spatial hash."""

    def __init__(self, bucket_size: float = 15.0):
        self.bucket_size = bucket_size
        self._inv_bucket = 1.0 / bucket_size
        self._table = {}
        self._positions = None
        self._n_cells = 0

    def build(self, positions: np.ndarray) -> None:
        self._positions = positions
        self._n_cells = len(positions)
        self._table.clear()

        if self._n_cells == 0:
            return

        keys = np.floor(positions * self._inv_bucket).astype(np.int64)

        for i in range(self._n_cells):
            k = (int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2]))
            if k not in self._table:
                self._table[k] = []
            self._table[k].append(i)

    def query_neighbors(self, idx: int, max_dist: float):
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


def _reference_pair_forces(cfg, positions, radii, alive, n_real):
    """Verbatim copy of the original ``_compute_forces_cpu``."""
    sh = _ReferenceSpatialHash(cfg.bucket_size_um)
    sh.build(positions)

    k_rep = cfg.repulsion_stiffness
    k_adh = cfg.adhesion_strength
    max_dist = cfg.bucket_size_um * 2
    adhesion_range = 2.0

    forces = np.zeros((n_real, 3), dtype=np.float32)

    for i in range(n_real):
        if not alive[i]:
            continue

        nbrs = sh.query_neighbors(i, max_dist)

        for j in nbrs:
            if not alive[j]:
                continue

            diff = positions[j] - positions[i]
            dist = np.sqrt(diff @ diff)
            if dist < 1e-8:
                continue

            normal = diff / dist
            contact_dist = radii[i] + radii[j]
            overlap = contact_dist - dist

            if overlap > 0:
                force_mag = k_rep * overlap
                forces[i] -= force_mag * normal
            elif overlap > -adhesion_range:
                force_mag = k_adh * (adhesion_range + overlap) / adhesion_range
                forces[i] += force_mag * normal

    return forces


def _reference_pressure(cfg, positions, radii, n_real):
    """Verbatim copy of the original ``compute_pressure`` inner loop."""
    sh = _ReferenceSpatialHash(cfg.bucket_size_um)
    sh.build(positions)

    pressure = np.zeros(n_real, dtype=np.float32)
    max_dist = cfg.bucket_size_um * 2

    for i in range(n_real):
        nbrs = sh.query_neighbors(i, max_dist)
        for j in nbrs:
            diff = positions[j] - positions[i]
            dist = np.sqrt(diff @ diff)
            overlap = (radii[i] + radii[j]) - dist
            if overlap > 0:
                pressure[i] += overlap * cfg.repulsion_stiffness

    return pressure


# ── Seeded configurations ─────────────────────────────────────────

def _case(positions, radii=5.0, alive=None, n_real=None):
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    n = len(positions)
    if np.isscalar(radii):
        radii = np.full(n, radii, dtype=np.float32)
    else:
        radii = np.asarray(radii, dtype=np.float32)
    if alive is None:
        alive = np.ones(n, dtype=bool)
    else:
        alive = np.asarray(alive, dtype=bool)
    return positions, radii, alive, n if n_real is None else n_real


def _random_cloud(seed, n, box, radius=5.0, origin=0.0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, box, size=(n, 3)) + origin
    return _case(pos, radius)


def configurations():
    """Named mechanics configurations, each ``(positions, radii, alive, n_real)``.

    Deliberately includes the boundary cases: a lone cell, cells with no
    neighbour at all, coincident cells, dead cells, ghost cells (so
    ``n_real < n_total``), far more than ``max_neighbors`` contacts, negative
    coordinates, mixed radii, and radii large enough that the contact range
    exceeds the hash bucket size.
    """
    cases = {}

    # Pairs straddling every branch of the force law.
    cases["pair_overlapping"] = _case([[0, 0, 0], [6, 0, 0]])
    cases["pair_just_touching"] = _case([[0, 0, 0], [9.5, 0, 0]])
    cases["pair_adhesive"] = _case([[0, 0, 0], [11.0, 0, 0]])
    cases["pair_out_of_range"] = _case([[0, 0, 0], [13.0, 0, 0]])
    cases["pair_far_apart"] = _case([[0, 0, 0], [400.0, 0, 0]])

    # Degenerate geometry.
    cases["single_cell"] = _case([[12.3, 45.6, 78.9]])
    cases["empty"] = _case(np.zeros((0, 3)))
    cases["coincident"] = _case([[5, 5, 5], [5, 5, 5], [8, 5, 5]])
    cases["isolated_plus_cluster"] = _case(
        [[0, 0, 0], [6, 0, 0], [500, 500, 500]]
    )

    # Dead cells are skipped as both source and neighbour.
    cases["dead_mixed"] = _case(
        [[0, 0, 0], [6, 0, 0], [12, 0, 0], [18, 0, 0]],
        alive=[True, False, True, True],
    )
    cases["all_dead"] = _case([[0, 0, 0], [6, 0, 0]], alive=[False, False])

    # Ghost cells: forces only for the real prefix, neighbours from all.
    ghost_pos = np.array(
        [[0, 0, 0], [6, 0, 0], [11, 0, 0], [16, 0, 0], [21, 0, 0]],
        dtype=np.float64,
    )
    cases["with_ghosts"] = _case(ghost_pos, n_real=2)

    # Bucket-boundary alignment: cells sitting exactly on multiples of 15.
    cases["on_bucket_boundaries"] = _case(
        [[15.0, 15.0, 15.0], [14.999, 15.0, 15.0], [22.0, 15.0, 15.0]]
    )

    # Negative coordinates (negative hash keys).
    cases["negative_coords"] = _random_cloud(7, 60, 60.0, origin=-300.0)

    # Dense: every cell sees far more than max_neighbors contacts.
    cases["dense_200_in_20um"] = _random_cloud(11, 200, 20.0)
    cases["dense_400_in_30um"] = _random_cloud(12, 400, 30.0)

    # Mixed radii.
    mixed = _random_cloud(13, 120, 60.0)
    rng = np.random.default_rng(99)
    cases["mixed_radii"] = (
        mixed[0],
        rng.uniform(2.0, 6.0, size=120).astype(np.float32),
        mixed[2],
        120,
    )

    # Radii whose contact range (2r + 2) exceeds the 15 um bucket: the
    # original stencil cannot see all of these contacts, and the vectorized
    # path must reproduce that, misses included.
    big = _random_cloud(17, 150, 60.0)
    cases["radii_beyond_bucket"] = (
        big[0], np.full(150, 8.0, dtype=np.float32), big[2], 150,
    )

    # Realistic tissue density (10k cells in a 1 mm box -> ~1e-5 cells/um^3).
    cases["sparse_2000"] = _random_cloud(23, 2000, 580.0)
    # Clustered, a few cells per bucket.
    cases["moderate_1500"] = _random_cloud(29, 1500, 150.0)

    return cases


# Captured from the pre-vectorization module (commit 38116bc) by running the
# live `MultiGPUCellMechanics.build_spatial_hash` + `.compute_forces` with
# `random_force=0`. Values are `_signature(...)` of the resulting forces.
GOLDEN = {
    "all_dead": (0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "coincident": (3, 56000.0, 28000.0, 56000.0, 0.0, 0.0),
    "dead_mixed": (2, 16000.0, 8000.0, 16000.0, 0.0, 0.0),
    "dense_200_in_20um": (200, 18943735.273969907, 162909.04607923346, 8898606.559890747, 9302906.764953613, 9494084.931060791),
    "dense_400_in_30um": (400, 23302902.611128837, 126465.33659539509, 11911047.959006786, 10525211.533615112, 11406386.79094696),
    "empty": (0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "isolated_plus_cluster": (2, 16000.0, 8000.0, 16000.0, 0.0, 0.0),
    "mixed_radii": (90, 382457.5053312887, 22545.50909918671, 197085.84413909912, 213986.00892543793, 153385.50111836195),
    "moderate_1500": (1391, 8967901.617221251, 26647.48865137826, 4338402.574697271, 4711484.785013404, 4443067.2492689025),
    "negative_coords": (47, 310763.2012400064, 17242.86556686249, 162885.3928886652, 135531.36609768867, 162639.1722369194),
    "on_bucket_boundaries": (3, 51992.0, 25996.0, 51992.0, 0.0, 0.0),
    "pair_adhesive": (2, 400.0, 200.0, 400.0, 0.0, 0.0),
    "pair_far_apart": (0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "pair_just_touching": (2, 2000.0, 1000.0, 2000.0, 0.0, 0.0),
    "pair_out_of_range": (0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "pair_overlapping": (2, 16000.0, 8000.0, 16000.0, 0.0, 0.0),
    "radii_beyond_bucket": (150, 4471662.067385806, 64319.336869879146, 2250028.8606796265, 2185024.531668663, 2238446.093017578),
    "single_cell": (0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "sparse_2000": (131, 358444.1914769779, 14992.758388409999, 174001.51092748344, 159447.7762913704, 190646.8537390232),
    "with_ghosts": (2, 9400.0, 7800.0, 9400.0, 0.0, 0.0),
}


def _mechanics(cfg=None):
    backend = MultiGPUBackend(n_gpus=1)
    return MultiGPUCellMechanics(
        backend, None, None,
        config=cfg or MechanicsConfig(random_force=0.0),
    )


def _cell_arrays(positions, radii, alive, n_real):
    return [TissueCellArrays(
        device_id=0,
        n_real=n_real,
        n_ghost=len(positions) - n_real,
        positions=positions.copy(),
        radii=radii.copy(),
        alive=alive.copy(),
        velocities=np.zeros((len(positions), 3), dtype=np.float32),
        cell_types=np.zeros(len(positions), dtype=np.int32),
        state=np.zeros((len(positions), 12), dtype=np.float32),
    )]


def _live_forces(positions, radii, alive, n_real, cfg=None):
    mech = _mechanics(cfg)
    arrays = _cell_arrays(positions, radii, alive, n_real)
    mech.build_spatial_hash(arrays)
    return np.asarray(mech.compute_forces(arrays)[0])


def _signature(forces):
    """A cancellation-free fingerprint of a force field.

    Net sums are useless here: Newton's third law makes them zero to within
    float32 noise, so they would pin the summation order rather than the
    physics. Magnitudes and per-axis absolute sums do not cancel.
    """
    if len(forces) == 0:
        return (0, 0.0, 0.0, 0.0, 0.0, 0.0)
    f = forces.astype(np.float64)
    mags = np.linalg.norm(f, axis=1)
    abs_sum = np.abs(f).sum(axis=0)
    return (
        int((mags > 0).sum()),
        float(mags.sum()), float(mags.max()),
        float(abs_sum[0]), float(abs_sum[1]), float(abs_sum[2]),
    )


def _assert_matches_reference(actual, expected):
    """Compare against the scalar loop at its own noise floor.

    The scalar loop accumulated each contact into a float32 array, so a cell
    with N contacts carried up to N * eps32 of rounding error. The
    vectorized kernel sums in float64 and rounds once, which is strictly
    more accurate — the residual below is the *old* code's error, not the
    new code's. The bound is eps32-scaled to the largest force present.
    """
    scale = float(np.abs(np.asarray(expected, dtype=np.float64)).max()) if len(
        np.asarray(expected)
    ) else 0.0
    np.testing.assert_allclose(
        actual, expected, rtol=1e-5, atol=max(1e-3, 2e-6 * scale),
    )


CASES = configurations()


@pytest.mark.parametrize("name", sorted(CASES))
def test_forces_match_the_original_python_loop(name):
    """The vectorized kernel reproduces the pre-change force loop."""
    positions, radii, alive, n_real = CASES[name]
    cfg = MechanicsConfig(random_force=0.0)

    expected = _reference_pair_forces(cfg, positions, radii, alive, n_real)
    actual = _live_forces(positions, radii, alive, n_real, cfg)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    _assert_matches_reference(actual, expected)


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_forces_match_values_captured_before_the_rewrite(name):
    """Pins the frozen reference itself to what the old module produced."""
    positions, radii, alive, n_real = CASES[name]
    actual = _signature(_live_forces(positions, radii, alive, n_real))
    expected = GOLDEN[name]

    assert actual[0] == expected[0], "cells carrying a force"
    np.testing.assert_allclose(actual[1:], expected[1:], rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("name", sorted(CASES))
def test_pressure_matches_the_original_python_loop(name):
    positions, radii, alive, n_real = CASES[name]
    cfg = MechanicsConfig(random_force=0.0)

    expected = _reference_pressure(cfg, positions, radii, n_real)

    mech = _mechanics(cfg)
    arrays = _cell_arrays(positions, radii, alive, n_real)
    mech.build_spatial_hash(arrays)
    actual = np.asarray(mech.compute_pressure(arrays)[0])

    assert actual.shape == expected.shape
    _assert_matches_reference(actual, expected)


def test_newtons_third_law_holds_for_a_pair():
    """Equal and opposite — a property the old loop also satisfied."""
    positions, radii, alive, n_real = CASES["pair_overlapping"]
    forces = _live_forces(positions, radii, alive, n_real)
    np.testing.assert_allclose(forces[0], -forces[1], rtol=1e-6, atol=1e-6)


def test_more_than_max_neighbors_contacts_are_all_summed():
    """The force path never capped contacts, and still does not."""
    positions, radii, alive, n_real = CASES["dense_200_in_20um"]
    cfg = MechanicsConfig(random_force=0.0)

    expected = _reference_pair_forces(cfg, positions, radii, alive, n_real)
    actual = _live_forces(positions, radii, alive, n_real, cfg)

    # Sanity: this configuration really does exceed the 64-contact cap.
    sh = _ReferenceSpatialHash(cfg.bucket_size_um)
    sh.build(positions)
    assert len(sh.query_neighbors(0, 12.0)) > cfg.max_neighbors

    _assert_matches_reference(actual, expected)


def test_integrate_moves_only_alive_cells_and_clamps_displacement():
    cfg = MechanicsConfig(random_force=0.0)
    mech = _mechanics(cfg)
    positions = np.array(
        [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [200.0, 0.0, 0.0]],
    )
    arrays = _cell_arrays(
        positions,
        np.full(3, 5.0, dtype=np.float32),
        np.array([True, False, True]),
        3,
    )
    forces = [np.array(
        [[1e6, 0.0, 0.0], [1e6, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )]

    mech.integrate(arrays, forces, dt=0.001)
    moved = np.asarray(arrays[0].positions) - positions

    # Clamped to max_displacement_um, dead cell pinned, zero force no move.
    np.testing.assert_allclose(moved[0], [cfg.max_displacement_um, 0.0, 0.0])
    np.testing.assert_allclose(moved[1], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(moved[2], [0.0, 0.0, 0.0])


def test_integrate_matches_the_original_formula():
    """dx = clamp(F * dt_seconds / gamma), computed in float32 as before."""
    rng = np.random.default_rng(5)
    cfg = MechanicsConfig(random_force=0.0)
    mech = _mechanics(cfg)

    positions = rng.uniform(0, 500, size=(50, 3))
    alive = rng.random(50) > 0.2
    arrays = _cell_arrays(
        positions, np.full(50, 5.0, dtype=np.float32), alive, 50,
    )
    f = rng.normal(0, 30, size=(50, 3)).astype(np.float32)

    dt = 0.0005
    disp = f * (dt * 3600.0) / cfg.damping_coeff
    mag = np.linalg.norm(disp, axis=1, keepdims=True)
    disp = disp * np.minimum(
        1.0, cfg.max_displacement_um / np.maximum(mag, 1e-10),
    )
    disp = disp * alive[:, None]
    expected = positions + disp.astype(np.float64)

    mech.integrate(arrays, [f.copy()], dt=dt)
    np.testing.assert_allclose(np.asarray(arrays[0].positions), expected)


def test_chemotaxis_and_noise_are_added_on_top_of_contact_forces():
    positions, radii, alive, n_real = CASES["sparse_2000"]
    cfg = MechanicsConfig(random_force=0.0)

    base = _live_forces(positions, radii, alive, n_real, cfg)

    rng = np.random.default_rng(3)
    grad = rng.normal(0, 1e-3, size=(n_real, 3)).astype(np.float32)
    mech = _mechanics(cfg)
    arrays = _cell_arrays(positions, radii, alive, n_real)
    mech.build_spatial_hash(arrays)
    with_grad = np.asarray(mech.compute_forces(arrays, [grad])[0])

    np.testing.assert_allclose(
        with_grad, base + grad * cfg.migration_force, rtol=1e-6, atol=1e-3,
    )


def test_noise_uses_the_numpy_global_stream_as_before():
    positions, radii, alive, n_real = CASES["sparse_2000"]
    cfg = MechanicsConfig(random_force=20.0)
    base = _live_forces(positions, radii, alive, n_real,
                        MechanicsConfig(random_force=0.0))

    np.random.seed(4242)
    noisy = _live_forces(positions, radii, alive, n_real, cfg)

    np.random.seed(4242)
    expected = base + np.random.randn(n_real, 3).astype(np.float32) * 20.0

    np.testing.assert_allclose(noisy, expected, rtol=1e-6, atol=1e-3)
