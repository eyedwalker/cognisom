"""Characterization tests for the multi-GPU diffusion solver.

The 7-point stencil in ``cognisom.gpu.multi_gpu_diffusion`` was rewritten to
update the sub-volume in place over contiguous 1-D views instead of building
a full-size Laplacian plus four whole-field temporaries per substep. These
tests pin the *output* of the original expression so the replacement can be
checked against it rather than against a re-derivation of the physics.

``_reference_diffuse_local``, ``_reference_sample`` and
``_reference_gradient`` below are a frozen, verbatim copy of the
implementation as of commit 27daa05. They are deliberately *not* imported
from the module under test — that is the whole point. The ``GOLDEN`` tables
were captured from the live pre-change module, so the frozen copies are
themselves pinned and cannot silently drift away from what shipped.

A handful of tests at the end check the scheme against the analytic solution
of the diffusion equation, so this is not purely self-referential: the
rewrite is held to both "same answer as before" and "still solving the heat
equation".
"""

from __future__ import annotations

import numpy as np
import pytest

from cognisom.gpu.domain_decomposition import DomainDecomposer
from cognisom.gpu.multi_gpu_backend import MultiGPUBackend
from cognisom.gpu.multi_gpu_diffusion import MultiGPUDiffusionSolver
from cognisom.gpu.nccl_ghost_exchange import GhostExchanger


# ── Frozen reference implementation (pre-rewrite) ─────────────────

def _reference_diffuse_local(field, D, dt, resolution):
    """Verbatim copy of the original ``_diffuse_local``."""
    xp = np
    if hasattr(field, '__cuda_array_interface__'):
        try:
            import cupy as cp
            xp = cp
        except ImportError:
            field = np.asarray(field)

    # Compute Laplacian (7-point stencil)
    lap = xp.zeros_like(field)
    lap[1:-1, 1:-1, 1:-1] = (
        field[2:, 1:-1, 1:-1] + field[:-2, 1:-1, 1:-1] +
        field[1:-1, 2:, 1:-1] + field[1:-1, :-2, 1:-1] +
        field[1:-1, 1:-1, 2:] + field[1:-1, 1:-1, :-2] -
        6 * field[1:-1, 1:-1, 1:-1]
    )

    # Euler update
    field = field + D * lap * dt / (resolution ** 2)

    # Non-negative clamp
    field = xp.maximum(field, 0.0)

    return field


def _reference_sample(local_field, positions, partition, gw, axis):
    """Verbatim copy of the original ``sample_at_positions`` inner body."""
    resolution = 10.0  # TODO: pass from config

    n_cells = len(positions)
    if n_cells == 0:
        return np.zeros((0,), dtype=np.float32)

    pos_np = np.asarray(positions)
    voxel = pos_np / resolution

    voxel[:, axis] -= partition.start
    voxel[:, axis] += gw

    field_np = np.asarray(local_field)
    shape = np.array(field_np.shape)
    idx = np.clip(voxel.astype(int), 0, shape - 1)
    values = field_np[idx[:, 0], idx[:, 1], idx[:, 2]]
    return values.astype(np.float32)


def _reference_gradient(local_field, positions, partition, gw, axis, resolution):
    """Verbatim copy of the original ``compute_gradient_at_positions`` body."""
    n_cells = len(positions)
    if n_cells == 0:
        return np.zeros((0, 3), dtype=np.float32)

    pos_np = np.asarray(positions)
    voxel = pos_np / resolution
    voxel[:, axis] -= partition.start
    voxel[:, axis] += gw

    field_np = np.asarray(local_field)
    shape = np.array(field_np.shape)
    idx = np.clip(voxel.astype(int), 1, shape - 2)

    grad = np.zeros((n_cells, 3), dtype=np.float32)
    for d in range(3):
        idx_plus = idx.copy()
        idx_minus = idx.copy()
        idx_plus[:, d] += 1
        idx_minus[:, d] -= 1
        idx_plus[:, d] = np.clip(idx_plus[:, d], 0, shape[d] - 1)
        idx_minus[:, d] = np.clip(idx_minus[:, d], 0, shape[d] - 1)

        c_plus = field_np[idx_plus[:, 0], idx_plus[:, 1], idx_plus[:, 2]]
        c_minus = field_np[idx_minus[:, 0], idx_minus[:, 1], idx_minus[:, 2]]
        grad[:, d] = (c_plus - c_minus) / (2 * resolution)

    return grad


# ── Seeded field configurations ───────────────────────────────────

# D, dt and resolution as the engine passes them at the default config:
# D in um^2/s, dt in *hours*, resolution in um. The resulting
# D*dt/dx^2 = 144 is far outside the 1/6 stability limit of explicit Euler
# (see the unit note in the module docstring of multi_gpu_diffusion), but it
# is what ships, so it is what gets characterized.
SHIPPED = dict(D=2000.0, dt=0.01 / 5, resolution=10.0)
# A self-consistent, stable parameter set for the physics checks.
STABLE = dict(D=1.0, dt=0.1, resolution=1.0)


def _blob(shape, centre, amplitude=1.0, sigma=1.5):
    grids = np.indices(shape, dtype=np.float64)
    r2 = sum((g - c) ** 2 for g, c in zip(grids, centre))
    return (amplitude * np.exp(-r2 / (2 * sigma ** 2))).astype(np.float32)


def field_configurations():
    """Named sub-volume configurations, each a float32 3-D array.

    Deliberately includes the boundary cases: an empty (all-zero) field, a
    uniform field, fields holding negatives so the clamp bites, a single hot
    voxel, a linear ramp, and shapes whose interior along one or more axes
    is empty so the stencil writes nothing at all.
    """
    rng = np.random.default_rng(20261010)
    cases = {}

    cases["zeros"] = np.zeros((9, 8, 7), dtype=np.float32)
    cases["uniform_021"] = np.full((9, 8, 7), 0.21, dtype=np.float32)
    cases["uniform_zero_edge"] = np.full((6, 6, 6), 1e-30, dtype=np.float32)
    cases["random_small"] = (rng.random((9, 8, 7), dtype=np.float32) * 0.21)
    cases["random_cube"] = (rng.random((12, 12, 12), dtype=np.float32))
    cases["point_source"] = np.zeros((11, 11, 11), dtype=np.float32)
    cases["point_source"][5, 5, 5] = 1.0
    cases["blob"] = _blob((13, 13, 13), (6, 6, 6))
    cases["linear_ramp"] = np.ascontiguousarray(
        np.broadcast_to(
            np.arange(10, dtype=np.float32)[:, None, None], (10, 7, 5),
        )
    )
    neg = rng.normal(0.0, 1.0, size=(9, 8, 7)).astype(np.float32)
    cases["signed_noise"] = neg
    cases["all_negative"] = -np.abs(neg)
    hot = np.zeros((8, 8, 8), dtype=np.float32)
    hot[1, 1, 1] = 1.0      # sits in the ghost ring, still updated
    hot[0, 4, 4] = 5.0      # sits on the frozen shell
    cases["edge_hotspots"] = hot
    cases["big_dynamic_range"] = np.array(
        rng.lognormal(0.0, 6.0, size=(9, 8, 7)), dtype=np.float32,
    )

    # Degenerate shapes: the stencil's interior is empty along >= 1 axis, so
    # the original wrote nothing and only the clamp applied.
    cases["flat_x"] = rng.random((2, 6, 6), dtype=np.float32) - 0.5
    cases["flat_y"] = rng.random((6, 2, 6), dtype=np.float32) - 0.5
    cases["flat_z"] = rng.random((6, 6, 2), dtype=np.float32) - 0.5
    cases["single_voxel"] = np.array([[[-3.0]]], dtype=np.float32)
    cases["thin_slab"] = rng.random((3, 3, 3), dtype=np.float32) - 0.5

    return cases


FIELDS = field_configurations()


def _field_signature(f):
    """A fingerprint that catches magnitude *and* spatial rearrangement."""
    a = np.asarray(f, dtype=np.float64)
    if a.size == 0:
        return (0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    total = a.sum()
    moments = []
    for ax in range(3):
        coord = np.arange(a.shape[ax], dtype=np.float64)
        shape = [1, 1, 1]
        shape[ax] = a.shape[ax]
        moments.append(float((a * coord.reshape(shape)).sum()))
    return (
        int((a != 0).sum()),
        float(total), float((a * a).sum()),
        float(a.max()), float(a.min()),
        moments[0], moments[1], moments[2],
    )


# Captured from the pre-rewrite module (commit 27daa05) by running the live
# `MultiGPUDiffusionSolver._diffuse_local` five times (the engine's
# diffusion_substeps) at the shipped D/dt/resolution. Values are
# `_field_signature(...)` of the resulting field.
GOLDEN_FIELDS = {
    'all_negative': (0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    'big_dynamic_range': (504, 727521177.109203, 6.7532698178801304e+16, 220610368.0, 2.3731278986360849e-07, 4333571157.309931, 3610943552.2665787, 2186099592.3175907),
    'blob': (2197, 53.0626398034299, 14.79967246459142, 0.7929220199584961, 3.775134371775124e-11, 318.3758388205795, 318.3758388265589, 318.3758390149389),
    'edge_hotspots': (107, 6.485220219980931, 25.503456935403058, 5.0, 0.0, 1.6829312855928151, 24.386753083279018, 24.386753083268104),
    'flat_x': (31, 8.28045803308487, 2.899636538678525, 0.4934713840484619, 0.0, 3.507585346698761, 16.991666674613953, 18.27955400943756),
    'flat_y': (36, 9.577834367752075, 3.243251761161787, 0.49239206314086914, 0.0, 16.33018946647644, 4.803119421005249, 23.367132663726807),
    'flat_z': (37, 9.533644556999207, 3.3181690398116714, 0.4878743290901184, 0.0, 25.410235106945038, 26.34960788488388, 5.5521479845047),
    'linear_ramp': (315, 1575.0, 9975.0, 9.0, 0.0, 9975.0, 4725.0, 3150.0),
    'point_source': (225, 0.9999993578549038, 0.12223696563321224, 0.29656973481178284, 0.0, 4.999996789274519, 4.9999967902058415, 4.999996789274519),
    'random_cube': (1728, 865.6904982477427, 501.05189337534375, 0.9938924312591553, 0.00032067298889160156, 4724.853532925248, 4778.087493285537, 4725.960294350982),
    'random_small': (504, 51.5431908526225, 6.376512852170979, 0.20893539488315582, 0.0003350668994244188, 210.83923557653907, 185.23328839123133, 153.33082755672513),
    'signed_noise': (364, 194.70997015945613, 185.8396059576069, 3.347071886062622, 0.0, 814.1524176746607, 684.3202028013766, 571.1988731846213),
    'single_voxel': (0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    'thin_slab': (13, 2.8504693135619164, 0.9129604380151677, 0.48140883445739746, 0.0, 2.7600866183638573, 3.6975180730223656, 1.8647763952612877),
    'uniform_021': (504, 105.8399966955185, 22.22639861211779, 0.20999999344348907, 0.20999999344348907, 423.359986782074, 370.4399884343147, 317.5199900865555),
    'uniform_zero_edge': (216, 2.160000006849526e-28, 2.160000013699052e-58, 1.0000000031710769e-30, 1.0000000031710769e-30, 5.400000017123815e-28, 5.400000017123815e-28, 5.400000017123815e-28),
    'zeros': (0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
}


# ── Helpers ───────────────────────────────────────────────────────

def _solver(grid_shape, n_gpus=4, ghost_width=2):
    backend = MultiGPUBackend(n_gpus=n_gpus)
    decomp = DomainDecomposer(
        grid_shape=grid_shape, n_gpus=n_gpus, ghost_width=ghost_width,
    )
    ghost = GhostExchanger(backend, decomp)
    return MultiGPUDiffusionSolver(backend, decomp, ghost)


def _live_diffuse(field, n_substeps=1, **params):
    """Run the module under test on a single sub-volume."""
    solver = _solver((16, 8, 7), n_gpus=1)
    out = field.copy()
    for _ in range(n_substeps):
        out = solver._diffuse_local(
            out, params["D"], params["dt"], params["resolution"],
        )
    return np.asarray(out)


def _reference_diffuse(field, n_substeps=1, **params):
    out = field.copy()
    for _ in range(n_substeps):
        out = _reference_diffuse_local(
            out, params["D"], params["dt"], params["resolution"],
        )
    return np.asarray(out)


def _assert_fields_match(actual, expected):
    """float32 fields produced by the same arithmetic, in the same order."""
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    scale = float(np.abs(expected).max()) if expected.size else 0.0
    np.testing.assert_allclose(
        actual, expected, rtol=1e-6, atol=max(1e-30, 1e-6 * scale),
    )


# ── Characterization: the local stencil ───────────────────────────

@pytest.mark.parametrize("name", sorted(FIELDS))
def test_one_substep_matches_the_original_expression(name):
    field = FIELDS[name]
    expected = _reference_diffuse(field, 1, **SHIPPED)
    actual = _live_diffuse(field, 1, **SHIPPED)
    _assert_fields_match(actual, expected)


@pytest.mark.parametrize("name", sorted(FIELDS))
def test_five_substeps_match_the_original_expression(name):
    """The engine runs diffusion_substeps=5 of these per step."""
    field = FIELDS[name]
    expected = _reference_diffuse(field, 5, **SHIPPED)
    actual = _live_diffuse(field, 5, **SHIPPED)
    _assert_fields_match(actual, expected)


@pytest.mark.parametrize("name", sorted(FIELDS))
def test_stable_parameters_match_the_original_expression(name):
    """Same, at a D*dt/dx^2 inside the CFL limit."""
    field = FIELDS[name]
    expected = _reference_diffuse(field, 20, **STABLE)
    actual = _live_diffuse(field, 20, **STABLE)
    _assert_fields_match(actual, expected)


@pytest.mark.parametrize("name", sorted(GOLDEN_FIELDS))
def test_field_matches_values_captured_before_the_rewrite(name):
    """Pins the frozen reference itself to what the old module produced."""
    actual = _field_signature(_live_diffuse(FIELDS[name], 5, **SHIPPED))
    expected = GOLDEN_FIELDS[name]
    assert actual[0] == expected[0], "non-zero voxel count"
    np.testing.assert_allclose(actual[1:], expected[1:], rtol=1e-6, atol=1e-30)


def test_the_substep_allocates_no_full_size_array():
    """The perf fix: the update is in place, so no new sub-volume per substep.

    The original allocated a zeroed Laplacian plus four whole-field
    temporaries for every substep of every field of every partition. This
    pins the replacement to reusing the caller's array.
    """
    field = FIELDS["random_cube"].copy()
    solver = _solver((16, 8, 7), n_gpus=1)
    out = solver._diffuse_local(field, 2000.0, 0.002, 10.0)
    assert out is field, "the sub-volume was reallocated"

    expected = _reference_diffuse(FIELDS["random_cube"], 1, **SHIPPED)
    _assert_fields_match(np.asarray(out), expected)


def test_a_non_contiguous_sub_volume_still_works():
    """Nothing in the solver produces one, but the kernel must not assume."""
    base = np.zeros((9, 8, 14), dtype=np.float32)
    rng = np.random.default_rng(23)
    view = base[:, :, ::2]
    view[...] = rng.random((9, 8, 7), dtype=np.float32)
    assert not view.flags["C_CONTIGUOUS"]

    solver = _solver((16, 8, 7), n_gpus=1)
    out = np.asarray(solver._diffuse_local(view, 2000.0, 0.002, 10.0))
    expected = _reference_diffuse(np.ascontiguousarray(view), 1, **SHIPPED)
    _assert_fields_match(out, expected)


# ── Characterization: the solver step across partitions ───────────

@pytest.mark.parametrize("n_gpus", [1, 2, 4, 8])
def test_solver_step_matches_the_original_per_partition_loop(n_gpus):
    grid = (24, 10, 9)
    rng = np.random.default_rng(7)
    global_field = rng.random(grid, dtype=np.float32)
    coeffs = {"oxygen": 2000.0, "glucose": 600.0, "cxcl12": 100.0}
    fields = {
        "oxygen": global_field,
        "glucose": global_field * 3.0,
        "cxcl12": np.zeros(grid, dtype=np.float32),
    }

    live = _solver(grid, n_gpus=n_gpus)
    live.initialize_fields({k: v.copy() for k, v in fields.items()}, coeffs)

    ref = _solver(grid, n_gpus=n_gpus)
    ref.initialize_fields({k: v.copy() for k, v in fields.items()}, coeffs)

    for _ in range(5):  # diffusion_substeps
        live.step(SHIPPED["dt"], SHIPPED["resolution"])
        # Frozen reference: same ghost exchange, original local kernel.
        for name, device_fields in ref._fields.items():
            D = ref._diff_coeffs[name]
            if D <= 0:
                continue
            ref._ghost_ex.exchange_field_ghosts(device_fields)
            for gpu_id, local in enumerate(device_fields):
                device_fields[gpu_id] = _reference_diffuse_local(
                    local, D, SHIPPED["dt"], SHIPPED["resolution"],
                )

    for name in fields:
        _assert_fields_match(live.gather_field(name), ref.gather_field(name))


def test_zero_diffusion_coefficient_leaves_the_field_untouched():
    grid = (12, 8, 8)
    rng = np.random.default_rng(3)
    f = rng.random(grid, dtype=np.float32)
    solver = _solver(grid, n_gpus=2)
    solver.initialize_fields({"inert": f.copy()}, {"inert": 0.0})
    solver.step(SHIPPED["dt"], SHIPPED["resolution"])
    np.testing.assert_array_equal(solver.gather_field("inert"), f)


def test_threaded_and_serial_partition_updates_agree():
    """Partitions are independent; running them concurrently must not show."""
    grid = (40, 16, 16)
    rng = np.random.default_rng(11)
    f = rng.random(grid, dtype=np.float32)

    serial = _solver(grid, n_gpus=8)
    serial._n_threads = 1
    serial.initialize_fields({"oxygen": f.copy()}, {"oxygen": 2000.0})

    threaded = _solver(grid, n_gpus=8)
    threaded._n_threads = 4
    threaded._thread_min_voxels = 0      # these sub-volumes are tiny
    threaded.initialize_fields({"oxygen": f.copy()}, {"oxygen": 2000.0})

    for _ in range(5):
        serial.step(SHIPPED["dt"], SHIPPED["resolution"])
        threaded.step(SHIPPED["dt"], SHIPPED["resolution"])

    assert threaded._pool is not None, "the threaded path never ran"
    assert serial._pool is None

    np.testing.assert_array_equal(
        threaded.gather_field("oxygen"), serial.gather_field("oxygen"),
    )


# ── Properties the scheme must keep ───────────────────────────────

def test_a_uniform_field_does_not_change():
    """Zero Laplacian everywhere: the field is a fixed point."""
    f = np.full((16, 12, 12), 0.21, dtype=np.float32)
    out = _live_diffuse(f, 10, **STABLE)
    np.testing.assert_array_equal(out, f)


def test_negatives_are_clamped_away_everywhere_including_the_shell():
    f = np.full((8, 7, 6), -2.0, dtype=np.float32)
    out = _live_diffuse(f, 1, **STABLE)
    assert out.min() == 0.0
    np.testing.assert_array_equal(out, np.zeros_like(f))


def test_the_outer_shell_is_frozen_apart_from_the_clamp():
    """The stencil only writes [1:-1, 1:-1, 1:-1]; the shell just gets clamped."""
    rng = np.random.default_rng(5)
    f = (rng.random((10, 9, 8), dtype=np.float32) - 0.3)
    out = _live_diffuse(f, 1, **STABLE)
    expected_shell = np.maximum(f, 0.0)
    for ax in range(3):
        for idx in (0, -1):
            sl = [slice(None)] * 3
            sl[ax] = idx
            sl = tuple(sl)
            np.testing.assert_array_equal(out[sl], expected_shell[sl])


def test_mass_is_conserved_in_a_closed_domain():
    """A blob that never reaches the shell neither gains nor loses mass."""
    f = _blob((41, 41, 41), (20, 20, 20), amplitude=1.0, sigma=1.5)
    before = float(np.asarray(f, dtype=np.float64).sum())
    out = _live_diffuse(f, 20, **STABLE)
    after = float(np.asarray(out, dtype=np.float64).sum())
    assert out.max() < f.max() * 0.9                 # it really did spread
    # The only loss is float32 rounding: the blob's tail (~1e-14 of peak)
    # never reaches the frozen shell, so no mass leaves through it.
    assert abs(after - before) / before < 1e-5


def test_a_point_source_spreads_symmetrically():
    f = np.zeros((21, 21, 21), dtype=np.float32)
    f[10, 10, 10] = 1.0
    out = _live_diffuse(f, 25, **STABLE)

    # Symmetric under reflection about the centre on every axis, and under
    # exchanging any two axes. Only to float32 rounding: the six neighbour
    # terms are summed in a fixed axis order, which a transpose permutes.
    for ax in range(3):
        np.testing.assert_allclose(out, np.flip(out, axis=ax),
                                   rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(out, out.transpose(1, 0, 2),
                               rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(out, out.transpose(0, 2, 1),
                               rtol=1e-5, atol=1e-12)
    assert out[10, 10, 10] == out.max()


def test_a_point_source_matches_the_analytic_gaussian():
    """Independent check that this is still the diffusion equation.

    The continuum solution of dC/dt = D grad^2 C from a point release of mass
    M at the origin is C(r, t) = M (4 pi D t)^-3/2 exp(-r^2 / 4 D t). The
    discrete 7-point stencil converges to it once the spread covers several
    voxels, which is what is compared here — on an independently derived
    analytic field, not on anything the solver produced.
    """
    n, c = 61, 30
    D, dt, dx = 1.0, 0.1, 1.0
    n_steps = 200
    t = n_steps * dt                      # sigma^2 = 2 D t = 40 voxels^2

    f = np.zeros((n, n, n), dtype=np.float32)
    f[c, c, c] = 1.0
    out = np.asarray(_live_diffuse(f, n_steps, D=D, dt=dt, resolution=dx),
                     dtype=np.float64)

    grids = np.indices((n, n, n), dtype=np.float64)
    r2 = sum((g - c) ** 2 for g in grids)
    analytic = (4 * np.pi * D * t) ** -1.5 * np.exp(-r2 / (4 * D * t))

    # Compare where there is signal to compare (within ~2 sigma of centre).
    sigma = np.sqrt(2 * D * t)
    mask = r2 <= (2 * sigma) ** 2
    rel = np.abs(out[mask] - analytic[mask]) / analytic[mask]
    assert rel.max() < 0.05, f"max relative error {rel.max():.3f}"
    # Second moment: <r^2> = 6 D t for a 3-D point release.
    total = out.sum()
    assert abs(total - 1.0) < 1e-4
    assert abs((out * r2).sum() / total - 6 * D * t) / (6 * D * t) < 1e-3


# ── Characterization: sampling and gradients ──────────────────────

def _position_cases(grid, n_gpus):
    """Position sets per partition, including the awkward ones."""
    rng = np.random.default_rng(31)
    nx, ny, nz = grid
    res = 10.0
    cases = {}
    cases["interior_random"] = rng.uniform(
        0.0, min(nx, ny, nz) * res, size=(40, 3),
    )
    cases["spread_over_grid"] = np.column_stack([
        rng.uniform(0, nx * res, 60),
        rng.uniform(0, ny * res, 60),
        rng.uniform(0, nz * res, 60),
    ])
    cases["out_of_range"] = np.array([
        [-500.0, -500.0, -500.0],
        [nx * res * 3, ny * res * 3, nz * res * 3],
        [0.0, 0.0, 0.0],
        [nx * res - 0.001, ny * res - 0.001, nz * res - 0.001],
    ])
    cases["on_voxel_centres"] = (
        np.indices((3, 3, 3)).reshape(3, -1).T.astype(np.float64) * res + 5.0
    )
    cases["empty"] = np.zeros((0, 3))
    cases["single"] = np.array([[37.0, 21.0, 63.0]])
    return cases


GRID = (24, 10, 9)
N_GPUS = 4
POSITIONS = _position_cases(GRID, N_GPUS)

# Captured from the pre-rewrite module (commit 27daa05): per-partition
# sampled values and gradients, summarized as
# (n, sum, sum|v|, max, min) and (n, sum|g|, max|g|) respectively.
GOLDEN_SAMPLES = {
    'empty': (0, 0.0, 0.0, 0.0, 0.0),
    'interior_random': (160, 69.79429590702057, 69.79429590702057, 0.9994426965713501, 0.017616331577301025),
    'on_voxel_centres': (108, 61.31383156776428, 61.31383156776428, 0.9848034381866455, 0.021659135818481445),
    'out_of_range': (16, 9.098469853401184, 9.098469853401184, 0.8959534168243408, 0.25611692667007446),
    'single': (4, 1.8917145133018494, 1.8917145133018494, 0.880756139755249, 0.13992822170257568),
    'spread_over_grid': (240, 117.32223129272461, 117.32223129272461, 0.9845334887504578, 0.0008170604705810547),
}
GOLDEN_GRADIENTS = {
    'empty': (0, 0.0, 0.0),
    'interior_random': (480, 7.603664730529999, 0.049161605536937714),
    'on_voxel_centres': (324, 5.394958864708315, 0.04720238596200943),
    'out_of_range': (48, 0.8172753970138729, 0.043463315814733505),
    'single': (12, 0.161793272360228, 0.0389668345451355),
    'spread_over_grid': (720, 11.497215884621255, 0.04715700075030327),
}


def _sampling_solver():
    rng = np.random.default_rng(13)
    f = rng.random(GRID, dtype=np.float32)
    solver = _solver(GRID, n_gpus=N_GPUS)
    solver.initialize_fields({"oxygen": f.copy()}, {"oxygen": 2000.0})
    return solver


@pytest.mark.parametrize("name", sorted(POSITIONS))
def test_sampling_matches_the_original_whole_field_copy_path(name):
    solver = _sampling_solver()
    positions = [POSITIONS[name].copy() for _ in range(N_GPUS)]

    actual = solver.sample_at_positions(positions, "oxygen")
    for gpu_id in range(N_GPUS):
        expected = _reference_sample(
            solver._fields["oxygen"][gpu_id], POSITIONS[name].copy(),
            solver._decomp.partitions[gpu_id], solver._decomp._ghost_width,
            solver._decomp.split_axis,
        )
        got = np.asarray(actual[gpu_id])
        assert got.shape == expected.shape
        assert got.dtype == expected.dtype
        np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("name", sorted(POSITIONS))
def test_gradients_match_the_original_whole_field_copy_path(name):
    solver = _sampling_solver()
    positions = [POSITIONS[name].copy() for _ in range(N_GPUS)]

    actual = solver.compute_gradient_at_positions(positions, "oxygen", 10.0)
    for gpu_id in range(N_GPUS):
        expected = _reference_gradient(
            solver._fields["oxygen"][gpu_id], POSITIONS[name].copy(),
            solver._decomp.partitions[gpu_id], solver._decomp._ghost_width,
            solver._decomp.split_axis, 10.0,
        )
        got = np.asarray(actual[gpu_id])
        assert got.shape == expected.shape
        assert got.dtype == expected.dtype
        np.testing.assert_array_equal(got, expected)


def _sample_signature(per_gpu):
    vals = np.concatenate([np.asarray(v, dtype=np.float64).ravel()
                           for v in per_gpu]) if per_gpu else np.zeros(0)
    if vals.size == 0:
        return (0, 0.0, 0.0, 0.0, 0.0)
    return (int(vals.size), float(vals.sum()), float(np.abs(vals).sum()),
            float(vals.max()), float(vals.min()))


@pytest.mark.parametrize("name", sorted(GOLDEN_SAMPLES))
def test_samples_match_values_captured_before_the_rewrite(name):
    solver = _sampling_solver()
    positions = [POSITIONS[name].copy() for _ in range(N_GPUS)]
    actual = _sample_signature(solver.sample_at_positions(positions, "oxygen"))
    np.testing.assert_allclose(actual, GOLDEN_SAMPLES[name], rtol=1e-7, atol=0)


@pytest.mark.parametrize("name", sorted(GOLDEN_GRADIENTS))
def test_gradients_match_values_captured_before_the_rewrite(name):
    solver = _sampling_solver()
    positions = [POSITIONS[name].copy() for _ in range(N_GPUS)]
    grads = solver.compute_gradient_at_positions(positions, "oxygen", 10.0)
    flat = np.concatenate(
        [np.abs(np.asarray(g, dtype=np.float64)).ravel() for g in grads]
    ) if grads else np.zeros(0)
    actual = (int(flat.size), float(flat.sum()),
              float(flat.max()) if flat.size else 0.0)
    np.testing.assert_allclose(actual, GOLDEN_GRADIENTS[name], rtol=1e-7, atol=0)


def test_empty_position_arrays_return_empty_per_gpu_arrays():
    solver = _sampling_solver()
    empty = [np.zeros((0, 3)) for _ in range(N_GPUS)]
    for v in solver.sample_at_positions(empty, "oxygen"):
        assert np.asarray(v).shape == (0,)
        assert np.asarray(v).dtype == np.float32
    for g in solver.compute_gradient_at_positions(empty, "oxygen"):
        assert np.asarray(g).shape == (0, 3)
        assert np.asarray(g).dtype == np.float32


def test_sampling_an_unknown_field_raises():
    solver = _sampling_solver()
    with pytest.raises(KeyError):
        solver.sample_at_positions([np.zeros((1, 3))] * N_GPUS, "nope")
    with pytest.raises(KeyError):
        solver.compute_gradient_at_positions([np.zeros((1, 3))] * N_GPUS, "nope")


def test_sampling_never_copies_the_whole_sub_volume():
    """The perf fix: a handful of voxels must not drag the field to the host.

    On GPU every ``to_numpy`` of a sub-volume is a device-to-host transfer of
    the whole slab; the original did one per sample and one per gradient
    query, per field, per step.
    """
    solver = _sampling_solver()
    sizes = []
    original = solver._mgpu.to_numpy

    def spy(arr):
        out = original(arr)
        sizes.append(np.asarray(out).size)
        return out

    solver._mgpu.to_numpy = spy
    positions = [POSITIONS["spread_over_grid"].copy() for _ in range(N_GPUS)]
    solver.sample_at_positions(positions, "oxygen")
    solver.compute_gradient_at_positions(positions, "oxygen", 10.0)
    solver._mgpu.to_numpy = original

    sub_volume = solver._fields["oxygen"][0].size
    assert all(s < sub_volume for s in sizes), (
        f"a whole {sub_volume}-voxel sub-volume was moved to the host: {sizes}"
    )


def test_sampling_resolution_defaults_to_ten_microns_as_before():
    """``resolution`` is now a parameter; the old hardcoded 10.0 is the default."""
    solver = _sampling_solver()
    positions = [POSITIONS["spread_over_grid"].copy() for _ in range(N_GPUS)]
    default = solver.sample_at_positions(positions, "oxygen")
    explicit = solver.sample_at_positions(positions, "oxygen", resolution=10.0)
    for a, b in zip(default, explicit):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_sampling_honours_a_non_default_resolution():
    """The old code ignored the configured resolution and always used 10 um."""
    solver = _sampling_solver()
    positions = [POSITIONS["spread_over_grid"].copy() for _ in range(N_GPUS)]
    coarse = solver.sample_at_positions(positions, "oxygen", resolution=25.0)
    expected = [
        _reference_sample_at(
            solver, gpu_id, POSITIONS["spread_over_grid"].copy(), 25.0,
        )
        for gpu_id in range(N_GPUS)
    ]
    for a, b in zip(coarse, expected):
        np.testing.assert_array_equal(np.asarray(a), b)


def _reference_sample_at(solver, gpu_id, positions, resolution):
    """The original indexing arithmetic, with the resolution made a parameter."""
    partition = solver._decomp.partitions[gpu_id]
    gw = solver._decomp._ghost_width
    axis = solver._decomp.split_axis
    field_np = np.asarray(solver._fields["oxygen"][gpu_id])
    voxel = positions / resolution
    voxel[:, axis] -= partition.start
    voxel[:, axis] += gw
    shape = np.array(field_np.shape)
    idx = np.clip(voxel.astype(int), 0, shape - 1)
    return field_np[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.float32)


# ── Gather paths (unchanged, pinned so the rewrite cannot disturb them) ──

def test_gather_round_trips_the_field():
    grid = (18, 7, 6)
    rng = np.random.default_rng(17)
    f = rng.random(grid, dtype=np.float32)
    solver = _solver(grid, n_gpus=4)
    solver.initialize_fields({"oxygen": f.copy()}, {"oxygen": 2000.0})
    np.testing.assert_array_equal(solver.gather_field("oxygen"), f)
    for axis in (0, 1, 2):
        for index in (0, 2, grid[axis] - 1, -1):
            np.testing.assert_array_equal(
                solver.gather_field_slice("oxygen", axis, index),
                np.take(f, index, axis=axis),
                err_msg=f"axis={axis} index={index}",
            )
    with pytest.raises(IndexError):
        solver.gather_field_slice("oxygen", 0, grid[0])
    with pytest.raises(ValueError):
        solver.gather_field_slice("oxygen", 3, 0)


def test_a_gathered_slice_is_not_a_live_view_of_the_field():
    grid = (18, 7, 6)
    rng = np.random.default_rng(19)
    f = rng.random(grid, dtype=np.float32)
    solver = _solver(grid, n_gpus=4)
    solver.initialize_fields({"oxygen": f.copy()}, {"oxygen": 2000.0})
    sl = solver.gather_field_slice("oxygen", 0, 5)
    solver.step(SHIPPED["dt"], SHIPPED["resolution"])
    np.testing.assert_array_equal(sl, f[5])
