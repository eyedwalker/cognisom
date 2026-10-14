"""The implicit diffusion solver, checked against analytic and exact-discrete answers.

Explicit Euler cannot integrate this engine's diffusion correctly at any speed.
With the hours/seconds conversion fixed, oxygen at the shipped configuration
gives D*dt/dx^2 = 720 against a 3-D stability limit of 1/6, so stability needs
4320 substeps per engine step instead of 5 -- 864x the work, ~1286 h for a 24 h
run against a 2 h cap. So the integrator changes, and these tests are what say
the replacement is right.

Four of these tests exist because the obvious implementation was wrong, and
each wrong version was caught by comparing against an exact answer rather than
against itself:

  * ``test_uptake_is_divided_by_active_sweeps_not_three`` -- dividing k*dt by a
    hard-coded 3 applies a third of the uptake on a 1-D column, which read as a
    640% steady-state error with an effective lambda of sqrt(3)*lambda.
  * ``test_a_single_interior_row_gets_both_dirichlet_terms`` -- with one
    interior point the first row is also the last, so both boundary terms fold
    into it. The compiled kernels applied only the left one and diverged from
    the NumPy reference by 73% on a (12,12,3) grid.
  * ``test_dirichlet_boundaries_are_never_written`` -- applying uptake as a
    factor after the sweeps scales the boundary-adjacent interior point while
    its pinned neighbour keeps its value, so the interior sees a boundary of
    C_b*(1-k*dt). On a cosh profile that lands at the centre as an error of
    exactly k*dt. Folding uptake onto the diagonal is what fixes it.
  * ``test_steady_state_is_exact_in_1d`` -- the test that proves the fix,
    rather than merely that the code runs.
"""
from __future__ import annotations

import numpy as np
import pytest

from cognisom.gpu.implicit_diffusion import adi_step, have_numba

# The shipped oxygen configuration: TissueScaleConfig.fields["oxygen"].
D_OXYGEN = 2000.0        # um^2/s
DX = 10.0                # um per voxel
DT_ENGINE_S = 0.01 * 3600.0   # cfg.dt is 0.01 HOURS; this is it in seconds
K_OXYGEN = 0.714 / 3600.0     # q/C_arterial per spatial_module, as 1/s

PATHS = [False] + ([True] if have_numba() else [])
PATH_IDS = ["numpy"] + (["numba"] if have_numba() else [])


def _exact_discrete_1d(n, D, k, dx, c_boundary):
    """Solve (D/dx^2)*L*C - k*C = 0 exactly, Dirichlet at both ends.

    This is the answer the scheme's fixed point must reproduce -- not the
    continuum cosh, which also carries the O(dx^2) truncation of the Laplacian.
    """
    m = n - 2
    c = D / (dx * dx)
    A = np.zeros((m, m))
    for i in range(m):
        A[i, i] = -2.0 * c - k
        if i > 0:
            A[i, i - 1] = c
        if i < m - 1:
            A[i, i + 1] = c
    b = np.zeros(m)
    b[0] -= c * c_boundary
    b[-1] -= c * c_boundary
    return np.linalg.solve(A, b)


def _relax_1d(n, dt, use_numba, k=K_OXYGEN, D=D_OXYGEN,
              tol=1e-15, max_steps=400_000):
    """Relax a 1-D column to steady state, stopping when it stops moving.

    Iterating a fixed wall-clock worth of steps wastes most of them at small
    dt: at dt = 0.05 s the column needs ~9k steps to settle and a fixed
    3000 s budget ran 60k, which cost 16 s of suite time per parameter for no
    extra evidence. Converging on the actual increment is both faster and a
    stronger statement -- it asserts the iteration reached a fixed point
    rather than assuming it did.
    """
    C = np.full((n, 1, 1), 0.21)
    cache = {}
    for step in range(max_steps):
        prev = C[1, 0, 0], C[n // 2, 0, 0], C[-2, 0, 0]
        C = adi_step(C, D, dt, DX, uptake_k=k, cache=cache, use_numba=use_numba)
        C[0, 0, 0] = 0.21
        C[-1, 0, 0] = 0.21
        now = C[1, 0, 0], C[n // 2, 0, 0], C[-2, 0, 0]
        if step > 2 and max(abs(a - b) for a, b in zip(prev, now)) <= tol:
            break
    else:  # pragma: no cover - a non-converging scheme is a failure, not a pass
        raise AssertionError(
            f"did not reach a fixed point in {max_steps} steps at dt={dt}")
    return C[:, 0, 0]


# ── the property that makes an implicit scheme worth having ──────────────

@pytest.mark.parametrize("use_numba", PATHS, ids=PATH_IDS)
@pytest.mark.parametrize("r_multiple", [1.0, 60.0, 6000.0, 6_000_000.0])
def test_stable_and_non_negative_far_beyond_the_explicit_limit(use_numba, r_multiple):
    """Explicit Euler needs D*dt/dx^2 <= 1/6. This must not care."""
    dt = r_multiple * (1.0 / 6.0) * DX * DX / D_OXYGEN
    C = np.zeros((21, 21, 21))
    C[10, 10, 10] = 1.0

    out = adi_step(C, D_OXYGEN, dt, DX, use_numba=use_numba)

    assert np.isfinite(out).all(), f"diverged at {r_multiple}x the explicit limit"
    assert out.min() >= 0.0


# ── accuracy against exact answers ───────────────────────────────────────

@pytest.mark.parametrize("use_numba", PATHS, ids=PATH_IDS)
@pytest.mark.parametrize("dt", [100.0, 5.0, 1.0, 0.2])
def test_steady_state_is_exact_in_1d(use_numba, dt):
    """With no directional splitting, the fixed point IS the steady-state system.

    Folding uptake onto the tridiagonal diagonal makes (I - rL + k*dt)C = C
    rearrange to (D/dx^2)*L*C = k*C, so this holds at any dt -- including
    dt = 100 s, where r = 2000 is twelve thousand times the explicit limit.
    """
    n = 61
    got = _relax_1d(n, dt, use_numba)
    want = _exact_discrete_1d(n, D_OXYGEN, K_OXYGEN, DX, 0.21)

    rel = np.abs(got[1:-1] - want) / want
    assert rel.max() < 1e-6, (
        f"dt={dt}: steady state is off by {rel.max():.3%}; the fixed point of "
        f"the scheme should equal the discrete steady-state solve exactly"
    )


@pytest.mark.parametrize("use_numba", PATHS, ids=PATH_IDS)
def test_free_diffusion_reproduces_the_analytic_second_moment(use_numba):
    """<x^2> = 2*D*t for a point source, and mass is conserved."""
    n, D, dx, dt = 81, 1.0, 1.0, 0.2
    C = np.zeros((n, n, n))
    C[n // 2, n // 2, n // 2] = 1.0
    cache = {}
    steps = 20
    for _ in range(steps):
        C = adi_step(C, D, dt, dx, cache=cache, use_numba=use_numba)
    t = steps * dt

    coords = (np.arange(n) - n // 2) * dx
    mass = C.sum()
    var = (C.sum(axis=(1, 2)) * coords ** 2).sum() / mass

    assert abs(mass - 1.0) < 1e-6, f"mass drifted to {mass}"
    assert abs(var - 2 * D * t) / (2 * D * t) < 1e-6, (
        f"second moment {var:.6f} should be 2*D*t = {2*D*t:.6f}"
    )


@pytest.mark.parametrize("use_numba", PATHS, ids=PATH_IDS)
def test_accuracy_at_the_engines_actual_operating_point(use_numba):
    """One step per engine step, at the shipped dt, must be accurate enough.

    The splitting error scales with k*dt, not r, which is why r = 720 is
    harmless here: k*dt = 0.0071 at the configured uptake.
    """
    n = 21
    C = np.full((n, n, n), 0.21)
    cache = {}
    for _ in range(400):
        C = adi_step(C, D_OXYGEN, DT_ENGINE_S, DX, uptake_k=K_OXYGEN,
                     cache=cache, use_numba=use_numba)
        C[0] = C[-1] = 0.21
        C[:, 0] = C[:, -1] = 0.21
        C[:, :, 0] = C[:, :, -1] = 0.21

    # Mild depletion only: lambda = sqrt(D/k) is ~3.2 mm, far larger than this
    # 210 um domain, so the field should sit just below arterial everywhere.
    interior = C[1:-1, 1:-1, 1:-1]
    assert (interior <= 0.21).all()
    assert interior.min() > 0.205, (
        "oxygen should barely deplete over 210 um when lambda is 3.2 mm; "
        f"got min {interior.min():.6f}"
    )


# ── the regressions ──────────────────────────────────────────────────────

@pytest.mark.parametrize("use_numba", PATHS, ids=PATH_IDS)
def test_uptake_is_divided_by_active_sweeps_not_three(use_numba):
    """A 1-D column must consume at k, not k/3.

    Dividing by a hard-coded 3 left two of three sweeps skipped (no interior on
    a length-1 axis) and applied a third of the uptake. The signature was a
    steady state matching lambda' = sqrt(3)*lambda -- a 640% error at the
    centre, and identical at every dt, which is what gave it away: a splitting
    error would shrink with dt.
    """
    n = 61
    # A deliberately strong sink so k/3 vs k is unmistakable.
    k = 0.5
    got = _relax_1d(n, 0.2, use_numba, k=k)
    want = _exact_discrete_1d(n, D_OXYGEN, k, DX, 0.21)
    third = _exact_discrete_1d(n, D_OXYGEN, k / 3.0, DX, 0.21)

    err = np.abs(got[1:-1] - want).max()
    err_third = np.abs(got[1:-1] - third).max()
    assert err < 1e-6, f"not consuming at k: off by {err:.3e}"
    assert err_third > 1e-3, (
        "the profile matches k/3, so uptake is being divided by 3 sweeps when "
        "fewer than 3 actually ran"
    )


@pytest.mark.skipif(not have_numba(), reason="compares the two paths")
@pytest.mark.parametrize("shape", [
    (31, 31, 31), (40, 25, 17), (61, 1, 1), (12, 12, 3), (3, 3, 3),
    (3, 40, 40), (40, 3, 40), (40, 40, 3), (1, 1, 1), (2, 2, 2),
    (5, 3, 1), (3, 1, 3),
])
def test_a_single_interior_row_gets_both_dirichlet_terms(shape):
    """The compiled kernels must agree with the NumPy reference on every shape.

    (12,12,3) is the case that failed: axis 2 has one interior point, so it is
    both the first and the last row and both Dirichlet terms fold into it. The
    kernels applied only the left one -- the right fold lived inside a loop
    that does not execute when m == 1 -- and diverged by 73%.
    """
    rng = np.random.default_rng(0)
    base = rng.random(shape) * 0.21 + 0.01

    ref = adi_step(base.copy(), D_OXYGEN, DT_ENGINE_S, DX,
                   uptake_k=K_OXYGEN, use_numba=False)
    got = adi_step(np.ascontiguousarray(base.copy()), D_OXYGEN, DT_ENGINE_S, DX,
                   uptake_k=K_OXYGEN, use_numba=True)

    assert np.array_equal(ref, got), (
        f"{shape}: compiled and NumPy paths disagree by "
        f"{np.abs(ref - got).max():.3e}"
    )


@pytest.mark.parametrize("use_numba", PATHS, ids=PATH_IDS)
def test_dirichlet_boundaries_are_never_written(use_numba):
    """Boundary planes hold their values, as the explicit solver's do.

    This is what makes a vessel voxel a boundary CONDITION rather than a source
    term, and it is why uptake cannot be applied as a blanket factor after the
    sweeps -- that would decay the pinned values too.
    """
    rng = np.random.default_rng(1)
    C = rng.random((11, 12, 13)) * 0.2 + 0.05
    before = C.copy()

    out = adi_step(C, D_OXYGEN, DT_ENGINE_S, DX, uptake_k=K_OXYGEN,
                   use_numba=use_numba)

    for axis in (0, 1, 2):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = 0
        hi[axis] = -1
        assert np.array_equal(out[tuple(lo)], before[tuple(lo)]), f"axis {axis} low face moved"
        assert np.array_equal(out[tuple(hi)], before[tuple(hi)]), f"axis {axis} high face moved"


# ── contract ─────────────────────────────────────────────────────────────

def test_a_uniform_field_with_no_uptake_does_not_move():
    """No gradient and no sink means nothing to do.

    To rounding, not bit-exactly: the exact solution of (I - rL)x = b for
    uniform b with matching boundaries is x = b, but the Thomas recursion
    reaches it through ~n divisions and accumulates float error. At r = 720
    that lands around 1e-16, which is the tolerance below.
    """
    C = np.full((17, 17, 17), 0.21)
    out = adi_step(C.copy(), D_OXYGEN, DT_ENGINE_S, DX, uptake_k=0.0)
    assert np.abs(out - 0.21).max() < 1e-12, (
        f"uniform field drifted by {np.abs(out - 0.21).max():.3e}"
    )


def test_a_uniform_field_with_uptake_does_move():
    """The companion to the above: the current tissue engine never gets here.

    With uniform fields and no sources or sinks the explicit solver computes a
    Laplacian that is identically zero, so tissue diffusion is a no-op today.
    Uptake is what makes the step mean something.
    """
    C = np.full((17, 17, 17), 0.21)
    out = adi_step(C.copy(), D_OXYGEN, DT_ENGINE_S, DX, uptake_k=K_OXYGEN)
    assert out[8, 8, 8] < 0.21


def test_a_two_dimensional_field_is_rejected():
    with pytest.raises(ValueError, match="3-D"):
        adi_step(np.zeros((4, 4)), D_OXYGEN, 1.0, DX)


def test_a_negative_timestep_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        adi_step(np.zeros((4, 4, 4)), D_OXYGEN, -1.0, DX)


@pytest.mark.parametrize("use_numba", PATHS, ids=PATH_IDS)
def test_the_coefficient_cache_does_not_change_the_answer(use_numba):
    """A shared cache across steps is the point; it must not alias wrongly."""
    rng = np.random.default_rng(2)
    base = rng.random((15, 15, 15)) * 0.2

    shared = {}
    a = base.copy()
    for _ in range(5):
        a = adi_step(a, D_OXYGEN, DT_ENGINE_S, DX, uptake_k=K_OXYGEN,
                     cache=shared, use_numba=use_numba)
    b = base.copy()
    for _ in range(5):
        b = adi_step(b, D_OXYGEN, DT_ENGINE_S, DX, uptake_k=K_OXYGEN,
                     cache=None, use_numba=use_numba)

    assert np.array_equal(a, b)
