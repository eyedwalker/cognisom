"""The oxygen field is solved for its steady state, and checked against a
reference assembled independently of the solver under test.

Why solve rather than step: oxygen crosses the 60 um gap between capillaries
in L^2/D = 1.8 s, and the engine's step is 180 s, so the transient is not
observable. Stepping it anyway pays the reaction-splitting error, and the cost
of that error is invisible in the usual way of quoting it -- 0.71% of the
FIELD at the engine's validated k*dt -- because the depletion it is meant to
compute is 0.002% of the field. Measured: the stepped scheme over-states the
depletion by about 180x on the geometry below, with every voxel still inside
0.71% of arterial. ``test_the_split_error_is_measured_against_the_wrong_thing``
pins that; this module pins the replacement.

The reference here is a sparse matrix assembled row by row and solved by
Newton with a direct factorisation. It shares no code with
``steady_diffusion`` beyond numpy, so agreement between them is evidence
rather than a tautology -- which matters, because an earlier round of this
work spent several measurements trusting a 1-D reference that could not see
the error it was looking for.
"""

import numpy as np
import pytest

from cognisom.gpu.steady_diffusion import (
    ARTERIAL, INSULATED, solve_steady_state,
)

N, DX, D = 16, 10.0, 2000.0
LEVEL = 0.21
KM = 1.0 / 760.0


def _vessels():
    """Capillaries as lines along k, so the field varies in i and j."""
    mask = np.zeros((N, N, N), bool)
    mask[2::6, 2::6, :] = True
    return mask


def _faces():
    mask = np.zeros((N, N, N), bool)
    for axis in range(3):
        sl = [slice(None)] * 3
        sl[axis] = 0
        mask[tuple(sl)] = True
        sl[axis] = -1
        mask[tuple(sl)] = True
    return mask


def _uptake(pin, scale=100.0):
    up = np.zeros((N, N, N))
    up[1::2, :, :] = (0.15 / LEVEL) * scale / 3600.0
    up[pin] = 0.0
    return up


def _reference(pin, up, insulated):
    """Sparse Newton on the 7-point operator. Independent of the solver.

    ``insulated`` drops the missing neighbour at a face AND the matching
    diagonal term, which is what a mirrored neighbour comes to: the +1 and the
    -1 cancel. Getting that cancellation wrong is how a zero-flux reference
    silently becomes a Dirichlet one.
    """
    sparse = pytest.importorskip("scipy.sparse")
    linalg = pytest.importorskip("scipy.sparse.linalg")

    idx = np.arange(N ** 3).reshape(N, N, N)
    rows, cols, vals = [], [], []
    for i in range(N):
        for j in range(N):
            for k in range(N):
                n = idx[i, j, k]
                if pin[i, j, k]:
                    rows.append(n), cols.append(n), vals.append(1.0)
                    continue
                diag = 0.0
                for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                          (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                    a, b, c = i + d[0], j + d[1], k + d[2]
                    inside = 0 <= a < N and 0 <= b < N and 0 <= c < N
                    if inside:
                        rows.append(n), cols.append(idx[a, b, c])
                        vals.append(D / DX ** 2)
                        diag -= D / DX ** 2
                    elif not insulated:
                        raise AssertionError(
                            "a non-insulated reference must have every face "
                            "voxel pinned, so no neighbour can be outside")
                rows.append(n), cols.append(n), vals.append(diag)
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(N ** 3,) * 2)

    flat_pin, flat_up = pin.ravel(), up.ravel()
    C = np.full(N ** 3, LEVEL)
    for _ in range(80):
        sink = np.where(flat_pin, 0.0,
                        flat_up * LEVEL * C / (KM + np.abs(C)))
        slope = np.where(flat_pin, 0.0,
                         flat_up * LEVEL * KM / (KM + np.abs(C)) ** 2)
        F = A.dot(C) - sink
        F[flat_pin] = C[flat_pin] - LEVEL
        J = (A - sparse.diags(slope)).tolil()
        for n in np.flatnonzero(flat_pin):
            J.rows[n], J.data[n] = [n], [1.0]
        step = linalg.spsolve(J.tocsr(), -F)
        C = C + step
        if np.abs(step).max() < 1e-14:
            break
    return C.reshape(N, N, N)


def _solve(vessels, up, faces, start=None):
    field = np.full((N, N, N), LEVEL) if start is None else start
    return solve_steady_state(
        field, D, DX, uptake_k=up, uptake_km=KM, uptake_ref=LEVEL,
        held=vessels, held_value=LEVEL, faces=faces)


@pytest.mark.parametrize("scale", [1.0, 10.0, 100.0, 1000.0])
@pytest.mark.parametrize("faces", [INSULATED, ARTERIAL])
def test_it_matches_an_independent_sparse_solve(scale, faces):
    """To machine precision, over four decades of consumption."""
    vessels = _vessels()
    insulated = faces == INSULATED
    pin = vessels if insulated else (vessels | _faces())
    up = _uptake(pin, scale)

    exact = _reference(pin, up, insulated)
    got = _solve(vessels, up, faces)

    free = ~pin
    depletion = LEVEL - exact[free].min()
    assert depletion > 0, "no depletion to compare against"
    error = np.abs(got.field[free] - exact[free]).max()
    assert error < 1e-12, (
        f"{error:.3e} absolute, which is {error / depletion:.2%} of the "
        f"depletion -- this should be round-off, not approximation"
    )
    assert got.converged
    assert got.newton_steps <= 4, got.newton_steps


def test_the_faces_are_insulated_by_default_and_it_changes_the_answer():
    """``adi_step`` never writes its outer layer, which makes every face a
    capillary. For a block cut from a larger tissue that is wrong, and it
    dominates once vessels are more than a few voxels apart.
    """
    vessels = _vessels()
    up = _uptake(vessels, 1000.0)
    free = ~(vessels | _faces())

    insulated = _solve(vessels, up, INSULATED)
    arterial = _solve(vessels, up, ARTERIAL)

    dep_ins = LEVEL - insulated.field[free].min()
    dep_art = LEVEL - arterial.field[free].min()
    assert dep_ins > 1.3 * dep_art, (
        f"insulating the faces should deepen the depletion materially; got "
        f"{dep_ins:.3e} against {dep_art:.3e}"
    )
    # And the default is the physical one.
    default = _solve(vessels, up, INSULATED)
    assert np.array_equal(
        default.field,
        solve_steady_state(
            np.full((N, N, N), LEVEL), D, DX, uptake_k=up, uptake_km=KM,
            uptake_ref=LEVEL, held=vessels, held_value=LEVEL).field)


def test_capillaries_hold_exactly_and_nothing_goes_negative():
    """The Dirichlet condition is exact, not approached."""
    vessels = _vessels()
    # Far beyond any physiological demand, so the solve is pushed to anoxia.
    up = _uptake(vessels, 100000.0)
    got = _solve(vessels, up, INSULATED)

    assert np.all(got.field[vessels] == LEVEL)
    assert got.field.min() >= 0.0, got.field.min()
    assert got.field.max() <= LEVEL + 1e-12, got.field.max()
    # Anoxia is reachable, which is the whole point of the exercise.
    assert got.field[~vessels].min() < 0.1 * LEVEL


def test_the_work_does_not_grow_with_the_grid():
    """The claim that makes gathering to the global grid affordable.

    A vessel lattice puts every free voxel within a few cells of a Dirichlet
    point, so the system is well conditioned however large the domain and the
    CG count stays flat. If this ever regresses, the solve becomes O(N) in
    iterations as well as O(N^3) in work per iteration, and the cost model in
    ``MultiGPUDiffusionSolver.solve_steady`` is wrong.
    """
    counts = {}
    for n in (16, 32, 48):
        vessels = np.zeros((n, n, n), bool)
        vessels[2::6, 2::6, :] = True
        up = np.zeros((n, n, n))
        up[1::2, :, :] = (0.15 / LEVEL) * 100.0 / 3600.0
        up[vessels] = 0.0
        got = solve_steady_state(
            np.full((n, n, n), LEVEL), D, DX, uptake_k=up, uptake_km=KM,
            uptake_ref=LEVEL, held=vessels, held_value=LEVEL)
        assert got.converged
        counts[n] = got.cg_steps
    assert max(counts.values()) < 3 * min(counts.values()), counts


def test_a_warm_start_costs_less_than_a_cold_one():
    """Why this is affordable once per engine step: cells move about a
    micrometre between steps, so the previous solution is a good guess."""
    vessels = _vessels()
    up = _uptake(vessels, 100.0)
    cold = _solve(vessels, up, INSULATED)
    warm = _solve(vessels, up, INSULATED, start=cold.field)
    assert warm.newton_steps <= cold.newton_steps
    assert warm.cg_steps < cold.cg_steps, (cold.cg_steps, warm.cg_steps)
    assert warm.converged


def test_a_first_order_sink_is_available_and_is_not_the_default_for_oxygen():
    """km = 0 keeps the linear sink, for fields that genuinely are linear.

    It must still solve exactly -- with km = 0 the problem is linear, so
    Newton should finish in one step.
    """
    vessels = _vessels()
    up = _uptake(vessels, 100.0)
    got = solve_steady_state(
        np.full((N, N, N), LEVEL), D, DX, uptake_k=up, uptake_km=0.0,
        uptake_ref=LEVEL, held=vessels, held_value=LEVEL, faces=INSULATED)
    assert got.converged
    assert got.newton_steps <= 2, (
        "a linear problem should not need Newton iteration")

    from cognisom.core.tissue_config import DiffusionFieldConfig
    assert DiffusionFieldConfig.oxygen().uptake_km > 0
    assert DiffusionFieldConfig.oxygen().steady_state is True
    # A secreted field keeps its transient: its history is the behaviour.
    assert DiffusionFieldConfig.cxcl12().steady_state is False


@pytest.mark.parametrize("kwargs,match", [
    ({"faces": "nonsense"}, "faces must be"),
    ({"diffusion_coeff": 0.0}, "diffusion_coeff must be positive"),
    ({"resolution_um": 0.0}, "resolution_um must be positive"),
    ({"uptake_km": -1.0}, "uptake_km must be non-negative"),
    ({"uptake_k": -1.0}, "uptake_k must be non-negative"),
])
def test_it_refuses_input_it_cannot_honour(kwargs, match):
    args = {"field": np.full((N, N, N), LEVEL), "diffusion_coeff": D,
            "resolution_um": DX}
    args.update(kwargs)
    field = args.pop("field")
    with pytest.raises(ValueError, match=match):
        solve_steady_state(field, args.pop("diffusion_coeff"),
                           args.pop("resolution_um"), **args)


def test_a_mismatched_mask_is_refused_rather_than_broadcast():
    with pytest.raises(ValueError, match="does not match field shape"):
        solve_steady_state(np.full((N, N, N), LEVEL), D, DX,
                           held=np.zeros((4, 4, 4), bool))
