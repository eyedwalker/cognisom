"""An error worth 0.71% of the field is 300x the depletion being computed.

``implicit_diffusion._uptake_field_step`` records that the reaction-diffusion
operator split carries an error of order k*dt, and that the engine runs at
k*dt = 0.0071 -- "0.71%". That number is correct, it is relative to the FIELD
VALUE, and it is the most reassuring thing in the module.

What the simulation actually wants from the oxygen field is the DEPLETION: how
far below arterial the tissue between vessels falls, because that is what
decides whether a cell is hypoxic. At the shipped consumption rate the true
depletion is 4.77e-06 against a field of 0.21 -- 0.002% of it. An error worth
0.71% of the field is therefore 300 times the entire quantity of interest.

Measured against an exact Newton solve of the same 7-point operator, at the
engine's dt the split's fixed point depletes by 1.49e-03 where the truth is
4.77e-06: over by 312x. The field still looks almost uniform, every voxel is
within 0.71% of arterial, and the depletion is wrong by two and a half orders
of magnitude.

Two corollaries, both pinned below:

* It is NOT the ADI directional split. One-dimensional and three-dimensional
  geometries give the same 1.49e-03 to three figures, so the multi-dimensional
  cross terms are not what is costing this.

* Iterating is not a remedy. The iterate goes bit-for-bit stationary -- a
  change of exactly 0.0 -- with the 7-point steady-state equation still out
  of balance, because the split's fixed point solves a modified equation.

The uncomfortable consequence is in ``_SPLIT_ERROR_LIMIT``: resolving the real
depletion to 10% needs dt near 0.005 s, about 7,000 substeps per engine step,
which is the same order as the 4,320 the EXPLICIT scheme needs for stability.
The implicit solver bought unconditional stability; it did not buy a long step
for this quantity.
"""

import numpy as np
import pytest

from cognisom.gpu.implicit_diffusion import adi_step

N, DX, D = 16, 10.0, 2000.0
ARTERIAL = 0.21
KM = 1.0 / 760.0
#: The engine's substep: cfg.dt = 0.05 h over 5 diffusion substeps.
ENGINE_DT = 36.0


def _held(kind):
    """``1d`` gives vessel PLANES, ``3d`` vessel LINES along k.

    The faces are held too, because adi_step keeps its outermost layer fixed
    and the exact reference has to apply the same condition there. Giving the
    reference no-flux faces instead leaves a constant offset that reads as
    solver error.
    """
    held = np.zeros((N, N, N), bool)
    if kind == "1d":
        held[2::6, :, :] = True
    else:
        held[2::6, 2::6, :] = True
    for axis in range(3):
        sl = [slice(None)] * 3
        sl[axis] = 0
        held[tuple(sl)] = True
        sl[axis] = -1
        held[tuple(sl)] = True
    return held


def _uptake(held, scale=1.0):
    """The SHIPPED rate by default: scale is a multiple of it."""
    up = np.zeros((N, N, N))
    up[1::2, :, :] = (0.15 / ARTERIAL) * scale / 3600.0
    up[held] = 0.0
    return up


def _exact(held, up):
    """Newton on D grad^2 C = Vmax C/(km+C) over the 7-point stencil."""
    sparse = pytest.importorskip("scipy.sparse")
    linalg = pytest.importorskip("scipy.sparse.linalg")

    idx = np.arange(N ** 3).reshape(N, N, N)
    rows, cols, vals = [], [], []
    for i in range(N):
        for j in range(N):
            for k in range(N):
                n = idx[i, j, k]
                if held[i, j, k]:
                    rows.append(n), cols.append(n), vals.append(1.0)
                    continue
                diag = 0.0
                for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                          (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                    rows.append(n)
                    cols.append(idx[i + d[0], j + d[1], k + d[2]])
                    vals.append(D / DX ** 2)
                    diag -= D / DX ** 2
                rows.append(n), cols.append(n), vals.append(diag)
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(N ** 3,) * 2)

    pin, upf = held.ravel(), up.ravel()
    C = np.full(N ** 3, ARTERIAL)
    for _ in range(80):
        sink = np.where(pin, 0.0, upf * ARTERIAL * C / (KM + np.abs(C)))
        dsink = np.where(pin, 0.0,
                         upf * ARTERIAL * KM / (KM + np.abs(C)) ** 2)
        F = A.dot(C) - sink
        F[pin] = C[pin] - ARTERIAL
        J = (A - sparse.diags(dsink)).tolil()
        for n in np.flatnonzero(pin):
            J.rows[n], J.data[n] = [n], [1.0]
        step = linalg.spsolve(J.tocsr(), -F)
        C = C + step
        if np.abs(step).max() < 1e-13:
            break
    return C.reshape(N, N, N)


def _fixed_point(held, up, dt, iters=4000):
    """Iterate until the field stops moving, vessels re-pinned each step."""
    C = np.full((N, N, N), ARTERIAL)
    stationary = False
    for _ in range(iters):
        before = C.copy()
        C = adi_step(C, D, dt, DX, uptake_k=up,
                     uptake_km=KM, uptake_ref=ARTERIAL)
        C = np.where(held, ARTERIAL, C)
        if np.abs(C - before).max() == 0.0:
            stationary = True
            break
    return C, stationary


def _depletion(field, held):
    return ARTERIAL - float(field[~held].min())


def test_the_error_is_tiny_against_the_field_and_huge_against_the_depletion():
    """The whole point, in one comparison, at the engine's own step.

    k*dt here is 0.0071 -- the value the docstring validates -- so this is
    not a pathological configuration. It is the shipped one.
    """
    held = _held("3d")
    up = _uptake(held)
    assert up.max() * ENGINE_DT == pytest.approx(0.0071, rel=0.05), (
        "this test is about the VALIDATED operating point; if k*dt has moved "
        "the premise needs restating"
    )

    exact = _exact(held, up)
    got, _ = _fixed_point(held, up, ENGINE_DT)
    free = ~held

    # Against the field value, the error is the reassuring 0.71%.
    against_field = (np.abs(got[free] - exact[free]).max()
                     / exact[free].mean())
    assert against_field < 0.02, (
        f"{against_field:.2%} of the field -- this is the number that gets "
        "quoted"
    )

    # Against the depletion, it is two and a half orders of magnitude.
    true_dep = _depletion(exact, held)
    got_dep = _depletion(got, held)
    assert true_dep > 0, "no depletion to compare against"
    assert got_dep / true_dep > 100, (
        f"over-depletion is only {got_dep / true_dep:.0f}x; if the solver has "
        "improved, this test should be retired rather than loosened"
    )


def test_a_shorter_step_recovers_the_depletion_and_a_long_one_cannot():
    """The over-depletion falls with dt, roughly linearly, from 312x."""
    held = _held("3d")
    up = _uptake(held)
    exact = _exact(held, up)
    true_dep = _depletion(exact, held)

    ratios = {}
    for dt in (ENGINE_DT, 1.68, 0.168):
        got, _ = _fixed_point(held, up, dt)
        ratios[dt] = _depletion(got, held) / true_dep

    assert ratios[ENGINE_DT] > ratios[1.68] > ratios[0.168], ratios
    assert ratios[ENGINE_DT] > 100, ratios
    assert ratios[0.168] < 10, ratios
    # Still not 1.0 even at dt = 0.168 s, which is the point of the note in
    # _SPLIT_ERROR_LIMIT about needing ~7,000 substeps.
    assert ratios[0.168] > 1.5, (
        f"dt = 0.168 s already resolves the depletion to "
        f"{ratios[0.168]:.2f}x; if so, the substep estimate is pessimistic "
        "and should be remeasured"
    )


def test_it_is_not_the_multidimensional_split():
    """1-D and 3-D over-deplete identically, so the cross terms are not it.

    Worth pinning because the ADI directional split is the natural suspect --
    its fixed point satisfies (I-rLx)(I-rLy)(I-rLz)C = C, so the cross terms
    genuinely do survive to steady state, and the engine runs at r = 720. It
    is still not what is costing the depletion here.
    """
    one_d = _held("1d")
    three_d = _held("3d")
    flat, _ = _fixed_point(one_d, _uptake(one_d), ENGINE_DT)
    cubed, _ = _fixed_point(three_d, _uptake(three_d), ENGINE_DT)

    assert _depletion(flat, one_d) == pytest.approx(
        _depletion(cubed, three_d), rel=0.05), (
        "if these now differ, the multi-dimensional split IS contributing "
        "and the docstring's attribution is wrong"
    )


def test_iterating_longer_is_not_a_remedy():
    """Stationary to the last bit, with the equation still unbalanced."""
    held = _held("3d")
    up = _uptake(held)
    got, stationary = _fixed_point(held, up, ENGINE_DT)
    assert stationary, (
        "expected a bit-for-bit stationary iterate; if it never settles, the "
        "claim that this is a fixed-point error needs revisiting"
    )

    lap = np.zeros_like(got)
    inner = (slice(1, -1),) * 3
    lap[inner] = (
        got[2:, 1:-1, 1:-1] + got[:-2, 1:-1, 1:-1] + got[1:-1, 2:, 1:-1]
        + got[1:-1, :-2, 1:-1] + got[1:-1, 1:-1, 2:] + got[1:-1, 1:-1, :-2]
        - 6.0 * got[inner]) / (DX * DX)
    sink = up * ARTERIAL * got / (KM + got)
    keep = np.zeros_like(got, bool)
    keep[inner] = True
    keep &= ~held

    imbalance = np.abs(D * lap - sink)[keep].max()
    assert imbalance > sink[keep].max(), (
        "a stationary iterate that satisfies the steady-state equation would "
        "mean the split error had gone away"
    )
