"""
Adaptive step-size control in the batched ODE solver.

The module advertised "CVODE-style adaptive time-stepping", "adaptive
step size control with PI controller" and "BDF order 1-5". None of it
was implemented:

  * ``rk45`` was classical fixed-step RK4 -- one solution, no embedded
    pair, so there was nothing to estimate error with.
  * ``rtol`` and ``atol`` were stored on the integrator and never read by
    any code path, so asking for a tighter tolerance changed nothing.
  * ``adams`` called ``_step_bdf()``: a request for a non-stiff method
    silently got a first-order implicit one.
  * The compiled ``_ERROR_ESTIMATION_KERNEL`` was never invoked.

``rk45`` is now Dormand-Prince 5(4) with a PI controller. These tests
check it against closed-form solutions, and check that the tolerance
argument actually constrains the answer -- the property whose absence
was invisible before.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cognisom.gpu.ode_solver import BatchedODEIntegrator, ODESystem


def decay_system(k: float = 2.0) -> ODESystem:
    """dy/dt = -k*y, whose solution is y0*exp(-k*t)."""
    return ODESystem(
        n_species=1,
        species_names=["y"],
        rhs_func=lambda t, y, p: -k * y,
        parameters={"k": k},
        stiff=False,
    )


def oscillator_system(omega: float = 3.0) -> ODESystem:
    """y'' = -w^2 y as a first-order pair. Energy is conserved exactly."""
    def rhs(t, y, p):
        out = np.empty_like(y)
        out[..., 0] = y[..., 1]
        out[..., 1] = -(omega ** 2) * y[..., 0]
        return out

    return ODESystem(
        n_species=2,
        species_names=["x", "v"],
        rhs_func=rhs,
        parameters={"omega": omega},
        stiff=False,
    )


def _integrate(system, y0, t_end, rtol, atol=1e-14, method="rk45"):
    integrator = BatchedODEIntegrator(
        system, n_cells=y0.shape[0], method=method, rtol=rtol, atol=atol
    )
    integrator.integrate((0.0, t_end), y0)
    return integrator


# ── Accuracy tracks tolerance ───────────────────────────────────────

@pytest.mark.parametrize("rtol,max_rel_error", [
    (1e-3, 1e-3),
    (1e-5, 1e-5),
    (1e-7, 1e-7),
    (1e-9, 1e-9),
])
def test_error_respects_the_requested_tolerance(rtol, max_rel_error):
    k, t_end, y_start = 2.0, 2.0, 10.0
    y0 = np.full((4, 1), y_start, dtype=np.float64)

    integrator = _integrate(decay_system(k), y0, t_end, rtol)

    exact = y_start * math.exp(-k * t_end)
    observed = float(integrator.get_state()[0, 0])
    assert abs(observed - exact) / exact < max_rel_error


def test_tighter_tolerance_costs_more_steps_and_buys_more_accuracy():
    """The defining behaviour of an adaptive solver.

    Under the old fixed-step RK4 both of these were identical, because
    rtol was never read.
    """
    k, t_end, y_start = 2.0, 2.0, 10.0
    y0 = np.full((2, 1), y_start, dtype=np.float64)
    exact = y_start * math.exp(-k * t_end)

    loose = _integrate(decay_system(k), y0, t_end, rtol=1e-3)
    tight = _integrate(decay_system(k), y0, t_end, rtol=1e-9)

    loose_err = abs(float(loose.get_state()[0, 0]) - exact)
    tight_err = abs(float(tight.get_state()[0, 0]) - exact)

    assert tight.get_state().shape == loose.get_state().shape
    assert tight._state.n_steps > loose._state.n_steps
    assert tight_err < loose_err


def test_step_size_actually_adapts():
    """dt must move away from its initial value."""
    y0 = np.full((2, 1), 10.0, dtype=np.float64)
    integrator = _integrate(decay_system(), y0, 2.0, rtol=1e-4)

    initial_dt = min(0.01, 2.0 / 100)
    assert integrator._state.dt != pytest.approx(initial_dt)


def test_a_hard_problem_causes_step_rejections():
    """Rejections are how error control differs from just stepping."""
    system = ODESystem(
        n_species=1, species_names=["y"],
        rhs_func=lambda t, y, p: -1000.0 * y,
        parameters={}, stiff=True,
    )
    y0 = np.full((2, 1), 1.0, dtype=np.float64)
    integrator = _integrate(system, y0, 0.1, rtol=1e-6, atol=1e-9)

    assert integrator._state.n_rejected > 0


# ── The interval boundary ───────────────────────────────────────────

def test_integration_stops_exactly_at_the_requested_end_time():
    """A rejected/grown step must not carry the solution past t_end.

    Nothing clamped the final step, so with an adapting dt the reported
    answer was evaluated wherever the last step happened to land -- for
    dy/dt=-2y to t=2 that was t=2.26.
    """
    y0 = np.full((2, 1), 10.0, dtype=np.float64)
    integrator = _integrate(decay_system(), y0, 2.0, rtol=1e-4)

    assert integrator._state.t == pytest.approx(2.0, abs=1e-12)


def test_clamping_the_last_step_does_not_shrink_dt_permanently():
    """The clamp is for the interval, not a controller decision."""
    y0 = np.full((2, 1), 10.0, dtype=np.float64)
    integrator = _integrate(decay_system(), y0, 2.0, rtol=1e-4)

    # Whatever the final partial step was, the retained dt should be the
    # controller's own suggestion, not that sliver.
    assert integrator._state.dt > 1e-4


# ── Precision ───────────────────────────────────────────────────────

def test_float64_input_is_not_downcast():
    """Single precision put a ~1e-7 floor under every solution."""
    y0 = np.full((2, 1), 10.0, dtype=np.float64)
    integrator = _integrate(decay_system(), y0, 2.0, rtol=1e-6)
    assert integrator.get_state().dtype == np.float64


def test_float32_input_is_preserved_for_the_gpu_path():
    y0 = np.full((2, 1), 10.0, dtype=np.float32)
    integrator = _integrate(decay_system(), y0, 2.0, rtol=1e-4)
    assert integrator.get_state().dtype == np.float32


# ── A two-variable system ───────────────────────────────────────────

def test_harmonic_oscillator_tracks_the_analytic_solution():
    omega, t_end = 3.0, 1.5
    y0 = np.tile(np.array([[1.0, 0.0]]), (3, 1))

    integrator = _integrate(oscillator_system(omega), y0, t_end, rtol=1e-10)
    x, v = integrator.get_state()[0]

    assert x == pytest.approx(math.cos(omega * t_end), abs=1e-7)
    assert v == pytest.approx(-omega * math.sin(omega * t_end), abs=1e-6)


def test_harmonic_oscillator_energy_drift_is_bounded():
    """Energy 0.5(v^2 + w^2 x^2) is invariant; drift measures global error."""
    omega, t_end = 3.0, 20.0
    y0 = np.tile(np.array([[1.0, 0.0]]), (2, 1))

    integrator = _integrate(oscillator_system(omega), y0, t_end, rtol=1e-10)
    x, v = integrator.get_state()[0]

    energy = 0.5 * (v ** 2 + omega ** 2 * x ** 2)
    assert energy == pytest.approx(0.5 * omega ** 2, rel=1e-6)


def test_each_cell_integrates_its_own_initial_condition():
    y0 = np.array([[1.0], [5.0], [20.0]], dtype=np.float64)
    k, t_end = 2.0, 1.0

    integrator = _integrate(decay_system(k), y0, t_end, rtol=1e-8)
    final = integrator.get_state()[:, 0]

    expected = y0[:, 0] * math.exp(-k * t_end)
    assert final == pytest.approx(expected, rel=1e-6)


# ── Method routing honesty ──────────────────────────────────────────

def test_adams_no_longer_silently_runs_the_stiff_solver(caplog):
    """It used to call _step_bdf() -- first-order and implicit."""
    y0 = np.full((2, 1), 10.0, dtype=np.float64)
    k, t_end = 2.0, 1.0

    with caplog.at_level("WARNING"):
        integrator = _integrate(
            decay_system(k), y0, t_end, rtol=1e-8, method="adams"
        )

    assert "Adams-Moulton is not implemented" in caplog.text
    # And it produced the accurate explicit result, not backward Euler's.
    exact = 10.0 * math.exp(-k * t_end)
    assert float(integrator.get_state()[0, 0]) == pytest.approx(exact, rel=1e-6)
