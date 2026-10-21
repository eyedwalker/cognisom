"""Tissue mixing was set by a numerical accuracy knob.

``mechanics_substeps`` exists to integrate the same physics more
finely. The engine's own docstring says taking snapshots more or less
often cannot change the physics. It did:

    substeps   mean net displacement over 2 simulated hours
         5            7.65 um
        10           11.21 um
        20           15.10 um
        40           19.04 um      2.49x, from an accuracy setting

The cause was motility being applied as a FORCE. A force gives a
displacement proportional to dt, which is the scaling of drift, not of
diffusion -- and the clamp at ``max_displacement_um`` then saturated,
so finer steps meant more capped kicks rather than the same total
motion resolved better.

Diffusion scales as sqrt(dt), and that is the only form which is
invariant under subdivision: n substeps of dt/n contribute a total
variance of n * 2D(dt/n) = 2D dt, exactly one step of dt.
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _run(substeps, *, adhesion, repulsion, hours=1.0, cells=400, seed=7):
    from cognisom.core.tissue_config import TissueScaleConfig
    from cognisom.core.tissue_engine import TissueSimulationEngine

    cfg = TissueScaleConfig(
        n_cells=cells, grid_shape=(12, 12, 12),
        tissue_size_um=(120., 120., 120.), n_gpus=1, dt=0.05,
        duration=hours, enable_ode=False, enable_diffusion=False,
        enable_cell_cycle=False, seed=seed)
    cfg.mechanics_substeps = substeps
    with contextlib.redirect_stdout(io.StringIO()):
        engine = TissueSimulationEngine(cfg)
        engine.initialize()
        engine._mechanics._config.random_force = 0.0
        engine._mechanics._config.adhesion_strength = adhesion
        engine._mechanics._config.repulsion_stiffness = repulsion
        arrays = engine._cell_arrays[0]
        start = np.array(arrays.positions[:arrays.n_real])
        engine.run()
        end = np.array(arrays.positions[:arrays.n_real])
    n = min(len(start), len(end))
    return (float(np.linalg.norm(end[:n] - start[:n], axis=1).mean()),
            engine._mechanics._clamped_fraction)


# ── the arithmetic, which needs no engine ────────────────────────────

def test_subdividing_a_diffusive_step_leaves_the_variance_alone():
    """The whole point, in one line: variance is linear in dt."""
    d_coeff, dt = 0.41, 3.0
    one = 2.0 * d_coeff * dt
    many = sum(2.0 * d_coeff * (dt / 8) for _ in range(8))
    assert one == pytest.approx(many)


def test_a_force_does_not_have_that_property():
    """A displacement proportional to dt adds in quadrature over random
    directions, so subdividing CHANGES the total -- which is why the
    old term could not be made step-independent by tuning it."""
    dt, n = 3.0, 8
    one_step = dt
    subdivided = np.sqrt(n) * (dt / n)
    assert not np.isclose(one_step, subdivided)


# ── the engine ───────────────────────────────────────────────────────

@pytest.mark.slow
def test_the_diffusive_term_is_invariant_under_subdivision():
    """With the contact forces off, only motility moves the cells, and
    the result must not depend on how the step was cut up."""
    runs = {s: _run(s, adhesion=0.0, repulsion=0.0) for s in (5, 20)}
    base = runs[5][0]
    assert base > 1.0, "nothing moved; the test proves nothing"
    for substeps, (displacement, clamped) in runs.items():
        assert displacement / base == pytest.approx(1.0, abs=0.12), (
            f"{substeps} substeps gave {displacement:.2f} um "
            f"against {base:.2f} at 5")
        assert clamped == 0.0, (
            f"the clamp fired on {clamped:.1%} of cells with no contact "
            "forces, so the diffusive step alone is saturating it")


@pytest.mark.slow
def test_the_contact_step_is_invariant_too_once_it_is_solved():
    """With the contact forces on as well.

    This test used to assert the opposite -- that the clamp still bound
    on most cells and some step dependence survived -- because it did.
    Solving the contact relaxation instead of stepping it explicitly
    removed that too: 2.49x before any of this, 1.17x with motility
    fixed alone, and now level with the diffusion-only control.
    """
    low, low_clamped = _run(5, adhesion=400.0, repulsion=2000.0)
    high, high_clamped = _run(20, adhesion=400.0, repulsion=2000.0)

    assert high / low == pytest.approx(1.0, abs=0.15), (
        f"{high:.2f} um at 20 substeps against {low:.2f} at 5")

    # The clamp is now a backstop rather than the model, and finer steps
    # need it LESS -- the opposite of the explicit integrator, where
    # more substeps meant more capped kicks.
    assert high_clamped < 0.4
    assert high_clamped <= low_clamped + 0.02


def test_the_explicit_contact_step_was_far_past_its_stability_limit():
    """Why subdividing could never have fixed it.

    Overdamped relaxation stepped explicitly is stable only while
    k.dt/gamma < 2. At this engine's settings a 0.05 h step cut into 40
    substeps still runs that ratio at 900 for repulsion -- 450x past the
    limit, and 18,000 substeps would be needed to get under it.
    """
    from cognisom.gpu.multi_gpu_mechanics import MechanicsConfig

    cfg = MechanicsConfig()
    dt_seconds = (0.05 / 40) * 3600.0
    ratio = cfg.repulsion_stiffness * dt_seconds / cfg.damping_coeff
    assert ratio > 2.0 * 100, (
        f"k.dt/gamma = {ratio:.0f}; if this is now below 2 the engine "
        "no longer needs the solved step and this test is obsolete")

    substeps_needed = 0.05 / ((2.0 * cfg.damping_coeff
                               / cfg.repulsion_stiffness) / 3600.0)
    assert substeps_needed > 10_000


def test_the_solved_step_recovers_the_explicit_one_for_a_small_step():
    """dx = (F/k)(1 - exp(-k dt/gamma)) must reduce to F dt/gamma when
    the step is short, or the fix would have changed the physics rather
    than the numerics."""
    gamma, k = 10.0, 2000.0
    for dt_seconds in (1e-5, 1e-4):
        solved = -np.expm1(-k * dt_seconds / gamma) / k
        explicit = dt_seconds / gamma
        assert solved == pytest.approx(explicit, rel=1e-2)


def test_the_solved_step_saturates_at_the_equilibrium_offset():
    """And for a long step it goes to F/k -- where the contact balances
    -- instead of overshooting. That bound is the physics; the 1 um
    clamp it replaces was a number someone picked."""
    gamma, k = 10.0, 2000.0
    long_step = 36.0
    solved = -np.expm1(-k * long_step / gamma) / k
    assert solved == pytest.approx(1.0 / k, rel=1e-6)
    assert (long_step / gamma) / solved > 1000


# ── the wiring ───────────────────────────────────────────────────────

def test_the_coefficient_comes_from_the_library_not_a_constant():
    from cognisom.core.tissue_config import TissueScaleConfig
    from cognisom.core.tissue_engine import TissueSimulationEngine
    from cognisom.library.seed_migration_modes import \
        diffusion_coefficient_um2_per_min as D

    cfg = TissueScaleConfig(
        n_cells=100, grid_shape=(8, 8, 8), tissue_size_um=(80., 80., 80.),
        n_gpus=1, dt=0.05, duration=0.2, enable_ode=False,
        enable_diffusion=False, enable_cell_cycle=False)
    with contextlib.redirect_stdout(io.StringIO()):
        engine = TissueSimulationEngine(cfg)
        engine.initialize()

    expected = D(0.234, TissueSimulationEngine.DEFAULT_PERSISTENCE_MIN)
    assert engine._mechanics._config.motility_diffusion_um2_per_min == \
        pytest.approx(expected)
    assert "PMID:25679760" in engine._motility_source


def test_an_unevidenced_speed_switches_the_term_off_rather_than_guessing():
    """42 of the 44 stored migration speeds are placeholders. Reading
    one would be inventing motion."""
    from cognisom.core.tissue_engine import TissueSimulationEngine

    source = (REPO_ROOT / "cognisom" / "core" / "tissue_engine.py").read_text()
    assert "Provenance.PLACEHOLDER" in source
    assert "no evidenced migration speed" in source


def test_the_persistence_time_is_labelled_an_estimate():
    """D = S^2 P / 2 needs P, and the library has no sourced persistence
    for any lineage. The estimate is named, not buried."""
    from cognisom.core.tissue_engine import TissueSimulationEngine

    assert TissueSimulationEngine.DEFAULT_PERSISTENCE_MIN == 15.0
    source = (REPO_ROOT / "cognisom" / "core" / "tissue_engine.py").read_text()
    block = source[source.index("DEFAULT_PERSISTENCE_MIN") - 700:
                   source.index("DEFAULT_PERSISTENCE_MIN") + 60]
    assert "ESTIMATE" in block and "PMID:25679760" in block
