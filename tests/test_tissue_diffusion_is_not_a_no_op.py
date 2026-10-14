"""Tissue diffusion has to actually do something, and in the right time unit.

Before this, the tissue engine's diffusion stage was a no-op that still cost
~0.4-1.5 h of wall clock per run. Two independent reasons, both measured here:

  * **No sources or sinks.** ``MultiGPUDiffusionSolver.step`` documents that
    sources and sinks "are **not implemented**, and never were", so the fields
    stayed at their uniform initial values and the Laplacian was identically
    zero at every step. Measured directly: 20 substeps left the field
    bit-identical to its initial state.

  * **Hours passed as seconds.** ``cfg.dt`` is in hours and the field
    coefficients are in um^2/s, so each step advanced 0.01 s of diffusion
    instead of 0.01 h -- 3600x too slow, which for oxygen is a diffusion
    length sqrt(3600) = 60x shorter than intended.

The second cannot be fixed inside the explicit scheme: with the conversion
right, oxygen gives D*dt/dx^2 = 720 against an explicit 3-D limit of 1/6, so
stability would need 4320 substeps per step instead of 5. Hence the implicit
solver, and hence ``TissueScaleConfig.implicit_diffusion``.

``test_the_explicit_scheme_is_still_a_no_op`` pins the old behaviour on
purpose. It is not an endorsement -- it is there so that the explicit path
cannot be quietly "fixed" by converting the units, which would make it
unstable rather than correct.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from cognisom.core.tissue_config import TissueScaleConfig
from cognisom.core.tissue_engine import TissueSimulationEngine


@pytest.fixture(autouse=True)
def _quiet(caplog):
    caplog.set_level(logging.ERROR)


def _run(implicit: bool, steps: int = 20, uptake_multiplier: float = 1.0):
    cfg = TissueScaleConfig.small_test()
    cfg.implicit_diffusion = implicit
    cfg.duration = cfg.dt * steps
    if uptake_multiplier != 1.0:
        ox = cfg.fields["oxygen"]
        ox.uptake_per_cell_per_hour *= uptake_multiplier
    engine = TissueSimulationEngine(cfg)
    engine.initialize()
    before = [f.copy() for f in engine._diffusion._fields["oxygen"]]
    for _ in range(steps):
        engine.step()
    after = engine._diffusion._fields["oxygen"]
    return before, after


def _max_change(before, after) -> float:
    return max(float(np.abs(a - b).max()) for a, b in zip(before, after))


def test_the_explicit_scheme_is_still_a_no_op():
    """Pinned deliberately: the explicit path consumes nothing and moves nothing.

    Kept so the unit bug cannot be "fixed" here by multiplying dt by 3600 --
    that would make this path unstable, not correct. The fix is the implicit
    scheme.
    """
    before, after = _run(implicit=False)

    assert _max_change(before, after) == 0.0, (
        "the explicit path has no sources or sinks, so a uniform field must "
        "stay exactly uniform; if this moved, something now injects mass"
    )


def test_the_implicit_scheme_actually_consumes_oxygen():
    """The whole point: cells take up oxygen, so the field must move."""
    before, after = _run(implicit=True)

    change = _max_change(before, after)
    assert change > 0.0, "oxygen did not move; uptake is not reaching the solver"

    lo = min(float(f.min()) for f in after)
    hi = max(float(f.max()) for f in after)
    assert lo < hi, "no gradient formed, so nothing is consuming locally"
    # Backward Euler is monotone -- (I - rL)^-1 has non-negative entries -- so
    # the field cannot genuinely exceed its boundary value. What is left is
    # float32 rounding: the fields are float32, 0.21 is 0.20999999 in float32,
    # and the measured overshoot is a constant 1.127e-07 (7.6 ULP) that does
    # NOT grow -- identical at 10, 40, 100 and 300 steps. The tolerance is
    # sized for that, not for drift, which this would still catch.
    assert hi <= 0.21 + 1e-6, f"oxygen exceeded arterial by {hi - 0.21:.3e}"
    assert lo >= 0.0, "oxygen went negative"


def test_depletion_tracks_the_diffusion_length():
    """Raising uptake must deepen the gradient, the way sqrt(D/k) says.

    This is the check that the coupling is physical rather than merely
    non-zero. lambda = sqrt(D/k) falls as k rises, so more of the domain sits
    beyond supply range and the minimum drops monotonically.

    It is also how the shipped rate was shown to be too low to produce
    hypoxia: at the configured k = 0.714/h, lambda is ~3.2 mm against a 1 mm
    domain, giving under 1% depletion. Physiological oxygen penetration is
    100-200 um, which this model reaches only near k = 714/h -- roughly 1000x
    the configured value. Calibrating that is a biology decision, not a solver
    one, so this test asserts the ordering rather than any particular depth.
    """
    depths = []
    for mult in (1.0, 10.0, 100.0):
        _, after = _run(implicit=True, steps=30, uptake_multiplier=mult)
        depths.append(min(float(f.min()) for f in after))

    assert depths[0] > depths[1] > depths[2], (
        f"oxygen minimum should fall as uptake rises; got {depths}"
    )


def test_the_implicit_scheme_advances_real_time_not_seconds_as_hours():
    """dt is hours; the solver is handed seconds.

    With uptake switched off, the only difference between the two paths is the
    time unit, and the implicit path must spread a perturbation markedly
    further for the same cfg.dt because it advances 3600x more diffusion.
    """
    cfg = TissueScaleConfig.small_test()
    cfg.implicit_diffusion = True
    for fc in cfg.fields.values():
        fc.uptake_per_cell_per_hour = 0.0
    cfg.duration = cfg.dt * 3

    engine = TissueSimulationEngine(cfg)
    engine.initialize()
    # Dent one interior voxel so there is a gradient for either path to move.
    field = engine._diffusion._fields["oxygen"][0]
    centre = tuple(s // 2 for s in field.shape)
    field[centre] = 0.0
    start = float(field[centre])

    engine.step()
    recovered_implicit = float(engine._diffusion._fields["oxygen"][0][centre])

    cfg2 = TissueScaleConfig.small_test()
    cfg2.implicit_diffusion = False
    for fc in cfg2.fields.values():
        fc.uptake_per_cell_per_hour = 0.0
    cfg2.duration = cfg2.dt * 3
    engine2 = TissueSimulationEngine(cfg2)
    engine2.initialize()
    field2 = engine2._diffusion._fields["oxygen"][0]
    field2[centre] = 0.0
    engine2.step()
    recovered_explicit = float(engine2._diffusion._fields["oxygen"][0][centre])

    assert recovered_implicit > recovered_explicit, (
        f"the implicit path advances dt in seconds (3600x more diffusion) so "
        f"the dent should refill much further: implicit {recovered_implicit:.6f} "
        f"vs explicit {recovered_explicit:.6f} from {start:.6f}"
    )


def test_a_chemokine_carries_no_uptake():
    """CXCL12 is secreted, not consumed, so it must not be given a sink.

    Secretion is still not modelled, so this field stays uniform -- which is
    worth pinning, because giving it an uptake would make it decay toward zero
    and look like a working gradient.
    """
    cfg = TissueScaleConfig()
    assert cfg.fields["cxcl12"].uptake_per_cell_per_hour == 0.0


def test_oxygen_and_glucose_uptake_follow_the_q_over_arterial_convention():
    """k = q_volumetric / C_arterial, as spatial_module argues it must be."""
    cfg = TissueScaleConfig()

    assert cfg.fields["oxygen"].uptake_per_cell_per_hour == pytest.approx(0.15 / 0.21)
    assert cfg.fields["glucose"].uptake_per_cell_per_hour == pytest.approx(0.2 / 5.0)
