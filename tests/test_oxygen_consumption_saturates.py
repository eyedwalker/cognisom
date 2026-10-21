"""Respiration draws at a rate the cell sets, not at a rate the supply sets.

The sink in ``implicit_diffusion`` was first order: R = k(x)*C. That form
consumes MOST where the concentration is HIGHEST, which in a vascularised
tissue is the voxel against the capillary wall, and LEAST in the tissue
between vessels where oxygen is already scarce. It relaxes to an exponential
with no front, so no depth is ever anoxic while its neighbours are fed, and
the Krogh profile the vessel geometry exists to produce cannot form at any
rate. Scaling k up only lowers the whole field together.

``tissue_config`` recorded the reason it was linearised anyway: "A zeroth-order
sink cannot be used in a steady-state solve -- it drives the concentration
negative wherever demand exceeds supply". That objection is real for a constant
sink and does not apply to a saturating one, which shuts itself off as the
field approaches zero. Michaelis-Menten gets both: effectively constant across
the whole physiological range, and provably non-negative.

Cytochrome c oxidase holds a near-maximal rate down to about 1 mmHg, 160x
below the arterial pO2 this field calls 0.21, so over the range the tissue
actually occupies the draw is flat.
"""

import numpy as np
import pytest

from cognisom.gpu.implicit_diffusion import adi_step
from cognisom.core.tissue_config import DiffusionFieldConfig

ARTERIAL = 0.21
KM = 1.0 / 760.0          # 1 mmHg, in the field's atmosphere-fraction units


def _one_voxel_many_steps(c0, k, km, ref, dt, nsteps):
    """Consume in a single cell with no diffusion, so only the sink acts."""
    field = np.full((1, 1, 1), c0, dtype=np.float64)
    uptake = np.full((1, 1, 1), k, dtype=np.float64)
    for _ in range(nsteps):
        field = adi_step(field, 0.0, dt, 10.0, uptake_k=uptake,
                         uptake_km=km, uptake_ref=ref)
    return float(field[0, 0, 0])


def test_the_sink_solves_the_equation_it_claims_to():
    """Backward Euler on dC/dt = -Vmax*C/(km+C), to machine precision.

    The closed form is the positive root of C^2 + (km - C0 + Vmax*dt)C
    - C0*km = 0. Written as (-b + sqrt(b^2+4q))/2 it loses the answer to
    cancellation whenever b is large and positive -- measured 1.8e-4 relative,
    against 2.7e-16 for the conjugate form. A sink that is only approximately
    the sink it documents is a sink nobody can reason about.
    """
    ref, dt = ARTERIAL, 36.0
    worst = 0.0
    for k in (0.7143 / 3600, 71.43 / 3600, 1.0, 1e4):
        for c0 in (ARTERIAL, 0.02, 1e-4, 1e-9):
            cn = _one_voxel_many_steps(c0, k, KM, ref, dt, 1)
            vmax = k * ref
            residual = cn - (c0 - dt * vmax * cn / (KM + cn))
            worst = max(worst, abs(residual) / c0)
    assert worst < 1e-12, f"backward Euler residual {worst:.2e}"


def test_the_saturating_sink_cannot_go_negative():
    """The one objection to a constant sink, removed rather than argued with.

    Demand here exceeds the entire content of the voxel by five orders of
    magnitude. A constant sink would return a large negative concentration and
    a first-order sink would stay positive only by refusing to deplete.
    """
    out = _one_voxel_many_steps(1e-6, 1e4, KM, ARTERIAL, 1e6, 1)
    assert out >= 0.0, out
    assert out < 1e-6, "an overdrawn voxel must still fall"


def test_the_draw_survives_where_it_has_to_do_its_work():
    """The mechanism, in one number: what each sink still draws at hypoxia.

    Both are configured to draw identically at the arterial level -- that is
    what ``uptake_ref`` means -- so this compares shape alone, not strength.
    At the hypoxia line the first-order sink has given up 90% of its demand
    precisely because the oxygen is scarce, while the saturating sink is still
    drawing at 94% of full rate. That is the difference between a tissue that
    can sustain a depleted region against resupply and one that cannot, and it
    is why no first-order rate produces a front.
    """
    k = (0.15 / ARTERIAL) / 3600.0        # the configured oxygen rate, 1/s
    dt = 1.0
    at_arterial_linear = ARTERIAL - _one_voxel_many_steps(
        ARTERIAL, k, 0.0, 1.0, dt, 1)
    at_hypoxia_linear = 0.02 - _one_voxel_many_steps(
        0.02, k, 0.0, 1.0, dt, 1)
    at_arterial_sat = ARTERIAL - _one_voxel_many_steps(
        ARTERIAL, k, KM, ARTERIAL, dt, 1)
    at_hypoxia_sat = 0.02 - _one_voxel_many_steps(
        0.02, k, KM, ARTERIAL, dt, 1)

    assert at_hypoxia_linear / at_arterial_linear < 0.15, (
        "a first-order sink must lose most of its draw as the field empties; "
        "that is the defect"
    )
    assert at_hypoxia_sat / at_arterial_sat > 0.90, (
        "the saturating sink must still be drawing near full rate at the "
        "hypoxia line"
    )


def test_the_crossover_to_supply_limited_happens_at_one_mmhg():
    """Both sinks end up first-order. The question is where the crossover is.

    Below km, C/(km+C) -> C/km and respiration does become supply-limited --
    that is what a Km means, and it is why the saturating sink also decays
    exponentially in the end rather than hitting zero. The difference is
    where: the linear sink crosses over at the arterial level, so it is
    supply-limited everywhere and its decades cost the same forever. The
    saturating sink crosses over at 1 mmHg, and below that its effective rate
    is Vmax/km -- 160x the linear one -- so the same four hours leave it 80-odd
    orders of magnitude lower. Functionally that is an anoxic voxel; formally
    it is still positive, and no mass was destroyed to get there.
    """
    k, dt = (0.15 / ARTERIAL) / 3600.0, 36.0
    nsteps = int(4.0 * 3600 / dt)
    linear = _one_voxel_many_steps(ARTERIAL, k, 0.0, 1.0, dt, nsteps)
    saturating = _one_voxel_many_steps(ARTERIAL, k, KM, ARTERIAL, dt, nsteps)

    assert linear > 0.0, "an exponential cannot reach zero in finite time"
    assert saturating > 0.0, "nor can this one -- but see how far it gets"
    assert linear / ARTERIAL > 1e-2, linear
    assert saturating / ARTERIAL < 1e-9, saturating

    # Equal time per decade is the formal statement of "it never arrives",
    # and the linear sink does it exactly.
    second = _one_voxel_many_steps(linear, k, 0.0, 1.0, dt, nsteps)
    assert second / linear == pytest.approx(linear / ARTERIAL, rel=1e-6)

    # The saturating sink's late rate is set by km, not by the arterial level:
    # Vmax/km against k = Vmax/ref, so the ratio is ref/km. Measured as drawn
    # amounts over a step short enough that backward Euler has not yet begun
    # to compress -- comparing discrete log-ratios at k_eff*dt = 1.1 instead
    # gives 107x, which is 160x * ln(1+x)/x and a statement about the
    # integrator rather than about the sink.
    tiny_c, tiny_dt = 1e-6 * KM, 0.01
    drawn_sat = tiny_c - _one_voxel_many_steps(
        tiny_c, k, KM, ARTERIAL, tiny_dt, 1)
    drawn_lin = tiny_c - _one_voxel_many_steps(
        tiny_c, k, 0.0, 1.0, tiny_dt, 1)
    assert drawn_sat / drawn_lin == pytest.approx(ARTERIAL / KM, rel=0.01), (
        f"the sub-km rate should be {ARTERIAL / KM:.0f}x the linear one, "
        f"got {drawn_sat / drawn_lin:.0f}x"
    )


def test_the_configured_demand_is_unchanged_at_the_reference_level():
    """Switching the sink on must not silently rescale the demand.

    ``build_uptake_from_cells`` bins a per-cell first-order rate onto the
    grid, and ``tissue_config`` chose that rate as q/C_arterial precisely so
    that the draw at arterial equals the measured zeroth-order consumption q.
    Vmax = k*ref keeps that promise: at C = ref the two sinks remove the same
    amount, so the saturating form reinterprets the number without moving it.
    """
    k, dt = (0.15 / ARTERIAL) / 3600.0, 1.0
    linear = _one_voxel_many_steps(ARTERIAL, k, 0.0, 1.0, dt, 1)
    saturating = _one_voxel_many_steps(ARTERIAL, k, KM, ARTERIAL, dt, 1)
    drawn_linear = ARTERIAL - linear
    drawn_saturating = ARTERIAL - saturating
    # km is 160x below arterial, so C/(km+C) = 0.994 here, not 1.000.
    assert drawn_saturating == pytest.approx(drawn_linear, rel=0.01)


def test_zero_km_is_the_untouched_first_order_path():
    """The default must be bit-for-bit what it was, not merely close.

    Every field that has not been given a km -- glucose, every chemokine --
    goes down the original branch. A saturating sink reaching them by default
    would change results nobody asked to change.
    """
    k, dt = 1e-3, 36.0
    with_zero = _one_voxel_many_steps(ARTERIAL, k, 0.0, 1.0, dt, 10)
    expected = ARTERIAL
    for _ in range(10):
        expected = expected / (1.0 + k * dt)
    assert with_zero == expected


def test_the_km_limit_is_a_constant_sink_with_a_floor():
    """As km -> 0 the root must become exactly max(0, C - Vmax*dt).

    This is the identity that makes the form trustworthy: it is the constant
    sink the biology calls for, with the negative values clipped by the
    algebra rather than by a hand-applied floor that would quietly destroy
    mass.
    """
    vmax, dt = 1e-4, 36.0
    for c0 in (ARTERIAL, 0.01, 1e-5, 0.0):
        k = vmax / ARTERIAL
        got = _one_voxel_many_steps(c0, k, 1e-14, ARTERIAL, dt, 1)
        assert got == pytest.approx(max(0.0, c0 - vmax * dt), abs=1e-12), c0


def test_oxygen_asks_for_saturation_and_glucose_does_not():
    """The km is a claim about a gap, and only oxygen has one here.

    Saturation only changes the answer when km sits far below the supply
    concentration. For oxygen that gap is 160x. GLUT1's Km is of the same
    order as blood glucose itself, so for glucose the linear and saturating
    forms are not far apart and picking one would be a guess dressed as a
    correction.
    """
    o2 = DiffusionFieldConfig.oxygen()
    assert o2.uptake_km > 0
    assert o2.uptake_km == pytest.approx(1.0 / 760.0)
    assert o2.initial_value / o2.uptake_km > 100, (
        "a km within two orders of the supply makes the sink first-order over "
        "the whole physiological range, which is what this replaces"
    )
    assert DiffusionFieldConfig.glucose().uptake_km == 0.0


def test_an_invalid_operator_split_says_so(caplog):
    """The limit was documented and unenforced, and the number reassured.

    ``_uptake_field_step`` states the split carries an error of order k*dt and
    that the engine operates at 0.0071, "0.71%". Raising the uptake to chase
    hypoxia drove it to 0.71, and the field came back with a 112% step between
    neighbouring voxels -- a checkerboard that looks exactly like structure.
    So the limit belongs in the code, where it fires.

    The warning also has to say that being UNDER the limit is not safety,
    because that 0.71% is a fraction of the field value while the depletion
    the field is being asked for is 0.002% of the field value. See
    test_the_error_is_tiny_against_the_field_and_huge_against_the_depletion.
    """
    import logging
    from cognisom.gpu.multi_gpu_diffusion import MultiGPUDiffusionSolver

    limit = MultiGPUDiffusionSolver._SPLIT_ERROR_LIMIT
    assert 0 < limit < 1

    dt = 36.0

    class _Stub:
        """Only what _check_split_validity touches."""
        def __init__(self):
            self._mgpu = self
            self._warned_split = set()
            self._SPLIT_ERROR_LIMIT = limit
        to_numpy = staticmethod(lambda a: np.asarray(a))
        _check_split_validity = MultiGPUDiffusionSolver._check_split_validity

    def warnings_for(k):
        stub = _Stub()
        with caplog.at_level(logging.WARNING):
            caplog.clear()
            for _ in range(2):          # twice: must warn once, not twice
                stub._check_split_validity(
                    "oxygen", [np.full((2, 2, 2), k)], dt)
        return [r.getMessage() for r in caplog.records
                if r.levelno >= logging.WARNING]

    assert warnings_for(0.01 * limit / dt) == [], "a valid split stays quiet"

    warned = warnings_for(10.0 * limit / dt)
    assert len(warned) == 1, "warn once per field, not once per step"
    assert "oxygen" in warned[0]
    assert "checkerboard" in warned[0], (
        "the warning has to name the symptom, or it will be read as a "
        "tolerance notice and ignored"
    )
    assert "312x" in warned[0], (
        "and it has to say that staying under the limit is not enough, or "
        "the limit itself becomes the reassurance"
    )
