"""A drug does not reach steady state at hour 0.

WHY THIS MATTERS MORE AS THE HORIZON GROWS. The engine applies a drug
as a step: ``transcriptional_activity`` is one number set at build time,
and ``drug_exposure_of`` varies only with distance to a capillary, never
with time. For a 24-hour run that is a reasonable simplification. For
the months of clinical response the engine wants to reach (limitation
11), it is not: enzalutamide's own pharmacokinetics, already sourced in
``exposure.py`` from Gibbons 2015 (PMID 25917876, full text), give a
half-life of 5.8 days and steady state by day 28. A 90-day run that
starts at full effect skips a ramp covering a third of itself -- and the
early PSA decline is exactly what a clinic follows.

WHAT IS ADDED. ``accumulation_fraction``, the standard repeated-dose
approach to steady state ``1 - 2**(-t/half_life)``, DERIVED from that
sourced half-life. It is off by default
(``drug_accumulation_half_life_days=None``), exactly as
``drug_half_distance_um=None`` means uniform penetration, so every
existing run is bit-identical and turning it on is a deliberate choice.

THE CONSEQUENCE, STATED: with it off, a one-week run has the drug at
full effect throughout while the sourced PK puts it near 57% of steady
state. The current default therefore OVERSTATES early drug effect. That
is a known-wrong simplification rather than a neutral one, which is why
the number is measured below rather than described.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import numpy as np
import pytest

from cognisom.engine.py.pharmacology.exposure import (
    ENZALUTAMIDE_PK, accumulation_fraction)

MODS = {"cellular": True, "spatial": True, "vascular": True,
        "ar_signaling": True, "immune": False, "lymphatic": False,
        "molecular": False, "epigenetic": False, "circadian": False,
        "morphogen": False, "cell_mechanics": False}


# ── the PK curve itself ────────────────────────────────────────────


def test_no_drug_has_accumulated_at_hour_zero():
    """A patient who has just taken the first dose has none on board."""
    assert accumulation_fraction(0.0, 5.8) == 0.0


def test_the_curve_is_monotonic_and_approaches_one():
    prev = -1.0
    for days in (0, 1, 3, 7, 14, 28, 56, 180):
        value = accumulation_fraction(days * 24.0, 5.8)
        assert 0.0 <= value <= 1.0, days
        assert value > prev, f"not monotonic at day {days}"
        prev = value
    assert accumulation_fraction(365 * 24.0, 5.8) > 0.999


def test_the_half_life_and_the_steady_state_day_agree():
    """A CROSS-CHECK between two independently reported numbers from the
    same paper, which is the strongest thing available without new data.

    Gibbons 2015 reports a 5.8-day half-life AND steady state by day 28.
    Those are separate statements, so the standard accumulation curve
    built from the first should land at essentially steady state on the
    second. It gives 96.5%, which is agreement rather than assumption.
    If this drifts, the PK model and its own source have parted company.
    """
    half_life = float(ENZALUTAMIDE_PK["half_life_days"])
    steady_day = float(ENZALUTAMIDE_PK["steady_state_day"])
    at_steady = accumulation_fraction(steady_day * 24.0, half_life)
    assert at_steady == pytest.approx(0.965, abs=0.02), (
        f"the curve from a {half_life} d half-life reaches "
        f"{at_steady:.3f} of steady state on day {steady_day:.0f}, "
        "where the same paper says steady state is reached")
    assert at_steady > 0.95


def test_a_week_long_run_is_far_from_steady_state():
    """The number that makes this worth having: the shipped demo's
    horizon sits near 57% of steady-state exposure, not 100%."""
    week = accumulation_fraction(168.0, 5.8)
    assert week == pytest.approx(0.567, abs=0.02), week
    assert week < 0.65, (
        "if a week now reads as near-steady-state, the half-life has "
        "been changed; check it against Gibbons 2015 before accepting")


def test_a_non_positive_half_life_is_refused():
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            accumulation_fraction(24.0, bad)


def test_the_curve_is_off_when_no_half_life_is_given():
    """None means "already at steady state", the same way
    drug_half_distance_um=None means uniform penetration. Exactly 1.0,
    so a run that does not ask for PK is unchanged to the bit."""
    assert accumulation_fraction(0.0, None) == 1.0
    assert accumulation_fraction(10_000.0, None) == 1.0


# ── wired into a run ───────────────────────────────────────────────


def _run(hours, accumulation=None, drugs=("Enzalutamide",), seed=11):
    from cognisom.dashboard.engine_runner import EngineRunner
    np.random.seed(seed)
    over = None
    if accumulation is not None:
        over = {"ar_signaling": {
            "drug_accumulation_half_life_days": accumulation}}
    runner = EngineRunner(tissue="ring", dt=0.05, duration=float(hours),
                          scenario="Baseline", drugs=drugs,
                          modules_enabled=MODS, overrides=over)
    with redirect_stdout(io.StringIO()):
        runner.build()
        runner.run()
    return runner


def _counts(runner):
    s = runner.engine.modules["cellular"].get_state()
    return tuple(s[k] for k in ("n_cancer", "n_normal", "total_divisions",
                                "total_deaths", "androgen_withdrawal_deaths"))


@pytest.mark.slow
def test_a_run_that_does_not_ask_for_pk_is_unchanged():
    assert _counts(_run(24.0)) == _counts(_run(24.0, accumulation=None))


@pytest.mark.slow
def test_pk_weakens_the_drug_over_a_short_run():
    """THE POINT. With accumulation modelled, a short run has less drug
    on board, so androgen signalling is suppressed less and fewer
    dependent cells die. A run where this changed nothing would mean
    the ramp is not reaching the drug's effect."""
    without = _run(48.0)
    with_pk = _run(48.0, accumulation=float(ENZALUTAMIDE_PK["half_life_days"]))

    ar_without = without.engine.modules["ar_signaling"]
    ar_with = with_pk.engine.modules["ar_signaling"]
    cel_without = without.engine.modules["cellular"]
    cel_with = with_pk.engine.modules["cellular"]

    def median_signalling(runner, ar):
        cel = runner.engine.modules["cellular"]
        vals = [ar.signalling_fraction(c) for c, v in cel.cells.items()
                if v.cell_type == "cancer"]
        vals = [v for v in vals if v is not None]
        return float(np.median(vals)) if vals else float("nan")

    s_without = median_signalling(without, ar_without)
    s_with = median_signalling(with_pk, ar_with)
    assert s_with > s_without, (
        f"signalling with PK {s_with:.3f} should be HIGHER than without "
        f"{s_without:.3f}: less drug has accumulated, so the receptor "
        "is blocked less")
    assert (cel_with.androgen_withdrawal_deaths
            <= cel_without.androgen_withdrawal_deaths), (
        "fewer cells should die of withdrawal when less drug is on "
        "board")


@pytest.mark.slow
def test_the_ramp_converges_so_a_long_run_ends_near_full_effect():
    """The ramp must not be a permanent handicap: by several half-lives
    the treated arm should look like the instant-steady-state one."""
    half_life = float(ENZALUTAMIDE_PK["half_life_days"])
    long_hours = 24.0 * 8 * half_life          # ~46 days, 8 half-lives
    assert accumulation_fraction(long_hours, half_life) > 0.996
    runner = _run(24.0, accumulation=half_life)
    ar = runner.engine.modules["ar_signaling"]
    assert ar.accumulated_fraction() < 0.25, (
        "a one-day run should still be early on the ramp")
