"""
T-cell kill probability unit tests.

Covers engine/py/immune/tcell_kill.py:
  * result is in [0, 1]
  * zero affinity, zero MHC, or zero costim with no checkpoint block
    yields zero kill probability
  * monotone in each of affinity, mhc_level, costimulation
  * checkpoint block rescues a PARTIALLY costimulated clone, and cannot
    rescue a CD28-null one (Kamphorst 2017, PMID 28280249) nor push
    past full costimulation
  * out-of-range inputs are clamped (no exceptions, no extrapolation)
  * Hill threshold lowers the EC50 (more permissive kills)
  * kill_outcome carries the full decomposition
"""
from __future__ import annotations

import math

import pytest

from cognisom.engine.py.immune.tcell_kill import (
    DEFAULT_HILL_THRESHOLD,
    PD1_CD28_RELIEF,
    KillOutcome,
    kill_outcome,
    kill_probability,
)


# ---------------------------------------------------------------------------
# Range invariants
# ---------------------------------------------------------------------------

def test_probability_in_unit_interval_for_all_inputs():
    for a in (0.0, 0.3, 0.7, 1.0):
        for m in (0.0, 0.5, 1.0):
            for c in (0.0, 0.5, 1.0):
                for b in (0.0, 0.5, 1.0):
                    p = kill_probability(a, m, c, b)
                    assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# Zero-signal axes
# ---------------------------------------------------------------------------

def test_zero_affinity_yields_zero_kill():
    assert kill_probability(0.0, 1.0, 1.0) == 0.0


def test_zero_mhc_yields_zero_kill():
    # No surface display -- but with checkpoint_block=0, signal collapses
    # only through the first product; the rescue term zeroes out because
    # affinity*mhc*costim=0 *and* checkpoint_block=0.
    assert kill_probability(1.0, 0.0, 1.0) == 0.0


def test_zero_costim_no_checkpoint_block_yields_zero_kill():
    assert kill_probability(1.0, 1.0, 0.0, checkpoint_block=0.0) == 0.0


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------

def test_monotone_in_affinity():
    prev = kill_probability(0.0, 1.0, 1.0)
    for a in (0.1, 0.3, 0.5, 0.8, 1.0):
        p = kill_probability(a, 1.0, 1.0)
        assert p >= prev
        prev = p


def test_monotone_in_mhc_level():
    prev = kill_probability(0.8, 0.0, 1.0)
    for m in (0.1, 0.3, 0.5, 0.8, 1.0):
        p = kill_probability(0.8, m, 1.0)
        assert p >= prev
        prev = p


def test_monotone_in_costim_when_checkpoint_block_zero():
    # With checkpoint_block=0, the rescue term is zero, so the entire
    # signal grows monotonically in costim.
    prev = kill_probability(0.8, 0.8, 0.0)
    for c in (0.1, 0.3, 0.5, 0.8, 1.0):
        p = kill_probability(0.8, 0.8, c)
        assert p >= prev
        prev = p


# ---------------------------------------------------------------------------
# Checkpoint rescue
# ---------------------------------------------------------------------------

def test_checkpoint_block_rescues_a_partly_costimulated_state():
    # Affinity and MHC are high and costimulation is PARTIAL: PD-1 is
    # holding down a CD28 signal that exists, which is the state
    # blockade relieves. Blockade must raise the kill probability.
    #
    # This test used to use costim 0.0 and assert rescue there, back
    # when the rescue term was additive. That is the one place the
    # literature is explicit there should be NO rescue -- see
    # test_checkpoint_block_cannot_rescue_a_cd28_null_clone.
    p_no_block = kill_probability(0.9, 0.9, 0.3, checkpoint_block=0.0)
    p_with_block = kill_probability(0.9, 0.9, 0.3, checkpoint_block=1.0)
    assert p_with_block > p_no_block


def test_checkpoint_block_cannot_rescue_a_cd28_null_clone():
    # Kamphorst 2017 (PMID 28280249): "Rescue of exhausted CD8 T cells
    # by PD-1-targeted therapies is CD28-dependent". With no CD28
    # signal there is nothing for PD-1 relief to restore, so blockade
    # does nothing. The additive rescue term was at its MAXIMUM here
    # and returned 0.885.
    assert kill_probability(0.9, 0.9, 0.0, checkpoint_block=0.0) == 0.0
    assert kill_probability(0.9, 0.9, 0.0, checkpoint_block=1.0) == 0.0


def test_checkpoint_block_cannot_exceed_full_costimulation():
    # Blockade restores CD28 signalling; it cannot push past the
    # ceiling, so a fully costimulated clone gains nothing from it.
    p_full_costim = kill_probability(0.9, 0.9, 1.0, checkpoint_block=0.0)
    p_both = kill_probability(0.9, 0.9, 1.0, checkpoint_block=1.0)
    assert p_full_costim == pytest.approx(p_both)
    # And a half-costimulated clone under full blockade reaches that
    # same ceiling, rather than overshooting it.
    assert kill_probability(0.9, 0.9, 0.5, checkpoint_block=1.0) == pytest.approx(
        p_full_costim)


# ---------------------------------------------------------------------------
# Hill curve tuning
# ---------------------------------------------------------------------------

def test_lower_hill_threshold_increases_kill_probability():
    p_high = kill_probability(0.6, 0.6, 0.6, hill_threshold=0.5)
    p_low = kill_probability(0.6, 0.6, 0.6, hill_threshold=0.1)
    assert p_low > p_high


def test_kill_at_default_threshold_is_half_when_signal_matches():
    # By construction, at signal == threshold and slope >= 1 the Hill
    # curve returns 0.5. Verify by picking inputs whose product equals
    # the default threshold.
    # signal = a*m*c = 0.3 with a=m=c=cbrt(0.3) ~ 0.6694
    base = DEFAULT_HILL_THRESHOLD ** (1.0 / 3.0)
    p = kill_probability(base, base, base, hill_threshold=DEFAULT_HILL_THRESHOLD)
    assert math.isclose(p, 0.5, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# Clamping
# ---------------------------------------------------------------------------

def test_negative_inputs_clamp_to_zero():
    assert kill_probability(-1.0, 1.0, 1.0) == 0.0
    assert kill_probability(0.5, -0.2, 1.0) == 0.0
    assert kill_probability(0.5, 0.5, -3.0, checkpoint_block=0.0) == 0.0


def test_above_one_inputs_clamp_to_one():
    p_above = kill_probability(5.0, 5.0, 5.0)
    p_one = kill_probability(1.0, 1.0, 1.0)
    assert math.isclose(p_above, p_one, abs_tol=1e-9)


def test_nan_inputs_treated_as_zero():
    assert kill_probability(float("nan"), 1.0, 1.0) == 0.0


# ---------------------------------------------------------------------------
# Outcome decomposition
# ---------------------------------------------------------------------------

def test_kill_outcome_exposes_components():
    out = kill_outcome(0.8, 0.7, 0.6, checkpoint_block=0.2)
    assert isinstance(out, KillOutcome)
    assert out.affinity == 0.8
    assert out.mhc_level == 0.7
    assert out.costimulation == 0.6
    assert out.checkpoint_block == 0.2
    # Blockade relieves PD-1 suppression of CD28, so it scales the
    # costimulation arm rather than adding a floor to the signal.
    expected_costim = min(1.0, 0.6 * (1.0 + PD1_CD28_RELIEF * 0.2))
    expected_signal = 0.8 * 0.7 * expected_costim
    assert math.isclose(out.effective_costimulation, expected_costim, abs_tol=1e-9)
    assert math.isclose(out.combined_signal, expected_signal, abs_tol=1e-9)
    assert 0.0 <= out.kill_probability <= 1.0
