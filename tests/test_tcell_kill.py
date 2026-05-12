"""
T-cell kill probability unit tests.

Covers engine/py/immune/tcell_kill.py:
  * result is in [0, 1]
  * zero affinity, zero MHC, or zero costim with no checkpoint block
    yields zero kill probability
  * monotone in each of affinity, mhc_level, costimulation
  * checkpoint block provides a partial rescue when costim is low
  * out-of-range inputs are clamped (no exceptions, no extrapolation)
  * Hill threshold lowers the EC50 (more permissive kills)
  * kill_outcome carries the full decomposition
"""
from __future__ import annotations

import math

import pytest

from engine.py.immune.tcell_kill import (
    DEFAULT_HILL_THRESHOLD,
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

def test_checkpoint_block_rescues_low_costim_state():
    # Affinity and MHC are high but costim is zero; without checkpoint
    # block, probability is 0. Adding checkpoint block should produce
    # a non-zero probability.
    p_no_block = kill_probability(0.9, 0.9, 0.0, checkpoint_block=0.0)
    p_with_block = kill_probability(0.9, 0.9, 0.0, checkpoint_block=1.0)
    assert p_no_block == 0.0
    assert p_with_block > 0.0


def test_checkpoint_block_below_full_costim_pathway():
    # Even at full checkpoint block, the rescue term is half-strength,
    # so it cannot exceed the regular pathway when costim is also full.
    p_full_costim = kill_probability(0.9, 0.9, 1.0, checkpoint_block=0.0)
    p_full_block = kill_probability(0.9, 0.9, 0.0, checkpoint_block=1.0)
    assert p_full_costim > p_full_block


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
    expected_signal = 0.8 * 0.7 * 0.6 + (1.0 - 0.6) * 0.2 * 0.5
    assert math.isclose(out.combined_signal, expected_signal, abs_tol=1e-9)
    assert 0.0 <= out.kill_probability <= 1.0
