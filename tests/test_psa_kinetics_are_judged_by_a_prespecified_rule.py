"""The serum PSA outcome test has its thresholds fixed in code, and the
readouts it is computed from are checked on curves with known answers.

The thresholds live in ``cognisom.validation.psa_kinetics`` and were
written before the runs they judge. These tests do two things: pin the
thresholds to the clinical record they came from, so moving one is a
visible diff, and check each readout against a synthetic curve whose
answer is known in closed form -- a test of a model is only as good as
the arithmetic that scores it.
"""

from __future__ import annotations

import math

import pytest

from cognisom.validation import psa_kinetics as pk

WEEK = 7.0 * 24.0


def _exp_fall_then_rise(weeks=26, fall_half_weeks=2.0, floor=0.1,
                        rise_from_week=None, rise_double_weeks=8.0):
    t = [h for h in range(0, int(weeks * WEEK) + 1, 12)]
    s = []
    for h in t:
        w = h / WEEK
        v = floor + (1.0 - floor) * 2.0 ** (-w / fall_half_weeks)
        if rise_from_week is not None and w > rise_from_week:
            v *= 2.0 ** ((w - rise_from_week) / rise_double_weeks)
        s.append(v)
    return t, s


def test_the_primary_threshold_is_the_trials_own():
    assert pk.TOMBAL_DECLINE == 0.80
    assert pk.TOMBAL_WEEK == 25
    assert "24739897" in pk.TOMBAL_SOURCE
    assert "28867562" in pk.TOMBAL_DURABILITY_SOURCE
    assert pk.PROGRESSION_RISE == 0.25


def test_the_crpc_numbers_are_context_and_not_scored():
    """They describe castration-resistant men; the engine's tumour is
    hormone-sensitive. Scoring against them would be a category error,
    so nothing in the verdict may read them."""
    import inspect
    src = inspect.getsource(pk.primary_verdict)
    assert "CRPC_CONTEXT" not in src
    for pmid in ("22894553", "24881730", "28188090"):
        assert any(pmid in v for v in pk.CRPC_CONTEXT.values()), pmid


def test_decline_at_a_week_reads_the_curve_there():
    t, s = _exp_fall_then_rise()
    # floor 0.1 with a 2-week half-life: by week 25 the transient is gone
    assert pk.decline_at(t, s, 25) == pytest.approx(0.9, abs=1e-3)
    assert pk.decline_at(t, s, 2) == pytest.approx(1 - 0.55, abs=1e-3)


def test_a_short_run_is_not_judged_at_week_25():
    t, s = _exp_fall_then_rise(weeks=12)
    assert pk.decline_at(t, s, 25) is None
    summary = pk.summarise("x", [(t, s)] * 3)
    assert pk.primary_verdict(summary)["verdict"] == "NOT JUDGED"


def test_too_few_seeds_are_not_judged():
    t, s = _exp_fall_then_rise()
    assert pk.primary_verdict(pk.summarise("x", [(t, s)] * 2))[
        "verdict"] == "NOT JUDGED"


def test_a_deep_durable_fall_passes():
    t, s = _exp_fall_then_rise(floor=0.1)
    v = pk.primary_verdict(pk.summarise("x", [(t, s)] * 3))
    assert v["verdict"] == "PASS", v


def test_a_shallow_fall_fails_on_depth_alone():
    """NON-VACUITY: the rule has to be able to say no. A 60% fall that
    never rebounds fails P1 and passes P2."""
    t, s = _exp_fall_then_rise(floor=0.4)
    v = pk.primary_verdict(pk.summarise("x", [(t, s)] * 3))
    assert v["verdict"] == "FAIL"
    assert v["P1_decline_at_week_25"]["passed"] is False
    assert v["P2_no_progression_by_week_25"]["passed"] is True


def test_a_deep_fall_that_rebounds_fails_on_durability():
    # A fast fall, so the curve is flat on its floor before the rise
    # begins and the turning point is where the rise was put.
    t, s = _exp_fall_then_rise(floor=0.02, fall_half_weeks=0.4,
                               rise_from_week=10, rise_double_weeks=6.0)
    summary = pk.summarise("x", [(t, s)] * 3)
    assert summary.progression_week[0] == pytest.approx(
        10 + 6.0 * math.log2(1.25), abs=0.6)
    v = pk.primary_verdict(summary)
    assert v["P2_no_progression_by_week_25"]["passed"] is False
    assert v["verdict"] == "FAIL"


def test_progression_is_censored_not_denied_when_it_never_comes():
    t, s = _exp_fall_then_rise()
    assert pk.weeks_to_progression(t, s) is None


def test_the_doubling_time_of_a_known_exponential():
    t = list(range(0, 2017, 12))
    s = [2.0 ** ((h / 24.0) / 76.0) for h in t]
    assert pk.doubling_time_days(t, s) == pytest.approx(76.0, rel=1e-6)
    falling = [1.0 / v for v in s]
    assert pk.doubling_time_days(t, falling) is None


def test_nadir_is_found_where_the_curve_turns():
    t, s = _exp_fall_then_rise(floor=0.02, fall_half_weeks=0.4,
                               rise_from_week=10)
    week, value = pk.nadir(t, s)
    assert week == pytest.approx(10.0, abs=0.2)
    assert value < 0.1


# ── the recorded result, re-scored from the stored series ──────────


def test_the_carrier_free_arm_passes_and_by_how_much():
    v = pk.stored_verdict("naive")
    assert v["verdict"] == "PASS", v
    p1 = v["P1_decline_at_week_25"]
    assert len(p1["per_seed"]) == 5
    assert p1["median"] == pytest.approx(0.956, abs=2e-3)
    assert v["P2_no_progression_by_week_25"]["seeds_progressed"] == 0


def test_the_pass_rests_mostly_on_an_estimated_number():
    """The claim the result must be reported WITH. The week-25 reading
    is (transcription the drug leaves) x (cells left): 0.10 x 656/1499.
    The 0.10 is an ESTIMATE in the drug mapping, and on its own it
    clears the 80% bar. If this stops holding, the description of what
    the pass is worth has to be rewritten, not just this number."""
    import json
    from pathlib import Path

    data = json.loads((Path(pk.__file__).parent / pk.RESULTS_FILE).read_text())
    run = data["arms"]["naive"]["seeds"]["1"]
    cells_left = run["n_cancer"][-1] / run["n_cancer"][0]
    serum_left = run["serum_psa_relative"][-1]
    block_left = 0.10
    assert serum_left == pytest.approx(block_left * cells_left, rel=0.02)
    assert 1.0 - block_left >= pk.TOMBAL_DECLINE, (
        "the estimated block alone clears the bar; say so")
    from cognisom.workflow.treatment import resolve_treatment
    block = resolve_treatment(("Enzalutamide",)).module_configs[
        "ar_signaling"]["ar_transcriptional_block"]
    assert 1.0 - float(block) == pytest.approx(block_left)


def test_the_ten_percent_carrier_arm_fails_both_criteria():
    v = pk.stored_verdict("carrier")
    assert v["verdict"] == "FAIL", v
    assert v["P1_decline_at_week_25"]["passed"] is False
    assert v["P1_decline_at_week_25"]["median"] < 0, (
        "serum PSA is ABOVE baseline at week 25 in this arm")
    assert v["P2_no_progression_by_week_25"]["seeds_progressed"] == 5
    s = pk.summarise("carrier", pk.stored_runs("carrier"))
    assert 0.55 < min(s.best) and max(s.best) < 0.62
    assert all(7.5 < w < 9.5 for w in s.progression_week)


def test_the_untreated_arm_is_too_short_to_be_judged_and_says_so():
    s = pk.summarise("none", pk.stored_runs("none"))
    assert pk.primary_verdict(s)["verdict"] == "NOT JUDGED"
    # recorded, not scored: no sourced clinical doubling time was found
    assert all(75 < d < 90 for d in s.doubling_days)
