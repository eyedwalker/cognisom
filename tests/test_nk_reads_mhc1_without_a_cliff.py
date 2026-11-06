"""The NK arm reads class I as a gradient, not as a cliff at 0.4.

What was wrong
--------------
Without a patient class-I typing the NK arm read MHC-I exactly twice:
once as ``mhc1_expression < 0.4`` -- an unsourced step lifted out of
``_recognize`` -- and then never again, because the per-contact kill
probability was FLAT in MHC-I. So an NK cell killed a target with
intact class I at the same rate as a class-I-null one right up to the
step, and not at all past it.

That threshold sat inside the population it was deciding. Measured on
the shipped 12 h Baseline slab run, cancer MHC-I spanned
0.3367-0.4160 with **80 of 128 cells within 0.05 of the step**, and
which side the population fell on changed with the peptide scorer
(max intact MHC-I 0.3885 under MHCflurry, 0.4093 under the PWM
fallback). A mechanism whose existence depends on which scorer is
installed is a threshold artefact.

The second bug, found while fixing the first
--------------------------------------------
``nk_kill_probability`` was passed as ``max_kill_probability`` to the
TYPED path, where the curve treats it as a ceiling, while the untyped
path used it as a flat rate. One constant, two meanings. The
consequence was backwards: supplying a patient's real HLA typing CUT
NK killing roughly twofold against an otherwise identical run. Adding
a measurement should not cost potency. The ceiling is now its own
constant, ``NK_GRADED_CEILING``.

Measured, 12 h Baseline slab, seeds 11/12/13, NK kills per run::

    untyped, step 0.4            [9, 9, 12]
    untyped, graded (default)    [7, 9,  9]
    typed, ceiling 0.15 (old)    [5, 4,  5]
    typed, ceiling 0.48 (new)    [7, 9,  7]

The numbers below are measurements, not invariants. If one fails,
RE-MEASURE and move the number with its claim -- do not re-pin it to
whatever the code now says.
"""

from __future__ import annotations

import io
from collections import Counter
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import pytest

from cognisom.dashboard.engine_runner import EngineRunner
from cognisom.engine.py.immune.kill_calibration import (
    NK_GRADED_CEILING, NK_KILL_PER_CONTACT,
)
from cognisom.engine.py.immune.nk_receptor import (
    KIR,
    KIR_AVIDITY,
    KIR_COGNATE_LIGANDS,
    POPULATION_PRIOR_KIR_PRESENCE,
    missing_self_recognized,
    population_prior_inhibitory_avidity,
    untyped_missing_self_decision,
)

#: The MHC-I the shipped presets actually occupy, and the burden a
#: default cancer cell carries. Both measured, both quoted in the
#: module docstring above.
SHIPPED_MHC1 = 0.3563
SHIPPED_MUTATIONS = 1


def _p(mhc1, n_mutations=SHIPPED_MUTATIONS, ceiling=None):
    return untyped_missing_self_decision(
        mhc1_expression=mhc1,
        n_mutations=n_mutations,
        max_kill_probability=(NK_GRADED_CEILING.value
                              if ceiling is None else ceiling),
    ).kill_probability


# ── the gradient ──────────────────────────────────────────────────


def test_killing_falls_as_class_i_rises_everywhere():
    """Missing self (Karre 1986): monotone, over the whole range."""
    grid = np.linspace(0.0, 1.0, 101)
    ps = [_p(m) for m in grid]
    assert all(b <= a + 1e-12 for a, b in zip(ps, ps[1:])), \
        "kill probability must never rise with target class I"
    assert ps[0] > ps[-1], "and it must actually vary"


def test_there_is_no_cliff_anywhere_in_the_range():
    """The point of the change: no MHC-I step flips the mechanism.

    The old rule changed the kill rate from its flat value to zero
    across a single 0.01 step at 0.4. Here the largest jump between
    neighbouring 0.01 samples is bounded by the curve's own slope.
    """
    grid = np.linspace(0.0, 1.0, 101)
    jumps = [abs(_p(b) - _p(a)) for a, b in zip(grid, grid[1:])]
    assert max(jumps) < 0.01, f"largest step {max(jumps):.4f} looks like a cliff"


def test_the_old_rule_really_was_a_cliff():
    """Guard on the claim this test file exists to justify."""
    assert missing_self_recognized(0.399) is True
    assert missing_self_recognized(0.401) is False


def test_intact_class_i_is_self_tolerated():
    """A licensed clone facing intact class I does not kill at all.

    Self-tolerance, not merely weak killing -- the distinction the
    receptor model's docstring draws.
    """
    assert _p(1.0, n_mutations=0) == 0.0
    assert _p(1.0, n_mutations=SHIPPED_MUTATIONS) == 0.0


def test_more_neoantigens_make_a_target_more_nk_visible():
    """Activating ligand load raises the class I needed to inhibit.

    NKG2D ligands are induced by the DNA-damage response (Gasser 2005);
    mutation count is the engine's honest stand-in for that drive.
    """
    for mhc1 in (0.0, 0.2, 0.4, 0.6):
        ps = [_p(mhc1, n_mutations=n) for n in (0, 1, 5, 50)]
        assert all(b >= a - 1e-12 for a, b in zip(ps, ps[1:])), (mhc1, ps)


# ── the avidity is derived, not written down ──────────────────────


def test_the_population_prior_avidity_derives_from_its_two_tables():
    """Re-derived here, so the constant cannot drift from its inputs.

    MEASURED 0.8276. If this moves, one of the two source tables moved
    -- re-measure the curve, do not re-pin this number.
    """
    num = den = 0.0
    for kir, freq in POPULATION_PRIOR_KIR_PRESENCE.items():
        avidity = max(KIR_AVIDITY[kir].get(g, 0.0)
                      for g in KIR_COGNATE_LIGANDS[kir])
        num += freq * avidity
        den += freq
    assert population_prior_inhibitory_avidity() == pytest.approx(num / den)
    assert population_prior_inhibitory_avidity() == pytest.approx(0.8276, abs=5e-4)


def test_the_prior_avidity_lies_inside_its_own_avidity_table():
    """A weighted mean of cognate avidities cannot escape their range."""
    cognate = [max(KIR_AVIDITY[k].get(g, 0.0) for g in KIR_COGNATE_LIGANDS[k])
               for k in POPULATION_PRIOR_KIR_PRESENCE]
    a = population_prior_inhibitory_avidity()
    assert min(cognate) <= a <= max(cognate)


def test_every_prior_receptor_has_a_cognate_avidity_to_contribute():
    """A receptor missing from either table would silently skew the mean."""
    for kir in POPULATION_PRIOR_KIR_PRESENCE:
        assert kir in KIR_COGNATE_LIGANDS, kir
        assert max(KIR_AVIDITY[kir].get(g, 0.0)
                   for g in KIR_COGNATE_LIGANDS[kir]) > 0.0, kir


# ── the ceiling is a choice, and it preserves the calibration ─────


def test_the_ceiling_is_declared_a_choice_not_a_measurement():
    """It has no source, and the status field must keep saying so."""
    assert NK_GRADED_CEILING.status == 'calibration-choice'
    assert NK_KILL_PER_CONTACT.status == 'ESTIMATE'
    assert NK_GRADED_CEILING.value != NK_KILL_PER_CONTACT.value, \
        "the ceiling and the flat typical rate are different quantities"


def test_the_ceiling_sits_at_the_top_of_the_sourced_range():
    assert NK_GRADED_CEILING.value == pytest.approx(NK_KILL_PER_CONTACT.high)


def test_the_graded_curve_reproduces_the_flat_rate_where_it_is_used():
    """Why removing the cliff did not re-potentiate the NK arm.

    At the MHC-I the presets occupy, the curve passes through 0.139
    against the flat 0.15 it replaces. MEASURED; if it drifts, the
    shipped NK rate has changed and the kill counts in the docstring
    need re-measuring.
    """
    p = _p(SHIPPED_MHC1)
    assert p == pytest.approx(0.1385, abs=2e-3)
    assert abs(p - NK_KILL_PER_CONTACT.value) < 0.02, (
        f"graded {p:.4f} vs flat {NK_KILL_PER_CONTACT.value}: the curve no "
        "longer lands on the rate the presets were calibrated with")


def test_anchoring_the_ceiling_at_the_typical_rate_would_gut_the_arm():
    """The mistake this constant exists to prevent, kept measurable."""
    assert _p(SHIPPED_MHC1, ceiling=NK_KILL_PER_CONTACT.value) < 0.5 * _p(SHIPPED_MHC1)


# ── the two paths agree ───────────────────────────────────────────


def _nk_kills(seed, patient=None, **overrides):
    np.random.seed(seed)
    kw = dict(dt=0.05, duration=12.0, scenario="Baseline", tissue="slab",
              snapshot_interval_hours=6.0)
    if overrides:
        kw["overrides"] = {"immune": overrides}
    if patient is not None:
        kw["patient"] = patient
    runner = EngineRunner(**kw)
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        runner.build()
        runner.run()
    seen, kills = set(), Counter()
    for ev in runner.recorder.events:
        if ev.kind == "kill" and ev.label not in seen:
            seen.add(ev.label)
            kills[ev.label.split()[0]] += 1
    return kills.get("NK_cell", 0), runner.engine.modules["immune"]


def _nk_module(seed, patient=None, **overrides):
    """The immune module as BUILT, without running the simulation.

    Every assertion below reads a flag that ``build()`` sets. Reaching
    them through ``_nk_kills`` ran 240 steps twice and discarded both
    kill counts: 62 s locally, over the suite's 120 s per-test budget on
    a CI runner, which is what blocked the deploy lane. The assertions
    are unchanged -- only the work nobody read is gone.
    """
    np.random.seed(seed)
    kw = dict(dt=0.05, duration=12.0, scenario="Baseline", tissue="slab",
              snapshot_interval_hours=6.0)
    if overrides:
        kw["overrides"] = {"immune": overrides}
    if patient is not None:
        kw["patient"] = patient
    runner = EngineRunner(**kw)
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        runner.build()
    return runner.engine.modules["immune"]


def test_the_graded_path_is_the_default_and_the_step_is_still_reachable():
    immune = _nk_module(11)
    assert immune.nk_graded_missing_self is True
    assert immune._nk_licensing is None, "no patient: the untyped path"
    assert immune.nk_graded_ceiling == pytest.approx(NK_GRADED_CEILING.value)
    stepped = _nk_module(11, nk_graded_missing_self=False)
    assert stepped.nk_graded_missing_self is False


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_removing_the_cliff_keeps_the_shipped_kill_rate():
    """Measured [9, 9, 12] stepped vs [7, 9, 9] graded, seeds 11/12/13.

    Pinned as a RELATION, because the per-seed counts are a
    measurement: the graded arm must stay within a third of the
    stepped one, which is what "the calibration survived" means.
    """
    stepped = [_nk_kills(s, nk_graded_missing_self=False)[0] for s in (11, 12, 13)]
    graded = [_nk_kills(s)[0] for s in (11, 12, 13)]
    assert sum(stepped) > 0 and sum(graded) > 0
    ratio = sum(graded) / sum(stepped)
    assert 0.66 <= ratio <= 1.5, (
        f"graded {graded} vs stepped {stepped} (ratio {ratio:.2f}): removing "
        "the cliff should not re-potentiate or gut the NK arm. Re-measure "
        "before re-pinning.")
