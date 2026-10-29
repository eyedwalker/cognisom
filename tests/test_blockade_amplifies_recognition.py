"""Checkpoint blockade amplifies recognition; it does not replace it.

The ICB-rescue term used to be ``(1 - costimulation) * block * 0.5``,
added to the signal. Three things followed, each pinned against here:

* blockade paid out where there was nothing to recognise (p = 0.165 at
  MHC-I 0.0), so therapy rewarded MHC-I loss -- the escape penalty went
  from 0.022 to 0.558 of an intact cell under full blockade;
* it was largest at ZERO costimulation, so suppressing CD28 could RAISE
  the kill signal, against Kamphorst 2017's "rescue ... is
  CD28-dependent";
* and it never read the target's PD-L1 -- which, per Klement 2023, is
  the one thing it was right about, so that is pinned too.

Everything here is pure arithmetic on tcell_kill, so unlike the
engine-level pins these numbers do not depend on which MHC-I predictor
is installed.
"""
import itertools

import pytest

from cognisom.engine.py.immune.tcell_kill import (
    DEFAULT_HILL_SLOPE,
    DEFAULT_HILL_THRESHOLD,
    PD1_CD28_RELIEF,
    kill_outcome,
    kill_probability,
)

# The shipped operating point: TCR affinity from the repertoire matcher
# and ImmuneModule's default costimulation.
A, C = 0.95, 0.6


def _hill(signal):
    """The Hill curve alone, so a test can state the expected signal."""
    if signal <= 0:
        return 0.0
    n = signal ** DEFAULT_HILL_SLOPE
    return n / (n + DEFAULT_HILL_THRESHOLD ** DEFAULT_HILL_SLOPE)


# ---------------------------------------------------------------------------
# What the additive term got wrong
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("affinity", [0.0, 0.5, 0.95, 1.0])
@pytest.mark.parametrize("costim", [0.0, 0.3, 0.6, 1.0])
def test_blockade_cannot_kill_what_cannot_be_seen(affinity, costim):
    """MHC-I 0 means no signal at ANY blockade. The old rule gave 0.165.

    This is the defect that mattered: the rescue term carried neither
    affinity nor MHC-I, so full blockade handed a kill probability to a
    cell displaying nothing. Only ImmuneModule._recognize's separate
    0.05 gate kept it off the engine's books.
    """
    for block in (0.0, 0.5, 1.0):
        assert kill_probability(affinity, 0.0, costim, block) == 0.0


def test_blockade_does_not_rescue_a_cd28_null_t_cell():
    """Kamphorst 2017 (PMID 28280249): rescue is CD28-dependent.

    The old rule was at its MAXIMUM here -- costimulation 0 made the
    rescue term (1-0)*block*0.5 = 0.5, which through the Hill curve is
    p = 0.885 for a T cell with no costimulation at all.
    """
    assert kill_probability(A, 1.0, 0.0, 1.0) == 0.0
    assert kill_probability(A, 1.0, 0.0, 0.5) == 0.0


@pytest.mark.parametrize("block", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_kill_signal_never_falls_as_costimulation_rises(block):
    """Monotone in costimulation, which the additive term was not.

    Past checkpoint_block 0.551 the old rule's signal violated this:
    it rose as costimulation fell, so a Treg acting on costimulation
    projected -27.4% in Baseline but +29.5% under blockade.
    """
    ps = [kill_probability(A, 1.0, c / 20.0, block) for c in range(21)]
    assert ps == sorted(ps), ps


def test_therapy_does_not_reward_mhc1_loss():
    """The escape penalty must stay a penalty under full blockade.

    OLD: 0.022 without therapy, 0.558 with -- blockade helped the
    escaped cell 26.7x against the intact cell's 1.1x.
    """
    untreated = kill_probability(A, 0.20, C, 0.0) / kill_probability(A, 1.0, C, 0.0)
    treated = kill_probability(A, 0.20, C, 1.0) / kill_probability(A, 1.0, C, 1.0)
    assert untreated == pytest.approx(0.0220, abs=5e-4)
    assert treated == pytest.approx(0.1400, abs=5e-4)
    # The loosened-but-still-a-penalty property, stated independently of
    # the exact constant above.
    assert treated < 0.5
    assert treated > untreated   # blockade DOES still help a dim cell some


# ---------------------------------------------------------------------------
# What must not change
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("affinity", [0.0, 0.1, 0.5, 0.95, 1.0])
@pytest.mark.parametrize("mhc", [0.0, 0.05, 0.2, 0.5, 1.0])
@pytest.mark.parametrize("costim", [0.0, 0.2, 0.6, 1.0])
def test_untreated_killing_is_bit_identical_to_the_old_rule(affinity, mhc, costim):
    """checkpoint_block = 0 must be exactly affinity * mhc * costim.

    Every shipped preset but Immunotherapy leaves checkpoint_block at
    its 0.0 default, so this is what makes the fix attributable: if a
    Baseline number moves, it was not this change.
    """
    assert kill_probability(affinity, mhc, costim, 0.0) == pytest.approx(
        _hill(affinity * mhc * costim), rel=0, abs=0)


def test_the_t_cell_path_ignores_the_targets_pdl1():
    """Klement 2023 (PMID 36917954): tumour PD-L1 does not shield a cell
    from a CTL.

    They deleted Cd274 in 4T1 and CT26: "tPD-L1 expression conferred no
    protection against ... cytotoxicity ... at early or late time
    points", and it "was not rescuable by pretreatment with IFNgamma to
    elevate surface PD-L1 expression" -- which is precisely the adaptive
    induction path ImmuneModule applies (+0.5 per recognition). So
    kill_probability takes no PD-L1 argument at all, and this test
    exists to keep it that way: the obvious "improvement" of threading
    the target's PD-L1 into the T-cell rule is refuted, and tumour PD-L1
    belongs on the MACROPHAGE path, where it already is.
    """
    import inspect
    params = inspect.signature(kill_probability).parameters
    assert not any('pdl1' in p or 'pd_l1' in p for p in params), sorted(params)


def test_exhausted_clones_get_no_relief():
    """Dolina 2021 and Kamphorst 2017 agree from two directions."""
    for block in (0.0, 0.5, 1.0):
        out = kill_outcome(A, 1.0, C, block, is_exhausted=True)
        assert out.effective_costimulation == pytest.approx(C)
        assert out.combined_signal == pytest.approx(A * 1.0 * C)


# ---------------------------------------------------------------------------
# The two entry points must not drift
# ---------------------------------------------------------------------------

def test_outcome_and_probability_cannot_disagree():
    """They each used to carry their own copy of the signal arithmetic."""
    for a, m, c, b, ex in itertools.product(
            (0.0, 0.4, 0.95), (0.0, 0.2, 1.0), (0.0, 0.6, 1.0),
            (0.0, 0.5, 1.0), (False, True)):
        out = kill_outcome(a, m, c, b, is_exhausted=ex)
        assert out.kill_probability == kill_probability(
            a, m, c, b, is_exhausted=ex)
        assert out.combined_signal == pytest.approx(
            a * m * out.effective_costimulation)


def test_relief_saturates_at_full_costimulation():
    """c_eff cannot exceed 1.0: nothing signals above full costimulation.

    At the shipped costimulation of 0.6 and PD1_CD28_RELIEF = 1.0 the
    cap binds, so the preset does not depend on the constant's precise
    value above 1/0.6 - 1 = 0.667. That is deliberate -- no paper prints
    the fold-change this scalar would come from.
    """
    assert PD1_CD28_RELIEF >= 0.667
    out = kill_outcome(A, 1.0, C, 1.0)
    assert out.effective_costimulation == pytest.approx(1.0)
    # And a fully costimulated cell is unaffected by blockade, because
    # there is nothing left to restore.
    assert (kill_probability(A, 1.0, 1.0, 0.0)
            == pytest.approx(kill_probability(A, 1.0, 1.0, 1.0)))
