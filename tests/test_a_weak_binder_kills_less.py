"""Predicted binding affinity changes the outcome of targeting a peptide.

Before this, a T cell assigned to a peptide binding at 12 nM and one
assigned to a peptide barely binding at 12,000 nM killed identically:
under cognate specificity the affinity is a fixed value for an assigned
pair, so predicted binding contributed nothing to a simulated target
ranking.

The fix is NOT a graded affinity. Measured 2026-11-27: the kill signal
is affinity x mhc_level x costimulation through a Hill curve with EC50
0.3 and slope 4, so above the 0.7 recognition threshold the curve is
saturated (0.70 -> 0.967, 0.95 -> 0.990) and grading inside the
recognised band changes nothing, while grading below it flips
is_recognized off and makes the peptide invisible rather than weakly
presented. Binding strength acts on occupancy instead: how much of the
cell's class I carries this peptide.
"""

import pytest

from cognisom.engine.py.immune.mhc_loading import (
    OCCUPANCY_REFERENCE_IC50_NM,
    OCCUPANCY_SOURCE,
    OCCUPANCY_STATUS,
    PEPTIDE_CONCENTRATION_NM,
    occupancy_from_ic50,
    occupancy_report,
)
from cognisom.engine.py.immune.tcell_kill import (
    DEFAULT_HILL_SLOPE,
    DEFAULT_HILL_THRESHOLD,
    kill_outcome,
    kill_probability,
)

STRONG_NM = 12.0
REFERENCE_NM = 50.0
WEAK_NM = 500.0
NON_NM = 12000.0


# ── why the affinity route was rejected ───────────────────────────

def test_the_hill_curve_is_saturated_above_the_recognition_threshold():
    """The measurement that ruled out grading the cognate affinity."""
    at_threshold = kill_probability(0.70, 1.0, 1.0, 0.0)
    at_cognate = kill_probability(0.95, 1.0, 1.0, 0.0)
    assert at_threshold == pytest.approx(0.967, abs=0.002)
    assert at_cognate == pytest.approx(0.990, abs=0.002)
    assert at_cognate - at_threshold < 0.05, (
        "if this gap ever widens, grading the affinity inside the "
        "recognised band becomes a real lever and this reasoning needs "
        "revisiting")


def test_the_hill_parameters_are_the_ones_that_reasoning_assumed():
    """Non-vacuity: the claim above depends on these two numbers."""
    assert DEFAULT_HILL_THRESHOLD == 0.3
    assert DEFAULT_HILL_SLOPE == 4.0


# ── the occupancy model ───────────────────────────────────────────

def test_a_strong_binder_is_unchanged():
    """A peptide at the strong-binder cut keeps the old behaviour."""
    assert occupancy_from_ic50(REFERENCE_NM) == pytest.approx(1.0, abs=1e-12)


def test_occupancy_falls_as_binding_weakens():
    values = [occupancy_from_ic50(x)
              for x in (STRONG_NM, REFERENCE_NM, 150.0, WEAK_NM, NON_NM)]
    assert values == sorted(values, reverse=True)
    assert values[0] == values[1] == 1.0, "both saturate, by construction"
    assert values[-1] < 0.02


def test_the_single_site_form_is_what_it_claims():
    """occupancy = theta(ic50) / theta(reference), theta = 1/(1 + ic50/P)."""
    p, ref = PEPTIDE_CONCENTRATION_NM, OCCUPANCY_REFERENCE_IC50_NM
    for ic50 in (75.0, 150.0, 500.0, 5000.0):
        expected = (1.0 / (1.0 + ic50 / p)) / (1.0 / (1.0 + ref / p))
        assert occupancy_from_ic50(ic50) == pytest.approx(expected, rel=1e-12)


def test_a_weak_binder_at_the_non_binder_cut_is_heavily_penalised():
    """500 nM is a tenth of the reference, so roughly a fifth occupied."""
    assert occupancy_from_ic50(WEAK_NM) == pytest.approx(0.1818, abs=0.001)


def test_a_missing_affinity_is_not_penalised():
    """None means "not measured", which must not read as "binds badly"."""
    assert occupancy_from_ic50(None) == 1.0
    assert occupancy_from_ic50(0.0) == 1.0


def test_occupancy_stays_in_range():
    for ic50 in (None, 0.0, 0.001, 1.0, 50.0, 1e6, 1e12):
        assert 0.0 <= occupancy_from_ic50(ic50) <= 1.0


def test_tighter_than_the_reference_cannot_exceed_one():
    """A stated limit: this is blind to differences among strong binders."""
    assert occupancy_from_ic50(1.0) == occupancy_from_ic50(50.0) == 1.0


def test_the_model_declares_itself_an_estimate():
    assert OCCUPANCY_STATUS == "ESTIMATE"
    assert "33038342" in OCCUPANCY_SOURCE, "the direction anchor"
    assert "ESTIMATE" in OCCUPANCY_SOURCE
    r = occupancy_report()
    assert r["status"] == "ESTIMATE"
    assert "saturate" in r["cannot_distinguish"]


# ── it reaches the kill probability ───────────────────────────────

def test_occupancy_moves_the_kill_probability_a_long_way():
    strong = kill_probability(0.95, 1.0, 1.0, 0.0,
                              peptide_occupancy=occupancy_from_ic50(STRONG_NM))
    weak = kill_probability(0.95, 1.0, 1.0, 0.0,
                            peptide_occupancy=occupancy_from_ic50(WEAK_NM))
    non = kill_probability(0.95, 1.0, 1.0, 0.0,
                           peptide_occupancy=occupancy_from_ic50(NON_NM))
    assert strong > weak > non
    assert strong == pytest.approx(0.990, abs=0.002)
    assert weak < 0.15, "a 500 nM cognate target must not kill like a 12 nM one"
    assert non < 0.001


def test_the_default_is_bit_identical_to_having_no_occupancy():
    for a in (0.3, 0.6, 0.95):
        for m in (0.2, 0.7, 1.0):
            assert (kill_probability(a, m, 1.0, 0.0)
                    == kill_probability(a, m, 1.0, 0.0,
                                        peptide_occupancy=1.0))


def test_zero_occupancy_cannot_be_rescued_by_blockade():
    """Same property the checkpoint term has: it multiplies, never adds."""
    assert kill_probability(0.95, 1.0, 1.0, 1.0, peptide_occupancy=0.0) == 0.0


def test_the_decomposition_reports_occupancy_separately_from_mhc():
    """Folding it into mhc_level would mislabel what was measured."""
    out = kill_outcome(affinity=0.95, mhc_level=0.5, costimulation=1.0,
                       peptide_occupancy=0.4)
    assert out.mhc_level == pytest.approx(0.5)
    assert out.peptide_occupancy == pytest.approx(0.4)
    assert out.combined_signal == pytest.approx(0.95 * 0.5 * 1.0 * 0.4)


def test_the_decomposition_defaults_to_one():
    out = kill_outcome(affinity=0.9, mhc_level=1.0, costimulation=1.0)
    assert out.peptide_occupancy == 1.0
    assert out.combined_signal == pytest.approx(0.9)


def test_occupancy_is_clamped_in_the_decomposition():
    out = kill_outcome(affinity=1.0, mhc_level=1.0, costimulation=1.0,
                       peptide_occupancy=5.0)
    assert out.peptide_occupancy == 1.0


# ── and the module keeps it off unless asked ──────────────────────

def test_the_immune_module_leaves_occupancy_off_by_default():
    """93% of displayed peptides are weak binders, so this is not idle."""
    from cognisom.modules.immune_module import ImmuneModule
    assert ImmuneModule({}).peptide_occupancy_enabled is False


def test_the_immune_module_can_be_asked_for_it():
    from cognisom.modules.immune_module import ImmuneModule
    assert ImmuneModule(
        {'peptide_occupancy': True}).peptide_occupancy_enabled is True


# ── the recorded experiment ───────────────────────────────────────

def test_the_recorded_experiment_passes_all_three_criteria():
    from cognisom.validation.affinity_changes_the_outcome import stored_verdict
    v = stored_verdict()
    assert v["verdict"] == "PASS", v
    assert v["A1_tight_beats_loose_with_occupancy"]["passed"]
    assert v["A2_the_peptides_are_otherwise_interchangeable"]["passed"]
    assert v["A3_the_tight_target_is_no_worse_than_no_target"]["passed"]


def test_the_two_peptides_are_identical_seed_for_seed_without_occupancy():
    """A2 passed more cleanly than its own criterion required.

    Equal medians would have satisfied it. The arms are in fact equal
    run for run, which is what makes the A1 difference attributable to
    affinity rather than to which clones the sequence hash picked.
    """
    from cognisom.validation.affinity_changes_the_outcome import stored_arms
    a = stored_arms()
    assert a["tight_off"].cancer_end == a["loose_off"].cancer_end


def test_a_peptide_at_full_occupancy_is_unaffected_by_the_flag():
    """The tight binder sits at occupancy 1.000, so nothing may change."""
    from cognisom.validation.affinity_changes_the_outcome import stored_arms
    a = stored_arms()
    assert a["tight_on"].cancer_end == a["tight_off"].cancer_end


def test_the_loose_target_lands_near_naming_no_target_at_all():
    """The size of the effect, beside the claim. Re-measure, do not re-pin."""
    from cognisom.validation.affinity_changes_the_outcome import stored_arms
    a = stored_arms()
    assert a["loose_on"].median_cancer() == 7.0
    assert a["tight_on"].median_cancer() == 0.0
    assert a["none"].median_cancer() == 9.0


def test_the_recorded_result_says_it_is_not_external_validation():
    from cognisom.validation.affinity_changes_the_outcome import stored_verdict
    v = stored_verdict()
    assert "arithmetic" in v["not_external_validation"]
    assert "ESTIMATE" in v["occupancy_magnitude"]
    assert "33038342" in v["direction_source"]


def test_the_verdict_refuses_to_judge_too_few_seeds():
    """Non-vacuity: the PASS above must depend on having the seeds."""
    from cognisom.validation.affinity_changes_the_outcome import (
        MIN_SEEDS, Arm, affinity_verdict,
    )
    thin = Arm(name="thin", occupancy=True, cancer_end=[0])
    v = affinity_verdict(thin, thin, thin, thin, thin)
    assert v["verdict"] == "NOT JUDGED"
    assert v["required"] == MIN_SEEDS == 5


def test_coverage_really_was_held_constant():
    """The whole design rests on both peptides being on every cell."""
    import json
    from pathlib import Path
    from cognisom.validation import affinity_changes_the_outcome as mod
    data = json.loads(
        (Path(mod.__file__).parent / mod.RESULTS_FILE).read_text())
    for name, records in data["arms"].items():
        for r in records:
            assert r["coverage_tight"] == r["coverage_loose"] == 10, (
                f"{name} seed {r['seed']} did not display both peptides "
                "on all 10 cancer cells")
