"""Which neoantigen a T cell is specific for, and why it did not matter.

MEASURED FIRST, 2026-11-27. The shipped TCR gate embeds both the
receptor and the pMHC as SHA-256-seeded random unit vectors in 16
dimensions, so the best match over a repertoire is an order statistic
of noise whose size depends on the CLONE COUNT and on nothing else. At
the shipped 1,000 clones it recognises everything: all 50 of the demo
patient's predicted neoantigens, 200 peptides drawn at random, a
non-mutant self peptide, and a peptide on an invented HLA allele. The
predicted binding affinity never enters -- the same peptide scores
identically at 1 nM and at 50,000 nM.

WHY THAT MATTERS BEYOND TIDINESS. It means the engine could not be
asked which neoantigen is worth targeting. MHCflurry's per-peptide
predictions reach the displayed pool and are discarded at the
recognition step, so two runs targeting different peptides were the
same run. These tests pin the measurement, so nobody reads the
closed-loop pipeline as a target-ranking capability, and they hold the
cognate gate that makes a target comparison possible.
"""

from __future__ import annotations

import random

import pytest

from cognisom.engine.py.immune import mhc_loading as ml
from cognisom.engine.py.immune.mhc_loading import MHCPresentation
from cognisom.engine.py.immune.tcr_repertoire import (
    BACKGROUND_CEILING_FRACTION,
    COGNATE_AFFINITY,
    SPECIFICITY_COGNATE,
    SPECIFICITY_HASH,
    TCRRepertoire,
)

AA = "ACDEFGHIKLMNPQRSTVWY"


def _pres(seq, ic50=50.0, score=0.9, allele="HLA-A*02:01", mutant=True):
    pep = ml.Peptide(
        sequence=seq, source_gene="X", length=len(seq), is_mutant=mutant,
        wild_type_sequence=seq, mutation_label="M1X" if mutant else None,
        anchor_position_in_peptide=2, parent_position_1based=0,
        cleavage_score=0.6)
    return MHCPresentation(peptide=pep, hla_allele=allele, ic50_nm=ic50,
                           binding_level="weak", presentation_score=score)


def _random_peptides(n, seed=5, length=9):
    rng = random.Random(seed)
    return ["".join(rng.choice(AA) for _ in range(length)) for _ in range(n)]


# ── the measurement, pinned ────────────────────────────────────────


def test_the_hash_gate_recognises_peptides_drawn_at_random():
    """The defect, stated as a claim and pinned beside its number."""
    rep = TCRRepertoire(size=1000, recognition_threshold=0.7, seed=0)
    affs = [rep.best_match(_pres(p)).affinity
            for p in _random_peptides(200)]
    assert all(a >= 0.7 for a in affs), (
        "the gate has become selective; re-measure this file's claims")
    assert min(affs) == pytest.approx(0.917, abs=0.01)
    assert max(affs) == pytest.approx(0.969, abs=0.01)


def test_the_hash_gates_recognition_tracks_clone_count_not_the_peptide():
    """Whether a peptide is 'seen' is an order statistic of noise."""
    peptides = _random_peptides(100, seed=11)
    seen = {}
    for size in (10, 100, 1000):
        rep = TCRRepertoire(size=size, recognition_threshold=0.7, seed=0)
        seen[size] = sum(rep.best_match(_pres(p)).is_recognized
                         for p in peptides)
    assert seen[10] < seen[100] <= seen[1000] == 100, seen


def test_the_hash_gate_ignores_the_predicted_binding_it_was_given():
    rep = TCRRepertoire(size=1000, recognition_threshold=0.7, seed=0)
    base = rep.best_match(_pres("SIINFEKLY")).affinity
    for ic50 in (1.0, 500.0, 50_000.0):
        assert rep.best_match(_pres("SIINFEKLY", ic50=ic50)
                              ).affinity == base
    for score in (0.0, 0.5, 0.99):
        assert rep.best_match(_pres("SIINFEKLY", score=score)
                              ).affinity == base
    # and a peptide that is not mutant at all is recognised
    assert rep.best_match(_pres("SIINFEKLY", mutant=False)).is_recognized


def test_every_one_of_this_patients_neoantigens_is_recognised():
    """So the displayed pool cannot rank its own candidates."""
    pytest.importorskip("mhcflurry")
    from cognisom.genomics.demo_patients import load_demo_patient

    neos = load_demo_patient().predicted_neoantigens
    assert len(neos) >= 20, len(neos)
    rep = TCRRepertoire(size=1000, recognition_threshold=0.7, seed=0)
    recognised = sum(
        rep.best_match(_pres(n.peptide, ic50=n.binding_affinity_nm,
                             score=n.presentation_score,
                             allele=n.best_hla_allele)).is_recognized
        for n in neos)
    assert recognised == len(neos), (recognised, len(neos))


# ── the cognate gate ───────────────────────────────────────────────


def test_a_cognate_repertoire_recognises_nothing_until_a_target_is_named():
    rep = TCRRepertoire(size=1000, recognition_threshold=0.7, seed=0,
                        specificity=SPECIFICITY_COGNATE)
    assert not any(rep.best_match(_pres(p)).is_recognized
                   for p in _random_peptides(200))
    assert rep.cognate_targets() == {}


def test_naming_a_target_makes_that_peptide_and_only_it_recognised():
    rep = TCRRepertoire(size=1000, recognition_threshold=0.7, seed=0,
                        specificity=SPECIFICITY_COGNATE)
    peptides = _random_peptides(50, seed=3)
    target = peptides[7]
    ids = rep.assign_cognate(target)
    assert len(ids) == 1
    match = rep.best_match(_pres(target))
    assert match.is_recognized
    assert match.affinity == COGNATE_AFFINITY
    assert match.tcr.tcr_id == ids[0]
    others = [p for p in peptides if p != target]
    assert not any(rep.best_match(_pres(p)).is_recognized for p in others)
    assert rep.cognate_targets() == {target: ids}


def test_the_background_cannot_reach_the_threshold_at_any_size():
    """What the hash gate lacked: a ceiling that holds as the
    repertoire grows, so 'not recognised' stays a real answer."""
    for size in (10, 1000, 20_000):
        for threshold in (0.5, 0.7, 0.9):
            rep = TCRRepertoire(size=size, recognition_threshold=threshold,
                                seed=0, specificity=SPECIFICITY_COGNATE)
            affs = [rep.best_match(_pres(p)).affinity
                    for p in _random_peptides(40, seed=size)]
            assert max(affs) <= BACKGROUND_CEILING_FRACTION * threshold
            assert max(affs) < threshold


def test_two_targets_differ_only_in_the_target():
    """The property a target comparison rests on: the assignment is
    deterministic from the sequence, and nothing else moves."""
    peptides = _random_peptides(2, seed=21)
    reps = []
    for target in peptides:
        rep = TCRRepertoire(size=1000, recognition_threshold=0.7, seed=0,
                            specificity=SPECIFICITY_COGNATE)
        rep.assign_cognate(target)
        reps.append(rep)
    a, b = reps
    assert [t.cdr3 for t in a.tcrs] == [t.cdr3 for t in b.tcrs], (
        "the repertoires themselves must be identical")
    assert a.cognate_targets() != b.cognate_targets()
    assert a.best_match(_pres(peptides[0])).is_recognized
    assert not a.best_match(_pres(peptides[1])).is_recognized
    assert b.best_match(_pres(peptides[1])).is_recognized
    assert not b.best_match(_pres(peptides[0])).is_recognized
    # repeating the same assignment is stable, not cumulative
    again = TCRRepertoire(size=1000, recognition_threshold=0.7, seed=0,
                          specificity=SPECIFICITY_COGNATE)
    again.assign_cognate(peptides[0])
    again.assign_cognate(peptides[0])
    assert again.cognate_targets() == a.cognate_targets()


def test_several_clones_can_carry_one_target():
    rep = TCRRepertoire(size=1000, recognition_threshold=0.7, seed=0,
                        specificity=SPECIFICITY_COGNATE)
    ids = rep.assign_cognate("SIINFEKLY", n_clones=5)
    assert len(set(ids)) == 5
    assert rep.best_match(_pres("SIINFEKLY")).tcr.tcr_id in ids


def test_a_targeting_experiment_that_would_change_nothing_is_refused():
    """Under the hash gate every peptide is recognised already, so an
    assignment there would read as targeting and do nothing."""
    rep = TCRRepertoire(size=100, seed=0, specificity=SPECIFICITY_HASH)
    with pytest.raises(ValueError, match="cognate"):
        rep.assign_cognate("SIINFEKLY")
    with pytest.raises(ValueError):
        TCRRepertoire(size=10, specificity="whatever")


def test_the_default_is_still_the_hash_gate():
    """Every pinned immune result was measured under it."""
    assert TCRRepertoire(size=10).specificity == SPECIFICITY_HASH
    from cognisom.modules.immune_module import ImmuneModule
    assert ImmuneModule({}).tcr_specificity == SPECIFICITY_HASH
    assert ImmuneModule({}).tcell_target_peptides == ()


def test_the_module_refuses_a_target_it_could_not_honour():
    from cognisom.modules.immune_module import ImmuneModule
    with pytest.raises(ValueError, match="cognate"):
        ImmuneModule({"tcell_target_peptides": ["SIINFEKLY"]})
    with pytest.raises(ValueError, match="not one string"):
        ImmuneModule({"tcell_target_peptides": "SIINFEKLY",
                      "tcr_specificity": SPECIFICITY_COGNATE})


# ── the pre-specified coverage test, re-scored from stored runs ─────


def test_the_coverage_rule_can_say_no():
    """NON-VACUITY for the verdict below. Hand it arms where the
    subclonal target did BETTER and it must fail."""
    from cognisom.validation import neoantigen_target_choice as ntc

    def arm(name, cancer, carriers, others):
        return ntc.Arm(name, cancer, carriers, others)

    none = arm("none", [10, 10, 10], [3, 3, 3], [7, 7, 7])
    clonal = arm("clonal", [8, 8, 8], [2, 2, 2], [6, 6, 6])
    subclonal = arm("subclonal", [2, 2, 2], [0, 0, 0], [2, 2, 2])
    v = ntc.coverage_verdict(none, clonal, subclonal)
    assert v["verdict"] == "FAIL"
    assert v["C2_clonal_beats_subclonal"]["passed"] is False
    # and too few seeds is not judged
    thin = arm("none", [1], [0], [1])
    assert ntc.coverage_verdict(thin, thin, thin)["verdict"] == "NOT JUDGED"


def test_a_subclonal_target_leaves_non_carriers_behind():
    """C3's arithmetic, without a run: survivors enriched for the cells
    that never displayed the target."""
    from cognisom.validation import neoantigen_target_choice as ntc

    sub = ntc.Arm("subclonal", [4, 5, 4], [0, 1, 0], [4, 4, 4])
    assert sub.median_non_carrier_share() > 1.0 - ntc.SUBCLONAL_FRACTION


def test_the_recorded_coverage_experiment_passes_all_three_criteria():
    from cognisom.validation import neoantigen_target_choice as ntc

    v = ntc.stored_verdict()
    assert v["verdict"] == "PASS", v
    arms = ntc.stored_arms()
    assert all(a.n_seeds == 5 for a in arms.values())
    assert arms["clonal"].median_cancer() < arms["subclonal"].median_cancer()
    assert arms["subclonal"].median_cancer() < arms["none"].median_cancer()


def test_the_subclonal_target_cleared_exactly_its_carriers():
    """The sharpest part of the result: selection by the target, in
    every seed, rather than a weaker version of the same effect."""
    from cognisom.validation import neoantigen_target_choice as ntc

    sub = ntc.stored_arms()["subclonal"]
    assert sub.carriers_end == [0] * 5, sub.carriers_end
    assert sub.median_non_carrier_share() == 1.0
    # NON-VACUITY: untargeted runs kept carriers, so zero is a result
    none = ntc.stored_arms()["none"]
    assert all(c and c > 0 for c in none.carriers_end), none.carriers_end


def test_the_recorded_result_says_it_is_not_external_validation():
    from cognisom.validation import neoantigen_target_choice as ntc

    v = ntc.stored_verdict()
    assert "construction" in str(v["not_external_validation"])
    assert "26940869" in str(v["direction_source"])

