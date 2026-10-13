"""The treatment simulator must predict honestly, or say that it does not.

Five defects this file pins down, each of which had shipped:

1. PROGRESSIVE DISEASE WAS UNREACHABLE. `best_response` is the nadir of a
   curve that starts at 1.0, so it can never exceed 1.0, and the old
   classifier's PD branch tested `best_response > 1.20`. A tumour 59 times
   its baseline volume at two years was reported as stable disease, and the
   PD indicator on three dashboard pages could not render.

2. RECIST THRESHOLDS WERE APPLIED TO THE WRONG QUANTITY. The curve is a
   relative volume; RECIST 1.1 is defined on the sum of longest diameters.
   V proportional to D**3, so a volume ratio of 0.70 is an 11.2% diameter
   shrink -- well inside stable disease -- and it was reported as a partial
   response.

3. PFS WAS RIGHT-CENSORED AND PRINTED AS A CLINICAL MONTHS FIGURE. On a
   180-day run it took two values, 1 or 180, and `agents.py` rendered
   "PFS: 180 days (6.0 months)" for every arm -- administrative end of
   follow-up presented to a clinician as a six-month PFS. The ClinicalAgent
   scored `min(0.25, pfs/720)`, which 180/720 saturates, so the term was a
   constant added to every arm.

4. THE NOISE SEED DIGESTED THE DRUG NAME ALONE, so every patient drew the
   same stochastic realisation.

5. THE EFFECTIVENESS MODEL DISPATCHED ON SUBSTRINGS OF A DISPLAY STRING.
   `"AR" in "PARP inhibitor"` is True -- "PARP" contains "AR" -- so both
   PARP arms took the androgen-receptor branch and olaparib was predicted
   to produce a partial response in a patient with no DNA-repair defect.
   `pembro_ipi_combo`, class "combination checkpoint", matched neither
   "PD-1" nor "CTLA-4" and so ignored every biomarker.
"""

import math

import pytest

from cognisom.genomics.patient_profile import PatientProfile
from cognisom.genomics.treatment_simulator import (
    CALIBRATED_GROWTH_RATE_PER_DAY,
    CALIBRATION_STATEMENT,
    DAYS_PER_MONTH,
    DERIVED_PARAMETER_COUNT,
    ILLUSTRATIVE_PARAMETER_COUNT,
    KNOWN_MECHANISMS,
    MECH_AR_ANTAGONIST,
    MECH_CTLA4,
    MECH_PARP,
    MECH_PD1,
    MODEL_CAVEATS,
    PARAMETER_PROVENANCE,
    PROFOUND_COHORT_A_CONTROL_RPFS_MONTHS,
    RECIST_PD_DIAMETER_RATIO,
    RECIST_PR_DIAMETER_RATIO,
    TREATMENT_PROFILES,
    TreatmentSimulator,
    format_progression_free_survival,
    treatment_mechanisms,
)
from cognisom.genomics.twin_config import DigitalTwinConfig
from cognisom.mad.agents import ClinicalAgent
from cognisom.mad.explainability import explain_effectiveness


def _twin(**kw):
    base = dict(
        patient=PatientProfile(patient_id="PT-TEST"),
        tumor_mutational_burden=5.0,
        microsatellite_instability="MSS",
        immune_profile_measured=True,
        immune_score="suppressed",
        t_cell_exhaustion_fraction=0.4,
        mean_exhaustion=0.5,
        treg_fraction=0.2,
        immune_cell_fraction=0.2,
        pd_l1_expression=0.5,
        mhc1_downregulation=0.1,
        vaccine_candidate_count=8,
        strong_binder_count=6,
        neoantigen_vaccine_candidate=True,
    )
    base.update(kw)
    return DigitalTwinConfig(**base)


# ── (1) progressive disease is reachable ─────────────────────────────

def test_a_tumour_that_only_grows_is_reported_as_progressive_disease():
    sim = TreatmentSimulator()
    curve = sim._simulate_tumor_dynamics(
        0.0, TREATMENT_PROFILES["pembrolizumab"], _twin(), 180)

    # It really does grow, far past the progression threshold.
    assert curve[-1] > RECIST_PD_DIAMETER_RATIO ** 3

    progression_day = sim._first_progression_day(curve)
    assert progression_day is not None
    assert sim._classify_response(min(curve), True) == "PD"


@pytest.mark.parametrize("duration", [180, 365, 730])
def test_unchecked_growth_is_never_called_stable_disease(duration):
    """At 730 days the old classifier called a 59x tumour "SD"."""
    sim = TreatmentSimulator()
    curve = sim._simulate_tumor_dynamics(
        0.0, TREATMENT_PROFILES["pembrolizumab"], _twin(), duration)
    progressed = sim._first_progression_day(curve) is not None
    assert sim._classify_response(min(curve), progressed) == "PD"


def test_all_three_reachable_categories_are_reachable():
    """PD, SD and PR must each occur somewhere in the effectiveness range."""
    sim = TreatmentSimulator()
    twin = _twin()
    seen = set()
    for i in range(0, 101):
        eff = i / 100.0
        curve = sim._simulate_tumor_dynamics(
            eff, TREATMENT_PROFILES["pembrolizumab"], twin, 180)
        progressed = sim._first_progression_day(curve) is not None
        seen.add(sim._classify_response(min(curve), progressed))
    assert seen == {"PD", "SD", "PR"}


def test_complete_response_is_never_reported():
    """The dynamics floor relative volume at 0.01, so disappearance of all
    target lesions is not representable and a CR would be an extrapolation."""
    sim = TreatmentSimulator()
    twin = _twin()
    for i in range(0, 101):
        curve = sim._simulate_tumor_dynamics(
            i / 100.0, TREATMENT_PROFILES["pembrolizumab"], twin, 365)
        progressed = sim._first_progression_day(curve) is not None
        assert sim._classify_response(min(curve), progressed) != "CR"


# ── (2) RECIST thresholds are diameter thresholds ────────────────────

def test_a_volume_ratio_of_seventy_percent_is_stable_disease_not_a_response():
    """0.70 volume is a 1 - 0.70**(1/3) = 11.2% diameter shrink."""
    diameter_change = 1 - 0.70 ** (1.0 / 3.0)
    assert diameter_change == pytest.approx(0.1121, abs=5e-4)
    assert TreatmentSimulator._classify_response(0.70, False) == "SD"


def test_the_partial_response_threshold_is_a_thirty_percent_diameter_change():
    boundary = RECIST_PR_DIAMETER_RATIO ** 3
    assert boundary == pytest.approx(0.343, abs=1e-9)
    assert TreatmentSimulator._classify_response(boundary - 1e-6, False) == "PR"
    assert TreatmentSimulator._classify_response(boundary + 1e-3, False) == "SD"


def test_progression_is_referenced_to_the_running_nadir_not_to_baseline():
    """RECIST 1.1 measures the increase from the smallest sum on study."""
    sim = TreatmentSimulator()
    # Shrinks to 0.5 volume, then regrows to 0.9 -- below baseline, but
    # 0.9 / 0.5 = 1.8 > 1.20**3 = 1.728 on the volume ratio, so it is a
    # RECIST progression even though it never re-crosses baseline. The old
    # test ("volume > 1.0") would have missed it entirely.
    curve = [1.0, 0.8, 0.5, 0.7, 0.9]
    assert sim._first_progression_day(curve) == 4

    # And a curve that merely exceeds baseline by a little is NOT PD.
    assert sim._first_progression_day([1.0, 1.05, 1.1, 1.15]) is None


def test_a_response_followed_by_progression_keeps_its_best_overall_response():
    """RECIST best overall response: a PR that later progresses is a PR."""
    assert TreatmentSimulator._classify_response(0.2, True) == "PR"
    assert TreatmentSimulator._classify_response(0.9, True) == "PD"


# ── (3) PFS carries its censoring, and is never printed as months ────

def test_a_run_without_progression_reports_a_censored_pfs():
    result = TreatmentSimulator().simulate("pembrolizumab", _twin(), 180)
    days, censored = result.progression_free_survival
    assert censored is True
    assert days == 180


def test_a_censored_pfs_is_never_rendered_as_a_months_figure():
    text = format_progression_free_survival(180, censored=True)
    assert "month" not in text
    assert "censored" in text
    # The exact string the old code printed for every arm of a 180-day run.
    assert "(6.0 months)" not in text


def test_an_uncensored_pfs_is_rendered_in_days_and_months():
    text = format_progression_free_survival(110, censored=False)
    assert "110 days" in text
    assert "3.6 months" in text


def _non_responder_twin(**kw):
    """A twin for whom the effectiveness model returns 0.0 on a vaccine arm:
    cold, no neoantigens, no TMB, PD-L1 absent, exhaustion outside the
    reactivation window, half of MHC-I lost."""
    return _twin(immune_score="cold", tumor_mutational_burden=0.0,
                 vaccine_candidate_count=0, strong_binder_count=0,
                 pd_l1_expression=0.0, t_cell_exhaustion_fraction=0.9,
                 mhc1_downregulation=0.5, **kw)


def test_a_progressing_arm_reports_an_uncensored_pfs_in_real_days():
    """The old PFS took only the values 1 and duration_days."""
    sim = TreatmentSimulator()
    twin = _non_responder_twin()
    assert sim._calculate_effectiveness(
        TREATMENT_PROFILES["neoantigen_vaccine_pembro"], twin) == 0.0
    result = sim.simulate("neoantigen_vaccine_pembro", twin, 180)
    days, censored = result.progression_free_survival
    assert censored is False
    assert 1 < days < 180
    assert result.response_category == "PD"
    assert "month" in result.format_progression_free_survival()


def test_the_clinical_agent_drops_a_censored_pfs_term():
    sim = TreatmentSimulator()
    twin = _twin()
    results = [sim.simulate(k, twin, 180) for k in sorted(TREATMENT_PROFILES)]
    assert all(r.progression_free_censored for r in results)

    opinion = ClinicalAgent().analyze(twin, results)
    terms = [c for r in opinion.treatment_rankings for c in r.contributions
             if c.feature_name == "progression_free_survival"]
    assert terms, "the term must still be reported, as explicitly absent"
    assert all(c.delta == 0.0 for c in terms)
    assert all(c.direction == "neutral" for c in terms)
    # The constant 0.25 that 180/720 used to saturate.
    assert not any(c.delta == pytest.approx(0.25) for c in terms)


def test_no_agent_contribution_prints_a_censored_pfs_as_months():
    sim = TreatmentSimulator()
    twin = _twin()
    results = [sim.simulate(k, twin, 180) for k in sorted(TREATMENT_PROFILES)]
    opinion = ClinicalAgent().analyze(twin, results)
    for ranking in opinion.treatment_rankings:
        for contribution in ranking.contributions:
            if contribution.feature_name != "progression_free_survival":
                continue
            assert "months" not in contribution.description
        for item in ranking.evidence:
            if item.source_type != "simulation":
                continue
            assert item.supporting_data["pfs_censored"] is True
            assert "months" not in item.claim


def test_the_clinical_agent_keeps_an_uncensored_pfs_term():
    sim = TreatmentSimulator()
    twin = _non_responder_twin()
    result = sim.simulate("neoantigen_vaccine_pembro", twin, 180)
    assert result.progression_free_censored is False

    opinion = ClinicalAgent().analyze(twin, [result])
    terms = [c for r in opinion.treatment_rankings for c in r.contributions
             if c.feature_name == "progression_free_survival"]
    assert len(terms) == 1
    # An OBSERVED progression is evidence against the arm, so the term is
    # a penalty. Scoring it positive ranked a day-110 progression (+0.153)
    # above an arm that never progressed (0.0). See the G3 tests below.
    assert terms[0].delta < 0.0
    assert terms[0].direction == "negative"


# ── (4) the seed depends on the patient ──────────────────────────────

def test_two_patients_do_not_share_one_noise_realisation():
    sim = TreatmentSimulator()
    a = sim._simulate_tumor_dynamics(
        0.5, TREATMENT_PROFILES["pembrolizumab"],
        _twin(patient=PatientProfile(patient_id="PT-A")), 60)
    b = sim._simulate_tumor_dynamics(
        0.5, TREATMENT_PROFILES["pembrolizumab"],
        _twin(patient=PatientProfile(patient_id="PT-B")), 60)
    assert a != b


def test_two_twins_with_no_patient_id_still_differ_on_their_biomarkers():
    sim = TreatmentSimulator()
    a = sim._simulate_tumor_dynamics(
        0.5, TREATMENT_PROFILES["pembrolizumab"],
        _twin(patient=None, tumor_mutational_burden=2.0), 60)
    b = sim._simulate_tumor_dynamics(
        0.5, TREATMENT_PROFILES["pembrolizumab"],
        _twin(patient=None, tumor_mutational_burden=30.0), 60)
    assert a != b


def test_one_patient_and_one_regimen_reproduce_exactly():
    sim = TreatmentSimulator()
    args = (0.5, TREATMENT_PROFILES["pembrolizumab"], _twin(), 60)
    assert sim._simulate_tumor_dynamics(*args) == sim._simulate_tumor_dynamics(*args)


# ── (5) mechanism dispatch, not substring matching ───────────────────

def test_the_substring_that_caused_the_collision_really_does_collide():
    """Documents why the dispatch had to change: this is True."""
    assert "AR" in "PARP inhibitor"
    assert "PD-1" not in "combination checkpoint"
    assert "CTLA-4" not in "combination checkpoint"


def test_every_shipped_regimen_declares_a_known_mechanism_set():
    for key, profile in TREATMENT_PROFILES.items():
        mechanisms = treatment_mechanisms(profile)
        assert mechanisms, key
        assert mechanisms <= KNOWN_MECHANISMS, key


def test_a_parp_inhibitor_takes_no_androgen_receptor_term():
    for key in ("olaparib", "olaparib_pembro_combo"):
        mechanisms = treatment_mechanisms(TREATMENT_PROFILES[key])
        assert MECH_PARP in mechanisms, key
        assert MECH_AR_ANTAGONIST not in mechanisms, key


def test_a_parp_inhibitor_is_not_predicted_active_without_a_dna_repair_defect():
    """0.5 - 0.3 (no HRD) + 0.2 ("wild-type AR responds well") = 0.4 used to
    give a 38% volume reduction, reported as a partial response: a PARP
    inhibitor predicted active with no synthetic-lethal target."""
    sim = TreatmentSimulator()
    twin = _twin(has_dna_repair_defect=False, has_ar_mutation=False)
    effectiveness = sim._calculate_effectiveness(
        TREATMENT_PROFILES["olaparib"], twin)
    assert effectiveness == pytest.approx(0.20)

    result = sim.simulate("olaparib", twin, 180)
    assert result.response_category != "PR"


def test_hrd_severity_is_visible_rather_than_clamped_away():
    """The spurious AR bonus pushed olaparib-with-HRD to the 1.0 clamp."""
    sim = TreatmentSimulator()
    effectiveness = sim._calculate_effectiveness(
        TREATMENT_PROFILES["olaparib"], _twin(has_dna_repair_defect=True))
    assert effectiveness < 1.0


def test_the_dual_checkpoint_combination_responds_to_the_patient():
    """`pembro_ipi_combo` took exactly two values ever: 0.5, or 0.7 with
    MSI-H -- invariant to PD-L1, TMB, exhaustion, Tregs and immune score."""
    sim = TreatmentSimulator()
    profile = TREATMENT_PROFILES["pembro_ipi_combo"]
    mechanisms = treatment_mechanisms(profile)
    assert mechanisms == {MECH_PD1, MECH_CTLA4}

    scores = {
        score: sim._calculate_effectiveness(
            profile,
            _twin(immune_score=score, tumor_mutational_burden=tmb,
                  pd_l1_expression=pd_l1),
        )
        for score, tmb, pd_l1 in (
            ("hot", 15.0, 0.9), ("cold", 0.0, 0.0), ("suppressed", 5.0, 0.5),
        )
    }
    assert len(set(scores.values())) == 3, scores
    assert scores["cold"] < scores["suppressed"] < scores["hot"]


def test_a_pd1_containing_combination_receives_the_pd1_terms():
    for key in ("pembro_ipi_combo", "olaparib_pembro_combo",
                "neoantigen_vaccine_pembro"):
        assert MECH_PD1 in treatment_mechanisms(TREATMENT_PROFILES[key]), key


def test_an_unmapped_treatment_class_gets_no_mechanism_rather_than_a_guess():
    assert treatment_mechanisms({"class": "chemotherapy"}) == frozenset()
    assert treatment_mechanisms({"class": "not a real class"}) == frozenset()


def test_a_profile_declaring_an_unknown_mechanism_is_refused():
    with pytest.raises(ValueError, match="unknown mechanism"):
        treatment_mechanisms({"name": "X", "mechanisms": frozenset({"magic"})})


def test_a_result_with_no_mechanism_term_says_it_is_not_patient_specific():
    sim = TreatmentSimulator()
    sim._entity_profiles = {"inert": {
        "name": "Inert Agent", "class": "chemotherapy",
        "mechanism": "none modelled", "exhaustion_reversal": 0.0,
        "irae_base_risk": 0.1, "effect_onset_days": 14,
    }}
    result = sim.simulate("inert", _twin(), 90)
    assert any("does not depend on the patient" in c for c in result.caveats)


# ── the audit decomposition must equal the score it decomposes ───────

def test_the_explainer_agrees_with_the_score_on_every_regimen():
    """The explainer guarded its AR branch with `"PARP" not in cls` and the
    simulator did not, so on both PARP arms the decomposition the model card
    calls "more auditable for FDA" disagreed with the score by 0.20."""
    sim = TreatmentSimulator()
    twins = [
        _twin(has_dna_repair_defect=hrd, has_ar_mutation=ar,
              immune_score=score, microsatellite_instability=msi,
              tumor_mutational_burden=tmb, vaccine_candidate_count=vc)
        for hrd in (False, True)
        for ar in (False, True)
        for score in ("hot", "cold", "suppressed")
        for msi in ("MSS", "MSI-H")
        for tmb in (0.0, 15.0)
        for vc in (0, 12)
    ]
    for twin in twins:
        for key, profile in TREATMENT_PROFILES.items():
            scored = sim._calculate_effectiveness(profile, twin)
            explained = explain_effectiveness(profile, twin).total_effectiveness
            assert scored == pytest.approx(explained, abs=1e-12), key


# ── calibration: one derived parameter, and the rest declared ────────

def test_the_growth_rate_is_derived_from_the_published_control_arm_rpfs():
    """PROfound cohort A control arm, median rPFS 3.6 months by BICR. The
    median patient reaches the RECIST progression threshold exactly then:

        exp(g T) = 1.20 ** 3   =>   g = 3 ln(1.20) / T
    """
    expected = (
        3.0 * math.log(RECIST_PD_DIAMETER_RATIO)
        / (PROFOUND_COHORT_A_CONTROL_RPFS_MONTHS * DAYS_PER_MONTH)
    )
    assert CALIBRATED_GROWTH_RATE_PER_DAY == pytest.approx(expected, rel=1e-12)
    assert CALIBRATED_GROWTH_RATE_PER_DAY == pytest.approx(0.004992, abs=5e-7)

    # Sensitivity to the one input. The abstract's rounded 3.6 months and the
    # unrounded BICR median of 3.55 months differ by 1.4% in g, and the two
    # bracket the 0.005 that this file used to assert without a source. The
    # derivation therefore does not hinge on which of the two is quoted.
    unrounded = 3.0 * math.log(1.20) / (3.55 * DAYS_PER_MONTH)
    assert unrounded == pytest.approx(0.005062, abs=5e-7)
    spread = abs(unrounded - CALIBRATED_GROWTH_RATE_PER_DAY) / unrounded
    assert spread < 0.015
    assert CALIBRATED_GROWTH_RATE_PER_DAY < 0.005 < unrounded


def test_an_untreated_tumour_progresses_at_the_published_median():
    """The calibration is a round trip, so it has to close.

    60 synthetic patients, no therapy. The target is
    3.6 months = 109.58 days. Measured over 200 patients: mean 110.36,
    median 111, sd 4.46, range 97-122 -- the spread is the noise term, and
    the mean sits 0.8 days from the target it was derived from.
    """
    sim = TreatmentSimulator()
    no_therapy = {"name": "none", "effect_onset_days": 10 ** 9}
    days = []
    for i in range(60):
        twin = DigitalTwinConfig(
            patient=PatientProfile(patient_id="CAL-%04d" % i))
        curve = sim._simulate_tumor_dynamics(0.0, no_therapy, twin, 400)
        progression_day = sim._first_progression_day(curve)
        assert progression_day is not None
        days.append(progression_day)

    target = PROFOUND_COHORT_A_CONTROL_RPFS_MONTHS * DAYS_PER_MONTH
    assert target == pytest.approx(109.58, abs=0.01)
    assert sum(days) / len(days) == pytest.approx(target, abs=3.0)
    assert all(abs(d - target) < 25 for d in days)


def test_exactly_one_parameter_claims_to_be_derived():
    assert DERIVED_PARAMETER_COUNT == 1
    derived = [p for p in PARAMETER_PROVENANCE if p.status == "derived"]
    assert [p.name for p in derived] == ["growth_rate_per_day"]
    assert "PROfound" in derived[0].basis


def test_every_provenance_entry_declares_a_recognised_status_and_a_basis():
    for entry in PARAMETER_PROVENANCE:
        assert entry.status in ("derived", "standard", "illustrative"), entry.name
        assert entry.count >= 1, entry.name
        assert len(entry.basis) > 20, entry.name


def test_the_illustrative_constants_are_counted_not_glossed():
    assert ILLUSTRATIVE_PARAMETER_COUNT == 80
    assert str(ILLUSTRATIVE_PARAMETER_COUNT) in CALIBRATION_STATEMENT
    assert "ILLUSTRATIVE" in CALIBRATION_STATEMENT
    assert "DEFERRED" in CALIBRATION_STATEMENT


def test_the_model_card_states_what_is_fitted_instead_of_claiming_it_all_is():
    from cognisom.mad.model_cards import TREATMENT_SIMULATOR_CARD as card
    assert card.training_data_equivalent == CALIBRATION_STATEMENT
    # The claim that used to stand here.
    assert "Parameters derived from published clinical trial endpoints" \
        not in card.training_data_equivalent


def test_the_model_card_no_longer_claims_results_vary_with_the_seed():
    from cognisom.mad.model_cards import TREATMENT_SIMULATOR_CARD as card
    limitations = " ".join(card.known_limitations)
    assert "results vary with random seed" not in limitations
    assert "Deterministic per (patient, regimen)" in limitations


def test_the_board_opinion_carries_the_calibration_statement():
    sim = TreatmentSimulator()
    twin = _twin()
    results = [sim.simulate("pembrolizumab", twin, 180)]
    opinion = ClinicalAgent().analyze(twin, results)
    assert any("ILLUSTRATIVE" in note for note in opinion.dissenting_notes)


# ── every result carries its structural limits ───────────────────────

def test_every_result_carries_the_model_caveats():
    sim = TreatmentSimulator()
    twin = _twin()
    for key in TREATMENT_PROFILES:
        result = sim.simulate(key, twin, 180)
        for caveat in MODEL_CAVEATS:
            assert caveat in result.caveats, key


def test_the_caveats_name_what_cannot_be_represented():
    joined = " ".join(MODEL_CAVEATS)
    assert "complete response" in joined
    assert "without ever reaching a partial response" in joined
    assert "5 mm" in joined


def test_the_summary_reports_both_units_and_the_censoring_flag():
    result = TreatmentSimulator().simulate("pembrolizumab", _twin(), 180)
    summary = result.summary()
    assert "volume reduction" in summary["best_response_volume_pct"]
    assert "diameter reduction" in summary["best_response_diameter_pct"]
    assert summary["pfs_censored"] is True
    assert "month" not in summary["pfs"]
    assert summary["caveats"]


def test_the_diameter_figure_is_the_smaller_of_the_two():
    """A 38% volume reduction is a 15% diameter reduction; reporting the
    volume figure beside a RECIST category overstated the shrinkage."""
    result = TreatmentSimulator().simulate("pembrolizumab", _twin(), 180)
    assert result.best_response < 1.0
    assert result.best_response_diameter_ratio > result.best_response
    assert result.best_response_diameter_ratio == pytest.approx(
        result.best_response ** (1.0 / 3.0))


# ── trial constants match the publications ───────────────────────────

def test_checkmate_650_reports_the_higher_response_rate_before_chemotherapy():
    """Sharma et al., Cancer Cell 2020: ORR 25% chemotherapy-naive (cohort 1),
    10% post-chemotherapy (cohort 2). The repo had the cohorts swapped, which
    inverts the clinical message."""
    from cognisom.mad.evidence import CHECKPOINT_TRIALS
    data = CHECKPOINT_TRIALS["checkmate_650"].supporting_data
    assert data["ORR_pre_chemo"] == 0.25
    assert data["ORR_post_chemo"] == 0.10
    assert data["ORR_pre_chemo"] > data["ORR_post_chemo"]


def test_keynote_158_tmb_high_median_pfs_is_the_published_value():
    """Median PFS in the tTMB-high subgroup was 2.1 months, not 4.1."""
    from cognisom.mad.evidence import CHECKPOINT_TRIALS
    data = CHECKPOINT_TRIALS["keynote_158"].supporting_data
    assert data["median_PFS_months"] == 2.1
    assert data["ORR_TMB_high"] == 0.29


def test_keynote_199_claims_no_cohort_three_response_rate():
    """Cohort 3 is bone-predominant: RECIST ORR is not its endpoint, and the
    0.05 recorded for it could not be verified against the publication."""
    from cognisom.mad.evidence import CHECKPOINT_TRIALS
    item = CHECKPOINT_TRIALS["keynote_199"]
    assert "ORR_cohort3" not in item.supporting_data
    assert item.supporting_data["ORR_cohort1"] == 0.05
    assert item.supporting_data["ORR_cohort2"] == 0.03
    assert item.supporting_data["DCR_cohort3"] == 0.22


def test_keynote_942_reports_its_interval_and_p_value():
    """HR 0.561 (95% CI 0.309-1.017, two-sided p=0.053): the interval crosses
    1, so "44% reduction in recurrence/death" alone overstated it."""
    from cognisom.mad.evidence import VACCINE_TRIALS
    data = VACCINE_TRIALS["keynote_942"].supporting_data
    assert data["HR_RFS"] == 0.561
    assert data["HR_RFS_95CI"] == [0.309, 1.017]
    assert data["p_two_sided"] == 0.053
    assert data["significant_at_0_05"] is False


def test_profound_records_the_control_arm_the_growth_rate_is_fitted_to():
    from cognisom.mad.evidence import PARP_TRIALS
    data = PARP_TRIALS["profound"].supporting_data
    assert data["rPFS_control"] == PROFOUND_COHORT_A_CONTROL_RPFS_MONTHS
    assert data["rPFS_olaparib"] == 7.4
    assert data["HR"] == 0.34
    assert data["HR_95CI"] == [0.25, 0.47]


# ─────────────────────────────────────────────────────────────────────
# Consolidation grafts
# ─────────────────────────────────────────────────────────────────────
#
# (6) THE ENTITY LIBRARY -- the parameter source this repo documents as
#     preferred -- ROUTED NO MECHANISM AT ALL. Its class labels are
#     "immunotherapy" for pembrolizumab and a bare "checkpoint inhibitor"
#     for nivolumab, ipilimumab, atezolizumab and relatlimab, so with
#     `TreatmentSimulator(store=EntityStore())` every checkpoint arm
#     scored the 0.5 baseline for every patient. Mechanisms are now
#     resolved by INN before the class string.
#
# (7) DROPPING THE PFS TERM ONLY WHEN CENSORED INVERTED ITS SIGN: an
#     observed day-110 progression scored +0.153 while an arm that never
#     progressed scored 0.0. An observed progression now penalises.
#
# (8) RESPONSE FOLLOWED BY PROGRESSION IS STILL NOT REACHABLE. That is
#     the largest remaining dishonesty in this module, and it is pinned
#     here so that whoever adds acquired resistance has to come and
#     delete this test rather than discover the gap later.

from cognisom.genomics.treatment_simulator import (  # noqa: E402
    DRUG_MECHANISMS,
    _inn_of,
)


def _entity_simulator():
    """A simulator reading the entity library, or a skip."""
    pytest = __import__("pytest")
    try:
        from cognisom.library.store import EntityStore
    except Exception as exc:                        # pragma: no cover
        pytest.skip(f"entity library unavailable: {exc}")
    sim = TreatmentSimulator(store=EntityStore())
    if not sim._entity_profiles:                    # pragma: no cover
        pytest.skip("entity library carries no drug profiles here")
    return sim


def test_the_entity_librarys_own_class_labels_really_are_unroutable():
    """Documents why an INN table is needed: these are the live labels."""
    sim = _entity_simulator()
    classes = {k: sim._profiles[k]["class"]
               for k in ("pembrolizumab", "nivolumab", "ipilimumab")}
    assert classes["pembrolizumab"] == "immunotherapy"
    assert classes["nivolumab"] == "checkpoint inhibitor"
    assert classes["ipilimumab"] == "checkpoint inhibitor"
    # An anti-PD-1 and an anti-CTLA-4 share the second label, so the
    # class string cannot carry the mechanism for either of them.
    assert classes["nivolumab"] == classes["ipilimumab"]


def test_entity_library_checkpoint_arms_are_resolved_by_drug_name():
    sim = _entity_simulator()
    assert treatment_mechanisms(sim._profiles["pembrolizumab"]) == {MECH_PD1}
    assert treatment_mechanisms(sim._profiles["nivolumab"]) == {MECH_PD1}
    assert treatment_mechanisms(sim._profiles["ipilimumab"]) == {MECH_CTLA4}
    assert treatment_mechanisms(sim._profiles["olaparib"]) == {MECH_PARP}
    assert treatment_mechanisms(
        sim._profiles["enzalutamide"]) == {MECH_AR_ANTAGONIST}


def test_an_anti_lag3_does_not_inherit_the_pd1_terms():
    """relatlimab shares ipilimumab's class label and must not share its
    mechanism, nor pick up PD-1 terms it has no claim to."""
    assert treatment_mechanisms(
        {"name": "Relatlimab", "class": "checkpoint inhibitor"}) == frozenset()


def test_entity_library_checkpoint_effectiveness_depends_on_the_patient():
    """Before the INN table these were 0.500 for every patient.

    Each arm is varied along the input its own mechanism reads: PD-1-axis
    blockade reads PD-L1, TMB and the immune score; anti-CTLA-4 works by
    depleting regulatory T-cells, so its patient term is the Treg
    fraction and it is correctly insensitive to PD-L1.
    """
    sim = _entity_simulator()
    for key in ("pembrolizumab", "nivolumab", "atezolizumab"):
        profile = sim._profiles[key]
        hot = sim._calculate_effectiveness(
            profile, _twin(immune_score="hot", tumor_mutational_burden=25.0,
                           pd_l1_expression=0.9))
        cold = sim._calculate_effectiveness(
            profile, _twin(immune_score="cold", tumor_mutational_burden=0.5,
                           pd_l1_expression=0.0))
        assert hot > cold, key

    ipilimumab = sim._profiles["ipilimumab"]
    treg_high = sim._calculate_effectiveness(
        ipilimumab, _twin(treg_fraction=0.8))
    treg_low = sim._calculate_effectiveness(
        ipilimumab, _twin(treg_fraction=0.0))
    assert treg_high > treg_low

    # And none of them sits on the bare baseline any more.
    for key in ("pembrolizumab", "nivolumab", "atezolizumab", "ipilimumab"):
        assert sim._calculate_effectiveness(
            sim._profiles[key], _twin()) != 0.5, key


def test_an_explicit_mechanisms_key_still_outranks_the_drug_name():
    assert treatment_mechanisms(
        {"name": "Pembrolizumab", "class": "immunotherapy",
         "mechanisms": frozenset({MECH_PARP})}) == {MECH_PARP}


def test_an_unknown_drug_and_an_unknown_class_resolve_to_nothing():
    assert treatment_mechanisms(
        {"name": "Fictionalimab", "class": "not a real class"}) == frozenset()


def test_the_inn_token_is_the_first_word_of_the_display_name():
    assert _inn_of({"name": "Pembrolizumab (Keytruda)"}) == "pembrolizumab"
    assert _inn_of({"name": "Lutetium-177 PSMA"}) == "lutetium-177"
    assert _inn_of({}) == ""
    # Every key in the table is reachable by that rule.
    for inn in DRUG_MECHANISMS:
        assert _inn_of({"name": inn.title()}) == inn


def test_every_entity_library_drug_resolves_without_a_warning(caplog):
    sim = _entity_simulator()
    import logging as _logging
    with caplog.at_level(_logging.WARNING,
                         logger="cognisom.genomics.treatment_simulator"):
        for key in sim._entity_profiles:
            treatment_mechanisms(sim._profiles[key])
    assert "maps to a mechanism" not in caplog.text


# ── (7) an observed progression must not outscore a censored one ──────

class _StubResult:
    """Only the attributes `ClinicalAgent.analyze` reads."""

    def __init__(self, pfs_days, censored, name="Olaparib (Lynparza)"):
        self.treatment_name = name
        self.treatment_class = "PARP inhibitor"
        self.duration_days = 180
        self.best_response = 0.9
        self.progression_free_days = pfs_days
        self.progression_free_censored = censored
        self.t_cell_reactivation_fraction = 0.0
        self.immune_related_adverse_events = 0.1
        self.response_category = "PD" if not censored else "SD"


def _pfs_term(result):
    opinion = ClinicalAgent().analyze(_twin(), [result])
    terms = [c for r in opinion.treatment_rankings for c in r.contributions
             if c.feature_name == "progression_free_survival"]
    assert len(terms) == 1
    return terms[0]


def test_an_observed_progression_penalises_rather_than_rewards():
    observed = _pfs_term(_StubResult(110, censored=False))
    censored = _pfs_term(_StubResult(180, censored=True))
    assert censored.delta == 0.0
    assert censored.direction == "neutral"
    assert observed.delta < 0.0
    assert observed.direction == "negative"
    # The defect: +0.153 for progressing, 0.0 for not progressing.
    assert observed.delta < censored.delta


def test_an_earlier_progression_is_penalised_more_heavily():
    deltas = [_pfs_term(_StubResult(d, censored=False)).delta
              for d in (10, 60, 120, 179)]
    assert deltas == sorted(deltas), deltas
    assert deltas[0] < deltas[-1] < 0.0


def test_the_penalty_meets_the_censored_branch_at_the_horizon():
    """No discontinuity: a progression on the last day scores exactly 0,
    with the same direction the censored branch reports."""
    at_horizon = _pfs_term(_StubResult(180, censored=False))
    censored = _pfs_term(_StubResult(180, censored=True))
    assert at_horizon.delta == 0.0
    assert censored.delta == 0.0
    assert at_horizon.direction == censored.direction == "neutral"


def test_the_penalty_cannot_exceed_the_magnitude_the_term_already_had():
    assert _pfs_term(_StubResult(0, censored=False)).delta == -0.25


# ── (8) response-then-progression is still unreachable ───────────────

def test_response_followed_by_progression_is_not_reachable_today():
    """Pinned so that adding acquired resistance has to delete this.

    The clinically central pattern -- respond, acquire resistance,
    progress -- cannot be produced by these dynamics: the resistance term
    is 1.8e-4/day at day 180 against a kill rate near 1.2e-2/day. The
    classifier can EXPRESS it (progression is measured against a running
    nadir, so a sub-baseline relapse is a progression), but the model
    cannot generate it. Every PD this model returns therefore means
    "never responded", which is what MODEL_CAVEATS says.
    """
    sim = TreatmentSimulator()
    twin = _twin()
    curve = sim._simulate_tumor_dynamics(
        0.5, TREATMENT_PROFILES["pembrolizumab"], twin, 3650)

    # It responds deeply ...
    assert min(curve) < 0.5
    # ... and never progresses from its nadir, over ten years.
    assert sim._first_progression_day(curve) is None

    # And the classifier would have caught it if it had: a synthetic
    # relapse from a nadir of 0.3 to 0.6 is a RECIST progression.
    nadir = 0.30
    relapse = [1.0, 0.5, nadir, nadir * (RECIST_PD_DIAMETER_RATIO ** 3) + 1e-6]
    assert sim._first_progression_day(relapse) == 3
    assert sim._classify_response(nadir, True) == "PR"


# ── (9) asking for the months of a censored PFS is an error ──────────

def test_the_months_accessor_refuses_a_censored_value():
    pytest = __import__("pytest")
    from cognisom.genomics.treatment_simulator import progression_free_months

    assert progression_free_months(110, False) == pytest.approx(
        110 / DAYS_PER_MONTH)
    with pytest.raises(ValueError, match="right-censored"):
        progression_free_months(180, True)


def test_a_censored_result_refuses_to_give_its_pfs_in_months():
    pytest = __import__("pytest")
    sim = TreatmentSimulator()
    result = sim.simulate("pembrolizumab", _twin(), 180)
    assert result.progression_free_censored is True
    with pytest.raises(ValueError, match="right-censored"):
        result.progression_free_months


def test_an_uncensored_result_gives_its_pfs_in_months():
    pytest = __import__("pytest")
    sim = TreatmentSimulator()
    result = sim.simulate("neoantigen_vaccine_pembro",
                          _non_responder_twin(), 180)
    assert result.progression_free_censored is False
    assert result.progression_free_months == pytest.approx(
        result.progression_free_days / DAYS_PER_MONTH)


def test_the_package_has_exactly_one_days_to_months_conversion():
    """Pinned so a second one cannot appear without this failing."""
    import pathlib
    import cognisom.genomics.treatment_simulator as ts
    root = pathlib.Path(ts.__file__).resolve().parent.parent
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        for needle in ("pfs / 30", "pfs/30", "pfs_days / 30",
                       "progression_free_days / 30"):
            if needle in text:
                offenders.append(f"{path.name}: {needle}")
    assert offenders == [], offenders
