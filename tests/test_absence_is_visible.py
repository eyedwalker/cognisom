"""Absence must be visible, not substituted by a plausible number.

Three mechanisms used to replace a value nobody measured with a number in
the plausible range: the tumor-only twin derived its immune score from TMB
and left five fractions at 0.0 with PD-L1 at 0.5; the parameter resolver
substituted literals for absent entity fields; and the segmenter reported
the algorithm that was *requested* rather than the one that ran.
"""

import importlib.util
import logging

import numpy as np
import pytest

from cognisom.genomics.patient_profile import PatientProfile
from cognisom.genomics.treatment_simulator import (
    TREATMENT_PROFILES, TreatmentSimulator,
)
from cognisom.genomics.twin_config import (
    IMMUNE_PROFILE_FIELDS, DigitalTwinConfig,
)
from cognisom.library.parameters import get_parameter
from cognisom.library.store import EntityStore
from cognisom.mad.agents import ImmuneAgent
from cognisom.mad.board import (
    IMMUNE_MEASURED_LIMITATION, IMMUNE_NOT_MEASURED_LIMITATION,
    BoardDecision, BoardModerator,
)
from cognisom.mad.errors import MADErrorCode
from cognisom.mad.explainability import explain_effectiveness


def _tumor_only(tmb=3.2, **kw):
    """A twin from the genomic path only -- no cell-state classifier ran."""
    kw.setdefault("hla_alleles", ["HLA-A*02:01"])
    p = PatientProfile(patient_id="TO", tumor_mutational_burden=tmb, **kw)
    p.tmb_caveats = []
    return p, DigitalTwinConfig.from_profile_only(p)


# ── (1) the tumor-only twin reports absence ──────────────────────────

@pytest.mark.parametrize("field_name", IMMUNE_PROFILE_FIELDS)
def test_an_unmeasured_immune_field_is_none_not_zero(field_name):
    # Parametrised over the field list so adding a field without teaching
    # from_profile_only to pass None for it fails here.
    _, twin = _tumor_only()
    assert getattr(twin, field_name) is None


def test_tumor_only_twin_reports_its_immune_profile_as_unmeasured():
    _, twin = _tumor_only()
    assert twin.immune_profile_measured is False
    assert twin.to_dict()["immune_profile_measured"] is False


@pytest.mark.parametrize("tmb", [0.0, 1.0, 9.9, 10.0, 25.0, 40.0])
def test_tmb_does_not_manufacture_an_immune_microenvironment(tmb):
    # `immune_score = "hot" if is_tmb_high else "cold"` laundered a
    # continuous genomic biomarker into a categorical about the immune
    # microenvironment, which no genomic input can support.
    _, twin = _tumor_only(tmb=tmb)
    assert twin.immune_score == "unknown"


def test_checkpoint_effectiveness_has_no_step_at_the_tmb_threshold():
    # The TMB -> immune_score fabrication put a 0.40-wide jump in
    # anti-PD-1 effectiveness across a 0.1 mut/Mb step at TMB = 10, then
    # saturated so TMB 10 and TMB 25 were indistinguishable.
    sim = TreatmentSimulator()
    effs = []
    for tmb in (1.0, 3.2, 9.9, 10.0, 25.0):
        _, twin = _tumor_only(tmb=tmb)
        effs.append(sim._calculate_effectiveness(
            TREATMENT_PROFILES["pembrolizumab"], twin))
    assert effs == sorted(effs), f"not monotone in TMB: {effs}"
    steps = [b - a for a, b in zip(effs, effs[1:])]
    assert max(steps) < 0.15, f"step discontinuity remains: {steps}"
    assert effs[-1] > effs[-2] - 1e-9


def test_unmeasured_infiltration_is_not_seeded_as_minimal():
    # `max(5, ...)` made an unmeasured tumor look identically, minimally
    # infiltrated rather than unknown.
    _, twin = _tumor_only()
    immune = twin.to_simulation_params()["immune"]
    assert immune["measured"] is False
    assert immune["n_t_cells"] is None
    assert immune["n_nk_cells"] is None
    assert immune["n_macrophages"] is None
    assert immune["t_cell_initial_exhaustion"] is None


def test_a_measured_twin_still_claims_its_measurement():
    twin = DigitalTwinConfig(
        immune_profile_measured=True, immune_score="hot",
        mean_exhaustion=0.45, pd_l1_expression=0.7,
    )
    assert twin.to_simulation_params()["immune"]["measured"] is True


# ── (2) MAD-003 and the limitations text ─────────────────────────────

def test_mad_003_fires_on_the_tumor_only_path():
    profile, twin = _tumor_only()
    opinion = ImmuneAgent().analyze(twin=twin, profile=profile)
    assert MADErrorCode.LOW_CONFIDENCE_IMMUNE.value in opinion.warnings


def test_mad_003_fires_when_a_score_is_present_but_unmeasured():
    # Something downstream may stamp a score onto an unmeasured twin; the
    # flag, not the label, is what records that no classifier ran.
    profile = PatientProfile(patient_id="X", hla_alleles=["HLA-A*02:01"])
    twin = DigitalTwinConfig(immune_score="cold", immune_profile_measured=False)
    opinion = ImmuneAgent().analyze(twin=twin, profile=profile)
    assert MADErrorCode.LOW_CONFIDENCE_IMMUNE.value in opinion.warnings


def test_mad_003_does_not_fire_for_a_measured_microenvironment():
    profile = PatientProfile(patient_id="X", hla_alleles=["HLA-A*02:01"])
    twin = DigitalTwinConfig(
        immune_profile_measured=True, immune_score="hot",
        t_cell_exhaustion_fraction=0.45, mean_exhaustion=0.42,
        treg_fraction=0.1, immune_cell_fraction=0.3, pd_l1_expression=0.7,
    )
    opinion = ImmuneAgent().analyze(twin=twin, profile=profile)
    assert MADErrorCode.LOW_CONFIDENCE_IMMUNE.value not in opinion.warnings


def test_limitations_do_not_claim_a_classifier_ran_when_none_did():
    profile, twin = _tumor_only()
    sim = TreatmentSimulator()
    recommended = sim.get_recommended_treatments(twin)
    results = sim.compare_treatments(recommended, twin)
    decision = BoardModerator().run_full_analysis(
        "TO", profile, twin, results)
    assert IMMUNE_MEASURED_LIMITATION not in decision.limitations
    assert IMMUNE_NOT_MEASURED_LIMITATION in decision.limitations


def test_a_decision_built_without_a_twin_makes_the_weaker_claim():
    # The dataclass default must be the honest text, so the stronger
    # claim cannot be reached by omitting the twin.
    import dataclasses
    factory = next(
        f.default_factory for f in dataclasses.fields(BoardDecision)
        if f.name == "limitations"
    )
    default = factory()
    assert IMMUNE_MEASURED_LIMITATION not in default
    assert IMMUNE_NOT_MEASURED_LIMITATION in default


def test_a_measured_twin_keeps_the_original_limitation_in_place():
    # The immune line must keep its index and the list its length, so a
    # measured decision's limitations are unchanged from before.
    twin = DigitalTwinConfig(immune_profile_measured=True, immune_score="hot")
    from cognisom.mad.board import _limitations_for
    lims = _limitations_for(twin)
    assert len(lims) == 5
    assert lims[3] == IMMUNE_MEASURED_LIMITATION


# ── (3) no treatment is escalated on the absence of evidence ─────────

def test_no_combination_is_escalated_on_an_unmeasured_microenvironment():
    # `if twin.immune_score != "cold"` treated "unknown" as positive
    # grounds to add a checkpoint combination.
    profile, twin = _tumor_only(affected_genes=["BRCA2"])
    assert twin.has_dna_repair_defect
    recommended = TreatmentSimulator().get_recommended_treatments(twin)
    assert "olaparib" in recommended
    assert "olaparib_pembro_combo" not in recommended
    assert "pembro_ipi_combo" not in recommended


def test_a_measured_hot_tumor_still_gets_the_combination():
    twin = DigitalTwinConfig(
        immune_profile_measured=True, immune_score="hot",
        has_dna_repair_defect=True, t_cell_exhaustion_fraction=0.45,
        mean_exhaustion=0.42, treg_fraction=0.1,
        immune_cell_fraction=0.3, pd_l1_expression=0.7,
    )
    recommended = TreatmentSimulator().get_recommended_treatments(twin)
    assert "olaparib_pembro_combo" in recommended


def test_an_unmeasured_biomarker_neither_raises_nor_lowers_the_score():
    # Skipping a term is not the same as scoring it zero: an unmeasured
    # tumor must land strictly between a measured cold and a measured hot.
    sim = TreatmentSimulator()
    pembro = TREATMENT_PROFILES["pembrolizumab"]
    base = dict(tumor_mutational_burden=5.0)
    cold = sim._calculate_effectiveness(pembro, DigitalTwinConfig(
        immune_profile_measured=True, immune_score="cold",
        t_cell_exhaustion_fraction=0.05, pd_l1_expression=0.05, **base))
    unknown = sim._calculate_effectiveness(
        pembro, _tumor_only(tmb=5.0)[1])
    hot = sim._calculate_effectiveness(pembro, DigitalTwinConfig(
        immune_profile_measured=True, immune_score="hot",
        t_cell_exhaustion_fraction=0.45, pd_l1_expression=0.8, **base))
    assert cold < unknown < hot


def test_a_tmb_high_patient_still_gets_a_checkpoint_inhibitor():
    # Guards against the fix destroying a genuine biomarker signal. The
    # GenomicsAgent owns TMB (KEYNOTE-158); the ImmuneAgent must not.
    profile, twin = _tumor_only(tmb=25.0)
    sim = TreatmentSimulator()
    recommended = sim.get_recommended_treatments(twin)
    assert "pembrolizumab" in recommended
    decision = BoardModerator().run_full_analysis(
        "H", profile, twin, sim.compare_treatments(recommended, twin))
    assert decision.recommended_treatment == "pembrolizumab"


# ── (4) the prediction names what it could not read ──────────────────

def test_a_prediction_names_the_inputs_it_could_not_read():
    _, twin = _tumor_only()
    result = TreatmentSimulator().simulate("pembrolizumab", twin)
    assert result.inputs_fully_measured is False
    assert "pd_l1_expression" in result.unmeasured_inputs
    assert "t_cell_exhaustion_fraction" in result.unmeasured_inputs
    assert result.final_exhaustion is None
    assert "unmeasured_inputs" in result.summary()


def test_a_measured_prediction_names_nothing_as_unmeasured():
    twin = DigitalTwinConfig(
        immune_profile_measured=True, immune_score="hot",
        t_cell_exhaustion_fraction=0.45, mean_exhaustion=0.42,
        treg_fraction=0.1, immune_cell_fraction=0.3,
        pd_l1_expression=0.7, mhc1_downregulation=0.2,
    )
    result = TreatmentSimulator().simulate("pembrolizumab", twin)
    assert result.unmeasured_inputs == []
    assert result.inputs_fully_measured is True


def test_the_rationale_says_exhaustion_was_not_measured():
    _, twin = _tumor_only()
    result = TreatmentSimulator().simulate("pembrolizumab", twin)
    assert "not measured" in result.rationale


def test_an_unmeasured_pd_l1_is_explained_as_absent_not_as_fifty_percent():
    _, twin = _tumor_only()
    expl = explain_effectiveness(TREATMENT_PROFILES["pembrolizumab"], twin)
    by_name = {c.feature_name: c for c in expl.contributions}
    pd_l1 = by_name["pd_l1_expression"]
    assert pd_l1.delta == 0.0
    assert pd_l1.direction == "neutral"
    assert "not measured" in pd_l1.description
    assert "50%" not in pd_l1.description


def test_every_regimen_simulates_and_explains_on_an_unmeasured_twin():
    # Drives every effectiveness branch, so a new unguarded read of an
    # immune field fails here rather than in production.
    _, twin = _tumor_only(tmb=25.0, msi_status="MSI-H",
                          affected_genes=["BRCA2", "AR"])
    twin.neoantigen_vaccine_candidate = True
    twin.vaccine_candidate_count = 12
    twin.strong_binder_count = 6
    sim = TreatmentSimulator()
    for key, profile in TREATMENT_PROFILES.items():
        result = sim.simulate(key, twin)
        assert 0.0 <= result.best_response <= 1.0, key
        assert explain_effectiveness(profile, twin) is not None, key
    assert sim.get_recommended_treatments(twin)


# ── (5) the resolver substitutes no literal ──────────────────────────

@pytest.fixture
def resolver():
    from cognisom.genomics.parameter_resolver import (
        SimulationParameterResolver, reset_missing_log,
    )
    reset_missing_log()
    return SimulationParameterResolver(EntityStore())


def test_an_uncurated_gene_asserts_nothing_about_growth_or_mhc1(resolver):
    # `tumor_growth_multiplier: 1.0` for an uncurated gene is not a
    # neutral placeholder; it asserts the mutation does not affect growth.
    effect = resolver.get_gene_immune_effect("AKT1")
    assert effect == {"_source": "absent"}


def test_an_unknown_gene_asserts_nothing(resolver):
    assert resolver.get_gene_immune_effect("NOT_A_REAL_GENE") == {
        "_source": "absent"}


def test_a_curated_gene_keeps_exactly_the_fields_it_has(resolver):
    effect = resolver.get_gene_immune_effect("PTEN")
    assert effect["mhc1_downregulation_effect"] == 0.3
    assert effect["tumor_growth_multiplier"] == 1.3
    # No gene in the library carries pd_l1_effect, so it was fabricated.
    assert "pd_l1_effect" not in effect


def test_a_genuinely_stored_zero_is_still_reported(resolver):
    # AR stores mhc1_downregulation_effect = 0.0. A measured zero and an
    # absent field must stay distinguishable in BOTH directions.
    assert resolver.get_gene_immune_effect("AR")[
        "mhc1_downregulation_effect"] == 0.0


def test_a_curated_gene_set_is_numerically_unchanged(resolver):
    agg = resolver.get_gene_effects_for_mutations(["PTEN", "TP53"])
    assert agg["mhc1_downregulation"] == pytest.approx(0.5)
    assert agg["tumor_growth_multiplier"] == pytest.approx(1.56)
    assert agg["_contributing_genes"] == ["PTEN", "TP53"]
    assert agg["_unresolved_genes"] == []


def test_an_all_uncurated_gene_set_yields_no_mhc1_number(resolver):
    agg = resolver.get_gene_effects_for_mutations(["EGFR", "KRAS"])
    assert agg["mhc1_downregulation"] is None
    assert agg["tumor_growth_multiplier"] is None
    assert agg["_unresolved_genes"] == ["EGFR", "KRAS"]


def test_no_mutations_is_a_different_absence_from_no_annotation(resolver):
    # Annotation ran and found nothing: zero effect is the right answer.
    agg = resolver.get_gene_effects_for_mutations([])
    assert agg["mhc1_downregulation"] == 0.0
    assert agg["_source"] == "no_mutations"


def test_a_twin_whose_genes_are_all_uncurated_reports_no_mhc1_value():
    # 0.0 here read as "MHC-I intact, no NK cell therapy" for a patient
    # nobody evaluated.
    _, twin = _tumor_only(affected_genes=["GATA3", "MAP3K1", "NF1"])
    assert twin.mhc1_downregulation is None


def test_a_cell_name_resolves_to_the_row_carrying_cytotoxic_params(resolver):
    # "NK cell" exists as both a cell_type and an immune_cell row, and
    # only the immune_cell row has kill_probability. Querying cell_type
    # first reported an NK cell that cannot kill anything.
    params = resolver.get_cell_type_params("NK cell")
    assert params["kill_probability"] == 0.3
    assert params["detection_radius_um"] == 30.0
    assert params["_entity_type"] == "immune_cell"
    assert params["_shadowed_by"] == "cell_type"


def test_a_non_cytotoxic_cell_claims_no_kill_probability(resolver):
    params = resolver.get_cell_type_params("Regulatory T cell")
    assert "kill_probability" not in params
    assert params["_entity_type"] == "cell_type"


def test_every_returned_number_is_backed_by_an_entity_parameter(resolver):
    # The structural form of the invariant: it fails if anyone
    # reintroduces a substituted literal under any name.
    store = EntityStore()
    cases = [
        (store.find_entity_by_name("CD8+ Effector T cell", "immune_cell"),
         resolver.get_cell_type_params("CD8+ Effector T cell")),
        (store.find_entity_by_name("Oxygen", "metabolite"),
         resolver.get_metabolite_params("Oxygen")),
        (store.find_entity_by_name("PTEN", "gene"),
         resolver.get_gene_immune_effect("PTEN")),
    ]
    for entity, params in cases:
        assert entity is not None
        for name, value in params.items():
            if name.startswith("_"):
                continue
            assert get_parameter(entity, name) is not None, (
                f"{entity.name}.{name} = {value!r} is backed by nothing")


def test_source_does_not_claim_a_curation_that_does_not_exist(resolver):
    # No entity in the library carries a provenance sidecar, so every
    # value resolves as PLACEHOLDER. "_source: entity_library" asserted
    # curation.
    for params in (resolver.get_cell_type_params("CD8+ Effector T cell"),
                   resolver.get_metabolite_params("Oxygen"),
                   resolver.get_gene_immune_effect("PTEN")):
        assert params["_source"] != "entity_library"
        assert "placeholder" in params["_source"]


def test_an_invented_metabolite_default_is_gone(resolver):
    # critical_low_mM is present on no metabolite in the library.
    assert "critical_low_mM" not in resolver.get_metabolite_params("Oxygen")


def test_a_missing_field_warns_once_per_entity_and_field(resolver, caplog):
    with caplog.at_level(
            logging.WARNING, logger="cognisom.genomics.parameter_resolver"):
        resolver.get_cell_type_params("NK cell")
        first = len(caplog.records)
        resolver.get_cell_type_params("NK cell")
        second = len(caplog.records)
    assert first > 0
    assert second == first, "the same absence was reported twice"
    assert any("kill_radius_um" in r.getMessage() for r in caplog.records)


def test_an_exome_scale_gene_list_reports_once_with_a_count(resolver, caplog):
    genes = [f"NOT_A_GENE_{i}" for i in range(2000)]
    with caplog.at_level(
            logging.WARNING, logger="cognisom.genomics.parameter_resolver"):
        resolver.get_gene_effects_for_mutations(genes)
    assert len(caplog.records) == 1, (
        f"{len(caplog.records)} lines for 2000 uncurated genes")
    assert "2000" in caplog.records[0].getMessage()


# ── (6) segmentation reports what ran ────────────────────────────────

def _blobs():
    img = np.zeros((128, 128), dtype=np.float32)
    yy, xx = np.ogrid[:128, :128]
    for cy, cx in [(30, 30), (30, 90), (90, 30), (90, 90), (60, 60)]:
        img[(yy - cy) ** 2 + (xx - cx) ** 2 < 12 ** 2] = 1.0
    return img + np.random.RandomState(0).rand(128, 128) * 0.05


def _have(mod):
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


@pytest.mark.parametrize("name", ["otsu", "watershed", "basic_threshold"])
def test_the_result_reports_the_algorithm_that_ran(name):
    from cognisom.imaging.segmentation import CellSegmenter, SegmentationMethod
    result = CellSegmenter(method=name).segment(_blobs())
    requested = CellSegmenter.METHOD_NAMES[name]
    if result.method is not requested:
        # A substitution must say so, and say what was asked for.
        assert result.fallback_from is requested
        assert result.is_fallback
    else:
        assert result.fallback_from is None
        assert not result.is_fallback
    assert result.n_cells == 5


def test_basic_threshold_is_nameable():
    # It is what actually runs without scikit-image, so a result cannot
    # report it truthfully while it has no name.
    from cognisom.imaging.segmentation import CellSegmenter, SegmentationMethod
    result = CellSegmenter(method="basic_threshold").segment(_blobs())
    assert result.method is SegmentationMethod.BASIC_THRESHOLD


@pytest.mark.parametrize("name", ["cellpose", "stardist"])
def test_a_learned_segmenter_is_refused_rather_than_approximated(name):
    from cognisom.imaging.segmentation import (
        CellSegmenter, SegmentationUnavailable,
    )
    if _have(name):
        pytest.skip(f"{name} is installed; nothing to refuse")
    with pytest.raises(SegmentationUnavailable):
        CellSegmenter(method=name).segment(_blobs())


def test_the_leaf_also_refuses_when_called_directly():
    from cognisom.imaging.segmentation import (
        CellSegmenter, SegmentationUnavailable,
    )
    if _have("cellpose"):
        pytest.skip("cellpose is installed")
    with pytest.raises(SegmentationUnavailable):
        CellSegmenter()._segment_cellpose(_blobs())


@pytest.mark.parametrize("name", ["cellpoze", "", "otsu2", "deep"])
def test_an_unknown_method_name_raises_instead_of_becoming_watershed(name):
    from cognisom.imaging.segmentation import CellSegmenter
    with pytest.raises(ValueError, match="Unknown segmentation method"):
        CellSegmenter(method=name)


def test_a_known_method_name_is_case_insensitive():
    from cognisom.imaging.segmentation import CellSegmenter, SegmentationMethod
    assert CellSegmenter(method="OTSU").method is SegmentationMethod.OTSU


def test_allow_fallback_false_demands_the_requested_algorithm():
    from cognisom.imaging.segmentation import (
        CellSegmenter, SegmentationUnavailable,
    )
    if _have("skimage"):
        pytest.skip("scikit-image is installed; otsu can run as requested")
    with pytest.raises(SegmentationUnavailable):
        CellSegmenter(method="otsu", allow_fallback=False).segment(_blobs())
    # basic_threshold needs only numpy, so it is always honourable.
    assert CellSegmenter(
        method="basic_threshold", allow_fallback=False
    ).segment(_blobs()).n_cells == 5


def test_unmeasured_morphology_is_flagged_not_reported_as_round_and_solid():
    from cognisom.imaging.segmentation import CellSegmenter
    result = CellSegmenter(method="basic_threshold").segment(_blobs())
    if _have("skimage"):
        assert result.morphology_measured
        return
    assert result.morphology_measured is False
    # These are dataclass defaults, not measurements of anything.
    assert {p.eccentricity for p in result.cell_properties} == {0.0}
    assert {p.solidity for p in result.cell_properties} == {1.0}


def test_the_morphology_flag_is_not_sticky_across_calls():
    from cognisom.imaging.segmentation import CellSegmenter
    segmenter = CellSegmenter(method="basic_threshold")
    segmenter.segment(_blobs())
    assert (segmenter.segment(_blobs()).morphology_measured
            is bool(_have("skimage")))


# ── (7) guards that nothing else pins ────────────────────────────────
# Each of these was measured to be invisible to the rest of the suite:
# reverting its guard left every other test green.

def _unmeasured_vaccine_twin(**kw):
    twin = DigitalTwinConfig(
        immune_score="unknown", immune_profile_measured=False,
        neoantigen_vaccine_candidate=True, vaccine_candidate_count=12,
        strong_binder_count=6, hla_alleles=["HLA-A*02:01"], **kw)
    return twin


def test_the_vaccine_combination_is_not_escalated_on_absence():
    # treatment_simulator's second `!= "cold"` gate.
    recommended = TreatmentSimulator().get_recommended_treatments(
        _unmeasured_vaccine_twin())
    assert "neoantigen_vaccine" in recommended
    assert "neoantigen_vaccine_pembro" not in recommended


def test_a_measured_hot_tumor_still_gets_the_vaccine_combination():
    twin = DigitalTwinConfig(
        immune_score="hot", immune_profile_measured=True,
        neoantigen_vaccine_candidate=True, vaccine_candidate_count=12,
        strong_binder_count=6, t_cell_exhaustion_fraction=0.45,
        mean_exhaustion=0.42, treg_fraction=0.1, immune_cell_fraction=0.3,
        pd_l1_expression=0.7, mhc1_downregulation=0.2,
    )
    assert "neoantigen_vaccine_pembro" in (
        TreatmentSimulator().get_recommended_treatments(twin))


def test_the_vaccine_agent_score_is_not_boosted_on_absence():
    # agents.py's `is_vaccine_eligible and immune_score != "cold"` added a
    # flat 0.10 to the vaccine+pembro score on no evidence at all. Asserted
    # against the un-bonused mean of its two components, so reinstating
    # `!= "cold"` fails here rather than merely shifting a comparison.
    profile = PatientProfile(patient_id="V", hla_alleles=["HLA-A*02:01"])

    def scores(twin):
        return {
            r.treatment_key: r.score for r in
            ImmuneAgent().analyze(twin=twin, profile=profile).treatment_rankings
        }

    unmeasured = scores(_unmeasured_vaccine_twin())
    expected = (unmeasured["neoantigen_vaccine"]
                + unmeasured["pembrolizumab"]) / 2
    assert unmeasured["neoantigen_vaccine_pembro"] == pytest.approx(expected), (
        "an eligibility bonus was applied without a measured microenvironment")

    measured_hot = DigitalTwinConfig(
        immune_score="hot", immune_profile_measured=True,
        neoantigen_vaccine_candidate=True, vaccine_candidate_count=12,
        strong_binder_count=6, t_cell_exhaustion_fraction=0.45,
        mean_exhaustion=0.42, treg_fraction=0.1, immune_cell_fraction=0.3,
        pd_l1_expression=0.7, mhc1_downregulation=0.2)
    hot = scores(measured_hot)
    hot_expected = (hot["neoantigen_vaccine"] + hot["pembrolizumab"]) / 2
    # A measured hot microenvironment is entitled to the bonus.
    assert hot["neoantigen_vaccine_pembro"] > hot_expected


def test_an_unmeasured_mhc1_does_not_lower_vaccine_effectiveness():
    # treatment_simulator's `eff -= twin.mhc1_downregulation * 0.3`.
    sim = TreatmentSimulator()
    vaccine = TREATMENT_PROFILES["neoantigen_vaccine"]
    absent = sim._calculate_effectiveness(
        vaccine, _unmeasured_vaccine_twin(mhc1_downregulation=None))
    measured_intact = sim._calculate_effectiveness(
        vaccine, _unmeasured_vaccine_twin(mhc1_downregulation=0.0))
    measured_lost = sim._calculate_effectiveness(
        vaccine, _unmeasured_vaccine_twin(mhc1_downregulation=0.8))
    # An absent measurement contributes nothing, exactly like a measured
    # zero; a measured loss must still reduce the score.
    assert absent == measured_intact
    assert measured_lost < absent


def test_an_unmeasured_mhc1_is_explained_as_absent_not_as_intact():
    # explainability's `if mhc1_down > 0`.
    expl = explain_effectiveness(
        TREATMENT_PROFILES["neoantigen_vaccine"],
        _unmeasured_vaccine_twin(mhc1_downregulation=None))
    assert "mhc1_downregulation" not in {
        c.feature_name for c in expl.contributions}
    expl_lost = explain_effectiveness(
        TREATMENT_PROFILES["neoantigen_vaccine"],
        _unmeasured_vaccine_twin(mhc1_downregulation=0.8))
    assert "mhc1_downregulation" in {
        c.feature_name for c in expl_lost.contributions}


def test_an_unmeasured_mhc1_does_not_lower_the_agent_vaccine_score():
    # agents.py's `if mhc1_down > 0.3`.
    profile = PatientProfile(patient_id="V", hla_alleles=["HLA-A*02:01"])
    def score(mhc1):
        op = ImmuneAgent().analyze(
            twin=_unmeasured_vaccine_twin(mhc1_downregulation=mhc1),
            profile=profile)
        return {r.treatment_key: r.score for r in op.treatment_rankings}[
            "neoantigen_vaccine"]
    assert score(None) == score(0.0)
    assert score(0.8) < score(None)


def test_a_hot_score_without_a_measured_fraction_renders_without_it():
    # agents.py's `f"... ({immune_fraction:.0%} immune cells)"` raised
    # `unsupported format string passed to NoneType.__format__`.
    profile = PatientProfile(patient_id="H", hla_alleles=["HLA-A*02:01"])
    twin = DigitalTwinConfig(
        immune_score="hot", immune_profile_measured=True,
        immune_cell_fraction=None,
    )
    opinion = ImmuneAgent().analyze(twin=twin, profile=profile)
    hot = [c for r in opinion.treatment_rankings for c in r.contributions
           if c.feature_name == "hot_tumor"]
    assert hot, "the hot_tumor contribution should still be emitted"
    assert "%" not in hot[0].description


def test_a_substituted_algorithm_never_reports_the_requested_one():
    # segmentation's `method=ran`. Written so it fails if `method` goes
    # back to echoing the request, whether or not scikit-image is here.
    from cognisom.imaging.segmentation import CellSegmenter
    requested = "otsu"
    result = CellSegmenter(method=requested).segment(_blobs())
    expected = CellSegmenter.METHOD_NAMES[requested]
    if _have("skimage"):
        assert result.method is expected and result.fallback_from is None
        return
    # scikit-image is absent, so Otsu CANNOT have run.
    assert result.method is not expected, (
        "reported OTSU although scikit-image is not installed")
    assert result.fallback_from is expected
