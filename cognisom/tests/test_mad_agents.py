"""
Tests for MAD Agent specialist agents.

Tests each agent independently with synthetic patient data,
verifying treatment rankings, evidence tracing, and edge cases.
"""

import pytest
from cognisom.genomics.patient_profile import PatientProfileBuilder
from cognisom.genomics.twin_config import DigitalTwinConfig
from cognisom.genomics.treatment_simulator import TreatmentSimulator
from cognisom.genomics.synthetic_vcf import get_synthetic_vcf
from cognisom.mad.agents import (
    GenomicsAgent, ImmuneAgent, ClinicalAgent, AgentOpinion,
)
from cognisom.mad.errors import MADErrorCode


@pytest.fixture
def demo_patient():
    """Build a complete patient profile from synthetic VCF."""
    builder = PatientProfileBuilder()
    vcf = get_synthetic_vcf()
    profile = builder.from_vcf_text(vcf, "TEST-001")
    twin = DigitalTwinConfig.from_profile_only(profile)
    sim = TreatmentSimulator()
    recommended = sim.get_recommended_treatments(twin)
    results = sim.compare_treatments(recommended, twin)
    return profile, twin, results


class TestGenomicsAgent:
    """Test the Genomics Agent."""

    def test_produces_opinion(self, demo_patient):
        profile, twin, _ = demo_patient
        agent = GenomicsAgent()
        opinion = agent.analyze(profile=profile, twin=twin)

        assert isinstance(opinion, AgentOpinion)
        assert opinion.agent_name == "genomics"
        assert 0 <= opinion.confidence <= 1
        assert len(opinion.treatment_rankings) > 0

    def test_rankings_are_sorted(self, demo_patient):
        profile, twin, _ = demo_patient
        agent = GenomicsAgent()
        opinion = agent.analyze(profile=profile, twin=twin)

        scores = [r.score for r in opinion.treatment_rankings]
        assert scores == sorted(scores, reverse=True)

    def test_hrd_patient_recommends_parp(self, demo_patient):
        """Synthetic patient has BRCA1/2 + ATM — should rank PARP highly."""
        profile, twin, _ = demo_patient
        assert profile.has_dna_repair_defect

        agent = GenomicsAgent()
        opinion = agent.analyze(profile=profile, twin=twin)

        top_key = opinion.top_treatment
        assert top_key in ("olaparib", "olaparib_pembro_combo")

    def test_evidence_items_populated(self, demo_patient):
        profile, twin, _ = demo_patient
        agent = GenomicsAgent()
        opinion = agent.analyze(profile=profile, twin=twin)

        assert len(opinion.evidence_items) > 0
        for ev in opinion.evidence_items:
            assert ev.source_type
            assert ev.source_name
            assert ev.claim

    def test_contributions_have_direction(self, demo_patient):
        profile, twin, _ = demo_patient
        agent = GenomicsAgent()
        opinion = agent.analyze(profile=profile, twin=twin)

        for ranking in opinion.treatment_rankings:
            for c in ranking.contributions:
                assert c.direction in ("positive", "negative")
                assert c.feature_name

    def test_model_versions_populated(self, demo_patient):
        profile, twin, _ = demo_patient
        agent = GenomicsAgent()
        opinion = agent.analyze(profile=profile, twin=twin)

        assert "variant_annotator" in opinion.model_versions
        assert "nccn_guidelines" in opinion.model_versions

    def test_serialization(self, demo_patient):
        profile, twin, _ = demo_patient
        agent = GenomicsAgent()
        opinion = agent.analyze(profile=profile, twin=twin)

        d = opinion.to_dict()
        assert isinstance(d, dict)
        assert d["agent_name"] == "genomics"
        assert isinstance(d["treatment_rankings"], list)
        assert isinstance(d["evidence_items"], list)


class TestImmuneAgent:
    """Test the Immune Agent."""

    def test_produces_opinion(self, demo_patient):
        profile, twin, _ = demo_patient
        agent = ImmuneAgent()
        opinion = agent.analyze(twin=twin, profile=profile)

        assert isinstance(opinion, AgentOpinion)
        assert opinion.agent_name == "immune"
        assert len(opinion.treatment_rankings) > 0

    def test_unknown_immune_score_lowers_confidence(self):
        """Explicitly test with unknown immune score."""
        from cognisom.genomics.twin_config import DigitalTwinConfig
        from cognisom.genomics.patient_profile import PatientProfile

        twin = DigitalTwinConfig(immune_score="unknown")
        profile = PatientProfile(patient_id="IMMUNE-TEST")

        agent = ImmuneAgent()
        opinion = agent.analyze(twin=twin, profile=profile)

        assert opinion.confidence <= 0.65  # Penalty for unknown immune
        assert MADErrorCode.LOW_CONFIDENCE_IMMUNE.value in opinion.warnings

    def test_parp_and_ar_are_neutral(self, demo_patient):
        profile, twin, _ = demo_patient
        agent = ImmuneAgent()
        opinion = agent.analyze(twin=twin, profile=profile)

        parp_ranking = next(
            (r for r in opinion.treatment_rankings if r.treatment_key == "olaparib"),
            None,
        )
        assert parp_ranking is not None
        assert parp_ranking.score == pytest.approx(0.3, abs=0.01)

    def test_vaccine_scoring_with_candidates(self, demo_patient):
        profile, twin, _ = demo_patient
        agent = ImmuneAgent()
        opinion = agent.analyze(twin=twin, profile=profile)

        vax = next(
            (r for r in opinion.treatment_rankings
             if r.treatment_key == "neoantigen_vaccine"),
            None,
        )
        assert vax is not None
        # Score depends on vaccine candidate count


class TestClinicalAgent:
    """Test the Clinical Agent."""

    def test_produces_opinion(self, demo_patient):
        _, twin, results = demo_patient
        agent = ClinicalAgent()
        opinion = agent.analyze(twin=twin, treatment_results=results)

        assert isinstance(opinion, AgentOpinion)
        assert opinion.agent_name == "clinical"
        assert len(opinion.treatment_rankings) == len(results)

    def test_rankings_by_response(self, demo_patient):
        _, twin, results = demo_patient
        agent = ClinicalAgent()
        opinion = agent.analyze(twin=twin, treatment_results=results)

        # Rankings should be assigned
        for r in opinion.treatment_rankings:
            assert r.rank >= 1

    def test_high_irae_generates_dissent(self, demo_patient):
        _, twin, results = demo_patient
        agent = ClinicalAgent()
        opinion = agent.analyze(twin=twin, treatment_results=results)

        # Pembro+ipi combo has 45% irAE — should generate dissent
        high_irae = [r for r in results if r.immune_related_adverse_events > 0.4]
        if high_irae:
            assert len(opinion.dissenting_notes) > 0

    def test_empty_results_returns_zero_confidence(self):
        from cognisom.genomics.twin_config import DigitalTwinConfig
        agent = ClinicalAgent()
        opinion = agent.analyze(twin=DigitalTwinConfig(), treatment_results=[])

        assert opinion.confidence == 0.0
        assert MADErrorCode.SIMULATION_FAILURE.value in opinion.warnings

    def test_simulation_evidence_attached(self, demo_patient):
        _, twin, results = demo_patient
        agent = ClinicalAgent()
        opinion = agent.analyze(twin=twin, treatment_results=results)

        for ranking in opinion.treatment_rankings:
            sim_ev = [e for e in ranking.evidence if e.source_type == "simulation"]
            assert len(sim_ev) >= 1, f"No simulation evidence for {ranking.treatment_key}"
