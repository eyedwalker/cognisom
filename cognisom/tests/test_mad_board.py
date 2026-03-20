"""
Tests for MAD Board Moderator consensus logic.

Tests agreement matrix computation, consensus level assessment,
rationale generation, and edge cases (unanimous, majority, split).
"""

import pytest
from cognisom.genomics.patient_profile import PatientProfileBuilder
from cognisom.genomics.twin_config import DigitalTwinConfig
from cognisom.genomics.treatment_simulator import TreatmentSimulator
from cognisom.genomics.synthetic_vcf import get_synthetic_vcf
from cognisom.mad.board import BoardModerator, BoardDecision
from cognisom.mad.agents import GenomicsAgent, ImmuneAgent, ClinicalAgent


@pytest.fixture
def demo_patient():
    """Build a complete patient profile from synthetic VCF."""
    builder = PatientProfileBuilder()
    vcf = get_synthetic_vcf()
    profile = builder.from_vcf_text(vcf, "BOARD-TEST-001")
    twin = DigitalTwinConfig.from_profile_only(profile)
    sim = TreatmentSimulator()
    recommended = sim.get_recommended_treatments(twin)
    results = sim.compare_treatments(recommended, twin)
    return profile, twin, results


@pytest.fixture
def board_decision(demo_patient):
    """Run full MAD Board analysis on demo patient."""
    profile, twin, results = demo_patient
    moderator = BoardModerator()
    return moderator.run_full_analysis(
        patient_id="BOARD-TEST-001",
        profile=profile,
        twin=twin,
        treatment_results=results,
    )


class TestBoardDecision:
    """Test BoardDecision structure."""

    def test_decision_has_required_fields(self, board_decision):
        d = board_decision
        assert d.session_id
        assert d.patient_id == "BOARD-TEST-001"
        assert d.timestamp
        assert d.recommended_treatment
        assert d.recommended_treatment_name
        assert d.consensus_level in ("unanimous", "majority", "split")
        assert 0 <= d.confidence <= 1

    def test_decision_has_three_agent_opinions(self, board_decision):
        assert len(board_decision.agent_opinions) == 3
        agent_names = {op.agent_name for op in board_decision.agent_opinions}
        assert agent_names == {"genomics", "immune", "clinical"}

    def test_evidence_chain_not_empty(self, board_decision):
        assert len(board_decision.evidence_chain) > 0

    def test_evidence_deduplicated(self, board_decision):
        ids = [(e.source_type, e.source_id) for e in board_decision.evidence_chain]
        assert len(ids) == len(set(ids)), "Duplicate evidence items found"

    def test_content_hash_deterministic(self, board_decision):
        h1 = board_decision.content_hash()
        h2 = board_decision.content_hash()
        assert h1 == h2
        assert len(h1) == 64  # SHA256

    def test_serialization(self, board_decision):
        d = board_decision.to_dict()
        assert isinstance(d, dict)
        assert d["patient_id"] == "BOARD-TEST-001"
        assert isinstance(d["agent_opinions"], list)
        assert isinstance(d["evidence_chain"], list)
        assert d["context_of_use"]  # FDA COU present

    def test_limitations_present(self, board_decision):
        assert len(board_decision.limitations) >= 3

    def test_context_of_use_present(self, board_decision):
        assert "independent review" in board_decision.context_of_use.lower()


class TestBoardModerator:
    """Test BoardModerator consensus logic."""

    def test_agreement_matrix_has_all_treatments(self, board_decision):
        matrix = board_decision.agreement_matrix
        assert isinstance(matrix, dict)
        for treatment, agents in matrix.items():
            assert isinstance(agents, dict)
            # Each treatment should have entries for all 3 agents
            for agent_name in ("genomics", "immune", "clinical"):
                assert agent_name in agents

    def test_consensus_level_reflects_agreement(self, board_decision):
        """Verify consensus level matches actual agent agreement."""
        top = board_decision.recommended_treatment
        agree_count = sum(
            1 for op in board_decision.agent_opinions
            if op.top_treatment == top
        )
        if agree_count == 3:
            assert board_decision.consensus_level == "unanimous"
        elif agree_count >= 2:
            assert board_decision.consensus_level == "majority"
        else:
            assert board_decision.consensus_level == "split"

    def test_rationale_mentions_treatment_name(self, board_decision):
        assert board_decision.recommended_treatment_name in board_decision.unified_rationale

    def test_rationale_mentions_consensus_level(self, board_decision):
        assert board_decision.consensus_level in board_decision.unified_rationale

    def test_custom_weights(self, demo_patient):
        """Test that custom agent weights affect the outcome."""
        profile, twin, results = demo_patient
        moderator = BoardModerator(weights={
            "genomics": 10.0,  # Heavily weight genomics
            "immune": 0.1,
            "clinical": 0.1,
        })
        decision = moderator.run_full_analysis(
            patient_id="WEIGHT-TEST",
            profile=profile,
            twin=twin,
            treatment_results=results,
        )
        # Genomics agent dominates — should match its top pick
        genomics_top = decision.agent_opinions[0].top_treatment
        assert decision.recommended_treatment == genomics_top

    def test_alternatives_populated(self, board_decision):
        assert isinstance(board_decision.alternative_treatments, list)

    def test_dissenting_views_prefixed_by_agent(self, board_decision):
        for dv in board_decision.dissenting_views:
            # Should start with [agent_name]
            assert dv.startswith("[")


class TestBoardEdgeCases:
    """Test edge cases for the board moderator."""

    def test_minimal_patient(self):
        """Test with a minimal patient profile (few variants)."""
        from cognisom.genomics.patient_profile import PatientProfile
        profile = PatientProfile(
            patient_id="MINIMAL",
            variants=[],
            coding_variants=[],
            cancer_driver_mutations=[],
            affected_genes=[],
            tumor_mutational_burden=0.5,
        )
        twin = DigitalTwinConfig.from_profile_only(profile)

        moderator = BoardModerator()
        decision = moderator.run_full_analysis(
            patient_id="MINIMAL",
            profile=profile,
            twin=twin,
            treatment_results=[],  # No treatments simulated
        )

        assert decision.recommended_treatment  # Should still produce a recommendation
        assert decision.confidence < 0.7  # Should be lower confidence

    def test_reproducible(self, demo_patient):
        """Two runs on same patient should produce same recommendation."""
        profile, twin, results = demo_patient
        moderator = BoardModerator()

        d1 = moderator.run_full_analysis("REPRO", profile, twin, results)
        d2 = moderator.run_full_analysis("REPRO", profile, twin, results)

        assert d1.recommended_treatment == d2.recommended_treatment
        assert d1.consensus_level == d2.consensus_level
