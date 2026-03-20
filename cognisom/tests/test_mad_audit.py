"""
Tests for MAD Agent audit trail integrity.

Verifies append-only behavior, hash consistency,
and query operations.
"""

import os
import tempfile
import pytest
from cognisom.mad.audit import AuditStore, AuditRecord


@pytest.fixture
def temp_audit_store():
    """Create an audit store with a temporary database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_audit.db")
        yield AuditStore(db_path=db_path)


@pytest.fixture
def sample_record():
    """Create a sample audit record."""
    return AuditRecord(
        session_id="test-session-001",
        timestamp="2026-03-20T12:00:00",
        user_id="dr.smith",
        patient_id="PATIENT-001",
        action="mad_board_convened",
        input_hash="abc123def456",
        decision_hash="789ghi012jkl",
        recommended_treatment="olaparib",
        consensus_level="majority",
        confidence=0.72,
        n_agents=3,
        n_evidence_items=7,
        n_warnings=1,
        model_versions={"treatment_simulator": "v1.0"},
    )


class TestAuditStore:
    """Test append-only audit store."""

    def test_record_and_retrieve(self, temp_audit_store, sample_record):
        temp_audit_store.record(sample_record)
        retrieved = temp_audit_store.get_by_session("test-session-001")

        assert retrieved is not None
        assert retrieved.session_id == "test-session-001"
        assert retrieved.patient_id == "PATIENT-001"
        assert retrieved.recommended_treatment == "olaparib"
        assert retrieved.confidence == pytest.approx(0.72)

    def test_duplicate_rejected(self, temp_audit_store, sample_record):
        temp_audit_store.record(sample_record)
        temp_audit_store.record(sample_record)  # Should not raise
        assert temp_audit_store.count() == 1

    def test_count(self, temp_audit_store, sample_record):
        assert temp_audit_store.count() == 0
        temp_audit_store.record(sample_record)
        assert temp_audit_store.count() == 1

    def test_get_by_patient(self, temp_audit_store):
        for i in range(3):
            record = AuditRecord(
                session_id=f"session-{i}",
                timestamp=f"2026-03-20T{12+i}:00:00",
                user_id="dr.smith",
                patient_id="PATIENT-001",
                action="mad_board_convened",
                input_hash=f"hash-{i}",
                decision_hash=f"dhash-{i}",
                recommended_treatment="olaparib",
                consensus_level="majority",
                confidence=0.7 + i * 0.05,
                n_agents=3,
                n_evidence_items=5,
                n_warnings=0,
            )
            temp_audit_store.record(record)

        results = temp_audit_store.get_by_patient("PATIENT-001")
        assert len(results) == 3

    def test_get_recent(self, temp_audit_store):
        for i in range(5):
            record = AuditRecord(
                session_id=f"recent-{i}",
                timestamp=f"2026-03-20T{10+i}:00:00",
                user_id="dr.jones",
                patient_id=f"PAT-{i}",
                action="mad_board_convened",
                input_hash=f"h-{i}",
                decision_hash=f"d-{i}",
                recommended_treatment="pembrolizumab",
                consensus_level="unanimous",
                confidence=0.85,
                n_agents=3,
                n_evidence_items=10,
                n_warnings=0,
            )
            temp_audit_store.record(record)

        recent = temp_audit_store.get_recent(3)
        assert len(recent) == 3

    def test_nonexistent_session(self, temp_audit_store):
        result = temp_audit_store.get_by_session("nonexistent")
        assert result is None

    def test_model_versions_preserved(self, temp_audit_store, sample_record):
        temp_audit_store.record(sample_record)
        retrieved = temp_audit_store.get_by_session("test-session-001")
        assert retrieved.model_versions == {"treatment_simulator": "v1.0"}


class TestAuditRecord:
    """Test AuditRecord dataclass."""

    def test_serialization(self, sample_record):
        d = sample_record.to_dict()
        assert d["session_id"] == "test-session-001"
        assert d["user_id"] == "dr.smith"
        assert d["recommended_treatment"] == "olaparib"
        assert isinstance(d["model_versions"], dict)

    def test_from_board_decision(self):
        """Test creating audit record from a real BoardDecision."""
        from cognisom.genomics.patient_profile import PatientProfileBuilder
        from cognisom.genomics.twin_config import DigitalTwinConfig
        from cognisom.genomics.treatment_simulator import TreatmentSimulator
        from cognisom.genomics.synthetic_vcf import get_synthetic_vcf
        from cognisom.mad.board import BoardModerator

        builder = PatientProfileBuilder()
        profile = builder.from_vcf_text(get_synthetic_vcf(), "AUDIT-TEST")
        twin = DigitalTwinConfig.from_profile_only(profile)
        sim = TreatmentSimulator()
        recommended = sim.get_recommended_treatments(twin)
        results = sim.compare_treatments(recommended, twin)

        moderator = BoardModerator()
        decision = moderator.run_full_analysis(
            "AUDIT-TEST", profile, twin, results
        )

        record = AuditRecord.from_board_decision(
            decision, user_id="test_user", input_data="test vcf data"
        )

        assert record.session_id == decision.session_id
        assert record.patient_id == "AUDIT-TEST"
        assert record.user_id == "test_user"
        assert record.input_hash  # SHA256 of input
        assert record.decision_hash  # SHA256 of decision
        assert record.n_agents == 3
        assert record.n_evidence_items > 0
