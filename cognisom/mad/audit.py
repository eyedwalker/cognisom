"""
Audit Trail for MAD Agent
==========================

Append-only audit logging for FDA 21 CFR Part 11 alignment.
Every MAD Board session is recorded with:
  - WHO: user/clinician ID
  - WHAT: action taken and decision made
  - WHEN: ISO 8601 timestamp
  - WHY: content hash linking to full decision

Storage: SQLite at /opt/cognisom/data/audit/mad_audit.db (volume-mounted)
Immutable: No UPDATE or DELETE operations.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_AUDIT_DIR = "/opt/cognisom/data/audit"
DEFAULT_DB_NAME = "mad_audit.db"


@dataclass
class AuditRecord:
    """Immutable audit record for a MAD Board session."""

    session_id: str
    timestamp: str
    user_id: str
    patient_id: str
    action: str

    # Data integrity
    input_hash: str
    """SHA256 of input VCF/data."""

    decision_hash: str
    """SHA256 of the BoardDecision content."""

    # Decision summary
    recommended_treatment: str
    consensus_level: str
    confidence: float
    n_agents: int
    n_evidence_items: int
    n_warnings: int

    # Provenance
    pipeline_version: str = "cognisom-mad-v0.1.0"
    model_versions: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "patient_id": self.patient_id,
            "action": self.action,
            "input_hash": self.input_hash,
            "decision_hash": self.decision_hash,
            "recommended_treatment": self.recommended_treatment,
            "consensus_level": self.consensus_level,
            "confidence": round(self.confidence, 4),
            "n_agents": self.n_agents,
            "n_evidence_items": self.n_evidence_items,
            "n_warnings": self.n_warnings,
            "pipeline_version": self.pipeline_version,
            "model_versions": self.model_versions,
        }

    @staticmethod
    def from_board_decision(decision, user_id: str, input_data: str) -> "AuditRecord":
        """Create an audit record from a BoardDecision."""
        input_hash = hashlib.sha256(input_data.encode()).hexdigest()
        decision_hash = decision.content_hash()

        model_versions = {}
        for opinion in decision.agent_opinions:
            model_versions.update(opinion.model_versions)

        return AuditRecord(
            session_id=decision.session_id,
            timestamp=decision.timestamp,
            user_id=user_id,
            patient_id=decision.patient_id,
            action="mad_board_convened",
            input_hash=input_hash,
            decision_hash=decision_hash,
            recommended_treatment=decision.recommended_treatment,
            consensus_level=decision.consensus_level,
            confidence=decision.confidence,
            n_agents=len(decision.agent_opinions),
            n_evidence_items=len(decision.evidence_chain),
            n_warnings=len(decision.warnings),
            model_versions=model_versions,
        )


class AuditStore:
    """Append-only audit log backed by SQLite.

    CRITICAL: This store only supports INSERT operations.
    No UPDATE or DELETE is ever performed.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            audit_dir = Path(DEFAULT_AUDIT_DIR)
            audit_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(audit_dir / DEFAULT_DB_NAME)

        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create the audit table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS mad_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL UNIQUE,
                    timestamp TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    patient_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    decision_hash TEXT NOT NULL,
                    recommended_treatment TEXT NOT NULL,
                    consensus_level TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    n_agents INTEGER NOT NULL,
                    n_evidence_items INTEGER NOT NULL,
                    n_warnings INTEGER NOT NULL,
                    pipeline_version TEXT NOT NULL,
                    model_versions_json TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_patient
                ON mad_audit(patient_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_timestamp
                ON mad_audit(timestamp)
            """)

    def record(self, audit: AuditRecord) -> None:
        """Append an audit record. Never updates or deletes."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO mad_audit (
                        session_id, timestamp, user_id, patient_id, action,
                        input_hash, decision_hash, recommended_treatment,
                        consensus_level, confidence, n_agents,
                        n_evidence_items, n_warnings, pipeline_version,
                        model_versions_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    audit.session_id,
                    audit.timestamp,
                    audit.user_id,
                    audit.patient_id,
                    audit.action,
                    audit.input_hash,
                    audit.decision_hash,
                    audit.recommended_treatment,
                    audit.consensus_level,
                    audit.confidence,
                    audit.n_agents,
                    audit.n_evidence_items,
                    audit.n_warnings,
                    audit.pipeline_version,
                    json.dumps(audit.model_versions),
                ))
            logger.info(f"Audit record created: session={audit.session_id}")
        except sqlite3.IntegrityError:
            logger.warning(f"Duplicate audit record: session={audit.session_id}")
        except Exception as e:
            logger.error(f"Audit record failed: {e}")

    def get_by_session(self, session_id: str) -> Optional[AuditRecord]:
        """Retrieve an audit record by session ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM mad_audit WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_record(row)

    def get_by_patient(self, patient_id: str, limit: int = 50) -> List[AuditRecord]:
        """Retrieve audit records for a patient, most recent first."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM mad_audit WHERE patient_id = ? "
                "ORDER BY timestamp DESC LIMIT ?",
                (patient_id, limit),
            ).fetchall()
            return [self._row_to_record(row) for row in rows]

    def get_recent(self, limit: int = 100) -> List[AuditRecord]:
        """Retrieve the most recent audit records."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM mad_audit ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._row_to_record(row) for row in rows]

    def count(self) -> int:
        """Total number of audit records."""
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM mad_audit").fetchone()[0]

    @staticmethod
    def _row_to_record(row) -> AuditRecord:
        return AuditRecord(
            session_id=row["session_id"],
            timestamp=row["timestamp"],
            user_id=row["user_id"],
            patient_id=row["patient_id"],
            action=row["action"],
            input_hash=row["input_hash"],
            decision_hash=row["decision_hash"],
            recommended_treatment=row["recommended_treatment"],
            consensus_level=row["consensus_level"],
            confidence=row["confidence"],
            n_agents=row["n_agents"],
            n_evidence_items=row["n_evidence_items"],
            n_warnings=row["n_warnings"],
            pipeline_version=row["pipeline_version"],
            model_versions=json.loads(row["model_versions_json"]),
        )
