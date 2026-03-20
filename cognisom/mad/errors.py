"""
Structured error codes for MAD Agent decisions.

Every error maps to a specific clinical edge case, enabling systematic
handling and audit logging of failure modes.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional


class MADErrorCode(str, Enum):
    """Structured error codes for MAD Agent edge cases."""

    INSUFFICIENT_VARIANTS = "MAD-001"
    """Too few variants for reliable TMB estimation (<30 coding variants)."""

    NO_HLA_DATA = "MAD-002"
    """HLA typing failed or returned no alleles — neoantigen prediction disabled."""

    LOW_CONFIDENCE_IMMUNE = "MAD-003"
    """Immune classification confidence below threshold — may be unreliable."""

    CONFLICTING_BIOMARKERS = "MAD-004"
    """Contradictory biomarker signals (e.g. MSI-H but low TMB)."""

    NO_ACTIONABLE_TARGETS = "MAD-005"
    """No druggable mutations or biomarker-driven recommendations found."""

    AGENT_DISAGREEMENT = "MAD-006"
    """All three agents disagree on top treatment — no majority consensus."""

    DATA_PROVENANCE_INCOMPLETE = "MAD-007"
    """Missing reference genome, annotation version, or input hash."""

    TUMOR_ONLY_FALLBACK = "MAD-008"
    """Running in tumor-only mode (no matched normal) — lower confidence."""

    SIMULATION_FAILURE = "MAD-009"
    """Treatment simulation failed for one or more regimens."""

    ENTITY_LIBRARY_UNAVAILABLE = "MAD-010"
    """Entity library not accessible — using hardcoded fallback parameters."""


class MADError(Exception):
    """Exception raised by MAD Agent components with structured error codes."""

    def __init__(
        self,
        code: MADErrorCode,
        message: str,
        details: Optional[dict] = None,
        recoverable: bool = True,
    ):
        self.code = code
        self.message = message
        self.details = details or {}
        self.recoverable = recoverable
        super().__init__(f"[{code.value}] {message}")

    def to_dict(self) -> dict:
        return {
            "code": self.code.value,
            "description": self.code.__doc__ or "",
            "message": self.message,
            "details": self.details,
            "recoverable": self.recoverable,
        }
