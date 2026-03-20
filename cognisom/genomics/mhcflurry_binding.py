"""
MHCflurry Binding Prediction
==============================

Production-grade MHC-I peptide binding prediction using MHCflurry 2.0.
Replaces the simplified position-weight matrix (PWM) in neoantigen_predictor.py.

MHCflurry:
  - Trained on >400,000 experimental binding measurements
  - Supports 300+ HLA-I alleles
  - AUC >0.95 for most common alleles
  - Includes presentation score (binding + processing)
  - Open source: pip install mhcflurry

Falls back to the existing PWM if MHCflurry is not installed or fails.

References:
  O'Donnell et al., Cell Systems 2020
  https://github.com/openvax/mhcflurry
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Lazy-loaded predictor (MHCflurry models are ~500 MB)
_predictor = None
_available = None


def is_mhcflurry_available() -> bool:
    """Check if MHCflurry is installed and models are downloaded."""
    global _available, _predictor
    if _available is not None:
        return _available

    try:
        from mhcflurry import Class1PresentationPredictor
        _predictor = Class1PresentationPredictor.load()
        _available = True
        logger.info("MHCflurry loaded: %d supported alleles",
                     len(_predictor.supported_alleles))
    except ImportError:
        logger.info("MHCflurry not installed (pip install mhcflurry)")
        _available = False
    except Exception as e:
        logger.warning("MHCflurry failed to load: %s", e)
        _available = False

    return _available


def reset_availability():
    """Reset cached availability check (useful after installing models)."""
    global _available, _predictor
    _available = None
    _predictor = None


def get_predictor():
    """Get the lazily-loaded MHCflurry predictor."""
    global _predictor
    if _predictor is None:
        from mhcflurry import Class1PresentationPredictor
        _predictor = Class1PresentationPredictor.load()
    return _predictor


@dataclass
class BindingResult:
    """Result of a peptide-MHC binding prediction."""

    peptide: str
    hla_allele: str
    affinity_nm: float
    """Predicted IC50 in nanomolar. Lower = stronger binding."""

    percentile_rank: float
    """Percentile rank (0-100). Lower = stronger binding."""

    presentation_score: float
    """Combined binding + processing score (0-1). Higher = more likely presented."""

    is_strong_binder: bool
    """IC50 < 50 nM."""

    is_weak_binder: bool
    """IC50 < 500 nM."""

    method: str = "mhcflurry"
    """'mhcflurry' or 'pwm' (fallback)."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "peptide": self.peptide,
            "hla_allele": self.hla_allele,
            "affinity_nm": round(self.affinity_nm, 2),
            "percentile_rank": round(self.percentile_rank, 4),
            "presentation_score": round(self.presentation_score, 6),
            "is_strong_binder": self.is_strong_binder,
            "is_weak_binder": self.is_weak_binder,
            "method": self.method,
        }


def predict_binding(
    peptide: str,
    hla_allele: str,
) -> BindingResult:
    """Predict peptide-MHC binding affinity.

    Uses MHCflurry if available, falls back to PWM.

    Args:
        peptide: Peptide sequence (8-14 AA)
        hla_allele: HLA allele (e.g., "HLA-A*02:01" or "HLA-A0201")

    Returns:
        BindingResult with affinity, percentile, and presentation score.
    """
    if is_mhcflurry_available():
        return _predict_mhcflurry(peptide, hla_allele)
    else:
        return _predict_pwm_fallback(peptide, hla_allele)


def predict_binding_batch(
    peptides: List[str],
    hla_alleles: List[str],
) -> List[BindingResult]:
    """Predict binding for multiple peptide-allele pairs.

    MHCflurry is much faster in batch mode (vectorized neural network).

    Args:
        peptides: List of peptide sequences
        hla_alleles: List of HLA alleles (same length as peptides,
                     or single allele applied to all)
    """
    if len(hla_alleles) == 1:
        hla_alleles = hla_alleles * len(peptides)

    if is_mhcflurry_available():
        return _predict_mhcflurry_batch(peptides, hla_alleles)
    else:
        return [_predict_pwm_fallback(p, a) for p, a in zip(peptides, hla_alleles)]


def predict_best_allele(
    peptide: str,
    patient_alleles: List[str],
) -> BindingResult:
    """Find the patient's HLA allele that best presents this peptide.

    Tests all patient alleles and returns the strongest binding.
    """
    if not patient_alleles:
        return _predict_pwm_fallback(peptide, "HLA-A*02:01")

    best = None
    for allele in patient_alleles:
        result = predict_binding(peptide, allele)
        if best is None or result.affinity_nm < best.affinity_nm:
            best = result
    return best


def get_supported_alleles() -> List[str]:
    """Return list of HLA alleles supported by MHCflurry."""
    if is_mhcflurry_available():
        return sorted(get_predictor().supported_alleles)
    return []


# --- MHCflurry implementation ---

def _normalize_allele(allele: str) -> str:
    """Normalize HLA allele to MHCflurry format (e.g., 'HLA-A*02:01').

    MHCflurry 2.x uses the standard IMGT format with asterisk and colon.
    Handles inputs like: HLA-A*02:01, HLA-A0201, A*02:01, A0201.
    """
    import re
    a = allele.replace("HLA-", "")

    # Already has asterisk and colon → just add prefix
    if "*" in a and ":" in a:
        return f"HLA-{a}"

    # Remove existing * and : for re-parsing
    a = a.replace("*", "").replace(":", "")

    # Parse: first letter = locus, then 2+ digit groups
    match = re.match(r"^([ABC])(\d{2})(\d{2,})$", a)
    if match:
        locus, group, protein = match.groups()
        return f"HLA-{locus}*{group}:{protein}"

    # Fallback: return as-is with prefix
    return f"HLA-{a}"


def _predict_mhcflurry(peptide: str, hla_allele: str) -> BindingResult:
    """Predict using MHCflurry neural network."""
    predictor = get_predictor()
    allele = _normalize_allele(hla_allele)

    # Check allele is supported
    if allele not in predictor.supported_alleles:
        logger.debug("Allele %s not supported by MHCflurry, using PWM fallback", allele)
        return _predict_pwm_fallback(peptide, hla_allele)

    try:
        df = predictor.predict(
            peptides=[peptide],
            alleles=[allele],
            verbose=0,
        )
        row = df.iloc[0]

        affinity = float(row.get("affinity", 5000))
        presentation = float(row.get("presentation_score", 0))
        percentile = float(row.get("affinity_percentile", 50))

        return BindingResult(
            peptide=peptide,
            hla_allele=allele,
            affinity_nm=affinity,
            percentile_rank=percentile,
            presentation_score=presentation,
            is_strong_binder=affinity < 50,
            is_weak_binder=affinity < 500,
            method="mhcflurry",
        )
    except Exception as e:
        logger.warning("MHCflurry prediction failed for %s/%s: %s", peptide, allele, e)
        return _predict_pwm_fallback(peptide, hla_allele)


def _predict_mhcflurry_batch(
    peptides: List[str],
    hla_alleles: List[str],
) -> List[BindingResult]:
    """Batch prediction using MHCflurry (faster than individual calls)."""
    predictor = get_predictor()
    normalized = [_normalize_allele(a) for a in hla_alleles]

    # Filter to supported alleles
    valid_indices = []
    valid_peptides = []
    valid_alleles = []
    results = [None] * len(peptides)

    for i, (pep, allele) in enumerate(zip(peptides, normalized)):
        if allele in predictor.supported_alleles and 8 <= len(pep) <= 14:
            valid_indices.append(i)
            valid_peptides.append(pep)
            valid_alleles.append(allele)
        else:
            results[i] = _predict_pwm_fallback(pep, hla_alleles[i])

    if valid_peptides:
        try:
            df = predictor.predict(
                peptides=valid_peptides,
                alleles=valid_alleles,
                verbose=0,
            )

            for idx, (_, row) in zip(valid_indices, df.iterrows()):
                affinity = float(row.get("affinity", 5000))
                results[idx] = BindingResult(
                    peptide=valid_peptides[valid_indices.index(idx)],
                    hla_allele=valid_alleles[valid_indices.index(idx)],
                    affinity_nm=affinity,
                    percentile_rank=float(row.get("affinity_percentile", 50)),
                    presentation_score=float(row.get("presentation_score", 0)),
                    is_strong_binder=affinity < 50,
                    is_weak_binder=affinity < 500,
                    method="mhcflurry",
                )
        except Exception as e:
            logger.warning("MHCflurry batch failed: %s", e)
            for idx in valid_indices:
                if results[idx] is None:
                    results[idx] = _predict_pwm_fallback(
                        peptides[idx], hla_alleles[idx]
                    )

    return results


# --- PWM fallback ---

def _predict_pwm_fallback(peptide: str, hla_allele: str) -> BindingResult:
    """Simplified position-weight matrix fallback.

    Used when MHCflurry is not installed. Less accurate (~75% IEDB concordance)
    but has zero dependencies.
    """
    import math

    # Simple anchor-based scoring
    score = 0.0

    # Position 2 and C-terminal anchors for HLA-A*02:01
    if len(peptide) >= 9:
        # P2 anchor: L, M, I, V, A preferred
        if peptide[1] in "LMIVA":
            score += 3.0
        # P9 (C-terminal): L, V, I preferred
        if peptide[-1] in "LVI":
            score += 3.0

    # Hydrophobicity
    hydro = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
             "E": -3.5, "Q": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
             "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
             "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}
    avg_hydro = sum(hydro.get(aa, 0) for aa in peptide) / len(peptide)
    if -1.0 < avg_hydro < 2.0:
        score += 0.5

    ic50 = 5000.0 * math.exp(-0.7 * score)
    ic50 = max(1.0, min(50000.0, ic50))

    return BindingResult(
        peptide=peptide,
        hla_allele=hla_allele,
        affinity_nm=ic50,
        percentile_rank=min(100, ic50 / 500 * 50),
        presentation_score=max(0, 1.0 - ic50 / 50000),
        is_strong_binder=ic50 < 50,
        is_weak_binder=ic50 < 500,
        method="pwm_fallback",
    )
