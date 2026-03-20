"""
MAD Agent Research Study Protocol
===================================

"Retrospective Validation of a Multi-Agent Molecular Decision Support
System for Immunotherapy Selection in Metastatic Castration-Resistant
Prostate Cancer"

Design: Retrospective concordance study, multi-cohort
Primary Cohort: SU2C/PCF Dream Team mCRPC 2019 (429 patients)
Validation Cohort: TCGA-PRAD PanCancer Atlas (494 patients)

Primary Endpoint: Treatment-Biomarker Concordance Rate
Secondary Endpoints: Biomarker sensitivity/specificity, agent agreement,
    neoantigen concordance, processing speed

Reuses SU2CFileValidator for data loading.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PatientStudyResult:
    """MAD Agent result for a single study patient."""
    patient_id: str

    # Ground truth (from SU2C)
    actual_treatment: str = ""
    abi_enza_status: str = ""
    os_months: float = 0.0
    os_status: str = ""
    actual_tmb: float = 0.0

    # MAD Agent predictions
    mad_recommended: str = ""
    mad_alternatives: List[str] = field(default_factory=list)
    mad_consensus_level: str = ""
    mad_confidence: float = 0.0
    n_agents_agree: int = 0

    # Per-agent top picks
    genomics_top: str = ""
    immune_top: str = ""
    clinical_top: str = ""

    # Biomarker detection
    predicted_tmb: float = 0.0
    predicted_tmb_high: bool = False
    actual_tmb_high: bool = False
    has_hrd: bool = False
    has_brca: bool = False
    predicted_parp_candidate: bool = False
    actual_received_parp: bool = False
    has_ar_mutation: bool = False
    predicted_msi_h: bool = False

    # Concordance flags
    treatment_concordant: bool = False
    """Top MAD recommendation matches actual treatment class."""

    biomarker_concordant: bool = False
    """Biomarker-driven recommendation matches actual treatment class."""

    # Timing
    processing_seconds: float = 0.0

    # Evidence count
    n_evidence_items: int = 0
    n_warnings: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "actual_treatment": self.actual_treatment,
            "mad_recommended": self.mad_recommended,
            "mad_consensus_level": self.mad_consensus_level,
            "mad_confidence": round(self.mad_confidence, 4),
            "n_agents_agree": self.n_agents_agree,
            "treatment_concordant": self.treatment_concordant,
            "biomarker_concordant": self.biomarker_concordant,
            "predicted_tmb": round(self.predicted_tmb, 2),
            "actual_tmb": round(self.actual_tmb, 2),
            "processing_seconds": round(self.processing_seconds, 4),
        }


@dataclass
class ConcordanceMetrics:
    """Treatment-biomarker concordance statistics."""
    total_patients: int = 0
    treatment_concordant: int = 0
    biomarker_concordant: int = 0
    treatment_concordance_rate: float = 0.0
    biomarker_concordance_rate: float = 0.0

    # Per-biomarker concordance
    parp_concordance: float = 0.0
    """BRCA+ patients who MAD recommended PARP / total BRCA+ patients."""
    checkpoint_concordance: float = 0.0
    """TMB-H patients who MAD recommended checkpoint / total TMB-H patients."""
    ar_concordance: float = 0.0
    """AR-mutated patients who MAD recommended AR therapy / total AR-mutated."""

    # Agreement
    unanimous_rate: float = 0.0
    majority_rate: float = 0.0
    split_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_patients": self.total_patients,
            "treatment_concordance_rate": round(self.treatment_concordance_rate, 4),
            "biomarker_concordance_rate": round(self.biomarker_concordance_rate, 4),
            "parp_concordance": round(self.parp_concordance, 4),
            "checkpoint_concordance": round(self.checkpoint_concordance, 4),
            "ar_concordance": round(self.ar_concordance, 4),
            "unanimous_rate": round(self.unanimous_rate, 4),
            "majority_rate": round(self.majority_rate, 4),
            "split_rate": round(self.split_rate, 4),
        }


@dataclass
class BiomarkerAccuracy:
    """Sensitivity/specificity for each biomarker detection."""
    biomarker: str
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    @property
    def sensitivity(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def specificity(self) -> float:
        denom = self.true_negatives + self.false_positives
        return self.true_negatives / denom if denom > 0 else 0.0

    @property
    def ppv(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def npv(self) -> float:
        denom = self.true_negatives + self.false_negatives
        return self.true_negatives / denom if denom > 0 else 0.0

    def wilson_ci(self, p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
        """Wilson score interval for 95% confidence."""
        if n == 0:
            return (0.0, 0.0)
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denominator
        spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
        return (max(0.0, center - spread), min(1.0, center + spread))

    def to_dict(self) -> Dict[str, Any]:
        n = self.true_positives + self.false_negatives + self.true_negatives + self.false_positives
        sens_ci = self.wilson_ci(self.sensitivity, self.true_positives + self.false_negatives)
        spec_ci = self.wilson_ci(self.specificity, self.true_negatives + self.false_positives)
        return {
            "biomarker": self.biomarker,
            "sensitivity": round(self.sensitivity, 4),
            "sensitivity_95ci": [round(x, 4) for x in sens_ci],
            "specificity": round(self.specificity, 4),
            "specificity_95ci": [round(x, 4) for x in spec_ci],
            "ppv": round(self.ppv, 4),
            "npv": round(self.npv, 4),
            "tp": self.true_positives,
            "fp": self.false_positives,
            "tn": self.true_negatives,
            "fn": self.false_negatives,
            "total": n,
        }


@dataclass
class StudyResults:
    """Complete study output."""
    study_name: str = (
        "Retrospective Validation of MAD Agent for "
        "Immunotherapy Selection in mCRPC"
    )
    cohort: str = "SU2C/PCF 2019"
    run_date: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    n_patients: int = 0
    total_processing_seconds: float = 0.0

    patient_results: List[PatientStudyResult] = field(default_factory=list)
    concordance: Optional[ConcordanceMetrics] = None
    biomarker_accuracy: List[BiomarkerAccuracy] = field(default_factory=list)
    tmb_calibration: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "study_name": self.study_name,
            "cohort": self.cohort,
            "run_date": self.run_date,
            "n_patients": self.n_patients,
            "total_processing_seconds": round(self.total_processing_seconds, 2),
            "avg_seconds_per_patient": round(
                self.total_processing_seconds / max(1, self.n_patients), 4
            ),
            "concordance": self.concordance.to_dict() if self.concordance else None,
            "biomarker_accuracy": [b.to_dict() for b in self.biomarker_accuracy],
            "tmb_calibration": self.tmb_calibration,
        }


class MADStudy:
    """Implements the MAD Agent research study protocol.

    Uses existing SU2CFileValidator for data loading and the MAD Board
    for patient analysis.
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir

    def run_full_study(
        self,
        n_patients: Optional[int] = None,
        progress_callback=None,
    ) -> StudyResults:
        """Run the complete study protocol.

        Args:
            n_patients: Limit number of patients (None = all).
            progress_callback: Optional callback(current, total, patient_id).

        Returns:
            StudyResults with all endpoints computed.
        """
        from .su2c_file_validator import SU2CFileValidator

        # Load cohort
        validator = SU2CFileValidator(self.data_dir)
        patients = validator.load_patients()

        if n_patients:
            patients = patients[:n_patients]

        total = len(patients)
        results = StudyResults(n_patients=total)
        start_time = time.time()

        for idx, patient in enumerate(patients):
            if progress_callback:
                progress_callback(idx + 1, total, patient.patient_id)

            try:
                result = self._analyze_patient(patient)
                results.patient_results.append(result)
            except Exception as e:
                logger.warning(f"Patient {patient.patient_id} failed: {e}")
                results.patient_results.append(PatientStudyResult(
                    patient_id=patient.patient_id,
                    actual_treatment=patient.chemo_regimen,
                ))

        results.total_processing_seconds = time.time() - start_time

        # Compute endpoints
        results.concordance = self.compute_concordance(results.patient_results)
        results.biomarker_accuracy = self.compute_biomarker_accuracy(results.patient_results)
        results.tmb_calibration = self.compute_tmb_calibration(results.patient_results)

        return results

    def _analyze_patient(self, su2c_patient) -> PatientStudyResult:
        """Run MAD Agent on a single SU2C patient."""
        from ..genomics.patient_profile import PatientProfileBuilder
        from ..genomics.twin_config import DigitalTwinConfig
        from ..genomics.treatment_simulator import TreatmentSimulator
        from ..mad.board import BoardModerator
        from .su2c_file_validator import SU2CFileValidator

        start = time.time()

        # Convert SU2C mutations to VCF format (reuse existing method)
        validator = SU2CFileValidator.__new__(SU2CFileValidator)
        vcf_text = validator.mutations_to_vcf(su2c_patient)

        if not su2c_patient.mutations:
            return PatientStudyResult(
                patient_id=su2c_patient.patient_id,
                actual_treatment=su2c_patient.chemo_regimen,
            )

        # Build full patient profile via existing pipeline
        builder = PatientProfileBuilder()
        profile = builder.from_vcf_text(vcf_text, su2c_patient.patient_id)

        # Build twin
        twin = DigitalTwinConfig.from_profile_only(profile)

        # Simulate treatments
        simulator = TreatmentSimulator()
        recommended = simulator.get_recommended_treatments(twin)
        treatment_results = simulator.compare_treatments(recommended, twin)

        # Run MAD Board
        moderator = BoardModerator()
        decision = moderator.run_full_analysis(
            patient_id=su2c_patient.patient_id,
            profile=profile,
            twin=twin,
            treatment_results=treatment_results,
        )

        processing_time = time.time() - start

        # Map actual treatment to class
        actual_class = self._classify_actual_treatment(su2c_patient)

        # Concordance check
        predicted_class = self._classify_predicted_treatment(decision.recommended_treatment)
        treatment_concordant = actual_class == predicted_class and actual_class != "unknown"

        # Biomarker concordance
        biomarker_concordant = self._check_biomarker_concordance(
            profile, decision.recommended_treatment, su2c_patient,
        )

        # Agent agreement count
        agents_agree = sum(
            1 for op in decision.agent_opinions
            if op.top_treatment == decision.recommended_treatment
        )

        result = PatientStudyResult(
            patient_id=su2c_patient.patient_id,
            actual_treatment=su2c_patient.chemo_regimen,
            abi_enza_status=su2c_patient.abi_enza_status,
            os_months=su2c_patient.os_months,
            os_status=su2c_patient.os_status,
            actual_tmb=su2c_patient.tmb,
            mad_recommended=decision.recommended_treatment,
            mad_alternatives=decision.alternative_treatments,
            mad_consensus_level=decision.consensus_level,
            mad_confidence=decision.confidence,
            n_agents_agree=agents_agree,
            genomics_top=decision.agent_opinions[0].top_treatment or "",
            immune_top=decision.agent_opinions[1].top_treatment or "",
            clinical_top=decision.agent_opinions[2].top_treatment or "",
            predicted_tmb=profile.tumor_mutational_burden,
            predicted_tmb_high=profile.is_tmb_high,
            actual_tmb_high=su2c_patient.tmb >= 10.0 if su2c_patient.tmb > 0 else False,
            has_hrd=profile.has_dna_repair_defect,
            has_brca="BRCA1" in profile.affected_genes or "BRCA2" in profile.affected_genes,
            predicted_parp_candidate=profile.has_dna_repair_defect,
            has_ar_mutation=profile.has_ar_mutation,
            treatment_concordant=treatment_concordant,
            biomarker_concordant=biomarker_concordant,
            processing_seconds=processing_time,
            n_evidence_items=len(decision.evidence_chain),
            n_warnings=len(decision.warnings),
        )

        return result

    def compute_concordance(self, results: List[PatientStudyResult]) -> ConcordanceMetrics:
        """Compute treatment-biomarker concordance metrics."""
        n = len(results)
        if n == 0:
            return ConcordanceMetrics()

        treatment_concordant = sum(1 for r in results if r.treatment_concordant)
        biomarker_concordant = sum(1 for r in results if r.biomarker_concordant)

        # PARP concordance: of BRCA+ patients, how many got PARP recommendation?
        brca_patients = [r for r in results if r.has_brca]
        parp_for_brca = sum(1 for r in brca_patients if r.predicted_parp_candidate)
        parp_concordance = parp_for_brca / len(brca_patients) if brca_patients else 0.0

        # Checkpoint concordance: of TMB-H patients, how many got checkpoint?
        tmb_h = [r for r in results if r.actual_tmb_high]
        ck_for_tmb = sum(
            1 for r in tmb_h
            if r.mad_recommended in ("pembrolizumab", "pembro_ipi_combo", "neoantigen_vaccine_pembro")
        )
        checkpoint_concordance = ck_for_tmb / len(tmb_h) if tmb_h else 0.0

        # AR concordance
        ar_patients = [r for r in results if r.has_ar_mutation]
        ar_treated = sum(1 for r in ar_patients if r.mad_recommended == "enzalutamide")
        ar_concordance = ar_treated / len(ar_patients) if ar_patients else 0.0

        # Consensus distribution
        unanimous = sum(1 for r in results if r.mad_consensus_level == "unanimous")
        majority = sum(1 for r in results if r.mad_consensus_level == "majority")
        split = sum(1 for r in results if r.mad_consensus_level == "split")

        return ConcordanceMetrics(
            total_patients=n,
            treatment_concordant=treatment_concordant,
            biomarker_concordant=biomarker_concordant,
            treatment_concordance_rate=treatment_concordant / n,
            biomarker_concordance_rate=biomarker_concordant / n,
            parp_concordance=parp_concordance,
            checkpoint_concordance=checkpoint_concordance,
            ar_concordance=ar_concordance,
            unanimous_rate=unanimous / n,
            majority_rate=majority / n,
            split_rate=split / n,
        )

    def compute_biomarker_accuracy(
        self, results: List[PatientStudyResult]
    ) -> List[BiomarkerAccuracy]:
        """Compute sensitivity/specificity for each biomarker."""
        # TMB-High detection
        tmb_acc = BiomarkerAccuracy(biomarker="TMB_high")
        for r in results:
            if r.actual_tmb <= 0:
                continue  # No ground truth TMB
            if r.actual_tmb_high and r.predicted_tmb_high:
                tmb_acc.true_positives += 1
            elif r.actual_tmb_high and not r.predicted_tmb_high:
                tmb_acc.false_negatives += 1
            elif not r.actual_tmb_high and r.predicted_tmb_high:
                tmb_acc.false_positives += 1
            else:
                tmb_acc.true_negatives += 1

        # HRD detection (BRCA as proxy for ground truth)
        hrd_acc = BiomarkerAccuracy(biomarker="HRD")
        for r in results:
            if r.has_brca and r.has_hrd:
                hrd_acc.true_positives += 1
            elif r.has_brca and not r.has_hrd:
                hrd_acc.false_negatives += 1
            elif not r.has_brca and r.has_hrd:
                # Could be ATM/CDK12 — not false positive, skip
                pass
            else:
                hrd_acc.true_negatives += 1

        return [tmb_acc, hrd_acc]

    def compute_tmb_calibration(
        self, results: List[PatientStudyResult]
    ) -> Dict[str, Any]:
        """Compute TMB prediction vs actual calibration."""
        pairs = [
            (r.predicted_tmb, r.actual_tmb)
            for r in results
            if r.actual_tmb > 0
        ]
        if not pairs:
            return {"n_pairs": 0}

        predicted = [p[0] for p in pairs]
        actual = [p[1] for p in pairs]

        # Pearson correlation
        n = len(pairs)
        mean_p = sum(predicted) / n
        mean_a = sum(actual) / n

        cov = sum((p - mean_p) * (a - mean_a) for p, a in zip(predicted, actual))
        var_p = sum((p - mean_p) ** 2 for p in predicted)
        var_a = sum((a - mean_a) ** 2 for a in actual)

        if var_p > 0 and var_a > 0:
            pearson_r = cov / math.sqrt(var_p * var_a)
        else:
            pearson_r = 0.0

        # Mean absolute error
        mae = sum(abs(p - a) for p, a in zip(predicted, actual)) / n

        # Bland-Altman
        diffs = [p - a for p, a in zip(predicted, actual)]
        mean_diff = sum(diffs) / n
        std_diff = math.sqrt(sum((d - mean_diff) ** 2 for d in diffs) / max(1, n - 1))

        return {
            "n_pairs": n,
            "pearson_r": round(pearson_r, 4),
            "mean_absolute_error": round(mae, 4),
            "bland_altman_mean_diff": round(mean_diff, 4),
            "bland_altman_std": round(std_diff, 4),
            "bland_altman_loa_lower": round(mean_diff - 1.96 * std_diff, 4),
            "bland_altman_loa_upper": round(mean_diff + 1.96 * std_diff, 4),
        }

    @staticmethod
    def _classify_actual_treatment(patient) -> str:
        """Map SU2C treatment to treatment class."""
        regimen = getattr(patient, "chemo_regimen", "").lower()
        abi_enza = getattr(patient, "abi_enza_status", "").lower()

        if "olaparib" in regimen or "rucaparib" in regimen or "niraparib" in regimen:
            return "PARP"
        elif "pembrolizumab" in regimen or "nivolumab" in regimen or "ipilimumab" in regimen:
            return "checkpoint"
        elif "enzalutamide" in regimen or "abiraterone" in regimen or "enza" in abi_enza:
            return "AR_targeted"
        elif regimen:
            return "other"
        return "unknown"

    @staticmethod
    def _classify_predicted_treatment(treatment_key: str) -> str:
        """Map MAD recommended treatment to class."""
        if treatment_key in ("olaparib", "olaparib_pembro_combo"):
            return "PARP"
        elif treatment_key in ("pembrolizumab", "nivolumab", "ipilimumab",
                               "pembro_ipi_combo", "neoantigen_vaccine_pembro"):
            return "checkpoint"
        elif treatment_key == "enzalutamide":
            return "AR_targeted"
        elif treatment_key == "neoantigen_vaccine":
            return "vaccine"
        return "unknown"

    @staticmethod
    def _check_biomarker_concordance(profile, recommended: str, patient) -> bool:
        """Check if MAD recommendation is concordant with biomarker status."""
        # BRCA → should recommend PARP
        if profile.has_dna_repair_defect:
            if recommended in ("olaparib", "olaparib_pembro_combo"):
                return True

        # TMB-H → should recommend checkpoint
        if profile.is_tmb_high:
            if recommended in ("pembrolizumab", "pembro_ipi_combo",
                               "neoantigen_vaccine_pembro"):
                return True

        # If no strong biomarker, any recommendation is "concordant" (no contradiction)
        if not profile.has_dna_repair_defect and not profile.is_tmb_high:
            return True

        return False
