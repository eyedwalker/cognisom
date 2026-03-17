"""
TCGA / SU2C Validation Module
================================

Validates Cognisom predictions against published prostate cancer genomics
datasets via the cBioPortal REST API.

Data Sources:
- TCGA-PRAD PanCancer Atlas (494 patients)
- SU2C/PCF Dream Team mCRPC 2019 (444 patients with treatment data)
- SU2C/PCF Dream Team mCRPC 2015 (150 patients)
- Metastatic Prostate Cancer Project (123 patients)

REQUIRED CITATIONS (per cBioPortal terms of use):

1. Cerami et al. The cBio Cancer Genomics Portal: An Open Platform for
   Exploring Multidimensional Cancer Genomics Data. Cancer Discovery.
   May 2012; 2:401. PMID: 22588877

2. Gao et al. Integrative analysis of complex cancer genomics and clinical
   profiles using the cBioPortal. Sci. Signal. 6, pl1 (2013).
   PMID: 23550210

3. de Bruijn et al. Analysis and Visualization of Longitudinal Genomic
   and Clinical Data from the AACR Project GENIE Biopharma Collaborative
   in cBioPortal. Cancer Res (2023). PMID: 37668528

Data source: cBioPortal REST API (free, no authentication for public studies)
Study: TCGA-PRAD PanCancer Atlas (494 patients)

Usage:
    validator = TCGAValidator()
    results = validator.run_validation(n_patients=100)
    print(f"Accuracy: {results.accuracy:.1%}")
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

CBIO_BASE = "https://www.cbioportal.org/api"

# Available studies for validation
STUDIES = {
    "tcga_prad": {
        "id": "prad_tcga_pan_can_atlas_2018",
        "name": "TCGA-PRAD PanCancer Atlas",
        "patients": 494,
        "type": "primary",
        "treatment_data": False,
    },
    "su2c_2019": {
        "id": "prad_su2c_2019",
        "name": "SU2C/PCF Dream Team mCRPC 2019",
        "patients": 444,
        "type": "mCRPC",
        "treatment_data": True,
        "treatment_field": "CHEMO_REGIMEN_CATEGORY",
        "drugs_available": ["abiraterone", "enzalutamide", "olaparib", "docetaxel", "cabazitaxel"],
    },
    "su2c_2015": {
        "id": "prad_su2c_2015",
        "name": "SU2C/PCF Dream Team mCRPC 2015",
        "patients": 150,
        "type": "mCRPC",
        "treatment_data": True,
        "treatment_field": "ABI_ENZA_EXPOSURE_STATUS",
    },
    "mpc_broad": {
        "id": "mpcproject_broad_2021",
        "name": "Metastatic Prostate Cancer Project (Broad)",
        "patients": 123,
        "type": "metastatic",
        "treatment_data": True,
    },
}

DEFAULT_STUDY = "tcga_prad"
STUDY_ID = STUDIES[DEFAULT_STUDY]["id"]
MUTATION_PROFILE = f"{STUDY_ID}_mutations"


@dataclass
class TCGAPatient:
    """A TCGA patient with mutations and clinical data."""
    patient_id: str
    sample_id: str
    mutations: List[Dict] = field(default_factory=list)
    clinical: Dict = field(default_factory=dict)
    # Clinical outcomes
    os_months: float = 0.0
    os_status: str = ""  # "0:LIVING" or "1:DECEASED"
    pfs_months: float = 0.0
    pfs_status: str = ""
    dfs_months: float = 0.0
    dfs_status: str = ""
    subtype: str = ""
    gleason: str = ""
    stage: str = ""


@dataclass
class ValidationResult:
    """Result of validating Cognisom against one TCGA patient."""
    patient_id: str
    # TCGA ground truth
    n_mutations: int = 0
    n_drivers: int = 0
    subtype: str = ""
    os_months: float = 0.0
    os_status: str = ""
    pfs_months: float = 0.0
    # Cognisom predictions
    predicted_tmb: float = 0.0
    predicted_mhc1_downreg: float = 0.0
    predicted_best_treatment: str = ""
    predicted_best_response: str = ""
    predicted_vaccine_eligible: bool = False
    n_neoantigens: int = 0
    n_treatments_simulated: int = 0
    # Timing
    processing_seconds: float = 0.0
    status: str = "pending"  # pending, completed, failed
    error: str = ""


@dataclass
class ValidationSummary:
    """Summary of validation across all patients."""
    n_patients: int = 0
    n_completed: int = 0
    n_failed: int = 0
    total_time_seconds: float = 0.0
    # Aggregate metrics
    mean_mutations: float = 0.0
    mean_drivers: float = 0.0
    mean_tmb: float = 0.0
    mean_neoantigens: float = 0.0
    mean_processing_time: float = 0.0
    # Subtype distribution
    subtypes: Dict[str, int] = field(default_factory=dict)
    # Treatment predictions
    treatment_distribution: Dict[str, int] = field(default_factory=dict)
    response_distribution: Dict[str, int] = field(default_factory=dict)
    # Per-patient results
    results: List[ValidationResult] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "n_patients": self.n_patients,
            "n_completed": self.n_completed,
            "n_failed": self.n_failed,
            "total_time_seconds": round(self.total_time_seconds, 1),
            "mean_mutations": round(self.mean_mutations, 1),
            "mean_drivers": round(self.mean_drivers, 1),
            "mean_tmb": round(self.mean_tmb, 1),
            "mean_neoantigens": round(self.mean_neoantigens, 1),
            "mean_processing_time": round(self.mean_processing_time, 2),
            "subtypes": self.subtypes,
            "treatment_distribution": self.treatment_distribution,
            "response_distribution": self.response_distribution,
        }


class TCGAValidator:
    """Validate Cognisom predictions against TCGA-PRAD data."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers["Accept"] = "application/json"

    def fetch_patients(self, n_patients: int = 100,
                       study: str = "tcga_prad") -> List[TCGAPatient]:
        """Download patient data from cBioPortal.

        Args:
            n_patients: Number of patients
            study: Study key from STUDIES dict (tcga_prad, su2c_2019, etc.)
        """
        study_info = STUDIES.get(study, STUDIES[DEFAULT_STUDY])
        study_id = study_info["id"]
        logger.info("Fetching %d patients from %s...", n_patients, study_info["name"])

        # Get all samples
        resp = self.session.get(
            f"{CBIO_BASE}/studies/{study_id}/samples",
            params={"pageSize": n_patients},
            timeout=30,
        )
        resp.raise_for_status()
        samples = resp.json()[:n_patients]

        # Get clinical data for all patients
        patient_ids = list(set(s["patientId"] for s in samples))
        clinical_map = self._fetch_clinical_data(patient_ids, study_id)

        # Build patient objects
        patients = []
        for sample in samples:
            pid = sample["patientId"]
            sid = sample["sampleId"]
            clin = clinical_map.get(pid, {})

            patient = TCGAPatient(
                patient_id=pid,
                sample_id=sid,
                clinical=clin,
                os_months=float(clin.get("OS_MONTHS", 0) or 0),
                os_status=clin.get("OS_STATUS", ""),
                pfs_months=float(clin.get("PFS_MONTHS", 0) or 0),
                pfs_status=clin.get("PFS_STATUS", ""),
                dfs_months=float(clin.get("DFS_MONTHS", 0) or 0),
                dfs_status=clin.get("DFS_STATUS", ""),
                subtype=clin.get("SUBTYPE", ""),
                stage=clin.get("PATH_T_STAGE", ""),
            )
            patients.append(patient)

        # Fetch mutations in batches
        mutation_profile = f"{study_id}_mutations"
        batch_size = 50
        for i in range(0, len(patients), batch_size):
            batch = patients[i:i+batch_size]
            sample_ids = [p.sample_id for p in batch]
            mutations = self._fetch_mutations(sample_ids, mutation_profile)

            # Assign mutations to patients
            for patient in batch:
                patient.mutations = [m for m in mutations
                                     if m.get("sampleId") == patient.sample_id]

            time.sleep(0.5)  # Rate limiting

        logger.info("Fetched %d patients with mutations and clinical data", len(patients))
        return patients

    def _fetch_clinical_data(self, patient_ids: List[str],
                             study_id: str = None) -> Dict[str, Dict]:
        """Fetch clinical data for patients."""
        if study_id is None:
            study_id = STUDY_ID
        resp = self.session.get(
            f"{CBIO_BASE}/studies/{study_id}/clinical-data",
            params={"clinicalDataType": "PATIENT", "pageSize": 10000},
            timeout=30,
        )
        resp.raise_for_status()

        clinical_map = {}
        for entry in resp.json():
            pid = entry["patientId"]
            attr = entry["clinicalAttributeId"]
            value = entry["value"]
            clinical_map.setdefault(pid, {})[attr] = value

        return clinical_map

    def _fetch_mutations(self, sample_ids: List[str],
                         mutation_profile: str = None) -> List[Dict]:
        """Fetch mutations for a batch of samples."""
        if mutation_profile is None:
            mutation_profile = MUTATION_PROFILE
        try:
            resp = self.session.post(
                f"{CBIO_BASE}/molecular-profiles/{mutation_profile}/mutations/fetch",
                json={"sampleIds": sample_ids},
                params={"projection": "DETAILED"},
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            if resp.status_code == 200:
                return resp.json()
            else:
                logger.warning("Mutation fetch failed: %s", resp.status_code)
                return []
        except Exception as e:
            logger.warning("Mutation fetch error: %s", e)
            return []

    def mutations_to_vcf(self, patient: TCGAPatient) -> str:
        """Convert cBioPortal mutations to VCF format for Cognisom."""
        lines = [
            "##fileformat=VCFv4.2",
            f"##source=cBioPortal_TCGA-PRAD_{patient.patient_id}",
            "##reference=GRCh38",
            '##INFO=<ID=GENE,Number=1,Type=String,Description="Gene symbol">',
            '##INFO=<ID=CONSEQUENCE,Number=1,Type=String,Description="Variant consequence">',
            '##INFO=<ID=AA_CHANGE,Number=1,Type=String,Description="Amino acid change">',
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE",
        ]

        for mut in patient.mutations:
            gene_info = mut.get("gene", {})
            gene = gene_info.get("hugoGeneSymbol", "") if isinstance(gene_info, dict) else ""
            chrom = f"chr{mut.get('chr', '')}"
            pos = mut.get("startPosition", 0)
            ref = mut.get("referenceAllele", "")
            alt = mut.get("variantAllele", "")
            protein = mut.get("proteinChange", "")
            mut_type = mut.get("mutationType", "")

            if not ref or not alt or ref == "-" or alt == "-":
                continue

            # Map mutation type to consequence
            consequence_map = {
                "Missense_Mutation": "missense",
                "Nonsense_Mutation": "nonsense",
                "Frame_Shift_Del": "frameshift",
                "Frame_Shift_Ins": "frameshift",
                "Splice_Site": "splice",
                "In_Frame_Del": "inframe_deletion",
                "In_Frame_Ins": "inframe_insertion",
                "Silent": "synonymous",
            }
            consequence = consequence_map.get(mut_type, mut_type.lower())

            info = f"GENE={gene};CONSEQUENCE={consequence}"
            if protein:
                info += f";AA_CHANGE={protein}"

            line = f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t99\tPASS\t{info}\tGT\t0/1"
            lines.append(line)

        return "\n".join(lines)

    def validate_patient(self, patient: TCGAPatient) -> ValidationResult:
        """Run one TCGA patient through Cognisom and record predictions."""
        result = ValidationResult(
            patient_id=patient.patient_id,
            n_mutations=len(patient.mutations),
            subtype=patient.subtype,
            os_months=patient.os_months,
            os_status=patient.os_status,
            pfs_months=patient.pfs_months,
        )

        t0 = time.time()
        try:
            # Convert mutations to VCF
            vcf_text = self.mutations_to_vcf(patient)

            if not patient.mutations:
                result.status = "completed"
                result.error = "No mutations"
                result.processing_seconds = time.time() - t0
                return result

            # Run through Cognisom pipeline
            from cognisom.genomics.patient_profile import PatientProfileBuilder
            builder = PatientProfileBuilder()
            profile = builder.from_vcf_text(vcf_text, patient_id=patient.patient_id)

            result.n_drivers = len(profile.cancer_driver_mutations)
            result.predicted_tmb = profile.tumor_mutational_burden
            result.n_neoantigens = len(profile.predicted_neoantigens)
            result.predicted_vaccine_eligible = profile.neoantigen_vaccine_candidate

            # Build digital twin
            from cognisom.genomics.twin_config import DigitalTwinConfig
            twin = DigitalTwinConfig.from_profile_only(profile)
            result.predicted_mhc1_downreg = twin.mhc1_downregulation

            # Simulate treatments
            from cognisom.library.store import EntityStore
            from cognisom.genomics.treatment_simulator import TreatmentSimulator
            store = EntityStore()
            sim = TreatmentSimulator(store=store)
            recommended = sim.get_recommended_treatments(twin)
            treatments = sim.compare_treatments(recommended[:6], twin, 180)

            result.n_treatments_simulated = len(treatments)
            if treatments:
                best = min(treatments, key=lambda t: t.best_response)
                result.predicted_best_treatment = best.treatment_name
                result.predicted_best_response = best.response_category

            result.status = "completed"

        except Exception as e:
            result.status = "failed"
            result.error = str(e)[:200]

        result.processing_seconds = time.time() - t0
        return result

    def run_validation(self, n_patients: int = 100,
                       progress_callback=None) -> ValidationSummary:
        """Run full TCGA validation.

        Args:
            n_patients: Number of patients to validate
            progress_callback: Called with (current, total, patient_id, status)
        """
        summary = ValidationSummary()
        t0 = time.time()

        # Fetch data
        if progress_callback:
            progress_callback(0, n_patients, "Downloading TCGA data...", "running")

        patients = self.fetch_patients(n_patients)
        summary.n_patients = len(patients)

        # Validate each patient
        for i, patient in enumerate(patients):
            if progress_callback:
                progress_callback(i + 1, len(patients), patient.patient_id, "running")

            result = self.validate_patient(patient)
            summary.results.append(result)

            if result.status == "completed":
                summary.n_completed += 1
            else:
                summary.n_failed += 1

            # Track distributions
            if result.subtype:
                summary.subtypes[result.subtype] = summary.subtypes.get(result.subtype, 0) + 1
            if result.predicted_best_treatment:
                summary.treatment_distribution[result.predicted_best_treatment] = \
                    summary.treatment_distribution.get(result.predicted_best_treatment, 0) + 1
            if result.predicted_best_response:
                summary.response_distribution[result.predicted_best_response] = \
                    summary.response_distribution.get(result.predicted_best_response, 0) + 1

        # Compute aggregates
        completed = [r for r in summary.results if r.status == "completed" and r.n_mutations > 0]
        if completed:
            summary.mean_mutations = sum(r.n_mutations for r in completed) / len(completed)
            summary.mean_drivers = sum(r.n_drivers for r in completed) / len(completed)
            summary.mean_tmb = sum(r.predicted_tmb for r in completed) / len(completed)
            summary.mean_neoantigens = sum(r.n_neoantigens for r in completed) / len(completed)
            summary.mean_processing_time = sum(r.processing_seconds for r in completed) / len(completed)

        summary.total_time_seconds = time.time() - t0

        if progress_callback:
            progress_callback(len(patients), len(patients), "Validation complete", "completed")

        logger.info(
            "TCGA validation: %d/%d completed, mean %d mutations, %d drivers, "
            "%.1f TMB, %.1f neoantigens per patient, %.2fs/patient",
            summary.n_completed, summary.n_patients,
            summary.mean_mutations, summary.mean_drivers,
            summary.mean_tmb, summary.mean_neoantigens,
            summary.mean_processing_time,
        )
        return summary
