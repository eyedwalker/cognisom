"""
SU2C File-Based Validator
===========================

Validates Cognisom against SU2C/PCF mCRPC 2019 dataset from cBioPortal
flat files (data_mutations.txt, data_clinical_patient.txt, data_clinical_sample.txt).

This is more reliable and complete than the API approach because:
- All 64,566 mutations in one file (no pagination)
- Sample-level data includes ARV7, NEPC score, AR score, TMB, ETS fusion
- Reproducible (frozen snapshot)
- Fast (~1s to load vs ~30s API)

Download: https://datahub.assets.cbioportal.org/prad_su2c_2019.tar.gz (98 MB)

REQUIRED CITATIONS:
1. Cerami et al. Cancer Discovery 2012; 2:401. PMID: 22588877
2. Gao et al. Sci. Signal. 6, pl1 (2013). PMID: 23550210
3. de Bruijn et al. Cancer Res (2023). PMID: 37668528
4. Abida et al. Proc Natl Acad Sci 2019; 116:11428-11436 (SU2C/PCF study)
"""

import csv
import io
import logging
import math
import os
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

DOWNLOAD_URL = "https://datahub.assets.cbioportal.org/prad_su2c_2019.tar.gz"
DATA_DIR_NAME = "prad_su2c_2019"

# Placeholder padding base for MAF->VCF indel conversion. See the note in
# mutations_to_vcf: the true base needs the reference FASTA, and nothing
# downstream of here reads it.
_PAD_BASE = "N"


@dataclass
class SU2CPatient:
    """A patient from the SU2C mCRPC dataset."""
    patient_id: str
    sample_id: str = ""
    # Clinical
    chemo_regimen: str = ""
    age_at_diagnosis: float = 0.0
    psa: str = ""
    os_status: str = ""      # "0:LIVING" or "1:DECEASED"
    os_months: float = 0.0
    race: str = ""
    # Sample-level
    abi_enza_status: str = ""  # "Naive", "Exposed", "On treatment"
    ets_fusion: str = ""       # "Positive", "Negative"
    nepc_score: float = 0.0
    ar_score: float = 0.0
    tmb: float = 0.0
    gleason: str = ""
    taxane_status: str = ""
    arv7_expression: float = 0.0
    tissue_site: str = ""
    # Mutations (raw MAF rows)
    mutations: List[Dict] = field(default_factory=list)


@dataclass
class SU2CValidationResult:
    """Result for one patient."""
    patient_id: str
    # Ground truth
    chemo_regimen: str = ""
    os_months: float = 0.0
    os_status: str = ""
    abi_enza_status: str = ""
    actual_tmb: float = 0.0
    ets_fusion: str = ""
    # Cognisom predictions
    n_mutations_parsed: int = 0
    n_drivers: int = 0
    predicted_tmb: float = 0.0
    predicted_mhc1_downreg: float = 0.0
    predicted_best_treatment: str = ""
    predicted_best_response: str = ""
    predicted_vaccine_eligible: bool = False
    n_neoantigens: int = 0
    n_treatments: int = 0
    # Concordance checks
    driver_genes_found: List[str] = field(default_factory=list)
    parp_candidate: bool = False
    immunotherapy_candidate: bool = False
    ar_mutation: bool = False
    # Timing
    processing_seconds: float = 0.0
    status: str = "pending"
    error: str = ""


@dataclass
class SU2CValidationSummary:
    """Summary across all patients."""
    n_patients: int = 0
    n_completed: int = 0
    n_failed: int = 0
    total_time_seconds: float = 0.0
    # Aggregates
    mean_mutations: float = 0.0
    mean_drivers: float = 0.0
    mean_tmb_predicted: float = 0.0
    mean_tmb_actual: float = 0.0
    mean_neoantigens: float = 0.0
    mean_processing_time: float = 0.0
    # TMB agreement against the study's own reported TMB_NONSYNONYMOUS.
    # These were the headline validation claim ("TMB r=0.987") and nothing
    # in this file ever computed a correlation -- the summary carried only
    # two means. The number is now produced here, so it can be reproduced
    # instead of asserted.
    tmb_pearson_r: float = 0.0
    tmb_spearman_rho: float = 0.0
    tmb_median_abs_error: float = 0.0
    tmb_within_25pct: float = 0.0
    tmb_n_compared: int = 0
    # Agreement on the >=10 mut/Mb line, which is what GenomicsAgent
    # actually scores checkpoint inhibitors on.
    tmb_high_true_positive: int = 0
    tmb_high_false_positive: int = 0
    tmb_high_false_negative: int = 0
    tmb_high_true_negative: int = 0

    # Treatment concordance
    parp_candidates_with_brca: int = 0
    total_brca_patients: int = 0
    ar_mutations_detected: int = 0
    total_ar_treated: int = 0
    # Ground truth read straight from the MAF, independent of anything
    # Cognisom derived. The pair above cannot measure agreement: its
    # denominator is "patient has a driver mutation in BRCA1/2, ATM or
    # CDK12" and its numerator is `parp_candidate`, which is *defined* as
    # having a mutation in those genes plus PALB2 and CHEK2. The numerator
    # is implied by the denominator, so it reports 100% by construction
    # whatever the pipeline does.
    hrd_true_positive: int = 0
    hrd_false_positive: int = 0
    hrd_false_negative: int = 0
    hrd_true_negative: int = 0
    # Distributions
    treatment_distribution: Dict[str, int] = field(default_factory=dict)
    response_distribution: Dict[str, int] = field(default_factory=dict)
    regimen_distribution: Dict[str, int] = field(default_factory=dict)
    driver_frequency: Dict[str, int] = field(default_factory=dict)
    # Per-patient
    results: List[SU2CValidationResult] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "study": "SU2C/PCF Dream Team mCRPC 2019",
            "citation": "Abida et al. PNAS 2019; Cerami et al. Cancer Discovery 2012",
            "n_patients": self.n_patients,
            "n_completed": self.n_completed,
            "n_failed": self.n_failed,
            "total_time_seconds": round(self.total_time_seconds, 1),
            "mean_mutations": round(self.mean_mutations, 1),
            "mean_drivers": round(self.mean_drivers, 1),
            "tmb_comparison": {
                "predicted_mean": round(self.mean_tmb_predicted, 2),
                "actual_mean": round(self.mean_tmb_actual, 2),
            },
            "mean_neoantigens": round(self.mean_neoantigens, 1),
            "tmb_agreement": {
                "n_compared": self.tmb_n_compared,
                "pearson_r": round(self.tmb_pearson_r, 4),
                "spearman_rho": round(self.tmb_spearman_rho, 4),
                "median_abs_error": round(self.tmb_median_abs_error, 3),
                "within_25pct": round(self.tmb_within_25pct, 4),
                "tmb_high_confusion": {
                    "tp": self.tmb_high_true_positive,
                    "fp": self.tmb_high_false_positive,
                    "fn": self.tmb_high_false_negative,
                    "tn": self.tmb_high_true_negative,
                },
                "note": (
                    "Agreement between this pipeline's TMB and the study's "
                    "reported TMB_NONSYNONYMOUS. Both are computed from the "
                    "same MAF, so this measures ingest and consequence "
                    "classification, NOT prediction of an independent outcome."
                ),
            },
            "concordance": {
                # Circular by construction -- kept only so its value is
                # visible next to the real one.
                "parp_candidates_with_brca": self.parp_candidates_with_brca,
                "total_brca_patients": self.total_brca_patients,
                "circular": True,
                "ar_mutations_detected": self.ar_mutations_detected,
                "total_ar_treated": self.total_ar_treated,
            },
            "hrd_detection": {
                "true_positive": self.hrd_true_positive,
                "false_positive": self.hrd_false_positive,
                "false_negative": self.hrd_false_negative,
                "true_negative": self.hrd_true_negative,
                "note": (
                    "Ground truth taken from MAF Hugo_Symbol plus "
                    "Variant_Classification, independent of the pipeline."
                ),
            },
            "treatment_distribution": self.treatment_distribution,
            "response_distribution": self.response_distribution,
            "regimen_distribution": self.regimen_distribution,
            "driver_frequency": dict(sorted(self.driver_frequency.items(), key=lambda x: -x[1])[:20]),
        }


class SU2CFileValidator:
    """Validate Cognisom against SU2C flat files."""

    def __init__(self, data_dir: Optional[str] = None):
        """
        Args:
            data_dir: Path to extracted prad_su2c_2019 directory.
                      If None, downloads and extracts to /tmp.
        """
        self.data_dir = data_dir
        self._patients: Optional[List[SU2CPatient]] = None

    def ensure_data(self) -> str:
        """Download and extract data if needed. Returns data directory path."""
        if self.data_dir and Path(self.data_dir).exists():
            return self.data_dir

        # Check /tmp first
        tmp_dir = Path("/tmp/cbio_data/prad_su2c_2019")
        if tmp_dir.exists() and (tmp_dir / "data_mutations.txt").exists():
            self.data_dir = str(tmp_dir)
            return self.data_dir

        # Download
        logger.info("Downloading SU2C 2019 dataset (98 MB)...")
        tmp_dir.parent.mkdir(parents=True, exist_ok=True)

        resp = requests.get(DOWNLOAD_URL, stream=True, timeout=120)
        resp.raise_for_status()

        # Extract tar.gz
        with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
            tar.extractall(path=str(tmp_dir.parent))

        self.data_dir = str(tmp_dir)
        logger.info("Extracted to %s", self.data_dir)
        return self.data_dir

    def load_patients(self) -> List[SU2CPatient]:
        """Load all patients from flat files."""
        if self._patients is not None:
            return self._patients

        data_dir = Path(self.ensure_data())

        # Load clinical patient data
        patient_map: Dict[str, SU2CPatient] = {}
        with open(data_dir / "data_clinical_patient.txt") as f:
            # Skip comment lines (start with #)
            reader = csv.DictReader(
                (line for line in f if not line.startswith("#")),
                delimiter="\t",
            )
            for row in reader:
                pid = row.get("PATIENT_ID", "")
                if not pid:
                    continue
                patient = SU2CPatient(
                    patient_id=pid,
                    chemo_regimen=row.get("CHEMO_REGIMEN_CATEGORY", ""),
                    age_at_diagnosis=float(row.get("AGE_AT_DIAGNOSIS", 0) or 0),
                    psa=row.get("PSA", ""),
                    os_status=row.get("OS_STATUS", ""),
                    os_months=float(row.get("OS_MONTHS", 0) or 0),
                    race=row.get("RACE", ""),
                )
                patient_map[pid] = patient

        # Load sample data (richer per-sample info)
        with open(data_dir / "data_clinical_sample.txt") as f:
            reader = csv.DictReader(
                (line for line in f if not line.startswith("#")),
                delimiter="\t",
            )
            for row in reader:
                pid = row.get("PATIENT_ID", "")
                sid = row.get("SAMPLE_ID", "")
                if pid in patient_map:
                    p = patient_map[pid]
                    p.sample_id = sid
                    p.abi_enza_status = row.get("ABI_ENZA_EXPOSURE_STATUS", "")
                    p.ets_fusion = row.get("ETS_FUSION_SEQ", "")
                    p.ar_score = float(row.get("AR_SCORE", 0) or 0)
                    p.nepc_score = float(row.get("NEPC_SCORE", 0) or 0)
                    p.tmb = float(row.get("TMB_NONSYNONYMOUS", 0) or 0)
                    p.gleason = row.get("GLEASON_SCORE", "")
                    p.taxane_status = row.get("TAXANE_EXPOSURE_STATUS", "")
                    p.tissue_site = row.get("TISSUE_SITE", "")
                    p.arv7_expression = float(row.get("POLYA_ARV7_SRPM", 0) or 0)

        # Load mutations
        with open(data_dir / "data_mutations.txt") as f:
            reader = csv.DictReader(
                (line for line in f if not line.startswith("#")),
                delimiter="\t",
            )
            for row in reader:
                sample_id = row.get("Tumor_Sample_Barcode", "")
                # Find patient by sample
                for p in patient_map.values():
                    if p.sample_id == sample_id or sample_id.startswith(p.patient_id.replace("-", ".")):
                        p.mutations.append(row)
                        break

        self._patients = list(patient_map.values())
        logger.info("Loaded %d patients with mutations from flat files", len(self._patients))
        return self._patients

    #: Homologous-recombination repair genes. Ground truth for these is
    #: taken from the MAF directly, never from a Cognisom-derived field.
    HRD_GENES = frozenset({"BRCA1", "BRCA2", "ATM", "CDK12", "PALB2", "CHEK2"})

    #: MAF Variant_Classification values that alter protein and therefore
    #: count toward TMB_NONSYNONYMOUS. Derived empirically from this cohort
    #: -- see PROTEIN_ALTERING_CONSEQUENCES in genomics/vcf_parser.py.
    PROTEIN_ALTERING_CLASSES = frozenset({
        "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del",
        "Frame_Shift_Ins", "Translation_Start_Site", "Splice_Site",
        "Splice_Region", "In_Frame_Del", "In_Frame_Ins", "Nonstop_Mutation",
    })

    @classmethod
    def maf_hrd_ground_truth(cls, patient: SU2CPatient) -> bool:
        """Does the raw MAF show a protein-altering HRD-gene mutation?

        Read straight from Hugo_Symbol and Variant_Classification, so it
        is independent of every stage of the pipeline being measured.
        """
        return any(
            m.get("Hugo_Symbol", "").upper() in cls.HRD_GENES
            and m.get("Variant_Classification") in cls.PROTEIN_ALTERING_CLASSES
            for m in patient.mutations
        )

    @staticmethod
    def _pearson(a: List[float], b: List[float]) -> float:
        n = len(a)
        if n < 2:
            return 0.0
        mean_a, mean_b = sum(a) / n, sum(b) / n
        num = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
        da = math.sqrt(sum((x - mean_a) ** 2 for x in a))
        db = math.sqrt(sum((y - mean_b) ** 2 for y in b))
        return num / (da * db) if da and db else 0.0

    @classmethod
    def _spearman(cls, a: List[float], b: List[float]) -> float:
        def ranks(values: List[float]) -> List[float]:
            order = sorted(range(len(values)), key=lambda i: values[i])
            out = [0.0] * len(values)
            i = 0
            while i < len(order):
                j = i
                while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                    j += 1
                average = (i + j) / 2 + 1
                for k in range(i, j + 1):
                    out[order[k]] = average
                i = j + 1
            return out

        return cls._pearson(ranks(a), ranks(b))

    def compare_tmb(self, n_patients: int = 500) -> Dict[str, Any]:
        """Compare predicted TMB against the study's reported TMB.

        Runs only VCF ingest and annotation -- not neoantigen prediction or
        treatment simulation -- because TMB depends on nothing else, and
        keeping it cheap is what lets this run as a test rather than as a
        one-off someone quotes a number from afterwards.
        """
        from cognisom.genomics.variant_annotator import VariantAnnotator
        from cognisom.genomics.vcf_parser import VCFParser

        predicted: List[float] = []
        reported: List[float] = []

        for patient in self.load_patients()[:n_patients]:
            if not patient.tmb or patient.tmb <= 0 or not patient.mutations:
                continue
            variants = VCFParser().parse_text(self.mutations_to_vcf(patient))
            VariantAnnotator(cancer_type="prostate").annotate(variants)
            predicted.append(sum(1 for v in variants if v.is_coding) / 30.0)
            reported.append(patient.tmb)

        if not predicted:
            return {"n_compared": 0}

        errors = sorted(abs(p - r) for p, r in zip(predicted, reported))
        mid = len(errors) // 2
        median_error = (errors[mid] if len(errors) % 2
                        else (errors[mid - 1] + errors[mid]) / 2)

        tp = sum(1 for p, r in zip(predicted, reported) if p >= 10 and r >= 10)
        fp = sum(1 for p, r in zip(predicted, reported) if p >= 10 and r < 10)
        fn = sum(1 for p, r in zip(predicted, reported) if p < 10 and r >= 10)
        tn = sum(1 for p, r in zip(predicted, reported) if p < 10 and r < 10)

        return {
            "n_compared": len(predicted),
            "pearson_r": self._pearson(predicted, reported),
            "spearman_rho": self._spearman(predicted, reported),
            "median_abs_error": median_error,
            "within_25pct": sum(
                1 for p, r in zip(predicted, reported) if abs(p - r) <= 0.25 * r
            ) / len(predicted),
            "mean_ratio": sum(p / r for p, r in zip(predicted, reported)) / len(predicted),
            "tmb_high": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        }

    def mutations_to_vcf(self, patient: SU2CPatient) -> str:
        """Convert MAF mutations to VCF format."""
        lines = [
            "##fileformat=VCFv4.2",
            f"##source=SU2C_mCRPC_2019_{patient.patient_id}",
            "##reference=GRCh37",
            '##INFO=<ID=GENE,Number=1,Type=String,Description="Gene symbol">',
            '##INFO=<ID=CONSEQUENCE,Number=1,Type=String,Description="Variant consequence">',
            '##INFO=<ID=AA_CHANGE,Number=1,Type=String,Description="Amino acid change">',
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
            '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read depth">',
            '##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths">',
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE",
        ]

        for mut in patient.mutations:
            gene = mut.get("Hugo_Symbol", "")
            chrom = f"chr{mut.get('Chromosome', '')}"
            pos = mut.get("Start_Position", "0")
            ref = mut.get("Reference_Allele", "")
            alt = mut.get("Tumor_Seq_Allele2", "")
            protein = mut.get("HGVSp_Short", "")
            var_class = mut.get("Variant_Classification", "")
            t_ref = mut.get("t_ref_count", "50")
            t_alt = mut.get("t_alt_count", "25")

            if not ref or not alt or ref == "NA":
                continue

            # MAF writes indels with "-" for the absent allele; VCF needs a
            # shared padding base on both sides. Skipping these rows -- as
            # this did -- silently dropped every indel in the cohort: 15,399
            # rows, including 3,394 frameshifts, 557 in-frame deletions and
            # 490 splice-region variants. Frameshifts in BRCA2 and CDK12 are
            # a central part of this cohort's biology, so the validation was
            # not exercising the indel path at all, and TMB came out ~16%
            # low against the study's own reported values.
            #
            # The padding base is a placeholder: recovering the real one
            # needs the GRCh37 reference FASTA, which this validator does
            # not carry. That is sufficient here because everything
            # downstream keys on the annotation (Hugo_Symbol, consequence,
            # HGVSp_Short) rather than the genomic base -- but it means
            # these records must not be used for anything that reads the
            # reference sequence itself.
            if alt == "-":          # deletion
                start = int(pos) if str(pos).isdigit() else 0
                pos = str(max(1, start - 1))
                ref, alt = _PAD_BASE + ref, _PAD_BASE
            elif ref == "-":        # insertion
                ref, alt = _PAD_BASE, _PAD_BASE + alt

            # Map MAF classification to VCF consequence
            consequence_map = {
                "Missense_Mutation": "missense",
                "Nonsense_Mutation": "nonsense",
                "Frame_Shift_Del": "frameshift",
                "Frame_Shift_Ins": "frameshift",
                "Splice_Site": "splice",
                "In_Frame_Del": "inframe_deletion",
                "In_Frame_Ins": "inframe_insertion",
                "Silent": "synonymous",
                "Nonstop_Mutation": "nonstop",
                "Translation_Start_Site": "start_lost",
            }
            consequence = consequence_map.get(var_class, var_class.lower())

            info_parts = [f"GENE={gene}", f"CONSEQUENCE={consequence}"]
            if protein:
                info_parts.append(f"AA_CHANGE={protein}")
            info = ";".join(info_parts)

            dp = int(t_ref or 0) + int(t_alt or 0)
            ad = f"{t_ref or 0},{t_alt or 0}"

            line = f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t99\tPASS\t{info}\tGT:DP:AD\t0/1:{dp}:{ad}"
            lines.append(line)

        return "\n".join(lines)

    def validate_patient(self, patient: SU2CPatient) -> SU2CValidationResult:
        """Run one SU2C patient through Cognisom."""
        result = SU2CValidationResult(
            patient_id=patient.patient_id,
            chemo_regimen=patient.chemo_regimen,
            os_months=patient.os_months,
            os_status=patient.os_status,
            abi_enza_status=patient.abi_enza_status,
            actual_tmb=patient.tmb,
            ets_fusion=patient.ets_fusion,
        )

        t0 = time.time()
        try:
            vcf_text = self.mutations_to_vcf(patient)

            if not patient.mutations:
                result.status = "completed"
                result.error = "No mutations"
                result.processing_seconds = time.time() - t0
                return result

            # Run Cognisom pipeline
            from cognisom.genomics.patient_profile import PatientProfileBuilder
            builder = PatientProfileBuilder()
            profile = builder.from_vcf_text(vcf_text, patient_id=patient.patient_id)

            result.n_mutations_parsed = len(profile.variants)
            result.n_drivers = len(profile.cancer_driver_mutations)
            result.predicted_tmb = profile.tumor_mutational_burden
            result.n_neoantigens = len(profile.predicted_neoantigens)
            result.predicted_vaccine_eligible = profile.neoantigen_vaccine_candidate
            result.driver_genes_found = sorted(set(
                v.gene for v in profile.cancer_driver_mutations if v.gene
            ))
            result.parp_candidate = profile.parp_inhibitor_candidate
            result.immunotherapy_candidate = profile.immunotherapy_candidate
            result.ar_mutation = profile.has_ar_mutation

            # Digital twin
            from cognisom.genomics.twin_config import DigitalTwinConfig
            twin = DigitalTwinConfig.from_profile_only(profile)
            result.predicted_mhc1_downreg = twin.mhc1_downregulation

            # Treatment simulation
            from cognisom.library.store import EntityStore
            from cognisom.genomics.treatment_simulator import TreatmentSimulator
            store = EntityStore()
            sim = TreatmentSimulator(store=store)
            recommended = sim.get_recommended_treatments(twin)
            treatments = sim.compare_treatments(recommended[:6], twin, 180)
            result.n_treatments = len(treatments)

            if treatments:
                best = min(treatments, key=lambda t: t.best_response)
                result.predicted_best_treatment = best.treatment_name
                result.predicted_best_response = best.response_category

            result.status = "completed"

        except Exception as e:
            result.status = "failed"
            result.error = str(e)[:200]
            logger.debug("Patient %s failed: %s", patient.patient_id, e)

        result.processing_seconds = time.time() - t0
        return result

    def run_validation(self, n_patients: int = 444,
                       progress_callback=None) -> SU2CValidationSummary:
        """Run full SU2C validation.

        Args:
            n_patients: Number of patients (max 444)
            progress_callback: Called with (current, total, patient_id, status)
        """
        summary = SU2CValidationSummary()
        t0 = time.time()

        if progress_callback:
            progress_callback(0, n_patients, "Loading SU2C data...", "running")

        patients = self.load_patients()[:n_patients]
        summary.n_patients = len(patients)

        # Track regimen distribution from ground truth
        for p in patients:
            if p.chemo_regimen:
                summary.regimen_distribution[p.chemo_regimen] = \
                    summary.regimen_distribution.get(p.chemo_regimen, 0) + 1

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
            if result.predicted_best_treatment:
                summary.treatment_distribution[result.predicted_best_treatment] = \
                    summary.treatment_distribution.get(result.predicted_best_treatment, 0) + 1
            if result.predicted_best_response:
                summary.response_distribution[result.predicted_best_response] = \
                    summary.response_distribution.get(result.predicted_best_response, 0) + 1

            # Track driver frequency
            for gene in result.driver_genes_found:
                summary.driver_frequency[gene] = summary.driver_frequency.get(gene, 0) + 1

            # Concordance: BRCA patients flagged as PARP candidates.
            # Retained for continuity, but see the note on the summary
            # fields -- this comparison is circular and always reports
            # 100%. The HRD confusion matrix below is the real measure.
            if any(g in result.driver_genes_found for g in ["BRCA1", "BRCA2", "ATM", "CDK12"]):
                summary.total_brca_patients += 1
                if result.parp_candidate:
                    summary.parp_candidates_with_brca += 1

            # Real concordance: ground truth from the MAF, prediction from
            # the pipeline. These come from different places, so this can
            # actually be wrong -- which is what makes it worth measuring.
            truth = self.maf_hrd_ground_truth(patient)
            predicted_hrd = result.parp_candidate
            if truth and predicted_hrd:
                summary.hrd_true_positive += 1
            elif truth and not predicted_hrd:
                summary.hrd_false_negative += 1
            elif not truth and predicted_hrd:
                summary.hrd_false_positive += 1
            else:
                summary.hrd_true_negative += 1

            # Concordance: AR mutations in AR-treated patients
            if "abiraterone" in patient.chemo_regimen.lower() or \
               "enzalutamide" in patient.chemo_regimen.lower():
                summary.total_ar_treated += 1
                if result.ar_mutation:
                    summary.ar_mutations_detected += 1

        # Aggregates
        completed = [r for r in summary.results
                     if r.status == "completed" and r.n_mutations_parsed > 0]
        if completed:
            summary.mean_mutations = sum(r.n_mutations_parsed for r in completed) / len(completed)
            summary.mean_drivers = sum(r.n_drivers for r in completed) / len(completed)
            summary.mean_tmb_predicted = sum(r.predicted_tmb for r in completed) / len(completed)
            summary.mean_tmb_actual = sum(r.actual_tmb for r in completed if r.actual_tmb > 0) / max(1, sum(1 for r in completed if r.actual_tmb > 0))
            summary.mean_neoantigens = sum(r.n_neoantigens for r in completed) / len(completed)
            summary.mean_processing_time = sum(r.processing_seconds for r in completed) / len(completed)

        # TMB agreement, computed rather than asserted.
        tmb = self.compare_tmb(n_patients=n_patients)
        if tmb.get("n_compared"):
            summary.tmb_n_compared = tmb["n_compared"]
            summary.tmb_pearson_r = tmb["pearson_r"]
            summary.tmb_spearman_rho = tmb["spearman_rho"]
            summary.tmb_median_abs_error = tmb["median_abs_error"]
            summary.tmb_within_25pct = tmb["within_25pct"]
            summary.tmb_high_true_positive = tmb["tmb_high"]["tp"]
            summary.tmb_high_false_positive = tmb["tmb_high"]["fp"]
            summary.tmb_high_false_negative = tmb["tmb_high"]["fn"]
            summary.tmb_high_true_negative = tmb["tmb_high"]["tn"]

        summary.total_time_seconds = time.time() - t0

        if progress_callback:
            progress_callback(len(patients), len(patients), "Validation complete", "completed")

        logger.info(
            "SU2C validation: %d/%d completed in %.0fs, "
            "mean %.0f mutations, %.1f drivers, "
            "PARP concordance: %d/%d BRCA patients flagged",
            summary.n_completed, summary.n_patients, summary.total_time_seconds,
            summary.mean_mutations, summary.mean_drivers,
            summary.parp_candidates_with_brca, summary.total_brca_patients,
        )
        return summary
