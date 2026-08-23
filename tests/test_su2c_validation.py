"""
Validation against the SU2C/PCF Dream Team mCRPC cohort (Abida et al.,
PNAS 2019; 116:11428-11436), via the cBioPortal flat-file export.

Two claims were carried on the investor page, in the dashboard and in
mad/compliance.py -- "TMB r=0.987" and "100% biomarker concordance"
across 429 patients -- and neither could be checked. Nothing in this
repository computed a correlation at all: SU2CValidationSummary carried
`mean_tmb_predicted` and `mean_tmb_actual`, two means and no r. The
concordance was computed, but circularly: its denominator was "patient
has a driver mutation in BRCA1/2, ATM or CDK12" and its numerator was
`parp_candidate`, which is *defined* as carrying a mutation in those
genes plus PALB2 and CHEK2, so it reported 100% whatever the pipeline
did.

That tautology was actively harmful, not merely uninformative. MAF
encodes indels with "-" for the absent allele, and mutations_to_vcf
skipped those rows rather than converting them -- dropping every indel
in the cohort, 15,399 rows including 3,394 frameshifts. 42 of the 95
HRD-positive patients here are HRD-positive *only* through an indel, so
the pipeline was missing 44% of the PARP-inhibitor candidates in this
cohort while the concordance metric read 100%.

The cohort is a 98 MB download and is not vendored. These tests skip
without it; the MAF/VCF conversion tests below need no data and always
run, because that is where the regression was.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cognisom.genomics.vcf_parser import VCFParser, is_protein_altering
from cognisom.validation.su2c_file_validator import (
    SU2CFileValidator,
    SU2CPatient,
)


def _data_dir():
    """Locate the extracted cohort, if it is present on this machine."""
    for candidate in (
        os.environ.get("COGNISOM_SU2C_DIR"),
        "/tmp/cbio_data/prad_su2c_2019",
    ):
        if candidate and (Path(candidate) / "data_mutations.txt").exists():
            return candidate
    return None


requires_cohort = pytest.mark.skipif(
    _data_dir() is None,
    reason=(
        "SU2C cohort not present. Download prad_su2c_2019.tar.gz from "
        "datahub.assets.cbioportal.org and set COGNISOM_SU2C_DIR."
    ),
)


# ── MAF -> VCF conversion (no cohort needed) ────────────────────────

def _patient(*mutations) -> SU2CPatient:
    return SU2CPatient(patient_id="TEST", mutations=list(mutations))


def _maf_row(**kwargs):
    row = {
        "Hugo_Symbol": "BRCA2", "Chromosome": "13",
        "Start_Position": "32914438", "Reference_Allele": "A",
        "Tumor_Seq_Allele2": "T", "HGVSp_Short": "p.E1143K",
        "Variant_Classification": "Missense_Mutation",
        "t_ref_count": "50", "t_alt_count": "25",
    }
    row.update(kwargs)
    return row


def test_a_deletion_survives_conversion():
    """MAF writes deletions as alt='-'; these were dropped outright."""
    validator = SU2CFileValidator(data_dir="/nonexistent")
    vcf = validator.mutations_to_vcf(_patient(_maf_row(
        Reference_Allele="ACTG", Tumor_Seq_Allele2="-",
        Variant_Classification="Frame_Shift_Del",
        HGVSp_Short="p.E1143fs",
    )))

    variants = VCFParser().parse_text(vcf)
    assert len(variants) == 1
    variant = variants[0]
    assert variant.is_deletion
    # Four bases removed, and the padding base is shared by REF and ALT.
    assert len(variant.ref) - len(variant.alt) == 4
    assert variant.gene == "BRCA2"
    assert variant.is_coding is True


def test_an_insertion_survives_conversion():
    validator = SU2CFileValidator(data_dir="/nonexistent")
    vcf = validator.mutations_to_vcf(_patient(_maf_row(
        Reference_Allele="-", Tumor_Seq_Allele2="GG",
        Variant_Classification="Frame_Shift_Ins",
        HGVSp_Short="p.L780fs",
    )))

    variants = VCFParser().parse_text(vcf)
    assert len(variants) == 1
    assert variants[0].is_insertion
    assert len(variants[0].alt) - len(variants[0].ref) == 2
    assert variants[0].is_coding is True


def test_a_deletion_is_anchored_one_base_earlier():
    """VCF indels carry a padding base, so POS shifts back by one."""
    validator = SU2CFileValidator(data_dir="/nonexistent")
    vcf = validator.mutations_to_vcf(_patient(_maf_row(
        Reference_Allele="AC", Tumor_Seq_Allele2="-",
        Start_Position="1000", Variant_Classification="In_Frame_Del",
    )))
    assert VCFParser().parse_text(vcf)[0].pos == 999


def test_substitutions_are_unaffected_by_the_indel_handling():
    validator = SU2CFileValidator(data_dir="/nonexistent")
    vcf = validator.mutations_to_vcf(_patient(_maf_row()))
    variant = VCFParser().parse_text(vcf)[0]

    assert variant.pos == 32914438
    assert (variant.ref, variant.alt) == ("A", "T")
    assert variant.protein_change == "p.E1143K"


def test_rows_with_no_usable_alleles_are_still_skipped():
    validator = SU2CFileValidator(data_dir="/nonexistent")
    vcf = validator.mutations_to_vcf(_patient(
        _maf_row(Reference_Allele="", Tumor_Seq_Allele2=""),
        _maf_row(Reference_Allele="NA", Tumor_Seq_Allele2="T"),
    ))
    assert VCFParser().parse_text(vcf) == []


# ── The consequence set that defines TMB ────────────────────────────

@pytest.mark.parametrize("consequence", [
    "missense", "nonsense", "frameshift", "splice", "splice_region",
    "inframe_deletion", "inframe_insertion", "nonstop", "start_lost",
])
def test_protein_altering_consequences_count_toward_tmb(consequence):
    """Dropping splice/in-frame/nonstop understated TMB by ~16%."""
    assert is_protein_altering(consequence) is True


@pytest.mark.parametrize("consequence", [
    "synonymous", "intron", "3'utr", "5'utr", "5'flank", "igr", "rna", "",
])
def test_non_coding_consequences_do_not_count(consequence):
    assert is_protein_altering(consequence) is False


def test_hrd_ground_truth_reads_the_maf_not_the_pipeline():
    """The measure must be independent of what it measures."""
    positive = _patient(_maf_row(
        Hugo_Symbol="CDK12", Variant_Classification="Frame_Shift_Del",
        Reference_Allele="AC", Tumor_Seq_Allele2="-",
    ))
    assert SU2CFileValidator.maf_hrd_ground_truth(positive) is True

    # Right gene, but a silent variant alters no protein.
    silent = _patient(_maf_row(Hugo_Symbol="BRCA2",
                               Variant_Classification="Silent"))
    assert SU2CFileValidator.maf_hrd_ground_truth(silent) is False

    # Protein-altering, but not an HRD gene.
    other = _patient(_maf_row(Hugo_Symbol="TP53"))
    assert SU2CFileValidator.maf_hrd_ground_truth(other) is False


# ── Against the real cohort ─────────────────────────────────────────

@pytest.fixture(scope="module")
def validator():
    return SU2CFileValidator(data_dir=_data_dir())


@requires_cohort
def test_cohort_is_the_429_patients_the_claim_cites(validator):
    assert len(validator.load_patients()) == 429


@requires_cohort
@pytest.mark.slow
def test_tmb_agrees_with_the_study_reported_values(validator):
    """The "r=0.987" claim, now computed rather than asserted.

    It reproduces -- and improves to 0.9998 once indels are converted
    instead of dropped. What it measures is agreement between this
    pipeline's TMB and cBioPortal's TMB_NONSYNONYMOUS over the same MAF,
    so it validates ingest and consequence classification. It is not a
    prediction of an independent outcome and must not be quoted as one.
    """
    result = validator.compare_tmb()

    assert result["n_compared"] == 427  # 429 patients, 2 without TMB
    assert result["pearson_r"] > 0.99
    assert result["spearman_rho"] > 0.99
    # No systematic bias: dropping indels used to leave TMB ~16% low.
    assert 0.95 < result["mean_ratio"] < 1.05
    assert result["median_abs_error"] < 0.1
    assert result["within_25pct"] > 0.95


@requires_cohort
@pytest.mark.slow
def test_tmb_high_calls_agree_at_the_threshold_that_drives_treatment(validator):
    """10 mut/Mb is the line GenomicsAgent scores checkpoint inhibitors on."""
    confusion = validator.compare_tmb()["tmb_high"]

    assert confusion["tp"] > 0
    assert confusion["fp"] == 0
    assert confusion["fn"] == 0


@requires_cohort
@pytest.mark.slow
def test_hrd_detection_recovers_every_patient_in_the_maf(validator):
    """Non-circular: ground truth from the MAF, prediction from the pipeline.

    42 of the 95 HRD-positive patients here carry their only
    PARP-relevant finding as an indel, so this fails outright if indels
    are dropped -- which is exactly what the circular metric it replaces
    could never show.
    """
    from cognisom.genomics.variant_annotator import VariantAnnotator

    hrd_genes = SU2CFileValidator.HRD_GENES
    tp = fp = fn = tn = 0

    for patient in validator.load_patients():
        truth = validator.maf_hrd_ground_truth(patient)
        variants = VCFParser().parse_text(validator.mutations_to_vcf(patient))
        VariantAnnotator(cancer_type="prostate").annotate(variants)
        genes = {v.gene for v in variants if v.is_coding and v.gene}
        predicted = bool(hrd_genes & genes)

        if truth and predicted:
            tp += 1
        elif truth:
            fn += 1
        elif predicted:
            fp += 1
        else:
            tn += 1

    assert tp + fn == 95, "cohort HRD prevalence changed"
    assert fn == 0, f"{fn} HRD patients missed"
    assert fp == 0, f"{fp} patients wrongly flagged"
