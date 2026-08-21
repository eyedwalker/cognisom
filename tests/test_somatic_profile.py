"""
The tumor/normal comparison, wired through to the patient profile.

Parsing both sides of a matched callset only matters if the pipeline
then acts on it. It did not: ``_build_profile`` went straight from
annotation to ``[v for v in variants if v.is_coding]``, so inherited
variants were carried into the driver list and into TMB.

On the SEQC2 HCC1395 callset this repo ships -- raw Parabricks Mutect2
output, never passed through FilterMutectCalls -- that inflated TMB
about sevenfold, from 7.1 to 49.8 mutations/Mb. The 10 mut/Mb line is
the one ``GenomicsAgent`` scores checkpoint inhibitors on
(``if tmb >= 10.0: ck_score += 0.35``), so the germline and the
unfiltered artifacts were what pushed this patient over the
immunotherapy threshold.

These tests pin the comparison, the exclusion, and the provenance the
profile now carries about both.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cognisom.genomics import gene_protein_mapper as gpm
from cognisom.genomics.patient_profile import PatientProfileBuilder
from cognisom.genomics.variant_annotator import VariantAnnotator
from cognisom.genomics.vcf_parser import VCFParser

SEQC2_VCF = REPO_ROOT / "cognisom" / "genomics" / "seqc2_demo.vcf"

TUMOR_NORMAL_VCF = "\n".join([
    "##fileformat=VCFv4.2",
    "##source=Mutect2",
    "##normal_sample=N",
    "##tumor_sample=T",
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tN\tT",
    # Somatic driver.
    "chr12\t25398284\t.\tC\tA\t.\t.\t"
    "GENE=KRAS;CONSEQUENCE=missense;AA_CHANGE=p.G12V;TLOD=40.0;NLOD=9.0;NALOD=2.0\t"
    "GT:AD:DP\t0/0:50,0:50\t0/1:60,40:100",
    # Inherited: the normal carries it at ~50%.
    "chr17\t7578406\t.\tC\tT\t.\t.\t"
    "GENE=TP53;CONSEQUENCE=missense;AA_CHANGE=p.R248W;TLOD=40.0;NLOD=0.5;NALOD=2.0\t"
    "GT:AD:DP\t0/1:25,25:50\t0/1:50,50:100",
])

SINGLE_SAMPLE_VCF = "\n".join([
    "##fileformat=VCFv4.2",
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE",
    "chr12\t25398284\t.\tC\tA\t99\tPASS\t"
    "GENE=KRAS;CONSEQUENCE=missense;AA_CHANGE=p.G12V\tGT:AD:DP\t0/1:40,60:100",
])


@pytest.fixture(autouse=True)
def offline_uniprot(monkeypatch):
    """Keep these tests off the network.

    Genes outside the built-in cache otherwise trigger a UniProt fetch
    per gene, which turns a unit test into a timeout when the host has
    no route to rest.uniprot.org.
    """
    monkeypatch.setattr(
        gpm.GeneProteinMapper, "_fetch_from_uniprot", lambda self, gene: None
    )


# ── The comparison happens ──────────────────────────────────────────

def test_profile_records_that_a_matched_normal_was_present():
    profile = PatientProfileBuilder().from_vcf_text(TUMOR_NORMAL_VCF, "p1")

    assert profile.has_matched_normal is True
    assert profile.tumor_sample == "T"
    assert profile.normal_sample == "N"


def test_inherited_variant_is_excluded_from_the_driver_set():
    profile = PatientProfileBuilder().from_vcf_text(TUMOR_NORMAL_VCF, "p1")

    genes = {v.gene for v in profile.coding_variants}
    assert "KRAS" in genes          # somatic
    assert "TP53" not in genes      # carried by the normal
    assert profile.germline_variants_excluded == 1
    assert profile.somatic_status_counts["SOMATIC"] == 1
    assert profile.somatic_status_counts["GERMLINE"] == 1


def test_tumor_only_callset_is_flagged_and_nothing_is_excluded():
    """No normal means the question was never asked, not that it passed."""
    profile = PatientProfileBuilder().from_vcf_text(SINGLE_SAMPLE_VCF, "p2")

    assert profile.has_matched_normal is False
    assert profile.normal_sample is None
    assert profile.germline_variants_excluded == 0
    # The variant is still carried forward -- a tumor-only run is usable,
    # it just cannot claim its calls are somatic.
    assert {v.gene for v in profile.coding_variants} == {"KRAS"}
    assert profile.somatic_status_counts == {"UNKNOWN": 1}


# ── Provenance is serialized ────────────────────────────────────────

def test_to_dict_carries_the_comparison_provenance():
    payload = PatientProfileBuilder().from_vcf_text(
        TUMOR_NORMAL_VCF, "p1"
    ).to_dict()

    assert payload["has_matched_normal"] is True
    assert payload["tumor_sample"] == "T"
    assert payload["normal_sample"] == "N"
    assert payload["germline_variants_excluded"] == 1
    assert payload["somatic_status_counts"]["GERMLINE"] == 1
    # An empty neoantigen list must be explicable.
    assert "neoantigen_diagnostics" in payload
    assert "binding_method" in payload


# ── The SEQC2 regression ────────────────────────────────────────────

@pytest.mark.skipif(not SEQC2_VCF.exists(), reason="SEQC2 demo VCF not present")
def test_germline_contamination_no_longer_inflates_the_seqc2_variant_set():
    """Two defects stacked here, and both are now closed.

    The germline was counted as tumor, which inflated everything
    downstream. Separately, coding status was invented -- any coordinate
    inside a driver gene's whole-gene span was marked coding, introns and
    UTRs included -- which is what turned that inflated variant count
    into a TMB of 49.8 and crossed the checkpoint-inhibitor threshold.

    With coding status no longer fabricated this raw callset yields no
    coding variants at all, so the honest statement about its TMB is that
    it cannot be computed. The germline exclusion is still checked here,
    at the variant level where it is real.
    """
    parser = VCFParser()
    variants = parser.parse_file(str(SEQC2_VCF))
    annotator = VariantAnnotator(cancer_type="breast")
    annotator.annotate(variants)

    somatic = parser.filter_somatic(variants)

    # Most of the raw callset does not survive the tumor/normal test.
    assert 0 < len(somatic) < len(variants) * 0.5

    # And nothing in this file is annotated, so no coding call is possible.
    assert annotator.variants_with_consequence == 0
    assert not any(v.is_coding for v in variants)


@pytest.mark.skipif(not SEQC2_VCF.exists(), reason="SEQC2 demo VCF not present")
def test_unannotated_callset_reports_tmb_as_not_estimable():
    """TMB 0.0 must not be readable as 'low'."""
    profile = PatientProfileBuilder(cancer_type="breast").from_vcf_file(
        str(SEQC2_VCF), patient_id="SEQC2"
    )

    assert profile.tmb_is_estimable is False
    assert profile.is_tmb_high is False
    assert profile.to_dict()["tmb_is_estimable"] is False


def test_annotated_callset_reports_tmb_as_estimable():
    """The synthetic corpus carries CONSEQUENCE/AA_CHANGE, so it does."""
    profile = PatientProfileBuilder().from_vcf_text(TUMOR_NORMAL_VCF, "p1")

    assert profile.tmb_is_estimable is True


@pytest.mark.skipif(not SEQC2_VCF.exists(), reason="SEQC2 demo VCF not present")
def test_seqc2_profile_uses_the_somatic_subset():
    profile = PatientProfileBuilder(cancer_type="breast").from_vcf_file(
        str(SEQC2_VCF), patient_id="SEQC2"
    )

    assert profile.has_matched_normal is True
    assert profile.is_tmb_high is False
    assert profile.germline_variants_excluded > 0
    # Drivers are drawn from the somatic set only.
    assert len(profile.cancer_driver_mutations) < len(profile.variants)
    assert all(v.is_somatic for v in profile.cancer_driver_mutations)
