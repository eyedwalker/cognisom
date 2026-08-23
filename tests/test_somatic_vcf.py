"""
Tumor/normal (somatic) VCF parsing.

The neoantigen pipeline's central claim is that it compares a tumor
against the patient's own germline and reports what is *new* in the
tumor. That claim rests entirely on reading the right sample column and
then actually performing the comparison.

Both halves were previously missing. ``VCFParser`` read ``fields[9]`` --
the first sample column -- and the SEQC2 HCC1395 callset this repo ships
is ordered ``normal`` first, so the "patient genotype" shown in the
dashboard was the germline. Nothing compared the two sides at all.

These tests pin the behaviour that fixes it. The inline fixtures below
are deliberately hand-written rather than loaded from disk so the
expected values can be computed by hand from the read counts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cognisom.genomics.vcf_parser import (
    MAX_NORMAL_AF,
    MAX_NORMAL_ALT_COUNT,
    NORMAL_LOD_THRESHOLD,
    TUMOR_LOD_THRESHOLD,
    VCFParser,
)

SEQC2_VCF = REPO_ROOT / "cognisom" / "genomics" / "seqc2_demo.vcf"


# ── Fixtures ────────────────────────────────────────────────────────

# Mutect2-style callset. Note the column order: `normal` comes FIRST,
# exactly as Parabricks emitted it for the SEQC2 run. A parser that
# assumes the tumor is column 9 reads the normal here.
TUMOR_NORMAL_VCF = "\n".join([
    "##fileformat=VCFv4.2",
    "##source=Mutect2",
    "##normal_sample=NORMAL_A",
    "##tumor_sample=TUMOR_A",
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tNORMAL_A\tTUMOR_A",
    # Clean somatic: 0 alt reads in normal, strong TLOD, normal is hom-ref.
    "chr1\t100\t.\tA\tT\t.\t.\tTLOD=40.0;NLOD=9.0;NALOD=2.0\t"
    "GT:AD:AF:DP\t0/0:50,0:0.03:50\t0/1:60,40:0.400:100",
    # Germline: normal carries the allele at ~50%.
    "chr1\t200\t.\tC\tG\t.\t.\tTLOD=40.0;NLOD=0.5;NALOD=2.0\t"
    "GT:AD:AF:DP\t0/1:25,25:0.03:50\t0/1:50,50:0.500:100",
    # Real in the tumor but below the high-confidence LOD_T bar.
    "chr1\t300\t.\tG\tA\t.\t.\tTLOD=4.0;NLOD=9.0;NALOD=2.0\t"
    "GT:AD:AF:DP\t0/0:50,0:0.03:50\t0/1:90,10:0.100:100",
    # Artifact seen in the normal (NALOD <= 0).
    "chr1\t400\t.\tT\tC\t.\t.\tTLOD=40.0;NLOD=9.0;NALOD=-1.5\t"
    "GT:AD:AF:DP\t0/0:50,0:0.03:50\t0/1:80,20:0.200:100",
])

# Single-sample germline callset (GATK/DeepVariant shape): no normal to
# compare against, and no AF field -- AF must be derived from AD.
SINGLE_SAMPLE_VCF = "\n".join([
    "##fileformat=VCFv4.2",
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE",
    "chr7\t500\t.\tA\tG\t99\tPASS\tGENE=EGFR\tGT:AD:DP\t0/1:40,60:100",
])

# Multi-allelic site. AD is Number=R -> [ref, alt1, alt2];
# AF is Number=A -> [alt1, alt2]. Splitting the site must index both.
MULTIALLELIC_VCF = "\n".join([
    "##fileformat=VCFv4.2",
    "##source=Mutect2",
    "##normal_sample=N",
    "##tumor_sample=T",
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tN\tT",
    "chr2\t900\t.\tA\tG,T\t.\t.\tTLOD=30.0,12.0;NLOD=9.0,9.0;NALOD=2.0,2.0\t"
    "GT:AD:AF:DP\t0/0:50,0,0:0.03,0.03:50\t0/1/2:40,35,25:0.350,0.250:100",
])


def parse(text: str):
    parser = VCFParser()
    return parser, parser.parse_text(text)


# ── Sample column resolution ────────────────────────────────────────

def test_tumor_column_resolved_by_name_not_position():
    """The tumor is found via ##tumor_sample, even when listed second."""
    parser, variants = parse(TUMOR_NORMAL_VCF)

    assert parser.sample_names == ["NORMAL_A", "TUMOR_A"]
    assert parser.tumor_sample == "TUMOR_A"
    assert parser.normal_sample == "NORMAL_A"
    assert parser.is_somatic_vcf is True


def test_genotype_comes_from_tumor_not_the_first_column():
    """Regression: `fields[9]` here is the NORMAL, whose GT is 0/0.

    If this ever reads 0/0 again, the dashboard is showing the patient's
    germline genotype and calling it the tumor.
    """
    _, variants = parse(TUMOR_NORMAL_VCF)
    v = variants[0]

    assert v.tumor_genotype == "0/1"
    assert v.normal_genotype == "0/0"
    # The long-standing public field must track the tumor.
    assert v.genotype == "0/1"


def test_both_sides_of_the_comparison_are_populated():
    _, variants = parse(TUMOR_NORMAL_VCF)
    v = variants[0]

    assert v.tumor_af == pytest.approx(0.400)
    assert v.tumor_ref_count == 60
    assert v.tumor_alt_count == 40
    assert v.tumor_depth == 100

    assert v.normal_alt_count == 0
    assert v.normal_ref_count == 50
    assert v.normal_depth == 50


def test_normal_af_is_derived_from_read_counts_not_the_af_field():
    """Mutect2's FORMAT/AF is 'allele fractions ... in the tumor'.

    It emits that same value in the normal column, so at a site with zero
    alt reads in the normal the field still reads 0.03. Trusting it there
    invents germline evidence: 0.03 sits exactly on MAX_NORMAL_AF.
    """
    _, variants = parse(TUMOR_NORMAL_VCF)
    v = variants[0]

    assert v.info == {} or "TLOD" in v.info  # sanity: INFO parsed
    assert v.normal_alt_count == 0
    # Observed fraction is 0/50, not the 0.03 the file claims.
    assert v.normal_af == pytest.approx(0.0)


# ── Somatic classification ──────────────────────────────────────────

def test_clean_somatic_call_is_classified_somatic():
    _, variants = parse(TUMOR_NORMAL_VCF)
    v = variants[0]

    assert v.somatic_status == "SOMATIC"
    assert v.somatic_filter_reasons == []
    assert v.is_somatic is True


def test_allele_present_in_normal_is_germline():
    _, variants = parse(TUMOR_NORMAL_VCF)
    v = variants[1]

    assert v.somatic_status == "GERMLINE"
    assert v.is_somatic is False
    assert "alt_allele_in_normal" in v.somatic_filter_reasons
    assert f"normal_lod<{NORMAL_LOD_THRESHOLD}" in v.somatic_filter_reasons


def test_weak_tumor_evidence_is_low_evidence_not_somatic():
    _, variants = parse(TUMOR_NORMAL_VCF)
    v = variants[2]

    assert v.somatic_status == "LOW_EVIDENCE"
    assert v.is_somatic is False
    assert f"tumor_lod<{TUMOR_LOD_THRESHOLD}" in v.somatic_filter_reasons


def test_normal_artifact_is_classified_artifact():
    _, variants = parse(TUMOR_NORMAL_VCF)
    v = variants[3]

    assert v.somatic_status == "ARTIFACT"
    assert "normal_artifact" in v.somatic_filter_reasons


def test_single_alt_read_in_normal_does_not_make_a_call_germline():
    """The count and fraction tests are a conjunction, deliberately.

    One alt read in a 30x normal is 3.3% by arithmetic alone, so a
    fraction-only test rejects most true somatic calls as inherited.
    """
    vcf = "\n".join([
        "##fileformat=VCFv4.2",
        "##normal_sample=N",
        "##tumor_sample=T",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tN\tT",
        "chr1\t100\t.\tA\tT\t.\t.\tTLOD=40.0;NLOD=9.0;NALOD=2.0\t"
        "GT:AD:DP\t0/0:29,1:30\t0/1:60,40:100",
    ])
    _, variants = parse(vcf)
    v = variants[0]

    assert v.normal_alt_count == 1
    assert v.normal_af > MAX_NORMAL_AF  # 1/30 = 0.033
    assert v.normal_alt_count < MAX_NORMAL_ALT_COUNT
    assert v.somatic_status == "SOMATIC"


# ── Tumor-only / single-sample callsets ─────────────────────────────

def test_single_sample_vcf_cannot_answer_the_somatic_question():
    """No normal => UNKNOWN, never SOMATIC.

    A tumor-only callset cannot separate a somatic mutation from a rare
    inherited one. Calling those somatic is how a self antigen ends up in
    a vaccine design.
    """
    parser, variants = parse(SINGLE_SAMPLE_VCF)

    assert parser.is_somatic_vcf is False
    v = variants[0]
    assert v.somatic_status == "UNKNOWN"
    assert v.is_somatic is False
    assert "no_matched_normal" in v.somatic_filter_reasons


def test_single_sample_af_is_derived_from_ad():
    """HaplotypeCaller/DeepVariant emit AD but no AF."""
    _, variants = parse(SINGLE_SAMPLE_VCF)
    v = variants[0]

    assert v.tumor_af == pytest.approx(0.6)  # 60 / (40 + 60)
    assert v.genotype == "0/1"


def test_filter_somatic_excludes_unknown_by_default():
    parser, variants = parse(SINGLE_SAMPLE_VCF)

    assert parser.filter_somatic(variants) == []
    assert len(parser.filter_somatic(variants, include_unknown=True)) == 1


# ── Multi-allelic splitting ─────────────────────────────────────────

def test_multiallelic_site_indexes_per_allele_fields():
    """Each ALT must get its own AD/AF/TLOD, not allele 1's values."""
    _, variants = parse(MULTIALLELIC_VCF)

    assert len(variants) == 2
    alt_g, alt_t = variants

    assert alt_g.alt == "G"
    assert alt_g.tumor_alt_count == 35          # AD[1]
    assert alt_g.tumor_af == pytest.approx(0.350)
    assert alt_g.somatic_evidence.tumor_lod == pytest.approx(30.0)

    assert alt_t.alt == "T"
    assert alt_t.tumor_alt_count == 25          # AD[2]
    assert alt_t.tumor_af == pytest.approx(0.250)
    assert alt_t.somatic_evidence.tumor_lod == pytest.approx(12.0)

    # Both share the REF count, which is AD[0].
    assert alt_g.tumor_ref_count == alt_t.tumor_ref_count == 40


# ── Filter honesty ──────────────────────────────────────────────────

def test_unfiltered_callset_is_reported_as_unfiltered():
    """FILTER '.' means 'not assessed', which is not the same as PASS."""
    parser, _ = parse(TUMOR_NORMAL_VCF)
    assert parser.filters_applied is False

    parser2, _ = parse(SINGLE_SAMPLE_VCF)
    assert parser2.filters_applied is True


def test_variant_summary_reports_the_tumor_normal_verdict():
    parser, variants = parse(TUMOR_NORMAL_VCF)
    summary = parser.variant_summary(variants)

    assert summary["tumor_sample"] == "TUMOR_A"
    assert summary["normal_sample"] == "NORMAL_A"
    assert summary["filters_applied"] is False
    assert summary["somatic_variants"] == 1
    assert summary["by_somatic_status"] == {
        "SOMATIC": 1, "GERMLINE": 1, "LOW_EVIDENCE": 1, "ARTIFACT": 1,
    }


# ── The shipped SEQC2 callset ───────────────────────────────────────

@pytest.mark.skipif(not SEQC2_VCF.exists(), reason="SEQC2 demo VCF not present")
def test_seqc2_demo_vcf_is_read_tumor_side_up():
    """Integration check against the real Parabricks output we ship.

    The file is ordered normal-then-tumor, so this is the case the old
    positional read got backwards.
    """
    parser = VCFParser()
    variants = parser.parse_file(str(SEQC2_VCF))

    assert parser.sample_names == ["normal", "sample"]
    assert parser.tumor_sample == "sample"
    assert parser.normal_sample == "normal"

    # Parabricks emitted this without FilterMutectCalls.
    assert parser.filters_applied is False

    # Every record has both sides populated.
    assert all(v.tumor_af is not None for v in variants)
    assert all(v.normal_depth is not None for v in variants)

    # The raw callset is emitted down to TLOD>=3.0, so most of it must not
    # survive the somatic bar. If this ever approaches the full 1495, the
    # comparison has stopped happening.
    somatic = parser.filter_somatic(variants)
    assert 0 < len(somatic) < len(variants) * 0.5

    # And nothing that survives may carry alt reads in the normal.
    for v in somatic:
        assert not (v.normal_alt_count >= MAX_NORMAL_ALT_COUNT
                    and v.normal_af >= MAX_NORMAL_AF)
