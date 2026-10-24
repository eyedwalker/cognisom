"""The real genome and the rest of the pipeline meet, for the first time.

``seqc2_demo.vcf`` is a real callset -- HCC1395, the SEQC2 breast line --
carrying coordinates and nothing else, so the profile builder correctly
produced no coding variants, no drivers and no neoantigens, and said why.
The consequence went unnoticed: every neoantigen number this platform
has shown came from the SYNTHETIC VCF.

``VEPAnnotator`` could close that gap all along -- 560 lines, two
backends, its own tests -- and was never passed in.
``annotate_demo_callset`` now runs it once and ships the result, so the
demo reaches the neoantigen path offline with its provenance in the
file's header.

What the chain yields, end to end, with every link real:

    1,495 variants in the callset
       29 carry a coding protein change (VEP, Ensembl REST, MANE)
        2 survive the somatic filter: ARID1A p.N218T, NOTCH1 p.L2065Q
          -- 811 low-evidence, 292 germline, 179 artifact are dropped
       50 candidate neoantigens from those two
        3 strong binders against the patient's MEASURED HLA
          (1 against a population-default panel)

The 29 -> 2 step is the one to keep in view. The raw annotation finds
ESR1 and ERBB2 among others, which would be an exciting result for a
breast line and is not a result at all: they do not survive filtering.
"""

from __future__ import annotations

import contextlib
import io
import os

import pytest

from cognisom.genomics.variant_annotator import ALL_CANCER_DRIVERS
from cognisom.genomics.vcf_parser import VCFParser

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "cognisom", "genomics", "seqc2_demo.vcf")
ANNOTATED = os.path.join(HERE, "cognisom", "genomics", "seqc2_demo.vep.vcf")

pytestmark = pytest.mark.skipif(
    not os.path.exists(ANNOTATED),
    reason="annotated demo callset not generated; "
           "run python -m cognisom.genomics.annotate_demo_callset")


def _parse(path):
    with contextlib.redirect_stdout(io.StringIO()):
        return VCFParser().parse_file(path)


# ── the annotated file ────────────────────────────────────────────


def test_the_annotated_callset_keeps_every_call_and_adds_only_consequences():
    raw, annotated = _parse(RAW), _parse(ANNOTATED)
    assert len(raw) == len(annotated) == 1495, "no call is added or dropped"
    sites = lambda vs: {(v.chrom, v.pos, v.ref, v.alt) for v in vs}
    assert sites(raw) == sites(annotated), "coordinates and alleles unchanged"
    assert not any(getattr(v, "protein_change", None) for v in raw)
    assert sum(1 for v in annotated if getattr(v, "protein_change", None)) == 29


def test_the_file_records_how_it_was_made():
    with open(ANNOTATED) as handle:
        header = [line for line in handle if line.startswith("##cognisom_vep")]
    text = "".join(header)
    for key in ("backend", "assembly", "date", "submitted", "annotated", "mane"):
        assert f"##cognisom_vep_{key}=" in text, key
    assert "ensembl-rest" in text and "GRCh38" in text


# ── the step that matters: 29 coding, 2 somatic ───────────────────


def test_somatic_filtering_is_what_takes_twenty_nine_coding_calls_to_two():
    parser = VCFParser()
    variants = _parse(ANNOTATED)
    coding = [v for v in variants if getattr(v, "protein_change", None)]
    assert len(coding) == 29

    somatic_coding = [v for v in parser.filter_somatic(variants)
                      if getattr(v, "protein_change", None)]
    genes = {v.gene for v in somatic_coding}
    assert genes == {"ARID1A", "NOTCH1"}, genes
    assert all(g in ALL_CANCER_DRIVERS for g in genes)

    # and the ones that do NOT survive include the exciting-looking ones
    dropped = {v.gene for v in coding} - genes
    assert {"ESR1", "ERBB2"} <= dropped, (
        "ESR1 and ERBB2 are in the raw annotation and are not somatic calls; "
        "reporting them as findings would be wrong")


def test_the_callset_is_mostly_not_somatic_and_the_parser_says_so():
    import collections
    status = collections.Counter(v.somatic_status for v in _parse(ANNOTATED))
    assert status["SOMATIC"] == 213
    assert status["LOW_EVIDENCE"] == 811
    assert status["GERMLINE"] == 292
    assert status["ARTIFACT"] == 179


# ── end to end ────────────────────────────────────────────────────


def _profile(**kwargs):
    from cognisom.genomics.patient_profile import PatientProfileBuilder
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        return PatientProfileBuilder(cancer_type="breast").from_vcf_file(
            ANNOTATED, **kwargs)


@pytest.mark.slow
def test_the_real_genome_now_reaches_the_neoantigen_path():
    profile = _profile()
    assert len(profile.coding_variants) == 2
    assert len(profile.cancer_driver_mutations) == 2
    roles = {d["gene"]: d["role"] for d in profile.get_driver_details()}
    assert roles == {"ARID1A": "tumor_suppressor", "NOTCH1": "oncogene"}, roles
    assert len(profile.predicted_neoantigens) > 0, "the path is reachable at last"
    assert profile.binding_method != "unknown"


@pytest.mark.slow
def test_the_patients_own_hla_finds_more_binders_than_a_population_panel():
    """The measured typing is reachable by patient id and is not the
    default. It is the same OptiType typing that licenses the NK
    receptor model, and it changes the answer: against this patient's
    HLA the same two mutations present more strong binders than against
    a population-default panel."""
    default = _profile()
    measured = _profile(patient_id="SEQC2")

    assert not default.hla_is_patient_specific
    assert default.hla_typing_method == "population-default"
    assert measured.hla_is_patient_specific
    assert measured.hla_typing_method == "optitype"
    assert set(measured.hla_alleles) == {"HLA-A*29:02", "HLA-B*45:01", "HLA-C*06:02"}

    assert len(measured.predicted_neoantigens) == len(default.predicted_neoantigens), (
        "the same peptides are generated; only the restriction changes")

    # The binder COUNT is only meaningful under the real predictor: the
    # PWM fallback is quantised to a handful of IC50 values and calls no
    # strong binder here at all. Measured under MHCflurry 2.0.6: 3 strong
    # and 9 weak against this patient's HLA, 1 and 4 against the
    # population panel.
    from cognisom.genomics.mhcflurry_binding import is_mhcflurry_available
    if not is_mhcflurry_available():
        pytest.skip("no MHCflurry: binder counts are the fallback's, not a prediction")
    assert measured.strong_binder_count > default.strong_binder_count
    assert measured.weak_binder_count > default.weak_binder_count


@pytest.mark.slow
def test_tmb_is_still_not_estimable_and_still_says_why():
    """Annotation does not make a raw callset a TMB. The numerator is
    still unfiltered and the assay footprint is still undeclared, so the
    number stays marked unusable rather than becoming a fact."""
    profile = _profile(patient_id="SEQC2")
    assert not profile.tmb_is_estimable
    assert len(profile.tmb_caveats) == 2
    joined = " ".join(profile.tmb_caveats)
    assert "FilterMutectCalls" in joined and "callable footprint" in joined
