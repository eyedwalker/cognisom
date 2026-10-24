"""What carries the cancer type through the genomics pipeline, and what does not.

The platform simulates prostate as its worked example but is meant to
take any tissue. These tests pin where that already holds, where it was
quietly broken, and where it is still a stub -- measured by running the
real demo callset and a constructed breast profile.

Found by running HCC1395 as breast: ``get_driver_details`` read
``PROSTATE_CANCER_DRIVERS`` whatever the cancer type, so four of six
canonical breast drivers (ERBB2, ESR1, GATA3, CDH1) came back
``role="unknown"`` despite all four being in the pan-cancer table the
annotator already ships. TP53 and BRCA2 resolved only because prostate
happens to share them -- which is exactly why it went unnoticed.
"""

from __future__ import annotations

import pytest

from cognisom.genomics.patient_profile import PatientProfile
from cognisom.genomics.variant_annotator import (
    ALL_CANCER_DRIVERS, PAN_CANCER_DRIVERS, PROSTATE_CANCER_DRIVERS)
from cognisom.genomics.vcf_parser import Variant

BREAST_DRIVERS = ("ERBB2", "ESR1", "GATA3", "CDH1")
SHARED_WITH_PROSTATE = ("TP53", "BRCA2")


def _variant(gene, change):
    v = Variant(chrom="17", pos=1, id=".", ref="A", alt="G",
                qual=99.0, filter_status="PASS")
    v.gene, v.protein_change = gene, change
    v.consequence, v.impact = "missense_variant", "HIGH"
    return v


def _breast_profile():
    drivers = [_variant(g, "p.X1Y") for g in BREAST_DRIVERS + SHARED_WITH_PROSTATE]
    return PatientProfile(patient_id="breast-test", cancer_type="breast",
                          variants=drivers, cancer_driver_mutations=drivers)


# ── the panels ────────────────────────────────────────────────────


def test_the_library_already_knows_the_breast_drivers():
    for gene in BREAST_DRIVERS:
        assert gene in ALL_CANCER_DRIVERS, gene
        assert gene in PAN_CANCER_DRIVERS, f"{gene} is pan-cancer, not prostate"
        assert gene not in PROSTATE_CANCER_DRIVERS, (
            f"{gene} is not in the prostate panel -- which is the whole point")
    for gene in SHARED_WITH_PROSTATE:
        assert gene in PROSTATE_CANCER_DRIVERS, "shared, so it resolved either way"


def test_driver_details_resolve_for_a_tissue_that_is_not_prostate():
    details = {d["gene"]: d for d in _breast_profile().get_driver_details()}
    for gene in BREAST_DRIVERS + SHARED_WITH_PROSTATE:
        assert details[gene]["role"] != "unknown", (
            f"{gene} came back unknown -- the driver table is prostate-only again")
        assert details[gene]["full_name"] != gene, (
            f"{gene} has no full name -- ditto")
    assert details["ERBB2"]["role"] == "oncogene"
    assert details["CDH1"]["role"] == "tumor_suppressor"


def test_the_profile_keeps_the_cancer_type_it_was_built_with():
    assert _breast_profile().cancer_type == "breast"
    assert PatientProfile(patient_id="p").cancer_type == "prostate", (
        "prostate is still the default, which is a default and not a constraint")


# ── what is still prostate-shaped, stated rather than hidden ──────


def test_the_trials_client_still_branches_on_prostate_alone():
    """Not fixed here, recorded: clinical_trials has a literal
    `if cancer_type == "prostate"` branch, so another tissue takes the
    generic path. Fixing it needs per-tissue trial queries, which is a
    data question rather than a code one."""
    import inspect
    from cognisom.genomics import clinical_trials
    source = inspect.getsource(clinical_trials)
    assert 'cancer_type == "prostate"' in source, (
        "if this branch is gone, generalise this test with it")


def test_the_real_demo_callset_yields_nothing_for_any_tissue_and_says_why():
    """HCC1395's VCF is a raw callset with no protein consequences, so it
    produces no coding variants, no drivers and no neoantigens -- for
    breast and prostate alike. That is correct behaviour, not a tissue
    problem: a consequence cannot be derived from a coordinate. The
    caveats are how a reader tells 'quiet tumour' from 'not evaluated'.
    """
    import contextlib
    import io
    import os

    from cognisom.genomics.patient_profile import PatientProfileBuilder

    vcf = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "cognisom", "genomics", "seqc2_demo.vcf")
    if not os.path.exists(vcf):
        pytest.skip("demo callset not present")
    for cancer_type in ("prostate", "breast"):
        with contextlib.redirect_stdout(io.StringIO()):
            profile = PatientProfileBuilder(cancer_type=cancer_type).from_vcf_file(vcf)
        assert len(profile.variants) > 1000, "the callset is read"
        assert profile.coding_variants == [], "and carries no consequences"
        assert profile.predicted_neoantigens == []
        assert not profile.tmb_is_estimable and profile.tmb_caveats, (
            "and the profile says so rather than reporting TMB 0.0 as a fact")
