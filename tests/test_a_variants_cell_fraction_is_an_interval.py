"""A variant's share of the cancer cells is an interval, or it is nothing.

The claim under test is not "the arithmetic runs". It is that reading a
variant allele frequency as a cell fraction is wrong, that this module
says so in both directions, and that a missing input produces no number
at all.
"""

import math

import pytest

from cognisom.genomics.cancer_cell_fraction import (
    AUTOSOMAL_NORMAL_COPIES,
    CCF_MODEL_SOURCE,
    CCF_MODEL_STATUS,
    HEMIZYGOUS_NORMAL_COPIES,
    CancerCellFraction,
    CopyState,
    PurityUnavailable,
    cancer_cell_fraction,
    ccf_report,
    mutant_copies_per_tumour_cell,
    vaf_confidence_interval,
)
from cognisom.genomics.demo_patients import SU2C_AR_F877L_CLONALITY

#: The AR segment's log2 ratio for SC_9184_T, from the cohort's
#: data_cna_hg19.seg as read by scripts/su2c_ar_clonality.py.
AR_SEGMENT_LOG2 = 1.934


def ar_copy_state(purity):
    return CopyState.from_copy_ratio(
        copy_ratio=2.0 ** AR_SEGMENT_LOG2,
        purity=purity,
        normal_copies=HEMIZYGOUS_NORMAL_COPIES,
        source="SU2C prad_su2c_2019 data_cna_hg19.seg, chrX segment "
               "containing the F877L call, log2 1.934",
    )


def ar_fraction():
    purity = SU2C_AR_F877L_CLONALITY["purity"]
    return cancer_cell_fraction(
        vaf=SU2C_AR_F877L_CLONALITY["vaf"],
        purity=purity,
        copy_state=ar_copy_state(purity),
    )


# ── it reproduces the one result the repo already derived by hand ──

def test_the_demo_patient_s_ar_variant_reproduces_the_recorded_range():
    """The hand-written script's answer, from the general module."""
    f = ar_fraction()
    assert f is not None
    assert f.lower == pytest.approx(
        SU2C_AR_F877L_CLONALITY["carrier_fraction_min"], abs=0.005)
    assert f.upper == pytest.approx(
        SU2C_AR_F877L_CLONALITY["carrier_fraction_max"], abs=1e-9)


def test_the_recorded_range_is_the_one_this_test_claims():
    """Non-vacuity: the constant really says 0.26 to 1.0.

    Without this, the test above would keep passing if the recorded
    numbers were replaced by whatever the code produced.
    """
    assert SU2C_AR_F877L_CLONALITY["carrier_fraction_min"] == 0.26
    assert SU2C_AR_F877L_CLONALITY["carrier_fraction_max"] == 1.0
    assert SU2C_AR_F877L_CLONALITY["vaf"] == 0.233


def test_the_intermediate_quantities_match_the_recorded_ones():
    purity = SU2C_AR_F877L_CLONALITY["purity"]
    state = ar_copy_state(purity)
    assert state.total_tumour_copies == pytest.approx(
        SU2C_AR_F877L_CLONALITY["ar_copies_per_tumour_cell"], abs=0.01)
    fm = mutant_copies_per_tumour_cell(
        SU2C_AR_F877L_CLONALITY["vaf"], purity, state)
    assert fm == pytest.approx(
        SU2C_AR_F877L_CLONALITY["mutant_copies_per_tumour_cell"], abs=0.01)


def test_from_copy_ratio_matches_the_script_s_inline_expression():
    """`(ratio - (1 - purity)) / purity`, which is this with N = 1."""
    ratio, purity = 2.0 ** AR_SEGMENT_LOG2, 0.65
    expected = (ratio - (1.0 - purity)) / purity
    state = CopyState.from_copy_ratio(
        ratio, purity, normal_copies=HEMIZYGOUS_NORMAL_COPIES, source="x")
    assert state.total_tumour_copies == pytest.approx(expected, rel=1e-12)


# ── the headline claim: a VAF is not a cell fraction ──────────────

def test_an_amplified_locus_carries_more_cells_than_its_vaf_suggests():
    """0.233 of reads is at least 0.26 of cells, and may be all of them."""
    f = ar_fraction()
    assert f.lower > f.vaf, (
        "on an amplified locus the carrier fraction must exceed the allele "
        "frequency; reading the VAF as a cell fraction understates it")
    assert f.may_be_clonal


def test_more_than_one_mutant_copy_is_read_off_the_data():
    """f*m above 1 cannot happen with one mutant copy per carrier."""
    f = ar_fraction()
    assert f.mutant_copies_per_tumour_cell > 1.0


def test_a_diploid_subclone_is_bounded_well_inside_zero_to_one():
    """A 15% VAF in a pure diploid tumour is 15-30% of cells, not 15%."""
    state = CopyState(total_tumour_copies=2.0,
                      normal_copies=AUTOSOMAL_NORMAL_COPIES,
                      source="assumed diploid for this test")
    f = cancer_cell_fraction(vaf=0.15, purity=1.0, copy_state=state)
    assert f.lower == pytest.approx(0.15, abs=1e-9)
    assert f.upper == pytest.approx(0.30, abs=1e-9)
    # non-vacuity: the interval has to say something
    assert 0.0 < f.lower and f.upper < 1.0
    assert f.is_certainly_subclonal
    assert not f.may_be_clonal


def test_a_clonal_heterozygous_diploid_variant_reaches_one():
    state = CopyState(total_tumour_copies=2.0, source="assumed diploid")
    f = cancer_cell_fraction(vaf=0.5, purity=1.0, copy_state=state)
    assert f.upper == pytest.approx(1.0, abs=1e-9)
    assert f.lower == pytest.approx(0.5, abs=1e-9)
    assert f.may_be_clonal


def test_normal_contamination_raises_the_fraction():
    """The same VAF in a less pure sample means more of the cancer cells."""
    state = CopyState(total_tumour_copies=2.0, source="assumed diploid")
    pure = cancer_cell_fraction(vaf=0.25, purity=1.0, copy_state=state)
    dilute = cancer_cell_fraction(vaf=0.25, purity=0.5, copy_state=state)
    assert dilute.lower > pure.lower
    assert dilute.upper > pure.upper


# ── it refuses rather than guesses ────────────────────────────────

@pytest.mark.parametrize("missing", ["vaf", "purity", "copy_state"])
def test_a_missing_input_gives_no_number_at_all(missing):
    kwargs = {
        "vaf": 0.3,
        "purity": 0.6,
        "copy_state": CopyState(total_tumour_copies=2.0, source="test"),
    }
    kwargs[missing] = None
    assert cancer_cell_fraction(**kwargs) is None


def test_demanding_a_fraction_without_purity_raises_its_own_error():
    state = CopyState(total_tumour_copies=2.0, source="test")
    with pytest.raises(PurityUnavailable):
        mutant_copies_per_tumour_cell(0.3, None, state)


@pytest.mark.parametrize("purity", [0.0, -0.1, 1.5])
def test_an_impossible_purity_is_rejected(purity):
    state = CopyState(total_tumour_copies=2.0, source="test")
    with pytest.raises(ValueError):
        mutant_copies_per_tumour_cell(0.3, purity, state)


def test_a_copy_state_must_say_where_it_came_from():
    """The missing provenance is what made the AR variant look subclonal."""
    with pytest.raises(ValueError, match="source is required"):
        CopyState(total_tumour_copies=2.0)


def test_a_locus_in_no_tumour_cell_is_rejected():
    with pytest.raises(ValueError, match="must be positive"):
        CopyState(total_tumour_copies=0.0, source="test")


def test_there_is_no_midpoint_scenario():
    """Collapsing the interval has to name an end, on purpose."""
    f = ar_fraction()
    assert f.scenario("lower") == f.lower
    assert f.scenario("fewest_carriers") == f.lower
    assert f.scenario("most_carriers") == f.upper
    with pytest.raises(ValueError, match="no midpoint"):
        f.scenario("midpoint")


# ── read depth widens the answer ──────────────────────────────────

def test_a_shallow_call_reports_a_wider_interval_than_a_deep_one():
    state = CopyState(total_tumour_copies=2.0, source="test")
    deep = cancer_cell_fraction(vaf=0.25, purity=1.0, copy_state=state,
                                alt_count=250, depth=1000)
    shallow = cancer_cell_fraction(vaf=0.25, purity=1.0, copy_state=state,
                                   alt_count=5, depth=20)
    assert shallow.width > deep.width


def test_the_vaf_interval_matches_the_recorded_one_for_ar():
    lo, hi = vaf_confidence_interval(190, 817)
    recorded_lo, recorded_hi = SU2C_AR_F877L_CLONALITY["vaf_ci95"]
    assert lo == pytest.approx(recorded_lo, abs=0.002)
    assert hi == pytest.approx(recorded_hi, abs=0.002)


def test_the_vaf_interval_rejects_impossible_counts():
    with pytest.raises(ValueError):
        vaf_confidence_interval(10, 0)
    with pytest.raises(ValueError):
        vaf_confidence_interval(50, 10)


# ── provenance travels with the answer ────────────────────────────

def test_the_answer_carries_its_status_and_source():
    f = ar_fraction()
    assert f.status == CCF_MODEL_STATUS == "derived"
    assert "28270531" in f.source
    assert "22544022" in f.source
    assert f.as_dict()["status"] == "derived"


def test_the_report_names_what_it_does_not_model():
    r = ccf_report()
    assert r["status"] == "derived"
    assert r["source"] == CCF_MODEL_SOURCE
    assert any("subclones" in s for s in r["not_modelled"])
    assert "tumour purity" in r["requires"]


def test_a_fraction_is_immutable():
    f = ar_fraction()
    with pytest.raises(Exception):
        f.lower = 0.0


def test_nothing_returns_a_fraction_outside_zero_to_one():
    state = CopyState(total_tumour_copies=7.0, source="test")
    for vaf in (0.0, 0.01, 0.4, 0.99, 1.0):
        f = cancer_cell_fraction(vaf=vaf, purity=0.3, copy_state=state)
        assert 0.0 <= f.lower <= 1.0
        assert 0.0 <= f.upper <= 1.0
        assert f.lower <= f.upper
        assert not math.isnan(f.lower) and not math.isnan(f.upper)
