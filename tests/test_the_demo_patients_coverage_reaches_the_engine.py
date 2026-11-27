"""The demo patient's coverage is computed, and the engine can take it.

Closes the loop Phase 1 opened: the genomics side now derives what share
of the cancer cells carries a variant, and the engine's
``clonal_fractions`` argument is where that number has to land. Before
this, the only route was a hand-set fraction with nothing behind it.

The committed profile exists so this works without the cohort, which is
gitignored. The last test in this file reads the cohort when it is
present and checks the committed numbers against it, so the constants
cannot drift away from the files they came from unnoticed.
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

import pytest

from cognisom.genomics.cancer_cell_fraction import (
    CLONALITY_POSSIBLY_CLONAL,
    HEMIZYGOUS_NORMAL_COPIES,
    CopyState,
    clonality_label,
    founding_fractions,
)
from cognisom.genomics.demo_patients import (
    SU2C_AR_F877L_CLONALITY as CLONALITY,
    su2c_ar_cell_fraction,
    su2c_sample_profile,
)

LABEL = "AR_F877L"
COHORT = Path("data/cbioportal/prad_su2c_2019")
AR_POS = 66943551          # hg19 chrX, the F877L call


# ── the committed profile stands in for the cohort ────────────────

def test_the_committed_profile_computes_without_the_cohort():
    f = su2c_ar_cell_fraction()
    assert f is not None
    assert f.lower == pytest.approx(
        CLONALITY["carrier_fraction_min_ci95"][0], abs=0.01)
    assert f.upper == pytest.approx(1.0, abs=1e-9)


def test_the_profile_knows_ar_is_on_a_single_x():
    """Two normal copies here would inflate C and shrink the fraction."""
    state = su2c_sample_profile().copy_states["AR"]
    assert state.normal_copies == HEMIZYGOUS_NORMAL_COPIES == 1


def test_only_ar_has_a_copy_state():
    """Every other gene gets no fraction rather than an assumed diploid."""
    profile = su2c_sample_profile()
    assert set(profile.copy_states) == {"AR"}
    assert profile.fraction_for_gene("PTEN", 0.4) is None
    assert profile.fraction_for_gene("BRCA2", 0.3) is None


def test_the_profile_carries_its_provenance():
    profile = su2c_sample_profile()
    assert "clonal" in profile.purity_source
    assert "3 markers" in profile.copy_states["AR"].source, (
        "the weakest input to this fraction has to stay visible")


def test_the_profile_is_memoised():
    assert su2c_sample_profile() is su2c_sample_profile()


def test_the_variant_is_not_provably_subclonal():
    """The honest label: the data cannot rule out every cancer cell."""
    f = su2c_ar_cell_fraction()
    assert clonality_label(f) == CLONALITY_POSSIBLY_CLONAL
    assert f.may_be_clonal
    assert not f.is_certainly_subclonal


# ── collapsing an interval into the engine's argument ─────────────

def test_founding_fractions_requires_an_end_to_be_named():
    labelled = {LABEL: su2c_ar_cell_fraction()}
    with pytest.raises(TypeError):
        founding_fractions(labelled)          # type: ignore[call-arg]
    with pytest.raises(ValueError, match="no midpoint"):
        founding_fractions(labelled, at="midpoint")


def test_founding_fractions_takes_the_end_it_was_told_to():
    f = su2c_ar_cell_fraction()
    assert founding_fractions({LABEL: f}, at="fewest_carriers") == {
        LABEL: f.lower}
    assert founding_fractions({LABEL: f}, at="most_carriers") == {
        LABEL: f.upper}


def test_the_two_ends_really_differ():
    """Non-vacuity: with a degenerate interval the test above proves nothing."""
    f = su2c_ar_cell_fraction()
    assert f.upper > f.lower


def test_a_variant_with_no_fraction_is_left_out_not_defaulted():
    out = founding_fractions({LABEL: su2c_ar_cell_fraction(),
                              "PTEN_LOSS": None}, at="lower")
    assert set(out) == {LABEL}


def test_a_zero_fraction_is_left_out():
    """The engine refuses 0.0, and a variant in no founder is not a subclone."""
    state = CopyState(total_tumour_copies=2.0, source="test")
    from cognisom.genomics.cancer_cell_fraction import cancer_cell_fraction
    zero = cancer_cell_fraction(vaf=0.0, purity=1.0, copy_state=state)
    assert zero.lower == 0.0
    assert founding_fractions({"NONE": zero}, at="lower") == {}


def test_an_empty_mapping_gives_an_empty_argument():
    assert founding_fractions({}, at="lower") == {}


# ── the engine accepts what comes out ─────────────────────────────

#: The engine validates a clonal_fractions label against the patient's
#: own mutations and refuses an unknown one, so a run has to declare the
#: mutation as well as its fraction.
PATIENT_MUTATIONS = [{"gene": "AR", "aa_change": "F877L", "label": LABEL,
                      "oncogenic": True}]


def _cellular(fractions):
    import contextlib
    import io
    from cognisom.modules.cellular_module import CellularModule
    with contextlib.redirect_stdout(io.StringIO()):
        return CellularModule({"patient_mutations": PATIENT_MUTATIONS,
                               "clonal_fractions": dict(fractions)})


@pytest.mark.parametrize("end", ["fewest_carriers", "most_carriers"])
def test_the_engine_accepts_the_derived_fraction(end):
    """The end of the loop: a measured interval configures a run."""
    fractions = founding_fractions({LABEL: su2c_ar_cell_fraction()}, at=end)
    assert fractions, "nothing to hand the engine"
    module = _cellular(fractions)
    assert module.clonal_fractions == pytest.approx(fractions)


def test_the_derived_fraction_is_inside_what_the_engine_allows():
    """The engine refuses 0.0 and anything above 1.0."""
    for end in ("fewest_carriers", "most_carriers"):
        value = founding_fractions(
            {LABEL: su2c_ar_cell_fraction()}, at=end)[LABEL]
        assert 0.0 < value <= 1.0


def test_the_fraction_decides_how_many_founders_carry_it():
    """Non-vacuity: the derived number has to change the cell population."""
    import numpy as np

    low = _cellular(founding_fractions(
        {LABEL: su2c_ar_cell_fraction()}, at="fewest_carriers"))
    high = _cellular(founding_fractions(
        {LABEL: su2c_ar_cell_fraction()}, at="most_carriers"))
    n = 100
    carriers_low = sum(LABEL in low._founding_mutations() for _ in range(n))
    carriers_high = sum(LABEL in high._founding_mutations() for _ in range(n))
    assert carriers_high > carriers_low
    assert carriers_low == int(np.floor(
        n * low.clonal_fractions[LABEL] + 1e-9))


def test_a_label_the_patient_lacks_is_still_refused():
    """The derived route must not bypass the engine's own guard."""
    with pytest.raises(ValueError, match="not one of this patient's"):
        _cellular({"AR_T878A": 0.3})


# ── and the committed numbers still match the cohort ──────────────

def _cohort_vaf(row):
    alt, ref = int(row["t_alt_count"]), int(row["t_ref_count"])
    return alt / (alt + ref)


@pytest.mark.skipif(not COHORT.exists(),
                    reason="the SU2C cohort is gitignored; run locally")
def test_the_committed_numbers_match_the_cohort_files():
    """Guards against the constants drifting from their source."""
    sample = "SC_9184_T"

    with open(COHORT / "data_cna.txt") as fh:
        rows = csv.reader(fh, delimiter="\t")
        col = next(rows).index(sample)
        cna = {r[0]: r[col] for r in rows}
    assert cna["AR"] == "2", "AR is amplified in this sample"

    with open(COHORT / "data_mutations.txt") as fh:
        muts = [m for m in csv.DictReader(
            (ln for ln in fh if not ln.startswith("#")), delimiter="\t")
            if m["Tumor_Sample_Barcode"] == sample]

    ar = next(m for m in muts if m["Hugo_Symbol"] == "AR")
    assert _cohort_vaf(ar) == pytest.approx(CLONALITY["vaf"], abs=0.001)
    assert int(ar["t_alt_count"]) == 190
    assert int(ar["t_alt_count"]) + int(ar["t_ref_count"]) == 817

    # purity, from clonal SNVs on the single X at normal copy number
    snvs = [m for m in muts if m["Variant_Type"] == "SNP"
            and m["t_alt_count"] not in ("", "0")]
    x_clonal = [_cohort_vaf(m) for m in snvs
                if m["Chromosome"] == "X"
                and cna.get(m["Hugo_Symbol"]) == "0"
                and _cohort_vaf(m) > 0.5]
    assert statistics.median(x_clonal) == pytest.approx(
        CLONALITY["purity"], abs=0.01)

    # copy number, from the segment containing the call
    with open(COHORT / "data_cna_hg19.seg") as fh:
        segs = [r for r in csv.DictReader(fh, delimiter="\t")
                if r["ID"] == sample and r["chrom"] in ("X", "23")]
    seg = next(r for r in segs
               if float(r["loc.start"]) <= AR_POS <= float(r["loc.end"]))
    derived = CopyState.from_copy_ratio(
        2.0 ** float(seg["seg.mean"]), CLONALITY["purity"],
        normal_copies=HEMIZYGOUS_NORMAL_COPIES, source="cohort")
    assert derived.total_tumour_copies == pytest.approx(
        CLONALITY["ar_copies_per_tumour_cell"], abs=0.02)
