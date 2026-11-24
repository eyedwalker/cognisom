"""The measurement behind limitation 2's fix, and its refutation rule.

Limitation 2 on ``?page=methods`` says the engine has no cell-to-cell
variation in AR drive and that the fix is to add some -- not to move
the dead band. Adding any needs a magnitude, and this project's
standing rule is that a capability blocked on an unsourced number stays
blocked rather than getting an invented one.

``ar_heterogeneity_singlecell`` sources it from 24 hormone-therapy-naive
prostate cancers. Most of what matters here is checkable without the
700 MB download: that the signatures are shared with the test the
measurement is meant to inform, and that the pre-specified verdict can
come out NEGATIVE. A verdict that can only say "sourced" is not a
verdict.

The live measurement runs only when the atlas is already cached, so the
suite never downloads it. To run it:

    COGNISOM_SC_CACHE=/some/dir python -m \\
        cognisom.validation.ar_heterogeneity_singlecell
"""

from __future__ import annotations

import pytest

from cognisom.validation import ar_heterogeneity_singlecell as sc
from cognisom.validation import ar_proliferation_tcga as cross


# ── the signatures must be the SAME OBJECTS ────────────────────────


def test_the_signatures_are_shared_with_the_cross_sectional_test():
    """Not merely equal -- the same objects.

    The whole point is to measure the spread of the pathway the TCGA
    test found graded. A gene set that drifted between the two files
    would make the measurement and the finding it informs
    incomparable, and equality would still pass after a copy-paste
    divergence was introduced in one place only.
    """
    assert sc.AR_ACTIVITY is cross.AR_ACTIVITY
    assert sc.PROLIFERATION is cross.PROLIFERATION


def test_the_controls_do_not_overlap_the_signature():
    """A control sharing genes with AR would carry the AR signal."""
    ar = set(sc.AR_ACTIVITY)
    assert not ar & set(sc.HOUSEKEEPING)
    assert not ar & set(sc.INTERFERON)
    assert not ar & set(sc.PROLIFERATION)


# ── the pre-specified thresholds ───────────────────────────────────


def test_the_thresholds_leave_room_to_fail():
    """The negative control's ceiling must sit below the signal's
    floor, or 'AR beats housekeeping' could be satisfied by noise."""
    assert sc.HOUSEKEEPING_EXCESS_MAX < sc.AR_EXCESS_MIN
    assert sc.AR_EXCESS_MIN > 1.0, (
        "an excess of 1.0 IS the null; a floor at or below it would "
        "accept pure counting noise as a sourced magnitude")
    assert sc.POSITIVE_CONTROL_MIN > 1.0


def _result(name, excess):
    return sc.SetResult(name=name, n_genes=7, n_cells=300, variance=excess,
                        null_variance=1.0, excess=excess, p_null=0.0)


def _report(ar, house, ifn, positive):
    donors = [sc.DonorResult(
        donor=f"P{i}", n_cells=300,
        sets={"AR": _result("AR", ar),
              "housekeeping": _result("housekeeping", house),
              "interferon": _result("interferon", ifn)},
        ar_sigma=0.34, ar_cv=0.46) for i in range(1, 4)]
    rep = sc.Report(donors=donors)
    rep.positive_control_pooled = positive
    rep.positive_control_pooled_p = 0.0
    return rep


def test_the_verdict_is_positive_when_every_condition_holds():
    rep = _report(ar=4.4, house=1.1, ifn=2.8, positive=3.5)
    assert rep.sourced
    assert "SOURCED" in rep.verdict()


@pytest.mark.parametrize("ar,house,ifn,positive,why", [
    (1.2, 1.1, 2.8, 3.5, "AR indistinguishable from the null"),
    (4.4, 3.0, 2.8, 3.5, "negative control as variable as AR"),
    (4.4, 1.1, 1.0, 1.0, "neither positive control detects a program"),
])
def test_the_verdict_can_come_out_negative(ar, house, ifn, positive, why):
    """REFUTATION. Each of the three pre-specified conditions must be
    able to sink the result on its own, or the stopping rule is
    decoration."""
    rep = _report(ar=ar, house=house, ifn=ifn, positive=positive)
    assert not rep.sourced, why
    assert "NOT SOURCED" in rep.verdict()


def test_biological_variance_never_goes_negative():
    """The null can exceed the real variance by chance; a negative
    variance would become a NaN sigma and quietly poison the median."""
    res = sc.SetResult(name="AR", n_genes=7, n_cells=120, variance=0.4,
                       null_variance=0.9, excess=0.44, p_null=0.8)
    assert res.biological_variance == 0.0


# ── what the number may be used for ────────────────────────────────


def test_the_module_says_the_sigma_is_an_upper_bound():
    """The engine's signalling_fraction is receptor occupancy; this
    measures the transcriptional READOUT, which is occupancy plus
    downstream noise. Anything wired from it has to say so, and the
    place that must not lose the caveat is the module itself."""
    doc = sc.__doc__ or ""
    assert "UPPER BOUND" in doc
    assert "readout" in doc.lower()
    rep = _report(ar=4.4, house=1.1, ifn=2.8, positive=3.5)
    assert "UPPER BOUND" in rep.verdict(), (
        "the verdict is what gets quoted; the caveat has to travel "
        "with it, not sit only in the docstring")


def test_the_dataset_is_hormone_therapy_naive():
    """An ADT-treated cohort would measure the spread of a SUPPRESSED
    population, which is a different quantity -- the engine's untreated
    arm is what 1.00 signalling corresponds to."""
    assert "naive" in sc.DATASET_TITLE.lower()
    assert sc.DATASET_ID in sc.SOURCE_URL
    assert sc.CENSUS_VERSION in sc.SOURCE_URL, (
        "the S3 mirror is per census release; a URL that drifts from "
        "CENSUS_VERSION would silently fetch another release")


def test_the_download_lands_outside_the_repository():
    """700 MB of third-party data stays out of the repository, and a
    committed dataset would need a NOTICE entry."""
    import os
    assert "COGNISOM_SC_CACHE" in (sc.cache_dir.__doc__ or "")
    assert sc.local_path().startswith(sc.cache_dir())
    repo_root = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(sc.__file__))))
    cache = os.path.abspath(sc.cache_dir())
    assert os.path.commonpath([cache, repo_root]) != repo_root, (
        f"the cache {cache} is inside the repository {repo_root}: a "
        "700 MB download would become a candidate for committing")


# ── the live measurement, only if already downloaded ───────────────


@pytest.mark.slow
@pytest.mark.skipif(not sc.is_cached(),
                    reason="atlas not cached; set COGNISOM_SC_CACHE")
def test_the_measured_spread_is_what_the_methods_page_quotes():
    """MEASURED 2026-11-24 on 3,147 malignant cells in 9 donors. The
    claims are asserted; the numbers are pinned beside them loosely,
    because they are medians over donors and a ventile-matched null.
    Re-measure, do not re-pin."""
    report = sc.run()
    assert len(report.donors) == 9, [d.donor for d in report.donors]

    # the claim: AR varies far more than a matched gene set
    assert report.ar_excess > 3.0
    # the claim: and far more than the negative control
    assert report.ar_excess > 3 * report.housekeeping_excess
    # the claim: the negative control sits at the null
    assert report.housekeeping_excess == pytest.approx(1.15, abs=0.35)
    # the claim: the method can see a real program when there is one
    assert report.best_positive_control >= sc.POSITIVE_CONTROL_MIN
    # every donor, not just the median
    significant = sum(1 for d in report.donors
                      if d.sets["AR"].p_null < 0.05)
    assert significant == 9, f"only {significant}/9 donors significant"

    assert report.sourced
    # the numbers the page quotes
    assert report.ar_sigma_median == pytest.approx(0.34, abs=0.08)
    assert report.ar_cv_median == pytest.approx(0.46, abs=0.10)
