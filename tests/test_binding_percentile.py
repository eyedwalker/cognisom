"""The percentile rank is the predictor's, not a plausible-looking constant.

``Class1PresentationPredictor.predict`` returns, per peptide: peptide,
peptide_num, sample_name, affinity, best_allele, processing_score,
presentation_score, presentation_percentile. The calibrated rank is
``presentation_percentile``; there is **no** ``affinity_percentile``
column.

The single-peptide path asked for ``affinity_percentile`` and defaulted a
miss to 50, so with the real predictor loaded it reported exactly 50.000 for
every peptide, while the batch path beside it read the right column.
Measured on HLA-A*02:01 through the repo's own code:

    NLVPMVATV    16.6 nM   single 50.000   batch 0.019
    KLVVVGADGV  110.8 nM   single 50.000   batch 0.870
    HMTEVVRHC  1386.3 nM   single 50.000   batch 3.946

It matters because the neoantigen ranker *prefers* a stored percentile over
its own affinity-derived estimate, on the grounds that the stored one is
calibrated. A constant therefore displaced a number that at least ordered
peptides correctly.

These tests use a stand-in predictor returning the real column set, so they
run anywhere; the numbers above came from the real one.
"""
from __future__ import annotations

import math

import pytest

from cognisom.genomics import mhcflurry_binding as mb
from cognisom.genomics.neoantigen_predictor import NeoantigenPredictor

PEPTIDES = ["NLVPMVATV", "KLVVVGADGV", "HMTEVVRHC"]
ALLELE = "HLA-A*02:01"
#: peptide -> (affinity nM, presentation_score, presentation_percentile),
#: as the real predictor returns them for this allele.
REAL = {
    "NLVPMVATV": (16.57, 0.9702, 0.0187),
    "KLVVVGADGV": (110.78, 0.4776, 0.8698),
    "HMTEVVRHC": (1386.29, 0.0672, 3.9462),
}


class _Frame:
    """The subset of a pandas frame these two paths use."""

    def __init__(self, rows):
        self._rows = rows
        self.columns = list(rows[0]) if rows else []

    def iterrows(self):
        return enumerate(self._rows)

    @property
    def iloc(self):
        return self._rows


class _Row(dict):
    @property
    def index(self):
        return list(self)


def _row(peptide, drop_percentile=False):
    affinity, presentation, percentile = REAL[peptide]
    row = _Row({
        "peptide": peptide, "peptide_num": 0, "sample_name": "sample1",
        "affinity": affinity, "best_allele": ALLELE,
        "processing_score": 0.5, "presentation_score": presentation,
        "presentation_percentile": percentile,
    })
    if drop_percentile:
        del row["presentation_percentile"]
    return row


class _Predictor:
    supported_alleles = [ALLELE]

    def __init__(self, drop_percentile=False):
        self.drop = drop_percentile

    def predict(self, peptides, alleles, verbose=0):
        return _Frame([_row(p, self.drop) for p in peptides])


@pytest.fixture
def predictor(monkeypatch):
    def install(drop_percentile=False):
        p = _Predictor(drop_percentile)
        monkeypatch.setattr(mb, "is_mhcflurry_available", lambda: True)
        monkeypatch.setattr(mb, "get_predictor", lambda: p)
        mb._PERCENTILE_COLUMN_WARNED.clear()
        return p
    return install


def test_the_percentile_is_the_predictors_not_a_constant(predictor):
    predictor()
    seen = {p: mb.predict_binding(p, ALLELE).percentile_rank for p in PEPTIDES}
    assert seen == {p: pytest.approx(REAL[p][2]) for p in PEPTIDES}
    assert len(set(seen.values())) == len(PEPTIDES), "every peptide ranked alike"


def test_the_single_and_batch_paths_agree(predictor):
    predictor()
    batch = mb.predict_binding_batch(list(PEPTIDES), [ALLELE])
    for peptide, batched in zip(PEPTIDES, batch):
        single = mb.predict_binding(peptide, ALLELE)
        assert single.percentile_rank == pytest.approx(batched.percentile_rank)
        assert single.presentation_score == pytest.approx(batched.presentation_score)
        assert single.affinity_nm == pytest.approx(batched.affinity_nm)


def test_the_percentile_orders_peptides_the_way_affinity_does(predictor):
    predictor()
    ranked = sorted(PEPTIDES, key=lambda p: mb.predict_binding(p, ALLELE).percentile_rank)
    assert ranked == sorted(PEPTIDES, key=lambda p: REAL[p][0])


def test_a_missing_column_is_reported_not_defaulted(predictor, caplog):
    """What hid this: a miss resolving to a plausible middle value."""
    predictor(drop_percentile=True)
    with caplog.at_level("WARNING", logger="cognisom.genomics.mhcflurry_binding"):
        result = mb.predict_binding(PEPTIDES[0], ALLELE)
    assert math.isnan(result.percentile_rank)
    assert result.percentile_rank != 50.0
    messages = [r.getMessage() for r in caplog.records
                if "presentation_percentile" in r.getMessage()]
    assert len(messages) == 1
    # Once per process, not once per peptide.
    for peptide in PEPTIDES:
        mb.predict_binding(peptide, ALLELE)
    assert len(messages) == 1


def test_an_uncalibrated_percentile_does_not_displace_the_estimate():
    """NaN passes an ``is not None`` test and then compares False against
    everything, so it has to be turned into an absence at the boundary."""
    calibrated = NeoantigenPredictor._calibrated
    assert calibrated(0.87) == pytest.approx(0.87)
    assert calibrated(0.0) == 0.0            # a real zero survives
    assert calibrated(None) is None
    assert calibrated(float("nan")) is None
    assert calibrated("not a number") is None


def test_the_cache_stores_the_predictors_percentile(predictor, monkeypatch):
    predictor()
    p = NeoantigenPredictor()
    p._warm_binding_cache(list(PEPTIDES), [ALLELE])
    for peptide in PEPTIDES:
        cached = p._binding_cache[(peptide, ALLELE)]
        assert cached.percentile_rank == pytest.approx(REAL[peptide][2])
        assert cached.presentation_score == pytest.approx(REAL[peptide][1])
