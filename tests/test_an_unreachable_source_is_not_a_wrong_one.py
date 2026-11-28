"""A curation check must separate "cannot reach" from "disagrees".

The live BioNumbers checks in test_bionumbers_curation reported both as
the same failure, so four consecutive nightly runs went red on a read
timeout while 4,542 of our own tests passed. Memory records an earlier
30+ night stretch of the same thing. A lane that cries wolf nightly is a
lane nobody reads, and a real crash was hiding under exactly that noise.

`_fetch_or_skip` now skips on a transport error. These tests hold the
other half of its contract -- that a source which ANSWERS and disagrees
is still a hard failure -- because a helper that swallowed both would
make the curation check worthless while looking green.
"""
from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
import requests

from cognisom.library.bionumbers import BioNumberRecord

curation = pytest.importorskip("tests.test_bionumbers_curation")


def _one_sourced(value, bnid="108569", conditions="", citation="PMID:111"):
    """A single (entity, name, parameter) triple shaped like the fixture's."""
    pv = SimpleNamespace(value=value, source=f"BioNumbers BNID:{bnid}",
                         conditions=conditions, citation=citation)
    return [(SimpleNamespace(name="Probe cell"), "diameter_um", pv)]


# ── the half that was broken: transport errors ────────────────────────

@pytest.mark.parametrize("error", [
    requests.exceptions.ReadTimeout("read timed out"),
    requests.exceptions.ConnectTimeout("connect timed out"),
    requests.exceptions.ConnectionError("refused"),
])
def test_a_transport_failure_skips_rather_than_fails(error):
    def boom():
        raise error
    with pytest.raises(pytest.skip.Exception) as caught:
        curation._fetch_or_skip(boom, "BNID 108569")
    assert "unreachable" in str(caught.value)
    assert "not contradicted" in str(caught.value), (
        "the skip must say the value is unverified, not vindicated")


# ── the half that must keep working: a source that disagrees ──────────

def test_a_value_that_disagrees_is_still_a_failure(monkeypatch):
    """The source answers, and says something else. That is a defect."""
    stored = 10.0
    monkeypatch.setattr(
        "cognisom.library.bionumbers.fetch",
        lambda bnid, session=None: BioNumberRecord(
            bnid=bnid, value=99.0, unit="um", reference="Smith 2020"))
    with pytest.raises(AssertionError) as caught:
        curation.test_a_sample_of_records_refetch_to_the_stored_value(
            _one_sourced(stored))
    assert "stored 10.0" in str(caught.value)


def test_a_record_that_stops_parsing_is_still_a_failure(monkeypatch):
    monkeypatch.setattr(
        "cognisom.library.bionumbers.fetch",
        lambda bnid, session=None: BioNumberRecord(bnid=bnid, value=None))
    with pytest.raises(AssertionError) as caught:
        curation.test_a_sample_of_records_refetch_to_the_stored_value(
            _one_sourced(10.0))
    assert "no longer parses" in str(caught.value)


def test_a_citation_that_resolves_elsewhere_is_still_a_failure(monkeypatch):
    monkeypatch.setattr(
        "cognisom.library.bionumbers.fetch",
        lambda bnid, session=None: BioNumberRecord(
            bnid=bnid, value=10.0, reference="Smith 2020"))
    monkeypatch.setattr("cognisom.library.bionumbers.resolve_pmid",
                        lambda ref, session=None: ("999", ref))
    with pytest.raises(AssertionError) as caught:
        curation.test_a_sample_of_citations_name_the_cited_paper(
            _one_sourced(10.0, citation="PMID:111"))
    assert "PMID:999" in str(caught.value)


def test_an_agreeing_value_passes(monkeypatch):
    """Non-vacuity for the three above: the same path can succeed."""
    monkeypatch.setattr(
        "cognisom.library.bionumbers.fetch",
        lambda bnid, session=None: BioNumberRecord(
            bnid=bnid, value=10.0, unit="um", reference="Smith 2020"))
    curation.test_a_sample_of_records_refetch_to_the_stored_value(
        _one_sourced(10.0))


# ── and the gate itself ───────────────────────────────────────────────

def test_the_live_checks_are_marked_network():
    """If the marker is dropped, the collection skip stops applying."""
    for name in ("test_a_sample_of_records_refetch_to_the_stored_value",
                 "test_a_sample_of_citations_name_the_cited_paper"):
        fn = getattr(curation, name)
        marks = {m.name for m in getattr(fn, "pytestmark", [])}
        assert "network" in marks, f"{name} lost its network marker"


def test_pytest_ini_no_longer_claims_an_unimplemented_default():
    """It said "deselected by default" while nothing deselected it."""
    from pathlib import Path
    ini = (Path(__file__).resolve().parent.parent / "pytest.ini").read_text()
    body = re.search(r"network:(.*?)(?=\n\w|\nasyncio|\Z)", ini, re.S).group(1)
    assert "--run-network" in body
    assert "deselected by default" not in body
