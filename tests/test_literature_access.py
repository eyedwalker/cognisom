"""Which paper you got, and how much of it.

A value in this library carries a PMID and the grade MEASURED, which
together imply somebody read the measurement. Frequently nobody could:
most of the papers cited here are abstract-only from any automated
route, and an abstract-derived number has no table behind it and no
method. Recording the TIER is what lets those be told apart.

The bug these tests exist for was found on the first run of the module
that was meant to fix the provenance hole, and it is the same shape as
every other one in this area -- a search returned something and
something was taken as the right thing.

Europe PMC indexes several databases, and ``EXT_ID`` matches across all
of them, so a PubMed identifier collides with an unrelated record
carrying the same number in another source:

    EXT_ID:308396   -> src=CBA, "A study on the extraction of
                       flavonoids from tea"
    EXT_ID:406262   -> src=CBA, "Upland rice variety diversity"

Those two PMIDs are the lymphocyte nuclear volume and the nuclear pore
count -- both values this library stores. Without ``SRC:MED`` the
retrieval would have attached a tea abstract to a cell measurement and
reported it as successfully fetched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cognisom.library import literature as lit    # noqa: E402


class _Response:
    def __init__(self, payload=None, text="", status=200, headers=None):
        self._payload = payload if payload is not None else {}
        self.text = text
        self.status_code = status
        self.headers = headers or {}

    def json(self):
        return self._payload

    @property
    def content(self):
        return self.text.encode()


class _Session:
    """Records the queries made, and answers them from a script."""

    def __init__(self, search=None, body=None):
        self.search = search or {}
        self.body = body or {}
        self.queries = []

    def get(self, url, params=None, timeout=None):
        params = params or {}
        if url.endswith("/search"):
            query = params.get("query", "")
            self.queries.append(query)
            return _Response(self.search.get(query, {
                "resultList": {"result": []}}))
        for key, text in self.body.items():
            if key in url:
                return _Response(text=text)
        return _Response(text="", status=404)


def _record(pmid, title, **extra):
    row = {"pmid": pmid, "title": title, "source": "MED",
           "abstractText": "An abstract.", "inEPMC": "N"}
    row.update(extra)
    return {"resultList": {"result": [row]}}


# ── the source qualifier ─────────────────────────────────────────────

def test_the_query_restricts_to_medline():
    """Without SRC:MED a PMID matches a record in another database that
    happens to carry the same number."""
    session = _Session(search={
        "EXT_ID:308396 AND SRC:MED": _record("308396", "Stereological model system"),
    })
    article = lit.fetch("308396", session)
    assert session.queries == ["EXT_ID:308396 AND SRC:MED"]
    assert "Stereological" in article.title
    assert article.tier == lit.ABSTRACT


def test_a_record_for_a_different_paper_is_refused():
    """Belt and braces: the qualifier could stop working, and a wrong
    paper retrieved confidently is worse than nothing retrieved."""
    session = _Session(search={
        "EXT_ID:406262 AND SRC:MED": _record("999999", "Upland rice variety diversity"),
    })
    article = lit.fetch("406262", session)
    assert article.tier == lit.NONE
    assert any("refusing" in a for a in article.attempts), article.attempts
    assert "rice" not in article.title.lower()


def test_no_medline_record_is_not_an_abstract():
    session = _Session(search={})
    article = lit.fetch("123", session)
    assert article.tier == lit.NONE
    assert article.text == ""


# ── the tiers ────────────────────────────────────────────────────────

def test_an_abstract_is_not_reported_as_full_text():
    """The distinction the module exists to make. A number from an
    abstract has no method and no table behind it."""
    session = _Session(search={
        "EXT_ID:1 AND SRC:MED": _record("1", "A paper"),
    })
    article = lit.fetch("1", session)
    assert article.tier == lit.ABSTRACT
    assert not article.full_text
    assert article.route == "europepmc:abstract"


def test_full_text_wins_over_the_abstract_when_it_is_there():
    body = "<article><body><sec><p>" + "x" * 9000 + "</p></sec></body></article>"
    session = _Session(
        search={"EXT_ID:2 AND SRC:MED": _record(
            "2", "A paper", inEPMC="Y", pmcid="PMC2")},
        body={"PMC2/fullTextXML": body})
    article = lit.fetch("2", session)
    assert article.full_text
    assert article.route == "europepmc:fullTextXML"
    assert "<" not in article.text and "x" * 50 in article.text


def test_every_route_tried_is_recorded():
    """A failure has to be diagnosable. "Nothing came back" is not a
    reason, and the three routes disagree often enough that which one
    refused matters."""
    session = _Session(search={
        "EXT_ID:3 AND SRC:MED": _record("3", "A paper", inEPMC="Y",
                                        pmcid="PMC3", doi="10.1/x")})
    article = lit.fetch("3", session)
    assert len(article.attempts) >= 2
    assert any("europepmc" in a for a in article.attempts)
    assert any("ncbi" in a or "unpaywall" in a for a in article.attempts)


# ── the credential ───────────────────────────────────────────────────

def test_the_key_raises_the_rate_limit(monkeypatch):
    """Free, and it is the difference between 3 requests a second and
    10. It is still a credential: it is read from the environment and
    never written into the repository."""
    monkeypatch.delenv(lit.API_KEY_ENV, raising=False)
    assert lit.api_key() == ""
    slow = lit._pause()

    monkeypatch.setenv(lit.API_KEY_ENV, "a-key")
    assert lit.api_key() == "a-key"
    fast = lit._pause()
    assert fast < slow
    assert 1 / fast > 9 and 1 / slow < 4


def test_the_key_is_sent_to_ncbi_and_nowhere_else(monkeypatch):
    monkeypatch.setenv(lit.API_KEY_ENV, "a-key")
    assert lit._ncbi_params(db="pmc")["api_key"] == "a-key"
    monkeypatch.delenv(lit.API_KEY_ENV)
    assert "api_key" not in lit._ncbi_params(db="pmc")


def test_xml_stripping_keeps_the_numbers():
    """Tables and figure legends are where the measurements are, which
    is the whole reason an abstract will not do."""
    xml = ("<article><front><journal>Ignore</journal></front><body>"
           "<table-wrap><label>Table 2</label><td>44.4%</td></table-wrap>"
           "<p>The nucleus occupied 44.4% of cell volume.</p></body>"
           "<back><ref-list><ref>Smith 1999</ref></ref-list></back></article>")
    text = lit._strip_xml(xml)
    assert "44.4%" in text and "Table 2" in text
    assert "Smith 1999" not in text, "the reference list is not the paper"
    assert "<" not in text
