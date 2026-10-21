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

import os
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


# ── a PDF fetched by hand ────────────────────────────────────────────

def test_a_local_pdf_is_read_before_anything_online(tmp_path, monkeypatch):
    """Most of what this library cites is paywalled to an automated
    fetch and readable to a person with institutional access. A PDF
    dropped on disk beats the abstract the online routes settle for, and
    costs no request."""
    pytest.importorskip("pypdf")
    from pypdf import PdfWriter

    papers = tmp_path / "papers"
    papers.mkdir()
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with open(papers / "6775712.pdf", "wb") as fh:
        writer.write(fh)

    # A blank page yields no text, so the module must fall through
    # rather than report a full text it did not get.
    monkeypatch.setattr(lit, "LOCAL_PDFS", str(papers))
    assert lit.local_pdf("6775712") is None
    assert lit.local_pdf("nonexistent") is None


def test_a_local_pdf_with_text_wins(tmp_path, monkeypatch):
    monkeypatch.setattr(lit, "local_pdf",
                        lambda pmid, root=None: "Table 2. Granules 23.0%" + "x" * 600
                        if pmid == "6775712" else None)
    session = _Session(search={
        "EXT_ID:6775712 AND SRC:MED": _record("6775712", "Morphometry of human leukocytes"),
    })
    article = lit.fetch("6775712", session)
    assert article.full_text
    assert article.route == "local:pdf"
    assert "23.0%" in article.text

    # and a paper with no local copy still goes online
    other = lit.fetch("1", _Session(search={
        "EXT_ID:1 AND SRC:MED": _record("1", "Another paper")}))
    assert other.tier == lit.ABSTRACT


def test_the_pdf_directory_is_not_committed():
    """These are licensed copies. They are not ours to redistribute."""
    ignore = (REPO_ROOT / ".gitignore").read_text()
    assert "docs/papers" in ignore


# ── the PDF a publisher actually produces ────────────────────────────

def _descriptor_with(*keys):
    """A font descriptor declaring each of ``keys`` as a font program."""
    from pypdf.generic import DictionaryObject, NameObject, NumberObject
    d = DictionaryObject()
    d[NameObject("/FontName")] = NameObject("/ZUPGTX+Times-Bold")
    for key in keys:
        d[NameObject(key)] = NumberObject(1)
    return d


class _FakeReader:
    def __init__(self, pages):
        self.pages = pages


def test_a_font_declared_twice_does_not_cost_the_paper():
    """pypdf refuses a page outright when one font descriptor carries two
    embedded programs -- PdfReadError: More than one /FontFile found. The
    spec does say at most one; publisher exports emit two anyway. Every
    page of the ScienceDirect copy of PMID 6775712 carries /FontFile and
    /FontFile3 for the same Times face, so ten pages of the paper that
    four stored granule fractions come from extracted to nothing."""
    from pypdf.generic import DictionaryObject, NameObject

    good = _descriptor_with("/FontFile2")
    bad = _descriptor_with("/FontFile", "/FontFile3")
    fonts = DictionaryObject()
    fonts[NameObject("/F1")] = DictionaryObject()
    fonts[NameObject("/F1")][NameObject("/FontDescriptor")] = bad
    fonts[NameObject("/F2")] = DictionaryObject()
    fonts[NameObject("/F2")][NameObject("/FontDescriptor")] = good
    resources = DictionaryObject()
    resources[NameObject("/Font")] = fonts

    repaired = lit._drop_duplicate_font_files(
        _FakeReader([{"/Resources": resources}]))
    assert repaired == 1
    # one program left, and it is the FIRST declared -- not whichever
    # the dict happened to yield last
    assert [k for k in lit._FONT_FILE_KEYS if k in bad] == ["/FontFile"]
    # a descriptor that was already legal is untouched
    assert "/FontFile2" in good


def test_the_repair_leaves_a_well_formed_pdf_alone():
    from pypdf.generic import DictionaryObject, NameObject
    fonts = DictionaryObject()
    fonts[NameObject("/F1")] = DictionaryObject()
    fonts[NameObject("/F1")][NameObject("/FontDescriptor")] = _descriptor_with("/FontFile")
    resources = DictionaryObject()
    resources[NameObject("/Font")] = fonts
    assert lit._drop_duplicate_font_files(_FakeReader([{"/Resources": resources}])) == 0


def test_one_unreadable_page_does_not_discard_the_others(tmp_path, monkeypatch):
    """A paper is worth nine tenths of itself. The old reader joined a
    generator inside one try, so any page that raised took the whole
    document with it and returned None."""
    class _Page:
        def __init__(self, text, boom=False):
            self.text, self.boom = text, boom
        def get(self, _key):
            return None
        def extract_text(self):
            if self.boom:
                raise ValueError("unparseable content stream")
            return self.text

    pages = [_Page("x" * 400), _Page("", boom=True), _Page("Table 2. 23.0%" + "y" * 400)]
    import pypdf
    monkeypatch.setattr(pypdf, "PdfReader", lambda _p: _FakeReader(pages))
    papers = tmp_path / "papers"
    papers.mkdir()
    (papers / "6775712.pdf").write_bytes(b"%PDF-1.4\n")

    text = lit.local_pdf("6775712", root=str(papers))
    assert text and "23.0%" in text


# ── credentials that are configured and unread ───────────────────────

def test_the_env_file_is_actually_loaded(tmp_path, monkeypatch):
    """Both credentials this module uses had been added to .env and
    neither ever reached the code, because nothing in the package read
    the file. The NCBI key sat there for days while every E-utilities
    call ran at the keyless 3 requests a second, and the Unpaywall
    address sat there with the third fetch route switched off.

    A credential that is configured and unread fails exactly like one
    that was never supplied, except that somebody believes it works.
    """
    env = tmp_path / ".env"
    env.write_text(
        "# a comment\n"
        "\n"
        "NCBI_API_KEY=from-the-file\n"
        'UNPAYWALL_EMAIL="quoted@example.org"\n'
        "MALFORMED\n")
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    monkeypatch.delenv("UNPAYWALL_EMAIL", raising=False)
    monkeypatch.setattr(lit, "_ENV_LOADED", False)

    assert lit.load_env(str(env)) == 2
    assert os.environ["NCBI_API_KEY"] == "from-the-file"
    assert os.environ["UNPAYWALL_EMAIL"] == "quoted@example.org"


def test_the_real_environment_wins_over_the_file(tmp_path, monkeypatch):
    """A value exported in a shell is more deliberate than one sitting
    in a file, so overriding it would be its own surprise."""
    env = tmp_path / ".env"
    env.write_text("NCBI_API_KEY=from-the-file\n")
    monkeypatch.setenv("NCBI_API_KEY", "from-the-shell")
    monkeypatch.setattr(lit, "_ENV_LOADED", False)

    lit.load_env(str(env))
    assert os.environ["NCBI_API_KEY"] == "from-the-shell"


def test_a_missing_env_file_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(lit, "_ENV_LOADED", False)
    assert lit.load_env(str(tmp_path / "nothing-here")) == 0


def test_api_key_loads_the_file_on_demand(tmp_path, monkeypatch):
    """The point of the whole thing: a caller asks for the key and gets
    it, without having to know a file exists."""
    env = tmp_path / ".env"
    env.write_text("NCBI_API_KEY=lazy-load\n")
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    monkeypatch.setattr(lit, "_ENV_LOADED", False)
    monkeypatch.chdir(tmp_path)
    assert lit.api_key() == "lazy-load"
    assert 1 / lit._pause() > 9


# ── a PDF Unpaywall finds is read, not just noted ────────────────────

def test_a_fetched_pdf_is_parsed_rather_than_described(monkeypatch):
    """This route used to record "pdf at publisher (5711 KB)" and stop,
    so a paper that had been found AND downloaded went on reporting as
    abstract-only. One of the sixteen the library still wants is
    reachable exactly this way."""
    pytest.importorskip("pypdf")

    body = "Km of (200 plus-minus 80) uM, kcat of (18 plus-minus 4) per s. " * 40
    monkeypatch.setattr(lit, "_pdf_text", lambda data: body)

    class _R:
        status_code = 200
        content = b"%PDF-1.4" + b"x" * 30000
        headers = {"content-type": "application/pdf"}

        def json(self):
            return {}

    class _S:
        def get(self, url, params=None, timeout=None):
            if "unpaywall" in url:
                class _J:
                    @staticmethod
                    def json():
                        return {"is_oa": True, "oa_status": "bronze",
                                "oa_locations": [{"host_type": "publisher",
                                                  "url_for_pdf": "http://x/y.pdf"}]}
                return _J()
            return _R()

    monkeypatch.setenv(lit.UNPAYWALL_EMAIL_ENV, "someone@example.org")
    article = lit.Article(pmid="1", doi="10.1/x")
    note = lit._unpaywall(article, _S())
    assert article.full_text
    assert article.route == "unpaywall:publisher"
    assert "kcat" in article.text
    assert "full text from a publisher pdf" in note


def test_a_scanned_pdf_says_so_rather_than_claiming_full_text(monkeypatch):
    """No text out of it is a scan, and reporting full_text for a scan
    would be worse than reporting nothing."""
    monkeypatch.setattr(lit, "_pdf_text", lambda data: "")

    class _S:
        def get(self, url, params=None, timeout=None):
            if "unpaywall" in url:
                class _J:
                    @staticmethod
                    def json():
                        return {"is_oa": True, "oa_status": "green",
                                "oa_locations": [{"host_type": "repository",
                                                  "url_for_pdf": "http://x/y.pdf"}]}
                return _J()

            class _R:
                status_code = 200
                content = b"%PDF" + b"x" * 30000
                headers = {"content-type": "application/pdf"}
            return _R()

    monkeypatch.setenv(lit.UNPAYWALL_EMAIL_ENV, "someone@example.org")
    article = lit.Article(pmid="1", doi="10.1/x")
    note = lit._unpaywall(article, _S())
    assert not article.full_text
    assert "no text could be extracted" in note
