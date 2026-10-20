"""Whether a BioNumbers record's reference resolves to the right paper.

``accept()`` refuses a record whose reference does not resolve, so this
function decides which measured values the library is allowed to hold.
It had three faults, and together they meant it never once searched for
the thing its own docstring said it searched for.

**It stripped one author, not the author list.** The query was
``re.sub(r"^[^,]*,\\s*", "", reference).split(".")[0]`` -- everything up
to the FIRST COMMA. On the records this repo actually cites:

    Maul GG, Deaven L. Quantitative...   -> "Deaven L"
    Fujioka A, Terai K, ... Matsuda M.   -> "Terai K, Itoh RE, Aoki K, ..."
    Luby-Phelps K. Cytoarchitecture...   -> "viscosity, diffusion, intra-
                                             cellular surface area"

The first is 8 characters, under the 20-character floor, so the whole
reference including ``J Cell Biol. 1977 Jun73(3):748-60`` went to esearch
and matched nothing: BNID 101402 was reported uncitable although its
paper is real and resolves as PMID 406262. The second is a bare author
list. The third is a fragment from the MIDDLE of the title, because that
title contains commas. The ones that worked worked by luck -- PubMed will
find a paper from its authors, and the title read-back then confirmed it.

**It ignored the identifier it already had.** Eight of the twelve records
here carry ``PubMed ID<digits>`` in the reference string. The four that do
not are a PhD thesis, a bench handbook, a textbook rule of thumb and an
unreferenced record -- exactly the four that should not be citable.

**retmax was 3.** esearch sorts by its own relevance and an old paper
sinks: the 1977 reference behind the mitochondrial volume record is 79
hits deep in a free-text search for its own exact title.

These tests are offline. The twelve resolutions below were checked
against live PubMed on 2026-10-20.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cognisom.library import bionumbers as bn

# Verbatim reference strings from the records, with the title each should
# yield and the PubMed id each should resolve to ("" = must not resolve).
REFERENCES = {
    "101402": (
        "Maul GG, Deaven L. Quantitative determination of nuclear pore "
        "complexes in cycling cells with differing DNA content. J Cell Biol. "
        "1977 Jun73(3):748-60PubMed ID406262",
        "Quantitative determination of nuclear pore complexes in cycling "
        "cells with differing DNA content", "406262"),
    "106665": (
        "Fujioka A, Terai K, Itoh RE, Aoki K, Nakamura T, Kuroda S, Nishida "
        "E, Matsuda M. Dynamics of the Ras/ERK MAPK cascade as monitored by "
        "fluorescent probes. J Biol Chem. 2006PubMed ID16418172",
        "Dynamics of the Ras/ERK MAPK cascade as monitored by fluorescent "
        "probes", "16418172"),
    "109399": (
        "Posakony JW, England JM, Attardi G. Mitochondrial growth and "
        "division during the cell cycle in HeLa cells. J Cell Biol. 1977 "
        "Aug74(2):468-91. p.485PubMed ID885911",
        "Mitochondrial growth and division during the cell cycle in HeLa "
        "cells", "885911"),
    "106078": (
        "Glunde K, Guggino SE, Solaiyappan M, Pathak AP, Ichikawa Y, "
        "Bhujwalla ZM. Extracellular acidification alters lysosomal "
        "trafficking in human breast cancer cells. Neoplasia. 2003 "
        "Nov-Dec5(6):533-45. p.540 figure 5aPubMed ID14965446",
        "Extracellular acidification alters lysosomal trafficking in human "
        "breast cancer cells", "14965446"),
    "102474": (
        "Menetret JF et al., The structure of ribosome-channel complexes "
        "engaged in protein translocation. Mol Cell. 2000 Nov6(5):1219-32. "
        "p.1231 left column 2nd paragraphPubMed ID11106759",
        "The structure of ribosome-channel complexes engaged in protein "
        "translocation", "11106759"),
    "105820": (
        "Mitra K, Ubarretxena-Belandia I, Taguchi T, Warren G, Engelman DM. "
        "Modulation of the bilayer thickness of exocytic pathway membranes "
        "by membrane proteins rather than cholesterol. PNAS 2004"
        "PubMed ID15016920",
        "Modulation of the bilayer thickness of exocytic pathway membranes "
        "by membrane proteins rather than cholesterol", "15016920"),
    # No identifier, and nothing that should resolve: a 2006 PhD thesis.
    "103061": (
        "Neil R. Voss. 2006 Geometric Studies of RNA and Ribosomes. "
        "Table 3.4 pp. 97", None, ""),
    # A bench handbook, and a textbook rule of thumb for no organism.
    "100440": ("Roskams and Rodgers, LabRef. 2002 p.98", None, ""),
    "103774": ("Phillips R, Kondev J, Theriot J. Physical Biology of the "
               "Cell. 2009 p.26", None, ""),
}

#: A title with commas in it, which the old first-comma rule cut into.
COMMA_TITLE = (
    "Luby-Phelps K. Cytoarchitecture and physical properties of cytoplasm: "
    "volume, viscosity, diffusion, intracellular surface area. Int Rev "
    "Cytol. 2000192:189-221PubMed ID10553280")


class _Session:
    """Stands in for requests, so these tests need no network."""

    def __init__(self, titles, search=None):
        self.titles = titles            # pmid -> title
        self.search = search or {}      # query -> [pmid]
        self.searches = []
        self.summaries = []

    def get(self, url, params=None, timeout=None):
        params = params or {}
        if url == bn.PUBMED_SEARCH:
            term = params.get("term", "")
            self.searches.append((term, params.get("retmax")))
            ids = self.search.get(term, [])
            return _Response({"esearchresult": {
                "idlist": ids, "count": str(len(ids))}})
        pmid = params.get("id", "")
        self.summaries.append(pmid)
        title = self.titles.get(pmid)
        if title is None:
            return _Response({"result": {}})
        return _Response({"result": {pmid: {"title": title}}})


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def _no_sleeping(monkeypatch):
    monkeypatch.setattr(bn.time, "sleep", lambda _s: None)


# ── pulling the title out of a reference ─────────────────────────────

@pytest.mark.parametrize("bnid", sorted(REFERENCES))
def test_the_title_comes_out_of_the_reference(bnid):
    reference, title, _pmid = REFERENCES[bnid]
    got = bn._title_of(reference)
    if title is None:
        # No author list we recognise, so there is no title to isolate.
        # It must not be a mangled fragment either -- better to send the
        # reference whole and let it fail honestly as uncitable.
        assert len(got) >= 20, got
        assert got in reference
        return
    assert got == title


def test_a_title_containing_commas_survives():
    """The old rule cut to the first comma, which for this record is inside
    the title: it searched for "viscosity, diffusion, intracellular surface
    area"."""
    got = bn._title_of(COMMA_TITLE)
    assert got.startswith("Cytoarchitecture and physical properties")
    assert "viscosity" in got
    assert not got.startswith("viscosity")


def test_a_two_author_reference_does_not_collapse_to_one_name():
    """The failure that discarded BNID 101402: "Deaven L" is 8 characters,
    under the floor, so the raw reference went to esearch."""
    got = bn._title_of(REFERENCES["101402"][0])
    assert got.startswith("Quantitative determination")
    assert "Deaven" not in got and "Maul" not in got


def test_an_eight_author_reference_does_not_yield_an_author_list():
    got = bn._title_of(REFERENCES["106665"][0])
    assert got.startswith("Dynamics of the Ras/ERK")
    for surname in ("Fujioka", "Terai", "Matsuda", "Nishida"):
        assert surname not in got


def test_et_al_is_an_author_list_too():
    got = bn._title_of(REFERENCES["102474"][0])
    assert got == "The structure of ribosome-channel complexes engaged in " \
                  "protein translocation"


def test_the_journal_and_pagination_are_never_in_the_query():
    """"J Cell Biol. 1977 Jun73(3):748-60" is what made the query match
    nothing when the author strip left too little."""
    for reference, _title, _pmid in REFERENCES.values():
        got = bn._title_of(reference)
        assert "PubMed ID" not in got
        for junk in ("J Cell Biol", "748-60", "1219-32", "figure 5a"):
            assert junk not in got, (junk, got)


# ── the identifier that was already there ────────────────────────────

def test_the_embedded_identifier_is_used_when_present():
    reference, title, pmid = REFERENCES["106078"]
    session = _Session({pmid: title})
    assert bn.resolve_pmid(reference, session) == (pmid, title)
    assert session.searches == [], "searched although the id was in the string"
    assert session.summaries == [pmid]


def test_the_embedded_identifier_is_still_verified():
    """An id scraped off a page is a claim about which paper this is. If
    its title does not match the reference, the search still runs."""
    reference, title, pmid = REFERENCES["106078"]
    right = "9999999"
    session = _Session({pmid: "An entirely unrelated paper about turbines",
                        right: title},
                       search={bn._title_of(reference): [right]})
    assert bn.resolve_pmid(reference, session) == (right, title)
    assert session.searches, "did not fall back to a search"


def test_a_reference_with_no_identifier_is_searched_by_title():
    reference, title, pmid = REFERENCES["109399"]
    stripped = reference.replace("PubMed ID885911", "")
    session = _Session({pmid: title}, search={bn._title_of(stripped): [pmid]})
    assert bn.resolve_pmid(stripped, session) == (pmid, title)
    assert session.searches[0][0] == bn._title_of(stripped)


def test_the_search_asks_for_more_than_three_results():
    """esearch sorts by relevance and an old paper sinks: the 1977 paper
    behind the mitochondrial volume is 79 hits deep in a free-text search
    for its own exact title."""
    reference, title, pmid = REFERENCES["109399"]
    stripped = reference.replace("PubMed ID885911", "")
    # the wanted paper last, behind four others
    others = [str(40000000 + i) for i in range(4)]
    session = _Session(
        dict({o: "Some unrelated recent paper" for o in others},
             **{pmid: title}),
        search={bn._title_of(stripped): others + [pmid]})
    assert bn.resolve_pmid(stripped, session) == (pmid, title)
    assert int(session.searches[0][1]) >= 10


def test_a_reference_with_no_paper_behind_it_does_not_resolve():
    """A thesis, a handbook and a textbook rule of thumb. These are the
    records accept() must refuse, and the reason the check exists."""
    for bnid in ("103061", "100440", "103774"):
        reference, _title, _pmid = REFERENCES[bnid]
        session = _Session({}, search={})
        assert bn.resolve_pmid(reference, session) == ("", ""), bnid


def test_a_wrong_paper_is_refused_rather_than_cited():
    """A citation naming the wrong paper is worse than no citation."""
    reference, _title, _pmid = REFERENCES["101402"]
    stripped = reference.replace("PubMed ID406262", "")
    session = _Session({"12345": "Something about the Golgi apparatus"},
                       search={bn._title_of(stripped): ["12345"]})
    assert bn.resolve_pmid(stripped, session) == ("", "")


# ── what the module promises callers ─────────────────────────────────

def test_every_reference_that_names_a_paper_resolves(caplog):
    """The eight records with an identifier all resolve; the three without
    one do not. Checked against live PubMed on 2026-10-20, and pinned here
    offline so a regression in the parsing is caught without a network."""
    resolved = refused = 0
    for bnid, (reference, title, pmid) in REFERENCES.items():
        session = _Session({pmid: title} if pmid else {},
                           search={bn._title_of(reference): [pmid]} if pmid else {})
        got_pmid, got_title = bn.resolve_pmid(reference, session)
        assert got_pmid == pmid, bnid
        if pmid:
            assert got_title == title
            resolved += 1
        else:
            refused += 1
    assert resolved == 6 and refused == 3, (resolved, refused)
