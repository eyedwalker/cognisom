"""Get a paper's text, and say how much of it you got.

Values in this library carry a PMID and the grade MEASURED. That pair
implies somebody read the measurement. Often nobody could: of the
eighteen papers the library currently cites, six are reachable in full
and twelve are abstract-only. The microvillus length of 0.35 um is
graded MEASURED with a PMID, and the note from the pass that sourced it
reads "Abstract only -- Blood's full text is behind Cloudflare. I have
the number but not the per-source table."

An abstract-derived number and a table-verified one are presently
indistinguishable in the library. That is the hole this module exists to
close: every retrieval reports which TIER it reached, so a value can
record what was actually read rather than only which paper it came from.

THE TIERS, best first
---------------------
``full_text``   the article body: methods, tables, figure legends. The
                only tier where a number can be checked against how it
                was measured.
``abstract``    the abstract alone. A number here has no table behind
                it and no method; it is a claim about a measurement
                rather than the measurement.
``none``        not retrievable.

WHY SEVERAL ROUTES
------------------
No single source covers it. Europe PMC serves full text for the
open-access subset; NCBI's efetch serves some PMC articles Europe PMC
refuses with a 500; and Unpaywall finds author manuscripts and
repository copies for articles in neither. They disagree, so all three
are tried, and the open-access flag is not trusted on its own: Unpaywall
reports the Blood paper above as open access -- "bronze", free to read
on the publisher's own site -- and the publisher returns 403 to any
automated fetch. An OA flag describes a licence, not a door.
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests

log = logging.getLogger(__name__)

#: An NCBI key raises the E-utilities limit from 3 requests a second to
#: 10. It is free and it is not a key to anything private, but it is
#: still a credential: it goes in the gitignored .env beside the BRENDA
#: ones and is read from the environment, never written into the repo.
API_KEY_ENV = "NCBI_API_KEY"

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
UNPAYWALL = "https://api.unpaywall.org/v2"

#: Unpaywall asks for a contact address; it is not authentication.
UNPAYWALL_EMAIL_ENV = "UNPAYWALL_EMAIL"

FULL_TEXT = "full_text"
ABSTRACT = "abstract"
NONE = "none"


def api_key() -> str:
    """The NCBI key, or "" if none is configured."""
    return (os.environ.get(API_KEY_ENV) or "").strip()


def _pause() -> float:
    """Seconds between NCBI calls: 10/s with a key, 3/s without."""
    return 0.11 if api_key() else 0.35


@dataclass
class Article:
    """What was retrieved for one paper, and how far it got."""

    pmid: str = ""
    pmcid: str = ""
    doi: str = ""
    title: str = ""
    tier: str = NONE
    text: str = ""
    #: Which route produced the text, for re-retrieval.
    route: str = ""
    #: Every route tried and what it said, so a failure is diagnosable
    #: rather than just a zero.
    attempts: List[str] = field(default_factory=list)

    @property
    def full_text(self) -> bool:
        return self.tier == FULL_TEXT

    def summary(self) -> str:
        return (f"PMID:{self.pmid or '-'} {self.tier} "
                f"({len(self.text)} chars via {self.route or 'nothing'})")


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "cognisom-library/1.0 (research use)"})
    return s


def _ncbi_params(**extra) -> Dict[str, str]:
    params = dict(extra)
    key = api_key()
    if key:
        params["api_key"] = key
    return params


def _europepmc(pmid: str, session) -> Tuple[Optional[Article], str]:
    try:
        found = session.get(f"{EPMC}/search", params={
            # SRC:MED is not optional. Europe PMC indexes several
            # databases and EXT_ID matches across all of them, so a PMID
            # collides with an unrelated record carrying the same number
            # in another source. Measured: EXT_ID:308396 returns a
            # Chinese Biological Abstracts record titled "A study on the
            # extraction of flavonoids from tea", and EXT_ID:406262
            # returns "Upland rice variety diversity". Those two PMIDs
            # are the lymphocyte nuclear volume and the nuclear pore
            # count -- two of the values this library stores. Without the
            # qualifier this module would have attached a tea abstract to
            # a cell measurement and reported it as retrieved.
            "query": f"EXT_ID:{pmid} AND SRC:MED", "format": "json",
            "resultType": "core", "pageSize": "1"}, timeout=30).json()
    except Exception as exc:
        return None, f"europepmc search failed: {exc}"
    results = (found.get("resultList") or {}).get("result") or []
    if not results:
        return None, "europepmc: no MEDLINE record"
    row = results[0]
    # And check the record returned is the one asked for, because a
    # qualifier that silently stops matching would fail the same way.
    if str(row.get("pmid") or "") != str(pmid):
        return None, (f"europepmc returned pmid {row.get('pmid')} for a "
                      f"request for {pmid}; refusing it")
    article = Article(pmid=pmid, pmcid=row.get("pmcid") or "",
                      doi=row.get("doi") or "", title=row.get("title") or "")
    if row.get("abstractText"):
        article.tier, article.text = ABSTRACT, row["abstractText"]
        article.route = "europepmc:abstract"

    if row.get("inEPMC") == "Y" and article.pmcid:
        try:
            body = session.get(f"{EPMC}/{article.pmcid}/fullTextXML", timeout=40)
            if body.status_code == 200 and len(body.content) > 5000:
                article.tier, article.text = FULL_TEXT, _strip_xml(body.text)
                article.route = "europepmc:fullTextXML"
                return article, "europepmc: full text"
            return article, (f"europepmc fullTextXML: HTTP "
                             f"{body.status_code}, {len(body.content)} bytes")
        except Exception as exc:
            return article, f"europepmc fullTextXML failed: {exc}"
    return article, f"europepmc: inEPMC={row.get('inEPMC')}, pmcid={article.pmcid or '-'}"


def _ncbi_fulltext(article: Article, session) -> str:
    """NCBI's efetch, which serves some PMC articles Europe PMC refuses."""
    if not article.pmcid:
        return "ncbi: no pmcid"
    time.sleep(_pause())
    try:
        got = session.get(f"{EUTILS}/efetch.fcgi", params=_ncbi_params(
            db="pmc", id=article.pmcid.replace("PMC", ""), retmode="xml"),
            timeout=40)
    except Exception as exc:
        return f"ncbi efetch failed: {exc}"
    body = got.text
    if got.status_code == 200 and ("<body" in body or "<sec " in body) \
            and len(body) > 20000:
        article.tier, article.text = FULL_TEXT, _strip_xml(body)
        article.route = "ncbi:efetch"
        return "ncbi: full text"
    return f"ncbi efetch: {len(body)} chars, no article body"


def _unpaywall(article: Article, session) -> str:
    """Where else a copy might be, when neither repository has it."""
    if not article.doi:
        return "unpaywall: no doi"
    email = (os.environ.get(UNPAYWALL_EMAIL_ENV) or "").strip()
    if not email:
        return "unpaywall: no contact address configured"
    try:
        data = session.get(f"{UNPAYWALL}/{article.doi}",
                           params={"email": email}, timeout=30).json()
    except Exception as exc:
        return f"unpaywall failed: {exc}"
    if not data.get("is_oa"):
        return "unpaywall: not open access"
    for location in (data.get("oa_locations") or []):
        url = location.get("url_for_pdf") or location.get("url")
        if not url:
            continue
        try:
            got = session.get(url, timeout=40)
        except Exception:
            continue
        if got.status_code == 200 and len(got.content) > 20000 \
                and "pdf" in (got.headers.get("content-type") or "").lower():
            article.route = f"unpaywall:{location.get('host_type')}"
            # The bytes are a PDF; this module does not parse one, so the
            # tier does not improve. Recording the location is still
            # worth it: it says a readable copy EXISTS, which is
            # different from the paper being unavailable.
            return f"unpaywall: pdf at {location.get('host_type')} ({len(got.content)//1024} KB)"
    return (f"unpaywall: {data.get('oa_status')} but no fetchable copy "
            "(a bronze licence is free to read on the publisher's own site "
            "and routinely blocks automated access)")


def fetch(pmid: str, session=None) -> Article:
    """Everything retrievable for one PMID, with the tier it reached."""
    session = session or _session()
    pmid = str(pmid).strip()

    # A PDF somebody already fetched beats anything online, and costs no
    # request at all.
    offline = local_pdf(pmid)

    article, note = _europepmc(pmid, session)
    if article is None:
        stub = Article(pmid=pmid, attempts=[note])
        if offline:
            stub.tier, stub.text, stub.route = FULL_TEXT, offline, "local:pdf"
        return stub
    article.attempts.append(note)

    if offline:
        article.tier, article.text = FULL_TEXT, offline
        article.route = "local:pdf"
        article.attempts.append(f"local pdf: {len(offline)} chars")
        return article

    if not article.full_text:
        article.attempts.append(_ncbi_fulltext(article, session))
    if not article.full_text:
        article.attempts.append(_unpaywall(article, session))
    return article


def _strip_xml(xml: str) -> str:
    """Readable text out of JATS, keeping tables and figure legends.

    The numbers worth having are usually in a table or a legend rather
    than the running prose, which is the whole reason an abstract is not
    good enough.
    """
    text = re.sub(r"<\?xml.*?\?>", " ", xml, flags=re.S)
    text = re.sub(r"<(ref-list|back|front)\b.*?</\1>", " ", text, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&amp;", "&", text)
    return re.sub(r"\s+", " ", text).strip()


def reachability(pmids: List[str], session=None) -> Dict[str, str]:
    """``pmid -> tier`` for a list, for auditing what a pass could read."""
    session = session or _session()
    out: Dict[str, str] = {}
    for pmid in pmids:
        out[str(pmid)] = fetch(pmid, session).tier
        time.sleep(_pause())
    return out


#: Where a PDF fetched by hand is looked for, named ``<PMID>.pdf``.
#:
#: Most of what this library cites is paywalled to an automated fetch and
#: readable to a person with institutional access. Rather than hand this
#: module a credential -- which would be broad, tied to one person, and
#: the sort of thing that ends up in a log -- a PDF dropped in this
#: directory is read from disk. The directory is gitignored: these are
#: licensed copies and they are not ours to redistribute.
LOCAL_PDFS = "docs/papers"


def local_pdf(pmid: str, root: str = LOCAL_PDFS) -> Optional[str]:
    """Text of a PDF placed here by hand, or None.

    Checked BEFORE the network, so a paper somebody has already fetched
    is never requested again, and so the full text wins over the
    abstract that the online routes would otherwise settle for.
    """
    import pathlib

    path = pathlib.Path(root) / f"{str(pmid).strip()}.pdf"
    if not path.exists():
        return None
    try:
        import pypdf
    except ImportError:                                 # pragma: no cover
        log.warning("pypdf is not installed; %s cannot be read", path)
        return None
    try:
        reader = pypdf.PdfReader(str(path))
        text = " ".join((page.extract_text() or "") for page in reader.pages)
    except Exception as exc:
        log.warning("could not read %s: %s", path, exc)
        return None
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) > 500 else None
