"""Candidate values from BioNumbers, for the quantities no API will hand you.

Most of what this library still needs -- how fast a cell divides, how fast a
neutrophil crawls, how long one lives -- is not in a protein database. It is in
the primary literature, and BioNumbers is the curated index of it: each record
carries a value, its units, the organism, the method, a verbatim quote from the
paper, and a reference that resolves to a single PubMed identifier.

This module deliberately does not write anything by itself. BioNumbers answers
"what has been measured", not "what applies to your model", and the difference
matters: the best-matching record for neutrophil migration speed reports 17
um/min in *burn patients*, measured in a microfluidic device. Whether that is
the right number for a neutrophil crawling through a tumour is a modelling
judgement, and a scraper that made it silently would be manufacturing evidence
at speed.

So :func:`candidates` fetches and ranks records with their context, and
:func:`accept` writes one only when a caller has chosen it. That is the same
line the UniProt sourcing draws at multi-chain proteins: fetch what is
unambiguous, surface what is not, and refuse to guess.

The reference is verified before it is cited. A reference string is resolved
through PubMed and the returned title is matched back against it, so a citation
only ever names a record that exists and says what the value claims.
"""
from __future__ import annotations

import html
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import requests

from .models import BioEntity
from .parameters import PARAMETER_SPECS, ParameterValue, Provenance, set_parameter

log = logging.getLogger(__name__)

SEARCH_URL = "https://bionumbers.hms.harvard.edu/search.aspx"
RECORD_URL = "https://bionumbers.hms.harvard.edu/bionumber.aspx"
PUBMED_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_SUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

#: BioNumbers is a small academic site. One request at a time, with a pause.
POLITE_PAUSE_S = 0.4

_CACHE: Dict[str, str] = {}


@dataclass
class BioNumberRecord:
    """One measured value, with everything needed to judge and cite it."""

    bnid: str
    #: What is being measured, from the record's heading. Named
    #: `quantity` and not `property` because the latter shadows the
    #: decorator inside a class body.
    quantity: str = ""
    value: Optional[float] = None
    unit: str = ""
    range_text: str = ""
    organism: str = ""
    reference: str = ""
    method: str = ""
    comments: str = ""
    pmid: str = ""
    pmid_title: str = ""

    @property
    def is_human(self) -> bool:
        return "homo sapiens" in self.organism.lower()

    @property
    def citable(self) -> bool:
        """A value and a verified reference. Without both it is a lead."""
        return self.value is not None and bool(self.pmid)

    def describe(self) -> str:
        bits = [f"BNID {self.bnid}", self.quantity or "(no quantity)"]
        if self.value is not None:
            bits.append(f"{self.value:g} {self.unit}".strip())
        if self.range_text:
            bits.append(self.range_text)
        if self.organism:
            bits.append(self.organism)
        if self.pmid:
            bits.append(f"PMID:{self.pmid}")
        return " | ".join(bits)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "Cognisom/1.0 (cognisom.com; research)"})
    return s


def _get(url: str, params: Optional[dict] = None,
         session: Optional[requests.Session] = None) -> str:
    key = url + repr(sorted((params or {}).items()))
    if key in _CACHE:
        return _CACHE[key]
    s = session or _session()
    response = s.get(url, params=params, timeout=30)
    response.raise_for_status()
    _CACHE[key] = response.text
    time.sleep(POLITE_PAUSE_S)
    return response.text


def _text(fragment: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", fragment))).strip()


def search(term: str, session: Optional[requests.Session] = None) -> List[str]:
    """BNIDs matching a free-text term, in the order BioNumbers ranks them."""
    page = _get(SEARCH_URL, {"task": "searchbytrmorg", "trm": term}, session)
    seen, out = set(), []
    for bnid in re.findall(r"bionumber\.aspx\?id=(\d+)", page):
        if bnid not in seen:
            seen.add(bnid)
            out.append(bnid)
    return out


#: "17 µm/min", "320 μm^3", "1.4e-10 m^2/s" -- a number then everything after.
_VALUE = re.compile(r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(.*)$")


def _split_value(raw: str):
    """``("17 µm/min Range: ±6 µm/min")`` -> ``(17.0, "µm/min", "±6 µm/min")``."""
    range_text = ""
    match = re.search(r"Range:\s*(.*)$", raw)
    if match:
        range_text = match.group(1).strip()
        raw = raw[:match.start()].strip()
    m = _VALUE.match(raw)
    if not m:
        return None, raw.strip(), range_text
    try:
        value = float(m.group(1))
    except ValueError:
        return None, raw.strip(), range_text
    return value, m.group(2).strip(), range_text


def fetch(bnid: str, session: Optional[requests.Session] = None) -> BioNumberRecord:
    """One record, parsed. Does not resolve the reference; see :func:`resolve_pmid`."""
    page = _get(RECORD_URL, {"id": bnid}, session)
    record = BioNumberRecord(bnid=str(bnid))

    heading = re.search(r"<h1[^>]*>(.*?)</h1>", page, re.S)
    if heading:
        record.quantity = _text(heading.group(1))
    else:
        title = re.search(r"<title>(.*?)</title>", page, re.S)
        if title:
            record.quantity = _text(title.group(1)).split(" - ")[0]

    cells = [_text(c) for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", page, re.S)]
    cells = [c for c in cells if c]
    fields = {}
    for i in range(len(cells) - 1):
        if cells[i] in ("Value", "Range", "Organism", "Reference", "Method",
                        "Comments", "ID", "Units"):
            fields.setdefault(cells[i], cells[i + 1])

    if "Value" in fields:
        record.value, record.unit, record.range_text = _split_value(fields["Value"])
    elif "Range" in fields:
        record.range_text = fields["Range"]
    record.organism = fields.get("Organism", "")
    record.reference = fields.get("Reference", "")
    record.method = fields.get("Method", "")
    record.comments = fields.get("Comments", "")
    return record


def _title_matches(reference: str, title: str) -> bool:
    """Enough overlap to be sure PubMed returned the cited paper."""
    def words(s):
        return {w for w in re.findall(r"[a-z]{4,}", s.lower())}
    ref_words, title_words = words(reference), words(title)
    if not title_words:
        return False
    return len(ref_words & title_words) / len(title_words) >= 0.6


#: One surname in a reference: optionally a particle (van, de, la), a
#: capitalised name, optionally a second one for a double-barrelled or
#: compound surname.
_NAME = r"[A-Z][\w'\u2019\-]*"
_SURNAME = rf"(?:(?:van|von|de|der|den|du|di|da|la|le|el)\s+)?{_NAME}(?:[ -]{_NAME})?"
#: A whole author list: surnames with initials, comma-separated, closed by
#: the period that ends the list and begins the title.
_AUTHOR_RUN = re.compile(
    rf"^(?:{_SURNAME}\s+[A-Z]{{1,4}}\s*,\s*)*{_SURNAME}\s+[A-Z]{{1,4}}"
    rf"(?:\s+et\s+al\.?)?\s*[,.]\s+")

#: Most BioNumbers references end with the identifier of the paper they
#: name. Eight of twelve records this repo cites carry one, and the four
#: that do not are a PhD thesis, a lab handbook, a textbook rule of thumb
#: and an unreferenced record -- exactly the four that should not be
#: citable. Using it is both exact and stable, where a free-text search
#: depends on PubMed's relevance ranking on the day.
_EMBEDDED_PMID = re.compile(r"PubMed\s*ID\s*:?\s*(\d{4,9})", re.I)
#: Trailing journal, year, volume and page junk, which makes an esearch
#: match nothing if it is left on.
_TAIL = re.compile(r"\.\s*(?:[A-Z][\w.\s&-]{1,40}\.\s*)?\d{4}.*$")


def _title_of(reference: str) -> str:
    r"""The title out of a BioNumbers reference string.

    This used to be ``re.sub(r"^[^,]*,\s*", "", reference).split(".")[0]``,
    which strips to the FIRST COMMA -- that is one author, not the author
    list, so the query it built was never the title the docstring below
    claims to use. Measured on records this repo actually cites:

        Maul GG, Deaven L. ...        -> "Deaven L" (8 chars, so the whole
                                         reference with page numbers went
                                         to esearch and matched nothing;
                                         BNID 101402 was reported
                                         uncitable although its paper
                                         resolves instantly as PMID 406262)
        Fujioka A, Terai K, ... M.    -> "Terai K, Itoh RE, Aoki K, ..."
                                         a bare author list
        Luby-Phelps K. Cytoarchi...   -> "viscosity, diffusion, intra-
                                         cellular surface area", a fragment
                                         from the MIDDLE of the title,
                                         because the title has commas in it

    Three of those happened to resolve anyway, because PubMed will find a
    paper from its author list and the title read-back then confirms it.
    So the function worked by luck and failed silently when the leftover
    was short, discarding good records.
    """
    title = _AUTHOR_RUN.sub("", reference, count=1)
    title = title.split(". ")[0].strip().rstrip(".")
    if len(title) >= 20:
        return title
    # No author list in a form we recognise. Try the reference with the
    # trailing journal and pagination removed, which is still far better
    # than the raw string -- "J Cell Biol. 1977 Jun73(3):748-60" matches
    # nothing. If that leaves too little to search for either, send the
    # reference as it stands rather than a mangled fragment: a book or a
    # thesis has no PubMed record to find anyway, and it is better for it
    # to fail as uncitable than to match some unrelated paper.
    trimmed = _TAIL.sub("", reference).strip()
    return trimmed if len(trimmed) >= 20 else reference


def resolve_pmid(reference: str,
                 session: Optional[requests.Session] = None) -> tuple:
    """``(pmid, title)`` for a reference string, or ``("", "")``.

    The title is read back and matched against the reference, because an
    esearch that returns something is not the same as an esearch that
    returned the right thing -- and a citation naming the wrong paper is
    worse than no citation.
    """
    s = session or _session()

    # The identifier is usually already in the string. Read its title back
    # and match it anyway: an identifier scraped from a page is still only
    # a claim about which paper this is.
    embedded = _EMBEDDED_PMID.search(reference)
    if embedded:
        pmid = embedded.group(1)
        title = _pubmed_title(pmid, s)
        if title and _title_matches(reference, title):
            return pmid, title
        if title:
            log.warning(
                "BioNumbers reference names PubMed ID %s, whose title %r does "
                "not match the reference; searching instead", pmid, title[:60])

    query = _title_of(reference)
    try:
        found = s.get(PUBMED_SEARCH, params={
            "db": "pubmed", "retmode": "json", "term": query, "retmax": "10",
        }, timeout=30).json()["esearchresult"]
    except Exception as exc:                            # pragma: no cover
        log.warning("PubMed search failed for %r: %s", query[:60], exc)
        return "", ""
    time.sleep(POLITE_PAUSE_S)

    # retmax is 10 rather than 3 because esearch sorts by its own relevance
    # and an old paper sinks: the 1977 reference behind the mitochondrial
    # volume record is 79 hits deep in a free-text search for its own title.
    for pmid in found.get("idlist", []):
        title = _pubmed_title(pmid, s)
        if title and _title_matches(reference, title):
            return pmid, title
    return "", ""


def _pubmed_title(pmid: str, session) -> str:
    """The title PubMed holds for ``pmid``, or "" if it cannot be read."""
    try:
        summary = session.get(PUBMED_SUMMARY, params={
            "db": "pubmed", "retmode": "json", "id": pmid,
        }, timeout=30).json()["result"]
    except Exception:                                   # pragma: no cover
        return ""
    time.sleep(POLITE_PAUSE_S)
    return summary.get(pmid, {}).get("title", "")


def candidates(term: str, limit: int = 8, human_only: bool = True,
               resolve: bool = True,
               session: Optional[requests.Session] = None) -> List[BioNumberRecord]:
    """Records for a term, ranked, with their references resolved.

    Returned for a caller to choose between. Nothing is written: which
    record applies to a given model is a judgement BioNumbers cannot make,
    and neither can this function.
    """
    s = session or _session()
    out: List[BioNumberRecord] = []
    for bnid in search(term, s):
        if len(out) >= limit:
            break
        try:
            record = fetch(bnid, s)
        except Exception as exc:                        # pragma: no cover
            log.warning("BNID %s: %s", bnid, exc)
            continue
        if human_only and record.organism and not record.is_human:
            continue
        if record.value is None:
            continue
        if resolve and record.reference:
            record.pmid, record.pmid_title = resolve_pmid(record.reference, s)
        out.append(record)
    # A record whose reference verified is worth more than one that did not.
    out.sort(key=lambda r: (not r.citable, r.bnid))
    return out


#: The same unit is written several ways across BioNumbers -- the micro
#: SIGN and Greek small mu both appear for micrometres, sometimes in
#: neighbouring records ("\u00b5m^3" for a nuclear volume, "\u03bcm^3" for a
#: mitochondrial one), and exponents come as carets or superscripts.
_UNIT_CHARS = {
    "\u00b5": "u", "\u03bc": "u",        # micro sign, Greek small mu
    "\u00b2": "2", "\u00b3": "3",        # superscript two, three
    "\u00c5": "angstrom", "\u00e5": "angstrom",
}


def _normalise_unit(unit: str) -> str:
    """A unit string reduced to what it means, for comparison only."""
    out = (unit or "").strip()
    for find, repl in _UNIT_CHARS.items():
        out = out.replace(find, repl)
    out = out.lower()
    for junk in ("^", " ", "*", "_"):
        out = out.replace(junk, "")
    return out.replace("angstroms", "angstrom")


def same_unit(a: str, b: str) -> bool:
    """Whether two unit strings denote the same unit."""
    return _normalise_unit(a) == _normalise_unit(b)


def accept(entity: BioEntity, parameter: str, record: BioNumberRecord,
           value: Optional[float] = None, note: str = "") -> List[str]:
    """Write a chosen record onto an entity, citing the paper behind it.

    ``value`` overrides the record's own number, for the case where a unit
    conversion is needed; state it in ``note`` when you do. The write goes
    through :func:`set_parameter`, so the spec's range still applies and a
    value that fails it is rejected rather than stored.

    A record whose unit is not the parameter's is refused. This used to
    pass the spec's unit straight through -- ``unit=spec.unit if spec else
    record.unit`` -- which *relabelled* the record rather than checking it:
    a Golgi membrane thickness of 39.5 angstroms, accepted into a
    parameter declared in micrometres, became "39.5 um", wrong by a factor
    of ten thousand with nothing in the stored value to show it. The
    docstring claimed "the spec's unit still applies", but a unit that is
    overwritten to match can never disagree. Supplying ``value`` is how a
    caller says it has done the conversion deliberately, and that is the
    only way past this.
    """
    if not record.citable:
        return [
            f"BNID {record.bnid} is not citable: "
            + ("no value parsed" if record.value is None
               else "its reference did not resolve to a PubMed record")
        ]
    spec = PARAMETER_SPECS.get(parameter)
    if spec is not None and value is None and record.unit \
            and not same_unit(record.unit, spec.unit):
        return [
            f"BNID {record.bnid} is in {record.unit!r} but {parameter} is "
            f"declared in {spec.unit!r}. Convert it and pass the converted "
            f"number as `value`, saying so in `note`; this is refused "
            f"rather than relabelled because the two differ by more than a "
            f"name ({record.quantity!r})."
        ]
    conditions = "; ".join(p for p in (
        record.organism,
        f"BioNumbers quantity: {record.quantity}" if record.quantity else "",
        f"range {record.range_text}" if record.range_text else "",
        f"method: {record.method[:160]}" if record.method else "",
        note,
    ) if p)
    return set_parameter(entity, parameter, ParameterValue(
        value=float(record.value if value is None else value),
        unit=spec.unit if spec else record.unit,
        provenance=Provenance.MEASURED,
        citation=f"PMID:{record.pmid}",
        source=f"BioNumbers BNID:{record.bnid} ({record.reference[:120]})",
        conditions=conditions,
        rationale=record.comments[:400],
    ), strict=False)
