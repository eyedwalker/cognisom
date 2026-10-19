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


def resolve_pmid(reference: str,
                 session: Optional[requests.Session] = None) -> tuple:
    """``(pmid, title)`` for a reference string, or ``("", "")``.

    The title is read back and matched against the reference, because an
    esearch that returns something is not the same as an esearch that
    returned the right thing -- and a citation naming the wrong paper is
    worse than no citation.
    """
    # The title is the part most likely to be unique; drop the author list.
    query = re.sub(r"^[^,]*,\s*", "", reference).split(".")[0]
    if len(query) < 20:
        query = reference
    s = session or _session()
    try:
        found = s.get(PUBMED_SEARCH, params={
            "db": "pubmed", "retmode": "json", "term": query, "retmax": "3",
        }, timeout=30).json()["esearchresult"]
    except Exception as exc:                            # pragma: no cover
        log.warning("PubMed search failed for %r: %s", query[:60], exc)
        return "", ""
    time.sleep(POLITE_PAUSE_S)

    for pmid in found.get("idlist", []):
        try:
            summary = s.get(PUBMED_SUMMARY, params={
                "db": "pubmed", "retmode": "json", "id": pmid,
            }, timeout=30).json()["result"]
        except Exception:                               # pragma: no cover
            continue
        time.sleep(POLITE_PAUSE_S)
        entry = summary.get(pmid, {})
        title = entry.get("title", "")
        if _title_matches(reference, title):
            return pmid, title
    return "", ""


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


def accept(entity: BioEntity, parameter: str, record: BioNumberRecord,
           value: Optional[float] = None, note: str = "") -> List[str]:
    """Write a chosen record onto an entity, citing the paper behind it.

    ``value`` overrides the record's own number, for the case where a unit
    conversion is needed; state it in ``note`` when you do. The write goes
    through :func:`set_parameter`, so the spec's unit and range still apply
    and a value that fails them is rejected rather than stored.
    """
    if not record.citable:
        return [
            f"BNID {record.bnid} is not citable: "
            + ("no value parsed" if record.value is None
               else "its reference did not resolve to a PubMed record")
        ]
    spec = PARAMETER_SPECS.get(parameter)
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
