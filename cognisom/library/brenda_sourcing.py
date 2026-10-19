"""Enzyme kinetics from BRENDA, once someone supplies credentials.

SABIO-RK retired its REST API -- every path now redirects to an HTML error
shell -- which left BRENDA as the remaining structured source for turnover
numbers and Michaelis constants. The library needs them: 26 entities carry
kcat or Km and not one of those values has evidence behind it.

BRENDA is free under CC BY 4.0 but its web service requires a registered email
and password, which this module reads from ``BRENDA_EMAIL`` and
``BRENDA_PASSWORD``. Nothing here will create that account: registering with an
outside service in somebody's name is theirs to do. Without credentials every
call raises :class:`BrendaCredentialsMissing` with the two things needed to fix
it, and the rest of the library is unaffected.

The join key is the EC number, which UniProt already supplies for every enzyme
in the library, so :func:`ec_numbers_for` is useful on its own even when
BRENDA is unreachable.

As with the other sourcing modules, candidates are returned and not written.
BRENDA reports every published measurement for an enzyme, across organisms,
substrates, mutants and assay conditions, and choosing among them is the
judgement the module cannot make. A kcat for a point mutant is a real
measurement of the wrong thing.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
from xml.etree import ElementTree

import requests

from .models import BioEntity
from .parameters import PARAMETER_SPECS, ParameterValue, Provenance, set_parameter

log = logging.getLogger(__name__)

#: The documented classic endpoint. The zeep variant exists and accepts the
#: same envelope, but BRENDA's own client examples use this one, and its
#: calling convention is what the service actually reads.
ENDPOINT = "https://www.brenda-enzymes.org/soap/brenda_server.php"
NAMESPACE = "https://www.brenda-enzymes.org/soap"

#: BRENDA packs every record of a response into one string: fields are
#: `name*value` joined by `#`, and records are joined by `!`. Reading the
#: blob as a single record -- which the element structure invites, since
#: there is only ever one <return> element -- silently returns one row of
#: fifty-three.
RECORD_SEPARATOR = "!"
FIELD_SEPARATOR = "#"

#: BRENDA writes this where a paper gave no number. Treating it as a value
#: would record a Km of minus one thousand millimolar.
NO_VALUE = -999.0
POLITE_PAUSE_S = 1.0        # BRENDA asks callers not to hammer a shared pool

#: Where the two credentials come from. Registration is at
#: https://www.brenda-enzymes.org/register.php and is free.
EMAIL_ENV = "BRENDA_EMAIL"
PASSWORD_ENV = "BRENDA_PASSWORD"


class BrendaCredentialsMissing(RuntimeError):
    """No credentials, so no call was attempted."""


class BrendaError(RuntimeError):
    """BRENDA answered, but not with data."""


@dataclass
class KineticRecord:
    """One published kinetic measurement, as BRENDA reports it."""

    ec_number: str
    parameter: str                  # "km" or "kcat"
    value: Optional[float] = None
    unit: str = ""
    substrate: str = ""
    organism: str = ""
    commentary: str = ""
    literature: str = ""
    pmid: str = ""
    title: str = ""
    year: str = ""

    @property
    def is_wild_type(self) -> bool:
        """BRENDA records mutants in the commentary, as "mutant X123Y"."""
        return not re.search(r"\bmutant\b|\bmutation\b|/[A-Z]\d+[A-Z]\b",
                             self.commentary or "", re.I)

    @property
    def is_human(self) -> bool:
        return "homo sapiens" in (self.organism or "").lower()

    @property
    def citable(self) -> bool:
        return self.value is not None and bool(self.pmid)

    def describe(self) -> str:
        bits = [f"{self.parameter}={self.value:g} {self.unit}".strip()
                if self.value is not None else f"{self.parameter} (no value)"]
        if self.substrate:
            bits.append(f"on {self.substrate}")
        if self.organism:
            bits.append(self.organism)
        if not self.is_wild_type:
            bits.append("MUTANT")
        if self.pmid:
            bits.append(f"PMID:{self.pmid}")
        if self.year:
            bits.append(str(self.year))
        return " | ".join(bits)


def _from_env_file(key: str) -> str:
    """Read one key from the project's gitignored .env.

    The library's other credentials live there, and requiring an export in
    every shell is how a working setup becomes a broken one three weeks
    later. Nothing is logged from this file.
    """
    for directory in (os.getcwd(),
                      os.path.dirname(os.path.dirname(os.path.dirname(
                          os.path.abspath(__file__))))):
        path = os.path.join(directory, ".env")
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    name, _, value = line.partition("=")
                    if name.strip() == key:
                        return value.strip().strip("'\"")
        except OSError:                                 # pragma: no cover
            continue
    return ""


def credentials() -> tuple:
    """``(email, sha256_of_password)``, or raise with how to supply them.

    The environment wins over ``.env``, so a shell export can override the
    file without editing it.
    """
    email = (os.environ.get(EMAIL_ENV, "") or _from_env_file(EMAIL_ENV)).strip()
    password = os.environ.get(PASSWORD_ENV, "") or _from_env_file(PASSWORD_ENV)
    if not email or not password:
        raise BrendaCredentialsMissing(
            f"BRENDA needs a registered account. Put {EMAIL_ENV} and "
            f"{PASSWORD_ENV} in the project's .env (which is gitignored), "
            f"or export them. Registration is free at "
            f"https://www.brenda-enzymes.org/register.php. Nothing was "
            f"requested."
        )
    return email, hashlib.sha256(password.encode("utf-8")).hexdigest()


def ec_numbers_for(accession: str,
                   session: Optional[requests.Session] = None) -> List[str]:
    """EC numbers UniProt lists for an accession.

    Useful without BRENDA: it is how the library learns that CD73 is a
    5'-nucleotidase and Caspase-3 a cysteine endopeptidase.
    """
    s = session or requests.Session()
    s.headers.setdefault("User-Agent", "Cognisom/1.0 (cognisom.com; research)")
    response = s.get(
        f"https://rest.uniprot.org/uniprotkb/{accession}.json",
        params={"fields": "protein_name,ec"}, timeout=30)
    response.raise_for_status()
    description = response.json().get("proteinDescription", {}) or {}

    out: List[str] = []
    blocks = [description.get("recommendedName", {})]
    blocks += description.get("alternativeNames", []) or []
    for part in description.get("includes", []) or []:
        blocks.append(part.get("recommendedName", {}) or {})
    for block in blocks:
        for ec in (block or {}).get("ecNumbers", []) or []:
            value = ec.get("value")
            # "3.1.3.-" is a family, not an enzyme; BRENDA cannot use it.
            if value and not value.endswith("-") and value not in out:
                out.append(value)
    return out


def _envelope(method: str, parameters: str) -> str:
    """BRENDA takes one comma-separated string, not separate XML parts.

    The zeep WSDL describes the arguments as separate message parts, which
    is what an earlier version of this module sent. The service parsed that
    envelope and answered with a fault saying the credentials were wrong --
    it had read no credentials at all -- which is a confusing way to learn
    that the convention is the one in BRENDA's own examples:
    ``email,sha256hash,key*value#key*value#``.
    """
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<soapenv:Envelope '
        'xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/" '
        f'xmlns:ws="{NAMESPACE}">'
        f"<soapenv:Body><ws:{method}>"
        '<parameters xsi:type="xsd:string" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xmlns:xsd="http://www.w3.org/2001/XMLSchema">'
        f"{_escape(parameters)}</parameters>"
        f"</ws:{method}></soapenv:Body></soapenv:Envelope>"
    )


def _escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def _call(method: str, arguments: str,
          session: Optional[requests.Session] = None) -> List[Dict[str, str]]:
    """Call a BRENDA method and return its records as dicts."""
    email, password_hash = credentials()
    s = session or requests.Session()
    s.headers.update({"User-Agent": "Cognisom/1.0 (cognisom.com; research)",
                      "Content-Type": "text/xml; charset=utf-8",
                      "SOAPAction": f"{NAMESPACE}#{method}"})
    payload = _envelope(method, f"{email},{password_hash},{arguments}")
    response = s.post(ENDPOINT, data=payload.encode("utf-8"), timeout=90)
    time.sleep(POLITE_PAUSE_S)

    tree = None
    try:
        tree = ElementTree.fromstring(response.content)
    except ElementTree.ParseError:
        tree = None

    if tree is not None:
        fault = tree.find(".//{http://schemas.xmlsoap.org/soap/envelope/}Fault")
        if fault is not None:
            code = fault.findtext("faultcode", "").strip()
            message = fault.findtext("faultstring", "").strip()
            detail = message or "".join(fault.itertext()).strip()[:200]
            if code == "401" or "password" in detail.lower():
                raise BrendaCredentialsMissing(
                    f"BRENDA rejected the credentials: {detail} "
                    f"A new account must be activated from the confirmation "
                    f"email before the web service accepts it; registration "
                    f"alone is not enough."
                )
            raise BrendaError(f"{method}: {detail}")

    if response.status_code != 200:
        raise BrendaError(
            f"{method} answered HTTP {response.status_code} with no SOAP "
            f"fault: {response.text[:200]}"
        )
    if tree is None:
        raise BrendaError(f"{method} returned unparseable XML")

    blob = "".join(el.text or "" for el in tree.iter()
                   if el.text and FIELD_SEPARATOR in el.text)
    return [d for d in (_parse(rec) for rec in blob.split(RECORD_SEPARATOR)) if d]


#: BRENDA packs a record as "field*value#field*value#...".
def _parse(row: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for field in row.split("#"):
        if "*" in field:
            key, _, value = field.partition("*")
            out[key.strip()] = value.strip()
    return out


_REFERENCE_CACHE: Dict[str, Dict[str, Dict[str, str]]] = {}


def _references(ec_number: str,
                session: Optional[requests.Session] = None) -> Dict[str, Dict[str, str]]:
    """BRENDA reference id -> its bibliography, including the PubMed id.

    A measurement's ``literature`` field is a BRENDA-internal reference
    number, not a PubMed id. Writing it as one would produce a citation
    that resolves to an unrelated paper, or to nothing.
    """
    if ec_number in _REFERENCE_CACHE:
        return _REFERENCE_CACHE[ec_number]
    out: Dict[str, Dict[str, str]] = {}
    for record in _call("getReference", f"ecNumber*{ec_number}#", session):
        ref_id = record.get("reference", "").strip()
        if ref_id:
            out[ref_id] = record
    _REFERENCE_CACHE[ec_number] = out
    return out


def _records(rows: List[Dict[str, str]], ec_number: str, parameter: str,
             value_key: str, unit: str,
             references: Optional[Dict[str, Dict[str, str]]] = None
             ) -> List[KineticRecord]:
    references = references or {}
    out = []
    for fields in rows:
        raw = fields.get(value_key, "")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = None
        # BRENDA writes -999 where the paper gave no number.
        if value is not None and value <= NO_VALUE:
            value = None
        literature = fields.get("literature", "").strip()
        bibliography = references.get(literature, {})
        pmid = str(bibliography.get("pubmedId", "")).strip()
        out.append(KineticRecord(
            ec_number=ec_number, parameter=parameter, value=value, unit=unit,
            substrate=fields.get("substrate", ""),
            organism=fields.get("organism", ""),
            commentary=fields.get("commentary", ""),
            literature=literature,
            pmid=pmid if pmid.isdigit() and pmid != "0" else "",
            title=bibliography.get("title", ""),
            year=bibliography.get("year", ""),
        ))
    return out


_EC_SHARE_CACHE: Dict[str, int] = {}


def proteins_sharing_ec(ec_number: str,
                        session: Optional[requests.Session] = None) -> int:
    """How many reviewed human proteins carry this EC number.

    BRENDA indexes measurements by EC, not by protein, so for a broad class
    its records belong to the class and not to any member of it. EC 2.7.10.2
    is "non-specific protein-tyrosine kinase" and 44 reviewed human proteins
    carry it: JAK1, JAK2, SRC and BTK all return the same nine turnover
    numbers, every one of them measured on ATP rather than on the protein
    substrate each kinase actually acts on. Writing those as JAK1's kcat
    would attach four real papers to a number that is not JAK1's.

    One protein means the records are attributable. More than one means
    they are not.
    """
    if ec_number in _EC_SHARE_CACHE:
        return _EC_SHARE_CACHE[ec_number]
    s = session or requests.Session()
    s.headers.setdefault("User-Agent", "Cognisom/1.0 (cognisom.com; research)")
    try:
        response = s.get(
            "https://rest.uniprot.org/uniprotkb/search",
            params={"query": f"ec:{ec_number} AND reviewed:true AND "
                             f"organism_id:9606",
                    "fields": "accession", "size": "500", "format": "list"},
            timeout=30)
        response.raise_for_status()
        count = len([line for line in response.text.splitlines() if line.strip()])
    except Exception as exc:                            # pragma: no cover
        log.warning("could not count proteins for EC %s: %s", ec_number, exc)
        return 0
    _EC_SHARE_CACHE[ec_number] = count
    return count


def ec_is_specific(ec_number: str,
                   session: Optional[requests.Session] = None) -> bool:
    """True when exactly one reviewed human protein carries this EC."""
    return proteins_sharing_ec(ec_number, session) == 1


def km_values(ec_number: str, organism: str = "Homo sapiens",
              session: Optional[requests.Session] = None) -> List[KineticRecord]:
    """Michaelis constants BRENDA holds for an enzyme, in mM as BRENDA reports.

    The unit is BRENDA's, not the library's: ``km_um`` is micromolar, a
    factor of a thousand away, and :func:`accept` requires the caller to
    apply it rather than doing so silently.
    """
    arguments = f"ecNumber*{ec_number}#"
    if organism:
        arguments += f"organism*{organism}#"
    rows = _call("getKmValue", arguments, session)
    return _records(rows, ec_number, "km", "kmValue", "mM",
                    _references(ec_number, session))


def turnover_numbers(ec_number: str, organism: str = "Homo sapiens",
                     session: Optional[requests.Session] = None) -> List[KineticRecord]:
    """Turnover numbers BRENDA holds for an enzyme, per second."""
    arguments = f"ecNumber*{ec_number}#"
    if organism:
        arguments += f"organism*{organism}#"
    rows = _call("getTurnoverNumber", arguments, session)
    return _records(rows, ec_number, "kcat", "turnoverNumber", "1/s",
                    _references(ec_number, session))


def candidates(ec_number: str, parameter: str = "km",
               human_only: bool = True, wild_type_only: bool = True,
               specific_ec_only: bool = True,
               session: Optional[requests.Session] = None) -> List[KineticRecord]:
    """Records for one enzyme, filtered and ranked, for a caller to choose from.

    Mutants are excluded by default. A kcat measured on a point mutant is a
    real measurement of the wrong protein, and BRENDA reports both without
    distinguishing them in any field but the commentary.
    """
    if specific_ec_only and not ec_is_specific(ec_number, session):
        shared = proteins_sharing_ec(ec_number, session)
        raise BrendaError(
            f"EC {ec_number} is carried by {shared} reviewed human proteins, "
            f"so BRENDA's records for it belong to the class and not to any "
            f"one of them. Pass specific_ec_only=False to see them anyway."
        )
    fetch = km_values if parameter == "km" else turnover_numbers
    records = fetch(ec_number, "Homo sapiens" if human_only else "", session)
    out = [r for r in records
           if r.value is not None
           and (not human_only or r.is_human)
           and (not wild_type_only or r.is_wild_type)]
    out.sort(key=lambda r: (not r.citable, r.substrate))
    return out


def accept(entity: BioEntity, parameter: str, record: KineticRecord,
           value: Optional[float] = None, note: str = "") -> List[str]:
    """Write a chosen record, citing the paper BRENDA names.

    ``value`` overrides the record's number for a unit conversion: BRENDA
    reports Km in millimolar and the library's ``km_um`` is micromolar, a
    factor of a thousand that has to be applied deliberately rather than
    assumed.
    """
    if not record.citable:
        return [
            f"EC {record.ec_number} record is not citable: "
            + ("no value" if record.value is None
               else "BRENDA lists no PubMed id for it")
        ]
    spec = PARAMETER_SPECS.get(parameter)
    conditions = "; ".join(p for p in (
        f"substrate: {record.substrate}" if record.substrate else "",
        record.organism,
        f"EC {record.ec_number}",
        f"BRENDA commentary: {record.commentary[:160]}" if record.commentary else "",
        note,
    ) if p)
    return set_parameter(entity, parameter, ParameterValue(
        value=float(record.value if value is None else value),
        unit=spec.unit if spec else record.unit,
        provenance=Provenance.MEASURED,
        citation=f"PMID:{record.pmid}",
        source=f"BRENDA EC {record.ec_number} (CC BY 4.0)",
        conditions=conditions,
    ), strict=False)


def accept_median(entity: BioEntity, parameter: str,
                  records: List[KineticRecord], factor: float = 1.0,
                  note: str = "") -> List[str]:
    """Write the median of several records as a DERIVED value.

    A single BRENDA record is one assay in one paper, and the spread across
    papers is usually wider than the difference the library cares about:
    IDO1's Michaelis constant for L-tryptophan runs from 0.72 to 6809 uM
    across 41 measurements. Picking one of those and calling it MEASURED
    would attach a real citation to an arbitrary choice.

    So the median is written as DERIVED, with the count, the paper count and
    the range in the rationale, and the most-represented paper as the
    citation. That is a statement someone can check and disagree with, which
    a cherry-picked record is not.

    ``factor`` converts BRENDA's unit to the spec's -- 1000 for millimolar
    to micromolar -- and is applied to the median, never assumed.
    """
    usable = [r for r in records if r.citable and r.value is not None]
    if not usable:
        return ["no citable records to aggregate"]

    values = sorted(r.value * factor for r in usable)
    middle = len(values) // 2
    median = (values[middle] if len(values) % 2
              else (values[middle - 1] + values[middle]) / 2.0)

    papers: Dict[str, int] = {}
    for record in usable:
        papers[record.pmid] = papers.get(record.pmid, 0) + 1
    dominant = max(papers.items(), key=lambda kv: kv[1])[0]
    substrates = sorted({r.substrate for r in usable})
    spec = PARAMETER_SPECS.get(parameter)

    # With one record there is no aggregation to describe: it is a single
    # measurement from a single paper, which is what MEASURED means.
    single = len(usable) == 1
    return set_parameter(entity, parameter, ParameterValue(
        value=float(median),
        unit=spec.unit if spec else usable[0].unit,
        provenance=Provenance.MEASURED if single else Provenance.DERIVED,
        citation=f"PMID:{dominant}",
        source=f"BRENDA EC {usable[0].ec_number} (CC BY 4.0), "
               f"{len(usable)} records across {len(papers)} papers",
        conditions="; ".join(p for p in (
            f"substrate: {substrates[0]}" if len(substrates) == 1
            else f"substrates: {', '.join(s[:28] for s in substrates[:3])}",
            "Homo sapiens, wild-type",
            note,
        ) if p),
        rationale="" if single else (
            f"median of {len(usable)} BRENDA measurements across "
            f"{len(papers)} papers, spanning {values[0]:.4g} to "
            f"{values[-1]:.4g} {spec.unit if spec else usable[0].unit}. "
            f"The citation names the paper contributing most records "
            f"({papers[dominant]} of {len(usable)}); a single record would "
            f"have been an arbitrary pick from that spread"
        ),
    ), strict=False)
