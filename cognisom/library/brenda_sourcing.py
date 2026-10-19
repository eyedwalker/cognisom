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

ENDPOINT = "https://www.brenda-enzymes.org/soap/brenda_server_zeep.php"
NAMESPACE = "ws.brenda"
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


def _envelope(method: str, parts: List[str]) -> str:
    body = "".join(
        f"<{name}>{_escape(value)}</{name}>" for name, value in
        zip(("email", "password", "ecNumber", "organism"), parts)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<soapenv:Envelope xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/"'
        f' xmlns:ws="{NAMESPACE}">'
        f"<soapenv:Body><ws:{method}>{body}</ws:{method}></soapenv:Body>"
        "</soapenv:Envelope>"
    )


def _escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def _call(method: str, ec_number: str, organism: str,
          session: Optional[requests.Session] = None) -> List[str]:
    email, password_hash = credentials()
    s = session or requests.Session()
    s.headers.update({"User-Agent": "Cognisom/1.0 (cognisom.com; research)",
                      "Content-Type": "text/xml; charset=utf-8",
                      "SOAPAction": f"{NAMESPACE}#{method}"})
    payload = _envelope(method, [email, password_hash, ec_number, organism])
    response = s.post(ENDPOINT, data=payload.encode("utf-8"), timeout=60)
    time.sleep(POLITE_PAUSE_S)
    # The fault is parsed before the status is judged. BRENDA returns its
    # most useful messages with HTTP 500 -- a bad password comes back as a
    # SOAP fault saying so, inside a 500 -- and reporting only the status
    # throws away the one line that says what to do about it.
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
    return [el.text or "" for el in tree.iter() if el.text and "*" in el.text]


#: BRENDA packs a record as "field*value#field*value#...".
def _parse(row: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for field in row.split("#"):
        if "*" in field:
            key, _, value = field.partition("*")
            out[key.strip()] = value.strip()
    return out


def _records(rows: List[str], ec_number: str, parameter: str,
             value_key: str, unit: str) -> List[KineticRecord]:
    out = []
    for row in rows:
        fields = _parse(row)
        raw = fields.get(value_key, "")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = None
        literature = fields.get("literature", "")
        pmid = literature.split(",")[0].strip() if literature else ""
        out.append(KineticRecord(
            ec_number=ec_number, parameter=parameter, value=value, unit=unit,
            substrate=fields.get("substrate", ""),
            organism=fields.get("organism", ""),
            commentary=fields.get("commentary", ""),
            literature=literature,
            pmid=pmid if pmid.isdigit() else "",
        ))
    return out


def km_values(ec_number: str, organism: str = "Homo sapiens",
              session: Optional[requests.Session] = None) -> List[KineticRecord]:
    """Michaelis constants BRENDA holds for an enzyme, in mM as BRENDA reports."""
    rows = _call("getKmValue", ec_number, organism, session)
    return _records(rows, ec_number, "km", "kmValue", "mM")


def turnover_numbers(ec_number: str, organism: str = "Homo sapiens",
                     session: Optional[requests.Session] = None) -> List[KineticRecord]:
    """Turnover numbers BRENDA holds for an enzyme, per second."""
    rows = _call("getTurnoverNumber", ec_number, organism, session)
    return _records(rows, ec_number, "kcat", "turnoverNumber", "1/s")


def candidates(ec_number: str, parameter: str = "km",
               human_only: bool = True, wild_type_only: bool = True,
               session: Optional[requests.Session] = None) -> List[KineticRecord]:
    """Records for one enzyme, filtered and ranked, for a caller to choose from.

    Mutants are excluded by default. A kcat measured on a point mutant is a
    real measurement of the wrong protein, and BRENDA reports both without
    distinguishing them in any field but the commentary.
    """
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
