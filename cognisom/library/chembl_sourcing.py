"""Measured binding constants from ChEMBL, for the Kd values the library guesses.

ChEMBL holds bioactivity extracted from the medicinal-chemistry literature,
indexed by target and by compound, and every measurement carries the document
it came from with a PubMed identifier already attached. That makes the citation
chain shorter than BioNumbers': there is no reference string to resolve and no
title to match back, because the record names the paper outright.

What it does not do is answer which number belongs in the library. A target
carries many constants -- the androgen receptor has 38 Kd values, 1,339 Ki and
4,664 IC50 -- and they are measurements of *different ligands*, most of them
synthetic compounds from a screening series. The library's ``kd_nm`` on a
receptor means its physiological ligand, so taking the first or the median
would attach a real citation to a number about a different molecule.

So :func:`candidates` returns activities with the compound, the assay
description and the paper, and :func:`accept` writes one a caller has chosen.
Measurements qualified by ``>`` or ``<`` are excluded by default: a bound is
not a value, and recording one as if it were is the quiet error this module
is otherwise well placed to make.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests

from .models import BioEntity
from .parameters import PARAMETER_SPECS, ParameterValue, Provenance, set_parameter

log = logging.getLogger(__name__)

BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
POLITE_PAUSE_S = 0.2

#: Types the library can use directly, and what each one means.
BINDING_TYPES = {
    "Kd": "equilibrium dissociation constant",
    "Ki": "inhibition constant",
}

_DOC_CACHE: Dict[str, dict] = {}


@dataclass
class Activity:
    """One measured constant, with the compound and the paper behind it."""

    target_chembl_id: str
    molecule_chembl_id: str
    standard_type: str
    value: Optional[float]
    units: str
    relation: str = "="
    assay_description: str = ""
    document_chembl_id: str = ""
    organism: str = ""
    pmid: str = ""
    doi: str = ""
    title: str = ""
    journal: str = ""
    year: Optional[int] = None
    molecule_name: str = ""

    @property
    def is_exact(self) -> bool:
        """A ``>`` or ``<`` measurement is a bound, not a value."""
        return self.relation in ("=", "~")

    @property
    def citable(self) -> bool:
        return self.value is not None and self.is_exact and bool(self.pmid)

    def describe(self) -> str:
        bits = [f"{self.standard_type} {self.relation}{self.value:g} {self.units}"
                if self.value is not None else f"{self.standard_type} (no value)"]
        bits.append(self.molecule_name or self.molecule_chembl_id)
        if self.pmid:
            bits.append(f"PMID:{self.pmid}")
        if self.year:
            bits.append(str(self.year))
        if self.assay_description:
            bits.append(self.assay_description[:70])
        return " | ".join(bits)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "Cognisom/1.0 (cognisom.com; research)",
                      "Accept": "application/json"})
    return s


def _get(path: str, params: dict, session: Optional[requests.Session] = None) -> dict:
    s = session or _session()
    response = s.get(f"{BASE_URL}/{path}", params=params, timeout=30)
    response.raise_for_status()
    time.sleep(POLITE_PAUSE_S)
    return response.json()


def target_for_accession(accession: str,
                         session: Optional[requests.Session] = None) -> Optional[str]:
    """The SINGLE PROTEIN target for a UniProt accession.

    Complexes and protein-protein interaction targets are skipped: a
    constant measured against "androgen receptor / BIRC2" is not the
    receptor's own ligand affinity.
    """
    data = _get("target.json", {"target_components__accession": accession,
                                "limit": 20}, session)
    for target in data.get("targets", []):
        if target.get("target_type") == "SINGLE PROTEIN":
            return target.get("target_chembl_id")
    return None


def document(doc_id: str, session: Optional[requests.Session] = None) -> dict:
    """The paper a measurement came from. Cached; ChEMBL is a shared service."""
    if doc_id in _DOC_CACHE:
        return _DOC_CACHE[doc_id]
    try:
        data = _get(f"document/{doc_id}.json", {}, session)
    except Exception as exc:                            # pragma: no cover
        log.warning("ChEMBL document %s: %s", doc_id, exc)
        data = {}
    _DOC_CACHE[doc_id] = data
    return data


def _molecule_name(mol_id: str, session: Optional[requests.Session] = None) -> str:
    try:
        data = _get(f"molecule/{mol_id}.json", {}, session)
    except Exception:                                   # pragma: no cover
        return ""
    name = data.get("pref_name") or ""
    if not name:
        synonyms = data.get("molecule_synonyms") or []
        if synonyms:
            name = synonyms[0].get("molecule_synonym", "")
    return name


def candidates(accession: str, standard_type: str = "Kd", limit: int = 10,
               exact_only: bool = True, with_names: bool = False,
               session: Optional[requests.Session] = None) -> List[Activity]:
    """Measured constants for a protein, newest first, with their papers.

    Returned for a caller to choose between. Which compound's affinity
    belongs in the library is not something ChEMBL can answer: the same
    target carries thousands of constants for synthetic analogues.
    """
    s = session or _session()
    target = target_for_accession(accession, s)
    if not target:
        return []
    data = _get("activity.json", {
        "target_chembl_id": target, "standard_type": standard_type,
        "standard_units": "nM", "limit": max(limit * 3, 30),
    }, s)

    out: List[Activity] = []
    for raw in data.get("activities", []):
        try:
            value = float(raw["standard_value"]) if raw.get("standard_value") else None
        except (TypeError, ValueError):
            value = None
        activity = Activity(
            target_chembl_id=target,
            molecule_chembl_id=raw.get("molecule_chembl_id", ""),
            standard_type=raw.get("standard_type", standard_type),
            value=value,
            units=raw.get("standard_units", ""),
            relation=raw.get("standard_relation") or "=",
            assay_description=raw.get("assay_description") or "",
            document_chembl_id=raw.get("document_chembl_id") or "",
            organism=raw.get("target_organism") or "",
        )
        if exact_only and not activity.is_exact:
            continue
        if activity.value is None:
            continue
        if activity.document_chembl_id:
            doc = document(activity.document_chembl_id, s)
            activity.pmid = str(doc.get("pubmed_id") or "")
            activity.doi = doc.get("doi") or ""
            activity.title = doc.get("title") or ""
            activity.journal = doc.get("journal") or ""
            activity.year = doc.get("year")
        if with_names and activity.molecule_chembl_id:
            activity.molecule_name = _molecule_name(activity.molecule_chembl_id, s)
        out.append(activity)
        if len(out) >= limit:
            break

    out.sort(key=lambda a: (not a.citable, -(a.year or 0)))
    return out


def accept(entity: BioEntity, parameter: str, activity: Activity,
           note: str = "") -> List[str]:
    """Write a chosen measurement, citing the paper ChEMBL names.

    The compound travels with the value in the conditions, because a
    dissociation constant without its ligand describes nothing.
    """
    if not activity.citable:
        reason = ("no value" if activity.value is None
                  else "a bound, not a measurement" if not activity.is_exact
                  else "no PubMed id on its document")
        return [f"{activity.molecule_chembl_id} is not citable: {reason}"]

    spec = PARAMETER_SPECS.get(parameter)
    ligand = activity.molecule_name or activity.molecule_chembl_id
    conditions = "; ".join(p for p in (
        f"ligand: {ligand}",
        f"{BINDING_TYPES.get(activity.standard_type, activity.standard_type)}"
        f" measured in nM",
        activity.organism,
        f"assay: {activity.assay_description[:160]}" if activity.assay_description else "",
        note,
    ) if p)
    return set_parameter(entity, parameter, ParameterValue(
        value=float(activity.value),
        unit=spec.unit if spec else "nM",
        provenance=Provenance.MEASURED,
        citation=f"PMID:{activity.pmid}",
        source=(f"ChEMBL {activity.target_chembl_id} / "
                f"{activity.molecule_chembl_id}"
                + (f"; {activity.title[:90]}" if activity.title else "")),
        conditions=conditions,
    ), strict=False)
