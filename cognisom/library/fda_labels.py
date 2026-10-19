"""Pharmacokinetics from FDA prescribing information, via openFDA.

A drug's elimination half-life, bioavailability and volume of distribution
are not in any kinetics database, but they are in the one document every
approved drug must have: its prescribing information, whose Clinical
Pharmacology section states them in regulatory prose. openFDA serves those
labels as JSON, each with a stable SPL set identifier, which is about as
citable as a source gets.

The prose is the hazard. A label states several half-lives when a drug has an
active metabolite, several bioavailabilities for fed and fasted dosing, and a
volume of distribution that may be per kilogram. So every extracted number is
returned with the sentence it came from, the sentence travels into the stored
conditions, and a label that yields more than one candidate for a parameter
is surfaced rather than resolved by taking the first. The same line as the
other sourcing modules: fetch what is unambiguous, show what is not.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import requests

from .models import BioEntity
from .parameters import PARAMETER_SPECS, ParameterValue, Provenance, set_parameter

log = logging.getLogger(__name__)

LABEL_URL = "https://api.fda.gov/drug/label.json"
POLITE_PAUSE_S = 0.3

#: Library display names that differ from the generic name a label is
#: indexed under.
GENERIC_NAMES: Dict[str, str] = {
    "Aldesleukin (IL-2)": "aldesleukin",
    "Interferon alfa-2b": "interferon alfa-2b",
    "Lutetium-177 PSMA": "lutetium lu 177 vipivotide tetraxetan",
}

#: Brand names to try before any full-text search. A full-text search for
#: "aldesleukin" returns a homeopathic preparation called GUNA-IL 2 with no
#: pharmacology section at all, while PROLEUKIN's label is indexed only by
#: brand.
BRAND_HINTS: Dict[str, str] = {
    "Aldesleukin (IL-2)": "proleukin",
    "Interferon alfa-2b": "intron a",
    "Infliximab": "remicade",
    "Lutetium-177 PSMA": "pluvicto",
}

#: Sections of a label that carry pharmacokinetics, in the order to read.
PK_SECTIONS = ("pharmacokinetics", "clinical_pharmacology",
               "clinical_pharmacology_table", "mechanism_of_action")


@dataclass
class PKCandidate:
    """One number read out of a label, with the sentence that states it."""

    parameter: str
    value: float                 # in the library's unit
    unit: str
    sentence: str
    raw_value: float
    raw_unit: str
    set_id: str
    brand: str = ""
    generic: str = ""
    effective_time: str = ""

    @property
    def citation(self) -> str:
        return f"FDA-SPL:{self.set_id}"

    def describe(self) -> str:
        return (f"{self.parameter}={self.value:g} {self.unit}  "
                f"({self.raw_value:g} {self.raw_unit})  "
                f"[{self.brand or self.generic}] \"{self.sentence[:80]}\"")


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "Cognisom/1.0 (cognisom.com; research)"})
    return s


def label_for(generic_name: str,
              session: Optional[requests.Session] = None,
              brand_first: bool = False) -> Optional[dict]:
    """The most recent label openFDA holds for a generic (or brand) name."""
    s = session or _session()
    # Some approved labels carry no openfda harmonisation block, so a search
    # on openfda.generic_name cannot find them even though the document is
    # there. Fall back to the brand field and then to the label's own text.
    queries = ((f'openfda.brand_name:"{generic_name}"',) if brand_first else ()) + (
               f'openfda.generic_name:"{generic_name}"',
               f'openfda.brand_name:"{generic_name}"',
               f'openfda.substance_name:"{generic_name}"',
               f'description:"{generic_name}"',
               f'"{generic_name}"')
    results: list = []
    for query in queries:
        try:
            response = s.get(LABEL_URL, params={"search": query, "limit": 5}, timeout=30)
            time.sleep(POLITE_PAUSE_S)
            if response.status_code == 404:
                continue
            response.raise_for_status()
            results = response.json().get("results", [])
        except Exception as exc:
            log.warning("openFDA %s (%s): %s", generic_name, query[:28], exc)
            continue
        # A full-text hit must actually be this drug's label, not one that
        # mentions it in a drug-interaction paragraph.
        if results and "openfda" not in query:
            name = generic_name.lower()
            results = [r for r in results
                       if name in " ".join(r.get("description", []) or []).lower()[:600]
                       or name in " ".join(r.get("indications_and_usage", []) or []).lower()[:600]]
        if results:
            break
    if not results:
        return None
    # Prefer the label with the longest PK section: it is usually the
    # full prescribing information rather than a patient leaflet.
    results.sort(key=lambda r: -len(" ".join(_pk_text(r))))
    return results[0]


def _pk_text(label: dict) -> List[str]:
    out = []
    for section in PK_SECTIONS:
        for chunk in label.get(section, []) or []:
            out.append(chunk)
    return out


def _sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text)
    return [s.strip() for s in re.split(r"(?<=[.;])\s+(?=[A-Z(])", text) if s.strip()]


#: A number as a label prints it: "4,864", "19,669", "0.5". The thousands
#: separator matters -- without it cabazitaxel's 4,864 L volume of
#: distribution reads as 864 L and abiraterone's 19,669 L as 669 L.
_NUM = r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
#: "12 ± 5 hours" is a mean and a standard deviation. The number nearest
#: the unit is the SD, so the pattern takes the one before the ±.
_PM = rf"{_NUM}(?:\s*(?:±|\+/-|\+/−)\s*\d{{1,3}}(?:,\d{{3}})*(?:\.\d+)?)?"
#: "7 to 9.5 days" is a range; both ends are kept and the midpoint stored,
#: with the range in the sentence that travels alongside. "2-hour" is the
#: same unit with a hyphen.
#: Labels put a CV or SD in parentheses between the number and its unit:
#: "26 (39%) hours", "90 (±58) hours". The parenthetical is skipped.
_PAREN = r"(?:\s*\([^)]{0,24}\))?"
_UNIT = r"(hours?|hrs?|h\b|days?|minutes?|min\b|weeks?)"
_RANGE = rf"{_PM}{_PAREN}(?:\s*(?:to|-|–)\s*{_NUM}{_PAREN})?"
#: "between 141 hours and 882 hours" repeats the unit on both ends.
_BETWEEN = re.compile(
    rf"(?:half[- ]?life|t\s*1/2|t½)[^.;,]{{0,60}}?between\s*{_NUM}\s*{_UNIT}\s*and\s*{_NUM}\s*{_UNIT}", re.I)
_HALF = re.compile(
    r"(?:half[- ]?life|t\s*1/2|t½)[^.;]{0,60}?"
    rf"(?:approximately|about|of|is|was|=|~)?\s*{_RANGE}[\s-]*"
    + _UNIT, re.I)
_HALF_PRE = re.compile(
    rf"{_PM}[\s-]*(hours?|hrs?|days?|minutes?|weeks?)[^.;,]{{0,40}}?half[- ]?life", re.I)
_BIO = re.compile(
    r"bioavailab\w*[^.;]{0,80}?(?:approximately|about|of|is|was|=|~)?\s*"
    rf"{_NUM}\s*%", re.I)
_VD = re.compile(
    r"volume of distribution[^.;]{0,80}?(?:approximately|about|of|is|was|=|~)?\s*"
    rf"{_PM}\s*(L|liters?|litres?)(?!\s*/\s*kg)\b", re.I)


def _to_float(text: str) -> float:
    return float(text.replace(",", ""))


_HOURS = {"hour": 1.0, "hours": 1.0, "hr": 1.0, "hrs": 1.0, "h": 1.0,
          "day": 24.0, "days": 24.0, "minute": 1 / 60, "minutes": 1 / 60,
          "min": 1 / 60, "week": 168.0, "weeks": 168.0}


def extract(label: dict) -> List[PKCandidate]:
    """Every PK number the label states, each with its sentence."""
    set_id = label.get("set_id", "")
    brand = (label.get("openfda", {}).get("brand_name") or [""])[0]
    generic = (label.get("openfda", {}).get("generic_name") or [""])[0]
    when = label.get("effective_time", "")
    out: List[PKCandidate] = []
    seen = set()

    def add(parameter, value, unit, sentence, raw, raw_unit):
        key = (parameter, round(value, 6), sentence[:60])
        if key in seen:
            return
        seen.add(key)
        out.append(PKCandidate(parameter=parameter, value=value, unit=unit,
                               sentence=sentence[:400], raw_value=raw,
                               raw_unit=raw_unit, set_id=set_id, brand=brand,
                               generic=generic, effective_time=when))

    names = [n.lower() for n in (brand, generic) if n]
    names += [w for n in names for w in n.split() if len(w) > 5]
    # Biosimilars carry a suffix ("infliximab-abda") their own sentences
    # drop ("infliximab"), so the stem before the hyphen counts as the name.
    names += [n.split("-")[0] for n in names if "-" in n and len(n.split("-")[0]) > 5]
    for sentence in _sentences(" ".join(_pk_text(label))):
        low = sentence.lower()
        # A sentence naming a different molecule's half-life is not about
        # the drug. Pembrolizumab's label says hyaluronan has a half-life of
        # about half a day; it also says pembrolizumab's is 22 days.
        if names and not any(n in low for n in names) and re.search(
                r"\b(hyaluron|metabolite|albumin|igg\b|endogenous)", low):
            continue
        for match in _BETWEEN.finditer(sentence):
            lo, u1, hi, u2 = match.group(1), match.group(2).lower(), match.group(3), match.group(4).lower()
            f1, f2 = _HOURS.get(u1.rstrip(".")), _HOURS.get(u2.rstrip("."))
            if f1 and f2:
                raw = (_to_float(lo) * f1 + _to_float(hi) * f2) / 2.0
                add("half_life_hours", raw, "h", sentence, raw,
                    f"h (midpoint of {lo} {u1}-{hi} {u2})")
        if _BETWEEN.search(sentence):
            continue
        for match in _HALF.finditer(sentence):
            low_end, high_end, unit = match.group(1), match.group(2), match.group(3).lower().rstrip(".")
            factor = _HOURS.get(unit)
            if not factor:
                continue
            raw = _to_float(low_end)
            if high_end:
                raw = (raw + _to_float(high_end)) / 2.0
                unit = f"{unit} (midpoint of {low_end}-{high_end})"
            add("half_life_hours", raw * factor, "h", sentence, raw, unit)
        for match in _HALF_PRE.finditer(sentence):
            raw, unit = _to_float(match.group(1)), match.group(2).lower().rstrip(".")
            factor = _HOURS.get(unit)
            if factor:
                add("half_life_hours", raw * factor, "h", sentence, raw, unit.split(" (")[0])
        for match in _BIO.finditer(sentence):
            raw = _to_float(match.group(1))
            if 0 < raw <= 100:
                add("bioavailability", raw / 100.0, "dimensionless", sentence, raw, "%")
        for match in _VD.finditer(sentence):
            raw = _to_float(match.group(1))
            add("volume_of_distribution_L", raw, "L", sentence, raw, "L")
    return out


def candidates(entity_name: str,
               session: Optional[requests.Session] = None) -> Dict[str, List[PKCandidate]]:
    """Candidates per parameter for a library drug, from its FDA label.

    More than one candidate for a parameter means the label states more
    than one number -- a parent and a metabolite, fed and fasted -- and a
    caller has to say which the library means.
    """
    generic = GENERIC_NAMES.get(entity_name, entity_name).lower()
    label = None
    brand = BRAND_HINTS.get(entity_name)
    if brand:
        label = label_for(brand, session, brand_first=True)
    if label is None:
        label = label_for(generic, session)
    if label is None:
        return {}
    by_parameter: Dict[str, List[PKCandidate]] = {}
    for candidate in extract(label):
        by_parameter.setdefault(candidate.parameter, []).append(candidate)
    return by_parameter


def accept(entity: BioEntity, candidate: PKCandidate, note: str = "") -> List[str]:
    """Write one candidate, citing the label it was read from."""
    spec = PARAMETER_SPECS.get(candidate.parameter)
    conditions = "; ".join(p for p in (
        f"{candidate.brand or candidate.generic} prescribing information, "
        f"effective {candidate.effective_time}",
        f"label states: \"{candidate.sentence[:220]}\"",
        note,
    ) if p)
    return set_parameter(entity, candidate.parameter, ParameterValue(
        value=float(candidate.value),
        unit=spec.unit if spec else candidate.unit,
        provenance=Provenance.MEASURED,
        citation=candidate.citation,
        source=f"FDA prescribing information via openFDA, SPL set {candidate.set_id}",
        conditions=conditions,
    ), strict=False)


# ── choosing, when a label states more than one number ──────────────────

#: Sentences describing a population or condition the library does not
#: model. A half-life measured under plasma exchange or in hepatic
#: impairment is a real number about a situation the simulation never
#: enters.
_SPECIAL_CASE = re.compile(
    r"impairment|plasma exchange|extended.release|\bXR\b|inhibitor|inducer|"
    r"metabolite|distribution half|with food|\bfed\b|pediatric|paediatric|"
    r"renal|hepatic|dialysis|elderly|japanese|chinese", re.I)
_PREFERRED = re.compile(r"terminal|elimination|steady.state", re.I)


def choose(parameter: str, options: List[PKCandidate]):
    """``(candidate, "")`` or ``(None, why)``.

    The rules are few and stated here so they can be argued with. A
    bioavailability below 100% on a monoclonal antibody is the subcutaneous
    product's: an intravenous dose is 100% by definition, and such a label
    only carries a lower figure when a subcutaneous formulation exists. The
    library doses these antibodies intravenously, so that number is
    surfaced rather than written -- whether or not the sentence happens to
    say "subcutaneous", which atezolizumab's does not.
    """
    if not options:
        return None, "the label states no value"
    if parameter == "bioavailability":
        generic = (options[0].generic or "").lower().split()
        if generic and re.search(r"mab$", generic[0]) and all(o.value < 1.0 for o in options):
            return None, ("absolute bioavailability below 100% on an antibody is the "
                          "subcutaneous product's; the library doses it intravenously")
    live = [o for o in options if not _SPECIAL_CASE.search(o.sentence)] or options
    distinct = {round(o.value, 3) for o in live}
    if len(distinct) == 1:
        return live[0], ""
    preferred = [o for o in live if _PREFERRED.search(o.sentence)]
    if len({round(o.value, 3) for o in preferred}) == 1:
        return preferred[0], ""
    return None, f"{len(distinct)} distinct values: {sorted(distinct)}"


def forget(entity: BioEntity, parameter: str) -> bool:
    """Remove a parameter and its provenance from an entity.

    For undoing a write that should not have happened. The library has no
    other way to retract a value, and leaving a wrong one in place because
    the API only adds is how a mistake becomes permanent.
    """
    from .parameters import PROVENANCE_KEY
    params = entity.physics_params or {}
    removed = params.pop(parameter, None) is not None
    sidecar = params.get(PROVENANCE_KEY) or {}
    if parameter in sidecar:
        del sidecar[parameter]
        removed = True
    return removed
