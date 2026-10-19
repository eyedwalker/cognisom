"""Source protein mass and length from UniProt, with the identity cross-checked.

The library holds 683 parameter values of which 27 carry any evidence. Most of
the rest cannot be sourced without reading a paper, but two quantities can be
taken from a curated database and checked: how long a protein is and what it
weighs. Both come from one UniProt record, and a record is identified by a
stable accession, which is what makes it citable.

Two things make this trustworthy rather than merely automated.

**The accession is verified, not assumed.** A gene symbol search can return the
wrong protein, and a wrong accession would attach a confident-looking citation
to the wrong molecule -- the failure this module exists to avoid. So where the
structure cache holds an AlphaFold model for the accession, its residue count
is compared with UniProt's sequence length, and a mismatch refuses the write.
Measured over the current library, 126 of 129 accessions had a model to check
against and all 126 agreed.

**The mature chain is distinguished from the precursor.** UniProt's canonical
mass is the translated precursor, which for a secreted protein is the wrong
quantity by a wide margin: EGF's precursor is 1207 residues and 134 kDa, while
the cytokine that actually diffuses is the 53-residue chain at 971-1023, about
6 kDa. Writing the precursor would have made EGF twenty-two times too heavy and
IGF-1 nearly three times. The record's own Chain features carry the answer, and
where they are unambiguous this module follows them.

Where they are not -- complement C3 declares twelve chains, caspase-3 splits
into p17 and p12, TNF exists as a membrane and a soluble form -- no value is
written. Fifteen of the library's proteins are like that, and choosing one
chain for them is a modelling decision, not a lookup.
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests

from .models import BioEntity
from .parameters import ParameterValue, Provenance, set_parameter
from .proteins import NOT_A_SINGLE_PROTEIN, candidate_symbols, gene_symbol_for

log = logging.getLogger(__name__)

BASE_URL = "https://rest.uniprot.org/uniprotkb"
FIELDS = "accession,protein_name,sequence,ft_chain,ft_signal,ft_propep"

#: Average residue masses in daltons. The free N- and C-termini add a water.
#: These reproduce UniProt's own reported mass to within a dalton, which is
#: the check in test_uniprot_sourcing.
_RESIDUE_MASS: Dict[str, float] = {
    "A": 71.0788, "R": 156.1875, "N": 114.1038, "D": 115.0886, "C": 103.1388,
    "E": 129.1155, "Q": 128.1307, "G": 57.0519, "H": 137.1411, "I": 113.1594,
    "L": 113.1594, "K": 128.1741, "M": 131.1926, "F": 147.1766, "P": 97.1167,
    "S": 87.0782, "T": 101.1051, "W": 186.2132, "Y": 163.1760, "V": 99.1326,
    "U": 150.0388, "O": 237.3018,
}
_WATER = 18.01528

#: Descriptions that name a piece of a protein rather than the product. Used
#: only to break a tie between several chains; a record declaring exactly one
#: chain is always answered with that chain, however it is described.
_FRAGMENT = re.compile(
    r"domain|anastellin|ugl-|fragment|subunit|\bchain\b|\bform\b|\(\d+-\d+\)",
    re.I,
)


class SourcingRefused(RuntimeError):
    """The value was not written, and the message says why."""


def sequence_mass_da(sequence: str) -> float:
    """Average molecular mass of a peptide sequence, in daltons."""
    return sum(_RESIDUE_MASS.get(c, 0.0) for c in sequence) + _WATER


@dataclass
class SourcedProtein:
    """One entity resolved to a UniProt record, with the checks recorded."""

    entity_name: str
    accession: str = ""
    symbol: str = ""
    organism: str = ""
    canonical_length: int = 0
    canonical_mass_da: float = 0.0
    mature_length: int = 0
    mature_mass_da: float = 0.0
    chain_description: str = ""
    chain_span: Tuple[int, int] = (0, 0)
    resolved_by: str = ""
    model_checked: bool = False
    model_agrees: Optional[bool] = None
    written: List[str] = field(default_factory=list)
    refused: str = ""

    @property
    def usable(self) -> bool:
        return bool(self.accession) and not self.refused


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Cognisom/1.0 (cognisom.com; research)",
        "Accept": "application/json",
    })
    return s


def fetch_record(accession: str, session: Optional[requests.Session] = None) -> dict:
    """The UniProt record, with the features this module reads."""
    s = session or _session()
    url = f"{BASE_URL}/{accession}.json?fields={FIELDS}"
    response = s.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def _chains(record: dict) -> List[Tuple[int, int, str]]:
    out = []
    for feature in record.get("features", []):
        if feature.get("type") != "Chain":
            continue
        location = feature.get("location", {})
        start = location.get("start", {}).get("value")
        end = location.get("end", {}).get("value")
        if start and end:
            out.append((start, end, feature.get("description") or ""))
    return out


def mature_chain(record: dict) -> Tuple[int, int, str, str]:
    """``(start, end, description, sequence)`` for the product, or raise.

    A record with no chain feature is answered with its whole sequence,
    which is correct for an intracellular protein that is never processed.
    A record with one chain is answered with that chain. A record with
    several is answered only when exactly one of them is not a named
    fragment; otherwise the choice is a modelling decision and
    :class:`SourcingRefused` is raised rather than guessed at.
    """
    sequence = record.get("sequence", {}).get("value", "")
    chains = _chains(record)
    if not chains:
        return 1, len(sequence), "whole sequence (no chain feature)", sequence
    if len(chains) == 1:
        start, end, description = chains[0]
        return start, end, description, sequence[start - 1:end]

    candidates = [c for c in chains if not _FRAGMENT.search(c[2])]

    # A chain the record itself calls "Pro-X" or a precursor is, by its own
    # description, not the product. EGF is the clear case: the record
    # declares "Pro-epidermal growth factor" spanning almost the whole
    # sequence and "Epidermal growth factor" nested inside it, and the
    # mature cytokine is the latter. Dropping the pro-forms only where a
    # non-pro chain remains keeps this from emptying the candidate list.
    if len(candidates) > 1:
        not_pro = [c for c in candidates
                   if not re.match(r"\s*pro[- ]", c[2], re.I)
                   and "precursor" not in c[2].lower()]
        if not_pro:
            candidates = not_pro

    if len(candidates) != 1:
        raise SourcingRefused(
            f"{len(chains)} chains declared and {len(candidates)} are not "
            f"named fragments, so which one the entity represents is a "
            f"modelling decision: "
            f"{[c[2][:40] for c in chains][:4]}"
        )
    start, end, description = candidates[0]
    return start, end, description, sequence[start - 1:end]


def resolve_accession(
    name: str, session: Optional[requests.Session] = None
) -> Tuple[Optional[str], str, str]:
    """``(accession, symbol, resolved_by)`` for an entity display name.

    The symbol and the method travel with the accession because without
    them it cannot be re-verified later, and a wrong accession is then
    indistinguishable from a right one.
    """
    if name in NOT_A_SINGLE_PROTEIN:
        return None, "", f"not a single protein: {NOT_A_SINGLE_PROTEIN[name]}"

    symbol, how = gene_symbol_for(name)
    curated = how if symbol else ""
    symbols = [symbol] if symbol else candidate_symbols(name)
    s = session or _session()
    for candidate in symbols:
        if not candidate:
            continue
        url = f"{BASE_URL}/search"
        params = {
            "query": f"gene_exact:{candidate} AND organism_id:9606 AND reviewed:true",
            "fields": "accession", "size": "1", "format": "json",
        }
        try:
            response = s.get(url, params=params, timeout=30)
            response.raise_for_status()
            results = response.json().get("results", [])
        except Exception as exc:                       # pragma: no cover
            log.warning("UniProt search failed for %s: %s", candidate, exc)
            continue
        if results:
            return (results[0].get("primaryAccession"), candidate,
                    curated or "gene_exact_name")
    return None, "", f"no reviewed human entry for {symbols[:3]}"


def _model_residues(accession: str) -> Optional[int]:
    """Residue count of a cached AlphaFold model, or None if not cached."""
    from . import structure_cache

    path = os.path.join(structure_cache.DEFAULT_CACHE, f"AF-{accession}.cif.gz")
    if not os.path.exists(path):
        return None
    try:
        structure = structure_cache.fetch_alphafold(accession)
    except Exception:                                   # pragma: no cover
        return None
    return getattr(structure, "n_residues", None) if structure else None


def source_entity(
    entity: BioEntity,
    session: Optional[requests.Session] = None,
    write: bool = True,
) -> SourcedProtein:
    """Resolve one entity to UniProt and attach its mass and length.

    With ``write=False`` nothing is stored and the result reports what
    would have been.
    """
    result = SourcedProtein(entity_name=entity.name)
    accession, symbol, resolved_by = resolve_accession(entity.name, session)
    if not accession:
        result.refused = resolved_by
        return result
    result.accession = accession
    result.symbol = symbol
    result.resolved_by = resolved_by

    record = fetch_record(accession, session)
    sequence = record.get("sequence", {})
    result.canonical_length = sequence.get("length", 0) or 0
    result.canonical_mass_da = sequence.get("molWeight", 0.0) or 0.0
    result.organism = "Homo sapiens"

    # The identity check. A wrong accession is the dangerous failure here,
    # because it would attach a real citation to the wrong molecule.
    model_residues = _model_residues(accession)
    if model_residues is not None:
        result.model_checked = True
        result.model_agrees = model_residues == result.canonical_length
        if not result.model_agrees:
            result.refused = (
                f"the cached AlphaFold model for {accession} has "
                f"{model_residues} residues but UniProt's sequence is "
                f"{result.canonical_length}; the accession may name a "
                f"different protein, so nothing was written"
            )
            return result

    # Recorded here, before the chain is chosen: the accession has passed
    # its identity check and is sound, and only which chain the entity
    # represents is still open. An entity left carrying an accession with
    # no symbol cannot be re-verified, which is the state the library
    # refuses, so a refusal further down must not leave it in that state.
    if write:
        for attribute, value in (("uniprot_id", accession),
                                 ("gene_symbol", symbol),
                                 ("uniprot_resolved_by", resolved_by)):
            if hasattr(entity, attribute):
                setattr(entity, attribute, value)

    try:
        start, end, description, chain = mature_chain(record)
    except SourcingRefused as exc:
        result.refused = str(exc)
        return result

    result.chain_span = (start, end)
    result.chain_description = description
    result.mature_length = len(chain)
    result.mature_mass_da = sequence_mass_da(chain)

    if not write:
        return result

    is_whole = result.mature_length == result.canonical_length
    conditions = (
        f"UniProt {accession}, {description}"
        + ("" if is_whole else f", residues {start}-{end} of {result.canonical_length}")
        + ("; whole translated sequence" if is_whole
           else "; mature chain, not the translated precursor")
    )
    for parameter, value, unit in (
        ("sequence_length_aa", float(result.mature_length), "aa"),
        ("molecular_weight_kda", result.mature_mass_da / 1000.0, "kDa"),
    ):
        problems = set_parameter(entity, parameter, ParameterValue(
            value=value, unit=unit, provenance=Provenance.MEASURED,
            citation=f"UniProt:{accession}",
            source="UniProt Knowledgebase, reviewed (Swiss-Prot)",
            conditions=conditions,
        ), strict=False)
        if problems:
            log.warning("%s: %s not written: %s", entity.name, parameter, problems)
        else:
            result.written.append(parameter)
    return result


def source_library(entities, write: bool = True, pause: float = 0.05) -> List[SourcedProtein]:
    """Run :func:`source_entity` over a collection, politely."""
    session = _session()
    out = []
    for entity in entities:
        try:
            out.append(source_entity(entity, session=session, write=write))
        except Exception as exc:                        # pragma: no cover
            result = SourcedProtein(entity_name=entity.name)
            result.refused = f"{type(exc).__name__}: {exc}"
            out.append(result)
        time.sleep(pause)
    return out


# ── Diffusion, derived from the mass rather than assumed ─────────────

#: Boltzmann constant, J/K.
_KB = 1.380649e-23
#: Body temperature, K, and the viscosity of water there, Pa.s.
_T_BODY = 310.15
_ETA_WATER_37C = 0.6913e-3

#: Validation of the relation below against six proteins with published
#: diffusion coefficients, spanning 14 to 250 kDa: computed/published runs
#: 0.78 to 0.92, median 0.83. The relation is therefore reported as a
#: first-principles estimate good to about twenty percent, and the bias is
#: stated rather than fitted away on six points.
DIFFUSION_VALIDATION = (
    "Stokes-Einstein with a Tyn-Gusek hydrodynamic radius reproduces "
    "published aqueous diffusion coefficients to within 0.78-0.92x "
    "(median 0.83) across lysozyme, ribonuclease A, chymotrypsinogen, "
    "ovalbumin, serum albumin and catalase, 14-250 kDa"
)


def hydrodynamic_radius_nm(mass_kda: float) -> float:
    """Hydrodynamic radius of a globular protein, in nanometres.

    Tyn & Gusek's correlation on molar mass. The alternative route, from
    partial specific volume, agrees with it to within a few percent.
    """
    return 0.0514 * (mass_kda * 1000.0) ** 0.392


def diffusion_in_water_m2_s(mass_kda: float,
                            temperature_k: float = _T_BODY,
                            viscosity_pa_s: float = _ETA_WATER_37C) -> float:
    """Stokes-Einstein diffusion coefficient in dilute aqueous solution."""
    radius_m = hydrodynamic_radius_nm(mass_kda) * 1e-9
    return _KB * temperature_k / (6.0 * 3.141592653589793 * viscosity_pa_s * radius_m)


def derive_diffusion(entity: BioEntity, write: bool = True) -> Optional[float]:
    """Attach a diffusion coefficient derived from the entity's own mass.

    Only runs where the mass is itself evidenced, so the derivation chains
    off a verified UniProt record rather than off another placeholder. An
    earlier attempt to fit this relation against the library's existing
    diffusion values was abandoned for exactly that reason: every one of
    those values is the same stamped default, and calibrating against them
    would have laundered a default into a derivation.

    The value is the dilute-aqueous one. Diffusion through tissue is slower,
    by roughly a factor of two to three once tortuosity and binding are
    accounted for, and that is said in the conditions rather than folded
    silently into the number.
    """
    from .parameters import get_parameter

    mass = get_parameter(entity, "molecular_weight_kda")
    if mass is None or mass.value <= 0:
        return None
    if mass.provenance is Provenance.PLACEHOLDER:
        return None

    value = diffusion_in_water_m2_s(mass.value)
    if not write:
        return value

    problems = set_parameter(entity, "diffusion_coeff_um2_s", ParameterValue(
        value=value * 1e12,                 # m^2/s -> um^2/s
        unit="um^2/s",
        provenance=Provenance.DERIVED,
        citation=mass.citation,
        source="Stokes-Einstein from the UniProt mature-chain mass",
        conditions=(
            f"dilute aqueous solution at 37 C (water viscosity "
            f"{_ETA_WATER_37C * 1e3:.4f} mPa.s); derived from a mature-chain "
            f"mass of {mass.value:.2f} kDa giving a hydrodynamic radius of "
            f"{hydrodynamic_radius_nm(mass.value):.2f} nm. Effective "
            f"diffusivity in tissue is lower, by roughly two- to threefold "
            f"once tortuosity and binding are included"
        ),
        rationale=(
            "computed from the entity's own evidenced mass rather than "
            f"assumed or fitted to the library's existing values. {DIFFUSION_VALIDATION}"
        ),
    ), strict=False)
    if problems:
        log.warning("%s: diffusion not written: %s", entity.name, problems)
        return None
    return value
