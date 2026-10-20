"""Write the measured cell ultrastructure onto the library's entities.

The morphometry behind the rendered cells was living as Python constants
in ``cognisom.visualization``, with its PMIDs in comments. That is not
where evidence belongs in this repository: a number in a comment cannot
be queried, cannot be exported with the rest of the curation, and cannot
be told apart from a number somebody liked the look of. Everything here
goes through :func:`set_parameter`, so the spec's unit and range apply
and the provenance obligation is enforced -- a MEASURED value without a
citation is refused rather than stored.

WHAT IS HERE AND WHAT IS NOT
----------------------------
Only values that survived verification. Each was re-retrieved by a second
agent and then re-checked by a critic whose job was to find what was
wrong, and the ones that failed are not here:

  * no Golgi dimension -- the citable record is a 39.5 angstrom LIPID
    BILAYER thickness, 760x too small for the organelle;
  * no ER radius -- the only record that could set one is non-human and
    non-citable, and the citable figure is a surface AREA;
  * no proteasome size -- the record is a yeast 20S shell volume from a
    PhD thesis with no PubMed entry;
  * no normal human nuclear VOLUME -- every citable human record is HeLa,
    hypertriploid and measured trypsinized in suspension, so the
    normal-versus-malignant contrast cannot be sourced here at all. The
    nuclear FRACTION below is a different quantity and is sourced.

ONE TABLE, ONE METHOD, ONE PAPER
--------------------------------
The granule fractions all come from Schmid-Schonbein PMID 6775712 Table
2, which matters more than any individual number in it: a 77-fold spread
measured the same way on the same day is a comparison you can trust,
where four numbers from four papers would not be.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .models import BioEntity
from .parameters import ParameterValue, Provenance, set_parameter

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Measurement:
    """One measured value and everything needed to judge it."""

    parameter: str
    value: float
    unit: str
    citation: str
    source: str
    conditions: str = ""
    rationale: str = ""

    def as_value(self) -> ParameterValue:
        return ParameterValue(
            value=float(self.value), unit=self.unit,
            provenance=Provenance.MEASURED, citation=self.citation,
            source=self.source, conditions=self.conditions,
            rationale=self.rationale,
        )


#: Resting lymphocytes: nucleus, surface and granules.
LYMPHOCYTE: Tuple[Measurement, ...] = (
    Measurement(
        "nuclear_volume_fraction", 0.444, "dimensionless",
        "PMID:6775712",
        "Schmid-Schonbein GW et al., Table 2 (BNID 115194)",
        "Human peripheral blood lymphocyte; stereology on electron "
        "micrographs",
        "The record reads 'nucleus 44.4%: granules 0.2%: mitochondria "
        "3.8%: cytoplasm 51.6%'. Petrzilka (PMID 308396) agrees "
        "independently at 0.458, from 47.5 um^3 of nucleus in a 103.8 "
        "um^3 cell. A RADIUS is the cube root of this; using the "
        "fraction as a radius gives a nucleus at 8.8% of the cell.",
    ),
    Measurement(
        "granule_volume_fraction", 0.002, "dimensionless",
        "PMID:6775712", "Schmid-Schonbein GW et al., Table 2 (BNID 115194)",
        "Human peripheral blood lymphocyte",
        "The floor of the 77-fold spread in this table; an eosinophil "
        "measured the same way is 23.0%.",
    ),
    Measurement(
        "microvillus_density_per_um2", 3.0, "1/um^2",
        "PMID:27647916",
        "Jung Y et al., PNAS 2016, Fig. S2 legend",
        "Resting primary human peripheral blood T cells, live and "
        "unfixed; variable-angle TIRF with 3D surface reconstruction",
        "3 +/- 0.5 per um^2. Independently the midpoint of the 2-4 range "
        "from SEM morphometry across lymphocyte sources two decades "
        "earlier (PMID 15130947) -- two labs, two unrelated methods, the "
        "same number. Measured on LIVE cells, so it escapes the "
        "critical-point-drying artefact; the same paper's own SEM gives "
        "4.5, which is the size of that artefact.",
    ),
    Measurement(
        "microvillus_length_um", 0.35, "um",
        "PMID:15130947", "Majstoravich S et al., Blood 2004",
        "Circulating mammalian lymphocytes; SEM morphometry",
        "A MEDIAN of a strongly right-skewed distribution, not a mean. "
        "Lengths run 'from 0.1 um to several micrometers' (PMID "
        "27647916, verbatim), so a constant-length coat has the wrong "
        "shape and not merely the wrong number.",
    ),
    Measurement(
        "microvillus_diameter_um", 0.085, "um",
        "PMID:27647916", "Jung Y et al., PNAS 2016",
        "Resting primary human T cells; SEM. Effector T cells measured "
        "'comparable' in the same paper",
        "Midpoint of a measured 70-100 nm range; the measurement is the "
        "range. The 150 nm figure in circulation traces to a review and "
        "could not be confirmed in any primary source.",
    ),
)

#: Cytotoxic lymphocytes carry the lymphocyte set plus their granules.
CYTOTOXIC: Tuple[Measurement, ...] = (
    Measurement(
        "lytic_granule_count", 63.0, "count",
        "PMID:29592963",
        "Gwalani LA, Orange JS, J Immunol 2018",
        "Primary human NK cells",
        "63 +/- 23. Explicitly NOT the 194-206 the same paper reports "
        "for the NK-92 and YTS lines, which is what reviews quote as "
        "'almost 200 granules': immortalised lines carry about three "
        "times the load.",
    ),
    Measurement(
        "lytic_granule_diameter_um", 0.251, "um",
        "PMID:21931537", "Brown ACN et al., 3D-SIM",
        "Primary human NK cells",
        "251 +/- 2 nm, where the +/- is a standard error and carries no "
        "spread information. Bracketed by two other methods: 333 +/- 103 "
        "nm by STED on NK-92 (PMID 21931536), and a 3.3 fF capacitance "
        "step on primary human CD8 T cells (PMID 26296096) which at 10 "
        "fF/um^2 works out at 324 nm.",
    ),
)

#: Granulocytes and phagocytes, all from the one table.
_TABLE = "Schmid-Schonbein GW et al., Table 2"
GRANULE_FRACTIONS: Dict[str, Tuple[float, str]] = {
    "eosinophil": (0.230, "BNID 115197"),
    "neutrophil": (0.154, "BNID 115195"),
    "monocyte": (0.026, "BNID 115196"),
    "lymphocyte": (0.002, "BNID 115194"),
}


def _granule(fraction: float, bnid: str, cell: str) -> Measurement:
    return Measurement(
        "granule_volume_fraction", fraction, "dimensionless",
        "PMID:6775712", f"{_TABLE} ({bnid})",
        f"Human peripheral blood {cell}; stereology on electron micrographs",
        "One table, one method, one paper, which is what makes the "
        "77-fold spread across these cells a comparison rather than four "
        "unrelated numbers.",
    )


#: Which entities get which set, matched on the library's own names.
#: Matching on name is deliberate: the library has no field saying "this
#: is a lymphocyte", and inferring it from mesh_type would make the
#: evidence depend on a rendering hint.
LYMPHOCYTE_NAMES = (
    "CD4+ T cell", "CD8+ T cell", "CD4+ Th1 cell", "CD4+ Th2 cell",
    "CD4+ Th17 cell", "CD8+ Naive T cell", "CD8+ Memory T cell",
    "CD8+ Effector T cell", "Regulatory T cell", "Regulatory T cell (Treg)",
    "T follicular helper (Tfh)", "Gamma-delta T cell", "Naive B cell",
    "Memory B cell", "ILC1", "ILC2", "ILC3", "NK cell", "NKT cell",
)
CYTOTOXIC_NAMES = ("NK cell", "NKT cell", "CD8+ Effector T cell")
EOSINOPHIL_NAMES = ("Eosinophil",)
NEUTROPHIL_NAMES = ("Neutrophil",)
MONOCYTE_NAMES = ("Macrophage", "Macrophage M1", "Macrophage M2", "Monocyte")


def measurements_for(name: str) -> List[Measurement]:
    """Every measurement that applies to the entity called ``name``."""
    out: List[Measurement] = []
    if name in LYMPHOCYTE_NAMES:
        out.extend(LYMPHOCYTE)
    if name in CYTOTOXIC_NAMES:
        out.extend(CYTOTOXIC)
    if name in EOSINOPHIL_NAMES:
        out.append(_granule(*GRANULE_FRACTIONS["eosinophil"], "eosinophil"))
    if name in NEUTROPHIL_NAMES:
        out.append(_granule(*GRANULE_FRACTIONS["neutrophil"], "neutrophil"))
    if name in MONOCYTE_NAMES:
        out.append(_granule(*GRANULE_FRACTIONS["monocyte"], "monocyte"))
    # A later entry wins, so a cytotoxic lymphocyte keeps the lymphocyte
    # granule fraction unless its own line overrides it.
    seen: Dict[str, Measurement] = {}
    for m in out:
        seen[m.parameter] = m
    return list(seen.values())


def seed(store, *, dry_run: bool = False) -> Tuple[int, int, List[str]]:
    """Write the measurements onto every entity they apply to.

    Returns ``(entities touched, values written, problems)``. A value that
    fails its spec is reported rather than stored -- the point of routing
    through ``set_parameter`` is that a wrong unit or an out-of-range
    number cannot reach the library whatever this module believes.
    """
    touched = written = 0
    problems: List[str] = []
    for entity in _entities(store):
        applicable = measurements_for(entity.name)
        if not applicable:
            continue
        changed = False
        for m in applicable:
            issues = set_parameter(entity, m.parameter, m.as_value(),
                                   strict=False)
            if issues:
                problems.extend(f"{entity.name}/{m.parameter}: {i}"
                                for i in issues)
                continue
            written += 1
            changed = True
        if changed:
            touched += 1
            if not dry_run:
                store.update_entity(entity, changed_by="seed_ultrastructure")
    return touched, written, problems


def _entities(store) -> Iterable[BioEntity]:
    rows = [r[0] for r in store._conn.execute("SELECT name FROM entities")]
    for name in rows:
        entity = store.find_entity_by_name(name)
        if entity is not None:
            yield entity
