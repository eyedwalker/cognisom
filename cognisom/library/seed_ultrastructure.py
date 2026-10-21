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
  * no excess membrane area, though PMID 6775712 measures it for all
    four cell types -- cell and nucleus: 84%/98% neutrophil, 130%/32%
    lymphocyte, 137%/73% monocyte, 92%/84% eosinophil. It is not a
    parameter the simulation reads; it is a CHECK on the renderer, and
    the renderer currently fails it. The microvillus coat built from the
    two sourced constants above (3 per um^2, 0.35 um) adds 30% to a
    smooth sphere, for every cell type, because count scales with area
    and the fraction is therefore scale-free. Measured is 84-137%, so
    the coat accounts for 22-35% of it, and the shortfall cannot be
    tuned away inside the measurements: closing it would need 1.60 um
    villi against a measured 0.35 um median, or 13.1 per um^2 against a
    measured 3 +/- 0.5. The reading is that most of the excess is not in
    villi at all -- the paper calls it 'membrane foldings' throughout
    and watches it unfold as the cell swells -- so the renderer needs a
    fold layer rather than different villi, and these four numbers are
    the per-type target it should hit. They also say the cell types
    DIFFER, 1.6-fold, where the renderer is flat;
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

That table was read on 2026-10-20, from a copy fetched by hand. Until
then every number here had reached the library through BioNumbers, and
the transcription turned out to be faithful -- all four match. What the
BNIDs did NOT carry is the Appendix, which says the four are
overestimates and by how much. A record of a measurement is not the
measurement.
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
        "fraction as a radius gives a nucleus at 8.8% of the cell. The "
        "same Appendix that discounts the granule fractions clears this "
        "one: the section-thickness error is 'negligible, because the "
        "section thickness is small in comparison to nuclear "
        "dimensions'. Checked against the paper's own Table 1, where an "
        "eosinophil nucleus of 37 um^3 in a 206 um^3 cell is 18.0% "
        "against Table 2's 17.9%.",
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
        "shape and not merely the wrong number. Confirmed from an "
        "unrelated direction by PMID 6775712, read 2026-10-20: TEM "
        "morphometry puts 'the height of the membrane foldings at 305 "
        "mosmole' at 'about 0.35 um', and checks it by arithmetic -- a "
        "6.72 um smooth-sphere diameter plus twice the folding height "
        "gives 7.42 um, which is what light microscopy measures at the "
        "outer edge. Two methods 24 years apart, one measuring villus "
        "length and the other folding height, agree to two figures.",
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

#: ``cell -> (fraction, BNID, stated maximum overestimate)``.
#:
#: The third number is the reason to read a paper rather than a record
#: of it. Table 2 gives the fractions bare, and so does every BioNumbers
#: entry derived from it; the paper's own Appendix then says they are
#: biased HIGH, and by how much. A granule's diameter is comparable to
#: the 0.08 um section, so a granule is caught by more sections than its
#: size warrants and is counted too often. The authors put the maximum
#: at 44% for lymphocytes, 29% for monocytes, 28% for neutrophils and
#: 19% for eosinophils -- a bias that runs the WRONG WAY for this
#: library, because it is largest exactly where the fraction is smallest.
GRANULE_FRACTIONS: Dict[str, Tuple[float, str, float]] = {
    "eosinophil": (0.230, "BNID 115197", 0.19),
    "neutrophil": (0.154, "BNID 115195", 0.28),
    "monocyte": (0.026, "BNID 115196", 0.29),
    "lymphocyte": (0.002, "BNID 115194", 0.44),
}


def _granule(fraction: float, bnid: str, bias: float, cell: str) -> Measurement:
    """One granule volume fraction, carrying its known bias.

    The value stays as Table 2 prints it. The bias is recorded beside it
    rather than subtracted out: the Appendix gives a MAXIMUM, not a
    point estimate, so correcting by it would trade a known overestimate
    for an unknown underestimate.
    """
    floor = fraction / (1.0 + bias)
    return Measurement(
        "granule_volume_fraction", fraction, "dimensionless",
        "PMID:6775712", f"{_TABLE} ({bnid})",
        f"Human peripheral blood {cell}; stereology on electron micrographs",
        "One table, one method, one paper, which is what makes the "
        "77-fold spread across these cells a comparison rather than four "
        "unrelated numbers. Table 2 as printed, which is an "
        f"OVERESTIMATE: the paper's Appendix puts the bias from finite "
        f"section thickness at up to {bias:.0%} for this cell, so the "
        f"true fraction is between {floor:.4f} and {fraction:.4f}. Not "
        "corrected here because the Appendix states a maximum rather "
        "than a point estimate.",
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
    # Still a lymphocyte: the nucleus, the microvilli and the resting
    # granule fraction all apply. Deliberately NOT in CYTOTOXIC_NAMES --
    # losing the lytic load is what exhaustion IS, so giving it the full
    # 63 granules would contradict the state the entity exists to model.
    "Exhausted CD8+ T cell",
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
        # Built the same way as the other three rather than written out
        # again. It had its own copy, which drifted the moment the table
        # grew a bias column: the lymphocyte carries the LARGEST stated
        # overestimate of the four, 44%, on the smallest fraction.
        out.append(_granule(*GRANULE_FRACTIONS["lymphocyte"], "lymphocyte"))
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
    """Every entity, by id.

    By NAME would skip duplicates, and the library has one: "NK cell"
    exists as both a cell_type and an immune_cell. Looking up by name
    returned the same record twice, so the immune_cell copy never got
    its granule count -- and that is the copy the immune module asks
    for, by LINEAGE_ENTITY['NK_cell'] = ('NK cell', 'immune_cell'). The
    NK cell, of all cells, ended up with no lytic granules.
    """
    rows = [r[0] for r in store._conn.execute("SELECT entity_id FROM entities")]
    for entity_id in rows:
        entity = store.get_entity(entity_id)
        if entity is not None:
            yield entity
