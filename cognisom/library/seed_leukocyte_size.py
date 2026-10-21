"""How big a white blood cell is, from the paper that measured them.

Twenty-three of this library's twenty-five immune cells carry a
PLACEHOLDER radius, and radius is the number every piece of geometry
multiplies through: membrane area, surface receptor counts, the volume
a granule fraction is a fraction OF, whether a cell fits down a
capillary. The two that were not placeholders were a macrophage and a
neutrophil.

Downey et al., J Appl Physiol 1990;69(5):1767-1778 (PMID 2272970)
measured three of these lineages directly, two ways, in 11,000-70,000
cells from each of three donors.

THE NEUTROPHIL WAS ALREADY CITED TO THIS PAPER
----------------------------------------------
It read 4.2 um, graded MEASURED, citation PMID:2272970 -- written
before anyone had the paper. It is nearly right, and it is the wrong
column: 4.2 is the Coulter counter mean. The paper's own text leads
with flow cytometry ("neutrophils were the largest cells with a mean
diameter of 8.3 um") and names a reason to distrust the Coulter figure
-- "possible overestimation of cell volume due to plasma trapping in
the surface folds". So the stored number came from the method the
authors trusted less, with nothing recording which method it was.

The gap is small for a neutrophil (4.17 against 4.20) and it is not
small for a monocyte: 4.06 by flow, 4.40 by Coulter, an 8% difference
in radius and 26% in volume. Each value below therefore records BOTH
and says which one it is.

TWO PAPERS, THE SAME CELLS, AND A REAL DISAGREEMENT
---------------------------------------------------
These diameters are larger than Schmid-Schonbein et al., Blood 1980
(PMID 6775712) -- the paper this library already uses for granule
volume fractions. Downey explains why, and it is not an error by
either side:

    "The measurements of Schmid-Schoenbein et al. determined the
    diameter of the mass of the cell excluding the membrane 'ruffles',
    the inclusion of which will increase the 'functional' diameter of
    the cell significantly. In vivo, the greater diameter (inclusive of
    membrane 'ruffles') may be the most relevant as the cell squeezes
    through a vessel the diameter of which is smaller than that of the
    cell."

Two conventions, each right for its own question. A cell's CONTENTS --
how much cytoplasm, what fraction is granule -- scale with the
ruffle-excluded mass. A cell's ENCOUNTERS -- what it collides with,
what it can squeeze past, how much membrane it presents -- scale with
the ruffle-inclusive diameter. This library uses the radius for
encounters and for drawing, so the ruffle-inclusive number is the
right one here. That choice is recorded on every value, because the
library also holds granule fractions from the other convention and
nothing else would reveal that the two were measured differently.

WHAT IS DELIBERATELY NOT FILLED IN
----------------------------------
Downey measured UNACTIVATED peripheral blood leukocytes: neutrophils,
monocytes, and a 98-99% pure elutriated lymphocyte fraction. A bulk
resting lymphocyte is a fair description of a naive or resting memory
cell. It is not a description of a blasting effector, and a Th1 cell
is not a small resting lymphocyte. So the effector subsets keep their
placeholders and gain a note saying what would be needed instead --
sixteen entities this pass could have filled and does not.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

CITATION = "PMID:2272970"
SOURCE = ("Downey GP, Doherty DE, Schwab B 3rd, Elson EL, Henson PM, "
          "Worthen GS. J Appl Physiol 1990;69(5):1767-1778, Table 1")

#: The other convention, for the note every value carries.
RUFFLE_EXCLUDED = "PMID:6775712"


@dataclass(frozen=True)
class Measured:
    """One lineage, both methods, as Table 1 prints them."""

    lineage: str
    #: Per-donor mean diameter in um, flow cytometry (axial pulse
    #: extinction), three donors.
    flow: Tuple[float, ...]
    #: The same three donors by Coulter counter.
    coulter: Tuple[float, ...]
    #: The narrowest and widest single cell across the flow rows.
    flow_range: Tuple[float, float]

    @property
    def flow_diameter(self) -> float:
        return sum(self.flow) / len(self.flow)

    @property
    def coulter_diameter(self) -> float:
        return sum(self.coulter) / len(self.coulter)

    @property
    def radius_um(self) -> float:
        return self.flow_diameter / 2.0

    def conditions(self) -> str:
        lo, hi = self.flow_range
        return (
            f"Unactivated human peripheral blood {self.lineage}s, isolated "
            f"by plasma-Percoll and counterflow elutriation, measured in "
            f"suspension at 22 C. Flow cytometry (axial pulse extinction, "
            f"calibrated against 5/10/15 um latex beads): mean of three "
            f"donors {self.flow_diameter:.2f} um, individual cells "
            f"{lo:.2f}-{hi:.2f} um, 11,000-70,000 cells per donor. The same "
            f"donors by Coulter counter read {self.coulter_diameter:.2f} um; "
            f"the flow figure is used because the authors lead with it and "
            f"name a bias in the other ('overestimation of cell volume due "
            f"to plasma trapping in the surface folds'). "
            f"RUFFLE-INCLUSIVE: this is the functional diameter including "
            f"membrane folds, which is larger than the ruffle-excluded "
            f"morphometry in {RUFFLE_EXCLUDED} that this library uses for "
            f"granule volume fractions.")


#: Table 1, transcribed. Three donor rows per lineage per method.
TABLE_1: Tuple[Measured, ...] = (
    Measured("neutrophil", (8.34, 8.22, 8.44), (8.34, 8.35, 8.50),
             (7.04, 10.25)),
    Measured("monocyte", (8.13, 8.13, 8.12), (8.86, 8.89, 8.62),
             (6.96, 9.09)),
    Measured("lymphocyte", (6.08, 6.09, 5.98), (6.94, 7.09, 7.04),
             (4.92, 7.25)),
)

BY_LINEAGE: Dict[str, Measured] = {m.lineage: m for m in TABLE_1}

#: entity name -> the lineage Downey measured.
#:
#: Lymphocyte covers only the RESTING small lymphocytes, which is what a
#: 98-99% pure elutriated blood lymphocyte fraction is made of.
APPLIES: Dict[str, str] = {
    "Neutrophil": "neutrophil",
    "Monocyte": "monocyte",
    "Naive B cell": "lymphocyte",
    "Memory B cell": "lymphocyte",
    "CD8+ Naive T cell": "lymphocyte",
    "CD8+ Memory T cell": "lymphocyte",
}

#: entity -> why this paper does not answer for it. Recorded rather than
#: guessed: a wrong size propagates into every area and volume.
DECLINED: Dict[str, str] = {
    "CD4+ Th1 cell": "differentiated effector, not a resting lymphocyte",
    "CD4+ Th2 cell": "differentiated effector, not a resting lymphocyte",
    "CD4+ Th17 cell": "differentiated effector, not a resting lymphocyte",
    "Regulatory T cell (Treg)": "differentiated effector",
    "T follicular helper (Tfh)": "differentiated effector",
    "CD8+ Effector T cell": "blasting effector; larger than a resting cell",
    "Gamma-delta T cell": "not resolved by the elutriation used here",
    "NKT cell": "not resolved by the elutriation used here",
    "NK cell": "a large granular lymphocyte; the bulk fraction is not it",
    "ILC1": "not a blood population this paper separated",
    "ILC2": "not a blood population this paper separated",
    "ILC3": "not a blood population this paper separated",
    "Plasma cell": "tissue-resident and much larger than a blood lymphocyte",
    "Basophil": "not measured here",
    "Mast cell": "tissue-resident; not a blood leukocyte",
    "Eosinophil": "not measured here (PMID 6775712 has it, ruffle-excluded)",
    "Conventional dendritic cell (cDC)": "not measured here",
    "Plasmacytoid dendritic cell (pDC)": "not measured here",
    "Macrophage M2": "tissue-resident; M1 is already sourced separately",
}


@dataclass
class Run:
    written: List[Tuple[str, float, float]] = field(default_factory=list)
    corrected: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    refused: List[str] = field(default_factory=list)
    #: Entities left placeholder on purpose, with the reason.
    declined: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (f"{len(self.written)} radii written "
                f"({len(self.corrected)} corrected), "
                f"{len(self.declined)} deliberately left placeholder, "
                f"{len(self.missing)} entities absent, "
                f"{len(self.refused)} refused by the spec")


def rationale(measured: Measured, entity_name: str) -> str:
    text = (
        f"Table 1, flow cytometry column, mean of three donors: "
        f"{measured.flow_diameter:.3f} um diameter -> "
        f"{measured.radius_um:.3f} um radius. The paper's own text rounds "
        f"this to '{measured.flow_diameter:.2f}', which is how the "
        f"transcription here was checked.")
    if measured.lineage == "lymphocyte":
        text += (
            "\nThis is a BULK resting lymphocyte: the fraction was 98-99% "
            "pure after elutriation and was not sorted into T and B, so the "
            "number describes a small resting lymphocyte and is applied "
            "only to entities that are one. The differentiated effector "
            "subsets keep their placeholders.")
    if entity_name == "Neutrophil":
        text += (
            f"\nThis replaces a stored 4.2 um that was already graded "
            f"MEASURED against this same PMID before the paper was read. "
            f"4.2 is the COULTER mean ({measured.coulter_diameter:.2f} um "
            f"diameter), not the flow mean the authors lead with, and "
            f"nothing recorded which of the two it was.")
    return text


def seed(store, *, dry_run: bool = False) -> Run:
    """Write a measured radius wherever this paper actually answers."""
    from .parameters import (ParameterError, ParameterValue, Provenance,
                             get_parameter, set_parameter)

    run = Run()
    for entity_name, lineage in APPLIES.items():
        entity = store.find_entity_by_name(entity_name)
        if entity is None:
            run.missing.append(entity_name)
            continue
        measured = BY_LINEAGE[lineage]
        before = get_parameter(entity, "radius_um")
        if dry_run:
            run.written.append((entity_name, before.value if before else 0.0,
                                measured.radius_um))
            continue
        try:
            set_parameter(entity, "radius_um", ParameterValue(
                value=round(measured.radius_um, 3), unit="um",
                provenance=Provenance.MEASURED,
                citation=CITATION, source=SOURCE,
                conditions=measured.conditions(),
                rationale=rationale(measured, entity_name)))
        except ParameterError as exc:
            run.refused.append(f"{entity_name}: {exc}")
            continue
        entity.sourced["diameter_convention"] = (
            "ruffle-inclusive (functional), per PMID:2272970; the granule "
            f"fractions in this library are ruffle-excluded, {RUFFLE_EXCLUDED}")
        store.update_entity(entity, changed_by="seed_leukocyte_size")
        run.written.append((entity_name, before.value if before else 0.0,
                            measured.radius_um))
        if before is not None and \
                abs(before.value - measured.radius_um) > 0.005:
            run.corrected.append(
                f"{entity_name}: {before.value} -> "
                f"{measured.radius_um:.3f} um "
                f"[was {before.provenance.value}]")

    for entity_name, why in DECLINED.items():
        if store.find_entity_by_name(entity_name) is not None:
            run.declined.append(f"{entity_name}: {why}")
    return run


def main(argv: Sequence[str] | None = None) -> int:     # pragma: no cover
    import argparse
    import contextlib
    import io

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    from .store import EntityStore
    with contextlib.redirect_stdout(io.StringIO()):
        store = EntityStore()
    run = seed(store, dry_run=args.dry_run)
    print(run.summary())
    print("\nWRITTEN:")
    for name, before, after in run.written:
        was = f"{before:g}" if before else "none"
        print(f"   {name:<24} {was:>6} -> {after:6.3f} um")
    if run.corrected:
        print(f"\nCORRECTED ({len(run.corrected)}):")
        for line in run.corrected:
            print("   ", line)
    print(f"\nLEFT PLACEHOLDER ON PURPOSE ({len(run.declined)}):")
    for line in run.declined[:8]:
        print("   ", line)
    if run.refused:
        print(f"\nREFUSED ({len(run.refused)}):")
        for line in run.refused:
            print("   ", line[:140])
    return 0


if __name__ == "__main__":                              # pragma: no cover
    raise SystemExit(main())
