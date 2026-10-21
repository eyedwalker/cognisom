"""The adhesion-cascade bonds, and what their numbers did not say.

The adhesion cascade's binding constants were already in this library
and they were already right -- Kd 320 nM for P-selectin, 62 uM for
E-selectin, 108 uM for L-selectin, each against the correct PMID. They
were sourced by an agent that had not read the papers, and the numbers
survived contact with them. What did not survive is the context.

PMID 9829984 (Mehta, Cummings & McEver, JBC 1998) and PMID 11404363
(Wild et al., JBC 2001) are now both on disk. Table IV of the second
sets all three interactions side by side, and it is the table that
makes the problems visible:

    Interaction            Species   T     Kd      kon      koff
    E-selectin-ESL-1       MOUSE     37    62 uM   7.4e4    4.6
    E-selectin-ESL-1       MOUSE     25    56 uM   4.8e4    2.7
    L-selectin-GlyCAM-1    MOUSE     25   108 uM  >=1e5    >=10
    P-selectin-PSGL-1      HUMAN     25  0.32 uM   4.4e6    1.4

WHAT THE STORED RECORDS GOT WRONG
---------------------------------
1. SPECIES. L-selectin's conditions read "human CD62L". Table IV says
   Mouse. E-selectin's read "mouse ligand", which understates it: the
   E-selectin constructs are mouse too, cloned from mouse cDNA
   (GenBank M87862). Both entities carry human UniProt accessions
   (P14151, P16581), so two of the three bonds in this library's
   adhesion cascade are mouse constants on human molecules.

2. TEMPERATURE. Nothing recorded one, and koff for E-selectin-ESL-1
   runs 1.1 / 1.9 / 2.7 / 4.6 s-1 at 5 / 15 / 25 / 37 C (Table III) --
   a factor of 4.2 across the measured range. The stored E-selectin
   values are the 37 C row, which is the right choice for a body-
   temperature simulation. The stored P-selectin values are 25 C,
   because that is the only temperature its paper reports. So the
   library silently mixes 25 C and 37 C constants across the three
   bonds of one cascade.

3. BOUNDS READ AS POINTS. L-selectin's koff and kon are published as
   >=10 s-1 and >=1e5 M-1 s-1: its authors say the kinetics were too
   fast to measure precisely. Stored as 10.0 and 1e5 they read as
   measurements. The grade was already ESTIMATED, which is right, but
   the ">=" is the whole content of the value and it was gone.

4. WHICH BOND. Every constant sat under a bare ``kd_nm``. Each selectin
   binds several ligands: E-selectin binds ESL-1 AND PSGL-1, and
   P-selectin binds PSGL-1 200x more tightly than E-selectin binds
   ESL-1. The values move to partner-named aliases so the pair travels
   with the number.

WHAT IS DELIBERATELY NOT CHANGED
--------------------------------
The numbers. All four stored values reproduce their papers exactly, and
kon stays DERIVED on P- and E-selectin because both papers say the
association phase was too fast to measure -- equilibrium inside 2-3 s --
so kon is computed as koff/Kd and is not an observation.

MONOMERIC, AND THAT IS THE POINT
--------------------------------
These are affinities of ONE selectin for ONE ligand, measured that way
on purpose: both papers go to lengths to exclude oligomers, because an
earlier radiolabel study of P-selectin reported ~70 nM and the authors
attribute the 4x discrepancy to a trace of dimers binding multivalently.
On a real cell the interaction is multivalent and stronger, so these
numbers are a floor on avidity, not an estimate of it. Each record now
says so, and carries the cellular off-rate beside the biochemical one:
they do not agree, and the papers discuss why at length.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

MEHTA = "PMID:9829984"
WILD = "PMID:11404363"
MEHTA_SOURCE = ("Mehta P, Cummings RD, McEver RP. J Biol Chem "
                "1998;273(49):32506-32513, Table I")
WILD_SOURCE = ("Wild MK, Huang MC, Schulze-Horsel U, van der Merwe PA, "
               "Vestweber D. J Biol Chem 2001;276(34):31602-31612, "
               "Tables II-IV")

#: The shared caveat. Every one of these is a monomeric affinity.
MONOMERIC = (
    "MONOMERIC affinity, deliberately: both papers exclude oligomers by "
    "gel filtration, high flow and short injections, because an earlier "
    "radiolabel study of P-selectin read ~70 nM and its authors ascribe "
    "the difference to a trace of dimers binding multivalently. On a cell "
    "the bond is multivalent and stronger, so this is a floor on avidity "
    "rather than an estimate of it.")


@dataclass(frozen=True)
class Bond:
    """One selectin-ligand pair as Table IV prints it."""

    entity: str
    ligand: str
    #: The alias that names the partner, so the pair travels with the value.
    kd_parameter: str
    species: str
    temperature_c: int
    kd_nm: float
    koff_per_s: float
    #: Published as a lower bound rather than a point estimate.
    is_bound: bool
    citation: str
    source: str
    #: Off-rate from flowing-cell tether lifetimes, which is a different
    #: experiment and does not agree.
    cellular_koff: str
    note: str = ""
    #: The grade for each of Kd, koff and kon, stated rather than
    #: inherited from whatever the library happened to hold. Inheriting
    #: it was a real bug: the first pass moved a MEASURED Kd to its
    #: substrate-named alias, the second pass read the bare name, got
    #: PLACEHOLDER from the alias fallback, and wrote that back over its
    #: own evidence. A grade is a fact about the EXPERIMENT, so it
    #: belongs beside the number it describes.
    grades: Tuple[str, str, str] = ("measured", "measured", "derived")
    #: Which of Kd and kon was observed. In the selectin papers the
    #: association phase was too fast to measure, so kon = koff/Kd. In
    #: the ICAM-1/LFA-1 paper it is the other way round: kass was fitted
    #: to the association phase and "Kd was calculated from the ratio
    #: kdiss/kass". Storing both as MEASURED would assert two
    #: observations where there was one.
    kd_is_calculated: bool = False
    #: The conformational state the measurement was made in, when the
    #: molecule has more than one and they differ by orders of magnitude.
    affinity_state: str = ""

    def conditions(self) -> str:
        bound = ">=" if self.is_bound else ""
        return (
            f"{self.entity} / {self.ligand}; surface plasmon resonance at "
            f"{self.temperature_c} C. SPECIES: {self.species.upper()}"
            + ("" if " " in self.species.strip() else " -- both partners")
            + f". Kd {self.kd_nm:g} nM, koff {bound}"
            f"{self.koff_per_s:g} s-1. Cellular off-rate from tether "
            f"lifetimes under flow: {self.cellular_koff}. {MONOMERIC}"
            + (f" STATE: {self.affinity_state}" if self.affinity_state else "")
            + (f" {self.note}" if self.note else ""))


TABLE_IV: Tuple[Bond, ...] = (
    Bond(
        entity="P-selectin (CD62P)", ligand="PSGL-1",
        kd_parameter="kd_psgl1_nm", species="human", temperature_c=25,
        kd_nm=320.0, koff_per_s=1.4, is_bound=False,
        citation=MEHTA, source=MEHTA_SOURCE,
        cellular_koff="0.95 +/- 0.17 s-1 for unstressed neutrophil tethers",
        grades=("measured", "measured", "derived"),
        note=("PSGL-1 purified from human NEUTROPHILS, so its O-glycans "
              "are those of a real leukocyte rather than of a transfected "
              "line expressing one fucosyltransferase. The nine consensus "
              "repeats contribute nothing measurable: Lec-EGF alone reads "
              "422 +/- 89 nM.")),
    Bond(
        entity="E-selectin (CD62E)", ligand="ESL-1",
        kd_parameter="kd_esl1_nm", species="mouse", temperature_c=37,
        kd_nm=62000.0, koff_per_s=4.6, is_bound=False,
        citation=WILD, source=WILD_SOURCE,
        cellular_koff="0.7 s-1 for human leukocyte tethers at 25 C, ~4x "
                      "slower than the 2.7 s-1 measured here at 25 C",
        grades=("measured", "measured", "derived"),
        note=("koff over the measured range: 1.1 / 1.9 / 2.7 / 4.6 s-1 at "
              "5 / 15 / 25 / 37 C. Affinity is flat with temperature "
              "(56-62 uM throughout), so binding is entropically driven "
              "-- unusual for a protein-carbohydrate interaction. ESL-1 "
              "sits on the LATERAL surfaces of microvilli, not the tips "
              "where PSGL-1 and L-selectin are, which fits a role in "
              "strengthening rolling rather than first capture.")),
    Bond(
        entity="L-selectin (CD62L)", ligand="GlyCAM-1",
        kd_parameter="kd_glycam1_nm",
        species="RAT CD62L binding MOUSE GlyCAM-1 -- a cross-species "
                "pair, not one organism", temperature_c=25,
        kd_nm=108000.0, koff_per_s=10.0, is_bound=True,
        citation="PMID:9422729", source=(
            "Nicholson MW, Barclay AN, Singer MS, Rosen SD, van der Merwe "
            "PA. J Biol Chem 1998;273(2):763-770, Tables II and III"),
        cellular_koff="~7 s-1 intrinsic, from L-selectin tethers "
                      "extrapolated to zero flow, which agrees with this",
        grades=("measured", "estimated", "estimated"),
        note=(
            "SPECIES, FROM THE PRIMARY SOURCE. PMID:11404363 Table IV "
            "lists this interaction as 'Mouse'. It is not: the authors "
            "expressed a soluble monomeric RAT CD62L and bound it to "
            "GlyCAM-1 purified from MOUSE serum, and flag it themselves "
            "-- 'One caveat is that we measured the affinity and "
            "kinetics of rat CD62L binding to mouse GlyCAM-1.' They "
            "argue it is tolerable because the two CD62L lectin domains "
            "are 93% identical. A secondary table simplified it, and "
            "this library had taken the simplification.\n"
            "WHY koff IS A BOUND AND NOT A MEASUREMENT: the observed "
            "decay, half-time ~0.07 s, is the time to WASH THE SAMPLE "
            "OUT OF THE FLOW CELL, not dissociation -- the control "
            "surface falls at the same rate, and the instrument cannot "
            "sample faster than 0.1 s. So >=10 s-1 is a floor set by "
            "the apparatus. kon >=1e5 is then calculated from one bound "
            "and one affinity, so it is a bound too.\n"
            "HOW FAR THE MONOMERIC FLOOR IS BELOW AVIDITY, measured in "
            "this same paper: soluble GlyCAM-1 binding to IMMOBILISED "
            "CD62L is detectable at ~30 nM and comes off with koff "
            "<=0.001 s-1 -- at least 10,000x slower than the 10 s-1 "
            "monovalent rate. That is multivalent binding at serum "
            "concentration, and it is how GlyCAM-1 is thought to signal "
            "by cross-linking CD62L.\n"
            "Affinity is flat with temperature (108 +/- 4 uM at 25 C, "
            "n=8; 100 +/- 6 at 37 C), so like E-selectin-ESL-1 this bond "
            "is entropically driven. If CD62L-CD4 were in fact a dimer "
            "the authors note both Kd and koff would be UNDERestimates.")),
    Bond(
        entity="ICAM-1 (CD54)", ligand="LFA-1",
        kd_parameter="kd_lfa1_nm", species="human", temperature_c=25,
        kd_nm=500.0, koff_per_s=0.1, is_bound=False,
        citation="PMID:9780171", source=(
            "Tominaga Y, Kita Y, Satoh A, Asai S, Kato K, Ishikawa K, "
            "Horiuchi T, Takashi T. J Immunol 1998;161(8):4016-4022, "
            "Table I"),
        cellular_koff="not reported; this is the firm-adhesion bond, not "
                      "a rolling tether, and it has no tether lifetime",
        kd_is_calculated=True,
        grades=("derived", "measured", "measured"),
        affinity_state=(
            "Mg2+-INDUCED HIGH-AFFINITY STATE, and that is a choice the "
            "authors made: 1 mM Ca2+ cut binding by 20%, so they ran "
            "'in the absence of Ca2+, but in the presence of Mg2+ to "
            "induce high affinity state of LFA-1'. LFA-1 on an "
            "UNSTIMULATED T cell binds ICAM-1 at about 100 uM -- 200x "
            "weaker than this. A resting lymphocyte modelled with 500 nM "
            "is wrong by that factor, and this is the only bond in the "
            "cascade with two states this far apart."),
        note=("Binding is to D1D2 of ICAM-1 as an IgG chimera; D1D5 has "
              "the same potency. Native ICAM-1 is a DIMER, and "
              "dimerising soluble ICAM-1 raises affinity for LFA-1 'by "
              "two orders of magnitude' -- so the monomeric floor is "
              "especially far below the cellular avidity here. Other "
              "reported values: 100 nM and 130 nM by receptor-binding "
              "assay, 400 nM on phorbol-activated T cells.")),
)


@dataclass
class Run:
    rewritten: List[str] = field(default_factory=list)
    moved: List[str] = field(default_factory=list)
    species_corrected: List[str] = field(default_factory=list)
    #: A constant that was graded as observed when its paper calculated
    #: it, or the reverse.
    grade_corrected: List[str] = field(default_factory=list)
    refused: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (f"{len(self.rewritten)} records given their species and "
                f"temperature, {len(self.moved)} Kd values moved to a "
                f"partner-named alias, {len(self.species_corrected)} "
                f"species errors corrected, {len(self.grade_corrected)} "
                f"grades corrected, {len(self.refused)} refused")


#: Entities that carry a bond's constants because they are the LIGAND.
ALSO_CARRIES: Dict[str, str] = {"PSGL-1 (CD162)": "PSGL-1",
                                "LFA-1 (CD11a/CD18)": "LFA-1"}


def seed(store, *, dry_run: bool = False) -> Run:
    """Give each selectin constant its species, temperature and partner."""
    from .parameters import (ParameterError, ParameterValue, get_parameter,
                             set_parameter)

    run = Run()
    by_ligand = {bond.ligand: bond for bond in TABLE_IV}
    targets = [(bond.entity, bond) for bond in TABLE_IV]
    targets += [(name, by_ligand[ligand])
                for name, ligand in ALSO_CARRIES.items()
                if ligand in by_ligand]

    for entity_name, bond in targets:
        entity = store.find_entity_by_name(entity_name)
        if entity is None:
            run.refused.append(f"{entity_name}: not in this library")
            continue
        conditions = bond.conditions()

        for parameter in ("kd_nm", "koff_per_s", "kon_per_nm_per_s",
                          "bond_lifetime_seconds"):
            existing = get_parameter(entity, parameter)
            if existing is None:
                continue
            if parameter == "kd_nm" and "human CD62L" in (
                    existing.conditions or ""):
                run.species_corrected.append(
                    f"{entity_name}: conditions said 'human CD62L'; "
                    f"PMID:11404363 Table IV says {bond.species}")

            # The Kd moves to the alias that names the ligand. The others
            # keep their canonical names -- there is no koff alias
            # vocabulary and inventing one here would be a bigger change
            # than the evidence asks for.
            destination = (bond.kd_parameter if parameter == "kd_nm"
                           else parameter)
            if dry_run:
                run.rewritten.append(f"{entity_name}.{destination}")
                continue
            provenance = _provenance(bond, parameter, existing.provenance)
            if provenance is not existing.provenance:
                run.grade_corrected.append(
                    f"{entity_name}.{destination}: "
                    f"{existing.provenance.value} -> {provenance.value}")
            try:
                set_parameter(entity, destination, ParameterValue(
                    value=existing.value, unit=existing.unit,
                    provenance=provenance,
                    citation=bond.citation, source=bond.source,
                    conditions=conditions,
                    rationale=_rationale(bond, parameter)))
            except ParameterError as exc:
                run.refused.append(f"{entity_name}.{destination}: {exc}")
                continue
            run.rewritten.append(f"{entity_name}.{destination}")
            if destination != parameter:
                params = getattr(entity, "physics_params", None) or {}
                params.pop(parameter, None)
                sidecar = params.get("parameter_provenance")
                if isinstance(sidecar, dict):
                    sidecar.pop(parameter, None)
                run.moved.append(
                    f"{entity_name}: {parameter} -> {destination}")

        if not dry_run:
            entity.sourced["selectin_bond"] = (
                f"{bond.ligand}, {bond.species}, {bond.temperature_c} C, "
                f"monomeric ({bond.citation})")
            store.update_entity(entity, changed_by="seed_selectin_kinetics")
    return run


def _provenance(bond: Bond, parameter: str, current):
    """Which of Kd and kon was actually observed, for this paper.

    The two families of measurement are mirror images and the library
    had both graded the same way. In the selectin papers the association
    phase reached equilibrium in 2-3 s, too fast to fit, so koff and Kd
    were measured and kon was computed as koff/Kd. In the ICAM-1/LFA-1
    paper kass WAS fitted to the association phase and "Kd was
    calculated from the ratio kdiss/kass" -- so there the derived
    quantity is the affinity, not the on-rate.

    Grading a calculated number MEASURED asserts an observation that was
    never made, and it is the affinity, not the rate, that a reader
    quotes.
    """
    from .parameters import Provenance

    index = {"kd_nm": 0, "koff_per_s": 1, "kon_per_nm_per_s": 2}.get(parameter)
    if index is None:                     # bond_lifetime_seconds is 1/koff
        return Provenance.DERIVED
    return Provenance(bond.grades[index])


def _rationale(bond: Bond, parameter: str) -> str:
    if parameter == "kon_per_nm_per_s":
        if bond.kd_is_calculated:
            return ("OBSERVED: fitted directly to the association phase. "
                    "It is the Kd here that is calculated, from "
                    "kdiss/kass -- the reverse of the selectin papers, "
                    "where the association was too fast to fit.")
        return (
            "CALCULATED as koff/Kd, not observed. The selectin papers "
            "state the association phase was too fast to measure -- "
            "equilibrium inside 2-3 s -- so kon there is not a direct "
            "observation.")
    if parameter == "kd_nm" and bond.kd_is_calculated:
        return ("CALCULATED, not measured: the paper says 'Kd was "
                "calculated from the ratio kdiss/kass'. Stored MEASURED "
                "it asserted an equilibrium experiment that was never "
                "run, on the one number a reader quotes.")
    if parameter == "bond_lifetime_seconds":
        return (f"1/koff = 1/{bond.koff_per_s:g} s-1, so it inherits "
                f"everything the off-rate does: species {bond.species}, "
                f"{bond.temperature_c} C, and a monomeric bond.")
    if bond.is_bound:
        return ("Published as a LOWER BOUND, not a point estimate. The "
                "decay the instrument saw was the washing time -- how "
                "long it takes to clear the sample from the flow cell, "
                "half-time ~0.07 s -- and the control surface falls at "
                "the same rate, so no dissociation was observed at all. "
                "The floor is set by the apparatus, which cannot sample "
                "faster than 0.1 s. Stored as a plain number because "
                "the library has no way to hold an inequality, which is "
                "exactly why the bound is spelled out here.")
    return (f"Table IV, the {bond.species} {bond.temperature_c} C row. "
            f"The number was already correct in this library; what it "
            f"lacked was the species, the temperature, and the name of "
            f"the ligand it is a constant for.")


def main(argv: Sequence[str] | None = None) -> int:    # pragma: no cover
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
    if run.grade_corrected:
        print("\nGRADES CORRECTED (calculated vs observed):")
        for line in run.grade_corrected:
            print("   ", line)
    if run.species_corrected:
        print("\nSPECIES CORRECTED:")
        for line in run.species_corrected:
            print("   ", line)
    if run.moved:
        print("\nMOVED TO A PARTNER-NAMED ALIAS:")
        for line in run.moved:
            print("   ", line)
    if run.refused:
        print("\nREFUSED:")
        for line in run.refused:
            print("   ", line[:140])
    return 0


if __name__ == "__main__":                             # pragma: no cover
    raise SystemExit(main())
