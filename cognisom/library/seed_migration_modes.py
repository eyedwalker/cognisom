"""A cell's speed is a property of how it is moving, not of what it is.

The library stores ``migration_speed_um_per_min = 0.234`` for the
stromal fibroblast, MEASURED against PMID 25679760. The number is
right: it is that paper's mesenchymal control. Its conditions came from
a BioNumbers record rather than from the paper, and the sentence quoted
as its rationale is about epithelial cells -- an adjacent paragraph.

What the record does not say is the paper's actual finding, and it is
the one that matters for a tissue simulation:

    mesenchymal (2D, adherent, unconfined)   0.234 +/- 0.09 um/min
    A1 amoeboid (confined, low adhesion)     1.7   +/- 0.4
    A2 amoeboid ("stable bleb", confined)    5.3   +/- 1.5

Twenty-three fold, and THE SAME CELLS. The authors put it plainly:
"these typical speeds are rather linked to the mode of migration than
to the type of cell, as a given cell type can adopt both types of
migration and associated speed, depending on its migration
environment."

WHY THIS IS NOT A FOOTNOTE
--------------------------
The switch is triggered by CONFINEMENT plus LOW ADHESION -- 3 um of
confinement on a non-adhesive surface. A solid tumour is a confined
environment, and the paper's closing argument is precisely that "some
physical characteristics of the tumour environment, such as the
confinement imposed by the surrounding tissue on the growing tumour,
might be enough to trigger a migratory switch toward a faster and more
invasive mode of migration". Reading 0.234 um/min into a confined
tissue simulation takes the one regime the paper says does not apply
there.

So the value stays -- it is a real measurement of a real regime -- and
it now carries the regime, the other two, and the trigger.

THE BRIDGE THE SIMULATION WAS MISSING
-------------------------------------
The same paper supplies the model. Cell tracks are fitted as a
PERSISTENT RANDOM WALK (the Fuerth model):

    MSD(t) = 2 S P (t - P (1 - exp(-t/P)))

which is ballistic below the persistence time P and diffusive above it,
and gives a diffusion coefficient

    D = S^2 P / 2

That is the missing conversion from a speed the library HAS to the
cell-level diffusion coefficient the mechanics needs, and it is why
speed alone was never enough: D depends on persistence too, and the
paper shows the two move in OPPOSITE directions as adhesion rises --
speed falls, persistence climbs, and D peaks in between.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

CITATION = "PMID:25679760"
SOURCE = ("Liu YJ, Le Berre M, Lautenschlaeger F, Maiuri P, Callan-Jones A, "
          "Heuze M, Takaki T, Voituriez R, Piel M. Cell "
          "2015;160(4):659-672, Figure 4C")


@dataclass(frozen=True)
class Mode:
    """One migration mode, as the paper measured it."""

    key: str
    name: str
    speed_um_per_min: float
    sd: float
    environment: str


MODES: Tuple[Mode, ...] = (
    Mode("mesenchymal", "mesenchymal", 0.234, 0.09,
         "2D adherent culture, unconfined: focal adhesions, stress "
         "fibres, spindle shape"),
    Mode("amoeboid_a1", "amoeboid A1", 1.7, 0.4,
         "3 um confinement on a non-adhesive surface, LOW contractility: "
         "a round body with a small leading edge, driven by a retrograde "
         "actin flow localised to that protrusion"),
    Mode("amoeboid_a2", "amoeboid A2 (stable bleb)", 5.3, 1.5,
         "3 um confinement on a non-adhesive surface, HIGH contractility: "
         "an elongated body with a uropod, driven by a GLOBAL cortical "
         "retrograde flow. Every leukocyte tested adopted this mode, and "
         "so did every transformed cell line"),
)

BY_KEY: Dict[str, Mode] = {m.key: m for m in MODES}

#: The entity whose stored speed this paper is the source for.
TARGET = "Stromal fibroblast"


def diffusion_coefficient_um2_per_min(speed_um_per_min: float,
                                      persistence_min: float) -> float:
    """``D = S^2 P / 2`` -- the Fuerth persistent-random-walk relation.

    The conversion the tissue mechanics needs and the library could not
    make. A speed on its own does not give a diffusion coefficient:
    a fast cell that turns constantly goes nowhere. Persistence is the
    second half of it, and the paper shows the two trade off -- as
    adhesion rises, instantaneous speed FALLS and persistence CLIMBS, so
    D peaks at an intermediate adhesion rather than at either end.
    """
    if speed_um_per_min < 0 or persistence_min < 0:
        raise ValueError("speed and persistence are non-negative")
    return speed_um_per_min ** 2 * persistence_min / 2.0


def mean_square_displacement_um2(speed_um_per_min: float,
                                 persistence_min: float,
                                 t_min: float) -> float:
    """The Fuerth model: ``MSD(t) = 2 S^2 P (t - P (1 - exp(-t/P)))``.

    S is SQUARED. The paper prints it as "2.S.P(...)" and the OCR of a
    superscript is lost, but the exponent is forced by the limits and by
    dimensions: below P this has to reduce to (S t)^2, and above it to
    4 D t with D = S^2 P / 2 in two dimensions. Writing it with a single
    S gives um.min, not um^2.

    Ballistic below P, diffusive above it. Kept beside the diffusion
    coefficient because D is only the long-time limit of this, and a
    tissue step shorter than P is in the ballistic regime where a plain
    random walk is the wrong model.
    """
    if persistence_min <= 0:
        return 0.0
    return (2.0 * speed_um_per_min ** 2 * persistence_min
            * (t_min - persistence_min * (1.0 - math.exp(-t_min / persistence_min))))


def conditions() -> str:
    rows = "; ".join(f"{m.name} {m.speed_um_per_min:g}+/-{m.sd:g}"
                     for m in MODES)
    return (
        "Normal human dermal fibroblasts and 20 other cell types, tracked "
        "by Hoechst-stained nuclei at 37 C and fitted as a persistent "
        "random walk (Fuerth model). THE SPEED IS THE MODE, NOT THE CELL: "
        f"{rows} um/min, all measured on the same cell types. The switch "
        "is triggered by 3 um confinement plus the absence of focal "
        "adhesions, which is why this matters for a tumour: the authors "
        "argue the confinement imposed by surrounding tissue 'might be "
        "enough to trigger a migratory switch toward a faster and more "
        "invasive mode of migration'. The 0.234 figure is the UNCONFINED "
        "2D adherent regime and must not be read into a confined one.")


@dataclass
class Run:
    rewritten: List[str] = field(default_factory=list)
    modes_recorded: int = 0
    refused: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (f"{len(self.rewritten)} records given their migration "
                f"regime, {self.modes_recorded} modes recorded, "
                f"{len(self.refused)} refused")


def seed(store, *, dry_run: bool = False) -> Run:
    """Give the stored speed its regime, and record the other two."""
    from .parameters import (ParameterError, ParameterValue, Provenance,
                             get_parameter, set_parameter)

    run = Run()
    entity = store.find_entity_by_name(TARGET)
    if entity is None:
        run.refused.append(f"{TARGET}: not in this library")
        return run

    existing = get_parameter(entity, "migration_speed_um_per_min")
    if existing is None:
        run.refused.append(f"{TARGET}: has no migration speed to correct")
        return run
    if dry_run:
        run.rewritten.append(TARGET)
        return run

    mesenchymal = BY_KEY["mesenchymal"]
    try:
        set_parameter(entity, "migration_speed_um_per_min", ParameterValue(
            value=mesenchymal.speed_um_per_min, unit="um/min",
            provenance=Provenance.MEASURED,
            citation=CITATION, source=SOURCE, conditions=conditions(),
            rationale=(
                "Figure 4C, the mesenchymal control. The number was "
                "already correct; its conditions came from a BioNumbers "
                "record rather than the paper, and the sentence stored as "
                "its rationale is about EPITHELIAL cells -- an adjacent "
                "paragraph that BioNumbers picked up.\n"
                "What was missing is the finding: the same cells reach "
                "1.7 um/min (A1) and 5.3 um/min (A2) once confined to "
                "3 um without focal adhesions. Twenty-three fold, with no "
                "genetic change. A confined tissue simulation that reads "
                "0.234 is using the one regime the paper excludes.")))
    except ParameterError as exc:
        run.refused.append(f"{TARGET}: {exc}")
        return run

    entity.sourced["migration_modes"] = {
        m.key: {"speed_um_per_min": m.speed_um_per_min, "sd": m.sd,
                "environment": m.environment, "citation": CITATION}
        for m in MODES
    }
    entity.sourced["migration_model"] = (
        "Persistent random walk (Fuerth): MSD(t) = 2 S P (t - P(1 - "
        "exp(-t/P))), D = S^2 P / 2. Speed alone does not give a "
        "diffusion coefficient; persistence is the other half, and the "
        f"two trade off with adhesion. {CITATION}")
    store.update_entity(entity, changed_by="seed_migration_modes")
    run.rewritten.append(TARGET)
    run.modes_recorded = len(MODES)
    return run


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
    print("\nTHE SAME CELLS, THREE REGIMES:")
    for mode in MODES:
        print(f"   {mode.name:<26} {mode.speed_um_per_min:>6.3f} "
              f"+/- {mode.sd:<5.2f} um/min")
    fast, slow = MODES[-1].speed_um_per_min, MODES[0].speed_um_per_min
    print(f"\n   ratio: {fast / slow:.0f}x, with no genetic change")
    for line in run.refused:
        print("   REFUSED:", line[:140])
    return 0


if __name__ == "__main__":                             # pragma: no cover
    raise SystemExit(main())
