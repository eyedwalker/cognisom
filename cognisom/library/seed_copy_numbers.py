"""Protein copies per cell, from relative abundance and one absolute count.

PaxDb measures what a proteome is made of, in parts per million of total
protein MOLECULES. That is a ratio, and a ratio cannot become a count
without knowing how many molecules a cell holds. The library's own
abundance note says exactly this, and refuses to make the leap:

    "Not copies per cell: that would be this number times the total
    protein molecules in the cell, which this library cannot cite for
    any of its cell types."

It can now, by a different route. Rather than sourcing a protein density
-- the obvious candidate record turned out to be "number of proteins
PRODUCED per cell", a flux rather than a census -- the scale is fixed
from one absolute measurement the library already holds a paper for.

THE ANCHOR
----------
PMID 9829984 (Mehta, Cummings & McEver 1998), already on disk for the
P-selectin binding constants, reports that radiolabelled soluble
P-selectin binds to approximately 25,000 sites on a human neutrophil,
and establishes in the same work that those sites all correspond to
PSGL-1 molecules. That is a direct count of one protein on one cell.

Combined with the neutrophil's measured radius (PMID 2272970) it gives
the number of protein molecules a cubic micrometre of cell contains, and
every other copy number follows from its ppm and its cell's volume.

WHAT THIS ASSUMES, STATED PLAINLY
---------------------------------
PaxDb has no neutrophil dataset for PSGL-1, so the ppm used for the
anchor is the MONOCYTE value, 37.5, as the nearest myeloid proxy. That
is a cross-cell-type substitution and it is the weakest link in the
chain, which is why nothing here is graded MEASURED. The grade is
CALIBRATED, and ``calibrated_against`` names the measurement.

THE INDEPENDENT CHECK
---------------------
The constant that falls out predicts the whole proteome of a HeLa cell
at its known volume. The literature range is 2-10e9 protein molecules;
if this calibration lands outside it, the chain is wrong somewhere and
:func:`seed` refuses to write anything.

WHAT THIS IS NOT
----------------
It is not ``surface_density_per_cell``, and the first version of this
module wrote it there. PaxDb is a shotgun proteome: it counts every
molecule in the cell, including a receptor's ER, Golgi, endosomal and
secretory pools alongside whatever is displayed on the membrane. The
mistake was visible the moment the numbers were checked against an area
-- 24 million CD14 on a macrophage is 43% of its entire membrane
covered by one protein -- so each value now records the membrane
occupancy it WOULD imply if it were all surface. Anything above a few
per cent cannot be, and that is most of them.

A receptor's surface fraction is a separate measurement, and the
library has it for nothing. So surface_density_per_cell is deliberately
left as it was.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

#: The absolute count the whole scale rests on.
ANCHOR_PROTEIN = "PSGL-1 (CD162)"
ANCHOR_COPIES = 25_000.0
ANCHOR_CELL = "Neutrophil"
ANCHOR_CITATION = "PMID:9829984"
#: PaxDb has no neutrophil dataset for PSGL-1; this is the monocyte
#: value, the nearest myeloid proxy, and the chain's weakest link.
ANCHOR_PPM = 37.5
ANCHOR_PPM_FROM = "Monocyte"

#: A HeLa cell is about this big, and holds this many protein molecules.
#: Used only to check the calibration, never to produce a value.
HELA_VOLUME_UM3 = 2425.0
HELA_PROTEOME_RANGE = (2e9, 1e10)

#: Area a membrane protein occupies, um^2. A few nanometres across, so
#: ~25 nm^2. Used only to report how much membrane a count would need if
#: it were all surface -- which is the check that caught the error.
FOOTPRINT_UM2 = 2.5e-5


def sphere_volume_um3(radius_um: float) -> float:
    return 4.0 / 3.0 * math.pi * radius_um ** 3


@dataclass
class Calibration:
    """The scale, and everything needed to check it."""

    proteins_per_um3: float = 0.0
    anchor_volume_um3: float = 0.0
    hela_proteome: float = 0.0

    @property
    def plausible(self) -> bool:
        low, high = HELA_PROTEOME_RANGE
        return low <= self.hela_proteome <= high

    def describe(self) -> str:
        return (
            f"{self.proteins_per_um3:.3g} protein molecules per um^3, fixed "
            f"by {ANCHOR_COPIES:,.0f} {ANCHOR_PROTEIN} copies on a "
            f"{ANCHOR_CELL} of {self.anchor_volume_um3:.0f} um^3 "
            f"({ANCHOR_CITATION}) at {ANCHOR_PPM:g} ppm. CHECK: this "
            f"predicts {self.hela_proteome:.2g} proteins in a HeLa cell of "
            f"{HELA_VOLUME_UM3:g} um^3, against a literature range of "
            f"{HELA_PROTEOME_RANGE[0]:.0g}-{HELA_PROTEOME_RANGE[1]:.0g}.")


def calibrate(store) -> Calibration:
    """Protein molecules per cubic micrometre, from the anchor."""
    from .parameters import Provenance, get_parameter

    cell = store.find_entity_by_name(ANCHOR_CELL)
    if cell is None:
        raise LookupError(f"{ANCHOR_CELL} is not in this library")
    radius = get_parameter(cell, "radius_um")
    if radius is None or radius.provenance is Provenance.PLACEHOLDER:
        raise LookupError(
            f"{ANCHOR_CELL} has no evidenced radius; the anchor needs a "
            "volume and a guessed one would put the whole scale on a guess")

    volume = sphere_volume_um3(radius.value)
    density = ANCHOR_COPIES / (ANCHOR_PPM * 1e-6 * volume)
    return Calibration(proteins_per_um3=density, anchor_volume_um3=volume,
                       hela_proteome=density * HELA_VOLUME_UM3)


@dataclass
class Counted:
    receptor: str = ""
    cell: str = ""
    ppm: float = 0.0
    volume_um3: float = 0.0
    copies: float = 0.0
    measured_in: str = ""
    #: Fraction of the cell membrane this count would cover if every
    #: molecule were on the surface. Above a few per cent it cannot be.
    membrane_fraction: float = 0.0


@dataclass
class Run:
    calibration: Optional[Calibration] = None
    written: List[Counted] = field(default_factory=list)
    no_sized_cell: List[str] = field(default_factory=list)
    refused: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (f"{len(self.written)} total copy numbers written, "
                f"{len(self.no_sized_cell)} with no cell of known size, "
                f"{len(self.refused)} refused")


def sized_cells(store) -> Dict[str, float]:
    """Cell name -> volume, for cells whose radius is evidence."""
    from .parameters import Provenance, get_parameter

    out: Dict[str, float] = {}
    entities, _ = store.search(limit=10000)
    for entity in entities:
        value = get_parameter(entity, "radius_um")
        if value is not None and value.value > 0 and \
                value.provenance is not Provenance.PLACEHOLDER:
            out[entity.name] = sphere_volume_um3(value.value)
    return out


def seed(store, *, dry_run: bool = False) -> Run:
    """Write a copy number wherever abundance meets a cell of known size."""
    from .models import EntityType
    from .parameters import (ParameterError, ParameterValue, Provenance,
                             set_parameter)

    run = Run()
    run.calibration = calibrate(store)
    if not run.calibration.plausible:
        run.refused.append(
            "calibration failed its own check: " + run.calibration.describe())
        return run

    volumes = sized_cells(store)
    entities, _ = store.search(limit=10000)
    for entity in sorted((e for e in entities
                          if e.entity_type is EntityType.RECEPTOR),
                         key=lambda e: e.name):
        abundance = (getattr(entity, "sourced", None) or {}).get("abundance_ppm")
        if not isinstance(abundance, dict) or not abundance:
            continue
        usable = {c: p for c, p in abundance.items() if c in volumes}
        if not usable:
            run.no_sized_cell.append(entity.name)
            continue

        # The cell this receptor is most abundant in, so the stored value
        # describes where the molecule actually matters. Every other cell
        # travels beside it rather than being averaged away: averaging a
        # count across cells of different volumes is meaningless.
        cell = max(usable, key=lambda c: usable[c])
        ppm = float(usable[cell])
        volume = volumes[cell]
        copies = ppm * 1e-6 * run.calibration.proteins_per_um3 * volume
        measured_in = str(((getattr(entity, "sourced", None) or {})
                           .get("abundance_measured_in") or {}).get(cell, ""))
        radius = (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0)
        area = 4.0 * math.pi * radius ** 2
        fraction = copies * FOOTPRINT_UM2 / area
        row = Counted(entity.name, cell, ppm, volume, copies, measured_in,
                      fraction)
        if dry_run:
            run.written.append(row)
            continue

        try:
            set_parameter(entity, "copies_per_cell", ParameterValue(
                value=round(copies, 1), unit="molecules/cell",
                provenance=Provenance.CALIBRATED,
                citation="PaxDb+" + ANCHOR_CITATION,
                source=("PaxDb abundance scaled by one absolute count; see "
                        "cognisom/library/seed_copy_numbers.py"),
                calibrated_against=(
                    f"{ANCHOR_COPIES:,.0f} {ANCHOR_PROTEIN} copies on a "
                    f"human {ANCHOR_CELL} ({ANCHOR_CITATION})"),
                conditions=(
                    f"{ppm:g} ppm of total protein molecules in "
                    f"{cell}"
                    + (f", measured in {measured_in}" if measured_in else "")
                    + f"; cell volume {volume:.0f} um^3 from its evidenced "
                    f"radius. {run.calibration.describe()}"),
                rationale=(
                    "A ppm is a RATIO and cannot become a count without "
                    "knowing how many molecules a cell holds. The scale "
                    "comes from one absolute measurement rather than a "
                    "sourced protein density -- the obvious BioNumbers "
                    "record for that turned out to be 'number of proteins "
                    "PRODUCED per cell', a flux and not a census.\n"
                    "CALIBRATED, not MEASURED, and the weakest link is "
                    f"named: PaxDb has no {ANCHOR_CELL} dataset for "
                    f"{ANCHOR_PROTEIN}, so the anchor's ppm is the "
                    f"{ANCHOR_PPM_FROM} value as the nearest myeloid "
                    "proxy. Every count here inherits that assumption.\n"
                    f"Stored for {cell}, where this receptor is most "
                    "abundant of the cells whose size is evidence. The "
                    "per-cell-type counts are in sourced['copies_per_cell'] "
                    "-- averaging a count across cells of different volumes "
                    "would mean nothing.\n"
                    f"TOTAL cellular copies, not surface. On the membrane "
                    f"alone this many would cover {fraction:.1%} of it, "
                    + ("which is impossible, so most of this pool is "
                       "intracellular." if fraction > 0.03 else
                       "which is small enough that a large surface "
                       "fraction is at least possible.")
                    + " surface_density_per_cell is left untouched; the "
                    "surface fraction is its own measurement and this "
                    "library has it for nothing.")))
        except ParameterError as exc:
            run.refused.append(f"{entity.name}: {exc}")
            continue

        entity.sourced["copies_per_cell"] = {
            c: round(p * 1e-6 * run.calibration.proteins_per_um3 * volumes[c], 1)
            for c, p in usable.items()
        }
        store.update_entity(entity, changed_by="seed_copy_numbers")
        run.written.append(row)
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
    if run.calibration:
        print("\n " + run.calibration.describe())
        print(f"  check {'PASSED' if run.calibration.plausible else 'FAILED'}")
    if run.written:
        print(f"\n  {'receptor':<34} {'cell':<19} {'copies':>12} "
              f"{'membrane if surface':>20}")
        for row in sorted(run.written, key=lambda r: -r.copies)[:12]:
            flag = "  IMPOSSIBLE" if row.membrane_fraction > 0.03 else ""
            print(f"  {row.receptor[:33]:<34} {row.cell[:18]:<19} "
                  f"{row.copies:>12,.0f} {row.membrane_fraction:>19.1%}{flag}")
        impossible = sum(1 for r in run.written if r.membrane_fraction > 0.03)
        print(f"\n  {impossible} of {len(run.written)} could not be all "
              f"surface. These are TOTAL cellular copies.")
    for line in run.refused:
        print("  REFUSED:", line[:150])
    return 0


if __name__ == "__main__":                             # pragma: no cover
    raise SystemExit(main())
