"""How big a molecule actually is, from its predicted structure.

This library records ``scale_um = 0.005`` -- 50 angstrom -- for **every
protein in the catalogue**, from cytochrome c to a 1972-residue myosin
heavy chain. It is not a measurement and never was: it is one type-level
default written once and inherited by everything.

The error is largest exactly where it shows. The molecules drawn at a
cell surface, where relative size is the whole visual signal, are the
big multi-domain receptors -- and those are the ones a single 50
angstrom default is most wrong about. Perforin alone measures 128
angstrom along its longest axis.

AlphaFold DB covers the human proteome and is **CC BY 4.0**, free for
commercial use with attribution. An audit agent reported 130 structures
already cached on this machine; there were none, and there still are
none until this module fetches them.

WHY THE CONFIDENCE GATE IS NOT OPTIONAL
---------------------------------------
AlphaFold returns a per-atom pLDDT score, and a low-confidence region is
usually a genuinely disordered one -- a floppy tail that has no fixed
position in the real molecule either. Such a tail stretches a bounding
box without making the molecule bigger. Perforin measured over all
atoms is 128 angstrom; over its confident atoms, 98. A quarter of its
apparent length is a tail.

So the extent is computed over atoms at pLDDT >= 70, which is
AlphaFold's own "confident" threshold, and the fraction of atoms that
survived the cut is recorded in the value's conditions. A reader can
then see when a number rests on three-quarters of a molecule.

TWO NUMBERS, AND THE SECOND IS THE BETTER ONE
---------------------------------------------
``molecular_extent_nm`` is the longest axis -- what you need to know
whether something fits through a pore. ``radius_of_gyration_nm`` is the
root-mean-square distance of the atoms from the centre of mass, and it
is what a packing or crowding calculation wants, because it is robust to
exactly the tails the gate exists for: perforin's longest axis moves
from 12.8 to 9.8 nm when the tail is cut, and its radius of gyration
stays at 3.5 either way.

DERIVED, NOT MEASURED
---------------------
These come from a PREDICTED structure. AlphaFold is extremely good and
it is still a model, so the grade is DERIVED: computed from something,
with the something named, rather than read off an instrument.
"""
from __future__ import annotations

import logging
import math
import os
import pathlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

FILES = "https://alphafold.ebi.ac.uk/files"
API = "https://alphafold.ebi.ac.uk/api/prediction"

#: Required by CC BY 4.0, stored on every value this module writes.
ATTRIBUTION = (
    "AlphaFold Protein Structure Database (alphafold.ebi.ac.uk), CC BY 4.0. "
    "Jumper et al. Nature 2021; Varadi et al. NAR 2024.")

#: AlphaFold's own threshold for a confident prediction.
CONFIDENT_PLDDT = 70.0

#: Below this fraction of confident atoms the number is reported but not
#: written: a molecule that is mostly disordered has no single size, and
#: storing one would be a statement the structure does not support.
MIN_CONFIDENT_FRACTION = 0.50

#: Where fetched structures are kept. Gitignored like the papers: these
#: are ~350 KB each and reproducible from the accession, so they are a
#: cache rather than an asset.
CACHE = "data/alphafold"

PAUSE = 0.2


@dataclass
class Shape:
    """What one structure says about one molecule."""

    accession: str = ""
    entity: str = ""
    atoms: int = 0
    confident_atoms: int = 0
    #: Longest axis of the bounding box, in nm, over confident atoms.
    extent_nm: float = 0.0
    #: Root-mean-square distance from the centre of mass, in nm.
    radius_of_gyration_nm: float = 0.0
    #: The same two over every atom, for comparison.
    extent_all_nm: float = 0.0
    mean_plddt: float = 0.0
    note: str = ""

    @property
    def confident_fraction(self) -> float:
        return self.confident_atoms / self.atoms if self.atoms else 0.0

    @property
    def ok(self) -> bool:
        return bool(self.extent_nm) and \
            self.confident_fraction >= MIN_CONFIDENT_FRACTION

    def conditions(self) -> str:
        return (f"AlphaFold model, {self.atoms} atoms, "
                f"{self.confident_fraction:.0%} at pLDDT>={CONFIDENT_PLDDT:g} "
                f"(mean pLDDT {self.mean_plddt:.0f}); extent computed over "
                f"the confident atoms. Over all atoms it is "
                f"{self.extent_all_nm:.1f} nm.")


def _session():
    import requests

    s = requests.Session()
    # Without a User-Agent the file host returns 403. The API does not.
    s.headers.update({"User-Agent": "cognisom-library/1.0 (research use)"})
    return s


def fetch_model(accession: str, session=None,
                cache: str = CACHE) -> Optional[str]:
    """The PDB text for one accession, from the cache or the network."""
    accession = accession.split("-")[0]
    path = pathlib.Path(cache) / f"AF-{accession}.pdb"
    if path.exists() and path.stat().st_size > 1000:
        return path.read_text()

    session = session or _session()
    # v6 is current; older entries exist only at earlier versions, so
    # the API is asked rather than a URL being guessed.
    url = ""
    try:
        meta = session.get(f"{API}/{accession}", timeout=30).json()
        if isinstance(meta, list) and meta:
            url = meta[0].get("pdbUrl") or ""
    except Exception as exc:
        log.debug("alphafold api for %s: %s", accession, exc)
    if not url:
        url = f"{FILES}/AF-{accession}-F1-model_v6.pdb"
    try:
        got = session.get(url, timeout=60)
    except Exception as exc:
        log.warning("alphafold fetch failed for %s: %s", accession, exc)
        return None
    if got.status_code != 200 or len(got.content) < 1000:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(got.text)
    return got.text


def measure(pdb_text: str) -> Shape:
    """Extent and radius of gyration, confident atoms and all atoms."""
    import numpy as np

    xyz, plddt = [], []
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        try:
            xyz.append((float(line[30:38]), float(line[38:46]),
                        float(line[46:54])))
            plddt.append(float(line[60:66]))
        except ValueError:
            continue
    shape = Shape()
    if len(xyz) < 10:
        shape.note = f"only {len(xyz)} atoms parsed"
        return shape

    coords = np.asarray(xyz, dtype=float)
    scores = np.asarray(plddt, dtype=float)
    shape.atoms = int(len(coords))
    shape.mean_plddt = float(scores.mean())

    def _extent(points):
        if len(points) < 2:
            return 0.0, 0.0
        box = points.max(axis=0) - points.min(axis=0)
        centred = points - points.mean(axis=0)
        rg = float(math.sqrt((centred ** 2).sum(axis=1).mean()))
        return float(box.max()), rg

    all_axis, _all_rg = _extent(coords)
    shape.extent_all_nm = all_axis / 10.0          # angstrom -> nm

    keep = scores >= CONFIDENT_PLDDT
    shape.confident_atoms = int(keep.sum())
    axis, rg = _extent(coords[keep])
    shape.extent_nm = axis / 10.0
    shape.radius_of_gyration_nm = rg / 10.0
    return shape


@dataclass
class Run:
    measured: List[Shape] = field(default_factory=list)
    too_disordered: List[Shape] = field(default_factory=list)
    unavailable: List[str] = field(default_factory=list)
    refused: List[str] = field(default_factory=list)
    written: int = 0

    def summary(self) -> str:
        return (f"{len(self.measured)} measured, {self.written} written, "
                f"{len(self.too_disordered)} too disordered to size, "
                f"{len(self.unavailable)} with no AlphaFold model, "
                f"{len(self.refused)} refused by the parameter spec")


def candidates(store) -> List[Tuple[str, str]]:
    """``(accession, entity name)`` for everything AlphaFold can answer."""
    import json

    out = []
    for blob, in store._conn.execute("SELECT data_json FROM entities"):
        data = json.loads(blob)
        props = data.get("properties") or {}
        accession = (props.get("uniprot_id") or "").strip()
        if accession:
            out.append((accession, data["name"]))
    return out


def source(store, *, dry_run: bool = False, limit: Optional[int] = None,
           session=None) -> Run:
    """Measure and store a size for every molecule with a structure."""
    from .parameters import (ParameterError, ParameterValue,
                             Provenance, set_parameter)

    session = session or _session()
    run = Run()
    for accession, name in (candidates(store)[:limit] if limit is not None
                            else candidates(store)):
        text = fetch_model(accession, session)
        if text is None:
            run.unavailable.append(f"{name} ({accession})")
            time.sleep(PAUSE)
            continue
        shape = measure(text)
        shape.accession, shape.entity = accession, name
        if not shape.atoms:
            run.unavailable.append(f"{name}: {shape.note}")
            continue
        if not shape.ok:
            run.too_disordered.append(shape)
            continue
        run.measured.append(shape)
        if dry_run:
            continue

        entity = store.find_entity_by_name(name)
        if entity is None:
            continue
        wrote = False
        for parameter, value in (
                ("molecular_extent_nm", shape.extent_nm),
                ("radius_of_gyration_nm", shape.radius_of_gyration_nm)):
            try:
                set_parameter(entity, parameter, ParameterValue(
                    value=float(value), unit="nm",
                    provenance=Provenance.DERIVED,
                    citation=f"AlphaFoldDB:{accession}",
                    source=ATTRIBUTION,
                    conditions=shape.conditions(),
                    rationale=(
                        "Computed from the predicted structure, not read "
                        "off an instrument, which is why this is DERIVED "
                        "and not MEASURED."),
                ))
                wrote = True
            except ParameterError as exc:
                run.refused.append(f"{name} {parameter}: {exc}")
        if wrote:
            entity.sourced.setdefault("alphafold_model", f"AF-{accession}-F1")
            store.update_entity(entity, changed_by="alphafold_sourcing")
            run.written += 1
    return run


def main(argv: Sequence[str] | None = None) -> int:     # pragma: no cover
    import argparse
    import contextlib
    import io

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    from .store import EntityStore
    with contextlib.redirect_stdout(io.StringIO()):
        store = EntityStore()
    run = source(store, dry_run=args.dry_run, limit=args.limit)
    print(run.summary())
    print(f"\nattribution: {ATTRIBUTION}")
    if run.measured:
        big = sorted(run.measured, key=lambda s: -s.extent_nm)[:8]
        print("\nLARGEST (against the 0.005 um = 5 nm this library "
              "records for every protein):")
        for s in big:
            print(f"   {s.extent_nm:6.1f} nm  Rg {s.radius_of_gyration_nm:5.1f}"
                  f"  {s.confident_fraction:>4.0%} confident  {s.entity[:44]}")
    if run.too_disordered:
        print(f"\nTOO DISORDERED TO SIZE ({len(run.too_disordered)}):")
        for s in run.too_disordered[:8]:
            print(f"   {s.confident_fraction:>4.0%} confident  {s.entity[:48]}")
    if run.refused:
        print(f"\nREFUSED BY THE SPEC ({len(run.refused)}):")
        for line in run.refused[:8]:
            print("   ", line[:140])
    return 0


if __name__ == "__main__":                              # pragma: no cover
    raise SystemExit(main())
