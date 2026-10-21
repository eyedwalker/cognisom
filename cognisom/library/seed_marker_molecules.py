"""The molecules the library's own marker lists name but never held.

The library described its cells by the molecules they carry -- a Th1 by
CXCR3 and T-bet, a macrophage by CD163 and arginase-1, a fibroblast by
FAP and alpha-SMA -- and almost none of those molecules was an entity.
Of 151 names in the marker and secretion lists, 16 resolved. Deriving
synonyms doubled that to 33 and the rest was absence, not naming.

These 87 close most of it. They come from a seven-agent curation pass,
and the reason to trust them is not that seven agents agreed: it is that
an eighth re-resolved every accession against UniProt afterwards and
checked the primary gene symbol, the recommended protein name AND the
sequence length against each submission. **86 of 86 matched on all four
fields.** After this library shipped a FasL entity carrying the
accession for fatty acid synthase, that is the check that matters.

WHAT THE VERIFIER CAUGHT, AND WHAT WAS DONE ABOUT IT
----------------------------------------------------
Two things, and neither was an identity error.

CD45RA and CD45RO were both submitted as P08575 at 1306 residues. They
are isoforms of one gene, so a shared accession is not wrong -- but 1306
is isoform 1, CD45RABC, which is the displayed canonical and neither of
them. The difference between these two isoforms IS the marker: it is how
a naive T cell is told from a memory one, and it is in the library's
marker lists for exactly that reason. Corrected here against UniProt's
own isoform table to P08575-8 (isoform 6, CD45RA, 1211 aa) and P08575-4
(isoform 2, CD45RO, 1145 aa).

Twenty-four would have rendered in the wrong compartment -- nuclear
transcription factors, granule cargo and secreted chemokines drawn as
studs on the plasma membrane. That was not a fault in these records,
whose declared compartments were all correct; it was that nothing read
them. ``cell_profiles.COMPARTMENT`` was a hand-written table of marker
strings and 66% of the vocabulary was missing from it. Fixed on the
rendering side, where the fault was, and these entities now answer for
their own placement.

WHAT IS NOT MERGED
------------------
``Collagen alpha-1(I) chain (COL1A1)`` is added beside the existing
``Collagen I`` rather than replacing it. They are different objects:
collagen I is a heterotrimer of two COL1A1 chains and one COL1A2, and
``proteins.NOT_A_SINGLE_PROTEIN`` already says so. Merging a chain into
its assembly would lose that.
"""
from __future__ import annotations

import json
import logging
import pathlib
from typing import Dict, List, Sequence, Tuple

log = logging.getLogger(__name__)

#: The curated records, beside this module.
#:
#: Beside it, and not in a ``data/`` subdirectory: ``cognisom/.gitignore``
#: ignores ``data/`` anywhere beneath it, so a file there is dropped from
#: the commit without complaint and the seeder then fails on a fresh
#: clone with a missing-file error nobody can reproduce locally.
DATA = pathlib.Path(__file__).with_name("marker_molecules.json")

#: Scale by entity type, matching what the library already uses for each.
#: These are the type-level placeholders this library carries throughout;
#: writing them here does not make them measurements, and they are
#: deliberately identical to what a sibling of the same type already has
#: rather than invented per molecule.
_SCALE_UM: Dict[str, float] = {
    "receptor": 0.008, "protein": 0.005, "cytokine": 0.003,
    "adhesion_molecule": 0.01, "mhc": 0.007,
}

#: Colour by compartment, so the placement is visible in a render rather
#: than only recorded. A molecule's compartment is the one thing about
#: these 87 that the geometry actually depends on.
_COLOUR: Dict[str, List[float]] = {
    "cell_membrane": [0.30, 0.60, 0.92],
    "nucleus": [0.62, 0.40, 0.85],
    "granule": [0.90, 0.45, 0.30],
    "cytoplasm": [0.45, 0.75, 0.55],
    "secreted": [0.95, 0.78, 0.35],
}


def records() -> List[dict]:
    """The curated molecule records, as written."""
    with open(DATA, encoding="utf-8") as handle:
        return json.load(handle)


def build(record: dict):
    """One record as a library entity."""
    from .models import BioEntity, EntityType

    entity = BioEntity(name=record["entity_name"],
                       entity_type=EntityType(record["entity_type"]))
    entity.display_name = record["entity_name"]
    entity.description = record.get("description", "")
    entity.synonyms = list(record.get("synonyms") or [])
    entity.compartments = list(record.get("compartments") or [])
    entity.tags = ["marker", f"curated_{record['entity_type']}"]
    entity.source = "curated"
    entity.source_url = (
        f"https://www.uniprot.org/uniprotkb/{record['accession'].split('-')[0]}")
    entity.properties = {
        "uniprot_id": record["accession"],
        "gene_symbol": record["gene"],
        "uniprot_recommended_name": record["uniprot_name"],
        "sequence_length_aa": record["length"],
        # What the marker list calls it. Without this the entity exists
        # and the reference to it still does not resolve, which was the
        # original failure.
        "referenced_as": record["referenced_as"],
    }
    if record.get("isoform_note"):
        entity.properties["isoform_note"] = record["isoform_note"]
    entity.scale_um = _SCALE_UM.get(record["entity_type"], 0.005)
    entity.mesh_type = "sphere"
    entity.usd_prim_type = "BioProtein"
    entity.color_rgb = _COLOUR.get(
        (record.get("compartments") or ["cytoplasm"])[0], [0.6, 0.6, 0.6])
    return entity


def seed(store, *, dry_run: bool = False) -> Tuple[int, List[str]]:
    """Add every molecule not already present.

    Returns ``(added, notes)``. An entity that already exists is left
    alone: this is additive, and silently overwriting curation would be
    the worse failure. Returning the notes rather than logging them is
    deliberate -- a caller should not be able to report the count without
    also having what was skipped.
    """
    notes: List[str] = []
    added = 0
    for record in records():
        name = record["entity_name"]
        if store.find_entity_by_name(name) is not None:
            notes.append(f"{name}: already present, left alone")
            continue
        if dry_run:
            added += 1
            continue
        store.upsert_entity(build(record))
        added += 1
    return added, notes


def main(argv: Sequence[str] | None = None) -> int:
    """Seed an EXISTING database, which the automatic pass will not.

    ``_auto_seed_if_empty`` runs only on a library built from nothing, so
    a database that already exists never sees a new seeder. Every other
    curation module here carries the same entry point for the same
    reason.
    """
    import argparse
    import contextlib
    import io

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="say what would be added and write nothing")
    parser.add_argument("--verbose", action="store_true",
                        help="list what was skipped as well as what was added")
    args = parser.parse_args(argv)

    from .store import EntityStore
    with contextlib.redirect_stdout(io.StringIO()):
        store = EntityStore()

    before = len(list(store._conn.execute("SELECT 1 FROM entities")))
    added, notes = seed(store, dry_run=args.dry_run)
    skipped = [n for n in notes if "already present" in n]

    print(f"marker molecules: {len(records())} curated, "
          f"{added} {'would be added' if args.dry_run else 'added'}, "
          f"{len(skipped)} already present")
    if args.verbose:
        for note in notes:
            print("  ", note)
    if not args.dry_run:
        after = len(list(store._conn.execute("SELECT 1 FROM entities")))
        print(f"entities: {before} -> {after}")
        # Synonyms are what make "CCL19" in a marker list reach the
        # entity named "CCL19 (MIP-3beta)". Seeding without them adds
        # records nothing can find, which is the failure this closes.
        from .seed_synonyms import seed as seed_synonyms
        touched, syn, dropped = seed_synonyms(store)
        print(f"synonyms: {touched} entities given {syn} forms, "
              f"{dropped} ambiguous dropped")
    return 0


if __name__ == "__main__":                              # pragma: no cover
    raise SystemExit(main())
