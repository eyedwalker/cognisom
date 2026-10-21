"""Where a protein actually is, from the Human Protein Atlas.

``cell_profiles.COMPARTMENT`` is a hand-written table of marker strings.
It was written by one person from memory, it covers 60 of the 173 names
this library uses, and everything it misses falls through to the plasma
membrane -- which put nuclear transcription factors, granule cargo and
cytoskeleton on the cell surface until the library's own
``compartments`` field was wired in behind it.

That field is better than the table and it is still an assertion. HPA is
the measurement: antibody-based immunofluorescence against a reference
cell line panel, 13,000-odd proteins, each location carrying a
reliability grade the atlas assigns from independent antibody agreement
and RNA support.

LICENCE
-------
**CC BY 4.0** -- free for commercial use with attribution. Any product
surface showing a location sourced here must credit the Human Protein
Atlas (proteinatlas.org). That is a condition of use, not a courtesy,
and it is why this module records ``hpa`` as the source on every value
it writes.

Of the four morphology and localisation sources considered, this is the
only one that is both reachable and commercially usable: Allen Cell is
explicitly non-commercial, the Cell Image Library is CC BY-NC-SA,
CellMarker grants no licence at all, and PanglaoDB does not answer.

WHAT HPA ANSWERS, AND WHAT IT DOES NOT
--------------------------------------
HPA reports where a protein was SEEN. For most proteins that is where it
works. For a secreted one it is where it was caught in transit:
``CXCL8`` -- interleukin-8, a chemokine whose entire function happens
outside the cell -- reads **Golgi apparatus**, because that is where an
antibody finds it on its way out.

So a naive mapping of HPA terms onto render compartments puts every
chemokine in the Golgi and every interleukin with it. This module
refuses that: an entity the library already records as secreted keeps
``secreted``, and HPA's Golgi/ER/vesicle reading is recorded as
CONFIRMING the secretory route rather than replacing the destination.

Equally, it does not overwrite a curated placement. The hand table is
small and hand-checked; where it and HPA disagree the disagreement is
REPORTED, because one of the two is wrong and finding out which is work
a person should do deliberately.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

API = "https://www.proteinatlas.org/api/search_download.php"

#: Attribution required by CC BY 4.0, stored on every sourced value.
ATTRIBUTION = ("Human Protein Atlas (proteinatlas.org), CC BY 4.0. "
               "Antibody-based immunofluorescence, subcellular main location.")

#: Seconds between requests. HPA publishes no documented rate limit, so
#: this is deliberate politeness rather than a measured ceiling.
PAUSE = 0.15

#: HPA's main-location vocabulary -> the bucket the renderer draws in.
#:
#: Derived from what HPA actually returns for THIS library's proteins
#: rather than from its full published list: 14 distinct terms across a
#: 45-protein sample. Terms not here are left unmapped and reported, so
#: the gap is visible instead of defaulting to the membrane, which is
#: the failure the hand table had.
BUCKET_FOR_HPA: Dict[str, str] = {
    # the plasma membrane and structures in it
    "Plasma membrane": "membrane",
    "Cell Junctions": "membrane",
    "Focal adhesion sites": "membrane",
    # the nucleus and its bodies
    "Nucleoplasm": "nucleus", "Nuclear bodies": "nucleus",
    "Nucleoli": "nucleus", "Nucleoli fibrillar center": "nucleus",
    "Nucleoli rim": "nucleus", "Nuclear speckles": "nucleus",
    "Nuclear membrane": "nucleus", "Mitotic spindle": "nucleus",
    "Mitotic chromosome": "nucleus", "Kinetochore": "nucleus",
    # stored cargo
    "Vesicles": "granule", "Lipid droplets": "granule",
    # the interior: cytosol, cytoskeleton, and the organelles in it
    "Cytosol": "cytoplasm", "Mitochondria": "cytoplasm",
    "Centrosome": "cytoplasm", "Basal body": "cytoplasm",
    "Calyx": "cytoplasm", "Intermediate filaments": "cytoplasm",
    "Actin filaments": "cytoplasm", "Microtubules": "cytoplasm",
    "Microtubule ends": "cytoplasm", "Cytokinetic bridge": "cytoplasm",
    "Midbody": "cytoplasm", "Midbody ring": "cytoplasm",
    "Endoplasmic reticulum": "cytoplasm", "Golgi apparatus": "cytoplasm",
    "Peroxisomes": "cytoplasm", "Lysosomes": "cytoplasm",
    "Endosomes": "cytoplasm", "Cytoplasmic bodies": "cytoplasm",
    "Aggresome": "cytoplasm", "Rods & Rings": "cytoplasm",
    "Cleavage furrow": "cytoplasm", "Cell Junctions ": "membrane",
}

#: HPA terms that are the SECRETORY ROUTE rather than a destination.
#: Seeing a secreted protein here confirms it is secreted; it does not
#: mean the protein lives there.
SECRETORY_TRANSIT = frozenset({
    "Golgi apparatus", "Endoplasmic reticulum", "Vesicles",
})


@dataclass
class Location:
    """What HPA says about one protein."""

    gene: str = ""
    accession: str = ""
    entity: str = ""
    hpa_terms: Tuple[str, ...] = ()
    #: The bucket those terms resolve to, or "" when none of them map.
    bucket: str = ""
    #: Why no bucket, or why one was not taken.
    note: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.bucket)


def _session():
    import requests

    s = requests.Session()
    s.headers.update({"User-Agent": "cognisom-library/1.0 (research use)"})
    return s


def fetch(gene: str, accession: str, session=None) -> Location:
    """HPA's main subcellular location for one protein.

    Matched on ACCESSION, not on the gene name HPA happened to return
    first. A gene search returns several rows -- a query for CD8A also
    returns CRTAM -- and taking the first is how a library ends up
    recording one protein's location against another's name.
    """
    session = session or _session()
    out = Location(gene=gene, accession=accession)
    url = (f"{API}?search={urllib.parse.quote(gene)}"
           "&format=json&columns=g,up,scml,scl&compress=no")
    try:
        rows = session.get(url, timeout=30).json()
    except Exception as exc:
        out.note = f"request failed: {exc}"
        return out
    if not isinstance(rows, list):
        out.note = "unexpected response shape"
        return out

    row = next((r for r in rows
                if accession in (r.get("Uniprot") or [])), None)
    if row is None:
        out.note = (f"HPA returned {len(rows)} rows for {gene!r} and none "
                    f"carried {accession}")
        return out

    terms = tuple(row.get("Subcellular main location") or ())
    out.hpa_terms = terms
    if not terms:
        out.note = "no subcellular main location recorded"
        return out

    buckets = [BUCKET_FOR_HPA[t] for t in terms if t in BUCKET_FOR_HPA]
    unmapped = [t for t in terms if t not in BUCKET_FOR_HPA]
    if not buckets:
        out.note = f"no term maps to a bucket: {list(terms)}"
        return out
    if unmapped:
        out.note = f"unmapped terms ignored: {unmapped}"

    # Same precedence the library's own compartment reader uses.
    from cognisom.visualization.cell_profiles import BUCKET_PRECEDENCE

    for bucket in BUCKET_PRECEDENCE:
        if bucket in buckets:
            out.bucket = bucket
            break
    return out


@dataclass
class Run:
    """What one pass over the library found."""

    located: List[Location] = field(default_factory=list)
    no_location: List[Location] = field(default_factory=list)
    unmatched: List[Location] = field(default_factory=list)
    #: Secreted entities whose HPA reading is the transit compartment.
    transit: List[str] = field(default_factory=list)
    #: HPA and the library's curated placement disagree. Reported, never
    #: silently resolved: one of the two is wrong.
    conflicts: List[str] = field(default_factory=list)
    written: int = 0

    def summary(self) -> str:
        return (f"{len(self.located)} located, {self.written} written, "
                f"{len(self.no_location)} with no HPA location, "
                f"{len(self.unmatched)} unmatched, "
                f"{len(self.transit)} secretory-transit, "
                f"{len(self.conflicts)} conflicts")


def candidates(store) -> List[Tuple[str, str, str]]:
    """``(gene, accession, entity name)`` for everything HPA can answer."""
    out = []
    for blob, in store._conn.execute("SELECT data_json FROM entities"):
        data = json.loads(blob)
        props = data.get("properties") or {}
        gene = (props.get("gene_symbol") or "").strip()
        accession = (props.get("uniprot_id") or "").strip().split("-")[0]
        if gene and accession:
            out.append((gene, accession, data["name"]))
    return out


def source(store, *, dry_run: bool = False, limit: Optional[int] = None,
           session=None) -> Run:
    """Record HPA's location on every entity it can answer for."""
    from cognisom.visualization.cell_profiles import (
        bucket_for_compartments, compartment_of)

    session = session or _session()
    run = Run()
    todo = candidates(store)
    if limit is not None:
        todo = todo[:limit]

    for gene, accession, name in todo:
        found = fetch(gene, accession, session)
        found.entity = name
        time.sleep(PAUSE)
        if found.hpa_terms and not found.bucket:
            run.no_location.append(found)
            continue
        if not found.hpa_terms:
            (run.unmatched if "none carried" in found.note
             else run.no_location).append(found)
            continue

        entity = store.find_entity_by_name(name)
        if entity is None:
            continue
        declared = bucket_for_compartments(
            getattr(entity, "compartments", None) or ())

        # A secreted protein seen in the Golgi is a secreted protein on
        # its way out, not a Golgi protein.
        if declared == "secreted" and set(found.hpa_terms) & SECRETORY_TRANSIT:
            run.transit.append(
                f"{name}: HPA {list(found.hpa_terms)} is the secretory route; "
                "kept as secreted")
            run.located.append(found)
            continue

        curated = compartment_of(name)
        if curated and curated != found.bucket:
            run.conflicts.append(
                f"{name}: curated {curated}, HPA {found.bucket} "
                f"({list(found.hpa_terms)})")
            run.located.append(found)
            continue

        run.located.append(found)
        if dry_run:
            continue
        # Into `sourced`, which always survives to_dict(). A loose
        # setattr does not: _extra_properties is a fixed key set per
        # subclass and silently drops anything it does not name, which
        # is how the first run of this module wrote 91 locations and
        # stored none of them.
        #
        # Beside the declared compartment rather than replacing it, so a
        # reader can see both and tell which one is measured.
        entity.sourced.update({
            "hpa_location": ", ".join(found.hpa_terms),
            "hpa_bucket": found.bucket,
            "hpa_source": ATTRIBUTION,
        })
        store.update_entity(entity, changed_by="hpa_sourcing")
        run.written += 1
    return run


def main(argv: Sequence[str] | None = None) -> int:    # pragma: no cover
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
    print(f"\nattribution required on any surface showing these: {ATTRIBUTION}")
    if run.conflicts:
        print(f"\nCONFLICTS ({len(run.conflicts)}) -- curated vs HPA, "
              "neither overwritten:")
        for line in run.conflicts:
            print("   ", line)
    if run.transit:
        print(f"\nSECRETORY TRANSIT ({len(run.transit)}):")
        for line in run.transit[:10]:
            print("   ", line)
    if run.unmatched:
        print(f"\nACCESSION NOT IN HPA ({len(run.unmatched)}):")
        for found in run.unmatched[:10]:
            print(f"    {found.entity}: {found.note}")
    return 0


if __name__ == "__main__":                             # pragma: no cover
    raise SystemExit(main())
