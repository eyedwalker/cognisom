"""Curated parameter values, kept in the repository.

Every value the sourcing modules write -- UniProt masses, ChEMBL and BRENDA
constants, openFDA pharmacokinetics, BioNumbers cell quantities -- lands in
the entity library's database, and that database is local state: the
SQLite file lives under data/, which is not in version control, and
production runs its own database. So until this module the evidence existed
on one machine only. A fresh clone, CI and the production server auto-seeded
a library with none of it, every simulation seam that reads evidence fell
back to its defaults there, and the tests that check the evidence failed in
CI while passing locally.

``curated_parameters.json`` beside this module is the repository's copy:
every parameter record whose provenance is not PLACEHOLDER, with the full
evidence trail (citation, source, conditions, rationale), plus the UniProt
identity fields the sourcing wrote. ``apply_curated`` brings any library up
to date with it, and EntityStore runs it on open whenever the file's
content has changed, so a deployed library picks up a new curation pass on
its next start. After a sourcing pass, regenerate the file with

NOT EVERY SOURCED FACT IS A PARAMETER
-------------------------------------
For a long time this module carried parameters and three identity fields
and nothing else, which quietly recreated on a smaller scale the exact
problem it was written to solve. A sourcing pass also writes to
``entity.sourced`` -- the generic slot that survives ``to_dict()`` -- and
none of it was exported: 148 PaxDb abundances, 113 AlphaFold accessions,
91 HPA subcellular locations and the leukocyte diameter convention existed
on one developer's machine. A fresh clone got the numbers and not the
record of where they came from, and twelve tests that check them were red
anywhere but there.

So the file now carries a ``sourced`` section per entity as well. The test
of what belongs in it is reproducibility: a seeded fact (a cytokine's
``signaling_pathway``, a drug's ``mechanism``) is written by a module in
this repository and any fresh library rebuilds it, so it is not exported.
A fact that came off the network cannot be rebuilt without going back to
the network, so it is.

This roughly doubles the file. That is the right trade: the alternative is
that a measured protein abundance, with the datasets it was measured in,
is lost the moment the machine that fetched it is reimaged.

    python -m cognisom.library.curated export

and commit it with the code that produced the values.

Application is additive: it writes the file's records and never deletes
one, because an export is a snapshot of one machine and a value missing
from it usually means that machine has not been sourced, not that the
value is wrong.

RETRACTION
----------
That leaves no way to take a value back. A fresh clone picks up a
correction immediately -- the bad record simply is not in the file -- but
a library that already applied it keeps it until it is rebuilt, and
production is never rebuilt. So a value found to be WRONG, as opposed to
merely absent, goes in ``RETRACTED`` below: an explicit list of
(entity, type, parameter, why) that apply_curated deletes. It is written
by hand, it is reviewed in a diff like any other code, and it is the only
thing in this module that removes data.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

CURATED_PATH = Path(__file__).resolve().parent / "curated_parameters.json"
FORMAT = 1
#: Identity fields the UniProt sourcing writes onto an entity. A mass with
#: no accession beside it cannot be re-verified, so they travel together.
#:
#: The last three were sourced and not exported: a recommended name, a
#: sequence length and the string a paper referred the protein by are each
#: the thing that lets a reader check the accession is the right protein.
IDENTITY_FIELDS = ("uniprot_id", "gene_symbol", "uniprot_resolved_by",
                   "uniprot_recommended_name", "sequence_length_aa",
                   "referenced_as")
META_KEY = "curated_parameters_sha256"

#: Values that are WRONG, not merely absent: (entity, entity_type,
#: parameter, why). apply_curated deletes each one wherever it finds it.
#: Dropping a record from the export is not enough -- application is
#: additive, so a library that already took the bad value would keep it.
#:
#: A retraction stays here permanently. Removing the entry would let the
#: value return to any library that had not yet applied this file.
RETRACTED: Tuple[Tuple[str, str, str, str], ...] = tuple(
    (name, "antibody", "molecular_weight_kda",
     "An IGH* accession translates to ONE heavy-chain constant region. "
     "Stored as the antibody's molecular weight it was wrong by 3x to "
     "19x -- IgM (pentameric) read 51.9 kDa against a real ~970. The "
     "chain mass is kept as constant_region_mass_kda; the assembled "
     "mass is not recorded, because this library has no source for it.")
    for name in ("IgA1 (secretory)", "IgD", "IgE", "IgG1", "IgG2",
                 "IgG3", "IgG4", "IgM (pentameric)")
)


@dataclass
class ApplyReport:
    digest: str = ""
    skipped: bool = False
    written: int = 0
    unchanged: int = 0
    #: Values deleted because RETRACTED names them as wrong.
    retracted: int = 0
    missing_entities: List[Tuple[str, str]] = field(default_factory=list)
    refused: List[str] = field(default_factory=list)


def _all_entities(store):
    ids = [row[0] if not isinstance(row, dict) else row["entity_id"]
           for row in store._execute("SELECT entity_id FROM entities").fetchall()]
    for entity_id in ids:
        entity = store.get_entity(entity_id)
        if entity is not None:
            yield entity


def export_curated(store, path: Path = CURATED_PATH) -> Tuple[int, int]:
    """Write every evidenced record in ``store`` to ``path``.

    Returns ``(parameter records, sourced facts)``.
    """
    from cognisom.library.parameters import Provenance, _sidecar

    entities = []
    for entity in _all_entities(store):
        records = {name: record for name, record in sorted(_sidecar(entity).items())
                   if record.get("provenance") != Provenance.PLACEHOLDER.value}
        identity = {f: getattr(entity, f) for f in IDENTITY_FIELDS
                    if getattr(entity, f, "")}
        # Everything a sourcing module fetched that is not a parameter.
        # Exported whole rather than by an allow-list of key names: the
        # slot exists precisely so a new source need not amend this
        # module, and an allow-list would silently drop the next one.
        sourced = {k: v for k, v in sorted(
            (getattr(entity, "sourced", None) or {}).items())
            if v not in (None, "", [], {})}
        if not (records or sourced):
            continue
        entities.append({
            "name": entity.name,
            "entity_type": entity.entity_type.value,
            **({"identity": identity} if identity else {}),
            "parameters": records,
            **({"sourced": sourced} if sourced else {}),
        })
    entities.sort(key=lambda e: (e["entity_type"], e["name"]))
    document = {
        "format": FORMAT,
        "about": ("Evidenced entity-library parameters, applied to any library "
                  "on open; see cognisom/library/curated.py."),
        "entities": entities,
        "retracted": [
            {"name": name, "entity_type": entity_type,
             "parameter": parameter, "why": why}
            for name, entity_type, parameter, why in RETRACTED
        ],
    }
    path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
    return (sum(len(e["parameters"]) for e in entities),
            sum(len(e.get("sourced", ())) for e in entities))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_curated(store, path: Path = CURATED_PATH,
                  force: bool = False) -> ApplyReport:
    """Bring ``store`` up to date with the curated file. Idempotent.

    Skips the work when the store records that this exact file content was
    already applied. Every record goes through set_parameter, so a record
    that no longer satisfies its spec is refused and reported rather than
    written.
    """
    from cognisom.library.parameters import (
        ParameterValue, _sidecar, set_parameter,
    )

    report = ApplyReport()
    if not path.exists():
        return report
    report.digest = _digest(path)
    if not force and store.get_meta(META_KEY) == report.digest:
        report.skipped = True
        return report

    document = json.loads(path.read_text())
    if document.get("format") != FORMAT:
        raise ValueError(f"{path} is format {document.get('format')!r}; "
                         f"this code reads format {FORMAT}")

    # Retractions first: a value that is wrong must go even if the same
    # pass then writes other records onto that entity.
    for item in document.get("retracted", []):
        entity = store.find_entity_by_name(item["name"], item["entity_type"])
        if entity is None:
            continue
        parameter = item["parameter"]
        params = getattr(entity, "physics_params", None) or {}
        sidecar = _sidecar(entity)
        if parameter not in params and parameter not in sidecar:
            continue
        params.pop(parameter, None)
        sidecar.pop(parameter, None)
        store.update_entity(entity, changed_by="curated_retraction")
        report.retracted += 1
        log.info("retracted %s.%s: %s", item["name"], parameter,
                 item.get("why", ""))

    for item in document.get("entities", []):
        entity = store.find_entity_by_name(item["name"], item["entity_type"])
        if entity is None:
            report.missing_entities.append((item["name"], item["entity_type"]))
            continue
        changed = False
        for name, value in (item.get("identity") or {}).items():
            if hasattr(entity, name) and getattr(entity, name) != value:
                setattr(entity, name, value)
                changed = True
        # Merged, not setattr'd. ``hasattr(entity, "abundance_ppm")`` is
        # False -- BioEntity has no __getattr__ and these never were
        # attributes -- so the identity loop above cannot carry them, and
        # a wholesale assignment would drop anything a later pass added.
        for key, value in (item.get("sourced") or {}).items():
            if entity.sourced.get(key) != value:
                entity.sourced[key] = value
                changed = True

        sidecar = _sidecar(entity)
        for name, record in item.get("parameters", {}).items():
            if sidecar.get(name) == record:
                report.unchanged += 1
                continue
            problems = set_parameter(entity, name, ParameterValue.from_dict(record),
                                     strict=False)
            if problems:
                report.refused.append(f"{item['name']}.{name}: {'; '.join(problems)}")
                continue
            # set_parameter stores pv.to_dict(); keep the file's exact record
            # so the next comparison is equality, not a re-serialisation.
            _sidecar(entity)[name] = record
            report.written += 1
            changed = True
        if changed:
            store.update_entity(entity, changed_by="curated_parameters")

    # Recorded only when every entity was found. A library opened while
    # another process is still seeding it sees some entities missing; marking
    # that partial pass as done would leave the library without the evidence
    # for good, so it is retried on the next open instead.
    if not report.missing_entities:
        store.set_meta(META_KEY, report.digest)
    for missing in report.missing_entities:
        log.warning("curated parameters name an entity this library lacks: %s (%s)",
                    *missing)
    for refusal in report.refused:
        log.warning("curated parameter refused: %s", refusal)
    log.info("curated parameters: %d written, %d already current, "
             "%d retracted, %d refused", report.written, report.unchanged,
             report.retracted, len(report.refused))
    return report


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    import contextlib
    import io

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("export", "apply"))
    args = parser.parse_args(argv)
    from cognisom.library.store import EntityStore
    with contextlib.redirect_stdout(io.StringIO()):
        store = EntityStore()
    if args.command == "export":
        params, sourced = export_curated(store)
        print(f"wrote {params} parameter records and {sourced} sourced "
              f"facts to {CURATED_PATH}")
    else:
        report = apply_curated(store, force=True)
        print(f"{report.written} written, {report.unchanged} current, "
              f"{len(report.refused)} refused, "
              f"{len(report.missing_entities)} entities missing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
