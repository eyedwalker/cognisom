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

    python -m cognisom.library.curated export

and commit it with the code that produced the values.

Application is additive: it writes the file's records and never deletes
one. A value retracted from the file stays in a library that had already
applied it until that library is rebuilt.
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
IDENTITY_FIELDS = ("uniprot_id", "gene_symbol", "uniprot_resolved_by")
META_KEY = "curated_parameters_sha256"


@dataclass
class ApplyReport:
    digest: str = ""
    skipped: bool = False
    written: int = 0
    unchanged: int = 0
    missing_entities: List[Tuple[str, str]] = field(default_factory=list)
    refused: List[str] = field(default_factory=list)


def _all_entities(store):
    ids = [row[0] if not isinstance(row, dict) else row["entity_id"]
           for row in store._execute("SELECT entity_id FROM entities").fetchall()]
    for entity_id in ids:
        entity = store.get_entity(entity_id)
        if entity is not None:
            yield entity


def export_curated(store, path: Path = CURATED_PATH) -> int:
    """Write every evidenced record in ``store`` to ``path``. Returns the count."""
    from cognisom.library.parameters import Provenance, _sidecar

    entities = []
    for entity in _all_entities(store):
        records = {name: record for name, record in sorted(_sidecar(entity).items())
                   if record.get("provenance") != Provenance.PLACEHOLDER.value}
        if not records:
            continue
        identity = {f: getattr(entity, f) for f in IDENTITY_FIELDS
                    if getattr(entity, f, "")}
        entities.append({
            "name": entity.name,
            "entity_type": entity.entity_type.value,
            **({"identity": identity} if identity else {}),
            "parameters": records,
        })
    entities.sort(key=lambda e: (e["entity_type"], e["name"]))
    document = {
        "format": FORMAT,
        "about": ("Evidenced entity-library parameters, applied to any library "
                  "on open; see cognisom/library/curated.py."),
        "entities": entities,
    }
    path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
    return sum(len(e["parameters"]) for e in entities)


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
        sidecar = _sidecar(entity)
        for name, record in item["parameters"].items():
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
    log.info("curated parameters: %d written, %d already current, %d refused",
             report.written, report.unchanged, len(report.refused))
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
        print(f"wrote {export_curated(store)} records to {CURATED_PATH}")
    else:
        report = apply_curated(store, force=True)
        print(f"{report.written} written, {report.unchanged} current, "
              f"{len(report.refused)} refused, "
              f"{len(report.missing_entities)} entities missing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
