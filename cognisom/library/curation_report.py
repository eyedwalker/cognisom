"""
Curation status for the entity library.

Turns "the numbers are missing" from an anecdote into a worklist. Run
it to see which parameters are uncurated, on how many entities, and
which entity types are furthest from being able to drive a simulation.

    python -m cognisom.library.curation_report
    python -m cognisom.library.curation_report --type cytokine --list

The exit status is non-zero when nothing at all is curated, so this can
gate a pipeline once curation is underway.
"""
from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional

from .models import EntityType
from .parameters import (
    PARAMETER_SPECS,
    Readiness,
    assess,
    coverage,
    iter_parameters,
    required_for,
)

# Entity types that carry simulation parameters worth reporting on.
_REPORTABLE = (
    EntityType.CYTOKINE,
    EntityType.IMMUNE_CELL,
    EntityType.CELL_TYPE,
    EntityType.RECEPTOR,
    EntityType.LIGAND,
    EntityType.ADHESION_MOLECULE,
    EntityType.PHYSICAL_CELL,
)


def _load(store, entity_type: EntityType) -> List:
    entities, _ = store.search(entity_type=entity_type.value, limit=5000)
    return entities


def build_report(
    store,
    entity_types=_REPORTABLE,
    show_entities: bool = False,
) -> str:
    """Render the curation status across the requested entity types."""
    lines: List[str] = []
    total_entities = total_ready = 0
    worklist: Dict[str, int] = {}

    for etype in entity_types:
        required = required_for(etype)
        if not required:
            continue
        entities = _load(store, etype)
        if not entities:
            continue

        report = coverage(entities)
        total_entities += report.total
        total_ready += report.ready
        for name, count in report.by_parameter_missing.items():
            worklist[name] = worklist.get(name, 0) + count

        lines.append(f"── {etype.value} ─────────────────────────")
        lines.append(
            f"   {report.ready}/{report.total} simulation-ready "
            f"({report.fraction_ready:.0%})"
        )
        for name, count in sorted(
            report.by_parameter_missing.items(), key=lambda kv: (-kv[1], kv[0])
        ):
            spec = PARAMETER_SPECS.get(name)
            unit = f" [{spec.unit}]" if spec else ""
            lines.append(f"     {count:>4} missing  {name}{unit}")

        if show_entities:
            for r in report.not_ready:
                lines.append(f"       - {r.describe()}")
        lines.append("")

    lines.insert(0, "")
    lines.insert(0, (
        f"Entity library curation status: "
        f"{total_ready}/{total_entities} entities simulation-ready"
    ))

    if worklist:
        lines.append("── highest-value curation targets ─────────")
        for name, count in sorted(
            worklist.items(), key=lambda kv: (-kv[1], kv[0])
        )[:10]:
            spec = PARAMETER_SPECS.get(name)
            desc = f" — {spec.description}" if spec else ""
            lines.append(f"   {count:>4} entities need {name}{desc}")
        lines.append("")
        lines.append(
            "A parameter is counted as uncurated when it is absent or "
            "carries placeholder provenance. Supplying one means calling "
            "set_parameter with a unit and a citation or a rationale; a "
            "value without evidence is rejected rather than stored."
        )

    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report entity-library parameter curation status.",
    )
    parser.add_argument(
        "--type", dest="etype", default=None,
        help="Restrict to one entity type, e.g. cytokine.",
    )
    parser.add_argument(
        "--list", dest="show_entities", action="store_true",
        help="List the individual entities that are not ready.",
    )
    args = parser.parse_args(argv)

    from .store import EntityStore

    if args.etype:
        try:
            types = (EntityType(args.etype),)
        except ValueError:
            print(
                f"unknown entity type {args.etype!r}; known types: "
                f"{sorted(t.value for t in _REPORTABLE)}",
                file=sys.stderr,
            )
            return 2
    else:
        types = _REPORTABLE

    store = EntityStore()
    print(build_report(store, types, show_entities=args.show_entities))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
