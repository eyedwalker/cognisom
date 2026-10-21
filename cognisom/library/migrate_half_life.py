"""Move each stored half-life to the name that says what it is.

``half_life_hours`` was one parameter carrying four quantities. The
stored data said so plainly once it was counted by entity type:

    drug      23 values, 22 MEASURED against FDA labels
    gene      18 values, all placeholder -- and ``gene`` was not even in
              the spec's applies_to, so every one of them was a write
              the spec should have refused
    cytokine  30 values, all placeholder, every single one 4.0 hours
    protein    4 values, all placeholder

A drug's elimination half-life is a whole-body clearance measured in
plasma. A gene row's is the half-life of its TRANSCRIPT -- that is what
``reaction_builder`` has always used AR's for, as the mRNA degradation
rate ``k_ar_m_deg``. A cytokine's is survival in circulation. A
protein's is intracellular turnover by proteasome or lysosome. They are
measured by unrelated experiments and differ by orders of magnitude.

WHY THIS IS A MIGRATION AND NOT A RENAME
----------------------------------------
The destination depends on the ENTITY, not on the key, so no alias can
express it: the same name on a drug and on a gene must end up in
different places. Each value is moved to the canonical name for its
type, keeping its provenance, citation and rationale exactly as they
were -- a migration that dropped the FDA citations off 22 measured drug
half-lives would be a worse outcome than the ambiguity it fixed.

WHAT IS DELIBERATELY NOT DONE
-----------------------------
Nothing is filled in. 91 entities have no half-life of any kind and
they still have none afterwards; this moves what exists and invents
nothing. The placeholders stay placeholders, including the thirty
cytokines that all read 4.0 -- which is now visible as thirty copies of
one default rather than thirty values.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

#: entity type -> the half-life that type actually has.
DESTINATION: Dict[str, str] = {
    "drug": "elimination_half_life_hours",
    "gene": "mrna_half_life_hours",
    "cytokine": "circulating_half_life_hours",
    "ligand": "circulating_half_life_hours",
    "protein": "protein_turnover_half_life_hours",
    "receptor": "protein_turnover_half_life_hours",
    "adhesion_molecule": "protein_turnover_half_life_hours",
    "antibody": "protein_turnover_half_life_hours",
    "complement_component": "protein_turnover_half_life_hours",
    "pattern_recognition_receptor": "protein_turnover_half_life_hours",
    "mhc_molecule": "protein_turnover_half_life_hours",
}

LEGACY = "half_life_hours"


@dataclass
class Move:
    entity: str = ""
    entity_type: str = ""
    to: str = ""
    value: float = 0.0
    provenance: str = ""
    citation: str = ""


@dataclass
class Run:
    moved: List[Move] = field(default_factory=list)
    #: Types with a stored half-life and nowhere defined to put it.
    unmapped: List[str] = field(default_factory=list)
    refused: List[str] = field(default_factory=list)

    def summary(self) -> str:
        by_dest: Dict[str, int] = {}
        for move in self.moved:
            by_dest[move.to] = by_dest.get(move.to, 0) + 1
        return (f"{len(self.moved)} moved {by_dest}, "
                f"{len(self.unmapped)} unmapped, {len(self.refused)} refused")


def migrate(store, *, dry_run: bool = False) -> Run:
    """Move every stored ``half_life_hours`` to its typed name."""
    from .parameters import (ParameterError, ParameterValue, Provenance,
                             get_parameter, set_parameter)

    run = Run()
    entities, _ = store.search(limit=10000)
    for entity in entities:
        params = getattr(entity, "physics_params", None) or {}
        if LEGACY not in params:
            continue
        existing = get_parameter(entity, LEGACY)
        if existing is None:
            continue
        destination = DESTINATION.get(entity.entity_type.value)
        if destination is None:
            run.unmapped.append(f"{entity.name} ({entity.entity_type.value})")
            continue

        move = Move(entity=entity.name, entity_type=entity.entity_type.value,
                    to=destination, value=float(existing.value),
                    provenance=existing.provenance.value,
                    citation=existing.citation)
        if dry_run:
            run.moved.append(move)
            continue
        try:
            # The whole record travels, not just the number. Dropping
            # the FDA citations off 22 measured drug half-lives would be
            # a worse outcome than the ambiguity this fixes.
            set_parameter(entity, destination, ParameterValue(
                value=float(existing.value), unit=existing.unit or "h",
                provenance=existing.provenance,
                citation=existing.citation, source=existing.source,
                conditions=existing.conditions,
                rationale=existing.rationale,
                calibrated_against=existing.calibrated_against,
                confidence=existing.confidence))
        except ParameterError as exc:
            run.refused.append(f"{entity.name}: {exc}")
            continue

        params.pop(LEGACY, None)
        sidecar = params.get("_provenance")
        if isinstance(sidecar, dict):
            sidecar.pop(LEGACY, None)
        store.update_entity(entity, changed_by="migrate_half_life")
        run.moved.append(move)
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
    run = migrate(store, dry_run=args.dry_run)
    print(run.summary())

    kept = [m for m in run.moved if m.provenance == "measured"]
    print(f"\n{len(kept)} MEASURED values kept their citation:")
    for move in kept[:8]:
        print(f"   {move.entity[:30]:<32} -> {move.to:<34} "
              f"{move.value:>6} h  {move.citation[:28]}")
    if run.unmapped:
        print(f"\nNO DESTINATION ({len(run.unmapped)}):")
        for line in run.unmapped[:8]:
            print("   ", line)
    if run.refused:
        print(f"\nREFUSED BY THE SPEC ({len(run.refused)}):")
        for line in run.refused[:8]:
            print("   ", line[:130])
    return 0


if __name__ == "__main__":                             # pragma: no cover
    raise SystemExit(main())
