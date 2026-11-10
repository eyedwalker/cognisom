"""A named drug, given to a named patient, as engine configuration.

``ParameterBridge`` already knows what each library drug does, including
the case where a receptor variant turns an anti-androgen into an agonist.
But the only way to reach it was ``EngineRunner.from_scenario``, which
takes a stored ``SimulationScenario`` and no patient. The path the demo
and every page use -- ``EngineRunner(patient=...)`` -- had no drug
argument at all, so the one patient-specific drug response the engine
models could not be shown on the patient it was written for.

This module is the join: drug names plus a patient profile in, the
bridge's module configs and its notes out. It adds no pharmacology of its
own; every number comes from ``parameter_bridge``.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class Treatment:
    """What a set of drugs does to this patient, as the bridge resolved it."""

    #: Library names of the drugs, as resolved (canonical capitalisation).
    drugs: List[str]
    #: ``GENE:CHANGE`` labels the bridge was shown. Empty without a patient.
    variant_labels: List[str]
    #: module name -> parameters, ready to merge into module_configs.
    module_configs: Dict[str, Dict[str, Any]]
    #: Why a drug was NOT applied as its class says (e.g. the agonist
    #: switch). Empty means every drug acted as its class.
    notes: List[str] = field(default_factory=list)
    #: Modules the bridge configured that the caller could not apply.
    #: Filled by the consumer, so a dropped config is stated, not silent.
    unapplied_modules: List[str] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {
            "drugs": list(self.drugs),
            "notes": list(self.notes),
            "module_configs": {m: dict(p) for m, p in self.module_configs.items()},
            "unapplied_modules": list(self.unapplied_modules),
        }


def _find_drug(store, name: str):
    """Exact name first, then case-insensitively: 'enzalutamide' on a
    command line should not fail against a library that says
    'Enzalutamide'."""
    wanted = (name or "").strip()
    entity = store.find_entity_by_name(wanted, "drug")
    if entity is None and wanted:
        entity = store.find_entity_by_name(wanted.capitalize(), "drug")
    return entity


def resolve_treatment(
    drugs: Sequence[str],
    patient: Optional[Any] = None,
    store: Optional[Any] = None,
) -> Treatment:
    """Resolve ``drugs`` for ``patient`` through ``ParameterBridge``.

    A drug the library does not hold RAISES. Skipping it would simulate
    an untreated patient under a treated label, which is the silent
    no-op this codebase has been bitten by more than once.
    """
    from cognisom.library.models import SimulationScenario
    from cognisom.workflow.parameter_bridge import ParameterBridge
    from cognisom.workflow.scenario_builder import variant_labels_from_profile

    if isinstance(drugs, str):
        drugs = [drugs]
    if store is None:
        from cognisom.library.store import EntityStore
        with contextlib.redirect_stdout(io.StringIO()):
            store = EntityStore()

    entities = []
    for name in drugs:
        entity = _find_drug(store, name)
        if entity is None:
            raise ValueError(
                f"drug {name!r} is not in the entity library, so it cannot "
                f"be simulated; refusing to run the patient untreated "
                f"under a treated label")
        entities.append(entity)

    labels = variant_labels_from_profile(patient) if patient is not None else []
    scenario = SimulationScenario(
        name="treatment", display_name="treatment",
        scenario_type="drug_response",
        drug_ids=[e.entity_id for e in entities],
        tags=["drug_response"] + [f"variant:{label}" for label in labels])

    bridge = ParameterBridge(store)
    resolved = bridge.resolve_scenario(scenario, include_defaults=False)
    return Treatment(
        drugs=[e.name for e in entities],
        variant_labels=labels,
        module_configs={m: dict(p) for m, p in resolved.items() if p},
        notes=list(bridge.last_notes),
    )
