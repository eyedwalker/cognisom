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
    clonal_fractions: Optional[Dict[str, float]] = None,
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
    treatment = Treatment(
        drugs=[e.name for e in entities],
        variant_labels=labels,
        module_configs={m: dict(p) for m, p in resolved.items() if p},
        notes=list(bridge.last_notes),
    )
    _split_for_a_subclone(treatment, entities, labels, clonal_fractions,
                          store)
    return treatment


def _split_for_a_subclone(treatment, entities, labels, clonal_fractions,
                          store) -> None:
    """When the variant that makes a drug an agonist is in only SOME
    cells, the drug does two things in one tumour.

    The bridge answers for a patient: "enzalutamide is an agonist on his
    AR F877L", one activity for the whole tumour. That is the right
    answer when every cell carries the variant. When a fraction do, the
    cells without it have the wild-type receptor and are BLOCKED, as the
    drug class says. So the tumour-wide config becomes the class effect
    -- resolved by the same bridge, shown the patient without that
    variant -- and the agonist activity moves to a per-cell rule keyed on
    the mutation label.
    """
    from cognisom.library.models import SimulationScenario
    from cognisom.workflow import parameter_bridge as pb

    if not clonal_fractions:
        return
    for entity in entities:
        change = pb.anti_androgen_is_agonist(entity.name, labels)
        if change is None:
            continue
        engine_label = f"AR_{change}"
        fraction = clonal_fractions.get(engine_label)
        if fraction is None or float(fraction) >= 1.0:
            continue
        others = [label for label in labels
                  if label.upper() != f"AR:{change}"]
        scenario = SimulationScenario(
            name="treatment", display_name="treatment",
            scenario_type="drug_response",
            drug_ids=[e.entity_id for e in entities],
            tags=["drug_response"] + [f"variant:{label}" for label in others])
        class_effect = pb.ParameterBridge(store).resolve_scenario(
            scenario, include_defaults=False)
        ar = dict(class_effect.get("ar_signaling") or {})
        ar["ar_variant_activity"] = {
            "label": engine_label,
            "activity": pb.AR_AGONIST_TRANSCRIPTIONAL_ACTIVITY,
        }
        treatment.module_configs["ar_signaling"] = ar
        # The bridge's note speaks for the whole tumour ("applied as
        # activation, NOT as a blockade"), which is now false for most
        # of it. Replaced, not appended to.
        treatment.notes = [n for n in treatment.notes
                           if not (entity.name in n and "AGONIST" in n)]
        treatment.notes.append(
            f"{entity.name} is an AGONIST on AR {change}, and that variant "
            f"is in {float(fraction):.0%} of the founding cancer cells, not "
            f"all of them. Those cells are driven at "
            f"{pb.AR_AGONIST_TRANSCRIPTIONAL_ACTIVITY:.2f}x the untreated "
            f"baseline ({pb.AR_AGONIST_ACTIVITY_SOURCE}); every other cell "
            f"has the wild-type receptor and is blocked as the drug class "
            f"says")
        return
