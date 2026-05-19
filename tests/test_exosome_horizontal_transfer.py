"""
Horizontal genome transfer via exosome uptake.

When a cancer cell releases an oncogenic exosome and a normal cell
uptakes it, the cargo's mutations must be applied as sparse deltas to
the recipient's CellGenomeView (not just as a phenotype flag). This is
the load-bearing invariant for the Claim-4 dependent on
exosome-mediated transfer: daughter cells of the transformed recipient
inherit the transmitted mutation through CellGenomeView.fork(), and
MUTATION_OCCURRED engages the Claim-1 neoantigen chain for the
recipient.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core import SimulationEngine, SimulationConfig
from core.event_bus import EventTypes
from modules.cellular_module import CellularModule
from modules.molecular_module import MolecularModule


def _build_engine() -> SimulationEngine:
    """Two-module engine with all stochastic background turned off so the
    exosome path is the only source of events under test."""
    engine = SimulationEngine(SimulationConfig(
        dt=0.01, duration=0.1, use_gpu=False,
    ))
    engine.register_module('molecular', MolecularModule, {
        'transcription_rate': 0.0,
        'exosome_release_rate': 0.0,
        'mutation_rate': 0.0,
    })
    engine.register_module('cellular', CellularModule, {
        'n_normal_cells': 0,
        'n_cancer_cells': 0,
        'hla_alleles': ['HLA-A*02:01'],
    })
    engine.initialize()
    return engine


def _stage_transfer(engine):
    """Build cancer source + normal recipient, package an oncogenic
    exosome from the source, mark it uptaken by the recipient, and
    drain the staging events so subsequent assertions only see
    post-uptake events.

    Returns (source_id, recipient_id, exosome).
    """
    molecular = engine.modules['molecular']
    cellular = engine.modules['cellular']
    cellular.set_molecular_module(molecular)

    source_id = cellular.add_cell(position=[10, 10, 10], cell_type='cancer')
    molecular.add_cell(source_id)
    molecular.introduce_mutation(source_id, 'KRAS', 'G12D')

    recipient_id = cellular.add_cell(position=[20, 20, 20], cell_type='normal')
    molecular.add_cell(recipient_id)

    exosome = molecular.create_exosome(source_id, oncogenic=True)
    molecular.exosome_system.release_exosome(
        exosome, np.array([20.0, 20.0, 20.0])
    )
    molecular.exosome_system.mark_uptaken(exosome, recipient_id)

    # Drain staging events (the source's MUTATION_OCCURRED + any
    # cascaded peptide events) and clear the log so the test only
    # inspects events caused by the uptake itself.
    engine.event_bus.process_events()
    engine.event_bus.event_log.clear()

    return source_id, recipient_id, exosome


def _drive_uptake(engine):
    """Emit EXOSOME_UPTAKEN via molecular.update(dt=0) and pump the bus
    enough rounds to drain the cascade (EXOSOME_UPTAKEN -> handler ->
    MUTATION_OCCURRED -> ... )."""
    engine.modules['molecular'].update(dt=0.0)
    for _ in range(6):
        if not engine.event_bus.event_queue:
            break
        engine.event_bus.process_events()


def test_recipient_view_carries_delta_after_uptake(monkeypatch):
    """Oncogenic exosome uptake writes a SubstitutionDelta into the
    recipient's CellGenomeView. Without this, Claim 4's sparse-delta
    architecture does not apply to horizontal transfer."""
    monkeypatch.setattr(np.random, 'random', lambda: 0.0)
    engine = _build_engine()
    molecular = engine.modules['molecular']
    _source_id, recipient_id, _exo = _stage_transfer(engine)

    _drive_uptake(engine)

    recipient_view = molecular.cell_views[recipient_id]
    kras_deltas = recipient_view.deltas_for_gene('KRAS')
    assert len(kras_deltas) == 1, (
        f"recipient view should carry exactly one KRAS delta, got "
        f"{len(kras_deltas)}: {kras_deltas}"
    )
    delta = kras_deltas[0]
    assert delta.position == 34, (
        f"KRAS G12D writes position 34 (0-indexed middle base of codon "
        f"12); got {delta.position}"
    )
    assert delta.new_base == 'A'
    assert delta.mutation_id == 'G12D'


def test_recipient_emits_mutation_occurred_event(monkeypatch):
    """The horizontal-transfer wire emits MUTATION_OCCURRED for the
    recipient cell_id so the closed-loop neoantigen chain (Claim 1)
    engages for the transformed recipient."""
    monkeypatch.setattr(np.random, 'random', lambda: 0.0)
    engine = _build_engine()
    _source_id, recipient_id, _exo = _stage_transfer(engine)

    _drive_uptake(engine)

    recipient_mutation_events = [
        data for evt, data in engine.event_bus.event_log
        if evt == EventTypes.MUTATION_OCCURRED
        and data.get('cell_id') == recipient_id
    ]
    assert len(recipient_mutation_events) == 1, (
        f"expected exactly one MUTATION_OCCURRED for recipient "
        f"{recipient_id}, got {len(recipient_mutation_events)}: "
        f"{recipient_mutation_events}"
    )
    payload = recipient_mutation_events[0]
    assert payload['gene'] == 'KRAS'
    assert payload['mutation'] == 'G12D'
    assert payload['oncogenic'] is True


def test_daughter_cell_inherits_transmitted_mutation(monkeypatch):
    """After horizontal transfer, the recipient's daughter view (via
    fork()) inherits the transmitted mutation. This is what makes
    exosomes a Claim-4 dependent: the transferred substitution
    propagates through lineage via the same fork operation used for
    intrinsic mutations."""
    monkeypatch.setattr(np.random, 'random', lambda: 0.0)
    engine = _build_engine()
    molecular = engine.modules['molecular']
    _source_id, recipient_id, _exo = _stage_transfer(engine)

    _drive_uptake(engine)

    recipient_view = molecular.cell_views[recipient_id]
    daughter_view = recipient_view.fork()

    assert daughter_view.n_deltas() == 1
    daughter_kras = daughter_view.deltas_for_gene('KRAS')
    assert len(daughter_kras) == 1
    assert daughter_kras[0].new_base == 'A'
    assert daughter_kras[0].mutation_id == 'G12D'
    # Daughter shares the parent's reference object (no genome copy).
    assert daughter_view.reference is recipient_view.reference


def test_non_oncogenic_cargo_does_not_transfer_delta(monkeypatch):
    """Negative control: a non-oncogenic exosome must NOT modify the
    recipient's genome view. Guards against the wire over-firing on
    routine vesicle traffic."""
    monkeypatch.setattr(np.random, 'random', lambda: 0.0)
    engine = _build_engine()
    molecular = engine.modules['molecular']
    cellular = engine.modules['cellular']
    cellular.set_molecular_module(molecular)

    # Source has no driver mutation; package a non-oncogenic exosome.
    source_id = cellular.add_cell(position=[10, 10, 10], cell_type='normal')
    molecular.add_cell(source_id)
    # Seed an mRNA in the cell pool so create_exosome(oncogenic=False)
    # has something to package.
    view = molecular.cell_views[source_id]
    molecular.cell_mrnas[source_id].append(
        molecular._build_mrna_from_view(view, 'KRAS', source_id)
    )

    recipient_id = cellular.add_cell(position=[20, 20, 20], cell_type='normal')
    molecular.add_cell(recipient_id)

    exosome = molecular.create_exosome(source_id, oncogenic=False)
    molecular.exosome_system.release_exosome(
        exosome, np.array([20.0, 20.0, 20.0])
    )
    molecular.exosome_system.mark_uptaken(exosome, recipient_id)
    engine.event_bus.process_events()
    engine.event_bus.event_log.clear()

    _drive_uptake(engine)

    recipient_view = molecular.cell_views[recipient_id]
    assert recipient_view.n_deltas() == 0
    # And no MUTATION_OCCURRED was emitted for the recipient.
    recipient_mutations = [
        data for evt, data in engine.event_bus.event_log
        if evt == EventTypes.MUTATION_OCCURRED
        and data.get('cell_id') == recipient_id
    ]
    assert recipient_mutations == []
