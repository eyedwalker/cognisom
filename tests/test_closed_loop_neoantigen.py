"""
End-to-end closed-loop neoantigen presentation test (Upgrade 2).

This is the patent-evidence anchor for the USC 101 closed-loop claim.
It exercises the full chain
    MUTATION_OCCURRED  ->  PEPTIDE_GENERATED  ->  PEPTIDE_PRESENTED
        ->  CELL_KILLED_BY_TCELL
across MolecularModule, CellularModule, and ImmuneModule, and asserts
the event trace appears in causal order in the simulation event log.

The test deliberately tunes the simulation for determinism: TCR
recognition threshold is 0.0 (every match recognized), the T cell is
colocated with the cancer cell so detection / kill ranges are
immediately satisfied, mhc1 expression and costimulation are high so
the Hill kill probability is essentially 1.0, and numpy.random is
seeded so the per-step kill roll is reproducible.

The test is NOT a sensitivity study of any individual stage -- those
are covered by the per-module unit tests
(test_peptidome.py, test_mhc_loading.py, test_tcr_repertoire.py,
test_tcell_kill.py). The point of *this* test is the event trace.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the repo root is on sys.path so the engine.py / modules / core
# packages resolve when pytest is run from arbitrary cwd.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core import SimulationEngine, SimulationConfig
from core.event_bus import EventTypes
from modules.cellular_module import CellularModule
from modules.immune_module import ImmuneModule
from modules.molecular_module import MolecularModule


def _build_engine() -> SimulationEngine:
    """Construct the minimal three-module engine the test needs."""
    engine = SimulationEngine(SimulationConfig(
        dt=0.01, duration=0.5, use_gpu=False,
    ))
    engine.register_module('molecular', MolecularModule, {
        'transcription_rate': 0.0,    # no stochastic transcription noise
        'exosome_release_rate': 0.0,  # no incidental exosome events
        'mutation_rate': 0.0,         # explicit mutation only
    })
    engine.register_module('cellular', CellularModule, {
        'n_normal_cells': 0,
        'n_cancer_cells': 0,
        'hla_alleles': ['HLA-A*02:01', 'HLA-A*24:02', 'HLA-B*07:02'],
    })
    engine.register_module('immune', ImmuneModule, {
        'n_t_cells': 1, 'n_nk_cells': 0, 'n_macrophages': 0,
        # Recognition threshold 0 -> the closest TCR clone is always
        # recognized. We're testing the trace, not the binding strength.
        'tcr_recognition_threshold': 0.0,
        'tcr_repertoire_size': 200,
        'tcr_seed': 0,
        # Tumor side: high costim, no checkpoint inhibition -> Hill
        # rule produces kill_probability close to 1.0 once affinity
        # and mhc clear the EC50.
        'costimulation': 1.0,
        'checkpoint_block': 0.0,
    })
    engine.initialize()
    return engine


def _colocate(cancer_cell, immune_cell, position=(100.0, 100.0, 50.0)) -> None:
    """Put the cancer and T cell at the same position so detection
    fires on the first immune update."""
    cancer_cell.position = np.array(position, dtype=np.float32)
    immune_cell.position = np.array(position, dtype=np.float32)


def test_closed_loop_event_trace_in_causal_order():
    # Reproducible numpy randomness for the per-step kill roll inside
    # immune.update.
    np.random.seed(0)

    engine = _build_engine()
    molecular = engine.modules['molecular']
    cellular = engine.modules['cellular']
    immune = engine.modules['immune']

    # Wire cross-module dependencies. The order matters:
    # cellular -> molecular for the WT protein lookup,
    # immune -> cellular for the cancer cell population view.
    cellular.set_molecular_module(molecular)
    immune.set_cellular_module(cellular)

    # Build the one cancer cell. mhc1_expression intentionally high so
    # the kill probability lands above ~0.9 once affinity drives the
    # signal -- the patent-evidence claim is the trace, not immune
    # escape. The mhc1 field is also what gates the cellular module's
    # display visibility ( > 0.05 ).
    cancer_id = cellular.add_cell(position=[100, 100, 50], cell_type='cancer')
    cellular.cells[cancer_id].mhc1_expression = 0.9

    # The molecular module tracks per-cell sequence state via
    # CellGenomeView. We must register the cancer cell explicitly --
    # cellular.add_cell doesn't cascade into molecular.
    molecular.add_cell(cancer_id)

    # Colocate the single T cell with the cancer cell.
    t_cell_id = next(iter(immune.immune_cells))
    _colocate(cellular.cells[cancer_id], immune.immune_cells[t_cell_id])

    # ---- Drive the chain --------------------------------------------------
    # introduce_mutation emits MUTATION_OCCURRED, which the cellular
    # handler converts into PEPTIDE_GENERATED + PEPTIDE_PRESENTED. We
    # then run the engine so the T cell's update step has a chance to
    # detect, activate, and kill.
    mut = molecular.introduce_mutation(cancer_id, 'KRAS', 'G12D')
    assert mut is not None, "molecular.introduce_mutation returned None"
    assert mut.effect.aa_change == 'G12D'

    engine.run(duration=0.5)

    # ---- Trace assertion --------------------------------------------------
    log = [evt for evt, _ in engine.event_bus.event_log]

    required = [
        EventTypes.MUTATION_OCCURRED,
        EventTypes.PEPTIDE_GENERATED,
        EventTypes.PEPTIDE_PRESENTED,
        EventTypes.CELL_KILLED_BY_TCELL,
    ]

    # Each required event must appear at least once.
    for evt in required:
        assert evt in log, (
            f"closed-loop trace missing event {evt!r}; "
            f"event log: {log}"
        )

    # And in this exact relative order. We check the FIRST occurrence
    # of each so a noisy log (e.g., many PEPTIDE_PRESENTED events for
    # the same mutation across alleles) does not perturb the assertion.
    first_indices = [log.index(evt) for evt in required]
    assert first_indices == sorted(first_indices), (
        f"closed-loop trace out of order; first-index per event = "
        f"{dict(zip(required, first_indices))}"
    )


def test_closed_loop_kill_carries_provenance():
    """The CELL_KILLED_BY_TCELL event must carry enough provenance to
    reconstruct the causal chain back to the originating mutation."""
    np.random.seed(0)
    engine = _build_engine()
    molecular = engine.modules['molecular']
    cellular = engine.modules['cellular']
    immune = engine.modules['immune']
    cellular.set_molecular_module(molecular)
    immune.set_cellular_module(cellular)

    cancer_id = cellular.add_cell(position=[100, 100, 50], cell_type='cancer')
    cellular.cells[cancer_id].mhc1_expression = 0.9
    molecular.add_cell(cancer_id)
    t_cell_id = next(iter(immune.immune_cells))
    _colocate(cellular.cells[cancer_id], immune.immune_cells[t_cell_id])

    molecular.introduce_mutation(cancer_id, 'KRAS', 'G12D')
    engine.run(duration=0.5)

    kill_events = [
        data for evt, data in engine.event_bus.event_log
        if evt == EventTypes.CELL_KILLED_BY_TCELL
    ]
    assert kill_events, "no CELL_KILLED_BY_TCELL event in log"
    kill = kill_events[0]
    # Provenance fields: peptide, mutation, source gene, HLA allele,
    # TCR id, affinity. All required for the patent-evidence chain.
    for field in ('peptide', 'mutation', 'source_gene', 'hla_allele',
                  'tcr_id', 'affinity', 'cell_id'):
        assert field in kill, f"kill event missing provenance field {field!r}"
    assert kill['source_gene'] == 'KRAS'
    assert kill['mutation'] == 'G12D'
    assert 0.0 <= kill['affinity'] <= 1.0
    # The mutated G must be at the anchor position of the recognized
    # peptide -- carrying the mutation_label implies this, but verify
    # the displayed peptide actually contains the mutant residue 'D'.
    assert 'D' in kill['peptide']


def test_no_displayed_peptides_means_no_tcell_kill():
    """Negative control: without the closed loop firing, a T cell
    sitting on top of a cancer cell should not record a TCR-mediated
    kill (it has nothing to recognize). The legacy NK / macrophage
    kill paths are not present in this configuration."""
    np.random.seed(0)
    engine = _build_engine()
    cellular = engine.modules['cellular']
    immune = engine.modules['immune']
    immune.set_cellular_module(cellular)

    cancer_id = cellular.add_cell(position=[100, 100, 50], cell_type='cancer')
    cellular.cells[cancer_id].mhc1_expression = 0.9
    # NOTE: no mutation introduced -> no displayed peptides -> the
    # T cell's TCR-pMHC recognition path is unreachable.
    t_cell_id = next(iter(immune.immune_cells))
    _colocate(cellular.cells[cancer_id], immune.immune_cells[t_cell_id])

    engine.run(duration=0.5)
    log = [evt for evt, _ in engine.event_bus.event_log]
    assert EventTypes.CELL_KILLED_BY_TCELL not in log
    # And, by symmetry, the activation event should also not fire on
    # the T cell since recognition was never satisfied.
    activations = [
        data for evt, data in engine.event_bus.event_log
        if evt == EventTypes.IMMUNE_ACTIVATED
    ]
    assert not any(a.get('immune_type') == 'T_cell' for a in activations)
