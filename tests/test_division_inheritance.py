"""Daughter cells inherit the state that makes them visible to T cells.

Regression cover for a silent immune-escape bias: ``_divide_cell``
used to copy only ``mutations`` and ``mhc1_expression`` onto the
daughter, dropping ``mhc1_displayed_peptides``. Since
``ImmuneModule._recognize`` gates the TCR-pMHC path on that list being
non-empty -- and MUTATION_OCCURRED never fires again for a mutation
the daughter merely inherited -- every daughter of a neoantigen-
displaying tumor cell was born invisible to T cells and stayed that
way. Nothing crashed; the tumor just escaped a little more with every
division.

These tests pin the daughter's recognizability, not the copy itself,
so a future rewrite that regenerates presentations from the forked
genome instead of copying the list still passes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cognisom.core import SimulationEngine, SimulationConfig
from cognisom.core.event_bus import EventTypes
from cognisom.modules.cellular_module import CellularModule
from cognisom.modules.immune_module import ImmuneModule
from cognisom.modules.molecular_module import MolecularModule


def _build_engine() -> SimulationEngine:
    """Three-module engine tuned for determinism, mirroring
    test_closed_loop_neoantigen: no stochastic transcription, mutation
    or exosome noise, and a TCR recognition threshold of 0 so the
    closest clone always recognizes."""
    engine = SimulationEngine(SimulationConfig(
        dt=0.01, duration=0.5, use_gpu=False,
    ))
    engine.register_module('molecular', MolecularModule, {
        'transcription_rate': 0.0,
        'exosome_release_rate': 0.0,
        'mutation_rate': 0.0,
    })
    engine.register_module('cellular', CellularModule, {
        'n_normal_cells': 0,
        'n_cancer_cells': 0,
        'hla_alleles': ['HLA-A*02:01', 'HLA-A*24:02', 'HLA-B*07:02'],
    })
    engine.register_module('immune', ImmuneModule, {
        'n_t_cells': 1, 'n_nk_cells': 0, 'n_macrophages': 0,
        'tcr_recognition_threshold': 0.0,
        'tcr_repertoire_size': 200,
        'tcr_seed': 0,
        'costimulation': 1.0,
        'checkpoint_block': 0.0,
    })
    engine.initialize()
    return engine


def _displaying_cancer_cell(engine):
    """Create one cancer cell and drive the closed loop far enough that
    it displays at least one neoantigen. Returns its cell id."""
    molecular = engine.modules['molecular']
    cellular = engine.modules['cellular']
    immune = engine.modules['immune']
    cellular.set_molecular_module(molecular)
    immune.set_cellular_module(cellular)

    cancer_id = cellular.add_cell(position=[100, 100, 50], cell_type='cancer')
    cellular.cells[cancer_id].mhc1_expression = 0.9
    molecular.add_cell(cancer_id)

    mut = molecular.introduce_mutation(cancer_id, 'KRAS', 'G12D')
    assert mut is not None
    # The bus is queued, not synchronous: MUTATION_OCCURRED only
    # reaches the cellular handler when the queue is drained.
    engine.event_bus.process_events()
    assert cellular.cells[cancer_id].mhc1_displayed_peptides, (
        "precondition failed: parent is not displaying any neoantigen"
    )
    return cancer_id


def test_daughter_of_displaying_cell_is_recognized_by_tcell():
    """The load-bearing assertion: a T cell that recognizes the parent
    must also recognize the daughter."""
    np.random.seed(0)
    engine = _build_engine()
    cellular = engine.modules['cellular']
    immune = engine.modules['immune']

    parent_id = _displaying_cancer_cell(engine)
    t_cell = immune.immune_cells[next(iter(immune.immune_cells))]

    recognized_parent, _ = immune._recognize(t_cell, cellular.cells[parent_id])
    assert recognized_parent, (
        "precondition failed: the T cell does not recognize the parent, "
        "so the daughter assertion below would prove nothing"
    )

    cellular._divide_cell(parent_id)
    daughter_id = max(cellular.cells)
    assert daughter_id != parent_id
    daughter = cellular.cells[daughter_id]

    recognized_daughter, match = immune._recognize(t_cell, daughter)
    assert recognized_daughter, (
        "daughter of a neoantigen-displaying tumor cell is invisible to "
        "T cells -- division dropped mhc1_displayed_peptides"
    )
    assert match is not None
    assert match.presentation.peptide.source_gene == 'KRAS'


def test_daughter_is_killable_end_to_end():
    """The same thing through the engine rather than through
    _recognize directly: remove the parent, leave only the daughter in
    front of the T cell, and require a TCR-mediated kill."""
    np.random.seed(0)
    engine = _build_engine()
    cellular = engine.modules['cellular']
    immune = engine.modules['immune']

    parent_id = _displaying_cancer_cell(engine)
    cellular._divide_cell(parent_id)
    daughter_id = max(cellular.cells)

    # The daughter is the only remaining target, so a
    # CELL_KILLED_BY_TCELL event can only be about her.
    cellular.remove_cell(parent_id)
    daughter = cellular.cells[daughter_id]
    t_cell = immune.immune_cells[next(iter(immune.immune_cells))]
    daughter.position = np.array([100, 100, 50], dtype=np.float32)
    t_cell.position = np.array([100, 100, 50], dtype=np.float32)
    # No vessels here, so the daughter is anoxic within two hours and dies
    # of it at a rate per hour -- measured: once with her lethal contact
    # already under way. The subject is whether she is killable.
    cellular.anoxic_death_per_hour = 0.0

    # A kill is now a contact with a sourced duration -- an exponential
    # with a 60 min mean when lethal, 15 min when not -- and a per-contact
    # probability, so a T cell may need several encounters. "Killable"
    # means it happens; it does not mean within half an hour. Run in
    # 2 h pieces until the event appears, up to a day.
    kills = []
    for _ in range(12):
        engine.run(duration=2.0)
        kills = [
            data for evt, data in engine.event_bus.event_log
            if evt == EventTypes.CELL_KILLED_BY_TCELL
        ]
        if kills:
            break
    assert kills, "the daughter was never killed by the T cell in 24 h"
    assert kills[0]['cell_id'] == daughter_id
    assert kills[0]['source_gene'] == 'KRAS'


def test_daughter_peptides_are_a_copy_not_an_alias():
    """Inheritance must not alias the parent's list -- a later mutation
    in one clone would otherwise appear in the other, which would
    silently undo CellGenomeView.fork()'s divergence guarantee."""
    np.random.seed(0)
    engine = _build_engine()
    cellular = engine.modules['cellular']

    parent_id = _displaying_cancer_cell(engine)
    parent = cellular.cells[parent_id]
    cellular._divide_cell(parent_id)
    daughter = cellular.cells[max(cellular.cells)]

    assert daughter.mhc1_displayed_peptides == parent.mhc1_displayed_peptides
    assert (daughter.mhc1_displayed_peptides
            is not parent.mhc1_displayed_peptides)

    n_before = len(daughter.mhc1_displayed_peptides)
    parent.mhc1_displayed_peptides.append(parent.mhc1_displayed_peptides[0])
    assert len(daughter.mhc1_displayed_peptides) == n_before


def test_daughter_inherits_pdl1_and_resamples_ecm():
    """PD-L1 is inherited (signalling state carried into the same
    cytokine field); ECM is re-sampled at the daughter's position
    rather than re-seeded from the module baseline."""
    np.random.seed(0)
    engine = _build_engine()
    cellular = engine.modules['cellular']
    # A stromal baseline that add_cell would apply to any new cell, so
    # a re-seeded daughter is distinguishable from a re-sampled one.
    cellular.cancer_baseline_ecm = 0.2

    parent_id = cellular.add_cell(position=[100, 100, 50], cell_type='cancer')
    parent = cellular.cells[parent_id]
    parent.pdl1_expression = 0.7
    # Parent has deposited well past the spawn baseline.
    parent.local_ecm_density = 0.8

    cellular._divide_cell(parent_id)
    daughter = cellular.cells[max(cellular.cells)]

    assert daughter.pdl1_expression == parent.pdl1_expression

    # The daughter lands ~5 um away, inside the 30 um sampling radius,
    # where the only cancer cell is the parent at 0.8 -- so it sees the
    # local stroma, not the 0.2 spawn baseline.
    assert daughter.local_ecm_density == 0.8
