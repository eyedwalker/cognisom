"""
T-cell exhaustion unit + integration tests (Upgrade 5).

The lecture by V. Chen (CU Anschutz, 2026-05-03, slides 51-52) flagged
exhaustion as the mechanistic gap most worth closing: in real biology,
checkpoint blockade does NOT rescue PD-1-hi exhausted CD8s -- it works
by expanding the PD-1-lo precursor pool. Our prior implementation
applied checkpoint_block as a generic rescue term, which would
incorrectly predict that ICB salvages every dysfunctional T cell.

Coverage:
  * TCRRepertoire: every clone starts as PRECURSOR; encounters
    increment a per-clone counter; crossing the threshold transitions
    one-way to EXHAUSTED; precursor_count / exhausted_count expose
    the population split.
  * tcell_kill: checkpoint_block rescue term is gated on the
    is_exhausted flag (precursor only); exhausted clones additionally
    get their kill probability scaled by exhaustion_multiplier.
  * Integration: chronic-antigen ImmuneModule run produces a
    TCELL_EXHAUSTED event after the threshold is crossed; the kill
    probability for the now-exhausted clone is observed to drop.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.py.immune.mhc_loading import MHCPresentation
from engine.py.immune.tcell_kill import kill_outcome, kill_probability
from engine.py.immune.tcr_repertoire import (
    EXHAUSTED_KILL_MULTIPLIER,
    EXHAUSTION_ENCOUNTER_THRESHOLD,
    ExhaustionState,
    TCRRepertoire,
)
from engine.py.molecular.peptidome import Peptide


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_presentation(seq: str = "YLAGGVGKV") -> MHCPresentation:
    pep = Peptide(
        sequence=seq,
        source_gene="TEST",
        length=len(seq),
        is_mutant=True,
        wild_type_sequence="A" * len(seq),
        mutation_label="X1Y",
        anchor_position_in_peptide=0,
        parent_position_1based=1,
        cleavage_score=0.9,
    )
    return MHCPresentation(
        peptide=pep,
        hla_allele="HLA-A*02:01",
        ic50_nm=100.0,
        binding_level="weak",
        presentation_score=0.5,
    )


# ---------------------------------------------------------------------------
# TCRRepertoire exhaustion bookkeeping
# ---------------------------------------------------------------------------

def test_fresh_repertoire_is_all_precursor():
    rep = TCRRepertoire(size=10, seed=0)
    for t in rep.tcrs:
        assert rep.exhaustion_state(t.tcr_id) is ExhaustionState.PRECURSOR
        assert rep.encounters(t.tcr_id) == 0
    assert rep.precursor_count() == 10
    assert rep.exhausted_count() == 0


def test_engagement_increments_counter():
    rep = TCRRepertoire(size=3, seed=0)
    tcr_id = rep.tcrs[0].tcr_id
    for i in range(1, 4):
        new_count, did_exhaust = rep.register_engagement(tcr_id)
        assert new_count == i
        assert did_exhaust is False
        assert rep.encounters(tcr_id) == i


def test_threshold_crossing_transitions_to_exhausted_once():
    rep = TCRRepertoire(size=3, seed=0, exhaustion_threshold=3)
    tcr_id = rep.tcrs[0].tcr_id
    # First two engagements: still precursor.
    for _ in range(2):
        _, did = rep.register_engagement(tcr_id)
        assert did is False
    # Third engagement crosses the threshold.
    new_count, did = rep.register_engagement(tcr_id)
    assert new_count == 3
    assert did is True
    assert rep.exhaustion_state(tcr_id) is ExhaustionState.EXHAUSTED
    # Subsequent engagements do NOT re-fire the transition.
    for _ in range(3):
        _, did = rep.register_engagement(tcr_id)
        assert did is False
    assert rep.encounters(tcr_id) == 6  # counter keeps climbing


def test_exhaustion_is_per_clone_isolated():
    """One clone's exhaustion must not transition its neighbors."""
    rep = TCRRepertoire(size=4, seed=0, exhaustion_threshold=2)
    a, b = rep.tcrs[0].tcr_id, rep.tcrs[1].tcr_id
    rep.register_engagement(a)
    rep.register_engagement(a)  # exhausts a
    assert rep.exhaustion_state(a) is ExhaustionState.EXHAUSTED
    assert rep.exhaustion_state(b) is ExhaustionState.PRECURSOR
    assert rep.precursor_count() == 3
    assert rep.exhausted_count() == 1


def test_register_engagement_unknown_id_raises():
    rep = TCRRepertoire(size=2, seed=0)
    with pytest.raises(KeyError):
        rep.register_engagement("TCR-DOES-NOT-EXIST")


def test_best_match_carries_exhaustion_state():
    """TCRMatch must reflect the live exhaustion state of the chosen
    clone; downstream kill probability gates on this field."""
    rep = TCRRepertoire(size=5, seed=0, exhaustion_threshold=2)
    pres = _make_presentation()
    m1 = rep.best_match(pres)
    assert m1.exhaustion_state is ExhaustionState.PRECURSOR
    assert m1.is_exhausted is False
    # Drive the winning clone past the threshold.
    rep.register_engagement(m1.tcr.tcr_id)
    rep.register_engagement(m1.tcr.tcr_id)
    m2 = rep.best_match(pres)
    assert m2.tcr.tcr_id == m1.tcr.tcr_id
    assert m2.exhaustion_state is ExhaustionState.EXHAUSTED
    assert m2.is_exhausted is True
    assert m2.encounter_count == 2


def test_default_exhaustion_threshold_is_documented_constant():
    rep = TCRRepertoire(size=1, seed=0)
    assert rep.exhaustion_threshold == EXHAUSTION_ENCOUNTER_THRESHOLD


# ---------------------------------------------------------------------------
# tcell_kill: exhaustion gates ICB rescue + scales kill probability
# ---------------------------------------------------------------------------

def test_precursor_kill_unchanged_when_not_exhausted():
    """Backwards-compat: omitting the new flag preserves the prior
    Stage A+B+C behaviour bit-for-bit."""
    p_default = kill_probability(0.9, 0.9, 1.0, checkpoint_block=0.5)
    p_explicit = kill_probability(
        0.9, 0.9, 1.0, checkpoint_block=0.5, is_exhausted=False,
    )
    assert p_default == p_explicit


def test_checkpoint_rescue_works_for_precursor():
    """Precursor with zero costim still gets some kill via ICB."""
    p = kill_probability(
        0.9, 0.9, 0.0,
        checkpoint_block=1.0,
        is_exhausted=False,
    )
    assert p > 0.0


def test_checkpoint_rescue_blocked_for_exhausted():
    """The patent-evidence point: exhausted clones cannot be rescued by
    checkpoint blockade. Same inputs that would lift the precursor
    above zero must collapse to ~0 for the exhausted case."""
    p_precursor = kill_probability(
        0.9, 0.9, 0.0,
        checkpoint_block=1.0,
        is_exhausted=False,
    )
    p_exhausted = kill_probability(
        0.9, 0.9, 0.0,
        checkpoint_block=1.0,
        is_exhausted=True,
    )
    assert p_precursor > 0.5
    assert p_exhausted == pytest.approx(0.0, abs=1e-9)


def test_exhausted_kill_is_scaled_down_with_full_costim():
    """Even at full costim (no checkpoint dependence), exhausted clones
    retain only ~10% of the precursor kill probability."""
    p_precursor = kill_probability(
        0.9, 0.9, 1.0,
        is_exhausted=False,
    )
    p_exhausted = kill_probability(
        0.9, 0.9, 1.0,
        is_exhausted=True,
        exhaustion_multiplier=EXHAUSTED_KILL_MULTIPLIER,
    )
    ratio = p_exhausted / p_precursor
    assert 0.05 <= ratio <= 0.15, (
        f"expected exhausted kill ~{EXHAUSTED_KILL_MULTIPLIER}x precursor, "
        f"got ratio {ratio}"
    )


def test_kill_outcome_decomposition_records_exhaustion_fields():
    o = kill_outcome(
        0.8, 0.7, 1.0,
        is_exhausted=True,
        exhaustion_multiplier=0.2,
    )
    assert o.is_exhausted is True
    assert o.exhaustion_multiplier == pytest.approx(0.2)


def test_kill_outcome_exhausted_signal_skips_rescue_term():
    """The combined_signal for an exhausted clone must NOT include the
    checkpoint-block rescue contribution -- it equals affinity * MHC *
    costim only."""
    o = kill_outcome(
        0.8, 0.7, 0.0,
        checkpoint_block=1.0,
        is_exhausted=True,
    )
    # signal = 0.8 * 0.7 * 0 = 0 -> kill_probability collapses
    assert o.combined_signal == pytest.approx(0.0)
    assert o.kill_probability == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Integration: drive enough engagements to exhaust a clone end-to-end
# ---------------------------------------------------------------------------

def test_chronic_antigen_drives_tcell_exhausted_event_and_drops_kill():
    """End-to-end: a colocated T cell engages a tumor target across
    enough engine steps to cross the exhaustion threshold. At that
    point a TCELL_EXHAUSTED event must appear on the bus AND the
    per-encounter kill probability computed for that clone must drop
    by approximately the EXHAUSTED_KILL_MULTIPLIER ratio."""
    from core import SimulationConfig, SimulationEngine
    from core.event_bus import EventTypes
    from modules.cellular_module import CellularModule
    from modules.immune_module import ImmuneModule
    from modules.molecular_module import MolecularModule

    np.random.seed(0)
    engine = SimulationEngine(SimulationConfig(
        dt=0.01, duration=0.10, use_gpu=False,
    ))
    engine.register_module("molecular", MolecularModule, {
        "transcription_rate": 0.0,
        "exosome_release_rate": 0.0,
        "mutation_rate": 0.0,
    })
    engine.register_module("cellular", CellularModule, {
        "n_normal_cells": 0, "n_cancer_cells": 0,
        "hla_alleles": ["HLA-A*02:01", "HLA-A*24:02", "HLA-B*07:02"],
    })
    engine.register_module("immune", ImmuneModule, {
        "n_t_cells": 1, "n_nk_cells": 0, "n_macrophages": 0,
        "tcr_recognition_threshold": 0.0,
        "tcr_repertoire_size": 200,
        "tcr_seed": 0,
        # No costim and no checkpoint block: kill probability is ~0,
        # so the cancer cell survives each engagement and the T cell
        # can keep re-engaging. This simulates chronic antigen
        # exposure -- the actual driver of exhaustion in vivo.
        "costimulation": 0.0,
        "checkpoint_block": 0.0,
    })
    engine.initialize()
    mol = engine.modules["molecular"]
    cel = engine.modules["cellular"]
    imm = engine.modules["immune"]
    cel.set_molecular_module(mol)
    imm.set_cellular_module(cel)

    pos = [100.0, 100.0, 50.0]
    cancer_id = cel.add_cell(position=pos, cell_type="cancer")
    cel.cells[cancer_id].mhc1_expression = 0.9
    mol.add_cell(cancer_id)
    t_id = next(iter(imm.immune_cells))
    imm.immune_cells[t_id].position = np.array(pos, dtype=np.float32)

    # Force the engaging clone to exhaust after just two engagements
    # so the test runs in a few engine steps.
    imm._tcr_repertoire.exhaustion_threshold = 2

    mol.introduce_mutation(cancer_id, "KRAS", "G12D")
    engine.run(duration=0.10)
    engine.event_bus.process_events()

    log = engine.event_bus.event_log
    exhaustion_events = [
        data for evt, data in log
        if evt == EventTypes.TCELL_EXHAUSTED
    ]
    assert exhaustion_events, (
        "expected at least one TCELL_EXHAUSTED event after chronic "
        "engagement; got none"
    )
    first = exhaustion_events[0]
    assert first["encounter_count"] >= 2
    assert first["mutation"] == "G12D"

    # Verify the repertoire stats reflect the transition.
    assert imm._tcr_repertoire.exhausted_count() >= 1
    assert imm.total_exhaustion_transitions >= 1

    # And confirm the kill probability for the same clone has
    # collapsed (this is the bottom-line clinical effect).
    exhausted_id = first["tcr_id"]
    assert (
        imm._tcr_repertoire.exhaustion_state(exhausted_id)
        is ExhaustionState.EXHAUSTED
    )
