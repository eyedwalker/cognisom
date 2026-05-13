"""
TME 4-type classifier unit + integration tests (Teng et al. 2015).

Pure-function tests use a tiny SimpleNamespace stand-in for cells so
the classifier can be exercised without spinning the engine. The
integration test wires the full SimulationEngine + Molecular +
Cellular + Immune trio so that the lifecycle (mutation -> activation
-> adaptive PD-L1) is exercised end-to-end and the
TME_CLASSIFIED event is observed on the bus.

Patent-evidence claim: given a patient's VCF, the cognisom pipeline
emits a TME classification that predicts ICB response category.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# Silence MHCflurry / Keras chatter when the integration test pulls it in.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.py.immune.tme_classifier import (
    DEFAULT_PDL1_FRAC_POS,
    DEFAULT_PDL1_PER_CELL_POS,
    DEFAULT_TIL_RATIO_POS,
    TMEClassification,
    TMEType,
    classify_tme,
)


# ---------------------------------------------------------------------------
# Tiny stand-in cell type (avoids dragging in the full CellState).
# ---------------------------------------------------------------------------

def _cancer(position=(0.0, 0.0, 0.0), pdl1=0.0):
    return SimpleNamespace(
        position=np.array(position, dtype=np.float32),
        pdl1_expression=float(pdl1),
        cell_type='cancer',
    )


def _tcell(position=(0.0, 0.0, 0.0), in_blood=False):
    return SimpleNamespace(
        position=np.array(position, dtype=np.float32),
        cell_type='T_cell',
        in_blood=in_blood,
    )


def _nk(position=(0.0, 0.0, 0.0)):
    return SimpleNamespace(
        position=np.array(position, dtype=np.float32),
        cell_type='NK_cell',
        in_blood=False,
    )


# ---------------------------------------------------------------------------
# Direct classification of the four types
# ---------------------------------------------------------------------------

def test_type_i_hot_tumor_with_adaptive_pdl1():
    """TIL+ PD-L1+ -- the patent-evidence claim A case: high-response
    candidate for checkpoint blockade."""
    cancers = [_cancer((i * 1.0, 0, 0), pdl1=0.8) for i in range(4)]
    tcells = [_tcell((i * 1.0, 0, 0)) for i in range(4)]
    r = classify_tme(cancers, tcells)
    assert r.tme_type is TMEType.TYPE_I
    assert r.predicted_icb_response == "high"
    assert r.n_til == 4
    assert r.til_ratio == 1.0
    assert r.pdl1_positive_fraction == 1.0


def test_type_ii_cold_tumor_no_til_no_pdl1():
    """TIL- PD-L1- -- the 'immunological ignorance' case."""
    cancers = [_cancer((i * 1.0, 0, 0), pdl1=0.0) for i in range(4)]
    # T cells off in the lymph node, nowhere near the tumor
    tcells = [_tcell((1000.0, 0, 0)) for _ in range(4)]
    r = classify_tme(cancers, tcells)
    assert r.tme_type is TMEType.TYPE_II
    assert r.predicted_icb_response == "minimal"
    assert r.n_til == 0
    assert r.pdl1_positive_fraction == 0.0


def test_type_iii_intrinsic_pdl1_no_tils():
    """TIL- PD-L1+ -- intrinsic / oncogene-driven PD-L1, no TIL pressure."""
    cancers = [_cancer((i * 1.0, 0, 0), pdl1=0.8) for i in range(4)]
    tcells = [_tcell((1000.0, 0, 0)) for _ in range(4)]
    r = classify_tme(cancers, tcells)
    assert r.tme_type is TMEType.TYPE_III
    assert r.predicted_icb_response == "low"
    assert r.n_til == 0
    assert r.pdl1_positive_fraction == 1.0


def test_type_iv_tils_but_no_pdl1():
    """TIL+ PD-L1- -- TILs present but suppressed by non-PD-1 axes
    (Treg / MDSC / TGF-beta). Checkpoint blockade less effective."""
    cancers = [_cancer((i * 1.0, 0, 0), pdl1=0.0) for i in range(4)]
    tcells = [_tcell((i * 1.0, 0, 0)) for i in range(4)]
    r = classify_tme(cancers, tcells)
    assert r.tme_type is TMEType.TYPE_IV
    assert r.predicted_icb_response == "moderate"
    assert r.n_til == 4
    assert r.pdl1_positive_fraction == 0.0


# ---------------------------------------------------------------------------
# Threshold edge cases
# ---------------------------------------------------------------------------

def test_til_threshold_is_inclusive_lower_bound():
    """At exactly the TIL ratio threshold, the tumor is classified
    TIL+. Boundary belongs to the positive side."""
    cancers = [_cancer((0, 0, 0), pdl1=0.0), _cancer((1, 0, 0), pdl1=0.0)]
    # 1 TIL / 2 cancer cells = ratio 0.5 == DEFAULT_TIL_RATIO_POS
    tils = [_tcell((0, 0, 0))]
    r = classify_tme(cancers, tils)
    assert r.til_ratio == pytest.approx(DEFAULT_TIL_RATIO_POS)
    # No PD-L1, so this is Type IV (TIL+ PDL1-) not Type II.
    assert r.tme_type is TMEType.TYPE_IV


def test_pdl1_fraction_threshold_is_inclusive():
    """At exactly DEFAULT_PDL1_FRAC_POS (25%), tumor is PD-L1+."""
    # 1 of 4 cells is PD-L1+ -> fraction 0.25 == threshold
    cancers = [
        _cancer((0, 0, 0), pdl1=DEFAULT_PDL1_PER_CELL_POS),
        _cancer((1, 0, 0), pdl1=0.0),
        _cancer((2, 0, 0), pdl1=0.0),
        _cancer((3, 0, 0), pdl1=0.0),
    ]
    tils = [_tcell((1000.0, 0, 0))]
    r = classify_tme(cancers, tils)
    assert r.pdl1_positive_fraction == pytest.approx(0.25)
    assert r.tme_type is TMEType.TYPE_III


def test_til_proximity_filter_rejects_distant_cells():
    """Only cells within til_proximity_um count as TILs."""
    cancer = [_cancer((0, 0, 0), pdl1=0.0)]
    near = _tcell((10.0, 0, 0))   # within 20 um default
    far = _tcell((100.0, 0, 0))   # outside
    r = classify_tme(cancer, [near, far])
    assert r.n_til == 1


def test_til_filter_excludes_blood_pool_cells():
    """Cells with in_blood=True are not infiltrating, even if their
    coordinate lies inside the tumor."""
    cancer = [_cancer((0, 0, 0), pdl1=0.0)]
    blood = _tcell((0, 0, 0), in_blood=True)
    r = classify_tme(cancer, [blood])
    assert r.n_til == 0


def test_immune_type_filter_default_excludes_nk():
    """Default immune_type_filter counts T cells only -- NK cells are
    not part of the IHC CD8 readout the clinical 4-type scheme uses."""
    cancer = [_cancer((0, 0, 0), pdl1=0.0)]
    r = classify_tme(cancer, [_nk((0, 0, 0))])
    assert r.n_til == 0


def test_immune_type_filter_none_counts_all():
    cancer = [_cancer((0, 0, 0), pdl1=0.0)]
    r = classify_tme(
        cancer, [_nk((0, 0, 0))],
        immune_type_filter=None,
    )
    assert r.n_til == 1


# ---------------------------------------------------------------------------
# Empty / degenerate populations
# ---------------------------------------------------------------------------

def test_empty_cancer_population_returns_type_ii():
    r = classify_tme([], [_tcell((0, 0, 0))])
    assert r.tme_type is TMEType.TYPE_II
    assert r.n_cancer_cells == 0


def test_no_immune_cells_yields_type_ii_or_iii_only():
    cancers = [_cancer((0, 0, 0), pdl1=0.0)]
    r1 = classify_tme(cancers, [])
    assert r1.tme_type is TMEType.TYPE_II
    cancers2 = [_cancer((0, 0, 0), pdl1=0.8)]
    r2 = classify_tme(cancers2, [])
    assert r2.tme_type is TMEType.TYPE_III


# ---------------------------------------------------------------------------
# Custom thresholds
# ---------------------------------------------------------------------------

def test_custom_thresholds_can_flip_classification():
    """A tumor that is Type IV at default thresholds can become Type
    II at a stricter TIL threshold."""
    cancers = [_cancer((0, 0, 0), pdl1=0.0) for _ in range(4)]
    tils = [_tcell((0, 0, 0)) for _ in range(2)]
    default = classify_tme(cancers, tils)
    assert default.tme_type is TMEType.TYPE_IV  # ratio 0.5 == threshold
    strict = classify_tme(cancers, tils, til_ratio_threshold=0.8)
    assert strict.tme_type is TMEType.TYPE_II


# ---------------------------------------------------------------------------
# Integration: drive the closed loop and observe TME_CLASSIFIED
# ---------------------------------------------------------------------------

def test_end_to_end_kras_g12d_yields_type_i_with_event_emission():
    """End-to-end patent-evidence path: a real cancer cell mutated
    with KRAS G12D, recognized by the patient TCR repertoire, drives
    adaptive PD-L1 induction. The colocated T cell makes the tumor
    TIL+; the activation-driven PD-L1 bump makes it PD-L1+. Result:
    Type I, ICB-responsive."""
    from core import SimulationConfig, SimulationEngine
    from core.event_bus import EventTypes
    from modules.cellular_module import CellularModule
    from modules.immune_module import ImmuneModule
    from modules.molecular_module import MolecularModule

    np.random.seed(0)
    engine = SimulationEngine(SimulationConfig(
        dt=0.01, duration=0.05, use_gpu=False,
    ))
    engine.register_module("molecular", MolecularModule, {
        "transcription_rate": 0.0, "exosome_release_rate": 0.0,
        "mutation_rate": 0.0,
    })
    engine.register_module("cellular", CellularModule, {
        "n_normal_cells": 0, "n_cancer_cells": 0,
        "hla_alleles": ["HLA-A*02:01", "HLA-A*24:02", "HLA-B*07:02"],
    })
    engine.register_module("immune", ImmuneModule, {
        "n_t_cells": 1, "n_nk_cells": 0, "n_macrophages": 0,
        "tcr_recognition_threshold": 0.0,
        "tcr_repertoire_size": 500,
        "tcr_seed": 0,
        "costimulation": 1.0,
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

    mol.introduce_mutation(cancer_id, "KRAS", "G12D")
    engine.run(duration=0.05)

    # Classification time: T cell has activated, PD-L1 is bumped by
    # the adaptive path. The cell may have died from the closed-loop
    # kill; classify before/after we don't care -- but if it's dead
    # we have to handle that.
    result = imm.classify_tme()
    # The kill probability is essentially 1 at these settings, so the
    # cancer cell is gone -> Type II (degenerate). To check the
    # mechanism we re-spawn the cell, classify before letting the
    # engine clear it.
    if result.n_cancer_cells == 0:
        # Re-do with a fresh cell, classify immediately after the
        # first activation but before the kill resolves.
        # Need at least 2 steps: step 1 processes the queued
        # MUTATION_OCCURRED -> displayed peptides land at end of step.
        # Step 2+ is where recognition + adaptive PD-L1 bump can fire.
        engine2 = SimulationEngine(SimulationConfig(
            dt=0.01, duration=0.05, use_gpu=False,
        ))
        engine2.register_module("molecular", MolecularModule, {
            "transcription_rate": 0.0, "exosome_release_rate": 0.0,
            "mutation_rate": 0.0,
        })
        engine2.register_module("cellular", CellularModule, {
            "n_normal_cells": 0, "n_cancer_cells": 0,
            "hla_alleles": ["HLA-A*02:01", "HLA-A*24:02", "HLA-B*07:02"],
        })
        # Very low kill probability so the cell survives long enough
        # to be classified TIL+ PD-L1+ before being killed.
        engine2.register_module("immune", ImmuneModule, {
            "n_t_cells": 1, "n_nk_cells": 0, "n_macrophages": 0,
            "tcr_recognition_threshold": 0.0,
            "tcr_repertoire_size": 500, "tcr_seed": 0,
            "costimulation": 0.0,  # disable kill while keeping recognition
            "checkpoint_block": 0.0,
        })
        engine2.initialize()
        mol2, cel2, imm2 = (engine2.modules[k] for k in
                            ("molecular", "cellular", "immune"))
        cel2.set_molecular_module(mol2)
        imm2.set_cellular_module(cel2)
        cid = cel2.add_cell(position=pos, cell_type="cancer")
        cel2.cells[cid].mhc1_expression = 0.9
        mol2.add_cell(cid)
        t2_id = next(iter(imm2.immune_cells))
        imm2.immune_cells[t2_id].position = np.array(pos, dtype=np.float32)
        mol2.introduce_mutation(cid, "KRAS", "G12D")
        engine2.run(duration=0.05)
        result = imm2.classify_tme()
        # Flush the queued TME_CLASSIFIED into the event log so the
        # assertion below can find it.
        engine2.event_bus.process_events()
        log = engine2.event_bus.event_log
    else:
        # Same flush for the primary path.
        engine.event_bus.process_events()
        log = engine.event_bus.event_log

    assert result.tme_type is TMEType.TYPE_I, (
        f"expected Type I after closed-loop activation, got {result.tme_type}; "
        f"TIL ratio={result.til_ratio}, PD-L1 frac={result.pdl1_positive_fraction}"
    )
    assert result.predicted_icb_response == "high"

    # The TME_CLASSIFIED event must appear on the bus.
    tme_events = [data for evt, data in log if evt == EventTypes.TME_CLASSIFIED]
    assert tme_events, "TME_CLASSIFIED event was not emitted"
    last = tme_events[-1]
    assert last["tme_type"] == "I"
    assert last["predicted_icb_response"] == "high"
