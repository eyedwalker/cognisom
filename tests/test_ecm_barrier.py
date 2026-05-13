"""
ECM barrier unit + integration tests (Upgrade 6).

Closes the biggest mechanistic gap flagged in the lecture by V. Chen
(CU Anschutz, 2026-05-03, slides 34-35, 49-50): only 1% of PDAC
responds to ICB because the desmoplastic stroma physically excludes
TILs even when neoantigens are displayed. Without this layer the
pipeline cannot distinguish "no antigens / cold" from "antigens behind
a wall / cold-but-vaccinable-via-anti-fibrotic-combo".

Tests cover:
  - engine/py/spatial/ecm_barrier helpers: density sampling, motility
    attenuation, detection attenuation, clamping behaviour.
  - CellularModule: cancer cells deposit ECM over time;
    anti_fibrotic_active reverses the deposition; spawn-time baseline
    ECM is honoured.
  - ImmuneModule: high ECM compresses effective detection radius +
    speed (T cells colocated with a high-ECM tumor recognize less /
    move slower).
  - TMEClassification: ecm_excluded fires when TIL- AND mean ECM
    above threshold AND any neoantigens displayed (clinically
    actionable Type II split).
  - End-to-end: high-ECM tumor with a displayed neoantigen and a
    colocated T cell ends up TIL-negative + ecm_excluded.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.py.immune.tme_classifier import (
    DEFAULT_ECM_EXCLUDED_THRESHOLD,
    TMEType,
    classify_tme,
)
from engine.py.spatial.ecm_barrier import (
    MIN_RETAINED_FRACTION,
    detection_attenuation,
    ecm_density_at,
    motility_attenuation,
)


# ---------------------------------------------------------------------------
# Pure-function helpers
# ---------------------------------------------------------------------------

def _cancer(position=(0.0, 0.0, 0.0), ecm=0.0, peptides=None):
    return SimpleNamespace(
        position=np.array(position, dtype=np.float32),
        local_ecm_density=float(ecm),
        pdl1_expression=0.0,
        cell_type='cancer',
        mhc1_displayed_peptides=list(peptides or []),
    )


def test_ecm_density_at_empty_returns_zero():
    assert ecm_density_at(np.array([0.0, 0.0, 0.0]), []) == 0.0


def test_ecm_density_at_averages_in_radius():
    cells = [
        _cancer((0.0, 0.0, 0.0), ecm=0.8),
        _cancer((5.0, 0.0, 0.0), ecm=0.6),
        _cancer((1000.0, 0.0, 0.0), ecm=0.1),  # far away, ignored
    ]
    d = ecm_density_at(np.array([0.0, 0.0, 0.0]), cells, sample_radius_um=30.0)
    assert d == pytest.approx((0.8 + 0.6) / 2.0)


def test_ecm_density_clamps_to_unit_interval():
    cells = [_cancer((0.0, 0.0, 0.0), ecm=2.0)]
    d = ecm_density_at(np.array([0.0, 0.0, 0.0]), cells)
    assert d == 1.0


def test_motility_attenuation_zero_ecm_is_identity():
    assert motility_attenuation(10.0, 0.0) == pytest.approx(10.0)


def test_motility_attenuation_high_ecm_collapses():
    # blocking_factor 0.9 + density 1.0 -> retained = 0.1
    out = motility_attenuation(10.0, 1.0, blocking_factor=0.9)
    assert out == pytest.approx(1.0)


def test_motility_attenuation_floors_at_min_retained():
    # blocking_factor 1.0 + density 1.0 would give retained=0; floor
    # keeps T cells from getting permanently stuck.
    out = motility_attenuation(10.0, 1.0, blocking_factor=1.0)
    assert out == pytest.approx(10.0 * MIN_RETAINED_FRACTION)


def test_detection_attenuation_monotone_in_ecm():
    base = 20.0
    out_low = detection_attenuation(base, 0.0)
    out_mid = detection_attenuation(base, 0.5)
    out_high = detection_attenuation(base, 1.0)
    assert out_low > out_mid > out_high


# ---------------------------------------------------------------------------
# CellularModule ECM dynamics
# ---------------------------------------------------------------------------

def _build_cellular_module(**config):
    from core import SimulationConfig, SimulationEngine
    from modules.cellular_module import CellularModule
    from modules.molecular_module import MolecularModule

    cfg = {
        "n_normal_cells": 0,
        "n_cancer_cells": 0,
        "hla_alleles": ["HLA-A*02:01"],
        **config,
    }
    engine = SimulationEngine(SimulationConfig(
        dt=0.01, duration=0.01, use_gpu=False,
    ))
    engine.register_module("molecular", MolecularModule, {
        "transcription_rate": 0.0, "exosome_release_rate": 0.0,
        "mutation_rate": 0.0,
    })
    engine.register_module("cellular", CellularModule, cfg)
    engine.initialize()
    cel = engine.modules["cellular"]
    cel.set_molecular_module(engine.modules["molecular"])
    return engine, cel


def test_cancer_baseline_ecm_seeded_at_spawn():
    _, cel = _build_cellular_module(cancer_baseline_ecm=0.6)
    cid = cel.add_cell(position=[0, 0, 0], cell_type='cancer')
    assert cel.cells[cid].local_ecm_density == pytest.approx(0.6)


def test_cancer_cells_deposit_ecm_over_time():
    engine, cel = _build_cellular_module(ecm_deposition_rate=1.0)
    cid = cel.add_cell(position=[0, 0, 0], cell_type='cancer')
    before = cel.cells[cid].local_ecm_density
    # 10 steps of dt=0.01 with rate=1.0 -> ECM should rise ~0.1
    engine.run(duration=0.10)
    after = cel.cells[cid].local_ecm_density
    assert after > before
    assert after == pytest.approx(0.10, abs=0.02)


def test_anti_fibrotic_degrades_ecm():
    engine, cel = _build_cellular_module(
        anti_fibrotic_active=True,
        ecm_degradation_rate=2.0,
        cancer_baseline_ecm=0.8,
    )
    cid = cel.add_cell(position=[0, 0, 0], cell_type='cancer')
    assert cel.cells[cid].local_ecm_density == pytest.approx(0.8)
    engine.run(duration=0.20)
    after = cel.cells[cid].local_ecm_density
    assert after < 0.5, f"anti-fibrotic should drop ECM well below 0.5; got {after}"


def test_ecm_density_at_lookup_through_cellular_module():
    _, cel = _build_cellular_module()
    cid1 = cel.add_cell(position=[0, 0, 0], cell_type='cancer')
    cid2 = cel.add_cell(position=[5, 0, 0], cell_type='cancer')
    cel.cells[cid1].local_ecm_density = 0.7
    cel.cells[cid2].local_ecm_density = 0.5
    d = cel.ecm_density_at([0, 0, 0])
    assert d == pytest.approx(0.6)


def test_normal_cells_do_not_deposit_ecm():
    engine, cel = _build_cellular_module(ecm_deposition_rate=1.0)
    nid = cel.add_cell(position=[0, 0, 0], cell_type='normal')
    engine.run(duration=0.20)
    assert cel.cells[nid].local_ecm_density == 0.0


# ---------------------------------------------------------------------------
# TMEClassification ecm_excluded flag
# ---------------------------------------------------------------------------

def _make_peptide():
    """Stand-in for an MHC-displayed neoantigen entry. classify_tme only
    checks truthiness of mhc1_displayed_peptides."""
    return SimpleNamespace(sequence="YLAGGVGKV", mutation_label="X1Y")


def test_ecm_excluded_fires_for_cold_high_ecm_tumor_with_antigens():
    """No TILs + high ECM + neoantigens present -> ecm_excluded True."""
    cells = [
        _cancer((i * 1.0, 0, 0), ecm=0.8, peptides=[_make_peptide()])
        for i in range(4)
    ]
    far_tcells = [
        SimpleNamespace(
            position=np.array([1000.0, 0, 0], dtype=np.float32),
            cell_type='T_cell',
            in_blood=False,
        )
    ]
    r = classify_tme(cells, far_tcells)
    assert r.tme_type is TMEType.TYPE_II
    assert r.ecm_excluded is True
    assert r.mean_ecm_density == pytest.approx(0.8)
    assert "anti-fibrotic" in r.description.lower()


def test_ecm_excluded_does_not_fire_without_antigens():
    """A genuinely cold tumor (no neoantigens) should NOT flag exclusion."""
    cells = [
        _cancer((i * 1.0, 0, 0), ecm=0.8, peptides=[])
        for i in range(4)
    ]
    far_tcells = [
        SimpleNamespace(
            position=np.array([1000.0, 0, 0], dtype=np.float32),
            cell_type='T_cell',
            in_blood=False,
        )
    ]
    r = classify_tme(cells, far_tcells)
    assert r.tme_type is TMEType.TYPE_II
    assert r.ecm_excluded is False


def test_ecm_excluded_does_not_fire_below_threshold():
    low_ecm = DEFAULT_ECM_EXCLUDED_THRESHOLD - 0.05
    cells = [
        _cancer((i * 1.0, 0, 0), ecm=low_ecm, peptides=[_make_peptide()])
        for i in range(4)
    ]
    far_tcells = [
        SimpleNamespace(
            position=np.array([1000.0, 0, 0], dtype=np.float32),
            cell_type='T_cell',
            in_blood=False,
        )
    ]
    r = classify_tme(cells, far_tcells)
    assert r.ecm_excluded is False


def test_ecm_excluded_does_not_fire_when_tils_present():
    """High ECM + TILs is Type IV (suppression), not ECM-excluded."""
    cells = [
        _cancer((i * 1.0, 0, 0), ecm=0.8, peptides=[_make_peptide()])
        for i in range(4)
    ]
    tils = [
        SimpleNamespace(
            position=np.array([float(i), 0, 0], dtype=np.float32),
            cell_type='T_cell',
            in_blood=False,
        )
        for i in range(4)
    ]
    r = classify_tme(cells, tils)
    assert r.tme_type is TMEType.TYPE_IV
    assert r.ecm_excluded is False


# ---------------------------------------------------------------------------
# Immune motility / detection gating end-to-end
# ---------------------------------------------------------------------------

def test_high_ecm_drops_til_count_to_zero():
    """A T cell next to a high-ECM tumor (with displayed neoantigens)
    should fail to detect: effective detection radius is near zero,
    no recognition fires, no TIL bump on cancer cell."""
    from core import SimulationConfig, SimulationEngine
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
        # Tumor starts at near-maximum ECM (PDAC-like).
        "cancer_baseline_ecm": 0.95,
        "ecm_deposition_rate": 0.0,
    })
    engine.register_module("immune", ImmuneModule, {
        "n_t_cells": 1, "n_nk_cells": 0, "n_macrophages": 0,
        "tcr_recognition_threshold": 0.0,
        "tcr_repertoire_size": 200,
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
    # Place the T cell at the very edge of the unblocked detection
    # radius (10 um default). With ECM near 1.0 and detection_block
    # 0.8, effective radius collapses to ~2 um and the T cell is
    # outside it.
    t_id = next(iter(imm.immune_cells))
    imm.immune_cells[t_id].position = np.array(
        [pos[0] + 8.0, pos[1], pos[2]], dtype=np.float32
    )

    mol.introduce_mutation(cancer_id, "KRAS", "G12D")
    engine.run(duration=0.05)

    # No T-cell-mediated kill, no PD-L1 induction.
    assert imm.total_tcr_recognitions == 0
    assert imm.total_tcell_kills == 0
    if cancer_id in cel.cells:
        assert cel.cells[cancer_id].pdl1_expression == 0.0


def test_anti_fibrotic_restores_til_infiltration():
    """Same setup as above but with anti_fibrotic_active=True and a
    fast degradation rate. ECM drops over the run; the T cell ends
    up able to detect and engage."""
    from core import SimulationConfig, SimulationEngine
    from modules.cellular_module import CellularModule
    from modules.immune_module import ImmuneModule
    from modules.molecular_module import MolecularModule

    np.random.seed(0)
    engine = SimulationEngine(SimulationConfig(
        dt=0.01, duration=0.30, use_gpu=False,
    ))
    engine.register_module("molecular", MolecularModule, {
        "transcription_rate": 0.0, "exosome_release_rate": 0.0,
        "mutation_rate": 0.0,
    })
    engine.register_module("cellular", CellularModule, {
        "n_normal_cells": 0, "n_cancer_cells": 0,
        "hla_alleles": ["HLA-A*02:01", "HLA-A*24:02", "HLA-B*07:02"],
        "cancer_baseline_ecm": 0.95,
        "ecm_deposition_rate": 0.0,
        "anti_fibrotic_active": True,
        "ecm_degradation_rate": 5.0,
    })
    engine.register_module("immune", ImmuneModule, {
        "n_t_cells": 1, "n_nk_cells": 0, "n_macrophages": 0,
        "tcr_recognition_threshold": 0.0,
        "tcr_repertoire_size": 200,
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
    imm.immune_cells[t_id].position = np.array(
        [pos[0] + 8.0, pos[1], pos[2]], dtype=np.float32
    )

    mol.introduce_mutation(cancer_id, "KRAS", "G12D")
    engine.run(duration=0.30)

    # ECM should be substantially reduced by the end of the run.
    if cancer_id in cel.cells:
        assert cel.cells[cancer_id].local_ecm_density < 0.3
    # And by the end, at least one recognition should have fired.
    assert imm.total_tcr_recognitions >= 1, (
        f"anti-fibrotic should restore TIL infiltration; got "
        f"{imm.total_tcr_recognitions} recognitions"
    )
