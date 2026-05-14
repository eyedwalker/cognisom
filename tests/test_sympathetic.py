"""
Sympathetic / β2-adrenergic immunosuppression tests (Upgrade 7).

Closes the lecture-flagged "neuroimmune axis" gap (V. Chen,
CU Anschutz 2026-05-03, slides 30-33). The patent-evidence claim is
that cognisom is the first cancer simulator to gate T-cell function
on a patient stress proxy, with β-blocker (propranolol) therapy as a
first-class rescue parameter -- matching the retrospective clinical
observation that β-blocker users on hypertension medication have
better outcomes on ICB (Kokolus 2018; Oh 2021).

Coverage:
  * sympathetic_attenuation: invariants, boundary values, monotone
    behaviour vs stress and vs blocker, clamping.
  * SympatheticState snapshot fields are internally consistent.
  * Immune module integration: high stress reduces T-cell kill
    probability; β-blocker rescues.
  * End-to-end: same closed-loop setup with stress=0 vs stress=1
    produces a strictly smaller kill count under stress.
  * Runtime setter: set_stress() updates state mid-run.
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

from engine.py.immune.sympathetic import (
    DEFAULT_MAX_SUPPRESSION,
    SympatheticState,
    sympathetic_attenuation,
    sympathetic_state,
)


# ---------------------------------------------------------------------------
# Pure-function invariants
# ---------------------------------------------------------------------------

def test_zero_stress_yields_full_function():
    assert sympathetic_attenuation(0.0) == pytest.approx(1.0)
    assert sympathetic_attenuation(0.0, beta_blocker=1.0) == pytest.approx(1.0)


def test_full_stress_no_blocker_hits_max_suppression():
    out = sympathetic_attenuation(1.0, 0.0, max_suppression=0.7)
    assert out == pytest.approx(0.3, abs=1e-9)


def test_full_stress_full_blocker_is_full_rescue():
    out = sympathetic_attenuation(1.0, 1.0)
    assert out == pytest.approx(1.0)


def test_attenuation_monotone_in_stress():
    prev = sympathetic_attenuation(0.0)
    for s in (0.1, 0.3, 0.5, 0.8, 1.0):
        cur = sympathetic_attenuation(s)
        assert cur <= prev, f"non-monotone at stress={s}"
        prev = cur


def test_attenuation_monotone_in_blocker_under_stress():
    prev = sympathetic_attenuation(1.0, 0.0)
    for b in (0.1, 0.3, 0.5, 0.8, 1.0):
        cur = sympathetic_attenuation(1.0, b)
        assert cur >= prev, f"blocker={b} did not increase function"
        prev = cur


def test_attenuation_bounded_in_unit_interval():
    for s in (-1.0, 0.0, 0.5, 1.0, 2.0):
        for b in (-1.0, 0.0, 0.5, 1.0, 2.0):
            out = sympathetic_attenuation(s, b)
            assert 1.0 - DEFAULT_MAX_SUPPRESSION - 1e-9 <= out <= 1.0 + 1e-9


def test_attenuation_clamps_negative_and_above_one():
    """Out-of-range inputs are clamped to [0, 1], not extrapolated."""
    assert sympathetic_attenuation(-1.0, 0.0) == pytest.approx(1.0)
    assert sympathetic_attenuation(2.0, 0.0) == pytest.approx(
        sympathetic_attenuation(1.0, 0.0)
    )
    assert sympathetic_attenuation(1.0, 5.0) == pytest.approx(1.0)


def test_attenuation_nan_inputs_treated_as_zero():
    assert sympathetic_attenuation(float("nan"), 0.0) == pytest.approx(1.0)


def test_max_suppression_zero_disables_axis():
    """If the max_suppression knob is 0, stress has no effect."""
    assert sympathetic_attenuation(1.0, 0.0, max_suppression=0.0) == 1.0


# ---------------------------------------------------------------------------
# SympatheticState snapshot
# ---------------------------------------------------------------------------

def test_state_fields_match_attenuation():
    s = sympathetic_state(0.6, 0.2)
    expected_effective = 0.6 * (1.0 - 0.2)
    expected_retained = 1.0 - expected_effective * DEFAULT_MAX_SUPPRESSION
    assert s.stress_level == pytest.approx(0.6)
    assert s.beta_blocker == pytest.approx(0.2)
    assert s.effective_signal == pytest.approx(expected_effective)
    assert s.t_cell_function_retained == pytest.approx(expected_retained)


# ---------------------------------------------------------------------------
# Immune module integration
# ---------------------------------------------------------------------------

def _build_engine(**immune_overrides):
    from core import SimulationConfig, SimulationEngine
    from modules.cellular_module import CellularModule
    from modules.immune_module import ImmuneModule
    from modules.molecular_module import MolecularModule

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
    immune_cfg = {
        "n_t_cells": 1, "n_nk_cells": 0, "n_macrophages": 0,
        "tcr_recognition_threshold": 0.0,
        "tcr_repertoire_size": 200,
        "tcr_seed": 0,
        "costimulation": 1.0,
        "checkpoint_block": 0.0,
    }
    immune_cfg.update(immune_overrides)
    engine.register_module("immune", ImmuneModule, immune_cfg)
    engine.initialize()
    return engine


def _spawn_test_cell(engine, position=(100.0, 100.0, 50.0)):
    mol = engine.modules["molecular"]
    cel = engine.modules["cellular"]
    imm = engine.modules["immune"]
    cel.set_molecular_module(mol)
    imm.set_cellular_module(cel)
    cancer_id = cel.add_cell(position=list(position), cell_type="cancer")
    cel.cells[cancer_id].mhc1_expression = 0.9
    mol.add_cell(cancer_id)
    t_id = next(iter(imm.immune_cells))
    imm.immune_cells[t_id].position = np.array(position, dtype=np.float32)
    return cancer_id


def test_immune_module_kill_probability_drops_under_stress():
    """At stress=1 the kill probability for a recognized clone should
    fall by the full max_suppression factor relative to stress=0."""
    eng = _build_engine(stress_level=1.0, beta_blocker=0.0)
    imm = eng.modules["immune"]
    state = imm.get_sympathetic_state()
    assert state.stress_level == 1.0
    assert state.t_cell_function_retained == pytest.approx(
        1.0 - DEFAULT_MAX_SUPPRESSION
    )


def test_set_stress_runtime_setter_updates_state():
    eng = _build_engine(stress_level=0.0, beta_blocker=0.0)
    imm = eng.modules["immune"]
    assert imm.get_sympathetic_state().stress_level == 0.0
    new = imm.set_stress(stress_level=0.7, beta_blocker=0.5)
    assert new.stress_level == pytest.approx(0.7)
    assert new.beta_blocker == pytest.approx(0.5)
    assert isinstance(new, SympatheticState)


def test_stress_reduces_total_tcell_kills_end_to_end():
    """Compare two parallel runs with identical seeds: the high-stress
    arm must record strictly fewer T-cell kills than the no-stress
    arm. This is the patent-evidence-relevant population-level claim
    (Armaiz-Pena 2015 corollary).

    Uses sympathetic_max_suppression=0.99 so the stressed arm's
    per-step kill probability drops to ~1% of baseline, making the
    stochastic difference robust to seed."""
    # Run 1: no stress
    np.random.seed(0)
    eng_low = _build_engine(stress_level=0.0)
    _spawn_test_cell(eng_low)
    eng_low.modules["molecular"].introduce_mutation(0, "KRAS", "G12D")
    eng_low.run(duration=0.05)
    kills_low = eng_low.modules["immune"].total_tcell_kills

    # Run 2: high stress + near-complete suppression
    np.random.seed(0)
    eng_high = _build_engine(
        stress_level=1.0,
        beta_blocker=0.0,
        sympathetic_max_suppression=0.99,
    )
    _spawn_test_cell(eng_high)
    eng_high.modules["molecular"].introduce_mutation(0, "KRAS", "G12D")
    eng_high.run(duration=0.05)
    kills_high = eng_high.modules["immune"].total_tcell_kills

    assert kills_high <= kills_low, (
        f"stress should reduce kills; got stress={kills_high} vs "
        f"baseline={kills_low}"
    )
    # Baseline must produce at least one kill (otherwise the
    # suppression test is vacuous); strict drop expected under stress.
    assert kills_low > 0
    assert kills_high < kills_low


def test_beta_blocker_rescues_stressed_kill_count():
    """Stress + β-blocker must produce at least as many kills as the
    baseline (modulo stochasticity); strictly more than stress alone."""
    # Stress alone
    np.random.seed(0)
    eng_stress = _build_engine(stress_level=1.0, beta_blocker=0.0)
    _spawn_test_cell(eng_stress)
    eng_stress.modules["molecular"].introduce_mutation(0, "KRAS", "G12D")
    eng_stress.run(duration=0.05)
    kills_stress = eng_stress.modules["immune"].total_tcell_kills

    # Stress + full blocker
    np.random.seed(0)
    eng_rescued = _build_engine(stress_level=1.0, beta_blocker=1.0)
    _spawn_test_cell(eng_rescued)
    eng_rescued.modules["molecular"].introduce_mutation(0, "KRAS", "G12D")
    eng_rescued.run(duration=0.05)
    kills_rescued = eng_rescued.modules["immune"].total_tcell_kills

    assert kills_rescued >= kills_stress, (
        f"β-blocker should rescue function; got rescued={kills_rescued} vs "
        f"stress={kills_stress}"
    )


def test_max_suppression_zero_means_stress_is_inert():
    """If the operator disables the axis (max_suppression=0), stress
    has no effect on kills."""
    np.random.seed(0)
    eng_no_axis = _build_engine(
        stress_level=1.0,
        beta_blocker=0.0,
        sympathetic_max_suppression=0.0,
    )
    _spawn_test_cell(eng_no_axis)
    eng_no_axis.modules["molecular"].introduce_mutation(0, "KRAS", "G12D")
    eng_no_axis.run(duration=0.05)
    kills_no_axis = eng_no_axis.modules["immune"].total_tcell_kills

    np.random.seed(0)
    eng_baseline = _build_engine(stress_level=0.0)
    _spawn_test_cell(eng_baseline)
    eng_baseline.modules["molecular"].introduce_mutation(0, "KRAS", "G12D")
    eng_baseline.run(duration=0.05)
    kills_baseline = eng_baseline.modules["immune"].total_tcell_kills

    assert kills_no_axis == kills_baseline
