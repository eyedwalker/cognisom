"""A cell chasing a target it recognised closes on it at its real speed.

Speed is um/min and the engine steps in hours, so one chase step is
speed x 60 x dt. Both chase sites -- ImmuneModule.update and the opt-in
GPU-patched update -- multiplied by 0.01 instead, the same wrong-direction
conversion already fixed in _patrol: 6,000x too slow. At the default
10 um/min and dt = 0.01 h a chasing cell moved 0.001 um per step, so a cell
that recognised a target from outside the 5 um kill radius never reached it
and only targets first seen from inside the kill radius could die.

The step is also capped at the remaining distance, so a cell closes on its
target rather than flying past it once the step is large.
"""
from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from cognisom.engine.py.spatial.ecm_barrier import motility_attenuation
from cognisom.modules.cellular_module import CellState
from cognisom.modules.immune_module import (
    MINUTES_PER_HOUR, ImmuneModule, chase_displacement,
)

TARGET = 7
SPEED = 10.0          # ImmuneCell.speed default, um/min


class _Tissue:
    def __init__(self, target, ecm=0.0):
        self.cells = {target.cell_id: target}
        self._ecm = ecm

    def ecm_density_at(self, position):
        return self._ecm


def _setup(gap_um, ecm=0.0):
    """One NK cell already engaged on a target ``gap_um`` away along x.

    Kill probability 0, so the run measures movement and nothing else."""
    target = CellState(cell_id=TARGET, cell_type="cancer", phase="G1",
                       position=np.array([100.0, 100.0, 50.0], dtype=np.float32))
    with contextlib.redirect_stdout(io.StringIO()):
        m = ImmuneModule({"n_t_cells": 0, "n_nk_cells": 0, "n_macrophages": 0,
                          "nk_kill_probability": 0.0, "kill_probability": 0.0,
                          "library_params": False})
    m.set_cellular_module(_Tissue(target, ecm))
    cell = m.immune_cells[m.add_immune_cell([100.0 - gap_um, 100.0, 50.0], "NK_cell")]
    cell.activated = True
    cell.target_cell_id = TARGET
    return m, cell, target


def _gap(cell, target):
    return float(np.linalg.norm(target.position - cell.position))


def test_the_step_is_speed_times_sixty_times_dt():
    step = chase_displacement(np.array([1.0, 0.0, 0.0]), 100.0, SPEED, 0.01)
    assert step == pytest.approx([SPEED * MINUTES_PER_HOUR * 0.01, 0.0, 0.0])


def test_the_step_never_passes_the_target():
    step = chase_displacement(np.array([0.0, 1.0, 0.0]), 3.0, SPEED, 0.05)
    assert step == pytest.approx([0.0, 3.0, 0.0])


def test_a_chasing_cell_closes_six_microns_per_default_step():
    m, cell, target = _setup(gap_um=30.0)
    m.update(0.01)
    assert _gap(cell, target) == pytest.approx(30.0 - 6.0, abs=1e-4)


def test_a_cell_detected_outside_the_kill_radius_now_reaches_it():
    """The case the bug made impossible: seen at 8 um, killed from < 5 um."""
    m, cell, target = _setup(gap_um=8.0)
    assert _gap(cell, target) > cell.kill_radius
    m.update(0.01)
    assert _gap(cell, target) < cell.kill_radius


def test_a_large_step_lands_on_the_target_instead_of_overshooting():
    m, cell, target = _setup(gap_um=3.0)
    m.update(0.05)                       # uncapped, this step is 30 um
    assert _gap(cell, target) == pytest.approx(0.0, abs=1e-4)


def test_ecm_still_slows_the_chase():
    m, cell, target = _setup(gap_um=30.0, ecm=0.5)
    m.update(0.01)
    expected = motility_attenuation(SPEED, 0.5) * MINUTES_PER_HOUR * 0.01
    assert expected < 6.0
    assert 30.0 - _gap(cell, target) == pytest.approx(expected, abs=1e-4)


def test_the_gpu_patched_update_chases_at_the_same_speed():
    from cognisom.gpu.orchestrator import GPUOrchestrator

    class _Engine:
        def __init__(self, modules):
            self.modules = modules

    m, cell, target = _setup(gap_um=30.0)
    with contextlib.redirect_stdout(io.StringIO()):
        GPUOrchestrator(force_cpu=True, accelerate_immune=True).accelerate(
            _Engine({"immune": m}))
    m.update(0.01)
    assert _gap(cell, target) == pytest.approx(30.0 - 6.0, abs=1e-4)
