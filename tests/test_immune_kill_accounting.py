"""A target dies once, however many immune cells reach it in one step.

CANCER_KILLED is queued and the cellular module removes the dead cell only
when the step's events are processed, so for the rest of the immune update
the target was still in the step's snapshot of cancer cells. A second cell
reaching it in the same step killed it again: total_kills and the
CANCER_KILLED / CELL_KILLED_BY_TCELL events overcounted, while the cellular
module's second _kill_cell silently returned. Measured on the Hypoxia
preset once the chase step moves cells at their real speed: 35 kills for 20
deaths. A cell that had not yet activated could also detect the corpse and
activate on it.

Both immune update paths are checked: ImmuneModule.update, and the opt-in
GPU-patched update in gpu/orchestrator.
"""
from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from cognisom.core.event_bus import EventBus, EventTypes
from cognisom.modules.cellular_module import CellState
from cognisom.modules.immune_module import ImmuneModule

TARGET = 7


class _Tissue:
    """What the immune update reads from a cellular module, and no more."""

    def __init__(self, *cells):
        self.cells = {c.cell_id: c for c in cells}

    def ecm_density_at(self, position):
        return 0.0


def _cancer(position, mhc1=0.1):
    """MHC-I low, so the legacy NK missing-self rule recognises it."""
    return CellState(cell_id=TARGET, cell_type="cancer", phase="G1",
                     position=np.array(position, dtype=np.float32),
                     mhc1_expression=mhc1)


def _module(target):
    with contextlib.redirect_stdout(io.StringIO()):
        m = ImmuneModule({"n_t_cells": 0, "n_nk_cells": 0, "n_macrophages": 0,
                          "nk_kill_probability": 1.0, "library_params": False})
    m.event_bus = EventBus()
    m.set_cellular_module(_Tissue(target))
    return m


def _engaged_nk(module, position):
    cell = module.immune_cells[module.add_immune_cell(position, "NK_cell")]
    cell.activated = True
    cell.target_cell_id = TARGET
    return cell


def _killed_events(module):
    return [d for e, d in module.event_bus.event_queue if e == EventTypes.CANCER_KILLED]


def test_two_cells_in_contact_with_one_target_kill_it_once():
    m = _module(_cancer([100, 100, 50]))
    first = _engaged_nk(m, [102, 100, 50])
    second = _engaged_nk(m, [98, 100, 50])
    m.update(0.01)
    assert m.total_kills == 1
    assert len(_killed_events(m)) == 1
    # The second lost its target, so it disengages and will patrol.
    assert (second.activated, second.target_cell_id) == (False, None)
    assert first.activated is False          # disengaged by its own kill


def test_a_cell_does_not_activate_on_a_target_killed_earlier_in_the_step():
    m = _module(_cancer([100, 100, 50]))
    _engaged_nk(m, [102, 100, 50])
    bystander = m.immune_cells[m.add_immune_cell([94, 100, 50], "NK_cell")]
    m.update(0.01)
    assert m.total_kills == 1
    assert bystander.activated is False
    assert m.total_activations == 0          # the engaged cell was set up by hand


def test_the_gpu_patched_update_kills_once_too():
    from cognisom.gpu.orchestrator import GPUOrchestrator

    class _Engine:
        def __init__(self, modules):
            self.modules = modules

    m = _module(_cancer([100, 100, 50]))
    m.kill_probability = 1.0                 # the GPU path reads the flat knob
    _engaged_nk(m, [102, 100, 50])
    _engaged_nk(m, [98, 100, 50])
    with contextlib.redirect_stdout(io.StringIO()):
        GPUOrchestrator(force_cpu=True, accelerate_immune=True).accelerate(
            _Engine({"immune": m}))
    m.update(0.01)
    assert m.total_kills == 1
    assert len(_killed_events(m)) == 1
