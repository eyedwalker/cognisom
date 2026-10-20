"""A target dies once, however many immune cells reach it in one step.

CANCER_KILLED is queued and the cellular module removes the dead cell only
when the step's events are processed, so for the rest of the immune update
the target was still in the step's snapshot of cancer cells. A second cell
whose contact resolved in the same step killed it again: total_kills and the
CANCER_KILLED / CELL_KILLED_BY_TCELL events overcounted, while the cellular
module's second _kill_cell silently returned. Measured on the Hypoxia preset
once the chase step moves cells at their real speed: 35 kills for 20
deaths. A cell that had not yet activated could also detect the corpse and
activate on it.

Contacts resolve after a measured duration, so two contacts end in the same
step only when they are short against the step; the tests make the lethal
contact a fraction of a step so that they do.

Both immune update paths are checked: ImmuneModule.update, and the opt-in
GPU-patched update in gpu/orchestrator.
"""
from __future__ import annotations

import contextlib
import io

import numpy as np

from cognisom.core.event_bus import EventBus, EventTypes
from cognisom.modules.cellular_module import CellState
from cognisom.modules.immune_module import ImmuneModule

TARGET = 7
DT_H = 0.01
#: Mean lethal contact, minutes: 0.06 min is a tenth of a 0.01 h step.
SHORT_CONTACT = {"NK_cell": 0.06}


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
                          "nk_kill_probability": 1.0, "library_params": False,
                          "lethal_contact_minutes": SHORT_CONTACT})
    m.event_bus = EventBus()
    m.set_cellular_module(_Tissue(target))
    return m


def _engaged_nk(module, position):
    cell = module.immune_cells[module.add_immune_cell(position, "NK_cell")]
    cell.activated = True
    cell.target_cell_id = TARGET
    return cell


def _step(module):
    """One update, then what the cellular module does when the step's
    events are processed: remove each killed cell."""
    module.update(DT_H)
    for event, data in module.event_bus.event_queue:
        if event == EventTypes.CANCER_KILLED:
            module.cellular_module.cells.pop(data["cell_id"], None)


def _killed_events(module):
    return [d for e, d in module.event_bus.event_queue if e == EventTypes.CANCER_KILLED]


def test_two_cells_in_contact_with_one_target_kill_it_once():
    np.random.seed(0)
    m = _module(_cancer([100, 100, 50]))
    first = _engaged_nk(m, [102, 100, 50])
    second = _engaged_nk(m, [98, 100, 50])
    _step(m)                           # both contacts begin, both lethal
    assert first.contact_ends_at is not None and second.contact_ends_at is not None
    _step(m)                           # both resolve in this step
    assert m.total_kills == 1
    assert len(_killed_events(m)) == 1
    # The second lost its target, so it disengages and will patrol.
    assert (second.activated, second.target_cell_id) == (False, None)
    assert first.activated is False          # disengaged by its own kill


def test_a_cell_does_not_activate_on_a_target_killed_earlier_in_the_step():
    np.random.seed(0)
    m = _module(_cancer([100, 100, 50]))
    _engaged_nk(m, [102, 100, 50])
    _step(m)                           # the contact begins
    bystander = m.immune_cells[m.add_immune_cell([94, 100, 50], "NK_cell")]
    _step(m)                           # it kills; the bystander senses next
    assert m.total_kills == 1
    assert bystander.activated is False
    assert m.total_activations == 0          # the engaged cell was set up by hand


def test_the_gpu_patched_update_kills_once_too():
    from cognisom.gpu.orchestrator import GPUOrchestrator

    class _Engine:
        def __init__(self, modules):
            self.modules = modules

    np.random.seed(0)
    m = _module(_cancer([100, 100, 50]))
    _engaged_nk(m, [102, 100, 50])
    _engaged_nk(m, [98, 100, 50])
    with contextlib.redirect_stdout(io.StringIO()):
        GPUOrchestrator(force_cpu=True, accelerate_immune=True).accelerate(
            _Engine({"immune": m}))
    _step(m)
    _step(m)
    assert m.total_kills == 1
    assert len(_killed_events(m)) == 1
