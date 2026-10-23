"""The lymphatic module ran in every preset and changed nothing.

A 24 h Baseline with it on and off differed only in a drainage counter
nobody reads. When its two stochastic events did fire they were wrong:
a cancer cell that 'entered a lymphatic' was counted as a metastasis and
left in place, alive and dividing; and both rates were rolled at five
sample points per vessel, so a cell near the middle of a vessel drew up
to five times a step. Two handlers subscribed to events and did nothing.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import numpy as np

from cognisom.core.event_bus import EventBus, EventTypes
from cognisom.modules.lymphatic_module import LymphaticModule, LymphaticVessel


class _Cell:
    def __init__(self, cell_id, position, cell_type="cancer"):
        self.cell_id = cell_id
        self.position = np.asarray(position, dtype=np.float32)
        self.cell_type = cell_type
        self.alive = True
        self.mutations = []


class _Cellular:
    """The two things the lymphatic module uses: the dict, and the exit."""

    def __init__(self, cells):
        self.cells = {c.cell_id: c for c in cells}
        self.departed = []
        self.total_departed = 0

    def depart_cell(self, cell_id, destination="lymphatic"):
        self.cells[cell_id].alive = False
        self.departed.append((cell_id, destination))
        self.total_departed += 1
        del self.cells[cell_id]


class _Immune:
    def __init__(self, cells):
        self.immune_cells = {c.cell_id: c for c in cells}


class _ImmuneCell:
    def __init__(self, cell_id, position, activated=True):
        self.cell_id = cell_id
        self.position = np.asarray(position, dtype=np.float32)
        self.cell_type = "t_cell"
        self.activated = activated
        self.in_blood = False


def _module(n_vessels=1, **config):
    module = LymphaticModule({"n_vessels": n_vessels, **config})
    bus = EventBus()
    module.set_event_bus(bus)
    with redirect_stdout(io.StringIO()):
        module.initialize()
    # one straight vessel along x, so geometry is legible
    module.vessels.clear()
    module.add_vessel(np.array([0.0, 0.0, 0.0]), np.array([30.0, 0.0, 0.0]))
    return module, bus


def test_distance_is_to_the_segment_not_to_sample_points():
    v = LymphaticVessel(0, np.array([0.0, 0.0, 0.0]), np.array([30.0, 0.0, 0.0]))
    assert v.distance_to(np.array([15.0, 4.0, 0.0])) == 4.0
    assert v.distance_to(np.array([40.0, 0.0, 0.0])) == 10.0      # past the end
    assert v.distance_to(np.array([-3.0, 4.0, 0.0])) == 5.0       # before the start


def test_one_cell_near_a_vessel_draws_once_per_step(monkeypatch):
    """With five sample points a cell at the vessel's middle was within
    20 um of all five and rolled five times. Count the rolls."""
    module, _ = _module()
    cell = _Cell(1, (15.0, 5.0, 0.0))
    module.set_cellular_module(_Cellular([cell]))
    rolls = []
    monkeypatch.setattr(np.random, "random", lambda: rolls.append(1) or 1.0)
    module.update(0.05)
    assert len(rolls) == 1


def test_an_intravasated_cell_leaves_the_tissue(monkeypatch):
    module, bus = _module(metastasis_probability=1.0)
    near, far = _Cell(1, (15.0, 5.0, 0.0)), _Cell(2, (15.0, 80.0, 0.0))
    cellular = _Cellular([near, far])
    module.set_cellular_module(cellular)
    seen = []
    bus.subscribe(EventTypes.METASTASIS_OCCURRED, lambda d: seen.append(d))
    monkeypatch.setattr(np.random, "random", lambda: 0.0)
    module.update(1.0)
    bus.process_events()
    assert cellular.departed == [(1, "lymphatic")]
    assert 1 not in cellular.cells and far.alive
    assert module.total_metastases == 1 and [e["cell_id"] for e in seen] == [1]


def test_a_departure_is_not_a_death():
    from cognisom.modules.cellular_module import CellularModule

    cellular = CellularModule({"n_normal_cells": 0, "n_cancer_cells": 0})
    bus = EventBus()
    cellular.set_event_bus(bus)
    with redirect_stdout(io.StringIO()):
        cellular.initialize()
    cid = cellular.add_cell(position=np.array([10.0, 10.0, 10.0]), cell_type="cancer")
    died = []
    bus.subscribe(EventTypes.CELL_DIED, lambda d: died.append(d))
    cellular.depart_cell(cid, destination="lymphatic")
    bus.process_events()
    assert cid not in cellular.cells
    assert cellular.total_departed == 1 and cellular.total_deaths == 0
    assert died and died[0]["cause"] == "intravasation" and died[0]["destination"] == "lymphatic"


def test_an_activated_immune_cell_near_a_vessel_egresses(monkeypatch):
    module, bus = _module(immune_trafficking_rate=1.0)
    hot, cold = _ImmuneCell(1, (15.0, 5.0, 0.0)), _ImmuneCell(2, (15.0, 5.0, 0.0), activated=False)
    module.set_immune_module(_Immune([hot, cold]))
    monkeypatch.setattr(np.random, "random", lambda: 0.0)
    module.update(1.0)
    assert hot.in_blood and not cold.in_blood
    assert module.total_immune_trafficked == 1


def test_it_subscribes_to_nothing_it_ignores():
    module, bus = _module()
    assert not any(module.on_immune_activated if hasattr(module, "on_immune_activated") else []
                   for _ in [0])
    assert not hasattr(module, "on_immune_activated") and not hasattr(module, "on_cell_transformed")
