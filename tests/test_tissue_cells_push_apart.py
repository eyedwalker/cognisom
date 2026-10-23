"""Cell mechanics, which was registered nowhere and could not have run.

Five things were wrong with the dormant module, none of which could show
while nothing registered it:

* its physics core's Warp kernels would not COMPILE on the installed
  Warp -- a nested function with `nonlocal` inside the Brownian kernel,
  and Warp compiles a module as a whole, so every kernel failed with
  Brownian forces on or off; the numpy path was never reached because
  the CPU backend reports itself available;
* it indexed the cellular module's dict of cells like a list;
* it subscribed to "CELL_CREATED", which the bus does not have, and to
  "CELL_DIED"/"CELL_DIVIDED" in a case the bus never emits;
* it derived a radius from a `volume` the cells do not carry;
* an ndarray position fell through an isinstance test into dict.get().

It is wired opt-in: its stiffnesses are unsourced placeholders, and on
by default it would move every cell in every preset on a guess.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import numpy as np
import pytest

from cognisom.core.event_bus import EventBus, EventTypes
from cognisom.modules.cell_mechanics_module import CellMechanicsModule


class _Cell:
    def __init__(self, cell_id, position, cell_type="normal"):
        self.cell_id = cell_id
        self.position = np.asarray(position, dtype=np.float32)
        self.cell_type = cell_type
        self.alive = True


class _Cellular:
    """The shape the real module has: a dict keyed by id."""

    def __init__(self, cells):
        self.cells = {c.cell_id: c for c in cells}


def _module(cells, **config):
    module = CellMechanicsModule(config)
    bus = EventBus()
    module.set_event_bus(bus)
    with redirect_stdout(io.StringIO()):
        module.initialize()
    module.set_cellular_module(_Cellular(cells))
    for c in cells:
        module.add_cell(c.cell_id)
    return module, bus


def _overlaps(cells, radius=5.0):
    pos = np.array([c.position for c in cells], dtype=float)
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    iu = np.triu_indices(len(cells), 1)
    return int((d[iu] < 2 * radius).sum())


# ── the physics core steps at all ───────────────────────────────────


def test_the_kernels_compile_and_overlapping_cells_separate():
    from cognisom.physics.cell_mechanics import (
        CellMechanics, CellMechanicsConfig, ForceType)

    m = CellMechanics(n_cells=2, config=CellMechanicsConfig(force_types=ForceType.ALL))
    m.set_positions(np.array([[0, 0, 0], [5, 0, 0]], np.float32))
    m.set_radii(np.full(2, 6.0, np.float32))
    m.set_cell_types(np.zeros(2, np.int32))
    for _ in range(10):
        m.step(0.005)
    gap = float(np.linalg.norm(m.get_positions()[1] - m.get_positions()[0]))
    assert gap > 5.0
    assert m._warp_failed is None, m._warp_failed  # Warp, if present, must not have died


def test_a_population_smaller_than_the_allocation_still_steps_on_warp():
    """The module allocates once and syncs whatever is alive. Before the
    buffers resized together, a 90-cell population on a 1000-row solver
    launched kernels at dim 1000 over 90 positions; Warp died on every
    runner step and the silent fallback hid it behind 'numpy'."""
    from cognisom.physics.cell_mechanics import CellMechanics, CellMechanicsConfig

    m = CellMechanics(n_cells=1000, config=CellMechanicsConfig())
    rng = np.random.default_rng(0)
    m.set_positions(rng.uniform(0, 200, (90, 3)).astype(np.float32))
    m.set_radii(np.full(90, 5.0, np.float32))
    m.set_cell_types(np.zeros(90, np.int32))
    assert m.n_cells == 90
    m.step(0.005)
    assert m._warp_failed is None, m._warp_failed
    assert m.get_positions().shape == (90, 3) and m.get_forces().shape == (90, 3)
    # and growing again, as a division would
    m.set_positions(rng.uniform(0, 200, (91, 3)).astype(np.float32))
    m.set_radii(np.full(91, 5.0, np.float32))
    m.set_cell_types(np.zeros(91, np.int32))
    m.step(0.005)
    assert m._warp_failed is None and m.get_positions().shape == (91, 3)


def test_a_warp_failure_falls_back_to_numpy_rather_than_raising(monkeypatch):
    from cognisom.physics.cell_mechanics import CellMechanics, CellMechanicsConfig

    m = CellMechanics(n_cells=2, config=CellMechanicsConfig())
    m.set_positions(np.array([[0, 0, 0], [5, 0, 0]], np.float32))
    m.set_radii(np.full(2, 6.0, np.float32))
    m.set_cell_types(np.zeros(2, np.int32))
    if not m._use_warp:
        pytest.skip("no Warp on this machine; nothing to fall back from")

    def boom(dt):
        raise RuntimeError("kernel refused")
    monkeypatch.setattr(m, "_step_warp", boom)
    m.step(0.005)
    assert m._use_warp is False and "kernel refused" in (m._warp_failed or "")
    gap = float(np.linalg.norm(m.get_positions()[1] - m.get_positions()[0]))
    assert gap > 5.0, "the numpy step did not run"


# ── the module against a dict population ────────────────────────────


def test_a_dict_of_cells_is_read_and_written_back():
    cells = [_Cell(10, (0, 0, 0)), _Cell(11, (4, 0, 0)), _Cell(12, (60, 0, 0))]
    module, _ = _module(cells)
    before = _overlaps(cells)
    assert before == 1
    for _ in range(5):
        module.update(0.05)
    assert _overlaps(cells) == 0, "the overlapping pair did not separate"
    assert isinstance(cells[0].position, np.ndarray)
    assert np.allclose(cells[2].position, (60, 0, 0), atol=1e-3), "a lone cell moved"
    state = module.get_state()
    assert state["n_active"] == 3 and state["total_displacement_um"] > 0


def test_a_dead_cell_is_left_alone():
    cells = [_Cell(1, (0, 0, 0)), _Cell(2, (4, 0, 0))]
    cells[1].alive = False
    module, _ = _module(cells)
    module.update(0.05)
    assert np.allclose(cells[1].position, (4, 0, 0))
    assert module.get_state()["n_active"] == 1


def test_division_and_death_arrive_under_the_names_the_bus_uses():
    cells = [_Cell(1, (0, 0, 0))]
    module, bus = _module(cells)
    bus.emit(EventTypes.CELL_DIVIDED, {"cell_id": 1, "daughter_id": 2,
                                       "cell_type": "normal", "position": [0, 0, 0]})
    bus.process_events()
    assert 2 in module._cell_id_to_index
    bus.emit(EventTypes.CELL_DIED, {"cell_id": 1, "cell_type": "normal",
                                    "cause": "test", "position": [0, 0, 0]})
    bus.process_events()
    assert 1 not in module._cell_id_to_index


def test_chemotaxis_and_brownian_are_off_for_tissue_cells():
    module, _ = _module([_Cell(1, (0, 0, 0))])
    assert module.module_config.enable_chemotaxis is False
    assert module.module_config.enable_brownian is False


# ── the runner: opt-in, and it changes state when opted in ─────────


def _runner(on):
    from cognisom.dashboard.engine_runner import EngineRunner

    np.random.seed(5)
    enabled = {"cellular": True, "spatial": True, "vascular": True,
               "immune": False, "lymphatic": False, "molecular": False,
               "epigenetic": False, "circadian": False, "morphogen": False,
               "cell_mechanics": on}
    runner = EngineRunner(dt=0.05, duration=1.0, scenario="Baseline",
                          modules_enabled=enabled)
    with redirect_stdout(io.StringIO()):
        runner.build()
        runner.run()
    return runner


def test_it_is_off_unless_asked_for():
    from cognisom.dashboard.engine_runner import EngineRunner

    runner = EngineRunner(dt=0.05, duration=0.1, scenario="Baseline")
    assert runner.modules_enabled.get("cell_mechanics") is False


def test_opting_in_separates_the_seeded_overlaps():
    """Baseline seeds cells at random in a box; some overlap. With the
    module on they push apart; off, they stay where they were put."""
    off, on = _runner(False), _runner(True)
    cells_off = [c for c in off.engine.modules["cellular"].cells.values() if c.alive]
    cells_on = [c for c in on.engine.modules["cellular"].cells.values() if c.alive]
    assert _overlaps(cells_on) < _overlaps(cells_off), (
        f"overlaps on {_overlaps(cells_on)} vs off {_overlaps(cells_off)}")
    state = on.engine.modules["cell_mechanics"].get_state()
    assert state["n_active"] > 0 and state["total_displacement_um"] > 0
    core = on.engine.modules["cell_mechanics"]._mechanics
    assert core._warp_failed is None, core._warp_failed
