"""A chemokine that cells secrete, that decays, and that immune cells follow.

Before this the modules engine had a ``cytokine`` field driven by nobody
-- identically zero, skipped as idle on every update -- a "morphogen"
module assigning embryonic axis fates from organisers no cell could
sense, and a chemotaxis consumer that was a literal ``TODO`` on a module
nothing registered. Measured on the shipped Baseline run: 0.0% of cells
sensed any morphogen above e^-1, and nothing read a fate.

Now: living cancer cells are sources on the cytokine field; the field has
a first-order decay, so it has a finite reach and a regular steady-state
operator; and the immune patrol mixes its random direction with the unit
gradient. The speed is still the sourced patrol speed -- chemotaxis here
aims the motility the cell already has.

The secretion rate (1.0 field-unit/cell/h) and the chemotactic bias
(0.5) are estimates and the code says so. The half-life (30 min) is a
circulating figure. Every test below is about structure and direction,
which is what those numbers do and do not determine.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import numpy as np
import pytest

from cognisom.modules.spatial_module import (
    CHEMOKINE_DECAY_PER_HOUR, SpatialField, SpatialModule)


# ── the field: decay ────────────────────────────────────────────────


def test_decay_bounds_the_steady_state():
    """Maximum principle: with one source s and decay k, C <= s / k.

    Without decay (and no vessel, no uptake) this field would have no
    steady state at all -- the operator is singular -- and the explicit
    path would accumulate without bound.
    """
    f = SpatialField("c", (12, 12, 6), 10.0, 100.0, decay_rate=2.0)
    f.add_source(np.array([60.0, 60.0, 30.0]), 1.0)
    f.update(1.0)
    assert f.last_solver == "steady_state", f.last_solver
    assert f.concentration.max() > 0.0
    assert f.concentration.max() <= 1.0 / 2.0 + 1e-6


def test_the_explicit_path_decays_too():
    """A blob with no sources decays as exp(-k dt) on the explicit path."""
    f = SpatialField("c", (6, 6, 6), 10.0, 0.0, decay_rate=1.0)
    f.concentration[3, 3, 3] = 1.0
    f.update(0.5)
    assert f.concentration[3, 3, 3] == pytest.approx(np.exp(-0.5), rel=1e-5)


def test_decay_is_part_of_the_operator_identity():
    """Two decays, two factorisations; otherwise a changed decay would
    silently reuse the old LU."""
    a = SpatialField("c", (6, 6, 6), 10.0, 100.0, decay_rate=1.0)
    b = SpatialField("c", (6, 6, 6), 10.0, 100.0, decay_rate=2.0)
    assert a._operator_key() != b._operator_key()


def test_the_cytokine_field_ships_with_a_decay():
    sm = SpatialModule({"grid_size": (6, 6, 3)})
    with redirect_stdout(io.StringIO()):
        sm.initialize()
    assert sm.fields["cytokine"].decay_rate == pytest.approx(CHEMOKINE_DECAY_PER_HOUR)
    assert sm.fields["oxygen"].decay_rate == 0.0


# ── the source side ─────────────────────────────────────────────────


class _Cell:
    def __init__(self, cell_id, position, cell_type, alive=True):
        self.cell_id = cell_id
        self.position = np.asarray(position, dtype=np.float32)
        self.cell_type = cell_type
        self.alive = alive


class _Cellular:
    def __init__(self, cells):
        self.cells = {c.cell_id: c for c in cells}


def _secreting_module(cells, grid=(20, 20, 10)):
    from cognisom.modules.morphogen_module import MorphogenModule

    spatial = SpatialModule({"grid_size": grid})
    with redirect_stdout(io.StringIO()):
        spatial.initialize()
    morph = MorphogenModule({})
    with redirect_stdout(io.StringIO()):
        morph.initialize()
    morph.set_cellular_module(_Cellular(cells))
    morph.set_spatial_module(spatial)
    for c in cells:
        morph.add_cell(c.cell_id)
    return spatial, morph


def test_only_living_secreting_cells_are_sources():
    cells = [_Cell(1, (50, 50, 50), "cancer"), _Cell(2, (60, 50, 50), "cancer"),
             _Cell(3, (150, 150, 50), "normal"), _Cell(4, (70, 50, 50), "cancer", alive=False)]
    spatial, morph = _secreting_module(cells)
    spatial.update(spatial.update_interval)
    assert len(spatial.fields["cytokine"].sources) == 2
    assert morph.n_secreting == 2


def test_a_cell_that_dies_stops_secreting_on_the_next_update():
    """The terms are rebuilt each update, so no retraction code is needed
    -- and a one-time add_source could never have done this."""
    cells = [_Cell(1, (50, 50, 50), "cancer"), _Cell(2, (60, 50, 50), "cancer")]
    spatial, morph = _secreting_module(cells)
    spatial.update(spatial.update_interval)
    assert len(spatial.fields["cytokine"].sources) == 2
    cells[0].alive = False
    spatial.update(spatial.update_interval)
    assert len(spatial.fields["cytokine"].sources) == 1


def test_the_field_is_highest_at_the_tumour_and_falls_away():
    cells = [_Cell(i, (50 + 5 * (i % 3), 50 + 5 * (i // 3), 50), "cancer")
             for i in range(9)]
    spatial, morph = _secreting_module(cells)
    spatial.update(spatial.update_interval)
    field = spatial.fields["cytokine"]
    near = field.get_concentration_at(np.array([55.0, 55.0, 50.0]))
    far = field.get_concentration_at(np.array([170.0, 170.0, 50.0]))
    assert near > far > 0.0, (near, far)
    morph.update(0.1)
    assert morph.get_state()["field_max"] > 0.0


def test_the_gradient_points_at_the_tumour():
    cells = [_Cell(i, (40 + 5 * (i % 3), 40 + 5 * (i // 3), 50), "cancer")
             for i in range(9)]
    spatial, _ = _secreting_module(cells)
    spatial.update(spatial.update_interval)
    field = spatial.fields["cytokine"]
    here = np.array([140.0, 140.0, 50.0])
    toward = np.array([45.0, 45.0, 50.0]) - here
    toward /= np.linalg.norm(toward)
    gradient = field.get_gradient_at(here)
    assert np.linalg.norm(gradient) > 0.0
    assert float(gradient @ toward) > 0.0


def test_decay_length_is_reported_and_right():
    cells = [_Cell(1, (50, 50, 50), "cancer")]
    spatial, morph = _secreting_module(cells)
    field = spatial.fields["cytokine"]
    expected = np.sqrt(field.diffusion_coeff * 3600.0 / field.decay_rate)
    assert morph.decay_length_um() == pytest.approx(expected)
    assert 400.0 < expected < 600.0, "30 min half-life at D=100 um^2/s is ~510 um"


def test_the_old_organisers_are_gone():
    import inspect

    from cognisom.modules import morphogen_module

    source = inspect.getsource(morphogen_module)
    # Code relics, not words: the docstring is allowed to say what was
    # there, the module is not allowed to still have it.
    for relic in ("anterior_posterior", "'BMP'", "'Shh'", "'Wnt'",
                  "def determine_fate", "self._registered_sources",
                  "source_position"):
        assert relic not in source, relic


# ── the consumer ────────────────────────────────────────────────────


class _ConstantGradientField:
    """A field whose gradient is +x everywhere."""

    decay_rate = 1.0
    diffusion_coeff = 100.0

    def get_gradient_at(self, position):
        return np.array([1.0, 0.0, 0.0])

    def get_concentration_at(self, position):
        return 1.0


class _Spatial:
    def __init__(self):
        self.fields = {"cytokine": _ConstantGradientField()}


def _immune(bias):
    from cognisom.modules.immune_module import ImmuneCell, ImmuneModule

    module = ImmuneModule({"chemotaxis_bias": bias, "library_params": False})
    module.set_spatial_module(_Spatial())
    cell = ImmuneCell(cell_id=1, position=np.array([100.0, 100.0, 50.0]),
                      cell_type="T_cell")
    cell.detection_radius = 0.0
    return module, cell


def test_a_full_bias_walks_straight_up_the_gradient():
    module, cell = _immune(1.0)
    np.random.seed(0)
    start = cell.position.copy()
    for _ in range(20):
        module._patrol(cell, 0.05)
    moved = cell.position - start
    assert moved[0] > 0.0
    assert abs(moved[1]) < 1e-6 and abs(moved[2]) < 1e-6
    assert module.chemotaxis_steps == 20


def test_no_bias_is_the_old_random_walk():
    module, cell = _immune(0.0)
    assert module._chemotaxis_gradient_field() is None
    np.random.seed(0)
    start = cell.position.copy()
    for _ in range(400):
        module._patrol(cell, 0.05)
    # unbiased: the x drift over 400 steps is noise, not a march
    per_step = (cell.position[0] - start[0]) / 400.0
    assert abs(per_step) < 0.1 * 5.0 * 60.0 * 0.05
    assert module.chemotaxis_steps == 0


def test_the_bias_does_not_change_the_speed():
    """Chemotaxis aims the sourced patrol speed; it does not add to it."""
    fast, cell_fast = _immune(1.0)
    slow, cell_slow = _immune(0.0)
    np.random.seed(1)
    fast._patrol(cell_fast, 0.05)
    np.random.seed(1)
    slow._patrol(cell_slow, 0.05)
    assert np.linalg.norm(cell_fast.velocity) == pytest.approx(
        np.linalg.norm(cell_slow.velocity))


# ── end to end: the dashboard's engine ──────────────────────────────


def _runner(bias, duration=6.0, seed=3):
    from cognisom.dashboard.engine_runner import EngineRunner

    np.random.seed(seed)
    runner = EngineRunner(tissue="ring", dt=0.05, duration=duration, scenario="Baseline",
                          overrides={"immune": {"chemotaxis_bias": bias}})
    with redirect_stdout(io.StringIO()):
        runner.build()
        runner.run()
    return runner


def _cancer_alive_at(runner, hour):
    """From the snapshot history: the end state is useless here, because
    with the bias on the tumour is gone long before the run ends."""
    times = np.array([h.time for h in runner.history])
    i = int(np.argmin(np.abs(times - hour)))
    return runner.history[i].n_cancer


@pytest.fixture(scope="module")
def biased():
    return _runner(0.5)


@pytest.fixture(scope="module")
def unbiased():
    return _runner(0.0)


def test_the_runner_links_source_and_consumer_to_the_field(biased):
    mods = biased.engine.modules
    assert mods["morphogen"].spatial_module is mods["spatial"]
    assert mods["immune"].spatial_module is mods["spatial"]
    assert mods["spatial"].fields["cytokine"].decay_rate > 0.0


def test_the_field_is_driven_and_the_patrol_followed_it(biased):
    """Read over the run, not at its end: once the last cancer cell is
    killed there is nothing secreting and the field decays away."""
    assert max(h.n_secreting for h in biased.history) > 0
    assert max(h.chemokine_max for h in biased.history) > 0.0
    assert biased.engine.modules["immune"].chemotaxis_steps > 0
    assert biased.engine.modules["morphogen"].get_state()["decay_length_um"] > 0


def test_chemotaxis_finds_the_tumour_sooner(biased, unbiased):
    """The Phase 6 bar: attached, it changes state, in the right direction.

    Measured over five seeds: with the bias every one of Baseline's ten
    cancer cells is dead by 2 h; without it 2-7 are still alive at 2 h
    and 0-4 at 4 h. Baseline's tumour is a compact cluster (position
    std 5.5 um), which is what a chemokine gradient is good at.
    """
    assert _cancer_alive_at(biased, 2.0) < _cancer_alive_at(unbiased, 2.0)
    assert _cancer_alive_at(biased, 4.0) <= _cancer_alive_at(unbiased, 4.0)
