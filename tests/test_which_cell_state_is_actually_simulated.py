"""Eleven state variables per cell. Four of them move.

``N_CELL_STATE = 12`` and the comment beside it names the columns:
oxygen, glucose, atp, lactate, age, phase, mhc1, division_timer,
damage, apoptotic_signal, migration_speed, reserved.

Running the engine with mechanics, diffusion, the ODE module and the
cell cycle all enabled, over 1,200 cells and six simulated hours:

    0  oxygen            varies   sampled from the diffusion field
    1  glucose           varies
    2  atp               set to 1.0 at init, read, never updated
    3  lactate           no writer
    4  age               varies
    5  phase             no writer
    6  mhc1              no writer
    7  division_timer    varies
    8  damage            no writer
    9  apoptotic_signal  no writer
    10 migration_speed   no writer

So the tissue simulation is mechanics, division on a timer, and two
fields. There is no metabolism -- ATP is an initial condition that is
consumed and never produced, and lactate is never written, so there is
no Warburg effect to model. There is no cell cycle: cells divide, but
``phase`` never advances. There is no death: nothing writes damage or
apoptotic_signal and every cell survives every run. And nothing writes
``mhc1``, which on a cancer immunology platform is the variable the
whole antigen-presentation story would run through.

This test does not assert that is acceptable. It records which columns
are live so the number cannot quietly go down, and so that making one
live is a visible change rather than an invisible one.

UPDATE, 2026-10-21. ``atp``, ``lactate`` and ``apoptotic_signal`` now
have writers: ``_step_metabolism`` reads each cell's local oxygen and
glucose and asks the FBA model in ``cognisom.gpu.fba_solver`` what is
achievable there. ``damage`` always had one, and the no-writer check
below missed it -- see ``test_a_dead_column_has_no_writer_anywhere``.

The LIVE set below is UNCHANGED, and deliberately so. In this
well-perfused default scenario the new columns are written and
constant: the ATP charge is 1.0 and lactate is 0 everywhere, because a
cell with oxygen to spare respires and does not ferment. That is the
physiology, not a dead column, and the honest way to show the coupling
works is a second scenario rather than a default tuned until the
numbers move. ``test_metabolism_moves_the_new_columns_under_hypoxia``
is that scenario.
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

COLUMNS = ["oxygen", "glucose", "atp", "lactate", "age", "phase", "mhc1",
           "division_timer", "damage", "apoptotic_signal",
           "migration_speed", "reserved"]

#: Columns that change over a run today. Raising this is the point of
#: the work; lowering it silently is what the test exists to stop.
LIVE = {"oxygen", "glucose", "age", "division_timer"}


@pytest.fixture(scope="module")
def run_state():
    from cognisom.core.tissue_config import TissueScaleConfig
    from cognisom.core.tissue_engine import TissueSimulationEngine

    cfg = TissueScaleConfig(
        n_cells=600, grid_shape=(16, 16, 16),
        tissue_size_um=(160., 160., 160.), n_gpus=1, dt=0.05,
        duration=4.0, seed=11, enable_ode=True, enable_diffusion=True,
        enable_cell_cycle=True, enable_mechanics=True)
    with contextlib.redirect_stdout(io.StringIO()):
        engine = TissueSimulationEngine(cfg)
        engine.initialize()
        arrays = engine._cell_arrays[0]
        before = int(arrays.n_real)
        engine.run()
        state = np.asarray(arrays.state[:arrays.n_real])
        alive = int(np.asarray(arrays.alive[:arrays.n_real]).sum())
        after = int(arrays.n_real)
    return state, before, after, alive


def test_the_column_count_matches_the_names():
    from cognisom.gpu.tissue_distributor import N_CELL_STATE

    assert N_CELL_STATE == len(COLUMNS) == 12


def test_exactly_the_recorded_columns_are_live(run_state):
    state, _b, _a, _alive = run_state
    live = {name for i, name in enumerate(COLUMNS)
            if state[:, i].std() > 1e-9}
    assert live == LIVE, (
        f"live columns changed: now {sorted(live)}, recorded {sorted(LIVE)}. "
        "If something was wired up, raise LIVE and say so in the docstring.")


#: Columns nothing assigns. ``damage`` was in this list and should not have
#: been: it has been written since before this test existed, at
#: ``state[hypoxic, damage_col] += dt * 0.5``. The old check was a regex for a
#: literal ``[:, 8] =``, which a named column constant and an augmented
#: assignment both slip past, so the test passed for the wrong reason -- the
#: same way a grep for "set_parameter" once matched a docstring saying there
#: was none. The check below walks the AST and resolves integer-valued names.
UNWRITTEN = ["mhc1", "migration_speed", "phase"]


def _column_writers(index: int):
    """Every assignment to column ``index`` of a cell-state array.

    Resolves names bound to integer literals anywhere in the file, so a
    writer indexing by ``damage_col`` is found. Collecting bindings file-wide
    rather than per scope can only over-match, which errs towards reporting a
    writer that is not there -- the safe direction for a test whose claim is
    that there is none.
    """
    import ast

    ARRAYS = {"local_state", "state", "cell_state", "st", "full_state"}
    found = []
    for path in (REPO_ROOT / "cognisom").rglob("*.py"):
        if "test" in path.name:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:                                # pragma: no cover
            continue
        ints = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                    and isinstance(node.targets[0], ast.Name) \
                    and isinstance(node.value, ast.Constant) \
                    and isinstance(node.value.value, int):
                ints[node.targets[0].id] = node.value.value

        def resolves_to(expr):
            if isinstance(expr, ast.Constant):
                return expr.value
            if isinstance(expr, ast.Name):
                return ints.get(expr.id)
            return None

        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AugAssign):
                targets = [node.target]
            for target in targets:
                if not isinstance(target, ast.Subscript):
                    continue
                base = target.value
                if not (isinstance(base, ast.Name) and base.id in ARRAYS):
                    continue
                key = target.slice
                parts = key.elts if isinstance(key, ast.Tuple) else [key]
                if any(resolves_to(part) == index for part in parts):
                    found.append(f"{path.name}:{node.lineno}")
    return found


@pytest.mark.parametrize("name", UNWRITTEN)
def test_a_dead_column_has_no_writer_anywhere(name):
    """Not merely unchanging -- there is no code that assigns it."""
    index = COLUMNS.index(name)
    writers = _column_writers(index)
    assert not writers, (
        f"{name} (column {index}) is assigned at {writers}; if it was wired "
        "up, take it out of UNWRITTEN and say so in the docstring")


@pytest.mark.parametrize("name", ["atp", "lactate", "apoptotic_signal",
                                  "damage"])
def test_a_column_claimed_live_really_is_assigned(name):
    """The other direction, so the AST check cannot pass by finding nothing."""
    assert _column_writers(COLUMNS.index(name)), (
        f"{name} is supposed to have a writer and the AST walk found none, "
        "so the walk is broken rather than the code")


def test_metabolism_moves_the_new_columns_under_hypoxia():
    """The coupling, shown where there is something to show.

    The default scenario is well perfused, so a correct metabolism leaves ATP
    at 1.0 and lactate at 0. Widening the vessel spacing and raising the
    oxygen demand produces an anoxic core, and there the columns move: cells
    ferment, lactate rises, and prolonged hypoxia accumulates damage.
    """
    from cognisom.core.tissue_config import TissueScaleConfig
    from cognisom.core.tissue_engine import TissueSimulationEngine

    cfg = TissueScaleConfig(
        n_cells=800, grid_shape=(16, 16, 16),
        tissue_size_um=(160., 160., 160.), n_gpus=1, dt=0.05,
        duration=6.0, seed=11, enable_ode=True, enable_diffusion=True,
        enable_cell_cycle=True, enable_mechanics=True)
    # x5000 is far above any physiological rate, and is here to make an
    # anoxic core form in a 16^3 grid within a few seconds. The point
    # being demonstrated is the COUPLING -- that the columns respond to
    # the field at all -- not the calibration, which is open.
    cfg.fields["oxygen"].uptake_per_cell_per_hour *= 5000.0
    original = TissueSimulationEngine.VESSEL_SPACING_UM
    TissueSimulationEngine.VESSEL_SPACING_UM = 160.0
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            engine = TissueSimulationEngine(cfg)
            engine.initialize()
            engine.run()
            arrays = engine._cell_arrays[0]
            state = np.asarray(arrays.state[:arrays.n_real])
    finally:
        TissueSimulationEngine.VESSEL_SPACING_UM = original

    oxygen = state[:, COLUMNS.index("oxygen")]
    lactate = state[:, COLUMNS.index("lactate")]
    damage = state[:, COLUMNS.index("damage")]

    assert oxygen.min() < 0.02, (
        f"no cell got hypoxic (min O2 {oxygen.min():.5f}), so this scenario "
        "cannot demonstrate anything about metabolism under hypoxia")
    assert lactate.max() > 0.0, "nothing fermented in an anoxic core"
    assert lactate.std() > 1e-9, "lactate is uniform, so it is not coupled"
    assert damage.max() > 0.0, "prolonged hypoxia accumulated no damage"


def test_atp_is_computed_and_happens_to_be_full_here(run_state):
    """1.0 for a different reason than before, and that distinction matters.

    It used to be 1.0 because nothing wrote it. It is now 1.0 because
    ``_step_metabolism`` asked the flux model what this cell can do and the
    answer, for a cell with oxygen and glucose to spare, is "everything it
    needs". The value is the same and the meaning is not, which is why this
    test changed rather than being deleted. Under hypoxia it moves -- see
    ``test_metabolism_moves_the_new_columns_under_hypoxia``.
    """
    state, _b, _a, _alive = run_state
    atp = state[:, COLUMNS.index("atp")]
    assert np.allclose(atp, 1.0), (
        f"ATP charge is {atp.min():.3f}-{atp.max():.3f} in a well-perfused "
        "scenario; it should be saturated, so either the substrate levels or "
        "the flux model have moved")
    assert _column_writers(COLUMNS.index("atp")), (
        "and it must be computed rather than merely initialised")


def test_nothing_dies_in_well_perfused_tissue(run_state):
    """Death paths exist now, and must not fire where tissue is healthy.

    This used to pass because nothing could kill a cell. Both routes are live
    -- damage under prolonged hypoxia, and an apoptotic signal when the flux
    problem is infeasible -- so the claim is now the stronger one: a
    well-perfused tissue stays alive. The companion is
    ``test_vascular_oxygen_delivery``, which pins the same requirement from
    the other side.
    """
    _state, before, after, alive = run_state
    assert alive == after, "cells died in tissue that is not oxygen-limited"
    assert after >= before


def test_the_oxygen_field_is_effectively_flat(run_state):
    """Cells sample it, but the configured uptake is far below a
    measured OCR and there is no intravascular boundary condition, so
    no gradient forms and hypoxia cannot be reached."""
    state, _b, _a, _alive = run_state
    oxygen = state[:, COLUMNS.index("oxygen")]
    spread = oxygen.std() / oxygen.mean()
    assert spread < 0.02, (
        f"oxygen now varies by {spread:.1%} across the tissue; if a "
        "gradient forms this test should become its assertion")
