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


@pytest.mark.parametrize("name", ["mhc1", "damage", "apoptotic_signal",
                                  "migration_speed", "lactate", "phase"])
def test_a_dead_column_has_no_writer_anywhere(name):
    """Not merely unchanging -- there is no code that assigns it."""
    import re

    index = COLUMNS.index(name)
    # Scoped to a CELL STATE array. An unanchored "[:, 3] =" also matches
    # ode_module's y0[:, 3], which is AR-DHT in a different array with a
    # different layout -- a coincidence that would read as a writer.
    pattern = re.compile(
        r"\b(?:local_state|state|cell_state|st)\s*\[\s*:\s*,\s*"
        + str(index) + r"\s*\]\s*=")
    hits = []
    for path in (REPO_ROOT / "cognisom").rglob("*.py"):
        if "test" in path.name:
            continue
        for line in path.read_text(errors="ignore").splitlines():
            if pattern.search(line):
                hits.append(f"{path.name}: {line.strip()[:70]}")
    assert not hits, f"{name} now has a writer: {hits}"


def test_atp_is_an_initial_condition_that_is_only_consumed(run_state):
    """Written once at distribution, read by the engine, never updated.
    A metabolism that cannot produce ATP is not a metabolism."""
    state, _b, _a, _alive = run_state
    atp = state[:, COLUMNS.index("atp")]
    assert np.allclose(atp, 1.0), (
        "ATP has moved off its initial value; metabolism may now be "
        "running and this test is out of date")


def test_nothing_dies(run_state):
    """No writer for damage or apoptotic_signal, so no cell death."""
    _state, before, after, alive = run_state
    assert alive == after
    assert after >= before, "cells were lost without a death pathway"


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
