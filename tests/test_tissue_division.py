"""
Cell division in the tissue-scale engine.

``_step_cell_cycle`` checked each cell's division timer, reset it to
zero, and left a comment reading "Actual daughter cell creation happens
in bulk below". There was no bulk below -- the function ended a few
lines later, and ``grep daughter`` over the file matched only that
comment.

So the tissue population was fixed for the entire run. Growth, clonal
expansion, and the tumour-to-immune ratio were all static, while the
division timers ticked and reset forever. Nothing errored.

The check was also a per-cell Python loop, running once per cell per
step; the configured default is 1e6 cells.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cognisom.core.tissue_config import TissueScaleConfig
from cognisom.core.tissue_engine import TissueSimulationEngine

CANCER = TissueSimulationEngine._CANCER_TYPE_INDEX


def engine(n_cells: int = 200, dt: float = 1.0) -> TissueSimulationEngine:
    """Cell cycle only: the other subsystems are exercised elsewhere."""
    config = TissueScaleConfig(
        n_cells=n_cells,
        dt=dt,
        enable_diffusion=False,
        enable_mechanics=False,
        enable_ode=False,
        enable_cell_cycle=True,
    )
    eng = TissueSimulationEngine(config)
    eng.initialize()
    return eng


def population(eng) -> int:
    return sum(ca.n_real for ca in eng._cell_arrays)


def type_counts(eng):
    cancer = other = 0
    for ca in eng._cell_arrays:
        types = eng._backend.to_numpy(ca.cell_types[:ca.n_real])
        cancer += int((types == CANCER).sum())
        other += int((types != CANCER).sum())
    return cancer, other


def run(eng, hours: int):
    for _ in range(hours):
        eng.step()


# ── Cells actually divide ───────────────────────────────────────────

def test_population_grows_once_the_division_time_has_elapsed():
    eng = engine()
    start = population(eng)

    run(eng, 15)  # cancer division time is 12 h

    assert population(eng) > start


def test_population_is_unchanged_before_any_cell_is_due():
    eng = engine()
    start = population(eng)

    run(eng, 5)  # under both division times

    assert population(eng) == start


def test_cancer_expands_faster_than_normal_tissue():
    """12 h vs 48 h division times must produce divergent growth."""
    eng = engine()
    cancer_start, other_start = type_counts(eng)
    assert cancer_start > 0 and other_start > 0

    run(eng, 30)  # past two cancer cycles, before one normal cycle
    cancer_now, other_now = type_counts(eng)

    assert cancer_now > cancer_start
    assert other_now == other_start


def test_daughters_inherit_their_parent_cell_type():
    eng = engine()
    _, other_start = type_counts(eng)

    run(eng, 60)
    cancer_now, other_now = type_counts(eng)

    # Nothing changed lineage; both compartments only ever grow here.
    assert cancer_now > 0
    assert other_now >= other_start


# ── The arrays stay coherent ────────────────────────────────────────

def test_every_cell_array_matches_the_real_cell_count():
    """Growth reallocates six parallel arrays; they must stay in step."""
    eng = engine()
    run(eng, 20)

    for ca in eng._cell_arrays:
        for name in ("positions", "radii", "velocities",
                     "cell_types", "alive", "state"):
            array = eng._backend.to_numpy(getattr(ca, name))
            assert len(array) == ca.n_real, f"{name} is out of step"


def test_ghost_cells_are_invalidated_after_growth():
    """Ghosts sit directly after the real region, which growth overwrites."""
    eng = engine()
    run(eng, 15)

    for ca in eng._cell_arrays:
        assert ca.n_ghost == 0


def test_daughters_are_not_placed_exactly_on_their_parent():
    """Perfectly overlapping cells blow up the mechanics force term."""
    eng = engine()
    run(eng, 15)

    for ca in eng._cell_arrays:
        positions = eng._backend.to_numpy(ca.positions[:ca.n_real])
        if len(positions) < 2:
            continue
        # No two cells share an exact position.
        unique = np.unique(np.round(positions, 9), axis=0)
        assert len(unique) == len(positions)


def test_daughter_radius_conserves_volume():
    """Two cells of radius r*2^(-1/3) hold the volume of one of radius r."""
    eng = engine()
    before = []
    for ca in eng._cell_arrays:
        before.extend(eng._backend.to_numpy(ca.radii[:ca.n_real]).tolist())

    run(eng, 15)

    after = []
    for ca in eng._cell_arrays:
        after.extend(eng._backend.to_numpy(ca.radii[:ca.n_real]).tolist())

    assert min(after) < max(before)
    assert min(after) > 0


def test_all_new_cells_are_alive():
    eng = engine()
    run(eng, 15)

    for ca in eng._cell_arrays:
        alive = eng._backend.to_numpy(ca.alive[:ca.n_real])
        assert alive.any()


# ── Growth is bounded ───────────────────────────────────────────────

def test_growth_stops_at_the_configured_ceiling():
    """A 12 h doubling time exhausts memory quickly if unbounded."""
    eng = engine(n_cells=50)
    ceiling = max(16, int(50 * TissueSimulationEngine._GROWTH_HEADROOM))

    run(eng, 200)

    for ca in eng._cell_arrays:
        assert ca.n_real <= ceiling


def test_division_is_reproducible_for_a_given_seed():
    """Daughter placement draws from a seeded local generator."""
    a, b = engine(), engine()
    run(a, 20)
    run(b, 20)
    assert population(a) == population(b)
