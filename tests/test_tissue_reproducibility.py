"""A tissue run is a function of its configuration, seed included.

The initial tissue and daughter placement were already drawn from the
engine's own seeded generator, but the mechanics random force came from the
global ``np.random`` stream. So two runs of one configuration diverged, and
anything else in the process that drew a number changed the trajectory --
including the visualisation subsample, which used ``np.random.choice`` on the
same stream whenever a snapshot held more cells than ``max_viz_cells``, so
the snapshot interval changed the physics.

Every stream now derives from ``TissueScaleConfig.seed``: layout and
division, mechanics noise (one stream per partition, addressed by partition
id) and the visualisation subsample, each separate.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from cognisom.core.tissue_config import TissueScaleConfig
from cognisom.core.tissue_engine import TissueSimulationEngine

STEPS = 8


def _cfg(**kw):
    base = dict(n_cells=800, grid_shape=(20, 20, 20), tissue_size_um=(200.0, 200.0, 200.0),
                n_gpus=2, dt=0.05, duration=1.0, enable_ode=False,
                snapshot_interval_steps=1000, max_viz_cells=100_000)
    base.update(kw)
    return TissueScaleConfig(**base)


def _digest(eng) -> str:
    h = hashlib.sha256()
    for ca in eng.cell_arrays:
        for arr in (ca.positions, ca.radii, ca.state, ca.cell_types, ca.alive):
            h.update(np.ascontiguousarray(eng._backend.to_numpy(arr)[:ca.n_real]).tobytes())
    return h.hexdigest()


def _run(cfg, disturb=None) -> str:
    eng = TissueSimulationEngine(cfg)
    eng.initialize()
    for i in range(STEPS):
        if disturb is not None:
            disturb(i)
        eng.step()
    return _digest(eng)


def test_the_same_seed_gives_the_same_run_whatever_else_draws():
    np.random.seed(1)
    first = _run(_cfg(seed=11))
    np.random.seed(2)                                   # a different global state ...
    second = _run(_cfg(seed=11),
                  disturb=lambda i: np.random.random(i + 1))   # ... and draws in between
    assert first == second


def test_a_different_seed_gives_a_different_run():
    assert _run(_cfg(seed=11)) != _run(_cfg(seed=12))


def test_how_often_snapshots_are_taken_does_not_change_the_physics():
    """800 cells against a 100-cell visualisation cap, so every snapshot
    subsamples -- which used to draw from the mechanics noise stream."""
    every_step = _run(_cfg(seed=5, snapshot_interval_steps=1, max_viz_cells=100))
    never = _run(_cfg(seed=5, snapshot_interval_steps=1000, max_viz_cells=100))
    assert every_step == never


def test_a_run_leaves_the_global_stream_alone():
    np.random.seed(77)
    before = np.random.get_state()
    _run(_cfg(seed=3, snapshot_interval_steps=1, max_viz_cells=100))
    after = np.random.get_state()
    assert before[1].tolist() == after[1].tolist() and before[2] == after[2]


def test_the_seed_travels_with_the_config_json():
    assert TissueScaleConfig.from_json(_cfg(seed=42).to_json()).seed == 42


def test_partition_streams_do_not_depend_on_visiting_order():
    """A partition's noise is addressed by its id, not by how many draws
    the partitions before it made."""
    from cognisom.gpu.multi_gpu_mechanics import MultiGPUCellMechanics

    def draws(order):
        mech = MultiGPUCellMechanics.__new__(MultiGPUCellMechanics)
        mech._seed = np.random.SeedSequence(9)
        mech._noise_rngs = {}
        return {g: mech._partition_noise(np, g, 4).tolist() for g in order}

    assert draws([0, 1, 2]) == draws([2, 0, 1])
