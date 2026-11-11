"""An immune cell must be able to get to every cell of the tissue it is in.

Why this file exists
--------------------
The immune module kept its cells inside one hard-coded box, 20-180 x
20-180 x 20-80 um, sized for the ring the engine first seeded. The section
layout is 243 um wide and its cells reach 0-260 um, so on it the
infiltrate could not enter the outer part of the tissue at all.

What it looked like, on the public demo (section, seed 7, 72 h): 92 kills
in the first 15 hours and then none. All 120 immune cells sat pinned
against the box wall, each locked on one of five cancer cells 5.1 um away
with a 5.0 um kill radius. Never in contact, so never released. The
"quiet" last two thirds of the recording was this, not biology, and a
viewer spotted it before any test did.

So the property is asserted directly: wherever a tissue cell is, an immune
cell is allowed to be.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from cognisom.dashboard.engine_runner import EngineRunner
from cognisom.modules.immune_module import ImmuneModule


def _stepped(tissue: str, steps: int = 2):
    runner = EngineRunner(dt=0.05, duration=1.0, scenario="Immunotherapy",
                          tissue=tissue)
    with contextlib.redirect_stdout(io.StringIO()):
        runner.build()
        for _ in range(steps):
            runner.engine.step()
    return runner.engine.modules["immune"], runner.engine.modules["cellular"]


@pytest.mark.parametrize("tissue", ["section", "slab"])
def test_every_tissue_cell_is_inside_the_box_the_infiltrate_roams(tissue):
    immune, cellular = _stepped(tissue)
    lo, hi = immune._roaming_box()
    positions = np.array([c.position for c in cellular.cells.values()])
    outside = np.any((positions < lo) | (positions > hi), axis=1)
    assert not outside.any(), (
        f"{int(outside.sum())} of {len(positions)} {tissue} cells lie "
        f"outside the box immune cells are confined to ({lo.round(1)} to "
        f"{hi.round(1)}); no effector can ever reach them")


def test_the_section_is_wider_than_the_old_fixed_box():
    """The guard on the premise: if the section ever fits inside the
    ring's box, the test above stops proving anything."""
    _immune, cellular = _stepped("section", steps=1)
    positions = np.array([c.position for c in cellular.cells.values()])
    lo, hi = ImmuneModule.WINDOW_ROAMING_BOX
    assert np.any((positions < lo) | (positions > hi))


def test_an_effector_can_stand_on_a_cell_at_the_tissue_edge():
    """The failure itself: an effector sent to the outermost cancer cell
    must be able to close to contact rather than stop at a wall."""
    immune, cellular = _stepped("section")
    cancer = [c for c in cellular.cells.values()
              if c.cell_type == "cancer" and c.alive]
    lo, hi = immune._roaming_box()
    for cell in cancer:
        reachable = np.clip(cell.position, lo, hi)
        assert np.linalg.norm(reachable - cell.position) == 0.0


def test_the_ring_keeps_the_box_it_always_had():
    """The ring presets are pinned across the suite and fit inside the
    historical box, so they are deliberately left where they were."""
    immune, _cellular = _stepped("ring", steps=1)
    lo, hi = immune._roaming_box()
    assert np.array_equal(lo, [20.0, 20.0, 20.0])
    assert np.array_equal(hi, [180.0, 180.0, 80.0])
