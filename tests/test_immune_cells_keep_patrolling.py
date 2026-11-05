"""An immune cell that has not engaged keeps migrating.

What was wrong
--------------
``update`` gated the random walk on TWO conditions::

    if not immune_cell.activated:
        if not self._target_within_reach(immune_cell, cancer_cells):
            self._patrol(immune_cell, dt)

The second gate suppressed migration whenever ANY cancer cell sat inside
the sensing radius, whether or not the cell engaged it. In a packed
tissue section that is almost always true, so the gate was a permanent
latch: a cell that drifted next to a tumour cell it could not recognise
never moved again.

MEASURED over a 24 h Immunotherapy section run, dt = 0.05 h, 480 steps,
59,478 immune-cell-steps:

    blocked by `activated` (genuinely engaged)   10,490   17.6%
    blocked by proximity alone                   47,476   79.8%
    actually patrolled                            1,512    2.5%

So the compartment was immobile 97.5% of the time and four fifths of
that was cells parked beside targets they had already declined. Every
patrol that did run moved its cell -- 1,512 of 1,512 -- so the movement
arithmetic was never the problem. In the shipped 24 h stage the whole
immune population stopped moving for good at h 12.25, and 64 of its 123
cells stopped without ever activating again.

Why `activated` alone is the right test
---------------------------------------
``activated`` is set only where a target is RECOGNISED, and cleared only
by ``_release``, whose docstring says what that means: "the effector
disengages and patrols again". A lymphocyte does arrest on contact, but
the arrest is conditional on recognition -- a non-recognising contact is
transient, the cell detaches and keeps scanning. Overshoot, which was
the stated reason for the proximity gate, is prevented where it belongs:
by the step cap in ``_patrol``, which never lets a cell cross more than
its own sensing radius in one step.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from cognisom.modules.cellular_module import CellState
from cognisom.modules.immune_module import ImmuneModule

TARGET = 7
HERE = np.array([100.0, 100.0, 50.0], dtype=np.float32)


class _Tissue:
    def __init__(self):
        self.cells = {}

    def ecm_density_at(self, position):
        return 0.0


def _module(lineage="T_cell", **cfg):
    base = {"n_t_cells": 0, "n_nk_cells": 0, "n_macrophages": 0}
    with contextlib.redirect_stdout(io.StringIO()):
        m = ImmuneModule({**base, **cfg})
    m.set_cellular_module(_Tissue())
    cell = m.immune_cells[m.add_immune_cell(list(HERE), lineage)]
    cell.position = HERE.copy()
    return m, cell


#: Intact class I. This is a target every lineage DECLINES: NK
#: missing-self does not fire on it, and a T cell has no TCR match
#: because the cell displays no neoantigen. That is the case these
#: tests are about -- a contact the cell inspects and walks away from.
#: An MHC-NULL target is the opposite: NK and macrophages recognise it
#: and correctly arrest on it, now that detection runs before the move.
DECLINED = 1.0
ENGAGED = 0.0


def _put_target(m, *, mhc1, at=None):
    """A cancer cell right on top of the immune cell, so it is always
    inside the sensing radius -- the exact configuration that used to
    freeze the cell for the rest of the run."""
    m.cellular_module.cells = {TARGET: CellState(
        cell_id=TARGET, cell_type="cancer", phase="G1",
        position=np.array(HERE if at is None else at, dtype=np.float32),
        mhc1_expression=mhc1)}


def _displacement_over(m, cell, steps=40, dt=0.05):
    start = cell.position.copy()
    path = 0.0
    for _ in range(steps):
        before = cell.position.copy()
        m.update(dt)
        path += float(np.linalg.norm(cell.position - before))
    return float(np.linalg.norm(cell.position - start)), path


# ── the bug, stated as a test ─────────────────────────────────────


def test_a_cell_beside_a_target_it_has_not_engaged_still_migrates():
    """The regression. With a cancer cell inside its sensing radius and
    no engagement, the cell must still move; it used to be frozen."""
    np.random.seed(0)
    m, cell = _module()
    _put_target(m, mhc1=DECLINED)
    assert cell.detection_radius > 0.0
    assert np.linalg.norm(cell.position - m.cellular_module.cells[TARGET].position) \
        < cell.detection_radius, "the target must be IN reach for this test to bite"
    cell.activated = False
    net, path = _displacement_over(m, cell)
    assert path > 0.0, "a disengaged cell beside a target did not move at all"


def test_the_sensing_radius_alone_does_not_arrest_a_cell():
    """Same cell, same seed, with and without a target in reach: the
    presence of an unengaged target must not change whether it walks."""
    results = {}
    for label in ("target_in_reach", "empty_tissue"):
        np.random.seed(7)
        m, cell = _module()
        if label == "target_in_reach":
            _put_target(m, mhc1=DECLINED)
        cell.activated = False
        results[label] = _displacement_over(m, cell)[1]
    assert results["target_in_reach"] > 0.0
    assert results["empty_tissue"] > 0.0


# ── what must still hold: an ENGAGED cell stays put ───────────────


def test_an_engaged_cell_does_not_wander_off_its_target():
    """Arrest on recognition is real biology and must survive the fix."""
    np.random.seed(0)
    m, cell = _module()
    _put_target(m, mhc1=ENGAGED)
    cell.activated = True
    cell.target_cell_id = TARGET
    before = cell.position.copy()
    for _ in range(20):
        m.update(0.05)
        if not cell.activated:
            break                      # the contact resolved; stop checking
        assert np.allclose(cell.position, before), (
            "an activated cell moved away from the target it is engaging")


# ── the step cap, which is what actually prevents overshoot ───────


def test_one_step_never_crosses_more_than_the_sensing_radius():
    """The guard that lets the proximity gate go. A cell may not step
    clean over a target between two detection passes."""
    np.random.seed(3)
    # Positions are float32, so the cap lands within single-precision
    # epsilon of the radius rather than exactly on it: a 10 um reach
    # realises as 10.0000038. The tolerance is that epsilon, scaled --
    # it is not slack in the guard.
    for dt in (0.05, 0.25, 1.0):
        m, cell = _module()
        reach = float(cell.detection_radius)
        tol = reach * 1e-5
        for _ in range(30):
            before = cell.position.copy()
            m._patrol(cell, dt)
            step = float(np.linalg.norm(cell.position - before))
            assert step <= reach + tol, (dt, step, reach)


# ── the compartment moves, and by a sane amount ───────────────────


@pytest.mark.parametrize("lineage", ["T_cell", "NK_cell", "macrophage"])
def test_every_lineage_migrates_when_disengaged(lineage):
    """Each lineage declines a target for its OWN reason, so each needs a
    different target to be a declined contact rather than an engagement:

      T cell      intact class I and no neoantigen, so no TCR match
      NK cell     intact class I, so missing-self does not fire
      macrophage  not antigen-restricted at all, so class I does nothing
                  to it. It declines when POLARIZED past
                  macrophage_recognition_max_polarization (0.5): an M2
                  macrophage is pro-tumour and stops recognising targets.
                  PD-L1 is the other don't-eat-me side, but it only
                  SCALES the phagocytic probability and never reaches
                  zero, so PD-L1 = 1.0 still engages.

    Getting this wrong is informative rather than fatal: an unpolarized
    macrophage RECOGNISED this target and arrested on it, which is now
    correct behaviour and is what the first version of this test tripped
    over.
    """
    np.random.seed(11)
    m, cell = _module(lineage)
    _put_target(m, mhc1=DECLINED)
    if lineage == "macrophage":
        cell.polarization = 1.0
    cell.activated = False
    net, path = _displacement_over(m, cell, steps=40)
    assert not cell.activated, (
        f"{lineage} engaged the target, so this is not testing a declined "
        "contact -- give it a target this lineage actually declines")
    assert path > 0.0, f"{lineage} never moved"


def test_a_disengaged_cell_covers_ground_over_an_hour():
    """A property, not a pin: 20 steps of dt=0.05 h is one hour, and a
    patrolling leukocyte should cover more than one cell diameter in
    that time rather than jittering on the spot."""
    np.random.seed(5)
    m, cell = _module()
    _put_target(m, mhc1=DECLINED)
    cell.activated = False
    net, path = _displacement_over(m, cell, steps=20, dt=0.05)
    assert path > 10.0, (
        f"path {path:.2f} um in 1 h: a patrolling cell should cover at "
        "least a cell diameter. Re-measure, do not re-pin.")
