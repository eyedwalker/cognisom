"""A warning that fires where nothing is wrong is a warning people learn
to scroll past.

Why this file exists
--------------------
Every run of the section preset logged this, once per arm:

    Field 'ifn_gamma': D*dt/dx^2 = 180.0 needs 1200 explicit sub-steps
    to stay stable, above the cap of 512. Running 512 stable sub-steps
    instead, so the field is UNDER-DIFFUSED over this interval (about
    42.7% of the transport applied). At this diffusion number the field
    is effectively at steady state within one step, so an implicit or
    steady-state solve is the right method here, not explicit Euler.

Read on its own that is alarming, and it was taken at face value and
queued as a defect to fix. It is arithmetically true and practically
empty, for a reason worth keeping written down.

``ifn_gamma`` has no Dirichlet voxels: nothing holds it at a boundary
value. It is driven only by cells that are secreting it, which means
activated T and NK cells engaged on a target. Before the first such
engagement the field has NO SOURCES, so ``_steady_state_applies``
correctly refuses the steady solve -- an undriven field's steady state
is zero, and replacing a real decaying transient with zero would discard
the thing being modelled. The explicit path then runs, hits its
sub-step cap, and warns. But at that moment the field is identically
zero: there is no transient, and nothing to under-diffuse.

The moment anything secretes IFN-gamma the field becomes driven and
every later step goes through the implicit solver, which is exactly what
the warning recommends. MEASURED on the 24 h section preset, comparing
the shipped behaviour against ALSO routing the undriven-and-capped case
to the steady solve: identical to five significant figures -- field peak
2.1e-4 and mean 2.1e-5 both ways, mean MHC-I 0.31020, max 0.35645,
2,746 cancer cells, 37 NK and 3 T-cell kills, and 135 factorisations in
both. The branch changes nothing.

So the fix is to the diagnostic, not the physics: do not warn about
under-diffusing a field that holds nothing. These tests hold that line
from both sides, because the easy mistake now is to silence the warning
for a field that genuinely is carrying signal through a capped step.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from cognisom.modules.spatial_module import SpatialField


def _capped_field(name="ifn_gamma", d=50.0):
    """A field whose explicit path is guaranteed to hit the sub-step cap."""
    f = SpatialField(name=name, grid_size=(12, 12, 6), resolution=10.0,
                     diffusion_coeff=d)
    f.concentration[:] = 0.0
    return f


def test_the_preset_that_warned_really_does_hit_the_cap():
    """The arithmetic in the warning is true; this is not a test that
    the condition went away."""
    f = _capped_field()
    assert f._explicit_is_capped(0.05), (
        "ifn_gamma's diffusion number no longer exceeds the sub-step "
        "cap, so the rest of this file is testing a branch that cannot "
        "be reached -- re-measure before deleting anything")


def test_an_empty_field_does_not_warn(caplog):
    f = _capped_field()
    assert not f._carries_signal()
    with caplog.at_level(logging.WARNING):
        f.update(0.05)
    assert not any("UNDER-DIFFUSED" in r.getMessage() for r in caplog.records), (
        "warned about under-diffusing a field with nothing in it")


def test_a_field_carrying_signal_still_warns(caplog):
    """The half that must not be lost: a capped explicit step on a field
    that DOES hold something is still worth reporting."""
    f = _capped_field()
    f.concentration[6, 6, 3] = 1.0
    assert f._carries_signal()
    with caplog.at_level(logging.WARNING):
        f.update(0.05)
    assert any("UNDER-DIFFUSED" in r.getMessage() for r in caplog.records), (
        "a capped explicit step on a field carrying signal must still "
        "say so")


def test_the_floor_is_an_absolute_one_and_very_small():
    f = _capped_field()
    assert SpatialField.SIGNAL_FLOOR == 1e-9
    f.concentration[6, 6, 3] = 1e-8
    assert f._carries_signal(), (
        "the floor must not be so high that a real but faint signal "
        "counts as empty")
    f.concentration[6, 6, 3] = 1e-12
    assert not f._carries_signal()


def test_warning_still_fires_only_once_per_field(caplog):
    """The behaviour this warning already had, kept."""
    f = _capped_field()
    f.concentration[6, 6, 3] = 1.0
    with caplog.at_level(logging.WARNING):
        for _ in range(4):
            f.concentration[6, 6, 3] = 1.0
            f.update(0.05)
    n = sum(1 for r in caplog.records if "UNDER-DIFFUSED" in r.getMessage())
    assert n == 1, f"warned {n} times across four updates"


def test_an_empty_field_is_still_sub_step_CAPPED():
    """The bug this nearly shipped.

    Making the warning conditional is right; making the CLAMP
    conditional is not. Gating the whole branch on "does this field
    carry signal" left an empty capped field running its full uncapped
    sub-step count -- 57,600 iterations per update for oxygen at
    dt = 0.1 -- and the existing contract test did not fail an
    assertion, it TIMED OUT at 120 s. Quietly not clamping is a
    performance cliff with no error attached, so it gets its own test.
    """
    f = _capped_field("oxygen", d=2000.0)
    assert not f._carries_signal()
    calls = {"n": 0}
    real = f._compute_laplacian

    def counted():
        calls["n"] += 1
        return real()

    f._compute_laplacian = counted
    f.update(0.1)
    assert calls["n"] <= SpatialField.MAX_SUBSTEPS, (
        f"an empty capped field took {calls['n']} sub-steps against a "
        f"cap of {SpatialField.MAX_SUBSTEPS}")


def test_checking_for_signal_costs_nothing_once_warned():
    """The guard runs on every capped update, so it must stop scanning
    the array once it can no longer change the outcome."""
    f = _capped_field()
    f.concentration[6, 6, 3] = 1.0
    f.update(0.05)
    assert f._cfl_warned
    seen = {"n": 0}
    real = f._carries_signal

    def counted():
        seen["n"] += 1
        return real()

    f._carries_signal = counted
    for _ in range(5):
        f.update(0.05)
    assert seen["n"] == 0, (
        f"scanned the field {seen['n']} times after the warning had "
        "already fired")


def test_an_undriven_field_is_still_refused_the_steady_state():
    """The reason the branch exists at all, pinned.

    Routing an undriven field to the steady solve would replace its
    transient with zero. That is right for ifn_gamma, whose transient is
    shorter than a step, and wrong in general -- so the refusal stays and
    this test says why anyone reading the warning should not 'fix' it by
    deleting the `driven` condition.
    """
    f = _capped_field()
    f.concentration[6, 6, 3] = 1.0
    assert not f.dirichlet and not f.sources
    assert f._steady_state_applies(0.05) is False
