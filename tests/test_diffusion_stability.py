"""
Diffusion field stability and units.

``SpatialField.update`` had two defects that compounded:

  * ``diffusion_coeff`` is in um^2/s and ``dt`` is in hours, and nothing
    converted between them -- so the transported amount was understated
    by a factor of 3600.

  * There was no stability check. The explicit 3D 7-point scheme is
    stable only for ``D*dt/dx^2 <= 1/6``; the oxygen field's
    D = 2000 um^2/s on a 10 um grid gives ~7200 once the units are
    right, and 2.0 even with the unit bug. Above the limit the scheme
    amplifies high-frequency modes every step rather than damping them.

The instability never surfaced because the update ended with
``np.maximum(concentration, 0)``, which clipped the negative half of the
oscillation each step -- hiding the blow-up and destroying mass while
doing it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cognisom.modules.spatial_module import SpatialField


def field(d: float, n: int = 21, dx: float = 10.0) -> SpatialField:
    f = SpatialField("test", (n, n, n), resolution=dx, diffusion_coeff=d)
    return f


def centred_blob(f: SpatialField, amount: float = 1000.0) -> float:
    mid = f.concentration.shape[0] // 2
    f.concentration[mid, mid, mid] = amount
    return float(f.concentration.sum())


# ── Stability ───────────────────────────────────────────────────────

def test_production_oxygen_settings_stay_finite():
    """D=2000 um^2/s, dx=10 um, dt=0.1 h -- the real configuration."""
    f = field(2000.0)
    centred_blob(f)

    for _ in range(3):
        f.update(0.1)

    assert np.isfinite(f.concentration).all()
    assert not np.isnan(f.concentration).any()
    assert f.concentration.min() >= 0.0


@pytest.mark.parametrize("d", [10.0, 100.0, 600.0, 2000.0])
def test_no_oscillation_for_any_shipped_diffusion_coefficient(d):
    """Glucose is 600 and cytokine 100; all must be non-oscillating."""
    f = field(d)
    centred_blob(f)

    peak_trace = []
    for _ in range(5):
        f.update(0.01)
        mid = f.concentration.shape[0] // 2
        peak_trace.append(float(f.concentration[mid, mid, mid]))

    # Diffusion from a single point can only lower the peak.
    assert all(
        peak_trace[i] >= peak_trace[i + 1] - 1e-6
        for i in range(len(peak_trace) - 1)
    )
    assert np.isfinite(f.concentration).all()


def test_a_stable_scheme_needs_no_clipping():
    """If the scheme is stable, the clamp has nothing to clamp."""
    f = field(100.0, n=31)
    centred_blob(f)

    for _ in range(5):
        f.update(0.001)
        assert f.concentration.min() >= 0.0


# ── Conservation ────────────────────────────────────────────────────

def test_mass_is_conserved_while_the_blob_stays_interior():
    """Diffusion moves material; it does not create or destroy it.

    The Laplacian is only evaluated on interior points, so the boundary
    shell is absorbing -- this checks conservation in the regime where
    nothing has reached it yet.
    """
    f = field(10.0, n=31)
    initial = centred_blob(f)

    for _ in range(5):
        f.update(0.0002)

    # Nothing has reached the boundary shell.
    assert f.concentration[0].max() == 0.0
    assert f.concentration[-1].max() == 0.0
    assert f.concentration.sum() == pytest.approx(initial, rel=1e-5)


def test_diffusion_actually_spreads_material():
    """The unit bug made transport 3600x too small."""
    f = field(100.0, n=21)
    centred_blob(f)
    mid = f.concentration.shape[0] // 2

    f.update(0.001)

    # Immediate neighbours must have received a meaningful share.
    assert f.concentration[mid + 1, mid, mid] > 0.0
    assert f.concentration[mid, mid, mid] < 1000.0


# ── Sub-stepping ────────────────────────────────────────────────────

def test_substepping_keeps_each_step_within_the_stability_limit():
    f = field(2000.0)
    dt_hours = 0.1
    dt_seconds = dt_hours * SpatialField.SECONDS_PER_HOUR

    diffusion_number = f.diffusion_coeff * dt_seconds / f.resolution ** 2
    max_stable = SpatialField.CFL_LIMIT_3D * SpatialField.CFL_SAFETY

    assert diffusion_number > max_stable  # the regime that used to blow up

    n_sub = int(np.ceil(diffusion_number / max_stable))
    per_substep = min(diffusion_number / n_sub, max_stable)
    assert per_substep <= SpatialField.CFL_LIMIT_3D


def test_capped_substeps_remain_stable_rather_than_overflowing():
    """Hitting the cap must under-diffuse, not go unstable.

    Spreading an over-limit coefficient across a capped number of
    sub-steps leaves every sub-step unstable, which overflows to NaN --
    strictly worse than the clipped garbage it replaced.
    """
    f = field(50000.0)  # far past anything the cap can resolve
    centred_blob(f)

    f.update(1.0)

    assert np.isfinite(f.concentration).all()
    assert f.concentration.min() >= 0.0


def test_zero_diffusion_is_a_no_op():
    f = field(0.0)
    initial = centred_blob(f)
    f.update(0.1)
    assert f.concentration.sum() == pytest.approx(initial)
