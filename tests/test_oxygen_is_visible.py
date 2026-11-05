"""The per-cell oxygen in the stage reaches the picture.

What was wrong
--------------
``/World/Cells`` carried ``cognisom:cell:oxygen`` time-sampled on every
frame, and on the shipped 24 h prostate run that attribute changed on
94 of its 96 frame-to-frame steps, over a measured range of
0.0781-0.2100. ``primvars:displayColor`` on the same instancer was
**byte-identical on all 97 frames**. So the tissue was a still image
carrying live data nobody could see: the only visible change in the
cell population was instances blinking in and out as cells divided and
died.

The fix shades each cell's own type colour by its own oxygen. It may
not pick the HUE -- the type colour is the legend, and a normal cell
has to stay peach while a cancer cell stays red -- so it scales
brightness, which leaves lineage readable and makes the gradient show.

The ramp is anchored, not stretched
-----------------------------------
0.21 is a volume fraction of air (~21 kPa) and is what every cell is
seeded at; 0.05 is the engine's OWN hypoxia cutoff, from
vascular_module's "hypoxic (< 0.05) from 25 to 50 um". Fixing the ramp
between those two means two runs shade the same way and can be
compared by eye. Stretching it to each run's own min and max would
make every run look equally hypoxic, including one that never drops.
"""

from __future__ import annotations

import numpy as np
import pytest

from cognisom.visualization.run_usd import (
    OXYGEN_FULL,
    OXYGEN_HYPOXIC,
    OXYGEN_SHADE_FLOOR,
    shade_by_oxygen,
)

#: A cancer cell's authored colour, from the stage's own appearance table.
CANCER_RGB = np.array([[[0.8, 0.26, 0.3]]], dtype="float32")

#: The oxygen the shipped 24 h prostate section run actually occupies,
#: measured over live cells on frames 0/24/48/96. MEASURED -- if this
#: moves, re-measure, do not re-pin.
SHIPPED_OXYGEN_MIN = 0.0781
SHIPPED_OXYGEN_MAX = 0.2100


def _brightness(oxygen, rgb=CANCER_RGB):
    out = shade_by_oxygen(rgb, [[oxygen]])
    return float(out[0, 0, 0] / rgb[0, 0, 0])


# ── the hue is the legend and survives ────────────────────────────


def test_shading_never_changes_the_hue():
    """Lineage is read off colour, so oxygen may only change brightness."""
    base = CANCER_RGB[0, 0] / np.linalg.norm(CANCER_RGB[0, 0])
    for oxygen in (0.0, 0.05, 0.09, 0.15, 0.21, 0.9):
        out = shade_by_oxygen(CANCER_RGB, [[oxygen]])[0, 0]
        got = out / np.linalg.norm(out)
        assert np.allclose(got, base, atol=1e-6), (oxygen, out)


def test_two_lineages_stay_distinguishable_at_the_same_oxygen():
    """A dark normal cell must not be mistakable for a bright cancer one."""
    normal = np.array([[[0.9, 0.7, 0.5]]], dtype="float32")
    cancer = CANCER_RGB
    for oxygen in (0.06, 0.12, 0.21):
        a = shade_by_oxygen(normal, [[oxygen]])[0, 0]
        b = shade_by_oxygen(cancer, [[oxygen]])[0, 0]
        a_h = a / np.linalg.norm(a)
        b_h = b / np.linalg.norm(b)
        assert np.linalg.norm(a_h - b_h) > 0.1, (oxygen, a, b)


# ── the ramp ──────────────────────────────────────────────────────


def test_more_oxygen_is_never_darker():
    grid = np.linspace(0.0, 0.3, 61)
    vals = [_brightness(o) for o in grid]
    assert all(b >= a - 1e-6 for a, b in zip(vals, vals[1:])), vals


def test_the_ends_of_the_ramp_are_the_anchors():
    """Air is full brightness; the engine's own hypoxia cutoff is the floor."""
    assert _brightness(OXYGEN_FULL) == pytest.approx(1.0, abs=1e-6)
    assert _brightness(OXYGEN_HYPOXIC) == pytest.approx(OXYGEN_SHADE_FLOOR,
                                                        abs=1e-6)


def test_the_ramp_is_clamped_at_both_ends():
    """Beyond the anchors nothing brightens further or goes black."""
    assert _brightness(0.9) == pytest.approx(1.0, abs=1e-6)
    assert _brightness(0.0) == pytest.approx(OXYGEN_SHADE_FLOOR, abs=1e-6)
    assert _brightness(-5.0) == pytest.approx(OXYGEN_SHADE_FLOOR, abs=1e-6)


def test_a_hypoxic_cell_is_still_drawn():
    """The floor is not 0: a black cell reads as an absent cell."""
    assert OXYGEN_SHADE_FLOOR > 0.2
    assert (shade_by_oxygen(CANCER_RGB, [[0.0]]) > 0.0).all()


def test_a_missing_oxygen_value_is_not_rendered_as_suffocation():
    """The writer passes OXYGEN_FULL for a cell with no recorded value."""
    assert _brightness(OXYGEN_FULL) == pytest.approx(1.0, abs=1e-6)


# ── it is actually visible on the range the runs occupy ───────────


def test_the_shipped_oxygen_range_spans_a_visible_brightness_range():
    """The point of the change: the gradient has to be SEEN, not merely
    be present. Across the oxygen the shipped run actually occupies the
    brightness spans 0.464-1.000, a 2.2x range. MEASURED."""
    lo = _brightness(SHIPPED_OXYGEN_MIN)
    hi = _brightness(SHIPPED_OXYGEN_MAX)
    assert lo == pytest.approx(0.464, abs=5e-3), lo
    assert hi == pytest.approx(1.000, abs=5e-3), hi
    assert hi / lo > 2.0, (
        f"brightness only spans {hi / lo:.2f}x over the run's own oxygen "
        "range; the gradient will not read on screen")


def test_a_degenerate_ramp_does_nothing_rather_than_dividing_by_zero():
    out = shade_by_oxygen(CANCER_RGB, [[0.1]], full=0.05, hypoxic=0.05)
    assert np.allclose(out, CANCER_RGB)


# ── shape and dtype, because this feeds Vt.Vec3fArray ─────────────


def test_shape_and_dtype_survive_for_the_usd_writer():
    """(frames, n, 3) float32 in, the same out -- Vt.Vec3fArray needs it."""
    rgb = np.zeros((4, 7, 3), dtype="float32") + 0.5
    oxygen = np.full((4, 7), 0.12)
    out = shade_by_oxygen(rgb, oxygen)
    assert out.shape == (4, 7, 3)
    assert out.dtype == np.dtype("float32")


def test_each_cell_is_shaded_by_its_own_oxygen_not_the_frame_mean():
    """Per instance, per frame -- a hypoxic core beside a fed rim."""
    rgb = np.ones((1, 3, 3), dtype="float32")
    out = shade_by_oxygen(rgb, [[0.21, 0.13, 0.05]])
    got = [float(out[0, i, 0]) for i in range(3)]
    assert got[0] > got[1] > got[2]
    assert got[0] == pytest.approx(1.0, abs=1e-6)
    assert got[2] == pytest.approx(OXYGEN_SHADE_FLOOR, abs=1e-6)
