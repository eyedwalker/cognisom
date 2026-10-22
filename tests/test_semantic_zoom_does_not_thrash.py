"""Semantic zoom switches representation, and switching is expensive.

``semantic_zoom.py`` changes an entity's representation TYPE with
screen coverage -- an organ becomes a surface mesh, then a
PointInstancer of cells, then molecules -- by authoring a VariantSet
selection. That recomposes the stage, which costs far more than a frame
of drawing, so a camera parked on a threshold must not flip back and
forth across it.

``ZoomThresholds.hysteresis`` has carried a 0.2 default since the
module was written and was never read: the code that should have
applied it was a ``pass`` under a comment saying "simple
implementation: just use raw thresholds". Measured, a camera jittering
+/-2% on the 1000 px anatomy/tissue boundary -- less than a hand tremor
on a trackpad -- switched representation on 206 of 400 frames.

The module had no tests at all before this file.
"""

from __future__ import annotations

import numpy as np
import pytest

from cognisom.omniverse.semantic_zoom import (
    SemanticZoomConfig, SemanticZoomController, ZoomableEntity,
    ZoomThresholds)

LEVELS = ["Hidden", "Icon", "Schematic", "Anatomy", "Tissue", "Cellular",
          "Subcellular", "Molecular", "Atomic"]


def _controller(thresholds):
    """A controller without a stage; only the pixel maths is under test."""
    controller = SemanticZoomController.__new__(SemanticZoomController)
    controller.config = SemanticZoomConfig(thresholds=thresholds)
    return controller


def _entity(thresholds, available=None):
    return ZoomableEntity(prim_path="/World/Heart",
                          available_representations=list(
                              available or LEVELS),
                          custom_thresholds=thresholds)


def _jitter_switches(hysteresis, *, spread=0.02, frames=400, seed=0):
    thresholds = ZoomThresholds(hysteresis=hysteresis)
    controller, entity = _controller(thresholds), _entity(thresholds)
    rng = np.random.default_rng(seed)
    switches = 0
    for _ in range(frames):
        pixels = thresholds.anatomy_max * (1.0 + rng.normal(0, spread))
        target = controller._pixels_to_representation(entity, pixels)
        if (entity.current_representation is not None
                and target != entity.current_representation):
            switches += 1
        entity.current_representation = target
    return switches


def test_raw_thresholds_thrash_on_a_boundary():
    """The defect, pinned so the fix cannot be quietly reverted."""
    assert _jitter_switches(0.0) > 100


def test_the_declared_hysteresis_stops_it():
    assert _jitter_switches(ZoomThresholds().hysteresis) == 0


def test_even_a_small_buffer_almost_stops_it():
    """It does not take the full 20% -- so the default is not load-bearing."""
    assert _jitter_switches(0.05) <= 5


def test_a_real_zoom_still_reaches_every_level():
    """Hysteresis must damp jitter without blocking a genuine zoom.

    A buffer applied as "never leave the current level" would pass the
    thrash test above and make the feature useless.
    """
    thresholds = ZoomThresholds(hysteresis=0.2)
    controller, entity = _controller(thresholds), _entity(thresholds)
    visited = []
    for pixels in np.geomspace(1.0, 1.0e6, 300):
        level = controller._pixels_to_representation(entity, pixels)
        entity.current_representation = level
        if not visited or visited[-1] != level:
            visited.append(level)
    assert visited == LEVELS, visited


def test_zooming_back_out_retraces_the_same_levels():
    thresholds = ZoomThresholds(hysteresis=0.2)
    controller, entity = _controller(thresholds), _entity(thresholds)
    entity.current_representation = "Atomic"
    visited = []
    for pixels in np.geomspace(1.0e6, 1.0, 300):
        level = controller._pixels_to_representation(entity, pixels)
        entity.current_representation = level
        if not visited or visited[-1] != level:
            visited.append(level)
    assert visited == list(reversed(LEVELS)), visited


# ── the band helper the hysteresis rests on ──────────────────────────


def _table(thresholds):
    controller = _controller(thresholds)
    entity = _entity(thresholds)
    # rebuild the same table _pixels_to_representation uses
    return [
        (thresholds.hidden_max, "Hidden"),
        (thresholds.icon_max, "Icon"),
        (thresholds.schematic_max, "Schematic"),
        (thresholds.anatomy_max, "Anatomy"),
        (thresholds.tissue_max, "Tissue"),
        (thresholds.cellular_max, "Cellular"),
        (thresholds.subcellular_max, "Subcellular"),
        (thresholds.molecular_max, "Molecular"),
        (float("inf"), "Atomic"),
    ], controller, entity


def test_the_bands_tile_without_gaps_or_overlap():
    """Each level's band must start where the one below it ends."""
    table, controller, _ = _table(ZoomThresholds())
    previous_high = 0.0
    for _, level in table:
        low, high = controller._band_of(level, table)
        assert low == previous_high, f"{level} starts at {low}, not {previous_high}"
        assert high > low
        previous_high = high
    assert previous_high == float("inf")


def test_an_unknown_level_cannot_be_held_on_to():
    """A level no longer in the table must not trap the entity.

    `_band_of` returns (inf, inf), and no finite pixel count lies in
    that band, so hysteresis declines to keep it.
    """
    table, controller, entity = _table(ZoomThresholds(hysteresis=0.2))
    low, high = controller._band_of("Holographic", table)
    assert low == float("inf") and high == float("inf")

    entity.current_representation = "Holographic"
    assert controller._pixels_to_representation(entity, 500.0) == "Anatomy"


def test_hysteresis_of_zero_is_the_old_behaviour():
    """The buffer is opt-out, so the raw thresholds stay reachable."""
    thresholds = ZoomThresholds(hysteresis=0.0)
    controller, entity = _controller(thresholds), _entity(thresholds)
    entity.current_representation = "Anatomy"
    just_over = thresholds.anatomy_max * 1.01
    assert controller._pixels_to_representation(entity, just_over) == "Tissue"


def test_a_negative_hysteresis_does_not_invert_the_bands():
    """Clamped at zero rather than shrinking the band below the level."""
    thresholds = ZoomThresholds(hysteresis=-0.5)
    controller, entity = _controller(thresholds), _entity(thresholds)
    entity.current_representation = "Anatomy"
    inside = thresholds.anatomy_max * 0.5
    assert controller._pixels_to_representation(entity, inside) == "Anatomy"


def test_hysteresis_never_returns_an_unavailable_level():
    """The availability constraint has to run after the buffer.

    An entity that has no Tissue variant must never be told to show
    one, however the hysteresis resolves.
    """
    thresholds = ZoomThresholds(hysteresis=0.2)
    available = [lvl for lvl in LEVELS if lvl != "Tissue"]
    controller = _controller(thresholds)
    entity = _entity(thresholds, available)
    for pixels in np.geomspace(1.0, 1.0e6, 200):
        level = controller._pixels_to_representation(entity, pixels)
        entity.current_representation = level
        assert level in available, f"{pixels:.0f} px -> {level}"


def test_the_first_frame_has_no_level_to_hold():
    """With no current representation the raw threshold decides."""
    thresholds = ZoomThresholds(hysteresis=0.2)
    controller, entity = _controller(thresholds), _entity(thresholds)
    assert entity.current_representation is None
    assert controller._pixels_to_representation(entity, 500.0) == "Anatomy"


@pytest.mark.parametrize("boundary", [
    ZoomThresholds().icon_max,
    ZoomThresholds().schematic_max,
    ZoomThresholds().anatomy_max,
    ZoomThresholds().tissue_max,
    ZoomThresholds().cellular_max,
    ZoomThresholds().subcellular_max,
])
def test_the_fix_holds_on_every_boundary(boundary):
    """Not just the one the defect was found on.

    Parked ON each threshold, because that is where thrash lives -- a
    camera at a random distance mostly sits in the middle of a band and
    would pass this whether or not the buffer worked.
    """
    thresholds = ZoomThresholds(hysteresis=0.2)
    controller, entity = _controller(thresholds), _entity(thresholds)
    rng = np.random.default_rng(int(boundary))
    centre = float(boundary)
    switches = 0
    for _ in range(300):
        pixels = centre * (1.0 + rng.normal(0, 0.02))
        target = controller._pixels_to_representation(entity, pixels)
        if (entity.current_representation is not None
                and target != entity.current_representation):
            switches += 1
        entity.current_representation = target
    assert switches <= 2, f"centre {centre:.0f} px gave {switches} switches"
