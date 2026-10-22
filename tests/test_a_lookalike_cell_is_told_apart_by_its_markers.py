"""Nineteen entities resolve to a smooth sphere. The library knows better.

Rendering all sixty entities and looking at them showed nineteen drawn as
the same sphere, collapsing to five distinct (scale, feature-count)
combinations: a Th1, a Th17, a naive B cell and an exhausted CD8 were the
same picture. The obvious reading is that this is honest -- a naive and a
memory B cell genuinely do look alike down a microscope -- and it is
wrong. **All nineteen carry DISTINCT marker sets, none shared with
another**: Th1 is CXCR3 and T-bet, Th17 is CCR6 and RORgt, exhausted CD8
is PD-1 and LAG-3.

``profile_for`` has partitioned those into membrane, nuclear and granule
compartments since it was written, and ``scene_for`` received the result
and drew none of it. The data to tell the cells apart was already arriving
and being discarded.

**The count is a glyph, not a measurement**, and that is what makes this
safe to draw. Surface receptor copy numbers run to hundreds of thousands
per cell and this library has none of them; the one abundance it does
have, B2M from PaxDb, is five bulk samples broadcast across twenty cell
types and implies CD4+ carries 2.79x the MHC-I of CD8+. Inventing a
density per marker would encode that artefact as geometry. So every marker
gets the same count and the COLOUR carries identity, with ``sources``
recording it as ``glyph:identity_not_density``.

Colour had to be measured too. Hashing a marker name to a hue at fixed
saturation put CCR5 and CCR6 an RGB distance of 0.032 apart -- the one
pair that has to separate a Th1 from a Th17 -- because 45 markers spread
over one hue circle land a few degrees apart. Using saturation and value
as well moved that pair to 0.797 and cut the sub-0.15 pairs from 59 to 9.
"""

from __future__ import annotations

import contextlib
import io
import itertools

import numpy as np
import pytest

from cognisom.visualization.cell_profiles import markers_of, partition_markers
from cognisom.visualization.cell_scene import (
    MARKER_GLYPHS, marker_colour, scene_for)
from cognisom.visualization.morphology_usd import CELLULAR_TYPES

#: Two cells of the same shape and size whose markers are the whole
#: difference between them.
LOOKALIKES = ("CD4+ Th1 cell", "CD4+ Th17 cell")

#: Below this an eye will not separate two glyph populations at card size.
MIN_SEPARATION = 0.15


@pytest.fixture(scope="module")
def store():
    with contextlib.redirect_stdout(io.StringIO()):
        from cognisom.library.store import EntityStore
        return EntityStore()


@pytest.fixture(scope="module")
def membrane_markers(store):
    """Every membrane marker in the library, and which cells carry it."""
    carried = {}
    for etype, name, _blob in store._conn.execute(
            "SELECT entity_type, name, data_json FROM entities"):
        if etype not in CELLULAR_TYPES or etype == "tissue_type":
            continue
        entity = store.find_entity_by_name(name)
        if entity is None:
            continue
        for marker in partition_markers(markers_of(entity.to_dict()))["membrane"]:
            carried.setdefault(marker, set()).add(name)
    assert len(carried) > 30, len(carried)
    return carried


def _scene(store, name):
    entity = store.find_entity_by_name(name)
    assert entity is not None, name
    with contextlib.redirect_stdout(io.StringIO()):
        return scene_for(entity)


def test_membrane_markers_reach_the_picture(store):
    """They were partitioned and then dropped."""
    scene = _scene(store, "CD4+ Th1 cell")
    drawn = {d.name for d in scene.details}
    assert {"CXCR3", "CCR5", "CD4"} <= drawn, drawn


def test_two_cells_of_one_shape_are_told_apart(store):
    """The whole point. Same builder, same diameter, different cell."""
    a, b = (_scene(store, n) for n in LOOKALIKES)
    assert a.mesh_type == b.mesh_type
    assert a.scale_um == b.scale_um
    assert [s.name for s in a.surfaces] == [s.name for s in b.surfaces]

    only_a = {d.name for d in a.details} - {d.name for d in b.details}
    only_b = {d.name for d in b.details} - {d.name for d in a.details}
    assert only_a and only_b, (
        f"{LOOKALIKES[0]} and {LOOKALIKES[1]} draw the same populations; "
        "nothing in the picture distinguishes them")

    # and the distinguishing populations must not share a colour
    for x in only_a:
        for y in only_b:
            cx, cy = np.array(marker_colour(x)), np.array(marker_colour(y))
            assert float(np.linalg.norm(cx - cy)) > MIN_SEPARATION, (
                f"{x} and {y} are the markers that separate these two cells "
                f"and they are the same colour")


def test_the_count_is_declared_a_glyph(store):
    """A density read off this picture would be read off nothing."""
    scene = _scene(store, "CD4+ Th1 cell")
    assert scene.sources.get("membrane_markers") == "glyph:identity_not_density"
    for detail in scene.details:
        if detail.name in ("microvilli", "lytic granules"):
            continue
        assert detail.actual == MARKER_GLYPHS
        assert detail.drawn == detail.actual, (
            "a marker glyph population is never capped, because the count "
            "carries no information to lose")


def test_one_marker_is_one_colour_everywhere(membrane_markers):
    """Stable across cells, or it identifies nothing."""
    for marker in list(membrane_markers)[:12]:
        assert marker_colour(marker) == marker_colour(marker)
    # the +/- suffix is a spelling, not a different molecule
    assert marker_colour("CD8") == marker_colour("CD8+")
    assert marker_colour("CTLA4") == marker_colour("CTLA-4+")


def test_the_colours_are_spread_well_enough_to_use(membrane_markers):
    """Pins the fix, so a regression to one-axis hashing fails here."""
    def base(name):
        return name.rstrip("+-").upper()

    colours = {m: np.array(marker_colour(m)) for m in membrane_markers}
    close = [(a, b) for a, b in itertools.combinations(sorted(colours), 2)
             if base(a) != base(b)
             and float(np.linalg.norm(colours[a] - colours[b])) < MIN_SEPARATION]
    assert len(close) <= 12, (
        f"{len(close)} marker pairs are within {MIN_SEPARATION} of each "
        f"other: {close[:6]}. One-axis hue hashing gave 59.")

    ccr5, ccr6 = colours["CCR5"], colours["CCR6"]
    assert float(np.linalg.norm(ccr5 - ccr6)) > 0.5, (
        "CCR5 and CCR6 separate a Th1 from a Th17 and must not look alike")
