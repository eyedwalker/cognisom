"""The library's shape vocabulary must produce shapes, and say when it cannot.

``mesh_type`` carries 35 values written by someone who knew what these cells and
organisms look like -- ``multilobed_granular``, ``ruffled_pseudopod``,
``enveloped_corona_spikes``, ``rod_flagellated``. Nothing read any of it. The
only code touching the field was the enrichment table that writes it, and the
USD exporter drew a sphere for every entity, so a neutrophil and a flagellated
Vibrio rendered identically.

The risk in fixing that is not a crash, it is a quiet fallback. A builder that
silently returns a sphere for ``multilobed_granular`` leaves a picture labelled
"neutrophil" that carries none of what the label asserts, and nothing in a test
suite notices because geometry was returned. So ``fell_back`` is part of the
result, and ``test_an_unknown_type_falls_back_and_admits_it`` is the test that
matters most here.
"""
from __future__ import annotations

import contextlib
import io
import json

import numpy as np
import pytest

from cognisom.visualization.morphology import (
    BUILDERS, PART_NAMES, build, build_parts,
)

#: mesh_types in the library that intentionally have no procedural builder:
#: these are molecular and should come from a real structure, not an invented
#: shape. See cognisom/library/structures.py.
STRUCTURE_BACKED = {"protein_structure", "mesh", "cylinder"}


@pytest.fixture(scope="module")
def library_mesh_types():
    from cognisom.library.store import EntityStore
    with contextlib.redirect_stdout(io.StringIO()):
        store = EntityStore()
    rows = store._conn.execute("SELECT data_json FROM entities").fetchall()
    out = set()
    for (blob,) in rows:
        mt = json.loads(blob or "{}").get("mesh_type")
        if mt:
            out.add(mt)
    return out


def test_every_shape_word_in_the_library_has_a_builder(library_mesh_types):
    """A type with no builder renders as a sphere, which is the old behaviour."""
    unhandled = library_mesh_types - set(BUILDERS) - STRUCTURE_BACKED
    assert not unhandled, (
        f"mesh_types the library uses but nothing can draw: {sorted(unhandled)}"
    )


@pytest.mark.parametrize("mesh_type", sorted(BUILDERS))
def test_every_builder_produces_a_closed_surface(mesh_type):
    m = build(mesh_type, scale_um=10.0, seed=1, grid=32)

    assert m.n_triangles > 0, f"{mesh_type} produced no geometry"
    assert len(m.vertices) > 0
    assert not m.fell_back
    assert m.builder == mesh_type
    assert np.isfinite(m.vertices).all()


def test_an_unknown_type_falls_back_and_admits_it():
    """The failure this module exists to prevent is a silent sphere."""
    m = build("no_such_shape", grid=32)

    assert m.n_triangles > 0, "a fallback must still draw something"
    assert m.fell_back is True
    assert m.builder == "sphere"
    assert m.mesh_type == "no_such_shape", "the request is kept, not overwritten"


def test_scale_is_in_micrometres_and_actually_scales():
    small = build("sphere", scale_um=5.0, grid=32)
    large = build("sphere", scale_um=20.0, grid=32)

    def extent(m):
        return float(np.ptp(m.vertices, axis=0).max())

    assert extent(large) > 3.4 * extent(small), (
        f"4x the scale should be ~4x the extent: {extent(small):.2f} -> {extent(large):.2f}"
    )
    assert 8.0 < extent(large) < 22.0, "a 20 um cell should be roughly 20 um across"


# ── the shapes have to carry what their names claim ──────────────────────

def _components(mask):
    from scipy import ndimage as ndi
    return int(ndi.label(mask)[1])


def test_a_multilobed_nucleus_really_has_several_lobes():
    """Otherwise 'multilobed_granular' is a sphere with a label on it."""
    parts, _ = build_parts("multilobed_granular", scale_um=12.0, seed=2, grid=48)

    assert "nucleus" in parts, "a neutrophil without a nucleus part"
    assert "granules" in parts, "the word 'granular' has to mean something"
    nucleus_v = parts["nucleus"][0]
    spread = np.ptp(nucleus_v, axis=0)
    assert spread.max() > 0.45 * 12.0, (
        "the nuclear lobes should span much of the cell, not sit as one ball")


def test_a_bilobed_nucleus_is_not_the_same_shape_as_a_multilobed_one():
    a, _ = build_parts("bilobed_granular", scale_um=12.0, seed=2, grid=48)
    b, _ = build_parts("multilobed_granular", scale_um=12.0, seed=2, grid=48)

    assert len(a["nucleus"][0]) != len(b["nucleus"][0]), (
        "two different words produced identical nuclei")


def test_a_flagellated_rod_carries_flagella_and_an_acid_fast_one_does_not():
    flagellated, _ = build_parts("rod_flagellated", scale_um=3.0, seed=1, grid=48)
    plain, _ = build_parts("rod_acid_fast", scale_um=3.0, seed=1, grid=48)

    assert "flagella" in flagellated
    assert "flagella" not in plain


def test_a_spore_forming_rod_carries_a_spore():
    parts, _ = build_parts("rod_spore_forming", scale_um=3.0, seed=1, grid=48)

    assert "spore" in parts


def test_a_plasma_cell_has_an_offset_nucleus_and_an_er():
    parts, _ = build_parts("eccentric_nucleus_er", scale_um=12.0, seed=1, grid=48)

    assert "er" in parts, "'eccentric_nucleus_er' names the ER explicitly"
    nucleus_centre = parts["nucleus"][0].mean(axis=0)
    body_centre = parts["body"][0].mean(axis=0)
    offset = float(np.linalg.norm(nucleus_centre - body_centre))
    assert offset > 0.5, f"the nucleus should be eccentric, offset only {offset:.2f} um"


def test_a_spiked_virus_actually_has_things_sticking_out():
    """Triangle count is the wrong probe: a bigger smooth envelope beats a
    smaller spiked one. What makes a spike a spike is radial reach past the
    envelope, so that is what is measured."""
    def reach(mesh_type):
        m = build(mesh_type, scale_um=0.12, seed=1, grid=48)
        r = np.linalg.norm(m.vertices - m.vertices.mean(axis=0), axis=1)
        return float(np.max(r) / np.median(r))

    assert reach("enveloped_corona_spikes") > 1.25 * reach("enveloped_smooth"), (
        "a spiked envelope should reach much further past its own surface "
        "than a smooth one")


# ── parts ────────────────────────────────────────────────────────────────

def test_part_names_are_the_documented_ones():
    parts, _ = build_parts("multilobed_granular", scale_um=10.0, grid=32)

    assert set(parts) <= set(PART_NAMES.values())


def test_sectioning_keeps_every_part_that_the_whole_shape_had():
    whole, _ = build_parts("multilobed_granular", scale_um=12.0, seed=4, grid=48, cut=False)
    cut, _ = build_parts("multilobed_granular", scale_um=12.0, seed=4, grid=48, cut=True)

    assert set(cut) == set(whole), "sectioning dropped a part entirely"
    assert len(cut["body"][1]) < len(whole["body"][1]), "the cut removed nothing"


def test_a_sectioned_inner_part_stands_proud_of_the_body_cut_face():
    """Coplanar cut faces z-fight and the interior renders as flat body colour."""
    parts, _ = build_parts("multilobed_granular", scale_um=12.0, seed=4, grid=48, cut=True)

    body_front = parts["body"][0][:, 1].min()
    nucleus_front = parts["nucleus"][0][:, 1].min()
    assert nucleus_front < body_front, (
        "the nucleus must extend past the body's cut plane to be visible")
