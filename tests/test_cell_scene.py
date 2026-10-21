"""The bridge from a library entity to something drawable.

This exists because the detail layer had no way in. A static import
closure over every dashboard page, every simulation module, the app, the
REST server and the orchestrator reached 249 of 409 modules and NONE of
morphology, cell_detail, cell_profiles, morphology_usd or molecular_usd.
The 32 shape builders, the measured microvillus density, the sourced
granule counts and the compartment placement were computed only by
tests.

What users actually saw came from a parallel system that picked a shape
by substring-matching the entity's name -- fibroblast and endothelial
both to macrophage_m2, epithelial to plasma_cell, every B cell and ILC
to t_cell -- and then printed the entity's real mesh_type underneath as
a caption. The page asserted a shape it had not drawn.

These tests pin the two claims that matter: the shape comes from the
library's own field, and the scene can say which of its numbers are
evidence.
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cognisom.visualization import cell_profiles as cp     # noqa: E402
from cognisom.visualization import cell_scene as cs        # noqa: E402


@pytest.fixture(scope="module")
def store():
    with contextlib.redirect_stdout(io.StringIO()):
        from cognisom.library.store import EntityStore
        return EntityStore()


@pytest.fixture(scope="module")
def index(store):
    return cp.library_compartment_index(store)


class _Stub:
    """An entity-shaped object with no builder, for the fallback path."""

    def __init__(self, name, mesh_type, scale_um=8.0):
        self.name = name
        self.mesh_type = mesh_type
        self.scale_um = scale_um
        self.properties = {}
        self.physics_params = {}

    def to_dict(self):
        # scene_for reads markers through to_dict, because only some
        # entity classes carry a loose .properties attribute and the
        # immune cells do not.
        return {"properties": self.properties}


def _scene(store, index, name):
    entity = store.find_entity_by_name(name)
    assert entity is not None, f"{name} is not in the library"
    return cs.scene_for(entity, compartments=index)


# ── the shape comes from the library ─────────────────────────────────

@pytest.mark.parametrize("name,mesh_type", [
    ("Cancer-associated fibroblast", "elongated_spindle"),
    ("Endothelial cell", "flat_squamous"),
    ("Neutrophil", "multilobed_granular"),
    ("Eosinophil", "bilobed_granular"),
    ("Plasma cell", "eccentric_nucleus_er"),
    ("Macrophage", "ruffled_pseudopod"),
])
def test_the_shape_is_the_one_the_entity_declares(store, index, name, mesh_type):
    """The substring router sent fibroblast and endothelial to
    macrophage_m2, and epithelial to plasma_cell. Each of these is now
    built from the field that names its shape."""
    scene = _scene(store, index, name)
    assert scene.mesh_type == mesh_type
    assert scene.from_library and not scene.fell_back
    assert scene.surfaces and scene.triangles > 100


def test_two_cells_with_different_mesh_types_are_different_objects(store, index):
    fibroblast = _scene(store, index, "Cancer-associated fibroblast")
    endothelium = _scene(store, index, "Endothelial cell")
    assert fibroblast.mesh_type != endothelium.mesh_type
    assert fibroblast.triangles != endothelium.triangles

    def extent(scene):
        v = np.vstack([s.vertices for s in scene.surfaces])
        return v.max(axis=0) - v.min(axis=0)

    # a spindle is long and thin; a squamous cell is wide and flat.
    f, e = extent(fibroblast), extent(endothelium)
    assert f.max() / f.min() > 2.0, f
    assert e.max() / e.min() > 2.0, e


def test_the_caption_never_claims_a_shape_that_was_not_built(store, index):
    """The old caption printed mesh_type beside a shape chosen by
    substring, so a fibroblast drawn as a macrophage was captioned
    'Elongated Spindle'."""
    scene = _scene(store, index, "Cancer-associated fibroblast")
    assert "elongated spindle" in scene.caption()

    fallback = cs.scene_for(_Stub("Something Unmeshed", "not_a_builder"))
    assert fallback.fell_back
    assert "no builder" in fallback.caption()
    assert "not_a_builder" not in fallback.caption()


def test_an_entity_with_no_builder_still_produces_something(store, index):
    scene = cs.scene_for(_Stub("Unmeshed", ""))
    assert scene.surfaces, "a fallback sphere is still a scene"
    assert not scene.from_library


def test_two_entities_sharing_a_mesh_type_are_not_the_same_object(store, index):
    """Everything was built with seed 0, so entities sharing a mesh_type
    and a size got byte-identical geometry."""
    a = _scene(store, index, "CD4+ Th1 cell")
    b = _scene(store, index, "CD4+ Th2 cell")
    if a.mesh_type != b.mesh_type:
        pytest.skip("these two no longer share a mesh_type")
    va = a.surfaces[0].vertices
    vb = b.surfaces[0].vertices
    assert va.shape != vb.shape or not np.allclose(va, vb), (
        "these two are byte-identical meshes -- shape_seed is not reaching "
        "this builder, which is what 17 of the 32 builders still do")


# ── the detail is the sourced detail ─────────────────────────────────

def test_a_cytotoxic_cell_carries_its_sourced_granules(store, index):
    """63 +/- 23 per primary human NK cell (PMID 29592963), and 251 nm
    across (PMID 21931537). Both read off the entity."""
    scene = _scene(store, index, "CD8+ Effector T cell")
    granules = [d for d in scene.details if "granule" in d.name]
    assert granules, [d.name for d in scene.details]
    assert granules[0].actual == 63
    assert granules[0].radius_um == pytest.approx(0.1255, abs=1e-3)
    assert scene.sources.get("lytic_granule_count") == "library"


def test_granules_are_drawn_once_and_not_twice(store, index):
    """A granular_sphere carves granules into its occupancy grid AND the
    profile reports a measured count, so both were drawn -- an NK cell's
    granules appeared on screen twice. Invisible to every numeric check
    and obvious the moment a cell was looked at.

    The instanced population wins: it carries the sourced count and
    diameter, and being instanced it does not depend on the grid, where
    a voxel-carved 0.25 um granule at a card's grid 32 is under one
    voxel and merges into a lump.
    """
    scene = _scene(store, index, "NK cell")
    carved = [s for s in scene.surfaces if s.name == "granules"]
    instanced = [d for d in scene.details if "granule" in d.name]
    assert not carved, "granules are drawn twice again"
    assert instanced and instanced[0].actual == 63
    # and the override is recorded rather than done quietly
    assert scene.sources.get("granules") == "library:superseded_builder"


def test_a_granulocyte_keeps_the_granules_its_builder_carves(store, index):
    """A neutrophil has no sourced lytic granule count -- its granules
    ARE its mesh_type -- so nothing supersedes them."""
    scene = _scene(store, index, "Neutrophil")
    assert [s for s in scene.surfaces if s.name == "granules"]
    assert "granules" not in scene.sources


def test_the_microvillus_count_follows_the_cell_size(store, index):
    """A measured density times this cell's own surface area, so the
    count stays right when the cell changes size."""
    small = _scene(store, index, "CD8+ Naive T cell")
    big = _scene(store, index, "NK cell")
    villi = {s.entity_name: [d for d in s.details if d.name == "microvilli"]
             for s in (small, big)}
    if not all(villi.values()):
        pytest.skip("one of these carries no microvilli")
    assert big.scale_um > small.scale_um
    assert (villi[big.entity_name][0].actual
            > villi[small.entity_name][0].actual)


def test_a_capped_population_reports_what_it_clipped(store, index):
    """Reporting only the drawn count makes a clipped population look
    sparse rather than clipped."""
    scene = _scene(store, index, "NK cell")
    for detail in scene.details:
        assert detail.actual >= detail.drawn > 0
        assert len(detail.positions) == detail.drawn


def test_the_scene_says_which_numbers_are_evidence(store, index):
    scene = _scene(store, index, "CD8+ Effector T cell")
    assert scene.sources
    assert any(v == "library" for v in scene.sources.values())
    assert any(v == "default" for v in scene.sources.values())
    assert "measured" in scene.caption()


# ── it must be affordable on a page ──────────────────────────────────

def test_a_card_builds_in_card_time(store, index):
    """Measured at grid 32: a median builder is 34 ms and the whole
    32-builder catalogue 1.48 s, against 163 ms and 12 s at grid 64.
    If this regresses, the dashboard wiring stalls behind it."""
    import time

    entity = store.find_entity_by_name("Neutrophil")
    start = time.perf_counter()
    cs.scene_for(entity, compartments=index)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"{elapsed:.2f}s for one card is too slow"
    assert cs.CARD_GRID == 32


def test_every_rendered_entity_produces_a_scene(store, index):
    """The catalogue has to survive the renderer, not just a sample."""
    from cognisom.visualization.morphology_usd import library_entities

    built = fell_back = 0
    from cognisom.visualization.morphology_usd import CELLULAR_TYPES

    for name, _etype, mesh_type, _scale in library_entities(store, CELLULAR_TYPES):
        entity = store.find_entity_by_name(name)
        if entity is None:
            continue
        scene = cs.scene_for(entity, grid=20, compartments=index)
        assert scene.surfaces, name
        built += 1
        fell_back += bool(scene.fell_back)
    assert built > 50
    # the fallbacks are the tissue_type rows whose mesh_type is the
    # literal string "mesh"; they are counted, not hidden
    assert fell_back <= 6, fell_back
