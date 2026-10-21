"""The rendered cells follow the library, not a constant beside them.

The morphometry behind these cells was sourced and then written into
``cognisom.visualization`` as Python constants with its PMIDs in
comments. That is the wrong place twice over: a number in a comment
cannot be queried or exported with the rest of the curation, and a
renderer holding its own copy is a second source of truth that drifts
from the first without anything noticing.

So the values live on the entities now, through ``set_parameter``, with
MEASURED provenance and a citation each -- and the profile reads them.
These tests exist to keep that true. The decisive one is
``test_changing_the_library_changes_the_geometry``: if the library can
be changed and the geometry does not move, the wiring is decorative and
the citations are describing something the renderer is not using.

What is NOT claimed: that every number is sourced. The mitochondrial
volume fraction, the lysosome count and the ruffle amplitude are still
declared estimates, and ``Profile.sources`` says "default" for them.
Telling the two apart is the point of recording the source at all.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cognisom.library.parameters import (                       # noqa: E402
    PARAMETER_SPECS, Provenance, get_parameter)
from cognisom.library import seed_ultrastructure as seeding     # noqa: E402
from cognisom.visualization import cell_detail as cd            # noqa: E402
from cognisom.visualization import cell_profiles as cp          # noqa: E402

#: Parameters the sourcing pass actually delivered.
SOURCED = ("nuclear_volume_fraction", "granule_volume_fraction",
           "microvillus_density_per_um2", "microvillus_length_um",
           "microvillus_diameter_um", "lytic_granule_count",
           "lytic_granule_diameter_um")


@pytest.fixture(scope="module")
def store():
    with contextlib.redirect_stdout(io.StringIO()):
        from cognisom.library.store import EntityStore
        return EntityStore()


def _record(store, name):
    rows = [tuple(r) for r in store._conn.execute(
        "SELECT data_json FROM entities WHERE name = ?", (name,))]
    return json.loads(rows[0][0] or "{}") if rows else {}


# ── the values are in the library, with their evidence ───────────────

@pytest.mark.parametrize("parameter", SOURCED)
def test_the_parameter_is_declared_with_a_unit_and_a_range(parameter):
    spec = PARAMETER_SPECS.get(parameter)
    assert spec is not None, f"{parameter} has no spec, so nothing checks it"
    assert spec.unit
    assert spec.minimum is not None and spec.maximum is not None


def test_a_lymphocyte_carries_the_measured_ultrastructure(store):
    entity = store.find_entity_by_name("CD8+ Effector T cell")
    assert entity is not None
    for parameter in ("nuclear_volume_fraction", "microvillus_density_per_um2",
                      "microvillus_length_um", "lytic_granule_count"):
        pv = get_parameter(entity, parameter)
        assert pv is not None, f"{parameter} is not on the entity"
        assert pv.provenance is Provenance.MEASURED
        assert pv.citation.startswith("PMID:"), pv.citation
        assert pv.unit == PARAMETER_SPECS[parameter].unit


def test_the_granule_fractions_keep_their_measured_spread(store):
    """One table, one method, one paper (PMID 6775712). The spread is the
    finding: eosinophil to lymphocyte is 115-fold, and a set of values
    that does not reproduce it is not this measurement however plausible
    each number looks on its own."""
    got = {}
    for name in ("Eosinophil", "Neutrophil", "Macrophage", "CD8+ T cell"):
        entity = store.find_entity_by_name(name)
        pv = get_parameter(entity, "granule_volume_fraction")
        assert pv is not None, name
        assert pv.citation == "PMID:6775712"
        got[name] = pv.value

    assert got["Eosinophil"] == pytest.approx(0.230)
    assert got["Neutrophil"] == pytest.approx(0.154)
    assert got["Macrophage"] == pytest.approx(0.026)
    assert got["CD8+ T cell"] == pytest.approx(0.002)
    assert got["Eosinophil"] / got["CD8+ T cell"] > 50


def test_the_nuclear_fraction_is_a_fraction_not_a_radius(store):
    """The trap this parameter's description exists to prevent: the
    radius is the cube root. Using 0.444 as a radius gives a nucleus at
    8.8% of the cell, which is the bug it replaced, reached from the
    right number."""
    entity = store.find_entity_by_name("CD4+ T cell")
    pv = get_parameter(entity, "nuclear_volume_fraction")
    assert pv.unit == "dimensionless"
    assert 0.40 < pv.value < 0.50

    # Through the live path, not against a second copy of the number.
    # This used to compare the entity's value with a module constant
    # holding the same value, which passed because one person wrote both
    # -- and said nothing about whether the renderer used either.
    from cognisom.visualization.morphology import ShapeParams

    shape = ShapeParams(nuclear_volume_fraction=pv.value)
    assert shape.nuclear_radius() == pytest.approx(0.92 * pv.value ** (1 / 3))
    assert shape.nuclear_radius() > pv.value, "radius set from the fraction"


def test_the_lytic_granule_count_is_the_primary_cell_one(store):
    """63 +/- 23 on primary human NK (PMID 29592963), not the 194-206 the
    same paper reports for NK-92 and YTS -- which is what reviews quote
    as 'almost 200 granules'."""
    entity = store.find_entity_by_name("NK cell")
    pv = get_parameter(entity, "lytic_granule_count")
    assert pv.value == pytest.approx(63)
    assert pv.value < 100, "this is a cell-line count, not a primary one"
    assert "NK-92" in pv.rationale or "cell line" in pv.rationale.lower()


# ── and the geometry follows them ────────────────────────────────────

def test_the_profile_reads_the_library(store):
    entity = store.find_entity_by_name("CD8+ Effector T cell")
    record = _record(store, "CD8+ Effector T cell")
    library = cp.library_detail(entity)
    assert library, "nothing provenanced came back off the entity"

    profile = cp.profile_for(
        "CD8+ Effector T cell", record["mesh_type"],
        cp.markers_of(json.dumps(record)),
        scale_um=float(record["scale_um"]), library=library)

    from_library = {k for k, v in profile.sources.items() if v == "library"}
    assert "microvilli_count" in from_library
    assert "lytic_granule_count" in from_library
    assert profile.lytic_granule_count == 63


def test_changing_the_library_changes_the_geometry(store):
    """The decisive one. If the library can be changed and the geometry
    does not move, the wiring is decorative and the citations describe
    something the renderer is not using."""
    record = _record(store, "CD8+ Effector T cell")
    markers = cp.markers_of(json.dumps(record))
    mesh_type, scale = record["mesh_type"], float(record["scale_um"])

    real = cp.library_detail(store.find_entity_by_name("CD8+ Effector T cell"))
    doubled = dict(real)
    doubled["microvillus_density_per_um2"] = real["microvillus_density_per_um2"] * 2
    doubled["lytic_granule_count"] = 7.0

    before = cp.profile_for("x", mesh_type, markers, scale_um=scale,
                            library=real)
    after = cp.profile_for("x", mesh_type, markers, scale_um=scale,
                           library=doubled)

    assert after.microvilli_count == pytest.approx(before.microvilli_count * 2,
                                                   rel=0.02)
    assert after.lytic_granule_count == 7
    assert before.lytic_granule_count == 63


def test_without_the_library_the_profile_says_so(store):
    """A default must be legible as a default. The fallback constants are
    the same numbers, which is exactly why the source has to be recorded:
    the values alone cannot tell you whether anything was read."""
    record = _record(store, "CD4+ T cell")
    profile = cp.profile_for(
        "CD4+ T cell", record["mesh_type"], cp.markers_of(json.dumps(record)),
        scale_um=float(record["scale_um"]), library=None)
    assert all(v != "library" for v in profile.sources.values())
    # and the estimates are still named as estimates
    assert "mitochondrial_volume_fraction" in cp.UNSOURCED
    assert "microvillus_length_um" not in cp.UNSOURCED


def test_the_microvillus_count_scales_with_the_cell(store):
    """A density times this cell's own area. The count it replaced was one
    fixed number, which was wrong at every diameter but one."""
    assert cd.microvillus_count(7.0) == pytest.approx(462, abs=2)
    assert cd.microvillus_count(8.0) == pytest.approx(603, abs=2)
    assert cd.microvillus_count(8.0) > cd.microvillus_count(7.0)
    # doubling the density doubles the count
    assert cd.microvillus_count(7.0, 6.0) == pytest.approx(
        2 * cd.microvillus_count(7.0, 3.0), rel=0.01)


# ── the seeding itself ───────────────────────────────────────────────

def test_every_seeded_value_passes_its_spec(store):
    """set_parameter is the gate: a wrong unit or an out-of-range number
    cannot reach the library whatever the seeding module believes."""
    touched, written, problems = seeding.seed(store, dry_run=True)
    assert problems == [], problems
    assert touched >= 20 and written >= 100


def test_the_seeding_refuses_what_could_not_be_sourced():
    """The absences are deliberate and each has a reason. A Golgi size
    taken from the available record would be a lipid-bilayer thickness,
    760x too small."""
    every = {m.parameter for name in seeding.LYMPHOCYTE_NAMES
             for m in seeding.measurements_for(name)}
    for absent in ("golgi_extent_um", "er_surface_area_um2",
                   "nucleus_volume_um3"):
        assert absent not in every, absent
    doc = seeding.__doc__ or ""
    assert "39.5 angstrom" in doc and "yeast" in doc


def test_a_measured_value_without_a_citation_is_refused(store):
    """The obligation that makes the provenance grade mean something."""
    from cognisom.library.parameters import ParameterValue, set_parameter
    entity = store.find_entity_by_name("CD4+ T cell")
    problems = set_parameter(
        entity, "microvillus_length_um",
        ParameterValue(value=0.35, unit="um",
                       provenance=Provenance.MEASURED, citation=""),
        strict=False)
    assert any("citation" in p for p in problems), problems


# ── the two that shape the cell, not only decorate it ────────────────

def _nucleus_share(mesh_type, fraction, grid=40):
    """What fraction of the built body the built nucleus occupies."""
    from cognisom.visualization.morphology import ShapeParams, build_masks

    masks, _ = build_masks(mesh_type, seed=1, grid=grid,
                           params=ShapeParams(nuclear_volume_fraction=fraction))
    return masks["nucleus"].sum() / masks["body"].sum()


def _granule_share(mesh_type, fraction, grid=44):
    from cognisom.visualization.morphology import ShapeParams, build_masks

    masks, _ = build_masks(mesh_type, seed=1, grid=grid,
                           params=ShapeParams(granule_volume_fraction=fraction))
    cytoplasm = masks["body"].sum() - masks["nucleus"].sum()
    return masks["granules"].sum() / cytoplasm


def test_the_nuclear_fraction_changes_the_mesh():
    """The decisive test of the claim that the library drives geometry.

    ``library_detail`` fetched seven parameters and ``FROM_LIBRARY``
    mapped two, so ``nuclear_volume_fraction`` was fetched, graded
    MEASURED on 20 entities, and then overridden by a module constant.
    The copies agreed -- they were written by one person in one pass --
    which is exactly why nothing caught it.
    """
    small, measured, large = (_nucleus_share("smooth_sphere", f)
                              for f in (0.20, 0.444, 0.70))
    assert small < measured < large
    # and it is the fraction itself that comes out, not something
    # proportional to it: a RADIUS is the cube root of a volume.
    assert measured == pytest.approx(0.444, abs=0.02)
    assert small == pytest.approx(0.20, abs=0.02)
    assert large == pytest.approx(0.70, abs=0.02)


def test_the_granule_fraction_changes_the_mesh():
    thin, thick = _granule_share("granular_sphere", 0.002), \
        _granule_share("granular_sphere", 0.23)
    assert thin < thick
    assert thin == pytest.approx(0.002, abs=0.004)
    assert thick == pytest.approx(0.23, abs=0.02)


def test_the_builder_no_longer_keeps_its_own_copy():
    """Two copies of one measurement is one copy too many, and the
    module's copy won. One had already drifted: granular_sphere carved
    granules at 0.020 while the NK cell entity stored 0.002."""
    import cognisom.visualization.morphology as mo

    assert not hasattr(mo, "GRANULE_FRACTION")
    assert not hasattr(mo, "LYMPHOCYTE_NUCLEAR_VOLUME_FRACTION")
    assert not hasattr(mo, "LYMPHOCYTE_NUCLEAR_RADIUS")
    # the fallbacks that remain are named as fallbacks
    assert mo.DEFAULT_NUCLEAR_VOLUME_FRACTION == 0.444
    assert mo.DEFAULT_GRANULE_FRACTION["lymphocyte"] == 0.002


def test_an_nk_cell_gets_its_stored_fraction_not_the_old_constant():
    """granular_sphere hardcoded 0.020 against a stored, measured 0.002
    -- a factor of ten, and the hardcoded one was what rendered."""
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        from cognisom.library.store import EntityStore
        store = EntityStore()
    from cognisom.visualization import cell_profiles as cp
    from cognisom.visualization import cell_scene as cs

    entity = store.find_entity_by_name("NK cell")
    scene = cs.scene_for(entity, grid=40,
                         compartments=cp.library_compartment_index(store))
    assert scene.profile.granule_volume_fraction == pytest.approx(0.002)
    assert scene.sources["granule_volume_fraction"] == "library"
    assert scene.sources["nuclear_volume_fraction"] == "library"


def test_a_caller_supplying_nothing_still_gets_a_cell():
    """A bare build is a legitimate call -- a test, a preview, an entity
    with no sourced proportions. It gets the documented fallback, and
    the profile records that it was a fallback rather than evidence."""
    from cognisom.visualization.morphology import build_masks

    masks, _ = build_masks("smooth_sphere", seed=1, grid=40)
    share = masks["nucleus"].sum() / masks["body"].sum()
    assert share == pytest.approx(0.444, abs=0.02)
