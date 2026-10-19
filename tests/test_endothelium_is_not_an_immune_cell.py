"""Vessel wall is not a leukocyte, and retyping it must not eat its biology.

Three rows -- high endothelial venule, inflamed endothelium and postcapillary
venule endothelium -- were typed ``IMMUNE_CELL`` while their own
``immune_type`` field said ``endothelial``. That made the parameter specs
demand a contact kill radius and a per-contact kill probability of a blood
vessel, and gave a junction-anchored sessile cell a lymphocyte's crawling
speed.

The retype is the fix, but it is also the risk. ``ImmuneCellEntity`` keeps its
markers under ``surface_markers`` and ``CellTypeEntity`` keeps them under
``markers``, and ``CellTypeEntity`` had no field for secreted mediators at
all, so a naive retype dropped seventeen property keys. For the HEV row those
keys are the mechanism it exists to describe: PNAd and GlyCAM-1 are the
L-selectin ligands, CCL21 and CCL19 the CCR7 ligands that arrest a rolling
lymphocyte. Losing them would leave a row that still had a name and nothing
else.
"""
from __future__ import annotations

import contextlib
import io

import pytest

from cognisom.library.models import BioEntity, CellTypeEntity, EntityType
from cognisom.library.parameters import assess

NAMES = ("High endothelial venule (HEV) cell",
         "Inflamed endothelial cell",
         "Postcapillary venule endothelial cell")

#: The HEV homing mechanism, in the row's own words.
HEV_MARKERS = ("PNAd", "GlyCAM-1", "MAdCAM-1", "CCL21")


@pytest.fixture(scope="module")
def rows():
    from cognisom.library.store import EntityStore
    with contextlib.redirect_stdout(io.StringIO()):
        store = EntityStore()
    ids = [r[0] for r in
           store._conn.execute("SELECT entity_id FROM entities").fetchall()]
    entities = [e for e in (store.get_entity(i) for i in ids) if e]
    return {e.name: e for e in entities if e.name in NAMES}


def test_all_three_rows_are_present(rows):
    assert set(rows) == set(NAMES), f"missing: {set(NAMES) - set(rows)}"


@pytest.mark.parametrize("name", NAMES)
def test_endothelium_is_typed_as_a_cell_not_an_immune_cell(name, rows):
    entity = rows[name]

    assert entity.entity_type is EntityType.CELL_TYPE, (
        f"{name} is typed {entity.entity_type.value}; vascular endothelium "
        "is not a leukocyte")
    assert isinstance(entity, CellTypeEntity)


@pytest.mark.parametrize("name", NAMES)
def test_the_retype_kept_the_markers(name, rows):
    """The seventeen property keys a naive retype drops."""
    entity = rows[name]

    assert entity.markers, f"{name} lost its surface markers in the retype"
    assert "CD31" in entity.markers, (
        f"{name} has markers {entity.markers} but not the endothelial CD31 "
        "it was seeded with")


def test_the_hev_row_kept_the_homing_mechanism(rows):
    hev = rows["High endothelial venule (HEV) cell"]

    for marker in HEV_MARKERS:
        assert marker in hev.markers, (
            f"HEV lost {marker}. PNAd and GlyCAM-1 are the L-selectin "
            "ligands and CCL21 the CCR7 ligand; without them the row no "
            "longer describes lymphocyte homing at all")
    assert set(hev.cytokines_secreting) >= {"CCL21", "CCL19"}, (
        f"HEV secretes {hev.cytokines_secreting}; the CCR7 ligands that "
        "arrest a rolling lymphocyte are missing")


@pytest.mark.parametrize("name", NAMES)
def test_no_vessel_wall_is_asked_for_a_kill_radius(name, rows):
    readiness = assess(rows[name])
    asked = set(readiness.missing) | set(readiness.placeholder)

    for killing in ("kill_radius_um", "kill_probability", "detection_radius_um"):
        assert killing not in asked, (
            f"{name} is still asked for {killing}; it is a blood vessel")


def test_a_retype_round_trip_preserves_markers_from_either_name():
    """CellTypeEntity must read surface_markers as well as markers.

    This is the mechanism the migration depended on, tested directly so it
    cannot be removed as an unused alias.
    """
    source = {
        "entity_id": "x", "entity_type": "cell_type", "name": "test row",
        "properties": {"surface_markers": ["PNAd", "CD31"],
                       "cytokines_secreting": ["CCL21"]},
    }

    entity = BioEntity.from_dict(source)

    assert entity.markers == ["PNAd", "CD31"]
    assert entity.cytokines_secreting == ["CCL21"]


def test_the_whole_library_still_loads(rows):
    """A row whose type the enum does not know raises for every listing."""
    from cognisom.library.store import EntityStore
    with contextlib.redirect_stdout(io.StringIO()):
        store = EntityStore()
    ids = [r[0] for r in
           store._conn.execute("SELECT entity_id FROM entities").fetchall()]

    loaded = [e for e in (store.get_entity(i) for i in ids) if e]

    assert len(loaded) == len(ids), (
        f"{len(ids) - len(loaded)} rows failed to load")


@pytest.mark.parametrize("name", NAMES)
def test_the_stored_type_agrees_with_the_stored_json(name, rows):
    """search() reads the column and get_entity() reads the JSON.

    A row where they disagree lists as one type and loads as another.
    """
    import json
    from cognisom.library.store import EntityStore
    with contextlib.redirect_stdout(io.StringIO()):
        store = EntityStore()
    row = store._conn.execute(
        "SELECT entity_type, data_json FROM entities WHERE entity_id=?",
        (rows[name].entity_id,)).fetchone()

    assert row is not None
    column, blob = row
    assert json.loads(blob).get("entity_type") == column == "cell_type"
