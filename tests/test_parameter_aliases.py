"""A parameter the library already states must not be reported as absent.

``coverage()`` drives curation: a parameter it calls missing sends someone to
find a number. Forty seeded cell types state their size in the dataclass field
``scale_um``, but the aliasing reader consulted only ``physics_params``, so
``radius_um`` came back absent for every one of them and the worklist asked for
forty values the library was already holding.

The factor is the dangerous part. ``scale_um`` is a full extent, not a radius:
:func:`morphology.build` lays a ``scale_um``-wide box over the voxel grid, and
the seeded numbers read as diameters against textbook sizes. Aliasing it at 1.0
would double every cell radius and eightfold every cell volume -- a silent
rescaling of contact, uptake and packing throughout the tissue model, with no
test failing. Hence :func:`test_the_factor_is_a_half_not_one`.
"""
from __future__ import annotations

import contextlib
import io

import pytest

from cognisom.library.models import EntityType
from cognisom.library.parameters import (
    PARAMETER_ALIASES, Provenance, get_parameter,
)

SIZED = (EntityType.CELL_TYPE, EntityType.IMMUNE_CELL, EntityType.PHYSICAL_CELL)

#: Textbook cell diameters in um. These check the orientation of the factor,
#: not the accuracy of the seeded values.
DIAMETERS = {
    "CD8+ Naive T cell": (6, 8),
    "Naive B cell": (6, 9),
    "NK cell": (10, 12),
    "Neutrophil": (12, 15),
    "Eosinophil": (12, 17),
    "Plasma cell": (12, 16),
    "Macrophage M1": (15, 25),
}


@pytest.fixture(scope="module")
def sized_entities():
    from cognisom.library.store import EntityStore
    with contextlib.redirect_stdout(io.StringIO()):
        store = EntityStore()
    ids = [r[0] for r in
           store._conn.execute("SELECT entity_id FROM entities").fetchall()]
    entities = [e for e in (store.get_entity(i) for i in ids) if e]
    return [e for e in entities if e.entity_type in SIZED and e.scale_um]


def test_the_factor_is_a_half_not_one():
    """scale_um is a diameter. At factor 1.0 every cell doubles."""
    alias = next(a for a in PARAMETER_ALIASES
                 if a.legacy == "scale_um" and a.canonical == "radius_um")

    assert alias.factor == 0.5, (
        "scale_um is the cell's full extent; treating it as a radius "
        "eightfolds every cell volume")
    assert alias.attribute, "scale_um is a dataclass field, not a parameter"


def test_a_stated_size_is_never_reported_absent(sized_entities):
    assert sized_entities, "no sized cell types in the library to check"

    for entity in sized_entities:
        pv = get_parameter(entity, "radius_um")
        assert pv is not None, (
            f"{entity.name} states scale_um={entity.scale_um} but radius_um "
            "reads as absent, so curation will ask for it again")
        assert pv.value == pytest.approx(entity.scale_um / 2.0)
        assert pv.unit == "um"


def test_the_radii_land_in_the_textbook_range(sized_entities):
    """Catches a factor applied in the wrong direction."""
    by_name = {e.name: e for e in sized_entities}
    checked = 0
    for name, (lo, hi) in DIAMETERS.items():
        entity = by_name.get(name)
        if entity is None:
            continue
        pv = get_parameter(entity, "radius_um")
        assert lo / 2.0 <= pv.value <= hi / 2.0, (
            f"{name} radius {pv.value} um is outside the {lo/2}-{hi/2} um "
            f"implied by a {lo}-{hi} um diameter")
        checked += 1
    assert checked >= 5, f"only {checked} known cell types were checked"


def test_a_conversion_does_not_become_evidence(sized_entities):
    """Dividing by two does not tell you where the number came from."""
    pv = get_parameter(sized_entities[0], "radius_um")

    assert pv.provenance is Provenance.PLACEHOLDER
    assert not pv.is_simulation_grade, (
        "an unevidenced number must not pass as simulation-grade just "
        "because it was unit-converted")
    assert "scale_um" in pv.rationale


def test_an_unset_size_stays_absent_rather_than_becoming_zero():
    """0.0 is the dataclass default, not a measurement of zero.

    A radius of zero is a dimensionless cell: it would divide by zero in
    packing and silently zero every surface-area term.
    """
    from cognisom.library.models import BioEntity

    entity = BioEntity(name="sizeless", entity_type=EntityType.CELL_TYPE)
    entity.scale_um = 0.0

    assert get_parameter(entity, "radius_um") is None
