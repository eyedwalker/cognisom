"""A ppm became a count, and the count was nearly stored as a surface.

PaxDb measures what a proteome is made of, in parts per million of
total protein MOLECULES. A ratio cannot become a count without knowing
how many molecules a cell holds, and the library's own abundance note
said so and refused to make the leap.

The scale now comes from one absolute measurement already on disk: PMID
9829984 reports ~25,000 P-selectin binding sites on a human neutrophil
and establishes they are all PSGL-1. With the neutrophil's measured
radius that fixes protein molecules per cubic micrometre, and every
other count follows.

The first version wrote the result to ``surface_density_per_cell``.
PaxDb is a shotgun proteome -- it counts the ER, Golgi, endosomal and
secretory pools along with whatever is displayed -- so that put 24
million CD14 on a macrophage surface, 43% of the entire membrane under
one protein. The number was right and the name was wrong.
"""
from __future__ import annotations

import contextlib
import io
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="module")
def store():
    with contextlib.redirect_stdout(io.StringIO()):
        from cognisom.library.store import EntityStore
        return EntityStore()


# ── the calibration, and its independent check ───────────────────────

def test_the_scale_rests_on_a_measurement_not_a_guess(store):
    from cognisom.library.seed_copy_numbers import calibrate

    cal = calibrate(store)
    assert cal.proteins_per_um3 > 0
    assert cal.anchor_volume_um3 == pytest.approx(303, abs=5)


def test_the_calibration_predicts_a_hela_proteome_in_range(store):
    """One anchor fixes the scale; an unrelated cell checks it. If this
    fails the chain is wrong somewhere and seed() writes nothing."""
    from cognisom.library.seed_copy_numbers import calibrate

    cal = calibrate(store)
    assert cal.plausible, cal.describe()
    assert 2e9 <= cal.hela_proteome <= 1e10


def test_the_anchor_needs_an_evidenced_radius(store):
    """A guessed radius would put the whole scale on a guess."""
    from cognisom.library.parameters import Provenance, get_parameter
    from cognisom.library.seed_copy_numbers import ANCHOR_CELL

    radius = get_parameter(store.find_entity_by_name(ANCHOR_CELL), "radius_um")
    assert radius.provenance is not Provenance.PLACEHOLDER
    assert radius.citation == "PMID:2272970"


def test_the_weakest_link_is_named_in_the_module(store):
    """PaxDb has no neutrophil dataset for PSGL-1, so the anchor's ppm
    is the monocyte value. Every count inherits that."""
    from cognisom.library import seed_copy_numbers as mod

    assert mod.ANCHOR_PPM_FROM == "Monocyte"
    assert "nearest myeloid proxy" in mod.__doc__


# ── the quantity is named correctly ──────────────────────────────────

def test_counts_are_stored_as_total_copies_not_surface(store):
    from cognisom.library.parameters import Provenance, get_parameter

    entity = store.find_entity_by_name("LPS co-receptor (CD14)")
    total = get_parameter(entity, "copies_per_cell")
    assert total is not None
    assert total.provenance is Provenance.CALIBRATED
    assert total.value > 1e6

    surface = get_parameter(entity, "surface_density_per_cell")
    assert surface is None or surface.provenance is Provenance.PLACEHOLDER, (
        "a whole-cell proteome count is back in the surface parameter")


def test_the_membrane_check_that_caught_it(store):
    """24 million CD14 would need 43% of a macrophage membrane. The
    check is kept because it is what made the error visible."""
    from cognisom.library.parameters import get_parameter
    from cognisom.library.seed_copy_numbers import FOOTPRINT_UM2

    cell = store.find_entity_by_name("Macrophage")
    radius = get_parameter(cell, "radius_um").value
    area = 4.0 * math.pi * radius ** 2

    copies = get_parameter(store.find_entity_by_name("LPS co-receptor (CD14)"),
                           "copies_per_cell").value
    fraction = copies * FOOTPRINT_UM2 / area
    assert fraction > 0.30, (
        f"CD14 would cover {fraction:.1%} of the membrane; if this is now "
        "small the abundance changed and the example needs revisiting")


def test_each_value_states_the_membrane_it_would_need(store):
    from cognisom.library.parameters import get_parameter

    pv = get_parameter(store.find_entity_by_name("LPS co-receptor (CD14)"),
                       "copies_per_cell")
    assert "TOTAL cellular copies, not surface" in pv.rationale
    assert "impossible" in pv.rationale.lower()
    assert pv.calibrated_against and "25,000" in pv.calibrated_against


def test_the_spec_says_what_it_is_not():
    from cognisom.library.parameters import PARAMETER_SPECS

    spec = PARAMETER_SPECS["copies_per_cell"]
    assert spec.unit == "molecules/cell"
    assert "surface_density_per_cell" in spec.description
    assert "43%" in spec.description


def test_coverage_is_limited_by_cells_of_known_size(store):
    """A count needs a volume, so only cells with an evidenced radius
    can carry one. That is the honest limit and it should be visible."""
    from cognisom.library.seed_copy_numbers import seed, sized_cells

    volumes = sized_cells(store)
    assert len(volumes) >= 6
    run = seed(store, dry_run=True)
    assert len(run.written) >= 20
    assert run.no_sized_cell, (
        "nothing was skipped for want of a sized cell; the limit has "
        "moved and this test no longer measures it")
