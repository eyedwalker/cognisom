"""How big a molecule is, instead of one number for all of them.

This library recorded ``scale_um = 0.005`` -- 50 angstrom -- for **every
protein in the catalogue**, from cytochrome c to a 1972-residue myosin
heavy chain. Not a measurement: one type-level default, written once and
inherited by everything.

Measured from AlphaFold models, the range is **3.7 nm to 21.8 nm**, a
six-fold spread, and the error is largest where it shows most -- the big
multi-domain proteins drawn at a cell surface, where relative size is
the whole visual signal.

THE CONFIDENCE GATE IS THE POINT
--------------------------------
A low-pLDDT region is usually a genuinely disordered one, and a floppy
tail stretches a bounding box without making the molecule bigger.
Perforin measures 12.8 nm over all its atoms and 9.8 nm over its
confident ones: a quarter of its apparent length is tail.

Twenty-four molecules are refused outright for being mostly disordered,
and the list reads like a textbook: PUMA at 0% confident is a BH3-only
protein that is intrinsically disordered by definition; HIF-1alpha,
NFATC1 and MCL-1 carry large disordered regulatory regions; CD44 and
CD34 are mucin-like stalks; serglycin is an extended proteoglycan core
whose whole function is to be a flexible string for glycosaminoglycans
to hang off. None of them HAS a single size, and the honest answer is to
decline rather than to store a bounding box.
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cognisom.library import alphafold_sourcing as af     # noqa: E402


@pytest.fixture(scope="module")
def store():
    with contextlib.redirect_stdout(io.StringIO()):
        from cognisom.library.store import EntityStore
        return EntityStore()


def _pdb(coords_and_plddt):
    """A minimal PDB with the columns measure() reads."""
    lines = []
    for i, (x, y, z, b) in enumerate(coords_and_plddt, 1):
        lines.append(
            f"ATOM  {i:>5}  CA  ALA A{i:>4}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00{b:>6.2f}           C")
    return "\n".join(lines)


# ── the measurement ──────────────────────────────────────────────────

def test_the_extent_is_the_longest_axis_in_nanometres():
    """A 100 angstrom rod is 10 nm, not 100."""
    rod = [(x * 10.0, 0.0, 0.0, 95.0) for x in range(11)]
    shape = af.measure(_pdb(rod))
    assert shape.extent_nm == pytest.approx(10.0, abs=0.01)
    assert shape.atoms == 11


def test_a_disordered_tail_is_cut_from_the_extent():
    """The whole reason for the gate: a floppy tail stretches a bounding
    box without making the molecule bigger."""
    core = [(x * 2.0, 0.0, 0.0, 95.0) for x in range(20)]      # 38 A
    tail = [(40.0 + x * 5.0, 0.0, 0.0, 30.0) for x in range(20)]  # out to 135 A
    shape = af.measure(_pdb(core + tail))
    assert shape.extent_all_nm > 13.0, "the tail should stretch the raw box"
    assert shape.extent_nm == pytest.approx(3.8, abs=0.2)
    assert shape.confident_fraction == pytest.approx(0.5, abs=0.01)


def test_a_mostly_disordered_molecule_is_refused_not_guessed():
    mostly_floppy = ([(x * 2.0, 0.0, 0.0, 95.0) for x in range(5)]
                     + [(x * 9.0, 0.0, 0.0, 20.0) for x in range(25)])
    shape = af.measure(_pdb(mostly_floppy))
    assert shape.confident_fraction < af.MIN_CONFIDENT_FRACTION
    assert not shape.ok


def test_the_radius_of_gyration_is_robust_to_the_tail():
    """Which is why it is the better single number for packing: cutting
    the tail moves the longest axis a long way and Rg hardly at all."""
    core = [(x * 2.0, 0.0, 0.0, 95.0) for x in range(30)]
    tail = [(60.0 + x * 4.0, 0.0, 0.0, 95.0) for x in range(3)]
    with_tail = af.measure(_pdb(core + tail))
    without = af.measure(_pdb(core))
    axis_change = with_tail.extent_nm / without.extent_nm
    rg_change = (with_tail.radius_of_gyration_nm
                 / without.radius_of_gyration_nm)
    assert axis_change > rg_change


def test_the_conditions_say_what_the_number_rests_on():
    """A reader has to be able to see when a size was computed over
    three-quarters of a molecule."""
    atoms = ([(x * 2.0, 0.0, 0.0, 95.0) for x in range(30)]
             + [(x * 2.0, 5.0, 0.0, 40.0) for x in range(10)])
    text = af.measure(_pdb(atoms)).conditions()
    assert "75% at pLDDT>=70" in text
    assert "Over all atoms" in text


# ── what the live pass stored ────────────────────────────────────────

def test_the_library_has_real_sizes_now(store):
    from cognisom.library.parameters import get_parameter

    entities, _ = store.search(limit=10000)
    sizes = [get_parameter(e, "molecular_extent_nm") for e in entities]
    sizes = [pv.value for pv in sizes if pv is not None]
    assert len(sizes) >= 100, f"only {len(sizes)} molecules carry a size"
    assert max(sizes) / min(sizes) > 4.0, (
        "the sizes are all alike again, which is the state this replaced")
    assert min(sizes) > 1.0 and max(sizes) < 60.0


def test_a_size_is_derived_and_says_where_from(store):
    """From a PREDICTED structure. AlphaFold is very good and it is
    still a model, so the grade is DERIVED, not MEASURED."""
    from cognisom.library.parameters import Provenance, get_parameter

    entity = store.find_entity_by_name("Perforin (PRF1)")
    pv = get_parameter(entity, "molecular_extent_nm")
    assert pv is not None
    assert pv.provenance is Provenance.DERIVED
    assert pv.citation == "AlphaFoldDB:P14222"
    assert "AlphaFold Protein Structure Database" in pv.source
    assert "pLDDT" in pv.conditions


def test_the_disordered_ones_carry_no_size(store):
    """Storing a bounding box for PUMA would be a statement the
    structure does not support."""
    from cognisom.library.parameters import get_parameter

    for name in ("PUMA (BBC3)", "HIF-1alpha", "Hyaluronan receptor (CD44)"):
        entity = store.find_entity_by_name(name)
        if entity is None:
            continue
        assert get_parameter(entity, "molecular_extent_nm") is None, (
            f"{name} is mostly disordered and should have no single size")


def test_the_spec_accepts_these_types_and_refuses_a_cell(store):
    """A cell has a radius_um; a molecule has an extent. Neither should
    take the other's parameter."""
    from cognisom.library.models import EntityType
    from cognisom.library.parameters import PARAMETER_SPECS

    spec = PARAMETER_SPECS["molecular_extent_nm"]
    assert EntityType.PROTEIN in spec.applies_to
    assert EntityType.RECEPTOR in spec.applies_to
    assert EntityType.CELL_TYPE not in spec.applies_to
    assert EntityType.IMMUNE_CELL not in spec.applies_to


def test_the_cache_is_not_committed():
    """A structure is ~350 KB and reproducible from its accession, so it
    is a cache rather than an asset."""
    ignore = (REPO_ROOT / ".gitignore").read_text()
    assert "data/" in ignore
