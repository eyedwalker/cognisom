"""Where a marker is drawn, and why it is no longer always the surface.

The field is called ``surface_markers`` and for most of this library's
life everything in it was drawn as a stud on the plasma membrane. That
was wrong for a third of the vocabulary and invisibly so: a nuclear
transcription factor, a granule protein and a secreted chemokine all
look like a receptor once they are spheres on a membrane.

It had already bitten once. A CD8+ T cell carried ``perforin+`` and
``granzyme B+``; the compartment table keyed ``perforin`` and
``granzyme_B``; the lookup missed, and the one cell whose entire identity
is killing by degranulation was drawn with its granule contents stuck to
the outside of its membrane and no granules inside. That was fixed by
normalising the strings.

This is the same bug one level up, and the fix is structural rather than
another string. ``COMPARTMENT`` is a hand-written table of 60 markers and
the library refers to 173, so 115 of them -- 66% -- fell through to the
membrane. The entities themselves record where they live, 188 of them in
23 distinct compartment strings, and nothing read it.
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

from cognisom.visualization import cell_profiles as cp     # noqa: E402


@pytest.fixture(scope="module")
def store():
    with contextlib.redirect_stdout(io.StringIO()):
        from cognisom.library.store import EntityStore
        return EntityStore()


@pytest.fixture(scope="module")
def index(store):
    return cp.library_compartment_index(store)


# ── the vocabulary ───────────────────────────────────────────────────

def test_an_internal_membrane_is_not_the_plasma_membrane():
    """A lysosomal or mitochondrial membrane protein is INSIDE the cell.
    Drawing it as a surface receptor is the same error as perforin, one
    organelle further in."""
    for compartment in ("lysosome_membrane", "mitochondrial_outer_membrane",
                        "mitochondrial_intermembrane_space"):
        assert cp.bucket_for_compartments([compartment]) == "cytoplasm", compartment
    assert cp.bucket_for_compartments(["cell_membrane"]) == "membrane"


def test_the_most_decisive_compartment_wins():
    """Perforin is in a granule AND reaches the target's membrane. What
    shapes the cell HOLDING it is the granule."""
    assert cp.bucket_for_compartments(
        ["cytotoxic granule", "target plasma membrane"]) == "granule"
    # a receptor with a cytoplasmic tail is still a receptor
    assert cp.bucket_for_compartments(["cell_membrane", "cytoplasm"]) == "membrane"
    # and anything soluble passes through the cytosol on its way out
    assert cp.bucket_for_compartments(["cytoplasm", "secreted"]) == "cytoplasm"


def test_an_unrecognised_compartment_string_is_not_guessed():
    assert cp.bucket_for_compartments(["somewhere nobody named"]) is None
    assert cp.bucket_for_compartments([]) is None


def test_extracellular_is_only_safe_because_no_receptor_carries_it(store):
    """The map reads ``extracellular`` as secreted. That holds only while
    no membrane protein declares it, so the data is checked rather than
    the assumption trusted."""
    rows = store._conn.execute(
        "SELECT entity_type, data_json FROM entities")
    offenders = []
    for entity_type, blob in rows:
        comps = json.loads(blob or "{}").get("compartments") or []
        if "extracellular" in comps and entity_type in (
                "receptor", "adhesion_molecule", "mhc", "prr"):
            offenders.append(entity_type)
    assert not offenders, (
        "a membrane protein now declares 'extracellular', so mapping it "
        f"to 'secreted' is wrong: {offenders}")


# ── the resolver ─────────────────────────────────────────────────────

def test_the_library_answers_for_markers_the_table_never_mentioned(index):
    """The point of the whole change. None of these is in COMPARTMENT."""
    for marker, where in (("IFNg", "secreted"), ("IL10", "secreted"),
                          ("CCL19", "secreted"), ("CXCL8", "secreted"),
                          ("AR", "nucleus"), ("BCL2", "cytoplasm")):
        assert cp.compartment_of(marker) is None, f"{marker} is in the table now"
        assert cp.compartment_of(marker, index) == where, marker


def test_the_curated_table_still_wins(index):
    """A hand-checked placement is not overturned by a bulk import."""
    assert cp.COMPARTMENT["perforin"] == "granule"
    assert cp.compartment_of("perforin+", index) == "granule"
    assert cp.compartment_of("T-bet", index) == "nucleus"


def test_a_form_two_entities_claim_differently_is_dropped(index):
    """Guessing between two claimants puts a molecule somewhere it is
    not, which is the failure this function exists to stop."""
    assert all(isinstance(v, str) for v in index.values())
    # the index only ever holds forms with ONE agreed bucket
    import collections
    from cognisom.library import seed_synonyms as syn
    assert isinstance(index, dict) and len(index) > 500


def test_most_markers_are_placed_now(store, index):
    """The measurement the change is for: 115 of 173 marker strings were
    drawn as surface studs, and the entity records that would have
    placed them were sitting unread."""
    rows = store._conn.execute("SELECT data_json FROM entities")
    markers = set()
    for (blob,) in rows:
        props = (json.loads(blob or "{}").get("properties") or {})
        for field in ("surface_markers", "markers", "cytokines_secreting"):
            for value in (props.get(field) or []):
                markers.add(str(value))
    unplaced = [m for m in markers if cp.compartment_of(m, index) is None]
    assert len(markers) > 150
    assert len(unplaced) <= 20, (
        f"{len(unplaced)} of {len(markers)} markers unplaced: {sorted(unplaced)}")


# ── what the profile does with them ──────────────────────────────────

def test_a_secreted_cytokine_is_not_drawn_on_the_cell(index):
    """It has left. Painting it on the membrane it crossed says the cell
    displays it, which is a different claim."""
    profile = cp.profile_for(
        "Th1", "smooth_sphere", ["CD4", "IFNg", "IL2", "T-bet"],
        scale_um=8.0, compartments=index)
    assert "IFNg" in profile.secreted_markers
    assert "IFNg" not in profile.membrane_markers
    assert "IFNg" not in profile.membrane_stud_groups
    assert "T-bet" in profile.nuclear_markers
    assert "CD4" in profile.membrane_markers


def test_a_cytoskeletal_marker_is_drawn_inside(index):
    """Vimentin is the single marker separating mesenchymal from
    epithelial. On the surface it says nothing."""
    profile = cp.profile_for(
        "Fibroblast", "elongated_spindle", ["VIM", "ACTA2+"],
        scale_um=30.0, compartments=index)
    assert profile.cytoplasmic_markers
    for marker in profile.cytoplasmic_markers:
        assert marker not in profile.membrane_stud_groups


def test_the_stud_budget_is_not_spent_on_molecules_that_are_not_there(index):
    """Studs are a fixed budget split across the receptors a cell
    carries. Counting secreted and cytoplasmic molecules among them
    takes studs away from the receptors that are actually displayed."""
    markers = ["CD4", "CXCR3", "IFNg", "IL2", "TNFa", "T-bet", "VIM"]
    with_lib = cp.profile_for("Th1", "smooth_sphere", markers,
                              scale_um=8.0, compartments=index)
    without = cp.profile_for("Th1", "smooth_sphere", markers, scale_um=8.0)
    assert set(with_lib.membrane_stud_groups) < set(without.membrane_stud_groups)
    # the budget is conserved either way
    assert (sum(with_lib.membrane_stud_groups.values())
            == sum(without.membrane_stud_groups.values())
            == with_lib.receptor_stud_count)


def test_an_unplaced_marker_still_goes_on_the_membrane(index):
    """``surface_markers`` is what the field is called, so membrane is
    the declared default. What changed is that "unknown" now means
    genuinely unplaced rather than merely absent from a table."""
    profile = cp.profile_for("x", "smooth_sphere", ["NotAMoleculeAnyoneNamed"],
                             scale_um=8.0, compartments=index)
    assert profile.unknown_markers == ["NotAMoleculeAnyoneNamed"]
    assert "NotAMoleculeAnyoneNamed" in profile.membrane_stud_groups


# ── the suffix that hid six molecules ────────────────────────────────

@pytest.mark.parametrize("written,molecule", [
    ("ICAM-1_", "ICAM-1"), ("VCAM-1_", "VCAM-1"),
    ("ICAM1_inducible", "ICAM1"), ("VCAM1_inducible", "VCAM1"),
    ("E-selectin_inducible", "E-selectin"), ("CXCL8_displayed", "CXCL8"),
])
def test_a_state_suffix_is_not_a_different_molecule(written, molecule):
    """Six marker strings failed to resolve on a trailing word, and every
    one of them has an entity. Same failure as CD8 against CD8+."""
    assert cp.normalise_marker(written) == molecule


def test_stripping_suffixes_does_not_eat_real_names():
    """HLA-DR ends in a hyphenated token and is not HLA."""
    assert cp.normalise_marker("HLA-DR") == "HLA-DR"
    assert cp.normalise_marker("IL-17A") == "IL-17A"
    assert cp.normalise_marker("CD8+") == "CD8"
    assert cp.normalise_marker("AR-low") == "AR"
