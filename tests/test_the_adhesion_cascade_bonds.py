"""Four binding constants that were right, and said almost nothing.

The adhesion cascade's affinities were already in this library against
the correct PMIDs, sourced by an agent that had not read the papers.
Every number survived contact with them. None of the context did.

    Bond                   Species              T     Kd
    P-selectin/PSGL-1      human                25 C  0.32 uM
    E-selectin/ESL-1       mouse                37 C    62 uM
    L-selectin/GlyCAM-1    RAT CD62L + MOUSE    25 C   108 uM
    ICAM-1/LFA-1           human                25 C   0.5 uM

Four things were missing, and each of them changes how the number may
be used: the species, the temperature, which ligand the constant is
for, and -- for LFA-1 -- which of its two conformational states.
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

BONDS = ("P-selectin (CD62P)", "E-selectin (CD62E)",
         "L-selectin (CD62L)", "ICAM-1 (CD54)")


@pytest.fixture(scope="module")
def store():
    with contextlib.redirect_stdout(io.StringIO()):
        from cognisom.library.store import EntityStore
        return EntityStore()


def _kd(store, name):
    from cognisom.library.parameters import get_parameter

    entity = store.find_entity_by_name(name)
    assert entity is not None, name
    pv = get_parameter(entity, "kd_nm")
    assert pv is not None, f"{name} has no Kd"
    return entity, pv


@pytest.mark.parametrize("name", BONDS)
def test_every_bond_names_its_species_and_temperature(store, name):
    _entity, pv = _kd(store, name)
    assert "SPECIES:" in pv.conditions
    assert " C." in pv.conditions, "no temperature recorded"


@pytest.mark.parametrize("name", BONDS)
def test_every_bond_names_the_ligand_it_is_a_constant_for(store, name):
    """Each selectin binds several ligands and they differ by orders of
    magnitude: P-selectin binds PSGL-1 200x more tightly than E-selectin
    binds ESL-1. A bare kd_nm cannot say which bond it is."""
    from cognisom.library.parameters import parameter_partner

    entity, _pv = _kd(store, name)
    assert parameter_partner(entity, "kd_nm"), f"{name}: no ligand named"


def test_the_cross_species_pair_is_not_called_one_species(store):
    """PMID:11404363 Table IV lists L-selectin/GlyCAM-1 as 'Mouse'. The
    primary source says rat CD62L bound to mouse GlyCAM-1, and flags it
    as a caveat itself. The library had taken the simplification."""
    _entity, pv = _kd(store, "L-selectin (CD62L)")
    assert "RAT" in pv.conditions and "MOUSE" in pv.conditions
    assert "cross-species" in pv.rationale.lower() or \
        "CROSS-SPECIES" in pv.conditions


def test_a_bound_is_not_stored_as_a_measurement(store):
    """L-selectin's koff is >=10 s-1 because the observed decay was the
    instrument's washout time, not dissociation. The floor is the whole
    content of the value."""
    from cognisom.library.parameters import Provenance, get_parameter

    entity = store.find_entity_by_name("L-selectin (CD62L)")
    koff = get_parameter(entity, "koff_per_s")
    assert koff.provenance is Provenance.ESTIMATED
    assert ">=" in koff.conditions
    assert "washing time" in koff.rationale.lower() or \
        "floor" in koff.rationale.lower()


def test_the_calculated_constant_is_not_graded_as_observed(store):
    """The two families of paper are mirror images. In the selectin
    papers the association phase was too fast to fit, so kon = koff/Kd.
    In the ICAM-1/LFA-1 paper kass WAS fitted and 'Kd was calculated
    from the ratio kdiss/kass'. The library had both graded the same."""
    from cognisom.library.parameters import Provenance, get_parameter

    icam = store.find_entity_by_name("ICAM-1 (CD54)")
    assert get_parameter(icam, "kd_nm").provenance is Provenance.DERIVED
    assert get_parameter(icam, "kon_per_nm_per_s").provenance is \
        Provenance.MEASURED

    psel = store.find_entity_by_name("P-selectin (CD62P)")
    assert get_parameter(psel, "kd_nm").provenance is Provenance.MEASURED
    assert get_parameter(psel, "kon_per_nm_per_s").provenance is \
        Provenance.DERIVED


def test_the_integrin_says_which_affinity_state_it_is(store):
    """LFA-1 has two states 200x apart. The 500 nM is the Mg2+-induced
    high-affinity one, measured deliberately without Ca2+; on an
    unstimulated T cell it is about 100 uM. A resting lymphocyte
    modelled with 500 nM is wrong by that factor."""
    _entity, pv = _kd(store, "ICAM-1 (CD54)")
    assert "HIGH-AFFINITY STATE" in pv.conditions
    assert "100 uM" in pv.conditions and "200x" in pv.conditions
    assert "UNSTIMULATED" in pv.conditions


@pytest.mark.parametrize("name", BONDS)
def test_every_bond_says_it_is_a_monomeric_floor(store, name):
    """All four are monomeric affinities, measured that way on purpose.
    The cell-surface bond is multivalent and stronger -- in the
    L-selectin paper, 10,000x slower off-rate at serum concentration."""
    _entity, pv = _kd(store, name)
    assert "MONOMERIC" in pv.conditions
    assert "floor" in pv.conditions


def test_the_seeder_is_idempotent(store):
    """It was not: the first pass moved a MEASURED Kd to its alias, the
    second read the bare name, got PLACEHOLDER from the alias fallback,
    and wrote that back over its own evidence. Grades are stated per
    bond now, because a grade is a fact about the experiment."""
    from cognisom.library.seed_adhesion_bonds import TABLE_IV, seed

    run = seed(store)
    assert not run.grade_corrected, (
        f"a repeat run changed grades: {run.grade_corrected}")
    assert not run.refused
    assert len(TABLE_IV) == 4


@pytest.mark.parametrize("pmid,probe", [
    ("9829984", "320"), ("11404363", "62"),
    ("9780171", "500"), ("9422729", "108"),
])
def test_each_paper_is_on_disk_and_holds_its_number(pmid, probe):
    from cognisom.library import literature as lit

    if not (REPO_ROOT / "docs" / "papers" / f"{pmid}.pdf").exists():
        pytest.skip(f"{pmid}.pdf is not on this machine")
    article = lit.fetch(pmid)
    assert article.full_text and probe in article.text
