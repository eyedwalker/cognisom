"""A structure in the library has to be the molecule the record names.

The library shipped one filled SMILES field, under the name Enzalutamide, and
it was the wrong molecule: it parses to C28H24F6N4O2 (InChIKey
KPEMKSCTFNDIQU), while enzalutamide is C21H16F4N4O2S (WXCXUHSOUPDCQV) -- 40
atoms against 32, five rings against three, and no sulfur where enzalutamide
carries a thiohydantoin.

Nothing caught it because a SMILES string is unreadable. Nobody looks at
``CC1(C2CCC3(...))`` and sees the wrong compound, and no test recomputed
anything from it. The record was internally consistent in the only sense the
suite checked: the field was non-empty.

So the test that matters here is not "is there a structure" but "does the
structure agree with its own record". ``test_every_library_structure_is_self_consistent``
walks the live library and recomputes formula and weight from each SMILES. It
would have failed the day the bad string was added.

The biologic tests exist because the lookup that fills these fields does not
fail cleanly: PubChem name search answers "Interferon alfa-2b" with CID
71306834, a 746 Da compound containing Cl3I2Na. A quarter of this library's
"drugs" are antibodies, so a confident wrong answer is the common case, not the
edge case.
"""
from __future__ import annotations

import contextlib
import io
import json

import pytest

from cognisom.library.structures import (
    BIOLOGIC_PATTERNS, StructureCheck, conformer_prim_count, is_biologic,
    validate_structure,
)

# Enzalutamide, from PubChem CID 15951529.
ENZALUTAMIDE = "CC1(C(=O)N(C(=S)N1C2=CC(=C(C=C2)C(=O)NC)F)C3=CC(=C(C=C3)C#N)C(F)(F)F)C"
# The string the library shipped under that name.
NOT_ENZALUTAMIDE = ("CC1(C2CCC3(C(=O)NC(=C3C2CC(C1)(F)F)C#N)C)"
                    "C4=CC(=C(C=C4)F)C(=O)NC5=CC=C(C=N5)C(F)(F)F")


# ── the check itself ─────────────────────────────────────────────────────

def test_a_correct_structure_passes_its_own_record():
    check = validate_structure(
        ENZALUTAMIDE, expected_formula="C21H16F4N4O2S", expected_mw=464.4)

    assert check.ok, check.problems
    assert check.formula == "C21H16F4N4O2S"
    assert check.inchikey.startswith("WXCXUHSOUPDCQV")


def test_the_wrong_molecule_under_the_right_name_is_caught():
    """The regression. This is what shipped, and it must not pass again."""
    check = validate_structure(
        NOT_ENZALUTAMIDE, expected_formula="C21H16F4N4O2S", expected_mw=464.4)

    assert not check.ok
    assert check.smiles_parses, "it parses fine -- that is why it went unnoticed"
    assert any("formula" in p for p in check.problems)
    assert check.formula != "C21H16F4N4O2S"


def test_an_unparseable_smiles_is_reported_rather_than_accepted():
    check = validate_structure("this is not a molecule")

    assert not check.ok
    assert not check.smiles_parses


def test_a_zero_molecular_weight_is_treated_as_absent_not_as_a_target():
    """0.0 is what the library stores for "never filled in", not a measurement.

    430 of its 598 numeric properties are exactly zero. Comparing a real weight
    against 0.0 would fail every correct molecule in the catalogue.
    """
    check = validate_structure(ENZALUTAMIDE, expected_mw=0.0)

    assert check.ok, check.problems
    assert check.molecular_weight == pytest.approx(464.44, abs=0.1)


def test_a_structure_with_no_record_to_compare_against_still_computes():
    check = validate_structure(ENZALUTAMIDE)

    assert check.ok
    assert check.formula == "C21H16F4N4O2S"


# ── biologics ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", [
    "Pembrolizumab", "Nivolumab", "Rituximab", "Atezolizumab", "Ipilimumab",
    "Adalimumab", "Infliximab", "Eculizumab", "Relatlimab",
    "Interferon alfa-2b", "Aldesleukin (IL-2)", "Lutetium-177 PSMA",
])
def test_protein_drugs_are_refused_before_a_small_molecule_lookup(name):
    """Name search answers for the wrong molecule rather than returning nothing.

    "Interferon alfa-2b" matches CID 71306834 -- 746 Da, containing Cl3I2Na,
    nothing to do with a 19 kDa cytokine. Refusing by name is the only
    defence, because the result itself looks like a successful hit.
    """
    assert is_biologic(name), f"{name} should be routed away from PubChem"


@pytest.mark.parametrize("name", [
    "Enzalutamide", "Olaparib", "Docetaxel", "Abiraterone", "Glucose",
    "Testosterone", "ATP", "Rucaparib", "Tofacitinib",
])
def test_real_small_molecules_are_not_refused(name):
    assert not is_biologic(name), f"{name} is a small molecule"


def test_the_biologic_patterns_are_not_so_broad_they_catch_everything():
    """A guard that refuses everything would silently empty the catalogue."""
    assert BIOLOGIC_PATTERNS
    refused = [n for n in ("Glucose", "Lactate", "Pyruvate", "Citrate", "NADH",
                           "Oxygen", "Glutamine", "DHT") if is_biologic(n)]
    assert not refused, f"metabolites wrongly refused: {refused}"


# ── prim budget ──────────────────────────────────────────────────────────

def test_a_molecule_costs_one_prim_per_atom_and_bond():
    """What a structure is worth visually, against the one sphere drawn today."""
    atoms, bonds = conformer_prim_count(ENZALUTAMIDE)

    assert atoms == 32 + 16, f"expected 48 atoms with hydrogens, got {atoms}"
    assert atoms + bonds > 90, "a ball-and-stick enzalutamide is ~100 prims"


# ── the live library ─────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def library_structures():
    from cognisom.library.store import EntityStore
    with contextlib.redirect_stdout(io.StringIO()):
        store = EntityStore()
    rows = store._conn.execute(
        "SELECT name, data_json FROM entities").fetchall()
    out = []
    for name, blob in rows:
        props = (json.loads(blob or "{}").get("properties") or {})
        smiles = props.get("smiles")
        if smiles:
            out.append((name, smiles, props.get("molecular_formula"),
                        props.get("molecular_weight")))
    return out


def test_every_library_structure_is_self_consistent(library_structures):
    """Walk the live library and recompute each structure from its SMILES.

    This is the test that would have caught the shipped Enzalutamide string on
    the day it was written, and the one that keeps the next bad paste out.
    """
    if not library_structures:
        pytest.skip("no SMILES in the library yet")

    failures = []
    for name, smiles, formula, mw in library_structures:
        check = validate_structure(smiles, expected_formula=formula,
                                   expected_mw=mw)
        if not check.ok:
            failures.append(f"{name}: {'; '.join(check.problems)}")

    assert not failures, "structures disagree with their own records:\n  " + \
        "\n  ".join(failures)
