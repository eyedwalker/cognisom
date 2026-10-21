"""Every cell-type string this repository emits, and whether it resolves.

Four modules name cell types in code, and none of them could reach the
library. Measured before the fix: **19 of 19 distinct labels resolved to
nothing**, and the causes were three different things wearing one face.

Six failed on CASE ALONE. ``find_entity_by_name`` matches exactly;
``cell_state_classifier`` lowercases every label before looking one up.
So ``Macrophage`` answered and ``macrophage`` did not.

Two were compartments of the tissue simulation itself.
``DEFAULT_COMPOSITION`` declares ten archetypes and ``ARCHETYPE_ENTITY``
mapped eight: ``smooth_muscle`` at 10% and ``cancer_epithelial`` at 10%
had no entity, so a FIFTH of every simulated tissue -- including the
malignant compartment, which is what this platform is about -- fell back
to a flat 5 um sphere with nothing to curate, cite or version.

Two more were cells the code names and the library never held:
``Monocyte``, which ``seed_ultrastructure`` already lists in
``MONOCYTE_NAMES`` and holds a measured granule fraction for, and the
exhausted CD8 T cell, which is the state checkpoint blockade is aimed
at.

The rest are left failing on purpose, and this file says which and why.
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


#: Emitted by spatial_transcriptomics, coloured by 30_spatial_tissue,
#: bucketed by cell_state_classifier. Every one of these must reach an
#: entity: they name a cell.
MUST_RESOLVE = (
    "tumor_cell", "t_cell_exhausted", "monocyte", "smooth_muscle",
    "cancer_epithelial", "cancer epithelial",
    "cd8+ t cell", "cd4+ t cell", "regulatory t cell", "nk cell",
    "macrophage", "plasma cell",
)

#: Generic, with two or more specific entities under each. A form two
#: entities claim cannot resolve -- whichever is indexed first silently
#: becomes the answer -- so these are left failing visibly rather than
#: guessed at. Resolving one means deciding it has a parent entity of
#: its own, which is a modelling decision and not a lookup fix.
AMBIGUOUS = ("t cell", "b cell", "dendritic cell",
             "fibroblast", "endothelial", "epithelial")

#: Not cell types at all: the three-way classification at
#: cellular_module.py:78 (``cell_type: str  # 'normal', 'cancer',
#: 'immune'``). There is no entity for these to be, and inventing one
#: would make a category look like a cell.
CATEGORIES = ("normal", "cancer", "immune")


@pytest.fixture(scope="module")
def resolver():
    with contextlib.redirect_stdout(io.StringIO()):
        from cognisom.genomics.parameter_resolver import (
            SimulationParameterResolver)
        from cognisom.library.store import EntityStore
        return SimulationParameterResolver(EntityStore())


def _resolves(resolver, label):
    with contextlib.redirect_stdout(io.StringIO()):
        return bool(resolver.get_cell_type_params(label))


# ── the tissue simulation's own compartments ─────────────────────────

def test_every_archetype_in_the_default_composition_has_a_row():
    """A fifth of every simulated tissue had no entity behind it."""
    from cognisom.core.tissue_config import (
        ARCHETYPE_ENTITY, DEFAULT_COMPOSITION)

    unmapped = [a for a in DEFAULT_COMPOSITION if a not in ARCHETYPE_ENTITY]
    assert not unmapped, (
        f"{unmapped} are {sum(DEFAULT_COMPOSITION[a] for a in unmapped):.0%} "
        "of the default composition and have no library row")


def test_every_mapped_archetype_points_at_an_entity_that_exists():
    """A map entry naming a row that is not there is worse than no
    entry: it reports a library lookup rather than a gap."""
    with contextlib.redirect_stdout(io.StringIO()):
        from cognisom.library.store import EntityStore
        store = EntityStore()
    from cognisom.core.tissue_config import ARCHETYPE_ENTITY

    for archetype, (name, entity_type) in ARCHETYPE_ENTITY.items():
        assert store.find_entity_by_name(name, entity_type) is not None, (
            f"{archetype} -> {name!r} ({entity_type}) is not in the library")


def test_the_malignant_compartment_exists():
    """The cell that displays the neoantigen, gets killed, downregulates
    MHC-I and goes hypoxic. It had no entity."""
    with contextlib.redirect_stdout(io.StringIO()):
        from cognisom.library.store import EntityStore
        store = EntityStore()

    entity = store.find_entity_by_name("Prostatic adenocarcinoma cell")
    assert entity is not None
    markers = set(getattr(entity, "markers", []) or [])
    # The diagnosis is an absence as much as a presence: no basal layer.
    assert "AMACR+" in markers, markers
    assert "p63-" in markers and "CK5-" in markers, (
        "loss of the basal markers is what separates carcinoma from a "
        "benign mimic, and it is recorded as explicitly as the gains")


# ── the vocabulary ───────────────────────────────────────────────────

@pytest.mark.parametrize("label", MUST_RESOLVE)
def test_a_label_the_code_emits_reaches_an_entity(resolver, label):
    assert _resolves(resolver, label), (
        f"{label!r} is emitted in code and reaches no entity")


@pytest.mark.parametrize("label", AMBIGUOUS)
def test_a_generic_label_is_left_failing_rather_than_guessed(resolver, label):
    """If one of these starts resolving, somebody has decided it has a
    parent entity. That is a modelling decision and it should arrive
    with a reason -- so this test fails and names the entity that now
    claims the form, rather than quietly approving it."""
    assert not _resolves(resolver, label), (
        f"{label!r} now resolves. If a generic parent entity was added "
        "deliberately, move it to MUST_RESOLVE and say why in the commit")


@pytest.mark.parametrize("label", CATEGORIES)
def test_a_category_is_not_a_cell_type(resolver, label):
    assert not _resolves(resolver, label)


def test_case_and_punctuation_no_longer_decide_the_answer(resolver):
    """Six of the nineteen failed on case alone, while the one caller
    that looks cell types up lowercases every label first."""
    for upper, lower in (("Macrophage", "macrophage"),
                         ("NK cell", "nk cell"),
                         ("Plasma cell", "plasma cell")):
        assert _resolves(resolver, upper)
        assert _resolves(resolver, lower), f"{lower!r} lost to case"
        with contextlib.redirect_stdout(io.StringIO()):
            a = resolver.get_cell_type_params(upper)
            b = resolver.get_cell_type_params(lower)
        assert a == b


def test_an_unknown_name_still_fails_loudly(resolver):
    """The index must not become a way to get an answer to a question
    nobody asked. A name nothing claims returns nothing."""
    assert not _resolves(resolver, "Not A Cell Anyone Named")


def test_the_measured_granule_fraction_finally_has_its_monocyte():
    """seed_ultrastructure listed 'Monocyte' in MONOCYTE_NAMES and held
    BNID 115196's 2.6% for it, with no entity to attach it to."""
    with contextlib.redirect_stdout(io.StringIO()):
        from cognisom.library.store import EntityStore
        store = EntityStore()
    from cognisom.library.parameters import get_parameter
    from cognisom.library.seed_ultrastructure import MONOCYTE_NAMES

    assert "Monocyte" in MONOCYTE_NAMES
    entity = store.find_entity_by_name("Monocyte")
    assert entity is not None
    value = get_parameter(entity, "granule_volume_fraction")
    assert value is not None and value.value == pytest.approx(0.026)
