"""Which cells are allowed to kill, and which were allowed to by accident.

Twenty-eight cells carried ``kill_probability = 0.3``. Among them: a
plasma cell, which is a dedicated antibody factory with no lytic
granules; a regulatory T cell, whose effector programme is suppression;
an M2 macrophage, whose programme is repair; both B cells; the Th2, Th17
and Tfh helpers; and **three endothelial rows**, which is not even a
question of lineage -- ``kill_probability`` is declared for
``immune_cell`` and those three are ``cell_type``.

Every one of them killed at the same per-contact rate as a CD8 effector,
so no simulated immune response could be sensitive to composition, which
is most of what a tumour microenvironment model is for.

The library already knew. ``models.py`` says of these fields that "for
the non-cytotoxic types that is not a missing value, it is a wrong one".
``kill_probability``'s spec carries ``inapplicable_when`` and
``set_parameter`` refuses a write that violates it. The enrichment pass
went around both by assigning ``entity.physics_params`` wholesale.

That produced the precise drift the single-writer rule exists to stop:
``assess()`` reported ``kill_probability`` NOT APPLICABLE for a plasma
cell while 0.3 sat in its ``physics_params``.
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

#: Cells whose whole function is something other than contact killing.
#: Each is in _NO_CONTACT_CYTOTOXICITY with a reason.
MUST_NOT_KILL = (
    "Plasma cell", "Naive B cell", "Memory B cell",
    "CD4+ Th2 cell", "CD4+ Th17 cell",
    "T follicular helper (Tfh)", "ILC2", "ILC3",
    "Conventional dendritic cell (cDC)", "Plasmacytoid dendritic cell (pDC)",
)

#: Contested, and deliberately NOT excluded. A Treg can express granzyme
#: B, a Th1 can be cytotoxic, an M2 kills in some settings --
#: test_parameter_applicability keeps them applicable on the grounds
#: that "a false exclusion deletes the question and nobody looks at the
#: row again", and that is the better call.
#:
#: They keep a PLACEHOLDER-graded 0.3, which is not simulation-grade, so
#: a curation report lists them as work to do rather than as evidence.
#: What changed around them is that the thirteen settled non-cytotoxic
#: lineages no longer share that number, so the value is now a question
#: about three contested cells rather than a property of everything.
CONTESTED_AND_UNSOURCED = (
    "Regulatory T cell (Treg)", "CD4+ Th1 cell", "Macrophage M2",
)

#: Cells that do kill by contact. They still carry a PLACEHOLDER 0.3 --
#: sourcing them individually is the next job, and this file does not
#: pretend otherwise.
MAY_KILL = ("CD8+ Effector T cell", "NK cell", "NKT cell",
            "Gamma-delta T cell", "ILC1")


@pytest.fixture(scope="module")
def store():
    with contextlib.redirect_stdout(io.StringIO()):
        from cognisom.library.store import EntityStore
        return EntityStore()


@pytest.mark.parametrize("name", MUST_NOT_KILL)
def test_a_non_cytotoxic_cell_carries_no_kill_probability(store, name):
    from cognisom.library.parameters import get_parameter

    entity = store.find_entity_by_name(name, "immune_cell")
    assert entity is not None, name
    assert get_parameter(entity, "kill_probability") is None, (
        f"{name} still kills by contact at the same rate as a CD8 effector")


@pytest.mark.parametrize("name", MUST_NOT_KILL)
def test_the_report_and_the_store_agree(store, name):
    """The drift this closes: assess() called kill_probability not
    applicable for a plasma cell while 0.3 sat in its physics_params."""
    from cognisom.library.parameters import assess, get_parameter

    entity = store.find_entity_by_name(name, "immune_cell")
    readiness = assess(entity)
    if "kill_probability" in readiness.not_applicable:
        assert get_parameter(entity, "kill_probability") is None, (
            f"{name}: the report calls it not applicable and a value is stored")


@pytest.mark.parametrize("name", MAY_KILL)
def test_a_cytotoxic_cell_keeps_its_parameter(store, name):
    """The point is selectivity, not deletion. Emptying every row would
    have made the resolver's collision-breaker useless in the other
    direction."""
    from cognisom.library.parameters import get_parameter

    entity = store.find_entity_by_name(name, "immune_cell")
    value = get_parameter(entity, "kill_probability")
    assert value is not None, f"{name} can no longer kill anything"
    assert 0.0 < value.value <= 1.0


def test_no_cell_type_row_carries_a_per_contact_kill_chance(store):
    """kill_probability is declared for immune_cell. Three endothelial
    rows carried one, because they were enriched with the immune_cell
    defaults -- a vessel wall with a per-contact kill chance."""
    from cognisom.library.parameters import get_parameter

    offenders = []
    for row in store._conn.execute(
            "SELECT name FROM entities WHERE entity_type = 'cell_type'"):
        entity = store.find_entity_by_name(row[0], "cell_type")
        if entity is not None and get_parameter(entity, "kill_probability"):
            offenders.append(row[0])
    assert not offenders, offenders


def test_the_defaults_are_filtered_before_they_are_written(store):
    """The fix is at the write, not only in a cleanup pass: enrichment
    now applies the same two checks set_parameter makes."""
    from cognisom.library.enrichment import EntityEnricher
    from cognisom.library.models import EntityType

    enricher = EntityEnricher(store)
    plasma = store.find_entity_by_name("Plasma cell", "immune_cell")
    applicable = enricher._applicable_defaults(plasma, "immune_cell")
    assert "kill_probability" not in applicable
    # and a cell that does kill still gets it
    nk = store.find_entity_by_name("NK cell", "immune_cell")
    assert "kill_probability" in enricher._applicable_defaults(nk, "immune_cell")
    # a default with no spec is left alone
    assert "migration_speed_um_per_min" in applicable


def test_stripping_is_idempotent_and_reports_what_it_removed(store):
    """A silent deletion of stored data is not something a caller should
    have to take on trust."""
    from cognisom.library.enrichment import EntityEnricher

    assert EntityEnricher(store).strip_inapplicable_parameters() == []


def test_the_collision_breaker_now_discriminates(store):
    """'NK cell' exists as both a cell_type and an immune_cell, and the
    resolver breaks the tie by asking which row carries cytotoxic
    fields. While all 28 rows carried them that test was always true.
    """
    from cognisom.library.parameters import get_parameter

    as_immune = store.find_entity_by_name("NK cell", "immune_cell")
    as_cell = store.find_entity_by_name("NK cell", "cell_type")
    assert as_immune is not None and as_cell is not None
    assert get_parameter(as_immune, "kill_probability") is not None
    assert get_parameter(as_cell, "kill_probability") is None

    with contextlib.redirect_stdout(io.StringIO()):
        from cognisom.genomics.parameter_resolver import (
            SimulationParameterResolver)
        params = SimulationParameterResolver(store).get_cell_type_params(
            "NK cell")
    assert params["_entity_type"] == "immune_cell"
    assert params["_shadowed_by"] == "cell_type"


def test_composition_can_now_change_the_answer(store):
    """The consequence worth stating. Twenty-eight of the library's cell
    rows carried one kill probability, so no simulated response could be
    sensitive to what it was made of -- which is most of what a tumour
    microenvironment model is for. Thirteen settled non-cytotoxic
    lineages have lost it, and a plasma cell, a B cell and a dendritic
    cell no longer kill like a CD8 effector.

    Three contested lineages keep a placeholder on purpose; they are
    counted separately here rather than quietly folded into either
    side.
    """
    from cognisom.library.parameters import Provenance, get_parameter

    def graded(names):
        out = {"none": 0, "placeholder": 0, "evidence": 0}
        for name in names:
            value = get_parameter(
                store.find_entity_by_name(name, "immune_cell"),
                "kill_probability")
            if value is None:
                out["none"] += 1
            elif value.provenance is Provenance.PLACEHOLDER:
                out["placeholder"] += 1
            else:
                out["evidence"] += 1
        return out

    assert graded(MUST_NOT_KILL)["none"] == len(MUST_NOT_KILL)
    assert graded(CONTESTED_AND_UNSOURCED)["placeholder"] == 3
    assert graded(MAY_KILL)["none"] == 0

    # And nothing anywhere is yet EVIDENCE: sourcing the genuinely
    # cytotoxic ones individually is the next job, and this says so
    # rather than letting a placeholder pass for a measurement.
    everything = MUST_NOT_KILL + CONTESTED_AND_UNSOURCED + MAY_KILL
    assert graded(everything)["evidence"] == 0


@pytest.mark.parametrize("name", CONTESTED_AND_UNSOURCED)
def test_a_contested_lineage_keeps_the_question_and_loses_the_answer(
        store, name):
    """The distinction this whole change turns on. These three stay
    APPLICABLE -- excluding them would delete a live question -- and
    they no longer carry a number nobody measured."""
    from cognisom.library.parameters import assess, get_parameter

    from cognisom.library.parameters import Provenance

    entity = store.find_entity_by_name(name, "immune_cell")
    readiness = assess(entity)
    assert "kill_probability" not in readiness.not_applicable, (
        f"{name} was excluded; it is contested, not settled")
    # It keeps a value, and the value keeps its grade. PLACEHOLDER is
    # not simulation-grade, so a curation report lists it as needing
    # work rather than counting it as evidence -- which is the question
    # staying open, stated in the one way a reader cannot misread.
    value = get_parameter(entity, "kill_probability")
    assert value is not None
    assert value.provenance is Provenance.PLACEHOLDER
    assert not value.is_simulation_grade
    assert "kill_probability" in readiness.placeholder
