"""A parameter an entity cannot have is not a curation gap.

``required_for_simulation`` was asserted against ``EntityType`` alone, which
lumps a CD8 effector T cell with a plasma cell, a Treg and three rows of
vascular endothelium. All 28 immune rows were therefore required to supply a
contact kill radius, a per-contact kill probability and a division time, and
the curation report read "28 entities need kill_radius_um" when sixteen of them
form no lytic synapse and six do not divide. Thirty-eight of those rows were
questions that do not arise.

The risk in fixing it is the opposite error. A false inclusion leaves a row on
a worklist where somebody eventually reads it; a false exclusion deletes the
question and nobody looks again. So the exclusion tables are deliberately
asymmetric -- only defensible exclusions are listed -- and the tests that
matter most here are :func:`test_the_change_cannot_inflate_the_ready_count`,
which pins that making the report kinder did not make it more flattering, and
:func:`test_a_contested_lineage_is_not_excluded`, which pins the twelve
deliberate non-exclusions.
"""
from __future__ import annotations

import contextlib
import io

import pytest

from cognisom.library.models import BioEntity, EntityType
from cognisom.library.parameters import (
    _NO_CONTACT_CYTOTOXICITY, _NOT_CYCLING, PARAMETER_SPECS, ParameterError,
    ParameterValue, Provenance, assess, coverage, get_parameter,
    required_for, required_for_entity, set_parameter, unspecced_names,
)

KILL_PARAMS = ("kill_radius_um", "kill_probability")

#: Lineages deliberately NOT excluded, each for a reason recorded in
#: parameters.py. Excluding any of these would delete a live question.
CONTESTED = (
    "Regulatory T cell (Treg)", "ILC1", "CD4+ Th1 cell", "Macrophage M1",
    "Macrophage M2", "Neutrophil", "Eosinophil", "CD8+ Memory T cell",
    "NKT cell", "Gamma-delta T cell", "NK cell", "CD8+ Effector T cell",
)


@pytest.fixture(scope="module")
def immune_rows():
    from cognisom.library.store import EntityStore
    with contextlib.redirect_stdout(io.StringIO()):
        store = EntityStore()
    ids = [r[0] for r in
           store._conn.execute("SELECT entity_id FROM entities").fetchall()]
    rows = [e for e in (store.get_entity(i) for i in ids) if e]
    return [e for e in rows if e.entity_type == EntityType.IMMUNE_CELL]


@pytest.fixture(scope="module")
def all_rows():
    from cognisom.library.store import EntityStore
    with contextlib.redirect_stdout(io.StringIO()):
        store = EntityStore()
    ids = [r[0] for r in
           store._conn.execute("SELECT entity_id FROM entities").fetchall()]
    return [e for e in (store.get_entity(i) for i in ids) if e]


# ── the tables have to actually match the library ────────────────────────

def test_the_exclusion_tables_name_no_lineage_absent_from_the_library(immune_rows):
    """A mistyped key is invisible: it simply means "no exclusion"."""
    present = {(e.immune_type or "", e.immune_subtype or "") for e in immune_rows}

    for table, label in ((_NO_CONTACT_CYTOTOXICITY, "cytotoxicity"),
                         (_NOT_CYCLING, "cycling")):
        spurious = set(table) - present
        assert not spurious, (
            f"the {label} table names lineages no library row has, so those "
            f"rows silently exclude nothing: {sorted(spurious)}")


def test_every_exclusion_carries_a_reason(immune_rows, all_rows):
    """An exclusion that cannot say why is as bad as an uncited number."""
    for table in (_NO_CONTACT_CYTOTOXICITY, _NOT_CYCLING):
        for key, reason in table.items():
            assert reason and reason.strip(), f"{key} excluded with no reason"

    for entity in all_rows:
        for spec in PARAMETER_SPECS.values():
            why = spec.inapplicable_to(entity)
            assert why is None or (isinstance(why, str) and why.strip()), (
                f"{spec.name} on {entity.name} excluded with an empty reason")


# ── the two directions of error ──────────────────────────────────────────

def test_a_non_cytotoxic_lineage_is_not_asked_for_a_kill_radius(immune_rows):
    by_name = {e.name: e for e in immune_rows}

    for name in ("Plasma cell", "Naive B cell", "CD8+ Naive T cell",
                 "Regulatory T cell (Treg)"):
        entity = by_name[name]
        r = assess(entity)
        if name == "Regulatory T cell (Treg)":
            continue
        for p in KILL_PARAMS:
            assert p in r.not_applicable, f"{name} still asked for {p}"
            assert p not in r.missing and p not in r.placeholder


@pytest.mark.parametrize("name", CONTESTED)
def test_a_contested_lineage_is_not_excluded(name, immune_rows):
    """These stay applicable on purpose. See parameters.py for each reason."""
    entity = {e.name: e for e in immune_rows}[name]
    r = assess(entity)

    for p in KILL_PARAMS:
        assert p not in r.not_applicable, (
            f"{name} was excluded from {p}; a false exclusion deletes the "
            "question and nobody looks at the row again")


def test_a_post_mitotic_cell_is_not_asked_for_a_division_time(immune_rows):
    by_name = {e.name: e for e in immune_rows}

    for name in ("Neutrophil", "Eosinophil", "Basophil", "Plasma cell"):
        r = assess(by_name[name])
        assert "division_time_hours" in r.not_applicable, (
            f"{name} is post-mitotic but is still asked when it divides")


def test_an_unknown_lineage_stays_applicable():
    """The default must be "applies", so a new row is never silently excused."""
    entity = BioEntity(name="novel subset", entity_type=EntityType.IMMUNE_CELL)
    entity.immune_type = "not_a_known_lineage"
    entity.immune_subtype = "novel"

    r = assess(entity)

    assert not r.not_applicable
    assert set(r.missing) == set(required_for(EntityType.IMMUNE_CELL))


# ── honesty of the headline numbers ──────────────────────────────────────

def test_the_change_cannot_inflate_the_ready_count(immune_rows):
    """Removing bogus worklist rows must not make anything look curated."""
    report = coverage(immune_rows)

    assert report.ready == 0, (
        "no immune row has a single curated parameter; if this reads "
        "non-zero, applicability has been mistaken for evidence")


def test_per_parameter_buckets_sum_to_the_entity_count(immune_rows):
    report = coverage(immune_rows)

    for name in required_for(EntityType.IMMUNE_CELL):
        curated = sum(1 for e in immune_rows if name in assess(e).curated)
        total = (report.by_parameter_missing.get(name, 0)
                 + report.by_parameter_placeholder.get(name, 0)
                 + report.by_parameter_not_applicable.get(name, 0)
                 + curated)
        assert total == report.total, (
            f"{name}: buckets sum to {total}, not {report.total}; an entity "
            "has fallen out of every bucket or been counted twice")


def test_not_applicable_is_not_a_curation_target(immune_rows):
    report = coverage(immune_rows)

    assert "kill_radius_um" not in report.by_parameter_uncurated or (
        report.by_parameter_uncurated["kill_radius_um"]
        < report.total), "the worklist still claims every row needs a kill radius"
    for name, count in report.by_parameter_not_applicable.items():
        assert report.by_parameter_uncurated.get(name, 0) + count <= report.total


def test_an_entity_nothing_is_required_of_is_not_reported_ready(all_rows):
    """157/294 "simulation-ready" was 148 entities that cannot fail.

    Twelve types declare no required parameter at all, so every gene, drug,
    pathway and mutation passed the check vacuously.
    """
    report = coverage(all_rows)

    assert report.not_assessable > 0
    assert report.assessable == report.total - report.not_assessable
    assert report.ready < 20, (
        "the ready count has returned to counting entities that were never "
        "assessed")
    assert report.fraction_ready == pytest.approx(
        report.ready / report.assessable)

    gene = next(e for e in all_rows if e.entity_type == EntityType.GENE)
    r = assess(gene)
    assert not r.assessable
    assert not r.is_ready
    assert "readiness does not apply" in r.describe()


# ── the writer must agree with the report ────────────────────────────────

def test_an_inapplicable_parameter_cannot_be_written(immune_rows):
    """Otherwise the stored number and the report disagree."""
    plasma = {e.name: e for e in immune_rows}["Plasma cell"]

    with pytest.raises(ParameterError) as exc:
        set_parameter(plasma, "kill_probability", ParameterValue(
            value=0.3, unit="dimensionless", provenance=Provenance.ESTIMATED,
            rationale="a default"))

    assert "does not apply" in str(exc.value)
    assert "widen the spec" in str(exc.value), (
        "a curator who disagrees must be told to widen the spec rather than "
        "reach for strict=False")


def test_a_cytotoxic_lineage_can_still_be_written(immune_rows):
    nk = {e.name: e for e in immune_rows}["NK cell"]

    problems = set_parameter(nk, "kill_probability", ParameterValue(
        value=0.4, unit="dimensionless", provenance=Provenance.ESTIMATED,
        rationale="test write on an applicable row"), strict=False)

    assert problems == []


def test_required_for_is_still_type_level():
    """curation_report uses it as a skip guard and a test pins its keys."""
    assert set(required_for(EntityType.IMMUNE_CELL)) == {
        "detection_radius_um", "division_time_hours", "kill_probability",
        "kill_radius_um", "radius_um",
    }


def test_required_for_entity_narrows_where_required_for_cannot(immune_rows):
    plasma = {e.name: e for e in immune_rows}["Plasma cell"]

    assert set(required_for(plasma.entity_type)) > set(required_for_entity(plasma))
    assert "kill_radius_um" not in required_for_entity(plasma)
    assert "radius_um" in required_for_entity(plasma)


# ── the read path is deliberately untouched ──────────────────────────────

def test_the_read_path_still_returns_the_stored_number(immune_rows):
    """parameter_resolver breaks a name collision on `is not None`.

    Suppressing reads is a separate change; if it happens here it takes the
    NK-cell tie-break with it.
    """
    plasma = {e.name: e for e in immune_rows}["Plasma cell"]

    pv = get_parameter(plasma, "kill_probability")
    assert pv is not None and pv.value == 0.3


def test_no_immune_parameter_is_carried_without_a_spec(immune_rows):
    """exhaustion_rate was the only one, and it now has a unit and a range."""
    assert unspecced_names(immune_rows) == {}
