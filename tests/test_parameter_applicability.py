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

    # A lineage nothing knows about is assumed to divide, so it owes a
    # division time and not a lifespan. Nothing else is excused.
    assert set(r.not_applicable) == {"lifespan_hours"}
    assert set(r.missing) == set(required_for_entity(entity))
    for p in KILL_PARAMS:
        assert p in r.missing, f"{p} was excused for an unknown lineage"


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
        "kill_radius_um", "radius_um", "lifespan_hours",
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


# ── enzymes typed as receptors ───────────────────────────────────────────

def test_an_enzyme_is_not_asked_for_a_ligand_kd(all_rows):
    """The same category error a third time, on a different axis.

    IDO1 and CD73 are enzymes by the library's own descriptions but are
    typed RECEPTOR, so kd_nm was required of them. Both carry a turnover
    number and a Michaelis constant, which is what actually describes them.
    """
    by_name = {e.name: e for e in all_rows}

    for name in ("IDO1", "CD73 (NT5E)"):
        entity = by_name[name]
        r = assess(entity)
        assert "kd_nm" in r.not_applicable, f"{name} still asked for a kd_nm"
        assert "enzyme" in r.not_applicable["kd_nm"]
        assert get_parameter(entity, "kcat_per_s") is not None, (
            f"{name} is excluded from kd_nm on the grounds that kcat "
            "describes it, so kcat had better be readable")
        assert get_parameter(entity, "km_um") is not None


def test_a_genuine_receptor_is_still_asked_for_its_kd(all_rows):
    """A regex over descriptions would have excluded these four too.

    They recruit phosphatases; they are not themselves enzymes.
    """
    by_name = {e.name: e for e in all_rows}

    for name in ("CD47", "CTLA-4 (CD152)", "TIGIT", "TIM-3 (HAVCR2)"):
        entity = by_name.get(name)
        if entity is None:
            continue
        assert "kd_nm" not in assess(entity).not_applicable, (
            f"{name} is a receptor and a Kd is exactly what it needs")


def test_the_catalytic_table_names_only_real_entities(all_rows):
    from cognisom.library.parameters import _CATALYTIC_NOT_A_RECEPTOR

    names = {e.name for e in all_rows}
    spurious = set(_CATALYTIC_NOT_A_RECEPTOR) - names
    assert not spurious, (
        f"these rows exclude nothing because no entity has the name: "
        f"{sorted(spurious)}")


# ── enzyme kinetics now carry a unit ─────────────────────────────────────

def test_a_per_minute_turnover_is_not_read_as_per_second(all_rows):
    """The 60x hazard: kcat was written in two different time units."""
    by_name = {e.name: e for e in all_rows}

    ezh2 = by_name["EZH2"]
    raw = ezh2.physics_params["methyltransferase_kcat_per_min"]
    pv = get_parameter(ezh2, "kcat_per_s")

    assert pv.value == pytest.approx(raw / 60.0), (
        f"EZH2 records {raw}/min; read as {pv.value}/s it should be "
        f"{raw / 60.0}/s, not {raw}/s")
    assert pv.unit == "1/s"


def test_every_entity_carrying_kinetics_resolves_to_the_canonical_names(all_rows):
    """Six kcat names and eleven Km names, all readable as two."""
    resolved = sum(
        1 for e in all_rows
        if get_parameter(e, "kcat_per_s") or get_parameter(e, "km_um")
    )

    assert resolved >= 26, (
        f"only {resolved} entities resolve a turnover or Michaelis "
        "constant; an alias has been dropped")


def test_the_kinetic_bounds_reject_a_misplaced_decimal():
    """Physical bounds, not biological guesses: they catch a unit slip."""
    from cognisom.library.parameters import PARAMETER_SPECS

    kcat, km = PARAMETER_SPECS["kcat_per_s"], PARAMETER_SPECS["km_um"]
    assert kcat.unit == "1/s" and km.unit == "uM"

    for spec, bad in ((kcat, 1e9), (km, 1e9)):
        pv = ParameterValue(value=bad, unit=spec.unit,
                            provenance=Provenance.ESTIMATED,
                            rationale="deliberately out of range")
        assert spec.check(pv), (
            f"{spec.name}={bad} {spec.unit} was accepted; the range check "
            "is not doing anything")


def test_aliased_names_are_not_counted_as_uninterpretable(all_rows):
    """Otherwise declaring an alias looks like it achieved nothing."""
    narrow = unspecced_names(all_rows)
    full = unspecced_names(all_rows, include_aliased=True)

    assert len(full) > len(narrow)
    assert "kinase_kcat_per_s" in full
    assert "kinase_kcat_per_s" not in narrow


# ── gaps the exclusions expose, pinned so they are not forgotten ─────────

def test_the_post_mitotic_rows_have_no_temporal_parameter(immune_rows):
    """Excluding their division time leaves them nothing. Pinned, not fixed.

    The right fix is a per-lineage lifespan, not a uniform default: a
    circulating neutrophil lives hours and a memory T cell years.
    ``apoptosis_rate_per_hour`` already applies to IMMUNE_CELL, so the
    parameter exists and only the values are missing. This test exists so
    that stays visible, and should be rewritten when they are supplied.
    """
    by_name = {e.name: e for e in immune_rows}

    for name in ("Neutrophil", "Eosinophil", "Basophil", "Plasma cell"):
        entity = by_name[name]
        assert "division_time_hours" in assess(entity).not_applicable
        assert get_parameter(entity, "apoptosis_rate_per_hour") is None, (
            f"{name} now has an apoptosis rate -- supply the rest and "
            "rewrite this test rather than leaving it asserting the gap")


def test_the_defaults_are_inverted_for_lifespan(immune_rows, all_rows):
    """immune_cell gets a division time; cell_type gets both. Backwards."""
    from cognisom.library.models import EntityType

    cells = [e for e in all_rows if e.entity_type == EntityType.CELL_TYPE]
    immune_apoptosis = sum(
        1 for e in immune_rows
        if (e.physics_params or {}).get("apoptosis_rate_per_hour"))
    cell_apoptosis = sum(
        1 for e in cells
        if (e.physics_params or {}).get("apoptosis_rate_per_hour"))

    # The three retyped endothelial rows are cell_types that carry no
    # apoptosis rate, so the count is now 12 of 15 rather than all of them.
    assert immune_apoptosis == 0, (
        f"{immune_apoptosis} immune rows now carry an apoptosis rate; the "
        "asymmetry described on _NOT_CYCLING has changed")
    assert cell_apoptosis >= 12, (
        f"only {cell_apoptosis}/{len(cells)} cell_types carry an apoptosis "
        "rate; update the note on _NOT_CYCLING")


def test_the_unitless_consumption_rate_is_still_one_default(all_rows):
    """Guards the reason oxygen_consumption_rate must not be aliased.

    If these ever become distinct per-cell measurements, a unit can be
    established and the alias becomes possible -- but not before.
    """
    values = {(e.physics_params or {}).get("oxygen_consumption_rate")
              for e in all_rows}
    values.discard(None)

    assert values == {0.05}, (
        f"oxygen_consumption_rate now holds {sorted(values)}. If these are "
        "real measurements, establish the unit and alias it to "
        "oxygen_consumption_fmol_per_cell_per_hour; until then no factor "
        "can be asserted")
