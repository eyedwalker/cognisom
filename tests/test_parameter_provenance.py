"""
Tests for provenanced simulation parameters.

The system exists to make one thing impossible: storing a number that
looks more authoritative than its evidence. Most of these tests are
about what the system *refuses* to do.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cognisom.library.models import (
    BioEntity, CellTypeEntity, Cytokine, EntityType, ImmuneCellEntity, Receptor,
)
from cognisom.library.parameters import (
    PARAMETER_SPECS,
    ParameterError,
    ParameterValue,
    Provenance,
    absorb_legacy_attribution,
    assess,
    coverage,
    from_kinetic_parameter,
    get_parameter,
    iter_parameters,
    required_for,
    set_parameter,
    specs_for,
)


def _measured(value=0.28, unit="h"):
    return ParameterValue(
        value=value, unit=unit, provenance=Provenance.MEASURED,
        source="SABIO-RK", citation="PMID:2985038",
        conditions="human serum, 37C",
    )


# ── Evidence obligations ─────────────────────────────────────────────

def test_measured_without_a_citation_is_rejected():
    """This is the core rule. A guess may not be recorded as a
    measurement, because downstream nothing can tell them apart."""
    pv = ParameterValue(value=0.28, unit="h", provenance=Provenance.MEASURED)
    problems = pv.validate(PARAMETER_SPECS["half_life_hours"])
    assert any("citation" in p for p in problems)


def test_estimated_without_a_rationale_is_rejected():
    pv = ParameterValue(value=0.28, unit="h", provenance=Provenance.ESTIMATED)
    assert any("rationale" in p for p in pv.validate())


def test_derived_without_a_rationale_is_rejected():
    pv = ParameterValue(value=0.28, unit="h", provenance=Provenance.DERIVED)
    assert any("rationale" in p for p in pv.validate())


def test_estimated_with_a_rationale_is_accepted():
    pv = ParameterValue(
        value=0.28, unit="h", provenance=Provenance.ESTIMATED,
        rationale="by analogy to IL-15, which shares the gamma chain",
    )
    assert pv.validate() == []
    assert pv.is_simulation_grade


def test_placeholder_needs_no_evidence_but_is_not_simulation_grade():
    """Placeholders must be sayable. Demos and smoke tests legitimately
    need a number; what they must not do is imply it is real."""
    pv = ParameterValue(value=1.0, unit="h", provenance=Provenance.PLACEHOLDER)
    assert pv.validate() == []
    assert not pv.is_simulation_grade
    assert not pv.is_publication_grade


def test_measured_and_derived_are_publication_grade_but_estimated_is_not():
    def pv(prov, **kw):
        return ParameterValue(value=1.0, unit="h", provenance=prov, **kw)

    assert pv(Provenance.MEASURED, citation="PMID:1").is_publication_grade
    assert pv(Provenance.DERIVED, rationale="x").is_publication_grade
    assert not pv(Provenance.ESTIMATED, rationale="x").is_publication_grade


def test_a_bare_number_is_not_a_quantity():
    pv = ParameterValue(value=0.28, unit="", provenance=Provenance.PLACEHOLDER)
    assert any("unit" in p for p in pv.validate())


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_values_are_rejected(bad):
    pv = ParameterValue(value=bad, unit="h", provenance=Provenance.PLACEHOLDER)
    assert any("finite" in p for p in pv.validate())


# ── Spec enforcement ─────────────────────────────────────────────────

def test_wrong_unit_is_rejected():
    """A diffusion coefficient in cm^2/s is off by eight orders of
    magnitude from one in um^2/s, and silently produces wrong dynamics."""
    il2 = Cytokine(name="IL-2")
    with pytest.raises(ParameterError, match="declared in"):
        set_parameter(il2, "diffusion_coeff_um2_s", ParameterValue(
            value=100.0, unit="cm^2/s", provenance=Provenance.ESTIMATED,
            rationale="literature range",
        ))


def test_value_outside_the_plausible_range_is_rejected():
    cell = ImmuneCellEntity(name="NK cell")
    with pytest.raises(ParameterError, match="above the plausible maximum"):
        set_parameter(cell, "kill_probability", ParameterValue(
            value=1.4, unit="dimensionless", provenance=Provenance.ESTIMATED,
            rationale="typo for 0.4",
        ))


def test_zero_kill_probability_is_allowed():
    """Zero is a real, meaningful value for a non-cytotoxic cell. It is
    the uniform 0.8 default that was wrong, not zero."""
    treg = ImmuneCellEntity(name="Regulatory T cell")
    set_parameter(treg, "kill_probability", ParameterValue(
        value=0.0, unit="dimensionless", provenance=Provenance.ESTIMATED,
        rationale="Tregs are suppressive, not cytotoxic",
    ))
    assert get_parameter(treg, "kill_probability").value == 0.0


def test_parameter_not_declared_for_this_entity_type_is_rejected():
    cyt = Cytokine(name="IL-2")
    with pytest.raises(ParameterError, match="not declared for"):
        set_parameter(cyt, "kill_probability", ParameterValue(
            value=0.5, unit="dimensionless", provenance=Provenance.ESTIMATED,
            rationale="nonsense on a cytokine",
        ))


def test_non_strict_mode_reports_instead_of_raising():
    il2 = Cytokine(name="IL-2")
    problems = set_parameter(
        il2, "half_life_hours",
        ParameterValue(value=0.28, unit="h", provenance=Provenance.MEASURED),
        strict=False,
    )
    assert problems
    assert "half_life_hours" not in il2.physics_params, (
        "an invalid value must not be stored even in non-strict mode"
    )


# ── Backward compatibility and round-tripping ────────────────────────

def test_flat_read_path_is_preserved():
    """Existing consumers read physics_params[name] as a scalar. That
    must keep working, or this system breaks the pipeline it is meant
    to improve."""
    il2 = Cytokine(name="IL-2")
    set_parameter(il2, "half_life_hours", _measured())
    assert il2.physics_params["half_life_hours"] == 0.28
    assert isinstance(il2.physics_params["half_life_hours"], float)


def test_provenance_survives_a_store_round_trip():
    il2 = Cytokine(name="IL-2")
    set_parameter(il2, "half_life_hours", _measured())

    restored = BioEntity.from_dict(il2.to_dict())
    pv = get_parameter(restored, "half_life_hours")

    assert pv.value == 0.28
    assert pv.provenance is Provenance.MEASURED
    assert pv.citation == "PMID:2985038"
    assert pv.conditions == "human serum, 37C"


def test_a_scalar_with_no_recorded_provenance_reads_as_placeholder():
    """Numbers already in physics_params have unknown origin. Reporting
    them as anything better would invent provenance."""
    cyt = Cytokine(name="IL-6")
    cyt.physics_params["half_life_hours"] = 1.0

    pv = get_parameter(cyt, "half_life_hours")
    assert pv.provenance is Provenance.PLACEHOLDER
    assert not pv.is_simulation_grade


def test_missing_parameter_reads_as_none():
    assert get_parameter(Cytokine(name="IL-2"), "half_life_hours") is None


def test_iter_parameters_skips_bookkeeping_keys():
    il2 = Cytokine(name="IL-2")
    set_parameter(il2, "half_life_hours", _measured())
    il2.physics_params["_source"] = "legacy"

    names = {name for name, _ in iter_parameters(il2)}
    assert "half_life_hours" in names
    assert "_source" not in names
    assert "parameter_provenance" not in names


# ── Readiness and coverage ───────────────────────────────────────────

def test_entity_is_not_ready_until_required_parameters_are_curated():
    cell = ImmuneCellEntity(name="NK cell")
    assert not assess(cell).is_ready
    assert set(assess(cell).missing) == set(required_for(EntityType.IMMUNE_CELL))


def test_placeholder_values_do_not_make_an_entity_ready():
    """The whole point: a number present but unevidenced still counts as
    uncurated."""
    cell = ImmuneCellEntity(name="NK cell")
    for name in required_for(EntityType.IMMUNE_CELL):
        cell.physics_params[name] = 1.0

    readiness = assess(cell)
    assert not readiness.is_ready
    assert not readiness.missing, "the values are present..."
    assert readiness.placeholder, "...but they carry no evidence"


def test_curating_every_required_parameter_makes_an_entity_ready():
    cell = ImmuneCellEntity(name="NK cell")
    for name, spec in required_for(EntityType.IMMUNE_CELL).items():
        set_parameter(cell, name, ParameterValue(
            value=1.0, unit=spec.unit, provenance=Provenance.ESTIMATED,
            rationale="test fixture",
        ))
    assert assess(cell).is_ready


def test_coverage_counts_and_prioritizes():
    ready = ImmuneCellEntity(name="curated")
    for name, spec in required_for(EntityType.IMMUNE_CELL).items():
        set_parameter(ready, name, ParameterValue(
            value=1.0, unit=spec.unit, provenance=Provenance.ESTIMATED,
            rationale="test fixture",
        ))
    blank = ImmuneCellEntity(name="uncurated")

    report = coverage([ready, blank])
    assert report.total == 2
    assert report.ready == 1
    assert report.fraction_ready == 0.5
    assert report.by_parameter_missing["kill_probability"] == 1


# ── Ingest paths ─────────────────────────────────────────────────────

class _FakeKinetic:
    """Shaped like kinetics_client.KineticParameter."""
    parameter_type = "Km"
    value = 25.0
    unit = "nM"
    source_db = "SABIO-RK"
    pubmed_id = "12482937"
    organism = "Homo sapiens"
    tissue = ""
    ph = 7.0
    temperature_c = 37.0
    conditions = ""


def test_measured_database_record_becomes_a_measured_value():
    pv = from_kinetic_parameter(_FakeKinetic())
    assert pv.provenance is Provenance.MEASURED
    assert pv.citation == "PMID:12482937"
    assert pv.source == "SABIO-RK"
    assert "37" in pv.conditions
    assert pv.validate() == []


def test_database_record_without_a_publication_is_not_measured():
    """No PMID means no citation, so it cannot claim MEASURED."""
    kp = _FakeKinetic()
    kp.pubmed_id = ""
    pv = from_kinetic_parameter(kp)

    assert pv.provenance is Provenance.ESTIMATED
    assert pv.rationale
    assert pv.validate() == []


def test_legacy_whole_entity_attribution_is_migrated_per_parameter():
    """kinetics_client stamped one _source/_pmid for the whole entity,
    so an entity with values from two papers could record only one."""
    r = Receptor(name="EGFR")
    r.physics_params.update({
        "kd_nm": 2.0,
        "_source": "SABIO-RK",
        "_pmid": "12482937",
        "_conditions": "37C, pH 7.0",
    })

    migrated = absorb_legacy_attribution(r)
    assert migrated == 1

    pv = get_parameter(r, "kd_nm")
    assert pv.provenance is Provenance.MEASURED
    assert pv.citation == "PMID:12482937"


def test_migration_does_not_overwrite_a_curated_value():
    r = Receptor(name="EGFR")
    set_parameter(r, "kd_nm", ParameterValue(
        value=2.0, unit="nM", provenance=Provenance.MEASURED,
        citation="PMID:99999", source="curated by hand",
    ))
    r.physics_params.update({"_source": "SABIO-RK", "_pmid": "12482937"})

    absorb_legacy_attribution(r)
    assert get_parameter(r, "kd_nm").citation == "PMID:99999"


def test_migration_is_a_noop_without_legacy_attribution():
    r = Receptor(name="EGFR")
    r.physics_params["kd_nm"] = 2.0
    assert absorb_legacy_attribution(r) == 0


# ── The spec registry itself ─────────────────────────────────────────

def test_every_spec_declares_a_unit_and_applies_to_something():
    for name, spec in PARAMETER_SPECS.items():
        assert spec.unit, f"{name} has no unit"
        assert spec.applies_to, f"{name} applies to no entity type"
        assert spec.description, f"{name} has no description"


def test_cell_related_types_declare_required_parameters():
    """If a type declares nothing required, coverage reporting silently
    calls every one of its entities ready."""
    for etype in (
        EntityType.CYTOKINE, EntityType.IMMUNE_CELL,
        EntityType.CELL_TYPE, EntityType.RECEPTOR,
    ):
        assert required_for(etype), f"{etype.value} declares no requirements"


def test_specs_for_filters_by_entity_type():
    assert "kill_probability" in specs_for(EntityType.IMMUNE_CELL)
    assert "kill_probability" not in specs_for(EntityType.CYTOKINE)
    assert "half_life_hours" in specs_for(EntityType.CYTOKINE)


def test_cell_type_entity_can_carry_division_time():
    """CellTypeEntity had no numeric fields at all, which is why a
    simulator could not instantiate a behaving cell from one."""
    luminal = CellTypeEntity(name="Luminal epithelial cell")
    set_parameter(luminal, "division_time_hours", ParameterValue(
        value=36.0, unit="h", provenance=Provenance.ESTIMATED,
        rationale="midpoint of the 24-48h range given in the entity "
                  "description; needs a primary source",
    ))
    assert get_parameter(luminal, "division_time_hours").value == 36.0
