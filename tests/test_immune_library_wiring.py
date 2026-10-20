"""The immune module reads one parameter from the entity library.

Each lineage's migration speed becomes its patrol speed. It is the only field
on the immune rows with a consumer in the module: immune cells never divide,
age or die here, and have no radius. The same rows also carry detection
radius, kill probability and exhaustion rate, but those are model constructs
the module calibrates, and they must never arrive through this seam.

What is pinned here:

- a measured library value reaches the running patrol. The probe is the
  Neutrophil row (17 um/min, PMID:20689600), the only simulation-grade
  migration speed on an immune row today. The module has no neutrophil
  recognition rule, so it patrols and never engages, which makes it a clean
  probe;
- a lineage whose library speed is a placeholder patrols at patrol_speed,
  exactly as before -- the cell_type Macrophage row's stamped 0.5 um/min
  would otherwise slow every macrophage tenfold;
- explicit config outranks the library, and library_params=False turns the
  library off;
- prepared for the next curation pass: when T, NK or macrophage acquires a
  simulation-grade speed, the module uses it, and a value stored in the wrong
  unit is refused rather than rescaled.

The end-to-end proof that a default run is unchanged is the existing
nine-module pin in test_spatial_field_coupling.py, which passes unmodified:
every speed on the three modelled lineages is still a placeholder.

Steps use dt = 0.005 h. At 17 um/min that is a 5.1 um step, under the 10 um
detection-radius step cap; at the engine's default dt = 0.01 h the cap would
bind and these tests would measure the cap instead of the speed.
"""
from __future__ import annotations

import contextlib
import copy
import io
import logging

import numpy as np
import pytest

from cognisom.genomics import parameter_resolver
from cognisom.genomics.parameter_resolver import SimulationParameterResolver
from cognisom.library.parameters import (
    PARAMETER_SPECS, SIMULATION_GRADE, ParameterValue, Provenance, _sidecar,
    set_parameter,
)
from cognisom.modules.immune_module import (
    LINEAGE_ENTITY, MINUTES_PER_HOUR, ImmuneCell, ImmuneModule,
)

DT_H = 0.005
SPEED = "migration_speed_um_per_min"
NO_CELLS = {"n_t_cells": 0, "n_nk_cells": 0, "n_macrophages": 0}


def _module(**config) -> ImmuneModule:
    with contextlib.redirect_stdout(io.StringIO()):
        return ImmuneModule({**NO_CELLS, **config})


def _step_length(module: ImmuneModule, cell: ImmuneCell) -> float:
    """Displacement of one patrol step. No cellular module is linked, so
    there is no ECM attenuation and the step is speed x 60 x dt exactly."""
    before = cell.position.copy()
    module._patrol(cell, DT_H)
    return float(np.linalg.norm(cell.position - before))


def _expected(speed_um_per_min: float) -> float:
    return speed_um_per_min * MINUTES_PER_HOUR * DT_H


@pytest.fixture(autouse=True)
def _fresh_logs():
    parameter_resolver.reset_missing_log()
    yield
    parameter_resolver.reset_missing_log()


@pytest.fixture(scope="module")
def shipped_rows():
    """Real library rows, copied, for planting values a test controls."""
    from cognisom.library.store import EntityStore
    with contextlib.redirect_stdout(io.StringIO()):
        store = EntityStore()
    rows = {}
    for name, etype in list(LINEAGE_ENTITY.values()) + [("Neutrophil", "immune_cell")]:
        entity = store.find_entity_by_name(name, etype)
        assert entity is not None, f"library has no {etype} row {name!r}"
        rows[name] = entity
    return rows


class _PlantedStore:
    """A store that holds exactly the rows a test planted."""

    def __init__(self, *entities):
        self._rows = {(e.name, e.entity_type.value): e for e in entities}

    def find_entity_by_name(self, name, entity_type=None):
        return self._rows.get((name, entity_type))


@pytest.fixture
def planted(monkeypatch, shipped_rows):
    """Point the module's shared resolver at copies of real rows, edited.

    Returns a function: plant(name, pv=None, wrong_unit=None) puts a speed
    (or none) on a copy of that row and returns the copy. Rows not planted
    are absent from the store.
    """
    entities = {}

    def plant(name, pv=None, *, wrong_unit=None):
        entity = copy.deepcopy(shipped_rows[name])
        _sidecar(entity).pop(SPEED, None)
        entity.physics_params.pop(SPEED, None)
        if pv is not None:
            assert set_parameter(entity, SPEED, pv) == []
        if wrong_unit is not None:
            # A row written before set_parameter validated units -- the
            # writer refuses this today, so it can only be planted directly.
            _sidecar(entity)[SPEED] = wrong_unit.to_dict()
            entity.physics_params[SPEED] = wrong_unit.value
        entities[name] = entity
        monkeypatch.setattr(
            parameter_resolver, "_SHARED",
            SimulationParameterResolver(_PlantedStore(*entities.values())),
        )
        return entity

    return plant


def _measured(value, unit="um/min"):
    return ParameterValue(value=value, unit=unit,
                          provenance=Provenance.MEASURED,
                          citation="PMID:12345678", source="planted by a test")


# ── the shipped library ─────────────────────────────────────────────────

def test_a_measured_library_speed_reaches_the_running_patrol():
    m = _module()
    neutrophil = m.immune_cells[m.add_immune_cell([100, 100, 50], "Neutrophil")]
    assert m.lineage_patrol_speed["Neutrophil"] == pytest.approx(17.0)
    assert m.patrol_speed_source["Neutrophil"] == (
        "Neutrophil: measured:PMID:20689600")
    assert _step_length(m, neutrophil) == pytest.approx(_expected(17.0), rel=1e-5)


def test_a_cell_built_outside_add_immune_cell_is_resolved_on_its_first_step():
    """The ingestion bridge constructs ImmuneCell directly."""
    m = _module()
    cell = ImmuneCell(cell_id=99, position=np.array([100, 100, 50], dtype=np.float32),
                      cell_type="Neutrophil")
    m.immune_cells[99] = cell
    assert "Neutrophil" not in m.patrol_speed_source
    assert _step_length(m, cell) == pytest.approx(_expected(17.0), rel=1e-5)


def test_no_modelled_lineage_patrols_on_a_placeholder():
    """Holds whatever the library says: a lineage either patrols at
    patrol_speed or at a value the resolver passed as evidence."""
    m = _module()
    for lineage, (name, _) in LINEAGE_ENTITY.items():
        source = m.patrol_speed_source[lineage]
        assert source.startswith(f"{name}: "), source
        if lineage in m.lineage_patrol_speed:
            grade = source.split(": ", 1)[1].split(":")[0]
            assert grade in {p.value for p in SIMULATION_GRADE}, source
        else:
            assert m._patrol_speed_for(lineage) == m.patrol_speed


def test_the_state_says_where_each_speed_came_from():
    m = _module()
    m.add_immune_cell([100, 100, 50], "Neutrophil")
    sources = m.get_state()["patrol_speed_source"]
    assert set(LINEAGE_ENTITY) | {"Neutrophil"} <= set(sources)
    assert "PMID:20689600" in sources["Neutrophil"]


def test_an_unknown_lineage_patrols_at_the_default_and_says_why():
    m = _module()
    cell = m.immune_cells[m.add_immune_cell([100, 100, 50], "basophil")]
    assert "no immune_cell named 'basophil'" in m.patrol_speed_source["basophil"]
    assert _step_length(m, cell) == pytest.approx(_expected(5.0), rel=1e-5)


# ── what the module reads, and what it must not ─────────────────────────

def test_only_measured_quantities_are_read(monkeypatch):
    """An allowlist: the migration speed and the two contact durations.
    The same rows carry model constructs, and the next widening must not
    silently replace a calibrated knob."""
    asked = []
    real = SimulationParameterResolver.sourced_value

    def spy(self, name, etype, field):
        asked.append(field)
        return real(self, name, etype, field)

    monkeypatch.setattr(SimulationParameterResolver, "sourced_value", spy)
    m = _module()
    neutrophil = m.immune_cells[m.add_immune_cell([100, 100, 50], "Neutrophil")]
    assert set(asked) == {SPEED, "lethal_contact_minutes", "nonlethal_contact_minutes"}
    assert not any(PARAMETER_SPECS[f].model_construct for f in set(asked))
    assert (neutrophil.detection_radius, neutrophil.kill_radius) == (10.0, 5.0)
    assert m.kill_probability == 0.8


def test_the_resolver_refuses_a_construct_however_it_is_graded(shipped_rows):
    row = copy.deepcopy(shipped_rows["NK cell"])
    set_parameter(row, "kill_probability", ParameterValue(
        value=0.4, unit="dimensionless", provenance=Provenance.CALIBRATED,
        calibrated_against="a planted observable", source="planted by a test"))
    resolver = SimulationParameterResolver(_PlantedStore(row))
    pv, note = resolver.sourced_value("NK cell", "immune_cell", "kill_probability")
    assert pv is None and "model construct" in note


# ── precedence ──────────────────────────────────────────────────────────

def test_explicit_config_outranks_the_library():
    m = _module(lineage_patrol_speed={"Neutrophil": 3.0})
    cell = m.immune_cells[m.add_immune_cell([100, 100, 50], "Neutrophil")]
    assert m.patrol_speed_source["Neutrophil"] == "config"
    assert _step_length(m, cell) == pytest.approx(_expected(3.0), rel=1e-5)


def test_library_params_off_reads_nothing(monkeypatch):
    def refuse(*_):
        raise AssertionError("the library was consulted")

    monkeypatch.setattr(parameter_resolver, "shared_resolver", refuse)
    m = _module(library_params=False)
    cell = m.immune_cells[m.add_immune_cell([100, 100, 50], "Neutrophil")]
    assert m.lineage_patrol_speed == {}
    assert _step_length(m, cell) == pytest.approx(_expected(5.0), rel=1e-5)


def test_patrol_speed_still_governs_every_lineage_without_its_own(planted):
    """A runtime change to patrol_speed (the calibrator's set_parameter)
    reaches fallback lineages, because the fallback is read per step."""
    planted("NK cell")
    m = _module()
    cell = m.immune_cells[m.add_immune_cell([100, 100, 50], "NK_cell")]
    m.patrol_speed = 8.0
    assert _step_length(m, cell) == pytest.approx(_expected(8.0), rel=1e-5)


# ── prepared for the next curation pass ─────────────────────────────────

@pytest.mark.parametrize("lineage", sorted(LINEAGE_ENTITY))
def test_a_curated_speed_on_a_modelled_lineage_is_used(planted, lineage):
    name, _ = LINEAGE_ENTITY[lineage]
    planted(name, _measured(9.0))
    m = _module()
    cell = m.immune_cells[m.add_immune_cell([100, 100, 50], lineage)]
    assert m.patrol_speed_source[lineage] == f"{name}: measured:PMID:12345678"
    assert _step_length(m, cell) == pytest.approx(_expected(9.0), rel=1e-5)


def test_a_placeholder_speed_is_not_used(planted):
    """The Macrophage row's real stamp: 0.5 um/min, unknown origin."""
    row = planted("Macrophage")
    row.physics_params[SPEED] = 0.5          # a bare number, no sidecar
    m = _module()
    cell = m.immune_cells[m.add_immune_cell([100, 100, 50], "macrophage")]
    assert m.patrol_speed_source["macrophage"] == "Macrophage: placeholder: not evidence"
    assert _step_length(m, cell) == pytest.approx(_expected(5.0), rel=1e-5)


def test_a_speed_stored_in_the_wrong_unit_is_refused_not_rescaled(planted, caplog):
    """540 um/h is 9 um/min. The seam must not guess that."""
    planted("CD8+ Effector T cell", wrong_unit=_measured(540.0, unit="um/h"))
    with caplog.at_level(logging.WARNING, logger="cognisom.genomics.parameter_resolver"):
        m = _module()
    cell = m.immune_cells[m.add_immune_cell([100, 100, 50], "T_cell")]
    assert "'um/h'" in m.patrol_speed_source["T_cell"]
    assert any("um/h" in r.getMessage() for r in caplog.records)
    assert _step_length(m, cell) == pytest.approx(_expected(5.0), rel=1e-5)


def test_a_refusal_is_logged_once_however_many_modules_are_built(planted, caplog):
    planted("CD8+ Effector T cell")
    with caplog.at_level(logging.INFO, logger="cognisom.genomics.parameter_resolver"):
        for _ in range(3):
            _module()
    lines = [r for r in caplog.records
             if r.getMessage().startswith("CD8+ Effector T cell " + SPEED)]
    assert len(lines) == 1
