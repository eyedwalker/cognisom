"""The library's evidence travels with the repository.

Sourced values land in the library database, which is local state, so they
existed on one machine only: CI and production ran on a library with none
of them, and eighteen tests that check the evidence failed in CI while
passing locally. ``curated_parameters.json`` is the repository's copy, and
every library applies it on open.
"""
from __future__ import annotations

import contextlib
import io
import json

import pytest

from cognisom.library import curated
from cognisom.library.parameters import (
    PARAMETER_SPECS, ParameterValue, Provenance, get_parameter,
)


@pytest.fixture
def fresh_store(tmp_path):
    from cognisom.library.store import EntityStore
    with contextlib.redirect_stdout(io.StringIO()):
        return EntityStore(db_path=str(tmp_path / "entities.db"))


@pytest.fixture(scope="module")
def document():
    return json.loads(curated.CURATED_PATH.read_text())


def test_a_fresh_library_carries_the_evidence(fresh_store):
    """One value from each kind of source."""
    for name, etype, field, citation in (
        ("Enzalutamide", "drug", "half_life_hours", "FDA-SPL:"),
        ("Neutrophil", "immune_cell", "migration_speed_um_per_min", "PMID:20689600"),
        ("Macrophage", "cell_type", "radius_um", "PMID:9400735"),
    ):
        pv = get_parameter(fresh_store.find_entity_by_name(name, etype), field)
        assert pv is not None and pv.provenance is Provenance.MEASURED, (name, field)
        assert pv.citation.startswith(citation), (name, field, pv.citation)


def test_applying_is_recorded_and_not_repeated(fresh_store):
    assert fresh_store.get_meta(curated.META_KEY) == curated._digest(curated.CURATED_PATH)
    again = curated.apply_curated(fresh_store)
    assert again.skipped


def test_a_forced_reapply_changes_nothing(fresh_store):
    report = curated.apply_curated(fresh_store, force=True)
    assert report.written == 0 and not report.refused and not report.missing_entities


def test_every_record_names_a_seeded_entity_and_satisfies_its_spec(fresh_store, document):
    for item in document["entities"]:
        entity = fresh_store.find_entity_by_name(item["name"], item["entity_type"])
        assert entity is not None, (item["name"], item["entity_type"])
        for name, record in item["parameters"].items():
            pv = ParameterValue.from_dict(record)
            assert pv.provenance is not Provenance.PLACEHOLDER, (item["name"], name)
            assert pv.validate(PARAMETER_SPECS.get(name)) == [], (item["name"], name)


def test_the_file_is_what_export_would_write(fresh_store, tmp_path, document):
    """Exporting a library that applied the file reproduces the file, so the
    file is the whole of the evidence, not a sample of it."""
    path = tmp_path / "round_trip.json"
    curated.export_curated(fresh_store, path)
    assert json.loads(path.read_text()) == document


def test_a_partial_application_is_retried(fresh_store, monkeypatch):
    """A library opened while another process is still seeding it must not
    record the file as applied, or the evidence would never arrive."""
    fresh_store.set_meta(curated.META_KEY, "")
    real = fresh_store.find_entity_by_name
    monkeypatch.setattr(fresh_store, "find_entity_by_name",
                        lambda name, etype=None: None if name == "Neutrophil" else real(name, etype))
    partial = curated.apply_curated(fresh_store)
    assert ("Neutrophil", "immune_cell") in partial.missing_entities
    assert fresh_store.get_meta(curated.META_KEY) == ""
    monkeypatch.setattr(fresh_store, "find_entity_by_name", real)
    assert not curated.apply_curated(fresh_store).skipped
    assert fresh_store.get_meta(curated.META_KEY) == curated._digest(curated.CURATED_PATH)
