"""Callers must handle `add_entity` refusing, and must not claim work it didn't do.

`add_entity` once minted a fresh UUID and inserted unconditionally, which is
how the shipped database reached five copies of all 141 entities. Making it
refuse a duplicate was correct, but it turned every call site that discarded
the return value into a latent failure:

  * `scenario_builder`, `run_manager` and `project_manager` discarded the
    boolean, so rebuilding a scenario under a name already used returned an
    in-memory object whose `entity_id` was never inserted. `create_run` then
    raised `ValueError: Scenario <id> not found`. Any user running the same
    named scenario twice hit it.

  * the seeders incremented their counters regardless, so a second seed run
    reported 141 entities seeded having inserted zero rows. That is the same
    defect the evaluation layer had -- a report claiming work that did not
    happen -- in a different file.

These tests pin both halves: a refusal has to reach the caller, and a count
has to mean something.
"""
from __future__ import annotations

import contextlib
import io
import tempfile

import pytest


@pytest.fixture
def fresh_store(tmp_path):
    """A store on its own database, auto-seeded as the constructor does."""
    from cognisom.library.store import EntityStore

    with contextlib.redirect_stdout(io.StringIO()):
        return EntityStore(db_path=str(tmp_path / "entities.db"))


@pytest.fixture
def workflow(fresh_store, tmp_path):
    from cognisom.workflow.artifact_store import ArtifactStore
    from cognisom.workflow.run_manager import RunManager
    from cognisom.workflow.scenario_builder import ScenarioBuilder

    with contextlib.redirect_stdout(io.StringIO()):
        builder = ScenarioBuilder(fresh_store)
        manager = RunManager(fresh_store, ArtifactStore(str(tmp_path / "artifacts")))
    return builder, manager


# ── the contract itself ──────────────────────────────────────────────────

def test_add_entity_reports_whether_it_inserted(fresh_store):
    from cognisom.library.models import EntityType, BioEntity

    entity = BioEntity(name="A Test Cytokine", entity_type=EntityType.CYTOKINE)

    assert fresh_store.add_entity(entity) is True, "a new entity should insert"


def test_add_entity_refuses_a_duplicate_name_and_type(fresh_store):
    from cognisom.library.models import EntityType, BioEntity

    first = BioEntity(name="A Test Cytokine", entity_type=EntityType.CYTOKINE)
    second = BioEntity(name="A Test Cytokine", entity_type=EntityType.CYTOKINE)

    assert fresh_store.add_entity(first) is True
    assert fresh_store.add_entity(second) is False, (
        "a second active entity with the same name and type must be refused"
    )


def test_a_homonym_can_be_forced_when_the_caller_means_it(fresh_store):
    """Some types legitimately share a name across categories."""
    from cognisom.library.models import EntityType, BioEntity

    a = BioEntity(name="Homonym", entity_type=EntityType.CYTOKINE)
    b = BioEntity(name="Homonym", entity_type=EntityType.CYTOKINE)

    assert fresh_store.add_entity(a) is True
    assert fresh_store.add_entity(b, allow_duplicate=True) is True


# ── the call sites that discarded it ─────────────────────────────────────

def test_the_same_named_scenario_can_be_built_repeatedly(workflow):
    """The regression the duplicate-prevention fix created.

    Each build must yield an entity_id that is actually in the database,
    whether or not the name was used before.
    """
    builder, _ = workflow

    with contextlib.redirect_stdout(io.StringIO()):
        ids = [
            builder.build_immunotherapy_scenario("Atezolizumab").entity_id
            for _ in range(3)
        ]

    assert len(set(ids)) == 3, f"builds collided on one id: {ids}"
    for i, entity_id in enumerate(ids, 1):
        assert builder._store.get_entity(entity_id) is not None, (
            f"build #{i} returned id {entity_id} which is not in the database"
        )


def test_a_run_can_be_created_from_any_of_those_scenarios(workflow):
    """This raised ValueError: Scenario <id> not found on the second build."""
    builder, manager = workflow

    with contextlib.redirect_stdout(io.StringIO()):
        ids = [
            builder.build_immunotherapy_scenario("Atezolizumab").entity_id
            for _ in range(3)
        ]
        for entity_id in ids:
            manager.create_run(entity_id)  # must not raise


# ── the seeders that counted regardless ──────────────────────────────────

def test_a_second_seed_run_reports_nothing_seeded(fresh_store):
    """It used to report the full catalogue having inserted zero rows."""
    from cognisom.library import seed_immunology

    with contextlib.redirect_stdout(io.StringIO()):
        counts = seed_immunology.seed_immunology_catalog(fresh_store)

    entity_counts = {k: v for k, v in counts.items() if k != "relationships"}
    assert sum(entity_counts.values()) == 0, (
        f"the store was already seeded by its constructor, so a seeder call "
        f"inserted nothing -- but it reported {entity_counts}"
    )


def test_seeding_an_empty_store_reports_what_it_inserted(tmp_path):
    """The other direction: a real insert must still be counted."""
    from cognisom.library.store import EntityStore
    from cognisom.library import seed_immunology

    # Build the store, then empty it, so the seeder has real work to do.
    # Relationships reference entities, so they go first.
    with contextlib.redirect_stdout(io.StringIO()):
        store = EntityStore(db_path=str(tmp_path / "e.db"))
        store._conn.execute("DELETE FROM relationships")
        store._conn.execute("DELETE FROM entities")
        store._conn.commit()
        counts = seed_immunology.seed_immunology_catalog(store)

    entity_counts = {k: v for k, v in counts.items() if k != "relationships"}
    seeded = sum(entity_counts.values())
    rows = store._conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]

    assert seeded > 0, "seeding an empty store reported nothing inserted"
    assert seeded == rows, (
        f"reported {seeded} seeded but the database holds {rows} rows"
    )


def test_the_catalogue_never_duplicates_itself(tmp_path):
    """The original defect: repeated seeding multiplied the catalogue."""
    from cognisom.library.store import EntityStore
    from cognisom.library import seed_immunology

    with contextlib.redirect_stdout(io.StringIO()):
        store = EntityStore(db_path=str(tmp_path / "e.db"))
        for _ in range(3):
            seed_immunology.seed_immunology_catalog(store)

    rows = store._conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    distinct = store._conn.execute(
        "SELECT COUNT(*) FROM (SELECT DISTINCT entity_type, name FROM entities)"
    ).fetchone()[0]

    assert rows == distinct, (
        f"{rows} rows for {distinct} distinct name/type pairs -- the "
        f"catalogue duplicated itself"
    )
