"""A library that already exists catches up with the seeds. It used not to.

Why this file exists
--------------------
The library seeded itself only when EMPTY. A deployed library is never
empty, so every entity a seed module gained after that library was created
skipped it for good -- and the curated values for those entities, which
can only be written onto an entity that exists, landed nowhere.
Development and CI build a fresh library each time and cannot see this.

It was found on production the day the public cells page went live
(2026-11-12): the prostate cancer cell had no entry, the page headed its
section "None", and 108 entities named by the curated file were absent.

The catch-up must be additive in the strictest sense, because it runs
unattended against a production database: it may add a row that is not
there and must never change one that is.
"""

from __future__ import annotations

import contextlib
import io
import logging

import pytest

from cognisom.library.store import SEED_META_KEY, EntityStore, _seed_digest

CANCER = ("Prostatic adenocarcinoma cell", "cell_type")
OTHER = ("NK cell", "immune_cell")


def _open(path) -> EntityStore:
    with contextlib.redirect_stdout(io.StringIO()):
        return EntityStore(db_path=str(path))


@pytest.fixture(autouse=True)
def _no_real_database(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("DB_SECRET_ARN", raising=False)


@pytest.fixture()
def old_library(tmp_path):
    """A library as production's was: built, then missing an entity a
    later seed added, with no record of having caught up."""
    path = tmp_path / "library" / "entities.db"
    path.parent.mkdir()
    store = _open(path)
    assert store.find_entity_by_name(*CANCER) is not None
    cancer_id = store.find_entity_by_name(*CANCER).entity_id
    store._execute("DELETE FROM relationships WHERE source_id = ? OR target_id = ?",
                   (cancer_id, cancer_id))
    store._execute("DELETE FROM entities WHERE entity_id = ?", (cancer_id,))
    store._execute("DELETE FROM library_meta WHERE key = ?", (SEED_META_KEY,))
    store.conn.commit()
    assert store.find_entity_by_name(*CANCER) is None
    store.close()
    return path


def test_a_fresh_library_has_nothing_to_catch_up_with(tmp_path):
    path = tmp_path / "library" / "entities.db"
    path.parent.mkdir()
    store = _open(path)
    assert store.get_meta(SEED_META_KEY) == _seed_digest()
    assert store._add_missing_seeded_entities() == 0


def test_an_entity_the_library_never_had_arrives_on_the_next_open(old_library):
    store = _open(old_library)
    entity = store.find_entity_by_name(*CANCER)
    assert entity is not None, "the cancer cell is still missing after an open"
    assert store.get_meta(SEED_META_KEY) == _seed_digest()


def test_it_arrives_with_its_curated_evidence(old_library):
    """The point of catching up: the curated radius had nowhere to land."""
    store = _open(old_library)
    entity = store.find_entity_by_name(*CANCER)
    assert "23626725" in repr(entity.to_dict()), (
        "the cancer cell came back without its measured size's citation")


def test_an_entity_already_there_is_not_touched(old_library):
    import sqlite3
    conn = sqlite3.connect(str(old_library))
    conn.execute("UPDATE entities SET description = ? WHERE name = ? AND entity_type = ?",
                 ("EDITED BY HAND", *OTHER))
    before = conn.execute(
        "SELECT entity_id, data_json, updated_at FROM entities WHERE name = ? "
        "AND entity_type = ?", OTHER).fetchone()
    n_before = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    conn.commit(); conn.close()

    _open(old_library).close()

    conn = sqlite3.connect(str(old_library))
    after = conn.execute(
        "SELECT entity_id, data_json, updated_at FROM entities WHERE name = ? "
        "AND entity_type = ?", OTHER).fetchone()
    described = conn.execute(
        "SELECT description FROM entities WHERE name = ? AND entity_type = ?",
        OTHER).fetchone()[0]
    n_after = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    conn.close()
    assert after[0] == before[0], "an existing entity was replaced"
    assert described == "EDITED BY HAND"
    assert n_after == n_before + 1, "exactly the one missing entity was added"


def test_an_entity_archived_on_purpose_stays_archived(old_library):
    import sqlite3
    conn = sqlite3.connect(str(old_library))
    conn.execute("UPDATE entities SET status = 'archived' WHERE name = ? AND "
                 "entity_type = ?", OTHER)
    conn.commit(); conn.close()

    _open(old_library).close()

    conn = sqlite3.connect(str(old_library))
    rows = conn.execute("SELECT status FROM entities WHERE name = ? AND "
                        "entity_type = ?", OTHER).fetchall()
    conn.close()
    assert rows == [("archived",)], rows


def test_the_second_open_does_no_work(old_library, caplog):
    _open(old_library).close()
    store = _open(old_library)
    with caplog.at_level(logging.INFO, logger="cognisom.library.store"):
        assert store._add_missing_seeded_entities() == 0
    assert "seeded entities this library lacked" not in caplog.text


def test_opening_twice_never_duplicates_anything(old_library):
    _open(old_library).close()
    store = _open(old_library)
    store._execute("DELETE FROM library_meta WHERE key = ?", (SEED_META_KEY,))
    store.conn.commit()
    assert store._add_missing_seeded_entities() == 0
    rows = store._execute(
        "SELECT lower(name), entity_type, COUNT(*) FROM entities "
        "WHERE entity_type NOT IN ('simulation_scenario','simulation_run',"
        "'research_project','parameter_set') "
        "GROUP BY lower(name), entity_type HAVING COUNT(*) > 1").fetchall()
    assert not rows, [tuple(r) for r in rows][:5]


def test_a_changed_seed_is_what_triggers_the_next_catch_up(old_library, monkeypatch):
    from cognisom.library import store as store_module
    _open(old_library).close()
    monkeypatch.setattr(store_module, "_seed_digest", lambda: "a-new-seed")
    store = _open(old_library)
    assert store.get_meta(SEED_META_KEY) == "a-new-seed"
