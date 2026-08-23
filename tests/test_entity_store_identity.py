"""
Entity identity in the biological library.

``entity_id`` is a fresh UUID minted on every insert, so it can never
collide -- which means the ``INSERT OR IGNORE`` / ``ON CONFLICT
(entity_id) DO NOTHING`` clauses in the store skipped nothing, ever. The
library's real identity key is ``(name, entity_type)``, and nothing
checked it.

The only guard was ``_auto_seed_if_empty``, which runs the seeders when
``total_entities == 0``. Any direct seeder call bypassed it and inserted
the whole catalogue again. The database shipped in this repo shows the
result: 705 rows over 141 distinct (name, type) pairs -- every entity
stored five times, from five seed runs.

The duplication was also invisible, because the two auditors that would
have reported it (``KnowledgeGraphAgent`` and ``OntologyAgent``) both
unpacked ``EntityStore.search()``'s ``(results, total)`` tuple as a list
and raised ``AttributeError`` on their first entity, inside a factory
that swallows exceptions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cognisom.library.models import EntityType, Gene, Protein
from cognisom.library.store import EntityStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    """An empty store that does not auto-seed (kept fast and isolated)."""
    monkeypatch.setattr(EntityStore, "_auto_seed_if_empty", lambda self: None)
    return EntityStore(db_path=str(tmp_path / "entities.db"))


def gene(name: str, description: str = "") -> Gene:
    return Gene(
        entity_type=EntityType.GENE,
        name=name,
        display_name=name,
        description=description or f"{name} description",
    )


# ── The identity key ────────────────────────────────────────────────

def test_a_second_entity_with_the_same_name_and_type_is_refused(store):
    assert store.add_entity(gene("TP53")) is True
    assert store.add_entity(gene("TP53")) is False
    assert store.stats()["total_entities"] == 1


def test_identity_is_case_insensitive_on_name(store):
    assert store.add_entity(gene("TP53")) is True
    assert store.add_entity(gene("tp53")) is False


def test_the_same_name_under_a_different_type_is_allowed(store):
    """Identity is the pair, not the name alone."""
    assert store.add_entity(gene("AR")) is True
    protein = Protein(
        entity_type=EntityType.PROTEIN, name="AR", display_name="AR",
        description="Androgen receptor protein",
    )
    assert store.add_entity(protein) is True
    assert store.stats()["total_entities"] == 2


def test_duplicates_can_still_be_forced_when_a_caller_means_it(store):
    assert store.add_entity(gene("MYC")) is True
    assert store.add_entity(gene("MYC"), allow_duplicate=True) is True
    assert store.stats()["total_entities"] == 2


def test_entity_exists_reports_the_identity_key(store):
    assert store.entity_exists("BRCA2", "gene") is False
    store.add_entity(gene("BRCA2"))
    assert store.entity_exists("BRCA2", "gene") is True
    assert store.entity_exists("  brca2 ", "gene") is True
    assert store.entity_exists("BRCA2", "protein") is False


# ── Batch inserts ───────────────────────────────────────────────────

def test_batch_insert_skips_entities_already_in_the_database(store):
    store.add_entity(gene("KRAS"))
    inserted = store.add_entities_batch([gene("KRAS"), gene("NRAS")])

    assert inserted == 1
    assert store.stats()["total_entities"] == 2


def test_batch_insert_deduplicates_within_the_batch(store):
    """Neither copy is committed while the other is checked."""
    inserted = store.add_entities_batch([gene("EGFR"), gene("EGFR"), gene("ALK")])

    assert inserted == 2
    assert store.stats()["total_entities"] == 2


# ── Re-seeding is idempotent ────────────────────────────────────────

@pytest.mark.slow
def test_running_a_seeder_repeatedly_does_not_duplicate_the_catalogue(tmp_path):
    """The exact sequence that produced the 5x shipped database."""
    from cognisom.library.seed_immunology import seed_immunology_catalog

    store = EntityStore(db_path=str(tmp_path / "seeded.db"))
    after_autoseed = store.stats()["total_entities"]
    assert after_autoseed > 0

    for _ in range(3):
        seed_immunology_catalog(store)

    assert store.stats()["total_entities"] == after_autoseed

    entities, _ = store.search(limit=10_000)
    identities = [(e.name.lower(), e.entity_type.value) for e in entities]
    assert len(identities) == len(set(identities))


# ── The auditors run at all ─────────────────────────────────────────

def test_knowledge_graph_agent_scans_every_entity(store):
    """It reported `entities_scanned == 2` -- the length of a 2-tuple."""
    from cognisom.agents.knowledge_graph_agent import KnowledgeGraphAgent

    for name in ("TP53", "KRAS", "EGFR", "BRCA1", "PTEN"):
        store.add_entity(gene(name))

    report = KnowledgeGraphAgent(store).maintain()
    assert report.entities_scanned == 5


def test_ontology_agent_completes_a_full_audit(store):
    """It raised AttributeError on its first entity, silently."""
    from cognisom.agent.ontology_sync import OntologyAgent

    for name in ("TP53", "KRAS"):
        store.add_entity(gene(name))

    report = OntologyAgent(store).full_audit()
    assert report.entities_scanned == 2
    assert isinstance(report.issues, list)


def test_knowledge_graph_agent_detects_forced_duplicates(store):
    from cognisom.agents.knowledge_graph_agent import KnowledgeGraphAgent

    store.add_entity(gene("TP53"))
    store.add_entity(gene("TP53"), allow_duplicate=True)

    report = KnowledgeGraphAgent(store).maintain()
    assert report.duplicates_found >= 1
