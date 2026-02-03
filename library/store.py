"""
Entity Store — SQLite + FTS5
=============================

Persistent storage for biological entities with full-text search,
faceted filtering, and relationship graph queries.

Tables:
    entities      — main entity table (JSON blob for flexibility)
    entities_fts  — FTS5 virtual table for text search
    relationships — typed edges between entities
    audit_log     — change history for every entity modification
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .models import (
    BioEntity,
    EntityStatus,
    EntityType,
    Relationship,
    RelationshipType,
)

log = logging.getLogger(__name__)


class EntityStore:
    """SQLite-backed store for biological entities."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            data_dir = os.environ.get(
                "COGNISOM_DATA_DIR",
                str(Path(__file__).resolve().parent.parent.parent / "data"),
            )
            db_path = os.path.join(data_dir, "library", "entities.db")

        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            # check_same_thread=False required for Streamlit multi-threaded environment
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def _init_db(self):
        """Create tables if they don't exist."""
        c = self.conn
        c.executescript("""
            CREATE TABLE IF NOT EXISTS entities (
                entity_id    TEXT PRIMARY KEY,
                entity_type  TEXT NOT NULL,
                name         TEXT NOT NULL,
                display_name TEXT NOT NULL DEFAULT '',
                description  TEXT NOT NULL DEFAULT '',
                status       TEXT NOT NULL DEFAULT 'active',
                source       TEXT NOT NULL DEFAULT '',
                data_json    TEXT NOT NULL DEFAULT '{}',
                created_at   REAL NOT NULL DEFAULT 0,
                updated_at   REAL NOT NULL DEFAULT 0,
                created_by   TEXT NOT NULL DEFAULT 'system'
            );

            CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
            CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
            CREATE INDEX IF NOT EXISTS idx_entities_status ON entities(status);

            CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(
                entity_id,
                name,
                display_name,
                description,
                synonyms,
                tags
            );

            CREATE TABLE IF NOT EXISTS relationships (
                rel_id      TEXT PRIMARY KEY,
                source_id   TEXT NOT NULL,
                target_id   TEXT NOT NULL,
                rel_type    TEXT NOT NULL,
                confidence  REAL NOT NULL DEFAULT 1.0,
                evidence    TEXT NOT NULL DEFAULT '',
                props_json  TEXT NOT NULL DEFAULT '{}',
                created_at  REAL NOT NULL DEFAULT 0,
                FOREIGN KEY (source_id) REFERENCES entities(entity_id),
                FOREIGN KEY (target_id) REFERENCES entities(entity_id)
            );

            CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_id);
            CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_id);
            CREATE INDEX IF NOT EXISTS idx_rel_type ON relationships(rel_type);

            CREATE TABLE IF NOT EXISTS audit_log (
                log_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id  TEXT NOT NULL,
                action     TEXT NOT NULL,
                old_data   TEXT,
                new_data   TEXT,
                changed_by TEXT NOT NULL DEFAULT 'system',
                changed_at REAL NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_audit_entity ON audit_log(entity_id);
        """)
        c.commit()
        log.info("Entity store initialized at %s", self.db_path)

    # ── CRUD: Entities ───────────────────────────────────────────────

    def add_entity(self, entity: BioEntity) -> bool:
        """Insert a new entity. Returns True on success."""
        data = entity.to_dict()
        try:
            self.conn.execute(
                """INSERT INTO entities
                   (entity_id, entity_type, name, display_name, description,
                    status, source, data_json, created_at, updated_at, created_by)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entity.entity_id,
                    entity.entity_type.value,
                    entity.name,
                    entity.display_name,
                    entity.description,
                    entity.status.value,
                    entity.source,
                    json.dumps(data),
                    entity.created_at,
                    entity.updated_at,
                    entity.created_by,
                ),
            )
            # Update FTS index
            self.conn.execute(
                """INSERT INTO entities_fts
                   (entity_id, name, display_name, description, synonyms, tags)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    entity.entity_id,
                    entity.name,
                    entity.display_name,
                    entity.description,
                    " ".join(entity.synonyms),
                    " ".join(entity.tags),
                ),
            )
            self._log_audit(entity.entity_id, "create", None, data)
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            log.warning("Entity %s already exists", entity.entity_id)
            return False

    def get_entity(self, entity_id: str) -> Optional[BioEntity]:
        """Retrieve an entity by ID."""
        row = self.conn.execute(
            "SELECT data_json FROM entities WHERE entity_id = ?", (entity_id,)
        ).fetchone()
        if row is None:
            return None
        return BioEntity.from_dict(json.loads(row["data_json"]))

    def update_entity(self, entity: BioEntity, changed_by: str = "system") -> bool:
        """Update an existing entity."""
        old_row = self.conn.execute(
            "SELECT data_json FROM entities WHERE entity_id = ?",
            (entity.entity_id,),
        ).fetchone()
        if old_row is None:
            return False

        entity.updated_at = time.time()
        data = entity.to_dict()

        self.conn.execute(
            """UPDATE entities
               SET name=?, display_name=?, description=?, status=?,
                   source=?, data_json=?, updated_at=?, entity_type=?
               WHERE entity_id=?""",
            (
                entity.name,
                entity.display_name,
                entity.description,
                entity.status.value,
                entity.source,
                json.dumps(data),
                entity.updated_at,
                entity.entity_type.value,
                entity.entity_id,
            ),
        )
        # Rebuild FTS for this entity
        self.conn.execute(
            "DELETE FROM entities_fts WHERE entity_id = ?", (entity.entity_id,)
        )
        self.conn.execute(
            """INSERT INTO entities_fts
               (entity_id, name, display_name, description, synonyms, tags)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                entity.entity_id,
                entity.name,
                entity.display_name,
                entity.description,
                " ".join(entity.synonyms),
                " ".join(entity.tags),
            ),
        )
        self._log_audit(
            entity.entity_id,
            "update",
            json.loads(old_row["data_json"]),
            data,
            changed_by,
        )
        self.conn.commit()
        return True

    def delete_entity(self, entity_id: str, changed_by: str = "system") -> bool:
        """Soft-delete: set status to deprecated."""
        entity = self.get_entity(entity_id)
        if entity is None:
            return False
        entity.status = EntityStatus.DEPRECATED
        return self.update_entity(entity, changed_by)

    def hard_delete_entity(self, entity_id: str) -> bool:
        """Permanently remove entity and its relationships."""
        self.conn.execute("DELETE FROM relationships WHERE source_id=? OR target_id=?",
                          (entity_id, entity_id))
        self.conn.execute("DELETE FROM entities_fts WHERE entity_id=?", (entity_id,))
        result = self.conn.execute("DELETE FROM entities WHERE entity_id=?", (entity_id,))
        self.conn.commit()
        return result.rowcount > 0

    # ── Search & Query ───────────────────────────────────────────────

    def search(
        self,
        query: str = "",
        entity_type: Optional[str] = None,
        status: str = "active",
        tags: Optional[List[str]] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[BioEntity], int]:
        """Search entities with optional filters.

        Returns (results, total_count).
        """
        if query:
            return self._fts_search(query, entity_type, status, limit, offset)
        return self._filtered_list(entity_type, status, tags, limit, offset)

    def _fts_search(
        self, query: str, entity_type: Optional[str], status: str,
        limit: int, offset: int,
    ) -> Tuple[List[BioEntity], int]:
        """Full-text search using FTS5."""
        # Build FTS query (simple prefix matching)
        fts_query = " OR ".join(f'"{w}"*' for w in query.split() if w)

        base = """
            FROM entities e
            JOIN entities_fts f ON e.entity_id = f.entity_id
            WHERE entities_fts MATCH ?
        """
        params: list = [fts_query]

        if entity_type:
            base += " AND e.entity_type = ?"
            params.append(entity_type)
        if status:
            base += " AND e.status = ?"
            params.append(status)

        # Count
        count_row = self.conn.execute(
            f"SELECT COUNT(*) as cnt {base}", params
        ).fetchone()
        total = count_row["cnt"] if count_row else 0

        # Fetch
        rows = self.conn.execute(
            f"SELECT e.data_json {base} ORDER BY rank LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()

        entities = [BioEntity.from_dict(json.loads(r["data_json"])) for r in rows]
        return entities, total

    def _filtered_list(
        self, entity_type: Optional[str], status: str,
        tags: Optional[List[str]], limit: int, offset: int,
    ) -> Tuple[List[BioEntity], int]:
        """List entities with optional type/status/tag filters."""
        base = "FROM entities WHERE 1=1"
        params: list = []

        if entity_type:
            base += " AND entity_type = ?"
            params.append(entity_type)
        if status:
            base += " AND status = ?"
            params.append(status)

        count_row = self.conn.execute(
            f"SELECT COUNT(*) as cnt {base}", params
        ).fetchone()
        total = count_row["cnt"] if count_row else 0

        rows = self.conn.execute(
            f"SELECT data_json {base} ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()

        entities = [BioEntity.from_dict(json.loads(r["data_json"])) for r in rows]

        # Post-filter by tags if specified
        if tags:
            tag_set = set(tags)
            entities = [e for e in entities if tag_set.intersection(e.tags)]
            total = len(entities)

        return entities, total

    def count_by_type(self) -> Dict[str, int]:
        """Return entity counts grouped by type."""
        rows = self.conn.execute(
            "SELECT entity_type, COUNT(*) as cnt FROM entities WHERE status='active' GROUP BY entity_type"
        ).fetchall()
        return {r["entity_type"]: r["cnt"] for r in rows}

    def get_all_tags(self) -> List[str]:
        """Return all unique tags across entities."""
        rows = self.conn.execute(
            "SELECT data_json FROM entities WHERE status='active'"
        ).fetchall()
        tags = set()
        for r in rows:
            data = json.loads(r["data_json"])
            for t in data.get("tags", []):
                tags.add(t)
        return sorted(tags)

    # ── Relationships ────────────────────────────────────────────────

    def add_relationship(self, rel: Relationship) -> bool:
        """Insert a relationship between two entities."""
        try:
            self.conn.execute(
                """INSERT INTO relationships
                   (rel_id, source_id, target_id, rel_type, confidence,
                    evidence, props_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    rel.rel_id,
                    rel.source_id,
                    rel.target_id,
                    rel.rel_type.value,
                    rel.confidence,
                    rel.evidence,
                    json.dumps(rel.properties),
                    rel.created_at,
                ),
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError as e:
            log.warning("Relationship insert failed: %s", e)
            return False

    def get_relationships(
        self, entity_id: str, direction: str = "both"
    ) -> List[Relationship]:
        """Get relationships for an entity.

        direction: 'outgoing', 'incoming', or 'both'
        """
        results = []
        if direction in ("outgoing", "both"):
            rows = self.conn.execute(
                "SELECT * FROM relationships WHERE source_id = ?", (entity_id,)
            ).fetchall()
            results.extend(self._rows_to_rels(rows))

        if direction in ("incoming", "both"):
            rows = self.conn.execute(
                "SELECT * FROM relationships WHERE target_id = ?", (entity_id,)
            ).fetchall()
            results.extend(self._rows_to_rels(rows))

        return results

    def get_relationships_by_type(
        self, rel_type: str, limit: int = 100
    ) -> List[Relationship]:
        """Get all relationships of a specific type."""
        rows = self.conn.execute(
            "SELECT * FROM relationships WHERE rel_type = ? LIMIT ?",
            (rel_type, limit),
        ).fetchall()
        return self._rows_to_rels(rows)

    def delete_relationship(self, rel_id: str) -> bool:
        result = self.conn.execute(
            "DELETE FROM relationships WHERE rel_id = ?", (rel_id,)
        )
        self.conn.commit()
        return result.rowcount > 0

    def _rows_to_rels(self, rows) -> List[Relationship]:
        return [
            Relationship(
                rel_id=r["rel_id"],
                source_id=r["source_id"],
                target_id=r["target_id"],
                rel_type=RelationshipType(r["rel_type"]),
                confidence=r["confidence"],
                evidence=r["evidence"],
                properties=json.loads(r["props_json"]),
                created_at=r["created_at"],
            )
            for r in rows
        ]

    # ── Audit log ────────────────────────────────────────────────────

    def _log_audit(
        self, entity_id: str, action: str,
        old_data: Optional[dict], new_data: Optional[dict],
        changed_by: str = "system",
    ):
        self.conn.execute(
            """INSERT INTO audit_log
               (entity_id, action, old_data, new_data, changed_by, changed_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                entity_id,
                action,
                json.dumps(old_data) if old_data else None,
                json.dumps(new_data) if new_data else None,
                changed_by,
                time.time(),
            ),
        )

    def get_audit_log(self, entity_id: str, limit: int = 20) -> List[dict]:
        """Get change history for an entity."""
        rows = self.conn.execute(
            """SELECT action, changed_by, changed_at, old_data, new_data
               FROM audit_log WHERE entity_id = ?
               ORDER BY changed_at DESC LIMIT ?""",
            (entity_id, limit),
        ).fetchall()
        return [
            {
                "action": r["action"],
                "changed_by": r["changed_by"],
                "changed_at": r["changed_at"],
                "old_data": json.loads(r["old_data"]) if r["old_data"] else None,
                "new_data": json.loads(r["new_data"]) if r["new_data"] else None,
            }
            for r in rows
        ]

    # ── Stats ────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Get library statistics."""
        entity_counts = self.count_by_type()
        rel_count = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM relationships"
        ).fetchone()["cnt"]
        total = sum(entity_counts.values())
        return {
            "total_entities": total,
            "total_relationships": rel_count,
            "by_type": entity_counts,
        }

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
