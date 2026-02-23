"""
Entity Store — PostgreSQL (production) + SQLite (fallback)
===========================================================

Persistent storage for biological entities with full-text search,
faceted filtering, and relationship graph queries.

PostgreSQL (RDS) is used in production for HIPAA-compliant persistent storage.
SQLite is used as fallback for local development.

Environment variables:
    DATABASE_URL       — PostgreSQL connection string (if set, uses PostgreSQL)
    COGNISOM_DATA_DIR  — Local data directory for SQLite fallback

Tables:
    entities      — main entity table (JSONB for flexibility)
    relationships — typed edges between entities
    audit_log     — change history for every entity modification
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

from .models import (
    BioEntity,
    EntityStatus,
    EntityType,
    Relationship,
    RelationshipType,
)

log = logging.getLogger(__name__)


def get_database_url() -> Optional[str]:
    """Get PostgreSQL connection URL from environment or AWS Secrets Manager."""
    # Direct environment variable
    if url := os.environ.get("DATABASE_URL"):
        return url

    # Try AWS Secrets Manager
    secret_arn = os.environ.get("DB_SECRET_ARN")
    if secret_arn:
        try:
            import boto3
            client = boto3.client("secretsmanager", region_name="us-east-1")
            response = client.get_secret_value(SecretId=secret_arn)
            secret = json.loads(response["SecretString"])
            return secret.get("url")
        except Exception as e:
            log.warning("Failed to fetch DB credentials from Secrets Manager: %s", e)

    return None


class EntityStore:
    """Database-backed store for biological entities.

    Uses PostgreSQL in production (AWS RDS) or SQLite for local development.
    """

    def __init__(self, db_path: Optional[str] = None, database_url: Optional[str] = None):
        self._conn: Any = None
        self._use_postgres = False
        self._db_path: Optional[str] = None
        self._database_url: Optional[str] = None

        # Check for PostgreSQL connection
        pg_url = database_url or get_database_url()
        if pg_url:
            self._database_url = pg_url
            self._use_postgres = True
            log.info("Using PostgreSQL database")
        else:
            # Fall back to SQLite
            if db_path is None:
                data_dir = os.environ.get(
                    "COGNISOM_DATA_DIR",
                    str(Path(__file__).resolve().parent.parent.parent / "data"),
                )
                db_path = os.path.join(data_dir, "library", "entities.db")
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            self._db_path = db_path
            log.info("Using SQLite database at %s", db_path)

        self._init_db()

    @property
    def conn(self) -> Any:
        if self._conn is None:
            if self._use_postgres:
                import psycopg2
                import psycopg2.extras
                self._conn = psycopg2.connect(self._database_url)
                self._conn.autocommit = False
            else:
                import sqlite3
                self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
                self._conn.row_factory = sqlite3.Row
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def _execute(self, query: str, params: tuple = ()) -> Any:
        """Execute query with database-specific placeholder handling."""
        cursor = self.conn.cursor()
        if self._use_postgres:
            # Convert ? placeholders to %s for PostgreSQL
            query = query.replace("?", "%s")
            import psycopg2.extras
            cursor = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute(query, params)
        return cursor

    def _executescript(self, script: str):
        """Execute multi-statement script."""
        if self._use_postgres:
            self.conn.cursor().execute(script)
        else:
            self.conn.executescript(script)

    def _init_db(self):
        """Create tables if they don't exist."""
        if self._use_postgres:
            self._init_postgres()
        else:
            self._init_sqlite()

    def _init_postgres(self):
        """Initialize PostgreSQL schema."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                entity_id    TEXT PRIMARY KEY,
                entity_type  TEXT NOT NULL,
                name         TEXT NOT NULL,
                display_name TEXT NOT NULL DEFAULT '',
                description  TEXT NOT NULL DEFAULT '',
                status       TEXT NOT NULL DEFAULT 'active',
                source       TEXT NOT NULL DEFAULT '',
                data_json    JSONB NOT NULL DEFAULT '{}',
                created_at   DOUBLE PRECISION NOT NULL DEFAULT 0,
                updated_at   DOUBLE PRECISION NOT NULL DEFAULT 0,
                created_by   TEXT NOT NULL DEFAULT 'system',
                -- Full-text search vector
                search_vector TSVECTOR
            );

            CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
            CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
            CREATE INDEX IF NOT EXISTS idx_entities_status ON entities(status);
            CREATE INDEX IF NOT EXISTS idx_entities_search ON entities USING GIN(search_vector);

            CREATE TABLE IF NOT EXISTS relationships (
                rel_id      TEXT PRIMARY KEY,
                source_id   TEXT NOT NULL REFERENCES entities(entity_id),
                target_id   TEXT NOT NULL REFERENCES entities(entity_id),
                rel_type    TEXT NOT NULL,
                confidence  DOUBLE PRECISION NOT NULL DEFAULT 1.0,
                evidence    TEXT NOT NULL DEFAULT '',
                props_json  JSONB NOT NULL DEFAULT '{}',
                created_at  DOUBLE PRECISION NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_id);
            CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_id);
            CREATE INDEX IF NOT EXISTS idx_rel_type ON relationships(rel_type);

            CREATE TABLE IF NOT EXISTS audit_log (
                log_id     SERIAL PRIMARY KEY,
                entity_id  TEXT NOT NULL,
                action     TEXT NOT NULL,
                old_data   JSONB,
                new_data   JSONB,
                changed_by TEXT NOT NULL DEFAULT 'system',
                changed_at DOUBLE PRECISION NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_audit_entity ON audit_log(entity_id);

            -- Create function to update search vector
            CREATE OR REPLACE FUNCTION entities_search_trigger() RETURNS trigger AS $$
            BEGIN
                NEW.search_vector :=
                    setweight(to_tsvector('english', COALESCE(NEW.name, '')), 'A') ||
                    setweight(to_tsvector('english', COALESCE(NEW.display_name, '')), 'A') ||
                    setweight(to_tsvector('english', COALESCE(NEW.description, '')), 'B') ||
                    setweight(to_tsvector('english', COALESCE(NEW.data_json->>'synonyms', '')), 'B') ||
                    setweight(to_tsvector('english', COALESCE(NEW.data_json->>'tags', '')), 'C');
                RETURN NEW;
            END
            $$ LANGUAGE plpgsql;

            -- Create trigger if not exists
            DROP TRIGGER IF EXISTS entities_search_update ON entities;
            CREATE TRIGGER entities_search_update
                BEFORE INSERT OR UPDATE ON entities
                FOR EACH ROW EXECUTE FUNCTION entities_search_trigger();
        """)
        self.conn.commit()
        log.info("PostgreSQL entity store initialized")

    def _init_sqlite(self):
        """Initialize SQLite schema with FTS5."""
        import sqlite3
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
        log.info("SQLite entity store initialized at %s", self._db_path)

    # ── CRUD: Entities ───────────────────────────────────────────────

    def add_entity(self, entity: BioEntity) -> bool:
        """Insert a new entity. Returns True on success."""
        data = entity.to_dict()
        try:
            cursor = self._execute(
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

            # Update FTS index for SQLite
            if not self._use_postgres:
                self._execute(
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
        except Exception as e:
            log.warning("Entity %s insert failed: %s", entity.entity_id, e)
            self.conn.rollback()
            return False

    def get_entity(self, entity_id: str) -> Optional[BioEntity]:
        """Retrieve an entity by ID."""
        cursor = self._execute(
            "SELECT data_json FROM entities WHERE entity_id = ?", (entity_id,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        data_json = row["data_json"] if isinstance(row, dict) else row[0]
        if isinstance(data_json, str):
            data_json = json.loads(data_json)
        return BioEntity.from_dict(data_json)

    def update_entity(self, entity: BioEntity, changed_by: str = "system") -> bool:
        """Update an existing entity."""
        cursor = self._execute(
            "SELECT data_json FROM entities WHERE entity_id = ?",
            (entity.entity_id,),
        )
        old_row = cursor.fetchone()
        if old_row is None:
            return False

        old_data = old_row["data_json"] if isinstance(old_row, dict) else old_row[0]
        if isinstance(old_data, str):
            old_data = json.loads(old_data)

        entity.updated_at = time.time()
        data = entity.to_dict()

        self._execute(
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

        # Rebuild FTS for SQLite
        if not self._use_postgres:
            self._execute(
                "DELETE FROM entities_fts WHERE entity_id = ?", (entity.entity_id,)
            )
            self._execute(
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

        self._log_audit(entity.entity_id, "update", old_data, data, changed_by)
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
        self._execute("DELETE FROM relationships WHERE source_id=? OR target_id=?",
                      (entity_id, entity_id))
        if not self._use_postgres:
            self._execute("DELETE FROM entities_fts WHERE entity_id=?", (entity_id,))
        cursor = self._execute("DELETE FROM entities WHERE entity_id=?", (entity_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    # ── Batch Operations ──────────────────────────────────────────────

    def add_entities_batch(self, entities: List[BioEntity]) -> int:
        """Batch insert entities in a single transaction. Returns count inserted.

        Skips entities that already exist (ON CONFLICT DO NOTHING for Postgres,
        INSERT OR IGNORE for SQLite). Updates FTS index for SQLite.
        """
        if not entities:
            return 0

        inserted = 0
        try:
            cursor = self.conn.cursor()
            for entity in entities:
                data = entity.to_dict()
                try:
                    if self._use_postgres:
                        cursor.execute(
                            """INSERT INTO entities
                               (entity_id, entity_type, name, display_name, description,
                                status, source, data_json, created_at, updated_at, created_by)
                               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                               ON CONFLICT (entity_id) DO NOTHING""",
                            (
                                entity.entity_id, entity.entity_type.value,
                                entity.name, entity.display_name, entity.description,
                                entity.status.value, entity.source,
                                json.dumps(data), entity.created_at,
                                entity.updated_at, entity.created_by,
                            ),
                        )
                    else:
                        cursor.execute(
                            """INSERT OR IGNORE INTO entities
                               (entity_id, entity_type, name, display_name, description,
                                status, source, data_json, created_at, updated_at, created_by)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                entity.entity_id, entity.entity_type.value,
                                entity.name, entity.display_name, entity.description,
                                entity.status.value, entity.source,
                                json.dumps(data), entity.created_at,
                                entity.updated_at, entity.created_by,
                            ),
                        )

                    if cursor.rowcount > 0:
                        inserted += 1
                        # FTS for SQLite
                        if not self._use_postgres:
                            cursor.execute(
                                """INSERT OR IGNORE INTO entities_fts
                                   (entity_id, name, display_name, description, synonyms, tags)
                                   VALUES (?, ?, ?, ?, ?, ?)""",
                                (
                                    entity.entity_id, entity.name,
                                    entity.display_name, entity.description,
                                    " ".join(entity.synonyms), " ".join(entity.tags),
                                ),
                            )
                except Exception as e:
                    log.debug("Batch insert skip %s: %s", entity.entity_id, e)

            self.conn.commit()
            log.info("Batch inserted %d/%d entities", inserted, len(entities))
            return inserted

        except Exception as e:
            log.error("Batch insert failed: %s", e)
            self.conn.rollback()
            return 0

    def add_relationships_batch(self, rels: List[Relationship]) -> int:
        """Batch insert relationships in a single transaction. Returns count inserted."""
        if not rels:
            return 0

        inserted = 0
        try:
            cursor = self.conn.cursor()
            for rel in rels:
                try:
                    if self._use_postgres:
                        cursor.execute(
                            """INSERT INTO relationships
                               (rel_id, source_id, target_id, rel_type, confidence,
                                evidence, props_json, created_at)
                               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                               ON CONFLICT (rel_id) DO NOTHING""",
                            (
                                rel.rel_id, rel.source_id, rel.target_id,
                                rel.rel_type.value, rel.confidence,
                                rel.evidence, json.dumps(rel.properties),
                                rel.created_at,
                            ),
                        )
                    else:
                        cursor.execute(
                            """INSERT OR IGNORE INTO relationships
                               (rel_id, source_id, target_id, rel_type, confidence,
                                evidence, props_json, created_at)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                rel.rel_id, rel.source_id, rel.target_id,
                                rel.rel_type.value, rel.confidence,
                                rel.evidence, json.dumps(rel.properties),
                                rel.created_at,
                            ),
                        )

                    if cursor.rowcount > 0:
                        inserted += 1
                except Exception as e:
                    log.debug("Batch rel skip %s: %s", rel.rel_id, e)

            self.conn.commit()
            log.info("Batch inserted %d/%d relationships", inserted, len(rels))
            return inserted

        except Exception as e:
            log.error("Batch rel insert failed: %s", e)
            self.conn.rollback()
            return 0

    def upsert_entity(self, entity: BioEntity) -> bool:
        """Insert or update entity. Merges non-empty fields on conflict."""
        existing = self.get_entity(entity.entity_id)
        if existing is None:
            return self.add_entity(entity)
        # Merge: only overwrite fields that are non-empty in the new entity
        return self.update_entity(entity, changed_by="enrichment")

    def find_entity_by_external_id(self, key: str, value: str) -> Optional[BioEntity]:
        """Find an entity by external_ids key/value (e.g., 'ncbi_gene'/'7157')."""
        if self._use_postgres:
            cursor = self.conn.cursor()
            import psycopg2.extras
            cursor = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute(
                "SELECT data_json FROM entities WHERE data_json->'external_ids'->>%s = %s LIMIT 1",
                (key, value),
            )
        else:
            cursor = self._execute(
                "SELECT data_json FROM entities WHERE json_extract(data_json, ?) = ? LIMIT 1",
                (f"$.external_ids.{key}", value),
            )
        row = cursor.fetchone()
        if row is None:
            return None
        data_json = row["data_json"] if isinstance(row, dict) else row[0]
        if isinstance(data_json, str):
            data_json = json.loads(data_json)
        return BioEntity.from_dict(data_json)

    def find_entity_by_name(self, name: str, entity_type: Optional[str] = None) -> Optional[BioEntity]:
        """Find an entity by exact name match."""
        if entity_type:
            cursor = self._execute(
                "SELECT data_json FROM entities WHERE name = ? AND entity_type = ? LIMIT 1",
                (name, entity_type),
            )
        else:
            cursor = self._execute(
                "SELECT data_json FROM entities WHERE name = ? LIMIT 1",
                (name,),
            )
        row = cursor.fetchone()
        if row is None:
            return None
        data_json = row["data_json"] if isinstance(row, dict) else row[0]
        if isinstance(data_json, str):
            data_json = json.loads(data_json)
        return BioEntity.from_dict(data_json)

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
        """Full-text search using FTS5 (SQLite) or tsvector (PostgreSQL)."""
        if self._use_postgres:
            return self._postgres_fts_search(query, entity_type, status, limit, offset)
        return self._sqlite_fts_search(query, entity_type, status, limit, offset)

    def _postgres_fts_search(
        self, query: str, entity_type: Optional[str], status: str,
        limit: int, offset: int,
    ) -> Tuple[List[BioEntity], int]:
        """PostgreSQL full-text search using tsvector."""
        # Build search query
        search_terms = " & ".join(f"{w}:*" for w in query.split() if w)

        base = """
            FROM entities e
            WHERE search_vector @@ to_tsquery('english', %s)
        """
        params: list = [search_terms]

        if entity_type:
            base += " AND e.entity_type = %s"
            params.append(entity_type)
        if status:
            base += " AND e.status = %s"
            params.append(status)

        # Count
        import psycopg2.extras
        cursor = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute(f"SELECT COUNT(*) as cnt {base}", params)
        count_row = cursor.fetchone()
        total = count_row["cnt"] if count_row else 0

        # Fetch with rank
        cursor.execute(
            f"""SELECT e.data_json, ts_rank(search_vector, to_tsquery('english', %s)) as rank
                {base} ORDER BY rank DESC LIMIT %s OFFSET %s""",
            [search_terms] + params + [limit, offset],
        )
        rows = cursor.fetchall()

        entities = []
        for r in rows:
            data = r["data_json"]
            if isinstance(data, str):
                data = json.loads(data)
            entities.append(BioEntity.from_dict(data))
        return entities, total

    def _sqlite_fts_search(
        self, query: str, entity_type: Optional[str], status: str,
        limit: int, offset: int,
    ) -> Tuple[List[BioEntity], int]:
        """SQLite FTS5 full-text search."""
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
        cursor = self._execute(f"SELECT COUNT(*) as cnt {base}", tuple(params))
        count_row = cursor.fetchone()
        total = count_row["cnt"] if isinstance(count_row, dict) else count_row[0]

        # Fetch
        cursor = self._execute(
            f"SELECT e.data_json {base} ORDER BY rank LIMIT ? OFFSET ?",
            tuple(params + [limit, offset]),
        )
        rows = cursor.fetchall()

        entities = []
        for r in rows:
            data = r["data_json"] if isinstance(r, dict) else r[0]
            if isinstance(data, str):
                data = json.loads(data)
            entities.append(BioEntity.from_dict(data))
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

        cursor = self._execute(f"SELECT COUNT(*) as cnt {base}", tuple(params))
        count_row = cursor.fetchone()
        total = count_row["cnt"] if isinstance(count_row, dict) else count_row[0]

        cursor = self._execute(
            f"SELECT data_json {base} ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            tuple(params + [limit, offset]),
        )
        rows = cursor.fetchall()

        entities = []
        for r in rows:
            data = r["data_json"] if isinstance(r, dict) else r[0]
            if isinstance(data, str):
                data = json.loads(data)
            entities.append(BioEntity.from_dict(data))

        # Post-filter by tags if specified
        if tags:
            tag_set = set(tags)
            entities = [e for e in entities if tag_set.intersection(e.tags)]
            total = len(entities)

        return entities, total

    def count_by_type(self) -> Dict[str, int]:
        """Return entity counts grouped by type."""
        cursor = self._execute(
            "SELECT entity_type, COUNT(*) as cnt FROM entities WHERE status='active' GROUP BY entity_type"
        )
        rows = cursor.fetchall()
        result = {}
        for r in rows:
            if isinstance(r, dict):
                result[r["entity_type"]] = r["cnt"]
            else:
                result[r[0]] = r[1]
        return result

    def get_all_tags(self) -> List[str]:
        """Return all unique tags across entities."""
        cursor = self._execute(
            "SELECT data_json FROM entities WHERE status='active'"
        )
        rows = cursor.fetchall()
        tags = set()
        for r in rows:
            data = r["data_json"] if isinstance(r, dict) else r[0]
            if isinstance(data, str):
                data = json.loads(data)
            for t in data.get("tags", []):
                tags.add(t)
        return sorted(tags)

    # ── Relationships ────────────────────────────────────────────────

    def add_relationship(self, rel: Relationship) -> bool:
        """Insert a relationship between two entities."""
        try:
            self._execute(
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
        except Exception as e:
            log.warning("Relationship insert failed: %s", e)
            self.conn.rollback()
            return False

    def get_relationships(
        self, entity_id: str, direction: str = "both"
    ) -> List[Relationship]:
        """Get relationships for an entity.

        direction: 'outgoing', 'incoming', or 'both'
        """
        results = []
        if direction in ("outgoing", "both"):
            cursor = self._execute(
                "SELECT * FROM relationships WHERE source_id = ?", (entity_id,)
            )
            results.extend(self._rows_to_rels(cursor.fetchall()))

        if direction in ("incoming", "both"):
            cursor = self._execute(
                "SELECT * FROM relationships WHERE target_id = ?", (entity_id,)
            )
            results.extend(self._rows_to_rels(cursor.fetchall()))

        return results

    def get_relationships_by_type(
        self, rel_type: str, limit: int = 100
    ) -> List[Relationship]:
        """Get all relationships of a specific type."""
        cursor = self._execute(
            "SELECT * FROM relationships WHERE rel_type = ? LIMIT ?",
            (rel_type, limit),
        )
        return self._rows_to_rels(cursor.fetchall())

    def delete_relationship(self, rel_id: str) -> bool:
        cursor = self._execute(
            "DELETE FROM relationships WHERE rel_id = ?", (rel_id,)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def _rows_to_rels(self, rows) -> List[Relationship]:
        results = []
        for r in rows:
            if isinstance(r, dict):
                props = r["props_json"]
                if isinstance(props, str):
                    props = json.loads(props)
                results.append(Relationship(
                    rel_id=r["rel_id"],
                    source_id=r["source_id"],
                    target_id=r["target_id"],
                    rel_type=RelationshipType(r["rel_type"]),
                    confidence=r["confidence"],
                    evidence=r["evidence"],
                    properties=props,
                    created_at=r["created_at"],
                ))
            else:
                # sqlite3.Row
                props = r["props_json"]
                if isinstance(props, str):
                    props = json.loads(props)
                results.append(Relationship(
                    rel_id=r["rel_id"],
                    source_id=r["source_id"],
                    target_id=r["target_id"],
                    rel_type=RelationshipType(r["rel_type"]),
                    confidence=r["confidence"],
                    evidence=r["evidence"],
                    properties=props,
                    created_at=r["created_at"],
                ))
        return results

    # ── Audit log ────────────────────────────────────────────────────

    def _log_audit(
        self, entity_id: str, action: str,
        old_data: Optional[dict], new_data: Optional[dict],
        changed_by: str = "system",
    ):
        self._execute(
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
        cursor = self._execute(
            """SELECT action, changed_by, changed_at, old_data, new_data
               FROM audit_log WHERE entity_id = ?
               ORDER BY changed_at DESC LIMIT ?""",
            (entity_id, limit),
        )
        rows = cursor.fetchall()
        results = []
        for r in rows:
            if isinstance(r, dict):
                old_data = r["old_data"]
                new_data = r["new_data"]
                if isinstance(old_data, str):
                    old_data = json.loads(old_data) if old_data else None
                if isinstance(new_data, str):
                    new_data = json.loads(new_data) if new_data else None
                results.append({
                    "action": r["action"],
                    "changed_by": r["changed_by"],
                    "changed_at": r["changed_at"],
                    "old_data": old_data,
                    "new_data": new_data,
                })
            else:
                results.append({
                    "action": r["action"],
                    "changed_by": r["changed_by"],
                    "changed_at": r["changed_at"],
                    "old_data": json.loads(r["old_data"]) if r["old_data"] else None,
                    "new_data": json.loads(r["new_data"]) if r["new_data"] else None,
                })
        return results

    # ── Stats ────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Get library statistics."""
        entity_counts = self.count_by_type()
        cursor = self._execute("SELECT COUNT(*) as cnt FROM relationships")
        row = cursor.fetchone()
        rel_count = row["cnt"] if isinstance(row, dict) else row[0]
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
