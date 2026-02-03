"""
Knowledge Graph Agent (Phase 7)
================================

Autonomous agent that maintains the biological knowledge graph:
- Entity relationships (gene->protein, drug->target, pathway->components)
- Consistency checking (no contradictory relationships)
- Redundancy detection (duplicate entities)
- Completeness analysis (missing expected relationships)
- Auto-linking (infer relationships from external databases)

Usage::

    from cognisom.agents import KnowledgeGraphAgent

    agent = KnowledgeGraphAgent(store)
    report = agent.maintain()
    print(report.summary())
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)


@dataclass
class GraphIssue:
    """An issue found in the knowledge graph."""
    severity: str = "warning"   # "error", "warning", "info"
    category: str = ""          # "duplicate", "orphan", "contradiction", "missing"
    entity_ids: List[str] = field(default_factory=list)
    message: str = ""
    auto_fixable: bool = False
    suggested_fix: str = ""


@dataclass
class GraphMaintenanceReport:
    """Results of knowledge graph maintenance."""
    timestamp: float = 0.0
    entities_scanned: int = 0
    relationships_scanned: int = 0
    duplicates_found: int = 0
    duplicates_merged: int = 0
    orphans_found: int = 0
    orphans_linked: int = 0
    contradictions_found: int = 0
    missing_links_found: int = 0
    missing_links_added: int = 0
    issues: List[GraphIssue] = field(default_factory=list)
    elapsed_sec: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()

    @property
    def health_score(self) -> float:
        """Graph health score (0-1, higher is better)."""
        total_issues = (self.duplicates_found + self.orphans_found +
                       self.contradictions_found * 2)  # Contradictions weighted
        if self.entities_scanned == 0:
            return 1.0
        penalty = min(total_issues / self.entities_scanned, 0.5)
        return 1.0 - penalty

    def summary(self) -> str:
        health = "HEALTHY" if self.health_score >= 0.9 else "NEEDS ATTENTION"
        lines = [
            f"Knowledge Graph {health} (score: {self.health_score:.2f})",
            f"  Scanned: {self.entities_scanned} entities, "
            f"{self.relationships_scanned} relationships",
            f"  Duplicates: {self.duplicates_found} found, "
            f"{self.duplicates_merged} merged",
            f"  Orphans: {self.orphans_found} found, "
            f"{self.orphans_linked} linked",
            f"  Contradictions: {self.contradictions_found}",
            f"  Missing links: {self.missing_links_found} found, "
            f"{self.missing_links_added} added",
            f"  Elapsed: {self.elapsed_sec:.1f}s",
        ]
        errors = [i for i in self.issues if i.severity == "error"]
        if errors:
            lines.append("  Errors:")
            for e in errors[:5]:
                lines.append(f"    - {e.category}: {e.message}")
        return "\n".join(lines)


class KnowledgeGraphAgent:
    """Autonomous agent for maintaining the biological knowledge graph.

    Ensures consistency, completeness, and quality of entity relationships.
    """

    # Expected relationship patterns (source_type -> rel_type -> target_type)
    EXPECTED_RELATIONSHIPS = {
        ("gene", "encodes", "protein"): 0.9,      # 90% of genes should encode proteins
        ("protein", "encoded_by", "gene"): 0.9,
        ("mutation", "affects", "gene"): 1.0,     # All mutations should affect a gene
        ("drug", "targets", "protein"): 0.8,      # Most drugs should have targets
        ("pathway", "contains", "gene"): 0.5,     # Pathways contain genes
        ("protein", "part_of", "pathway"): 0.3,
    }

    # Contradictory relationship pairs
    CONTRADICTIONS = [
        (("activates",), ("inhibits",)),          # Can't both activate and inhibit
        (("upregulates",), ("downregulates",)),
    ]

    def __init__(self, store=None) -> None:
        """Initialize the knowledge graph agent.

        Args:
            store: cognisom.library.store.EntityStore instance
        """
        self._store = store

    def maintain(self, auto_fix: bool = False) -> GraphMaintenanceReport:
        """Run full knowledge graph maintenance.

        Args:
            auto_fix: If True, automatically fix simple issues
        """
        t0 = time.time()
        report = GraphMaintenanceReport()

        if self._store is None:
            report.issues.append(GraphIssue(
                severity="error",
                category="config",
                message="No entity store configured",
            ))
            return report

        # Load all entities
        entities = self._store.search(limit=10000)
        report.entities_scanned = len(entities)

        # Build index by name for duplicate detection
        by_name: Dict[str, List] = {}
        by_type: Dict[str, List] = {}
        for entity in entities:
            name_key = entity.name.lower().strip()
            if name_key not in by_name:
                by_name[name_key] = []
            by_name[name_key].append(entity)

            type_key = entity.entity_type.value
            if type_key not in by_type:
                by_type[type_key] = []
            by_type[type_key].append(entity)

        # Check for duplicates
        dup_issues = self._find_duplicates(by_name)
        report.duplicates_found = len(dup_issues)
        report.issues.extend(dup_issues)
        if auto_fix:
            report.duplicates_merged = self._merge_duplicates(dup_issues)

        # Check for orphans
        orphan_issues = self._find_orphans(entities)
        report.orphans_found = len(orphan_issues)
        report.issues.extend(orphan_issues)
        if auto_fix:
            report.orphans_linked = self._link_orphans(orphan_issues)

        # Check for contradictions
        contra_issues = self._find_contradictions(entities)
        report.contradictions_found = len(contra_issues)
        report.issues.extend(contra_issues)

        # Check for missing expected relationships
        missing_issues = self._find_missing_links(entities, by_type)
        report.missing_links_found = len(missing_issues)
        report.issues.extend(missing_issues)
        if auto_fix:
            report.missing_links_added = self._add_missing_links(missing_issues)

        # Count relationships
        report.relationships_scanned = self._count_relationships()

        report.elapsed_sec = time.time() - t0
        return report

    def check_entity(self, entity_id: str) -> List[GraphIssue]:
        """Check a single entity for graph issues."""
        issues = []

        entity = self._store.get_entity(entity_id)
        if entity is None:
            issues.append(GraphIssue(
                severity="error",
                category="not_found",
                entity_ids=[entity_id],
                message=f"Entity {entity_id} not found",
            ))
            return issues

        # Check if orphaned
        rels = self._store.get_relationships(entity_id)
        if not rels:
            issues.append(GraphIssue(
                severity="warning",
                category="orphan",
                entity_ids=[entity_id],
                message=f"Entity '{entity.name}' has no relationships",
                auto_fixable=True,
                suggested_fix="Auto-link based on entity type and external IDs",
            ))

        # Check for expected relationships based on type
        type_issues = self._check_type_requirements(entity)
        issues.extend(type_issues)

        return issues

    # ── Duplicate Detection ──────────────────────────────────────────

    def _find_duplicates(self, by_name: Dict[str, List]) -> List[GraphIssue]:
        """Find potential duplicate entities."""
        issues = []

        for name, entities in by_name.items():
            if len(entities) > 1:
                # Check if same type
                types = set(e.entity_type for e in entities)
                if len(types) == 1:
                    issues.append(GraphIssue(
                        severity="warning",
                        category="duplicate",
                        entity_ids=[e.entity_id for e in entities],
                        message=f"Potential duplicates: '{name}' ({len(entities)} entities)",
                        auto_fixable=True,
                        suggested_fix="Merge entities, keeping most complete record",
                    ))

        return issues

    def _merge_duplicates(self, issues: List[GraphIssue]) -> int:
        """Merge duplicate entities. Returns count of merges."""
        merged = 0
        for issue in issues:
            if issue.category != "duplicate" or not issue.auto_fixable:
                continue

            entity_ids = issue.entity_ids
            if len(entity_ids) < 2:
                continue

            try:
                # Keep the first (usually oldest), merge others into it
                primary = self._store.get_entity(entity_ids[0])
                if primary is None:
                    continue

                for eid in entity_ids[1:]:
                    secondary = self._store.get_entity(eid)
                    if secondary is None:
                        continue

                    # Merge external IDs
                    for source, ids in secondary.external_ids.items():
                        if source not in primary.external_ids:
                            primary.external_ids[source] = ids

                    # Merge ontology IDs
                    primary.ontology_ids = list(set(
                        primary.ontology_ids + secondary.ontology_ids
                    ))

                    # Move relationships to primary
                    rels = self._store.get_relationships(eid)
                    for rel in rels:
                        if rel.source_id == eid:
                            rel.source_id = entity_ids[0]
                        if rel.target_id == eid:
                            rel.target_id = entity_ids[0]

                    # Delete secondary
                    self._store.delete_entity(eid)
                    merged += 1

                # Save primary with merged data
                self._store.update_entity(primary)

            except Exception as e:
                log.warning("Failed to merge duplicates: %s", e)

        return merged

    # ── Orphan Detection ─────────────────────────────────────────────

    def _find_orphans(self, entities: List) -> List[GraphIssue]:
        """Find entities with no relationships."""
        issues = []

        for entity in entities:
            # Some types are expected to be standalone
            if entity.entity_type.value in ("parameter_set", "simulation_scenario", "organ"):
                continue

            rels = self._store.get_relationships(entity.entity_id)
            if not rels:
                issues.append(GraphIssue(
                    severity="info",
                    category="orphan",
                    entity_ids=[entity.entity_id],
                    message=f"Orphaned entity: '{entity.name}' ({entity.entity_type.value})",
                    auto_fixable=True,
                    suggested_fix=f"Link based on external IDs: {list(entity.external_ids.keys())}",
                ))

        return issues

    def _link_orphans(self, issues: List[GraphIssue]) -> int:
        """Attempt to auto-link orphaned entities. Returns count of links added."""
        linked = 0

        for issue in issues:
            if issue.category != "orphan" or not issue.auto_fixable:
                continue

            entity_id = issue.entity_ids[0]
            entity = self._store.get_entity(entity_id)
            if entity is None:
                continue

            # Try to find related entities based on type
            try:
                if entity.entity_type.value == "gene":
                    # Link to proteins with matching gene source
                    proteins = self._store.search(
                        entity_type="protein",
                        limit=100
                    )
                    for prot in proteins:
                        if hasattr(prot, "gene_source") and prot.gene_source == entity.name:
                            self._store.add_relationship(
                                source_id=entity.entity_id,
                                target_id=prot.entity_id,
                                rel_type="encodes",
                            )
                            linked += 1
                            break

                elif entity.entity_type.value == "protein":
                    # Link to gene with matching name
                    gene_source = getattr(entity, "gene_source", "")
                    if gene_source:
                        genes = self._store.search(
                            query=gene_source,
                            entity_type="gene",
                            limit=5
                        )
                        for gene in genes:
                            if gene.name.lower() == gene_source.lower():
                                self._store.add_relationship(
                                    source_id=gene.entity_id,
                                    target_id=entity.entity_id,
                                    rel_type="encodes",
                                )
                                linked += 1
                                break

            except Exception as e:
                log.warning("Failed to link orphan %s: %s", entity_id, e)

        return linked

    # ── Contradiction Detection ──────────────────────────────────────

    def _find_contradictions(self, entities: List) -> List[GraphIssue]:
        """Find contradictory relationships."""
        issues = []

        for entity in entities:
            rels = self._store.get_relationships(entity.entity_id)

            # Group by target
            by_target: Dict[str, List] = {}
            for rel in rels:
                if rel.source_id == entity.entity_id:
                    if rel.target_id not in by_target:
                        by_target[rel.target_id] = []
                    by_target[rel.target_id].append(rel.rel_type.value)

            # Check for contradictions
            for target_id, rel_types in by_target.items():
                for contra_pair in self.CONTRADICTIONS:
                    types_a, types_b = contra_pair
                    has_a = any(t in rel_types for t in types_a)
                    has_b = any(t in rel_types for t in types_b)
                    if has_a and has_b:
                        issues.append(GraphIssue(
                            severity="error",
                            category="contradiction",
                            entity_ids=[entity.entity_id, target_id],
                            message=f"Contradictory relationships: "
                                   f"{entity.name} both {types_a[0]} and {types_b[0]} target",
                            auto_fixable=False,
                        ))

        return issues

    # ── Missing Link Detection ───────────────────────────────────────

    def _find_missing_links(self, entities: List, by_type: Dict[str, List]) -> List[GraphIssue]:
        """Find expected relationships that are missing."""
        issues = []

        for (src_type, rel_type, tgt_type), min_coverage in self.EXPECTED_RELATIONSHIPS.items():
            src_entities = by_type.get(src_type, [])
            if not src_entities:
                continue

            missing_count = 0
            for entity in src_entities:
                rels = self._store.get_relationships(entity.entity_id)
                has_expected = any(
                    r.rel_type.value == rel_type
                    for r in rels
                    if r.source_id == entity.entity_id
                )
                if not has_expected:
                    missing_count += 1

            coverage = 1 - (missing_count / len(src_entities))
            if coverage < min_coverage:
                issues.append(GraphIssue(
                    severity="warning",
                    category="missing",
                    message=f"Low coverage for {src_type}->{rel_type}->{tgt_type}: "
                           f"{coverage * 100:.0f}% (expected {min_coverage * 100:.0f}%)",
                    auto_fixable=True,
                    suggested_fix=f"Add {rel_type} relationships for "
                                 f"{missing_count} {src_type} entities",
                ))

        return issues

    def _add_missing_links(self, issues: List[GraphIssue]) -> int:
        """Add missing expected links. Returns count of links added."""
        # This would require more sophisticated inference
        # For now, just return 0 (manual review needed)
        return 0

    # ── Type Requirements ────────────────────────────────────────────

    def _check_type_requirements(self, entity) -> List[GraphIssue]:
        """Check type-specific relationship requirements."""
        issues = []
        etype = entity.entity_type.value

        rels = self._store.get_relationships(entity.entity_id)
        outgoing = [r for r in rels if r.source_id == entity.entity_id]
        outgoing_types = set(r.rel_type.value for r in outgoing)

        # Gene should encode protein
        if etype == "gene" and "encodes" not in outgoing_types:
            issues.append(GraphIssue(
                severity="info",
                category="missing",
                entity_ids=[entity.entity_id],
                message=f"Gene '{entity.name}' has no 'encodes' relationship",
                auto_fixable=True,
            ))

        # Mutation should affect gene
        if etype == "mutation" and "affects" not in outgoing_types:
            issues.append(GraphIssue(
                severity="warning",
                category="missing",
                entity_ids=[entity.entity_id],
                message=f"Mutation '{entity.name}' has no 'affects' relationship",
                auto_fixable=False,
            ))

        return issues

    # ── Utilities ────────────────────────────────────────────────────

    def _count_relationships(self) -> int:
        """Count total relationships in the store."""
        stats = self._store.stats()
        return stats.get("total_relationships", 0)
