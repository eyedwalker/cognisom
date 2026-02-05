"""
Ontology Sync & Validation Agent (Phase 7 — pulled forward)
============================================================

Autonomous agent that validates and enriches the Entity Library
by cross-referencing against standard ontologies and public databases.

Responsibilities:
    1. Ontology alignment  — Verify GO/ChEBI/UBERON/CL terms are valid
    2. Cross-reference sync — Ensure NCBI/UniProt/Ensembl IDs are current
    3. Relationship audit   — Flag orphaned or contradictory edges
    4. Completeness check   — Identify entities missing key annotations
    5. Auto-enrichment      — Fetch missing metadata from APIs

The agent runs in batch mode (full library scan) or incremental mode
(validate a single entity after insertion/update).

Usage::

    from cognisom.agent.ontology_sync import OntologyAgent

    agent = OntologyAgent(store)
    report = agent.full_audit()
    print(report.summary())

    # Incremental: validate one entity
    issues = agent.validate_entity(entity_id)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from cognisom.library.models import (
    BioEntity,
    Drug,
    EntityStatus,
    EntityType,
    Gene,
    Metabolite,
    Mutation,
    Pathway,
    ParameterSet,
    Protein,
    Receptor,
    Relationship,
    RelationshipType,
)

log = logging.getLogger(__name__)


# ── Known ontology prefixes ──────────────────────────────────────────

KNOWN_ONTOLOGY_PREFIXES = {
    "GO": "Gene Ontology",
    "CHEBI": "Chemical Entities of Biological Interest",
    "CL": "Cell Ontology",
    "UBERON": "Uber-anatomy Ontology",
    "HP": "Human Phenotype Ontology",
    "DOID": "Disease Ontology",
    "SO": "Sequence Ontology",
    "PR": "Protein Ontology",
    "MONDO": "MONDO Disease Ontology",
}

VALID_EXTERNAL_ID_SOURCES = {
    "ncbi_gene", "uniprot", "ensembl", "pdb", "chebi",
    "drugbank", "reactome", "kegg", "pubchem", "hgnc",
    "omim", "clinvar", "cosmic",
}


@dataclass
class AuditIssue:
    """A single finding from the ontology audit."""
    severity: str = "warning"  # "error", "warning", "info"
    category: str = ""         # "ontology", "xref", "relationship", "completeness"
    entity_id: str = ""
    entity_name: str = ""
    message: str = ""
    suggestion: str = ""       # auto-fixable suggestion


@dataclass
class AuditReport:
    """Results of a full library audit."""
    timestamp: float = 0.0
    entities_scanned: int = 0
    relationships_scanned: int = 0
    issues: List[AuditIssue] = field(default_factory=list)
    elapsed_sec: float = 0.0

    @property
    def errors(self) -> List[AuditIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> List[AuditIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def info_items(self) -> List[AuditIssue]:
        return [i for i in self.issues if i.severity == "info"]

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"Ontology Audit {status}",
            f"  Scanned: {self.entities_scanned} entities, "
            f"{self.relationships_scanned} relationships",
            f"  Errors: {len(self.errors)}, "
            f"Warnings: {len(self.warnings)}, "
            f"Info: {len(self.info_items)}",
            f"  Elapsed: {self.elapsed_sec:.2f}s",
        ]
        for issue in self.errors[:15]:
            lines.append(f"  [ERROR] {issue.category}: {issue.entity_name} — {issue.message}")
        for issue in self.warnings[:15]:
            lines.append(f"  [WARN]  {issue.category}: {issue.entity_name} — {issue.message}")
        if len(self.issues) > 30:
            lines.append(f"  ... and {len(self.issues) - 30} more findings")
        return "\n".join(lines)

    def by_category(self) -> Dict[str, int]:
        """Count issues by category."""
        counts: Dict[str, int] = {}
        for i in self.issues:
            counts[i.category] = counts.get(i.category, 0) + 1
        return counts


class OntologyAgent:
    """Validates and enriches the entity library against ontologies.

    Operates against an EntityStore instance.
    """

    def __init__(self, store):
        """
        Args:
            store: cognisom.library.store.EntityStore instance
        """
        self._store = store

    def full_audit(self) -> AuditReport:
        """Run a complete audit of all entities and relationships."""
        t0 = time.time()
        report = AuditReport(timestamp=t0)

        # Load all entities
        entities = self._store.search(limit=10000)
        report.entities_scanned = len(entities)

        for entity in entities:
            issues = self._validate_entity_obj(entity)
            report.issues.extend(issues)

        # Audit relationships
        rel_issues = self._audit_relationships()
        report.issues.extend(rel_issues)
        report.relationships_scanned = self._count_relationships()

        # Check for orphaned entities
        orphan_issues = self._check_orphaned_entities(entities)
        report.issues.extend(orphan_issues)

        # Check parameter set consistency
        param_issues = self._audit_parameter_sets(entities)
        report.issues.extend(param_issues)

        report.elapsed_sec = time.time() - t0
        return report

    def validate_entity(self, entity_id: str) -> List[AuditIssue]:
        """Validate a single entity by ID."""
        entity = self._store.get_entity(entity_id)
        if entity is None:
            return [AuditIssue(
                severity="error",
                category="existence",
                entity_id=entity_id,
                message=f"Entity {entity_id} not found in store",
            )]
        return self._validate_entity_obj(entity)

    # ── Per-entity validation ─────────────────────────────────

    def _validate_entity_obj(self, entity: BioEntity) -> List[AuditIssue]:
        """Run all checks on a single entity."""
        issues = []

        issues.extend(self._check_naming(entity))
        issues.extend(self._check_ontology_ids(entity))
        issues.extend(self._check_external_ids(entity))
        issues.extend(self._check_completeness(entity))

        # Type-specific checks
        if isinstance(entity, Gene):
            issues.extend(self._check_gene(entity))
        elif isinstance(entity, Protein):
            issues.extend(self._check_protein(entity))
        elif isinstance(entity, Mutation):
            issues.extend(self._check_mutation(entity))
        elif isinstance(entity, Drug):
            issues.extend(self._check_drug(entity))
        elif isinstance(entity, ParameterSet):
            issues.extend(self._check_parameter_set(entity))

        return issues

    def _check_naming(self, entity: BioEntity) -> List[AuditIssue]:
        """Check naming conventions."""
        issues = []

        if not entity.name:
            issues.append(AuditIssue(
                severity="error", category="completeness",
                entity_id=entity.entity_id, entity_name=entity.display_name,
                message="Entity has no name",
            ))

        if not entity.description and entity.entity_type not in (
            EntityType.PARAMETER_SET, EntityType.SIMULATION_SCENARIO,
        ):
            issues.append(AuditIssue(
                severity="info", category="completeness",
                entity_id=entity.entity_id, entity_name=entity.name,
                message="Entity missing description",
                suggestion="Add a brief description for searchability",
            ))

        return issues

    def _check_ontology_ids(self, entity: BioEntity) -> List[AuditIssue]:
        """Validate ontology IDs have known prefixes."""
        issues = []

        for ont_id in entity.ontology_ids:
            if ":" not in ont_id:
                issues.append(AuditIssue(
                    severity="warning", category="ontology",
                    entity_id=entity.entity_id, entity_name=entity.name,
                    message=f"Ontology ID '{ont_id}' missing prefix (expected PREFIX:ID)",
                    suggestion=f"Format as PREFIX:{ont_id}",
                ))
                continue

            prefix = ont_id.split(":")[0]
            if prefix not in KNOWN_ONTOLOGY_PREFIXES:
                issues.append(AuditIssue(
                    severity="info", category="ontology",
                    entity_id=entity.entity_id, entity_name=entity.name,
                    message=f"Unknown ontology prefix '{prefix}' in '{ont_id}'",
                ))

        # Type-specific expected ontologies
        expected = self._expected_ontologies(entity)
        if expected and not entity.ontology_ids:
            issues.append(AuditIssue(
                severity="warning", category="ontology",
                entity_id=entity.entity_id, entity_name=entity.name,
                message=f"Entity type {entity.entity_type.value} typically has "
                        f"ontology IDs ({', '.join(expected)})",
                suggestion=f"Add {expected[0]} identifier",
            ))

        return issues

    def _expected_ontologies(self, entity: BioEntity) -> List[str]:
        """Return expected ontology prefixes for an entity type."""
        mapping = {
            EntityType.GENE: ["GO"],
            EntityType.PROTEIN: ["GO", "PR"],
            EntityType.METABOLITE: ["CHEBI"],
            EntityType.CELL_TYPE: ["CL"],
            EntityType.TISSUE_TYPE: ["UBERON"],
            EntityType.ORGAN: ["UBERON"],
            EntityType.PATHWAY: ["GO"],
        }
        return mapping.get(entity.entity_type, [])

    def _check_external_ids(self, entity: BioEntity) -> List[AuditIssue]:
        """Validate external ID sources are known."""
        issues = []

        for source in entity.external_ids:
            if source not in VALID_EXTERNAL_ID_SOURCES:
                issues.append(AuditIssue(
                    severity="info", category="xref",
                    entity_id=entity.entity_id, entity_name=entity.name,
                    message=f"Non-standard external ID source: '{source}'",
                ))

        # Genes should have at least ncbi_gene or ensembl
        if entity.entity_type == EntityType.GENE:
            if not entity.external_ids.get("ncbi_gene") and not entity.external_ids.get("ensembl"):
                issues.append(AuditIssue(
                    severity="warning", category="xref",
                    entity_id=entity.entity_id, entity_name=entity.name,
                    message="Gene missing NCBI Gene ID or Ensembl ID",
                    suggestion="Fetch from NCBI Gene API",
                ))

        # Proteins should have UniProt
        if entity.entity_type == EntityType.PROTEIN:
            if not entity.external_ids.get("uniprot"):
                issues.append(AuditIssue(
                    severity="warning", category="xref",
                    entity_id=entity.entity_id, entity_name=entity.name,
                    message="Protein missing UniProt ID",
                    suggestion="Fetch from UniProt API",
                ))

        return issues

    def _check_completeness(self, entity: BioEntity) -> List[AuditIssue]:
        """Check for entities with minimal data that need enrichment."""
        issues = []

        # Count populated fields (rough heuristic)
        props = entity._extra_properties() if hasattr(entity, '_extra_properties') else {}
        populated = sum(1 for v in props.values() if v)
        total = len(props)

        if total > 0 and populated < total * 0.3:
            issues.append(AuditIssue(
                severity="info", category="completeness",
                entity_id=entity.entity_id, entity_name=entity.name,
                message=f"Only {populated}/{total} type-specific fields populated",
                suggestion="Consider enriching from public APIs",
            ))

        return issues

    # ── Type-specific checks ──────────────────────────────────

    def _check_gene(self, gene: Gene) -> List[AuditIssue]:
        issues = []
        if not gene.symbol:
            issues.append(AuditIssue(
                severity="warning", category="completeness",
                entity_id=gene.entity_id, entity_name=gene.name,
                message="Gene missing HGNC symbol",
            ))
        if not gene.chromosome:
            issues.append(AuditIssue(
                severity="info", category="completeness",
                entity_id=gene.entity_id, entity_name=gene.name,
                message="Gene missing chromosome location",
            ))
        valid_gene_types = {"oncogene", "tumor_suppressor", "housekeeping",
                           "transcription_factor", "kinase", "receptor", ""}
        if gene.gene_type and gene.gene_type not in valid_gene_types:
            issues.append(AuditIssue(
                severity="info", category="ontology",
                entity_id=gene.entity_id, entity_name=gene.name,
                message=f"Non-standard gene_type: '{gene.gene_type}'",
            ))
        return issues

    def _check_protein(self, prot: Protein) -> List[AuditIssue]:
        issues = []
        if not prot.gene_source:
            issues.append(AuditIssue(
                severity="info", category="completeness",
                entity_id=prot.entity_id, entity_name=prot.name,
                message="Protein missing gene_source reference",
            ))
        if prot.amino_acid_length == 0:
            issues.append(AuditIssue(
                severity="info", category="completeness",
                entity_id=prot.entity_id, entity_name=prot.name,
                message="Protein missing amino_acid_length",
            ))
        return issues

    def _check_mutation(self, mut: Mutation) -> List[AuditIssue]:
        issues = []
        if not mut.gene_symbol:
            issues.append(AuditIssue(
                severity="warning", category="completeness",
                entity_id=mut.entity_id, entity_name=mut.name,
                message="Mutation missing gene_symbol",
            ))
        if not mut.consequence:
            issues.append(AuditIssue(
                severity="info", category="completeness",
                entity_id=mut.entity_id, entity_name=mut.name,
                message="Mutation missing functional consequence",
            ))
        valid_consequences = {"loss_of_function", "gain_of_function", "neutral",
                              "dominant_negative", "unknown", ""}
        if mut.consequence and mut.consequence not in valid_consequences:
            issues.append(AuditIssue(
                severity="info", category="ontology",
                entity_id=mut.entity_id, entity_name=mut.name,
                message=f"Non-standard consequence: '{mut.consequence}'",
            ))
        return issues

    def _check_drug(self, drug: Drug) -> List[AuditIssue]:
        issues = []
        if not drug.targets:
            issues.append(AuditIssue(
                severity="warning", category="completeness",
                entity_id=drug.entity_id, entity_name=drug.name,
                message="Drug missing target gene/protein list",
            ))
        if not drug.mechanism:
            issues.append(AuditIssue(
                severity="info", category="completeness",
                entity_id=drug.entity_id, entity_name=drug.name,
                message="Drug missing mechanism of action",
            ))
        return issues

    def _check_parameter_set(self, ps: ParameterSet) -> List[AuditIssue]:
        issues = []
        if not ps.parameters:
            issues.append(AuditIssue(
                severity="error", category="completeness",
                entity_id=ps.entity_id, entity_name=ps.name,
                message="ParameterSet has no parameters defined",
            ))
        # Validate ranges
        violations = ps.validate_ranges()
        for v in violations:
            issues.append(AuditIssue(
                severity="error", category="range",
                entity_id=ps.entity_id, entity_name=ps.name,
                message=f"Parameter out of range: {v}",
            ))
        if not ps.module:
            issues.append(AuditIssue(
                severity="warning", category="completeness",
                entity_id=ps.entity_id, entity_name=ps.name,
                message="ParameterSet missing module assignment",
            ))
        return issues

    # ── Relationship audit ────────────────────────────────────

    def _audit_relationships(self) -> List[AuditIssue]:
        """Check all relationships for integrity."""
        issues = []

        # Get all entity IDs
        all_entities = self._store.search(limit=10000)
        valid_ids = {e.entity_id for e in all_entities}

        # Check each entity's relationships
        for entity in all_entities:
            rels = self._store.get_relationships(entity.entity_id)
            for rel in rels:
                if rel.source_id not in valid_ids:
                    issues.append(AuditIssue(
                        severity="error", category="relationship",
                        entity_id=rel.rel_id,
                        entity_name=f"rel:{rel.source_id}->{rel.target_id}",
                        message=f"Relationship source_id '{rel.source_id}' not found",
                    ))
                if rel.target_id not in valid_ids:
                    issues.append(AuditIssue(
                        severity="error", category="relationship",
                        entity_id=rel.rel_id,
                        entity_name=f"rel:{rel.source_id}->{rel.target_id}",
                        message=f"Relationship target_id '{rel.target_id}' not found",
                    ))
                if rel.confidence < 0 or rel.confidence > 1:
                    issues.append(AuditIssue(
                        severity="warning", category="relationship",
                        entity_id=rel.rel_id,
                        entity_name=f"rel:{rel.source_id}->{rel.target_id}",
                        message=f"Confidence {rel.confidence} outside [0, 1]",
                    ))

        return issues

    def _count_relationships(self) -> int:
        """Count total relationships in the store."""
        stats = self._store.stats()
        return stats.get("total_relationships", 0)

    def _check_orphaned_entities(self, entities: List[BioEntity]) -> List[AuditIssue]:
        """Find entities with no relationships (isolated nodes)."""
        issues = []

        for entity in entities:
            # ParameterSets and Scenarios are expected to be connected
            if entity.entity_type in (EntityType.PARAMETER_SET, EntityType.SIMULATION_SCENARIO):
                continue

            rels = self._store.get_relationships(entity.entity_id)
            if not rels and entity.entity_type not in (
                EntityType.ORGAN,  # Organs may be standalone
            ):
                issues.append(AuditIssue(
                    severity="info", category="relationship",
                    entity_id=entity.entity_id, entity_name=entity.name,
                    message=f"Entity has no relationships (isolated node)",
                    suggestion="Consider adding relationships to related entities",
                ))

        return issues

    def _audit_parameter_sets(self, entities: List[BioEntity]) -> List[AuditIssue]:
        """Check parameter set consistency across scenarios."""
        issues = []
        param_sets = [e for e in entities if isinstance(e, ParameterSet)]

        # Check for version conflicts
        by_context = {}
        for ps in param_sets:
            key = (ps.context, ps.module)
            if key not in by_context:
                by_context[key] = []
            by_context[key].append(ps)

        for (ctx, mod), sets in by_context.items():
            if len(sets) > 1:
                versions = [s.version for s in sets]
                issues.append(AuditIssue(
                    severity="info", category="parameter_set",
                    entity_id=sets[0].entity_id,
                    entity_name=f"{ctx}/{mod}",
                    message=f"Multiple parameter sets for context='{ctx}', "
                            f"module='{mod}': versions {versions}",
                ))

        return issues
