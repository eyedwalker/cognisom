"""
Biological Entity Library
=========================

Phase 5A — A comprehensive catalog of biological entities (genes, proteins,
metabolites, cell types, pathways, drugs, mutations) with typed relationships,
ontology references, and full-text search.

Storage: SQLite with FTS5 for text search.
Sources: NCBI Gene, UniProt, PDB, Reactome, cBioPortal, DrugBank (via API).

Extensibility:
    New entity types can be registered at runtime:

        from cognisom.library import register_entity, BioEntity

        @register_entity("virus")
        class VirusEntity(BioEntity):
            ...
"""

from .models import (
    # Core classes
    BioEntity,
    Relationship,
    # Enums
    EntityType,
    RelationshipType,
    EntityStatus,
    # Entity types
    Gene,
    Protein,
    Metabolite,
    Drug,
    CellTypeEntity,
    TissueTypeEntity,
    Pathway,
    Mutation,
    Receptor,
    Ligand,
    OrganEntity,
    ParameterSet,
    SimulationScenario,
    # Registry
    entity_registry,
    register_entity,
    get_entity_class,
    list_entity_types,
    create_entity,
    # Legacy
    ENTITY_CLASS_MAP,
)

__version__ = "0.1.0"

__all__ = [
    # Core
    "BioEntity",
    "Relationship",
    # Enums
    "EntityType",
    "RelationshipType",
    "EntityStatus",
    # Entity types
    "Gene",
    "Protein",
    "Metabolite",
    "Drug",
    "CellTypeEntity",
    "TissueTypeEntity",
    "Pathway",
    "Mutation",
    "Receptor",
    "Ligand",
    "OrganEntity",
    "ParameterSet",
    "SimulationScenario",
    # Registry API
    "entity_registry",
    "register_entity",
    "get_entity_class",
    "list_entity_types",
    "create_entity",
    # Legacy
    "ENTITY_CLASS_MAP",
]
