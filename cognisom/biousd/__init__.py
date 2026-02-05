"""
Bio-USD: Biological Entity Schema for OpenUSD
==============================================

Cognisom Phase 5-6 — A standardized OpenUSD schema for representing
biological entities (cells, proteins, molecules, tissues, spatial fields)
in a format compatible with NVIDIA Omniverse and the broader USD ecosystem.

This package provides:
- ``schema``: Python dataclass definitions mirroring the USDA schema
- ``converter``: Export Cognisom simulation state to .usda files
- ``sbml_converter``: Convert SBML models to Bio-USD scene format
- ``sync``: Bidirectional sync between simulation engine and USD scene
- ``validator``: Schema validation for CI + reference scenes
- ``schemas/``: Reference .usda schema definition files

Bio-USD is designed to become an open standard submitted to the
Alliance for OpenUSD (AOUSD) in Phase 9.

Extensibility:
    New prim types can be registered at runtime:

        from cognisom.biousd import register_prim, BioUnit

        @register_prim("bio_virus")
        class BioVirusParticle(BioUnit):
            ...
"""

from .schema import (
    # Base types
    BioUnit,
    # Core prims
    BioCell,
    BioImmuneCell,
    BioGene,
    BioProtein,
    BioMolecule,
    BioTissue,
    BioCapillary,
    BioSpatialField,
    BioExosome,
    BioScene,
    # API schemas
    BioMetabolicAPI,
    BioGeneExpressionAPI,
    BioEpigeneticAPI,
    BioImmuneAPI,
    BioInteractionAPI,
    # Enums
    CellType,
    CellPhase,
    ImmuneCellType,
    SpatialFieldType,
    GeneType,
    # Registry
    prim_registry,
    api_registry,
    register_prim,
    register_api_schema,
    get_prim_class,
    get_api_class,
    list_prim_types,
    list_api_schemas,
    create_prim,
)

__version__ = "0.2.0"

__all__ = [
    # Base
    "BioUnit",
    # Core prims
    "BioCell",
    "BioImmuneCell",
    "BioGene",
    "BioProtein",
    "BioMolecule",
    "BioTissue",
    "BioCapillary",
    "BioSpatialField",
    "BioExosome",
    "BioScene",
    # API schemas
    "BioMetabolicAPI",
    "BioGeneExpressionAPI",
    "BioEpigeneticAPI",
    "BioImmuneAPI",
    "BioInteractionAPI",
    # Enums
    "CellType",
    "CellPhase",
    "ImmuneCellType",
    "SpatialFieldType",
    "GeneType",
    # Registry API
    "prim_registry",
    "api_registry",
    "register_prim",
    "register_api_schema",
    "get_prim_class",
    "get_api_class",
    "list_prim_types",
    "list_api_schemas",
    "create_prim",
]
