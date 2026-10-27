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

IMPORT THIS BEFORE OPENING A STAGE
----------------------------------
``cognisom.biousd.standard`` is the publishable Bio-USD package, and
importing it hands the codeless ``BioCellAPI`` schema plugin to USD's
``Plug.Registry``. This package imports it below, so ``import
cognisom.biousd`` is the registration. The order matters, and silently:
USD builds its schema registry once per process, the first time anything
needs a prim definition -- the first ``Usd.Stage.CreateNew`` or ``Open``
that defines or composes a typed prim does -- and a plugin registered
after that is never consulted. Measured on usd-core 25.11 and 26.11:
``RegisterPlugins`` returns the plugin and ``FindAppliedAPIPrimDefinition``
still answers ``None``, however many times it is called.

So the rule for a Cognisom process is: import this package before the
first stage. Cognisom's own writers obey it by construction --
:mod:`.usd_writer` and :mod:`cognisom.visualization.run_usd` import
``cognisom.biousd`` at module level, before any ``pxr`` call -- and the
dashboard imports it in ``app.py``, because its pages open stages lazily
and any of them may be the first. A late import costs one WARNING and
nothing else: stages still write, open and validate (a writer needs no
plugin; the validator reads authored data), only the schema's fallbacks
and documentation are missing in that process. Without ``usd-core``
installed the import is quiet and :data:`schema_registered` is ``False``.
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

# Registers the BioCellAPI plugin as a side effect (see the docstring);
# bound after .schema so the registration cannot be skipped by a partial
# import of this package.
from . import standard                                          # noqa: E402
from .standard import register_schema, schema_is_known          # noqa: E402

#: Whether the plugin was handed to ``Plug.Registry`` at import time.
#: ``False`` means usd-core is absent, the plugin files are missing, or
#: ``BIOUSD_NO_PLUGIN`` is set. Whether the registry then KNOWS the
#: schema is ``schema_is_known()``'s answer, asked when a caller asks
#: (asking builds the registry, which the import deliberately does not
#: do); a stage opened before the import makes that answer ``False``
#: with one WARNING, and the written files are the same either way.
schema_registered = standard.schema_registered

__version__ = "0.2.0"

__all__ = [
    # Bio-USD 0.2.0 schema plugin
    "standard",
    "schema_registered",
    "register_schema",
    "schema_is_known",
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
