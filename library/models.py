"""
Biological Entity Models
========================

Dataclass definitions for all entity types in the Biological Entity Library.
Each entity has a unique ID, standard identifiers, ontology references,
and metadata for FAIR compliance.

Entity types (20 total):
    Core biological:
        Gene, Protein, Metabolite, CellType, TissueType, Pathway,
        Drug, Mutation, Receptor, Ligand, Organ

    Simulation-first:
        ParameterSet, SimulationScenario

    Bio-USD aligned (Phase 0):
        PhysicalCell, ImmuneCellEntity, Exosome, Capillary,
        SpatialField, Tissue, PhysicsModelEntity

Relationship types (26 total):
    Core: binds_to, activates, inhibits, part_of, located_in,
          expressed_in, metabolizes, encodes, targets, regulates,
          produces, transports, catalyzes, phosphorylates, mutated_in,
          parameterizes, uses_parameters

    Bio-USD: supplies, kills, divides_into, releases, receives,
             contains, senses, secretes, uses_physics

Extensibility:
    New entity types can be registered at runtime using the entity_registry:

        from cognisom.library.models import entity_registry

        @entity_registry.register("virus")
        class VirusEntity(BioEntity):
            virus_family: str = ""
            genome_type: str = ""  # DNA, RNA, etc.
            ...
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Type

from cognisom.core.registry import Registry, registry_manager

log = logging.getLogger(__name__)


# ── Entity types ─────────────────────────────────────────────────────

class EntityType(str, Enum):
    GENE = "gene"
    PROTEIN = "protein"
    METABOLITE = "metabolite"
    CELL_TYPE = "cell_type"
    TISSUE_TYPE = "tissue_type"
    PATHWAY = "pathway"
    DRUG = "drug"
    MUTATION = "mutation"
    RECEPTOR = "receptor"
    LIGAND = "ligand"
    ORGAN = "organ"
    PARAMETER_SET = "parameter_set"
    SIMULATION_SCENARIO = "simulation_scenario"
    # Bio-USD aligned types (Phase 0)
    PHYSICAL_CELL = "physical_cell"      # Maps to BioCell
    IMMUNE_CELL = "immune_cell"          # Maps to BioImmuneCell
    EXOSOME = "exosome"                  # Maps to BioExosome
    CAPILLARY = "capillary"              # Maps to BioCapillary
    SPATIAL_FIELD = "spatial_field"      # Maps to BioSpatialField
    TISSUE = "tissue"                    # Maps to BioTissue
    PHYSICS_MODEL = "physics_model"      # Physics configuration reference


class RelationshipType(str, Enum):
    BINDS_TO = "binds_to"
    ACTIVATES = "activates"
    INHIBITS = "inhibits"
    PART_OF = "part_of"
    LOCATED_IN = "located_in"
    EXPRESSED_IN = "expressed_in"
    METABOLIZES = "metabolizes"
    ENCODES = "encodes"
    TARGETS = "targets"
    REGULATES = "regulates"
    PRODUCES = "produces"
    TRANSPORTS = "transports"
    CATALYZES = "catalyzes"
    PHOSPHORYLATES = "phosphorylates"
    MUTATED_IN = "mutated_in"
    PARAMETERIZES = "parameterizes"  # ParameterSet -> Entity (provides rates)
    USES_PARAMETERS = "uses_parameters"  # Scenario -> ParameterSet
    # Bio-USD aligned relationships
    SUPPLIES = "supplies"            # Capillary -> Tissue (oxygen/glucose)
    KILLS = "kills"                  # ImmuneCell -> Cell
    DIVIDES_INTO = "divides_into"    # Cell -> Cell (daughter cells)
    RELEASES = "releases"            # Cell -> Exosome
    RECEIVES = "receives"            # Cell -> Exosome (uptake)
    CONTAINS = "contains"            # Tissue -> PhysicalCell
    SENSES = "senses"                # Cell -> SpatialField
    SECRETES = "secretes"            # Cell -> SpatialField (cytokine release)
    USES_PHYSICS = "uses_physics"    # Scenario -> PhysicsModel


class EntityStatus(str, Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    REVIEW = "review"  # pending curation


# ── Entity Type Registry ─────────────────────────────────────────────
# Dynamic registry for entity types, enabling plugin-based extensibility.
# The EntityType enum is preserved for backward compatibility, but new
# types can be added at runtime without modifying this file.

# Forward declaration - will be set after BioEntity is defined
entity_registry: Registry = None  # type: ignore


def _init_entity_registry():
    """Initialize the entity registry. Called after BioEntity is defined."""
    global entity_registry
    # Import here to avoid circular dependency
    entity_registry = Registry(
        name="entities",
        base_class=BioEntity,
        allow_override=False,
    )
    registry_manager.add_registry("entities", entity_registry)
    return entity_registry


def register_entity(
    name: str,
    version: str = "1.0.0",
    description: str = "",
    **metadata
):
    """
    Decorator to register a new entity type.

    This enables plugin authors to add new biological entity types
    without modifying the core cognisom codebase.

    Parameters
    ----------
    name : str
        Entity type name (e.g., "virus", "bacterium")
    version : str
        Version string for the entity type
    description : str
        Human-readable description
    **metadata
        Additional metadata

    Returns
    -------
    Callable
        Decorator function

    Examples
    --------
    >>> from cognisom.library.models import register_entity, BioEntity
    >>>
    >>> @register_entity("virus", version="1.0.0")
    ... class VirusEntity(BioEntity):
    ...     virus_family: str = ""
    ...     genome_type: str = ""
    ...
    ...     def _extra_properties(self):
    ...         return {"virus_family": self.virus_family, "genome_type": self.genome_type}
    """
    def decorator(cls: Type[BioEntity]) -> Type[BioEntity]:
        if entity_registry is not None:
            entity_registry.register_class(
                name, cls, version=version, description=description, **metadata
            )
        else:
            # Registry not yet initialized - defer registration
            _DEFERRED_REGISTRATIONS.append((name, cls, version, description, metadata))
        return cls
    return decorator


# Deferred registrations for entities defined before registry init
_DEFERRED_REGISTRATIONS: List = []


def _register_deferred():
    """Register any entities that were decorated before registry init."""
    for name, cls, version, description, metadata in _DEFERRED_REGISTRATIONS:
        entity_registry.register_class(
            name, cls, version=version, description=description, **metadata
        )
    _DEFERRED_REGISTRATIONS.clear()


# ── Base entity ──────────────────────────────────────────────────────

@dataclass
class BioEntity:
    """Base class for all biological entities.

    Every entity has:
    - A unique internal ID (UUID)
    - A canonical name and optional synonyms
    - Ontology cross-references (GO, ChEBI, CL, UBERON, etc.)
    - Source attribution (which database/paper it came from)
    - FAIR metadata tags
    """
    entity_id: str = ""
    entity_type: EntityType = EntityType.GENE
    name: str = ""
    display_name: str = ""
    description: str = ""
    synonyms: List[str] = field(default_factory=list)

    # External identifiers
    external_ids: Dict[str, str] = field(default_factory=dict)
    # e.g. {"ncbi_gene": "7157", "uniprot": "P04637", "ensembl": "ENSG00000141510"}

    # Ontology references
    ontology_ids: List[str] = field(default_factory=list)
    # e.g. ["GO:0006915", "GO:0005634", "CHEBI:15377"]

    # Source and provenance
    source: str = ""  # "ncbi_gene", "uniprot", "manual", etc.
    source_url: str = ""
    references: List[str] = field(default_factory=list)  # PubMed IDs, DOIs

    # Status
    status: EntityStatus = EntityStatus.ACTIVE
    created_at: float = 0.0
    updated_at: float = 0.0
    created_by: str = "system"

    # Tags for categorization
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.entity_id:
            self.entity_id = str(uuid.uuid4())[:12]
        if not self.created_at:
            self.created_at = time.time()
        if not self.updated_at:
            self.updated_at = self.created_at
        if not self.display_name:
            self.display_name = self.name

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type.value,
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "synonyms": self.synonyms,
            "external_ids": self.external_ids,
            "ontology_ids": self.ontology_ids,
            "source": self.source,
            "source_url": self.source_url,
            "references": self.references,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "created_by": self.created_by,
            "tags": self.tags,
            "properties": self._extra_properties(),
        }

    def _extra_properties(self) -> dict:
        """Override in subclasses to add type-specific properties."""
        return {}

    @classmethod
    def from_dict(cls, data: dict) -> BioEntity:
        """Reconstruct entity from dict. Dispatches to subclass if known.

        Uses the entity registry for lookup, falling back to the legacy
        ENTITY_CLASS_MAP for backward compatibility.
        """
        etype = data.get("entity_type", "gene")

        # Try registry first (supports dynamic types)
        klass = None
        if entity_registry is not None and etype in entity_registry:
            klass = entity_registry.get(etype)

        # Fallback to legacy map
        if klass is None:
            klass = ENTITY_CLASS_MAP.get(etype, BioEntity)

        props = data.get("properties", {})

        entity = klass.__new__(klass)
        entity.entity_id = data.get("entity_id", "")
        entity.entity_type = EntityType(etype)
        entity.name = data.get("name", "")
        entity.display_name = data.get("display_name", "")
        entity.description = data.get("description", "")
        entity.synonyms = data.get("synonyms", [])
        entity.external_ids = data.get("external_ids", {})
        entity.ontology_ids = data.get("ontology_ids", [])
        entity.source = data.get("source", "")
        entity.source_url = data.get("source_url", "")
        entity.references = data.get("references", [])
        entity.status = EntityStatus(data.get("status", "active"))
        entity.created_at = data.get("created_at", 0.0)
        entity.updated_at = data.get("updated_at", 0.0)
        entity.created_by = data.get("created_by", "system")
        entity.tags = data.get("tags", [])

        # Apply type-specific properties
        entity._apply_properties(props)
        return entity

    def _apply_properties(self, props: dict):
        """Override in subclasses to restore type-specific fields."""
        pass


# ── Concrete entity types ────────────────────────────────────────────

@dataclass
class Gene(BioEntity):
    """A gene with genomic location and expression data."""
    entity_type: EntityType = EntityType.GENE
    symbol: str = ""
    full_name: str = ""
    chromosome: str = ""
    gene_type: str = ""  # oncogene, tumor_suppressor, housekeeping, etc.
    map_location: str = ""
    organism: str = "Homo sapiens"

    def _extra_properties(self) -> dict:
        return {
            "symbol": self.symbol,
            "full_name": self.full_name,
            "chromosome": self.chromosome,
            "gene_type": self.gene_type,
            "map_location": self.map_location,
            "organism": self.organism,
        }

    def _apply_properties(self, props: dict):
        self.symbol = props.get("symbol", "")
        self.full_name = props.get("full_name", "")
        self.chromosome = props.get("chromosome", "")
        self.gene_type = props.get("gene_type", "")
        self.map_location = props.get("map_location", "")
        self.organism = props.get("organism", "Homo sapiens")


@dataclass
class Protein(BioEntity):
    """A protein with sequence and structural data."""
    entity_type: EntityType = EntityType.PROTEIN
    gene_source: str = ""
    uniprot_id: str = ""
    pdb_ids: List[str] = field(default_factory=list)
    amino_acid_length: int = 0
    sequence_preview: str = ""  # first 100 residues
    function_summary: str = ""
    go_terms: List[str] = field(default_factory=list)
    pathways: List[str] = field(default_factory=list)

    def _extra_properties(self) -> dict:
        return {
            "gene_source": self.gene_source,
            "uniprot_id": self.uniprot_id,
            "pdb_ids": self.pdb_ids,
            "amino_acid_length": self.amino_acid_length,
            "sequence_preview": self.sequence_preview,
            "function_summary": self.function_summary,
            "go_terms": self.go_terms,
            "pathways": self.pathways,
        }

    def _apply_properties(self, props: dict):
        self.gene_source = props.get("gene_source", "")
        self.uniprot_id = props.get("uniprot_id", "")
        self.pdb_ids = props.get("pdb_ids", [])
        self.amino_acid_length = props.get("amino_acid_length", 0)
        self.sequence_preview = props.get("sequence_preview", "")
        self.function_summary = props.get("function_summary", "")
        self.go_terms = props.get("go_terms", [])
        self.pathways = props.get("pathways", [])


@dataclass
class Metabolite(BioEntity):
    """A small molecule: metabolite, cofactor, or signaling molecule."""
    entity_type: EntityType = EntityType.METABOLITE
    smiles: str = ""
    inchi: str = ""
    molecular_weight: float = 0.0
    molecular_formula: str = ""
    chebi_id: str = ""

    def _extra_properties(self) -> dict:
        return {
            "smiles": self.smiles,
            "inchi": self.inchi,
            "molecular_weight": self.molecular_weight,
            "molecular_formula": self.molecular_formula,
            "chebi_id": self.chebi_id,
        }

    def _apply_properties(self, props: dict):
        self.smiles = props.get("smiles", "")
        self.inchi = props.get("inchi", "")
        self.molecular_weight = props.get("molecular_weight", 0.0)
        self.molecular_formula = props.get("molecular_formula", "")
        self.chebi_id = props.get("chebi_id", "")


@dataclass
class Drug(BioEntity):
    """A drug or therapeutic compound."""
    entity_type: EntityType = EntityType.DRUG
    drug_class: str = ""  # e.g. "anti-androgen", "chemotherapy"
    mechanism: str = ""
    targets: List[str] = field(default_factory=list)  # gene/protein names
    smiles: str = ""
    molecular_weight: float = 0.0
    approval_status: str = ""  # "approved", "clinical_trial", "experimental"
    drugbank_id: str = ""

    def _extra_properties(self) -> dict:
        return {
            "drug_class": self.drug_class,
            "mechanism": self.mechanism,
            "targets": self.targets,
            "smiles": self.smiles,
            "molecular_weight": self.molecular_weight,
            "approval_status": self.approval_status,
            "drugbank_id": self.drugbank_id,
        }

    def _apply_properties(self, props: dict):
        self.drug_class = props.get("drug_class", "")
        self.mechanism = props.get("mechanism", "")
        self.targets = props.get("targets", [])
        self.smiles = props.get("smiles", "")
        self.molecular_weight = props.get("molecular_weight", 0.0)
        self.approval_status = props.get("approval_status", "")
        self.drugbank_id = props.get("drugbank_id", "")


@dataclass
class CellTypeEntity(BioEntity):
    """A cell type classification."""
    entity_type: EntityType = EntityType.CELL_TYPE
    cell_ontology_id: str = ""  # CL:0000000
    tissue_origin: str = ""
    markers: List[str] = field(default_factory=list)  # surface markers
    lineage: str = ""  # epithelial, mesenchymal, etc.

    def _extra_properties(self) -> dict:
        return {
            "cell_ontology_id": self.cell_ontology_id,
            "tissue_origin": self.tissue_origin,
            "markers": self.markers,
            "lineage": self.lineage,
        }

    def _apply_properties(self, props: dict):
        self.cell_ontology_id = props.get("cell_ontology_id", "")
        self.tissue_origin = props.get("tissue_origin", "")
        self.markers = props.get("markers", [])
        self.lineage = props.get("lineage", "")


@dataclass
class TissueTypeEntity(BioEntity):
    """A tissue type."""
    entity_type: EntityType = EntityType.TISSUE_TYPE
    uberon_id: str = ""  # UBERON:0001266
    organ: str = ""
    cell_types: List[str] = field(default_factory=list)

    def _extra_properties(self) -> dict:
        return {
            "uberon_id": self.uberon_id,
            "organ": self.organ,
            "cell_types": self.cell_types,
        }

    def _apply_properties(self, props: dict):
        self.uberon_id = props.get("uberon_id", "")
        self.organ = props.get("organ", "")
        self.cell_types = props.get("cell_types", [])


@dataclass
class Pathway(BioEntity):
    """A biological pathway (signaling, metabolic, regulatory)."""
    entity_type: EntityType = EntityType.PATHWAY
    pathway_type: str = ""  # signaling, metabolic, regulatory
    reactome_id: str = ""
    kegg_id: str = ""
    genes: List[str] = field(default_factory=list)
    proteins: List[str] = field(default_factory=list)

    def _extra_properties(self) -> dict:
        return {
            "pathway_type": self.pathway_type,
            "reactome_id": self.reactome_id,
            "kegg_id": self.kegg_id,
            "genes": self.genes,
            "proteins": self.proteins,
        }

    def _apply_properties(self, props: dict):
        self.pathway_type = props.get("pathway_type", "")
        self.reactome_id = props.get("reactome_id", "")
        self.kegg_id = props.get("kegg_id", "")
        self.genes = props.get("genes", [])
        self.proteins = props.get("proteins", [])


@dataclass
class Mutation(BioEntity):
    """A specific genetic mutation."""
    entity_type: EntityType = EntityType.MUTATION
    gene_symbol: str = ""
    mutation_type: str = ""  # missense, nonsense, frameshift, etc.
    position: str = ""  # e.g. "R175H", "del exon 5"
    consequence: str = ""  # loss_of_function, gain_of_function, neutral
    frequency: float = 0.0  # population frequency
    clinical_significance: str = ""  # pathogenic, benign, VUS

    def _extra_properties(self) -> dict:
        return {
            "gene_symbol": self.gene_symbol,
            "mutation_type": self.mutation_type,
            "position": self.position,
            "consequence": self.consequence,
            "frequency": self.frequency,
            "clinical_significance": self.clinical_significance,
        }

    def _apply_properties(self, props: dict):
        self.gene_symbol = props.get("gene_symbol", "")
        self.mutation_type = props.get("mutation_type", "")
        self.position = props.get("position", "")
        self.consequence = props.get("consequence", "")
        self.frequency = props.get("frequency", 0.0)
        self.clinical_significance = props.get("clinical_significance", "")


@dataclass
class Receptor(BioEntity):
    """A receptor protein on the cell surface or in the cytoplasm."""
    entity_type: EntityType = EntityType.RECEPTOR
    receptor_type: str = ""  # GPCR, RTK, nuclear, ion_channel
    ligands: List[str] = field(default_factory=list)
    signaling_pathway: str = ""
    gene_source: str = ""

    def _extra_properties(self) -> dict:
        return {
            "receptor_type": self.receptor_type,
            "ligands": self.ligands,
            "signaling_pathway": self.signaling_pathway,
            "gene_source": self.gene_source,
        }

    def _apply_properties(self, props: dict):
        self.receptor_type = props.get("receptor_type", "")
        self.ligands = props.get("ligands", [])
        self.signaling_pathway = props.get("signaling_pathway", "")
        self.gene_source = props.get("gene_source", "")


@dataclass
class Ligand(BioEntity):
    """A ligand that binds to a receptor."""
    entity_type: EntityType = EntityType.LIGAND
    ligand_type: str = ""  # hormone, cytokine, growth_factor
    receptors: List[str] = field(default_factory=list)
    molecular_weight: float = 0.0

    def _extra_properties(self) -> dict:
        return {
            "ligand_type": self.ligand_type,
            "receptors": self.receptors,
            "molecular_weight": self.molecular_weight,
        }

    def _apply_properties(self, props: dict):
        self.ligand_type = props.get("ligand_type", "")
        self.receptors = props.get("receptors", [])
        self.molecular_weight = props.get("molecular_weight", 0.0)


@dataclass
class OrganEntity(BioEntity):
    """An organ in the body."""
    entity_type: EntityType = EntityType.ORGAN
    uberon_id: str = ""
    system: str = ""  # reproductive, digestive, etc.
    tissue_types: List[str] = field(default_factory=list)

    def _extra_properties(self) -> dict:
        return {
            "uberon_id": self.uberon_id,
            "system": self.system,
            "tissue_types": self.tissue_types,
        }

    def _apply_properties(self, props: dict):
        self.uberon_id = props.get("uberon_id", "")
        self.system = props.get("system", "")
        self.tissue_types = props.get("tissue_types", [])


# ── Simulation-first entity types ────────────────────────────────────

@dataclass
class ParameterSet(BioEntity):
    """A versioned collection of simulation parameters.

    Parameter sets are first-class objects tied to biological context
    (e.g., "normal_prostate_epithelium", "CRPC_AR_amplified").
    They define rate constants, thresholds, and initial conditions
    that drive the simulation engine.

    Parameters are stored as a flat dict of name -> value pairs,
    each with units and valid ranges for Contract v1 validation.
    """
    entity_type: EntityType = EntityType.PARAMETER_SET
    version: str = "1.0"
    context: str = ""  # "normal", "cancer_CRPC", "drug_enzalutamide"
    module: str = ""   # "cellular", "spatial", "immune", "intracellular"
    parameters: Dict[str, float] = field(default_factory=dict)
    units: Dict[str, str] = field(default_factory=dict)
    ranges: Dict[str, list] = field(default_factory=dict)  # name -> [min, max]
    parent_set_id: str = ""  # fork from another ParameterSet
    is_validated: bool = False

    def _extra_properties(self) -> dict:
        return {
            "version": self.version,
            "context": self.context,
            "module": self.module,
            "parameters": self.parameters,
            "units": self.units,
            "ranges": self.ranges,
            "parent_set_id": self.parent_set_id,
            "is_validated": self.is_validated,
        }

    def _apply_properties(self, props: dict):
        self.version = props.get("version", "1.0")
        self.context = props.get("context", "")
        self.module = props.get("module", "")
        self.parameters = props.get("parameters", {})
        self.units = props.get("units", {})
        self.ranges = props.get("ranges", {})
        self.parent_set_id = props.get("parent_set_id", "")
        self.is_validated = props.get("is_validated", False)

    def get_param(self, name: str, default: float = 0.0) -> float:
        """Get a parameter value by name."""
        return self.parameters.get(name, default)

    def validate_ranges(self) -> list:
        """Check all parameters against their declared ranges.
        Returns list of violation strings (empty = all valid).
        """
        violations = []
        for name, val in self.parameters.items():
            if name in self.ranges:
                lo, hi = self.ranges[name]
                if val < lo or val > hi:
                    violations.append(
                        f"{name}={val} outside [{lo}, {hi}]"
                    )
        return violations


@dataclass
class SimulationScenario(BioEntity):
    """A complete simulation setup linking entities to parameter sets.

    A scenario defines:
    - Which cell types, genes, drugs are present
    - Which parameter sets govern their behavior
    - Initial conditions (cell counts, field concentrations)
    - Duration and resolution

    Scenarios are the primary unit of reproducible simulation.
    """
    entity_type: EntityType = EntityType.SIMULATION_SCENARIO
    scenario_type: str = ""  # "baseline", "drug_response", "immune_challenge"
    duration_hours: float = 48.0
    time_step_hours: float = 0.01
    grid_shape: list = field(default_factory=lambda: [200, 200, 100])
    resolution_um: float = 10.0
    # Entity references (IDs of entities in this scenario)
    cell_type_ids: List[str] = field(default_factory=list)
    gene_ids: List[str] = field(default_factory=list)
    drug_ids: List[str] = field(default_factory=list)
    parameter_set_ids: List[str] = field(default_factory=list)
    # Initial conditions
    initial_cell_counts: Dict[str, int] = field(default_factory=dict)
    initial_field_values: Dict[str, float] = field(default_factory=dict)

    def _extra_properties(self) -> dict:
        return {
            "scenario_type": self.scenario_type,
            "duration_hours": self.duration_hours,
            "time_step_hours": self.time_step_hours,
            "grid_shape": self.grid_shape,
            "resolution_um": self.resolution_um,
            "cell_type_ids": self.cell_type_ids,
            "gene_ids": self.gene_ids,
            "drug_ids": self.drug_ids,
            "parameter_set_ids": self.parameter_set_ids,
            "initial_cell_counts": self.initial_cell_counts,
            "initial_field_values": self.initial_field_values,
        }

    def _apply_properties(self, props: dict):
        self.scenario_type = props.get("scenario_type", "")
        self.duration_hours = props.get("duration_hours", 48.0)
        self.time_step_hours = props.get("time_step_hours", 0.01)
        self.grid_shape = props.get("grid_shape", [200, 200, 100])
        self.resolution_um = props.get("resolution_um", 10.0)
        self.cell_type_ids = props.get("cell_type_ids", [])
        self.gene_ids = props.get("gene_ids", [])
        self.drug_ids = props.get("drug_ids", [])
        self.parameter_set_ids = props.get("parameter_set_ids", [])
        self.initial_cell_counts = props.get("initial_cell_counts", {})
        self.initial_field_values = props.get("initial_field_values", {})


# ── Bio-USD Aligned Entity Types (Phase 0) ────────────────────────────

@dataclass
class PhysicalCell(BioEntity):
    """A physical cell instance with spatial position and state.

    Maps to BioCell in the Bio-USD schema. Unlike CellTypeEntity which
    represents a cell classification, PhysicalCell represents an actual
    cell in a simulation with position, cycle phase, and metabolic state.
    """
    entity_type: EntityType = EntityType.PHYSICAL_CELL
    cell_type_ref: str = ""         # Reference to CellTypeEntity
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    phase: str = "G1"               # Cell cycle: G0, G1, S, G2, M
    age_hours: float = 0.0
    alive: bool = True
    volume: float = 1.0             # Relative volume (1.0 = normal)
    division_time: float = 24.0     # Hours between divisions
    # Metabolic state
    oxygen: float = 0.21
    glucose: float = 5.0
    atp: float = 1000.0
    lactate: float = 0.0
    # Gene expression
    expression_level: float = 0.5
    active_genes: List[str] = field(default_factory=list)
    mutations: List[str] = field(default_factory=list)

    def _extra_properties(self) -> dict:
        return {
            "cell_type_ref": self.cell_type_ref,
            "position": self.position,
            "phase": self.phase,
            "age_hours": self.age_hours,
            "alive": self.alive,
            "volume": self.volume,
            "division_time": self.division_time,
            "oxygen": self.oxygen,
            "glucose": self.glucose,
            "atp": self.atp,
            "lactate": self.lactate,
            "expression_level": self.expression_level,
            "active_genes": self.active_genes,
            "mutations": self.mutations,
        }

    def _apply_properties(self, props: dict):
        self.cell_type_ref = props.get("cell_type_ref", "")
        self.position = props.get("position", [0.0, 0.0, 0.0])
        self.phase = props.get("phase", "G1")
        self.age_hours = props.get("age_hours", 0.0)
        self.alive = props.get("alive", True)
        self.volume = props.get("volume", 1.0)
        self.division_time = props.get("division_time", 24.0)
        self.oxygen = props.get("oxygen", 0.21)
        self.glucose = props.get("glucose", 5.0)
        self.atp = props.get("atp", 1000.0)
        self.lactate = props.get("lactate", 0.0)
        self.expression_level = props.get("expression_level", 0.5)
        self.active_genes = props.get("active_genes", [])
        self.mutations = props.get("mutations", [])


@dataclass
class ImmuneCellEntity(BioEntity):
    """An immune cell with activation and targeting state.

    Maps to BioImmuneCell in the Bio-USD schema. Represents T cells,
    NK cells, macrophages, and other immune cells with their targeting
    and killing behaviors.
    """
    entity_type: EntityType = EntityType.IMMUNE_CELL
    immune_type: str = "T_cell"     # T_cell, NK_cell, macrophage, dendritic, B_cell
    cell_type_ref: str = ""         # Reference to CellTypeEntity
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    activated: bool = False
    target_cell_id: str = ""        # ID of cell being targeted
    detection_radius: float = 10.0  # Scanning radius in um
    kill_radius: float = 5.0        # Effective kill distance in um
    kill_probability: float = 0.8   # Per-contact kill chance
    mhc1_expression: float = 1.0    # MHC class I surface expression
    activation_state: float = 0.0   # 0-1 activation level
    # Inherited from PhysicalCell
    phase: str = "G0"
    alive: bool = True

    def _extra_properties(self) -> dict:
        return {
            "immune_type": self.immune_type,
            "cell_type_ref": self.cell_type_ref,
            "position": self.position,
            "activated": self.activated,
            "target_cell_id": self.target_cell_id,
            "detection_radius": self.detection_radius,
            "kill_radius": self.kill_radius,
            "kill_probability": self.kill_probability,
            "mhc1_expression": self.mhc1_expression,
            "activation_state": self.activation_state,
            "phase": self.phase,
            "alive": self.alive,
        }

    def _apply_properties(self, props: dict):
        self.immune_type = props.get("immune_type", "T_cell")
        self.cell_type_ref = props.get("cell_type_ref", "")
        self.position = props.get("position", [0.0, 0.0, 0.0])
        self.activated = props.get("activated", False)
        self.target_cell_id = props.get("target_cell_id", "")
        self.detection_radius = props.get("detection_radius", 10.0)
        self.kill_radius = props.get("kill_radius", 5.0)
        self.kill_probability = props.get("kill_probability", 0.8)
        self.mhc1_expression = props.get("mhc1_expression", 1.0)
        self.activation_state = props.get("activation_state", 0.0)
        self.phase = props.get("phase", "G0")
        self.alive = props.get("alive", True)


@dataclass
class Exosome(BioEntity):
    """An extracellular vesicle carrying molecular cargo.

    Maps to BioExosome in the Bio-USD schema. Exosomes are small vesicles
    released by cells that carry mRNA, miRNA, and proteins for cell-cell
    communication.
    """
    entity_type: EntityType = EntityType.EXOSOME
    source_cell_id: str = ""        # Cell that released this exosome
    target_cell_id: str = ""        # Cell that will receive this exosome
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    velocity: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    radius: float = 0.05            # Exosome radius in um (30-150nm typical)
    cargo_mrna: List[str] = field(default_factory=list)
    cargo_mirna: List[str] = field(default_factory=list)
    cargo_proteins: List[str] = field(default_factory=list)
    released_at: float = 0.0        # Simulation time of release

    def _extra_properties(self) -> dict:
        return {
            "source_cell_id": self.source_cell_id,
            "target_cell_id": self.target_cell_id,
            "position": self.position,
            "velocity": self.velocity,
            "radius": self.radius,
            "cargo_mrna": self.cargo_mrna,
            "cargo_mirna": self.cargo_mirna,
            "cargo_proteins": self.cargo_proteins,
            "released_at": self.released_at,
        }

    def _apply_properties(self, props: dict):
        self.source_cell_id = props.get("source_cell_id", "")
        self.target_cell_id = props.get("target_cell_id", "")
        self.position = props.get("position", [0.0, 0.0, 0.0])
        self.velocity = props.get("velocity", [0.0, 0.0, 0.0])
        self.radius = props.get("radius", 0.05)
        self.cargo_mrna = props.get("cargo_mrna", [])
        self.cargo_mirna = props.get("cargo_mirna", [])
        self.cargo_proteins = props.get("cargo_proteins", [])
        self.released_at = props.get("released_at", 0.0)


@dataclass
class Capillary(BioEntity):
    """A blood vessel segment in the vascular network.

    Maps to BioCapillary in the Bio-USD schema. Represents segments of
    the vascular network that supply oxygen and glucose to tissues.
    """
    entity_type: EntityType = EntityType.CAPILLARY
    start_point: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    end_point: List[float] = field(default_factory=lambda: [100.0, 0.0, 0.0])
    radius: float = 5.0             # Vessel radius in um
    oxygen_conc: float = 0.21       # O2 concentration (0-0.21)
    glucose_conc: float = 5.0       # Glucose in mM
    flow_rate: float = 0.5          # Blood flow rate (arbitrary units)
    permeability: float = 1.0       # Vessel wall permeability
    vessel_type: str = "capillary"  # capillary, arteriole, venule

    def _extra_properties(self) -> dict:
        return {
            "start_point": self.start_point,
            "end_point": self.end_point,
            "radius": self.radius,
            "oxygen_conc": self.oxygen_conc,
            "glucose_conc": self.glucose_conc,
            "flow_rate": self.flow_rate,
            "permeability": self.permeability,
            "vessel_type": self.vessel_type,
        }

    def _apply_properties(self, props: dict):
        self.start_point = props.get("start_point", [0.0, 0.0, 0.0])
        self.end_point = props.get("end_point", [100.0, 0.0, 0.0])
        self.radius = props.get("radius", 5.0)
        self.oxygen_conc = props.get("oxygen_conc", 0.21)
        self.glucose_conc = props.get("glucose_conc", 5.0)
        self.flow_rate = props.get("flow_rate", 0.5)
        self.permeability = props.get("permeability", 1.0)
        self.vessel_type = props.get("vessel_type", "capillary")


@dataclass
class SpatialField(BioEntity):
    """A 3D concentration field (oxygen, glucose, cytokine, etc.).

    Maps to BioSpatialField in the Bio-USD schema. Represents continuous
    fields of molecules that cells sense and respond to.
    """
    entity_type: EntityType = EntityType.SPATIAL_FIELD
    field_type: str = "oxygen"      # oxygen, glucose, cytokine, lactate, morphogen
    grid_shape: List[int] = field(default_factory=lambda: [200, 200, 100])
    voxel_size: float = 10.0        # Size of each voxel in um
    diffusion_coeff: float = 2000.0 # Diffusion coefficient in um^2/s
    decay_rate: float = 0.0         # First-order decay rate
    min_value: float = 0.0
    max_value: float = 0.21         # Default for oxygen
    units: str = "fraction"         # mM, fraction, arbitrary
    data_file: str = ""             # Path to binary data file if stored

    def _extra_properties(self) -> dict:
        return {
            "field_type": self.field_type,
            "grid_shape": self.grid_shape,
            "voxel_size": self.voxel_size,
            "diffusion_coeff": self.diffusion_coeff,
            "decay_rate": self.decay_rate,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "units": self.units,
            "data_file": self.data_file,
        }

    def _apply_properties(self, props: dict):
        self.field_type = props.get("field_type", "oxygen")
        self.grid_shape = props.get("grid_shape", [200, 200, 100])
        self.voxel_size = props.get("voxel_size", 10.0)
        self.diffusion_coeff = props.get("diffusion_coeff", 2000.0)
        self.decay_rate = props.get("decay_rate", 0.0)
        self.min_value = props.get("min_value", 0.0)
        self.max_value = props.get("max_value", 0.21)
        self.units = props.get("units", "fraction")
        self.data_file = props.get("data_file", "")


@dataclass
class Tissue(BioEntity):
    """A collection of cells forming tissue architecture.

    Maps to BioTissue in the Bio-USD schema. Different from TissueTypeEntity
    which is a classification; this represents an actual tissue region in
    a simulation.
    """
    entity_type: EntityType = EntityType.TISSUE
    tissue_type_ref: str = ""       # Reference to TissueTypeEntity
    cell_ids: List[str] = field(default_factory=list)
    extent_min: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    extent_max: List[float] = field(default_factory=lambda: [100.0, 100.0, 50.0])
    cell_count: int = 0
    volume_um3: float = 0.0         # Tissue volume in cubic micrometers

    def _extra_properties(self) -> dict:
        return {
            "tissue_type_ref": self.tissue_type_ref,
            "cell_ids": self.cell_ids,
            "extent_min": self.extent_min,
            "extent_max": self.extent_max,
            "cell_count": self.cell_count,
            "volume_um3": self.volume_um3,
        }

    def _apply_properties(self, props: dict):
        self.tissue_type_ref = props.get("tissue_type_ref", "")
        self.cell_ids = props.get("cell_ids", [])
        self.extent_min = props.get("extent_min", [0.0, 0.0, 0.0])
        self.extent_max = props.get("extent_max", [100.0, 100.0, 50.0])
        self.cell_count = props.get("cell_count", 0)
        self.volume_um3 = props.get("volume_um3", 0.0)


@dataclass
class PhysicsModelEntity(BioEntity):
    """Reference to a physics model configuration.

    Links simulation scenarios to specific physics backends and models
    defined in the physics_interface module.
    """
    entity_type: EntityType = EntityType.PHYSICS_MODEL
    model_name: str = ""            # Registered model name (e.g., "cupy_diffusion")
    backend: str = "cupy"           # cupy, warp, newton, numpy
    model_type: str = "particle"    # particle, rigid_body, soft_body, fluid, diffusion
    config: Dict[str, float] = field(default_factory=dict)
    version: str = "1.0.0"
    is_gpu: bool = True

    def _extra_properties(self) -> dict:
        return {
            "model_name": self.model_name,
            "backend": self.backend,
            "model_type": self.model_type,
            "config": self.config,
            "version": self.version,
            "is_gpu": self.is_gpu,
        }

    def _apply_properties(self, props: dict):
        self.model_name = props.get("model_name", "")
        self.backend = props.get("backend", "cupy")
        self.model_type = props.get("model_type", "particle")
        self.config = props.get("config", {})
        self.version = props.get("version", "1.0.0")
        self.is_gpu = props.get("is_gpu", True)


# ── Relationship ─────────────────────────────────────────────────────

@dataclass
class Relationship:
    """A typed edge between two entities."""
    rel_id: str = ""
    source_id: str = ""
    target_id: str = ""
    rel_type: RelationshipType = RelationshipType.BINDS_TO
    confidence: float = 1.0  # 0-1, how confident we are
    evidence: str = ""  # source of evidence
    properties: Dict[str, str] = field(default_factory=dict)
    created_at: float = 0.0

    def __post_init__(self):
        if not self.rel_id:
            self.rel_id = str(uuid.uuid4())[:12]
        if not self.created_at:
            self.created_at = time.time()

    def to_dict(self) -> dict:
        return {
            "rel_id": self.rel_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "rel_type": self.rel_type.value,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "properties": self.properties,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Relationship:
        return cls(
            rel_id=data.get("rel_id", ""),
            source_id=data.get("source_id", ""),
            target_id=data.get("target_id", ""),
            rel_type=RelationshipType(data.get("rel_type", "binds_to")),
            confidence=data.get("confidence", 1.0),
            evidence=data.get("evidence", ""),
            properties=data.get("properties", {}),
            created_at=data.get("created_at", 0.0),
        )


# ── Class map for deserialization ────────────────────────────────────
# Legacy map maintained for backward compatibility.
# New code should use entity_registry.get() instead.

ENTITY_CLASS_MAP = {
    "gene": Gene,
    "protein": Protein,
    "metabolite": Metabolite,
    "cell_type": CellTypeEntity,
    "tissue_type": TissueTypeEntity,
    "pathway": Pathway,
    "drug": Drug,
    "mutation": Mutation,
    "receptor": Receptor,
    "ligand": Ligand,
    "organ": OrganEntity,
    "parameter_set": ParameterSet,
    "simulation_scenario": SimulationScenario,
    # Bio-USD aligned types
    "physical_cell": PhysicalCell,
    "immune_cell": ImmuneCellEntity,
    "exosome": Exosome,
    "capillary": Capillary,
    "spatial_field": SpatialField,
    "tissue": Tissue,
    "physics_model": PhysicsModelEntity,
}


# ── Initialize Registry ──────────────────────────────────────────────
# Initialize the entity registry and register all built-in types.

def _bootstrap_entity_registry():
    """Bootstrap the entity registry with all built-in entity types."""
    _init_entity_registry()

    # Register built-in entity types
    for name, cls in ENTITY_CLASS_MAP.items():
        entity_registry.register_class(
            name,
            cls,
            version="1.0.0",
            tags=["builtin"],
            description=cls.__doc__.strip().split("\n")[0] if cls.__doc__ else "",
        )

    # Register any deferred types (from @register_entity decorators)
    _register_deferred()

    log.debug(f"Entity registry initialized with {len(entity_registry)} types")


# Bootstrap on module load
_bootstrap_entity_registry()


# ── Public API ───────────────────────────────────────────────────────

def get_entity_class(entity_type: str) -> Type[BioEntity]:
    """
    Get the entity class for a given type name.

    This is the recommended way to look up entity classes,
    as it supports both built-in and plugin-provided types.

    Parameters
    ----------
    entity_type : str
        Entity type name (e.g., "gene", "virus")

    Returns
    -------
    Type[BioEntity]
        The entity class

    Raises
    ------
    KeyError
        If entity type is not registered

    Examples
    --------
    >>> cls = get_entity_class("gene")
    >>> entity = cls(name="TP53", symbol="TP53")
    """
    if entity_type in entity_registry:
        return entity_registry.get(entity_type)
    raise KeyError(f"Unknown entity type: {entity_type}. "
                   f"Available: {entity_registry.list_names()}")


def list_entity_types() -> List[str]:
    """
    List all registered entity types.

    Returns
    -------
    List[str]
        Names of all registered entity types
    """
    return entity_registry.list_names()


def create_entity(entity_type: str, **kwargs) -> BioEntity:
    """
    Create an entity instance by type name.

    Parameters
    ----------
    entity_type : str
        Entity type name (e.g., "gene", "virus")
    **kwargs
        Arguments passed to the entity constructor

    Returns
    -------
    BioEntity
        New entity instance

    Examples
    --------
    >>> gene = create_entity("gene", name="TP53", symbol="TP53")
    """
    return entity_registry.create(entity_type, **kwargs)
