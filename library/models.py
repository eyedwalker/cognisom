"""
Biological Entity Models
========================

Dataclass definitions for all entity types in the Biological Entity Library.
Each entity has a unique ID, standard identifiers, ontology references,
and metadata for FAIR compliance.

Entity types:
    Gene, Protein, Metabolite, CellType, TissueType, Pathway,
    Drug, Mutation, Receptor, Ligand, Organ, ParameterSet,
    SimulationScenario

Relationship types:
    binds_to, activates, inhibits, part_of, located_in,
    expressed_in, metabolizes, encodes, targets, regulates,
    parameterizes, uses_parameters

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
