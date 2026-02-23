"""
Biological Entity Models
========================

Dataclass definitions for all entity types in the Biological Entity Library.
Each entity has a unique ID, standard identifiers, ontology references,
and metadata for FAIR compliance.

Entity types (31 total):
    Core biological:
        Gene, Protein, Metabolite, CellType, TissueType, Pathway,
        Drug, Mutation, Receptor, Ligand, Organ

    Simulation-first:
        ParameterSet, SimulationScenario

    Bio-USD aligned (Phase 0):
        PhysicalCell, ImmuneCellEntity, Exosome, Capillary,
        SpatialField, Tissue, PhysicsModelEntity

    Researcher workflow:
        SimulationRun, ResearchProject

    Immunology:
        Virus, Bacterium, Antibody, Antigen, Cytokine,
        PatternRecognitionReceptor, ComplementComponent, MHCMolecule,
        AdhesionMolecule

Relationship types (36 total):
    Core: binds_to, activates, inhibits, part_of, located_in,
          expressed_in, metabolizes, encodes, targets, regulates,
          produces, transports, catalyzes, phosphorylates, mutated_in,
          parameterizes, uses_parameters

    Bio-USD: supplies, kills, divides_into, releases, receives,
             contains, senses, secretes, uses_physics

    Immunology: presents, recognizes, neutralizes, opsonizes,
                activates_complement, polarizes, differentiates

    Diapedesis: adheres_to, transmigrates_through, expresses_on_surface

Extensibility:
    New entity types can be registered at runtime using the entity_registry.
    See register_entity() and entity_registry.
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
    # Researcher workflow types
    SIMULATION_RUN = "simulation_run"
    RESEARCH_PROJECT = "research_project"
    # Immunology entity types
    VIRUS = "virus"
    BACTERIUM = "bacterium"
    ANTIBODY = "antibody"
    ANTIGEN = "antigen"
    CYTOKINE = "cytokine"
    PATTERN_RECOGNITION_RECEPTOR = "prr"
    COMPLEMENT_COMPONENT = "complement"
    MHC_MOLECULE = "mhc"
    ADHESION_MOLECULE = "adhesion_molecule"


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
    # Researcher workflow relationships
    EXECUTES = "executes"            # SimulationRun -> SimulationScenario
    BELONGS_TO = "belongs_to"        # SimulationRun -> ResearchProject
    COMPARES_TO = "compares_to"      # SimulationRun -> SimulationRun
    # Immunology relationships
    PRESENTS = "presents"            # MHC -> Antigen
    RECOGNIZES = "recognizes"        # TCR/BCR -> Antigen
    NEUTRALIZES = "neutralizes"      # Antibody -> Pathogen
    OPSONIZES = "opsonizes"          # Antibody/Complement -> Pathogen
    ACTIVATES_COMPLEMENT = "activates_complement"  # Antibody -> Complement
    POLARIZES = "polarizes"          # Cytokine -> Macrophage M1/M2
    DIFFERENTIATES = "differentiates"  # Cytokine -> T cell subset
    # Diapedesis / leukocyte migration relationships
    ADHERES_TO = "adheres_to"                        # Leukocyte -> Endothelium (via selectin/integrin)
    TRANSMIGRATES_THROUGH = "transmigrates_through"  # Leukocyte -> Endothelial junction
    EXPRESSES_ON_SURFACE = "expresses_on_surface"    # Cell -> AdhesionMolecule


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
    # Extended identifiers for cross-referencing
    ncbi_gene_id: str = ""
    ensembl_id: str = ""
    hgnc_id: str = ""
    refseq_mrna: str = ""
    refseq_protein: str = ""
    # Rich content for human interpretation
    summary: str = ""  # NCBI Gene functional summary
    gene_group: str = ""
    phenotype_ids: List[str] = field(default_factory=list)
    expression_tissues: List[str] = field(default_factory=list)

    def _extra_properties(self) -> dict:
        return {
            "symbol": self.symbol,
            "full_name": self.full_name,
            "chromosome": self.chromosome,
            "gene_type": self.gene_type,
            "map_location": self.map_location,
            "organism": self.organism,
            "ncbi_gene_id": self.ncbi_gene_id,
            "ensembl_id": self.ensembl_id,
            "hgnc_id": self.hgnc_id,
            "refseq_mrna": self.refseq_mrna,
            "refseq_protein": self.refseq_protein,
            "summary": self.summary,
            "gene_group": self.gene_group,
            "phenotype_ids": self.phenotype_ids,
            "expression_tissues": self.expression_tissues,
        }

    def _apply_properties(self, props: dict):
        self.symbol = props.get("symbol", "")
        self.full_name = props.get("full_name", "")
        self.chromosome = props.get("chromosome", "")
        self.gene_type = props.get("gene_type", "")
        self.map_location = props.get("map_location", "")
        self.organism = props.get("organism", "Homo sapiens")
        self.ncbi_gene_id = props.get("ncbi_gene_id", "")
        self.ensembl_id = props.get("ensembl_id", "")
        self.hgnc_id = props.get("hgnc_id", "")
        self.refseq_mrna = props.get("refseq_mrna", "")
        self.refseq_protein = props.get("refseq_protein", "")
        self.summary = props.get("summary", "")
        self.gene_group = props.get("gene_group", "")
        self.phenotype_ids = props.get("phenotype_ids", [])
        self.expression_tissues = props.get("expression_tissues", [])


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
    # 3D structure for RTX visualization
    alphafold_id: str = ""
    alphafold_url: str = ""
    mass_daltons: float = 0.0
    subcellular_location: str = ""
    domains: List[str] = field(default_factory=list)
    tissue_specificity: str = ""
    disease_associations: List[str] = field(default_factory=list)
    # Best experimental structure
    structure_method: str = ""  # X-ray, Cryo-EM, NMR, AlphaFold
    resolution_angstrom: float = 0.0
    structure_url: str = ""
    chain_count: int = 0

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
            "alphafold_id": self.alphafold_id,
            "alphafold_url": self.alphafold_url,
            "mass_daltons": self.mass_daltons,
            "subcellular_location": self.subcellular_location,
            "domains": self.domains,
            "tissue_specificity": self.tissue_specificity,
            "disease_associations": self.disease_associations,
            "structure_method": self.structure_method,
            "resolution_angstrom": self.resolution_angstrom,
            "structure_url": self.structure_url,
            "chain_count": self.chain_count,
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
        self.alphafold_id = props.get("alphafold_id", "")
        self.alphafold_url = props.get("alphafold_url", "")
        self.mass_daltons = props.get("mass_daltons", 0.0)
        self.subcellular_location = props.get("subcellular_location", "")
        self.domains = props.get("domains", [])
        self.tissue_specificity = props.get("tissue_specificity", "")
        self.disease_associations = props.get("disease_associations", [])
        self.structure_method = props.get("structure_method", "")
        self.resolution_angstrom = props.get("resolution_angstrom", 0.0)
        self.structure_url = props.get("structure_url", "")
        self.chain_count = props.get("chain_count", 0)


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
    # PubChem properties for visualization
    pubchem_cid: int = 0
    inchi: str = ""
    inchi_key: str = ""
    logp: float = 0.0  # XLogP (lipophilicity → surface texture)
    tpsa: float = 0.0  # topological polar surface area
    hbd: int = 0  # H-bond donors
    hba: int = 0  # H-bond acceptors
    rotatable_bonds: int = 0
    molecular_formula: str = ""
    indication: str = ""
    # 3D structure URLs for RTX rendering
    conformer_3d_url: str = ""
    structure_2d_url: str = ""

    def _extra_properties(self) -> dict:
        return {
            "drug_class": self.drug_class,
            "mechanism": self.mechanism,
            "targets": self.targets,
            "smiles": self.smiles,
            "molecular_weight": self.molecular_weight,
            "approval_status": self.approval_status,
            "drugbank_id": self.drugbank_id,
            "pubchem_cid": self.pubchem_cid,
            "inchi": self.inchi,
            "inchi_key": self.inchi_key,
            "logp": self.logp,
            "tpsa": self.tpsa,
            "hbd": self.hbd,
            "hba": self.hba,
            "rotatable_bonds": self.rotatable_bonds,
            "molecular_formula": self.molecular_formula,
            "indication": self.indication,
            "conformer_3d_url": self.conformer_3d_url,
            "structure_2d_url": self.structure_2d_url,
        }

    def _apply_properties(self, props: dict):
        self.drug_class = props.get("drug_class", "")
        self.mechanism = props.get("mechanism", "")
        self.targets = props.get("targets", [])
        self.smiles = props.get("smiles", "")
        self.molecular_weight = props.get("molecular_weight", 0.0)
        self.approval_status = props.get("approval_status", "")
        self.drugbank_id = props.get("drugbank_id", "")
        self.pubchem_cid = props.get("pubchem_cid", 0)
        self.inchi = props.get("inchi", "")
        self.inchi_key = props.get("inchi_key", "")
        self.logp = props.get("logp", 0.0)
        self.tpsa = props.get("tpsa", 0.0)
        self.hbd = props.get("hbd", 0)
        self.hba = props.get("hba", 0)
        self.rotatable_bonds = props.get("rotatable_bonds", 0)
        self.molecular_formula = props.get("molecular_formula", "")
        self.indication = props.get("indication", "")
        self.conformer_3d_url = props.get("conformer_3d_url", "")
        self.structure_2d_url = props.get("structure_2d_url", "")


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
    # Extended pathway metadata
    reaction_count: int = 0
    diagram_url: str = ""
    go_terms: List[str] = field(default_factory=list)
    disease_associations: List[str] = field(default_factory=list)

    def _extra_properties(self) -> dict:
        return {
            "pathway_type": self.pathway_type,
            "reactome_id": self.reactome_id,
            "kegg_id": self.kegg_id,
            "genes": self.genes,
            "proteins": self.proteins,
            "reaction_count": self.reaction_count,
            "diagram_url": self.diagram_url,
            "go_terms": self.go_terms,
            "disease_associations": self.disease_associations,
        }

    def _apply_properties(self, props: dict):
        self.pathway_type = props.get("pathway_type", "")
        self.reactome_id = props.get("reactome_id", "")
        self.kegg_id = props.get("kegg_id", "")
        self.genes = props.get("genes", [])
        self.proteins = props.get("proteins", [])
        self.reaction_count = props.get("reaction_count", 0)
        self.diagram_url = props.get("diagram_url", "")
        self.go_terms = props.get("go_terms", [])
        self.disease_associations = props.get("disease_associations", [])


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


# ── Researcher Workflow Entity Types ─────────────────────────────────


@dataclass
class SimulationRun(BioEntity):
    """A single execution of a simulation scenario with persisted results.

    Captures the full context needed for reproducibility: frozen config
    snapshot, hardware/software info, random seed, and references to
    artifact files on disk.  Final metrics and eval grades are stored
    inline (JSONB) for fast listing and filtering.
    """
    entity_type: EntityType = EntityType.SIMULATION_RUN

    # Links
    scenario_id: str = ""           # -> SimulationScenario
    project_id: str = ""            # -> ResearchProject (optional grouping)

    # Execution state
    run_status: str = "pending"     # pending | running | completed | failed
    started_at: float = 0.0         # epoch timestamp
    completed_at: float = 0.0
    elapsed_seconds: float = 0.0

    # Configuration snapshot (frozen at run creation for reproducibility)
    config_snapshot: Dict = field(default_factory=dict)
    modules_enabled: Dict[str, bool] = field(default_factory=dict)
    dt: float = 0.05
    duration_hours: float = 6.0
    random_seed: int = 0

    # Reproducibility metadata
    hardware_info: Dict = field(default_factory=dict)
    software_versions: Dict = field(default_factory=dict)
    git_commit: str = ""

    # Results summary (lightweight, queryable via JSONB)
    final_metrics: Dict = field(default_factory=dict)
    event_summary: Dict = field(default_factory=dict)

    # Evaluation grades
    accuracy_grade: str = ""        # A-F from AccuracyReport
    fidelity_grade: str = ""        # A-F from FidelityReport

    # Artifact paths (relative to /app/data/runs/<entity_id>/)
    artifacts_dir: str = ""
    time_series_file: str = "timeseries.csv"
    cell_snapshots_file: str = "cell_snapshots.npz"
    event_log_file: str = "events.jsonl"
    eval_report_file: str = "eval_report.json"

    def _extra_properties(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "project_id": self.project_id,
            "run_status": self.run_status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "elapsed_seconds": self.elapsed_seconds,
            "config_snapshot": self.config_snapshot,
            "modules_enabled": self.modules_enabled,
            "dt": self.dt,
            "duration_hours": self.duration_hours,
            "random_seed": self.random_seed,
            "hardware_info": self.hardware_info,
            "software_versions": self.software_versions,
            "git_commit": self.git_commit,
            "final_metrics": self.final_metrics,
            "event_summary": self.event_summary,
            "accuracy_grade": self.accuracy_grade,
            "fidelity_grade": self.fidelity_grade,
            "artifacts_dir": self.artifacts_dir,
            "time_series_file": self.time_series_file,
            "cell_snapshots_file": self.cell_snapshots_file,
            "event_log_file": self.event_log_file,
            "eval_report_file": self.eval_report_file,
        }

    def _apply_properties(self, props: dict):
        self.scenario_id = props.get("scenario_id", "")
        self.project_id = props.get("project_id", "")
        self.run_status = props.get("run_status", "pending")
        self.started_at = props.get("started_at", 0.0)
        self.completed_at = props.get("completed_at", 0.0)
        self.elapsed_seconds = props.get("elapsed_seconds", 0.0)
        self.config_snapshot = props.get("config_snapshot", {})
        self.modules_enabled = props.get("modules_enabled", {})
        self.dt = props.get("dt", 0.05)
        self.duration_hours = props.get("duration_hours", 6.0)
        self.random_seed = props.get("random_seed", 0)
        self.hardware_info = props.get("hardware_info", {})
        self.software_versions = props.get("software_versions", {})
        self.git_commit = props.get("git_commit", "")
        self.final_metrics = props.get("final_metrics", {})
        self.event_summary = props.get("event_summary", {})
        self.accuracy_grade = props.get("accuracy_grade", "")
        self.fidelity_grade = props.get("fidelity_grade", "")
        self.artifacts_dir = props.get("artifacts_dir", "")
        self.time_series_file = props.get("time_series_file", "timeseries.csv")
        self.cell_snapshots_file = props.get("cell_snapshots_file", "cell_snapshots.npz")
        self.event_log_file = props.get("event_log_file", "events.jsonl")
        self.eval_report_file = props.get("eval_report_file", "eval_report.json")


@dataclass
class ResearchProject(BioEntity):
    """A collection of simulation runs grouped for analysis and publication.

    Projects link multiple runs together (e.g., baseline + treatment),
    store researcher-authored text for paper sections, and manage
    citations for the final manuscript.
    """
    entity_type: EntityType = EntityType.RESEARCH_PROJECT

    # Project metadata
    title: str = ""
    hypothesis: str = ""
    methodology: str = ""

    # Linked runs
    run_ids: List[str] = field(default_factory=list)
    baseline_run_id: str = ""

    # Paper content (editable text sections)
    abstract: str = ""
    introduction: str = ""
    discussion: str = ""

    # Citations
    bibtex_entries: Dict[str, str] = field(default_factory=dict)

    # Publication status
    paper_status: str = "draft"     # draft | in_review | published
    paper_artifact_dir: str = ""

    def _extra_properties(self) -> dict:
        return {
            "title": self.title,
            "hypothesis": self.hypothesis,
            "methodology": self.methodology,
            "run_ids": self.run_ids,
            "baseline_run_id": self.baseline_run_id,
            "abstract": self.abstract,
            "introduction": self.introduction,
            "discussion": self.discussion,
            "bibtex_entries": self.bibtex_entries,
            "paper_status": self.paper_status,
            "paper_artifact_dir": self.paper_artifact_dir,
        }

    def _apply_properties(self, props: dict):
        self.title = props.get("title", "")
        self.hypothesis = props.get("hypothesis", "")
        self.methodology = props.get("methodology", "")
        self.run_ids = props.get("run_ids", [])
        self.baseline_run_id = props.get("baseline_run_id", "")
        self.abstract = props.get("abstract", "")
        self.introduction = props.get("introduction", "")
        self.discussion = props.get("discussion", "")
        self.bibtex_entries = props.get("bibtex_entries", {})
        self.paper_status = props.get("paper_status", "draft")
        self.paper_artifact_dir = props.get("paper_artifact_dir", "")


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
    # Expanded immunology fields
    immune_subtype: str = ""        # Th1, Th2, Th17, Treg, Tfh, naive, effector, memory, M1, M2
    polarization_state: str = ""    # M1/M2 for macrophages, Th1/Th2/Th17 for CD4+ T cells
    cytokines_secreting: List[str] = field(default_factory=list)  # e.g. ["IFNg", "TNFa"]
    surface_markers: List[str] = field(default_factory=list)      # e.g. ["CD8", "CD45"]
    exhaustion_level: float = 0.0   # 0-1, T cell exhaustion (PD-1/LAG-3/TIM-3)

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
            "immune_subtype": self.immune_subtype,
            "polarization_state": self.polarization_state,
            "cytokines_secreting": self.cytokines_secreting,
            "surface_markers": self.surface_markers,
            "exhaustion_level": self.exhaustion_level,
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
        self.immune_subtype = props.get("immune_subtype", "")
        self.polarization_state = props.get("polarization_state", "")
        self.cytokines_secreting = props.get("cytokines_secreting", [])
        self.surface_markers = props.get("surface_markers", [])
        self.exhaustion_level = props.get("exhaustion_level", 0.0)


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


# ── Immunology Entity Types ──────────────────────────────────────────


@dataclass
class Virus(BioEntity):
    """A virus with genome type, tropism, and immune evasion data.

    Represents viral pathogens for immunology simulations including
    replication dynamics, host cell tropism, and immune evasion strategies.
    """
    entity_type: EntityType = EntityType.VIRUS
    virus_family: str = ""          # Coronaviridae, Orthomyxoviridae, Retroviridae
    genome_type: str = ""           # dsDNA, ssDNA, dsRNA, ssRNA+, ssRNA-, retro
    capsid_type: str = ""           # icosahedral, helical, complex
    envelope: bool = False          # Enveloped or non-enveloped
    genome_size_kb: float = 0.0     # Genome size in kilobases
    replication_rate: float = 0.0   # Virions per cell per hour
    host_tropism: List[str] = field(default_factory=list)  # ["epithelial", "T_cell"]
    target_receptors: List[str] = field(default_factory=list)  # ["ACE2", "CD4"]
    evasion_mechanisms: List[str] = field(default_factory=list)  # ["antigenic_drift", "MHC_downregulation"]
    incubation_hours: float = 0.0
    virulence_genes: List[str] = field(default_factory=list)
    taxonomy_id: str = ""           # NCBI Taxonomy ID

    def _extra_properties(self) -> dict:
        return {
            "virus_family": self.virus_family,
            "genome_type": self.genome_type,
            "capsid_type": self.capsid_type,
            "envelope": self.envelope,
            "genome_size_kb": self.genome_size_kb,
            "replication_rate": self.replication_rate,
            "host_tropism": self.host_tropism,
            "target_receptors": self.target_receptors,
            "evasion_mechanisms": self.evasion_mechanisms,
            "incubation_hours": self.incubation_hours,
            "virulence_genes": self.virulence_genes,
            "taxonomy_id": self.taxonomy_id,
        }

    def _apply_properties(self, props: dict):
        self.virus_family = props.get("virus_family", "")
        self.genome_type = props.get("genome_type", "")
        self.capsid_type = props.get("capsid_type", "")
        self.envelope = props.get("envelope", False)
        self.genome_size_kb = props.get("genome_size_kb", 0.0)
        self.replication_rate = props.get("replication_rate", 0.0)
        self.host_tropism = props.get("host_tropism", [])
        self.target_receptors = props.get("target_receptors", [])
        self.evasion_mechanisms = props.get("evasion_mechanisms", [])
        self.incubation_hours = props.get("incubation_hours", 0.0)
        self.virulence_genes = props.get("virulence_genes", [])
        self.taxonomy_id = props.get("taxonomy_id", "")


@dataclass
class Bacterium(BioEntity):
    """A bacterium with virulence factors and antibiotic resistance.

    Represents bacterial pathogens for immunology and infection simulations.
    """
    entity_type: EntityType = EntityType.BACTERIUM
    gram_stain: str = ""            # positive, negative
    shape: str = ""                 # rod (bacillus), coccus, spiral, filamentous
    oxygen_requirement: str = ""    # aerobic, anaerobic, facultative
    pathogenicity_factors: List[str] = field(default_factory=list)  # ["LPS", "type_III_secretion"]
    antibiotic_resistance: List[str] = field(default_factory=list)  # ["methicillin", "vancomycin"]
    growth_rate: float = 0.0        # Doublings per hour
    virulence_genes: List[str] = field(default_factory=list)
    toxins: List[str] = field(default_factory=list)  # ["exotoxin_A", "endotoxin"]
    host_niche: str = ""            # "intracellular", "extracellular", "mucosal"
    taxonomy_id: str = ""           # NCBI Taxonomy ID
    genome_size_mb: float = 0.0

    def _extra_properties(self) -> dict:
        return {
            "gram_stain": self.gram_stain,
            "shape": self.shape,
            "oxygen_requirement": self.oxygen_requirement,
            "pathogenicity_factors": self.pathogenicity_factors,
            "antibiotic_resistance": self.antibiotic_resistance,
            "growth_rate": self.growth_rate,
            "virulence_genes": self.virulence_genes,
            "toxins": self.toxins,
            "host_niche": self.host_niche,
            "taxonomy_id": self.taxonomy_id,
            "genome_size_mb": self.genome_size_mb,
        }

    def _apply_properties(self, props: dict):
        self.gram_stain = props.get("gram_stain", "")
        self.shape = props.get("shape", "")
        self.oxygen_requirement = props.get("oxygen_requirement", "")
        self.pathogenicity_factors = props.get("pathogenicity_factors", [])
        self.antibiotic_resistance = props.get("antibiotic_resistance", [])
        self.growth_rate = props.get("growth_rate", 0.0)
        self.virulence_genes = props.get("virulence_genes", [])
        self.toxins = props.get("toxins", [])
        self.host_niche = props.get("host_niche", "")
        self.taxonomy_id = props.get("taxonomy_id", "")
        self.genome_size_mb = props.get("genome_size_mb", 0.0)


@dataclass
class Antibody(BioEntity):
    """An antibody/immunoglobulin with isotype and binding properties.

    Represents all antibody isotypes (IgG1-4, IgA, IgE, IgM, IgD) with
    their effector functions, binding affinities, and producing cell types.
    """
    entity_type: EntityType = EntityType.ANTIBODY
    isotype: str = ""               # IgG1, IgG2, IgG3, IgG4, IgA1, IgA2, IgE, IgM, IgD
    heavy_chain: str = ""           # gamma1-4, alpha1-2, epsilon, mu, delta
    light_chain: str = ""           # kappa, lambda
    target_antigen: str = ""        # Name or ID of target antigen
    affinity_kd: float = 0.0        # Dissociation constant (M)
    fab_sequence: str = ""          # Variable region sequence (preview)
    fc_function: List[str] = field(default_factory=list)  # ["ADCC", "CDC", "opsonization", "neutralization"]
    producing_cell_type: str = ""   # "plasma_cell", "B_cell_memory"
    half_life_days: float = 0.0     # Serum half-life
    valency: int = 2                # Number of antigen-binding sites (IgM=10, IgG=2)
    complement_fixation: bool = False

    def _extra_properties(self) -> dict:
        return {
            "isotype": self.isotype,
            "heavy_chain": self.heavy_chain,
            "light_chain": self.light_chain,
            "target_antigen": self.target_antigen,
            "affinity_kd": self.affinity_kd,
            "fab_sequence": self.fab_sequence,
            "fc_function": self.fc_function,
            "producing_cell_type": self.producing_cell_type,
            "half_life_days": self.half_life_days,
            "valency": self.valency,
            "complement_fixation": self.complement_fixation,
        }

    def _apply_properties(self, props: dict):
        self.isotype = props.get("isotype", "")
        self.heavy_chain = props.get("heavy_chain", "")
        self.light_chain = props.get("light_chain", "")
        self.target_antigen = props.get("target_antigen", "")
        self.affinity_kd = props.get("affinity_kd", 0.0)
        self.fab_sequence = props.get("fab_sequence", "")
        self.fc_function = props.get("fc_function", [])
        self.producing_cell_type = props.get("producing_cell_type", "")
        self.half_life_days = props.get("half_life_days", 0.0)
        self.valency = props.get("valency", 2)
        self.complement_fixation = props.get("complement_fixation", False)


@dataclass
class Antigen(BioEntity):
    """An antigen — a molecule recognized by the adaptive immune system.

    Represents protein, lipid, or carbohydrate antigens with their epitopes,
    MHC restriction, and immunogenicity properties.
    """
    entity_type: EntityType = EntityType.ANTIGEN
    antigen_type: str = ""          # protein, lipid, carbohydrate, nucleic_acid
    epitope: str = ""               # Epitope sequence or structure description
    mhc_restriction: str = ""       # "MHC_I", "MHC_II", "both", "none" (lipids via CD1)
    source_organism: str = ""       # Organism or cell of origin
    immunogenicity: float = 0.0     # 0-1 how immunogenic
    cross_reactivity: List[str] = field(default_factory=list)  # Related antigens
    peptide_length: int = 0         # Length of presented peptide (8-10 for MHC-I, 13-25 for MHC-II)
    processing_pathway: str = ""    # "proteasome", "endosomal", "cross-presentation"
    t_cell_response: str = ""       # "CD8_cytotoxic", "CD4_helper", "both"
    b_cell_epitope: bool = False    # Can be recognized by BCR directly

    def _extra_properties(self) -> dict:
        return {
            "antigen_type": self.antigen_type,
            "epitope": self.epitope,
            "mhc_restriction": self.mhc_restriction,
            "source_organism": self.source_organism,
            "immunogenicity": self.immunogenicity,
            "cross_reactivity": self.cross_reactivity,
            "peptide_length": self.peptide_length,
            "processing_pathway": self.processing_pathway,
            "t_cell_response": self.t_cell_response,
            "b_cell_epitope": self.b_cell_epitope,
        }

    def _apply_properties(self, props: dict):
        self.antigen_type = props.get("antigen_type", "")
        self.epitope = props.get("epitope", "")
        self.mhc_restriction = props.get("mhc_restriction", "")
        self.source_organism = props.get("source_organism", "")
        self.immunogenicity = props.get("immunogenicity", 0.0)
        self.cross_reactivity = props.get("cross_reactivity", [])
        self.peptide_length = props.get("peptide_length", 0)
        self.processing_pathway = props.get("processing_pathway", "")
        self.t_cell_response = props.get("t_cell_response", "")
        self.b_cell_epitope = props.get("b_cell_epitope", False)


@dataclass
class Cytokine(BioEntity):
    """A cytokine — soluble signaling protein of the immune system.

    Represents interleukins, interferons, chemokines, TNF family members,
    and colony-stimulating factors with their receptor, source, and target info.
    """
    entity_type: EntityType = EntityType.CYTOKINE
    cytokine_family: str = ""       # interleukin, interferon, chemokine, TNF, CSF, TGF
    receptor: str = ""              # Primary receptor (e.g. "IL2RA/IL2RB/IL2RG")
    producing_cells: List[str] = field(default_factory=list)   # ["Th1", "macrophage_M1"]
    target_cells: List[str] = field(default_factory=list)      # ["T_cell", "NK_cell"]
    function: str = ""              # Primary biological function
    signaling_pathway: str = ""     # JAK-STAT, NF-kB, MAPK
    half_life_hours: float = 0.0    # Serum half-life
    pro_inflammatory: bool = True
    gene_symbol: str = ""           # Gene encoding this cytokine (e.g. "IL2", "IFNG")
    molecular_weight_kda: float = 0.0

    def _extra_properties(self) -> dict:
        return {
            "cytokine_family": self.cytokine_family,
            "receptor": self.receptor,
            "producing_cells": self.producing_cells,
            "target_cells": self.target_cells,
            "function": self.function,
            "signaling_pathway": self.signaling_pathway,
            "half_life_hours": self.half_life_hours,
            "pro_inflammatory": self.pro_inflammatory,
            "gene_symbol": self.gene_symbol,
            "molecular_weight_kda": self.molecular_weight_kda,
        }

    def _apply_properties(self, props: dict):
        self.cytokine_family = props.get("cytokine_family", "")
        self.receptor = props.get("receptor", "")
        self.producing_cells = props.get("producing_cells", [])
        self.target_cells = props.get("target_cells", [])
        self.function = props.get("function", "")
        self.signaling_pathway = props.get("signaling_pathway", "")
        self.half_life_hours = props.get("half_life_hours", 0.0)
        self.pro_inflammatory = props.get("pro_inflammatory", True)
        self.gene_symbol = props.get("gene_symbol", "")
        self.molecular_weight_kda = props.get("molecular_weight_kda", 0.0)


@dataclass
class PatternRecognitionReceptor(BioEntity):
    """A pattern recognition receptor (PRR) of the innate immune system.

    Represents TLRs, RLRs, CLRs, NLRs, and cGAS-STING sensors that detect
    PAMPs (pathogen-associated) and DAMPs (damage-associated) molecular patterns.
    """
    entity_type: EntityType = EntityType.PATTERN_RECOGNITION_RECEPTOR
    prr_type: str = ""              # TLR, RLR, CLR, NLR, cGAS_STING
    ligands: List[str] = field(default_factory=list)        # ["LPS", "dsRNA", "flagellin"]
    pamp_or_damp: str = ""          # "PAMP", "DAMP", "both"
    signaling_pathway: str = ""     # "MyD88-dependent", "TRIF-dependent", "inflammasome"
    cell_expression: List[str] = field(default_factory=list) # ["macrophage", "dendritic"]
    downstream_effectors: List[str] = field(default_factory=list)  # ["NF-kB", "IRF3", "IL1B"]
    subcellular_location: str = ""  # "cell_surface", "endosomal", "cytoplasmic"
    gene_symbol: str = ""           # e.g. "TLR4", "NLRP3", "DDX58"

    def _extra_properties(self) -> dict:
        return {
            "prr_type": self.prr_type,
            "ligands": self.ligands,
            "pamp_or_damp": self.pamp_or_damp,
            "signaling_pathway": self.signaling_pathway,
            "cell_expression": self.cell_expression,
            "downstream_effectors": self.downstream_effectors,
            "subcellular_location": self.subcellular_location,
            "gene_symbol": self.gene_symbol,
        }

    def _apply_properties(self, props: dict):
        self.prr_type = props.get("prr_type", "")
        self.ligands = props.get("ligands", [])
        self.pamp_or_damp = props.get("pamp_or_damp", "")
        self.signaling_pathway = props.get("signaling_pathway", "")
        self.cell_expression = props.get("cell_expression", [])
        self.downstream_effectors = props.get("downstream_effectors", [])
        self.subcellular_location = props.get("subcellular_location", "")
        self.gene_symbol = props.get("gene_symbol", "")


@dataclass
class ComplementComponent(BioEntity):
    """A complement system component for innate immunity.

    Represents proteins of the classical, alternative, and lectin complement
    pathways including their activation steps and effector functions.
    """
    entity_type: EntityType = EntityType.COMPLEMENT_COMPONENT
    pathway: str = ""               # classical, alternative, lectin, terminal
    activation_step: int = 0        # Order in cascade (1=initiator, 9=MAC)
    cleavage_products: List[str] = field(default_factory=list)  # ["C3a", "C3b"]
    function: str = ""              # "opsonization", "anaphylatoxin", "MAC_formation", "C3_convertase"
    deficiency_phenotype: str = ""  # Clinical consequence of deficiency
    gene_symbol: str = ""           # e.g. "C3", "CFB", "MBL2"
    serum_concentration_ug_ml: float = 0.0
    half_life_hours: float = 0.0

    def _extra_properties(self) -> dict:
        return {
            "pathway": self.pathway,
            "activation_step": self.activation_step,
            "cleavage_products": self.cleavage_products,
            "function": self.function,
            "deficiency_phenotype": self.deficiency_phenotype,
            "gene_symbol": self.gene_symbol,
            "serum_concentration_ug_ml": self.serum_concentration_ug_ml,
            "half_life_hours": self.half_life_hours,
        }

    def _apply_properties(self, props: dict):
        self.pathway = props.get("pathway", "")
        self.activation_step = props.get("activation_step", 0)
        self.cleavage_products = props.get("cleavage_products", [])
        self.function = props.get("function", "")
        self.deficiency_phenotype = props.get("deficiency_phenotype", "")
        self.gene_symbol = props.get("gene_symbol", "")
        self.serum_concentration_ug_ml = props.get("serum_concentration_ug_ml", 0.0)
        self.half_life_hours = props.get("half_life_hours", 0.0)


@dataclass
class MHCMolecule(BioEntity):
    """An MHC/HLA molecule for antigen presentation.

    Represents MHC class I and class II molecules with their peptide-binding
    properties, tissue distribution, and disease associations.
    """
    entity_type: EntityType = EntityType.MHC_MOLECULE
    mhc_class: str = ""             # "I", "II"
    hla_allele: str = ""            # "HLA-A*02:01", "HLA-DRB1*04:01"
    gene_symbol: str = ""           # "HLA-A", "HLA-DRA", "B2M"
    peptide_length_range: List[int] = field(default_factory=lambda: [8, 10])  # MHC-I: 8-10, MHC-II: 13-25
    tissue_distribution: List[str] = field(default_factory=list)  # ["all_nucleated" for MHC-I]
    presenting_cell_types: List[str] = field(default_factory=list)  # ["dendritic", "macrophage", "B_cell"]
    associated_diseases: List[str] = field(default_factory=list)
    binding_groove: str = ""        # Structural description
    population_frequency: float = 0.0  # Allele frequency in population

    def _extra_properties(self) -> dict:
        return {
            "mhc_class": self.mhc_class,
            "hla_allele": self.hla_allele,
            "gene_symbol": self.gene_symbol,
            "peptide_length_range": self.peptide_length_range,
            "tissue_distribution": self.tissue_distribution,
            "presenting_cell_types": self.presenting_cell_types,
            "associated_diseases": self.associated_diseases,
            "binding_groove": self.binding_groove,
            "population_frequency": self.population_frequency,
        }

    def _apply_properties(self, props: dict):
        self.mhc_class = props.get("mhc_class", "")
        self.hla_allele = props.get("hla_allele", "")
        self.gene_symbol = props.get("gene_symbol", "")
        self.peptide_length_range = props.get("peptide_length_range", [8, 10])
        self.tissue_distribution = props.get("tissue_distribution", [])
        self.presenting_cell_types = props.get("presenting_cell_types", [])
        self.associated_diseases = props.get("associated_diseases", [])
        self.binding_groove = props.get("binding_groove", "")
        self.population_frequency = props.get("population_frequency", 0.0)


# ── Adhesion Molecule ────────────────────────────────────────────────

@dataclass
class AdhesionMolecule(BioEntity):
    """Cell adhesion molecule involved in leukocyte-endothelial interactions and diapedesis."""
    entity_type: EntityType = EntityType.ADHESION_MOLECULE
    molecule_family: str = ""  # selectin, integrin, immunoglobulin_superfamily, cadherin, JAM
    expressed_on: List[str] = field(default_factory=list)  # ["endothelial", "leukocyte", "platelet"]
    ligands: List[str] = field(default_factory=list)  # binding partners
    binding_affinity_kd: float = 0.0  # Kd in μM
    on_rate: float = 0.0  # kon in μM⁻¹s⁻¹
    off_rate: float = 0.0  # koff in s⁻¹
    regulation: str = ""  # constitutive, cytokine_induced, chemokine_activated, histamine_induced
    signaling_pathway: str = ""
    diapedesis_step: str = ""  # rolling, activation, arrest, transmigration
    gene_symbol: str = ""
    structure_type: str = ""  # type_I_transmembrane, heterodimer, GPI_anchored, homophilic

    def _extra_properties(self) -> Dict[str, Any]:
        return {
            "molecule_family": self.molecule_family,
            "expressed_on": self.expressed_on,
            "ligands": self.ligands,
            "binding_affinity_kd": self.binding_affinity_kd,
            "on_rate": self.on_rate,
            "off_rate": self.off_rate,
            "regulation": self.regulation,
            "signaling_pathway": self.signaling_pathway,
            "diapedesis_step": self.diapedesis_step,
            "gene_symbol": self.gene_symbol,
            "structure_type": self.structure_type,
        }

    def _apply_properties(self, props: Dict[str, Any]):
        self.molecule_family = props.get("molecule_family", "")
        self.expressed_on = props.get("expressed_on", [])
        self.ligands = props.get("ligands", [])
        self.binding_affinity_kd = props.get("binding_affinity_kd", 0.0)
        self.on_rate = props.get("on_rate", 0.0)
        self.off_rate = props.get("off_rate", 0.0)
        self.regulation = props.get("regulation", "")
        self.signaling_pathway = props.get("signaling_pathway", "")
        self.diapedesis_step = props.get("diapedesis_step", "")
        self.gene_symbol = props.get("gene_symbol", "")
        self.structure_type = props.get("structure_type", "")


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
    # Researcher workflow types
    "simulation_run": SimulationRun,
    "research_project": ResearchProject,
    # Immunology entity types
    "virus": Virus,
    "bacterium": Bacterium,
    "antibody": Antibody,
    "antigen": Antigen,
    "cytokine": Cytokine,
    "prr": PatternRecognitionReceptor,
    "complement": ComplementComponent,
    "mhc": MHCMolecule,
    "adhesion_molecule": AdhesionMolecule,
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
