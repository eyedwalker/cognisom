"""
Bio-USD Python Schema Definitions
==================================

Python dataclass representations of the Bio-USD schema types.
These mirror the .usda schema definitions and are used by the
converter module to export simulation state to USD format.

Hierarchy:
    BioUnit (abstract)
    ├── BioCell
    │   ├── BioImmuneCell
    │   └── BioEndothelialCell
    ├── BioGene
    ├── BioProtein
    ├── BioMolecule
    ├── BioTissue
    ├── BioCapillary
    ├── BioSpatialField
    ├── BioExosome
    ├── BioAntibody
    ├── BioVirusParticle
    └── BioCytokineField

Applied API Schemas (composable metadata):
    BioMetabolicAPI      — oxygen, glucose, ATP, lactate per cell
    BioGeneExpressionAPI — expression levels, methylation, mutations
    BioImmuneAPI         — MHC-I, activation, detection radius
    BioEpigeneticAPI     — methylation, histone marks, chromatin state
    BioInteractionAPI    — binding relationships and affinities

Extensibility:
    New prim types can be registered at runtime using the prim_registry:

        from cognisom.biousd.schema import prim_registry, BioUnit

        @prim_registry.register("bio_virus")
        class BioVirusParticle(BioUnit):
            virus_type: str = ""
            capsid_proteins: List[str] = field(default_factory=list)
            genome_rna: str = ""
            ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Type

from cognisom.core.registry import Registry, registry_manager

log = logging.getLogger(__name__)


# ── Enums ─────────────────────────────────────────────────────────────

class CellType(str, Enum):
    NORMAL = "normal"
    CANCER = "cancer"
    IMMUNE = "immune"
    STROMAL = "stromal"
    ENDOTHELIAL = "endothelial"
    BASAL = "basal"
    LUMINAL = "luminal"
    NEUROENDOCRINE = "neuroendocrine"
    STEM = "stem"
    FIBROBLAST = "fibroblast"


class CellPhase(str, Enum):
    G0 = "G0"
    G1 = "G1"
    S = "S"
    G2 = "G2"
    M = "M"


class ImmuneCellType(str, Enum):
    # Legacy aliases (backward compat)
    T_CELL = "T_cell"
    NK_CELL = "NK_cell"
    MACROPHAGE = "macrophage"
    DENDRITIC = "dendritic"
    B_CELL = "B_cell"
    # T cell subtypes
    T_CELL_CD8_NAIVE = "T_cell_CD8_naive"
    T_CELL_CD8_EFFECTOR = "T_cell_CD8_effector"
    T_CELL_CD8_MEMORY = "T_cell_CD8_memory"
    T_CELL_CD4_TH1 = "T_cell_CD4_Th1"
    T_CELL_CD4_TH2 = "T_cell_CD4_Th2"
    T_CELL_CD4_TH17 = "T_cell_CD4_Th17"
    T_CELL_TREG = "T_cell_Treg"
    T_CELL_TFH = "T_cell_Tfh"
    T_CELL_GAMMA_DELTA = "T_cell_gamma_delta"
    # NK subtypes
    NKT_CELL = "NKT_cell"
    # Macrophage subtypes
    MACROPHAGE_M1 = "macrophage_M1"
    MACROPHAGE_M2 = "macrophage_M2"
    # Dendritic subtypes
    DENDRITIC_CONVENTIONAL = "dendritic_cDC"
    DENDRITIC_PLASMACYTOID = "dendritic_pDC"
    # B cell subtypes
    B_CELL_NAIVE = "B_cell_naive"
    B_CELL_MEMORY = "B_cell_memory"
    PLASMA_CELL = "plasma_cell"
    # Granulocytes
    NEUTROPHIL = "neutrophil"
    MAST_CELL = "mast_cell"
    BASOPHIL = "basophil"
    EOSINOPHIL = "eosinophil"
    # Innate lymphoid cells
    ILC1 = "ILC1"
    ILC2 = "ILC2"
    ILC3 = "ILC3"


class SpatialFieldType(str, Enum):
    OXYGEN = "oxygen"
    GLUCOSE = "glucose"
    CYTOKINE = "cytokine"
    LACTATE = "lactate"
    MORPHOGEN = "morphogen"


class GeneType(str, Enum):
    ONCOGENE = "oncogene"
    TUMOR_SUPPRESSOR = "tumor_suppressor"
    HOUSEKEEPING = "housekeeping"
    SIGNALING = "signaling"


# ── Prim Type Registry ───────────────────────────────────────────────
# Dynamic registry for Bio-USD prim types, enabling plugin-based extensibility.
# New prim types can be registered at runtime without modifying this file.

# Forward declarations - initialized after BioUnit is defined
prim_registry: Registry = None  # type: ignore
api_registry: Registry = None   # type: ignore


def _init_prim_registries():
    """Initialize the prim and API schema registries."""
    global prim_registry, api_registry

    prim_registry = Registry(
        name="prims",
        base_class=BioUnit,
        allow_override=False,
    )
    registry_manager.add_registry("prims", prim_registry)

    # API schemas don't have a common base class - use no base
    api_registry = Registry(
        name="api_schemas",
        base_class=None,
        allow_override=False,
    )
    registry_manager.add_registry("api_schemas", api_registry)

    return prim_registry, api_registry


def register_prim(
    name: str,
    version: str = "1.0.0",
    description: str = "",
    **metadata
):
    """
    Decorator to register a new Bio-USD prim type.

    Parameters
    ----------
    name : str
        Prim type name (e.g., "bio_virus", "bio_ctc")
    version : str
        Version string
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
    >>> from cognisom.biousd.schema import register_prim, BioUnit
    >>>
    >>> @register_prim("bio_virus", version="1.0.0")
    ... class BioVirusParticle(BioUnit):
    ...     virus_type: str = ""
    ...     genome_type: str = ""  # DNA, RNA
    """
    def decorator(cls: Type[BioUnit]) -> Type[BioUnit]:
        if prim_registry is not None:
            prim_registry.register_class(
                name, cls, version=version, description=description, **metadata
            )
        else:
            _DEFERRED_PRIM_REGISTRATIONS.append((name, cls, version, description, metadata))
        return cls
    return decorator


def register_api_schema(
    name: str,
    version: str = "1.0.0",
    description: str = "",
    **metadata
):
    """
    Decorator to register a new Bio-USD API schema.

    API schemas are composable metadata that can be applied to prim types.

    Parameters
    ----------
    name : str
        API schema name (e.g., "bio_chemotaxis_api")
    version : str
        Version string
    description : str
        Human-readable description

    Examples
    --------
    >>> @register_api_schema("bio_chemotaxis_api")
    ... @dataclass
    ... class BioChemotaxisAPI:
    ...     receptor_expression: float = 0.5
    ...     gradient_sensitivity: float = 1.0
    """
    def decorator(cls):
        if api_registry is not None:
            api_registry.register_class(
                name, cls, version=version, description=description, **metadata
            )
        else:
            _DEFERRED_API_REGISTRATIONS.append((name, cls, version, description, metadata))
        return cls
    return decorator


# Deferred registrations for prims/APIs defined before registry init
_DEFERRED_PRIM_REGISTRATIONS: List = []
_DEFERRED_API_REGISTRATIONS: List = []


def _register_deferred():
    """Register any prims/APIs that were decorated before registry init."""
    for name, cls, version, description, metadata in _DEFERRED_PRIM_REGISTRATIONS:
        prim_registry.register_class(
            name, cls, version=version, description=description, **metadata
        )
    _DEFERRED_PRIM_REGISTRATIONS.clear()

    for name, cls, version, description, metadata in _DEFERRED_API_REGISTRATIONS:
        api_registry.register_class(
            name, cls, version=version, description=description, **metadata
        )
    _DEFERRED_API_REGISTRATIONS.clear()


# ── Core Prim Types (IsA schemas) ────────────────────────────────────

@dataclass
class BioUnit:
    """Abstract base for all Bio-USD prims.

    Every biological entity in the scene graph inherits from BioUnit.
    Maps to: ``class BioUnit`` in the USDA schema.
    """
    prim_path: str = ""
    display_name: str = ""
    bio_version: str = "0.1.0"


@dataclass
class BioCell(BioUnit):
    """A single biological cell with spatial position and state.

    Maps to: ``class BioCell "Represents a single cell"``
    Inherits UsdGeomXformable for 3D positioning.

    Properties:
        cell_id: Unique integer identifier
        position: (x, y, z) in micrometres
        cell_type: Classification token
        phase: Cell cycle phase
        age: Age in simulation hours
        alive: Viability state
        volume: Relative cell volume (1.0 = normal)
        division_time: Hours between divisions
    """
    cell_id: int = 0
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    cell_type: CellType = CellType.NORMAL
    phase: CellPhase = CellPhase.G1
    age: float = 0.0
    alive: bool = True
    volume: float = 1.0
    division_time: float = 24.0

    # Applied APIs (composable)
    metabolic: Optional[BioMetabolicAPI] = None
    gene_expression: Optional[BioGeneExpressionAPI] = None
    epigenetic: Optional[BioEpigeneticAPI] = None


@dataclass
class BioImmuneCell(BioCell):
    """Immune cell with activation and targeting state.

    Maps to: ``class BioImmuneCell "Immune cell with activation"``
    Inherits BioCell.

    Properties:
        immune_type: T_cell, NK_cell, macrophage, etc.
        activated: Whether the cell is actively targeting
        target_cell_id: ID of the cell being targeted (-1 = none)
        detection_radius: Scanning radius in um
        kill_radius: Effective kill distance in um
        kill_probability: Per-step kill chance when in range
    """
    immune_type: ImmuneCellType = ImmuneCellType.T_CELL
    activated: bool = False
    target_cell_id: int = -1
    detection_radius: float = 10.0
    kill_radius: float = 5.0
    kill_probability: float = 0.8
    mhc1_expression: float = 1.0

    def __post_init__(self):
        self.cell_type = CellType.IMMUNE


@dataclass
class BioGene(BioUnit):
    """A gene with sequence and expression metadata.

    Maps to: ``class BioGene "Gene with expression state"``

    Properties:
        gene_name: Standard gene symbol (e.g. TP53, BRCA1)
        gene_type: oncogene, tumor_suppressor, etc.
        chromosome: Chromosome number or name
        sequence_length: Length in base pairs
        expression_level: Normalised expression (0-1)
        mutations: List of mutation identifiers
    """
    gene_name: str = ""
    gene_type: GeneType = GeneType.HOUSEKEEPING
    chromosome: str = ""
    sequence_length: int = 0
    expression_level: float = 0.5
    mutations: List[str] = field(default_factory=list)


@dataclass
class BioProtein(BioUnit):
    """A protein molecule with optional 3D structure.

    Maps to: ``class BioProtein "Protein with structure"``
    Inherits UsdGeomMesh when 3D structure is available.

    Properties:
        protein_name: Standard protein name
        gene_source: Gene that encodes this protein
        amino_acid_length: Number of residues
        pdb_id: PDB identifier if structure is known
        binding_affinity: Binding strength (arbitrary units)
        active_sites: List of residue positions
    """
    protein_name: str = ""
    gene_source: str = ""
    amino_acid_length: int = 0
    pdb_id: str = ""
    binding_affinity: float = 0.0
    active_sites: List[int] = field(default_factory=list)


@dataclass
class BioMolecule(BioUnit):
    """A small molecule (drug, metabolite, ligand).

    Maps to: ``class BioMolecule "Small molecule"``

    Properties:
        smiles: SMILES string representation
        inchi: InChI identifier
        molecular_weight: Daltons
        logp: Octanol-water partition coefficient
        charge: Net charge at physiological pH
    """
    smiles: str = ""
    inchi: str = ""
    molecular_weight: float = 0.0
    logp: float = 0.0
    charge: int = 0


@dataclass
class BioTissue(BioUnit):
    """A collection of cells forming tissue architecture.

    Maps to: ``class BioTissue "Tissue collection"``

    Properties:
        tissue_type: Tissue classification
        cell_ids: IDs of cells belonging to this tissue
        extent: Bounding box (min_x, min_y, min_z, max_x, max_y, max_z)
        cell_count: Number of cells
    """
    tissue_type: str = "unknown"
    cell_ids: List[int] = field(default_factory=list)
    extent: Tuple[float, ...] = (0.0, 0.0, 0.0, 100.0, 100.0, 50.0)
    cell_count: int = 0


@dataclass
class BioCapillary(BioUnit):
    """A blood vessel segment in the vascular network.

    Maps to: ``class BioCapillary "Blood vessel segment"``

    Properties:
        start: (x, y, z) start point in um
        end: (x, y, z) end point in um
        radius: Vessel radius in um
        oxygen_conc: O2 concentration (normalised 0-0.21)
        glucose_conc: Glucose concentration (mM)
        flow_rate: Blood flow rate (arbitrary units)
    """
    capillary_id: int = 0
    start: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    end: Tuple[float, float, float] = (100.0, 0.0, 0.0)
    radius: float = 5.0
    oxygen_conc: float = 0.21
    glucose_conc: float = 5.0
    flow_rate: float = 0.5


@dataclass
class BioSpatialField(BioUnit):
    """A 3D concentration field (oxygen, glucose, cytokine, etc.).

    Maps to: ``class BioSpatialField "3D concentration field"``

    Properties:
        field_type: oxygen, glucose, cytokine, etc.
        grid_shape: (nx, ny, nz) voxel dimensions
        voxel_size: Size of each voxel in um
        diffusion_coeff: Diffusion coefficient in um^2/s
        min_value: Minimum concentration in the field
        max_value: Maximum concentration in the field
        data_ref: File path to the raw float32 array data
    """
    field_type: SpatialFieldType = SpatialFieldType.OXYGEN
    grid_shape: Tuple[int, int, int] = (200, 200, 100)
    voxel_size: float = 10.0
    diffusion_coeff: float = 2000.0
    min_value: float = 0.0
    max_value: float = 0.21
    data_ref: str = ""  # path to binary array file


@dataclass
class BioExosome(BioUnit):
    """An extracellular vesicle carrying molecular cargo.

    Maps to: ``class BioExosome "Extracellular vesicle"``

    Properties:
        source_cell_id: Cell that released this exosome
        target_cell_id: Cell that will receive this exosome (-1 = unbound)
        position: Current (x, y, z) in um
        cargo_mrna: List of mRNA gene names carried
        cargo_mirna: List of miRNA identifiers carried
        cargo_proteins: List of protein names carried
    """
    source_cell_id: int = -1
    target_cell_id: int = -1
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    cargo_mrna: List[str] = field(default_factory=list)
    cargo_mirna: List[str] = field(default_factory=list)
    cargo_proteins: List[str] = field(default_factory=list)


# ── Immunology Prim Types ────────────────────────────────────────────


@dataclass
class BioAntibody(BioUnit):
    """An antibody molecule in the simulation scene.

    Represents IgG, IgM, IgA, IgE, IgD immunoglobulins with their
    binding state and target information for visualization.
    """
    isotype: str = "IgG1"           # IgG1-4, IgA, IgE, IgM, IgD
    target_antigen: str = ""
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bound: bool = False             # True if bound to target
    affinity_kd: float = 1e-9       # Dissociation constant (M)
    source_cell_id: int = -1        # Plasma cell that produced it


@dataclass
class BioVirusParticle(BioUnit):
    """A virus particle in the simulation scene.

    Represents individual virions for infection dynamics visualization.
    """
    virus_type: str = ""            # SARS-CoV-2, Influenza, HIV-1
    genome_type: str = "ssRNA+"     # dsDNA, ssDNA, dsRNA, ssRNA+, ssRNA-, retro
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    capsid_intact: bool = True
    attached_to_cell: int = -1      # Cell ID if attached (-1 = free)
    replication_stage: str = ""     # attachment, entry, replication, assembly, budding


@dataclass
class BioCytokineField(BioUnit):
    """A cytokine concentration field for immune signaling visualization.

    Represents the spatial distribution of a specific cytokine, enabling
    gradient-based chemotaxis and immune cell activation visualization.
    """
    cytokine_name: str = ""         # IL-2, IFNg, TNFa, etc.
    concentration: float = 0.0      # Local concentration (pg/mL)
    source_cell_ids: List[int] = field(default_factory=list)
    diffusion_coeff: float = 100.0  # um^2/s
    half_life_hours: float = 1.0
    grid_shape: Tuple[int, int, int] = (200, 200, 100)


@dataclass
class BioEndothelialCell(BioCell):
    """Endothelial cell lining blood vessel wall.

    Models the endothelial cell with adhesion molecule expression levels
    for diapedesis simulation. Tracks inflammation-driven upregulation
    of selectins (E/P-selectin) and Ig-superfamily ligands (ICAM-1, VCAM-1),
    as well as junction integrity for transmigration.
    """
    cell_subtype: str = ""                # postcapillary_venule, HEV, arterial, capillary
    e_selectin_expression: float = 0.0    # 0-1 normalized
    p_selectin_expression: float = 0.0
    icam1_expression: float = 0.0
    vcam1_expression: float = 0.0
    inflammation_state: float = 0.0       # 0=resting, 1=fully activated
    junction_integrity: float = 1.0       # 1=intact, 0=fully open

    def __post_init__(self):
        self.cell_type = CellType.ENDOTHELIAL


# ── Applied API Schemas ───────────────────────────────────────────────

@dataclass
class BioMetabolicAPI:
    """Metabolic state applied to a BioCell.

    Maps to: ``class "BioMetabolicAPI" (inherits = </APISchemaBase>)``

    Properties with biologically meaningful ranges:
        oxygen: Normalised O2 level (0-0.21, arterial=0.21, hypoxic<0.02)
        glucose: Glucose concentration (0-10 mM, threshold=0.5)
        atp: ATP energy units (0-2000)
        lactate: Lactate produced (0-20 mM, Warburg effect marker)
    """
    oxygen: float = 0.21
    glucose: float = 5.0
    atp: float = 1000.0
    lactate: float = 0.0


@dataclass
class BioGeneExpressionAPI:
    """Gene expression state applied to a BioCell.

    Maps to: ``class "BioGeneExpressionAPI" (inherits = </APISchemaBase>)``

    Properties:
        expression_level: Normalised overall expression (0-1)
        methylation_state: Global methylation level (0-1)
        active_genes: List of actively transcribed gene names
        mutations: List of mutation identifiers (e.g. 'TP53_R175H')
    """
    expression_level: float = 0.5
    methylation_state: float = 0.0
    active_genes: List[str] = field(default_factory=list)
    mutations: List[str] = field(default_factory=list)


@dataclass
class BioEpigeneticAPI:
    """Epigenetic state applied to a BioCell.

    Maps to: ``class "BioEpigeneticAPI" (inherits = </APISchemaBase>)``

    Histone marks and chromatin accessibility:
        methylation_level: CpG methylation (0-1)
        h3k4me3: Active promoter mark (0-1)
        h3k27me3: Polycomb repressive mark (0-1)
        h3k9ac: Active acetylation mark (0-1)
        chromatin_open: True = euchromatin, False = heterochromatin
    """
    methylation_level: float = 0.0
    h3k4me3: float = 1.0
    h3k27me3: float = 0.0
    h3k9ac: float = 1.0
    chromatin_open: bool = True


@dataclass
class BioImmuneAPI:
    """Immune surveillance state applied to an immune cell.

    Maps to: ``class "BioImmuneAPI" (inherits = </APISchemaBase>)``

    Properties:
        mhc1_expression: MHC class I surface expression (0-1)
        activation_state: Activation level (0-1)
        immune_cell_type: T_cell, NK_cell, macrophage, etc.
        detection_radius: Scanning radius in um
        kill_probability: Per-contact kill probability
    """
    mhc1_expression: float = 1.0
    activation_state: float = 0.0
    immune_cell_type: ImmuneCellType = ImmuneCellType.T_CELL
    detection_radius: float = 10.0
    kill_probability: float = 0.8


@dataclass
class BioInteractionAPI:
    """Molecular interaction metadata.

    Maps to: ``class "BioInteractionAPI" (inherits = </APISchemaBase>)``

    Used to annotate protein-protein, protein-ligand, and
    receptor-ligand binding relationships.

    Properties:
        binds_to: Prim path of the binding partner
        binding_affinity: Kd in nM (lower = stronger)
        interaction_type: binding, inhibition, activation, etc.
    """
    binds_to: str = ""
    binding_affinity: float = 0.0
    interaction_type: str = "binding"


@dataclass
class BioPrecisePointAPI:
    """High-precision coordinate storage for simulation-grade accuracy.

    Maps to: ``class "BioPrecisePointAPI" (inherits = </APISchemaBase>)``

    This API schema enables storing double-precision (float64) coordinates
    alongside the standard float32 render coordinates. Essential for:
    - Molecular dynamics with sub-angstrom precision
    - Universe-scale visualization (11 orders of magnitude)
    - Thermodynamic analysis requiring velocity data

    The visualization uses float32 points for rendering performance,
    while the simulation uses float64 precise:positions for accuracy.

    Properties:
        precise_positions: Double-precision atomic/particle coordinates
        precise_velocities: High-fidelity velocity vectors
        precise_masses: Atomic masses for momentum calculations
        precision_scale: Scale factor applied to coordinates
        reference_origin: Origin point for local coordinate system
    """
    precise_positions: List[Tuple[float, float, float]] = field(default_factory=list)
    precise_velocities: List[Tuple[float, float, float]] = field(default_factory=list)
    precise_masses: List[float] = field(default_factory=list)
    precision_scale: float = 1.0
    reference_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class BioSimulationBridgeAPI:
    """Bridge between simulation and visualization precision levels.

    Maps to: ``class "BioSimulationBridgeAPI" (inherits = </APISchemaBase>)``

    Manages the mixed-precision approach where:
    - Position integration uses float64 (prevents drift)
    - Force calculation can use float32 (local interactions)
    - Rendering uses float32 (GPU optimization)

    Properties:
        position_precision: Precision for position integration ('float64' or 'float32')
        force_precision: Precision for force calculations
        render_precision: Precision for visualization
        last_sync_time: Last time simulation synced to visualization
        drift_correction: Accumulated position drift correction
    """
    position_precision: str = "float64"
    force_precision: str = "float32"
    render_precision: str = "float32"
    last_sync_time: float = 0.0
    drift_correction: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class BiologicalScaleAPI:
    """Scale information for multi-scale visualization.

    Maps to: ``class "BiologicalScaleAPI" (inherits = </APISchemaBase>)``

    Defines the biological scale level for semantic zooming:
    - ATOMIC (1e-10m): Bond lengths, atomic positions
    - MOLECULAR (1e-9m): Protein structures, small molecules
    - ORGANELLE (1e-7m): Mitochondria, nuclei, vesicles
    - CELLULAR (1e-5m): Whole cells
    - TISSUE (1e-3m): Cell aggregates, tissues
    - ORGAN (1e-1m): Organs, organ systems
    - ORGANISM (1m): Whole organisms

    Properties:
        scale_level: Biological scale level name
        typical_size_meters: Typical entity size at this scale
        coordinate_precision: Required precision for this scale
        representation_type: Default representation (mesh, instancer, etc.)
    """
    scale_level: str = "CELLULAR"
    typical_size_meters: float = 1e-5
    coordinate_precision: str = "float64"
    representation_type: str = "mesh"


# ── Scene Graph Helper ────────────────────────────────────────────────

@dataclass
class BioScene:
    """Top-level Bio-USD scene containing all biological prims.

    Represents a complete simulation snapshot that can be serialised
    to a .usda file.

    Properties:
        simulation_time: Current simulation time in hours
        time_step: Time step size in hours
        step_count: Number of simulation steps completed
        cells: All cells in the scene
        immune_cells: Immune cells (also in cells list)
        genes: Gene catalogue
        proteins: Protein catalogue
        molecules: Small molecule catalogue
        tissues: Tissue collections
        capillaries: Vascular network segments
        spatial_fields: 3D concentration fields
        exosomes: Extracellular vesicles in transit
    """
    simulation_time: float = 0.0
    time_step: float = 0.1
    step_count: int = 0
    cells: List[BioCell] = field(default_factory=list)
    immune_cells: List[BioImmuneCell] = field(default_factory=list)
    genes: List[BioGene] = field(default_factory=list)
    proteins: List[BioProtein] = field(default_factory=list)
    molecules: List[BioMolecule] = field(default_factory=list)
    tissues: List[BioTissue] = field(default_factory=list)
    capillaries: List[BioCapillary] = field(default_factory=list)
    spatial_fields: List[BioSpatialField] = field(default_factory=list)
    exosomes: List[BioExosome] = field(default_factory=list)

    @property
    def total_cells(self) -> int:
        return len(self.cells) + len(self.immune_cells)

    @property
    def alive_cells(self) -> int:
        alive = sum(1 for c in self.cells if c.alive)
        alive += sum(1 for c in self.immune_cells if c.alive)
        return alive


# ── Initialize Registries ────────────────────────────────────────────
# Initialize registries and register all built-in prim and API types.

# Built-in prim types to register
_BUILTIN_PRIMS = {
    "bio_unit": BioUnit,
    "bio_cell": BioCell,
    "bio_immune_cell": BioImmuneCell,
    "bio_gene": BioGene,
    "bio_protein": BioProtein,
    "bio_molecule": BioMolecule,
    "bio_tissue": BioTissue,
    "bio_capillary": BioCapillary,
    "bio_spatial_field": BioSpatialField,
    "bio_exosome": BioExosome,
    # Immunology prim types
    "bio_antibody": BioAntibody,
    "bio_virus_particle": BioVirusParticle,
    "bio_cytokine_field": BioCytokineField,
    # Diapedesis prim types
    "bio_endothelial_cell": BioEndothelialCell,
}

# Built-in API schemas to register
_BUILTIN_APIS = {
    "bio_metabolic_api": BioMetabolicAPI,
    "bio_gene_expression_api": BioGeneExpressionAPI,
    "bio_epigenetic_api": BioEpigeneticAPI,
    "bio_immune_api": BioImmuneAPI,
    "bio_interaction_api": BioInteractionAPI,
    # Precision-related APIs (Phase B5)
    "bio_precise_point_api": BioPrecisePointAPI,
    "bio_simulation_bridge_api": BioSimulationBridgeAPI,
    "biological_scale_api": BiologicalScaleAPI,
}


def _bootstrap_prim_registries():
    """Bootstrap the prim and API registries with all built-in types."""
    _init_prim_registries()

    # Register built-in prim types
    for name, cls in _BUILTIN_PRIMS.items():
        prim_registry.register_class(
            name,
            cls,
            version="1.0.0",
            tags=["builtin"],
            description=cls.__doc__.strip().split("\n")[0] if cls.__doc__ else "",
        )

    # Register built-in API schemas
    for name, cls in _BUILTIN_APIS.items():
        api_registry.register_class(
            name,
            cls,
            version="1.0.0",
            tags=["builtin"],
            description=cls.__doc__.strip().split("\n")[0] if cls.__doc__ else "",
        )

    # Register any deferred types (from decorators applied before init)
    _register_deferred()

    log.debug(f"Prim registry initialized with {len(prim_registry)} types")
    log.debug(f"API registry initialized with {len(api_registry)} schemas")


# Bootstrap on module load
_bootstrap_prim_registries()


# ── Public API ───────────────────────────────────────────────────────

def get_prim_class(prim_type: str) -> Type[BioUnit]:
    """
    Get the prim class for a given type name.

    Parameters
    ----------
    prim_type : str
        Prim type name (e.g., "bio_cell", "bio_virus")

    Returns
    -------
    Type[BioUnit]
        The prim class

    Raises
    ------
    KeyError
        If prim type is not registered
    """
    if prim_type in prim_registry:
        return prim_registry.get(prim_type)
    raise KeyError(f"Unknown prim type: {prim_type}. "
                   f"Available: {prim_registry.list_names()}")


def get_api_class(api_name: str):
    """
    Get an API schema class by name.

    Parameters
    ----------
    api_name : str
        API schema name (e.g., "bio_metabolic_api")

    Returns
    -------
    Type
        The API schema class
    """
    if api_name in api_registry:
        return api_registry.get(api_name)
    raise KeyError(f"Unknown API schema: {api_name}. "
                   f"Available: {api_registry.list_names()}")


def list_prim_types() -> List[str]:
    """List all registered prim types."""
    return prim_registry.list_names()


def list_api_schemas() -> List[str]:
    """List all registered API schemas."""
    return api_registry.list_names()


def create_prim(prim_type: str, **kwargs) -> BioUnit:
    """
    Create a prim instance by type name.

    Parameters
    ----------
    prim_type : str
        Prim type name (e.g., "bio_cell", "bio_virus")
    **kwargs
        Arguments passed to the prim constructor

    Returns
    -------
    BioUnit
        New prim instance
    """
    return prim_registry.create(prim_type, **kwargs)
