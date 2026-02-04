# Cognisom Strategic Implementation Plan
## From Bio-Simulation Platform to Full Bio-Digital Twin

**Version**: 2.0
**Date**: February 2026
**Status**: Strategic Roadmap

---

## Executive Summary

This document presents a comprehensive analysis of Cognisom's current state versus the strategic vision outlined in the "Architectural Viability and Strategic Roadmap" research document, along with a detailed phased implementation plan.

### Key Finding

**Cognisom is a well-architected biochemistry simulator with GPU acceleration, but it is NOT yet a physics simulator or a complete Bio-Digital Twin.**

| Dimension | Current State | Vision Requirement | Gap |
|-----------|---------------|-------------------|-----|
| Stochastic Chemistry | ✅ 90% complete | SSA, FBA, kinetics | Minimal |
| Bio-USD Schema | ✅ 70% complete | Cell/tissue level | Major (no atomic) |
| GPU Acceleration | ✅ 85% complete | CUDA kernels | Minimal |
| Physics/Mechanics | ❌ 5% complete | Warp/Newton | Critical |
| AI/LLM Integration | ⚠️ 40% complete | ESM-3, Boltz, Cosmos | Moderate |
| Lab-in-the-Loop | ❌ 0% complete | Robotics + validation | Critical |
| **Extensibility** | ⚠️ **40% complete** | **Plugin architecture** | **Moderate** |

---

## Part 1: Current State Assessment

### 1.1 What We've Built (Real, Production-Ready)

| Module | LOC | Status | Description |
|--------|-----|--------|-------------|
| **GPU Backend** | 2,280 | Production | CuPy/NumPy abstraction, auto-fallback |
| **SSA Kernel** | 350 | Production | GPU Gillespie + tau-leaping, 7-gene model |
| **Domain Decomposition** | 230 | Production | Multi-GPU slab partitioning |
| **Diffusion Solver** | 200 | Production | 3D Laplacian with CFL stability |
| **Spatial Operations** | 400 | Production | N-body immune detection |
| **Bio-USD Schema** | 450 | Production | 9 prim types, 5 API schemas |
| **Bio-USD Converter** | 700 | Production | Simulation → USDA export |
| **Bio-USD Validator** | 380 | Production | 11 validation checks, CI gate |
| **Bidirectional Sync** | 440 | Production | Engine ↔ USD state sync |
| **Entity Library** | 1,600 | Production | SQLite + FTS5, 15 entity types |
| **Research Agent** | 500+ | Production | Multi-tool orchestrator |
| **NIM Integrations** | 1,500+ | Partial | 12 NIMs (API clients, some stubbed) |
| **Dashboard** | 46K+ | Production | 14 pages, full auth, multi-tenant |
| **Intracellular Model** | 800 | Production | Genome, transcriptome, proteome |
| **Membrane Receptors** | 600 | Production | Kinetic model, 5 pathways |

**Total Production Code**: ~42,000 lines of Python

### 1.2 What's Stubbed or Missing

| Component | Status | Impact |
|-----------|--------|--------|
| **Atomic-level Bio-USD** | Missing | Cannot represent PDB structures |
| **Force Field Parameters** | Missing | Cannot do molecular dynamics |
| **Neural Force Fields** | Missing | Cannot do differentiable physics |
| **Cell Mechanics** | Missing | Cells don't collide or deform |
| **Fluid Dynamics** | Missing | No blood flow, no advection |
| **Membrane Geometry** | Missing | Receptors are abstract, no 3D membrane |
| **Cryo-ET Integration** | Missing | Cannot import tomograms |
| **Warp Integration** | Missing | No autodiff through physics |
| **Newton Integration** | Missing | No soft-body/collision solver |
| **Lab Robotics** | Missing | No physical automation |

### 1.3 Architectural Strengths (Keep As-Is)

1. **Module Registry Pattern** — Clean plugin architecture for simulation modules
2. **GPU Backend Abstraction** — Transparent CPU/GPU switching
3. **SoA Memory Layout** — Optimal for GPU parallelism
4. **EventBus Architecture** — Decoupled module communication
5. **Bio-USD Schema Design** — Extensible, USD-compliant foundation
6. **Multi-Tenant Auth** — Production security infrastructure
7. **Dashboard Framework** — Streamlit-based, easily extensible

### 1.4 Extensibility Assessment

**Current extensibility rating: 6/10 (Moderate)**

The platform was built with some extensibility in mind, but relies too heavily on hardcoded enums and static registries. For Cognisom to become a true framework expandable to other cells, tissues, organs, and organisms, architectural improvements are needed.

| Layer | Current State | Plugin-Ready? | Effort to Add New Component |
|-------|--------------|---------------|----------------------------|
| **Entity Types** | Enum-based, hardcoded map | ❌ No | 4 hours (requires code edits) |
| **Bio-USD Schema** | Dataclass-based, hardcoded BioScene | ❌ No | 3-4 hours (requires code edits) |
| **Simulation Modules** | Abstract base class, manual registration | ✅ Partial | 15 hours (module dev) + 1 hour registration |
| **GPU Kernels** | Vectorized with hardcoded types | ❌ No | 6-10 hours (requires kernel dev) |
| **Dashboard UI** | Hardcoded checks + CSS | ⚠️ Partial | 2-3 hours (styling) |

**Key Bottlenecks**:
- `EntityType` enum must be modified to add new entity types
- `BioScene` has hardcoded lists for each prim type
- GPU kernels assume specific property names (`oxygen`, `glucose`, `atp`, etc.)
- Dashboard module checkboxes and type colors are hardcoded

**Effort to add a complete new biological component** (e.g., BioVirus, BioNeuron):
- Current architecture: **35-46 hours** (~1-1.5 weeks)
- After extensibility refactoring: **15-20 hours** (mostly biological modeling)

---

## Part 2: Gap Analysis vs. Strategic Vision

### 2.1 Bio-USD Schema Gaps

The research document specifies a "BioUSD" standard. Here's where our schema stands:

| Schema Type | Current | Needed | Priority |
|-------------|---------|--------|----------|
| **Cell-level prims** | ✅ Complete | BioCell, BioImmuneCell | - |
| **Gene/Protein prims** | ✅ Complete | BioGene, BioProtein | - |
| **Spatial fields** | ✅ Complete | BioSpatialField | - |
| **Atomic coordinates** | ❌ Missing | BioAtom, BioAtomicStructure | P1 |
| **Force field API** | ❌ Missing | BioForceFieldAPI | P1 |
| **DNA/RNA 3D structure** | ❌ Missing | BioDNAStructure, BioRNAStructure | P2 |
| **Lipid membranes** | ❌ Missing | BioMembraneLayer, BioLipid | P2 |
| **Cryo-ET volumes** | ❌ Missing | BioCryoETTomogram | P3 |
| **Multiscale hierarchy** | ⚠️ Partial | BioComplex, BioOrganelle, BioOrgan | P2 |

### 2.2 Physics Engine Gaps

The research document specifies NVIDIA Warp (micro-scale) + Newton (macro-scale):

| Physics Capability | Current | Needed | Priority |
|-------------------|---------|--------|----------|
| **Stochastic kinetics** | ✅ SSA, tau-leaping | Done | - |
| **FBA metabolism** | ⚠️ Approx GPU | cuOpt LP solver | P2 |
| **Diffusion PDE** | ✅ Laplacian | Done | - |
| **Molecular dynamics** | ❌ Missing | Warp kernels | P1 |
| **Neural force fields** | ❌ Missing | cuEquivariance | P2 |
| **Cell-cell collisions** | ❌ Missing | Newton contact | P1 |
| **Soft-body membranes** | ❌ Missing | Newton VBD | P2 |
| **Fluid flow** | ❌ Missing | PhysX/Flow | P3 |
| **Autodifferentiation** | ❌ Missing | Warp autodiff | P1 |

### 2.3 AI/LLM Integration Gaps

| Model | Current | Needed | Priority |
|-------|---------|--------|----------|
| **ESM-2** | ✅ NIM client | Sequence embeddings | Done |
| **ESM-3** | ❌ Missing | Multimodal protein generation | P1 |
| **Boltz-1/2** | ✅ NIM client | Complex structure prediction | Done |
| **AlphaFold2** | ✅ NIM client | Single protein structure | Done |
| **DiffDock** | ✅ NIM client | Molecular docking | Done |
| **MolMIM** | ⚠️ Stubbed | Small molecule generation | P2 |
| **Cosmos Reason** | ❌ Missing | Physical reasoning VLM | P2 |
| **Geneformer** | ❌ Missing | Transcriptomic prediction | P3 |

### 2.4 Lab-in-the-Loop Gaps

| Component | Current | Needed | Priority |
|-----------|---------|--------|----------|
| **Lab robotics connector** | ❌ Missing | Automata LINQ, OpenTrons | P3 |
| **Experiment queue** | ❌ Missing | Batch job scheduling | P3 |
| **Validation loop** | ❌ Missing | Simulation → experiment → feedback | P3 |
| **Equipment digital twin** | ❌ Missing | Lab asset models | P4 |

---

## Part 3: Components to Rework vs. Extend

### 3.1 Keep As-Is (No Changes)

- `cognisom/gpu/backend.py` — Solid abstraction
- `cognisom/gpu/spatial_ops.py` — N-body queries work well
- `cognisom/gpu/diffusion.py` — Correct PDE solver
- `cognisom/biousd/schema.py` — Foundation is good (extend, don't replace)
- `cognisom/biousd/converter.py` — Works for current schema
- `cognisom/library/` — Entity store is production-ready
- `cognisom/auth/` — Security infrastructure complete
- `cognisom/dashboard/` — UI framework is solid

### 3.2 Extend (Add New Capabilities)

| Module | Extension Needed |
|--------|------------------|
| `cognisom/biousd/schema.py` | Add atomic, force field, membrane prim types |
| `cognisom/gpu/ssa_kernel.py` | Port to Warp for autodiff |
| `cognisom/gpu/fba_solver.py` | Replace with cuOpt LP or Warp solver |
| `cognisom/nim/` | Add ESM-3, Cosmos Reason, Geneformer clients |
| `cognisom/engine/py/` | Add physics modules (mechanics, fluid) |

### 3.3 Rework (Significant Changes)

| Module | Reason | Approach |
|--------|--------|----------|
| `cognisom/engine/py/cell.py` | Uses ODE, not SSA | Replace with SSA-based cell model |
| `cognisom/engine/py/membrane/` | Kinetic-only, no 3D | Add geometric membrane with Warp |
| None of the GPU kernels | Need autodiff | Port to Warp (not CuPy) |

### 3.4 New Modules Required

| New Module | Purpose |
|------------|---------|
| `cognisom/physics/warp_kernels.py` | Warp-based physics kernels |
| `cognisom/physics/newton_bridge.py` | Newton solver integration |
| `cognisom/physics/cell_mechanics.py` | Soft-sphere collisions, adhesion |
| `cognisom/physics/membrane_solver.py` | Lipid bilayer dynamics |
| `cognisom/biousd/atomic_schema.py` | Atomic-level prim definitions |
| `cognisom/biousd/forcefield_schema.py` | Force field parameters |
| `cognisom/biousd/cryoet_schema.py` | Cryo-ET tomogram support |
| `cognisom/agent/cosmos_agent.py` | Cosmos Reason VLM agent |
| `cognisom/agent/esm3_generator.py` | ESM-3 protein generation |
| `cognisom/lab/robotics_connector.py` | Lab automation interface |

---

## Part 4: Phased Implementation Plan

### Overview: 6 Phases Over 20 Months

| Phase | Duration | Focus | Key Deliverable |
|-------|----------|-------|-----------------|
| **Phase 0** | 2 months | Extensibility Framework | Plugin architecture for all layers |
| **Phase A** | 3 months | Physics Foundation | Warp integration, cell mechanics |
| **Phase B** | 3 months | Atomic Bio-USD | Full molecular representation |
| **Phase C** | 4 months | AI Intelligence Layer | ESM-3, Cosmos Reason, virtual scientist |
| **Phase D** | 4 months | Multi-Scale Coupling | Atom → cell → tissue dynamics |
| **Phase E** | 4 months | Lab-in-the-Loop | Robotics, validation, AOUSD submission |

---

## Phase 0: Extensibility Framework (Months 1-2)

### Goal
Transform Cognisom from a hardcoded platform into a true plugin-based framework that can easily expand to new cells, tissues, organs, and organisms without modifying core code.

### Why This Comes First
Every subsequent phase (Physics, Atomic Bio-USD, AI, Multi-Scale, Lab-in-the-Loop) will add new entity types, prim types, simulation modules, and GPU kernels. Without the extensibility framework, each addition requires touching 5-6 files across the codebase. With the framework, new components are self-contained packages.

### 0.1 Entity Type Registry (Week 1-2)

**Current Problem**:
- `EntityType` enum with 13 hardcoded values
- Static `ENTITY_CLASS_MAP` dictionary
- Adding a new entity type requires editing `models.py`

**Solution**: Dynamic registry pattern

**Tasks**:
1. Replace `EntityType` enum with `EntityTypeRegistry` class
2. Implement `registry.register(type_name, entity_class)` for runtime registration
3. Add decorator `@register_entity("virus")` for declarative registration
4. Migrate existing 13 entity types to new registry
5. Add type validation and schema versioning
6. Update serialization to use registry lookup

**Files Modified**:
- `cognisom/library/models.py` — Replace enum with registry
- `cognisom/library/store.py` — Use registry for deserialization

**Files Created**:
- `cognisom/core/registry.py` — Generic registry base class

**API After Change**:
```python
# Before (hardcoded):
class EntityType(str, Enum):
    GENE = "gene"
    VIRUS = "virus"  # Must edit this file!

# After (dynamic):
from cognisom.core.registry import entity_registry

@entity_registry.register("virus")
class Virus(BioEntity):
    ...  # Self-contained, no core edits needed
```

**Success Criteria**:
- All existing entity types work unchanged
- New entity types can be added from external packages
- Zero edits to `models.py` required to add new types

### 0.2 Bio-USD Prim Registry (Week 3-4)

**Current Problem**:
- `BioScene` has hardcoded lists: `cells`, `immune_cells`, `genes`, etc.
- Adding a prim type requires editing `schema.py` AND `converter.py`
- Enums (`CellType`, `SpatialFieldType`) are not extensible

**Solution**: Protocol-based prim registry with auto-discovery

**Tasks**:
1. Create `BioPrim` protocol defining required interface
2. Implement `PrimRegistry` with `register_prim(prim_class)`
3. Make `BioScene` dynamically discover registered prim types
4. Replace hardcoded enums with extensible registries
5. Update converter to iterate registered prim types
6. Update validator to validate any registered prim

**Files Modified**:
- `cognisom/biousd/schema.py` — Add registry, make BioScene dynamic
- `cognisom/biousd/converter.py` — Use prim registry for export
- `cognisom/biousd/validator.py` — Validate registered prims

**Files Created**:
- `cognisom/biousd/prim_registry.py` — Prim type registration

**API After Change**:
```python
# Before (hardcoded):
@dataclass
class BioScene:
    cells: List[BioCell]
    viruses: List[BioVirus]  # Must edit schema.py!

# After (dynamic):
from cognisom.biousd.prim_registry import prim_registry

@prim_registry.register
class BioVirus(BioUnit):
    ...  # Auto-discovered, no schema.py edits

scene = BioScene()
scene.get_prims("virus")  # Works automatically
```

**Success Criteria**:
- All existing prims work unchanged
- New prim types auto-register via decorator
- Converter/validator handle new prims without modification

### 0.3 Simulation Module Auto-Discovery (Week 5)

**Current Problem**:
- Modules require manual `engine.register_module()` calls
- Dashboard has hardcoded module checkboxes
- No plugin directory convention

**Solution**: Entry-point based auto-discovery

**Tasks**:
1. Define module entry point in `pyproject.toml`
2. Implement `ModuleDiscovery` to scan entry points at startup
3. Add plugin directory (`cognisom/plugins/`) for local modules
4. Update dashboard to dynamically list discovered modules
5. Add module metadata (name, description, category) for UI

**Files Modified**:
- `cognisom/core/simulation_engine.py` — Auto-discover modules
- `cognisom/dashboard/pages/3_simulation.py` — Dynamic checkboxes
- `pyproject.toml` — Define entry points

**Files Created**:
- `cognisom/core/module_discovery.py` — Entry point scanner

**API After Change**:
```python
# Before (manual registration):
engine.register_module("viral", ViralModule, config)

# After (auto-discovery via entry points):
# In external package's pyproject.toml:
[project.entry-points."cognisom.modules"]
viral = "my_plugin:ViralModule"

# Module auto-loads at engine startup
```

**Success Criteria**:
- Existing modules load without changes
- External packages can provide modules via entry points
- Dashboard shows all discovered modules

### 0.4 GPU Physics Model Interface (Week 6-7)

**Current Problem**:
- Kernels hardcode type constants: `TYPE_NORMAL=0, TYPE_CANCER=1, TYPE_IMMUNE=2`
- Property access assumes specific names: `oxygen`, `glucose`, `atp`
- Adding particle type requires editing kernel code

**Solution**: PhysicsModel interface with kernel composition

**Tasks**:
1. Define `PhysicsModel` protocol (update rules, property requirements)
2. Create property schema registry for cell/particle types
3. Implement kernel composer that generates code from models
4. Migrate metabolism, death detection to PhysicsModel interface
5. Add type code registry for GPU kernels

**Files Modified**:
- `cognisom/gpu/cell_ops.py` — Use physics model interface
- `cognisom/gpu/ssa_kernel.py` — Parameterize type handling

**Files Created**:
- `cognisom/gpu/physics_model.py` — PhysicsModel protocol
- `cognisom/gpu/property_schema.py` — Property definitions
- `cognisom/gpu/kernel_composer.py` — Dynamic kernel generation

**API After Change**:
```python
# Before (hardcoded):
TYPE_VIRUS = 3  # Must edit cell_ops.py!
# ... and update every kernel function

# After (declarative):
@physics_registry.register("virus")
class ViralPhysics(PhysicsModel):
    properties = ["envelope_integrity", "replication_stage"]

    def update_rule(self, state, dt):
        ...  # Self-contained physics
```

**Success Criteria**:
- Existing cell types work unchanged
- New particle types define their own physics
- Zero kernel edits for new types

### 0.5 Dashboard Plugin System (Week 8)

**Current Problem**:
- Type colors hardcoded in CSS
- Visualization logic has type-specific if/elif branches
- Module checkboxes are static

**Solution**: Dashboard component registry

**Tasks**:
1. Create `UIComponentRegistry` for type-specific renderers
2. Implement color scheme registry (auto-generate or explicit)
3. Add visualization plugin interface for 3D view
4. Make module controls data-driven from module metadata
5. Support dashboard page plugins from external packages

**Files Modified**:
- `cognisom/dashboard/pages/14_entity_library.py` — Use component registry
- `cognisom/dashboard/pages/12_3d_visualization.py` — Plugin renderers
- `cognisom/dashboard/pages/3_simulation.py` — Dynamic module UI

**Files Created**:
- `cognisom/dashboard/component_registry.py` — UI component plugins
- `cognisom/dashboard/visualization_plugins.py` — 3D renderer interface

**API After Change**:
```python
# Before (hardcoded CSS):
.type-virus { background-color: #???; }  # Must edit CSS!

# After (registered):
@ui_registry.register("virus")
class VirusUI:
    color = "#8B0000"
    icon = "🦠"

    def render_card(self, entity):
        ...  # Custom rendering
```

**Success Criteria**:
- New entity types get automatic UI without CSS edits
- Custom visualizations can be plugged in
- Dashboard pages can be added from external packages

### 0.6 Integration & Documentation (Week 8)

**Tasks**:
1. Create "Building a Plugin" developer guide
2. Build example plugin package: `cognisom-virus-plugin`
3. Add plugin template generator CLI: `cognisom create-plugin`
4. Write migration guide for existing custom code
5. Update all tests for registry-based architecture

**Files Created**:
- `docs/developer/building-plugins.md`
- `examples/virus_plugin/` — Complete example plugin
- `cognisom/cli/create_plugin.py` — Template generator

**Success Criteria**:
- Developer can create new biological component in < 4 hours
- Example plugin demonstrates all extension points
- Existing tests pass with registry architecture

### Phase 0 Summary

| Component | Before | After | Effort |
|-----------|--------|-------|--------|
| Entity Types | Hardcoded enum | Dynamic registry | 8-10 hrs |
| Bio-USD Prims | Hardcoded BioScene | Protocol + auto-discovery | 12-15 hrs |
| Simulation Modules | Manual registration | Entry-point auto-discovery | 6-8 hrs |
| GPU Kernels | Hardcoded types | PhysicsModel interface | 15-20 hrs |
| Dashboard UI | Hardcoded CSS/logic | Component registry | 10-12 hrs |
| **Total** | | | **~50-65 hrs** |

**After Phase 0, adding a new biological component**:
- Before: 35-46 hours (edit 5-6 core files)
- After: **15-20 hours** (self-contained plugin package)

---

## Phase A: Physics Foundation (Months 3-5)

> **Note**: Phase A now builds on the extensibility framework from Phase 0. New physics models (cell mechanics, Warp kernels) will be implemented as plugins using the `PhysicsModel` interface.

### Goal
Transform Cognisom from a biochemistry simulator into a physics simulator with differentiable mechanics.

### A.1 NVIDIA Warp Integration (Weeks 1-4)

**Objective**: Port core kernels to Warp for autodifferentiation.

**Tasks**:
1. Install Warp SDK, configure build
2. Port `tau_leap_step` kernel to Warp
3. Port `direct_ssa_step` kernel to Warp
4. Port `laplacian_3d` kernel to Warp
5. Add gradient computation through SSA (parameter sensitivity)
6. Benchmark against CuPy implementation

**Files Created**:
- `cognisom/physics/__init__.py`
- `cognisom/physics/warp_backend.py`
- `cognisom/physics/warp_ssa.py`
- `cognisom/physics/warp_diffusion.py`

**Success Criteria**:
- All existing GPU tests pass with Warp backend
- Can compute ∂(output)/∂(rate_constant) through 100 SSA steps
- Performance within 2x of CuPy implementation

### A.2 Cell Mechanics Module (Weeks 5-8)

**Objective**: Add force-based cell-cell interactions.

**Physics Model**:
```
F_total = F_repulsion + F_adhesion + F_chemotaxis + F_random

F_repulsion = k_rep * (R_i + R_j - d_ij) * n_ij   if d_ij < R_i + R_j
F_adhesion = -k_adh * E_cadherin * n_ij           if d_ij < d_contact
F_chemotaxis = μ * ∇[cytokine]
F_random = √(2kT/γ) * ξ(t)                        Brownian motion
```

**Tasks**:
1. Add position, velocity, force vectors to cell state
2. Implement soft-sphere repulsion in Warp
3. Implement adhesion force (E-cadherin dependent)
4. Implement chemotactic gradient following
5. Add Langevin noise (Brownian motion)
6. Implement Verlet/XPBD integration

**Files Created**:
- `cognisom/physics/cell_mechanics.py`
- `cognisom/physics/force_kernels.py`
- `cognisom/physics/integrator.py`

**Success Criteria**:
- 10,000 cells simulate at 30 fps on RTX 4090
- Cells maintain volume (no overlap > 10%)
- Chemotaxis produces correct migration patterns

### A.3 Newton Integration (Weeks 9-12)

**Objective**: Add soft-body deformation and contact constraints.

**Tasks**:
1. Build Newton SDK bridge for Python
2. Model cells as deformable ellipsoids
3. Implement membrane as thin-shell VBD mesh
4. Add contact constraints for cell-cell adhesion junctions
5. Couple Newton solver output back to Bio-USD scene

**Files Created**:
- `cognisom/physics/newton_bridge.py`
- `cognisom/physics/deformable_cell.py`
- `cognisom/physics/contact_solver.py`

**Success Criteria**:
- Cell deformation under compression is physically plausible
- Contact forces are stable (no jitter)
- Can simulate tissue compression (100 cells in confined space)

---

## Phase B: Atomic Bio-USD + Metastasis Framework (Months 6-8)

> **Note**: Phase B leverages the prim registry from Phase 0. New schema types (BioAtom, BioMembraneLayer, BioCryoETTomogram) will be self-registering plugins. This phase now includes the **Prostate Cancer Metastasis Framework** as the primary validation use case.

### Goal
Extend Bio-USD schema to atomic resolution and implement the multi-scale "Seed and Soil" metastasis simulation framework for prostate cancer bone metastasis.

### B.1 Atomic Schema Extension (Weeks 1-3)

**New Schema Types**:

```python
@dataclass
class BioAtom(BioUnit):
    """Single atom with coordinates and properties"""
    element: str  # "C", "N", "O", "H", "S", "P"
    position: Tuple[float, float, float]  # Ångströms
    residue_id: int
    chain_id: str
    occupancy: float = 1.0
    b_factor: float = 0.0
    formal_charge: int = 0

@dataclass
class BioAtomicStructure(BioUnit):
    """Atomic-level representation of a molecule"""
    atoms: List[BioAtom]
    bonds: List[Tuple[int, int, str]]  # (idx1, idx2, bond_type)
    source_pdb: str = ""
    resolution_angstroms: float = 0.0

@dataclass
class BioForceFieldAPI:
    """Applied API for molecular dynamics parameters"""
    force_field_name: str  # "AMBER99SB", "CHARMM36"
    partial_charges: List[float]
    vdw_epsilon: List[float]
    vdw_sigma: List[float]
    masses: List[float]
```

**Tasks**:
1. Define atomic prim types in `cognisom/biousd/atomic_schema.py`
2. Build PDB → BioAtomicStructure converter
3. Build GROMACS topology → ForceFieldAPI converter
4. Update validator for atomic constraints
5. Update USDA exporter for atomic prims

**Files Created**:
- `cognisom/biousd/atomic_schema.py`
- `cognisom/biousd/pdb_converter.py`
- `cognisom/biousd/forcefield_loader.py`

### B.2 Membrane Schema (Weeks 4-6)

**New Schema Types**:

```python
@dataclass
class BioLipidMolecule(BioUnit):
    """Membrane lipid type"""
    lipid_type: str  # "POPC", "DOPC", "cholesterol"
    headgroup: str
    chain_lengths: Tuple[int, int]
    unsaturation: Tuple[int, int]

@dataclass
class BioMembraneLayer(BioUnit):
    """Lipid bilayer segment"""
    lipid_composition: Dict[str, float]
    normal: Tuple[float, float, float]
    area_nm2: float
    embedded_proteins: List[str]  # prim paths

@dataclass
class BioDNAStructure(BioUnit):
    """Double helix with base pairs"""
    sequence: str
    base_pairs: List[Tuple[int, int]]
    form: str = "B"  # A, B, Z DNA

@dataclass
class BioRNAStructure(BioUnit):
    """RNA with secondary structure"""
    sequence: str
    dot_bracket: str  # Secondary structure
    tertiary_coords: Optional[np.ndarray]
```

**Tasks**:
1. Define membrane/nucleic acid schemas
2. Build FASTA → BioRNA/DNA converter
3. Build Martini/CHARMM lipid library
4. Add membrane visualization to Omniverse

### B.3 Cryo-ET Schema (Weeks 7-9)

**New Schema Types**:

```python
@dataclass
class BioCryoETTomogram(BioUnit):
    """Cryo-EM tomogram volume"""
    grid_shape: Tuple[int, int, int]
    voxel_size_angstroms: float
    data_ref: str  # Path to MRC file
    microscope: str
    voltage_kv: int
    resolution_angstroms: float

@dataclass
class BioCryoETSegmentation(BioUnit):
    """Segmented structure from tomogram"""
    tomogram_ref: str
    structure_type: str
    mask_ref: str
    confidence_ref: str
    linked_structure: Optional[str]  # BioAtomicStructure path
```

**Tasks**:
1. Define cryo-ET schemas
2. Build MRC/STAR file importers
3. Add segmentation overlay to converter
4. Integrate with fVDB for volume rendering

### B.4 Prostate Cancer Metastasis Framework (Weeks 10-13)

**Goal**: Implement the complete "Seed and Soil" metastasis cascade for prostate cancer → bone.

**Background**: Prostate cancer uniquely metastasizes to bone via osteoblastic (bone-forming) lesions. Before tumor cells arrive, they send exosome "advance teams" containing DNA/RNA/proteins that reprogram bone marrow cells (the "Genometastasis Hypothesis").

#### B.4.1 New Bio-USD Prim Types for Metastasis

```python
# Circulating Tumor Cell (CTC) with EMT state
@prim_registry.register
class BioCirculatingTumorCell(BioCell):
    """Cancer cell in circulation"""
    emt_state: float = 0.0  # 0=epithelial, 1=mesenchymal
    membrane_integrity: float = 1.0
    cxcr4_expression: float = 0.5  # Chemokine receptor for bone homing
    mmp_secretion: float = 0.0  # Matrix metalloproteinase for invasion
    extravasation_competent: bool = False

# Bone Microenvironment Niche
@prim_registry.register
class BioBoneNiche(BioUnit):
    """Pre-metastatic niche in bone marrow"""
    niche_id: int = 0
    position: Tuple[float, float, float] = (0, 0, 0)
    stiffness: float = 1.0  # Bone matrix stiffness (GPa)
    cxcl12_concentration: float = 0.5  # Chemokine gradient
    fibronectin_level: float = 0.1  # "Soil" preparation marker
    primed_state: float = 0.0  # 0=healthy, 1=fully primed by exosomes
    immune_surveillance: float = 1.0  # 0=suppressed, 1=active

# Osteoblast with biomineralization
@prim_registry.register
class BioOsteoblast(BioCell):
    """Bone-forming cell"""
    osteoblast_id: int = 0
    activation_state: float = 0.0  # Tumor-induced activation
    bone_matrix_production: float = 1.0  # Normal rate
    et1_receptor_expression: float = 0.5  # Endothelin-1 receptor
    bmp_responsiveness: float = 0.5  # BMP pathway activation

# Osteoclast for bone remodeling
@prim_registry.register
class BioOsteoclast(BioCell):
    """Bone-resorbing cell"""
    osteoclast_id: int = 0
    resorption_activity: float = 1.0
    rankl_sensitivity: float = 0.5
```

#### B.4.2 New API Schemas for Metastasis

```python
# Chemotaxis API for CXCR4/CXCL12 homing
@dataclass
class BioChemotaxisAPI:
    """Applied to cells that follow chemical gradients"""
    receptor_type: str = "CXCR4"
    receptor_expression: float = 0.5
    sensitivity: float = 1.0
    chemokine_target: str = "CXCL12"
    velocity_bias: Tuple[float, float, float] = (0, 0, 0)

# EMT (Epithelial-Mesenchymal Transition) API
@dataclass
class BioEMT_API:
    """Tracks EMT state for invasion"""
    e_cadherin: float = 1.0  # Epithelial marker (high = stuck)
    n_cadherin: float = 0.0  # Mesenchymal marker (high = motile)
    vimentin: float = 0.0  # Mesenchymal marker
    snail_expression: float = 0.0  # EMT transcription factor
    emt_score: float = 0.0  # Composite: 0=epithelial, 1=mesenchymal

# Payload Carrier API for exosomes
@dataclass
class BioPayloadCarrierAPI:
    """Extended exosome cargo for metastasis"""
    # Existing cargo fields in BioExosome
    # Additional metastasis-specific fields:
    dna_fragments: List[str] = field(default_factory=list)  # Cancer DNA
    mir_141: float = 0.0  # Bone metastasis-promoting miRNA
    mir_21: float = 0.0  # Immunosuppressive miRNA
    integrin_alpha_v_beta_3: float = 0.0  # Bone-targeting integrin
    reprogramming_potential: float = 0.0  # Ability to prime niche

# Biomineralization API for osteoblasts
@dataclass
class BioBiomineralizationAPI:
    """Osteoblast bone formation state"""
    alkaline_phosphatase: float = 1.0
    osteocalcin: float = 0.5
    collagen_type_1: float = 1.0
    hydroxyapatite_deposition: float = 1.0
    matrix_volume_produced: float = 0.0
```

#### B.4.3 Metastasis Simulation Modules

**Module 1: Exosome Cloud & Pre-Metastatic Niche Formation**

```python
class ExosomeMetastasisModule(SimulationModule):
    """
    Simulates exosome bombardment and bone niche priming.

    Physics: NVIDIA Warp fluid advection (particles follow blood flow).
    Logic: When exosome collides with BoneMarrowCell, trigger Transfer event.
    Effect: BoneMarrowCell state changes from Healthy → Primed.
    """

    def update(self, dt: float):
        # 1. Primary tumor releases exosomes (10^9/day)
        self.release_exosomes_from_tumor(rate=1e6 * dt)

        # 2. Advect exosomes through blood flow field
        self.advect_exosomes(self.blood_flow_field, dt)

        # 3. Check collisions with bone marrow cells
        for exosome in self.exosomes:
            nearby_cells = self.spatial_query(exosome.position, radius=5.0)
            for cell in nearby_cells:
                if cell.type in ['bone_marrow_stem_cell', 'osteoblast']:
                    self.transfer_payload(exosome, cell)

        # 4. Update niche priming state
        for niche in self.bone_niches:
            niche.primed_state = self.calculate_priming(niche)
            niche.fibronectin_level = self.calculate_fibronectin(niche)
```

**Module 2: Intravasation & Circulation Survival**

```python
class IntravasationModule(SimulationModule):
    """
    Simulates cancer cell escape into bloodstream.

    Physics: Newton VBD Soft Body Dynamics.
    Constraint: Cell must deform to pass through 1/10th its diameter.
    Failure: If shear_stress > membrane_integrity → cell ruptures.
    """

    def update(self, dt: float):
        for ctc in self.circulating_tumor_cells:
            # 1. Check EMT state (must be mesenchymal to intravasate)
            if ctc.emt_score < 0.7:
                continue  # Still epithelial, can't squeeze through

            # 2. Deformation physics (Newton VBD)
            gap_width = self.get_endothelial_gap(ctc.position)
            required_deformation = ctc.diameter / gap_width

            if required_deformation > 10:  # Extreme deformation
                stress = self.calculate_shear_stress(ctc, gap_width)
                if stress > ctc.membrane_integrity:
                    ctc.alive = False  # Rupture (99.9% die)
                    continue

            # 3. Successful intravasation
            ctc.in_circulation = True
            ctc.extravasation_competent = True
```

**Module 3: Chemotaxis Bone Homing**

```python
class BoneHomingModule(SimulationModule):
    """
    Simulates CXCR4/CXCL12-mediated bone homing.

    Physics: Warp kernel for chemotactic gradient following.
    Environment: OpenVDB voxel grid of CXCL12 concentration.
    """

    def update(self, dt: float):
        # GPU kernel for chemotaxis
        for ctc in self.circulating_tumor_cells:
            if ctc.cxcr4_expression > 0.3:
                # Sample CXCL12 gradient at cell position
                gradient = self.sample_grid_gradient(
                    self.cxcl12_field, ctc.position
                )

                # Apply chemotactic force
                direction = normalize(gradient)
                force = ctc.cxcr4_expression * self.attraction_strength
                ctc.velocity += direction * force * dt
```

**Module 4: Vicious Cycle (Osteoblastic Lesion)**

```python
class ViciousCycleModule(SimulationModule):
    """
    Simulates the prostate cancer → osteoblast → tumor growth feedback loop.

    Unique to prostate cancer: OSTEOBLASTIC (bone-forming) not osteolytic.

    Cycle:
    1. Tumor releases ET-1, BMPs
    2. Osteoblasts hyper-activated → lay disordered bone
    3. New bone releases IGFs from matrix
    4. Tumor consumes IGFs → grows faster → more ET-1
    """

    def update(self, dt: float):
        for tumor_cell in self.bone_metastasis_cells:
            # 1. Tumor secretes osteoblast-activating factors
            et1 = tumor_cell.secrete('ET1', rate=0.1)
            bmp = tumor_cell.secrete('BMP2', rate=0.05)

            # 2. Activate nearby osteoblasts
            nearby_osteoblasts = self.spatial_query(
                tumor_cell.position, radius=50.0, type='osteoblast'
            )
            for ob in nearby_osteoblasts:
                ob.activation_state += (et1 + bmp) * ob.responsiveness * dt
                ob.bone_matrix_production *= (1 + 0.5 * ob.activation_state)

            # 3. Generate new bone matrix (mesh generation)
            for ob in nearby_osteoblasts:
                if ob.activation_state > 0.5:
                    self.generate_bone_matrix(ob.position, volume=ob.matrix_volume)

            # 4. Release IGFs from new bone
            igf_released = self.bone_matrix_volume * self.igf_density

            # 5. Tumor uptakes IGFs → accelerates growth
            for tumor_cell in nearby_tumor_cells:
                tumor_cell.proliferation_rate *= (1 + 0.3 * igf_released)
```

#### B.4.4 Metastasis Visualization in Omniverse

**Visual Output**:
- Healthy bone: Porous trabecular structure (normal mesh)
- Primed niche: Subtle color change (fibronectin accumulation)
- Metastatic lesion: Dense, chaotic "concrete-like" geometry (sclerotic)
- Exosome cloud: Particle system following blood flow
- CXCL12 gradient: Volume rendering (red = high concentration)

#### B.4.5 Success Criteria for Metastasis Framework

- [ ] Exosome cloud follows blood flow physics (Warp advection)
- [ ] Bone niche priming visualized over time (fibronectin accumulation)
- [ ] CTC deformation physics during intravasation (Newton VBD)
- [ ] 99.9% CTC death rate in circulation matches biology
- [ ] CXCR4/CXCL12 chemotaxis produces correct bone homing
- [ ] Osteoblastic lesion formation visible in Omniverse (sclerotic bone)
- [ ] Vicious cycle produces exponential tumor growth
- [ ] End-to-end metastasis cascade: Primary → Exosomes → CTCs → Bone → Lesion

---

## Phase C: AI Intelligence Layer (Months 9-12)

> **Note**: Phase C uses the entity registry from Phase 0. Generated proteins and molecules auto-register as library entities.

### Goal
Integrate generative biology models (ESM-3, Cosmos Reason) to create a "virtual scientist" agent.

### C.1 ESM-3 Integration (Weeks 1-4)

**Capability**: Multimodal protein generation from sequence/structure/function constraints.

**Tasks**:
1. Build ESM-3 NIM client (when available) or self-host
2. Create prompt templates for protein generation
3. Implement sequence → structure → BioAtomicStructure pipeline
4. Add "Generate Protein" tool to Research Agent
5. Cache generated structures in Entity Library

**Files Created**:
- `cognisom/nim/esm3_client.py`
- `cognisom/agent/tools/protein_generator.py`

**Use Case**:
```
User: "I need an enzyme that catalyzes the hydrolysis of X in the prostate"
Agent:
  1. Queries literature for similar enzymes
  2. Extracts functional constraints
  3. Calls ESM-3 to generate candidate sequences
  4. Validates structure with Boltz-1
  5. Inserts into simulation for testing
```

### C.2 Cosmos Reason Integration (Weeks 5-8)

**Capability**: Physical reasoning over simulation state, anomaly detection, hypothesis generation.

**Tasks**:
1. Build Cosmos Reason NIM client
2. Create simulation → video stream for VLM input
3. Implement observation → hypothesis → action loop
4. Add reasoning tool to Research Agent
5. Build anomaly detection pipeline

**Files Created**:
- `cognisom/nim/cosmos_reason_client.py`
- `cognisom/agent/cosmos_agent.py`
- `cognisom/agent/tools/simulation_observer.py`

**Use Case**:
```
Cosmos Reason observes: "Membrane is rupturing at t=12.3h"
Hypothesis: "Osmotic pressure exceeds elastic limit"
Action: "Recommend reducing solute concentration by 20%"
```

### C.3 Virtual Scientist Agent (Weeks 9-12)

**Architecture**:
```
┌─────────────────────────────────────────────────────┐
│                 Virtual Scientist                    │
├─────────────────────────────────────────────────────┤
│  Perception:   Cosmos Reason (observes simulation)  │
│  Knowledge:    Entity Library + Literature RAG      │
│  Generation:   ESM-3 (proteins), MolMIM (drugs)     │
│  Validation:   Boltz-1 (structures), DiffDock       │
│  Action:       Modify simulation, queue experiments │
└─────────────────────────────────────────────────────┘
```

**Tasks**:
1. Build agent orchestrator with tool selection
2. Implement observation → hypothesis → experiment loop
3. Add "explain this result" capability
4. Add automatic parameter tuning via gradient descent
5. Dashboard integration (agent chat panel)

**Files Created**:
- `cognisom/agent/virtual_scientist.py`
- `cognisom/agent/hypothesis_engine.py`
- `cognisom/dashboard/pages/14_scientist_agent.py`

---

## Phase D: Multi-Scale Coupling (Months 13-16)

> **Note**: Phase D demonstrates the power of Phase 0's extensibility — multi-scale models are composed from registered physics models at each scale.

### Goal
Achieve true bidirectional coupling between atomic, molecular, cellular, and tissue scales.

### D.1 Atomic ↔ Molecular Coupling (Weeks 1-4)

**Challenge**: Molecular dynamics (femtoseconds) vs. cell dynamics (hours).

**Approach**: Adaptive multi-rate integration
- Run MD on active sites only (binding events)
- Coarse-grain rest of protein as rigid bodies
- Use neural network surrogate for common conformations

**Tasks**:
1. Implement adaptive time-stepping
2. Build MD → coarse-grain projection
3. Train neural surrogate on common proteins
4. Couple binding events to receptor kinetics

### D.2 Molecular ↔ Cellular Coupling (Weeks 5-8)

**Challenge**: Protein copy numbers (thousands) vs. cell forces (continuous).

**Approach**: Event-driven coupling
- When gene expression changes → update force parameters
- When mechanical stress exceeds threshold → trigger signaling

**Tasks**:
1. Map gene expression to mechanical properties (stiffness, adhesion)
2. Implement mechanotransduction (force → signaling)
3. Couple SSA output to mechanics parameters
4. Validate with known mechanobiology

### D.3 Cellular ↔ Tissue Coupling (Weeks 9-12)

**Challenge**: Individual cells (thousands) vs. tissue continuum.

**Approach**: Hybrid discrete-continuum
- Dense regions → continuum (tissue mechanics)
- Sparse regions → discrete cells
- Dynamic switching based on density

**Tasks**:
1. Implement continuum tissue solver (FEM)
2. Build adaptive mesh refinement
3. Couple cell dynamics to tissue stress
4. Add tissue-level visualization

---

## Phase E: Lab-in-the-Loop (Months 17-20)

> **Note**: Phase E uses module auto-discovery from Phase 0. Lab connectors are packaged as optional plugins.

### Goal
Close the loop between digital simulation and physical laboratory.

### E.1 Lab Robotics Connector (Weeks 1-4)

**Integration Targets**:
- Automata LINQ (high-throughput)
- OpenTrons OT-2 (accessible)
- Hamilton STAR (precision)

**Tasks**:
1. Build REST API adapter for each platform
2. Create experiment job queue
3. Implement protocol translation (simulation → robot commands)
4. Add equipment status monitoring

**Files Created**:
- `cognisom/lab/__init__.py`
- `cognisom/lab/automata_connector.py`
- `cognisom/lab/opentrons_connector.py`
- `cognisom/lab/experiment_queue.py`

### E.2 Validation Pipeline (Weeks 5-8)

**Workflow**:
```
Simulation predicts: "Drug X inhibits pathway Y at IC50 = 10 nM"
    ↓
Lab validates: Robot runs dose-response assay
    ↓
Results return: Actual IC50 = 15 nM
    ↓
Feedback: Update simulation parameters
    ↓
Iterate until prediction matches experiment
```

**Tasks**:
1. Build experiment result ingestion
2. Implement parameter fitting from experimental data
3. Add confidence calibration
4. Dashboard for validation status

### E.3 AOUSD Standardization (Weeks 9-12)

**Objective**: Submit BioUSD as open standard to Alliance for OpenUSD.

**Tasks**:
1. Finalize BioUSD schema specification
2. Write formal schema documentation
3. Create reference implementation
4. Submit to AOUSD working group
5. Open-source all schema code

### E.4 Publication & Release (Weeks 13-16)

**Tasks**:
1. Benchmark against Lattice Microbes
2. Validate against Visual Proteomics test
3. Write technical paper
4. Create demo videos
5. Launch public beta

---

## Part 5: Resource Requirements

### Team Structure

| Role | FTE | Phase Focus |
|------|-----|-------------|
| Platform/Architecture Engineer | 0.5 | 0 (Extensibility) |
| Physics Engineer | 1.0 | A, D |
| Graphics/USD Engineer | 0.5 | 0, B |
| ML/AI Engineer | 1.0 | C |
| Bio/Simulation Engineer | 1.0 | A, B, D |
| Platform Engineer | 0.5 | E |
| **Total** | 4.5 FTE | |

### Infrastructure

| Resource | Specification | Purpose |
|----------|---------------|---------|
| Development GPU | 4x RTX 4090 | Physics dev, benchmarking |
| Cloud GPU | 8x A100 (on-demand) | Large-scale testing |
| NIM Access | NVIDIA Developer tier | ESM-3, Cosmos Reason |
| Lab Equipment | OpenTrons OT-2 | Validation experiments |

### Budget Estimate (20 months)

| Category | Cost |
|----------|------|
| Personnel (4.5 FTE × 20 mo) | $1.5M |
| Cloud Compute | $120K |
| Equipment | $50K |
| NVIDIA Developer Program | $25K |
| **Total** | ~$1.7M |

---

## Part 6: Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| ESM-3 not available as NIM | Medium | High | Self-host or use ESM-2 + AlphaFold |
| Warp performance insufficient | Low | High | Fall back to custom CUDA |
| Newton license issues | Medium | Medium | Use open-source alternatives (XPBD) |
| Lab integration delays | High | Low | Delay Phase E, prioritize simulation |
| AOUSD rejection | Medium | Low | Maintain as proprietary standard |

---

## Part 7: Success Metrics

### Phase 0 (Extensibility)
- [ ] New entity type can be added without editing `models.py`
- [ ] New Bio-USD prim auto-registers via decorator
- [ ] Simulation modules discovered via entry points
- [ ] New particle type works in GPU kernels without kernel edits
- [ ] Example "virus plugin" package demonstrates all extension points
- [ ] Adding a new biological component takes < 20 hours (down from 35-46)

### Phase A (Physics)
- [ ] 10,000 cells at 30 fps with mechanics
- [ ] Gradient computation through 100 SSA steps
- [ ] Cell deformation under compression

### Phase B (Atomic Bio-USD + Metastasis)
- [ ] Import 1,000 PDB structures automatically
- [ ] Force field parameters for AMBER/CHARMM
- [ ] Cryo-ET tomogram visualization
- [ ] **Metastasis**: Exosome cloud follows blood flow physics
- [ ] **Metastasis**: Bone niche priming visualized (fibronectin)
- [ ] **Metastasis**: CTC intravasation with Newton VBD deformation
- [ ] **Metastasis**: CXCR4/CXCL12 chemotaxis to bone
- [ ] **Metastasis**: Osteoblastic lesion formation visible
- [ ] **Metastasis**: End-to-end cascade: Primary → Exosomes → CTCs → Bone → Lesion

### Phase C (AI)
- [ ] ESM-3 generates functional protein variants
- [ ] Cosmos Reason detects simulation anomalies
- [ ] Virtual scientist proposes hypothesis from observation

### Phase D (Multi-Scale)
- [ ] Binding event triggers signaling cascade
- [ ] Mechanical stress modulates gene expression
- [ ] Tissue continuum emerges from cell dynamics

### Phase E (Lab-in-the-Loop)
- [ ] Robot executes simulation-designed experiment
- [ ] Experimental result updates simulation parameters
- [ ] BioUSD submitted to AOUSD

---

## Appendix A: File Structure After Phase E

```
cognisom/
├── core/                    # NEW (Phase 0): Core registries
│   ├── __init__.py
│   ├── registry.py          # Generic registry base class
│   ├── module_discovery.py  # Entry-point based module discovery
│   └── plugin_manager.py    # Plugin lifecycle management
├── auth/                    # (existing) Multi-tenant auth
├── dashboard/               # (existing + Phase 0 extensions)
│   ├── component_registry.py     # NEW: UI component plugins
│   ├── visualization_plugins.py  # NEW: 3D renderer interface
│   └── pages/
│       └── 15_scientist_agent.py # NEW (Phase C)
├── engine/                  # (existing) Simulation modules
├── gpu/                     # (existing + Phase 0 extensions)
│   ├── physics_model.py     # NEW: PhysicsModel protocol
│   ├── property_schema.py   # NEW: Property definitions
│   ├── kernel_composer.py   # NEW: Dynamic kernel generation
│   └── ...                  # (existing kernels)
├── physics/                 # NEW (Phase A): Warp/Newton integration
│   ├── __init__.py
│   ├── warp_backend.py
│   ├── warp_ssa.py
│   ├── warp_diffusion.py
│   ├── cell_mechanics.py
│   ├── force_kernels.py
│   ├── integrator.py
│   ├── newton_bridge.py
│   ├── deformable_cell.py
│   └── contact_solver.py
├── biousd/                  # (existing + Phase 0 + Phase B)
│   ├── schema.py            # (modified: uses prim registry)
│   ├── prim_registry.py     # NEW (Phase 0): Prim type registration
│   ├── atomic_schema.py     # NEW (Phase B)
│   ├── forcefield_schema.py # NEW (Phase B)
│   ├── membrane_schema.py   # NEW (Phase B)
│   ├── cryoet_schema.py     # NEW (Phase B)
│   ├── pdb_converter.py     # NEW (Phase B)
│   └── forcefield_loader.py # NEW (Phase B)
├── nim/                     # (existing + Phase C)
│   ├── esm3_client.py       # NEW
│   └── cosmos_reason_client.py # NEW
├── agent/                   # (existing + Phase C)
│   ├── virtual_scientist.py # NEW
│   ├── cosmos_agent.py      # NEW
│   └── hypothesis_engine.py # NEW
├── lab/                     # NEW (Phase E): Lab automation
│   ├── __init__.py
│   ├── automata_connector.py
│   ├── opentrons_connector.py
│   └── experiment_queue.py
├── library/                 # (existing + Phase 0 modifications)
│   ├── models.py            # (modified: uses entity registry)
│   └── ...
├── plugins/                 # NEW (Phase 0): Local plugin directory
│   └── README.md
├── cli/                     # NEW (Phase 0): CLI tools
│   └── create_plugin.py     # Plugin template generator
├── docs/
│   └── developer/
│       └── building-plugins.md  # NEW (Phase 0)
└── examples/
    └── virus_plugin/        # NEW (Phase 0): Example plugin
        ├── __init__.py
        ├── entity.py        # BioVirus entity
        ├── prim.py          # Bio-USD prim
        ├── module.py        # Simulation module
        ├── physics.py       # GPU physics model
        └── ui.py            # Dashboard components
```

---

## Appendix B: Key Technical Decisions

### Decision 1: Warp vs. JAX for Autodiff
**Choice**: NVIDIA Warp
**Rationale**: Native USD integration, better Omniverse compatibility, NVIDIA support.

### Decision 2: Newton vs. Custom Physics
**Choice**: Newton SDK
**Rationale**: Production-quality soft-body solver, maintained by NVIDIA, VBD algorithm for membranes.

### Decision 3: ESM-3 vs. ESM-2 + AlphaFold
**Choice**: ESM-3 (when available), ESM-2 fallback
**Rationale**: ESM-3 is multimodal (seq + structure + function), enables generative biology.

### Decision 4: Cryo-ET via fVDB vs. Custom
**Choice**: NVIDIA fVDB
**Rationale**: Optimized sparse volume library, integrates with Omniverse natively.

### Decision 5: Metastasis Framework as Primary Use Case
**Choice**: Prostate Cancer → Bone Metastasis
**Rationale**:
- Prostate cancer is the primary validation use case for Cognisom
- Bone metastasis is the lethal endpoint (90% of prostate cancer deaths)
- Unique osteoblastic (bone-forming) mechanism differentiates from breast cancer
- Exosome-mediated "Genometastasis" showcases Bio-USD molecular capabilities
- Multi-scale cascade (molecular → cellular → tissue) demonstrates full platform

---

## Appendix C: Extensibility Architecture

### Registry Pattern Overview

After Phase 0, all major subsystems use a consistent registry pattern:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Cognisom Plugin System                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   Entity    │  │  Bio-USD    │  │ Simulation  │              │
│  │  Registry   │  │   Prim      │  │   Module    │              │
│  │             │  │  Registry   │  │  Discovery  │              │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
│         │                │                │                      │
│  ┌──────┴──────┐  ┌──────┴──────┐  ┌──────┴──────┐              │
│  │   GPU       │  │  Dashboard  │  │   Entry     │              │
│  │  Physics    │  │  Component  │  │   Points    │              │
│  │  Models     │  │  Registry   │  │   Loader    │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Adding a New Biological Component (Post-Phase 0)

**Example: Adding "BioVirus" to Cognisom**

1. **Create plugin package** (5 minutes):
   ```bash
   cognisom create-plugin virus
   cd cognisom-virus-plugin
   ```

2. **Define entity** (`entity.py`, 30 minutes):
   ```python
   from cognisom.core.registry import entity_registry

   @entity_registry.register("virus")
   class Virus(BioEntity):
       genome_type: str = "RNA"  # DNA, RNA
       envelope: bool = True
       host_tropism: List[str] = field(default_factory=list)
       replication_rate: float = 0.1
   ```

3. **Define Bio-USD prim** (`prim.py`, 30 minutes):
   ```python
   from cognisom.biousd.prim_registry import prim_registry

   @prim_registry.register
   class BioVirus(BioUnit):
       virus_id: int = 0
       position: Tuple[float, float, float] = (0, 0, 0)
       genome_type: str = "RNA"
       envelope_integrity: float = 1.0
       attached_to_cell_id: int = -1
   ```

4. **Define simulation module** (`module.py`, 4-8 hours):
   ```python
   from cognisom.core.module_base import SimulationModule

   class ViralReplicationModule(SimulationModule):
       def update(self, dt: float):
           # Viral dynamics: attachment, entry, replication, budding
           ...
   ```

5. **Define GPU physics** (`physics.py`, 2-4 hours, optional):
   ```python
   from cognisom.gpu.physics_model import physics_registry

   @physics_registry.register("virus")
   class ViralPhysics(PhysicsModel):
       properties = ["envelope_integrity", "replication_stage", "position"]

       def update_rule(self, state, dt):
           # GPU-accelerated viral dynamics
           ...
   ```

6. **Define UI components** (`ui.py`, 1 hour):
   ```python
   from cognisom.dashboard.component_registry import ui_registry

   @ui_registry.register("virus")
   class VirusUI:
       color = "#8B0000"
       icon = "🦠"

       def render_card(self, entity):
           return f"**{entity.name}** ({entity.genome_type})"
   ```

7. **Register entry points** (`pyproject.toml`, 5 minutes):
   ```toml
   [project.entry-points."cognisom.entities"]
   virus = "cognisom_virus:Virus"

   [project.entry-points."cognisom.prims"]
   virus = "cognisom_virus:BioVirus"

   [project.entry-points."cognisom.modules"]
   viral = "cognisom_virus:ViralReplicationModule"
   ```

8. **Install and use**:
   ```bash
   pip install -e .
   # Plugin auto-discovered at Cognisom startup
   ```

**Total effort: ~15-20 hours** (down from 35-46 hours before Phase 0)

### Supported Extension Points

| Extension Point | Registry | Decorator | Entry Point |
|-----------------|----------|-----------|-------------|
| Entity Types | `entity_registry` | `@entity_registry.register("name")` | `cognisom.entities` |
| Bio-USD Prims | `prim_registry` | `@prim_registry.register` | `cognisom.prims` |
| Simulation Modules | `ModuleDiscovery` | — | `cognisom.modules` |
| Physics Models | `physics_registry` | `@physics_registry.register("name")` | `cognisom.physics` |
| UI Components | `ui_registry` | `@ui_registry.register("name")` | `cognisom.ui` |
| Dashboard Pages | `page_registry` | `@page_registry.register` | `cognisom.pages` |

### Plugin Compatibility

Each registry enforces a version contract:

```python
@entity_registry.register("virus", min_cognisom_version="2.1.0")
class Virus(BioEntity):
    ...
```

Cognisom validates plugin compatibility at startup and warns about version mismatches.

---

## Appendix D: Prostate Cancer Metastasis Use Case

### Existing Capabilities (Already Built)

| Component | Status | Location |
|-----------|--------|----------|
| **BioExosome prim** | ✅ Complete | `biousd/schema.py:286-304` |
| **Exosome cargo** | ✅ Complete | `cargo_mrna`, `cargo_mirna`, `cargo_proteins` |
| **VascularModule** | ✅ Complete | `modules/vascular_module.py` |
| **LymphaticModule** | ✅ Complete | `modules/lymphatic_module.py` |
| **Metastasis tracking** | ✅ Complete | `METASTASIS_OCCURRED` event |
| **CellType enum** | ✅ Complete | Includes CANCER, IMMUNE, STROMAL |
| **BioInteractionAPI** | ✅ Complete | Binding affinity, interaction type |
| **GPU spatial queries** | ✅ Complete | N-body detection in `spatial_ops.py` |
| **Diffusion solver** | ✅ Complete | 3D Laplacian for gradients |
| **Molecular transmission docs** | ✅ Complete | `MOLECULAR_CANCER_TRANSMISSION.md` |

### New Capabilities Required (Phase B.4)

| Component | Status | Priority |
|-----------|--------|----------|
| **BioCirculatingTumorCell** | ❌ New prim | P1 |
| **BioBoneNiche** | ❌ New prim | P1 |
| **BioOsteoblast/Osteoclast** | ❌ New prims | P1 |
| **BioChemotaxisAPI** | ❌ New API | P1 |
| **BioEMT_API** | ❌ New API | P1 |
| **BioBiomineralizationAPI** | ❌ New API | P2 |
| **ExosomeMetastasisModule** | ❌ New module | P1 |
| **IntravasationModule** | ❌ New module | P1 |
| **BoneHomingModule** | ❌ New module | P1 |
| **ViciousCycleModule** | ❌ New module | P1 |
| **Blood flow field (OpenVDB)** | ⚠️ Partial | P2 |
| **Newton VBD for cell deformation** | ❌ Phase A dep | P1 |
| **Warp advection for exosomes** | ❌ Phase A dep | P1 |

### The "Seed and Soil" Metastasis Cascade

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PROSTATE CANCER → BONE METASTASIS                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHASE 1: EXOSOME BOMBARDMENT ("Seed Preparation")                          │
│  ────────────────────────────────────────────────────                       │
│  • Primary tumor releases 10⁹ exosomes/day                                  │
│  • Cargo: miR-141, cancer DNA, integrins                                    │
│  • Travel via blood flow (Warp advection)                                   │
│  • Target: Bone marrow stem cells, osteoblasts                              │
│  • Effect: Niche "priming" → fibronectin↑, immune surveillance↓            │
│                                                                             │
│  PHASE 2: INTRAVASATION ("Seed Release")                                    │
│  ────────────────────────────────────────                                   │
│  • Cancer cell undergoes EMT (epithelial→mesenchymal)                       │
│  • Squeezes through endothelial gap (Newton VBD)                            │
│  • Enters blood as CTC (Circulating Tumor Cell)                             │
│  • 99.9% die from shear stress → membrane rupture                           │
│                                                                             │
│  PHASE 3: CHEMOTAXIS ("Seed Homing")                                        │
│  ────────────────────────────────────                                       │
│  • Bone marrow secretes CXCL12 chemokine                                    │
│  • CTCs express CXCR4 receptor                                              │
│  • Gradient-following (∇CXCL12 → velocity bias)                             │
│  • CTC "smells" bone, swims upstream                                        │
│                                                                             │
│  PHASE 4: VICIOUS CYCLE ("Seed → Plant → Harvest")                          │
│  ─────────────────────────────────────────────────                          │
│  • Tumor releases ET-1, BMPs → activate osteoblasts                         │
│  • Osteoblasts hyper-produce bone matrix (OSTEOBLASTIC)                     │
│  • New bone releases IGFs from matrix                                       │
│  • Tumor consumes IGFs → accelerated growth                                 │
│  • Visual: Porous bone → Dense "concrete" (sclerotic lesion)                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why This Use Case Validates the Full Platform

| Cognisom Capability | Metastasis Validation |
|--------------------|-----------------------|
| **Bio-USD Schema** | 4 new prim types, 3 new API schemas |
| **GPU Physics (Warp)** | Exosome advection, chemotaxis gradients |
| **Newton Soft-Body** | CTC deformation during intravasation |
| **SSA Kinetics** | Gene expression changes during EMT |
| **Spatial Fields** | CXCL12 concentration gradient (OpenVDB) |
| **Multi-Scale** | Molecular (exosome cargo) → Cellular (CTC) → Tissue (bone lesion) |
| **Omniverse Viz** | Bone matrix generation, exosome particle cloud |
| **Event Bus** | EMT_TRIGGERED, INTRAVASATION, METASTASIS_OCCURRED |

### Clinical Relevance

> "Over 90% of men who die from prostate cancer have bone metastases. Understanding the molecular mechanisms of bone homing and the osteoblastic vicious cycle is critical for developing effective treatments."
> — American Cancer Society, 2025

---

## Conclusion

Cognisom has a solid foundation for biochemistry simulation. The strategic vision requires adding:

0. **Extensibility framework** (plugin architecture) — 2 months ⭐ *Critical for scalability*
1. **Physics layer** (Warp + Newton) — 3 months
2. **Atomic Bio-USD** — 3 months
3. **AI intelligence** (ESM-3 + Cosmos Reason) — 4 months
4. **Multi-scale coupling** — 4 months
5. **Lab-in-the-Loop** — 4 months

Total timeline: **20 months** to full Bio-Digital Twin capability.

### Why Extensibility Comes First

The extensibility framework (Phase 0) is a strategic investment that pays dividends throughout the project:

| Without Phase 0 | With Phase 0 |
|-----------------|--------------|
| Adding new particle type: 35-46 hours | Adding new particle type: 15-20 hours |
| Must edit 5-6 core files | Self-contained plugin package |
| Core developers bottleneck all additions | Community can contribute independently |
| Hard to adapt to other biological domains | Easy expansion to neurons, viruses, bacteria, etc. |

**Example**: After Phase 0, a research partner could add a "BioNeuron" type with synapse dynamics by creating a single plugin package — no pull request to core Cognisom required.

### Extensibility Investment

| Refactoring Area | Effort | Payoff |
|------------------|--------|--------|
| Entity Type Registry | 8-10 hrs | New entities without code edits |
| Bio-USD Prim Registry | 12-15 hrs | New prims auto-register |
| Module Auto-Discovery | 6-8 hrs | External module packages |
| GPU Physics Interface | 15-20 hrs | Custom particle physics |
| Dashboard Plugins | 10-12 hrs | Custom UI components |
| **Total Phase 0** | **~50-65 hrs** | **2x faster additions** |

The existing ~42,000 lines of code remain valuable and require minimal rework. The primary investment is in **extensibility infrastructure** (Phase 0) followed by **new capabilities** (physics, atomics, AI agents).

**Likelihood of success: 85%** — contingent on NVIDIA NIM availability and successful Warp integration. Phase 0 de-risks subsequent phases by enabling parallel development.
