# cognisom — GPU-Accelerated Cellular Simulation Platform

## Vision
Understanding communication at the cellular scale to advance cancer research and treatment.

## Mission
Build a mechanistic, GPU-first simulation platform that models:
- Normal cellular function and intercellular communication
- Immune system recognition of "self" vs "non-self"
- Cancer evolution, immune evasion, and treatment resistance
- Multi-million cell tissue-scale dynamics

## Platform Architecture

### Phase 1: Single Cell Foundation (Months 0-6)
**Goal**: Rigorous single-cell model with validated biology

#### Intracellular Engine
- **Transcription/Translation**: Stochastic simulation (SSA/tau-leap) on GPU
- **Metabolism**: Dynamic FBA with GPU LP solver (glycolysis, TCA, OXPHOS)
- **Cell Cycle**: Checkpoints, DNA damage/repair, apoptosis
- **Antigen Presentation**: MHC-I peptide loading from protein pool
- **State Management**: Compact species counts (~2-8k species per cell)

#### Technical Stack
- **Compute**: CUDA kernels for batched SSA, FBA, PDE solvers
- **Memory**: Structure-of-arrays layout, ~8KB/cell base
- **I/O**: SBML import, Zarr/HDF5 checkpoints
- **Validation**: Unit tests vs literature (MAPK, p53, apoptosis)

### Phase 2: Immune System & Communication (Months 4-10)
**Goal**: Model immune surveillance and "self" recognition

#### Self-ID Mechanisms
- **MHC-I Presentation**: Neoantigens from mutated proteins
- **Expression Levels**: Baseline MHC-I, β2-microglobulin
- **Stress Signals**: NKG2D ligands, calreticulin exposure
- **Immune Checkpoints**: PD-L1/PD-1, CTLA-4

#### Immune Agents
- **NK Cells**: Missing-self detection + stress ligand recognition
- **CD8 T Cells**: TCR specificity to presented peptides
- **Macrophages**: M1/M2 polarization, phagocytosis
- **Dendritic Cells**: Antigen pickup → T-cell priming

#### Spatial Layer
- **Diffusion Fields**: O₂, glucose, lactate, cytokines (IFN-γ, IL-2, TGF-β)
- **CUDA PDE Solvers**: 2D→3D stencil methods
- **Secretion/Uptake**: Per-cell ports to field grid
- **Domain Decomposition**: Multi-GPU halo exchange

### Phase 3: Cancer Progression (Months 8-18)
**Goal**: Model oncogenesis, immune evasion, therapy resistance

#### Prostate Cancer Focus
**Normal → Oncogenic Stress → Immune Evasion → Castration Resistance**

##### Oncogenic Pathways
- PTEN loss → PI3K/AKT metabolic advantage
- TP53 dysfunction → DNA damage tolerance
- AR signaling modulation
- Clonal mutation sampling at division

##### Immune Evasion Strategies
- MHC-I downregulation
- PD-L1 upregulation
- Tumor-associated macrophage polarization (M1→M2)
- TGF-β microenvironmental suppression
- Antigen loss variants

##### Therapy Simulation
- **Androgen Deprivation Therapy (ADT)**
- **AR Antagonists** (enzalutamide)
- **Radiation** (antigen release)
- **Checkpoint Blockade** (anti-PD-1/PD-L1)
- **PK/PD Fields**: Drug diffusion, uptake, resistance emergence

#### Pancreatic Cancer Extension
**Desmoplastic, Immune-Excluded Microenvironment**

- Dense stromal fibroblasts (CAFs)
- Poor perfusion, hypoxia-driven EMT
- KRAS/TP53/CDKN2A/SMAD4 pathways
- Gemcitabine/FOLFIRINOX response curves

### Phase 4: Scale & Surrogates (Months 10-24)
**Goal**: Million+ cell simulations with ML acceleration

#### Multi-GPU Scaling
- Domain decomposition with NCCL/MPI
- Ghost cell halo exchange
- Overlap compute/communication
- Synthetic million-cell benchmarks

#### ML Surrogates
- **RN Surrogate**: GNN/RNN emulates stiff subnets (e.g., MAPK)
- **FBA Surrogate**: MLP predicts fluxes, periodic exact correction
- **PDE Surrogate**: UNet-style diffusion stepper
- **Accuracy Enforcement**: Periodic exact solver + error monitors

## Data Model

### Per-Cell State (~8-32 KB)
```
Species counts:    int32[2000-8000]  // molecular species
Parameters:        compact indices to global tables
State flags:       cell phase, stress level, mutation bitset
MHC-I peptides:    sparse array of presented antigens
Immune markers:    PD-L1 level, stress ligands
Secretion ports:   uptake/release rates per field
```

### Spatial Grid
```
Fields:            float32[Nx][Ny][Nz][n_species]
                   O₂, glucose, lactate, cytokines, drugs
Diffusion:         CUDA stencil solver
Resolution:        10-50 μm voxels
```

## Hardware Roadmap

### Phase A: MVP Development ($3.5k-$5k)
- **GPU**: RTX 4090 (24GB)
- **RAM**: 128-192GB
- **Storage**: 2-4TB NVMe
- **Capability**: 10k-50k cells, kernel development

### Phase B: Serious R&D ($32k-$50k)
- **GPU**: NVIDIA H100 80GB PCIe
- **RAM**: 256-512GB
- **Storage**: 4-8TB NVMe RAID
- **Capability**: 0.5-5M cells, realistic tumor spheroids

### Phase C: Scale Node ($130k-$210k)
- **GPU**: 4× H100 80GB SXM + NVLink
- **RAM**: 512GB-1TB
- **Network**: 200-400 Gbps InfiniBand
- **Capability**: 5-50M cells, tissue microenvironment

### Phase D: Cluster ($500k-$6M+)
- **Nodes**: 4-32 nodes × 4× H100
- **Capability**: 100M+ cells, full tumor ecology

## Funding Strategy

### Free Compute & Credits (Apply Now)
1. **NVIDIA Inception** — GPU credits + engineer consults
2. **AWS Cloud Credit for Research** — Compute credits
3. **Google Cloud Research Credits** — H100/A100 time
4. **Microsoft Azure Research** — Compute + ML tools
5. **NIH STRIDES** — Discounted HPC (w/ NIH-funded collaborator)

### Grant Targets (Non-Dilutive)
1. **NIH NCI ITCR** ($100k-$400k) — Cancer research software platforms
2. **NSF CSSI** ($80k-$300k) — Scientific simulation frameworks
3. **DoD PCRP** ($400k-$1.2M) — Prostate cancer research
4. **NIH R21** ($275k/2yr) — Exploratory high-risk research
5. **Cancer Grand Challenges** ($1M-$25M) — Multi-institution teams

### Budget Scenarios

#### Lean Start (12 months): $250k-$400k
- 1 workstation + cloud bursts
- 2-3 contract developers
- Prototype → grant applications

#### Grant-Ready (12 months): $230k-$340k
- Sustained H100 access
- 1 FTE comp bio + 0.5 FTE GPU engineer
- Publication + open-source release

## Validation & Rigor

### Unit Tests
- Conservation laws (mass, energy)
- Steady-state checks
- SSA vs ODE agreement at high copy numbers

### Biological Benchmarks
- Lac operon, MAPK cascade, p53 oscillations
- Growth curves vs literature (doubling time 18-36h)
- Hypoxia gradient formation in spheroids
- Immune surveillance vs escape dynamics

### Calibration
- Bayesian parameter inference
- Uncertainty quantification (ensembles)
- Cross-scale validation (intracellular → phenotype)

## Milestones

### Month 3
✅ Single-cell GPU SSA engine
✅ Basic transcription/translation/degradation
✅ Minimal metabolism (toy FBA)
✅ Checkpoint/restore to Zarr

### Month 6
✅ DNA damage/p53/apoptosis
✅ Calibrated doubling times
✅ 2D diffusion (O₂, glucose)
✅ Small colony growth (10-1000 cells)

### Month 9
✅ MHC-I presentation system
✅ NK + CD8 T-cell agents
✅ Immune surveillance → escape
✅ 3D spatial fields

### Month 12
✅ Prostate cancer clonal evolution
✅ ADT + checkpoint blockade simulation
✅ Multi-GPU domain decomposition
✅ Million-cell synthetic runs

### Month 18
✅ PDAC immune-excluded model
✅ ML surrogates (3-10× speedup)
✅ Published validation study
✅ Open-source MVP release

## Competitive Advantage

### vs Existing Approaches
| Approach | Limitation | Our Solution |
|----------|-----------|--------------|
| ML-only prediction | No mechanistic interpretability | Mechanistic + ML hybrid |
| Agent-based models | Limited intracellular detail | Full biochemical fidelity |
| Whole-cell models | Single bacteria only | Multicellular + immune |
| PhysiCell/BioDynaMo | Simplified intracellular | GPU-batched SSA + dFBA |

### Unique Differentiators
1. **GPU-first architecture** — 10-100× faster than CPU ABMs
2. **Mechanistic fidelity** — Real biochemical networks, not phenomenology
3. **Immune integration** — Self/non-self recognition from first principles
4. **Therapy prediction** — PK/PD + resistance emergence
5. **Open science** — SBML import, Zarr export, reproducible configs

## Integration with Cogs Platform

### Shared Architecture
Both cognisom and Cogs use:
- **pgvector** for relational/semantic memory
- **NVIDIA GPU acceleration** (H100 for simulation, Jetson for embodiment)
- **Microservices architecture** (Docker, FastAPI)
- **Communication models** — Cellular signaling ↔ Human interaction

### Unified Vision
**Understanding communication from cells to minds**
- cognisom: How cells communicate and recognize each other
- Cogs: How humans and AI communicate and form relationships

## Repository Structure
```
cognisom/
├── engine/
│   ├── cuda/              # GPU kernels (SSA, PDE, FBA)
│   ├── cpp/               # C++ bindings
│   └── py/                # Python API, schedulers
├── models/
│   ├── pathways/          # SBML pathway definitions
│   ├── metabolism/        # Genome-scale metabolic models
│   └── presets/           # Cell type configurations
├── immune/
│   ├── agents/            # NK, T-cell, macrophage models
│   ├── recognition/       # MHC-I, TCR, NK receptor logic
│   └── cytokines/         # Signaling field definitions
├── cancer/
│   ├── prostate/          # Prostate cancer specific models
│   ├── pancreatic/        # PDAC models
│   └── mutations/         # Clonal evolution logic
├── ml/
│   ├── surrogates/        # Neural network surrogates
│   └── training/          # Training scripts
├── io/
│   ├── sbml_import.py     # SBML parser
│   └── storage.py         # Zarr/HDF5 handlers
├── spatial/
│   ├── diffusion/         # PDE solvers
│   └── domain/            # Multi-GPU decomposition
├── tests/
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── benchmarks/        # Validation vs literature
├── docs/
│   ├── biology/           # Biological specifications
│   ├── architecture/      # Technical design docs
│   └── validation/        # Calibration & validation
├── funding/
│   ├── grants/            # Grant applications
│   ├── pitch/             # Pitch decks
│   └── budgets/           # Cost breakdowns
└── examples/
    ├── single_cell/       # Single cell demos
    ├── spheroid/          # Tumor spheroid growth
    └── immune_escape/     # Cancer immune evasion
```

## VCell Parity: GPU-Accelerated Solvers

Cognisom achieves feature parity with [VCell](https://vcell.org) with 5 GPU-accelerated solver types:

| Solver | VCell Equivalent | GPU Speedup | Key File |
|--------|------------------|-------------|----------|
| **ODE Solver** | CVODE | 10-50× | `cognisom/gpu/ode_solver.py` |
| **Smoldyn Spatial** | Smoldyn | 20-100× | `cognisom/gpu/smoldyn_solver.py` |
| **Hybrid ODE/SSA** | Hybrid Solvers | 5-20× | `cognisom/gpu/hybrid_solver.py` |
| **BNGL Rules** | BioNetGen | ~1× (rule parsing) | `cognisom/bngl/` |
| **Imaging Pipeline** | Image-based | 10-50× | `cognisom/imaging/` |

### ODE Solver — Batched Deterministic Integration
GPU-accelerated ODE integration for simulating **thousands of cells in parallel**:
```python
from cognisom.gpu.ode_solver import BatchedODEIntegrator, ODESystem

system = ODESystem.gene_expression_2species()
solver = BatchedODEIntegrator(system, n_cells=10000, method='rk45')
solution = solver.integrate(t_span=(0, 10), y0=y0)
```
- **Methods**: RK45, BDF (stiff), Adams-Moulton
- **Heterogeneity**: Per-cell parameter randomization
- **Use Cases**: Gene regulatory networks, parameter sensitivity, drug response

### Smoldyn Spatial — Particle Brownian Dynamics
Simulate **individual molecules** diffusing in 3D with bimolecular reactions:
```python
from cognisom.gpu.smoldyn_solver import SmoldynSolver, SmoldynSystem, SmoldynSpecies

species = [SmoldynSpecies(name='A', diffusion_coeff=1.0)]
system = SmoldynSystem(species=species, reactions=[], compartment=compartment)
solver = SmoldynSolver(system, n_max_particles=100000)
solver.add_particles('A', positions)
solver.step(dt)
```
- **Scale**: 100K+ particles on GPU
- **Features**: Reflective/absorbing boundaries, bimolecular reactions
- **Use Cases**: Receptor-ligand kinetics, single-molecule tracking, spatial patterning

### Hybrid ODE/SSA — Automatic Partitioning
Combines **deterministic ODE** for high-copy species with **stochastic SSA** for low-copy:
```python
from cognisom.gpu.hybrid_solver import HybridSolver, HybridSystem

system = HybridSystem.gene_regulatory_network()
solver = HybridSolver(system, n_cells=5000, threshold=100)
solver.initialize()
solver.step(dt)
```
- **Partitioning**: Haseltine-Rawlings automatic fast/slow separation
- **Dynamic**: Species repartitioned as populations change
- **Use Cases**: Gene expression with transcription bursts, mixed abundance systems

### BNGL Rules — Combinatorial Complexity
Handle **combinatorial complexity** in signaling pathways using reaction rules:
```python
from cognisom.bngl import BNGLModel, BNGLParser

model = BNGLModel.egfr_signaling()
# or parse from file:
parser = BNGLParser()
model = parser.parse_file("model.bngl")
```
- **Features**: Molecule types with components/states, rule expansion
- **Observables**: Pattern-based counting (Molecules, Species)
- **Use Cases**: Receptor signaling, phosphorylation cascades, protein networks

### Imaging Pipeline — Image to Geometry
Convert **microscopy images** into simulation-ready geometries:
```python
from cognisom.imaging import CellSegmenter, MeshGenerator, GPUImageProcessor

proc = GPUImageProcessor()
blurred = proc.gaussian_blur(image, sigma=2.0)
binary = proc.threshold_otsu(blurred)

segmenter = CellSegmenter(method='watershed')
result = segmenter.segment(image)  # Returns SegmentationResult

generator = MeshGenerator(resolution=0.5)
mesh = generator.labels_to_mesh(result.labels)  # Returns SimulationMesh
```
- **Formats**: TIFF, OME-TIFF, CZI (Zeiss), ND2 (Nikon), PNG/JPEG
- **Methods**: Otsu, watershed, Cellpose, StarDist (if installed)
- **Output**: 3D mesh with compartments for spatial simulations

### Dashboard Access
All VCell solvers are accessible via the Streamlit dashboard:
- **Page 20**: VCell Solvers — Interactive configuration and visualization
- **URL**: `http://localhost:8501` or your Brev deployment URL

### Integration with Entity Model
VCell solvers integrate with Cognisom's entity model for data management:
- **`ParameterSet`**: Store kinetic parameters as entities
- **`SimulationScenario`**: Define complete simulation setups
- **`PhysicsModelEntity`**: Reference specific solver configurations

```python
from cognisom.library.models import SimulationScenario, ParameterSet

params = ParameterSet(
    name="GRN_baseline",
    context="gene_regulatory_network",
    parameters={"k_transcription": 1.0, "gamma_mrna": 0.1}
)
scenario = SimulationScenario(
    name="GRN_1000_cells",
    duration_hours=24.0,
    parameter_set_ids=[params.entity_id],
)
```

---

## Getting Started

### Prerequisites
- NVIDIA GPU (RTX 4090 or better recommended, or cloud L40S/H100)
- CUDA 12.0+
- Python 3.10+
- Docker & Docker Compose

### Quick Start
```bash
# Clone repository
git clone https://github.com/eyentelligence/cognisom.git
cd cognisom

# Build Docker containers
docker-compose build

# Run single-cell demo
python examples/single_cell/basic_growth.py

# Run tests
pytest tests/
```

## Citation
If you use cognisom in your research, please cite:
```
@software{cognisom2025,
  title = {cognisom: GPU-Accelerated Cellular Simulation Platform},
  author = {eyentelligence},
  year = {2025},
  url = {https://github.com/eyentelligence/cognisom}
}
```

## License
MIT License — Open science, open source

## Contact
- **Website**: https://eyentelligence.ai
- **Email**: research@eyentelligence.ai
- **GitHub**: https://github.com/eyentelligence

---

*Understanding communication from cells to minds.*
