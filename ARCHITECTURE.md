# cognisom Technical Architecture

## System Overview

cognisom is a GPU-accelerated, mechanistic cellular simulation platform designed to model biological systems from single cells to tissue-scale multicellular dynamics, with a focus on cancer-immune interactions.

## Core Design Principles

### 1. GPU-First Architecture
- All performance-critical paths run on CUDA
- Batched operations across thousands of cells
- Structure-of-arrays memory layout
- Mixed precision where safe (FP32/FP16/INT32)

### 2. Modular & Composable
- Each biological subsystem is a swappable module
- Clean interfaces between layers
- Independent testing and validation
- Standards-based I/O (SBML, CellML, HDF5, Zarr)

### 3. Hybrid Physics + AI
- Start with mechanistic models (stochastic chemistry, FBA, PDEs)
- Progressively introduce ML surrogates for hot spots
- Periodic exact correction to prevent drift
- Error monitors enforce accuracy bounds

### 4. Deterministic Envelopes + Stochastic Cores
- Deterministic ODE/PDE where valid (high copy numbers)
- SSA/tau-leaping for low-copy-number species
- Reproducible via seed control
- Checkpoint/restore for long runs

## System Layers

### Layer 1: Intracellular Engine (Per-Cell)

#### Reaction Network (RN)
**Purpose**: Model biochemical reactions inside each cell

**Implementation**:
- Stochastic Simulation Algorithm (Gillespie SSA)
- Tau-leaping for faster approximate sampling
- GPU batching across N cells in parallel
- Species: 2,000-8,000 molecular species per cell
- Reactions: 5,000-20,000 reactions

**Data Structure**:
```cuda
struct CellState {
    int32_t species[MAX_SPECIES];      // Molecular counts
    uint16_t param_indices[MAX_PARAMS]; // Compact parameter refs
    uint8_t phase;                      // Cell cycle phase
    uint8_t stress_level;               // Cumulative stress
    uint32_t mutation_flags;            // Bitset of active mutations
    float32_t mhc1_expression;          // MHC-I surface level
    float32_t pdl1_expression;          // PD-L1 checkpoint level
};
```

**CUDA Kernel Strategy**:
- One warp per cell (32 threads)
- Shared memory for reaction propensities
- Atomic updates for species counts
- Event binning to reduce divergence

#### Central Dogma Module
**Purpose**: Transcription, translation, protein folding, degradation

**Components**:
1. **Transcription**
   - Promoter states (active/repressed)
   - RNA polymerase elongation (stochastic)
   - Transcription factor binding
   
2. **Splicing**
   - Rule-based alternative splicing
   - Isoform generation
   
3. **Translation**
   - Ribosome queueing model
   - Codon-specific elongation rates
   - Protein folding kinetics (coarse-grained)
   
4. **Degradation**
   - RNase-mediated mRNA decay
   - Proteasome-mediated protein degradation
   - Ubiquitin tagging

#### Metabolism Module
**Purpose**: Energy production, biosynthesis, metabolic flux

**Implementation**:
- Genome-scale metabolic model (GSM) or core model
- Dynamic Flux Balance Analysis (dFBA)
- GPU-accelerated linear programming solver
- Exchange fluxes with extracellular media

**Key Metabolic Pathways**:
- Glycolysis
- TCA cycle
- Oxidative phosphorylation
- Lactate shuttle (Warburg effect)
- Amino acid synthesis
- Nucleotide synthesis

**Constraints**:
```python
# Stoichiometric matrix S: m reactions × n metabolites
# Flux vector v: reaction rates
# S·v = 0  (steady-state)
# v_min ≤ v ≤ v_max  (bounds)
# Maximize: c^T·v  (objective, e.g., biomass)
```

#### Cell Cycle & Fate Module
**Purpose**: Cell division, death, senescence decisions

**Cell Cycle Checkpoints**:
1. **G1/S**: DNA integrity, growth signals
2. **G2/M**: DNA replication completion
3. **Metaphase**: Chromosome alignment
4. **Restriction Point**: Commitment to division

**Fate Decisions**:
- **Proliferation**: Pass checkpoints → divide
- **Apoptosis**: Intrinsic (mitochondrial) or extrinsic (death receptor)
- **Senescence**: Permanent growth arrest
- **Necrosis**: Uncontrolled death (releases DAMPs)

**Key Pathways**:
- p53 DNA damage response
- Rb/E2F cell cycle control
- Caspase cascade (apoptosis)
- BCL-2 family (apoptosis regulation)

#### Antigen Presentation Module
**Purpose**: MHC-I peptide loading and surface expression

**Pipeline**:
1. **Protein Degradation** → peptide pool
2. **TAP Transport** → ER lumen
3. **MHC-I Loading** → peptide-MHC complex
4. **Surface Expression** → immune recognition

**Neoantigen Generation**:
- Mutated proteins generate novel peptides
- Binding affinity prediction (netMHCpan-like)
- Immunogenicity scoring

**Immune Evasion Mechanisms**:
- MHC-I downregulation (loss of HLA genes)
- β2-microglobulin loss
- TAP transporter defects
- Interferon pathway defects

### Layer 2: Intercellular Communication

#### Spatial Grid
**Purpose**: Diffusion of soluble factors

**Implementation**:
- 2D/3D regular lattice (voxels)
- CUDA PDE solvers (finite difference stencil)
- Explicit or implicit time-stepping
- Multi-GPU domain decomposition

**Diffusing Species**:
- **Nutrients**: O₂, glucose, amino acids
- **Waste**: Lactate, CO₂, ammonia
- **Cytokines**: IFN-γ, IL-2, IL-10, TGF-β
- **Chemokines**: CXCL9, CXCL10, CCL2, CCL5
- **Growth Factors**: VEGF, EGF, TGF-α
- **Drugs**: Chemotherapy, checkpoint inhibitors

**Diffusion Equation**:
```
∂C/∂t = D·∇²C - k_decay·C + S(x,y,z,t)
```
where:
- C: concentration
- D: diffusion coefficient
- k_decay: degradation rate
- S: source term (cell secretion)

#### Cell-Cell Communication
**Purpose**: Direct signaling between adjacent cells

**Mechanisms**:
1. **Paracrine**: Secreted factors (via spatial grid)
2. **Juxtacrine**: Direct contact (Notch/Delta, cadherins)
3. **Gap Junctions**: Small molecule exchange (optional, later)

**Secretion/Uptake**:
- Each cell has secretion rates per species
- Uptake modeled as Michaelis-Menten kinetics
- Deposited into local voxel
- Read from local voxel concentration

### Layer 3: Immune System

#### Immune Agent Types

##### NK (Natural Killer) Cells
**Recognition Logic**:
```python
def nk_activation(target_cell):
    # Missing-self detection
    mhc1_signal = target_cell.mhc1_expression
    inhibition = mhc1_signal * KIR_affinity
    
    # Stress ligand detection
    stress_signal = target_cell.stress_ligands  # NKG2D, etc.
    activation = stress_signal * NKG2D_affinity
    
    # Net activation
    net = activation - inhibition
    return net > threshold
```

**Killing Mechanism**:
- Perforin/granzyme release
- Stochastic dwell time
- Serial killing capacity (limited)

##### CD8 T Cells
**Recognition Logic**:
```python
def tcr_recognition(target_cell, tcr_specificity):
    # Sample presented peptides
    for peptide in target_cell.mhc1_peptides:
        if matches(peptide, tcr_specificity):
            # Check co-stimulation
            if target_cell.cd80_cd86 > threshold:
                # Check checkpoint
                if not (target_cell.pdl1 * tcell.pd1 > inhibit_threshold):
                    return ACTIVATED
    return NOT_ACTIVATED
```

**States**:
- Naive → Primed (by DC)
- Effector → Killing
- Memory → Long-lived
- Exhausted → PD-1^high, dysfunctional

##### Macrophages
**Polarization**:
- **M1** (pro-inflammatory): IFN-γ, LPS → TNF-α, IL-12, ROS
- **M2** (anti-inflammatory): IL-4, IL-13 → IL-10, TGF-β, arginase

**Functions**:
- Phagocytosis of apoptotic/necrotic cells
- Antigen presentation (MHC-II, not modeled initially)
- Cytokine secretion
- Tissue remodeling

##### Dendritic Cells
**Function**:
- Pickup antigens from dead/dying cells
- Migrate to lymph node (abstracted as delay)
- Prime naive T cells
- Expand T-cell clones

**Simplified Model**:
```python
# When DC encounters tumor antigen:
delay = 3-7 days  # Migration + priming
after(delay):
    spawn_effector_tcells(specificity=tumor_antigen, count=N)
```

#### Cytokine Network
**Key Cytokines**:
- **IFN-γ**: T-cell/NK secretion → M1 polarization, MHC-I upregulation
- **IL-2**: T-cell growth factor
- **IL-10**: Immunosuppressive (Treg, M2)
- **TGF-β**: Immunosuppressive, fibrosis
- **TNF-α**: Pro-inflammatory, apoptosis
- **IL-12**: NK/T-cell activation

**Cytokine Effects**:
- Modulate gene expression (transcription rates)
- Affect cell migration (chemotaxis)
- Polarize immune cells
- Induce/inhibit apoptosis

### Layer 4: Cancer-Specific Modules

#### Clonal Evolution
**Mutation Model**:
- Per-division mutation probability
- Driver mutations (fitness advantage)
- Passenger mutations (neutral)
- Mutation types: SNV, indel, CNV, LOH

**Fitness Effects**:
- Proliferation rate
- Apoptosis resistance
- Metabolic efficiency
- Immune evasion
- Drug resistance

**Selection Dynamics**:
- Clonal expansion under selection pressure
- Branching evolution
- Convergent evolution (same phenotype, different genotypes)

#### Immune Evasion Strategies
1. **Antigen Loss**
   - Loss of immunogenic neoantigens
   - HLA loss of heterozygosity
   
2. **MHC-I Downregulation**
   - β2-microglobulin loss
   - TAP deficiency
   - HLA gene silencing
   
3. **Checkpoint Upregulation**
   - PD-L1 overexpression
   - CTLA-4 ligands
   
4. **Immunosuppressive Microenvironment**
   - TGF-β secretion
   - IL-10 secretion
   - Recruitment of Tregs, MDSCs
   - M2 macrophage polarization
   
5. **Fas/FasL Counterattack**
   - Express FasL → kill T cells

#### Tumor Microenvironment (TME)
**Components**:
- **Tumor Cells**: Heterogeneous clones
- **Stromal Cells**: Fibroblasts (CAFs), endothelial
- **Immune Cells**: T cells, NK, macrophages, DCs, Tregs
- **ECM**: Collagen, fibronectin (affects diffusion)
- **Vasculature**: Oxygen/nutrient delivery (simplified)

**Gradients**:
- **Hypoxia**: Low O₂ → HIF-1α → VEGF, glycolysis, EMT
- **Acidity**: High lactate → pH drop → immune suppression
- **Nutrient Depletion**: Low glucose → metabolic competition

**Spatial Patterns**:
- Proliferative rim (high O₂, nutrients)
- Quiescent zone (moderate hypoxia)
- Necrotic core (severe hypoxia, no nutrients)

#### Therapy Simulation
**Pharmacokinetics (PK)**:
- Drug concentration in blood (compartment model)
- Diffusion into tissue (PDE)
- Cellular uptake (Michaelis-Menten)

**Pharmacodynamics (PD)**:
- Drug effect on target (e.g., DNA damage, AR inhibition)
- Dose-response curves
- Combination effects (synergy, antagonism)

**Resistance Mechanisms**:
- **Intrinsic**: Pre-existing resistant clones
- **Acquired**: Mutation under selection
- **Adaptive**: Reversible phenotypic changes

**Therapy Types**:
1. **Cytotoxic Chemo**: DNA damage → apoptosis
2. **Targeted Therapy**: Specific pathway inhibition
3. **Hormone Therapy**: AR blockade (prostate)
4. **Immunotherapy**: Checkpoint inhibitors, CAR-T
5. **Radiation**: DNA double-strand breaks

## Scaling Strategy

### Within-GPU Optimization
1. **Batched SSA**: Process thousands of cells in parallel
2. **Event Binning**: Group cells by next event time
3. **Tau-Leaping**: Approximate multiple reactions per step
4. **Shared Memory**: Reaction propensity lookup tables
5. **Warp-Level Primitives**: Reduce, scan, ballot
6. **Mixed Precision**: FP16 for non-critical paths

### Multi-GPU Scaling
1. **Domain Decomposition**: Spatial partitioning
2. **Halo Exchange**: Ghost cells at boundaries
3. **Overlap Compute/Comm**: Async NCCL/MPI
4. **Load Balancing**: Dynamic repartitioning
5. **Compression**: Quantize halo data

### ML Surrogate Acceleration
**Target**: 3-10× speedup on hot kernels

**Approach**:
1. **Profile**: Identify 10% of code taking 90% of time
2. **Train**: Learn surrogate on exact solver traces
3. **Deploy**: Replace hot kernel with surrogate
4. **Correct**: Periodic exact solver + error check
5. **Fallback**: Revert to exact if error exceeds threshold

**Surrogate Types**:
- **RN Surrogate**: GNN/RNN for stiff signaling networks
- **FBA Surrogate**: MLP for common metabolic states
- **PDE Surrogate**: UNet for diffusion stepping

## Data Management

### State Representation
**Per-Cell State**: ~8-32 KB
- Species counts: 2-8k × 4 bytes = 8-32 KB
- Parameters: compact indices
- Flags: phase, stress, mutations

**Spatial Grid**: ~1-10 GB
- Fields: Nx × Ny × Nz × n_species × 4 bytes
- Example: 512³ × 10 × 4 = 5.4 GB

**Total Memory**:
- 1M cells × 16 KB = 16 GB
- Grid: 5 GB
- **Total: ~21 GB** (fits on H100 80GB with headroom)

### Checkpointing
**Format**: Zarr (chunked, compressed HDF5-like)

**Strategy**:
- Full checkpoint every N steps
- Differential snapshots between
- Metadata: config, seed, timestamp
- Reproducibility: exact state restore

**Compression**:
- Blosc/Zstd for species counts
- Quantization for fields (FP32 → FP16)

### Metrics & Logging
**Per-Step Metrics**:
- Cell counts by type/phase
- Field statistics (mean, min, max)
- Event counts (division, death, mutation)
- Immune interactions (kills, activations)

**Output**:
- Parquet files (columnar, queryable)
- Time-series database (InfluxDB/Prometheus)
- MLflow for experiment tracking

## Validation Framework

### Unit Tests
1. **Conservation Laws**
   - Mass conservation in reactions
   - Energy conservation in metabolism
   
2. **Steady-State Checks**
   - Known equilibria (e.g., lac operon)
   
3. **SSA vs ODE Agreement**
   - High copy numbers should match ODE
   
4. **Diffusion Accuracy**
   - Analytical solutions for simple geometries

### Biological Benchmarks
1. **Pathway Dynamics**
   - MAPK cascade (pulses, oscillations)
   - p53 oscillations under DNA damage
   - Apoptosis kinetics (caspase cascade)
   
2. **Cell-Level Phenotypes**
   - Doubling time (18-36h for epithelial)
   - Apoptosis rate under stress
   - Metabolic flux distributions
   
3. **Tissue-Level Patterns**
   - Tumor spheroid growth curves
   - Hypoxia gradient formation
   - Necrotic core emergence
   - Immune infiltration patterns

### Cross-Scale Validation
- **Intracellular → Phenotype**: Gene expression → growth rate
- **Phenotype → Tissue**: Cell behavior → spatial patterns
- **Tissue → Clinical**: Tumor dynamics → survival curves

## Performance Targets

### Throughput
- **Single Cell**: 1-10 ms/step (1000 steps/sec)
- **1k Cells**: 10-100 ms/step
- **100k Cells**: 1-10 sec/step
- **1M Cells**: 10-100 sec/step (with multi-GPU)

### Scaling Efficiency
- **Within-GPU**: 80-95% of peak (batching)
- **Multi-GPU**: 70-85% (communication overhead)
- **ML Surrogates**: 3-10× speedup on hot paths

### Memory Efficiency
- **Per-Cell**: 8-32 KB (target: 16 KB average)
- **1M Cells**: 16 GB (fits on single H100)
- **10M Cells**: 160 GB (2× H100 or compression)

## Technology Stack

### Core Compute
- **CUDA**: 12.0+
- **cuBLAS**: Dense linear algebra
- **cuSPARSE**: Sparse matrices
- **cuSOLVER**: Linear programming (FBA)
- **NCCL**: Multi-GPU communication

### Languages
- **C++/CUDA**: Performance-critical kernels
- **Python**: API, orchestration, analysis
- **JAX** (optional): Fast prototyping, autodiff

### I/O & Data
- **SBML/CellML**: Pathway import
- **Zarr**: Checkpoints
- **HDF5**: Legacy compatibility
- **Parquet**: Metrics/logs
- **PostgreSQL + pgvector**: Metadata, embeddings

### Orchestration
- **Docker**: Containerization
- **Docker Compose**: Local dev
- **Kubernetes** (optional): Cluster deployment
- **MLflow**: Experiment tracking
- **DVC**: Data versioning

### Visualization
- **Jupyter**: Interactive analysis
- **Plotly/Altair**: 2D plots
- **napari/ParaView**: 3D volumes
- **FastAPI + React**: Web dashboard

## Development Workflow

### Phase 1: Prototype (Months 0-3)
1. Implement GPU SSA (batched)
2. Add transcription/translation/degradation
3. Minimal metabolism (toy FBA)
4. Unit tests + benchmarks
5. Single-cell validation

### Phase 2: Biology (Months 3-6)
1. DNA damage/p53/apoptosis
2. Full dFBA with core metabolic network
3. 2D diffusion (O₂, glucose)
4. Small colony growth (10-1000 cells)
5. Calibration to literature

### Phase 3: Immune (Months 6-9)
1. MHC-I presentation
2. NK + CD8 T-cell agents
3. Cytokine fields
4. Immune surveillance → escape
5. 3D spatial grid

### Phase 4: Cancer (Months 9-12)
1. Clonal evolution
2. Immune evasion mechanisms
3. Prostate cancer pathways
4. ADT + checkpoint blockade
5. Multi-GPU scaling

### Phase 5: Scale (Months 12-18)
1. Million-cell benchmarks
2. ML surrogates
3. PDAC extension
4. Validation study
5. Open-source release

## Risk Mitigation

### Technical Risks
1. **Performance Cliffs**
   - Mitigation: Tight profiling, mixed precision, surrogates
   
2. **Biological Complexity**
   - Mitigation: Modular design, swappable components
   
3. **Validation Debt**
   - Mitigation: Unit tests first, golden benchmarks
   
4. **Data Sprawl**
   - Mitigation: Strict versioning, MLflow tracking

### Scientific Risks
1. **Parameter Uncertainty**
   - Mitigation: Bayesian inference, sensitivity analysis
   
2. **Model Validity**
   - Mitigation: Cross-scale validation, expert review
   
3. **Scope Creep**
   - Mitigation: Lock module APIs, phased releases

## Future Extensions

### Year 2+
1. **Adaptive Immunity**: B cells, antibodies, germinal centers
2. **Vasculature**: Angiogenesis, blood flow, oxygen delivery
3. **Mechanics**: Cell deformability, ECM remodeling, migration
4. **Spatial Heterogeneity**: Tissue architecture, organ-level
5. **Patient-Specific**: Omics integration, personalized models
6. **Clinical Trials**: Virtual trial simulation, drug optimization

### Integration with Cogs
- **Shared Memory Architecture**: pgvector for both platforms
- **Unified Communication Models**: Cell signaling ↔ Human interaction
- **Embodied Simulation**: Cogs as interface to cognisom results
- **Educational Tools**: Interactive cellular biology teaching

---

*This architecture enables mechanistic understanding of cellular communication from first principles, accelerated by GPU computing and guided by biological validation.*
