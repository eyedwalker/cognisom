# cognisom Quick Start Guide

## Get Started in 15 Minutes

This guide will help you set up your development environment and understand the project structure.

---

## Prerequisites

### Hardware
- **Minimum**: NVIDIA GPU with 8GB+ VRAM (GTX 1080, RTX 3060, etc.)
- **Recommended**: RTX 4090 (24GB) or better
- **RAM**: 16GB+ system RAM
- **Storage**: 50GB+ free space (for datasets, checkpoints)

### Software
- **OS**: Linux (Ubuntu 22.04+), macOS, or Windows with WSL2
- **CUDA**: 12.0+ ([Download](https://developer.nvidia.com/cuda-downloads))
- **Python**: 3.10+ ([Download](https://www.python.org/downloads/))
- **Docker**: Optional but recommended ([Download](https://www.docker.com/))
- **Git**: For version control

---

## Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/eyentelligence/cognisom.git
cd cognisom
```

### Step 2: Create Python Environment
```bash
# Using venv
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n biosim python=3.10
conda activate biosim
```

### Step 3: Install Dependencies
```bash
# Core dependencies
pip install -r requirements.txt

# Development dependencies (optional)
pip install -r requirements-dev.txt
```

### Step 4: Verify CUDA Installation
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

Expected output:
```
CUDA available: True
CUDA version: 12.1
GPU: NVIDIA GeForce RTX 4090
```

---

## Project Structure

```
cognisom/
├── README.md                 # Main project documentation
├── ARCHITECTURE.md           # Technical architecture details
├── QUICKSTART.md            # This file
├── requirements.txt         # Python dependencies
├── setup.py                 # Package installation
│
├── engine/                  # Core simulation engine
│   ├── cuda/               # CUDA kernels
│   │   ├── ssa.cu         # Stochastic simulation algorithm
│   │   ├── diffusion.cu   # Reaction-diffusion PDE solver
│   │   └── fba.cu         # Flux balance analysis
│   ├── cpp/                # C++ bindings
│   └── py/                 # Python API
│       ├── cell.py        # Cell state management
│       ├── scheduler.py   # Event scheduler
│       └── simulation.py  # Main simulation loop
│
├── models/                  # Biological models
│   ├── pathways/           # SBML pathway definitions
│   │   ├── mapk.xml       # MAPK signaling
│   │   ├── p53.xml        # p53 DNA damage response
│   │   └── apoptosis.xml  # Apoptosis cascade
│   ├── metabolism/         # Metabolic models
│   │   ├── core_ecoli.xml # E. coli core model (test)
│   │   └── recon3d.xml    # Human metabolism (large)
│   └── presets/            # Cell type configurations
│       ├── epithelial.yaml
│       ├── fibroblast.yaml
│       └── immune.yaml
│
├── immune/                  # Immune system models
│   ├── agents/             # Immune cell types
│   │   ├── nk_cell.py     # Natural killer cells
│   │   ├── tcell.py       # T cells (CD8, CD4)
│   │   └── macrophage.py  # Macrophages
│   ├── recognition/        # Recognition logic
│   │   ├── mhc1.py        # MHC-I presentation
│   │   ├── tcr.py         # T-cell receptor
│   │   └── nk_receptors.py # NK activating/inhibitory
│   └── cytokines/          # Cytokine definitions
│       └── fields.yaml
│
├── cancer/                  # Cancer-specific modules
│   ├── prostate/           # Prostate cancer
│   │   ├── pathways.yaml  # PTEN, TP53, AR
│   │   └── therapy.py     # ADT, enzalutamide
│   ├── pancreatic/         # Pancreatic cancer
│   │   ├── pdac.yaml      # KRAS, TP53, SMAD4
│   │   └── stroma.py      # CAFs, ECM
│   └── mutations/          # Mutation models
│       └── clonal_evolution.py
│
├── spatial/                 # Spatial simulation
│   ├── diffusion/          # PDE solvers
│   │   ├── explicit.py    # Explicit time-stepping
│   │   └── implicit.py    # Implicit (Crank-Nicolson)
│   └── domain/             # Multi-GPU decomposition
│       └── partition.py
│
├── ml/                      # Machine learning surrogates
│   ├── surrogates/         # Trained models
│   │   ├── rn_gnn.py      # Reaction network GNN
│   │   ├── fba_mlp.py     # FBA surrogate
│   │   └── pde_unet.py    # Diffusion UNet
│   └── training/           # Training scripts
│       └── train_surrogate.py
│
├── io/                      # Input/output
│   ├── sbml_import.py      # SBML parser
│   ├── storage.py          # Zarr/HDF5 handlers
│   └── metrics.py          # Logging & telemetry
│
├── tests/                   # Test suite
│   ├── unit/               # Unit tests
│   │   ├── test_ssa.py
│   │   ├── test_diffusion.py
│   │   └── test_fba.py
│   ├── integration/        # Integration tests
│   │   └── test_single_cell.py
│   └── benchmarks/         # Validation benchmarks
│       ├── mapk_validation.py
│       └── spheroid_growth.py
│
├── examples/                # Example simulations
│   ├── single_cell/        # Single cell demos
│   │   ├── basic_growth.py
│   │   └── stress_response.py
│   ├── spheroid/           # Tumor spheroid
│   │   └── hypoxia_gradient.py
│   └── immune_escape/      # Cancer immune evasion
│       └── nk_escape.py
│
├── docs/                    # Documentation
│   ├── biology/            # Biological specifications
│   ├── architecture/       # Technical design
│   └── tutorials/          # User guides
│
├── funding/                 # Grant applications
│   ├── NVIDIA_APPLICATION.md
│   ├── GRANT_TARGETS.md
│   ├── PITCH_DECK_CONTENT.md
│   └── eyentelligence_pitch_deck_B2.pptx
│
└── scripts/                 # Utility scripts
    ├── setup_env.sh        # Environment setup
    ├── run_tests.sh        # Run test suite
    └── benchmark.sh        # Performance benchmarks
```

---

## Your First Simulation

### Example 1: Single Cell Growth

Create a file `my_first_sim.py`:

```python
from biosim import Cell, Simulation
from biosim.models import load_preset

# Load epithelial cell preset
cell_config = load_preset('epithelial')

# Create a single cell
cell = Cell(config=cell_config)

# Initialize simulation
sim = Simulation(
    cells=[cell],
    duration=24.0,  # hours
    dt=0.01,        # time step (hours)
    output_dir='./output/first_sim'
)

# Run simulation
print("Starting simulation...")
sim.run()

# Analyze results
results = sim.get_results()
print(f"Final cell count: {len(results.cells)}")
print(f"Divisions: {results.events['division']}")
print(f"Deaths: {results.events['death']}")

# Plot growth curve
import matplotlib.pyplot as plt
plt.plot(results.time, results.cell_count)
plt.xlabel('Time (hours)')
plt.ylabel('Cell Count')
plt.title('Single Cell Growth')
plt.savefig('./output/first_sim/growth_curve.png')
print("✓ Results saved to ./output/first_sim/")
```

Run it:
```bash
python my_first_sim.py
```

Expected output:
```
Starting simulation...
[████████████████████] 100% | 2400/2400 steps | 12.3s
Final cell count: 2
Divisions: 1
Deaths: 0
✓ Results saved to ./output/first_sim/
```

---

### Example 2: Tumor Spheroid with Hypoxia

```python
from biosim import Simulation, SpatialGrid
from biosim.models import load_preset
from biosim.cancer.prostate import ProstateCancerCell

# Create spatial grid (3D)
grid = SpatialGrid(
    size=(100, 100, 100),  # 100x100x100 voxels
    voxel_size=10.0,       # 10 microns per voxel
    species=['O2', 'glucose', 'lactate']
)

# Initialize with oxygen and glucose
grid.set_field('O2', 0.2)      # 20% oxygen (normoxia)
grid.set_field('glucose', 5.0)  # 5 mM glucose

# Create initial tumor cell at center
center = (50, 50, 50)
initial_cell = ProstateCancerCell(
    position=center,
    mutations=['PTEN_loss']  # Start with PTEN loss
)

# Run simulation
sim = Simulation(
    cells=[initial_cell],
    grid=grid,
    duration=240.0,  # 10 days
    dt=0.1,
    output_dir='./output/spheroid'
)

print("Growing tumor spheroid...")
sim.run()

# Analyze spatial patterns
results = sim.get_results()
print(f"Final cell count: {len(results.cells)}")
print(f"Hypoxic cells: {results.count_hypoxic()}")
print(f"Necrotic core: {results.has_necrotic_core()}")

# Visualize oxygen gradient
results.plot_field_slice('O2', z=50, save='./output/spheroid/oxygen.png')
print("✓ Spheroid simulation complete")
```

---

### Example 3: Immune Surveillance

```python
from biosim import Simulation
from biosim.cancer.prostate import ProstateCancerCell
from biosim.immune import NKCell, CD8TCell

# Create tumor cell with mutations
tumor_cell = ProstateCancerCell(
    mutations=['TP53_loss', 'PTEN_loss'],
    mhc1_expression=0.8  # Normal MHC-I initially
)

# Create immune cells
nk_cell = NKCell(position=(10, 10, 10))
tcell = CD8TCell(
    position=(15, 15, 15),
    tcr_specificity='neoantigen_1'  # Recognizes specific neoantigen
)

# Run simulation
sim = Simulation(
    cells=[tumor_cell, nk_cell, tcell],
    duration=48.0,  # 2 days
    dt=0.01,
    track_interactions=True,
    output_dir='./output/immune_surveillance'
)

print("Simulating immune surveillance...")
sim.run()

# Analyze immune interactions
results = sim.get_results()
print(f"NK activations: {results.events['nk_activation']}")
print(f"T-cell kills: {results.events['tcell_kill']}")
print(f"Tumor escaped: {results.tumor_survived()}")

# Plot MHC-I expression over time
results.plot_marker('mhc1_expression', cell_id=0, 
                   save='./output/immune_surveillance/mhc1.png')
```

---

## Running Tests

### Unit Tests
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/unit/test_ssa.py

# Run with coverage
pytest --cov=biosim tests/
```

### Benchmarks
```bash
# Run validation benchmarks
python tests/benchmarks/mapk_validation.py
python tests/benchmarks/spheroid_growth.py

# Performance benchmarks
./scripts/benchmark.sh
```

---

## Configuration

### Cell Configuration (YAML)

Example: `models/presets/epithelial.yaml`

```yaml
name: "Generic Epithelial Cell"
species_count: 2000
reactions_count: 5000

# Cell cycle
doubling_time: 24.0  # hours
g1_duration: 11.0
s_duration: 8.0
g2_duration: 4.0
m_duration: 1.0

# Metabolism
glucose_uptake: 10.0  # mmol/gDW/hr
oxygen_uptake: 5.0
lactate_secretion: 15.0  # Warburg effect

# Immune markers
mhc1_baseline: 1.0
pdl1_baseline: 0.1
stress_ligands: 0.05

# Pathways
pathways:
  - mapk
  - p53
  - apoptosis
  - cell_cycle

# Parameters
parameters:
  k_transcription: 0.1
  k_translation: 0.5
  k_degradation_mrna: 0.05
  k_degradation_protein: 0.01
```

---

## Common Tasks

### Task 1: Add a New Pathway

1. Create SBML file in `models/pathways/my_pathway.xml`
2. Or define in Python:

```python
from biosim.models import Pathway, Reaction, Species

pathway = Pathway(name='my_pathway')

# Define species
A = Species('A', initial=100)
B = Species('B', initial=0)

# Define reaction: A -> B
reaction = Reaction(
    reactants={'A': 1},
    products={'B': 1},
    rate_constant=0.1
)

pathway.add_species([A, B])
pathway.add_reaction(reaction)

# Save to SBML
pathway.export_sbml('models/pathways/my_pathway.xml')
```

### Task 2: Customize Immune Recognition

```python
from biosim.immune import NKCell

class MyCustomNKCell(NKCell):
    def activation_logic(self, target_cell):
        """Custom NK activation logic"""
        # Missing-self detection
        mhc1_inhibition = target_cell.mhc1_expression * self.kir_affinity
        
        # Stress ligand activation
        stress_activation = target_cell.stress_ligands * self.nkg2d_affinity
        
        # Custom: also check for specific marker
        custom_signal = target_cell.get_marker('my_marker') * 0.5
        
        # Net activation
        net = stress_activation + custom_signal - mhc1_inhibition
        return net > self.activation_threshold
```

### Task 3: Run on Multiple GPUs

```python
from biosim import Simulation
from biosim.spatial import MultiGPUDomain

# Create domain decomposition
domain = MultiGPUDomain(
    grid_size=(200, 200, 200),
    num_gpus=4,  # Use 4 GPUs
    decomposition='spatial'  # or 'cell-based'
)

sim = Simulation(
    cells=initial_cells,
    domain=domain,
    duration=100.0
)

sim.run()  # Automatically distributes across GPUs
```

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution 1**: Reduce cell count or grid size
```python
# Before
grid = SpatialGrid(size=(200, 200, 200))  # Too large

# After
grid = SpatialGrid(size=(100, 100, 100))  # Smaller
```

**Solution 2**: Use mixed precision
```python
sim = Simulation(
    cells=cells,
    precision='mixed',  # Use FP16 where safe
    ...
)
```

**Solution 3**: Enable checkpointing
```python
sim = Simulation(
    cells=cells,
    checkpoint_every=1000,  # Save every 1000 steps
    checkpoint_dir='./checkpoints'
)
```

### Issue: Slow Performance

**Check 1**: Verify GPU is being used
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.current_device())  # Should show GPU
```

**Check 2**: Profile the simulation
```python
sim.run(profile=True)
sim.print_profile()  # Shows time spent in each component
```

**Check 3**: Enable tau-leaping (approximate but faster)
```python
sim = Simulation(
    cells=cells,
    ssa_method='tau_leap',  # Instead of 'exact'
    tau=0.01  # Leap size
)
```

### Issue: Results Don't Match Literature

**Check 1**: Verify parameter values
```python
# Load pathway and inspect parameters
pathway = load_pathway('mapk')
print(pathway.parameters)
```

**Check 2**: Run validation benchmark
```bash
python tests/benchmarks/mapk_validation.py
# Should show agreement with published data
```

**Check 3**: Check random seed
```python
# For reproducibility
sim = Simulation(
    cells=cells,
    seed=42  # Fixed seed
)
```

---

## Next Steps

### Learn More
1. Read `ARCHITECTURE.md` for technical details
2. Browse `examples/` for more complex simulations
3. Check `docs/tutorials/` for step-by-step guides

### Contribute
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

### Get Help
- **GitHub Issues**: https://github.com/eyentelligence/cognisom/issues
- **Email**: research@eyentelligence.ai
- **Documentation**: https://biosim.readthedocs.io (coming soon)

---

## Resources

### Biological Background
- **Cancer Biology**: Weinberg, "The Biology of Cancer"
- **Immunology**: Murphy, "Janeway's Immunobiology"
- **Systems Biology**: Alon, "An Introduction to Systems Biology"

### Computational Methods
- **Stochastic Simulation**: Gillespie (1977) "Exact stochastic simulation"
- **Flux Balance Analysis**: Orth et al. (2010) "What is flux balance analysis?"
- **GPU Computing**: NVIDIA CUDA Programming Guide

### Related Tools
- **PhysiCell**: Multicellular simulation (CPU-based)
- **VCell**: Virtual Cell modeling platform
- **COBRApy**: Constraint-based metabolic modeling
- **Lattice Microbes**: GPU reaction-diffusion

---

**Welcome to cognisom! Let's understand life at the cellular scale.** 🧬🔬💻
