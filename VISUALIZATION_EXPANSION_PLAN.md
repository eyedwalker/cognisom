# Cognisom Visualization Expansion Plan

## What You Asked

> How to see proteins interact, cellular docking, cell membrane transport,
> a single cell fully simulated, the prostate with sub-tissues, lymphatic,
> blood vessels, nerves... can you simulate how cancer metastasizes and spreads?

## Answer: Yes — and most of the simulation already exists

The 9-module engine already simulates:
- Cancer cell division, metabolism, death, and transformation
- Immune surveillance (T cells, NK cells, macrophages) with spatial tracking
- Vascular network with O2/glucose diffusion (Fick's law)
- Lymphatic drainage with **metastasis** (cancer entering lymph vessels)
- Exosome-mediated cell-to-cell communication (horizontal gene transfer)
- Epigenetic silencing of tumor suppressors
- Circadian gating of cell division and immune activity
- Morphogen gradients and cell fate determination
- 3D spatial fields with PDE diffusion

**The gap is visualization, not simulation.** The engine runs all this biology
but the dashboard currently shows mock data instead of real engine output.

---

## Phase 1: Wire Real Engine to Dashboard (Foundation)

**What**: Replace the mock `run_demo_simulation()` with the actual 9-module engine.

**Files to modify**:
- `cognisom/dashboard/pages/3_simulation.py` — connect to real engine
- New: `cognisom/dashboard/engine_runner.py` — thin wrapper to run engine and collect results

**What you'll see**:
- Real cancer growth curves (not toy ODE)
- Real immune kill events
- Real oxygen gradients from vascular network
- Real metastasis events from lymphatic module
- Real circadian gating effects
- All 9 modules running simultaneously

**Data captured per step**: cell positions, types, states, oxygen field,
immune positions, vascular network, lymphatic events, epigenetic states,
circadian phases, morphogen gradients.

**Why this matters**: Everything after this depends on having real simulation
data flowing into the dashboard. Without it, all visualization is fake.

---

## Phase 2: Interactive 3D Tissue Viewer (Core Visual)

**What**: Replace the static Plotly scatter with a rich, layered 3D tissue view.

**Technology**: Plotly 3D with layer toggles (runs in browser, no GPU needed)

**Layers (toggleable)**:
1. **Epithelial cells** — green spheres (normal) / red spheres (cancer)
2. **Stromal cells** — gray spheres (fibroblasts, smooth muscle)
3. **Immune cells** — cyan (T cells), magenta (NK), orange (macrophages)
4. **Capillary network** — red tubes connecting vessel endpoints
5. **Lymphatic network** — blue tubes with drainage direction arrows
6. **Oxygen field** — 3D isosurface or slice heatmap (blue=oxygenated, red=hypoxic)
7. **Exosomes** — tiny yellow dots moving between cells
8. **Morphogen gradients** — transparent color overlay

**Controls**:
- Time scrubber slider (step through simulation)
- Layer visibility checkboxes
- Color-by selector: cell type / cell cycle phase / oxygen level / mutations / MHC-I expression
- Click-to-select any cell (opens Phase 3 detail view)
- Animation play/pause (auto-advance time steps)

**Files**:
- New: `cognisom/dashboard/pages/6_tissue_viewer.py`
- New: `cognisom/dashboard/tissue_viz.py` — 3D rendering utilities

---

## Phase 3: Single-Cell Deep Dive

**What**: Click any cell in the 3D viewer → see its complete internal state.

**Panel contents**:

### Cell Identity
- Cell ID, type (normal/cancer/immune), age, position
- Cell cycle phase (G1/S/G2/M) with visual indicator

### Metabolism
- O2 level (gauge), glucose (gauge), ATP (gauge), lactate (gauge)
- Warburg effect indicator (cancer cells)
- Distance to nearest capillary

### Genetics & Molecular
- Active mutations (KRAS_G12D, etc.)
- Gene expression levels (TP53, KRAS, BRAF)
- MHC-I expression level (immune evasion status)
- Exosomes released / received

### Epigenetic State
- Per-gene methylation levels (bar chart)
- Histone marks (H3K4me3, H3K27me3, H3K9ac)
- Chromatin state (open/closed per gene)
- Silenced genes highlighted in red

### Circadian Clock
- Current phase (clock dial visualization)
- CLOCK/BMAL1 and PER/CRY oscillation
- Synchrony with master clock
- Division gating status (permissive/restrictive)

### Neighborhood
- Nearest immune cells (distance, type, activated?)
- Nearest capillary (distance, O2 at that point)
- Nearest lymphatic vessel (metastasis risk)
- Morphogen levels sensed (BMP, Shh, Wnt)

**Files**:
- New: `cognisom/dashboard/pages/7_cell_inspector.py`

---

## Phase 4: Protein Interactions & Membrane Transport

**What**: Visualize the molecular mechanisms that the simulation models.

### 4A: Protein-Protein Interaction Viewer

Show the actual protein complexes relevant to simulation events:

| Simulation Event | Protein Complex | PDB |
|-----------------|----------------|-----|
| Immune evasion | MHC-I + peptide | 1HHK |
| T cell killing | TCR + MHC-I/peptide | 1OGA |
| NK missing-self | KIR + MHC-I | 1IM9 |
| Immune checkpoint | PD-1 + PD-L1 | 4ZQK |
| AR signaling | AR-LBD + DHT | 2AM9 |
| Drug binding | AR + enzalutamide | 3V49 |
| DNA repair | PARP1 + DNA | 4DQY |
| Growth signaling | KRAS + GTP | 4OBE |

**Implementation**: When a simulation event fires (e.g., CANCER_KILLED),
show the relevant protein complex in py3Dmol with the interacting surfaces
highlighted. Clicking "T cell kills cancer cell" shows TCR engaging MHC-I.

### 4B: Cell Membrane Transport Visualization

Interactive cross-section showing:
- **Lipid bilayer** (schematic SVG or 3D)
- **Ion channels**: Na+/K+ ATPase, Ca2+ channels
- **Glucose transporters**: GLUT1 (upregulated in cancer/Warburg)
- **Oxygen diffusion**: Passive through membrane
- **Drug transport**: How enzalutamide enters cell
- **Exosome fusion**: Membrane fusion and cargo delivery
- **Receptor-ligand**: AR receptor binding, PD-L1 surface display

**Implementation**: SVG-based interactive diagram with Streamlit.
Click any transporter → shows 3D protein structure.
Drug molecules shown entering via their transport mechanism.

### 4C: Docking Visualization

When DiffDock results are available:
- Protein surface with binding pocket highlighted
- Ligand positioned in pocket
- Binding energy / affinity score
- Comparison of multiple poses

**Files**:
- New: `cognisom/dashboard/pages/8_molecular_interactions.py`
- Update: `cognisom/dashboard/mol_viz.py` — add membrane diagram functions

---

## Phase 5: Cancer Metastasis Visualization

**What**: Animate the complete metastatic cascade over time.

### The Metastatic Cascade (already modeled in engine):
1. **Primary tumor growth** — cancer cells dividing (cellular module)
2. **Immune evasion** — MHC-I downregulation (cellular + immune modules)
3. **Angiogenesis** — tumor needs blood supply (vascular module)
4. **Local invasion** — cancer cells at tissue boundary
5. **Lymphatic entry** — cancer cell enters lymph vessel (lymphatic module, METASTASIS_OCCURRED event)
6. **CTC transport** — circulating tumor cell in lymph/blood
7. **Distant seeding** — colonization at new site

### Visualization:
- **Timeline view**: Horizontal timeline with key events marked
  - "Day 3: First immune evasion (MHC-I < 0.2)"
  - "Day 5: Hypoxia detected in tumor core"
  - "Day 8: First metastasis — cancer cell enters lymphatic vessel #2"
- **Spatial animation**: 3D tissue view with time scrubber
  - Watch tumor grow from single cell
  - See immune cells attacking (and failing)
  - See cancer cell enter lymphatic vessel (highlighted moment)
- **Event log**: Scrollable list of all simulation events with filtering
- **Statistics dashboard**: Tumor doubling time, immune kill rate, metastasis count

### What Is Learned:
- **Immune escape timing**: When does the tumor outpace immune surveillance?
- **Hypoxia threshold**: At what tumor size does the core become hypoxic?
- **Metastatic window**: When do cancer cells first access lymphatics?
- **Circadian vulnerability**: Is there a time of day when immune defense is weakest?
- **Epigenetic progression**: Which tumor suppressors get silenced first?
- **Drug intervention timing**: When is treatment most effective?

**Files**:
- New: `cognisom/dashboard/pages/9_metastasis.py`

---

## Phase 6: Prostate Organ Architecture

**What**: Anatomically accurate prostate visualization with tissue layers.

### Prostate Anatomy Layers:
1. **Peripheral zone** (70% of gland, where most cancer starts)
2. **Transition zone** (BPH occurs here)
3. **Central zone** (surrounds ejaculatory ducts)
4. **Anterior fibromuscular stroma**
5. **Prostatic urethra** (central tube)

### Sub-tissue Systems:
- **Epithelium**: Luminal + basal layers (from scRNA-seq archetypes)
- **Stroma**: Fibroblasts + smooth muscle
- **Vasculature**: Arterial supply → capillary bed → venous drainage
- **Lymphatics**: Drainage to obturator & internal iliac nodes
- **Nerves**: Neurovascular bundles (cavernous nerves) — critical for surgery planning
- **Immune**: Resident macrophages, patrolling T cells

### Implementation Approach:
- **3D mesh**: Simplified prostate geometry (ellipsoid with zones)
- **Cross-section view**: 2D slice through the gland showing all layers
- **Zoom levels**:
  - Organ level: prostate shape with zones colored
  - Zone level: tissue architecture (acini, ducts, stroma)
  - Acinus level: individual gland with cell types (this is where simulation runs)
  - Cell level: single-cell deep dive (Phase 3)

### Nerve Modeling (new module needed):
The engine does NOT currently model nerves. Adding this would require:
- `cognisom/modules/nerve_module.py` — neurovascular bundle model
- Nerve fiber positions along posterolateral prostate
- Neurotransmitter signaling (acetylcholine, norepinephrine)
- Perineural invasion (cancer spreading along nerves — major clinical concern)
- This is a significant addition but medically important

**Files**:
- New: `cognisom/dashboard/pages/10_organ_view.py`
- New: `cognisom/dashboard/prostate_viz.py` — anatomical rendering
- Optional: `cognisom/modules/nerve_module.py` — nerve tissue simulation

---

## Implementation Order & Dependencies

```
Phase 1: Wire Real Engine          ← MUST DO FIRST (everything depends on this)
   │
   ├── Phase 2: 3D Tissue Viewer   ← Core visual experience
   │      │
   │      └── Phase 3: Cell Inspector  ← Click-to-inspect from tissue view
   │
   ├── Phase 5: Metastasis Viz     ← Uses real engine event data
   │
   ├── Phase 4: Protein/Membrane   ← Independent (uses py3Dmol + PDB)
   │
   └── Phase 6: Organ Architecture ← Largest scope, can be incremental
```

**Phase 1** is the critical path. Without real engine data, everything else
is just more mock visualizations.

---

## Technology Stack

| Component | Technology | Cost |
|-----------|-----------|------|
| 3D tissue rendering | Plotly 3D (already installed) | Free |
| Protein structures | py3Dmol (already installed) | Free |
| Molecule rendering | RDKit (already installed) | Free |
| Membrane diagrams | SVG + Streamlit components | Free |
| Organ anatomy mesh | Plotly 3D mesh3d traces | Free |
| Time-series animation | Plotly frames + slider | Free |
| PDB structures | RCSB PDB (public database) | Free |
| Simulation engine | Existing 9-module engine | Free |
| Dashboard | Streamlit (already running) | Free |

**Total additional cost: $0**

Everything uses tools already installed or freely available.

---

## What You Learn From Each Phase

### Phase 1 (Real Engine):
- Actual tumor growth dynamics under 9-module interaction
- Whether immune system can contain tumor at given parameters

### Phase 2 (3D Tissue):
- Spatial relationship between tumor, vessels, and immune cells
- How hypoxic regions form as tumor outgrows vasculature

### Phase 3 (Cell Inspector):
- Why specific cells survive vs die
- Which epigenetic changes drive cancer progression
- How circadian phase affects individual cell behavior

### Phase 4 (Protein Interactions):
- Molecular basis for immune evasion (MHC-I downregulation)
- How drugs physically bind to targets
- Why certain mutations confer resistance

### Phase 5 (Metastasis):
- Timing: when tumor transitions from localized to metastatic
- Route: which lymphatic vessels are invasion paths
- Vulnerability: circadian windows for treatment
- Epigenetic progression: order of tumor suppressor silencing

### Phase 6 (Organ View):
- Anatomical context for simulation results
- Surgical planning (nerve-sparing prostatectomy)
- Why peripheral zone cancer is different from transition zone
- Lymph node drainage patterns (predicts metastatic sites)

---

## Summary

| Phase | What You See | Effort | Dependencies |
|-------|-------------|--------|--------------|
| 1. Real Engine | Real biology, not fake math | Medium | None |
| 2. 3D Tissue | Interactive layered tissue view | Medium | Phase 1 |
| 3. Cell Inspector | Complete cell internal state | Small | Phase 2 |
| 4. Protein/Membrane | 3D protein complexes, membrane transport | Medium | None |
| 5. Metastasis | Animated cancer spread timeline | Medium | Phase 1 |
| 6. Organ View | Prostate anatomy with all systems | Large | Phase 1 |

The engine already simulates cancer metastasis. The work is making it visible.
