# Cognisom — Patent Disclosure Scoping Document

**Prepared for:** David Walker
**Subject:** Cognisom multi-scale cellular simulation platform
**Repository:** `/Users/davidwalker/CascadeProjects/cognisom/`
**Date:** 2026-05-10
**Purpose:** Internal scoping document for prior-art research and patent attorney review.
**Status:** First-pass technical inventory based on direct source-code review. All claims are cited to file:line. Implementation maturity is honestly graded so that filings can be sized to what is actually enabled in code.

---

## Context

Cognisom is a GPU-aware, multi-scale cellular biology simulation platform aimed at predicting cancer-immunotherapy outcomes by modeling biology mechanistically from genome → molecule → cell → tissue. Some subsystems are working code with tests; others are aspirational scaffolding described in markdown. Patent filings must be sized to what the code actually does, not what the docs claim.

This document exists to (a) give you a single artifact to hand to a patent attorney or to feed into Google Patents / Lens.org / IEEE Xplore prior-art searches, (b) clearly separate signal (real, novel, implemented) from noise (documented but not built; well-known prior art), and (c) flag what should likely be trade-secret rather than patent.

> **Critical pre-filing posture:** Roughly 20–30% of the system is production-grade implemented code; 50% is scaffolding (class shells, `update(dt)` stubs); 20–30% is documentation-only. Patent claims must track *implemented* code, or USPTO §112 enablement / utility will reject. Use this document to identify which inventions are mature enough to claim now and which need additional implementation before filing.

---

## 1. Field of Invention

Computational systems biology / biomedical simulation, specifically:

1. Multi-scale stochastic-deterministic simulation of cellular and tissue-level cancer biology.
2. GPU-accelerated batched stochastic chemistry for populations of distinguishable cells.
3. Sequence-grounded simulation of mutations and their phenotypic consequences.
4. Mechanistic immune-recognition and therapy-response prediction in silico.

---

## 2. Repository Inventory (For Reference)

| Layer | Path | LOC | Maturity |
|---|---|---|---|
| Core engine | `core/` (event_bus, module_base, simulation_engine) | ~1.5k | **Implemented + tested** (243k steps/s claim in `INTEGRATION_COMPLETE.md`) |
| Molecular engine | `engine/py/molecular/` (nucleic_acids, exosomes, receptors) | ~1.4k | **Implemented**, partial test coverage |
| Cell / intracellular | `engine/py/cell.py`, `intracellular.py`, `simulation.py` | ~950 | **Implemented** (Poisson stochastic transcription/translation; metabolic gating) |
| Spatial grid (PDE) | `engine/py/spatial/grid.py` | ~280 | **Implemented** (explicit FDM, Neumann BC, sub-stepping) |
| Modules (cellular/immune/cancer/circadian/morphogen/epigenetic/etc.) | `modules/` (15 modules) | ~5k | Mixed: data classes implemented, dynamics partly missing |
| GPU stack | `gpu/` (13 files: ssa_kernel, ode_solver, hybrid_solver, smoldyn, fba, diffusion, domain_decomposition, ...) | 6,180 | **CUDA C strings present; CPU fallback used in tests; no `.cu` files; multi-GPU comms not wired** |
| Genomics / neoantigen | `cognisom/genomics/neoantigen_predictor.py` | ~120 | **Implemented** (PWM-based MHC-I binding) |
| Visualization | `engine/py/live_visualizer.py`, `visualization/`, `omniverse/` | ~3k | Core renderer implemented; Omniverse kit is scaffolding |
| Tests | `tests/` (229 `def test_` across 8 files) | — | CPU-only; GPU not validated in CI |

---

## 3. Implemented and Patent-Relevant Inventions

For each item below: (a) what it is, (b) where the code lives, (c) what is mechanistically novel, (d) what is textbook / prior art, (e) implementation maturity, (f) draft claim sketch, (g) prior-art search terms.

### Invention A — Sequence-Grounded Mutation→Phenotype Model

**File evidence**
- `engine/py/molecular/nucleic_acids.py:42–221` — `NucleicAcid` base class. Stores actual ATCG/AUCG strings; `_calculate_gc_content()` (83–88), `_calculate_tm()` Wallace rule (90–97), `mutate(position, base)` (106–148), `is_complementary(other, threshold=0.8)` (150–180), `translate_codon()` (188–209) full 64-codon table, stochastic decay `update()` 10%/cycle (211–221).
- `engine/py/molecular/nucleic_acids.py:224–261` — `DNA`: reverse complement (238–240), `transcribe()` T→U (242–261), mutation inheritance into transcripts.
- `engine/py/molecular/nucleic_acids.py:264–329` — `RNA` (mRNA/miRNA/tRNA): start-codon ORF discovery + stop-codon termination `translate()` (291–319); miRNA::mRNA `can_bind_to(threshold=0.7)` (321–329); type-specific half-lives (mRNA 8h, miRNA 24h).
- `engine/py/molecular/nucleic_acids.py:332–426` — `Gene`: `introduce_oncogenic_mutation()` (389–426) hard-codes KRAS G12D (pos 35, GGT→GAT), G13D (pos 38), BRAF V600E (pos 1799), TP53 R175H, TP53 R248W (404–415); stochastic `transcribe()` with promoter strength (360–387).

**Mechanism (novelty axis)**
The cell-level state in cognisom carries the *actual nucleotide sequence at the mutation locus*, not a phenomenological "mutated/not mutated" boolean or "activity = 1.5×". Chemical properties (GC%, Tm, stability) are computed from the sequence; protein consequences (oncogenic activation, tumor-suppressor loss) are tied to specific codon edits. Daughter cells inherit and can further mutate the sequence.

**Why this is plausibly novel**
PhysiCell, BioDynaMo, Compucell3D, COPASI, Virtual Cell, Lattice Microbes, Gell — none of the leading platforms ground their cell-state in actual ATCG sequence with codon-level oncogenic consequences. They use lumped parameters. The combination of (sequence storage + sequence-derived chemistry + sequence-tied phenotypic consequence + multi-cell stochastic tracking) appears to be unoccupied space.

**Textbook elements that are NOT novel and must be disclaimed**
Wallace's Tm rule (1979); the standard 64-codon translation table; reverse complementation; oncogenic mutation catalogs (COSMIC, OncoKB). Claims must focus on the *integration* — using sequence as the primary state variable through which simulation propagates — not the constituents.

**Maturity:** Implemented; can be exercised in `examples/` and through `tests/test_validation.py`. **Patent-ready.**

**Draft independent claim (sketch — not legal language)**
> A computer-implemented method for simulating tumor progression, comprising: (i) instantiating, in computer memory, a population of cell objects, each cell object including a per-locus nucleotide-sequence record for one or more genes; (ii) stochastically introducing site-specific substitutions into said nucleotide-sequence record at simulated cell division; (iii) deriving, from said nucleotide-sequence record, at least one chemical property selected from GC content, melting temperature, and stability index; (iv) deriving a phenotypic modifier — applied to a transcription rate, translation rate, or surface-marker expression of said cell object — from a lookup of the codon at one or more designated oncogenic positions in said record; and (v) advancing the population in time using a stochastic chemical kinetics integrator wherein said phenotypic modifier modulates reaction propensities.

**Prior-art search seeds**
- "sequence-resolved cellular simulation" / "nucleotide-level cell simulation"
- "in silico oncogenic mutation phenotype prediction agent-based"
- "PhysiCell genomic / sequence" (likely ∅)
- "BioDynaMo nucleotide" (likely ∅)
- COSMIC, OncoKB, CIViC integrations into agent-based simulators
- USPTO classes G16B 25/00 (Bioinformatics — gene expression), G16B 35/00 (Bioinformatics — sequence analysis)
- Inventors / companies to check: Mathematical Cell (Macklin lab), Allen Institute, Virtual Cell (Loew lab)

---

### Invention B — Exosome-Mediated Horizontal Transformation Model

**File evidence**
- `engine/py/molecular/exosomes.py:26–51` — `ExosomeCargo` dataclass: mRNA, miRNA, protein, DNA fragments. `is_oncogenic` (line 43) inspects mRNA for mutations and miRNAs targeting tumor suppressors (47–49).
- `engine/py/molecular/exosomes.py:54–207` — `Exosome` class. Brownian step σ = √(2·D·dt) (lines 158–161) with D_exosome = 4 µm²/s (line 90); 100 nm particle assumption. Lifetime 24h (97). Uptake via cell-type marker matching (169–186). 408 lines total.
- `modules/molecular_module.py` — `Exosome release` rate config-tunable; events `EXOSOME_RELEASED`, `EXOSOME_UPTAKEN`, `MUTATION_OCCURRED` published on the event bus (40–48).

**Mechanism (novelty axis)**
The model treats exosomes as discrete particles with explicit molecular cargo (sequence-bearing mRNA/miRNA/DNA fragments from Invention A), Brownian-diffuse them through the spatial grid, deliver cargo into recipient cells via stochastic surface-marker–matched uptake, and *transform* the recipient by translating the cargo into ectopic protein in the new host. This couples Inventions A + B + spatial diffusion in a way that, on direct inspection of competitor codebases, no published platform replicates.

**Textbook elements**
Einstein–Stokes Brownian dynamics; receptor–ligand binding for uptake. The biology of exosome cargo transfer is well known (Valadi et al. 2007); the simulation embodiment is what is potentially novel.

**Maturity:** Implemented; `MOLECULAR_CANCER_TRANSMISSION.md` documents demonstrated transformation of 3/4 recipient cells. **Patent-ready when paired with Invention A.**

**Draft independent claim (sketch)**
> A method for simulating horizontal transformation of cells in a tissue model, comprising: representing extracellular vesicles as particle objects each carrying a cargo record including one or more nucleotide-sequence-bearing items; advancing said particle objects through a spatial grid using Brownian dynamics; effecting uptake of a particle object into a recipient cell object based on a stochastic match between cargo surface markers and recipient surface markers; on uptake, translating at least one cargo nucleotide-sequence record into a protein quantity in the recipient cell object; and propagating downstream phenotypic consequences in the recipient cell object based on said protein quantity.

**Prior-art search seeds**
- "in silico exosome / extracellular vesicle simulation"
- "agent-based horizontal gene transfer cancer simulation"
- "computational model exosomal mRNA cargo"
- Biology references that are NOT prior art on the simulation: Valadi 2007; Tkach & Théry 2016
- USPTO classes G16B 5/20 (Stochastic simulation of biological systems)

---

### Invention C — Hybrid Stochastic/Deterministic Solver with Hysteresis-Banded Auto-Repartitioning

**File evidence — single highest-priority file for the patent attorney to read**
- `gpu/hybrid_solver.py:450–543` — `HaseltineRawlingsPartitioner.partition()`. Threshold = 100 copies (286). **Hysteresis** = ±20%: lower 80, upper 120 (483–484). Species at 90 copies stays in its current partition; only species clearing 120 are promoted to fast and only species falling below 80 are demoted to slow.
- `gpu/hybrid_solver.py:531–537` — Minimum-fast-species rule: if all species would partition slow, the highest-mean species is forcibly promoted to fast to preserve a deterministic anchor.
- `gpu/hybrid_solver.py:582, 747–772` — Repartition trigger: every `repartition_interval=100` steps the partition is re-evaluated; partition change is logged.
- `gpu/hybrid_solver.py:774–826` — Branching dispatch: pure ODE / pure SSA / coupled hybrid based on partition cardinality.
- `gpu/hybrid_solver.py:827–858` — Coupled step: ODE (RK4 or Euler, with non-negativity clamp at 858) for fast subset, then SSA for slow subset using current fast values.
- `gpu/hybrid_solver.py:65–251` — Four CUDA kernels (`compute_propensities`, `ssa_step_gillespie`, `ode_rhs_reactions`, `partition_species`) defined as runtime-compiled C strings.

**Mechanism (novelty axis)**
Standard Haseltine–Rawlings (2002) and Cao et al. (2006) are static partitioners. The pieces in cognisom that are plausibly novel as a *combination*:

1. **Hysteresis deadband** around the partitioning threshold to suppress oscillation of borderline species across solver regimes (lines 480–484);
2. **Periodic, scheduled re-evaluation** at a fixed `repartition_interval` rather than per-step or fully static (582, 747);
3. **Minimum-fast invariant** to guarantee a deterministic substrate even in extreme stochastic regimes (531–537);
4. **Implicit feedback coupling** in which each subsystem reads the other subsystem's current state without explicit truncation-error correction terms (787–791).

**Textbook elements that are NOT novel**
Haseltine–Rawlings partitioning itself; tau-leaping (Gillespie 2001); RK4; Adams–Moulton; CVODE/SUNDIALS-style adaptive stepping.

**Honest maturity caveat (IMPORTANT)**
The CUDA kernels exist as C strings but, per `gpu/hybrid_solver.py:939–946`, GPU dispatch falls back to CPU with the comment `# For now, fall back to CPU`. The hysteresis + minimum-fast logic is fully implemented in Python and exercised in tests. **A claim limited to "computer-implemented" and not "GPU" is enabled today.** A claim asserting GPU acceleration would currently fail enablement until kernel launch is wired up.

**Draft independent claim (sketch)**
> A computer-implemented method for hybrid stochastic-deterministic simulation of a chemical reaction network, comprising: (i) classifying each chemical species into a fast subset or a slow subset by comparing a recent abundance estimate to an upper threshold and a lower threshold defining a hysteresis band, wherein a species currently in the fast subset is reclassified to the slow subset only on its abundance estimate falling below the lower threshold, and a species currently in the slow subset is reclassified to the fast subset only on its abundance estimate exceeding the upper threshold; (ii) maintaining a minimum cardinality of one or more species in the fast subset by promoting a highest-abundance slow species when said cardinality would otherwise be zero; (iii) advancing fast-subset species using a deterministic ordinary-differential-equation integrator and slow-subset species using a stochastic event integrator, each subsystem reading the present state of the other subsystem without correction terms; and (iv) re-classifying species at a fixed step interval thereby producing a sequence of species partitions over simulation time.

**Prior-art search seeds**
- "hybrid stochastic deterministic solver hysteresis"
- "adaptive partitioning Gillespie tau-leaping"
- "Haseltine Rawlings dynamic repartition"
- Cao, Gillespie, Petzold (multi-scale stochastic simulation papers 2002–2010) — all prior art; novelty must lie outside their specific algorithms
- Existing solvers to differentiate from: STOCKS, COPASI, BioNetGen NFsim, StochKit2, MesoRD, Lattice Microbes

---

### Invention D — Batched GPU Gillespie SSA with Per-Cell xoshiro256** RNG and Threshold-Switched Poisson Sampler

**File evidence**
- `gpu/ssa_kernel.py:46–138` — CUDA `tau_leap_step` kernel. One thread per cell; grid = ⌈n_cells/256⌉ blocks of 256.
- `gpu/ssa_kernel.py:67–74` — Inline xoshiro256** macros (ROTL/NEXT) with per-cell 4×uint64 state.
- `gpu/ssa_kernel.py:80–98` — **Threshold-switched Poisson sampler**: λ < 30 uses Knuth's product-of-uniforms; λ ≥ 30 uses Box–Muller normal approximation. Threshold hard-coded at 30.
- `gpu/ssa_kernel.py:140–221` — CUDA `direct_ssa_step` kernel: per-cell exponential τ = −log(r₁)/a_total (197), cumulative-sum reaction selection on a 64-slot stack-allocated array (178), max 100,000 steps (558).
- `gpu/ssa_kernel.py:235–328` — `GeneExpressionModel` and `ProstateCellModel` (7 genes × 4 reactions = 28 reactions), with cancer variant scaling rate constants (MYC 3×, AR 2×, PTEN 0.2×, TP53 0.3×).
- `gpu/ssa_kernel.py:333–615` — `BatchSSA` orchestrator class.

**Mechanism (novelty axis)**
Batched SSA on GPU is published prior art (Komarov & D'Souza 2012; Lattice Microbes; Nobile et al. 2017). What is potentially distinctive in cognisom:

1. **Per-cell xoshiro256\*\* RNG** rather than the more common counter-based Philox or per-cell xorshift. Inline macros in CUDA C source (67–74).
2. **Threshold-switched Knuth/Box-Muller Poisson** — well-known statistical recipe but adapted to single-warp branchless dispatch.
3. **64-slot stack-allocated cumulative propensity array** (180) avoiding global-memory round-trip per reaction selection.
4. **Tight integration with the structure-of-arrays cell layout** (`gpu/cell_ops.py`) so per-cell state has no pointer indirection.

These are micro-optimization patterns; novelty here is thin. **This is more of a trade-secret candidate than a patent candidate.**

**Maturity:** CUDA source compiles; CPU fallback (`tests/test_gpu.py`) is what runs in CI. GPU performance not validated in repo.

**Recommended IP posture:** Hold as trade secret. Disclosing the kernel layout via patent buys little defensive value since patenting micro-optimizations of well-known algorithms tends to result in narrow claims that are easy to design around.

---

### Invention E — Spatial Stochastic Reaction–Diffusion with Grid-Hashed Bimolecular Pairing

**File evidence**
- `gpu/smoldyn_solver.py:59–108` — `brownian_motion_step` CUDA kernel: σ = √(2D·dt) (76); inline xorshift64 + Box-Muller for 3 Gaussians (94–100).
- `gpu/smoldyn_solver.py:184–221` — `build_spatial_hash`: `cx = (x − x_min) / cell_size`; `atomicAdd` to per-cell counters (220).
- `gpu/smoldyn_solver.py:224–280+` — `bimolecular_reactions` kernel: scans 27-cell neighborhood for partner; reacts if distance < binding_radius and species pair matches.

**Mechanism**
Smoldyn (Andrews & Bray 2004) and ReaDDy (Schöneberg & Noé 2013) are the dominant prior art for particle-based reaction–diffusion. Spatial hashing for collision broad-phase is standard in molecular dynamics (LAMMPS, GROMACS) and graphics (cf. NVIDIA particle samples). cognisom's implementation appears competent but not novel.

**Maturity:** CUDA kernels defined; no kernel launcher visible from the `SmoldynSolver` Python class — CPU fallback effectively used.

**Recommended IP posture:** Not a strong patent candidate. Trade secret at best.

---

### Invention F — Mechanistic Antigen-Presentation + Neoantigen Pipeline

**File evidence**
- `cognisom/genomics/neoantigen_predictor.py:116–235` — Implemented PWM-based MHC-I binding predictor. Generates 8-, 9-, 10-, 11-mer peptides spanning a mutation site, scores each against patient HLA alleles via position-weight matrices, and ranks by predicted affinity.
- `library/models.py:826` — Per-cell `mhc1_expression: float = 1.0` field.
- `modules/cellular_module.py:119` — Cancer-cell initial `mhc1_expression = 0.3` (immune-evasion baseline).
- `gpu/spatial_ops.py:200–240` — Immune detection: T cells use TCR/MHC-I match heuristic (235); NK cells use missing-self.
- `dashboard/pages/22_research_briefing.py:197` — references SEQC2 breast cancer dataset for validation.

**Mechanism (novelty axis)**
The strong claim is the *coupling*: simulated mutation in Invention A → simulated peptidome in F → MHC-I binding prediction → immune-cell recognition outcome → tumor population dynamics → therapy-response readout, all in one closed loop within a single simulator. Standalone neoantigen predictors (NetMHCpan, MHCflurry, MARIA) do not feed dynamic tissue simulations; tissue simulators do not consume per-cell sequence-derived peptidomes.

**Textbook elements**
PWM-based MHC binding (Parker 1994); NetMHCpan-style architectures; standard immunopeptidomics. Cognisom's PWM scorer is not state-of-the-art relative to NetMHCpan-4.1 / MHCflurry-2.0.

**Honest maturity caveat**
The closed-loop coupling is *partial*. The neoantigen predictor exists and runs. The immune-recognition kernel in `gpu/spatial_ops.py` is a thresholding heuristic, not actual TCR-affinity-matched binding. Until the loop is closed end-to-end with a working test, claims must be drafted carefully.

**Recommended posture:** File when the loop is closed; current state is patent-pending-quality but not yet patent-filing-quality.

**Prior-art search seeds**
- NetMHCpan, MHCflurry, MARIA, IEDB Analysis Resource
- "agent-based simulation neoantigen presentation"
- "in silico immunotherapy response prediction tumor microenvironment"
- USPTO class G16B 20/00 (Bioinformatics — Genetic engineering)

---

### Invention G — Coupled Circadian Oscillator with Cell-Cycle Gating

**File evidence**
- `modules/circadian_module.py:30–82` — `CircadianClock`: phase advance `phase += (dt/period)·24` (53); Kuramoto-style master coupling `phase += coupling_strength·(master_phase − phase)·dt` (63); CLOCK/BMAL1 = 0.5+0.5·amp·cos(t), PER/CRY = 0.5+0.5·amp·cos(t+π) (66–68); division-permissive window ZT 6–12 (81).

**Mechanism**
Per-cell circadian phase coupled to a master clock, modulating a cell-cycle checkpoint. Documented in `CIRCADIAN_AND_MORPHOGENS.md`. Code exists at the dataclass level. Whether the dynamics are *exercised* in a closed simulation loop is unclear.

**Honest maturity caveat**
Implementation is shallow: phase advance is explicit Euler, not a Goodwin or Leloup-Goldbeter ODE; no protein-level oscillation is simulated; immune circadian modulation is documented, not coded.

**Prior art:** Goodwin 1965; Leloup & Goldbeter 2003; SCN-coupling papers. Patent path unlikely until dynamics are deeper.

**Recommended posture:** Not yet patent-ready. Reassess when oscillator equations are fully wired.

---

### Invention H — Epigenetic Regulation of Gene Expression

**File evidence**
- `modules/epigenetic_module.py:28–68` — `EpigeneticState`: per-gene DNA methylation [0,1] (33), H3K4me3 active (36), H3K27me3 repressive (37), H3K9ac active (38). `is_silenced()`, `is_active()`, `get_expression_modifier()` (43–68) compute a multiplier on transcription rate from chromatin marks.
- `modules/epigenetic_module.py:140–146` — Cancer-specific hypermethylation of CDKN2A and MLH1 to 0.7–0.8.

**Mechanism**
A scalar chromatin-state modifier on transcription rate is a *modeling choice* rather than a novel algorithm; epigenetic regulation in agent-based simulators has been explored (e.g., Zaidi 2017). Patent posture is weak unless integrated with sequence-grounded mutation tracking (Invention A) so that mutation × methylation × transcription rate forms an irreducible joint state. That integration is not yet evident.

**Recommended posture:** Trade secret + future patent if joint state with A is built.

---

### Invention I — Event-Bus-Driven Modular Multi-Scale Simulation Engine

**File evidence**
- `core/simulation_engine.py:56–200` — `SimulationEngine` master orchestrator; `step()` triple-phase: pre_step → update → post_step, then `event_bus.process_events()`, then `time += dt` (177–195).
- `core/registry.py:1–100` — Plugin registry (decorators, auto-discovery, versioning). Marked "Phase 0 of Strategic Implementation Plan" (38).
- `INTEGRATION_COMPLETE.md:170–172` — claimed 243k steps/s, 100 steps in <1ms (CPU).

**Mechanism**
Event-bus modular simulation is well-known (DEVS, Repast, NetLogo extensions). Cognisom's combination of (event bus + module triple-phase + hot-pluggable backends + biology-specific semantics) is implementation polish, not invention.

**Recommended posture:** Not patentable. Use as ergonomic differentiator.

---

## 4. Aspirational Subsystems (NOT Patent-Ready)

These are documented in markdown and have class scaffolding but lack operative dynamics. Filing on these now would fail enablement.

| Subsystem | Where | Gap |
|---|---|---|
| Multi-GPU domain decomposition | `gpu/domain_decomposition.py` 339 lines | Partition geometry only; no NCCL/MPI/peer-access communication; no working multi-GPU run |
| ML surrogates (PDE / metabolism / signaling) | `ARCHITECTURE.md`, `FRONTIER_CAPABILITIES.md` | Zero PyTorch/JAX/TensorFlow code; no training pipeline; no surrogate inference path |
| Newton physics / cell mechanics | `gpu/physics_interface.py:495` | Literal `raise NotImplementedError` |
| Full clonal evolution lineage | `ARCHITECTURE.md:289–306` | No lineage data structure; mutations are parameter scalings |
| Omniverse real-time integration | `omniverse/` 8 files | Kit extension scaffolding; not driven by the live sim |
| End-to-end immune killing kinetics (TCR affinity-matched) | `modules/immune_module.py`, `gpu/spatial_ops.py` | Thresholding heuristics, not biophysical binding |
| Patient-data → simulation pipeline (VCF parsing → per-cell sequences) | `bridge/` | Bridge files exist; pipeline not closed |

**Implication:** Do not include claims that depend on these subsystems until they are working end-to-end. Mention them in the *specification* as planned embodiments only if your attorney advises (some patents draft claims narrowly while embodying broader future scope in the spec).

---

## 5. Differentiation From Prior Art (At-A-Glance)

| Capability | cognisom | PhysiCell | BioDynaMo | Lattice Microbes | Gell | COPASI | Virtual Cell |
|---|---|---|---|---|---|---|---|
| Per-cell ATCG nucleotide state | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Codon-level mutation phenotype | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Exosome cargo + horizontal transfer | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Hybrid SSA/ODE with hysteresis-banded repartition | ✅ | ❌ | ❌ | partial | ❌ | partial | ❌ |
| Batched GPU SSA across N cells | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
| Spatial PDE diffusion grid | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ |
| Coupled neoantigen → tumor dynamics | partial | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Multi-GPU scaling | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Million-cell tissues | aspirational | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |

**Punchline for the patent attorney:** Cognisom's defensible space is rows 1–4 and 7. Rows 5–6 are crowded; row 8–9 are not yet built.

---

## 6. Summary: Prioritized Filing Recommendations

| Rank | Invention | File a patent? | Posture |
|---|---|---|---|
| 1 | A — Sequence-grounded mutation→phenotype | **Yes — file first** | Independent claim + dependents tying to specific oncogenic loci |
| 2 | B — Exosome horizontal transformation | **Yes — file alongside A** | Pair with A in same continuation family |
| 3 | C — Hybrid SSA/ODE with hysteresis-banded auto-repartition | **Yes — narrow claim** | "Computer-implemented", not "GPU" until kernels are wired |
| 4 | F — Neoantigen → tumor dynamics closed loop | **Defer** until loop is closed | Patent-pending posture; finish implementation first |
| 5 | D — Batched GPU SSA micro-optimizations | **No — trade secret** | Disclose in technical white-papers if marketing benefit, not in patents |
| 6 | E — Spatial RD with grid-hashed pairing | **No** | Standard HPC; weak claim |
| 7 | G, H — Circadian, epigenetic | **Defer** | Build out dynamics first |
| 8 | I — Event-bus engine architecture | **No** | Engineering, not invention |

**Trade-secret-better-than-patent:**
- Specific oncogenic mutation parameter sets (KRAS/TP53/BRAF rate scalings)
- Cancer-cell `mhc1_expression = 0.3` and analogous immune-evasion calibrations
- Hybrid solver thresholds (100 copies, ±20% hysteresis, 100-step repartition interval)
- GPU kernel block sizes, max-reaction stack depths (64), max-steps (100k)
- Validation benchmark sets and any patient-data calibration

---

## 7. Pre-Filing Workstreams (Things to Do Before You Talk to a Patent Attorney)

1. **Run the inventor-side prior-art search** using the search seeds in §3 above. Spend ~4 hours on:
   - Google Patents — search seeds + USPTO class G16B 5/00, 5/20, 20/00, 25/00, 35/00
   - Lens.org — same searches; read the closest 20–30 hits
   - PubMed + bioRxiv — "agent-based cancer simulation", "exosome simulation", "neoantigen tumor model"
   - PhysiCell, BioDynaMo, Lattice Microbes, Compucell3D — read their *recent* paper abstracts and feature lists; note any 2024–2026 additions
2. **Compile a "freedom-to-operate" notebook** of the closest 10 prior-art references for each invention.
3. **Run the example scripts** in `examples/` end-to-end with a fresh checkout, capture outputs, and save them — this is the §112 enablement evidence.
4. **Decide on geographic scope** (US-only vs PCT) and on a defensive vs offensive posture, since this affects how aggressive the claims should be.
5. **Take a snapshot of the repo** at the filing date; commit a git tag like `patent-snapshot-2026-05-XX`. Anything implemented after the priority date can go in a continuation or CIP.

---

## 8. Risks & Caveats Re-Stated Plainly

- **Enablement risk (§112):** Multiple subsystems are documented but not built. Claims must track implemented code. Ignore the markdown roadmaps for filing scope.
- **Novelty risk (§102/103):** PhysiCell and BioDynaMo are aggressive 2024–2026; check their latest releases — they may have added sequence-aware features since this scoping document was written.
- **Subject-matter risk (§101):** Pure simulation methods sometimes get rejected as "abstract ideas" in the US (Alice/Mayo). Tying claims to a *specific computational improvement* (e.g., the hysteresis-banded partitioner reduces solver oscillation, or the per-cell sequence storage enables phenotype prediction not previously possible) helps. Use language about "improving the functioning of the computer / simulator".
- **Disclosure risk:** This document, the GitHub repo, and any conference talks may already constitute public disclosure under foreign-jurisdiction novelty rules. Confirm filing dates against any prior public disclosures (papers, demos, READMEs visible to the public).
- **PhysiCell directory risk:** The repo bundles a full unmodified PhysiCell clone. Verify it is treated as a third-party reference (BSD-3 license — permissive) and not woven into any claim language.

---

## 9. Verification (How To Sanity-Check This Document)

1. Confirm Invention A is implemented:
   - `python -c "from cognisom.engine.py.molecular.nucleic_acids import Gene; g = Gene(name='KRAS', sequence='ATG...'); g.introduce_oncogenic_mutation('G12D'); print(g.sequence[33:36])"` should show `GAT` (KRAS G12D codon edit).
   - Cross-check `engine/py/molecular/nucleic_acids.py:389–426` for the hard-coded oncogenic-mutation table.
2. Confirm Invention B runs:
   - Look for an example/demo in `examples/` exercising `ExosomeSystem`; run it; observe reported transformation events.
3. Confirm Invention C's hysteresis logic:
   - Read `gpu/hybrid_solver.py:480–543`; write a 30-line reproduction that flips a species back and forth around the threshold and verifies it does NOT change partitions until it crosses ±20%.
4. Confirm GPU enablement gap:
   - Read `gpu/hybrid_solver.py:939–946` and confirm the literal `# For now, fall back to CPU` comment. This is the line that constrains GPU claims.
5. Confirm Invention F maturity:
   - Run the predictor at `cognisom/genomics/neoantigen_predictor.py` against a sample mutation; verify it returns a ranked peptide list. Then attempt the *closed loop* (mutation → peptide → MHC presentation → T-cell kill in tissue sim) and document where it breaks.

When the attorney reviews this document, the §3 file:line citations are the verifiable backbone — every claim sketched above can be checked against the cited code in under a minute.

---

## 10. End-of-Document Posture

This is a scoping document, not a patent application and not legal advice. Hand it (or a redacted version) to a patent attorney along with:
- the repo at a tagged snapshot,
- the prior-art notebook from §7.1,
- a 1-page commercial rationale (which markets, which buyers, which competitors).

Files A, B, and C are the strongest candidates for immediate filing as a single application family. F is the strongest deferred candidate. D, E are trade secrets. G, H, I are not patent material as currently implemented.
