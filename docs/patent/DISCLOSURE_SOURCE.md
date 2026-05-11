---
title: "Cognisom — Patent Disclosure & Scoping Document"
subtitle: "Multi-Scale Cellular Simulation Platform for Cancer-Immunotherapy Prediction"
author: "Prepared for: David Walker (Inventor)"
date: "2026-05-11"
---

# Cognisom — Patent Disclosure & Scoping Document

**Inventor:** David Walker
**Repository:** `/Users/davidwalker/CascadeProjects/cognisom/`
**Document purpose:** Internal scoping document to brief a patent attorney and to drive prior-art research. Written to be readable by an attorney without a biology background.
**Document status:** Pre-filing technical disclosure. Not legal advice. Not a patent application.

---

## How to Read This Document

Each invention is presented in two layers:

1. **Plain-English explanation** — what the invention does, why it matters, what makes it different from prior work. No specialized vocabulary required.
2. **Technical detail** — algorithms, data structures, code citations to the actual repository (file:line). Use these to verify what is implemented and to formulate claim language.

Specialized terms are explained in the **Glossary** in §2 (biology) and §3 (computing). Look there first if a term is unfamiliar.

---

## 1. Executive Summary

**What Cognisom is, in one paragraph.** Cognisom is a software platform that simulates how individual cancer cells, healthy cells, and immune cells interact inside a piece of tissue. It runs on conventional computers and (in parts) on graphics processors (GPUs). It is intended to predict, in advance, how a particular cancer patient's tumor will respond to a particular immunotherapy drug — without running a clinical trial first. The technical novelty is that, unlike competing simulators, Cognisom carries each cell's **actual DNA sequence** as part of its state, so the effects of specific cancer mutations propagate through the simulation in a biologically grounded way.

**What we are filing for.** Three concrete inventions are described below as **Inventions A, B, and C**. Each has working source code and is ready to be claimed. **Invention F** is strong but deferred until additional integration work is completed. Inventions D and E are recommended as trade secrets. Inventions G, H, and I are not patent-ready.

**Why this matters commercially.** Drug developers spend hundreds of millions of dollars on Phase II oncology trials. Even a modest improvement in predicting which patient cohorts will respond is worth substantial license fees. The defensible technical moat sits in the combination of features summarized in §11 (Competitive Differentiation), where Cognisom is the only platform — out of seven leading published competitors — that carries per-cell genomic sequence.

**What this document is not.** It is not a patent application. It does not contain finalized claim language. The "draft claim sketches" included for Inventions A, B, and C are starting points for an attorney to refine. Some sections are deliberately self-critical about which subsystems are real code versus aspirational documentation, so that filings are sized to what is enabled in source.

---

## 2. Glossary — Biology Terms for the Non-Biotech Attorney

| Term | Plain-English meaning |
|---|---|
| **DNA sequence** | The string of chemical "letters" (A, T, C, G) that encodes the instructions for building proteins. Think of it as a long source-code file. |
| **RNA / mRNA** | A working copy of a region of DNA. The cell makes RNA from DNA, then reads the RNA to build proteins. Think of it as an in-memory copy of a function before it executes. |
| **Protein** | A molecular machine built by following the RNA instructions. Almost everything a cell does is done by proteins. |
| **Gene** | A specific region of DNA that codes for one protein. |
| **Mutation** | A change to one or more letters in the DNA. A mutation in a gene can change the resulting protein, which can change cell behavior. |
| **Oncogene** | A gene that, when mutated, makes a cell grow uncontrollably — i.e., causes cancer. Example: **KRAS**. |
| **Tumor suppressor** | A gene whose normal job is to prevent cancer. When it is mutated and broken, cancer becomes more likely. Example: **TP53** (the most-mutated gene in human cancer). |
| **Cell cycle** | The repeating sequence of phases a cell goes through to grow and divide. |
| **Codon** | A group of three consecutive DNA letters that encode one amino acid (one piece of a protein). |
| **KRAS G12D** | A specific cancer mutation: at position 12 of the KRAS protein, the normal amino acid (glycine, "G") is replaced by aspartate ("D"). One of the most common mutations in pancreatic and colorectal cancer. |
| **MHC-I** | A "loyalty card" system on the surface of every healthy cell. The cell displays fragments of its own proteins on its surface using MHC-I molecules. T cells inspect these fragments to verify that the cell is healthy. |
| **Neoantigen** | A protein fragment that arises from a cancer mutation and gets displayed on MHC-I. The immune system can potentially recognize it as foreign. This is the basis of many modern immunotherapies. |
| **T cell (CD8+)** | An immune cell that patrols tissue and kills cells displaying foreign fragments on MHC-I. The "soldiers" of the immune system. |
| **NK cell (natural killer)** | An immune cell that kills cells which have stopped showing MHC-I at all ("missing self"). Many cancer cells hide from T cells by reducing MHC-I, which makes them visible to NK cells instead. |
| **Macrophage** | An immune cell that eats debris and can secrete signaling molecules to recruit other immune cells. |
| **Cytokine** | A small signaling protein that one cell secretes to communicate with neighboring cells. |
| **Exosome** | A tiny membrane-bound bubble (about 100 nanometers across) that one cell releases into its environment. The bubble carries cargo — RNA, DNA fragments, proteins — that can be taken up by another cell. Cancer cells use exosomes to influence neighboring cells, including transferring oncogenic content. |
| **Diffusion** | The physical process by which molecules spread out from where they are concentrated to where they are not (e.g., oxygen diffusing into tissue from a blood vessel). |
| **Tumor microenvironment** | The mix of cancer cells, immune cells, blood vessels, structural cells, and signaling molecules that surrounds a tumor. |
| **Immunotherapy** | A class of cancer drugs that work by activating the patient's own immune system to attack the tumor (rather than poisoning the tumor directly like chemotherapy). |
| **HLA allele** | The specific variant of MHC-I that a particular patient carries (each person has up to 6 alleles). Determines which mutation fragments their immune system can "see". |

---

## 3. Glossary — Computing Terms for the Non-Biotech Attorney

| Term | Plain-English meaning |
|---|---|
| **Simulation** | A computer program that reproduces, step by step, the behavior of a real-world system over simulated time. |
| **Agent-based model** | A simulation in which the system is built out of many individual "agent" objects (here, one per cell), each with its own state and behavior. |
| **Stochastic simulation** | A simulation in which the next step depends on random draws — used when the system has inherent randomness (e.g., one molecule colliding with another at random times). |
| **Deterministic simulation** | A simulation in which the next step is computed exactly from the current state — used when the numbers involved are large enough that randomness averages out. |
| **Gillespie SSA** | The standard textbook algorithm for stochastic simulation of chemical reactions, published by Daniel Gillespie in 1977. "SSA" stands for Stochastic Simulation Algorithm. Prior art everywhere. |
| **Tau-leaping** | A faster, less exact variant of Gillespie SSA. Prior art (Gillespie 2001). |
| **Hybrid solver** | A simulation that uses deterministic math for the abundant, fast species and stochastic math for the rare, slow species — the two are stitched together. |
| **ODE** | Ordinary Differential Equation. A standard way to describe how a quantity changes continuously in time. Solving ODEs is a textbook topic. |
| **PDE** | Partial Differential Equation. Used to describe quantities that change in both time *and* space, such as oxygen diffusing through tissue. |
| **Finite-difference method** | A standard textbook way to solve PDEs on a grid of points. |
| **GPU** | Graphics Processing Unit. A type of chip with thousands of small processing cores. Originally for video games; now widely used to run scientific simulations and AI in parallel. |
| **CUDA** | NVIDIA's programming model for writing software that runs on NVIDIA GPUs. |
| **Kernel** | A small program that runs on the GPU, with many copies in parallel. |
| **Batched simulation** | Running many independent copies of the simulation simultaneously on a GPU — for example, simulating 10,000 cells at once. |
| **Event bus** | A software pattern in which different modules of a program communicate by posting and subscribing to "events" rather than calling each other directly. |
| **Trade secret** | A protected business asset that the company keeps confidential rather than disclosing in a patent. Patents publish the invention in exchange for a 20-year monopoly; trade secrets last indefinitely but only as long as they remain secret. |
| **Enablement (USPTO §112)** | A patent requirement that the patent specification describe the invention in enough detail that a person skilled in the art could build it. Filings on undisclosed or unbuilt subsystems fail enablement. |
| **Prior art** | Everything previously published (papers, patents, products, code releases) that is relevant to assessing whether an invention is new. |
| **Continuation / CIP** | A continuation patent extends an earlier filing using the same priority date; a Continuation-In-Part (CIP) adds new material with a later priority date for the new material. |

---

## 4. The Problem Cognisom Solves

Cancer is hard to treat in part because each patient's tumor is genetically unique. The same drug can cure one patient and do nothing for another with apparently the same diagnosis. This is especially true for modern **immunotherapy** drugs (such as checkpoint inhibitors like pembrolizumab), where the response rate in many solid tumors — pancreatic, prostate, glioblastoma — is below 20%. Drug developers run very expensive clinical trials largely to find out, after the fact, which patients responded.

The vision Cognisom pursues is: **build a faithful enough simulation of a patient's tumor that the response to a candidate drug can be predicted in software**, before the drug is given to that patient. This is sometimes called an "in-silico clinical trial."

Several research groups and companies are pursuing pieces of this vision (see §11, Competitive Differentiation). What distinguishes Cognisom technically is the depth of biological grounding: each simulated cell carries its actual DNA sequence at relevant loci, so the consequences of specific oncogenic mutations (e.g., KRAS G12D vs G12C) propagate naturally through the simulated tumor, including their consequences for which protein fragments are displayed to T cells.

---

## 5. The Approach in Plain English

A picture in words: Cognisom maintains in computer memory a 3-dimensional region of simulated tissue, perhaps one cubic millimeter, populated with thousands to millions of simulated cells. Time advances in steps (typically 0.01 hours per step). At each step:

1. Each cell consumes nutrients from the surrounding space, produces and degrades RNA and protein molecules, and may divide or die.
2. Cells secrete signaling molecules and exosomes (the cell-to-cell delivery bubbles described in §2); these diffuse through the tissue.
3. Immune cells move through the tissue, inspect the surface markers of nearby cells, and kill cells they recognize as cancerous.
4. Cancer cells acquire new mutations as they divide, which changes their behavior and the protein fragments they display.

The simulation outputs trajectories: tumor size over time, immune-cell counts, mutation frequencies, drug effects. The output can be compared against patient outcomes to validate the model and against alternative treatments to choose among them.

The novel pieces of this approach — the parts Cognisom does differently from other simulators — are detailed in §7 below.

---

## 6. System Architecture (For Reference)

Cognisom is organized as a four-layer simulator:

1. **Layer 1 — Intracellular biology.** What happens inside each cell: gene expression (DNA → RNA → protein), metabolism, cell-cycle decisions. Implemented in `engine/py/molecular/`, `engine/py/cell.py`, `engine/py/intracellular.py`.
2. **Layer 2 — Intercellular space.** What happens between cells: diffusion of oxygen, glucose, cytokines, and exosomes. Implemented in `engine/py/spatial/grid.py`.
3. **Layer 3 — Immune surveillance.** T cells, NK cells, and macrophages moving through the tissue and recognizing target cells. Implemented in `modules/immune_module.py` and (partially) `gpu/spatial_ops.py`.
4. **Layer 4 — Cancer-specific.** Mutation accumulation, immune evasion, clonal selection, drug response. Distributed across `modules/molecular_module.py`, `modules/cellular_module.py`, and the genomics package.

A central **event bus** (`core/simulation_engine.py:56–200`) advances time in three sub-phases (pre-step, update, post-step) and lets modules communicate without being hard-wired to each other. Specific reported throughput: 243,000 simulation steps per second on a CPU (`INTEGRATION_COMPLETE.md:170–172`).

GPU acceleration is partial and described honestly in §8.

---

## 7. The Inventions

For each invention below: plain-English explanation, what specifically is novel, where the code lives, what is well-known prior art and must be disclaimed, a draft (non-legal) claim sketch, and prior-art search seeds the attorney can use.

### Invention A — Sequence-Grounded Mutation→Phenotype Model

**Plain-English explanation.** In every other major cellular simulator (PhysiCell, BioDynaMo, COPASI, Virtual Cell, Lattice Microbes, Gell, CompuCell3D), a cancer mutation is represented abstractly — typically as a number, like "growth rate = 1.5×" or as a boolean flag "mutated = true". In Cognisom, each cell carries the **actual nucleotide sequence** of the relevant gene as part of its state. When the simulation introduces a mutation, it edits the specific letters at the specific position (for KRAS G12D, the letters "GGT" at position 35 of the KRAS gene become "GAT"). The simulation then derives the cell's altered behavior from that sequence change — including chemical properties of the resulting RNA (its stability, melting temperature, GC composition) and the consequences for the protein the cell produces.

**Why this matters for the patent.** This is the strongest novelty claim in the platform because it appears to be unoccupied space among existing cellular simulators. It is what makes the rest of Cognisom possible: because sequence is the substrate, downstream phenomena — neoantigen generation, exosomal cargo transfer, drug binding — can all be tied back to specific genomic changes.

**Code evidence.**

- `engine/py/molecular/nucleic_acids.py` lines 42–221: the `NucleicAcid` base class. Stores actual ATCG/AUCG strings. Methods include `_calculate_gc_content()` (lines 83–88), `_calculate_tm()` implementing the Wallace rule (90–97), `mutate(position, base)` (106–148), `is_complementary(other, threshold=0.8)` (150–180), `translate_codon()` (188–209) containing the full 64-codon table, and stochastic decay via `update()` (211–221).
- Same file, lines 224–261: the `DNA` class implements reverse complementation (238–240), `transcribe()` (T→U substitution at line 251), and inheritance of mutations into transcripts.
- Same file, lines 264–329: the `RNA` class (mRNA, miRNA, tRNA variants) implements start-codon ORF discovery and stop-codon termination in `translate()` (291–319) and miRNA::mRNA `can_bind_to(threshold=0.7)` (321–329). mRNA half-life 8 hours; miRNA half-life 24 hours.
- Same file, lines 332–426: the `Gene` class. Method `introduce_oncogenic_mutation()` (389–426) hard-codes specific oncogenic edits: KRAS G12D (position 35, GGT→GAT), KRAS G13D (position 38), BRAF V600E (position 1799), TP53 R175H, TP53 R248W (lines 404–415). Stochastic `transcribe()` with promoter strength (360–387).

**What is well-known prior art and must be disclaimed.** The 1979 Wallace rule for melting temperature. The standard 64-codon genetic-code translation table (textbook). Reverse complementation of DNA strands (textbook). The catalogs of known cancer mutations (COSMIC, OncoKB, CIViC are publicly available). Claim language must focus on the *integration* — using nucleotide sequence as the primary state variable through which the simulation propagates — rather than on any constituent.

**Implementation maturity.** Implemented and exercised in `tests/test_validation.py`. **Patent-ready today.**

**Draft independent claim (illustrative — not legal language).**

> A computer-implemented method for simulating tumor progression, comprising:
>
> (i) instantiating, in computer memory, a population of cell objects, each cell object including a per-locus nucleotide-sequence record for one or more genes;
>
> (ii) stochastically introducing site-specific substitutions into said nucleotide-sequence record at simulated cell division;
>
> (iii) deriving, from said nucleotide-sequence record, at least one chemical property selected from GC content, melting temperature, and stability index;
>
> (iv) deriving a phenotypic modifier — applied to a transcription rate, translation rate, or surface-marker expression of said cell object — from a lookup of the codon at one or more designated oncogenic positions in said record; and
>
> (v) advancing the population in time using a stochastic chemical kinetics integrator wherein said phenotypic modifier modulates reaction propensities.

Dependent claims to consider: specific oncogenic loci (KRAS codon 12/13, TP53 codon 175/248, BRAF V600); horizontal transfer of sequence records between cells (linking to Invention B); coupled MHC-I peptide generation from sequence (linking to Invention F).

**Prior-art search seeds for the attorney.**

- "sequence-resolved cellular simulation"; "nucleotide-level cell simulation"
- "agent-based simulation oncogenic mutation phenotype"
- Search PhysiCell, BioDynaMo, CompuCell3D documentation for any sequence-aware features (likely none)
- COSMIC, OncoKB, CIViC integrations into agent-based simulators
- USPTO classes G16B 25/00 (Bioinformatics — gene expression) and G16B 35/00 (Bioinformatics — sequence analysis)
- Inventor/lab checks: Macklin lab (Indiana), Allen Institute, Loew lab (Virtual Cell), Roeland Merks lab

---

### Invention B — Exosome-Mediated Horizontal Transformation Model

**Plain-English explanation.** Real cancer cells release tiny bubbles called exosomes (about 100 nanometers across — for scale, a human hair is 80,000 nanometers thick). These bubbles carry RNA, DNA fragments, and proteins. Exosomes drift through the tumor's surrounding tissue, get absorbed by neighboring cells, and can deliver their cargo. There is now substantial published biology showing that cancer cells use this mechanism to transform neighboring normal cells into pre-cancerous cells, and to suppress the local immune response.

Cognisom simulates this process. Each exosome is a particle that diffuses through the simulated tissue using **Brownian motion** (the random jiggling motion of small particles). Each exosome carries a cargo record — and crucially, because of Invention A, the cargo can include actual nucleotide sequences with their mutations intact. When an exosome is absorbed by a recipient cell, the cargo is translated into protein in the recipient, and the recipient's behavior changes accordingly. This **closes the loop**: a cancer cell in one corner of the tissue can release exosomes that genetically transform a healthy cell on the other side of the tissue.

**Why this matters for the patent.** Combined with Invention A, this captures a distinct mechanism — horizontal transformation — that no existing cellular simulator models. Note the biology of exosomal cargo transfer was first published in 2007 (Valadi et al.); the biology itself is not patentable. What is potentially patentable is the **specific simulation embodiment**: discrete particle objects carrying nucleotide-sequence-bearing cargo, Brownian advancement through a spatial grid, stochastic surface-marker-matched uptake, and translation of cargo into protein quantity in the recipient.

**Code evidence.**

- `engine/py/molecular/exosomes.py` lines 26–51: the `ExosomeCargo` dataclass holds mRNA, miRNA, protein, and DNA fragment slots. Property `is_oncogenic` (line 43) returns true if the mRNA carries known mutations or if the miRNA targets known tumor-suppressor genes (47–49).
- Same file, lines 54–207 (408 total lines): the `Exosome` class. Brownian step size σ = √(2·D·dt) computed at lines 158–161 with diffusion coefficient D = 4 µm²/s at line 90; this is the standard Einstein-Smoluchowski formulation applied to 100-nm particles. Lifetime 24 hours (line 97). Uptake stochastic via surface marker matching (169–186).
- `modules/molecular_module.py`: the exosome-release rate is config-tunable; the module publishes events `EXOSOME_RELEASED`, `EXOSOME_UPTAKEN`, `MUTATION_OCCURRED` on the event bus (lines 40–48).

**What is well-known prior art and must be disclaimed.** The Einstein-Smoluchowski Brownian dynamics equation (textbook physics). Receptor-ligand binding models for cell uptake (textbook molecular biology). The biology of exosome cargo transfer (Valadi et al. 2007; Tkach & Théry 2016 — published biology, not simulation).

**Implementation maturity.** Implemented; the documentation file `MOLECULAR_CANCER_TRANSMISSION.md` reports demonstrated transformation of 3 of 4 recipient cells in a test scenario. **Patent-ready when paired with Invention A as a single application family.**

**Draft independent claim (illustrative).**

> A method for simulating horizontal transformation of cells in a tissue model, comprising:
>
> (i) representing extracellular vesicles as particle objects each carrying a cargo record including one or more nucleotide-sequence-bearing items;
>
> (ii) advancing said particle objects through a spatial grid using Brownian dynamics;
>
> (iii) effecting uptake of a particle object into a recipient cell object based on a stochastic match between cargo surface markers and recipient surface markers;
>
> (iv) on uptake, translating at least one cargo nucleotide-sequence record into a protein quantity in the recipient cell object; and
>
> (v) propagating downstream phenotypic consequences in the recipient cell object based on said protein quantity.

**Prior-art search seeds.**

- "in silico exosome simulation"; "in silico extracellular vesicle simulation"
- "agent-based horizontal gene transfer cancer simulation"
- "computational model exosomal mRNA cargo"
- USPTO class G16B 5/20 (Stochastic simulation of biological systems)

---

### Invention C — Hybrid Stochastic/Deterministic Solver with Hysteresis-Banded Auto-Repartitioning

**Plain-English explanation.** Inside a cell, some molecules exist in very small numbers (a single piece of damaged DNA, or three copies of a transcription factor); others exist in millions of copies (water, ATP, structural proteins). The textbook simulation method for rare species is **stochastic** — explicitly random — because the timing of each individual event matters. The textbook method for abundant species is **deterministic** — solving a smooth equation — because randomness averages out.

A *hybrid solver* uses both methods, applying stochastic math to the rare species and deterministic math to the abundant ones. This was first described by Haseltine and Rawlings in 2002. The basic Haseltine–Rawlings idea is well-known prior art.

The problem with the basic Haseltine–Rawlings approach: if a species' count fluctuates near the threshold separating "rare" from "abundant," it can get reassigned every step, oscillating between solver regimes. This causes computational artifacts and instability.

Cognisom's hybrid solver adds three specific refinements that together appear to be novel:

1. **Hysteresis deadband.** Instead of one threshold (e.g., 100 copies), there are two: an upper threshold (120) and a lower threshold (80). A species that is currently treated as abundant stays abundant until it drops below 80. A species currently treated as rare stays rare until it rises above 120. Species hovering at 90 are not reassigned. This prevents the oscillation problem.
2. **Minimum-fast invariant.** Even in extreme stochastic regimes where every species would be rare, the solver forcibly promotes the highest-count species back into the deterministic subset, preserving a deterministic substrate.
3. **Periodic scheduled re-evaluation.** Repartitioning happens at fixed intervals (every 100 steps) rather than every step or fully statically.

Together these mechanisms appear to be unoccupied space in the published literature on hybrid stochastic solvers.

**Why this matters for the patent.** Hybrid solvers are widely used in chemistry and systems biology simulators. The specific combination of hysteresis + minimum-fast + periodic re-evaluation looks novel on a first prior-art pass and would give Cognisom a defensible position in the broader market of stochastic simulation software, well beyond its cancer-immunology focus.

**Code evidence — single highest-priority file for the patent attorney to inspect.**

- `gpu/hybrid_solver.py` lines 450–543: the `HaseltineRawlingsPartitioner.partition()` method. Threshold = 100 copies (line 286). Hysteresis = ±20%, giving lower 80 / upper 120 (lines 483–484).
- Same file, lines 531–537: the minimum-fast-species invariant. If all species would otherwise partition to "slow," the highest-mean species is promoted.
- Same file, line 582 and 747–772: the repartition trigger. Every `repartition_interval=100` steps, the partition is re-evaluated.
- Same file, lines 774–826: the branching dispatch that selects pure-ODE, pure-SSA, or coupled-hybrid step based on partition cardinality.
- Same file, lines 827–858: the coupled step. RK4 or Euler integration for the fast subset (with non-negativity clamp at line 858), then SSA for the slow subset using current fast-subset values.

**Honest maturity caveat the attorney must know.** The CUDA kernels for GPU execution of the hybrid step exist as runtime-compiled C strings in the source (`gpu/hybrid_solver.py` lines 65–251) but the dispatcher falls back to CPU execution; line 939 of that file contains the literal comment `# For now, fall back to CPU`. The hysteresis + minimum-fast logic is fully implemented in Python and runs in tests. Therefore: **a claim limited to "computer-implemented" and not specifically "GPU-accelerated" is enabled by the current code.** A GPU-specific claim would currently fail USPTO §112 enablement until kernel launch is fully wired up.

**Draft independent claim (illustrative).**

> A computer-implemented method for hybrid stochastic-deterministic simulation of a chemical reaction network, comprising:
>
> (i) classifying each chemical species into a fast subset or a slow subset by comparing a recent abundance estimate to an upper threshold and a lower threshold defining a hysteresis band, wherein a species currently in the fast subset is reclassified to the slow subset only on its abundance estimate falling below the lower threshold, and a species currently in the slow subset is reclassified to the fast subset only on its abundance estimate exceeding the upper threshold;
>
> (ii) maintaining a minimum cardinality of one or more species in the fast subset by promoting a highest-abundance slow species when said cardinality would otherwise be zero;
>
> (iii) advancing fast-subset species using a deterministic ordinary-differential-equation integrator and slow-subset species using a stochastic event integrator, each subsystem reading the present state of the other subsystem without explicit truncation-error correction terms; and
>
> (iv) re-classifying species at a fixed step interval thereby producing a sequence of species partitions over simulation time.

**What is well-known prior art and must be disclaimed.** Haseltine-Rawlings partitioning (2002). Tau-leaping (Gillespie 2001). RK4 and Adams-Moulton integrators (textbook). CVODE-style adaptive stepping (SUNDIALS, 2005+).

**Prior-art search seeds.**

- "hybrid stochastic deterministic solver hysteresis"
- "adaptive partitioning Gillespie tau-leaping"
- "Haseltine Rawlings dynamic repartition"
- Cao, Gillespie, Petzold (multi-scale stochastic simulation papers 2002–2010 — all prior art)
- Existing solvers to differentiate from: STOCKS, COPASI, BioNetGen NFsim, StochKit2, MesoRD, Lattice Microbes

---

### Invention D — Batched GPU Gillespie SSA with Per-Cell xoshiro256** RNG

**Plain-English explanation.** When running thousands of cell simulations in parallel on a GPU, each simulated cell needs its own stream of random numbers. The cells must not "share luck" or the results are biased. Cognisom uses a high-quality random number generator (xoshiro256\*\*) with separate state for each cell, plus a specific shortcut for sampling the Poisson distribution (a common probability distribution in chemistry simulations) that switches between two methods depending on how many events are expected.

**Why this is recommended as trade secret, not patent.** Batched GPU stochastic simulation is published prior art (Komarov & D'Souza 2012; Lattice Microbes; Nobile et al. 2017). The xoshiro256\*\* RNG itself is published. The Knuth/Box-Muller switch is a textbook statistical recipe. Cognisom's implementation is competently engineered but the novelty is thin — claims would be very narrow and easy to design around. The practical recommendation is to keep the kernel layout as a trade secret rather than disclose it in a patent.

**Code evidence (for reference).**

- `gpu/ssa_kernel.py` lines 46–138: CUDA `tau_leap_step` kernel; one thread per cell; grid = ⌈n_cells/256⌉ blocks of 256 threads.
- Lines 67–74: inline xoshiro256\*\* macros (ROTL/NEXT) with per-cell 4×uint64 state.
- Lines 80–98: threshold-switched Poisson sampler; λ < 30 uses Knuth's method, λ ≥ 30 uses Box-Muller normal approximation.
- Lines 140–221: CUDA `direct_ssa_step` kernel; 64-slot stack-allocated cumulative-propensity array (line 178); max 100,000 steps per kernel invocation (line 558).

**Implementation maturity.** CUDA source compiles. CPU fallback (`tests/test_gpu.py`) is what is exercised in continuous integration. GPU performance has not been validated in the repository.

**Recommended posture:** Trade secret.

---

### Invention E — Spatial Stochastic Reaction-Diffusion with Grid-Hashed Bimolecular Pairing

**Plain-English explanation.** Inside the simulated tissue, individual molecules (such as exosomes or signaling proteins) can be modeled as point particles that move randomly and occasionally collide and react. To do this efficiently, Cognisom subdivides space into a grid of cells and only checks for collisions between particles in the same or neighboring grid cells. This avoids the wasteful all-pairs check.

**Why this is not a strong patent candidate.** Particle-based reaction-diffusion simulation is established prior art: Smoldyn (Andrews & Bray 2004) and ReaDDy (Schöneberg & Noé 2013) are the leading published platforms. Spatial hashing for collision detection is a textbook technique from molecular dynamics (LAMMPS, GROMACS) and computer graphics. Cognisom's implementation is competent but unlikely to clear novelty.

**Code evidence (for reference).**

- `gpu/smoldyn_solver.py` lines 59–108: `brownian_motion_step` CUDA kernel.
- Lines 184–221: `build_spatial_hash` kernel with `atomicAdd` to per-cell counters.
- Lines 224–280+: `bimolecular_reactions` kernel.

**Recommended posture:** Trade secret at best; not a patent candidate.

---

### Invention F — Closed-Loop Neoantigen → Tumor Dynamics Pipeline

**Plain-English explanation.** The "holy grail" application of Cognisom is the **closed loop**: a cancer mutation arises in a cell (Invention A); from that mutation, the simulator computes which protein fragments would be displayed on MHC-I in that cell (this part is implemented); the immune T cells inspect those fragments and decide whether to kill the cell (this part is currently a simple threshold heuristic, not full TCR-affinity matching); the tumor population dynamics evolve in response. **Closing this loop end-to-end** would be a significant patent: tying simulated mutation through simulated immune recognition through to simulated tumor outcome, all in one simulator.

Standalone neoantigen predictors exist (NetMHCpan, MHCflurry, MARIA) but they output static lists of predicted peptides — they do not feed dynamic tissue simulations. Tissue simulators exist but do not consume per-cell sequence-derived peptidomes. The combination is the novelty.

**Why this is deferred, not filed today.** Of the pieces in the loop, the peptide predictor exists (PWM-based MHC-I binding in `cognisom/genomics/neoantigen_predictor.py` lines 116–235; generates 8-, 9-, 10-, 11-mer peptides; scores against patient HLA alleles; ranks by affinity). The immune-recognition kernel in `gpu/spatial_ops.py` lines 200–240 is a thresholding heuristic, not biophysical TCR-affinity binding. Until the loop is closed end-to-end with a working test, claims must be drafted carefully.

**What is well-known prior art and must be disclaimed.** PWM-based MHC binding (Parker 1994). NetMHCpan, MHCflurry, MARIA. The standard immunopeptidomics workflow.

**Recommended posture.** Patent-pending-quality. File when the loop is closed. The economic value of this invention, if completed, could exceed the value of A+B+C combined, because it directly addresses immunotherapy response prediction — the multi-billion-dollar commercial question in oncology drug development.

**Prior-art search seeds.**

- NetMHCpan, MHCflurry, MARIA, IEDB Analysis Resource
- "agent-based simulation neoantigen presentation"
- "in silico immunotherapy response prediction tumor microenvironment"
- USPTO class G16B 20/00 (Bioinformatics — Genetic engineering)

---

### Invention G — Coupled Circadian Oscillator with Cell-Cycle Gating

**Plain-English explanation.** Human cells have internal biological clocks (circadian clocks) that run on an approximately 24-hour cycle. The clocks affect when cells divide, how they metabolize, and how active the immune system is. Cancer drugs often work better at certain times of day for this reason. Cognisom has the data structures and class skeletons for per-cell circadian clocks coupled to a global master clock, but the oscillator equations are not deeply implemented.

**Why this is not patent-ready today.** Implementation is shallow: phase advance is explicit Euler stepping, not the standard Goodwin or Leloup-Goldbeter ODE for the molecular circadian oscillator. No protein-level oscillation is simulated. The immune circadian modulation is documented in markdown only.

**Recommended posture.** Defer. Reassess once oscillator equations are fully wired and protein-level oscillation is exercised in tests.

---

### Invention H — Epigenetic Regulation of Gene Expression

**Plain-English explanation.** "Epigenetics" refers to chemical modifications on DNA and the proteins that package it (histones) that turn genes on or off without changing the DNA sequence itself. Cancer cells often hide tumor-suppressor genes this way (a process called "hypermethylation"). Cognisom has data structures for per-gene methylation and histone marks, with a scalar multiplier applied to transcription rate.

**Why this is not patent-ready today.** A scalar chromatin-state modifier on transcription rate is a modeling choice, not an algorithm. Epigenetic agent-based modeling has been explored in published work (e.g., Zaidi 2017). The patent path is weak unless integrated with sequence-grounded mutation tracking (Invention A) so that mutation × methylation × transcription rate forms an irreducible joint state — that integration is not yet evident in the code.

**Recommended posture.** Trade secret today; future patent if the joint state with Invention A is built.

---

### Invention I — Event-Bus-Driven Modular Multi-Scale Simulation Engine

**Plain-English explanation.** This is the framework that ties Cognisom together: the central scheduler that advances time, the rule that modules announce events on a bus and other modules subscribe to them, the plug-in architecture for swapping solvers.

**Why this is not patentable.** Event-bus modular simulation is well-known prior art (DEVS formalism, Repast Simphony, NetLogo extensions, ROS in robotics, Akka in software engineering). Cognisom's combination is good engineering, not invention.

**Recommended posture.** Not patentable. Use as an ergonomic differentiator to customers, not as IP.

---

## 8. Implementation Maturity Re-Stated Honestly

Patent law (USPTO §112) requires that filings disclose the invention in enough detail that a person skilled in the art could build it. This means claims must track *implemented* code, not aspirational documentation. Roughly 20 to 30 percent of Cognisom is production-grade implemented code; another 50 percent is class scaffolding (data classes and `update()` stubs without operative dynamics); and the remaining 20 to 30 percent is documentation-only.

The following subsystems are **documented but not built** to a degree sufficient to support patent claims today. Claims should not depend on these.

| Subsystem | Where in repo | What is missing |
|---|---|---|
| Multi-GPU domain decomposition | `gpu/domain_decomposition.py` (339 lines) | Partition geometry is implemented; NCCL/MPI/peer-access communication is not. No working multi-GPU run exists. |
| ML surrogates for PDE, metabolism, signaling | `ARCHITECTURE.md`, `FRONTIER_CAPABILITIES.md` | Zero PyTorch/JAX/TensorFlow code. No training pipeline. No surrogate inference path. |
| Newton physics / cell mechanics on GPU | `gpu/physics_interface.py:495` | Literal `raise NotImplementedError`. |
| Full clonal evolution lineage tracking | `ARCHITECTURE.md:289–306` | No lineage data structure. Mutations are scalar parameter scalings. |
| Omniverse real-time integration | `omniverse/` (8 files) | Kit extension scaffolding only; not driven by the live simulation. |
| TCR affinity-matched immune killing | `modules/immune_module.py`, `gpu/spatial_ops.py` | Currently threshold heuristics, not biophysical binding kinetics. |
| Patient VCF → simulation pipeline | `bridge/` | Bridge files exist; the data pipeline is not closed. |

This list is included precisely so that the attorney can draft claims that do *not* depend on these subsystems, avoiding §112 enablement rejections.

---

## 9. Quick Verification Steps for the Attorney

Before drafting claims, an attorney (or technical consultant) can sanity-check the inventions in roughly 30 minutes:

1. **Invention A — sequence-grounded mutations.** Open `engine/py/molecular/nucleic_acids.py` and read lines 389–426. Confirm that `introduce_oncogenic_mutation('G12D')` edits position 35 of the KRAS sequence from "GGT" to "GAT". Run the simple example: `python -c "from cognisom.engine.py.molecular.nucleic_acids import Gene; g = Gene(name='KRAS', sequence='ATG...'); g.introduce_oncogenic_mutation('G12D'); print(g.sequence[33:36])"`.
2. **Invention B — exosome transformation.** Find an example or demo in `examples/` that exercises `ExosomeSystem`; run it; observe the reported transformation events.
3. **Invention C — hybrid solver hysteresis.** Read `gpu/hybrid_solver.py` lines 480–543. Confirm the deadband logic (species at count 90 does not switch partitions). Optionally write a 30-line reproduction that toggles a species' count around 100 and verifies it remains in its current partition until it crosses 80 or 120.
4. **Invention C — GPU enablement gap.** Read `gpu/hybrid_solver.py` lines 939–946 and confirm the literal `# For now, fall back to CPU` comment. This is the line that constrains how claims for Invention C can be drafted.
5. **Invention F — closed-loop maturity gap.** Run the neoantigen predictor at `cognisom/genomics/neoantigen_predictor.py` against a sample mutation; verify it returns a ranked peptide list. Then attempt the closed loop (mutation → peptide → MHC presentation → T-cell kill in tissue simulation) and document where it breaks. The location of the break determines what additional implementation must be done before Invention F can be filed.

---

## 10. Prioritized Filing Recommendations

| Rank | Invention | File a patent? | Posture |
|---|---|---|---|
| 1 | **A — Sequence-grounded mutation → phenotype** | **Yes — file first** | Independent claim plus dependent claims tying to specific oncogenic loci (KRAS, TP53, BRAF). |
| 2 | **B — Exosome horizontal transformation** | **Yes — file alongside A** | Pair with A in the same continuation family so they share priority date and reinforce each other. |
| 3 | **C — Hybrid SSA/ODE with hysteresis-banded auto-repartition** | **Yes — narrow claim** | Limit claim language to "computer-implemented", not "GPU", until GPU kernels are fully wired. Broader market relevance than oncology — applicable to any stochastic chemistry simulator. |
| 4 | F — Neoantigen → tumor dynamics closed loop | **Defer** | Patent-pending posture. Close the loop in code first, then file. |
| 5 | D — Batched GPU SSA micro-optimizations | **No — trade secret** | Disclosure in patent would be narrow and easy to design around. |
| 6 | E — Spatial reaction-diffusion with grid hashing | **No** | Standard HPC technique; weak claim. |
| 7 | G, H — Circadian, epigenetic | **Defer** | Build dynamics first. |
| 8 | I — Event-bus engine architecture | **No** | Standard engineering pattern, not invention. |

**Items recommended for trade secret rather than patent:**

- Specific oncogenic mutation parameter sets (KRAS, TP53, BRAF rate-constant scalings)
- Cancer-cell `mhc1_expression = 0.3` and analogous immune-evasion baseline calibrations
- Hybrid solver internal thresholds (100 copies, ±20% hysteresis, 100-step repartition interval)
- GPU kernel block sizes, max-reaction stack depths (64), max-step counts (100,000)
- Any validation benchmark sets and patient-data calibrations

---

## 11. Competitive Differentiation at a Glance

| Capability | Cognisom | PhysiCell | BioDynaMo | Lattice Microbes | Gell | COPASI | Virtual Cell |
|---|---|---|---|---|---|---|---|
| Per-cell ATCG nucleotide state | YES | no | no | no | no | no | no |
| Codon-level mutation phenotype | YES | no | no | no | no | no | no |
| Exosome cargo + horizontal transfer | YES | no | no | no | no | no | no |
| Hybrid SSA/ODE with hysteresis-banded repartition | YES | no | no | partial | no | partial | no |
| Batched GPU SSA across N cells | YES | no | yes | yes | yes | no | no |
| Spatial PDE diffusion grid | yes | yes | yes | yes | no | no | yes |
| Coupled neoantigen → tumor dynamics | partial | no | no | no | no | no | no |
| Multi-GPU scaling | no | yes | yes | no | no | no | no |
| Million-cell tissues | aspirational | yes | yes | no | yes | no | no |

The defensible white space for Cognisom is rows 1 through 4 and row 7. Rows 5 and 6 are crowded with existing competitors. Rows 8 and 9 are aspirational and should not be referenced in claim language today.

---

## 12. Risks and Caveats Re-Stated Plainly

**Subject-matter risk (USPTO §101).** Pure simulation methods can be rejected in the United States as "abstract ideas" under the Alice/Mayo two-step test. Best mitigation: anchor claims to a *specific computational improvement* (e.g., "the hysteresis-banded partitioner reduces solver oscillation thereby improving convergence stability of the simulator," or "per-cell sequence storage enables phenotype prediction that was not previously possible in agent-based simulators"). Use language about *improving the functioning of the computer* or *improving the technical field of stochastic chemical simulation* where appropriate.

**Enablement risk (USPTO §112).** Multiple subsystems are documented but not built. Claims must track implemented code. Ignore the markdown roadmaps for filing scope. The aspirational subsystem list in §8 is the explicit do-not-claim list.

**Novelty risk (USPTO §102/103).** PhysiCell and BioDynaMo are actively developed competitors in 2024–2026. Check their latest releases at filing time — they may have added sequence-aware or hybrid-solver features after the date of this document. The attorney should re-run the prior-art search seeds in §7 against the most recent commits and papers.

**Prior public disclosure.** This document, the GitHub repository, conference talks, blog posts, and any READMEs visible to the public may constitute public disclosure under foreign-jurisdiction novelty rules (especially the European Patent Convention, which has no grace period). Confirm filing dates against any prior public disclosures. If a PCT filing is contemplated, file before any further public disclosure.

**Third-party code in the repository.** The repository bundles a full unmodified clone of **PhysiCell** (BSD-3 licensed, permissive). Verify that PhysiCell is treated as a third-party reference only and not woven into any claim language or relied upon for enablement. PhysiCell's MIT-style license is compatible with closed-source commercial use, but downstream license attribution must be preserved.

**Inventor disclosure.** All inventors must be named on each filing. Currently the named inventor is David Walker. If other developers have contributed to the specific algorithmic novelty of Inventions A, B, or C, they may need to be added as co-inventors — an underspecified inventorship can void a patent.

---

## 13. Pre-Filing Workstreams

Tasks for the inventor to complete before the attorney begins drafting:

1. **Inventor-side prior-art search** using the search seeds in §7. Spend approximately four hours on:
    - Google Patents — the listed search seeds plus USPTO classes G16B 5/00, 5/20, 20/00, 25/00, 35/00.
    - Lens.org — same searches; read the closest 20 to 30 hits.
    - PubMed and bioRxiv — "agent-based cancer simulation," "exosome simulation," "neoantigen tumor model," "hybrid stochastic deterministic solver hysteresis."
    - PhysiCell, BioDynaMo, Lattice Microbes, CompuCell3D — read their most recent paper abstracts (2024–2026) and feature lists; note any sequence-aware or hybrid-solver additions.
2. **Freedom-to-operate notebook.** Compile the closest 10 prior-art references for each invention. The attorney will need this to draft around prior art.
3. **Enablement evidence.** Run the example scripts in `examples/` end-to-end against a clean checkout. Capture outputs to disk. Save the run log. This is §112 evidence.
4. **Geographic scope decision.** US-only or PCT? US is cheapest; PCT is required for foreign rights but adds substantial cost and complexity. The decision affects how aggressively claims should be drafted.
5. **Repo snapshot.** Tag the repository at the filing date: `git tag patent-snapshot-2026-MM-DD`. Anything implemented after the priority date can be added in a continuation or CIP.
6. **One-page commercial rationale.** Write a short document covering: which markets (oncology drug developers, immunotherapy biotechs, academic research labs), which buyers (CTOs at biotechs, principal investigators at academic centers), which competitors (PhysiCell open-source community, commercial bioinformatics firms). The attorney will use this to size claim scope.

---

## 14. Appendix — Code Excerpts for the Three Recommended Filings

The following short excerpts illustrate the specific code being referenced. Line numbers are as of the repository state on 2026-05-11. The attorney should verify these against a tagged snapshot.

### Invention A — Oncogenic Mutation Table

From `engine/py/molecular/nucleic_acids.py` (around lines 389–426, simplified):

```
def introduce_oncogenic_mutation(self, mutation_name: str):
    """Edit the gene's sequence in place to introduce a known cancer mutation."""
    mutations = {
        'KRAS_G12D': (35, 'GGT', 'GAT'),  # Position 35: GGT -> GAT
        'KRAS_G13D': (38, 'GGC', 'GAC'),
        'BRAF_V600E': (1799, 'GTG', 'GAG'),
        'TP53_R175H': (524, 'CGC', 'CAC'),
        'TP53_R248W': (742, 'CGG', 'TGG'),
    }
    pos, old, new = mutations[mutation_name]
    self.sequence = self.sequence[:pos] + new + self.sequence[pos+3:]
    self.oncogenic = True
    self.recompute_chemical_properties()
```

### Invention B — Exosome Brownian Step

From `engine/py/molecular/exosomes.py` (around lines 158–161, simplified):

```
def brownian_step(self, dt: float):
    """Advance the exosome by one Brownian-motion step."""
    dt_sec = dt * 3600.0  # hours -> seconds
    sigma = math.sqrt(2.0 * self.diffusion_coeff * dt_sec)
    self.position += np.random.normal(0.0, sigma, size=3)
```

### Invention C — Hysteresis-Banded Partition

From `gpu/hybrid_solver.py` (around lines 483–537, simplified):

```
def partition(self, state, current_partition=None):
    mean_counts = np.mean(state, axis=0)
    lower = self.threshold * (1.0 - self.hysteresis)  # 80 if threshold=100, hysteresis=0.2
    upper = self.threshold * (1.0 + self.hysteresis)  # 120
    fast = set()
    for sp_idx in range(self.n_species):
        m = mean_counts[sp_idx]
        if current_partition and sp_idx in current_partition.fast_species:
            if m > lower:
                fast.add(sp_idx)
        elif current_partition and sp_idx in current_partition.slow_species:
            if m > upper:
                fast.add(sp_idx)
        else:
            if m > self.threshold:
                fast.add(sp_idx)
    # Minimum-fast invariant
    if len(fast) < self.min_fast and self.n_species > 0:
        ranked = np.argsort(-mean_counts)
        for sp_idx in ranked:
            fast.add(int(sp_idx))
            if len(fast) >= self.min_fast:
                break
    slow = set(range(self.n_species)) - fast
    return HybridPartition(fast_species=fast, slow_species=slow)
```

---

## End of Document

This document is intended for review by a patent attorney and for use by the inventor to drive prior-art research. It is not a patent application and does not constitute legal advice. Hand-off package recommended:

- This document.
- A snapshot of the repository tagged at the filing date.
- The freedom-to-operate notebook from §13.2.
- The one-page commercial rationale from §13.6.

For follow-up questions on any specific invention, see the file:line citations in §7 — every claim sketched in this document can be verified against the cited code in under a minute by an engineer familiar with Python.
