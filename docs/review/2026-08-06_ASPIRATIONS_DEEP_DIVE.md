# Cognisom — Deep Dive on All Aspirations

**Date:** 2026-08-06
**Question asked:** *"Perhaps there are many different applications here — deep dive on all of its aspirations."*
**Answer:** Yes. There are **at least 26 distinct product aspirations** in this repository,
spanning **8 unrelated markets** and **two different companies**. This document enumerates all
of them, grades each on real-vs-promised, and proposes a portfolio triage.

---

## 0. The central finding

The repository is not one application. It is **roughly 6–8 companies' worth of product surface
area** built into a single tree:

| Measure | Value |
|---|---|
| Python LOC | ~246,000 (~151k in the newer `cognisom/` tree, rest duplicated legacy) |
| Distinct subpackages | 34 |
| Dashboard pages (product surfaces) | 43 |
| Markdown strategy/roadmap docs | ~90 |
| Distinct product aspirations | ~26 |
| Distinct addressable markets | ~8 |
| Companies described | 2 (eyentelligence / "Cognisom Therapeutics") |

**This is the root cause of the quality findings in the companion review.** The problem is not
laziness or lack of ambition — it is that ~26 aspirations share one team's attention. Every
aspiration got *plumbing*; almost none got *validation*. That is exactly the failure signature
seen in the simulation review: real HTTP clients, real USD, real CUDA kernels, real Postgres —
wrapped around a T-cell recognition step that is a coin flip and a validation module that
grades itself with fabricated PMIDs.

Notably, **stub density is genuinely low** (0–19 TODO/NotImplementedError hits per package).
The code is *written*. It is the *scientific grounding and validation* that is thin — because
attention was spread across 26 fronts.

---

## 1. Full aspiration inventory

Graded: **REAL** (works, evidence-backed) · **BUILT** (code runs, unvalidated) ·
**SCAFFOLD** (structure present, thin) · **DOC** (documentation only)

### Cluster A — Core simulation science

| # | Aspiration | Evidence | Grade |
|---|---|---|---|
| 1 | Multi-scale cellular simulator (molecule→organ) | `engine/`, `modules/` (15 modules), ~100 cells CPU | BUILT |
| 2 | VCell-parity GPU solver suite (ODE, SSA, hybrid, Smoldyn, BNGL, FBA, PDE) | `gpu/` 8.7k LOC — real CUDA for SSA/Smoldyn/diffusion; ODE+FBA GPU paths broken | MIXED |
| 3 | "Digital organs" — 1M+ cell full-organ simulation | Roadmap only; current ceiling ~100 cells CPU | DOC |
| 4 | Prostate cancer specialization (AR, zonal anatomy, CRPC) | `PROSTATE_TISSUE_ARCHITECTURE.md`, page 19 | SCAFFOLD |
| 5 | Metastasis modelling (EMT, CTC, bone tropism, diapedesis) | Page 25 diapedesis, strategic plan Phase B | SCAFFOLD |
| 6 | The "Digital Human" (all organs, 10+ yr) | `VISION_AND_ASPIRATIONS.md` only | DOC |

### Cluster B — Precision oncology / clinical

| # | Aspiration | Evidence | Grade |
|---|---|---|---|
| 7 | Molecular Digital Twin (VCF→treatment prediction) | `genomics/` 7.7k LOC — front half real, predictor is a heuristic | MIXED |
| 8 | MAD clinical decision support + FDA Non-Device CDS | `mad/` 3.4k LOC — real audit/provenance; validation tautological | BUILT |
| 9 | Neoantigen vaccine design | `neoantigen_predictor.py`, MHCflurry, page 31 | BUILT |
| 10 | Clinical genomics cloud pipeline (Parabricks, HealthOmics, OptiType) | Real `docker run --gpus`, real boto3 | BUILT |
| 11 | Virtual clinical trials / in-silico trials | Vision docs; `clinical_trials.py` | SCAFFOLD |
| 12 | Retrospective cohort validation (TCGA/SU2C/GIAB/IEDB) | `validation/` 5.7k LOC, real cBioPortal + IEDB APIs | REAL |

### Cluster C — Visualization & standards

| # | Aspiration | Evidence | Grade |
|---|---|---|---|
| 13 | NVIDIA Omniverse 3D visualization + real-time streaming | `omniverse/` **15.9k LOC — 2nd largest package**, real `pxr` USD | BUILT |
| 14 | **Bio-USD as a ratified AOUSD industry standard** | `biousd/` 3.2k LOC, 31 schema types, 0 stubs; Roadmap Phase 9 | BUILT |
| 15 | Semantic zoom (representation switching by camera scale) | `omniverse/semantic_zoom.py` — genuinely novel UX idea | BUILT |
| 16 | Microscopy imaging → simulation geometry | `imaging/` 1.9k LOC (Otsu, watershed, Cellpose/StarDist, mesh) | BUILT |

### Cluster D — Knowledge & data platform

| # | Aspiration | Evidence | Grade |
|---|---|---|---|
| 17 | Biological Entity Library (31 entity types, 36 relations, FAIR) | `library/` **12.2k LOC, 2 stub hits**, Postgres+SQLite, audit log | REAL |
| 18 | Entity-driven simulation (add a drug entity → new treatment) | `parameter_resolver.py` — genuinely elegant architecture | REAL |
| 19 | External database federation (NCBI, cBioPortal, OncoKB, IEDB) | `ncbi/`, `ingestion/`, page 18 | BUILT |

### Cluster E — AI / agents / MLOps

| # | Aspiration | Evidence | Grade |
|---|---|---|---|
| 20 | Autonomous research agents (literature monitor, knowledge graph, data sync) | `agents/` 2.1k LOC, real PubMed/bioRxiv/arXiv | BUILT |
| 21 | Data flywheel (distillation, model registry, A/B, feedback) | `flywheel/` 1.6k LOC — an MLOps product in its own right | SCAFFOLD |
| 22 | NVIDIA BioNeMo NIM layer (11 models: RFdiffusion, DiffDock, ESM2, Evo2, OpenFold3, Boltz-2…) | `nims/` — real `health.api.nvidia.com` calls | BUILT |
| 23 | MCP server exposing MAD to LLM clients | `mad/mcp_server.py`, `mcp_tools.py` | BUILT |

### Cluster F — Researcher workflow

| # | Aspiration | Evidence | Grade |
|---|---|---|---|
| 24 | Paper Studio — automated LaTeX manuscript + figure generation | `workflow/` 2.9k LOC, Jinja2 templates, BibTeX | BUILT |
| 25 | Reproducible run management / artifact store / scenario builder | `workflow/artifact_store.py`, `run_manager.py` | BUILT |

### Cluster G — Commercial platform

| # | Aspiration | Evidence | Grade |
|---|---|---|---|
| 26 | Multi-tenant SaaS (Cognito auth, orgs, subscriptions, security, admin) | `auth/` 2.3k LOC, real Cognito | BUILT |
| 27 | Patent pipeline / IP evidence generation | Page 42, `docs/patent/` | BUILT |
| 28 | Investor relations portal | Page 41 (hardcoded password) | BUILT |

### Cluster H — Outside biology entirely

| # | Aspiration | Evidence | Grade |
|---|---|---|---|
| 29 | **eyentell** — privacy-first AI client management for lawyers/therapists | Business plans only; ~no code in repo | DOC |
| 30 | Multi-platform content operation (LinkedIn/X/TikTok/YouTube scripts) | `MULTI_PLATFORM_CONTENT_STRATEGY.md` 36k, `linkedin_articles/` | REAL(!) |
| 31 | Education / training platform | Vision docs | DOC |

---

## 2. What this inventory reveals

### 2.1 The strongest assets are not the ones being marketed

The marketing centers on **"GPU-accelerated cellular digital twin for precision oncology"** —
which is the *weakest-validated* claim in the repo. Meanwhile three genuinely strong,
under-marketed assets sit in plain sight:

1. **The Biological Entity Library** (#17/#18) — 12.2k LOC, near-zero stubs, Postgres-backed,
   FAIR-compliant, 31 entity types with typed relationships, an audit log, and an elegant
   *entity-driven simulation* pattern where adding a drug entity automatically creates a
   treatment option. This is a real data-and-architecture asset that stands alone and would
   still be valuable if every simulation claim were withdrawn.

2. **Bio-USD + the AOUSD standardization play** (#14) — this is the single most strategically
   distinctive idea in the repository and it is buried at "Phase 9" behind nine other phases.
   A ratified USD extension for biology would create ecosystem position that no amount of
   simulation accuracy can buy, it aligns naturally with the existing NVIDIA relationship, and
   it is *cheap* relative to clinical validation. `biousd/` already has 3.2k LOC and zero stub
   markers.

3. **The retrospective validation harness** (#12) — real cBioPortal, IEDB, and GIAB
   integrations already written. The infrastructure to *honestly* validate already exists; it
   is simply pointed at a tautological metric today.

### 2.2 The portfolio is structurally incoherent

- **Two companies.** `eyentell` (SaaS for lawyers and therapists, projected to out-earn the
  cancer platform by Year 5: $10–18M vs $8–12M) shares essentially nothing with cellular
  simulation except the words "GPU" and "vector database." In diligence this reads as lack of
  focus, and it is the first thing an investor will challenge.
- **Three identities.** "eyentelligence" (dual-platform holdco) vs "Cognisom Therapeutics"
  (investor page — implies drug development) vs "cognisom" (open-source research tool). These
  imply three different regulatory postures and three different valuation multiples.
- **Eight markets.** Research software, clinical decision support, drug discovery (NIM),
  3D visualization, data standards, MLOps, scientific publishing, and professional-services
  SaaS. No team of this size serves eight markets.

### 2.3 Aspiration inflation is measurable

The roadmaps promise a **10-phase sequence** ending in "clinical scale and personalized twins,"
plus a 6-phase GPU roadmap, plus a 6-phase biology roadmap, plus a 6-phase strategic
implementation plan — four overlapping master plans, each with its own phase numbering. The
`CELLULAR_BIOLOGY_ROADMAP.md` schedules 12 weeks of work; `GPU_SCALING_ROADMAP.md` schedules
16 weeks; the strategic plan schedules 8+ months; the complete roadmap schedules 10 phases.
These are not reconciled with each other.

Meanwhile `WHAT_IT_CAN_DO.md` — the most honest document in the set — records the actual
tested state: single-cell growth, 7 genes, ~100 cells, CPU.

---

## 3. Portfolio triage — a recommendation

The strategic problem is not "which aspiration is best" but "how many can one team validate."
Realistically: **one, maybe two.**

### Tier 1 — Keep and go deep (the defensible core)

**Bio-USD / AOUSD standards + Entity Library + Omniverse visualization.**
Rationale: it is real code today, it is genuinely differentiated, it requires *no clinical
validation* to be valuable, it plays to the existing NVIDIA relationship, and standards
positions compound. A ratified biology extension to OpenUSD is a durable moat; a slightly
better tumor-growth heuristic is not. This also converts the 15.9k-LOC Omniverse investment
from "a demo" into "the reference implementation of a standard."

**Concretely:** promote Phase 9 to Phase 1. Publish the `.usda` schemas, apply to AOUSD, form
the Life Sciences Interest Group, and let the simulator be the reference implementation rather
than the product.

### Tier 2 — Keep, but reframe as Research Use Only

**Molecular variant-effect engine + neoantigen prediction + retrospective validation.**
These are scientifically legitimate. Reframed as a *research* tool with honest retrospective
metrics (not "clinical decision support," not "digital twin"), they support grants and
academic collaborations. Repoint the existing validation harness at real outcome labels.

### Tier 3 — Park (code stays, marketing stops)

Solver suite (fix or retract the GPU-ODE/FBA claims first), imaging, NIM layer, Paper Studio,
research agents, flywheel, MCP server, SaaS/billing, patent pipeline, investor portal. All are
built; none is a business today. Freeze feature work; keep them as platform capability.

### Tier 4 — Cut

- **eyentell.** Different market, different buyer, different regulatory surface, no code here.
  Spin it out or shelve it. Its presence materially damages the cancer-platform narrative.
- **"Digital Human," multi-organ, virtual clinical trials, FDA clearance roadmap.** Keep them
  in a vision document explicitly labeled as vision — remove them from capability and investor
  materials.
- **The content operation at current volume.** 36k words of TikTok/YouTube/X strategy plus a
  LinkedIn series is a marketing department's output for a product that cannot yet substantiate
  its headline claim. Marketing ahead of evidence is what created the overclaiming problem.

### The single highest-value action

**Reduce the phase count from ~30 to 3, and reduce the claim surface to what the validation
harness can actually prove.** Every quality problem in the companion review traces back to
attention spread thin. Cutting aspirations is the fix — not adding engineering.

---

## 4. Summary table — the 8 markets, ranked by defensibility

| Market | Asset quality | Validation needed | Capital needed | Recommendation |
|---|---|---|---|---|
| Bio data standards (AOUSD) | High | None | Low | **Go** |
| Biological knowledge base | High | Low | Low | **Go** |
| 3D bio-visualization | High | None | Medium | **Go (as ref. impl.)** |
| Research simulation software | Medium | Medium | Medium | Reframe RUO |
| Drug discovery (NIM/BioNeMo) | Medium | High | High | Park |
| Scientific publishing automation | Medium | Low | Low | Park |
| Clinical decision support | Low | Very high | Very high | Reframe or exit |
| Professional-services SaaS (eyentell) | None here | N/A | High | **Cut / spin out** |

---

## 5. Cross-reference

See `2026-08-06_SIMULATION_QUALITY_AND_BUSINESS_REVIEW.md` for the code-level fidelity findings
that motivated this portfolio analysis, including the five ranked correctness/integrity fixes.
