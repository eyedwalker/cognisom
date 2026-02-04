# COGNISOM COMPLETE ROADMAP
## Updated February 2026 — eyentelligence inc.

---

## CURRENT STATE — What's Built

| Layer | Components | Status |
|-------|-----------|--------|
| **Simulation Engine** | 9 modules (molecular, cellular, immune, vascular, lymphatic, spatial, epigenetic, circadian, morphogen) | Working |
| **NIM Clients** | 11 NVIDIA BioNeMo APIs (MolMIM, GenMol, DiffDock, ESM2, OpenFold3, Boltz-2, Evo2, RFdiffusion, ProteinMPNN, AlphaFold2-Multimer, MSA-Search) | Working |
| **Dashboard** | 14 Streamlit pages (ingestion, discovery, simulation, admin, molecular lab, research feed, research agent, subscriptions, account, security, validation, 3D viz, organization, entity library) | Working |
| **API** | Flask REST with auth middleware, 6 protected + 4 auth endpoints | Working |
| **Research** | Agent with 10 tools (5 DB + 5 NIM), PubMed/bioRxiv/arXiv feeds, 22-source subscription manager | Working |
| **Auth** | PBKDF2 auth, 4 roles (viewer/researcher/org_admin/admin), 4-tier subscriptions, multi-tenant orgs | Working |
| **Bridge** | Discovery pipeline, structure bridge, drug bridge | Working |
| **GPU** | Backend selection, cell ops, spatial ops, FBA solver, diffusion solver | Working |
| **Bio-USD** | OpenUSD schema (16 prim types), simulation-to-USD converter, SBML-to-USD converter, bidirectional sync | Working |
| **Entity Library** | 99 curated entities (genes, proteins, drugs, pathways, cell types, metabolites, mutations), SQLite+FTS5, NCBI/UniProt loaders | Working |
| **Deploy** | Dockerfile.prod, ECS Fargate, ECR, Route53, ACM, live at cognisom.com | Production |

**Completed since last update:** Bio-USD schema (Phase 5C), biological entity library with 99 entities (Phase 5A), library management UI (Phase 5B), SBML-to-USD converter (Phase 6), bidirectional sync engine (Phase 6), security hardening, multi-tenant organizations, 4-tier subscriptions, 14 dashboard pages, AWS deployment.

**Remaining:** AI maintenance agents (Phase 7), full Omniverse integration (Phase 8), AOUSD governance (Phase 9), FAIR data enrichment, kinetics-to-physics mapping, clinical scale (Phase 10).

---

## PHASE 1 — DEPLOY & STABILIZE
*Get real users on the platform*

| Task | Detail |
|------|--------|
| Docker build test | Validate Dockerfile.prod builds clean |
| AWS ECS deploy | Run deploy.sh, stand up Fargate service |
| Security hardening | Change default admin, rotate SECRET_KEY, restrict CORS to real domain |
| Smoke test | All 10 pages load, NIM endpoints respond, auth flow works |
| Domain + TLS | Route53 domain, ACM certificate, nginx TLS termination |

**Deliverable:** Live at `https://cognisom.eyentelligence.com`

---

## PHASE 2 — DATA VALIDATION & BENCHMARKING
*Prove the simulation produces biologically meaningful results*

| Task | Detail |
|------|--------|
| Real scRNA-seq pipeline | Load actual prostate cancer datasets (TCGA-PRAD, GSE176031) through ingestion pipeline |
| Simulation benchmarks | Validate 9 modules against published experimental data — tumor spheroid growth curves, immune infiltration rates, metabolic flux distributions |
| Parameter calibration | Bayesian inference on key rate constants using real data |
| Performance profiling | Identify bottlenecks — which modules dominate wall-clock time |
| Test suite expansion | Integration tests with golden-reference outputs |

**Deliverable:** Validation report showing simulation matches 3+ published experimental datasets

---

## PHASE 3 — GPU ACCELERATION
*Scale from ~100 cells to 100,000+ cells*

| Task | Detail |
|------|--------|
| CUDA SSA kernels | Batched Gillespie algorithm — 1 warp per cell |
| GPU PDE solver | Finite-difference diffusion on 3D grid (O2, glucose, cytokines) |
| GPU FBA solver | cuSOLVER for metabolic flux balance |
| Memory layout | Structure-of-arrays for coalesced GPU reads |
| Multi-GPU | Domain decomposition + NCCL halo exchange |

**Deliverable:** 100k cells at 1-10 sec/step on single H100

---

## PHASE 4 — 3D VISUALIZATION FOUNDATION
*Learn spatial bio rendering before tackling USD*

| Task | Detail |
|------|--------|
| Mol* integration | Protein structure viewer embedded in molecular lab page |
| vtk.js cell renderer | 3D cell population rendering (position, type, state as color) |
| Spatial field viewer | Volume rendering for O2/glucose/cytokine concentration fields |
| Contact map / network viz | Interactive graphs for cell-cell interactions |
| Export formats | PDB, glTF, VTK for external tools |

**Deliverable:** Interactive 3D visualization of simulation state in the browser

---

## PHASE 5 — BIO-USD SCHEMA FOUNDATION + BIOLOGICAL LIBRARY
*This is where the Bio-USD proposal enters the roadmap*

### 5A — Biological Entity Library

| Task | Detail |
|------|--------|
| Entity catalog schema | Define the complete taxonomy: Gene, Protein, Metabolite, Cell Type, Tissue Type, Organ, Pathway, Receptor, Ligand, Drug, Mutation |
| Entity database | PostgreSQL + pgvector — every entity has: name, synonyms, description, ontology IDs (GO, ChEBI, CL, UBERON), embedding vector, source references |
| Data loaders | Batch import from UniProt, NCBI Gene, Reactome, CellMarker, DrugBank, ChEMBL |
| FAIR metadata | Every entity tagged with Bioschemas.org vocabulary — FAIR (Findable, Accessible, Interoperable, Reusable) compliance |
| Entity relationships | Typed edges: binds_to, activates, inhibits, part_of, located_in, expressed_in, metabolizes |

### 5B — Library Management UI

| Task | Detail |
|------|--------|
| Entity browser page | Search/filter/browse all biological entities with faceted navigation |
| Entity detail view | Full card with properties, relationships, 3D structure preview, linked papers |
| Relationship graph | Interactive force-directed graph of entity relationships |
| Curation interface | Admin UI to add/edit/deprecate entities, merge duplicates, add annotations |
| Import/export | Bulk CSV/JSON import, SBML export, SBOL export |
| Version history | Track all changes to entity definitions with audit trail |

### 5C — Bio-USD Schema Drafting

| Task | Detail |
|------|--------|
| `BioUnit` IsA schema | Abstract `UsdTyped` base class — all biological prims inherit from this |
| `BioCell` IsA schema | Inherits `UsdGeomXformable` — position, orientation, scale in 3D space. Properties: cell_type, phase, volume, viability |
| `BioProtein` IsA schema | Inherits `UsdGeomMesh` or `UsdGeomBasisCurves` — 3D structure, active sites, binding pockets |
| `BioMolecule` IsA schema | Small molecules, metabolites — SMILES, InChI, 3D conformer |
| `BioTissue` IsA schema | Collection of BioCell prims with spatial constraints, architecture type |
| `BioMetabolicAPI` Applied schema | Namespaced properties: `bio:metabolic:atp_concentration`, `bio:metabolic:oxygen_level`, `bio:metabolic:glucose_uptake`, `bio:metabolic:lactate_production` |
| `BioInteractionAPI` Applied schema | Relationship targets to other prims: `bio:interaction:binds_to`, `bio:interaction:signal_target`, `bio:interaction:binding_affinity` |
| `BioGeneExpressionAPI` Applied schema | `bio:gene:expression_level`, `bio:gene:methylation_state`, `bio:gene:mutation_flags` |
| `BioSemantics` | `UsdSemantics` mapping to Bioschemas.org types — Gene, Protein, ChemicalSubstance, MedicalCondition, Taxon |
| Schema files | `.usda` definitions generated via `usdGenSchema` or hand-authored for codeless registration |

**Deliverable:** Complete biological entity library with management UI, and `.usda` schema files defining the Bio-USD standard

---

## PHASE 6 — OMNIVERSE CONNECTOR & CONVERTERS
*Bridge existing models into the Bio-USD world*

| Task | Detail |
|------|--------|
| Cognisom-to-USD exporter | Convert simulation state to USD scene: each cell becomes a `BioCell` prim, fields become `UsdVol`, metabolic state applied as `BioMetabolicAPI` |
| SBML-to-USD converter | Parse SBML XML, instantiate `BioMolecule` + `BioMetabolicAPI` prims with reaction network as relationships. Uses OpenUSD Exchange SDK 2.0 |
| SBOL-to-USD converter | Parse SBOL RDF, instantiate `BioProtein` + `BioGeneExpressionAPI` prims with genetic circuit topology |
| CellML-to-USD converter | Map CellML mathematical models to USD physics properties |
| Kinetics-to-Physics engine | Map biochemical rate equations to USD physics triggers: Michaelis-Menten `v = V_max * [S] / (K_m + [S])` maps to motor drive force, Hill function maps to spring constant modulation |
| Bidirectional sync | USD scene changes (user manipulation in Omniverse) update Cognisom simulation state, re-export |
| Omniverse Kit extension | Custom Omniverse extension that loads Bio-USD scenes, renders biological entities with appropriate materials/shaders |

**Deliverable:** Round-trip between Cognisom simulation and NVIDIA Omniverse via Bio-USD

---

## PHASE 7 — AI AGENTS FOR BIO-USD MAINTENANCE
*Autonomous agents that keep the biological library and schemas current*

| Agent | Role | How It Works |
|-------|------|-------------|
| **Literature Scout** | Discover new biological entities and relationships | Extends existing subscription system (22 sources). Monitors PubMed, bioRxiv, Nature, Cell daily. Uses LLM to extract new genes, proteins, interactions, mutations from abstracts. Flags items for curation. |
| **Schema Evolution Agent** | Propose Bio-USD schema updates when new entity types emerge | Monitors entity catalog for patterns — if >N entities of a new category appear (e.g., "organoid"), proposes a new IsA schema. Generates `.usda` draft for human review. |
| **Ontology Sync Agent** | Keep entity metadata aligned with canonical ontologies | Periodically checks Gene Ontology, ChEBI, UBERON, Cell Ontology for updates. Propagates ID changes, new terms, deprecated terms into the entity database. |
| **Validation Agent** | Verify Bio-USD attribute values against experimental data | Cross-references entity properties (binding affinities, expression levels, metabolic rates) against latest publications. Flags stale or contradicted values. |
| **Parameter Fitting Agent** | Auto-calibrate simulation parameters from new data | When new experimental datasets are published (e.g., new scRNA-seq study), downloads data, runs parameter estimation, proposes updated rate constants. |
| **Relationship Discovery Agent** | Infer missing entity relationships | Uses protein-protein interaction databases (STRING, BioGRID) + NIM embeddings (ESM2 cosine similarity) to predict missing edges in the entity graph. |

**Architecture:** Each agent is a scheduled task (cron or Celery beat) that runs autonomously, produces a "change proposal" (JSON diff), and queues it for human review in the Library Management UI. No unsupervised changes to production data.

**Deliverable:** 6 autonomous AI agents maintaining the biological knowledge base, with human-in-the-loop approval

---

## PHASE 8 — OMNIVERSE REAL-TIME SIMULATION
*Full physics-based simulation in NVIDIA's platform*

| Task | Detail |
|------|--------|
| Isaac Sim integration | Run Cognisom simulation inside Isaac Sim — cells as deformable bodies, tissue as soft-body physics |
| `UsdSkel` for cells | Cellular deformations — mitosis animation, membrane blebbing, pseudopod extension |
| `UsdPhysics` for dynamics | Rigid body (cell nuclei), deformable body (cell membrane), fluid dynamics (blood flow in vasculature) |
| Multiscale hierarchy | Molecular prims nested inside Cell prims nested inside Tissue prims nested inside Organ prims — USD composition arcs handle LOD |
| Lab-in-a-loop | Physical robot (via Isaac Sim) observes real experiment, feeds data to digital twin, digital twin predicts next state, robot validates |
| Real-time interaction | Researcher manipulates cells/drugs/conditions in VR/AR, simulation responds in real time |

**Deliverable:** Interactive, physics-based biological simulation running in NVIDIA Omniverse

---

## PHASE 9 — AOUSD STANDARDIZATION & COMMUNITY
*Formalize Bio-USD as an industry standard*

| Step | Detail |
|------|--------|
| Join AOUSD | Apply as member, attend working group meetings |
| Form Life Sciences Interest Group | Recruit pharma (Roche, Novartis), academic (Stanford, MIT), tech (NVIDIA) partners |
| Publish codeless schemas | Release `.usda` files on GitHub, register at runtime in Omniverse |
| Build reference implementation | Open-source Python package `biousd` with converters, validators, example scenes |
| Seek ratification | Submit to AOUSD Steering Committee as Core Specification 1.x extension |
| Community adoption | Workshops at ISMB, Bio-IT World, GTC. Tutorials, documentation, example datasets |

**Deliverable:** Bio-USD ratified as AOUSD extension, adopted by 10+ institutions

---

## PHASE 10 — CLINICAL SCALE & PERSONALIZED TWINS
*The ultimate goal*

| Task | Detail |
|------|--------|
| Patient-specific models | Import patient genomics (WGS/WES), imaging (MRI/CT), and pathology into personalized digital twin |
| Multi-organ modeling | Prostate to bone to liver to lung metastatic cascade |
| 1M+ cell simulation | Distributed multi-GPU with ML surrogates for hot paths |
| Virtual clinical trials | Simulate treatment protocols in silico before real trials |
| Clinical decision support | Real-time treatment recommendations based on patient's digital twin |
| FDA/regulatory pathway | Digital twin as FDA-qualified tool for drug development |

**Deliverable:** Clinically validated personalized digital twins for prostate cancer treatment planning

---

## DEPENDENCY GRAPH

```
Phase 1 (Deploy)
    +---> Phase 2 (Validate)
            +---> Phase 3 (GPU)
            |       +---> Phase 8 (Omniverse RT)
            +---> Phase 4 (3D Viz)
                    +---> Phase 5 (Bio-USD + Library)
                            +---> Phase 6 (Connectors)
                            |       +---> Phase 8 (Omniverse RT)
                            +---> Phase 7 (AI Agents)
                            +---> Phase 9 (AOUSD)
                                    +---> Phase 10 (Clinical)
```

Phases 3 and 4 can run in parallel. Phases 6, 7, and 9 can run in parallel after Phase 5.

---

## BIO-USD PROPOSAL CONCEPT CROSS-REFERENCE

Every concept from the Bio-USD proposal is accounted for:

| Proposal Concept | Where in Roadmap |
|-----------------|-----------------|
| IsA Schemas (BioUnit, BioCell) | Phase 5C |
| Applied API Schemas (BioMetabolicAPI, BioInteractionAPI) | Phase 5C |
| BioSemantics / Bioschemas.org | Phase 5A (FAIR metadata) + Phase 5C |
| SBML-to-USD converter | Phase 6 |
| SBOL-to-USD converter | Phase 6 |
| OpenUSD Exchange SDK 2.0 | Phase 6 |
| Codeless schemas / usdGenSchema | Phase 5C + Phase 9 |
| AOUSD Life Sciences Interest Group | Phase 9 |
| AOUSD Core Specification ratification | Phase 9 |
| Kinetics-to-Physics mapping | Phase 6 |
| UsdSkel for cellular deformations | Phase 8 |
| UsdPhysics for dynamics | Phase 8 |
| Multiscale hierarchy (molecular to organ) | Phase 8 |
| Isaac Sim 5.0 integration | Phase 8 |
| Lab-in-a-loop | Phase 8 |
| FAIR interoperability | Phase 5A |
| 70% cost reduction (pharma) | Phase 10 |
| Personalized digital twins | Phase 10 |
| Physical AI training | Phase 8 |
| Complete biological library | Phase 5A (entity catalog) |
| Library management UI | Phase 5B |
| AI agents for maintenance | Phase 7 (6 specialized agents) |

---

## RESUME POINT

When restarting, the immediate next action is:
1. Ensure Docker Desktop is running (no other Claude instances using it)
2. Run: `docker build -f Dockerfile.prod -t cognisom:latest .`
3. If build succeeds, test locally: `docker-compose up`
4. Then deploy to AWS: `bash deploy/deploy.sh`

All deployment infrastructure files are built and ready in:
- `Dockerfile.prod` — production image
- `entrypoint.sh` — starts gunicorn + streamlit
- `docker-compose.yml` — local testing
- `deploy/deploy.sh` — AWS ECS Fargate deployment
- `deploy/task-definition.json` — ECS task definition
- `deploy/apprunner.yaml` — App Runner config (alternative)
- `deploy/nginx.conf` — reverse proxy

---

*Cognisom: From cells to organs, from simulation to cure.*
*eyentelligence inc. | February 2026*
