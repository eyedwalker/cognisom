# Resume Point - Cognisom Pre-Filing Patent Sprint

**Last session:** 2026-05-13
**Branch:** main
**Latest commit:** Upgrade 3 Stage C (ESM-2 zero-shot stability composition)
**Tests passing:** 271 (+1 opt-in real-ESM smoke skipped)
**Filing readiness:** all four pre-filing gaps closed; remaining items are housekeeping (duplicate package tree, cleanup of pre-existing test_registry failures).

## How to resume

When you start the next session, **cd into this directory first** so Claude's
session state is keyed to cognisom (not wabah):

```bash
cd /Users/davidwalker/CascadeProjects/cognisom
```

Then say something like: "resume cognisom patent sprint - check RESUME.md".

## Where we are

The pre-filing audit on the cognisom molecular layer is now complete.
All three upgrades from `UPGRADES_SPEC.md` are landed (Upgrade 1 +
Upgrade 2 = USC 101 anchors; Upgrade 3 Stages A/B/C = USC 103
differentiator). All four pre-filing checklist gaps -- authentic
NCBI CDSes, VCF round-trip, expanded driver-gene panel, ESM-2
stability -- are closed.

### Sprints completed

| Sprint | Commit | Patent surface |
|---|---|---|
| 0 - prerequisites (Tm fix, snapshot tag) | 4797fb5 | enablement hygiene |
| 0.5 - patent artifact consolidation | 40bebd0 | inventorship trail |
| 1 - rule-based mutation effect classifier | 7d63412 | Upgrade 3 Stage A |
| 1b - KRAS reference fix | 3a72946 | enablement |
| 1c - TP53 and BRAF CDSes covering hotspots | 7c7dea2 | enablement |
| 1c.1 - off-by-one assertion fix | b88016a | regression-net consistency |
| 2 - reference-genome + per-cell delta architecture | 261aa4d | Upgrade 1 (USC 101 anchor) |
| 2b - wire CellGenomeView into MolecularModule | dc442da | Upgrade 1 integration |
| 2c - drop hardcoded driver set + sprint2 demo | 610dc4b | enablement + Upgrade 1 evidence |
| 2d - Upgrade 2 part 1: peptidome / MHC / TCR / kill | c131efc | Upgrade 2 primitives |
| 2e - Upgrade 2 part 2: closed-loop integration + e2e | 9f47de7 | Upgrade 2 (USC 101 anchor) |
| 2f - Upgrade 2 patent-evidence demo (closed loop) | 1e561a9 | Upgrade 2 visible evidence |
| 3B - Upgrade 3 Stage B: UniProt domain multiplier | 2e17718 | Upgrade 3 Stage B |
| pre-filing #1 - authentic NCBI CDSes (KRAS/TP53/BRAF) | fe49568 | enablement / §112 |
| pre-filing #2 - VCF round-trip end-to-end test | 3bdd733 | end-to-end provenance |
| pre-filing #4 - 21-gene driver panel for Stage B | 0b15a87 | patent claim breadth |
| 3C - Upgrade 3 Stage C: ESM-2 stability composition | 9489543 | Upgrade 3 Stage C (USC 103 differentiator) |

### Closed-loop neoantigen pipeline (Upgrade 2)

Event trace `MUTATION_OCCURRED -> PEPTIDE_GENERATED -> PEPTIDE_PRESENTED ->
CELL_KILLED_BY_TCELL` is asserted end-to-end in
`tests/test_closed_loop_neoantigen.py`. Pipeline modules:

- `engine/py/molecular/peptidome.py` - protein -> 8-11mer peptide pool
  with simple proteasomal cleavage scoring (NetChop deferred)
- `engine/py/immune/mhc_loading.py` - MHC-I scoring via the same PWM
  scorer the neoantigen predictor uses (MHCflurry picked up when
  installed; PWM fallback)
- `engine/py/immune/tcr_repertoire.py` - stochastic TCR repertoire,
  16-dim feature embeddings keyed on CDR3 + pMHC, sigmoid-of-cosine
  affinity (TCRdist3 deferred)
- `engine/py/immune/tcell_kill.py` - Hill-curve kill probability
  from affinity x MHC-I level x costimulation, with optional
  checkpoint-block rescue

Integration:
- `modules/cellular_module.py` subscribes to MUTATION_OCCURRED and
  populates `CellState.mhc1_displayed_peptides`, emitting
  PEPTIDE_GENERATED and PEPTIDE_PRESENTED. HLA panel and max
  displayed-per-mutation are config-driven.
- `modules/immune_module.py` builds a per-simulation `TCRRepertoire`
  on initialize; T cells recognize cancer cells via TCR-pMHC against
  displayed peptides; kill probability uses `tcell_kill.kill_outcome`;
  T-cell kills emit `CELL_KILLED_BY_TCELL` with full provenance.

### Stage C composition (Upgrade 3)

The classifier composes three orthogonal signals into the missense
impact score (a USC 103 differentiator vs prior-art classifiers that
use at most one or two):

  Stage A (BLOSUM62)        -- evolutionary conservation
  Stage B (UniProt domains) -- functional-region proximity (1.5x-4x)
  Stage C (ESM-2 dLL)       -- biophysics-grounded stability

Stage C is wired through `MolecularModule.set_esm_scorer(scorer)`.
The scorer interface is duck-typed; the production path uses
`RealESMStabilityScorer` (HuggingFace transformers + ESM-2 150M),
test fixtures use `StubESMStabilityScorer`. Opt-in real-ESM smoke
test: `ENABLE_ESM_SMOKE=1 pytest tests/test_esm_stability.py`.

### Remaining work (housekeeping only -- not patent-load-bearing)

- Delete or shim the duplicate `cognisom/engine/` and `cognisom/modules/`
  package tree (causes the 9 pre-existing `test_registry` failures).
- Pre-existing `test_ode_solver::test_cell_heterogeneity` failure
  (parameter-noise CV; unrelated to patent surface).

## Key file pointers

- `DECISIONS.md` - inventorship log (USPTO 2024 AI-assisted-invention
  trail). Every architectural decision recorded here.
- `UPGRADES_SPEC.md` - the full 12-week sprint plan
- `docs/patent/SCOPING.md` - technical scoping document
- `docs/patent/DISCLOSURE_SOURCE.md` - markdown source of the attorney docx
- `COGNISOM_PATENT_DISCLOSURE.docx` - attorney-facing Word version
- `engine/py/molecular/reference_genome.py` - shared reference genome (Sprint 2)
- `engine/py/molecular/sequence_view.py` - per-cell sparse delta view (Sprint 2)
- `engine/py/molecular/mutation_effect.py` - classifier (Sprint 1)
- `engine/py/molecular/reference_cds.py` - KRAS/TP53/BRAF reference CDSes
- `modules/molecular_module.py` - refactored to use views (Sprint 2b);
  exposes `get_reference_protein(gene)` for the Upgrade 2 chain
- `modules/cellular_module.py` - subscribes to MUTATION_OCCURRED;
  populates `CellState.mhc1_displayed_peptides`; emits PEPTIDE_GENERATED
  and PEPTIDE_PRESENTED
- `modules/immune_module.py` - builds TCRRepertoire on initialize;
  TCR-pMHC recognition for T cells; emits CELL_KILLED_BY_TCELL with
  provenance
- `engine/py/immune/{mhc_loading,tcr_repertoire,tcell_kill}.py` -
  closed-loop neoantigen primitives (Upgrade 2)
- `engine/py/molecular/peptidome.py` - peptide generation around
  mutation sites (Upgrade 2)
- `engine/py/molecular/protein_domains.py` - 21-gene UniProt domain
  panel for Stage B multiplier
- `engine/py/molecular/esm_stability.py` - ESM-2 zero-shot stability
  scoring (Stage C); Real + Stub scorers
- `examples/patent_evidence/sprint2_module_demo.py` - end-to-end
  evidence for Upgrade 1 (reference-identity + per-cell delta)
- `examples/patent_evidence/upgrade2_closed_loop_demo.py` - end-to-end
  evidence for Upgrade 2 (claims A sensitivity, B HLA restriction,
  C wild-type negative)
- `tests/test_closed_loop_neoantigen.py` - event-trace assertion
- `tests/test_vcf_round_trip.py` - raw VCF -> simulation -> kill
  with provenance (closes the "we never read DNA" critique)
- `tests/test_esm_stability.py` - Stage C unit tests + opt-in
  real-ESM smoke test
- `tests/test_*.py` - 271 patent-evidence + Upgrade 2 unit +
  closed-loop + Stage B/C tests

## Snapshot tags

- `patent-snapshot-pre-upgrades-2026-05-11` - state before any upgrade work

## Filing strategy (per DECISIONS.md)

User chose: **patch all holes before filing**. Single non-provisional
when all upgrades land + pre-filing checklist is clean. Risk accepted on
priority date (no provisional filed first). Re-run prior-art search at
each sprint completion; if PhysiCell / BioDynaMo announces a competing
sequence-aware feature, file immediately on what is then patentable.

## Inventorship note (USPTO Feb 2024 AI-assisted-invention guidance)

David Walker is the sole inventor. AI tools (Claude) act as implementer
at the inventor's direction. Every architectural decision is logged in
`DECISIONS.md` with "Conceived by: David Walker" attribution so the
inventorship file is contemporaneous.
