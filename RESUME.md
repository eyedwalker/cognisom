# Resume Point - Cognisom Pre-Filing Patent Sprint

**Last session:** 2026-05-12 (continued)
**Branch:** main
**Latest commit:** Upgrade 2 part 2 (closed-loop neoantigen integration)
**Tests passing:** 214 (patent-evidence + Upgrade-2 unit + closed-loop end-to-end)

## How to resume

When you start the next session, **cd into this directory first** so Claude's
session state is keyed to cognisom (not wabah):

```bash
cd /Users/davidwalker/CascadeProjects/cognisom
```

Then say something like: "resume cognisom patent sprint - check RESUME.md".

## Where we are

The pre-filing audit on the cognisom molecular layer is more than half done.
Upgrades 1 and 2 from `UPGRADES_SPEC.md` are complete and landed (USC 101
anchors). Upgrade 3 Stage A is in. Stages B and C remain.

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
| 2e - Upgrade 2 part 2: closed-loop integration + e2e | (this session) | Upgrade 2 (USC 101 anchor) |

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

### Remaining work

- **Upgrade 3 Stage B** (domain-aware impact) - annotate genes with
  UniProt domains; mutations in critical regions get 2-5x impact
  multiplier
- **Upgrade 3 Stage C** (ESM-2 protein language model) - zero-shot
  stability prediction for arbitrary mutations; replaces the current
  rule-based-only impact score with biophysics-derived modifiers.
  Optional GPU; ESM-2-150M runs on CPU in ~2s per inference. The
  strongest USC 103 differentiator vs PhysiCell + COSMIC-lookup
  competitors.
- **Verify-before-filing items**:
  - Replace synthetic TP53_CDS and BRAF_CDS with authentic NM_000546.6
    / NM_004333.6 CDSes via Biopython (`engine/py/molecular/reference_cds.py`
    has the VERIFY-BEFORE-FILING note)
  - Delete or shim the duplicate `cognisom/engine/` and `cognisom/modules/`
    package tree (causes the 9 pre-existing test_registry failures)

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
- `examples/patent_evidence/sprint2_module_demo.py` - end-to-end
  evidence for Upgrade 1 (reference-identity + per-cell delta)
- `tests/test_closed_loop_neoantigen.py` - end-to-end event-trace
  assertion for Upgrade 2
- `tests/test_*.py` - 214 patent-evidence + Upgrade 2 unit + closed-loop tests

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
