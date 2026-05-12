# Resume Point - Cognisom Pre-Filing Patent Sprint

**Last session:** 2026-05-12
**Branch:** main
**Latest commit:** dc442da (sprint 2b: wire CellGenomeView into MolecularModule)
**Tests passing:** 122 (patent-evidence suite)

## How to resume

When you start the next session, **cd into this directory first** so Claude's
session state is keyed to cognisom (not wabah):

```bash
cd /Users/davidwalker/CascadeProjects/cognisom
```

Then say something like: "resume cognisom patent sprint - check RESUME.md".

## Where we are

The pre-filing audit on the cognisom molecular layer is half-done.
Three of the four planned upgrades from `UPGRADES_SPEC.md` are complete and
landed; the fourth (closed-loop neoantigen presentation) is the next big
patent claim to build.

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

### Remaining work

Two quick fixes flagged at the Sprint 2b checkpoint, then Upgrade 2.

**Quick fix #1 (~30 min, ~5 lines)** - Derive driver-mutation set in
`modules/molecular_module.py:introduce_mutation` from
`Gene.ONCOGENIC_SUBSTITUTIONS.keys()` instead of hardcoding
`{"G12D","G12V","G13D","V600E","R175H","R248W"}`. The hardcoded set
duplicates knowledge with the table; adding a new entry to one would
require remembering the other.

**Quick fix #2 (~30 min)** - Write a new patent-evidence demo at
`examples/patent_evidence/sprint2_module_demo.py` that drives
SimulationEngine + MolecularModule with ~20 cells, induces multi-
generation transformation, and prints reference-identity + delta counts.
The existing `cancer_transmission_demo.py` uses raw Gene/Exosome and
does NOT exercise the MolecularModule refactor end-to-end.

**Upgrade 2 - closed-loop neoantigen presentation** (2-3 sessions).
This is the strongest USC 101 patent claim: concrete medical output
(predicted neoantigens + immunotherapy response trajectories). Per
UPGRADES_SPEC.md Section 2:

1. NEW `engine/py/molecular/peptidome.py` - protein -> peptide pool
   (sliding window 8-11mers around mutation site, simple proteasomal
   cleavage rules; NetChop integration deferred)
2. NEW `engine/py/immune/mhc_loading.py` - score peptides vs HLA
   alleles (use existing PWM scorer at
   `cognisom/genomics/neoantigen_predictor.py:116-235`)
3. NEW `engine/py/immune/tcr_repertoire.py` - stochastic TCR-pMHC
   affinity (16-dim feature vectors; TCRdist3 deferred)
4. NEW `engine/py/immune/tcell_kill.py` - kill probability from
   affinity x MHC-I level x costimulation
5. Refactor `modules/cellular_module.py` to populate
   `mhc1_displayed_peptides` per cell
6. Refactor `modules/immune_module.py` to use TCR-pMHC matching instead
   of the current threshold heuristic at `gpu/spatial_ops.py:200-240`
7. End-to-end test asserting event trace
   MUTATION_OCCURRED -> PEPTIDE_GENERATED -> PEPTIDE_PRESENTED ->
   CELL_KILLED_BY_TCELL in causal order

### Remaining after Upgrade 2

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
- `modules/molecular_module.py` - refactored to use views (Sprint 2b)
- `tests/test_*.py` - 122 patent-evidence tests

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
