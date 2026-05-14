# Cognisom — Inventorship & Design-Decision Log

**Purpose.** Under USPTO guidance on AI-assisted inventions (February 2024), a
human is the named inventor only if they "significantly contributed" to the
conception of each claimed invention. Tooling that codes at human direction
is allowed; tooling that originates inventive concepts is not.

This file is the contemporaneous record of who decided what and why during
the pre-filing implementation sprints. The inventor (David Walker) records
every architectural decision, alternatives considered, and the reason for
the choice. AI tools (Claude) act as implementer at the inventor's direction
and may draft text, but the conception of each decision is the inventor's
and is signed below.

Convention: one heading per decision, dated, with a "Conceived by:" line.
Keep it terse. The point is provenance, not prose.

---

## 2026-05-11 — Decision: Marmur-Doty for sequences > 14 bases; Wallace below

**Conceived by:** David Walker.

**Context.** Self-test of `nucleic_acids.py` printed Tm = 386 C for a 136-base
KRAS fragment. Cause: Wallace rule was applied unconditionally; valid only for
oligos <= 14 bases.

**Alternatives considered.**
- Clamp Tm output to [0, 100] C. Rejected: hides bug rather than fixing it.
- Implement full nearest-neighbor model (Sugimoto/Allawi). Rejected for v1:
  too much scope; Marmur-Doty is the standard simple long-sequence
  approximation and is sufficient for the simulator's needs.
- Keep Wallace and ignore. Rejected: a USPTO examiner who runs the disclosed
  code and sees Tm=386 C will reject the application for non-enablement.

**Decision.** Branch on sequence length at the historical Wallace cutoff
(<= 14 bases). Use Marmur-Doty above.

**Patent relevance.** Defensive fix; does not affect any claim. Ensures the
disclosed code prints physical values when an examiner exercises it.

---

## 2026-05-11 — Decision: Three-upgrade pre-filing sprint plan

**Conceived by:** David Walker (after consultation with patent advisor).

**Context.** The current codebase supports Inventions A (sequence-grounded
mutation->phenotype), B (exosome horizontal transformation), and C (hybrid
SSA/ODE with hysteresis-banded repartition), but each is narrower than ideal
for patent strength. Three identified upgrades would substantially strengthen
the filing:
- Upgrade 1: Reference-genome + per-cell-delta memory architecture.
- Upgrade 2: Closed-loop neoantigen->tissue coupling (Invention F).
- Upgrade 3: Zero-shot mutation-effect biophysics replacing the hardcoded
  oncogenic-mutation table.

**Decision.** Execute the three upgrades. Build out the implementation until
all holes are patched, then file once. See UPGRADES_SPEC.md for the
implementation plan.

**Patent relevance.** Defines the entire filing strategy.

---

## 2026-05-11 - Decision: File once after all holes patched (not provisional-now)

**Conceived by:** David Walker.

**Context.** UPGRADES_SPEC.md originally recommended a two-stage filing
strategy: provisional now on narrow scope, non-provisional in week 14 with
strengthened claims. The advisor's spec leaned heavily on locking priority
date early.

**Alternatives considered.**
- Provisional-now + non-provisional later. Rejected: filing narrow claims
  invites prosecution history that may be used against broader future
  claims; cheaper to file once strongly.
- File a second provisional after each upgrade. Rejected: complexity, cost.

**Decision.** Defer all filing until all three upgrades pass their tests in
tests/test_patent_evidence.py. File a single strong non-provisional then.

**Risk accepted.** Priority date for Inventions A, B, C does not start
running until the eventual filing date. If PhysiCell, BioDynaMo, or another
competitor publishes a sequence-aware tissue simulator before our filing,
that publication becomes prior art against our application. Mitigation:
re-run prior-art search at each sprint completion; if a near-collision is
detected, file immediately on what is then patentable.

**Patent relevance.** Replaces the filing strategy from the prior entry.

---

## 2026-05-11 - Finding: Pre-existing hardcoded mutation table was biologically incorrect

**Conceived by:** David Walker (issue surfaced during Sprint 1 classifier wiring).

**Context.** While integrating the rule-based MutationEffectClassifier
(Sprint 1 / Upgrade 3 Stage A) into Gene.introduce_oncogenic_mutation, the
classifier flagged that every entry in the existing hardcoded oncogenic-
mutation table at engine/py/molecular/nucleic_acids.py:415-428 had off-by-one
position errors. Examples:

  - KRAS G12D was encoded as (position 35, G->A). Position 35 of the KRAS
    reference is 'T' (third base of GGT codon 12). A G->A at position 35
    actually fails the base check; the resulting DNA mutation is recorded
    but the codon change is GGT->GGA (silent, Gly->Gly), not the intended
    GGT->GAT (Gly->Asp).
  - All 6 entries (KRAS G12D/G12V/G13D, BRAF V600E, TP53 R175H/R248W) had
    the same pattern: position off by 1 (one too high). Likely root cause:
    original author used 1-indexed positions without converting to Python's
    0-indexed convention.

The pre-existing `is_oncogene = True` flag was being set externally on
introduce_oncogenic_mutation, regardless of what codon edit actually
occurred. This masked the bug for downstream demo behavior.

**Decision.** Fix the table to correct positions (subtract 1 from each).
Validate via the classifier: introduce_oncogenic_mutation now asserts that
the resulting amino-acid change matches the named mutation. Future
corruption of the table will be caught at runtime.

**Patent relevance.** Strengthens the §112 enablement story: the disclosed
simulation does what it claims (G12D actually produces a Gly->Asp at codon
12). Pre-fix, an examiner running the demo and inspecting the resulting
sequence would have seen silent mutations labelled "oncogenic" — a credibility
hit. Post-fix the simulation matches the biology.

---

## 2026-05-11 - Finding: Demo's KRAS reference sequence was biologically broken (FIXED)

**Conceived by:** David Walker (surfaced during Sprint 1; fixed same day).

**Context.** The KRAS reference used in
modules/molecular_module.py:97-100 and engine/py/molecular/nucleic_acids.py
self-test started: ATG-GAC-TGA-... Codon 3 was TGA, which is a stop codon.
The demo's translate() correctly halted at codon 3, producing "MD" (2 AA)
instead of a full KRAS protein. Real KRAS CDS begins ATG-ACT-GAA-... (the
demo had an erroneous extra G at position 3, shifting the reading frame).

**Decision.** Replaced the synthetic KRAS reference in both files with
the real NM_004985.5 CDS prefix (51 codons, 153 bases, covering codons
1-51 including the G12/G13 hotspot). Added import-time sanity-check
assertions that codon 12 is GGT and codon 13 is GGC. Added a new test
module tests/test_gene_library_sequences.py with 7 regression tests:

  - test_kras_codon_12_is_GGT
  - test_kras_codon_13_is_GGC
  - test_kras_no_premature_stop_before_codon_15
  - test_kras_translates_to_real_protein_prefix (MTEYKLVVVG)
  - test_kras_g12d_via_named_mutation_does_not_warn
  - test_kras_g12v_via_named_mutation_does_not_warn
  - test_kras_g13d_via_named_mutation_does_not_warn

After the fix, the nucleic_acids __main__ demo:
  - prints Tm = 79.1 C (physically reasonable)
  - prints "Protein length: 51 amino acids" (was 2)
  - prints "Sequence: MTEYKLVVVGADGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVEDAFY"
    showing the G12D substitution visible at codon 12 (the D in ...VVG[A][D]GVG...)
  - prints "Classified: missense (G12D); BLOSUM62 score: -1; Impact score: 0.51"

The cancer_transmission_demo also runs end-to-end with the new KRAS reference:
4/4 normal cells transformed via oncogenic exosomes (same behavior as before
but now built on biologically correct KRAS).

**TP53 and BRAF deferred** [updated entry below].

**Patent relevance.** Strengthens enablement: the disclosed simulator
now produces biologically sensible output when exercised. An examiner
running the self-test will see a real 51-AA KRAS protein with the G12D
substitution clearly visible, classifier-validated as missense with
correct BLOSUM62 score and impact estimate.

---

## 2026-05-11 - Decision: Sprint 2 / Upgrade 1 - Reference + Sparse Delta architecture

**Conceived by:** David Walker.

**Context.** UPGRADES_SPEC.md Upgrade 1 calls for a memory architecture
that scales per-cell memory with mutation count, not with genome size.
This is the patent-claim anchor under USPTO 35 USC 101 (Alice/Mayo):
a specific technical improvement to computer functioning that defeats
abstract-idea rejection.

**Decision.** Implemented as two new modules:

  engine/py/molecular/reference_genome.py
    - ReferenceGenome (canonical, immutable after freeze())
    - GeneMetadata (per-gene shared annotations)
    - SubstitutionDelta (frozen dataclass with validation)
    - build_default_reference_genome() builds KRAS/TP53/BRAF from
      reference_cds.py

  engine/py/molecular/sequence_view.py
    - CellGenomeView with base_at(), codon_at(), iter_codons(),
      materialize(), add_substitution(), fork()
    - O(1) per-base lookup via dict-indexed delta map
    - Chronological delta log preserved for daughter inheritance and
      provenance
    - materialize() is the escape hatch for callers that need a full
      sequence string (e.g., for an ML model)

Both classes are read-mostly. Mutations are tracked via append-only
delta records.

**Patent-relevant invariants tested:**

  - fork() shares ReferenceGenome by Python object identity
    (not by copy)
  - Sequence of substitutions applied through view's API gives identical
    answers to the deep-copy + apply approach (tested across 5 random
    seeds with 50 substitutions each)
  - Independence after fork: parent mutations after fork do not affect
    daughter, and vice versa
  - Multi-generation fork chains carry cumulative ancestor deltas

**Benchmark numbers (psutil RSS):**

  - 10k cells with 3 mutations each: 8.2 MB total (budget was 50 MB)
  - At 100-gene synthetic genome (~300 KB total ref): naive 61.0 MB vs
    view 2.6 MB = 23x more efficient
  - 83x larger genome under fixed mutation count produces 0x extra
    memory (asymptotic O(deltas) confirmed)
  - Forking 10k times allocates no per-fork reference copies

**Not yet integrated.** ModuleularModule still uses the legacy
per-cell-Gene-copy approach. Sprint 2b will wire MolecularModule to use
CellGenomeView. The new classes are exercised only by their own tests
in this sprint.

**Total tests added:** 46 (18 reference_genome + 24 sequence_view + 4
memory benchmarks). Total project tests now: 104 (was 58).

**Patent relevance.** This sprint produces the load-bearing claim. The
disclosure now demonstrates a specific computational improvement
verifiable at the bench (psutil RSS measurement) over the prior-art
approach (per-cell deep copy).

---

## 2026-05-12 - Decision: Sprint 2b - Wire CellGenomeView into MolecularModule

**Conceived by:** David Walker.

**Context.** Sprint 2 built the ReferenceGenome + CellGenomeView
architecture as a standalone library. Sprint 2b integrates it into the
live simulator so cell-level state actually flows through the new
data structures.

**Decision.** Refactored modules/molecular_module.py:

  - Removed self.cell_genes (per-cell deep Gene copies)
  - Added self.reference_genome: ReferenceGenome (shared across cells)
  - Added self.cell_views: Dict[int, CellGenomeView]
  - Added self.cell_oncogene_flags: Dict[int, Set[str]]
  - Added self.classifier: MutationEffectClassifier

  Refactored methods:
    - add_cell: creates CellGenomeView pointing at the shared reference
    - on_cell_divided: invokes view.fork() (key memory-architecture op)
    - on_cell_died: cleans up all per-cell dicts
    - introduce_mutation: writes a delta to the view AND classifies via
      the classifier; sets oncogene flag based on classifier impact
      score (not external belief)
    - update: transcribes by materializing the cell's view-derived
      sequence at transcription time
    - create_exosome: packages mRNA materialized from view (so the
      mRNA carries the cell's specific deltas)
    - get_state: reports view population

  Kept self.genes (legacy Gene library) for backward-compat with the
  existing tests in test_gene_library_sequences.py. It is populated at
  initialize() from the same reference_cds source.

**Patent relevance.** This sprint converts the simulator from "demonstrates
the architecture in tests" to "uses the architecture as its actual
runtime data substrate". An examiner running the cancer_transmission
demo now sees the architecture in operation: 5 cells share one
ReferenceGenome; mutations are recorded as sparse deltas; cell division
produces forks that share the reference by Python identity.

**Tests added (18):** tests/test_molecular_module_views.py
  - Reference genome present after initialize and shared by identity
  - cell_genes attribute is gone
  - cells use views
  - introduce_mutation writes deltas, classifies, flags oncogene, isolates
    to target cell, rejects unknown/missing inputs
  - on_cell_divided forks; parent/daughter diverge after; share ref
  - on_cell_died cleans up
  - oncogenic exosomes carry mutated mRNA (with G12D substitution in the
    sequence) and the mutation log records the mutation name
  - normal exosomes from non-mutated cells carry reference mRNA
  - get_state reports view population

**Total tests now:** 122 (was 104). Cancer transmission demo passes
end-to-end with 3 of 4 cells transformed (stochastic, within prior range).

**Pre-filing audit progress:**

  - [DONE] Sprint 0: Tm fix, repo hygiene, snapshot tag
  - [DONE] Sprint 0.5: patent artifact consolidation
  - [DONE] Sprint 1: rule-based mutation effect classifier (Upgrade 3 Stage A)
  - [DONE] Sprint 1b: KRAS reference fix
  - [DONE] Sprint 1c: TP53 and BRAF reference CDSes
  - [DONE] Sprint 2: reference-genome + per-cell-delta memory architecture
  - [DONE] Sprint 2b: integrate views into MolecularModule
  - [REMAINING] Upgrade 2: closed-loop neoantigen presentation
  - [REMAINING] Upgrade 3 Stage B/C: domain-aware + ESM-2 stability
  - [VERIFY-BEFORE-FILING] replace synthetic TP53/BRAF with NCBI authentic
  - [DEFERRED] full clean-up of duplicate cognisom/modules/ tree

---

## 2026-05-11 - Decision: TP53 and BRAF synthetic CDSes with real hotspots

**Conceived by:** David Walker (Sprint 1c, following the "patch all holes"
filing strategy).

**Context.** TP53 R175H/R248W and BRAF V600E reside at codons 175, 248,
and 600 of their respective canonical CDSes (NM_000546.6 length 1182 bases,
NM_004333.6 length 2301 bases). The demo placeholders were 134 bases
each, far short of reaching these positions. With those placeholders,
calling Gene.introduce_oncogenic_mutation('R175H') etc. raises ValueError
at the DNA layer (position out of range).

**Alternatives considered.**

  - Embed authentic NCBI CDSes from memory. Rejected for engineering risk:
    reproducing ~1200 and ~2300 bases of DNA character-for-character from
    memory is unreliable. A single error anywhere could introduce a frame
    issue or wrong codon at a hotspot, undetected.
  - Synthetic CDSes with all-alanine filler. Accepted with caveat: easy
    to write, easy to verify, gets hotspot positions exactly right. The
    resulting protein outside the hotspot region is biologically
    meaningless (poly-Ala), but the patent claims do not depend on
    biological authenticity of off-hotspot codons -- they depend on
    position-correct hotspots and the classifier's ability to evaluate
    arbitrary substitutions.
  - Reverse-translate from authentic protein. Rejected for v1: requires
    embedding ~393 + ~766 amino acids of protein sequence accurately,
    which is the same memory-risk problem one step up. The user can
    upgrade to this approach (or to authentic NCBI CDSes via Biopython)
    before patent filing.

**Decision.** Constructed synthetic TP53_CDS (393 codons + stop, 1182 b)
and BRAF_CDS (766 codons + stop, 2301 b) using:

  - first 30 codons = best-recall canonical N-terminal sequence
  - codons 31..n with single filler GCG (Ala), except hotspots
  - canonical hotspot codons inserted at biologically correct positions:
    TP53 codon 175 = CGC (R), codon 248 = CGG (R); BRAF codon 600 = GTG (V)
  - TAA stop codon appended

Implementation in engine/py/molecular/data/reference_cds.py with module-
load-time assertions that hotspot codons are correct, lengths match, and
no premature stops exist. KRAS uses authentic NM_004985.5 prefix (51
codons) as before.

modules/molecular_module.py refactored to import from reference_cds.py.

After this change, all six entries in ONCOGENIC_SUBSTITUTIONS produce
correct AA changes with zero warnings:

  KRAS G12D: missense, BLOSUM -1, impact 0.51
  KRAS G12V: missense, BLOSUM -3, impact 0.74
  KRAS G13D: missense, BLOSUM -1, impact 0.51
  TP53 R175H: missense, BLOSUM 0, impact 0.40
  TP53 R248W: missense, BLOSUM -3, impact 0.74
  BRAF V600E: missense, BLOSUM -2, impact 0.62

Tests: 58 total pass (was 49). Cancer_transmission_demo still runs
end-to-end with the new gene library.

**Pre-filing checklist item added.** TP53_CDS and BRAF_CDS should be
replaced with authentic NM_000546.6 and NM_004333.6 CDSes via Biopython
or an NCBI download before patent filing. The reference_cds.py module
has a VERIFY-BEFORE-FILING note. The classifier-driven tests will catch
any hotspot regression during that replacement.

**Patent relevance.** Closes the molecular-layer enablement holes
identified during Sprint 1. All named oncogenic mutations now produce
biologically correct AA changes verifiable by the classifier. The
disclosed simulation can be exercised end-to-end on the canonical KRAS
G12D / G12V / G13D, TP53 R175H / R248W, and BRAF V600E mutations.

---

## 2026-05-11 - Decision: All cognisom patent artifacts live in cognisom repo

**Conceived by:** David Walker.

**Context.** Earlier scoping work produced two artifacts outside the cognisom
directory: ~/.claude/plans/do-a-much-deeper-swift-lerdorf.md (Claude Code's
session-scoped plan file, associated with a Claude Code session started in
the wabah working directory) and /tmp/cognisom_patent_disclosure.md (the
markdown source for the .docx). Mixing patent work across project
directories creates inventorship trail confusion and risks accidental
inclusion of unrelated wabah/wubba work.

**Decision.** All cognisom patent artifacts live under
/Users/davidwalker/CascadeProjects/cognisom/. Specifically:
- docs/patent/SCOPING.md - the technical scoping document.
- docs/patent/DISCLOSURE_SOURCE.md - markdown source of the .docx.
- COGNISOM_PATENT_DISCLOSURE.docx - attorney-facing Word version.
- UPGRADES_SPEC.md - implementation spec.
- DECISIONS.md - this file.
- tests/test_tm_calculation.py and similar patent-evidence tests.

Going forward: no patent-related artifact is to be written to .claude/plans
or /tmp. Working directory for cognisom Claude Code sessions must be
/Users/davidwalker/CascadeProjects/cognisom/ so that Claude's session state
is associated with the cognisom project, not wabah.

**Patent relevance.** Defensive / inventorship-trail hygiene.

---

## 2026-05-12 - Decision: Sprint 2c - Derive driver-mutation set from ONCOGENIC_SUBSTITUTIONS; add Sprint 2 patent-evidence demo

**Conceived by:** David Walker.

**Context.** Two carry-over items from the Sprint 2b checkpoint:
(a) modules/molecular_module.py:introduce_mutation carried a hardcoded
``{"G12D","G12V","G13D","V600E","R175H","R248W"}`` set used to flag
"oncogenic driver" status. Duplicating the curated knowledge in
``Gene.ONCOGENIC_SUBSTITUTIONS`` invites table drift on every future
hotspot addition. (b) The Sprint 2 reference-genome architecture had
no human-readable end-to-end demo separate from
``cancer_transmission_demo.py``, which bypasses MolecularModule
entirely and operates on raw Gene/Exosome objects.

**Alternatives considered.**
- Leave the hardcoded set; document the duplication. Rejected:
  inventorship-trail and §112 enablement-credibility risk grow with
  every future hotspot.
- Add the demo at ``examples/molecular/`` next to the legacy demo.
  Rejected: muddies the patent-evidence story; a dedicated
  ``examples/patent_evidence/`` directory makes the
  reduction-to-practice evidence explicit.

**Decision.** Replaced the hardcoded set with
``Gene.ONCOGENIC_SUBSTITUTIONS[gene_name]`` membership test (curated
table is the single source of truth -- any future hotspot added to the
table automatically inherits driver status). New patent-evidence demo
at ``examples/patent_evidence/sprint2_module_demo.py`` drives
SimulationEngine + MolecularModule end-to-end with 20 cells; prints
the two Upgrade-1 invariants explicitly:
  (A) reference-identity shared across all cell views (no copy)
  (B) per-cell sparse delta counts (O(deltas), not O(genome))
Includes a Generation-2 fork via CELL_DIVIDED and a Generation-3
divergence via second-hit mutation, with parent/daughter delta-count
assertions.

**Patent relevance.** Invention A (Sequence-Grounded Mutation->
Phenotype) §112 enablement — the demo is the human-readable
reduction-to-practice artifact attorneys can run during a disclosure
walkthrough. Commit ``610dc4b``.

**Tests after this commit:** 160 (was 122).

---

## 2026-05-12 - Decision: Upgrade 2 part 1 - Closed-loop neoantigen pipeline primitives

**Conceived by:** David Walker.

**Context.** UPGRADES_SPEC.md Upgrade 2 specifies a closed-loop chain:
MUTATION_OCCURRED -> PEPTIDE_GENERATED -> PEPTIDE_PRESENTED ->
CELL_KILLED_BY_TCELL. Part 1 builds the four pure new-module files
without touching the existing cellular/immune integration; part 2
wires them into modules/{cellular,immune}_module.py.

**Alternatives considered.**
- One big commit landing primitives + integration + tests + demo
  together. Rejected: too large to review; each primitive is itself
  patent-evidence and benefits from independent tests.
- Per-primitive commits (4 commits + integration + tests + demo).
  Rejected: too many commits with cross-references; the right grain
  is "all primitives land together" / "integration lands together".

**Decision.** Four new modules + 3 new event types + 51 unit tests
landed in a single commit:

  engine/py/molecular/peptidome.py
    Protein -> peptide pool with sliding 8-11mer windows. Two modes:
    full sliding window (self-peptide pool) and mutation-anchored
    window (neoantigen pool, primary patent-evidence path). Simple
    proteasomal cleavage scoring favors C-terminal hydrophobic /
    aromatic residues, penalizes basic residues at P1; NetChop
    integration deferred per spec line 503.

  engine/py/immune/mhc_loading.py
    MHC-I scoring via the same PWM scorer at
    ``cognisom/genomics/neoantigen_predictor.py:325`` that the
    clinical neoantigen path uses, so the simulation and clinical
    paths share a single source of truth. MHCflurry is picked up
    automatically when installed (already in this env); PWM fallback
    when not.

  engine/py/immune/tcr_repertoire.py
    Stochastic TCR repertoire with deterministic 16-dim feature
    embeddings derived via SHA-256 from CDR3 sequences and pMHC
    keys. Affinity is sigmoid of cosine similarity (scale=4 keeps
    the recognition_threshold knob meaningful). TCRdist3 deferred
    per spec line 506.

  engine/py/immune/tcell_kill.py
    Per-encounter kill probability = Hill function of
    affinity * mhc_level * costimulation, with optional checkpoint-
    block rescue when costim is weak. Default Hill threshold 0.3,
    slope 4. All inputs clamped to [0, 1].

  core/event_bus.py
    New event types: PEPTIDE_GENERATED, PEPTIDE_PRESENTED,
    CELL_KILLED_BY_TCELL — ordered causally to support the end-to-end
    trace assertion in part 2.

**Patent relevance.** Upgrade 2 §101 anchor (closed-loop neoantigen
presentation). The primitive files are the concrete computational
machinery the patent specification references. Commit ``c131efc``.

**Tests added (51):** test_peptidome.py (15), test_mhc_loading.py (8),
test_tcr_repertoire.py (13), test_tcell_kill.py (15). Total suite: 211.

---

## 2026-05-12 - Decision: Upgrade 2 part 2 - Closed-loop integration + end-to-end trace + patent-evidence demo

**Conceived by:** David Walker.

**Context.** Part 1 built the primitives in isolation. Part 2 wires
them into the live simulator so a mutation introduced through
``molecular.introduce_mutation`` actually traverses the full chain
through to CELL_KILLED_BY_TCELL.

**Alternatives considered.**
- Push closed-loop integration through a separate event-bus
  middleware layer (e.g., a NeoantigenPipeline module). Rejected:
  introduces a fourth module just for orchestration when the
  cellular and immune modules naturally carry the per-cell state
  and immune-encounter state respectively.
- Cellular module owns the peptide / MHC scoring AND the TCR
  matching. Rejected: violates the "molecular owns sequence,
  cellular owns per-cell state, immune owns immune cells"
  responsibility split established in Sprint 2.

**Decision.** Three integration changes:

  modules/molecular_module.py
    + ``get_reference_protein(gene_name) -> str`` — translates the
      reference DNA for a gene into the canonical WT protein. Used by
      the cellular handler to seed neoantigen peptide generation.

  modules/cellular_module.py
    CellState gains ``mhc1_displayed_peptides: List[MHCPresentation]``.
    Module gains ``hla_alleles``, ``max_displayed_per_mutation``, and a
    lazily-constructed ``MHCLoader``. New ``on_mutation_occurred``
    handler subscribes to MUTATION_OCCURRED, parses the
    ``aa_change`` field, generates 8-11mer neoantigen peptides via
    the peptidome, emits PEPTIDE_GENERATED, scores against the
    patient HLA panel, drops non-binders, stores top-k by IC50 on
    the cell, and emits PEPTIDE_PRESENTED per kept presentation.

  modules/immune_module.py
    ImmuneCell gains ``active_tcr_match``. Module builds a
    TCRRepertoire on initialize (size, seed, recognition threshold
    config-driven). T-cell recognition uses TCR-pMHC matching
    against ``target.mhc1_displayed_peptides`` rather than the
    legacy ``mhc1_expression > 0.2`` threshold. Kill probability
    uses ``tcell_kill.kill_outcome(affinity, mhc, costim, block)``.
    Successful T-cell kills additionally emit CELL_KILLED_BY_TCELL
    with full provenance (peptide, mutation, source_gene,
    hla_allele, tcr_id, affinity).

End-to-end test ``tests/test_closed_loop_neoantigen.py`` (3 tests)
asserts:
  - causal event trace appears in MUTATION_OCCURRED ->
    PEPTIDE_GENERATED -> PEPTIDE_PRESENTED -> CELL_KILLED_BY_TCELL
    order in the engine event log
  - the kill event provenance traces back to the originating
    mutation (gene, label, peptide, HLA, TCR clone, affinity)
  - negative control: no displayed peptides -> no T-cell kill, no
    T-cell activation

Plus the consolidated patent-evidence demo at
``examples/patent_evidence/upgrade2_closed_loop_demo.py`` drives the
chain against three canonical drivers (KRAS G12D, BRAF V600E, TP53
R248W) + a wild-type control. Demonstrates three claims in one trace:
  Claim A (sensitivity): KRAS G12D produces a presented peptide
    KLVVVGADGV on HLA-A*02:01, recognized by a TCR clone, killed.
  Claim B (HLA restriction): BRAF V600E and TP53 R248W generate 38
    peptides each; ZERO bind the demo HLA panel; trace correctly
    stops at PEPTIDE_GENERATED with no kill. This is a USC 101
    differentiator vs uniform "predict immune response" prior art --
    BRAF V600E is clinically hard for ICB and is treated with
    vemurafenib (small-molecule kinase inhibition) instead.
  Claim C (negative control): WT cancer cell -> no mutation -> no
    peptides -> no T-cell kill even with the T cell colocated.

**Patent relevance.** Upgrade 2 §101 anchor (closed-loop chain) +
USC 101 specificity differentiator (HLA restriction). Commits
``9f47de7`` + ``1e561a9``.

**Tests after this commit:** 234. Closed-loop trace asserted
end-to-end in 3 dedicated tests. Demo prints attorney-readable
output.

---

## 2026-05-13 - Decision: Upgrade 3 Stage B - UniProt domain-aware impact multiplier

**Conceived by:** David Walker.

**Context.** Stage A (Sprint 1, ``7d63412``) gave the classifier
BLOSUM62-derived missense impact scores. Empirically a conservative
substitution can still be catastrophic when it lands in a critical
functional region (KRAS G12 in the P-loop, TP53 R175 in the DNA-
binding domain, BRAF V600 in the activation loop), and a radical
substitution can be tolerated when it lands in a disordered linker.
Rule-based-only impact systematically misranks these cases.

**Alternatives considered.**
- Hardcode "is this a known hotspot" -> 0.85 impact override.
  Rejected: the override would not generalize to new mutations and
  bakes the answer rather than computing it.
- Use AlphaMissense or another foundation-model annotation. Rejected:
  Stage C does that; Stage B's job is the cheap-and-general domain
  multiplier from UniProt annotations.
- Per-mutation curated multipliers (look up COSMIC). Rejected:
  COSMIC has spotty coverage of less-studied genes and would re-
  create the prior-art rule-based lookup pattern this whole pipeline
  is trying to avoid.

**Decision.** New module
``engine/py/molecular/protein_domains.py``:
  - ``ProteinDomain`` dataclass (gene, name, role, codon range,
    UniProt id, source attribution).
  - Role -> multiplier table: ``critical`` 4.0x, ``functional`` 2.5x,
    ``regulatory`` 1.5x, ``structural`` 1.5x. Spec requires "2-5x";
    table sits inside that range.
  - 13 curated domains across KRAS / TP53 / BRAF for v1.
    KRAS: P-loop (critical), Switch I/II (functional), HVR
      (structural).
    TP53: TAD1/2 + PRR + Reg (regulatory), DBD (critical),
      tetramerization (functional).
    BRAF: RBD + kinase (functional), CRD (regulatory),
      activation loop (critical).
  - ``domain_at_codon(gene, codon_1based) -> Optional[ProteinDomain]``
    resolves overlap (e.g., BRAF activation loop sits inside the
    kinase domain) by picking the highest-multiplier match.

MutationEffect dataclass gains three Stage B fields:
``domain_name``, ``domain_role``, ``domain_multiplier``. The classifier
gains an optional ``gene_name`` parameter; when supplied, the missense
path multiplies the BLOSUM impact by the resolved domain multiplier
and clamps at the missense ceiling (0.85). Stage A is bit-for-bit
preserved when ``gene_name`` is omitted -- existing tests pass
unchanged.

molecular_module.introduce_mutation forwards ``gene_name`` to the
classifier so the per-cell delta and per-cell oncogene flags pick up
Stage B immediately.

Effect on canonical hotspots:
  KRAS G12D  Stage A 0.51 -> Stage B 0.85 (P-loop x4.0)
  KRAS G12V  Stage A 0.74 -> Stage B 0.85 (P-loop x4.0)
  BRAF V600E Stage A 0.63 -> Stage B 0.85 (Activation loop x4.0)
  TP53 R175H Stage A 0.40 -> Stage B 0.85 (DBD x4.0)
  TP53 R248W Stage A 0.74 -> Stage B 0.85 (DBD x4.0)

**Patent relevance.** Upgrade 3 Stage B (resolves the known
Stage-A weakness where a conservative BLOSUM substitution in a
critical functional region was scored as "mild"). Defensible §103
differentiator -- prior-art rule-based classifiers either ignore
domain context or hardcode known hotspots; Stage B is the smallest
principled bridge from BLOSUM to functional annotation that
generalizes to new genes. Commit ``2e17718``.

**Tests added (20):** tests/test_protein_domains.py. Total suite: 234.

---

## 2026-05-13 - Decision: Replace synthetic TP53/BRAF CDSes with authentic NCBI sequences (pre-filing gap #1)

**Conceived by:** David Walker.

**Context.** Sprint 1c (commit ``7c7dea2``) added synthetic CDSes for
TP53 and BRAF -- the real first-30-codon protein prefix + GCG (Ala)
filler chosen so the canonical hotspots landed at the correct codon
positions. Tagged VERIFY-BEFORE-FILING. Filing with synthetic
off-hotspot codons would be defensible (the patent claims do not
depend on biological authenticity of off-hotspot codons) but is
visibly weaker §112 enablement evidence than authentic CDSes.

**Alternatives considered.**
- Ship Sprint 1c synthetic CDSes; defend in filing prosecution if
  an examiner asks. Rejected: cheap to fix, removes a visible
  enablement gap.
- Use AlphaFold-derived sequences. Rejected: AlphaFold predicts
  structures, not authoritative protein sequences -- UniProt /
  RefSeq is the right authority.
- Fetch CDSes from NCBI at module-import time. Rejected: introduces
  a network dependency on import, which would break offline /
  air-gapped test runs and patent-evidence demos.

**Decision.** Fetched authentic CDSes via Biopython Entrez and
inlined into engine/py/molecular/reference_cds.py:
  KRAS  NM_004985.5  567 bp / 188 aa (extended from prior 153-bp
                                        51-codon prefix)
  TP53  NM_000546.6  1182 bp / 393 aa
  BRAF  NM_004333.6  2301 bp / 766 aa
Hotspot codons and protein lengths match the prior synthetic
placeholders bit-for-bit, so all downstream tests pass unchanged.
Import-time invariants strengthened: start codon, hotspot codons,
stop codon position, no-premature-stop, exact CDS length all
asserted against the authentic NCBI sequences.

Side benefit: KRAS now exposes its full GTPase fold rather than
just the N-terminal P-loop window, so the Switch I / Switch II /
HVR domains annotated in protein_domains.py are now reachable for
classification rather than dangling.

**Patent relevance.** §112 enablement — closes the
VERIFY-BEFORE-FILING note from Sprint 1c. The patent disclosure
can now reference authentic RefSeq accessions for every reference
sequence the simulator uses. Commit ``fe49568``.

**Tests after this commit:** 234 (unchanged — bit-for-bit
compatible with the prior synthetic placeholders).

---

## 2026-05-13 - Decision: VCF round-trip end-to-end test (pre-filing gap #2)

**Conceived by:** David Walker.

**Context.** The closed-loop test at
tests/test_closed_loop_neoantigen.py drove the pipeline via direct
``molecular.introduce_mutation(cell_id, gene, mutation_label)``
calls -- it skipped the VCF ingest stage. A patent reviewer could
reasonably ask "the cognisom pipeline claims VCF -> simulation, but
what test demonstrates that?" — and the honest answer would have been
"the VCF parser is unit-tested in isolation, and the simulation
pipeline is integration-tested separately, but no test bridges
them". Closes the "we never read DNA in any test" critique.

**Alternatives considered.**
- Add VCF parsing to the existing closed-loop test. Rejected:
  conflates two distinct test surfaces and obscures provenance
  assertion targets.
- Add a separate VCF-driven demo. Rejected: a test is stronger
  evidence than a demo for patent purposes.

**Decision.** New ``tests/test_vcf_round_trip.py`` (5 tests):
  - VCFParser populates Variant.gene and Variant.protein_change for
    every supported-gene row (sanity).
  - Extraction surfaces all four canonical hotspots that exist in
    both the VCF and the curated ONCOGENIC_SUBSTITUTIONS table:
      KRAS G12V (chr12:25398284)
      TP53 R175H (chr17:7577539)
      TP53 R248W (chr17:7578406)
      BRAF V600E (chr7:140753336)
  - Intronic / intergenic VCF rows are silently dropped.
  - End-to-end: at least one CELL_KILLED_BY_TCELL event fires whose
    ``source_gene`` + ``mutation`` fields match the originating VCF
    row's gene + protein_change. Provenance flows
      VCF chr:pos REF>ALT  ->  Variant.protein_change
        ->  molecular.introduce_mutation
        ->  MutationEffect (with Stage B domain multiplier)
        ->  peptidome -> MHC loading -> TCR recognition
        ->  CELL_KILLED_BY_TCELL.source_gene + .mutation
  - Negative control: a VCF whose rows all map to genes outside
    SUPPORTED_GENES produces zero peptide events.

**Patent relevance.** §112 enablement (VCF -> kill end-to-end is now
asserted) + USC 101 specificity (negative control prevents the
pipeline from fabricating neoantigens for unsupported genes).
Commit ``3bdd733``.

**Tests after this commit:** 239 (was 234).

---

## 2026-05-13 - Decision: Expand domain panel to 21 cancer drivers (pre-filing gap #4)

**Conceived by:** David Walker.

**Context.** Stage B (commit ``2e17718``) annotated three genes
(KRAS, TP53, BRAF). For the patent breadth claim, three genes is
thin -- production cancer panels cover 50-500 drivers. The
classifier's gene_name-driven path is gene-agnostic; only the
domain TABLE was small.

**Alternatives considered.**
- Inline ~500 driver genes immediately. Rejected: heavy data load
  for unclear marginal patent value; ~20 is the clinically
  meaningful core (TCGA Pan-Cancer Atlas top drivers, Bailey et al.
  Cell 2018).
- Auto-fetch UniProt features at import. Rejected: same offline
  reason as the NCBI-CDS decision.
- Curate by hand from UniProt. Accepted -- ensures every entry has
  a verifiable source citation.

**Decision.** Added 18 new genes to
engine/py/molecular/protein_domains.py:
  PIK3CA  ABD, RBD, C2, helical (critical), kinase (critical)
  PTEN    phosphatase (critical), C2, C-terminal regulatory
  EGFR    L1+L2, TM, kinase domain, activation loop (critical)
  NRAS    P-loop (critical), Switch I, Switch II (critical, Q61)
  IDH1    substrate-binding (critical, R132), NADP+ binding
  IDH2    substrate-binding (critical, R140 / R172)
  APC     ARM repeats, beta-catenin / MCR region (critical)
  RB1     pocket A (critical), pocket B (critical)
  BRCA1   RING (critical), BRCT repeats (critical)
  BRCA2   BRC repeats (critical), DNA-binding (critical)
  ATM     HEAT, FAT, kinase (critical)
  STK11   kinase (critical)
  CDKN2A  ankyrin repeats (critical)
  FGFR3   Ig-II, Ig-III (critical, S249 / G370), kinase (critical)
  MYC     transactivation, bHLH-LZ (critical)
  CDH1    EC1 (critical), cytoplasmic catenin-binding (critical)
  AR      NTD, DBD (critical), LBD (critical, T877A / L702H)
  NF1     CSRD, GAP-related (critical)

Every gene gets at least one critical-role domain so the 4x
multiplier fires for its canonical hotspots. UniProt accessions are
cited per domain in the ``source`` field for audit traceability.

**Patent relevance.** Breadth of claim. Now 21 genes / 57 domains
total; this is the panel an attorney would reasonably claim "the
classifier supports as in-spec" at filing time. Commit ``0b15a87``.

**Tests added (15):** parametrized hotspot lookups across the new
genes + a panel-size invariant assertion (>= 20 genes; every gene
has at least one critical-role domain). Total: 254.

---

## 2026-05-13 - Decision: Upgrade 3 Stage C - ESM-2 zero-shot stability composition (pre-filing gap #3)

**Conceived by:** David Walker.

**Context.** Spec UPGRADES_SPEC.md line 479 names Stage C ESM-2 as
the strongest §103 differentiator vs PhysiCell + COSMIC-lookup
competitors. ESM-2 is Meta's protein language model; for a missense
substitution it provides
  delta_log_likelihood = log P(mut | context) - log P(WT | context)
as a zero-shot proxy for change in free energy -- negative means
destabilizing, positive means well-tolerated. ESM-2 requires neither
a structure (unlike FoldX) nor a labelled training set (unlike
supervised stability predictors), so the same scorer works on every
cancer driver in the 21-gene panel without per-gene fine-tuning.

**Alternatives considered.**
- ESM-2 replaces the Stage A+B impact score entirely. Rejected:
  losing BLOSUM and domain-multiplier signal would weaken
  discrimination on cases ESM-2 is uncertain about; composition is
  more robust and more defensible as patent claim.
- Use AlphaFold + FoldX for ΔΔG. Rejected: structure-dependent,
  expensive, and not available for many tumor-specific isoforms.
- Use a smaller language model (e.g., ESM-1b). Rejected: ESM-2 is
  the documented stronger model and runs on CPU at ~2s/inference
  for the 150M variant.

**Decision.** New module ``engine/py/molecular/esm_stability.py``:
  ESMStabilityScorer protocol — duck-typed; anything with
    ``score_substitution(seq, pos, wt, mut) -> ESMStabilityResult``.
  StubESMStabilityScorer — constant-score scorer for tests +
    graceful-degradation paths.
  RealESMStabilityScorer — wraps HuggingFace transformers + ESM-2
    150M (``facebook/esm2_t30_150M_UR50D``, ~600MB, ~1-2s CPU
    inference). Lazy-loaded; raises ImportError at construction if
    transformers/torch are not installed.
  delta_ll_to_stability_modifier — sigmoid mapping with [-50, 50]
    clamp for numerical stability.
  apply_stability_to_impact — composes the modifier with the
    Stage B impact: destabilizing modifier pushes impact toward the
    missense ceiling (0.85); well-tolerated modifier pulls it toward 0.

MutationEffect gains three Stage C fields:
``esm_delta_log_likelihood``, ``esm_stability_modifier``,
``esm_model_name``. The classifier gains optional
``protein_sequence`` and ``esm_scorer`` parameters. When both are
supplied AND the substitution is missense, the classifier composes
ESM-2's stability score with the Stage B impact. Synonymous /
nonsense / start-loss paths skip Stage C. If the scorer raises, the
classifier falls back to Stage B impact and emits no ESM fields --
never crashes.

molecular_module gains ``set_esm_scorer(scorer)``. When set, every
introduce_mutation also passes the translated WT protein + the
scorer to the classifier. Default scorer is None, so the simulation
is identical to pre-Stage-C behaviour unless the caller wires Stage C
in explicitly.

Effect on KRAS G12D (P-loop, Stage B at 0.85 ceiling):
  Stage B + neutral (dLL=0)        : impact 0.85 (unchanged)
  Stage B + destabilizing (dLL=-3) : impact 0.85 (already at ceiling)
  Stage B + well-tolerated (dLL=+3): impact 0.08 (ESM overrides)
The third case is the patent-evidence point: biophysics corrects
rule-based-only overconfidence.

**Patent relevance.** Upgrade 3 Stage C — the strongest USC 103
differentiator on the spec. Patent claim is the *composition* of
three orthogonal signals (BLOSUM evolutionary, domain functional-
region, ESM-2 biophysical) -- no prior-art rule-based classifier
uses more than one or two. Commit ``9489543``.

**Tests added (22):** tests/test_esm_stability.py + 1 opt-in
real-ESM smoke test gated on ENABLE_ESM_SMOKE=1. Total: 271 + 1
skipped.

---

## 2026-05-13 - Decision: Upgrade 4 - TME 4-type classification as simulation output (Teng et al. 2015)

**Conceived by:** David Walker (motivated by V. Chen lecture,
CU Anschutz 2026-05-03, slides 53-54).

**Context.** Oncology classifies tumors into four TME types based on
TIL density and PD-L1 expression (Teng et al., Cancer Res 2015):
  Type I    TIL+ PDL1+  adaptive immune resistance       HIGH ICB
  Type II   TIL- PDL1-  immunological ignorance          MINIMAL
  Type III  TIL- PDL1+  intrinsic PD-L1 induction        LOW
  Type IV   TIL+ PDL1-  tolerance / non-PD-1 suppression MODERATE
This is the standard clinical readout used to triage patients for
checkpoint blockade. Prior to this upgrade, cognisom's simulation
produced raw events (TIL counts, PD-L1 expression per cell) but no
classification mapping back to the clinical framework.

**Alternatives considered.**
- Output raw TIL / PDL1 metrics only; let the caller classify.
  Rejected: misses the patent-evidence opportunity to claim the
  classifier as a first-class simulation output.
- Output a single response score (e.g., 0..1). Rejected: oncologists
  read the 4-type scheme; matching their language is what makes the
  output clinically actionable.
- Re-run the classifier continuously. Rejected: classification is
  expensive (O(cells * immune cells)); on-demand is the right grain.

**Decision.** New ``engine/py/immune/tme_classifier.py``:
  TMEType enum (I/II/III/IV) + TMEClassification dataclass carrying
  TIL ratio, PD-L1 positive fraction, type, predicted ICB response
  ("high" / "moderate" / "low" / "minimal"), and a human-readable
  mechanism description.
  ``classify_tme(cancer_cells, immune_cells, **kwargs)`` takes any
  cell-like iterables and returns the classification. Default
  thresholds: TIL ratio >= 0.5, PD-L1 fraction >= 0.25, per-cell
  PD-L1 >= 0.5, proximity <= 20 um. All overridable.

modules/cellular_module.py
  CellState gains ``pdl1_expression: float`` in [0, 1].
  _INTRINSIC_PDL1_INDUCERS table covers PIK3CA / MYC / EGFR
  (Spranger 2015, Casey 2016, Akbay 2013). On MUTATION_OCCURRED
  with one of these oncogenes, intrinsic PD-L1 is bumped on the
  cell -- this is the Type III mechanism.

modules/immune_module.py
  On T-cell activation (TCR-pMHC recognized), the target cancer
  cell's pdl1_expression is bumped by 0.5 (clamped at 1.0) -- the
  adaptive PD-L1 mechanism (Type I) driven by IFN-gamma.
  New ImmuneModule.classify_tme() entrypoint reads cancer cells
  from the linked cellular module + immune cells from this module's
  population, calls the classifier, and emits TME_CLASSIFIED with
  the resolved type + clinical readout fields.

core/event_bus.py
  New TME_CLASSIFIED event type.

**Patent relevance.** Upgrade 4 — patent-claim: given a patient
VCF, the cognisom pipeline emits a TME 4-type classification that
predicts ICB response category. The novelty vs clinical state of the
art (IHC of tumor biopsy) is that cognisom computes this
*prospectively from the genome*, where IHC requires a biopsy. No
known cancer simulator outputs the Teng 4-type classification.
Commit ``7d3aa58``.

**Tests added (14):** tests/test_tme_classifier.py. Total: 290 + 1
skipped.

---

## 2026-05-13 - Decision: Upgrade 5 - T-cell exhaustion 2-state model with ICB-rescue gating (Dolina et al. 2021)

**Conceived by:** David Walker (motivated by V. Chen lecture, slide 52).

**Context.** Dolina 2021 / Wherry / Bengsch establish that
checkpoint blockade does NOT rescue PD-1-hi exhausted CD8 T cells.
ICB works by *expanding the PD-1-lo precursor pool* in peripheral
lymphoid organs. Rule-based immune modules in prior art (PhysiCell
etc.) apply checkpoint_block as a generic kill-probability
multiplier — implying ICB rescues every dysfunctional T cell, which
would be a clinically dangerous prediction.

**Alternatives considered.**
- 3-state model (naive / effector / exhausted). Rejected: the
  patent claim is the rescuable/unrescuable binary distinction;
  3-state adds complexity without proportional claim value.
- Per-encounter exhaustion (Bayesian decay). Rejected: empirically
  exhaustion is epigenetically enforced and effectively irreversible
  (Bengsch 2018 chromatin work) — a one-way state transition is the
  right model.
- Time-based threshold (exhaust after X simulation hours).
  Rejected: in vivo exhaustion is antigen-encounter-driven, not
  time-driven. Encounter count is the right axis.

**Decision.** engine/py/immune/tcr_repertoire.py:
  ExhaustionState enum (PRECURSOR / EXHAUSTED).
  TCRMatch gains ``exhaustion_state`` and ``encounter_count`` fields.
  TCRRepertoire tracks per-clone encounter count + state. New
  register_engagement(tcr_id) increments the counter and one-way
  transitions to EXHAUSTED when crossing exhaustion_threshold
  (default 5, configurable). Helpers: exhaustion_state, encounters,
  precursor_count, exhausted_count.

engine/py/immune/tcell_kill.py:
  kill_probability / kill_outcome gain ``is_exhausted`` +
  ``exhaustion_multiplier`` params:
    * is_exhausted=True suppresses the checkpoint_block rescue term
      ENTIRELY (precursor-only path). Matches the lecture's central
      observation that ICB cannot un-exhaust PD-1-hi cells.
    * exhausted clones additionally have their kill probability
      scaled by exhaustion_multiplier (default 0.1) — residual
      cytotoxicity at order-of-magnitude lower rates per Dolina.
  Backwards-compat: existing callers that pass neither flag get
  pre-Upgrade-5 behaviour bit-for-bit.

modules/immune_module.py:
  Two registration points: once on the initial recognition
  transition, and once per simulation step while the T cell is in
  continuous contact with its target (chronic-antigen-exposure
  model). When the threshold is crossed, emit TCELL_EXHAUSTED with
  full provenance (tcr_id, cdr3, encounter_count, target_id,
  peptide, mutation).
  _target_kill_probability forwards is_exhausted from the active
  TCR match into kill_outcome.

core/event_bus.py: New TCELL_EXHAUSTED event type.

**Patent relevance.** Upgrade 5 — patent-claim: a 2-state TCR
repertoire that gates checkpoint-blockade rescue on the precursor
state. **Strong §103 differentiator** — corrects the rule-based-only
error documented in clinical literature; no other cancer simulator
does this. Commit ``ea8caef``.

**Tests added (14):** tests/test_exhaustion.py. Total: 304 + 1
skipped.

---

## 2026-05-13 - Decision: Upgrade 6 - ECM / desmoplasia barrier with ecm_excluded TME sub-classification

**Conceived by:** David Walker (motivated by V. Chen lecture,
slides 34-35 + 49-50).

**Context.** Only 1% of PDAC patients respond to ICB despite
carrying targetable mutations, because the desmoplastic stroma
physically excludes CD8 T cells from reaching tumor cells (PMC7956114
Cdh11 knockout result; Herzog 2023 NSCLC). Prior to this upgrade,
the simulator could not distinguish "cold because no antigens" from
"cold because TILs are walled out" -- a clinically critical split.
The first takes a vaccine; the second takes anti-fibrotic + ICB combo.

**Alternatives considered.**
- Grid-based ECM concentration field with diffusion. Rejected:
  heavy new solver; the patent claim is the *gating* of immune
  function on stromal density, not the spatial diffusion model.
- Per-cell local ECM density. Accepted: cell-grained representation
  composes with the existing cellular module without requiring a
  new ECM field solver.
- Single global ECM scalar per simulation. Rejected: loses spatial
  heterogeneity, which is the whole point of the exclusion claim.

**Decision.** New ``engine/py/spatial/ecm_barrier.py``:
  ecm_density_at(position, cancer_cells, sample_radius_um) averages
    cancer cells' local_ecm_density within a sample sphere -> [0, 1].
  motility_attenuation(base_speed, ecm, blocking_factor=0.9)
    retained_fraction = max(MIN_RETAINED, 1 - ecm * blocking).
  detection_attenuation(base_radius, ecm, blocking_factor=0.8)
    same shape; high ECM compresses sensing radius.

modules/cellular_module.py:
  CellState gains local_ecm_density: float in [0, 1].
  Config knobs: ecm_deposition_rate, ecm_degradation_rate,
  anti_fibrotic_active, cancer_baseline_ecm. Cancer cells deposit
  ECM continuously while alive; anti_fibrotic_active flips
  deposition to degradation. New cellular method ecm_density_at
  delegates to the spatial helper for immune module lookups.

modules/immune_module.py:
  T-cell patrol speed, attack-mode speed, and effective detection
  radius are now ECM-attenuated. High-ECM region compresses
  detection radius to near zero, reproducing the clinically observed
  "TIL absence despite antigen presence" pattern.

engine/py/immune/tme_classifier.py:
  TMEClassification gains mean_ecm_density and ecm_excluded.
  ecm_excluded fires only when (TIL-neg) AND (mean ECM >=
  DEFAULT_ECM_EXCLUDED_THRESHOLD=0.4) AND (any displayed
  neoantigens). When ecm_excluded fires, the description string is
  overridden with the anti-fibrotic recommendation:
    classic Type II ("ignored", no antigens) -> vaccine
    ECM-excluded Type II                     -> anti-fibrotic + ICB

**Patent relevance.** Upgrade 6 — **the most defensible §103
differentiator from the lecture-review pass**. Prior-art rule-based
immune modules (PhysiCell, BioDynaMo) treat infiltration as a
function of chemokine gradients alone — none model the physical ECM
barrier that explains why 99% of PDAC patients fail ICB. Cognisom
now predicts the *right reason* a tumor is cold, which determines
the right therapy. Commit ``0f37690``.

**Tests added (18):** tests/test_ecm_barrier.py. Total: 322 + 1
skipped.

---

## 2026-05-14 - Decision: Upgrade 7 - Sympathetic / beta2-adrenergic immunosuppression with beta-blocker rescue

**Conceived by:** David Walker (motivated by V. Chen lecture,
slides 30-33).

**Context.** Chronic stress -> sympathetic nervous system activation
-> norepinephrine at nerve terminals -> beta2AR engagement on T
cells, dendritic cells, macrophages, B cells -> suppression of
T-cell activation, survival, proliferation, cytokine release. The
mechanism is documented (Wu 2018; Farooq 2023; Armaiz-Pena 2015) and
clinically observable (retrospective cohorts show beta-blocker users
have better ICB outcomes; Kokolus 2018; Oh 2021). No other known
cancer simulator models the neuroimmune axis.

**Alternatives considered.**
- Tie stress to cortisol level + immune dysfunction via a separate
  endocrine module. Rejected: scope creep; the patent claim is the
  beta2AR-mediated T-cell kill attenuation, not the full HPA axis.
- Model stress as a per-cell modifier on immune cells. Rejected:
  in vivo the drive is systemic (catecholamines in circulation);
  per-cell makes the math harder without changing the prediction.
- Multiple-axis suppression (beta2AR + glucocorticoid + alpha-AR).
  Rejected: beta2AR is the documented dominant pathway; minimal
  model is more defensible as patent claim.

**Decision.** New ``engine/py/immune/sympathetic.py``:
  sympathetic_attenuation(stress, blocker, max_suppression=0.7)
    Returns the multiplier in [1 - max_suppression, 1.0]:
       effective = stress * (1 - blocker)
       retained  = 1 - effective * max_suppression
    All inputs clamped to [0, 1]; NaN treated as zero;
    max_suppression of 0 cleanly disables the axis.
  sympathetic_state() returns SympatheticState dataclass
    (stress_level, beta_blocker, effective_signal,
    t_cell_function_retained, max_suppression) for emit + audit.

modules/immune_module.py:
  Config knobs: stress_level (default 0), beta_blocker (default 0),
  sympathetic_max_suppression (default 0.7).
  _target_kill_probability multiplies the final kill probability by
  sympathetic_attenuation -- composes cleanly with the existing
  affinity / mhc / costim / exhaustion gates.
  Runtime API: set_stress(stress_level, beta_blocker) for
  treatment-arm experiments (e.g., add propranolol mid-run).
  get_sympathetic_state() returns the current snapshot.

**Patent relevance.** Upgrade 7 — **highest-novelty differentiator
in the whole patent surface**. No cancer simulator in the published
literature models neuroimmune effects at all. The retrospective
clinical evidence base for beta-blocker / ICB outcomes is well
established, but no prior computational model gates immunotherapy
response on a patient stress proxy. Commit ``6b7a994``.

**Tests added (15):** tests/test_sympathetic.py. Total: 337 + 1
skipped.

---

## 2026-05-14 - Decision: Upgrade 8 - INDEL + fusion neoantigen peptide generation

**Conceived by:** David Walker (motivated by V. Chen lecture,
slide 44).

**Context.** Prior to this upgrade, the peptidome only generated
missense neoantigens, ignoring frameshifts and gene fusions.
Clinically this systematically under-predicts response in the
cancers ICB works best for:
  * MMR-deficient / MSI-high tumors (Lynch syndrome, MSI-high CRC,
    endometrial) carry hundreds of frameshifts per tumor -- these
    generate long stretches of completely foreign C-terminal AA
    sequence, which are the highest-quality neoantigens (no self-
    tolerance issues).
  * Translocation-driven cancers (CML BCR-ABL1, ALK-fusion NSCLC,
    EWSR1 sarcomas) generate fusion neoantigens absent from either
    parent proteome.

**Alternatives considered.**
- Require the caller to compute the novel C-terminal sequence and
  pass it in. Accepted -- keeps the peptidome framework-free; the
  cellular module / VCF parser layer composes the novel sequence
  from the actual indel info when that integration lands later.
- Reuse the missense generator with a "long synthetic mutation".
  Rejected: the WT-comparison semantics differ (frameshift WT is
  the original protein's same-position window, possibly past WT
  end; fusion WT is the left partner extended) -- different paths
  are clearer.

**Decision.** Two new generators in
``engine/py/molecular/peptidome.py``:

  generate_frameshift_peptides(wt_protein, fs_position_1based,
      novel_c_terminal, source_gene, mutation_label, lengths,
      min_cleavage_score):
    Emits peptides whose windows include at least one novel
    residue. Windows upstream of the FS site have a WT prefix +
    novel suffix; windows at or after the FS site are 100% novel.
    wild_type_sequence is the WT protein's same-position window,
    X-padded past WT length. anchor_position_in_peptide marks the
    first novel residue (0 for pure-novel windows).

  generate_fusion_peptides(left_protein, right_protein,
      left_breakpoint_1based, right_breakpoint_1based, source_gene,
      mutation_label, lengths, min_cleavage_score):
    Builds the chimeric protein as left[:left_break-1] +
    right[right_break-1:] and emits only the windows that span the
    junction. wild_type_sequence uses left_protein extended with
    X-padding past its length. anchor_position_in_peptide marks the
    first right-partner residue within the peptide.

Peptide gains ``mutation_type`` field with documented values:
"missense" / "frameshift" / "fusion" / "self". __post_init__
validates the field. Existing missense / self generators stamp the
correct value (backwards-compat for callers that constructed
Peptide without the field).

**Patent relevance.** Upgrade 8 — completes the §101 patent claim
surface to ALL three clinically important mutation classes (SNVs,
INDELs, fusions). Patent disclosure can now claim complete
mutation-class coverage per the lecture's slide-44 taxonomy. Commit
``bacdfd8``.

**Tests added (25):** tests/test_peptidome_indel_fusion.py. Total:
362 + 1 skipped.

---

## 2026-05-14 - Decision: Patent Pipeline dashboard page (production exposure of all 8 upgrades)

**Conceived by:** David Walker.

**Context.** Upgrades 3 Stage B/C, 4, 5, 6, 7, 8 lived as backend
Python modules + pytest tests + two CLI demo scripts. Zero UI
exposure -- a patent attorney walking through the .docx would have
no live demo to point at for any of the new claims. Asked Claude to
build "the cheapest way to make all the new IP visible".

**Alternatives considered.**
- One page per upgrade (5 new pages). Rejected: the patent claim is
  the *composition*; the attorney needs ONE trace to point at.
- A new CLI demo per upgrade. Rejected: doesn't show the composition
  visually and isn't accessible to non-technical reviewers.
- Wire upgrades into the existing pages (31_neoantigen_vaccine,
  28_digital_twin). Rejected: those pages use the older
  ``cognisom/genomics/`` neoantigen path, not the new closed-loop
  pipeline; conflating the two would confuse the demo.

**Decision.** New ``cognisom/dashboard/_pages/42_patent_pipeline.py``
(originally drafted at ``dashboard/pages/32_patent_pipeline.py`` and
migrated to the production tree with a number bump to avoid
``32_parabricks.py`` collision). Registered in
``cognisom/dashboard/app.py`` under the "Digital Twin" group as
"Patent Pipeline".

Page structure:
  Sidebar: mutation class radio (Upgrade 8), driver selector,
    treatment levers (costim, checkpoint blockade, stress,
    beta-blocker), TME knobs (baseline ECM, anti-fibrotic toggle),
    run button.
  Per-upgrade cards (top to bottom):
    1 Memory architecture: ref bases, deltas, identity invariant
    3 Multi-axis impact: BLOSUM + domain + Stage C ESM-2 stub
    8 Multi-class neoantigens: peptide table per class
    2 Closed-loop trace
    5 Exhaustion: precursor / exhausted clone counts + transitions
    6 ECM: cancer-cell density vs T-cell-position density
    7 Sympathetic: stress, blocker, effective signal, function
      retained
    4 TME 4-type: type + ICB response category + ecm_excluded flag

TME UX fix: simulation runs in two phases (30% / 70%) so that
classify_tme() captures a peak-engagement snapshot before successful
kills empty the population. Avoids the degenerate "everything was
killed -> Type II ignorance" classification.

Deployed to cognisom.com via SSM RunShellScript (SSH key was
rejected; SSM is the safer auditable prod-deploy path going forward).
Build artifact sha256:c17cc8718a72. Cognisom.com HTTP 200 verified
post-swap; engine module imports verified in the container.

**Patent relevance.** Production reduction-to-practice evidence. An
attorney logged into cognisom.com can run the full closed-loop
pipeline across all 8 upgrades in one trace. Commits ``4f8aeda`` +
``412af82``.

---

<!--
Template for future entries:

## YYYY-MM-DD - Decision: <one-line summary>

**Conceived by:** <name>.

**Context.** <Why a decision was needed.>

**Alternatives considered.**
- <Option A>. Accepted / Rejected because <reason>.
- <Option B>. ...

**Decision.** <The choice made.>

**Patent relevance.** <Which claim or invention this affects, or "defensive".>

---
-->
