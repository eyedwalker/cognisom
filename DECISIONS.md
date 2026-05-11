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
