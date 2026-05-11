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

## 2026-05-11 - Finding: Demo's KRAS reference sequence is biologically broken

**Conceived by:** David Walker (surfaced during Sprint 1).

**Context.** The KRAS reference used in
modules/molecular_module.py:97-100 and engine/py/molecular/nucleic_acids.py
self-test starts: ATG-GAC-TGA-... Codon 3 is TGA, which is a stop codon.
The demo's translate() correctly halts at codon 3, producing "MD" (2 AA)
instead of a full KRAS protein. Real KRAS CDS begins ATG-ACT-GAA-... (the
demo has an erroneous extra G at position 3).

This is unrelated to the prior off-by-one bug in
ONCOGENIC_SUBSTITUTIONS (fixed in the previous DECISIONS.md entry). It is
a separate problem in the seed data.

**Decision (deferred to a later sprint).** Replace the synthetic KRAS
sequence in molecular_module._create_gene_library and the
nucleic_acids __main__ demo with the real KRAS CDS prefix. Same for TP53
and BRAF. After replacement, run the classifier with the named-mutation
warning enabled; expect zero warnings.

**Patent relevance.** Same as previous entry: the disclosed demo must
produce a biologically sensible protein when an examiner runs it.
Defensive / enablement hygiene. Not blocking Sprint 1.

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
