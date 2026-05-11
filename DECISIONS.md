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
