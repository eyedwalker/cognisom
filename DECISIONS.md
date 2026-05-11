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

**Alternatives considered.**
- File now on narrow scope. Rejected as primary plan; weak Section 101 posture.
- Implement upgrades first, then file. Rejected: priority date risk if a
  competitor publishes a sequence-aware tissue simulator in the meantime.

**Decision.** File a provisional NOW on current narrow scope to lock priority
date. Execute the three upgrades over 12 weeks. File the non-provisional with
strengthened claims in week 14, well inside the 12-month provisional deadline.

**Patent relevance.** Defines the entire filing strategy. See UPGRADES_SPEC.md
for the full implementation plan.

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
