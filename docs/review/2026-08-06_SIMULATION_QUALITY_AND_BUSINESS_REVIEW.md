# Cognisom — Simulation Quality & Business Value Review

**Date:** 2026-08-06
**Scope:** Fidelity of the "cellular twin" simulations + commercial/IP evaluation
**Method:** Direct code reading (no files modified). Claims below cite `file:line`.

---

## TL;DR

Cognisom is a ~246,000-line research prototype that wraps a genuinely good cancer-genomics
front-end and some real numerical solvers around a "digital twin" that, at its core, is a
**hand-tuned rule engine, not a mechanistic simulation** — and whose most load-bearing
biological step (T-cell recognition of neoantigens) is currently **a coin flip keyed by a
hash**. The engineering is a real mixture: some components are legitimate and well-thought-out;
several flagship pieces are overstated, decoupled, or — in two cases — actively fabricate
their own validation. Business-wise there is a fundable *kernel* here, but the current
investor/clinical framing outruns the evidence badly enough to be a liability.

---

## 1. What the "cellular twin" actually is

```
patient VCF → variant annotation → driver / TMB / MSI biomarkers
           → HLA typing → neoantigen prediction
           → DigitalTwinConfig → TreatmentSimulator → response curve + RECIST call
```

The **front half is clinically grounded and is the real asset**: VCF parsing, driver
annotation, TMB/MSI computation, and therapy matching whose rules mirror actual FDA labels
(`cognisom/genomics/patient_profile.py:133`). Neoantigen prediction uses MHCflurry when
installed with an honestly-labeled PWM fallback (`neoantigen_predictor.py:325`). The
molecular variant-effect pipeline behind it is high quality (correct BLOSUM62, real UniProt
domain ranges covering KRAS G12 / BRAF V600 / TP53 hotspots, correct zero-shot ESM-2 ΔLL).

The **back half — the part branded "digital twin" and sold as "predict personalized
treatment response" — is the weak link.**

---

## 2. Simulation-quality problems, ranked

### Severity 1 — The twin's "simulation" is a decoupled heuristic; personalization is partly illusory

`TreatmentSimulator._simulate_tumor_dynamics` (`cognisom/genomics/treatment_simulator.py:431`)
is a difference equation: `growth 0.5%/day − effectiveness × 2.5%/day`, linear ramp, small
Gaussian noise. `effectiveness` is an additive score of biomarker flags with invented
constants (`exhaustion_reversal=0.55`, etc.). Three compounding problems:

- **It never runs the mechanistic engine.** `twin.to_simulation_params()` exists but is only
  ever *displayed* via `st.json(...)` (`_pages/28_digital_twin.py:340,354`). `TreatmentSimulator`
  contains zero references to the cellular/immune/spatial engine. The "twin" and the
  "simulator" are disconnected.
- **Noise is seeded off the drug name only:**
  `np.random.seed(hash(profile.get("name","")) % 2**31)` (`treatment_simulator.py:439`).
  Every patient gets an *identical* trajectory for a given drug, and it mutates global NumPy
  RNG state as a side effect.
- **No outcome calibration.** Constants aren't fit to any cohort.

**Fix:** either make the predictor actually drive the ABM via `to_simulation_params()` and run
ensembles, or drop the mechanistic framing and *calibrate* against the SU2C mCRPC
treatment-outcome cohort already wired up. Replace the global seed with a per-call
`np.random.Generator`.

### Severity 1 — T-cell recognition is random-number theater

`engine/py/immune/tcr_repertoire.py:134-193`: recognition is computed by SHA-256-hashing the
CDR3 and the peptide/HLA string into **random Gaussian vectors** and taking a sigmoid of their
dot product. Two unrelated random vectors → cosine ≈ 0 → a cognate TCR–neoantigen pair is *no
more likely to be recognized than any random pair*. The advertised loop
`MUTATION → PEPTIDE → PRESENTED → TCR_RECOGNIZED → KILL` is real plumbing around a coin flip.

**Fix:** replace with a real sequence-based model (TCRdist3, which the docstring anticipates),
or disable and label the step until it exists. Highest-priority scientific fix.

### Severity 1 — Two modules fabricate their own validation (integrity, not just quality)

- `eval/simulation_accuracy.py`: "published benchmarks" carry **fake sequential PMIDs**
  (`PMID:12345678`, `23456789`, …, ~lines 305-401), and with no engine supplied (the default)
  the "simulated" value is literally `observed × (1 ± uniform(0.1))` (~line 461) — guaranteeing
  <20% error and a manufactured grade "A".
- The MAD flagship metric **"429-patient SU2C validation, 100% biomarker concordance,
  TMB r=0.987"** is tautological: `_check_biomarker_concordance` returns `True` for any patient
  with no strong biomarker (`cognisom/validation/mad_study.py:547-549`, verified) and only
  compares the recommendation against the same rules that generated it.
  `docs/patent/MAD_INTEGRATION_AUDIT.md` admits no test reproduces it — yet it appears on the
  investor page and in the compliance module.

**Fix:** rebuild `simulation_accuracy.py` with real citations and real simulator output;
replace concordance with retrospective prediction-vs-outcome scoring (harness already exists
in `eval/drug_response_eval.py`).

### Severity 2 — The "GPU adaptive ODE/BDF solver" is overstated and silently wrong

`cognisom/gpu/ode_solver.py`: no adaptivity (fixed `dt`; PI controller absent;
error-estimation kernel compiled but never called), "BDF 1–5" is backward-Euler only,
"Adams-Moulton" just calls BDF, "GPU RK45" is CPU NumPy with GPU round-trips. Worst: the GPU
BDF Newton kernel updates state **only `if (n_species == 2)`** (line 145) — so the advertised
6-species `ar_signaling_pathway` with default `method='bdf'` on GPU is a **no-op that still
returns `success=True`**. The CPU RK4 path (what tests exercise) is correct.

**Fix:** retract the GPU-ODE claims or implement a general LU/Newton solve; add a regression
test on the default BDF path. This also weakens the patent story (§112 enablement).

### Severity 2 — Smaller correctness bugs

- **Diffusion desync:** `engine/py/spatial/grid.py:121-123` — when requested `dt` exceeds the
  stable step it takes one fractional step instead of sub-cycling, so fields advance far
  slower than the sim clock.
- **Dead ribosome accounting:** `engine/py/intracellular.py:306` reads `is_translating`, never
  set anywhere; constraint is a no-op capped at the hardcoded 100.
- **Empty stubs:** `engine/py/molecular/mutations.py`, `proteins.py` are `class …: pass`.
- **GPU FBA:** CPU path is real `scipy.linprog`; GPU path is an unconstrained 5-step ascent
  that breaks mass balance.

### Severity 3 — Structural

Duplicated tree: top-level `modules/`, `engine/`, `dashboard/` *and* a parallel `cognisom/`
package (~151k of 246k LOC). 15 modules exist in both. ~90 markdown roadmap files against a
much smaller body of working code. Pick one tree; archive the other.

### What's genuinely good (keep and build on)

Real CUDA in the particle/stochastic/diffusion layer: Smoldyn-style Brownian reaction-diffusion
with GPU spatial hashing, correct direct-Gillespie SSA and tau-leaping, correct explicit 3D
Laplacian. `tme_classifier.py` faithfully implements Teng et al. 2015 (correctly cited). The
molecular pipeline (mutation effect, protein domains, ESM-2, peptidome, MHC PWM) is legitimate.
MAD audit/provenance/model-card plumbing is real with honest in-code disclaimers.
`docs/patent/SCOPING.md` is unusually candid (self-grades ~20-30% production-grade).

---

## 3. Business value

**Honest value:** a genomics-informed, *explainable* therapy-prioritization and
neoantigen-vaccine-design **research/education tool**, with a real molecular-effect engine and
real spatial/stochastic solvers underneath. Framed as Research Use Only and retrospectively
validated against public cohorts, it is a credible grant and academic-collaboration asset.

**Liabilities in the current framing:**

- **"Digital twin" + "clinical decision support" + "treatment response prediction"** applied to
  an uncalibrated rule engine with zero prospective validation.
- **Self-declared regulatory posture.** "Non-Device CDS under Cures Act §3060" and the "FDA
  7-Step Credibility Framework marked executed" are internal assertions resting on the
  tautological metric above. No FDA submission exists.
- **Scale claims contradict reality.** "Production-ready" / "GPU-accelerated" against ~100
  cells on CPU and a no-op GPU ODE kernel.
- **Identity confusion.** "eyentelligence" vs "Cognisom Therapeutics" vs "cognisom" — three
  different companies. "Therapeutics" implies drug development, a far more regulated posture.
- **Governance:** hardcoded investor-room password in
  `cognisom/dashboard/_pages/41_investors.py`; extreme docs-to-code ratio; projections with
  450–750% Y1→Y2 growth and no LOIs/pilots.

**IP:** at best one narrow, computer-implemented provisional (sequence-as-primary-state +
exosome transfer), explicitly *not* GPU-specific. Novelty rests on integration of textbook
components; §101/§102 and prior-public-disclosure risks are real and acknowledged internally.

**Bottom line:** a pre-seed research prototype with one interesting narrow idea and a
technically self-aware author. Converting it to defensible value requires *narrowing and
truth-in-labeling*, not more features.

---

## 4. Recommended fix order

1. Fix the twin determinism bug (per-call `np.random.Generator`).
2. Quarantine the fabricated validation (`eval/simulation_accuracy.py`; MAD concordance).
3. Honestly label the TCR recognition step at the API boundary; scaffold TCRdist3.
4. Fix diffusion sub-cycling (`grid.py`); delete the two empty stubs.
5. Gate or retract GPU-ODE claims; add a regression test on the default BDF path.

---

## 5. Status — correctness fixes applied (2026-08-06)

Fixed in this branch, with regression tests in `tests/test_correctness_fixes.py`
(mirrored to `cognisom/tests/`). All changed files exist in both the top-level and
nested `cognisom/` trees and were kept byte-identical.

| # | Defect | Fix |
|---|---|---|
| 1 | Twin RNG seeded on drug name only; used global `np.random.seed` | Per-`(patient, treatment)` `np.random.default_rng` seeded via `hashlib` (stable across processes; global RNG untouched) |
| 2 | `eval/simulation_accuracy.py` fabricated `simulated = observed ± noise` in two places; fake sequential PMIDs | Fabrication removed — benchmarks that cannot run are reported as `NOT EVALUATED`, excluded from pass rate/metrics; grade returns `N/A` when nothing ran; fake PMIDs replaced with `UNVERIFIED — citation required` and every built-in benchmark marked `verified: False` |
| 3 | Diffusion took one clamped sub-step instead of covering the requested `dt` | Sub-cycles `ceil(dt/dt_stable)` stable steps, with a capped-substep warning |
| 4 | **Diffusion did not conserve mass** (1000 → 1634 units) — `np.roll` implies periodic BCs, then the boundary copy injected material each step | Replaced with replicated ghost-cell (`np.pad(..., mode='edge')`) Laplacian, which *is* the no-flux condition and conserves mass to round-off |
| 5 | Ribosome constraint was a no-op (`is_translating` never written) | Finite ribosome pool consumed across all mRNAs within a step, reset each `step()`; flags now maintained |
| 6 | GPU BDF kernels no-op for `n_species != 2` yet report success | `ODESystem.gpu_bdf_compatible` opt-in; incompatible systems fall back to the real CPU solver with a warning |
| 7 | Empty `mutations.py` / `proteins.py` stubs **shadowed** the real `Mutation` dataclass via `molecular/__init__.py` | Stubs deleted; the real `Mutation` from `nucleic_acids.py` is re-exported |

Defect #4 was not in the original review — it was uncovered by writing a
mass-conservation test for #3.

**Not addressed** (needs a design decision, not a bug fix): the hash-random TCR
recognition in `engine/py/immune/tcr_repertoire.py`, which requires substituting a
real sequence-based model such as TCRdist3. Also unaddressed: the tautological MAD
concordance metric in `cognisom/validation/mad_study.py:547-549`, which needs real
outcome labels rather than a code change.

Pre-existing unrelated failures, untouched: `tests/test_ode_solver.py::test_cell_heterogeneity`
and `::test_get_species` fail identically before and after these changes.
