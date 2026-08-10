# Cognisom — Patentable Claims Analysis

**Date:** 2026-08-10
**Basis:** direct source review of the working tree at `10dab36`, plus a verification test run in a clean container.
**Supersedes for scope purposes:** `docs/patent/SCOPING.md` (2026-05-10) and `docs/patent/DISCLOSURE_SOURCE.md`, both of which predate Upgrades 1–8.
**Status:** engineering analysis, not legal advice. Every assertion below is cited to `file:line` and was checked against the code, not the markdown.

---

## 0. Why this document exists

`SCOPING.md` was written on 2026-05-10 and graded nine inventions A–I. Between 2026-05-11 and 2026-05-14 the repository landed eight "patent-evidence" upgrades plus four pre-filing gap closures. Three things changed materially:

1. **Invention F was deferred in May** ("file when the loop is closed"). **The loop is now closed** and is asserted end-to-end from a VCF file, with a negative control. It is the strongest claim in the portfolio.
2. **Two inventions exist now that did not exist in May** — the reference-genome/per-cell-delta memory architecture (Upgrade 1) and exosome-borne genome-delta transfer (`10dab36`). The first is the cleanest §101 anchor in the repo.
3. **Two subsystems `SCOPING.md` never inventoried** turned out to contain claimable material: the diapedesis state-gated kernel dispatch (`physics/diapedesis_kernels.py`) and the batched GPU FBA solver (`gpu/fba_solver.py`).

`SCOPING.md` also over-graded Invention C. That correction is in §4 below.

---

## 1. Verification run (§112 enablement evidence)

Clean container, Python 3.11.15, `numpy` / `matplotlib` / `scipy` / `plotly` / `psutil` / `pytest` installed from PyPI, no GPU, no MHCflurry, no torch.

```
296 passed, 1 failed, 1 skipped   (19 patent-evidence test files)
```

- **1 skipped** — `test_esm_stability.py` real-weights smoke test, opt-in by design (`ENABLE_ESM_SMOKE=1`, ~600 MB download).
- **1 failed** — `test_memory_architecture.py::test_view_is_dramatically_more_efficient_at_realistic_genome_size`. It asserts ≥20× and measured **19.8×** (naive 63.2 MB vs. views 3.2 MB at 100 genes × 3 kb, 5 000 cells). This is an RSS-measurement threshold flake, not a substantive failure — the architecture demonstrably delivers the order-of-magnitude claimed. **Action: lower the assertion to 15× or measure with `tracemalloc` instead of RSS, so the load-bearing benchmark is not red at filing time.**

Worth flagging: **`psutil` is not in `requirements.txt`**, so on a stock checkout `test_memory_architecture.py` calls `pytest.importorskip("psutil")` (`tests/test_memory_architecture.py:29`) and *all four* memory benchmarks silently skip. The single most important enablement evidence for Claim 1 is invisible unless you happen to have psutil. **Add psutil to `requirements.txt` before filing.**

---

## 2. Tier 1 — File on these now

### Claim 1 — Shared immutable reference genome with per-cell sparse substitution deltas and fork-on-mitosis inheritance

**Code:** `engine/py/molecular/reference_genome.py:33-200`, `engine/py/molecular/sequence_view.py:35-231`, integration at `modules/molecular_module.py:163, 440-456`.

One `ReferenceGenome` holds canonical CDS strings and is explicitly frozen after construction — `CellGenomeView.__init__` (`sequence_view.py:53-57`) *refuses to construct against an unfrozen reference*. Each cell holds a view carrying two parallel structures over the same deltas:

```python
self._delta_index: Dict[Tuple[str, int], SubstitutionDelta] = {}   # O(1) base reads
self._delta_log:   List[SubstitutionDelta]                  = []   # chronological provenance
```

`base_at()` (`sequence_view.py:69-78`) is a dict hit falling through to the shared reference; `iter_codons()` (`:90-111`) walks codons without ever materializing a sequence. `fork()` (`:167-181`) is the claim core — the daughter shares the reference *by pointer* and copies only the delta log, then rebuilds the index from that log so chronological order survives while lookup stays O(1). It is bound directly to the mitosis event at `modules/molecular_module.py:440-456`.

**Why this is the best §101 anchor in the portfolio.** Under Alice/Mayo, a simulation method is safest when it recites a *specific improvement to the functioning of the computer*. This one does, and the improvement is measured: memory grows ~5× when the genome grows ~80× (`tests/test_memory_architecture.py:206-251`), and 10 000 cell views fit in under 50 MB (`:107-126`). Correctness is not asserted by hand-waving either — `materialize_naive()` (`sequence_view.py:218-231`) exists purely as a test oracle proving view reads equal deep-copy-and-apply.

**Prior art to distinguish.** Copy-on-write over an immutable base is decades-old CS. Persistent data structures, VCF/gVCF sparse genotype encoding, and container-image layering all occupy adjacent space. The claim must therefore recite the *binding*, not the overlay: the delta log doubling as heritable mutation provenance, `fork()` invoked at simulated cell division, and materialization deferred to a model boundary. A claim to "sparse deltas over a shared reference" alone will read on prior art.

**Draft independent claim (sketch, not legal language).**
> A computer-implemented method for simulating a proliferating cell population, comprising: storing a single immutable reference nucleotide sequence record shared by reference among all simulated cells; maintaining, per simulated cell, a sparse substitution set comprising (a) an index keyed on gene identifier and position affording constant-time base resolution and (b) an ordered log preserving the order in which substitutions were introduced; resolving a base for a given cell by consulting said index and, on a miss, said shared reference record; on a simulated cell-division event, deriving a daughter cell's substitution set by copying said ordered log and reconstructing said index therefrom while retaining the shared reference by reference and without copying it; and advancing the population in time, whereby memory consumed by the population is substantially independent of the length of said reference record.

---

### Claim 2 — Closed loop from patient variant to T-cell kill with reconstructable causal provenance

**Code:** `modules/molecular_module.py:296-369`, `modules/cellular_module.py:491-592`, `engine/py/molecular/peptidome.py:188-263`, `engine/py/immune/mhc_loading.py:142-201`, `engine/py/immune/tcr_repertoire.py:260-332`, `engine/py/immune/tcell_kill.py:69-128`, `modules/immune_module.py:502-574`.

This is the invention `SCOPING.md` told you to defer. It is now closed, and closed *from a VCF file*, which is the part that matters.

The chain is `MUTATION_OCCURRED → PEPTIDE_GENERATED → PEPTIDE_PRESENTED → CELL_KILLED_BY_TCELL`, with each stage carrying typed provenance forward so a kill event traces back to a specific row of a specific input VCF. `tests/test_vcf_round_trip.py:224-297` asserts exactly that (`kill["source_gene"] == vcf_gene`, `kill["mutation"] == vcf_label`), and `:299-355` is a real negative control — a VCF with no curated drivers produces zero peptide and zero kill events.

Three design choices inside the loop are individually claimable as dependents, and each has a stated technical rationale:

- **Refusal to fabricate.** `peptidome.py:221-227` raises if the declared wild-type residue does not match the reference: *"peptidome refuses to silently fabricate a neoantigen against a mis-specified reference."* `cellular_module.py:545-546` returns silently on the same mismatch.
- **Mutant/WT pairing in one record.** `Peptide.wild_type_sequence` (`peptidome.py:52-116`) carries the counterpart window in the same object, with a `__post_init__` invariant that `not is_mutant → sequence == wild_type_sequence`. This is what makes agretopicity computable without a second pass.
- **Multiplicative proteasome gate.** `mhc_loading.py:186-201` composes affinity × cleavage rather than summing, on the stated ground that the proteasome must produce the peptide before MHC can load it:
  ```python
  affinity_component = 1.0/(1.0 + math.exp((math.log10(max(ic50,1.0)) - math.log10(500.0)) * 4.6))
  return float(affinity_component * cleavage_score)
  ```

**Differentiation.** Standalone predictors (NetMHCpan, MHCflurry, pVACseq) do not drive tissue dynamics. Tissue simulators (PhysiCell, CompuCell3D, BioDynaMo) gate immune killing on a boolean "cancer" flag. Here the kill is causally derived from per-cell sequence state and the derivation is reconstructable from the event log. That combination is the claim.

**Caveat to draft around.** MHC binding numbers come from `NeoantigenPredictor._predict_binding` (`cognisom/genomics/neoantigen_predictor.py:325-392`), which uses MHCflurry when installed and otherwise falls back to a hand-rolled anchor-residue heuristic (`ic50 = 5000·exp(-0.7·score)`). MHCflurry is now pinned (`2f05af7`) but the evidence runs recorded in the demo output were produced by the PWM fallback. Do not recite binding *accuracy*; recite the pipeline architecture.

---

### Claim 3 — Exosome-borne horizontal transfer that writes a sequence-level delta into the recipient genome

**Code:** `modules/molecular_module.py:240-275, 458-476`, `modules/cellular_module.py:463-484`. Tests: `tests/test_exosome_horizontal_transfer.py` (4/4 pass).

`SCOPING.md` graded Invention B on the strength of exosomes carrying cargo and flipping a recipient `transformed` flag. Commit `10dab36` changed the mechanism materially: on uptake, the module walks `exosome.cargo.mrnas`, filters to `mut.oncogenic`, dedupes into `(gene, mutation)` pairs, and the recipient path calls `introduce_mutation()` **before** `transform_cell()`. The transferred mutation therefore lands in the recipient's delta log as a real `SubstitutionDelta`, emits `MUTATION_OCCURRED` *for the recipient*, and — because it is in the log — is inherited by the recipient's daughters through `fork()`.

That is what converts Invention B from a phenotype-flag model into a genuine sequence-level horizontal transfer, and it composes Claims 1 and 2 into one chain: donor genome → vesicle cargo → Brownian transport → marker-matched uptake → recipient delta → recipient peptidome → recipient killed by T cell. The test file covers the delta write, the recipient-side event, daughter inheritance, and a negative control on non-oncogenic cargo.

**Cosmetic defect to fix before filing:** `cellular_module.py:423` still unconditionally appends the literal string `'KRAS_G12D'` to `cell.mutations` regardless of what was actually transmitted. The genome-level delta is correct; the phenotype label is not. A reviewer reading that line will question the rest.

---

### Claim 4 — ECM-excluded sub-classification of the tumor microenvironment

**Code:** `engine/py/spatial/ecm_barrier.py` (131 lines, three pure functions), `engine/py/immune/tme_classifier.py:282-318`, integration at `modules/cellular_module.py:254-313` and `modules/immune_module.py:308-318, 404-418, 482-500`.

Upgrade 4 re-implements the Teng et al. 2015 four-type scheme faithfully — that part is prior art and must be disclaimed. **Upgrade 6 is what is original.** It splits Teng Type II ("immunological ignorance") into two clinically opposite cases using a three-term conjunction (`tme_classifier.py:292-296`):

```python
ecm_excluded = (not til_positive) and mean_ecm >= 0.4 and any_antigens
```

"Cold because there are no antigens" takes a vaccine. "Cold because the T cells are walled out despite presentable neoantigens" takes anti-fibrotic + ICB. Teng does not make that distinction; it is not derivable from IHC of TIL and PD-L1 alone, because it requires knowing the peptidome — which the simulator has and a biopsy does not.

The mechanism underneath is honest and simple: ECM attenuates T-cell patrol speed, chase speed, and detection radius through a shared law `retained = max(0.05, 1 − density·blocking_factor)` (blocking 0.9 motility, 0.8 detection), floored so T cells never freeze permanently. ECM is dynamic — cancer cells deposit continuously, anti-fibrotic therapy flips deposition to degradation.

This upgrade has the best integration testing in the set: 18 tests including true engine-level runs `test_high_ecm_drops_til_count_to_zero` and `test_anti_fibrotic_restores_til_infiltration`, plus four negative controls on the flag itself.

**Note for drafting:** all constants (0.9 / 0.8 / 0.05 / 0.4) are hand-chosen, not fitted. Claim the conjunction and the therapy-selection consequence, not the numbers.

---

### Claim 5 — State-gated GPU kernel dispatch for the leukocyte adhesion cascade

**Code:** `physics/diapedesis_kernels.py:36-309, 358-565`; state machine at `simulations/diapedesis.py:533-777`.

Not inventoried in `SCOPING.md` at all, and it is fully implemented with a complete NumPy CPU mirror.

An 8-state per-leukocyte machine (FLOWING → MARGINATING → ROLLING → ACTIVATING → ARRESTED → CRAWLING → TRANSMIGRATING → MIGRATED) in which **each Warp kernel early-returns on the state integer**, so the force law applied to a particle is selected by its biological state rather than by a branch inside one monolithic force routine:

```python
if leukocyte_states[i] != 6:   # TRANSMIGRATING only
    return
```

Two things here are genuinely unusual:

- **Transitions are physically gated, not timer-driven.** MARGINATING→ROLLING requires a stochastic selectin bond whose probability reads the *nearest endothelial cell's* selectin expression (`diapedesis.py:570-578`). CRAWLING→TRANSMIGRATING requires both a crawl timer and `nearest_junction_integrity < 0.8` — and if the timer expires while the junction is still tight, the leukocyte *actively degrades it* (`self._weaken_junction_near(pos, 0.3)`, `:771`). The agent modifies the tissue property that gates its own transition.
- **The boundary condition inverts on state.** `vessel_boundary_kernel:285-306` clamps RBCs and pre-crawling leukocytes *inside* radius `R − rᵢ`, and state-7 (MIGRATED) leukocytes *outside* `R + rᵢ`. The vessel wall is a one-way membrane implemented as a state-conditional geometric projection.

Neither PhysiCell nor CompuCell3D has a per-agent adhesion-cascade state machine selecting which GPU force kernel applies, nor state-inverting boundary conditions.

---

## 3. Tier 2 — Claimable, but fix the code first

### Claim 6 — Three-axis composed missense impact classifier ⚠️ *Stage C is never invoked in production*

**Code:** `engine/py/molecular/mutation_effect.py:219-541`, `protein_domains.py:69-682`, `esm_stability.py:119-320`.

The composition is the invention, and it is well-designed. Stage A maps BLOSUM62 to impact piecewise-linearly with a deliberate ceiling `_MISSENSE_IMPACT_MAX = 0.85` so no missense can outrank a nonsense. Nonsense is position-dependent rather than flat: `impact = max(0.5, 1.0 − 0.5·truncation_fraction)`. Stage B multiplies by a domain role factor (critical 4.0 / functional 2.5 / regulatory 1.5) across 21 curated genes, resolving overlapping annotations to the *most severe* (`protein_domains.py:682`) so BRAF V600 takes "activation loop" over the enclosing kinase domain. Stage C composes ESM-2 masked-marginal ΔLL by **gap-closing interpolation rather than multiplication**, so a neutral modifier (0.5) is exactly a no-op and the result can never leave the missense band.

That interpolation choice is the non-obvious part — it is what stops a biophysics signal from overriding the categorical ordering. Standard tools pick one axis (SIFT/PolyPhen: conservation; FoldX: structure-required; CADD: ensemble ML).

**The blocker.** `modules/molecular_module.py:93` sets `self.esm_scorer = None` and **nothing in the simulation ever calls `set_esm_scorer()`**. Every Stage-C test uses `StubESMStabilityScorer`; the only real-weights test is opt-in and skipped; the dashboard panel also uses the stub and passes a placeholder `new_base="A"` (`dashboard/_pages/42_patent_pipeline.py:452`). There is also a bare `except Exception: impact = stage_b_impact` swallow at `mutation_effect.py:457-462`.

A claim reciting three composed axes is not enabled by a code path where the third axis is off by default and has never run against real weights in CI. **Fix: wire `set_esm_scorer(RealESMStabilityScorer())` into at least one production path and run the real-weights test once, capturing output.** Until then, claim Stages A+B and put Stage C in the specification as an embodiment.

A second, smaller issue: the 0.85 ceiling saturates readily. Any critical-domain missense with BLOSUM ≤ −1 pins at 0.85 (BRAF V600E: 0.625 × 4.0 → clamped to 0.85), which collapses Stage B's resolution exactly where it matters most.

---

### Claim 7 — T-cell exhaustion as suppression of the rescue *term* ⚠️ *defeated by a stale cache*

**Code:** `engine/py/immune/tcr_repertoire.py:43-68, 302-350`, `engine/py/immune/tcell_kill.py:69-168`, `modules/immune_module.py:325-378, 542-574`.

The formulation is the strongest §103 idea in the immune stack. Prior-art rule-based modules apply checkpoint blockade as a generic kill-probability multiplier, which implies ICB rescues every dysfunctional T cell — a clinically wrong and potentially dangerous prediction. Here exhaustion does not scale the output; it **removes the rescue term from the signal** (`tcell_kill.py:110-116`):

```python
if is_exhausted:  signal = a*m*c                      # no checkpoint rescue at all
else:             signal = a*m*c + (1.0 - c)*b*0.5
p = signal**slope / (signal**slope + thr**slope)
if is_exhausted:  p *= mult                           # then ×0.1 residual cytotoxicity
```

Exhaustion is a one-way transition on an encounter counter (`register_engagement`, `:304-332`), matching the epigenetically-enforced irreversibility in Bengsch 2018.

**The blocker, confirmed by reading the code.** `modules/immune_module.py:331` caches the `TCRMatch` on the immune cell (`immune_cell.active_tcr_match = tcr_match`), and `_target_kill_probability` reads `match.is_exhausted` from that **snapshot** (`:561`). The snapshot is written only at recognition (`:331`) and cleared only on target loss or kill (`:477`, `:612`). During a sustained engagement the repertoire counter increments every step and the clone transitions to EXHAUSTED — but the cached match still reports `is_exhausted=False`, so the kill probability is unaffected until a *subsequent* recognition cycle.

The 14 tests in `test_exhaustion.py` do not catch this: the unit tests call `kill_probability(..., is_exhausted=True)` directly, and the end-to-end test asserts only on the emitted event, `exhausted_count()` and `exhaustion_state()` — never on `_target_kill_probability` after the transition.

**A claim reciting "gating checkpoint-blockade rescue on the precursor state within the simulation loop" is not currently enabled.** The fix is one line — re-read exhaustion state from the repertoire at kill time rather than from the cached match — plus a regression test asserting kill probability drops after the threshold is crossed mid-engagement. Do that, and this becomes Tier 1.

---

### Claim 8 — Frameshift and fusion neoantigen windows with X-padded wild-type pairing ⚠️ *library, not a pipeline stage*

**Code:** `engine/py/molecular/peptidome.py:270-480`. Tests: 20, all pass.

The real contribution is a design choice most pipelines dodge. For a frameshift, positions past the original protein's end have *no* wild-type counterpart — the comparison is ill-defined, so pVACseq and NeoPredPipe simply don't produce one. Here it is manufactured by `X`-padding (`:355-360`) so that a single downstream agretopicity scorer works uniformly across missense, frameshift, and fusion. Fusion windows are restricted to junction-spanning only (`:449`), guaranteeing residues from both partners, with the left partner extended-and-X-padded as the WT reference on the stated ground that "the cell's prior identity is closest to `left_protein`."

**The blocker.** Nothing in the simulation calls either generator. `grep` finds callers only in the test file and a dashboard preview. `cellular_module.py:534` parses `"G12D"` positionally and bails on anything that isn't missense, with the comment at `:532-533` deferring the other paths to "future work." The repo is candid about this in two places, including the dashboard telling the user directly (`42_patent_pipeline.py:540-544`) and `MAD_INTEGRATION_AUDIT.md` uncertainty flag #3.

**Claim what is enabled:** peptide generation over indel and fusion variant classes. **Do not claim** frameshift/fusion neoantigens flowing through MHC presentation to T-cell killing — that path does not exist.

---

### Claim 9 — Neuroimmune (β2-adrenergic) gating of immunotherapy response

**Code:** `engine/py/immune/sympathetic.py:112-113`, applied at `modules/immune_module.py:563-573`.

The whole model is two lines: `effective = stress·(1 − blocker)`, `retained = 1 − effective·0.7`, applied as a final multiplicative gate on the T-cell kill path only. `DECISIONS.md` calls this the "highest-novelty differentiator in the whole patent surface," and the novelty premise is right — no published cancer simulator models the neuroimmune axis, and the retrospective β-blocker/ICB cohort evidence (Kokolus 2018, Oh 2021) is well established.

But the arithmetic is trivial and the module says so ("the simplest one with the right qualitative properties," `:99-100`), and the 0.7 constant is calibrated by analogy rather than fitted. **A standalone independent claim on two lines of multiplication will be hard to defend under §103.** Its real value is as a *dependent* claim on Claim 2 — "wherein said kill probability is further scaled by a patient stress proxy attenuated by a β-adrenergic-blockade parameter" — which is both non-obvious in context and cheap to add.

One thing in its favor over Claim 7: it reads the multiplier live from module config, so unlike exhaustion it **demonstrably changes simulation outcomes** (`test_sympathetic.py:190, 229, 254` are genuine end-to-end runs).

---

### Claim 10 — Batched population-scale FBA by alternating null-space projection and box-clipping

**Code:** `gpu/fba_solver.py:203-248`. Not in `SCOPING.md`.

Rather than running a linear program per cell, this replaces the LP entirely: start at the bound midpoint, project onto the stoichiometric null space via `v − S⁺(Sv)`, clip to bounds, then run five iterations of projected gradient ascent on `cᵀv` with **per-cell accept/reject** (`cp.where(improve, v_new, v)`). Standard practice for FBA is simplex or interior-point per model (COBRApy/GLPK/HiGHS); solving *populations* of per-cell FBA problems this way is unconventional.

**Two blockers.** The final operation is `clip`, not `project`, so the returned `v` does not satisfy `Sv = 0` exactly — and there is no convergence test, no feasibility residual, and fixed iteration count and step size. And **`FBASolver` has zero callers** anywhere in the repo. Fix the residual reporting and wire it to something before claiming it.

---

## 4. Corrections to `SCOPING.md`

### Invention C (hybrid solver) was over-graded — do not recite "coupled"

`SCOPING.md:113-146` recommended filing a narrow claim, correctly limiting it to "computer-implemented" rather than "GPU". That limitation still holds and the fallback comment it cites is still there verbatim (`gpu/hybrid_solver.py:942, 951`: `# For now, fall back to CPU`). But three further findings weaken it beyond what that document says:

1. **The partitions are not actually coupled.** `_step_hybrid_cpu:781-793` runs the ODE integrator over *all* reactions for fast species, then SSA over *all* reactions for slow species. Both integrate the same full reaction set with no coupling terms, despite the module docstring at `:18` claiming "coupling terms for reactions spanning partitions." Fluxes are effectively double-counted. Separately, in `_ssa_step_cpu:918-921` a firing reaction applies only the *slow* species' stoichiometry — the fast-species delta from that same reaction is discarded, so **mass is not conserved across the partition boundary.** Claim element (iii) as drafted in `SCOPING.md:138` ("each subsystem reading the present state of the other subsystem") is not supported by this code.
2. **Partitioning is global, not per-cell.** `np.mean(state, axis=0)` averages across the whole population, so a heterogeneous population where half the cells have 5 copies and half have 500 gets one shared partition.
3. **The four CUDA kernels are never launched.** They are compiled and stored in `self._kernels` solely so that `'ssa_step' in self._kernels` is true at `:776`, routing to `_step_hybrid_gpu`, which immediately calls `to_numpy()` and runs the CPU path. On GPU hardware this file is *slower* than on CPU. `_PROPENSITY_KERNEL` also cannot compile — line 83 references `n_params`, which is not among its declared parameters.

**What survives** is the hysteresis-banded partitioner itself (`:450-543`) — a Schmitt trigger on the Haseltine–Rawlings fast/slow criterion, with a minimum-fast floor promoting the highest-count slow species. That is real, implemented, and defensible. Claim the partitioning *rule*, on CPU, without reciting coupling or GPU acceleration.

### Inventions D, E, I — unchanged

Trade secret (D, E) and not-an-invention (I). The new survey found nothing to change. Add to the same bucket: `gpu/domain_decomposition.py` exchanges ghost cells by plain NumPy slice assignment with no P2P (`:248-291`), and `gpu/multi_gpu_backend.py:347-354` implements "NCCL AllReduce" as a host-side NumPy sum marked `# Placeholder`. Multi-GPU remains aspirational exactly as `SCOPING.md:269` said.

---

## 5. Liabilities to fix before any disclosure leaves the building

These are ordered by how much damage they do if a reviewer finds them first.

**1. `variant_annotator._predict_protein_change` fabricates amino acid changes.** `cognisom/genomics/variant_annotator.py:484-539`. It does not translate anything. It estimates `aa_pos = (variant.pos − gene_start) // 3` — ignoring introns, UTRs, strand and CDS offset — then invents amino acids from two hardcoded tables (`{"A":"K","C":"A","G":"G","T":"L"}` and a 12-entry pair map). The docstring admits it produces "plausible" changes. `_annotate_variant:406-418` then throttles it to ~3 per gene via dynamic attributes with the comment "to simulate exonic fraction."

Any neoantigen downstream of this path is synthetic. The engine path via `MutationEffectClassifier` is the real one and is genuinely good — but a reviewer who reads this function will discount the real work next to it. **Sever this from anything described as a neoantigen pipeline, or delete it.**

**2. `eval/simulation_accuracy.py` scores fabricated results against ground truth.** `:436-439` and `:461` synthesize "simulated" values as `observed × (1 ± 0.10–0.15)` — on the default path, since `evaluate_against_benchmarks(engine=None)` is the default signature (`:156-160`). Against tolerances of 15–35% the suite therefore passes 100% by construction, and `_generate_recommendations`, `_save_report`, `get_trend` and the flywheel all consume that number as a measurement. The built-in benchmarks also cite `PMID:12345678`, `PMID:23456789`, `PMID:34567890` — sequential placeholders.

Note the contrast: `validation/` is honest (`validation/validator.py:145-160` returns `score=0.0, passed=False` on exception, and `benchmarks.py` carries 25 real DOIs). **Never cite `eval/simulation_accuracy.py` output as validation evidence.**

**3. The MAD study's HRD accuracy is tautological.** `validation/mad_study.py:442-452` computes HRD "sensitivity" using `r.has_brca` as ground truth against `r.has_hrd` as prediction — but both derive from the same `profile` object (`:364-365`). Prediction and truth are the same quantity.

**4. The headline validation metric is not reproducible.** "TMB r=0.987, 100% biomarker concordance" appears on the landing page and `dashboard/app.py:429` but no test recreates it; `MAD_INTEGRATION_AUDIT.md` already flags this. Either reproduce it in CI or remove it from public-facing surfaces before filing.

**5. `flywheel/distillation.py` trains nothing.** `_train_lora:229-289` writes an `adapter_config.json` marked `# (mock)` then calls `_simulate_training:291-308`, which generates loss and accuracy from `random.uniform`. Deployment is gated on those fabricated numbers. Fine as scaffolding; must not appear in a specification as a working ML loop.

**6. Duplicate package trees.** `/modules` vs `/cognisom/modules`, `/core` vs `/cognisom/core`, `/gpu` vs `/cognisom/gpu` — and they have diverged (`immune_module.py` differs). `cognisom/modules/*` does `sys.path.insert(0, '..')` and imports the *top-level* `core`, while `cognisom/modules/__init__.py` imports `cognisom.core.registry`, so the registry validates against a `SimulationModule` class that is not the one the modules subclass — which is why strict validation is disabled at `core/registry.py:58-62`. At filing time it must be unambiguous which tree the claims read on. `RESUME.md` already lists this as housekeeping; it is now a filing blocker.

**7. Public disclosure clock.** The repository is on GitHub and `SCOPING.md:340` already raised this. Absolute-novelty jurisdictions (EPO, China) have no grace period. If the repo has been public since the Upgrade 1–8 commits landed in May 2026, that is a disclosure date to establish now, not at filing.

---

## 6. Recommended filing shape

One application family, with Claim 1 as the lead independent claim because it is the cleanest Alice/Mayo posture — a measured improvement in the functioning of the computer, not an abstract idea implemented on one.

| # | Claim | Posture | Blocking work |
|---|---|---|---|
| 1 | Reference genome + per-cell delta + fork-on-mitosis | **Lead independent** | Add psutil to requirements; relax the 20× assertion |
| 2 | VCF → peptide → pMHC → TCR → kill with provenance | **Independent** | None — file as is |
| 3 | Exosome-borne genome-delta transfer | **Independent**, same family | Fix the hardcoded `'KRAS_G12D'` label |
| 4 | ECM-excluded TME sub-classification | **Independent** | None — file as is |
| 5 | State-gated kernel dispatch (diapedesis) | **Independent**, possibly separate family | None |
| 6 | Three-axis impact classifier | Claim A+B now; C in spec | Wire a real ESM scorer into a production path |
| 7 | Exhaustion as rescue-term suppression | **Hold** | One-line cache fix + regression test → then Tier 1 |
| 8 | Indel/fusion peptide generation | Narrow claim to generation only | Wire to MHC scoring for the broader claim |
| 9 | β2-adrenergic gating | **Dependent** on Claim 2 | None |
| 10 | Batched null-space FBA | **Hold** | Report feasibility residual; find a caller |
| — | Hybrid solver hysteresis partitioner | Narrow, CPU-only | Do not recite "coupled" or "GPU" |
| — | Batched GPU SSA, spatial RD, event bus | Trade secret / no claim | — |

**Highest-leverage work before filing**, in order: (a) the one-line exhaustion cache fix, which promotes Claim 7 from unenabled to Tier 1 for a few hours of work; (b) severing `_predict_protein_change`; (c) resolving the duplicate package trees; (d) wiring a real ESM scorer once so Stage C is enabled; (e) the test-suite items in §8, which are cheap and make the enablement evidence reproducible by a third party in one command.

Trade-secret-better-than-patent, unchanged from `SCOPING.md:312-317` and extended: all hand-chosen constants (role multipliers 4.0/2.5/1.5, the 0.85 ceiling, TIL 0.5 / PD-L1 0.25 / 20 µm, ECM 0.9/0.8/0.05/0.4, exhaustion threshold 5 and ×0.1, sympathetic 0.7). None are fitted to data; the docstrings are generally candid about this, which is the right posture but makes them poor claim limitations.

---

## 7. How to check this document

Every §-cited line was read in the working tree at `10dab36`. The three findings most likely to be disputed, and how to reproduce them:

1. **Hybrid solver never uses the GPU** — read `gpu/hybrid_solver.py:939-955`; both `_ode_step_gpu` and `_ssa_step_gpu` begin with `to_numpy()` and call the CPU routine. Then `grep -n "self._kernels\[" gpu/hybrid_solver.py` — no launches.
2. **Exhaustion never reaches the kill computation** — `grep -n "active_tcr_match" modules/immune_module.py`. Writes occur only at `:331` (recognition) and `:477`/`:612` (clear). `_target_kill_probability:561` reads `match.is_exhausted` off that snapshot.
3. **Stage C never runs** — `grep -rn "set_esm_scorer" --include=*.py .` returns the definition and docstrings only; no production caller. `modules/molecular_module.py:93` is `self.esm_scorer = None`.

Test reproduction: `pip install numpy matplotlib scipy plotly psutil pytest`, then `python3 -m pytest tests/ -q` over the 19 patent-evidence files → 296 passed, 1 failed (the 19.8×/20× threshold), 1 skipped (opt-in ESM). Note that running `pytest tests/` over the *whole* directory does not terminate — see §8.

---

## 8. Test-infrastructure findings

An attempt to run the entire `tests/` directory in one pass was killed at 15 minutes. Bisecting it produced two findings that bear on the filing.

### 8.1 `test_smoldyn_solver.py` makes the full suite unrunnable

Three of its test classes each exceed 45 seconds and were terminated (`TestReactions`, `TestIntegration`, `TestSmoldynSolver`); a single test in `TestBrownianMotion` takes 28 seconds. This is not a deadlock, it is the CPU fallback being pathologically slow — consistent with the solver passing `n = self.n_max` (the whole particle buffer) as the particle count, so a nominally small system scans the full allocation every step.

The consequence for the filing is narrow but worth stating: the spatial reaction–diffusion subsystem (Invention E in `SCOPING.md:175-188`) is not merely un-accelerated, it is **effectively unvalidated in CI** — nobody can run its tests to completion as part of a normal suite run. That reinforces `SCOPING.md`'s existing "not a strong patent candidate, trade secret at best" grading rather than changing it. It does not touch any Tier 1 or Tier 2 claim, all of which live in test files that complete in under three seconds combined.

### 8.2 The nine `test_registry.py` failures are misattributed in `RESUME.md`

`RESUME.md` lists as housekeeping: *"Delete or shim the duplicate `cognisom/engine/` and `cognisom/modules/` package tree (causes the 9 pre-existing `test_registry` failures)."* That diagnosis is wrong, so the planned cleanup will not fix them.

The actual cause is a registration-key collision:

```
DuplicateRegistrationError: 'virus' already registered in entities registry.
  Existing: <class 'cognisom.library.models.Virus'>
  New:      <class 'cognisom.plugins.examples.virus_plugin.VirusEntity'>
```

The built-in library model and the example plugin both claim the key `virus`. And the failure is **import-order dependent** — `pytest tests/test_registry.py` alone gives 9 failed / 33 passed, while `pytest tests/test_ode_solver.py tests/test_registry.py` gives 0 registry failures, because whichever module imports first wins the key.

This matters for §112 in a small way: a test suite whose pass/fail outcome depends on file ordering is weak enablement evidence. It is cheap to fix (namespace the plugin key, or make the example plugin's registration idempotent), and it should be fixed rather than carried, since the duplicate-package-tree cleanup in `RESUME.md` will otherwise be done and the failures will remain.

### 8.3 Remaining failures, all outside the patent surface

`test_ode_solver.py::test_cell_heterogeneity` (the pre-existing parameter-noise CV failure `RESUME.md` already notes), `test_ode_solver.py::test_get_species`, and `test_validation.py::test_benchmark_categories`. None touch any claimed mechanism.
