# How the simulation computes what it shows

This document is for a scientist deciding whether anything here is worth
their time. It describes how each quantity in a Cognisom run is
calculated, where in the source to find it, and — the part that matters
most — how far each number can be trusted and how to check that claim
yourself. Every file path is clickable in the repository.

It is not a results paper. Where a mechanism has been tested against
something outside the model, section 5 says so and gives the result,
including the test that came back null.

**Status: research use only.** Nothing here is a clinical finding, a
diagnosis, or a treatment recommendation.

---

## 0. What this is, in one paragraph

A multi-scale agent-based simulation of a tumour and its
microenvironment, run cell by cell. Each cell is an object with a
position, a type, a genome derived from a real patient callset, and a
state that is advanced on a fixed time step by nine modules sharing one
event bus. The modules are independent: each reads the shared cell
population, computes its own physics or biology, and emits events. There
is no global "tumour growth equation" — population changes are what
falls out of per-cell division and death decisions. The run exports a
USD stage, which is what the ray-traced visualisation renders.

**What it is not.** It is not fitted to outcome data. It is not
calibrated to reproduce a tumour growth curve. Several of its mechanisms
are, on measurement, inert in shipped configurations — section 5 lists
them, because knowing which parts of a model do nothing is more useful
than being told the model works.

---

## 1. The engine

Ten modules, one clock, one event bus
([`cognisom/core/simulation_engine.py`](../cognisom/core/simulation_engine.py)).
Nine of them change what a cell does. The tenth is listed with the rest
because leaving it out is how a module that drives nothing stays
invisible — the **Drives** column is the one to read first.

| Module | What it computes | Drives | Source |
|---|---|---|---|
| `cellular` | Cell population: division, death, state, mutations, inheritance | yes | [`cellular_module.py`](../cognisom/modules/cellular_module.py) |
| `vascular` | Capillary network; per-cell oxygen and glucose delivery by Krogh geometry | yes | [`vascular_module.py`](../cognisom/modules/vascular_module.py) |
| `spatial` | Diffusing fields (oxygen, glucose, testosterone, lactate, IFN-γ, chemokine) | yes | [`spatial_module.py`](../cognisom/modules/spatial_module.py) |
| `immune` | T cell, NK cell and macrophage patrol, recognition, contact and killing | yes | [`immune_module.py`](../cognisom/modules/immune_module.py) |
| `ar_signaling` | Per-cell androgen receptor ODE → PSA output; drug action | yes | [`ar_signaling_module.py`](../cognisom/modules/ar_signaling_module.py) |
| `molecular` | Transcription, translation, exosome release | yes | [`molecular_module.py`](../cognisom/modules/molecular_module.py) |
| `circadian` | Per-cell clock phase, which gates division | yes | [`circadian_module.py`](../cognisom/modules/circadian_module.py) |
| `lymphatic` | Lymphatic vessels, intravasation, metastasis | yes | [`lymphatic_module.py`](../cognisom/modules/lymphatic_module.py) |
| `morphogen` | Secreted gradients: who secretes, how it decays, who reads it | yes | [`morphogen_module.py`](../cognisom/modules/morphogen_module.py) |
| `epigenetic` | Per-cell methylation and histone marks for five genes | **no** | [`epigenetic_module.py`](../cognisom/modules/epigenetic_module.py) |

`cell_mechanics` is registered but **off by default**: its stiffnesses
have nothing in the parameter library behind them, so enabling it would
move every cell in every preset on a guess. Four solver back-ends
(`ode`, `bngl`, `hybrid`, `smoldyn`) are alternative integrators rather
than biology.

Each step is explicit and ordered: `pre_step` on every module, then
`update`, then `post_step`, then the event bus drains
([`simulation_engine.py:163`](../cognisom/core/simulation_engine.py#L163)).
The default step is `dt = 0.05 h` (3 minutes). The engine draws from one
shared RNG stream, which has a consequence worth stating: changing the
*order* of random draws anywhere changes every later draw, so two runs
that differ only in a mechanism's presence are not otherwise identical.
Several measured results in this repository exist because that was
mistaken for a real effect once.

---

## 2. How one cell's step is computed

Trace a single cancer cell through one `dt`.

**Oxygen.** Not read from the diffusion field. The vascular module finds
the cell's nearest capillary and applies the Krogh-cylinder steady state
for that radial distance
([`vascular_module.py:35`](../cognisom/modules/vascular_module.py#L35)
`krogh_shape`). The vessel is a finite reservoir, not an infinite
source. The diffusion field for oxygen is solved and read by nothing —
stated, not hidden, in the signal inventory (section 3).

**Androgen receptor and PSA.** Each mapped cell carries a row in a
stiff ODE system — AR transcription, translation, DHT binding, PSA
output — integrated per cell. The system is stiff, and explicit solvers
silently fabricate mass on it, so it is integrated with SciPy's implicit
BDF method, one batched solve per engine step over all live cells, with
the block-diagonal sparsity declared (the cells do not couple inside the
solve). A failed step holds state and logs rather than proceeding
silently. The cell's tissue DHT comes from the testosterone field,
converted by the cellular module.

**What a drug does.** `activity_of(cell_id)`
([`ar_signaling_module.py`](../cognisom/modules/ar_signaling_module.py))
returns a multiplier on AR-driven transcription, resolved against *this
cell's own receptor*: a wild-type cell under enzalutamide is blocked, a
cell carrying AR F877L is *driven* at 3.05× the untreated baseline,
because the drug is an agonist on that variant (Balbas 2013,
PMID 23580326, AR-dependent EGFP reporter: 1051 vs 345 vehicle). Drug
exposure can also fall with distance from the nearest capillary
([`engine/py/pharmacology/exposure.py`](../cognisom/engine/py/pharmacology/exposure.py));
it is uniform by default, for reasons given in that file's docstring.

**Division.** Three gates compose. The cell's own clock must be in its
permissive window (circadian gating of division, Matsuo 2003,
PMID 12934012); AR signalling must be above the androgen-dependence
threshold for an AR-dependent cell; and its division timer must have
elapsed. Two clocks are available: a compressed one (12 h / 90 h) and
the measured labelling-index turnover (`clock="measured"`: cancer
1111 h, normal 8333 h). The *ratio* between normal and cancer, 7.5:1, is
the measured part (Meyer 1982, PMID 6185700).

**Death.** A cell dies from one of several named causes, and the engine
records which. Measured on the demo configuration (48 h, seed 7):

| cause | enzalutamide | no drug |
|---|---|---|
| `androgen_withdrawal` | 155 | 0 |
| `immune_killed` | 20 | 21 |
| `intravasation` | 2 | 5 |

Read that table before believing any claim about what a treatment arm
shows: immune killing is the *same* in both arms, so the drug's effect
here is its own, not the immune system's.

**Being killed.** An immune cell patrols (scan-then-move, biased up the
chemokine gradient), makes contact, and the contact resolves by a rule
specific to its lineage:

- **CD8 T cell** — recognition requires a receptor match against the
  cell's presented peptides. A contact then delivers Poisson-distributed
  perforin hits; damage decays with a 56.7 min half-life and the target
  dies when accumulated damage crosses the threshold three hits 50 min
  apart would leave (Weigelin 2021, PMID 34471116). This replaced a flat
  per-contact probability, which was ten times the measured solid-tumour
  figure.
- **NK cell** — graded in the target's MHC class I ("missing self"),
  with a KIR–HLA licensing model when the patient's HLA was measured
  from their own reads. Each NK cell is spent after a few kills
  (1 + Poisson, capped at 6; Prager 2019, PMID 31270246, measured
  average 2.53 kills per cell).
- **Macrophage** — phagocytosis requires the target to be *marked*
  (opsonised) **and** its CD47 "don't eat me" signal blocked. Both, not
  either: Dooling 2023 (PMID 37095318) measured CD47-intact tumouroid
  growth as "unaffected ... with or without added macrophages", and
  eradication only under CD47 knockout *plus* antibody. In a shipped
  configuration neither is supplied, so macrophages make **zero** kills
  — which is the measured biology, not an oversight.

---

## 3. Where every number comes from

This is the part to attack first, and the part built to survive it.

Every number on the kill path is a record carrying its value, a range,
a **status** and a citation
([`engine/py/immune/kill_calibration.py`](../cognisom/engine/py/immune/kill_calibration.py),
`KillParameter`). The statuses are deliberately few:

| status | meaning |
|---|---|
| `sourced` / `measured` | taken from a named paper, usually read in full text |
| `derived` | computed from sourced numbers; the derivation is in the docstring |
| `ESTIMATE` | no measurement found; a range is given and the label is loud |
| `placeholder` | a literal with no source recorded beside it |

`calibration_report()` returns all of them, and a test fails if a number
is added without one
([`tests/test_kill_calibration.py`](../tests/test_kill_calibration.py)).
The same discipline covers diffusing signals — every field the engine
solves has an inventory row saying who makes it, who reads it, what it
changes, and **whether anything in a default run depends on it at all**
([`modules/signalling.py`](../cognisom/modules/signalling.py), the
`wired` column). Three of six signals are honestly marked as read by
nothing.

Library-derived parameters travel through
[`library/curated_parameters.json`](../cognisom/library/curated_parameters.json)
with their evidence attached.

**You can read all of this without running anything:** the public page
at `cognisom.com/?page=cells` is generated from these records, so every
cell type lists its dimensions, what it does, how it is simulated, each
parameter's status and source, and a `missing` list of what is *not*
modelled for that cell. It is built from the code, not written by hand
([`dashboard/cell_catalog.py`](../cognisom/dashboard/cell_catalog.py)),
and a test fails if the page and the engine disagree.

---

## 4. How the visualisation is computed

The two videos on the public demo page are **one simulation**, not two
runs sharing a seed.

- Cell colours come from `lineage_appearance`, and tissue-cell
  brightness is that cell's own oxygen — the same two functions the
  USD/RTX path uses, so the 2D recording and the 3D render cannot
  disagree about what a cell looks like.
- Both are produced from one engine state by
  [`scripts/record_demo_run.py`](../scripts/record_demo_run.py) `--usd`:
  the stage is exported from the run already in memory, not re-simulated.
- Every number the page states is read from a JSON sidecar the recorder
  writes while recording. Nothing on that page is a number somebody
  remembered.
- **A viewer can check it.** The engine's kill count, the red crosses
  drawn in the recording, and the kill markers authored into the ray-traced
  stage are three separate code paths and must be one number; a test
  asserts it ([`tests/test_the_two_demo_videos_are_one_run.py`](../tests/test_the_two_demo_videos_are_one_run.py)).
  Two of those paths once double-counted every T-cell kill.
- The RTX render is made on a different machine, so the run's identity is
  carried along: a fingerprint over the facts that define a run is
  recorded in a manifest beside the video, and the page refuses to claim
  "the same run" unless the two match
  ([`dashboard/demo_provenance.py`](../cognisom/dashboard/demo_provenance.py)).
- The capture hashes its own frames and reports `DISTINCT n/m`, because a
  stuck render is indistinguishable from a working one in a single still.
  A wrong GIF was published from this project before that check existed.

---

## 5. What has been tested against something outside the model

**One external test passed, in two cohorts.** A pre-specified,
falsifiable prediction about hypoxia and MHC class I held on 592 TCGA
colorectal tumours, using the same study, assay and statistic as the
source paper
([`cognisom/validation/hypoxia_mhc1_tcga.py`](../cognisom/validation/hypoxia_mhc1_tcga.py)).
Four analysis traps on the way each produced a confident wrong answer
first.

It was then **replicated on 493 TCGA prostate tumours** (2026-11-13) —
the tissue this engine actually models, which is the first place the
mechanism should be checked. Nothing was re-tuned: same genes, same
reference gene, same statistic, same refutation rule, only the cohort
changed. No MHC-I transcript falls with hypoxia in prostate either
(HLA-A −0.03, HLA-B +0.11, HLA-C +0.28, B2M +0.27, IFN-γ partialled
out). Exactly two cohorts have been run and both are reported; adding
cohorts until one fails would turn a test into a search.

**What the replication does not add.** New tissue and new patients, but
not a new method — the same ACTB ratio, signature definition and IFN-γ
partialling are applied, so a methodological artefact would most likely
survive into it. Two cohorts agreeing says the finding is not specific
to colorectal; it does not say the analysis is right.

**One thing neither source paper predicts.** In colorectal, TAP1/TAP2
track hypoxia mostly *through* IFN-γ — TAP1 falls from +0.32 raw to
+0.12 once IFN-γ is partialled out. In prostate they barely move
(+0.44 → +0.44, +0.50 → +0.49), so the peptide-loading machinery tracks
hypoxia independently of IFN-γ there. That contradicts Sethumadhavan
(hypoxia downregulates TAP1/2) more strongly than the colorectal data,
and is not Estephan's IFN-γ reading either. The engine does not model
TAP, so nothing in it turns on this; it is recorded rather than dropped
for being hard to classify.

**One outcome test was null.** The single pre-specified attempt to
predict patient outcome, on the SU2C prostate cohort, came back **null
at p = 0.55**. It is reported on the capabilities page as `ABSENT`, not
buried.

**Mechanisms measured to be inert.** Reported because a model's
believability depends on knowing which parts are load-bearing:

- The lactate gate exists and is provably inert at this tissue size —
  the shipped block is too well perfused to become acidic.
- Oxygen and glucose diffusion fields are solved every step and read by
  nothing; per-cell oxygen comes from the vascular path instead.
- The lymphatic module changed nothing in any 24 h preset. Over a
  simulated week it does (14 metastases), but both its rates remain
  estimates.
- Drug penetration cannot limit anything in any shipped preset: no cell
  sits more than 44.5 µm from a capillary, against the 90–140 µm at which
  Primeau 2005 found tumour cells seeing no detectable drug. The vessel
  lattice is built so that no cell is outside a Krogh cylinder, so good
  perfusion is guaranteed by construction. Same root cause as the inert
  lactate gate.
- 73% of one agent-generated set of literature citations failed
  re-retrieval, and only 2 of 34 stored kinetic constants survived
  re-checking against their papers. The library records that.

**Test suite.** 236 test files, ~4,360 tests. The convention is to
assert the *claim* and pin the number beside it, with a guard whose
message says "re-measure, do not re-pin"; to measure under both peptide
scorers before calling anything scorer-independent; and never to widen a
tolerance to make a test pass.

---

## 6. Known limitations

Stated here so they are not discovered as surprises:

1. The simulated vasculature is denser than real tumour vasculature
   (section 5). Three mechanisms are inert because of it.
2. ~~The IFN-γ field is under-diffused.~~ **Measured and withdrawn
   2026-11-13.** The run logged a warning saying so, but the field is
   solved implicitly on every step where anything secretes IFN-γ; the
   warning fired once, on an early step before the first immune
   engagement, when the field was identically zero. Routing that case
   to the steady solve too gives an identical run to five significant
   figures. The warning no longer fires on an empty field.
3. The demo patient's HLA typing is a population default, not derived
   from their reads. The NK licensing model refuses an unmeasured typing
   on purpose rather than inventing one.
4. The T-cell receptor repertoire is generated, not sequenced.
5. No CD4 helper T cells and no lymph-node priming: T cells arrive
   already able to kill.
6. The macrophage per-contact rate's *magnitude* is unsourced, though
   the gate in front of it is measured.
7. The share of T cells that recognise the tumour (~10%, Scheper 2019)
   is from four patients in ovarian and colorectal cancer, not prostate.
8. Subclones are seeded spread evenly through the tumour; a real
   subclone is one contiguous patch.
9. No fibroblasts or stroma as cells; no nerves; no endocrine axis
   beyond the androgen pathway.
10. **The epigenetic module computes state that nothing reads.**
    Measured 2026-11-13: it tracks ~2,000 cells, costs ~2.5% of every
    engine step and takes ~31,000 random draws per step (~104 million
    over a week-long run), and
    [`get_expression_modifier()`](../cognisom/modules/epigenetic_module.py)
    — the one function by which methylation could reach transcription —
    has no caller anywhere outside the module that defines it. The
    "Avg Methylation" and "Silenced Genes" curves on the Simulation
    page move; no cell's fate depends on them. Lifting this needs a
    sourced magnitude for how much promoter methylation suppresses
    expression, not just the wire. Held by
    `tests/test_the_methods_page_matches_the_engine.py`, which fails
    the day it acquires a caller.

---

## 7. Checking any of this yourself

```bash
# the whole suite (needs the real peptide predictor, Python 3.11)
SSL_CERT_FILE=$($HOME/.venvs/mhcflurry/bin/python -c "import certifi;print(certifi.where())") \
  $HOME/.venvs/mhcflurry/bin/python -m pytest tests -q -n 4 --dist loadfile

# every kill-path number with its status and source
python -c "from cognisom.engine.py.immune.kill_calibration import calibration_report; \
           import json; print(json.dumps(calibration_report(), indent=2))"

# every diffusing signal, and whether anything reads it
python -c "from cognisom.modules.signalling import inventory; \
           [print(s.name, s.wired, s.diffusion.status) for s in inventory()]"

# one run, reproducibly
python scripts/record_demo_run.py --duration 168 --tissue section \
    --scenario Baseline --drug Enzalutamide --subclone AR_F877L=0.1 \
    --clock measured --tumour-fraction 0.735 --reactive-t-fraction 0.1
```

A run that cannot find the MHCflurry predictor falls back to a
position-weight-matrix scorer, under which **no peptide is ever a strong
binder**. That is a different biology, not a rounding difference, and
the recorder refuses to publish a recording scored that way.
