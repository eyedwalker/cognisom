"""What Cognisom simulates today, what it only partly does, and what it does not.

This is the single source for the public "what's built / what's targeted"
page (``?page=roadmap``). It is data, not prose, so it can be checked:

* every row marked BUILT names the evidence for it -- a test, or the
  analysis script that produced the result -- and
  ``tests/test_the_roadmap_cites_its_evidence.py`` fails if a BUILT row
  cites nothing, or cites a file that is not in the repository;
* every PARTIAL row says what limits it, in its note. An ABSENT row may
  carry a note, but "not modelled" is a complete statement.

So the page cannot claim more than the repository shows, and when a
capability lands the honest move is to add its test here and flip the
status -- not to edit a sentence.

The aspiration is a simulation of human tissue that responds to cancer and
to immunotherapy the way a real patient's tumour does. Rows are grouped by
biological scale, from the genome up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

BUILT = "built"
PARTIAL = "partial"
ABSENT = "absent"
STATUSES = (BUILT, PARTIAL, ABSENT)

#: What each status means on the page. "Built" is never "validated":
#: validation is its own layer below, and almost nothing is in it yet.
STATUS_MEANING = {
    BUILT: "Built and wired into runs, with a measured or published basis "
           "and a test that holds it.",
    PARTIAL: "Present but limited: an estimate, an optional feature, or a "
             "mechanism that exists but does not change outcomes yet.",
    ABSENT: "Not modelled.",
}


@dataclass(frozen=True)
class Capability:
    name: str
    status: str
    note: str = ""
    #: Repository paths that hold the claim: tests, or analysis scripts.
    evidence: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Layer:
    title: str
    question: str
    rows: Tuple[Capability, ...] = field(default_factory=tuple)


C = Capability

LAYERS: Tuple[Layer, ...] = (
    Layer(
        "Patient genome",
        "Which of this patient's mutations can the immune system see?",
        (
            C("VCF ingest and tumour/normal somatic filtering", BUILT,
              "On the HCC1395 reference callset: 1,495 variants, 29 coding, "
              "2 somatic (ARID1A, NOTCH1).",
              ("tests/test_somatic_vcf.py",
               "tests/test_the_real_callset_reaches_the_neoantigen_path.py")),
            C("Protein consequences across the whole exome", BUILT,
              "19,293 reference proteins (MANE); no consequence is invented "
              "for a variant that cannot be placed.",
              ("tests/test_exome_wide_proteome.py",
               "tests/test_frameshift_not_fabricated.py")),
            C("Neoantigen peptides from missense, frameshift and fusion "
              "variants", BUILT, "",
              ("tests/test_peptidome.py",
               "tests/test_peptidome_indel_fusion.py")),
            C("HLA typing from sequencing reads", BUILT,
              "OptiType when reads exist; otherwise a population default, "
              "shown as a default and never as the patient's type.",
              ("tests/test_optitype_runner.py",
               "tests/test_an_untyped_patient_is_not_shown_as_typed.py")),
            C("Peptide-HLA binding prediction (MHCflurry)", BUILT, "",
              ("tests/test_binding_percentile.py",
               "tests/test_binding_prediction_is_cached.py")),
            C("The patient's own neoantigens and HLA drive the simulation",
              BUILT,
              "Cancer cells present this patient's peptides on this "
              "patient's HLA; a run says whose genome it is.",
              ("tests/test_the_patient_reaches_the_simulation.py",
               "tests/test_a_run_says_whose_genome_it_is.py")),
            C("T-cell receptor repertoire", PARTIAL,
              "A seeded synthetic repertoire decides recognition; a "
              "patient's own TCR sequencing is not read.",
              ("tests/test_tcr_repertoire.py",)),
            C("Cell state from expression data", PARTIAL,
              "A marker-based classifier. No foundation model runs."),
            C("A prostate patient on prostate tissue", PARTIAL,
              "The public demo puts a breast-cancer callset (HCC1395) on "
              "prostate tissue. 338 prostate genomes (SU2C) are available "
              "but carry no HLA typing."),
        ),
    ),
    Layer(
        "Inside the cell",
        "What is each cell doing internally?",
        (
            C("Androgen-receptor signalling and PSA", BUILT,
              "A stiff kinetic model, solved implicitly.",
              ("tests/test_ar_signaling_responds_to_androgen.py",
               "tests/test_castration_reaches_psa_in_the_engine.py")),
            C("Energy metabolism, including the Warburg shift", BUILT,
              "Flux balance; glycolysis takes over as oxygen runs out.",
              ("tests/test_metabolism_is_coupled_to_the_fields.py",)),
            C("Tissue testosterone and DHT", BUILT, "",
              ("tests/test_a_castrate_vessel_leaves_a_fifth_of_the_tissue_androgen.py",)),
            C("Circadian gating of division", BUILT,
              "The division window is an estimate.",
              ("tests/test_the_clock_gates_division.py",)),
            C("Epigenetic state", PARTIAL,
              "Runs; not yet audited against data.",
              ("tests/test_epigenetic_state_lifecycle.py",)),
            C("Developmental (morphogen) signalling", PARTIAL,
              "Computes fates from signalling centres no cell in adult "
              "tissue can sense.",
              ("tests/test_circadian_and_morphogen_measure_reality.py",)),
            C("Rule-based, spatial-stochastic and hybrid solvers", PARTIAL,
              "Built and tested, but not among the modules a tissue run "
              "uses.",
              ("tests/test_bngl_rule_expansion.py",
               "tests/test_smoldyn_solver.py",
               "tests/test_ssa_correctness.py")),
            C("Protein structure and binding physics", PARTIAL,
              "Standalone tools; their output does not feed a run.",
              ("tests/test_library_structures.py",)),
        ),
    ),
    Layer(
        "Cells",
        "How does each cell live, move and die?",
        (
            C("Division at sourced relative rates", BUILT,
              "Normal to cancer turnover 7.5 : 1 (thymidine labelling, "
              "PMID 6185700). The absolute pace is compressed so a day of "
              "simulation shows divisions.",
              ("tests/test_division_inheritance.py",
               "tests/test_asynchronous_cell_cycle.py")),
            C("Death from anoxia", BUILT,
              "Below 0.01% oxygen (Papandreou 2005); hypoxia alone does "
              "not kill.",
              ("tests/test_kill_calibration.py",)),
            C("Immune-cell migration and patrol", BUILT,
              "3.3 um/min, inside the measured 2-12 um/min band.",
              ("tests/test_immune_cells_keep_patrolling.py",
               "tests/test_motility_does_not_depend_on_the_step.py")),
            C("Cell sizes", PARTIAL,
              "Six immune-cell sizes read from papers. The macrophage uses "
              "a lung measurement and the luminal cell is unsourced.",
              ("tests/test_cell_sizes_read_from_their_papers.py",)),
            C("Contact mechanics", PARTIAL,
              "Optional and off by default; the stiffnesses are unsourced.",
              ("tests/test_tissue_cells_push_apart.py",)),
            C("Programmed death from damage or drugs", ABSENT,
              "Cells die of anoxia or immune attack only."),
            C("Senescence", ABSENT, ""),
        ),
    ),
    Layer(
        "Tissue and microenvironment",
        "What surrounds the cells, and what reaches them?",
        (
            C("Real tissue architecture", BUILT,
              "Occupancy measured from a human prostate section (10x "
              "Xenium): glands pack, lumens stay empty, about five "
              "neighbours per cell. The default layout.",
              ("tests/test_tissue_section_layout.py",
               "tests/test_every_preset_runs_on_tissue.py")),
            C("Immune composition", BUILT,
              "A population prior from TCGA prostate tumours (CIBERSORT), "
              "labelled as a prior and never as the patient's.",
              ("tests/test_kill_calibration.py",)),
            C("Blood vessels and per-cell oxygen", BUILT,
              "Each cell is supplied by its nearest capillary across a "
              "Krogh distance, and burns oxygen at its own rate.",
              ("tests/test_vascular_oxygen_delivery.py",
               "tests/test_a_vessel_is_not_an_infinite_reservoir.py")),
            C("Chemokine gradients guiding immune cells", BUILT, "",
              ("tests/test_a_secreted_gradient_pulls_immune_cells.py",)),
            C("Extracellular matrix", PARTIAL,
              "It slows immune cells down, and how much is an unsourced "
              "tuning constant: a tenfold slowdown, where the nearest "
              "published measurement is threefold and measures a "
              "different quantity. Cancer cells deposit it, which is a "
              "pancreatic-cancer behaviour \u2014 in prostate, stroma is "
              "no greater in tumour than in normal tissue (p = 0.3).",
              ("tests/test_ecm_barrier.py",
               "tests/test_prostate_is_mostly_stroma_and_the_engine_has_none.py")),
            C("Stroma: fibroblasts, smooth muscle, the matrix they make",
              ABSENT,
              "Measured on human prostatectomy sections, stroma is "
              "54.8% of normal prostate tissue and epithelium only "
              "31.4%. The engine models the epithelium and the immune "
              "infiltrate, so slightly over half of real prostate is "
              "absent \u2014 and it is the half that decides whether "
              "immune cells and drugs can reach a tumour at all.",
              ("tests/test_prostate_is_mostly_stroma_and_the_engine_has_none.py",)),
            C("Oxygen and glucose diffusion fields", PARTIAL,
              "Solved, but nothing reads them: per-cell values come from "
              "the vessels instead.",
              ("tests/test_two_spatial_fields_have_no_consumer.py",)),
            C("Tumour acidity (lactate)", PARTIAL,
              "The mechanism exists and has no effect in the shipped "
              "scenarios.",
              ("tests/test_tumour_acid_blunts_the_killers.py",)),
            C("Lymphatic drainage", PARTIAL,
              "Runs, changes no outcome yet; its rates are estimates.",
              ("tests/test_a_cell_that_enters_a_lymphatic_leaves_the_tissue.py",)),
            C("Cancer-associated fibroblasts", ABSENT,
              "A distinct activated state of the absent stromal "
              "compartment above."),
            C("New blood vessels (angiogenesis)", ABSENT, ""),
            C("Whole-tissue composition (stroma versus epithelium)", ABSENT,
              "Single-cell RNA atlases cannot supply it; it needs imaging.",
              ("tests/test_an_atlas_cannot_say_what_a_prostate_is_made_of.py",)),
        ),
    ),
    Layer(
        "Immune system",
        "Who attacks the tumour, and what holds them back?",
        (
            C("CD8 T-cell killing", BUILT,
              "Perforin damage that adds up across contacts (Weigelin "
              "2021) rather than one lethal roll.",
              ("tests/test_tcell_kill.py", "tests/test_kill_calibration.py")),
            C("NK cells reading MHC class I (missing self)", BUILT,
              "Graded, not a threshold; the per-contact rate is an "
              "estimate.",
              ("tests/test_nk_reads_mhc1_without_a_cliff.py",)),
            C("Macrophages and M1/M2 polarisation", BUILT, "",
              ("tests/test_macrophage_polarization.py",)),
            C("T-cell exhaustion", BUILT,
              "A clock from first engagement; 7-14 days is an estimate.",
              ("tests/test_exhaustion.py",)),
            C("Interferon-gamma raising MHC class I", BUILT, "",
              ("tests/test_interferon_reaches_the_bystanders.py",)),
            C("PD-1 / PD-L1 biology", BUILT,
              "Blockade relieves CD28 inhibition (Hui 2017); tumour PD-L1 "
              "acts through myeloid cells only (Klement 2023).",
              ("tests/test_blockade_amplifies_recognition.py",)),
            C("Suppression by regulatory T cells and arginase", BUILT, "",
              ("tests/test_suppressors_act_where_there_is_room.py",)),
            C("Regulatory T cells and MDSCs as cells", PARTIAL,
              "Regulatory T cells act as a fraction, not as cells. MDSCs "
              "were declined: their mechanism shares an already saturated "
              "gate."),
            C("Recruitment from blood", PARTIAL,
              "Immune cells are recruited as the tumour grows; there is no "
              "circulation."),
            C("T-cell clonal expansion after recognition", PARTIAL,
              "A T cell that recognises its antigen now divides, on the "
              "programme its first recognition starts \u2014 it continues "
              "without further antigen, as measured. Off by default: both "
              "numbers (6 h per division, 7 divisions) are estimates from "
              "mouse lymph nodes in acute infection, and no per-cell rate "
              "inside a tumour has been published.",
              ("tests/test_a_t_cell_that_recognises_divides.py",
               "cognisom/engine/py/immune/tcell_expansion.py")),
            C("Dendritic cells, antigen presentation and priming", ABSENT,
              "T cells start already primed, so the start of a response "
              "is not simulated."),
            C("CD4 helper T cells, B cells, antibodies, neutrophils", ABSENT,
              ""),
            C("Immunological memory", ABSENT, ""),
        ),
    ),
    Layer(
        "Tumour evolution",
        "How does the tumour change under pressure?",
        (
            C("Heritable immune escape", BUILT,
              "Loss of HLA / MHC class I at 1.55e-6 per division (five "
              "published measurements), and PD-L1 gain.",
              ("tests/test_escape_can_be_inherited.py",)),
            C("Immunoediting, measured against a drift control", BUILT,
              "Finding: T cells select for escape, but immune cells that "
              "cannot read MHC class I do most of the killing and dilute "
              "the pressure.",
              ("tests/test_escape_is_selected_not_imposed.py",)),
            C("New driver mutations and clonal expansion", PARTIAL,
              "Only transformation during a run."),
            C("Therapy resistance (for example AR amplification)", ABSENT,
              ""),
        ),
    ),
    Layer(
        "Treatment",
        "What happens when the patient is treated?",
        (
            C("Anti-PD-1 checkpoint blockade, cell by cell", BUILT, "",
              ("tests/test_blockade_amplifies_recognition.py",)),
            C("Castration and enzalutamide, cell by cell", BUILT, "",
              ("tests/test_castration_slows_and_kills_the_dependent_tumour.py",
               "tests/test_an_antiandrogen_acts_downstream.py")),
            C("Patient outcome model for nine regimens", PARTIAL,
              "A tumour-size model calibrated to one published control arm "
              "(PROfound, 3.6 months rPFS). It does not call the cell-level "
              "simulation.",
              ("tests/test_treatment_predictions_are_honest.py",
               "tests/test_trial_constants_match_publications.py")),
            C("Drug washout at label half-life, and combinations", PARTIAL,
              "In the outcome model only.",
              ("tests/test_treatment_pk_from_library.py",
               "tests/test_combination_washout.py")),
            C("Chemotherapy, radiation, cell therapy and vaccines, cell by "
              "cell", ABSENT, ""),
            C("Drug delivery into tissue (dose to local concentration)",
              ABSENT, ""),
        ),
    ),
    Layer(
        "Whole body",
        "What happens beyond the tissue?",
        (
            C("Serum hormones as an input", BUILT, "Testosterone.",
              ("tests/test_castration_reaches_psa_in_the_engine.py",)),
            C("Metastasis", PARTIAL, "Nominal; no metastases form."),
            C("Circulation, lymph nodes, spleen, bone marrow", ABSENT, ""),
            C("Systemic cytokines and immune-related side effects", ABSENT,
              ""),
        ),
    ),
    Layer(
        "Scale and time",
        "How much tissue, for how long?",
        (
            C("Fast engine with exact checkpoint and restart", BUILT,
              "Cost grows about linearly with cells; roughly two seconds "
              "per simulated hour at 2,000 cells.",
              ("tests/test_a_run_can_be_resumed.py",)),
            C("Tissue size", PARTIAL,
              "About 2,000 cells, a quarter of a millimetre across. A 1 cm3 "
              "tumour holds about a billion."),
            C("Simulated duration", PARTIAL,
              "Typical runs cover a day; clinical responses take months."),
            C("Multi-GPU tissue engine", PARTIAL,
              "Built, but has no immune system yet."),
        ),
    ),
    Layer(
        "Validation",
        "Does it predict anything?",
        (
            C("Hypoxia and MHC class I, against human tumours", BUILT,
              "A falsifiable prediction held on 592 TCGA colorectal "
              "tumours. The only external test passed so far.",
              ("cognisom/validation/hypoxia_mhc1_tcga.py",)),
            C("Every kill-path number carries its source or an estimate "
              "label", BUILT, "",
              ("tests/test_kill_calibration.py",
               "tests/test_parameter_provenance.py")),
            C("Kinetic constants in the parameter library", PARTIAL,
              "Only 2 of 34 survived re-checking against their papers."),
            C("Predicting a patient's outcome", ABSENT,
              "The one pre-specified outcome test (SU2C prostate cohort) "
              "was null, p = 0.55."),
        ),
    ),
    Layer(
        "Platform",
        "Can people see and use it?",
        (
            C("Web app with patient and tissue choices", BUILT, "",
              ("tests/test_dashboard_pages_are_loadable.py",)),
            C("Public recorded demo, plotted and ray-traced", BUILT,
              "Both videos are one run.",
              ("tests/test_the_two_demo_videos_are_one_run.py",
               "tests/test_media_is_actually_served.py")),
            C("Bio-USD, an open standard for cells in OpenUSD", BUILT,
              "Version 0.2.0, archived with a DOI.",
              ("tests/test_biousd_is_publishable.py",
               "tests/test_biousd_validators.py")),
            C("Live ray-traced viewer", PARTIAL,
              "Works; the web app does not link to it yet.",
              ("tests/test_the_rtx_streamer_can_open_a_run.py",)),
        ),
    ),
)

#: The largest distances between the platform and the aspiration. ORDERED,
#: with what each one needs before it can start, because a list of five
#: goals with no sequencing is an inventory and not a plan.
TARGETS: Tuple[Tuple[str, str], ...] = (
    ("Treatments, cell by cell",
     "Chemotherapy, radiation, cell therapy and vaccines, and above all a "
     "drug concentration: today a treatment is a dimensionless 0-1 knob, "
     "so the engine cannot answer a question about dose or schedule. "
     "NEEDS NOTHING NEW \u2014 the label half-lives are already sourced and "
     "tested, and the vessel geometry a drug would diffuse from already "
     "exists. This is first because it is unblocked."),
    ("The tumour's surroundings",
     "A stromal compartment: fibroblasts, smooth muscle and the matrix "
     "they make. Measured on human sections, stroma is 54.8% of prostate "
     "tissue and the engine has none of it, which is the largest single "
     "piece of missing biology. Independent of everything else here, and "
     "second because it changes every contact-mediated number in the "
     "model, so it is better done before more numbers are pinned."),
    ("The start and growth of an immune response",
     "T cells that multiply when they recognise their target (now built, "
     "off by default), and the dendritic cells and lymph-node priming "
     "that start a response at all. NEEDS A NEW COMPARTMENT outside the "
     "tissue, plus trafficking into and out of it, which is why it is "
     "third despite being the deepest gap: vaccines act on exactly this."),
    ("Scale and time",
     "From thousands of cells over a day to millions over months. This is "
     "an ALGORITHM problem before it is a hardware one: the engine is "
     "already near-linear in cell count, the one super-linear term is the "
     "immune target scan, and a spatial index for it exists in the "
     "codebase with no callers. The oxygen grid is a separate, genuinely "
     "hard problem \u2014 a capillary is one voxel wide, so a coarser grid "
     "is not the fix. Cuts across the three above."),
    ("Validation",
     "Predictions tested against held-out patient data rather than the "
     "literature the model was built from. Gated on the others: there is "
     "little point testing an outcome prediction while half the tissue "
     "and all of the drug pharmacology are missing."),
)

#: Answering "is scale a GPU problem, an algorithm problem or a biology
#: problem?" with measurements rather than an impression, because it is
#: the first question a technical reader asks and the page used to state
#: only the gap.
SCALE_NOTE = (
    "**Where the scale problem actually is.** Measured, not estimated: the "
    "engine costs about 2.1 seconds per simulated hour at 2,000 cells and "
    "grows very nearly linearly with cell count, so the arithmetic to a "
    "million cells is not the obstacle. Two specific things are. The "
    "immune system's search for targets is the one part that grows faster "
    "than linearly, and a spatial index that would fix it already exists "
    "in this codebase with nothing calling it. And the oxygen diffusion "
    "grid is a genuinely hard problem rather than a slow one \u2014 a "
    "capillary is a single voxel wide, so a coarser grid does not "
    "approximate it, it deletes it. Neither is a question of buying a "
    "bigger GPU. Time is the harder half: a clinical response takes "
    "months, and nothing in the model yet argues that a month of "
    "simulated tissue behaves like a month of a patient."
)

#: The patient-outcome model is shaped like a product and is NOT the
#: cell-level engine, so it gets its own statement rather than one row in
#: a table where it reads as if the two were one thing.
OUTCOME_MODEL_NOTE = (
    "**A second, separate model \u2014 read this before quoting any survival "
    "number.** Alongside the cell-by-cell simulation there is a "
    "patient-level outcome model covering nine treatment regimens. It is "
    "a tumour-size equation calibrated to a single published control arm "
    "(PROfound, 3.6 months median radiographic progression-free "
    "survival), it carries each drug's label half-life so a stopped "
    "course washes out, and it handles combinations. It **does not call "
    "the cell-level engine at all.** Nothing a simulated tumour does "
    "reaches it, and nothing it predicts is derived from this patient's "
    "genome beyond the regimen choice. Connecting the two is the point of "
    "the treatment work above; until then they are two models under one "
    "name, and only the cell-level one is what the rest of this page "
    "describes."
)


def counts() -> dict:
    """How many rows have each status, for the page's summary line."""
    out = {s: 0 for s in STATUSES}
    for layer in LAYERS:
        for row in layer.rows:
            out[row.status] += 1
    return out


def all_rows() -> List[Tuple[Layer, Capability]]:
    return [(layer, row) for layer in LAYERS for row in layer.rows]
