"""What the engine computes each step, and what each number rests on.

The single source for the public methods page (``?page=methods``), and
the companion to ``docs/simulation_methods.md``: the prose lives in the
document, the CHECKABLE parts live here.

Same discipline as ``capabilities.py``. This is data, not prose, so a
test can hold it to the repository:

* every module row names the source file that computes it, and
  ``tests/test_the_methods_page_matches_the_engine.py`` fails if a file
  is missing, or if a module an ordinary run enables has no row here;
* the signal table and the kill table are READ from
  ``modules/signalling.py`` and ``engine/py/immune/kill_calibration.py``
  rather than retyped, so a number cannot be right here and wrong there;
* the validation rows are read from ``capabilities.py``, so the methods
  page and the roadmap page cannot disagree about what has been tested;
* every limitation names what would have to change to lift it, because
  "known limitation" with no remedy is an excuse rather than a plan.

A scientist's first question is not what the model does, it is which
parts of it are load-bearing and which are decoration. The page is
ordered to answer that: what is computed, which of it changes a run,
what each number rests on, what has been tested against something
outside the model, and what is known to be wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

#: Where the narrative lives. Rendered by the page; the path is checked
#: by the test, so a moved file breaks the build and not the reader.
NARRATIVE = "docs/simulation_methods.md"


# ── What the engine runs ─────────────────────────────────────────────

#: A module whose output changes what some cell does in a default run.
CHANGES = "changes a run"
#: A module that is enabled, costs time, and reports numbers that
#: nothing reads back into a cell's fate. Not a bug to hide: a reader
#: looking at a dashboard plot deserves to know which curves are inert.
REPORTS_ONLY = "reported only"
#: A module registered but off in a default run.
OFF = "off by default"

WIRING = (CHANGES, REPORTS_ONLY, OFF)

WIRING_MEANING = {
    CHANGES: "Its output is read by another module and changes what a "
             "cell does. Turning it off changes the result.",
    REPORTS_ONLY: "It runs and reports numbers, and nothing reads them "
                  "back into any cell's fate. Its plots move; the "
                  "simulation does not.",
    OFF: "Registered but not enabled in a default run.",
}


@dataclass(frozen=True)
class ModuleRow:
    """One engine module, and the file that computes it."""

    key: str            # the name the engine registers it under
    title: str          # what to call it on the page
    computes: str       # what it works out each step, in plain words
    source: str         # repository path
    wiring: str         # one of WIRING
    #: The quantity a reader is most likely to want to check, and how it
    #: is obtained. Empty where the module is plumbing.
    worked_example: str = ""
    #: What to distrust about it. Empty is a claim, so leave it empty
    #: only where there is nothing to say.
    caveat: str = ""

    def __post_init__(self) -> None:
        if self.wiring not in WIRING:
            raise ValueError(f"unknown wiring {self.wiring!r}")


#: Every module the engine can register, apart from the four solver
#: back-ends (ODE, BNGL, hybrid, Smoldyn), which are alternative
#: integrators rather than biology. The test asserts that every module
#: an ordinary run ENABLES has a row here, so a new one cannot ship
#: undescribed.
MODULES: Tuple[ModuleRow, ...] = (
    ModuleRow(
        "cellular", "Cells",
        "The cell population: division, death, state, mutations, and "
        "what a daughter inherits.",
        "cognisom/modules/cellular_module.py", CHANGES,
        "Division needs two things at once: the cell's own timer "
        "elapsed, and its circadian clock in the permissive window. "
        "Androgen-receptor signalling does not gate it -- it STRETCHES "
        "the timer. Below a dead band of 0.95 the interval is divided "
        "by the signalling fraction, so a cell at half signalling takes "
        "twice as long to divide and one at 0.05 takes twenty times as "
        "long. The coupling is one-sided on purpose: above the dead "
        "band nothing speeds up, which is both the measured biology and "
        "why a Baseline run is unchanged by the mechanism existing. "
        "That is how androgen deprivation slows growth here without "
        "killing outright.",
        "Cell cycle is a timer with a gate, not a modelled "
        "G1/S/G2/M with checkpoints."),
    ModuleRow(
        "vascular", "Blood supply",
        "The capillary network, and how much oxygen and glucose each "
        "cell receives.",
        "cognisom/modules/vascular_module.py", CHANGES,
        "A cell's oxygen is NOT read from the diffusion field. It comes "
        "from the Krogh-cylinder steady state at that cell's distance "
        "to its nearest capillary, from a vessel treated as a finite "
        "reservoir rather than an infinite source. The field exists for "
        "the molecules that genuinely are transport-limited.",
        "At the default 22 um vessel pitch no cell is hypoxic and none "
        "sits more than ~16 um from a capillary, which is better "
        "perfused than any real tumour. The pitch is a run parameter "
        "and a sparser lattice does produce hypoxia; no shipped preset "
        "asks for one. See the limitations below."),
    ModuleRow(
        "spatial", "Diffusing signals",
        "The concentration fields, and which cells can sense them.",
        "cognisom/modules/spatial_module.py", CHANGES,
        "A field that equilibrates within one step is solved implicitly, "
        "by LU factorisation of the steady-state operator, and the "
        "result is REFUSED if it violates the maximum principle that "
        "operator is supposed to guarantee. Fields that do not "
        "equilibrate take explicit sub-steps under a CFL bound.",
        "Six fields are solved; three are read by anything. The signal "
        "table below says which."),
    ModuleRow(
        "immune", "Immune cells",
        "T cell, NK cell and macrophage patrol, recognition, contact "
        "and killing.",
        "cognisom/modules/immune_module.py", CHANGES,
        "Each lineage kills by a different measured rule, and the kill "
        "table below gives every number. An immune cell scans before it "
        "moves, which is what lets it find a target rather than sit "
        "beside one, and it biases its walk up the chemokine gradient.",
        "By default every T cell is treated as tumour-reactive, which "
        "is almost certainly too generous; see the limitations."),
    ModuleRow(
        "ar_signaling", "Androgen receptor and PSA",
        "Each cell's androgen-receptor pathway and its PSA output, and "
        "what a drug does to it.",
        "cognisom/modules/ar_signaling_module.py", CHANGES,
        "A stiff ODE per cell, integrated with an implicit BDF method "
        "in one batched solve per step -- explicit solvers fabricate "
        "mass on this system. A drug is resolved against THIS cell's "
        "receptor, so the same dose blocks a wild-type cell and acts as "
        "an agonist on one carrying AR F877L.",
        "Drug exposure is uniform unless a run asks for a penetration "
        "distance, and on the shipped vessel lattice even then almost "
        "every cell is well perfused."),
    ModuleRow(
        "molecular", "DNA, RNA and peptides",
        "Transcription, translation, mutation, and the peptides a "
        "cancer cell puts on MHC class I.",
        "cognisom/modules/molecular_module.py", CHANGES,
        "This is the path from the patient's VCF to a kill: a mutation "
        "changes a protein sequence, the sequence is cut into peptides, "
        "the peptides are scored against the patient's HLA alleles, and "
        "what binds is displayed for a T cell to recognise.",
        "Binding is predicted by MHCflurry where it is installed, and "
        "falls back to a position-weight matrix where it is not -- a "
        "much weaker scorer. Runs state which one they used."),
    ModuleRow(
        "circadian", "The clock",
        "Each cell's clock phase, which gates division.",
        "cognisom/modules/circadian_module.py", CHANGES,
        "Division is gated on the clock (Matsuo 2003, PMID 12934012). "
        "Only the gate is wired: the module also computes a metabolic "
        "and an immune modifier, and both are unsourced magnitudes left "
        "as documented hooks rather than switched on.",
        "The permissive window is an ESTIMATE, and it moves the "
        "baseline division count more than any other single choice "
        "here."),
    ModuleRow(
        "morphogen", "Chemokine secretion",
        "Which cells secrete the chemokine that recruits immune cells, "
        "and at what rate.",
        "cognisom/modules/morphogen_module.py", CHANGES,
        "Cancer cells are sources on the spatial module's cytokine "
        "field; immune cells read its gradient during patrol. This is "
        "the CXCL12/CXCR4 class of mechanism written with the cell "
        "types this engine has.",
        "The secretion rate is an ESTIMATE and the field is therefore "
        "dimensionless -- no usable per-cell CXCL12 rate was found. "
        "What the consumer uses is the gradient's shape, not its "
        "units."),
    ModuleRow(
        "lymphatic", "Lymphatics",
        "Lymphatic vessels, intravasation, and a metastasis counter.",
        "cognisom/modules/lymphatic_module.py", CHANGES,
        "A cancer cell that meets a lymphatic can leave the tissue. "
        "Measured over 48 h on a ~1,500-cell tumour, this removes 2 "
        "cells under enzalutamide and 5 untreated -- a real sink, and "
        "a small one beside the 175 and 26 deaths from other causes in "
        "those same two arms.",
        "Both of its rates are ESTIMATES, and the metastasis counter is "
        "a mechanism demonstration, not a prediction. Do not read "
        "'Metastasis Risk' as a risk."),
    ModuleRow(
        "epigenetic", "Methylation and histone marks",
        "Per-cell methylation and histone state for five genes, "
        "inherited at division.",
        "cognisom/modules/epigenetic_module.py", REPORTS_ONLY,
        "",
        "MEASURED, 2026-11-13: it tracks ~2,000 cells, costs about 2.5% "
        "of every engine step and takes ~31,000 random draws per step "
        "(~104 million over a week-long run), and "
        "get_expression_modifier() -- the one function by which "
        "methylation could change transcription -- has no caller "
        "anywhere outside this module. The 'Avg Methylation' and "
        "'Silenced Genes' curves on the Simulation page move; no cell's "
        "fate depends on them."),
    ModuleRow(
        "cell_mechanics", "Contact and adhesion",
        "Physical forces between neighbouring tissue cells.",
        "cognisom/modules/cell_mechanics_module.py", OFF,
        "",
        "Off by default on purpose: its stiffnesses have nothing in the "
        "parameter library behind them, so switching it on would move "
        "every cell in every preset on a guess."),
)


def modules_by_wiring(wiring: str) -> List[ModuleRow]:
    return [m for m in MODULES if m.wiring == wiring]


# ── What is known to be wrong ────────────────────────────────────────

@dataclass(frozen=True)
class Limitation:
    """Something known to be wrong or missing, and what would fix it."""

    what: str
    why_it_matters: str
    #: What would have to change. Never "more work".
    to_lift_it: str


LIMITATIONS: Tuple[Limitation, ...] = (
    Limitation(
        "Every shipped preset is better perfused than a real tumour.",
        "At the default 22 um vessel pitch no cell sits more than about "
        "16 um from a capillary and none is hypoxic, against the "
        "90-140 um at which Primeau 2005 found tumour cells seeing no "
        "detectable doxorubicin. Three mechanisms are therefore nearly "
        "inert in every preset: drug penetration, the lactate gate, and "
        "most hypoxia. This is a DEFAULT, not a missing capability -- "
        "the vessel pitch is already a run parameter, and measured on "
        "the section layout a 60 um pitch puts the worst cell 46 um "
        "from a vessel and a 180 um pitch puts it at 113-143 um across "
        "five seeds, inside Primeau's band.",
        "Not new code, but two things before trusting such a run. The "
        "Krogh calibration was anchored at the 22 um pitch, where every "
        "cell sits inside a Krogh cylinder; a sparse lattice runs it "
        "exactly where it was never checked. And the vessel jitter is a "
        "FRACTION of the pitch, so the worst distance varies by 1 um "
        "across seeds at the default and by 60 um at a 400 um pitch -- "
        "past about 180 um which cells starve depends more on the seed "
        "than on the pitch, so a sparse preset has to be reported "
        "across seeds rather than quoted from one."),
    Limitation(
        "Methylation is computed and read by nothing.",
        "The epigenetic module tracks five genes in every cell and "
        "plots two curves, and no cell's behaviour depends on either. A "
        "reader watching those curves move would reasonably assume they "
        "are driving something.",
        "Wire get_expression_modifier() into transcription in the "
        "molecular module -- which needs a sourced magnitude for how "
        "much promoter methylation suppresses expression, not just the "
        "wire."),
    Limitation(
        "The demo patient's HLA typing is a population default.",
        "It is not derived from that patient's own reads, so the "
        "neoantigen results are conditioned on an assumption about them "
        "rather than a measurement of them.",
        "Type the HLA from the patient's sequencing. The NK licensing "
        "model already refuses an unmeasured typing rather than "
        "inventing one, so that half is waiting on the data."),
    Limitation(
        "The T-cell receptor repertoire is generated, not sequenced.",
        "Which clone recognises which peptide is a property of the "
        "generator, not of the patient.",
        "Ingest real TCR sequencing and match against it."),
    Limitation(
        "Every T cell is tumour-reactive by default.",
        "Scheper 2019 measured about 10% in ovarian and colorectal "
        "tumours, and found none at all in two of four patients. The "
        "engine's default is certainly wrong in the generous direction. "
        "The number is opt-in rather than applied silently, and the "
        "recorded demo does set it to 10% -- so the demo is not running "
        "on the generous default, but another run might.",
        "A prostate measurement, or a sensitivity analysis across the "
        "range that paper allows."),
    Limitation(
        "No helper T cells and no lymph-node priming.",
        "T cells arrive already able to kill, so the simulation cannot "
        "represent a response that fails because it was never primed.",
        "A lymph node compartment with antigen presentation, and "
        "trafficking in and out of the tissue."),
    Limitation(
        "No fibroblasts, stroma, nerves, or endocrine axis beyond "
        "androgen.",
        "The cells that make up much of a real tumour's mass are "
        "absent, so anything they would do -- stiffening the matrix, "
        "suppressing an immune response, feeding a tumour cell -- is "
        "absent with them.",
        "Each as a cell type with sourced parameters, in the order the "
        "evidence allows. The signalling module exists so that adding "
        "one is a declaration rather than four edits in two files."),
    Limitation(
        "Subclones are spread evenly through the tumour.",
        "A real resistant subclone is usually one contiguous patch, "
        "which changes how quickly immune cells and drug can reach it.",
        "Seed the subclone as a contiguous region and measure how much "
        "the geometry matters."),
    Limitation(
        "The macrophage per-contact rate's magnitude is unsourced.",
        "The gate in front of it is measured -- a macrophage eats "
        "nothing that is not both opsonised and stripped of CD47 "
        "protection (Dooling 2023) -- but how often it succeeds once "
        "licensed is an estimate. In every shipped preset the gate is "
        "shut, so macrophages kill nothing at all.",
        "A measured per-contact phagocytosis fraction against a solid "
        "tumour cell. Searches have not found one."),
    Limitation(
        "A day to a week of simulated time, against months of clinical "
        "response.",
        "The longest shipped run is a simulated week. Resistance, "
        "castration escape and radiographic response all play out over "
        "months, so nothing here can be compared to a clinical endpoint "
        "directly.",
        "Longer runs, which cost roughly 2.1 seconds per simulated hour "
        "at 2,000 cells -- or a coarser representation above the cell "
        "for the slow part."),
)


# ── Tables read from the engine, never retyped ───────────────────────

def signal_rows() -> List[Dict[str, object]]:
    """The diffusing signals, READ from the signalling inventory.

    "Changes a run" is the column to read first: no means the engine
    solves this field every step and nothing in a default run consumes
    it.
    """
    from cognisom.modules.signalling import inventory

    rows = []
    for entry in inventory():
        rows.append({
            "Signal": entry.molecule,
            "Field": entry.name,
            "Made by": entry.made_by,
            "Read by": entry.read_by,
            "Changes a run": "yes" if entry.wired else "no",
            "Transport": f"{entry.diffusion.value:g} {entry.diffusion.unit}",
            "Status": entry.diffusion.status,
        })
    return rows


def kill_rows() -> List[Dict[str, object]]:
    """Every number on the kill path, READ from kill_calibration."""
    from cognisom.engine.py.immune.kill_calibration import calibration_report

    rows = []
    for key, row in sorted(calibration_report().items()):
        low, high = row["range"]
        rows.append({
            "Parameter": key.replace("_", " "),
            "Value": f"{row['value']:g}",
            "Plausible range": f"{low:g} - {high:g}",
            "Status": row["status"],
            "Source": row["source"],
        })
    return rows


def status_counts() -> Dict[str, int]:
    """How many kill-path numbers sit at each provenance status."""
    counts: Dict[str, int] = {}
    for row in kill_rows():
        key = str(row["Status"])
        counts[key] = counts.get(key, 0) + 1
    return counts


def validation_rows() -> List[Dict[str, str]]:
    """What has been tested against something outside the model.

    Read from ``capabilities.py`` so the methods page and the roadmap
    page cannot disagree about what has been validated.
    """
    from cognisom.dashboard.capabilities import LAYERS

    rows = []
    for layer in LAYERS:
        if layer.title != "Validation":
            continue
        for cap in layer.rows:
            rows.append({
                "Claim": cap.name,
                "Status": cap.status,
                "What it rests on": cap.note or "—",
                "Evidence": tuple(cap.evidence),
            })
    return rows


def cited_paths() -> List[str]:
    """Every repository path this module names, for the test to check."""
    paths = [row.source for row in MODULES] + [NARRATIVE]
    for row in validation_rows():
        paths.extend(row["Evidence"])
    return paths
