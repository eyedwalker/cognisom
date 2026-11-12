"""The cells the engine simulates: size, role, and how each is modelled.

This is the data behind the public "Cells" page (``?page=cells``). It is a
module rather than prose in the page so that the numbers a visitor reads
are the numbers the engine runs:

* diameters, drawn shapes and colours come from
  ``visualization.lineage_appearance``, the one resolver the renderer uses;
* kill-path numbers come from ``engine.py.immune.kill_calibration``, each
  with its own status and source;
* contact durations, granule reserves and patrol speeds are read off a
  constructed ``ImmuneModule``, with the provenance note that module
  records for each;
* division periods, death rates and default populations come from the
  cellular module and the scenario presets.

Only the two paragraphs of plain description per cell are written here.
``tests/test_the_cell_page_matches_the_engine.py`` holds the page to that:
it re-reads the sources and fails if a number here is a second copy.

Every parameter row carries a STATUS, and the vocabulary is deliberately
blunt. A visitor should be able to tell a measurement from a guess without
reading a footnote.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

#: Status vocabulary, most to least trustworthy, with what each means.
STATUS_MEANING: Dict[str, str] = {
    "measured": "Taken from a published measurement, cited.",
    "derived": "Computed from published measurements, cited.",
    "estimate": "A judgement inside a published range, or with no "
                "measurement behind its magnitude. Labelled as such in "
                "the code.",
    "model choice": "A construct of the model with no biological "
                    "counterpart to measure (a sensing radius, a "
                    "compressed clock). Chosen, not sourced.",
    "placeholder": "No usable value yet; a default stands in.",
}

STATUS_ORDER: Tuple[str, ...] = tuple(STATUS_MEANING)


@dataclass(frozen=True)
class Parameter:
    """One number the engine uses for a cell, and where it came from."""

    name: str
    value: str
    status: str
    source: str

    def __post_init__(self) -> None:
        if self.status not in STATUS_MEANING:
            raise ValueError(f"unknown status {self.status!r} on {self.name!r}")


@dataclass(frozen=True)
class CellEntry:
    """One simulated cell type."""

    key: str                      # the engine's cell_type token
    name: str                     # what it is drawn and labelled as
    diameter_um: float
    diameter_status: str
    diameter_source: str
    shape: str
    colour: Tuple[float, float, float]
    #: Cells of this type at the start of the default (Baseline, section)
    #: run, and what share of the tissue that is.
    count: int
    does: str                     # what the cell does in the body
    simulated: str                # how the engine represents that
    parameters: List[Parameter] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)


#: Things a reader may expect to find as cells and will not.
NOT_CELLS: List[Tuple[str, str]] = [
    ("Blood vessels (endothelial cells)",
     "Capillaries are line segments that deliver oxygen and let immune "
     "cells in. No individual endothelial cell exists."),
    ("Fibroblasts and the matrix they lay down",
     "A density field that slows immune cells and shortens how far they "
     "can sense. No individual fibroblast exists."),
    ("Regulatory T cells",
     "A single fraction that weakens T-cell killing. They are not "
     "separate cells that move or make contact."),
    ("Basal and neuroendocrine prostate cells",
     "Not represented. Every non-cancer tissue cell is one 'normal' type, "
     "drawn as a luminal cell."),
    ("B cells, dendritic cells, neutrophils, MDSCs",
     "Not represented."),
]


#: What a cell is called when the entity library has no entry for it.
PLAIN_NAMES: Dict[str, str] = {
    "normal": "Normal prostate cell",
    "cancer": "Prostate cancer cell",
    "T_cell": "CD8 T cell",
    "NK_cell": "NK cell",
    "macrophage": "Macrophage",
}

NO_ENTITY_NOTE = (
    "The entity library on this server has no entry for this cell, so "
    "the size and shape shown are the engine's defaults, not sourced "
    "values.")


def _name(looks: Dict[str, Dict[str, object]], key: str) -> str:
    return str(looks[key].get("entity") or PLAIN_NAMES[key])


def _without_entity(entry: "CellEntry") -> "CellEntry":
    from dataclasses import replace
    return replace(entry, missing=[NO_ENTITY_NOTE, *entry.missing])


def _pmid(note: str) -> str:
    """'radius_um PMID:123' or 'x: measured:PMID:123' -> 'PMID 123'."""
    if "PMID:" not in note:
        return note
    return "PMID " + note.split("PMID:")[-1].strip()


def _size(appearance: Dict[str, object]) -> Tuple[float, str, str]:
    diameter = float(appearance["diameter_um"])
    if appearance["cited"]:
        return diameter, "measured", _pmid(str(appearance["size_source"]))
    return diameter, "placeholder", "No sourced width; a library default."


def _contact_status(note: str) -> str:
    for word in ("measured", "derived"):
        if f"{word}:" in note:
            return word
    if "estimated:" in note:
        return "estimate"
    return "placeholder"


def _kill_status(status: str) -> str:
    return {"sourced": "measured", "derived": "derived",
            "ESTIMATE": "estimate",
            "calibration-choice": "model choice"}[status]


def build_catalog() -> List[CellEntry]:
    """Assemble the catalogue from the modules the engine runs.

    Constructs one ``ImmuneModule`` (no simulation is started and no
    patient is read) because that module resolves contact durations,
    granule reserves and patrol speeds from the entity library at
    construction and records where each came from.
    """
    from cognisom.dashboard.engine_runner import SCENARIOS, section_variant
    from cognisom.engine.py.immune import kill_calibration as kc
    from cognisom.modules import cellular_module as cm
    from cognisom.modules.immune_module import ImmuneCell, ImmuneModule
    from cognisom.visualization import lineage_appearance as la

    looks = la.appearance_table()
    with contextlib.redirect_stdout(io.StringIO()):
        immune = ImmuneModule({})
        cellular = cm.CellularModule({})
    preset = section_variant(SCENARIOS["Baseline"])
    n_normal = int(preset["cellular"]["n_normal_cells"])
    n_cancer = int(preset["cellular"]["n_cancer_cells"])
    n_t = int(preset["immune"]["n_t_cells"])
    n_nk = int(preset["immune"]["n_nk_cells"])
    n_mac = int(preset["immune"]["n_macrophages"])

    fields = ImmuneCell.__dataclass_fields__
    sensing = float(fields["detection_radius"].default)
    reach = float(fields["kill_radius"].default)
    chase = float(fields["speed"].default)
    calib = kc.calibration_report()

    def contact(lineage: str, lethal: bool) -> Parameter:
        key = "lethal_contact_minutes" if lethal else "nonlethal_contact_minutes"
        minutes = immune._contact_minutes_for(lineage, lethal)
        note = immune.contact_time_source[lineage][key]
        status = _contact_status(note)
        return Parameter(
            "Contact that kills, mean" if lethal
            else "Contact that does not kill, mean",
            f"{minutes:g} min", status,
            _pmid(note) if status != "placeholder"
            else "Illustrative default; nothing in the library.")

    def movement(lineage: str) -> List[Parameter]:
        note = immune.patrol_speed_source.get(lineage, "")
        speed = immune._patrol_speed_for(lineage)
        sourced = "placeholder" not in note and "PMID" in note
        return [
            Parameter("Patrol speed (searching)", f"{speed:g} µm/min",
                      "measured" if sourced else "placeholder",
                      _pmid(note) if sourced else
                      "The library row is marked 'not evidence'; the "
                      "module default applies."),
            Parameter("Chase speed (locked on a target)",
                      f"{chase:g} µm/min", "model choice",
                      "Hard-coded; not sourced."),
            Parameter("Sensing radius", f"{sensing:g} µm", "model choice",
                      "About one cell diameter: a cell senses what it "
                      "touches."),
            Parameter("Kill reach", f"{reach:g} µm", "model choice",
                      "Hard-coded; not sourced."),
        ]

    def kill(key: str, name: str, fmt: str = "{:.2f}") -> Parameter:
        row = calib[key]
        return Parameter(name, fmt.format(row["value"]),
                         _kill_status(row["status"]), row["source"])

    def division(hours: float, real: str) -> Parameter:
        return Parameter(
            "Time between divisions", f"{hours:g} h", "model choice",
            f"Compressed clock. Measured turnover is about {real}; the "
            "ratio between normal and cancer (7.5 to 1) is the measured "
            "part (Meyer 1982, PMID 6185700).")

    anoxia = Parameter(
        "Death when starved of oxygen",
        f"{cm.ANOXIC_DEATH_PER_HOUR:.1%} per hour, below "
        f"{cm.ANOXIC_O2_LEVEL / 0.21 * 100:.2f}% of air-level oxygen",
        "estimate",
        "Threshold from Papandreou 2005 (PMID 15833847); the rate is an "
        "estimate.")

    def share(key: str) -> str:
        return f"{kc.PRAD_IMMUNE_SHARE[key] * 100:.0f}%"

    mix_source = f"{kc.PRAD_SOURCE}. A deconvolution estimate, not a count."
    mix_note = ("A cohort average, used for every run. It is not this "
                "patient's own infiltrate.")

    entries: List[CellEntry] = []

    # ── Normal prostate epithelium ───────────────────────────────────
    d, ds, dsrc = _size(looks["normal"])
    entries.append(CellEntry(
        key="normal", name=_name(looks, "normal"),
        diameter_um=d, diameter_status=ds, diameter_source=dsrc,
        shape=str(looks["normal"]["mesh_type"]),
        colour=tuple(looks["normal"]["colour"]), count=n_normal,
        does=("The working tissue of the prostate. Luminal cells line the "
              "glands, respond to androgen through the androgen receptor "
              "and secrete PSA. They turn over very slowly."),
        simulated=("One cell per lattice site, laid out to match the "
                   "occupancy of a measured human prostate section, so "
                   "glands pack and lumens stay empty. Each cell runs the "
                   "androgen-receptor pathway and makes PSA, divides on a "
                   "clock, and dies if its oxygen supply fails. Immune "
                   "cells ignore it."),
        parameters=[
            division(cellular.division_time_normal, "347 days"),
            anoxia,
        ],
        missing=[
            "No measured width: three papers give the cell's height "
            "(19–28 µm) and none its width.",
            "One type stands for all non-cancer tissue; basal cells and "
            "stroma are not distinguished.",
            "The tissue layout is prostate even when the patient's biopsy "
            "came from a metastasis elsewhere.",
        ]))

    # ── Cancer ───────────────────────────────────────────────────────
    d, ds, dsrc = _size(looks["cancer"])
    entries.append(CellEntry(
        key="cancer", name=_name(looks, "cancer"),
        diameter_um=d, diameter_status=ds, diameter_source=dsrc,
        shape=str(looks["cancer"]["mesh_type"]),
        colour=tuple(looks["cancer"]["colour"]), count=n_cancer,
        does=("A prostate epithelial cell that has escaped growth "
              "control. It divides faster than its neighbours, usually "
              "still depends on androgen signalling, and displays "
              "fragments of its mutated proteins on its surface (on "
              "MHC class I), which is how T cells can find it. It can "
              "evade the immune system by showing less MHC class I or "
              "more PD-L1."),
        simulated=("Seeded as one nodule inside the normal tissue. Each "
                   "cell carries the patient's own missense mutations; a "
                   "binding predictor (MHCflurry) decides which mutated "
                   "peptides the patient's HLA types can display. Its "
                   "division rate follows its androgen-receptor "
                   "signalling, which is where anti-androgen drugs and "
                   "receptor mutations such as AR F877L act. Its MHC "
                   "class I rises when nearby T cells release "
                   "interferon-gamma, and daughters inherit their "
                   "parent's mutations and escape traits."),
        parameters=[
            division(cellular.division_time_cancer, "46 days"),
            anoxia,
            Parameter("Extra death when androgen is withdrawn",
                      f"{cellular.androgen_withdrawal_death_per_hour:g} "
                      f"per hour, in {cellular.ar_dependent_fraction:.0%} "
                      "of cells",
                      "estimate",
                      "Direction from Westin 1995 (PMID 7778676); the "
                      "magnitudes are estimates."),
        ],
        missing=[
            "The size is from circulating tumour cells; no one has "
            "published a width for this cell measured in tissue.",
            "Starting MHC class I is one unsourced value for every "
            "cancer cell.",
            "Gene amplification (for example of AR) is not modelled; "
            "only missense mutations cross from the genome.",
            "The tumour is a nodule of about a hundred cells, so the "
            "whole tissue's immune cells can converge on it.",
        ]))

    # ── CD8 T cell ───────────────────────────────────────────────────
    d, ds, dsrc = _size(looks["T_cell"])
    entries.append(CellEntry(
        key="T_cell", name=_name(looks, "T_cell"),
        diameter_um=d, diameter_status=ds, diameter_source=dsrc,
        shape=str(looks["T_cell"]["mesh_type"]),
        colour=tuple(looks["T_cell"]["colour"]), count=n_t,
        does=("The immune system's specific killer. Each T cell carries "
              "one receptor; if it matches a mutated peptide displayed on "
              "a cancer cell, the T cell attaches and delivers perforin "
              "and granzymes. One contact rarely kills a solid tumour "
              "cell: damage from several hits has to add up. After long "
              "stimulation T cells become exhausted and stop working."),
        simulated=("Walks at random through the tissue, slowed by dense "
                   "matrix. On touching a cancer cell it checks the "
                   f"patient's repertoire of {immune.tcr_repertoire_size:,} "
                   "receptor clones against the peptides that cell "
                   "displays. A recognised contact lasts a drawn time and "
                   "adds perforin hits to the target; hits decay, and the "
                   "target dies only when enough accumulate. Kill "
                   "strength scales with receptor affinity, the target's "
                   "MHC class I, checkpoint blockade and local "
                   "suppression. A clone that has been engaged long "
                   "enough is exhausted for good."),
        parameters=[
            contact("T_cell", True), contact("T_cell", False),
            kill("tcell_hits_per_contact",
                 "Perforin hits per recognised contact, mean"),
            kill("tcell_damage_half_life_min",
                 "Half-life of sublethal damage", "{:.1f} min"),
            kill("tcell_lethal_damage",
                 "Damage that kills (about three hits in 50 min)"),
            Parameter("Exhaustion after first engagement",
                      f"{immune.tcell_exhaustion_onset_hours:g} h",
                      "estimate",
                      "State from Philip 2017 (PMID 28514453); the onset "
                      "time is an estimate (7–14 days)."),
            Parameter("Share of the infiltrate", share("n_t_cells"),
                      "derived", mix_source),
            *movement("T_cell"),
        ],
        missing=[
            mix_note,
            "The receptor repertoire is generated, not sequenced from "
            "the patient.",
            "No helper (CD4) T cells, and no priming in a lymph node: "
            "T cells arrive already able to kill.",
        ]))

    # ── NK cell ──────────────────────────────────────────────────────
    d, ds, dsrc = _size(looks["NK_cell"])
    low, high = immune.GRANULES_PER_KILL
    reserve = immune._lytic_reserve("NK_cell")
    entries.append(CellEntry(
        key="NK_cell", name=_name(looks, "NK_cell"),
        diameter_um=d, diameter_status=ds, diameter_source=dsrc,
        shape=str(looks["NK_cell"]["mesh_type"]),
        colour=tuple(looks["NK_cell"]["colour"]), count=n_nk,
        does=("An innate killer that needs no specific receptor match. "
              "It attacks cells that show too little MHC class I "
              "('missing self') or signs of stress, which makes it the "
              "counterweight to tumours that hide from T cells. It "
              "kills with the same granules a T cell uses, several "
              "targets in a row."),
        simulated=("Walks and makes contact like a T cell. Whether a "
                   "contact kills is graded by the target's MHC class I: "
                   "the less it shows, the likelier the kill. When the "
                   "patient's HLA typing was measured from their reads, "
                   "a KIR–HLA licensing model decides engagement "
                   "instead. Each kill spends a few of the cell's "
                   "granules. After a few kills the cell is spent: it "
                   "keeps moving but kills nothing until it recovers."),
        parameters=[
            contact("NK_cell", True), contact("NK_cell", False),
            kill("nk_graded_ceiling",
                 "Kill chance per contact, at most (MHC-I absent)"),
            kill("nk_kill_per_contact",
                 "Kill chance per contact, typical solid target"),
            kill("nk_serial_kills_mean",
                 "Targets killed before the cell is spent, mean"),
            Parameter("Most targets one cell kills before it is spent",
                      f"{kc.NK_SERIAL_KILLS_MAX}", "measured",
                      "Prager 2019 (PMID 31270246); Bhat & Watzl 2007 "
                      "(PMID 17389917)"),
            kill("nk_recovery_hours",
                 "Time a spent cell takes to kill again", "{:g} h"),
            Parameter("Granules carried", f"{reserve:g}", "measured",
                      "PMID 29592963"),
            Parameter("Granules spent per kill", f"{low}–{high}",
                      "measured", "PMID 29592963"),
            Parameter("Share of the infiltrate", share("n_nk_cells"),
                      "derived", mix_source),
            *movement("NK_cell"),
        ],
        missing=[
            "The kill limit is measured on NK cells activated with "
            "IL-2 in a dish, which is the generous end; NK cells inside "
            "a prostate tumour are not known to do as well.",
            "Recovery in two days was measured with added IL-2, away "
            "from targets. Inside a tumour it may be slower or not "
            "happen.",
            "The size is the weakest in this table: a single mean with "
            "the underlying data not shown.",
            mix_note,
        ]))

    # ── Macrophage ───────────────────────────────────────────────────
    d, ds, dsrc = _size(looks["macrophage"])
    pol = kc.PRAD_MACROPHAGE_POLARIZATION
    entries.append(CellEntry(
        key="macrophage", name=_name(looks, "macrophage"),
        diameter_um=d, diameter_status=ds, diameter_source=dsrc,
        shape=(f"{looks['macrophage_M1']['mesh_type']} (M1) / "
               f"{looks['macrophage_M2']['mesh_type']} (M2)"),
        colour=tuple(looks["macrophage"]["colour"]), count=n_mac,
        does=("A large cell that engulfs other cells and debris. In a "
              "tumour it comes in two leanings: M1, which attacks, and "
              "M2, which suppresses T cells and helps the tumour. Most "
              "macrophages in prostate tumours lean M2. Cancer cells "
              "protect themselves with 'don't eat me' signals."),
        simulated=("The most numerous immune cell in the run. It walks, "
                   "makes contact and occasionally engulfs a cancer "
                   "cell; a target's PD-L1 lowers that chance. Each "
                   "macrophage has a position between M1 and M2 that "
                   "drifts toward what its local signals favour over "
                   "about a day. M2 macrophages weaken the T cells near "
                   "them (arginase)."),
        parameters=[
            contact("macrophage", True), contact("macrophage", False),
            kill("macrophage_kill_per_contact",
                 "Chance a contact ends in engulfment"),
            Parameter("Starting mix, M1 / M0 / M2",
                      f"{pol['M1']:.0%} / {pol['M0']:.0%} / {pol['M2']:.0%}",
                      "derived", mix_source),
            Parameter("Share of the infiltrate", share("n_macrophages"),
                      "derived", mix_source),
            *movement("macrophage"),
        ],
        missing=[
            "The size is an alveolar (lung) macrophage's; no "
            "tumour-associated macrophage measurement is in use.",
            "Contact times are illustrative defaults, not measurements.",
            "The 'don't eat me' signal CD47 is not represented; its "
            "effect is folded into one low engulfment chance.",
            mix_note,
        ]))

    # A cell the library on THIS machine has no entry for still runs --
    # the engine falls back to defaults -- so it is still listed, under a
    # plain name, and says so. Found on production the day this page went
    # live: the cancer cell's entry had never been seeded there, and the
    # section was headed "None".
    return [
        entry if looks[entry.key].get("entity") else _without_entity(entry)
        for entry in entries
    ]


def status_counts(entries: Optional[List[CellEntry]] = None) -> Dict[str, int]:
    """How many parameter rows (sizes included) carry each status."""
    entries = build_catalog() if entries is None else entries
    counts = {status: 0 for status in STATUS_ORDER}
    for entry in entries:
        counts[entry.diameter_status] += 1
        for parameter in entry.parameters:
            counts[parameter.status] += 1
    return counts
