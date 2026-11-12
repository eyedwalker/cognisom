"""How cells talk to each other: one registry, and one way to add a signal.

Why this module exists
----------------------
The engine diffuses six signals, and each was wired by hand in its own
way: a field created by name in ``SpatialModule.initialize``, a secretion
provider registered from whichever module thought of it, an uptake term
written into ``_sync_sources_sinks``, and an attribute pushed to cells
through ``FIELDS_TO_CELLS``. Nothing lists them. Adding a seventh means
touching four places in two modules, and whether a given signal is read by
anything at all is a fact only the code's history knows -- oxygen and
glucose are solved every step and read by nothing.

Blood vessels, stroma, nerves and new immune cells all communicate through
secreted molecules, so every one of them needs this path first. This
module provides it in two parts.

1. ``SIGNALS`` -- the inventory. Every field the spatial module solves has
   an entry saying what the molecule is, who makes it, who reads it and
   what that changes, with each number's status. The numbers are READ from
   the modules that own them, never retyped, and a test fails if a field
   exists without an entry. ``wired`` is the honest column: False means the
   field is solved and nothing consumes it.

2. ``SecretedSignal`` -- the generic path. A new signal is a declaration:
   its transport numbers, which cells secrete it and how fast, and how
   strongly cells take it up. ``SpatialModule.register_signal`` then
   creates the field, sources it from the declared cells every update,
   applies the uptake, and gives every tissue cell its local level in
   ``cell.signals[name]``; an immune cell reads its own with
   ``SpatialModule.get_concentration``. No signal is declared by default,
   so adding this path changes no existing run.

What this module deliberately does not contain is any new biology. A
signal's EFFECT -- what a cell does differently when it senses one -- lives
with the cell that responds, where its evidence can be stated. This is the
plumbing, and the catalogue of what flows through it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

#: Status vocabulary, shared with dashboard/cell_catalog.py.
STATUSES = ("measured", "derived", "estimate", "model choice", "placeholder")


@dataclass(frozen=True)
class Quantity:
    """One transport number, with how far it can be trusted."""

    value: float
    unit: str
    status: str
    source: str

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"unknown status {self.status!r}")


@dataclass(frozen=True)
class SignalEntry:
    """One row of the inventory: a field the engine already solves."""

    name: str                      # the spatial field's name
    molecule: str                  # what it stands for, in plain words
    kind: str                      # gas / nutrient / metabolite / hormone / cytokine / chemokine
    made_by: str
    read_by: str
    effect: str
    #: True when something in a default run changes because of this field.
    wired: bool
    diffusion: Quantity
    #: Loss that sets the range: decay, uptake, or neither.
    loss: Optional[Quantity] = None
    note: str = ""


def inventory() -> List[SignalEntry]:
    """Every signal the spatial module solves, read from the code."""
    from cognisom.modules import spatial_module as sm

    return [
        SignalEntry(
            name="oxygen", molecule="Oxygen", kind="gas",
            made_by="Delivered by capillaries.",
            read_by="Nothing reads the diffusion field. Each cell's oxygen "
                    "is computed separately, from its distance to the "
                    "nearest capillary.",
            effect="None from this field. Per-cell oxygen (the separate "
                   "calculation) drives the metabolic shift, hypoxia and "
                   "death from anoxia.",
            wired=False,
            diffusion=Quantity(2000.0, "µm²/s", "placeholder",
                               "A literal in the code with no source "
                               "recorded beside it."),
            loss=Quantity(sm.SpatialModule.O2_DEMAND_NORMAL, "per hour, per cell",
                          "model choice",
                          "Matched to the cellular module's own demand."),
            note="Solved every update and consumed by nothing; kept because "
                 "the per-cell calculation was validated against it."),
        SignalEntry(
            name="glucose", molecule="Glucose", kind="nutrient",
            made_by="Delivered by capillaries.",
            read_by="Nothing reads the diffusion field.",
            effect="None from this field.",
            wired=False,
            diffusion=Quantity(600.0, "µm²/s", "placeholder",
                               "A literal in the code with no source "
                               "recorded beside it."),
            note="Solved and consumed by nothing, like oxygen."),
        SignalEntry(
            name="testosterone", molecule="Free testosterone", kind="hormone",
            made_by="Delivered by capillaries at the serum level the run "
                    "is given.",
            read_by="Every tissue cell, which converts it to DHT.",
            effect="Sets each cell's androgen-receptor signalling, and "
                   "through it PSA output and how fast a cancer cell "
                   "divides.",
            wired=True,
            diffusion=Quantity(sm.TESTOSTERONE_DIFFUSION_UM2_S, "µm²/s",
                               "estimate",
                               "From molecular size; no tissue "
                               "measurement was found.")),
        SignalEntry(
            name="lactate", molecule="Lactate", kind="metabolite",
            made_by="Every tissue cell, cancer cells faster.",
            read_by="T cells, as an acidity gate on killing.",
            effect="None in the shipped scenarios: the tissue block is too "
                   "well perfused to become acidic.",
            wired=False,
            diffusion=Quantity(sm.LACTATE_DIFFUSION_UM2_S, "µm²/s", "estimate",
                               "From molecular size, halved for tissue."),
            note="The mechanism exists and is provably inert at this "
                 "tissue size."),
        SignalEntry(
            name="ifn_gamma", molecule="Interferon-gamma", kind="cytokine",
            made_by="CD8 T cells and NK cells while they are engaged on a "
                    "target.",
            read_by="Every tissue cell.",
            effect="Raises a cancer cell's MHC class I, making it more "
                   "visible to T cells and less so to NK cells.",
            wired=True,
            diffusion=Quantity(sm.IFN_GAMMA_DIFFUSION_UM2_S, "µm²/s",
                               "estimate", "From molecular size."),
            loss=Quantity(sm.IFN_GAMMA_UPTAKE_PER_H, "per hour, per cell",
                          "estimate",
                          "Chosen so the signal's range in packed tissue "
                          "is the measured tens of micrometres "
                          "(Oyler-Yaniv 2017, PMID 28389069).")),
        SignalEntry(
            name="cytokine", molecule="Chemokine (stands for CXCL12-like "
                                      "attractants)", kind="chemokine",
            made_by="Cancer cells, through the morphogen module.",
            read_by="Patrolling immune cells, which bias their walk up "
                    "its gradient.",
            effect="Steers immune cells toward the tumour. The strength "
                   "of the bias is an estimate.",
            wired=True,
            diffusion=Quantity(100.0, "µm²/s", "placeholder",
                               "A literal in the code with no source "
                               "recorded beside it."),
            loss=Quantity(sm.CHEMOKINE_HALF_LIFE_HOURS, "hours (half-life)",
                          "estimate",
                          "CXCL12's half-life in blood (PMC6582792); "
                          "tissue-bound chemokine likely lasts longer.")),
    ]


#: Molecules a tumour uses that the engine has no signal for yet. Listed
#: so the gap is a row someone can see, not an absence.
NOT_YET: Tuple[Tuple[str, str], ...] = (
    ("TGF-beta", "Made by tumour cells, fibroblasts and regulatory T "
                 "cells; suppresses T cells and NK cells. Today a single "
                 "fraction stands in for it."),
    ("IL-2", "Made by activated T cells; sustains T cells and restores "
             "spent NK cells. NK recovery runs on a fixed clock instead."),
    ("IL-10", "Made by M2 macrophages and regulatory T cells; damps "
              "killing."),
    ("VEGF", "Made by hypoxic cells; grows new blood vessels. There is "
             "no vessel growth to drive."),
    ("Noradrenaline", "Released by sympathetic nerve endings; weakens "
                      "T cells. Today one global stress dial."),
    ("PGE2, adenosine, CCL2, CXCL9/10", "Not represented."),
)


# ── The generic path ─────────────────────────────────────────────────

@dataclass(frozen=True)
class Secretion:
    """Which cells make a signal, and how fast.

    ``population`` is 'tissue' (the cellular module's cells) or 'immune'.
    ``cell_type`` is that module's own token ('cancer', 'normal',
    'T_cell', ...). ``rate`` is per cell per hour in the field's units.
    ``when`` narrows it further and receives the cell; None means always.
    """

    population: str
    cell_type: str
    rate: float
    when: Optional[Callable[[object], bool]] = None

    def __post_init__(self) -> None:
        if self.population not in ("tissue", "immune"):
            raise ValueError(
                f"population must be 'tissue' or 'immune', not {self.population!r}")
        if self.rate < 0.0:
            raise ValueError("a secretion rate cannot be negative")


def is_engaged(cell) -> bool:
    """A ``when`` for 'only while locked on a target'.

    A named function and not a lambda on purpose: a declared signal rides
    in the run's configuration, and a checkpoint pickles that.
    """
    return bool(getattr(cell, "activated", False))


@dataclass(frozen=True)
class SecretedSignal:
    """A signal declared rather than hand-wired.

    Transport is diffusion with two possible losses, and at least one is
    needed or the signal fills the tissue and carries no position:
    ``half_life_hours`` (decay everywhere) and ``uptake_per_hour`` (each
    tissue cell consumes it at its own voxel, which makes the range
    shorter where the tissue is denser).
    """

    name: str
    diffusion_um2_s: float
    secreted_by: Tuple[Secretion, ...]
    half_life_hours: Optional[float] = None
    uptake_per_hour: float = 0.0

    def __post_init__(self) -> None:
        if not self.name or not self.name.isidentifier():
            raise ValueError(f"a signal needs a plain name, not {self.name!r}")
        if self.diffusion_um2_s <= 0.0:
            raise ValueError("diffusion must be positive")
        if self.half_life_hours is not None and self.half_life_hours <= 0.0:
            raise ValueError("a half-life must be positive")
        if self.uptake_per_hour < 0.0:
            raise ValueError("uptake cannot be negative")
        if self.half_life_hours is None and self.uptake_per_hour == 0.0:
            raise ValueError(
                f"signal {self.name!r} has no decay and no uptake: it would "
                "fill the tissue evenly and tell a cell nothing about where "
                "it is. Give it a half-life or an uptake.")

    @property
    def decay_per_hour(self) -> float:
        if self.half_life_hours is None:
            return 0.0
        return float(np.log(2.0) / self.half_life_hours)

    def sources(self, cellular_module, immune_module) -> List[Tuple[np.ndarray, float]]:
        """``(position, rate)`` for every cell secreting right now."""
        out: List[Tuple[np.ndarray, float]] = []
        for rule in self.secreted_by:
            if rule.rate <= 0.0:
                continue
            if rule.population == "tissue":
                cells = getattr(cellular_module, "cells", None) or {}
                pool = (c for c in cells.values() if getattr(c, "alive", False))
            else:
                cells = getattr(immune_module, "immune_cells", None) or {}
                pool = (c for c in cells.values()
                        if not getattr(c, "in_blood", False))
            for cell in pool:
                if cell.cell_type != rule.cell_type:
                    continue
                if rule.when is not None and not rule.when(cell):
                    continue
                out.append((cell.position, rule.rate))
        return out


def decay_length_um(signal: SecretedSignal, uptake_volume_fraction: float = 0.0) -> float:
    """How far the signal reaches, sqrt(D / k), in micrometres.

    ``uptake_volume_fraction`` is the share of the tissue volume occupied
    by consuming cells; uptake acts only there, so the effective loss is
    the decay plus the volume-averaged uptake.
    """
    k_per_s = (signal.decay_per_hour
               + signal.uptake_per_hour * uptake_volume_fraction) / 3600.0
    if k_per_s <= 0.0:
        return float("inf")
    return float(np.sqrt(signal.diffusion_um2_s / k_per_s))
