"""Secreted gradients: cells that secrete, a field that decays, cells that read it.

WHAT THIS WAS
-------------
Three embryonic organisers -- BMP, Shh, Wnt -- at hard-coded corners of
a 200 um box, an analytic ``exp(-d / 50 um)`` gradient, and a French-flag
rule assigning anterior / middle / posterior fates. Measured on the
shipped Baseline run (2 h, 92 cells): the nearest cell to any organiser
was 50 um, exactly one decay length, so 0.0% of cells sensed anything
above e^-1; fates split 76 "anterior" / 16 "middle" by distance noise;
and nothing anywhere consumed a fate. ``set_spatial_module`` had no
callers, so the gradient was never a diffused field. The checkbox on the
Simulation page was labelled "BMP/Wnt gradients, cell fate" and
recorded one counter. An adult prostate tumour has no anterior.

WHAT IT IS
----------
The secretion side of a diffusible signal. Living cells of the secreting
types (cancer, by default) are sources on the spatial module's
``cytokine`` field; the field diffuses and decays; immune cells read its
gradient in their patrol step (``immune_module._patrol``) and bias
their random walk up it at their sourced patrol speed. That is the
CXCL12 / CXCR4 class of mechanism -- tumour- and stroma-derived
chemokine recruiting leukocytes -- expressed with the cell types this
engine has. Nothing here moves a cell; this module says where the
signal comes from and reports what cells are exposed to.

Sources are supplied through ``SpatialModule.register_secretion`` and
rebuilt every spatial update, so a cell that dies stops secreting and
one that moves secretes from where it is. The old ``_registered_sources``
guard could do neither.

NUMBERS, AND WHICH ARE MEASURED
-------------------------------
* Secretion rate: **1.0 field-unit per secreting cell per hour. An
  ESTIMATE.** No per-cell CXCL12 secretion rate was found in the
  literature in a usable form (ELISA concentrations of conditioned
  medium only, without cell counts or harvest times). The field is
  therefore dimensionless: by the maximum principle an isolated
  secreting voxel saturates at rate / decay, and what the consumer uses
  is the gradient's DIRECTION, not its magnitude.
* Decay: a 30-minute half-life (``spatial_module.CHEMOKINE_HALF_LIFE_HOURS``),
  the circulating figure, the only measured one. With D = 100 um^2/s
  the decay length is 510 um; see :meth:`MorphogenModule.decay_length_um`.
* The chemotactic bias lives on the immune module and is also an
  estimate; this module does not set it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from cognisom.core.module_base import SimulationModule
from cognisom.core.event_bus import EventTypes

#: Field-units per secreting cell per hour. ESTIMATE; see the module
#: docstring for why it can be, and what the number does and does not
#: affect.
SECRETION_RATE_PER_HOUR = 1.0

#: Cell types that secrete. The engine's cellular module knows 'normal',
#: 'cancer' and 'immune'; stromal fibroblasts, the other major CXCL12
#: source, are not a type it has.
SECRETING_TYPES: Tuple[str, ...] = ("cancer",)

#: The spatial field secreted into. Created by SpatialModule with its
#: decay; this module does not own it.
DEFAULT_FIELD = "cytokine"


@dataclass
class Exposure:
    """What one tracked cell currently sees of the field."""

    cell_id: int
    concentration: float = 0.0
    gradient_magnitude: float = 0.0


class MorphogenModule(SimulationModule):
    """Secretion into, and exposure to, a diffusible signal field.

    Events Subscribed:
    - CELL_DIVIDED: track the daughter
    - CELL_DIED: stop tracking (the source list is rebuilt from the
      living population every update, so death also ends secretion
      without this module doing anything)
    """

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.field_name: str = str(self.config.get("field", DEFAULT_FIELD))
        self.secretion_rate: float = float(
            self.config.get("secretion_rate", SECRETION_RATE_PER_HOUR))
        self.secreting_types: Tuple[str, ...] = tuple(
            self.config.get("secreting_types", SECRETING_TYPES))

        self.exposures: Dict[int, Exposure] = {}
        self.n_secreting: int = 0

        self.spatial_module = None
        self.cellular_module = None

    # ── lifecycle ─────────────────────────────────────────────────────

    def initialize(self):
        self.subscribe(EventTypes.CELL_DIVIDED, self.on_cell_divided)
        self.subscribe(EventTypes.CELL_DIED, self.on_cell_died)
        print(f"  Secreted gradient: {self.secreting_types} -> "
              f"'{self.field_name}' at {self.secretion_rate:g}/h per cell")

    def set_spatial_module(self, spatial_module):
        """Link to the spatial module and become a source provider."""
        self.spatial_module = spatial_module
        register = getattr(spatial_module, "register_secretion", None)
        if register is not None:
            register(self.field_name, self.secretion_sources)

    def set_cellular_module(self, cellular_module):
        self.cellular_module = cellular_module

    # ── the secretion side ────────────────────────────────────────────

    def secretion_sources(self) -> List[Tuple[np.ndarray, float]]:
        """``(position, rate)`` for every living secreting cell, now.

        Called by the spatial module each update after it clears the
        field's terms. Counting here rather than in update() keeps
        ``n_secreting`` equal to what the field was actually given.
        """
        cells = getattr(self.cellular_module, "cells", None) or {}
        sources: List[Tuple[np.ndarray, float]] = []
        for cell in cells.values():
            if not getattr(cell, "alive", True):
                continue
            if getattr(cell, "cell_type", None) not in self.secreting_types:
                continue
            sources.append((np.asarray(cell.position, dtype=float),
                            self.secretion_rate))
        self.n_secreting = len(sources)
        return sources

    # ── exposure tracking ─────────────────────────────────────────────

    def add_cell(self, cell_id: int, position: Any = None):
        """Track a cell's exposure. ``position`` is accepted for the
        caller's convenience and not stored: positions are re-read from
        the cellular module every update, which is the lesson the old
        module learned the hard way."""
        self.exposures[cell_id] = Exposure(cell_id=cell_id)

    def remove_cell(self, cell_id: int):
        self.exposures.pop(cell_id, None)

    def update(self, dt: float):
        field = self._field()
        if field is None or self.cellular_module is None:
            return
        cells = getattr(self.cellular_module, "cells", {}) or {}
        for cell_id, exposure in list(self.exposures.items()):
            cell = cells.get(cell_id)
            if cell is None or not getattr(cell, "alive", True):
                del self.exposures[cell_id]
                continue
            position = np.asarray(cell.position, dtype=float)
            exposure.concentration = float(field.get_concentration_at(position))
            exposure.gradient_magnitude = float(
                np.linalg.norm(field.get_gradient_at(position)))

    # ── reporting ─────────────────────────────────────────────────────

    def _field(self):
        if self.spatial_module is None:
            return None
        return (getattr(self.spatial_module, "fields", None) or {}).get(
            self.field_name)

    def decay_length_um(self) -> Optional[float]:
        """sqrt(D / k): how far the signal reaches before it is gone.

        ``None`` when the field has no decay, in which case it has no
        finite reach and the steady state is unbounded.
        """
        field = self._field()
        if field is None:
            return None
        decay = float(getattr(field, "decay_rate", 0.0) or 0.0)
        if decay <= 0.0:
            return None
        d_per_hour = float(field.diffusion_coeff) * 3600.0
        return float(np.sqrt(d_per_hour / decay))

    def get_exposure(self, cell_id: int) -> Optional[Exposure]:
        return self.exposures.get(cell_id)

    def get_state(self) -> Dict[str, Any]:
        field = self._field()
        if field is not None:
            concentration = np.asarray(field.concentration, dtype=float)
            field_max = float(concentration.max()) if concentration.size else 0.0
            field_mean = float(concentration.mean()) if concentration.size else 0.0
        else:
            field_max = field_mean = 0.0
        exposures = list(self.exposures.values())
        return {
            "field": self.field_name,
            "secreting_types": list(self.secreting_types),
            "secretion_rate_per_hour": self.secretion_rate,
            "n_secreting": self.n_secreting,
            "field_max": field_max,
            "field_mean": field_mean,
            "decay_length_um": self.decay_length_um(),
            "n_cells_tracked": len(exposures),
            "mean_exposure": (float(np.mean([e.concentration for e in exposures]))
                              if exposures else 0.0),
            "mean_gradient_at_cells": (
                float(np.mean([e.gradient_magnitude for e in exposures]))
                if exposures else 0.0),
        }

    # ── events ────────────────────────────────────────────────────────

    def on_cell_divided(self, data):
        daughter = (data or {}).get("daughter_id")
        if daughter is not None:
            self.add_cell(int(daughter))

    def on_cell_died(self, data):
        cell_id = (data or {}).get("cell_id")
        if cell_id is not None:
            self.remove_cell(int(cell_id))
