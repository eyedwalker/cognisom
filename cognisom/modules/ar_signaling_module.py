"""Per-cell androgen receptor signalling, driven by each cell's tissue DHT.

This is the consumer of the testosterone signal. The field puts free
testosterone at every cell; CellularModule turns that into a tissue DHT
level per cell (``cell.dht_nM``, Titus 2005 at eugonadal supply, with
the intracrine share castration does not touch); and this module runs
the library-sourced AR pathway -- AR transcription, translation, DHT
binding, PSA induction -- in every live cell with that DHT as its
androgen supply. PSA is the readout.

The coupling
------------
The pathway (``ReactionNetworkBuilder.build_ar_signaling``) feeds its
DHT species at a constant rate ``DHT_input`` and degrades the AR-DHT
complex at ``k_complex_deg``; nothing else removes DHT once bound, so
the complex can never exceed ``DHT_input / k_complex_deg`` (the supply
bound the AR tests pin). This module sets, for each cell,

    DHT_input = k_complex_deg * dht_nM

so that bound is the cell's tissue DHT: at steady state the complex IS
the tissue DHT the measurements report (13.7 nM eugonadal, Titus 2005),
and a castrate tissue at 25% carries a complex at 25%.

What it predicts, and what it does not
--------------------------------------
With the library's ``K_psa`` of 0.5 nM the PSA induction term is 96%
saturated at 13.7 nM and 87% at 3.4 nM, so castration lowers steady
PSA by about 10%. The clinic sees serum PSA fall by well over 90% under
androgen deprivation. The gap is the K_psa category error recorded on
2026-10-17 (an AR-DNA affinity used as a transcriptional Km), already
marked xfail in the AR tests, and now a number the engine shows rather
than a note.

Cost: one batched BDF solve per engine step over all live cells, with
the block-diagonal Jacobian sparsity declared -- about 2 ms for 100
cells, so it is on by default. The system is stiff (DHT turns over in
seconds against AR protein in days) and an explicit solver silently
fabricates mass; see the 10-17 note before changing the method.
"""

from __future__ import annotations

import contextlib
import io
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from cognisom.core.event_bus import EventTypes
from cognisom.core.module_base import SimulationModule
from cognisom.modules.cellular_module import TISSUE_DHT_REFERENCE_NM

log = logging.getLogger(__name__)

#: Hours of integration used to find a steady state. The slowest species,
#: AR protein at k_deg 0.1 /h, is settled to 1e-8 well before this.
STEADY_STATE_HOURS = 4000.0


class ARSignalingModule(SimulationModule):
    """The AR/PSA pathway in every live cell, at that cell's tissue DHT."""

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.name = "ARSignalingModule"
        self.rtol = float(self.config.get('rtol', 1e-6))
        self.atol = float(self.config.get('atol', 1e-9))

        self.system = None
        self.rhs = None
        self.params: Dict[str, float] = {}
        self.species: List[str] = []
        self._i: Dict[str, int] = {}

        # One row per mapped cell, in `ids` order.
        self.ids: List[int] = []
        self.index: Dict[int, int] = {}
        self.Y = np.zeros((0, 0), dtype=float)
        self.eugonadal_state: Optional[np.ndarray] = None

        self.cellular_module = None
        self._sparsity_n = -1
        self._sparsity = None
        self.total_steps = 0
        self.total_rhs_evals = 0
        self.failed_solves = 0

    # ── setup ─────────────────────────────────────────────────────

    def initialize(self):
        from cognisom.genomics.reaction_builder import library_ar_system

        self.system = library_ar_system()
        self.rhs = self.system.get_rhs_func()
        self.params = dict(self.system.parameters)
        self.species = list(self.system.species_names)
        self._i = {name: i for i, name in enumerate(self.species)}
        self.Y = np.zeros((0, len(self.species)), dtype=float)

        # Every cell starts in equilibrium with the eugonadal supply. AR
        # protein settles over days; a run that started from zeros would
        # show PSA rising for its whole length and call it biology.
        self.eugonadal_state = self.steady_state(TISSUE_DHT_REFERENCE_NM)

        self.subscribe(EventTypes.CELL_DIVIDED, self._on_cell_divided)
        self.subscribe(EventTypes.CELL_DIED, self._on_cell_died)
        log.info("ARSignalingModule: %d species, eugonadal PSA %.1f",
                 len(self.species), self.eugonadal_state[self._i["PSA"]])

    def set_cellular_module(self, cellular_module):
        self.cellular_module = cellular_module

    def dht_input_for(self, dht_nM: float) -> float:
        """The supply rate whose complex bound is this tissue DHT."""
        return float(self.params['k_complex_deg']) * max(0.0, float(dht_nM))

    def steady_state(self, dht_nM: float) -> np.ndarray:
        """The pathway's equilibrium at a tissue DHT, by stiff integration."""
        from scipy.integrate import solve_ivp
        params = dict(self.params)
        params['DHT_input'] = self.dht_input_for(dht_nM)
        y0 = np.array(self.system.species_initial, dtype=float)
        sol = solve_ivp(lambda t, y: self.rhs(t, y, params), (0.0, STEADY_STATE_HOURS),
                        y0, method="LSODA", rtol=1e-8, atol=1e-10)
        if not sol.success:
            raise RuntimeError(f"AR steady state failed: {sol.message}")
        return sol.y[:, -1]

    # ── population ────────────────────────────────────────────────

    def add_cell(self, cell_id: int, state: Optional[np.ndarray] = None) -> None:
        if cell_id in self.index or self.eugonadal_state is None:
            return
        row = np.array(self.eugonadal_state if state is None else state, dtype=float)
        self.index[cell_id] = len(self.ids)
        self.ids.append(int(cell_id))
        self.Y = np.vstack([self.Y, row[None, :]]) if len(self.Y) else row[None, :].copy()

    def remove_cell(self, cell_id: int) -> None:
        i = self.index.pop(cell_id, None)
        if i is None:
            return
        last = len(self.ids) - 1
        if i != last:
            self.ids[i] = self.ids[last]
            self.index[self.ids[i]] = i
            self.Y[i] = self.Y[last]
        self.ids.pop()
        self.Y = self.Y[:last]

    def state_of(self, cell_id: int) -> Optional[np.ndarray]:
        i = self.index.get(cell_id)
        return None if i is None else self.Y[i].copy()

    def species_of(self, cell_id: int, name: str) -> Optional[float]:
        i = self.index.get(cell_id)
        return None if i is None else float(self.Y[i, self._i[name]])

    def psa_of(self, cell_id: int) -> Optional[float]:
        return self.species_of(cell_id, "PSA")

    # ── stepping ──────────────────────────────────────────────────

    def _dht_of_live_cells(self) -> np.ndarray:
        """Tissue DHT per mapped row; drops rows whose cell is gone."""
        cells = getattr(self.cellular_module, 'cells', None) or {}
        for cell_id in list(self.ids):
            cell = cells.get(cell_id)
            if cell is None or not getattr(cell, 'alive', False):
                self.remove_cell(cell_id)
        return np.array([float(getattr(cells[c], 'dht_nM', TISSUE_DHT_REFERENCE_NM))
                         for c in self.ids], dtype=float)

    def _sparsity_for(self, n: int):
        if n != self._sparsity_n:
            from scipy.sparse import block_diag, csr_matrix
            k = len(self.species)
            self._sparsity = block_diag([csr_matrix(np.ones((k, k)))] * n, format="csr")
            self._sparsity_n = n
        return self._sparsity

    def update(self, dt: float):
        if self.rhs is None or self.cellular_module is None:
            return
        dht = self._dht_of_live_cells()
        n = len(self.ids)
        if n == 0:
            return
        from scipy.integrate import solve_ivp
        k = len(self.species)
        params = dict(self.params)
        params['DHT_input'] = np.array([self.dht_input_for(v) for v in dht])

        def f(t, yf):
            return self.rhs(t, yf.reshape(n, k), params).ravel()

        sol = solve_ivp(f, (0.0, float(dt)), self.Y.ravel(), method="BDF",
                        rtol=self.rtol, atol=self.atol,
                        jac_sparsity=self._sparsity_for(n))
        self.total_steps += 1
        self.total_rhs_evals += int(sol.nfev)
        if not sol.success:
            self.failed_solves += 1
            if self.failed_solves == 1:
                log.warning("ARSignalingModule: BDF step failed (%s); state held.", sol.message)
            return
        self.Y = sol.y[:, -1].reshape(n, k)

    # ── events ────────────────────────────────────────────────────

    def _on_cell_divided(self, data):
        parent = data.get('cell_id') if isinstance(data, dict) else None
        daughter = data.get('daughter_id') if isinstance(data, dict) else None
        if daughter is None:
            return
        # The daughter shares the parent's cytoplasm: same receptor, same
        # complex, same PSA transcript -- not a fresh eugonadal cell.
        self.add_cell(int(daughter), state=self.state_of(parent) if parent is not None else None)

    def _on_cell_died(self, data):
        cell_id = data.get('cell_id') if isinstance(data, dict) else None
        if cell_id is not None:
            self.remove_cell(int(cell_id))

    # ── state ─────────────────────────────────────────────────────

    def get_state(self) -> Dict[str, Any]:
        n = len(self.ids)
        state: Dict[str, Any] = {
            'n_cells': n,
            'total_steps': self.total_steps,
            'rhs_evals_per_step': (self.total_rhs_evals / self.total_steps
                                   if self.total_steps else 0.0),
            'failed_solves': self.failed_solves,
            'eugonadal_psa': (float(self.eugonadal_state[self._i["PSA"]])
                              if self.eugonadal_state is not None else 0.0),
            'K_psa_nM': float(self.params.get('K_psa', 0.0)),
        }
        if n:
            for name in ("AR", "AR_DHT", "PSA"):
                col = self.Y[:, self._i[name]]
                state[f'{name.lower()}_mean'] = float(col.mean())
            state['psa_min'] = float(self.Y[:, self._i["PSA"]].min())
            state['psa_max'] = float(self.Y[:, self._i["PSA"]].max())
            state['psa_relative_to_eugonadal'] = (
                state['psa_mean'] / state['eugonadal_psa'] if state['eugonadal_psa'] else 0.0)
        else:
            state.update(psa_mean=0.0, ar_dht_mean=0.0, ar_mean=0.0)
        return state
