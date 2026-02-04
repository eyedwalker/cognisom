#!/usr/bin/env python3
"""
BNGL Module
===========

SimulationModule wrapper for rule-based modeling using BioNetGen Language.

VCell Parity Phase 4 - Rule-based modeling.

Features:
- BNGL model parsing and rule expansion
- Network generation from rule-based specification
- ODE or SSA simulation of expanded network
- Observable computation
- Support for combinatorial complexity

Usage::

    from cognisom.core import SimulationEngine, SimulationConfig
    from cognisom.modules.bngl_module import BNGLModule

    engine = SimulationEngine(SimulationConfig(dt=0.01, duration=100.0))
    engine.register_module('bngl', BNGLModule, {
        'model': 'egfr_signaling',
        'method': 'ode',
        'max_species': 1000,
    })
    engine.initialize()
    engine.run()
"""

import sys
sys.path.insert(0, '..')

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from core.module_base import SimulationModule
from core.event_bus import EventTypes

log = logging.getLogger(__name__)


class BNGLModule(SimulationModule):
    """
    Rule-based modeling module using BNGL.

    Parses BNGL models, expands rules to generate reaction networks,
    and simulates using ODE or SSA methods.

    Configuration
    -------------
    model : str or Path
        Model name ('egfr_signaling', 'simple_receptor') or path to .bngl file
    method : str
        Simulation method: 'ode' or 'ssa' (default: 'ode')
    max_species : int
        Maximum species in network expansion (default: 10000)
    ode_rtol : float
        ODE relative tolerance (default: 1e-6)
    ode_atol : float
        ODE absolute tolerance (default: 1e-8)

    Events Emitted
    --------------
    - NETWORK_GENERATED: When rule expansion completes
    - SPECIES_THRESHOLD: When a species count crosses threshold

    Example
    -------
    >>> engine.register_module('bngl', BNGLModule, {
    ...     'model': 'egfr_signaling',
    ...     'method': 'ode',
    ...     'max_species': 5000,
    ... })
    """

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.name = "BNGLModule"

        # Configuration
        self.model_spec = self.config.get('model', 'simple_receptor')
        self.method = self.config.get('method', 'ode')
        self.max_species = self.config.get('max_species', 10000)
        self.ode_rtol = self.config.get('ode_rtol', 1e-6)
        self.ode_atol = self.config.get('ode_atol', 1e-8)

        # State
        self.model = None
        self.reactions = None
        self.species_list = None
        self.state = None  # Current concentrations
        self.current_time = 0.0

        # Statistics
        self.total_steps = 0

    def initialize(self):
        """Initialize BNGL model and expand rules."""
        from bngl import BNGLParser, BNGLModel, RuleExpander

        log.info(f"Initializing BNGLModule: model={self.model_spec}, "
                 f"method={self.method}, max_species={self.max_species}")

        # Load or create model
        if isinstance(self.model_spec, (str, Path)):
            spec = str(self.model_spec)
            if spec == 'egfr_signaling':
                self.model = BNGLModel.egfr_signaling()
            elif spec == 'simple_receptor':
                self.model = BNGLModel.simple_receptor()
            elif Path(spec).exists():
                parser = BNGLParser()
                self.model = parser.parse_file(spec)
            else:
                # Try as built-in model name
                self.model = BNGLModel.simple_receptor()

        log.info(self.model.summary())

        # Expand rules
        expander = RuleExpander(
            molecule_types=self.model.molecule_types,
            rules=self.model.rules,
            parameters=self.model.parameters,
        )

        self.reactions, self.species_list = expander.expand(
            seed_species=self.model.get_seed_species_list(),
            max_species=self.max_species,
        )

        log.info(f"  Network: {len(self.species_list)} species, "
                 f"{len(self.reactions)} reactions")

        # Initialize state (concentrations)
        initial_counts = self.model.get_initial_counts()
        self.state = np.zeros(len(self.species_list), dtype=np.float64)

        for i, sp in enumerate(self.species_list):
            sp_key = str(sp)
            if sp_key in initial_counts:
                self.state[i] = initial_counts[sp_key]

        # Build stoichiometry matrix for ODE
        if self.method == 'ode':
            self._build_stoichiometry()

        # Emit network generated event
        self.emit_event('NETWORK_GENERATED', {
            'n_species': len(self.species_list),
            'n_reactions': len(self.reactions),
        })

        log.info(f"  Initial state: {np.sum(self.state > 0)} non-zero species")

    def _build_stoichiometry(self):
        """Build stoichiometry matrix for ODE simulation."""
        n_species = len(self.species_list)
        n_reactions = len(self.reactions)

        # Species string to index mapping
        self._species_idx = {
            str(sp): i for i, sp in enumerate(self.species_list)
        }

        # Stoichiometry matrix
        self._stoich = np.zeros((n_reactions, n_species), dtype=np.float64)

        # Rate constants
        self._rates = np.zeros(n_reactions, dtype=np.float64)

        # Reactant indices for rate computation
        self._reactant_indices = []

        for r_idx, rxn in enumerate(self.reactions):
            self._rates[r_idx] = rxn.rate

            # Reactant indices
            r_indices = []
            for reactant in rxn.reactants:
                r_key = str(reactant)
                if r_key in self._species_idx:
                    sp_idx = self._species_idx[r_key]
                    r_indices.append(sp_idx)
                    self._stoich[r_idx, sp_idx] -= 1

            self._reactant_indices.append(r_indices)

            # Product indices
            for product in rxn.products:
                p_key = str(product)
                if p_key in self._species_idx:
                    sp_idx = self._species_idx[p_key]
                    self._stoich[r_idx, sp_idx] += 1

    def update(self, dt: float):
        """
        Update BNGL simulation by one timestep.

        Parameters
        ----------
        dt : float
            Time step
        """
        if not self.enabled or self.state is None:
            return

        if self.method == 'ode':
            self._ode_step(dt)
        else:
            self._ssa_step(dt)

        self.current_time += dt
        self.total_steps += 1

    def _ode_step(self, dt: float):
        """ODE integration step using RK4."""
        y = self.state.copy()

        # RK4 integration
        k1 = self._ode_rhs(y)
        k2 = self._ode_rhs(y + 0.5 * dt * k1)
        k3 = self._ode_rhs(y + 0.5 * dt * k2)
        k4 = self._ode_rhs(y + dt * k3)

        self.state = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        self.state = np.maximum(self.state, 0)  # Non-negative

    def _ode_rhs(self, y: np.ndarray) -> np.ndarray:
        """Compute ODE right-hand side (dy/dt)."""
        dydt = np.zeros_like(y)

        for r_idx, r_indices in enumerate(self._reactant_indices):
            # Compute rate
            rate = self._rates[r_idx]
            for sp_idx in r_indices:
                rate *= max(0, y[sp_idx])

            # Apply stoichiometry
            dydt += rate * self._stoich[r_idx, :]

        return dydt

    def _ssa_step(self, dt: float):
        """SSA step using Gillespie algorithm."""
        target_time = self.current_time + dt

        while self.current_time < target_time:
            # Compute propensities
            propensities = np.zeros(len(self.reactions))
            for r_idx, r_indices in enumerate(self._reactant_indices):
                prop = self._rates[r_idx]
                for sp_idx in r_indices:
                    prop *= self.state[sp_idx]
                propensities[r_idx] = max(0, prop)

            a0 = np.sum(propensities)
            if a0 <= 0:
                break

            # Time to next reaction
            tau = -np.log(np.random.random()) / a0
            if self.current_time + tau > target_time:
                break

            self.current_time += tau

            # Choose reaction
            r_val = np.random.random() * a0
            cumsum = 0.0
            chosen = -1
            for r in range(len(propensities)):
                cumsum += propensities[r]
                if cumsum >= r_val:
                    chosen = r
                    break

            if chosen >= 0:
                self.state += self._stoich[chosen, :]
                self.state = np.maximum(self.state, 0)

    def get_state(self) -> Dict[str, Any]:
        """Return current simulation state."""
        if self.state is None:
            return {'initialized': False}

        state = {
            'initialized': True,
            'current_time': self.current_time,
            'total_steps': self.total_steps,
            'n_species': len(self.species_list),
            'n_reactions': len(self.reactions),
            'method': self.method,
            'model': str(self.model_spec),
        }

        # Add observable values
        observables = self.get_observables()
        state.update(observables)

        return state

    def get_observables(self) -> Dict[str, float]:
        """
        Compute observable values.

        Returns
        -------
        Dict[str, float]
            Observable name -> value
        """
        if self.state is None or self.model is None:
            return {}

        # Build species counts dict
        species_counts = {
            str(sp): self.state[i]
            for i, sp in enumerate(self.species_list)
        }

        return self.model.observables.compute_all(
            species_counts, self.species_list
        )

    def get_species_count(self, species_str: str) -> float:
        """
        Get count for a specific species.

        Parameters
        ----------
        species_str : str
            Species string representation

        Returns
        -------
        float
            Species count/concentration
        """
        if species_str in self._species_idx:
            return self.state[self._species_idx[species_str]]
        return 0.0

    def get_all_species_counts(self) -> Dict[str, float]:
        """Get counts for all species."""
        return {
            str(sp): self.state[i]
            for i, sp in enumerate(self.species_list)
            if self.state[i] > 0
        }

    def get_reactions(self) -> List[str]:
        """Get list of all reactions."""
        return [str(rxn) for rxn in self.reactions]


# ── Module Registration ──────────────────────────────────────────────────

def register_bngl_module():
    """Register BNGL module in the module registry."""
    try:
        from modules import module_registry
        module_registry.register('bngl')(BNGLModule)
        log.debug("BNGLModule registered in module registry")
    except ImportError:
        pass


# Auto-register on import
register_bngl_module()
