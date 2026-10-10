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

from cognisom.core.module_base import SimulationModule
from cognisom.core.event_bus import EventTypes

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
        # Exposed so the iteration-limit refusal below can name it, and
        # so a genuinely deep network can be given more room.
        self.max_iterations = self.config.get('max_iterations', 100)
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
        from cognisom.bngl import BNGLParser, BNGLModel, RuleExpander

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
            max_iterations=self.max_iterations,
        )

        log.info(f"  Network: {len(self.species_list)} species, "
                 f"{len(self.reactions)} reactions")

        # Refuse to integrate a truncated network.
        #
        # Expansion that terminates on the species cap has not finished: the
        # reaction set is a prefix of something larger, mass is not conserved
        # across it, and integrating it produces confident numbers for a model
        # nobody wrote. The expander no longer runs away on a well-posed rule
        # set -- the bundled simple_receptor model converges at 3 species and
        # egfr_signaling at 15 -- so reaching this now means the network may
        # genuinely be unbounded. It stays as a backstop, because a rule set
        # really can be.
        if getattr(expander, 'hit_species_limit', False):
            raise RuntimeError(
                f"BNGL rule expansion hit the species limit "
                f"({self.max_species}) instead of converging, so the network "
                f"is truncated: {len(self.species_list)} species and "
                f"{len(self.reactions)} reactions were generated from "
                f"{len(self.model.rules)} rule(s). Integrating a truncated "
                f"network would give meaningless results, so this refuses to "
                f"run. The rule set may be genuinely unbounded -- check for a "
                f"molecule carrying two binding sites for the same partner, "
                f"which polymerises without limit -- or raise max_species if "
                f"the network really is this large."
            )

        # Refuse a network truncated by the ITERATION limit.
        #
        # The species cap only fires if the network reaches max_species
        # within max_iterations. A rule set that adds few species per
        # iteration -- monomer addition onto a growing filament, where one
        # reactant pattern names every site free so only a monomer matches --
        # is unbounded yet never reaches the cap. This is the other half of
        # the same refusal, not a duplicate of it.
        if getattr(expander, 'hit_iteration_limit', False):
            raise RuntimeError(
                f"BNGL rule expansion stopped at the iteration limit "
                f"({self.max_iterations}) without converging, so the network "
                f"is truncated: {len(self.species_list)} species and "
                f"{len(self.reactions)} reactions from "
                f"{len(self.model.rules)} rule(s). Species were still being "
                f"added when the loop ended, which usually means the rule set "
                f"is unbounded -- a pattern that matches its own product, or "
                f"monomer addition onto a growing complex. Integrating a "
                f"truncated network would give meaningless results. Raise "
                f"max_iterations only if the network genuinely converges "
                f"beyond it."
            )

        # Refuse a network whose species identity is not exact.
        #
        # Species are deduplicated on a canonical string built from a
        # Weisfeiler-Leman refinement of the complex. Within each tie class
        # the smallest rendering over all orderings is taken, which is
        # exact, but a complex with more than Species.CANON_ORDERING_BUDGET
        # orderings in one class falls back to an ordering that is not
        # invariant, so two spellings of one complex could be counted as
        # two species. That inflates every count derived from it, so refuse
        # rather than approximate.
        if getattr(expander, 'canonicalization_degraded', False):
            raise RuntimeError(
                "BNGL rule expansion produced at least one complex too "
                "symmetric to canonicalize exactly, so two spellings of the "
                "same species may have been counted twice and every count "
                "derived from the network would be inflated. This needs a "
                "complete canonical labelling (see "
                "Species.CANON_ORDERING_BUDGET in cognisom/bngl/molecules.py), "
                "not a larger max_species."
            )

        # Initialize state (concentrations).
        #
        # Seed amounts are keyed on the canonical string -- the same identity
        # the expander interns species on -- and SUMMED over seed lines.
        # Keying on str(sp) silently dropped a seed whenever two seed lines
        # spell one complex differently, which BNGL permits: now that species
        # identity is invariant to bond ids, molecule order and
        # within-molecule component order, the two spellings collapse to one
        # species but only one spelling matched. Measured on a model seeding
        # `L(r!1).R(l!1,Y~U) 300` and `L(r!1).R(Y~U,l!1) 700`: 300 of the
        # declared 1000 landed, so 70% of the mass was absent before the first
        # step -- a regression against the pre-canonicalisation code.
        #
        # model.seed_species carries the parsed Species, so key off that
        # rather than off get_initial_counts(), which has already stringified.
        self.state = np.zeros(len(self.species_list), dtype=np.float64)

        def _identity(spec):
            canon = getattr(spec, 'canonical_string', None)
            return canon() if callable(canon) else str(spec)

        seeded: Dict[str, float] = {}
        for ss in self.model.seed_species:
            key = _identity(ss.species)
            seeded[key] = seeded.get(key, 0.0) + float(ss.count)

        landed = set()
        for i, sp in enumerate(self.species_list):
            key = _identity(sp)
            if key in seeded:
                self.state[i] = seeded[key]
                landed.add(key)

        # A declared seed that matched no expanded species is mass the model
        # asked for and the network does not contain. Refuse rather than
        # integrate a system that silently starts light.
        unlanded = sorted(set(seeded) - landed)
        if unlanded:
            raise RuntimeError(
                f"{len(unlanded)} of {len(seeded)} declared seed species "
                f"matched no species in the expanded network, so their mass "
                f"would be silently absent from the initial state: "
                f"{unlanded[:5]}. This usually means a seed pattern is not "
                f"reachable from the rules, or is spelled in a way the parser "
                f"read differently."
            )

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

        # Species string to index mapping. This is keyed on str(sp), while
        # the expander dedups on Species.canonical_string(), so a collision
        # here would silently merge two species and a miss would silently
        # drop a reaction endpoint -- which is how 390 of 900 endpoints went
        # missing and the integrator was handed non-conservative rows.
        self._species_idx = {
            str(sp): i for i, sp in enumerate(self.species_list)
        }
        if len(self._species_idx) != n_species:
            raise RuntimeError(
                f"two of the {n_species} species render to the same string, "
                f"so the stoichiometry index is incomplete "
                f"({len(self._species_idx)} distinct keys). This is a bug in "
                f"the species naming, not in the model."
            )
        missing = sorted({
            str(e)
            for rxn in self.reactions
            for e in list(rxn.reactants) + list(rxn.products)
            if str(e) not in self._species_idx
        })
        if missing:
            raise RuntimeError(
                f"{len(missing)} reaction endpoint(s) are not in the species "
                f"index, so their stoichiometry would be dropped and mass "
                f"would not be conserved: {missing[:3]}"
            )

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
        from cognisom.modules import module_registry
        module_registry.register('bngl')(BNGLModule)
        log.debug("BNGLModule registered in module registry")
    except ImportError:
        pass


# Auto-register on import
register_bngl_module()
