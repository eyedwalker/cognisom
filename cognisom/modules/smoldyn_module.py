#!/usr/bin/env python3
"""
Smoldyn Module
==============

SimulationModule wrapper for GPU-accelerated particle-based spatial stochastic
simulation (Smoldyn-style).

VCell Parity Phase 2 - Spatial stochastic dynamics.

Features:
- GPU-accelerated Brownian dynamics
- Particle-based reactions (uni/bimolecular)
- Spatial hashing for efficient neighbor finding
- Compartment support (cytoplasm, nucleus, membrane)
- Pre-defined reaction systems

Usage::

    from cognisom.core import SimulationEngine, SimulationConfig
    from cognisom.modules.smoldyn_module import SmoldynModule

    engine = SimulationEngine(SimulationConfig(dt=0.001, duration=1.0))
    engine.register_module('smoldyn', SmoldynModule, {
        'system': 'enzyme_kinetics',
        'domain_size': (10.0, 10.0, 10.0),
        'max_particles': 100000,
    })
    engine.initialize()
    engine.run()
"""

import sys
sys.path.insert(0, '..')

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.module_base import SimulationModule
from core.event_bus import EventTypes

log = logging.getLogger(__name__)


class SmoldynModule(SimulationModule):
    """
    Particle-based spatial stochastic simulation module.

    Simulates molecular-scale dynamics with:
    - Brownian motion (diffusion)
    - Stochastic reactions (zero/first/second order)
    - Spatial compartments
    - Surface interactions

    Configuration
    -------------
    system : str
        Reaction system: 'simple_binding', 'enzyme_kinetics', 'min_oscillator'
    domain_size : Tuple[float, float, float]
        Simulation domain dimensions in micrometers (default: (10, 10, 10))
    max_particles : int
        Maximum number of particles (default: 100000)
    boundary : str
        Boundary condition: 'reflective' or 'periodic' (default: 'reflective')
    species_counts : Dict[str, int]
        Initial particle counts per species (overrides system defaults)
    diffusion_coefficients : Dict[str, float]
        Override diffusion coefficients (um^2/s)

    Events Emitted
    --------------
    - REACTION_OCCURRED: When a reaction fires
    - PARTICLE_CREATED: New particle from zero-order reaction
    - PARTICLE_DESTROYED: Particle consumed in reaction
    - COMPARTMENT_CROSSED: Particle moves between compartments

    Events Subscribed
    -----------------
    - HYPOXIA_DETECTED: May reduce diffusion rates
    - CELL_DIVIDED: Split particles between daughter cells
    - SIGNALING_EVENT: Add/remove ligand particles

    Example
    -------
    >>> engine.register_module('smoldyn', SmoldynModule, {
    ...     'system': 'enzyme_kinetics',
    ...     'domain_size': (5.0, 5.0, 5.0),
    ...     'species_counts': {'E': 100, 'S': 1000},
    ... })
    """

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.name = "SmoldynModule"

        # Configuration with defaults
        self.system_name = self.config.get('system', 'simple_binding')
        self.domain_size = tuple(self.config.get('domain_size', (10.0, 10.0, 10.0)))
        self.max_particles = self.config.get('max_particles', 100000)
        self.boundary = self.config.get('boundary', 'reflective')
        self.species_counts_override = self.config.get('species_counts', {})
        self.diffusion_override = self.config.get('diffusion_coefficients', {})

        # State
        self.solver = None
        self.system = None
        self.current_time = 0.0

        # Statistics
        self.total_steps = 0
        self.total_reactions = 0
        self.reaction_counts = {}

    def initialize(self):
        """Initialize Smoldyn solver and particle system."""
        from gpu.smoldyn_solver import (
            SmoldynSolver,
            SmoldynSystem,
            SmoldynCompartment,
            BoundaryType,
        )

        log.info(f"Initializing SmoldynModule: system={self.system_name}, "
                 f"domain={self.domain_size}, max_particles={self.max_particles}")

        # Create reaction system
        if self.system_name == 'enzyme_kinetics':
            self.system = SmoldynSystem.enzyme_kinetics()
        elif self.system_name == 'min_oscillator':
            self.system = SmoldynSystem.min_oscillator()
        else:
            self.system = SmoldynSystem.simple_binding()

        # Override compartment bounds with domain_size if provided
        btype = BoundaryType.PERIODIC if self.boundary == 'periodic' else BoundaryType.REFLECT
        self.system.compartment = SmoldynCompartment(
            name="domain",
            bounds=(0, self.domain_size[0], 0, self.domain_size[1], 0, self.domain_size[2]),
            boundary_type=btype,
        )

        # Apply diffusion overrides
        for species_name, D in self.diffusion_override.items():
            for species in self.system.species:
                if species.name == species_name:
                    species.diffusion_coeff = D
                    break

        # Create solver
        self.solver = SmoldynSolver(
            system=self.system,
            n_max_particles=self.max_particles,
        )

        # Add initial particles based on species_counts_override or defaults
        for i, species in enumerate(self.system.species):
            count = self.species_counts_override.get(species.name, 500)  # default 500
            if count > 0:
                positions = np.random.uniform(
                    low=[0, 0, 0],
                    high=self.domain_size,
                    size=(count, 3)
                ).astype(np.float32)
                self.solver.add_particles(species.name, positions)

        # Initialize reaction counters
        for rxn in self.system.reactions:
            self.reaction_counts[rxn.name] = 0

        # Subscribe to events
        self.subscribe(EventTypes.HYPOXIA_DETECTED, self._on_hypoxia)
        self.subscribe(EventTypes.CELL_DIVIDED, self._on_cell_divided)

        log.info(f"  Species: {[s.name for s in self.system.species]}")
        log.info(f"  Reactions: {[r.name for r in self.system.reactions]}")
        log.info(f"  Initial particles: {self.solver.particles.n_alive}")

    def update(self, dt: float):
        """
        Update particle system by one timestep.

        Parameters
        ----------
        dt : float
            Time step in seconds (typically 1e-6 to 1e-3)
        """
        if not self.enabled or self.solver is None:
            return

        # Step the solver
        result = self.solver.step(dt)
        self.current_time += dt
        self.total_steps += 1

        # Track reactions
        if hasattr(result, 'reactions_fired'):
            for rxn_name, count in result.reactions_fired.items():
                self.total_reactions += count
                self.reaction_counts[rxn_name] = self.reaction_counts.get(rxn_name, 0) + count

        # Emit events for significant changes
        self._check_events()

    def _check_events(self):
        """Check for conditions that should trigger events."""
        if self.event_bus is None:
            return

        # Emit event if particle counts changed significantly
        # (Implementation depends on what's useful for other modules)
        pass

    def get_state(self) -> Dict[str, Any]:
        """Return current simulation state."""
        if self.solver is None:
            return {'initialized': False}

        state = {
            'initialized': True,
            'current_time': self.current_time,
            'total_steps': self.total_steps,
            'total_reactions': self.total_reactions,
            'n_particles': self.solver.particles.n_alive,
            'max_particles': self.max_particles,
            'domain_size': self.domain_size,
            'boundary': self.boundary,
            'system': self.system_name,
        }

        # Add species counts
        species_counts = self.get_species_counts()
        for name, count in species_counts.items():
            state[f'{name}_count'] = count

        # Add reaction counts
        for rxn_name, count in self.reaction_counts.items():
            state[f'rxn_{rxn_name}'] = count

        return state

    def get_species_counts(self) -> Dict[str, int]:
        """
        Get particle counts for each species.

        Returns
        -------
        Dict[str, int]
            Species name -> count mapping
        """
        if self.solver is None:
            return {}

        counts = {}
        particles = self.solver.particles

        for i, species in enumerate(self.system.species):
            mask = (particles.species == i) & particles.alive
            counts[species.name] = int(np.sum(mask))

        return counts

    def get_particle_positions(self, species_name: Optional[str] = None) -> np.ndarray:
        """
        Get positions of particles.

        Parameters
        ----------
        species_name : str, optional
            Filter by species name. If None, returns all particles.

        Returns
        -------
        np.ndarray
            (N, 3) array of positions
        """
        if self.solver is None:
            return np.zeros((0, 3), dtype=np.float32)

        particles = self.solver.particles
        mask = particles.alive.copy()

        if species_name is not None:
            species_idx = None
            for i, s in enumerate(self.system.species):
                if s.name == species_name:
                    species_idx = i
                    break
            if species_idx is None:
                raise KeyError(f"Unknown species: {species_name}")
            mask &= (particles.species == species_idx)

        return particles.positions[mask].copy()

    def get_species_positions(self) -> Dict[str, np.ndarray]:
        """
        Get positions organized by species.

        Returns
        -------
        Dict[str, np.ndarray]
            Species name -> (N, 3) positions array
        """
        result = {}
        for species in self.system.species:
            result[species.name] = self.get_particle_positions(species.name)
        return result

    def add_particles(
        self,
        species_name: str,
        count: int,
        positions: Optional[np.ndarray] = None,
        compartment: Optional[str] = None,
    ):
        """
        Add particles to the simulation.

        Parameters
        ----------
        species_name : str
            Species type for new particles
        count : int
            Number of particles to add
        positions : np.ndarray, optional
            (count, 3) array of positions. If None, random positions used.
        compartment : str, optional
            Restrict to compartment bounds
        """
        if self.solver is None:
            raise RuntimeError("SmoldynModule not initialized")

        # Find species index
        species_idx = None
        for i, s in enumerate(self.system.species):
            if s.name == species_name:
                species_idx = i
                break
        if species_idx is None:
            raise KeyError(f"Unknown species: {species_name}")

        # Generate positions if not provided
        if positions is None:
            positions = np.random.uniform(
                low=[0, 0, 0],
                high=self.domain_size,
                size=(count, 3)
            ).astype(np.float32)

        # Add to particle system
        self.solver.add_particles(species_idx, positions)

        log.debug(f"Added {count} particles of species {species_name}")

    def remove_particles(
        self,
        species_name: Optional[str] = None,
        count: Optional[int] = None,
        indices: Optional[np.ndarray] = None,
    ):
        """
        Remove particles from the simulation.

        Parameters
        ----------
        species_name : str, optional
            Remove from this species only
        count : int, optional
            Number to remove (random selection)
        indices : np.ndarray, optional
            Specific particle indices to remove
        """
        if self.solver is None:
            raise RuntimeError("SmoldynModule not initialized")

        particles = self.solver.particles

        if indices is not None:
            particles.alive[indices] = False
            return

        if species_name is not None:
            species_idx = None
            for i, s in enumerate(self.system.species):
                if s.name == species_name:
                    species_idx = i
                    break
            if species_idx is None:
                raise KeyError(f"Unknown species: {species_name}")

            mask = (particles.species == species_idx) & particles.alive
            valid_indices = np.where(mask)[0]
        else:
            valid_indices = np.where(particles.alive)[0]

        if count is not None and count < len(valid_indices):
            to_remove = np.random.choice(valid_indices, size=count, replace=False)
        else:
            to_remove = valid_indices

        particles.alive[to_remove] = False

    def set_diffusion_coefficient(self, species_name: str, D: float):
        """
        Change diffusion coefficient for a species.

        Parameters
        ----------
        species_name : str
            Species name
        D : float
            New diffusion coefficient (um^2/s)
        """
        for species in self.system.species:
            if species.name == species_name:
                species.diffusion_coefficient = D
                log.debug(f"Set {species_name} D = {D}")
                return
        raise KeyError(f"Unknown species: {species_name}")

    def set_reaction_rate(self, reaction_name: str, rate: float):
        """
        Change reaction rate constant.

        Parameters
        ----------
        reaction_name : str
            Reaction name
        rate : float
            New rate constant
        """
        for rxn in self.system.reactions:
            if rxn.name == reaction_name:
                rxn.rate = rate
                log.debug(f"Set {reaction_name} rate = {rate}")
                return
        raise KeyError(f"Unknown reaction: {reaction_name}")

    def get_concentration(self, species_name: str) -> float:
        """
        Get concentration of a species.

        Parameters
        ----------
        species_name : str
            Species name

        Returns
        -------
        float
            Concentration in molecules/um^3
        """
        count = self.get_species_counts().get(species_name, 0)
        volume = self.domain_size[0] * self.domain_size[1] * self.domain_size[2]
        return count / volume

    def get_spatial_distribution(
        self,
        species_name: str,
        bins: int = 20,
        axis: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get spatial distribution histogram along an axis.

        Parameters
        ----------
        species_name : str
            Species name
        bins : int
            Number of bins
        axis : int
            Axis (0=x, 1=y, 2=z)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (bin_edges, counts) histogram data
        """
        positions = self.get_particle_positions(species_name)
        if len(positions) == 0:
            return np.linspace(0, self.domain_size[axis], bins + 1), np.zeros(bins)

        counts, edges = np.histogram(
            positions[:, axis],
            bins=bins,
            range=(0, self.domain_size[axis])
        )
        return edges, counts

    # ── Event Handlers ───────────────────────────────────────────────

    def _on_hypoxia(self, data: Dict):
        """Handle hypoxia event - reduce diffusion."""
        severity = data.get('severity', 0.5)
        for species in self.system.species:
            # Reduce diffusion under hypoxia
            species.diffusion_coefficient *= (1.0 - 0.5 * severity)
        log.debug(f"Hypoxia: reduced diffusion by {50 * severity:.0f}%")

    def _on_cell_divided(self, data: Dict):
        """Handle cell division - partition particles."""
        # For simplicity, randomly assign half to daughter
        # In practice, would use spatial partitioning
        particles = self.solver.particles
        alive_indices = np.where(particles.alive)[0]
        n_to_remove = len(alive_indices) // 2

        if n_to_remove > 0:
            to_daughter = np.random.choice(
                alive_indices, size=n_to_remove, replace=False
            )
            particles.alive[to_daughter] = False
            log.debug(f"Cell division: {n_to_remove} particles to daughter")


# ── Module Registration ──────────────────────────────────────────────────

def register_smoldyn_module():
    """Register Smoldyn module in the module registry."""
    try:
        from modules import module_registry
        module_registry.register('smoldyn')(SmoldynModule)
        log.debug("SmoldynModule registered in module registry")
    except ImportError:
        pass  # Registry not available


# Auto-register on import
register_smoldyn_module()
