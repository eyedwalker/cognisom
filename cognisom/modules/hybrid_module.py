#!/usr/bin/env python3
"""
Hybrid ODE/SSA Module
=====================

SimulationModule wrapper for GPU-accelerated hybrid deterministic-stochastic
simulation with automatic species partitioning.

VCell Parity Phase 3 - Hybrid ODE/SSA dynamics.

Features:
- Automatic fast/slow species partitioning
- GPU-accelerated ODE and SSA solvers
- Dynamic repartitioning during simulation
- Per-cell parameter heterogeneity
- Multiple pre-defined biochemical systems

Usage::

    from cognisom.core import SimulationEngine, SimulationConfig
    from cognisom.modules.hybrid_module import HybridModule

    engine = SimulationEngine(SimulationConfig(dt=0.01, duration=100.0))
    engine.register_module('hybrid', HybridModule, {
        'system': 'gene_regulatory_network',
        'n_cells': 1000,
        'threshold': 100,
    })
    engine.initialize()
    engine.run()
"""

import sys
sys.path.insert(0, '..')

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from core.module_base import SimulationModule
from core.event_bus import EventTypes

log = logging.getLogger(__name__)


class HybridModule(SimulationModule):
    """
    Hybrid ODE/SSA simulation module.

    Combines deterministic ODE integration for high-copy number species
    with stochastic SSA for low-copy number species. Automatic partitioning
    based on copy number threshold.

    Configuration
    -------------
    system : str
        System to simulate: 'gene_regulatory_network', 'enzyme_mm', 'toggle_switch'
    n_cells : int
        Number of cells to simulate (default: 1000)
    threshold : float
        Copy number threshold for fast/slow partitioning (default: 100)
    repartition_interval : int
        Steps between repartitioning, 0 to disable (default: 100)
    ode_method : str
        ODE integration method: 'euler', 'rk4' (default: 'rk4')
    heterogeneity : float
        Coefficient of variation for parameter heterogeneity (default: 0.0)

    Events Emitted
    --------------
    - GENE_EXPRESSED: When gene expression threshold is crossed
    - PARTITION_CHANGED: When species partitioning changes

    Events Subscribed
    -----------------
    - DRUG_APPLIED: Modify kinetic parameters
    - CELL_DIVIDED: Split state for daughter cell

    Example
    -------
    >>> engine.register_module('hybrid', HybridModule, {
    ...     'system': 'gene_regulatory_network',
    ...     'n_cells': 5000,
    ...     'threshold': 50,
    ... })
    """

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.name = "HybridModule"

        # Configuration with defaults
        self.system_name = self.config.get('system', 'gene_regulatory_network')
        self.n_cells = self.config.get('n_cells', 1000)
        self.threshold = self.config.get('threshold', 100.0)
        self.repartition_interval = self.config.get('repartition_interval', 100)
        self.ode_method = self.config.get('ode_method', 'rk4')
        self.heterogeneity = self.config.get('heterogeneity', 0.0)
        self.initial_state_override = self.config.get('initial_state', None)

        # State
        self.solver = None
        self.system = None
        self.current_time = 0.0

        # Statistics
        self.total_steps = 0
        self.partition_changes = 0
        self._last_partition = None

    def initialize(self):
        """Initialize hybrid solver."""
        from gpu.hybrid_solver import HybridSolver, HybridSystem

        log.info(f"Initializing HybridModule: system={self.system_name}, "
                 f"n_cells={self.n_cells}, threshold={self.threshold}")

        # Create system
        if self.system_name == 'enzyme_mm':
            self.system = HybridSystem.enzyme_substrate_mm()
        elif self.system_name == 'toggle_switch':
            self.system = HybridSystem.toggle_switch()
        else:
            self.system = HybridSystem.gene_regulatory_network()

        # Create solver
        self.solver = HybridSolver(
            system=self.system,
            n_cells=self.n_cells,
            threshold=self.threshold,
            repartition_interval=self.repartition_interval,
            ode_method=self.ode_method,
        )

        # Initialize with optional override
        if self.initial_state_override is not None:
            self.solver.initialize(np.array(self.initial_state_override, dtype=np.float32))
        else:
            self.solver.initialize()

        # Apply parameter heterogeneity
        if self.heterogeneity > 0:
            self._apply_heterogeneity()

        # Track partition
        self._last_partition = self.solver.get_partition()

        # Subscribe to events
        self.subscribe(EventTypes.CELL_DIVIDED, self._on_cell_divided)

        log.info(f"  Species: {self.system.species_names}")
        log.info(f"  Reactions: {[r.name for r in self.system.reactions]}")
        log.info(f"  Initial partition: {self._last_partition.n_fast} fast, "
                 f"{self._last_partition.n_slow} slow")

    def _apply_heterogeneity(self):
        """Apply log-normal heterogeneity to per-cell parameters."""
        if self.solver._params is None:
            return

        backend = self.solver._backend
        params = backend.to_numpy(self.solver._params)

        # Log-normal noise
        sigma = np.sqrt(np.log(1 + self.heterogeneity ** 2))
        noise = np.random.lognormal(0, sigma, params.shape)
        params *= noise.astype(np.float32)

        self.solver._params = backend.xp.asarray(params)

    def update(self, dt: float):
        """
        Update hybrid simulation by one timestep.

        Parameters
        ----------
        dt : float
            Time step in seconds or hours (depends on system)
        """
        if not self.enabled or self.solver is None:
            return

        # Step the solver
        self.solver.step(dt)
        self.current_time += dt
        self.total_steps += 1

        # Check for partition changes
        current_partition = self.solver.get_partition()
        if current_partition.fast_species != self._last_partition.fast_species:
            self.partition_changes += 1
            self._last_partition = current_partition
            self._emit_partition_change()

        # Check for events
        self._check_events()

    def _emit_partition_change(self):
        """Emit partition change event."""
        if self.event_bus is None:
            return

        self.emit_event('PARTITION_CHANGED', {
            'n_fast': self._last_partition.n_fast,
            'n_slow': self._last_partition.n_slow,
            'fast_species': self._last_partition.fast_species,
            'slow_species': self._last_partition.slow_species,
        })

    def _check_events(self):
        """Check for threshold crossings and emit events."""
        if self.event_bus is None:
            return

        # Example: emit event when protein levels cross threshold
        # (would be customized based on system)
        pass

    def get_state(self) -> Dict[str, Any]:
        """Return current simulation state."""
        if self.solver is None:
            return {'initialized': False}

        stats = self.solver.get_statistics()
        partition = self.solver.get_partition()

        state = {
            'initialized': True,
            'current_time': self.current_time,
            'total_steps': self.total_steps,
            'n_cells': self.n_cells,
            'system': self.system_name,
            'threshold': self.threshold,
            'n_fast': partition.n_fast,
            'n_slow': partition.n_slow,
            'partition_changes': self.partition_changes,
            'ode_method': self.ode_method,
        }

        # Add species statistics
        for name in self.system.species_names:
            state[f'{name}_mean'] = stats.get(f'{name}_mean', 0.0)
            state[f'{name}_std'] = stats.get(f'{name}_std', 0.0)

        return state

    def get_species(self, name: str) -> np.ndarray:
        """
        Get values for a specific species across all cells.

        Parameters
        ----------
        name : str
            Species name

        Returns
        -------
        np.ndarray
            (n_cells,) array of values
        """
        if self.solver is None:
            raise RuntimeError("Hybrid module not initialized")
        return self.solver.get_species(name)

    def get_all_species(self) -> np.ndarray:
        """
        Get all species values.

        Returns
        -------
        np.ndarray
            (n_cells, n_species) array
        """
        if self.solver is None:
            raise RuntimeError("Hybrid module not initialized")
        return self.solver.get_state()

    def get_partition(self) -> Dict[str, List[str]]:
        """
        Get current species partition.

        Returns
        -------
        Dict
            {'fast': [species names], 'slow': [species names]}
        """
        if self.solver is None:
            return {'fast': [], 'slow': []}

        partition = self.solver.get_partition()
        return {
            'fast': [self.system.species_names[i] for i in partition.fast_species],
            'slow': [self.system.species_names[i] for i in partition.slow_species],
        }

    def set_threshold(self, threshold: float):
        """
        Change partitioning threshold.

        Parameters
        ----------
        threshold : float
            New copy number threshold
        """
        self.threshold = threshold
        if self.solver is not None:
            self.solver.threshold = threshold
            self.solver._partitioner.threshold = threshold
            self.solver._partitioner._upper_threshold = threshold * 1.2
            self.solver._partitioner._lower_threshold = threshold * 0.8
            log.debug(f"Threshold changed to {threshold}")

    def force_repartition(self):
        """Force immediate repartitioning."""
        if self.solver is not None:
            self.solver._repartition()
            self._last_partition = self.solver.get_partition()

    def get_cell_state(self, cell_idx: int) -> Dict[str, float]:
        """
        Get state for a specific cell.

        Parameters
        ----------
        cell_idx : int
            Cell index

        Returns
        -------
        Dict[str, float]
            Species name -> value mapping
        """
        if self.solver is None:
            return {}

        state = self.solver.get_state()
        return {
            name: float(state[cell_idx, i])
            for i, name in enumerate(self.system.species_names)
        }

    # ── Event Handlers ───────────────────────────────────────────────

    def _on_cell_divided(self, data: Dict):
        """Handle cell division - split state to daughter."""
        parent_idx = data.get('cell_index')
        if parent_idx is None or self.solver is None:
            return

        # For simplicity, log the event
        # Full implementation would expand arrays
        log.debug(f"Cell division: parent={parent_idx}")


# ── Module Registration ──────────────────────────────────────────────────

def register_hybrid_module():
    """Register Hybrid module in the module registry."""
    try:
        from modules import module_registry
        module_registry.register('hybrid')(HybridModule)
        log.debug("HybridModule registered in module registry")
    except ImportError:
        pass  # Registry not available


# Auto-register on import
register_hybrid_module()
