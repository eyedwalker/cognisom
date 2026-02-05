#!/usr/bin/env python3
"""
ODE Module
==========

SimulationModule wrapper for GPU-accelerated ODE solving.
Integrates with the simulation engine for cellular dynamics.

VCell Parity Phase 1 - ODE-based cellular dynamics.

Features:
- GPU-accelerated batched ODE solving
- Per-cell heterogeneous parameters
- Event-driven coupling with other modules
- Multiple pre-defined biochemical models

Usage::

    from cognisom.core import SimulationEngine, SimulationConfig
    from cognisom.modules.ode_module import ODEModule

    engine = SimulationEngine(SimulationConfig(dt=0.01, duration=10.0))
    engine.register_module('ode', ODEModule, {
        'system': 'ar_signaling',
        'n_cells': 10000,
        'method': 'bdf',
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


class ODEModule(SimulationModule):
    """
    ODE-based cellular dynamics module.

    Solves ordinary differential equations for intracellular biochemistry
    across a population of cells. Each cell can have heterogeneous parameters.

    Configuration
    -------------
    system : str
        ODE system to use: 'gene_expression', 'ar_signaling', or ODESystem instance
    n_cells : int
        Number of cells to simulate (default: 1000)
    method : str
        Integration method: 'bdf' (stiff), 'adams', 'rk45' (default: 'bdf')
    rtol : float
        Relative tolerance (default: 1e-3)
    atol : float
        Absolute tolerance (default: 1e-6)
    heterogeneity : float
        Coefficient of variation for parameter heterogeneity (default: 0.1)

    Events Emitted
    --------------
    - GENE_EXPRESSED: When gene expression threshold is crossed
    - PROTEIN_LEVEL_CHANGE: Significant protein level change

    Events Subscribed
    -----------------
    - CELL_DIVIDED: Add ODE state for daughter cell
    - CELL_DIED: Remove ODE state for dead cell
    - HYPOXIA_DETECTED: Modify metabolic parameters

    Example
    -------
    >>> engine.register_module('ode', ODEModule, {
    ...     'system': 'ar_signaling',
    ...     'n_cells': 5000,
    ...     'method': 'bdf',
    ... })
    """

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.name = "ODEModule"

        # Configuration with defaults
        self.system_name = self.config.get('system', 'gene_expression')
        self.n_cells = self.config.get('n_cells', 1000)
        self.method = self.config.get('method', 'bdf')
        self.rtol = self.config.get('rtol', 1e-3)
        self.atol = self.config.get('atol', 1e-6)
        self.heterogeneity = self.config.get('heterogeneity', 0.1)

        # State
        self.integrator = None
        self.system = None
        self.current_time = 0.0
        self.cell_params = None

        # Statistics
        self.total_steps = 0
        self.total_rhs_evals = 0

    def initialize(self):
        """Initialize ODE solver and state."""
        from gpu.ode_solver import (
            BatchedODEIntegrator,
            ODESystem,
            ODEState,
        )

        log.info(f"Initializing ODEModule: {self.n_cells} cells, "
                 f"system={self.system_name}, method={self.method}")

        # Create ODE system
        if self.system_name == 'ar_signaling':
            self.system = ODESystem.ar_signaling_pathway()
        else:
            self.system = ODESystem.gene_expression_2species()

        # Create integrator
        self.integrator = BatchedODEIntegrator(
            system=self.system,
            n_cells=self.n_cells,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
        )

        # Initialize with heterogeneous parameters
        self._initialize_cell_parameters()

        # Set initial conditions
        y0 = self._create_initial_conditions()

        self.integrator._state = ODEState(
            y=y0,
            t=0.0,
            dt=0.01,
            order=1,
        )

        # Subscribe to events
        self.subscribe(EventTypes.CELL_DIVIDED, self._on_cell_divided)
        self.subscribe(EventTypes.CELL_DIED, self._on_cell_died)
        self.subscribe(EventTypes.HYPOXIA_DETECTED, self._on_hypoxia)

        log.info(f"  Species: {self.system.species_names}")
        log.info(f"  Parameters: {list(self.system.parameters.keys())}")

    def _initialize_cell_parameters(self):
        """Create heterogeneous per-cell parameters."""
        # Get base parameters from system
        base_params = np.array(list(self.system.parameters.values()), dtype=np.float32)
        n_params = len(base_params)

        # Add log-normal heterogeneity
        if self.heterogeneity > 0:
            # CV = sigma/mu for lognormal: sigma = sqrt(log(1 + CV^2))
            sigma = np.sqrt(np.log(1 + self.heterogeneity ** 2))
            noise = np.random.lognormal(0, sigma, (self.n_cells, n_params))
            self.cell_params = base_params * noise.astype(np.float32)
        else:
            self.cell_params = np.tile(base_params, (self.n_cells, 1))

        # Pass to integrator
        self.integrator.set_cell_parameters(self.cell_params)

    def _create_initial_conditions(self) -> np.ndarray:
        """Create initial conditions for all cells."""
        n_species = self.system.n_species
        y0 = np.zeros((self.n_cells, n_species), dtype=np.float32)

        # Set reasonable initial values based on system
        if self.system_name == 'ar_signaling':
            # AR pathway: start with some AR mRNA and DHT
            y0[:, 0] = 10.0   # AR_mRNA
            y0[:, 1] = 50.0   # AR_protein
            y0[:, 2] = 5.0    # DHT
            y0[:, 3] = 0.0    # AR_DHT (complex forms dynamically)
            y0[:, 4] = 0.0    # PSA_mRNA
            y0[:, 5] = 0.0    # PSA
        else:
            # Gene expression: start with some mRNA
            y0[:, 0] = 10.0   # mRNA
            y0[:, 1] = 100.0  # Protein

        # Add noise
        y0 *= (1 + 0.1 * np.random.randn(self.n_cells, n_species)).astype(np.float32)
        y0 = np.maximum(y0, 0)  # Non-negative

        return y0

    def update(self, dt: float):
        """
        Update ODE state by integrating for dt.

        Parameters
        ----------
        dt : float
            Time step in hours
        """
        if not self.enabled or self.integrator is None:
            return

        # Integrate
        self.integrator.step(dt)
        self.current_time += dt
        self.total_steps += 1
        self.total_rhs_evals += self.integrator._state.n_rhs_evals

        # Check for events to emit
        self._check_events()

    def _check_events(self):
        """Check for conditions that should trigger events."""
        if self.event_bus is None:
            return

        y = self.integrator.get_state()

        # Example: emit event if PSA exceeds threshold (for AR signaling)
        if self.system_name == 'ar_signaling' and y.shape[1] >= 6:
            psa = y[:, 5]
            high_psa_cells = np.where(psa > 100)[0]
            if len(high_psa_cells) > 0:
                self.emit_event('PSA_HIGH', {
                    'cell_indices': high_psa_cells.tolist(),
                    'psa_levels': psa[high_psa_cells].tolist(),
                })

    def get_state(self) -> Dict[str, Any]:
        """Return current ODE state."""
        if self.integrator is None:
            return {'initialized': False}

        y = self.integrator.get_state()

        state = {
            'initialized': True,
            'n_cells': self.n_cells,
            'n_species': self.system.n_species,
            'species_names': self.system.species_names,
            'current_time': self.current_time,
            'total_steps': self.total_steps,
            'method': self.method,
        }

        # Add mean/std for each species
        for i, name in enumerate(self.system.species_names):
            state[f'{name}_mean'] = float(y[:, i].mean())
            state[f'{name}_std'] = float(y[:, i].std())

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
        if self.integrator is None:
            raise RuntimeError("ODE module not initialized")
        return self.integrator.get_species(name)

    def get_all_species(self) -> np.ndarray:
        """
        Get all species values.

        Returns
        -------
        np.ndarray
            (n_cells, n_species) array
        """
        if self.integrator is None:
            raise RuntimeError("ODE module not initialized")
        return self.integrator.get_state()

    def set_cell_parameter(self, cell_idx: int, param_name: str, value: float):
        """
        Set parameter for a specific cell.

        Parameters
        ----------
        cell_idx : int
            Cell index
        param_name : str
            Parameter name
        value : float
            New value
        """
        param_names = list(self.system.parameters.keys())
        if param_name not in param_names:
            raise KeyError(f"Unknown parameter: {param_name}")

        param_idx = param_names.index(param_name)
        self.cell_params[cell_idx, param_idx] = value
        self.integrator.set_cell_parameters(self.cell_params)

    def set_parameter_for_cells(
        self,
        cell_mask: np.ndarray,
        param_name: str,
        value: float
    ):
        """
        Set parameter for multiple cells.

        Parameters
        ----------
        cell_mask : np.ndarray
            Boolean mask or indices of cells to modify
        param_name : str
            Parameter name
        value : float
            New value
        """
        param_names = list(self.system.parameters.keys())
        if param_name not in param_names:
            raise KeyError(f"Unknown parameter: {param_name}")

        param_idx = param_names.index(param_name)
        self.cell_params[cell_mask, param_idx] = value
        self.integrator.set_cell_parameters(self.cell_params)

    # ── Event Handlers ───────────────────────────────────────────────

    def _on_cell_divided(self, data: Dict):
        """Handle cell division event."""
        parent_idx = data.get('cell_index')
        if parent_idx is None or self.integrator is None:
            return

        # Daughter inherits parent's state with slight perturbation
        y = self.integrator._state.y

        if parent_idx < len(y) and self.n_cells < y.shape[0]:
            # Find empty slot or expand
            log.debug(f"Cell division: parent={parent_idx}")
            # For simplicity, daughter takes perturbed state
            # Full implementation would expand arrays

    def _on_cell_died(self, data: Dict):
        """Handle cell death event."""
        cell_idx = data.get('cell_index')
        if cell_idx is None or self.integrator is None:
            return

        # Zero out dead cell's state
        y = self.integrator._state.y
        if cell_idx < len(y):
            y[cell_idx, :] = 0.0
            log.debug(f"Cell died: idx={cell_idx}")

    def _on_hypoxia(self, data: Dict):
        """Handle hypoxia event - modify metabolic parameters."""
        cell_indices = data.get('cell_indices', [])
        if not cell_indices or self.cell_params is None:
            return

        # Under hypoxia, reduce translation rate
        param_names = list(self.system.parameters.keys())
        if 'k_trans' in param_names:
            k_trans_idx = param_names.index('k_trans')
            self.cell_params[cell_indices, k_trans_idx] *= 0.5
            self.integrator.set_cell_parameters(self.cell_params)
            log.debug(f"Hypoxia: reduced k_trans for {len(cell_indices)} cells")


# ── Module Registration ──────────────────────────────────────────────────

def register_ode_module():
    """Register ODE module in the module registry."""
    try:
        from modules import module_registry
        module_registry.register('ode')(ODEModule)
        log.debug("ODEModule registered in module registry")
    except ImportError:
        pass  # Registry not available


# Auto-register on import
register_ode_module()
