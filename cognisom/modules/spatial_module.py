#!/usr/bin/env python3
"""
Spatial Module
==============

Handles 3D spatial fields and diffusion.

Features:
- 3D grid for diffusible molecules
- Diffusion solver (explicit)
- Multiple fields (O2, glucose, cytokines)
- Gradient calculation
- Source/sink management
"""

import sys
sys.path.insert(0, '..')

import logging

import numpy as np
from typing import Dict, List, Any, Tuple

from cognisom.core.module_base import SimulationModule
from cognisom.core.event_bus import EventTypes

_log = logging.getLogger(__name__)


class SpatialField:
    """3D diffusible field"""
    
    def __init__(self, name: str, grid_size: Tuple[int, int, int], 
                 resolution: float, diffusion_coeff: float):
        self.name = name
        self.grid_size = grid_size
        self.resolution = resolution  # μm per voxel
        self.diffusion_coeff = diffusion_coeff  # μm²/s
        
        # Concentration field
        self.concentration = np.zeros(grid_size, dtype=np.float32)
        
        # Sources and sinks
        self.sources = []  # [(position, rate), ...]
        self.sinks = []    # [(position, rate), ...]
    
    def add_source(self, position: np.ndarray, rate: float):
        """Add source"""
        idx = self._position_to_index(position)
        self.sources.append((idx, rate))
    
    def add_sink(self, position: np.ndarray, rate: float):
        """Add sink"""
        idx = self._position_to_index(position)
        self.sinks.append((idx, rate))
    
    def clear_sources_sinks(self):
        """Clear all sources and sinks"""
        self.sources.clear()
        self.sinks.clear()
    
    #: Stability limit for the explicit 3D 7-point Laplacian:
    #: D*dt/dx^2 <= 1/6. Above it the scheme amplifies high-frequency
    #: modes every step instead of damping them.
    CFL_LIMIT_3D = 1.0 / 6.0
    #: Fraction of the limit to actually run at.
    CFL_SAFETY = 0.9
    #: Cap on sub-steps per update, so a pathological diffusion number
    #: degrades into a warning rather than an unbounded loop.
    MAX_SUBSTEPS = 512

    SECONDS_PER_HOUR = 3600.0

    def update(self, dt: float):
        """Update field via diffusion.

        ``dt`` is in HOURS (the simulation's time unit) while
        ``diffusion_coeff`` is in um^2/s, so the conversion below is
        required. It was missing: ``D * lap * dt / dx^2`` was evaluated
        with mixed units, understating the transported amount 3600-fold.

        The step is then sub-divided to satisfy the explicit-Euler
        stability limit. Previously there was no check at all -- with the
        oxygen field's D = 2000 um^2/s on a 10 um grid the diffusion
        number is ~7200 against a limit of 1/6, so the scheme was
        unconditionally unstable. It did not visibly blow up only because
        ``np.maximum(concentration, 0)`` clipped the resulting oscillation
        every step, which also silently destroys mass.
        """
        dt_seconds = dt * self.SECONDS_PER_HOUR
        dx2 = self.resolution ** 2

        # Sources and sinks act over the whole interval, once.
        for idx, rate in self.sources:
            if self._is_valid_index(idx):
                self.concentration[idx] += rate * dt

        for idx, rate in self.sinks:
            if self._is_valid_index(idx):
                removal = min(rate * dt, self.concentration[idx])
                self.concentration[idx] -= removal

        if self.diffusion_coeff <= 0 or dt_seconds <= 0:
            np.maximum(self.concentration, 0, out=self.concentration)
            return

        diffusion_number = self.diffusion_coeff * dt_seconds / dx2
        max_stable = self.CFL_LIMIT_3D * self.CFL_SAFETY

        n_sub = max(1, int(np.ceil(diffusion_number / max_stable)))
        coeff = diffusion_number / n_sub

        if n_sub > self.MAX_SUBSTEPS:
            _log.warning(
                "Field '%s': D*dt/dx^2 = %.1f needs %d explicit sub-steps to "
                "stay stable, above the cap of %d. Running %d stable "
                "sub-steps instead, so the field is UNDER-DIFFUSED over this "
                "interval (about %.1f%% of the transport applied). At this "
                "diffusion number the field is effectively at steady state "
                "within one step, so an implicit or steady-state solve is the "
                "right method here, not explicit Euler.",
                self.name, diffusion_number, n_sub, self.MAX_SUBSTEPS,
                self.MAX_SUBSTEPS,
                100.0 * self.MAX_SUBSTEPS * max_stable / diffusion_number,
            )
            n_sub = self.MAX_SUBSTEPS
            # Each sub-step must still satisfy the stability limit --
            # spreading an over-limit coefficient across capped sub-steps
            # leaves every one of them unstable, which overflows to NaN
            # rather than merely being inaccurate.
            coeff = max_stable

        for _ in range(n_sub):
            self.concentration += coeff * self._compute_laplacian()

        # Non-negative
        np.maximum(self.concentration, 0, out=self.concentration)
    
    def _compute_laplacian(self) -> np.ndarray:
        """Compute Laplacian using finite differences"""
        laplacian = np.zeros_like(self.concentration)
        
        # 3D Laplacian: ∇²C = ∂²C/∂x² + ∂²C/∂y² + ∂²C/∂z²
        # Interior points only
        laplacian[1:-1, 1:-1, 1:-1] = (
            self.concentration[2:, 1:-1, 1:-1] + self.concentration[:-2, 1:-1, 1:-1] +
            self.concentration[1:-1, 2:, 1:-1] + self.concentration[1:-1, :-2, 1:-1] +
            self.concentration[1:-1, 1:-1, 2:] + self.concentration[1:-1, 1:-1, :-2] -
            6 * self.concentration[1:-1, 1:-1, 1:-1]
        )
        
        return laplacian
    
    def get_concentration_at(self, position: np.ndarray) -> float:
        """Get concentration at position"""
        idx = self._position_to_index(position)
        if self._is_valid_index(idx):
            return self.concentration[idx]
        return 0.0
    
    def get_gradient_at(self, position: np.ndarray) -> np.ndarray:
        """Get gradient at position"""
        idx = self._position_to_index(position)
        i, j, k = idx
        
        if not self._is_valid_index((i, j, k)):
            return np.zeros(3)
        
        # Central differences
        if i > 0 and i < self.grid_size[0] - 1:
            dx = (self.concentration[i+1, j, k] - self.concentration[i-1, j, k]) / (2 * self.resolution)
        else:
            dx = 0
        
        if j > 0 and j < self.grid_size[1] - 1:
            dy = (self.concentration[i, j+1, k] - self.concentration[i, j-1, k]) / (2 * self.resolution)
        else:
            dy = 0
        
        if k > 0 and k < self.grid_size[2] - 1:
            dz = (self.concentration[i, j, k+1] - self.concentration[i, j, k-1]) / (2 * self.resolution)
        else:
            dz = 0
        
        return np.array([dx, dy, dz])
    
    def _position_to_index(self, position: np.ndarray) -> Tuple[int, int, int]:
        """Convert position to grid index"""
        idx = tuple((position / self.resolution).astype(int))
        return idx
    
    def _is_valid_index(self, idx: Tuple[int, int, int]) -> bool:
        """Check if index is valid"""
        return (0 <= idx[0] < self.grid_size[0] and
                0 <= idx[1] < self.grid_size[1] and
                0 <= idx[2] < self.grid_size[2])


class SpatialModule(SimulationModule):
    """
    Spatial simulation module
    
    Manages:
    - 3D grid
    - Multiple diffusible fields
    - Diffusion solver
    - Gradient calculation
    
    Events Emitted:
    - None (provides service to other modules)
    
    Events Subscribed:
    - CELL_DIVIDED: Update sources/sinks
    - CELL_DIED: Remove sources/sinks
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        
        # Grid configuration
        self.grid_size = config.get('grid_size', (20, 20, 10))
        self.resolution = config.get('resolution', 10.0)  # μm per voxel
        
        # Fields
        self.fields: Dict[str, SpatialField] = {}
        
        # Parameters
        self.update_interval = config.get('update_interval', 0.1)  # hours
        self.time_since_update = 0.0
    
    def initialize(self):
        """Initialize spatial system"""
        print("  Creating spatial fields...")
        
        # Create fields
        self._create_field('oxygen', diffusion_coeff=2000.0)  # μm²/s
        self._create_field('glucose', diffusion_coeff=600.0)
        self._create_field('cytokine', diffusion_coeff=100.0)
        
        # Subscribe to events
        self.subscribe(EventTypes.CELL_DIVIDED, self.on_cell_divided)
        self.subscribe(EventTypes.CELL_DIED, self.on_cell_died)
        
        print(f"    ✓ Grid: {self.grid_size} voxels")
        print(f"    ✓ Resolution: {self.resolution} μm/voxel")
        print(f"    ✓ {len(self.fields)} fields created")
    
    def _create_field(self, name: str, diffusion_coeff: float):
        """Create diffusible field"""
        field = SpatialField(name, self.grid_size, self.resolution, diffusion_coeff)
        self.fields[name] = field
    
    def update(self, dt: float):
        """Update spatial fields"""
        self.time_since_update += dt
        
        # Update at intervals (diffusion is expensive)
        if self.time_since_update >= self.update_interval:
            for field in self.fields.values():
                field.update(self.time_since_update)
            
            self.time_since_update = 0.0
    
    def get_concentration(self, field_name: str, position: np.ndarray) -> float:
        """Get concentration at position"""
        if field_name in self.fields:
            return self.fields[field_name].get_concentration_at(position)
        return 0.0
    
    def get_gradient(self, field_name: str, position: np.ndarray) -> np.ndarray:
        """Get gradient at position"""
        if field_name in self.fields:
            return self.fields[field_name].get_gradient_at(position)
        return np.zeros(3)
    
    def add_source(self, field_name: str, position: np.ndarray, rate: float):
        """Add source to field"""
        if field_name in self.fields:
            self.fields[field_name].add_source(position, rate)
    
    def add_sink(self, field_name: str, position: np.ndarray, rate: float):
        """Add sink to field"""
        if field_name in self.fields:
            self.fields[field_name].add_sink(position, rate)
    
    def get_state(self) -> Dict[str, Any]:
        """Return current spatial state"""
        field_stats = {}
        for name, field in self.fields.items():
            field_stats[name] = {
                'mean': float(np.mean(field.concentration)),
                'max': float(np.max(field.concentration)),
                'min': float(np.min(field.concentration)),
                'n_sources': len(field.sources),
                'n_sinks': len(field.sinks)
            }
        
        return {
            'grid_size': self.grid_size,
            'resolution': self.resolution,
            'n_fields': len(self.fields),
            'fields': field_stats
        }
    
    # Event handlers
    def on_cell_divided(self, data):
        """Handle cell division"""
        pass
    
    def on_cell_died(self, data):
        """Handle cell death"""
        pass


# Test
if __name__ == '__main__':
    from cognisom.core import SimulationEngine, SimulationConfig
    
    print("=" * 70)
    print("Spatial Module Test")
    print("=" * 70)
    print()
    
    # Create engine
    engine = SimulationEngine(SimulationConfig(dt=0.01, duration=0.5))
    
    # Register spatial module
    engine.register_module('spatial', SpatialModule, {
        'grid_size': (20, 20, 10),
        'resolution': 10.0,
        'update_interval': 0.1
    })
    
    # Initialize
    engine.initialize()
    
    # Add some sources
    spatial = engine.modules['spatial']
    
    # Add O2 source (blood vessel)
    spatial.add_source('oxygen', np.array([100, 100, 50]), rate=0.5)
    
    # Add glucose source
    spatial.add_source('glucose', np.array([100, 100, 50]), rate=0.3)
    
    # Add cytokine source (immune cell)
    spatial.add_source('cytokine', np.array([80, 80, 50]), rate=0.1)
    
    print("Added sources")
    print()
    
    # Run simulation
    engine.run()
    
    # Check concentrations
    print("\nConcentrations at different positions:")
    positions = [
        np.array([100, 100, 50]),  # At source
        np.array([120, 120, 50]),  # Near source
        np.array([150, 150, 50])   # Far from source
    ]
    
    for pos in positions:
        O2 = spatial.get_concentration('oxygen', pos)
        glucose = spatial.get_concentration('glucose', pos)
        print(f"  Position {pos}: O2={O2:.3f}, glucose={glucose:.3f}")
    
    print()
    
    # Get results
    print("Spatial State:")
    spatial_state = spatial.get_state()
    for key, value in spatial_state.items():
        if key != 'fields':
            print(f"  {key}: {value}")
    
    print("\nField Statistics:")
    for field_name, stats in spatial_state['fields'].items():
        print(f"  {field_name}:")
        for stat_name, stat_value in stats.items():
            print(f"    {stat_name}: {stat_value}")
    
    print()
    print("=" * 70)
    print("✓ Spatial module working!")
    print("=" * 70)
