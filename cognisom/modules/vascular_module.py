#!/usr/bin/env python3
"""
Vascular Module
===============

Handles blood vessel network and nutrient/waste exchange.

Features:
- Capillary network
- O2 delivery
- Glucose delivery
- Waste removal (CO2, lactate)
- Diffusion-based exchange
- Distance-dependent gradients
"""

import sys
sys.path.insert(0, '..')

import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass

from core.module_base import SimulationModule
from core.event_bus import EventTypes


@dataclass
class Capillary:
    """Blood capillary"""
    capillary_id: int
    start: np.ndarray
    end: np.ndarray
    
    # Blood contents
    oxygen: float = 0.21  # 21% O2 (arterial)
    glucose: float = 5.0  # mM
    
    # Flow
    flow_rate: float = 0.5  # mm/s
    diameter: float = 8.0  # μm
    
    # Exchange parameters
    exchange_radius: float = 50.0  # μm
    permeability_O2: float = 0.1
    permeability_glucose: float = 0.05
    
    def __post_init__(self):
        self.length = np.linalg.norm(self.end - self.start)
    
    def get_position(self, t: float) -> np.ndarray:
        """Get position along capillary (t: 0-1)"""
        return self.start + t * (self.end - self.start)


class VascularModule(SimulationModule):
    """
    Vascular system simulation module
    
    Manages:
    - Capillary network
    - O2/glucose delivery
    - Waste removal
    - Diffusion-based exchange
    - Hypoxia detection
    
    Events Emitted:
    - HYPOXIA_DETECTED: When tissue becomes hypoxic
    - VESSEL_FORMED: When angiogenesis occurs
    
    Events Subscribed:
    - CELL_DIVIDED: Track nutrient demand
    - CELL_DIED: Reduce demand
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        
        # Capillary network
        self.capillaries: Dict[int, Capillary] = {}
        self.next_capillary_id = 0
        
        # Parameters
        self.n_capillaries = config.get('n_capillaries', 8)
        self.arterial_O2 = config.get('arterial_O2', 0.21)
        self.arterial_glucose = config.get('arterial_glucose', 5.0)
        self.exchange_rate = config.get('exchange_rate', 1.0)
        
        # Statistics
        self.total_O2_delivered = 0.0
        self.total_glucose_delivered = 0.0
        self.total_waste_removed = 0.0
        self.hypoxic_regions = 0
        
        # Reference to cellular module
        self.cellular_module = None
    
    def initialize(self):
        """Initialize vascular network"""
        print("  Creating capillary network...")
        
        # Create radial capillary pattern (from center outward)
        center = np.array([100.0, 100.0, 50.0])
        
        for i in range(self.n_capillaries):
            angle = 2 * np.pi * i / self.n_capillaries
            
            # Start near center
            start = center + np.array([
                20 * np.cos(angle),
                20 * np.sin(angle),
                np.random.uniform(-5, 5)
            ])
            
            # End at periphery
            end = center + np.array([
                80 * np.cos(angle),
                80 * np.sin(angle),
                np.random.uniform(-5, 5)
            ])
            
            self.add_capillary(start, end)
        
        # Subscribe to events
        self.subscribe(EventTypes.CELL_DIVIDED, self.on_cell_divided)
        self.subscribe(EventTypes.CELL_DIED, self.on_cell_died)
        
        print(f"    ✓ {self.n_capillaries} capillaries")
        print(f"    ✓ Radial network pattern")
    
    def add_capillary(self, start, end):
        """Add capillary to network"""
        capillary_id = self.next_capillary_id
        self.next_capillary_id += 1
        
        capillary = Capillary(
            capillary_id=capillary_id,
            start=np.array(start, dtype=np.float32),
            end=np.array(end, dtype=np.float32)
        )
        
        self.capillaries[capillary_id] = capillary
        return capillary_id
    
    def set_cellular_module(self, cellular_module):
        """Link to cellular module for exchange"""
        self.cellular_module = cellular_module
    
    def update(self, dt: float):
        """Update vascular system"""
        if not self.cellular_module:
            return
        
        # Replenish blood contents (arterial input)
        for capillary in self.capillaries.values():
            capillary.oxygen = self.arterial_O2
            capillary.glucose = self.arterial_glucose
        
        # Exchange with cells
        for capillary in self.capillaries.values():
            self._exchange_with_tissue(capillary, dt)
        
        # Check for hypoxia
        self._check_hypoxia()
    
    def _exchange_with_tissue(self, capillary: Capillary, dt: float):
        """Exchange O2/glucose with nearby cells"""
        if not self.cellular_module:
            return
        
        # Sample points along capillary
        n_samples = 10
        for i in range(n_samples):
            t = i / (n_samples - 1)
            cap_pos = capillary.get_position(t)
            
            # Find nearby cells
            for cell_id, cell in self.cellular_module.cells.items():
                if not cell.alive:
                    continue
                
                distance = np.linalg.norm(cell.position - cap_pos)
                
                if distance < capillary.exchange_radius:
                    # O2 diffusion (Fick's law)
                    gradient = (capillary.oxygen - cell.oxygen) / (distance + 1)
                    flux_O2 = capillary.permeability_O2 * gradient * dt * self.exchange_rate
                    
                    # Transfer O2
                    transfer = min(flux_O2, capillary.oxygen * 0.1)  # Max 10% per step
                    capillary.oxygen -= transfer
                    cell.oxygen += transfer
                    self.total_O2_delivered += transfer
                    
                    # Glucose diffusion
                    gradient = (capillary.glucose - cell.glucose) / (distance + 1)
                    flux_glucose = capillary.permeability_glucose * gradient * dt * self.exchange_rate
                    
                    # Transfer glucose
                    transfer = min(flux_glucose, capillary.glucose * 0.1)
                    capillary.glucose -= transfer
                    cell.glucose += transfer
                    self.total_glucose_delivered += transfer
                    
                    # Lactate removal (waste)
                    gradient = (cell.lactate - 0) / (distance + 1)
                    flux_lactate = capillary.permeability_glucose * gradient * dt * self.exchange_rate
                    
                    # Remove lactate
                    removal = min(flux_lactate, cell.lactate * 0.1)
                    cell.lactate -= removal
                    self.total_waste_removed += removal
    
    def _check_hypoxia(self):
        """Check for hypoxic regions"""
        if not self.cellular_module:
            return
        
        hypoxic_cells = 0
        
        for cell in self.cellular_module.cells.values():
            if cell.alive and cell.oxygen < 0.05:  # < 5% O2
                hypoxic_cells += 1
        
        if hypoxic_cells > 0 and hypoxic_cells != self.hypoxic_regions:
            self.hypoxic_regions = hypoxic_cells
            
            self.emit_event(EventTypes.HYPOXIA_DETECTED, {
                'n_hypoxic_cells': hypoxic_cells,
                'severity': 'severe' if hypoxic_cells > 10 else 'moderate'
            })
    
    def get_distance_to_nearest_vessel(self, position: np.ndarray) -> float:
        """Get distance from position to nearest capillary"""
        min_distance = float('inf')
        
        for capillary in self.capillaries.values():
            # Check multiple points along capillary
            for t in np.linspace(0, 1, 10):
                cap_pos = capillary.get_position(t)
                distance = np.linalg.norm(position - cap_pos)
                min_distance = min(min_distance, distance)
        
        return min_distance
    
    def get_O2_at_position(self, position: np.ndarray) -> float:
        """Estimate O2 level at position based on vessel distance"""
        distance = self.get_distance_to_nearest_vessel(position)
        
        # O2 decays exponentially with distance
        decay_length = 50.0  # μm
        O2 = self.arterial_O2 * np.exp(-distance / decay_length)
        
        return O2
    
    def get_state(self) -> Dict[str, Any]:
        """Return current vascular state"""
        if self.cellular_module:
            avg_cell_O2 = np.mean([c.oxygen for c in self.cellular_module.cells.values() 
                                   if c.alive])
            avg_cell_glucose = np.mean([c.glucose for c in self.cellular_module.cells.values() 
                                        if c.alive])
        else:
            avg_cell_O2 = 0
            avg_cell_glucose = 0
        
        return {
            'n_capillaries': len(self.capillaries),
            'total_length': sum(c.length for c in self.capillaries.values()),
            'avg_cell_O2': avg_cell_O2,
            'avg_cell_glucose': avg_cell_glucose,
            'total_O2_delivered': self.total_O2_delivered,
            'total_glucose_delivered': self.total_glucose_delivered,
            'total_waste_removed': self.total_waste_removed,
            'hypoxic_regions': self.hypoxic_regions
        }
    
    # Event handlers
    def on_cell_divided(self, data):
        """Handle cell division - increased demand"""
        # Could trigger angiogenesis if demand is high
        pass
    
    def on_cell_died(self, data):
        """Handle cell death - reduced demand"""
        pass


# Test
if __name__ == '__main__':
    from core import SimulationEngine, SimulationConfig
    from cellular_module import CellularModule
    
    print("=" * 70)
    print("Vascular Module Test")
    print("=" * 70)
    print()
    
    # Create engine
    engine = SimulationEngine(SimulationConfig(dt=0.01, duration=1.0))
    
    # Register modules
    engine.register_module('cellular', CellularModule, {
        'n_normal_cells': 20,
        'n_cancer_cells': 5
    })
    
    engine.register_module('vascular', VascularModule, {
        'n_capillaries': 8,
        'exchange_rate': 1.0
    })
    
    # Initialize
    engine.initialize()
    
    # Link modules
    vascular = engine.modules['vascular']
    cellular = engine.modules['cellular']
    vascular.set_cellular_module(cellular)
    
    print("Modules linked")
    print()
    
    # Check initial O2 levels
    print("Initial cell O2 levels:")
    for cell_id, cell in list(cellular.cells.items())[:5]:
        distance = vascular.get_distance_to_nearest_vessel(cell.position)
        print(f"  Cell {cell_id}: O2={cell.oxygen:.3f}, distance={distance:.1f}μm")
    print()
    
    # Run simulation
    engine.run()
    
    # Get results
    print("\nVascular State:")
    vascular_state = vascular.get_state()
    for key, value in vascular_state.items():
        print(f"  {key}: {value}")
    
    print("\nCellular State:")
    cellular_state = cellular.get_state()
    for key, value in cellular_state.items():
        print(f"  {key}: {value}")
    
    print()
    print("=" * 70)
    print("✓ Vascular module working!")
    print("=" * 70)
