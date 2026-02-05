#!/usr/bin/env python3
"""
Lymphatic Module
================

Handles lymphatic system: drainage, immune trafficking, metastasis.

Features:
- Lymphatic vessel network
- Fluid drainage
- Immune cell trafficking
- Cancer cell collection (metastasis!)
- Transport to lymph nodes
"""

import sys
sys.path.insert(0, '..')

import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass

from core.module_base import SimulationModule
from core.event_bus import EventTypes


@dataclass
class LymphaticVessel:
    """Lymphatic vessel"""
    vessel_id: int
    start: np.ndarray
    end: np.ndarray
    
    # Drainage
    drainage_rate: float = 0.01  # μL/min
    
    # Contents
    immune_cells_collected: int = 0
    cancer_cells_collected: int = 0
    
    # Collection parameters
    collection_radius: float = 20.0  # μm
    
    def __post_init__(self):
        self.length = np.linalg.norm(self.end - self.start)
    
    def get_position(self, t: float) -> np.ndarray:
        """Get position along vessel (t: 0-1)"""
        return self.start + t * (self.end - self.start)


class LymphaticModule(SimulationModule):
    """
    Lymphatic system simulation module
    
    Manages:
    - Lymphatic vessel network
    - Fluid drainage
    - Immune cell trafficking
    - Cancer metastasis
    
    Events Emitted:
    - METASTASIS_OCCURRED: When cancer enters lymphatic
    - IMMUNE_RECRUITED: When immune cell traffics
    
    Events Subscribed:
    - IMMUNE_ACTIVATED: Track immune trafficking
    - CELL_TRANSFORMED: Monitor for metastasis
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        
        # Lymphatic network
        self.vessels: Dict[int, LymphaticVessel] = {}
        self.next_vessel_id = 0
        
        # Parameters
        self.n_vessels = config.get('n_vessels', 4)
        self.drainage_rate = config.get('drainage_rate', 0.01)
        self.metastasis_probability = config.get('metastasis_probability', 0.001)
        self.immune_trafficking_rate = config.get('immune_trafficking_rate', 0.05)
        
        # Statistics
        self.total_fluid_drained = 0.0
        self.total_immune_trafficked = 0
        self.total_metastases = 0
        
        # References
        self.cellular_module = None
        self.immune_module = None
    
    def initialize(self):
        """Initialize lymphatic network"""
        print("  Creating lymphatic network...")
        
        # Create lymphatic vessels (fewer, larger than capillaries)
        # Positioned between capillaries
        center = np.array([100.0, 100.0, 50.0])
        
        for i in range(self.n_vessels):
            angle = 2 * np.pi * i / self.n_vessels + np.pi/4  # Offset from capillaries
            
            # Start mid-radius
            start = center + np.array([
                60 * np.cos(angle),
                60 * np.sin(angle),
                np.random.uniform(-5, 5)
            ])
            
            # End at periphery
            end = center + np.array([
                90 * np.cos(angle),
                90 * np.sin(angle),
                np.random.uniform(-5, 5)
            ])
            
            self.add_vessel(start, end)
        
        # Subscribe to events
        self.subscribe(EventTypes.IMMUNE_ACTIVATED, self.on_immune_activated)
        self.subscribe(EventTypes.CELL_TRANSFORMED, self.on_cell_transformed)
        
        print(f"    ✓ {self.n_vessels} lymphatic vessels")
        print(f"    ✓ Drainage and trafficking ready")
    
    def add_vessel(self, start, end):
        """Add lymphatic vessel"""
        vessel_id = self.next_vessel_id
        self.next_vessel_id += 1
        
        vessel = LymphaticVessel(
            vessel_id=vessel_id,
            start=np.array(start, dtype=np.float32),
            end=np.array(end, dtype=np.float32)
        )
        
        self.vessels[vessel_id] = vessel
        return vessel_id
    
    def set_cellular_module(self, cellular_module):
        """Link to cellular module"""
        self.cellular_module = cellular_module
    
    def set_immune_module(self, immune_module):
        """Link to immune module"""
        self.immune_module = immune_module
    
    def update(self, dt: float):
        """Update lymphatic system"""
        # Drain fluid
        for vessel in self.vessels.values():
            drainage = vessel.drainage_rate * dt
            self.total_fluid_drained += drainage
        
        # Collect immune cells
        if self.immune_module:
            self._collect_immune_cells(dt)
        
        # Collect cancer cells (metastasis!)
        if self.cellular_module:
            self._collect_cancer_cells(dt)
    
    def _collect_immune_cells(self, dt: float):
        """Collect activated immune cells for trafficking"""
        for vessel in self.vessels.values():
            # Sample points along vessel
            for t in np.linspace(0, 1, 5):
                vessel_pos = vessel.get_position(t)
                
                # Find nearby activated immune cells
                for immune_id, immune_cell in self.immune_module.immune_cells.items():
                    if immune_cell.in_blood or not immune_cell.activated:
                        continue
                    
                    distance = np.linalg.norm(immune_cell.position - vessel_pos)
                    
                    if distance < vessel.collection_radius:
                        # Activated immune cells enter lymphatics
                        if np.random.random() < self.immune_trafficking_rate * dt:
                            immune_cell.in_blood = True
                            vessel.immune_cells_collected += 1
                            self.total_immune_trafficked += 1
                            
                            self.emit_event(EventTypes.IMMUNE_RECRUITED, {
                                'immune_id': immune_id,
                                'immune_type': immune_cell.cell_type,
                                'vessel_id': vessel.vessel_id,
                                'destination': 'lymph_node'
                            })
    
    def _collect_cancer_cells(self, dt: float):
        """Collect cancer cells (metastasis pathway)"""
        for vessel in self.vessels.values():
            # Sample points along vessel
            for t in np.linspace(0, 1, 5):
                vessel_pos = vessel.get_position(t)
                
                # Find nearby cancer cells
                for cell_id, cell in self.cellular_module.cells.items():
                    if cell.cell_type != 'cancer' or not cell.alive:
                        continue
                    
                    distance = np.linalg.norm(cell.position - vessel_pos)
                    
                    if distance < vessel.collection_radius:
                        # Cancer cells can enter lymphatics (METASTASIS!)
                        if np.random.random() < self.metastasis_probability * dt:
                            vessel.cancer_cells_collected += 1
                            self.total_metastases += 1
                            
                            self.emit_event(EventTypes.METASTASIS_OCCURRED, {
                                'cell_id': cell_id,
                                'vessel_id': vessel.vessel_id,
                                'position': cell.position.tolist(),
                                'mutations': cell.mutations,
                                'destination': 'lymph_node'
                            })
                            
                            print(f"⚠️  METASTASIS: Cancer cell {cell_id} entered lymphatic!")
    
    def get_state(self) -> Dict[str, Any]:
        """Return current lymphatic state"""
        return {
            'n_vessels': len(self.vessels),
            'total_length': sum(v.length for v in self.vessels.values()),
            'total_fluid_drained': self.total_fluid_drained,
            'total_immune_trafficked': self.total_immune_trafficked,
            'total_metastases': self.total_metastases,
            'immune_in_vessels': sum(v.immune_cells_collected for v in self.vessels.values()),
            'cancer_in_vessels': sum(v.cancer_cells_collected for v in self.vessels.values())
        }
    
    # Event handlers
    def on_immune_activated(self, data):
        """Handle immune activation - may traffic to lymph node"""
        # Immune cells near lymphatics may enter
        pass
    
    def on_cell_transformed(self, data):
        """Handle cell transformation - monitor for metastasis"""
        # Transformed cells near lymphatics are at risk
        pass


# Test
if __name__ == '__main__':
    from core import SimulationEngine, SimulationConfig
    from cellular_module import CellularModule
    from immune_module import ImmuneModule
    
    print("=" * 70)
    print("Lymphatic Module Test")
    print("=" * 70)
    print()
    
    # Create engine
    engine = SimulationEngine(SimulationConfig(dt=0.01, duration=2.0))
    
    # Register modules
    engine.register_module('cellular', CellularModule, {
        'n_normal_cells': 10,
        'n_cancer_cells': 5
    })
    
    engine.register_module('immune', ImmuneModule, {
        'n_t_cells': 5,
        'n_nk_cells': 3
    })
    
    engine.register_module('lymphatic', LymphaticModule, {
        'n_vessels': 4,
        'metastasis_probability': 0.01  # Higher for testing
    })
    
    # Initialize
    engine.initialize()
    
    # Link modules
    lymphatic = engine.modules['lymphatic']
    cellular = engine.modules['cellular']
    immune = engine.modules['immune']
    
    lymphatic.set_cellular_module(cellular)
    lymphatic.set_immune_module(immune)
    immune.set_cellular_module(cellular)
    
    print("Modules linked")
    print()
    
    # Run simulation
    engine.run()
    
    # Get results
    print("\nLymphatic State:")
    lymphatic_state = lymphatic.get_state()
    for key, value in lymphatic_state.items():
        print(f"  {key}: {value}")
    
    print()
    print("=" * 70)
    print("✓ Lymphatic module working!")
    print("=" * 70)
