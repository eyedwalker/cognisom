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
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from cognisom.core.module_base import SimulationModule
from cognisom.core.event_bus import EventTypes


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

    def distance_to(self, position: np.ndarray) -> float:
        """Distance from a point to the vessel segment.

        Collection used to test a cell against five sample points along
        the vessel and roll the dice at each, so a cell near the middle
        of a 30 um vessel was within 20 um of all five points and drew
        five times a step: the configured per-hour rate was silently
        multiplied by up to five, and by a number that changed with
        vessel length. One distance, one draw.
        """
        seg = self.end - self.start
        length_sq = float(seg @ seg)
        if length_sq == 0.0:
            return float(np.linalg.norm(position - self.start))
        t = float(np.clip((position - self.start) @ seg / length_sq, 0.0, 1.0))
        return float(np.linalg.norm(position - (self.start + t * seg)))


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
    
    Events Subscribed: none. It used to subscribe to IMMUNE_ACTIVATED and
    CELL_TRANSFORMED with handlers that were ``pass``.

    Rates (both ESTIMATE -- no source; per cell per hour, for a cell
    within ``collection_radius`` of a vessel):
    - ``metastasis_probability``: a cancer cell intravasates and LEAVES
      the tissue. It used to be counted as a metastasis and left in
      place, alive and dividing: double-counted.
    - ``immune_trafficking_rate``: an activated immune cell egresses to
      the draining node. The immune module has no return path, so this
      is a one-way exit from the tissue.
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        
        # Lymphatic network
        self.vessels: Dict[int, LymphaticVessel] = {}
        self.next_vessel_id = 0
        
        # Parameters
        self.n_vessels = self.config.get('n_vessels', 4)
        self.drainage_rate = self.config.get('drainage_rate', 0.01)
        self.metastasis_probability = self.config.get('metastasis_probability', 0.001)
        self.immune_trafficking_rate = self.config.get('immune_trafficking_rate', 0.05)
        # Where the vessels sit: a ring about ``centre`` from ``inner`` to
        # ``outer`` radius, in um. The defaults put them at 60-90 um from
        # (100, 100, 50) while every preset seeds its tumour about 20 um
        # from that centre with a ~17 um radius -- so a 20 um collection
        # radius reached 0-6 of 25 cancer cells, by seed, and the preset
        # called "Metastasis Risk ... tumour near lymphatic vessels" could
        # not show a metastasis in its 24 h. A preset that means it sets
        # the ring through the tumour margin.
        self.ring_centre = np.asarray(self.config.get('ring_centre_um', (100.0, 100.0, 50.0)), dtype=float)
        self.ring_inner_um = float(self.config.get('ring_inner_um', 60.0))
        self.ring_outer_um = float(self.config.get('ring_outer_um', 90.0))
        
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
        center = self.ring_centre
        inner, outer = self.ring_inner_um, self.ring_outer_um

        for i in range(self.n_vessels):
            angle = 2 * np.pi * i / self.n_vessels + np.pi/4  # Offset from capillaries

            start = center + np.array([
                inner * np.cos(angle),
                inner * np.sin(angle),
                np.random.uniform(-5, 5)
            ])

            end = center + np.array([
                outer * np.cos(angle),
                outer * np.sin(angle),
                np.random.uniform(-5, 5)
            ])
            
            self.add_vessel(start, end)
        
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
        for immune_id, immune_cell in list(self.immune_module.immune_cells.items()):
            if immune_cell.in_blood or not immune_cell.activated:
                continue
            vessel = self._nearest_vessel(immune_cell.position)
            if vessel is None:
                continue
            # Activated immune cells egress to the draining node
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

    def _nearest_vessel(self, position) -> Optional[LymphaticVessel]:
        """The vessel within collection radius of ``position``, nearest
        first, or ``None``. A cell near two vessels is still one cell with
        one chance to enter a lymphatic this step."""
        best, best_d = None, None
        for vessel in self.vessels.values():
            d = vessel.distance_to(np.asarray(position, dtype=float))
            if d < vessel.collection_radius and (best_d is None or d < best_d):
                best, best_d = vessel, d
        return best
    
    def _collect_cancer_cells(self, dt: float):
        """Collect cancer cells (metastasis pathway)"""
        for cell_id, cell in list(self.cellular_module.cells.items()):
            if cell.cell_type != 'cancer' or not cell.alive:
                continue
            vessel = self._nearest_vessel(cell.position)
            if vessel is None:
                continue
            # A cancer cell intravasates: it leaves the tissue
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
                depart = getattr(self.cellular_module, 'depart_cell', None)
                if depart is not None:
                    depart(cell_id, destination='lymphatic')
    
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
    


# Test
if __name__ == '__main__':
    from cognisom.core import SimulationEngine, SimulationConfig
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
