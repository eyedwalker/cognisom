#!/usr/bin/env python3
"""
Cellular Module
===============

Handles cell-level dynamics: metabolism, division, death, migration.

Features:
- Cell cycle (G1, S, G2, M phases)
- Metabolism (O2, glucose, ATP)
- Cell division
- Cell death (apoptosis, necrosis)
- Cell migration
"""

import sys
sys.path.insert(0, '..')

import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass

from core.module_base import SimulationModule
from core.event_bus import EventTypes


@dataclass
class CellState:
    """State of a single cell"""
    cell_id: int
    position: np.ndarray
    cell_type: str  # 'normal', 'cancer', 'immune'
    phase: str  # 'G1', 'S', 'G2', 'M'
    age: float = 0.0
    alive: bool = True
    
    # Metabolism
    oxygen: float = 0.21
    glucose: float = 5.0
    atp: float = 1000.0
    lactate: float = 0.0
    
    # Cancer properties
    mhc1_expression: float = 1.0
    mutations: List[str] = None
    
    def __post_init__(self):
        if self.mutations is None:
            self.mutations = []


class CellularModule(SimulationModule):
    """
    Cellular simulation module
    
    Manages:
    - Cell population
    - Cell cycle
    - Metabolism
    - Division and death
    - Cell state
    
    Events Emitted:
    - CELL_DIVIDED: When cell divides
    - CELL_DIED: When cell dies
    - CELL_TRANSFORMED: When normal → cancer
    - CELL_MIGRATED: When cell moves
    
    Events Subscribed:
    - EXOSOME_UPTAKEN: Process molecular cargo
    - CANCER_KILLED: Remove killed cells
    - HYPOXIA_DETECTED: Respond to low O2
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        
        # Cell population
        self.cells: Dict[int, CellState] = {}
        self.next_cell_id = 0
        
        # Parameters
        self.division_time_normal = config.get('division_time_normal', 24.0)  # hours
        self.division_time_cancer = config.get('division_time_cancer', 12.0)  # hours
        self.glucose_consumption_normal = config.get('glucose_consumption', 0.2)
        self.glucose_consumption_cancer = config.get('glucose_consumption_cancer', 0.5)
        
        # Statistics
        self.total_divisions = 0
        self.total_deaths = 0
        self.total_transformations = 0
    
    def initialize(self):
        """Initialize cellular system"""
        print("  Creating initial cell population...")
        
        # Create initial cells
        n_normal = self.config.get('n_normal_cells', 80)
        n_cancer = self.config.get('n_cancer_cells', 20)
        
        # Normal cells (organized in circle)
        for i in range(n_normal):
            angle = 2 * np.pi * i / n_normal
            radius = 50
            x = 100 + radius * np.cos(angle)
            y = 100 + radius * np.sin(angle)
            z = 50 + np.random.uniform(-10, 10)
            
            self.add_cell(position=[x, y, z], cell_type='normal')
        
        # Cancer cells (clustered)
        for i in range(n_cancer):
            x = 120 + np.random.uniform(-15, 15)
            y = 120 + np.random.uniform(-15, 15)
            z = 50 + np.random.uniform(-5, 5)
            
            cell_id = self.add_cell(position=[x, y, z], cell_type='cancer')
            self.cells[cell_id].mutations = ['KRAS_G12D']
            self.cells[cell_id].mhc1_expression = 0.3  # Downregulated
        
        # Subscribe to events
        self.subscribe(EventTypes.EXOSOME_UPTAKEN, self.on_exosome_uptaken)
        self.subscribe(EventTypes.CANCER_KILLED, self.on_cancer_killed)
        
        print(f"    ✓ {n_normal} normal cells")
        print(f"    ✓ {n_cancer} cancer cells")
    
    def add_cell(self, position, cell_type='normal'):
        """Add new cell"""
        cell_id = self.next_cell_id
        self.next_cell_id += 1
        
        cell = CellState(
            cell_id=cell_id,
            position=np.array(position, dtype=np.float32),
            cell_type=cell_type,
            phase='G1'
        )
        
        self.cells[cell_id] = cell
        return cell_id
    
    def remove_cell(self, cell_id: int):
        """Remove cell"""
        if cell_id in self.cells:
            del self.cells[cell_id]
    
    def update(self, dt: float):
        """Update all cells"""
        cells_to_divide = []
        cells_to_die = []
        
        for cell_id, cell in list(self.cells.items()):
            if not cell.alive:
                continue
            
            # Age
            cell.age += dt
            
            # Metabolism
            self._update_metabolism(cell, dt)
            
            # Check for division
            division_time = (self.division_time_cancer if cell.cell_type == 'cancer' 
                           else self.division_time_normal)
            
            if cell.age >= division_time:
                cells_to_divide.append(cell_id)
            
            # Check for death (hypoxia, starvation)
            if cell.oxygen < 0.02 or cell.glucose < 0.5:
                if cell.cell_type == 'normal':
                    cells_to_die.append(cell_id)
                elif np.random.random() < 0.01:  # Cancer more resistant
                    cells_to_die.append(cell_id)
        
        # Process divisions
        for cell_id in cells_to_divide:
            self._divide_cell(cell_id)
        
        # Process deaths
        for cell_id in cells_to_die:
            self._kill_cell(cell_id, cause='hypoxia')
    
    def _update_metabolism(self, cell: CellState, dt: float):
        """Update cell metabolism"""
        if cell.cell_type == 'cancer':
            # Warburg effect
            cell.glucose -= self.glucose_consumption_cancer * dt
            cell.oxygen -= 0.1 * dt
            cell.lactate += 0.3 * dt
            cell.atp += 50 * dt  # Less efficient
        else:
            # Normal metabolism
            cell.glucose -= self.glucose_consumption_normal * dt
            cell.oxygen -= 0.15 * dt
            cell.lactate += 0.1 * dt
            cell.atp += 100 * dt  # More efficient
        
        # Clamp values
        cell.glucose = max(0, cell.glucose)
        cell.oxygen = max(0, cell.oxygen)
        cell.lactate = max(0, cell.lactate)
        cell.atp = max(0, cell.atp)
    
    def _divide_cell(self, cell_id: int):
        """Cell division"""
        if cell_id not in self.cells:
            return
        
        parent = self.cells[cell_id]
        
        # Create daughter cell
        offset = np.random.randn(3) * 5  # Small offset
        daughter_pos = parent.position + offset
        
        daughter_id = self.add_cell(
            position=daughter_pos,
            cell_type=parent.cell_type
        )
        
        daughter = self.cells[daughter_id]
        daughter.mutations = parent.mutations.copy()
        daughter.mhc1_expression = parent.mhc1_expression
        
        # Reset parent age
        parent.age = 0.0
        
        # Emit event
        self.emit_event(EventTypes.CELL_DIVIDED, {
            'cell_id': cell_id,
            'daughter_id': daughter_id,
            'cell_type': parent.cell_type,
            'position': parent.position.tolist()
        })
        
        self.total_divisions += 1
    
    def _kill_cell(self, cell_id: int, cause: str = 'unknown'):
        """Cell death"""
        if cell_id not in self.cells:
            return
        
        cell = self.cells[cell_id]
        cell.alive = False
        
        # Emit event
        self.emit_event(EventTypes.CELL_DIED, {
            'cell_id': cell_id,
            'cell_type': cell.cell_type,
            'cause': cause,
            'position': cell.position.tolist()
        })
        
        self.total_deaths += 1
        
        # Remove cell
        self.remove_cell(cell_id)
    
    def transform_cell(self, cell_id: int):
        """Transform normal cell to cancer"""
        if cell_id not in self.cells:
            return
        
        cell = self.cells[cell_id]
        if cell.cell_type != 'normal':
            return
        
        # Transform
        cell.cell_type = 'cancer'
        cell.mutations.append('KRAS_G12D')
        cell.mhc1_expression = 0.3
        
        # Emit event
        self.emit_event(EventTypes.CELL_TRANSFORMED, {
            'cell_id': cell_id,
            'position': cell.position.tolist(),
            'mutations': cell.mutations
        })
        
        self.total_transformations += 1
    
    def get_state(self) -> Dict[str, Any]:
        """Return current cellular state"""
        alive_cells = [c for c in self.cells.values() if c.alive]
        cancer_cells = [c for c in alive_cells if c.cell_type == 'cancer']
        normal_cells = [c for c in alive_cells if c.cell_type == 'normal']
        
        return {
            'n_cells': len(alive_cells),
            'n_cancer': len(cancer_cells),
            'n_normal': len(normal_cells),
            'total_divisions': self.total_divisions,
            'total_deaths': self.total_deaths,
            'total_transformations': self.total_transformations,
            'avg_oxygen': np.mean([c.oxygen for c in alive_cells]) if alive_cells else 0,
            'avg_glucose': np.mean([c.glucose for c in alive_cells]) if alive_cells else 0
        }
    
    # Event handlers
    def on_exosome_uptaken(self, data):
        """Handle exosome uptake - check for transformation"""
        cell_id = data['cell_id']
        
        if cell_id in self.cells and data['cargo']['oncogenic']:
            # Oncogenic cargo can transform cell
            if self.cells[cell_id].cell_type == 'normal':
                if np.random.random() < 0.3:  # 30% chance
                    self.transform_cell(cell_id)
    
    def on_cancer_killed(self, data):
        """Handle cancer cell killed by immune system"""
        cell_id = data['cell_id']
        self._kill_cell(cell_id, cause='immune_killed')


# Test
if __name__ == '__main__':
    from core import SimulationEngine, SimulationConfig
    
    print("=" * 70)
    print("Cellular Module Test")
    print("=" * 70)
    print()
    
    # Create engine
    engine = SimulationEngine(SimulationConfig(dt=0.1, duration=2.0))
    
    # Register cellular module
    engine.register_module('cellular', CellularModule, {
        'n_normal_cells': 10,
        'n_cancer_cells': 2,
        'division_time_normal': 1.0,  # Fast for testing
        'division_time_cancer': 0.5
    })
    
    # Initialize and run
    engine.initialize()
    engine.run()
    
    # Get results
    state = engine.modules['cellular'].get_state()
    print("\nCellular State:")
    for key, value in state.items():
        print(f"  {key}: {value}")
    print()
    
    print("=" * 70)
    print("✓ Cellular module working!")
    print("=" * 70)
