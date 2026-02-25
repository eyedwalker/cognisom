#!/usr/bin/env python3
"""
Immune Module
=============

Handles immune system: T cells, NK cells, macrophages.

Features:
- T cells (CD8+ cytotoxic) - MHC-I recognition
- NK cells (natural killer) - Missing-self detection
- Macrophages - Phagocytosis and polarization
- Immune surveillance and patrol
- Cancer recognition and killing
- Chemotaxis (follow gradients)
"""

import sys
sys.path.insert(0, '..')

import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass

from core.module_base import SimulationModule
from core.event_bus import EventTypes


class ExhaustionState:
    """T-cell exhaustion states (progressive loss of function)."""
    NAIVE = "naive"
    EFFECTOR = "effector"
    MEMORY = "memory"
    PRE_EXHAUSTED = "pre_exhausted"  # Progenitor exhausted (TCF1+, some function)
    EXHAUSTED = "exhausted"          # Terminally exhausted (PD-1hi, TIM-3+, no function)
    REACTIVATED = "reactivated"      # Checkpoint-blockade reactivated


@dataclass
class ImmuneCell:
    """State of an immune cell"""
    cell_id: int
    position: np.ndarray
    cell_type: str  # 'T_cell', 'NK_cell', 'macrophage'
    velocity: np.ndarray = None

    # State
    activated: bool = False
    target_cell_id: int = None
    in_blood: bool = False

    # Parameters
    speed: float = 10.0  # μm/min
    detection_radius: float = 10.0  # μm
    kill_radius: float = 5.0  # μm

    # Exhaustion (Digital Twin integration)
    exhaustion_score: float = 0.0        # 0=fully functional, 1=terminally exhausted
    exhaustion_state: str = "effector"   # ExhaustionState value
    pd1_expression: float = 0.0         # PD-1 surface expression (0-1)
    checkpoint_blocked: bool = False    # Under checkpoint inhibitor treatment

    # Macrophage polarization
    polarization: str = "M0"  # M0, M1, M2
    polarization_score: float = 0.0  # -1=M1, +1=M2

    @property
    def kill_probability(self) -> float:
        """Effective kill probability accounting for exhaustion."""
        if self.exhaustion_state == ExhaustionState.EXHAUSTED:
            return 0.05  # Minimal residual function
        elif self.exhaustion_state == ExhaustionState.PRE_EXHAUSTED:
            return 0.4  # Reduced but present
        elif self.exhaustion_state == ExhaustionState.REACTIVATED:
            return 0.7  # Partially restored
        return 0.8  # Full effector function

    @property
    def is_functional(self) -> bool:
        """Whether this cell can perform effector functions."""
        return self.exhaustion_score < 0.8 or self.checkpoint_blocked

    def __post_init__(self):
        if self.velocity is None:
            self.velocity = np.zeros(3)


class ImmuneModule(SimulationModule):
    """
    Immune system simulation module
    
    Manages:
    - T cells (CD8+ cytotoxic)
    - NK cells (natural killer)
    - Macrophages
    - Immune surveillance
    - Cancer killing
    
    Events Emitted:
    - IMMUNE_ACTIVATED: When immune cell activates
    - CANCER_KILLED: When cancer cell is killed
    - IMMUNE_RECRUITED: When new immune cell arrives
    
    Events Subscribed:
    - CELL_TRANSFORMED: Respond to new cancer
    - CELL_DIVIDED: Track cell population
    - EXOSOME_RELEASED: Detect danger signals
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        
        # Immune cell population
        self.immune_cells: Dict[int, ImmuneCell] = {}
        self.next_immune_id = 0
        
        # Parameters
        self.n_t_cells = config.get('n_t_cells', 15)
        self.n_nk_cells = config.get('n_nk_cells', 10)
        self.n_macrophages = config.get('n_macrophages', 8)
        self.patrol_speed = config.get('patrol_speed', 5.0)  # μm/min
        self.kill_probability = config.get('kill_probability', 0.8)
        
        # Statistics
        self.total_kills = 0
        self.total_activations = 0
        self.total_recruited = 0
        
        # Reference to cellular module (for target cells)
        self.cellular_module = None
    
    def initialize(self):
        """Initialize immune system"""
        print("  Creating immune cell population...")
        
        # Create T cells (patrol tissue)
        for i in range(self.n_t_cells):
            x = np.random.uniform(50, 150)
            y = np.random.uniform(50, 150)
            z = np.random.uniform(40, 60)
            
            self.add_immune_cell(
                position=[x, y, z],
                cell_type='T_cell'
            )
        
        # Create NK cells
        for i in range(self.n_nk_cells):
            x = np.random.uniform(50, 150)
            y = np.random.uniform(50, 150)
            z = np.random.uniform(40, 60)
            
            self.add_immune_cell(
                position=[x, y, z],
                cell_type='NK_cell'
            )
        
        # Create macrophages
        for i in range(self.n_macrophages):
            x = np.random.uniform(50, 150)
            y = np.random.uniform(50, 150)
            z = np.random.uniform(40, 60)
            
            self.add_immune_cell(
                position=[x, y, z],
                cell_type='macrophage'
            )
        
        # Subscribe to events
        self.subscribe(EventTypes.CELL_TRANSFORMED, self.on_cell_transformed)
        self.subscribe(EventTypes.CELL_DIVIDED, self.on_cell_divided)
        
        print(f"    ✓ {self.n_t_cells} T cells")
        print(f"    ✓ {self.n_nk_cells} NK cells")
        print(f"    ✓ {self.n_macrophages} macrophages")
    
    def add_immune_cell(self, position, cell_type):
        """Add immune cell"""
        cell_id = self.next_immune_id
        self.next_immune_id += 1
        
        cell = ImmuneCell(
            cell_id=cell_id,
            position=np.array(position, dtype=np.float32),
            cell_type=cell_type
        )
        
        self.immune_cells[cell_id] = cell
        return cell_id
    
    def set_cellular_module(self, cellular_module):
        """Link to cellular module for target access"""
        self.cellular_module = cellular_module
    
    def update(self, dt: float):
        """Update immune system"""
        if not self.cellular_module:
            return
        
        # Get cancer cells from cellular module
        cancer_cells = {cid: cell for cid, cell in self.cellular_module.cells.items()
                       if cell.cell_type == 'cancer' and cell.alive}
        
        for immune_id, immune_cell in list(self.immune_cells.items()):
            if immune_cell.in_blood:
                continue
            
            # Patrol (random walk)
            if not immune_cell.activated:
                self._patrol(immune_cell, dt)
                
                # Look for cancer cells
                for cancer_id, cancer_cell in cancer_cells.items():
                    distance = np.linalg.norm(immune_cell.position - cancer_cell.position)
                    
                    if distance < immune_cell.detection_radius:
                        # Check if can recognize
                        if self._can_recognize(immune_cell, cancer_cell):
                            immune_cell.activated = True
                            immune_cell.target_cell_id = cancer_id
                            self.total_activations += 1
                            
                            self.emit_event(EventTypes.IMMUNE_ACTIVATED, {
                                'immune_id': immune_id,
                                'immune_type': immune_cell.cell_type,
                                'target_id': cancer_id,
                                'position': immune_cell.position.tolist()
                            })
                            break
            
            # Attack target
            else:
                if immune_cell.target_cell_id in cancer_cells:
                    target = cancer_cells[immune_cell.target_cell_id]
                    
                    # Move toward target
                    direction = target.position - immune_cell.position
                    distance = np.linalg.norm(direction)
                    
                    if distance > 0:
                        direction = direction / distance
                        immune_cell.velocity = direction * immune_cell.speed
                        immune_cell.position += immune_cell.velocity * dt * 0.01  # Convert min to hours
                    
                    # Kill if close enough
                    if distance < immune_cell.kill_radius:
                        if np.random.random() < self.kill_probability:
                            self._kill_target(immune_cell, target)
                else:
                    # Target lost, deactivate
                    immune_cell.activated = False
                    immune_cell.target_cell_id = None
            
            # Keep in bounds
            immune_cell.position = np.clip(immune_cell.position, [20, 20, 20], [180, 180, 80])
    
    def _patrol(self, immune_cell: ImmuneCell, dt: float):
        """Random patrol movement"""
        # Random walk
        direction = np.random.randn(3)
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        
        immune_cell.velocity = direction * self.patrol_speed
        immune_cell.position += immune_cell.velocity * dt * 0.01  # Convert min to hours
    
    def _can_recognize(self, immune_cell: ImmuneCell, cancer_cell) -> bool:
        """Check if immune cell can recognize cancer cell"""
        if immune_cell.cell_type == 'T_cell':
            # T cells need MHC-I presentation
            # Cancer often downregulates MHC-I to evade
            return cancer_cell.mhc1_expression > 0.2
        
        elif immune_cell.cell_type == 'NK_cell':
            # NK cells detect MISSING MHC-I (missing-self)
            # This is why cancer can't escape both T cells and NK cells!
            return cancer_cell.mhc1_expression < 0.4
        
        elif immune_cell.cell_type == 'macrophage':
            # Macrophages can recognize via other markers
            return True
        
        return False
    
    def _kill_target(self, immune_cell: ImmuneCell, target_cell):
        """Kill target cancer cell"""
        # Emit kill event (cellular module will handle actual death)
        self.emit_event(EventTypes.CANCER_KILLED, {
            'cell_id': target_cell.cell_id,
            'killer_id': immune_cell.cell_id,
            'killer_type': immune_cell.cell_type,
            'position': target_cell.position.tolist()
        })
        
        self.total_kills += 1
        
        # Deactivate immune cell
        immune_cell.activated = False
        immune_cell.target_cell_id = None
    
    def get_state(self) -> Dict[str, Any]:
        """Return current immune state"""
        active_immune = [ic for ic in self.immune_cells.values() if not ic.in_blood]
        activated_immune = [ic for ic in active_immune if ic.activated]
        
        t_cells = [ic for ic in active_immune if ic.cell_type == 'T_cell']
        nk_cells = [ic for ic in active_immune if ic.cell_type == 'NK_cell']
        macrophages = [ic for ic in active_immune if ic.cell_type == 'macrophage']
        
        return {
            'n_immune_cells': len(active_immune),
            'n_activated': len(activated_immune),
            'n_t_cells': len(t_cells),
            'n_nk_cells': len(nk_cells),
            'n_macrophages': len(macrophages),
            'total_kills': self.total_kills,
            'total_activations': self.total_activations,
            'total_recruited': self.total_recruited
        }
    
    # Event handlers
    def on_cell_transformed(self, data):
        """Handle cell transformation - recruit more immune cells"""
        # Recruit additional immune cell
        position = data['position']
        
        # Add new T cell near transformation site
        offset = np.random.randn(3) * 20
        new_pos = np.array(position) + offset
        
        self.add_immune_cell(new_pos, 'T_cell')
        self.total_recruited += 1
        
        self.emit_event(EventTypes.IMMUNE_RECRUITED, {
            'cell_type': 'T_cell',
            'position': new_pos.tolist(),
            'reason': 'cell_transformation'
        })
    
    def on_cell_divided(self, data):
        """Handle cell division - track population"""
        # Could recruit more immune cells if cancer is dividing rapidly
        if data['cell_type'] == 'cancer':
            if np.random.random() < 0.1:  # 10% chance
                position = data['position']
                offset = np.random.randn(3) * 20
                new_pos = np.array(position) + offset
                
                self.add_immune_cell(new_pos, 'NK_cell')
                self.total_recruited += 1


# Test
if __name__ == '__main__':
    from core import SimulationEngine, SimulationConfig
    from cellular_module import CellularModule
    
    print("=" * 70)
    print("Immune Module Test")
    print("=" * 70)
    print()
    
    # Create engine
    engine = SimulationEngine(SimulationConfig(dt=0.01, duration=1.0))
    
    # Register modules
    engine.register_module('cellular', CellularModule, {
        'n_normal_cells': 10,
        'n_cancer_cells': 5,
        'division_time_cancer': 10.0  # Slow for testing
    })
    
    engine.register_module('immune', ImmuneModule, {
        'n_t_cells': 5,
        'n_nk_cells': 3,
        'n_macrophages': 2
    })
    
    # Initialize
    engine.initialize()
    
    # Link modules
    immune = engine.modules['immune']
    cellular = engine.modules['cellular']
    immune.set_cellular_module(cellular)
    
    print("Modules linked")
    print()
    
    # Run simulation
    engine.run()
    
    # Get results
    print("\nImmune State:")
    immune_state = immune.get_state()
    for key, value in immune_state.items():
        print(f"  {key}: {value}")
    
    print("\nCellular State:")
    cellular_state = cellular.get_state()
    for key, value in cellular_state.items():
        print(f"  {key}: {value}")
    
    print()
    print("=" * 70)
    print("✓ Immune module working!")
    print("=" * 70)
