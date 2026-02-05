"""
Cell class - Represents a single biological cell with internal state
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
import uuid


@dataclass
class CellState:
    """
    Represents the internal state of a single cell
    
    Attributes:
        cell_id: Unique identifier
        species_counts: Molecular species counts (proteins, mRNAs, etc.)
        position: (x, y, z) position in microns
        phase: Cell cycle phase ('G1', 'S', 'G2', 'M')
        age: Cell age in hours
        volume: Cell volume in femtoliters (fL)
        mhc1_expression: MHC-I surface expression level (0-1)
        stress_level: Cumulative stress (0-1)
    """
    cell_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    species_counts: np.ndarray = field(default=None)
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    phase: str = 'G1'
    age: float = 0.0
    volume: float = 2000.0  # Typical mammalian cell ~2000 fL
    mhc1_expression: float = 1.0
    stress_level: float = 0.0
    
    def __post_init__(self):
        if self.species_counts is None:
            # Initialize with basic species
            # Index 0: mRNA, Index 1: Proteins, Index 2: ATP
            self.species_counts = np.array([100, 1000, 5000], dtype=np.int32)


class Cell:
    """
    Biological cell with growth, division, and death capabilities
    
    This is a simplified model for prototyping. Full version will include:
    - Stochastic gene expression (SSA)
    - Metabolism (FBA)
    - Cell cycle checkpoints
    - Apoptosis pathways
    """
    
    def __init__(self, initial_state: Optional[CellState] = None):
        self.state = initial_state if initial_state else CellState()
        self.history = []
        
        # Parameters (will be loaded from config files later)
        self.params = {
            'k_transcription': 15.0,       # mRNA synthesis rate
            'k_translation': 80.0,         # Protein synthesis rate
            'k_mrna_decay': 0.03,          # mRNA degradation rate
            'k_protein_decay': 0.005,      # Protein degradation rate
            'division_threshold': 2500,    # Protein count for division
            'doubling_time': 24.0,         # Target doubling time (hours)
        }
    
    def step(self, dt: float = 0.01) -> Optional['Cell']:
        """
        Advance cell state by one time step
        
        Args:
            dt: Time step in hours
            
        Returns:
            New daughter cell if division occurred, None otherwise
        """
        # Extract current state
        mrna = self.state.species_counts[0]
        proteins = self.state.species_counts[1]
        atp = self.state.species_counts[2]
        
        # Simple transcription/translation model
        # (Will be replaced with full SSA later)
        
        # Transcription: DNA -> mRNA
        new_mrna = int(self.params['k_transcription'] * dt)
        mrna += new_mrna
        
        # Translation: mRNA -> Protein (requires ATP)
        if atp > 10:
            new_proteins = int(mrna * self.params['k_translation'] * dt)
            proteins += new_proteins
            atp -= min(new_proteins, atp)  # Consume ATP
        
        # Degradation
        mrna = int(mrna * (1 - self.params['k_mrna_decay'] * dt))
        proteins = int(proteins * (1 - self.params['k_protein_decay'] * dt))
        
        # ATP regeneration (simplified metabolism)
        atp += int(100 * dt)
        atp = min(atp, 10000)  # Cap at 10k
        
        # Update state
        self.state.species_counts[0] = max(0, mrna)
        self.state.species_counts[1] = max(0, proteins)
        self.state.species_counts[2] = max(0, atp)
        self.state.age += dt
        
        # Cell cycle progression
        self._update_cell_cycle(dt)
        
        # Check for division
        if proteins > self.params['division_threshold'] and self.state.phase == 'M':
            return self.divide()
        
        return None
    
    def _update_cell_cycle(self, dt: float):
        """Update cell cycle phase based on age"""
        # Simplified cell cycle (will be replaced with checkpoint model)
        cycle_times = {
            'G1': 11.0,  # G1 phase duration
            'S': 8.0,    # S phase
            'G2': 4.0,   # G2 phase
            'M': 1.0,    # Mitosis
        }
        
        total_time = sum(cycle_times.values())
        cycle_age = self.state.age % total_time
        
        cumulative = 0
        for phase, duration in cycle_times.items():
            cumulative += duration
            if cycle_age < cumulative:
                self.state.phase = phase
                break
    
    def divide(self) -> 'Cell':
        """
        Cell division - creates daughter cell
        
        Returns:
            New daughter cell
        """
        # Split molecular content
        self.state.species_counts = self.state.species_counts // 2
        self.state.age = 0.0
        self.state.phase = 'G1'
        self.state.volume = self.state.volume / 2
        
        # Create daughter with same content
        daughter_state = CellState(
            species_counts=self.state.species_counts.copy(),
            position=self.state.position,
            phase='G1',
            age=0.0,
            volume=self.state.volume,
            mhc1_expression=self.state.mhc1_expression,
            stress_level=self.state.stress_level
        )
        
        return Cell(daughter_state)
    
    def apply_stress(self, stress_amount: float):
        """Apply stress to cell (DNA damage, hypoxia, etc.)"""
        self.state.stress_level = min(1.0, self.state.stress_level + stress_amount)
        
        # Stress reduces MHC-I expression (immune evasion)
        if self.state.stress_level > 0.5:
            self.state.mhc1_expression *= 0.95
    
    def is_alive(self) -> bool:
        """Check if cell is alive"""
        # Cell dies if critical species drop too low
        proteins = self.state.species_counts[1]
        atp = self.state.species_counts[2]
        
        if proteins < 100 or atp < 10:
            return False
        
        # High stress triggers apoptosis
        if self.state.stress_level > 0.9:
            return False
        
        return True
    
    def get_state_dict(self) -> Dict:
        """Get cell state as dictionary for logging"""
        return {
            'cell_id': self.state.cell_id,
            'age': self.state.age,
            'phase': self.state.phase,
            'mrna': int(self.state.species_counts[0]),
            'proteins': int(self.state.species_counts[1]),
            'atp': int(self.state.species_counts[2]),
            'volume': self.state.volume,
            'mhc1': self.state.mhc1_expression,
            'stress': self.state.stress_level,
            'position': self.state.position,
        }
    
    def __repr__(self) -> str:
        return (f"Cell(id={self.state.cell_id}, phase={self.state.phase}, "
                f"age={self.state.age:.1f}h, proteins={self.state.species_counts[1]})")
