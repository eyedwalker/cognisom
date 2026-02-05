"""
Unit tests for Cell class
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add engine to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from engine.py.cell import Cell, CellState


def test_cell_creation():
    """Test basic cell creation"""
    cell = Cell()
    
    assert cell.state.phase == 'G1'
    assert cell.state.age == 0.0
    assert len(cell.state.species_counts) == 3
    assert cell.state.mhc1_expression == 1.0
    assert cell.state.stress_level == 0.0


def test_cell_with_custom_state():
    """Test cell creation with custom state"""
    custom_state = CellState(
        species_counts=np.array([200, 2000, 10000], dtype=np.int32),
        position=(10.0, 20.0, 30.0),
        phase='S',
        age=5.0
    )
    
    cell = Cell(custom_state)
    
    assert cell.state.phase == 'S'
    assert cell.state.age == 5.0
    assert cell.state.position == (10.0, 20.0, 30.0)
    assert cell.state.species_counts[0] == 200


def test_cell_step():
    """Test cell time stepping"""
    cell = Cell()
    initial_proteins = cell.state.species_counts[1]
    
    # Step forward
    daughter = cell.step(dt=0.01)
    
    # Age should increase
    assert cell.state.age > 0.0
    
    # Proteins should change (increase due to synthesis)
    assert cell.state.species_counts[1] != initial_proteins
    
    # Should not divide immediately
    assert daughter is None


def test_cell_division():
    """Test cell division"""
    cell = Cell()
    
    # Force high protein count and age to M phase
    cell.state.species_counts[1] = 3000
    cell.state.age = 23.5  # Near end of 24h cycle, should be in M phase
    
    # Step multiple times to ensure we hit M phase with high proteins
    daughter = None
    for _ in range(100):  # Try up to 100 steps
        daughter = cell.step(dt=0.01)
        if daughter is not None:
            break
    
    # Should eventually divide
    assert daughter is not None
    assert isinstance(daughter, Cell)
    
    # Both cells should have reduced proteins
    assert cell.state.species_counts[1] < 3000
    assert daughter.state.species_counts[1] < 3000
    
    # Both should be in G1 phase after division
    assert cell.state.phase == 'G1'
    assert daughter.state.phase == 'G1'


def test_cell_stress():
    """Test stress application"""
    cell = Cell()
    initial_mhc1 = cell.state.mhc1_expression
    
    # Apply moderate stress
    cell.apply_stress(0.3)
    assert cell.state.stress_level == 0.3
    assert cell.state.mhc1_expression == initial_mhc1  # No change yet
    
    # Apply high stress
    cell.apply_stress(0.3)  # Now at 0.6
    assert cell.state.stress_level == 0.6
    assert cell.state.mhc1_expression < initial_mhc1  # Should decrease


def test_cell_death():
    """Test cell death conditions"""
    cell = Cell()
    
    # Healthy cell should be alive
    assert cell.is_alive() is True
    
    # Deplete proteins
    cell.state.species_counts[1] = 50
    assert cell.is_alive() is False
    
    # Reset and test ATP depletion
    cell.state.species_counts[1] = 1000
    cell.state.species_counts[2] = 5
    assert cell.is_alive() is False
    
    # Reset and test high stress
    cell.state.species_counts[2] = 5000
    cell.state.stress_level = 0.95
    assert cell.is_alive() is False


def test_cell_cycle_progression():
    """Test cell cycle phase transitions"""
    cell = Cell()
    
    # Start in G1
    assert cell.state.phase == 'G1'
    
    # Age through cell cycle
    for _ in range(1200):  # 12 hours
        cell.step(dt=0.01)
    
    # Should have progressed through phases
    # (exact phase depends on timing, but age should increase)
    assert cell.state.age > 10.0


def test_get_state_dict():
    """Test state dictionary export"""
    cell = Cell()
    state_dict = cell.get_state_dict()
    
    assert 'cell_id' in state_dict
    assert 'age' in state_dict
    assert 'phase' in state_dict
    assert 'mrna' in state_dict
    assert 'proteins' in state_dict
    assert 'atp' in state_dict
    assert 'mhc1' in state_dict
    assert 'stress' in state_dict
    
    assert isinstance(state_dict['mrna'], int)
    assert isinstance(state_dict['proteins'], int)


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
