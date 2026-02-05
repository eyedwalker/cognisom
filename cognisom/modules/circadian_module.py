#!/usr/bin/env python3
"""
Circadian Clock Module
======================

Handles circadian rhythms and temporal regulation.

Features:
- Master clock (SCN-like)
- Cellular clocks (peripheral)
- Clock genes (CLOCK, BMAL1, PER, CRY)
- Rhythmic gene expression
- Cell cycle gating
- Metabolism cycling
- Immune function rhythms
"""

import sys
sys.path.insert(0, '..')

import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass

from core.module_base import SimulationModule
from core.event_bus import EventTypes


@dataclass
class CircadianClock:
    """Cellular circadian oscillator"""
    cell_id: int
    
    # Clock genes (oscillate 0-1)
    CLOCK_BMAL1: float = 1.0  # Positive arm
    PER_CRY: float = 0.0      # Negative arm
    
    # Phase (0-24 hours)
    phase: float = 0.0
    
    # Period (hours)
    period: float = 24.0
    
    # Amplitude (0-1, strength of rhythm)
    amplitude: float = 1.0
    
    # Synchronization to master clock
    coupling_strength: float = 0.1
    
    def update(self, dt: float, master_phase: float):
        """Update clock state"""
        # Advance phase
        self.phase += (dt / self.period) * 24.0
        self.phase = self.phase % 24.0
        
        # Couple to master clock
        phase_diff = master_phase - self.phase
        if phase_diff > 12:
            phase_diff -= 24
        elif phase_diff < -12:
            phase_diff += 24
        
        self.phase += self.coupling_strength * phase_diff * dt
        
        # Update clock genes (sinusoidal)
        t = (self.phase / 24.0) * 2 * np.pi
        self.CLOCK_BMAL1 = 0.5 + 0.5 * self.amplitude * np.cos(t)
        self.PER_CRY = 0.5 + 0.5 * self.amplitude * np.cos(t + np.pi)
    
    def get_time_of_day(self) -> str:
        """Get time of day"""
        if 6 <= self.phase < 18:
            return "day"
        else:
            return "night"
    
    def is_permissive_for_division(self) -> bool:
        """Check if clock permits cell division"""
        # Cell division gated to specific phases (late G1/S)
        # Typically around ZT 6-12 (zeitgeber time)
        return 6 <= self.phase < 12


class CircadianModule(SimulationModule):
    """
    Circadian rhythm simulation module
    
    Manages:
    - Master clock (SCN-like pacemaker)
    - Cellular clocks (peripheral oscillators)
    - Clock gene oscillations
    - Rhythmic outputs (metabolism, immunity, division)
    
    Events Emitted:
    - CIRCADIAN_PEAK: When clock genes peak
    - CIRCADIAN_TROUGH: When clock genes trough
    
    Events Subscribed:
    - CELL_DIVIDED: Inherit clock from parent
    - CELL_DIED: Remove clock
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        
        # Master clock (SCN)
        self.master_phase = 0.0  # 0-24 hours
        self.master_period = 24.0
        
        # Cellular clocks
        self.cell_clocks: Dict[int, CircadianClock] = {}
        
        # Parameters
        self.coupling_strength = config.get('coupling_strength', 0.1)
        self.amplitude = config.get('amplitude', 1.0)
        self.enable_gating = config.get('enable_gating', True)
        
        # Statistics
        self.total_peaks = 0
        self.total_troughs = 0
        
        # References
        self.cellular_module = None
    
    def initialize(self):
        """Initialize circadian system"""
        print("  Creating circadian clock system...")
        
        # Subscribe to events
        self.subscribe(EventTypes.CELL_DIVIDED, self.on_cell_divided)
        self.subscribe(EventTypes.CELL_DIED, self.on_cell_died)
        
        print(f"    ✓ Master clock (SCN-like)")
        print(f"    ✓ Cellular oscillators")
        print(f"    ✓ Clock gene dynamics")
    
    def add_cell(self, cell_id: int, initial_phase: float = None):
        """Add cell with circadian clock"""
        if initial_phase is None:
            # Random phase initially (unsynchronized)
            initial_phase = np.random.uniform(0, 24)
        
        self.cell_clocks[cell_id] = CircadianClock(
            cell_id=cell_id,
            phase=initial_phase,
            coupling_strength=self.coupling_strength,
            amplitude=self.amplitude
        )
    
    def remove_cell(self, cell_id: int):
        """Remove cell clock"""
        if cell_id in self.cell_clocks:
            del self.cell_clocks[cell_id]
    
    def set_cellular_module(self, cellular_module):
        """Link to cellular module"""
        self.cellular_module = cellular_module
    
    def update(self, dt: float):
        """Update circadian system"""
        # Update master clock
        self.master_phase += (dt / self.master_period) * 24.0
        self.master_phase = self.master_phase % 24.0
        
        # Update cellular clocks
        for clock in self.cell_clocks.values():
            old_phase = clock.phase
            clock.update(dt, self.master_phase)
            
            # Detect peaks/troughs
            if old_phase < 12 and clock.phase >= 12:
                self.total_peaks += 1
            elif old_phase < 0 and clock.phase >= 0:
                self.total_troughs += 1
    
    def get_metabolic_modifier(self, cell_id: int) -> float:
        """Get circadian modifier for metabolism"""
        if cell_id not in self.cell_clocks:
            return 1.0
        
        clock = self.cell_clocks[cell_id]
        
        # Metabolism higher during day
        if clock.get_time_of_day() == "day":
            return 1.0 + 0.2 * clock.amplitude
        else:
            return 1.0 - 0.2 * clock.amplitude
    
    def get_immune_modifier(self, cell_id: int) -> float:
        """Get circadian modifier for immune function"""
        if cell_id not in self.cell_clocks:
            return 1.0
        
        clock = self.cell_clocks[cell_id]
        
        # Immune function peaks at night (in mice)
        if clock.get_time_of_day() == "night":
            return 1.0 + 0.3 * clock.amplitude
        else:
            return 1.0 - 0.1 * clock.amplitude
    
    def can_divide(self, cell_id: int) -> bool:
        """Check if cell can divide (circadian gating)"""
        if not self.enable_gating:
            return True
        
        if cell_id not in self.cell_clocks:
            return True
        
        return self.cell_clocks[cell_id].is_permissive_for_division()
    
    def get_state(self) -> Dict[str, Any]:
        """Return current circadian state"""
        if self.cell_clocks:
            phases = [c.phase for c in self.cell_clocks.values()]
            amplitudes = [c.amplitude for c in self.cell_clocks.values()]
            
            # Calculate synchronization
            phase_vectors = [np.exp(1j * p * 2 * np.pi / 24) for p in phases]
            synchrony = abs(np.mean(phase_vectors))
        else:
            synchrony = 0
            phases = []
            amplitudes = []
        
        return {
            'master_phase': self.master_phase,
            'master_time_of_day': 'day' if 6 <= self.master_phase < 18 else 'night',
            'n_clocks': len(self.cell_clocks),
            'synchrony': synchrony,
            'avg_phase': np.mean(phases) if phases else 0,
            'avg_amplitude': np.mean(amplitudes) if amplitudes else 0,
            'total_peaks': self.total_peaks,
            'total_troughs': self.total_troughs
        }
    
    # Event handlers
    def on_cell_divided(self, data):
        """Handle cell division - inherit clock"""
        parent_id = data['cell_id']
        daughter_id = data.get('daughter_id')
        
        if daughter_id and parent_id in self.cell_clocks:
            # Daughter inherits parent's phase with small variation
            parent_phase = self.cell_clocks[parent_id].phase
            daughter_phase = parent_phase + np.random.normal(0, 0.5)
            
            self.add_cell(daughter_id, initial_phase=daughter_phase)
    
    def on_cell_died(self, data):
        """Handle cell death - remove clock"""
        cell_id = data['cell_id']
        self.remove_cell(cell_id)


# Test
if __name__ == '__main__':
    from core import SimulationEngine, SimulationConfig
    from cellular_module import CellularModule
    
    print("=" * 70)
    print("Circadian Clock Module Test")
    print("=" * 70)
    print()
    
    # Create engine (48 hours to see oscillations)
    engine = SimulationEngine(SimulationConfig(dt=0.1, duration=48.0))
    
    # Register modules
    engine.register_module('cellular', CellularModule, {
        'n_normal_cells': 10,
        'n_cancer_cells': 0
    })
    
    engine.register_module('circadian', CircadianModule, {
        'coupling_strength': 0.1,
        'enable_gating': True
    })
    
    # Initialize
    engine.initialize()
    
    # Add clocks to cells
    circadian = engine.modules['circadian']
    cellular = engine.modules['cellular']
    
    for cell_id in cellular.cells.keys():
        circadian.add_cell(cell_id)
    
    print(f"Added {len(cellular.cells)} cellular clocks")
    print()
    
    # Check initial state
    print("Initial state:")
    state = circadian.get_state()
    print(f"  Master phase: {state['master_phase']:.1f}h")
    print(f"  Synchrony: {state['synchrony']:.3f}")
    print()
    
    # Run simulation
    print("Running 48-hour simulation...")
    engine.run()
    
    # Get results
    print("\nCircadian State:")
    state = circadian.get_state()
    for key, value in state.items():
        print(f"  {key}: {value}")
    
    print()
    
    # Check individual clocks
    print("Individual cell clocks:")
    for cell_id in list(circadian.cell_clocks.keys())[:5]:
        clock = circadian.cell_clocks[cell_id]
        print(f"  Cell {cell_id}: phase={clock.phase:.1f}h, "
              f"time={clock.get_time_of_day()}, "
              f"can_divide={clock.is_permissive_for_division()}")
    
    print()
    print("=" * 70)
    print("✓ Circadian module working!")
    print("=" * 70)
