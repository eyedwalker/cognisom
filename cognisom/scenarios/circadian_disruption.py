#!/usr/bin/env python3
"""
Circadian Disruption Scenario
==============================

Simulate jet lag / shift work effects.
"""

import sys
sys.path.insert(0, '..')

from core import SimulationEngine, SimulationConfig
from modules import CellularModule, CircadianModule


def run_circadian_disruption_scenario():
    """Run circadian disruption scenario"""
    print("=" * 70)
    print("SCENARIO: Circadian Disruption (Jet Lag)")
    print("=" * 70)
    print()
    print("Intervention: Shift master clock by 12 hours")
    print("  - Simulate jet lag")
    print("  - Observe desynchronization")
    print("  - Duration: 72 hours (3 days)")
    print()
    
    config = SimulationConfig(dt=0.1, duration=72.0)
    engine = SimulationEngine(config)
    
    engine.register_module('cellular', CellularModule)
    engine.register_module('circadian', CircadianModule)
    
    engine.initialize()
    
    circadian = engine.modules['circadian']
    cellular = engine.modules['cellular']
    
    for cell_id in cellular.cells.keys():
        circadian.add_cell(cell_id)
    
    # SHIFT CLOCK (jet lag!)
    circadian.master_phase = 12.0
    
    print("✓ Modules linked")
    print("✓ Clock shifted by 12 hours (jet lag)")
    print()
    print("Running simulation...")
    engine.run()
    
    state = engine.get_state()
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nSynchrony: {state['circadian']['synchrony']:.3f}")
    print(f"Master phase: {state['circadian']['master_phase']:.1f}h")
    
    if state['circadian']['synchrony'] < 0.5:
        print("\n⚠ DESYNCHRONIZED: Cells out of phase")
    else:
        print("\n✓ RESYNCHRONIZED: Cells adapted")
    
    return state


if __name__ == '__main__':
    run_circadian_disruption_scenario()
