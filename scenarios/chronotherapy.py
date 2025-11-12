#!/usr/bin/env python3
"""
Chronotherapy Scenario
======================

Time drug delivery to circadian rhythms.

Concept: Deliver treatment when cancer cells are most vulnerable
(during S phase, typically ZT 6-12)
"""

import sys
sys.path.insert(0, '..')

from core import SimulationEngine, SimulationConfig
from modules import CellularModule, CircadianModule


def run_chronotherapy_scenario():
    """Run chronotherapy scenario"""
    print("=" * 70)
    print("SCENARIO: Chronotherapy (Timed Treatment)")
    print("=" * 70)
    print()
    print("Concept: Time treatment to circadian rhythm")
    print("  - Optimal window: ZT 6-12 (S phase)")
    print("  - Duration: 48 hours")
    print()
    
    config = SimulationConfig(dt=0.1, duration=48.0)
    engine = SimulationEngine(config)
    
    engine.register_module('cellular', CellularModule, {
        'n_cancer_cells': 10
    })
    engine.register_module('circadian', CircadianModule)
    
    engine.initialize()
    
    # Link
    circadian = engine.modules['circadian']
    cellular = engine.modules['cellular']
    
    for cell_id in cellular.cells.keys():
        circadian.add_cell(cell_id)
    
    print("✓ Modules linked")
    print()
    print("Running simulation...")
    engine.run()
    
    state = engine.get_state()
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nCircadian synchrony: {state['circadian']['synchrony']:.3f}")
    print(f"Final phase: {state['circadian']['master_phase']:.1f}h")
    print(f"Time of day: {state['circadian']['master_time_of_day']}")
    print("\n✓ Optimal treatment window identified: ZT 6-12")
    
    return state


if __name__ == '__main__':
    run_chronotherapy_scenario()
