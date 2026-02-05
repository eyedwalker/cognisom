#!/usr/bin/env python3
"""
Hypoxia Response Scenario
==========================

Simulate low oxygen environment.
"""

import sys
sys.path.insert(0, '..')

from core import SimulationEngine, SimulationConfig
from modules import CellularModule, VascularModule


def run_hypoxia_scenario():
    """Run hypoxia scenario"""
    print("=" * 70)
    print("SCENARIO: Hypoxia Response")
    print("=" * 70)
    print()
    print("Intervention: Reduce oxygen supply")
    print("  - Arterial O2: 21% → 10%")
    print("  - Capillaries: 8 → 3")
    print("  - Duration: 24 hours")
    print()
    
    config = SimulationConfig(dt=0.01, duration=24.0)
    engine = SimulationEngine(config)
    
    engine.register_module('cellular', CellularModule, {
        'n_normal_cells': 20,
        'n_cancer_cells': 5
    })
    engine.register_module('vascular', VascularModule, {
        'n_capillaries': 3,      # REDUCED!
        'arterial_O2': 0.10      # LOW!
    })
    
    engine.initialize()
    
    vascular = engine.modules['vascular']
    cellular = engine.modules['cellular']
    vascular.set_cellular_module(cellular)
    
    print("✓ Modules linked")
    print()
    print("Running simulation...")
    engine.run()
    
    state = engine.get_state()
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nAvg O2: {state['vascular']['avg_cell_O2']:.3f}")
    print(f"Hypoxic regions: {state['vascular']['hypoxic_regions']}")
    print(f"Cell deaths: {state['cellular']['total_deaths']}")
    print(f"Surviving cells: {state['cellular']['n_cells']}")
    
    if state['vascular']['hypoxic_regions'] > 10:
        print("\n⚠ SEVERE HYPOXIA: Major cell death")
    else:
        print("\n✓ MODERATE HYPOXIA: Some adaptation")
    
    return state


if __name__ == '__main__':
    run_hypoxia_scenario()
