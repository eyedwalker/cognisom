#!/usr/bin/env python3
"""
Cancer Immunotherapy Scenario
==============================

Simulate boosting immune system to fight cancer.

Intervention:
- Increase T cells (5 → 50)
- Increase NK cells (3 → 30)
- Run for 48 hours
- Measure cancer elimination
"""

import sys
sys.path.insert(0, '..')

from core import SimulationEngine, SimulationConfig
from modules import (MolecularModule, CellularModule, ImmuneModule,
                    VascularModule, LymphaticModule)


def run_immunotherapy_scenario():
    """Run immunotherapy scenario"""
    print("=" * 70)
    print("SCENARIO: Cancer Immunotherapy")
    print("=" * 70)
    print()
    print("Intervention: Boost immune system")
    print("  - T cells: 5 → 50 (10x increase)")
    print("  - NK cells: 3 → 30 (10x increase)")
    print("  - Duration: 48 hours")
    print()
    
    # Create engine
    config = SimulationConfig(dt=0.01, duration=48.0)
    engine = SimulationEngine(config)
    
    # Register modules
    engine.register_module('molecular', MolecularModule)
    engine.register_module('cellular', CellularModule, {
        'n_normal_cells': 20,
        'n_cancer_cells': 10  # Significant cancer burden
    })
    engine.register_module('immune', ImmuneModule, {
        'n_t_cells': 50,      # BOOSTED!
        'n_nk_cells': 30,     # BOOSTED!
        'n_macrophages': 15   # BOOSTED!
    })
    engine.register_module('vascular', VascularModule)
    engine.register_module('lymphatic', LymphaticModule)
    
    # Initialize
    engine.initialize()
    
    # Link modules
    molecular = engine.modules['molecular']
    cellular = engine.modules['cellular']
    immune = engine.modules['immune']
    vascular = engine.modules['vascular']
    lymphatic = engine.modules['lymphatic']
    
    for cell_id in cellular.cells.keys():
        molecular.add_cell(cell_id)
    
    immune.set_cellular_module(cellular)
    vascular.set_cellular_module(cellular)
    lymphatic.set_cellular_module(cellular)
    lymphatic.set_immune_module(immune)
    
    print("✓ Modules initialized and linked")
    print()
    
    # Run simulation
    print("Running simulation...")
    engine.run()
    
    # Analyze results
    state = engine.get_state()
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"Duration: {state['time']:.1f} hours")
    print()
    print("CANCER CONTROL:")
    print(f"  Initial cancer cells: 10")
    print(f"  Final cancer cells: {state['cellular']['n_cancer']}")
    print(f"  Cancer eliminated: {10 - state['cellular']['n_cancer']}")
    print(f"  Elimination rate: {((10 - state['cellular']['n_cancer']) / 10 * 100):.1f}%")
    print()
    print("IMMUNE RESPONSE:")
    print(f"  Total immune cells: {state['immune']['n_immune_cells']}")
    print(f"  Activated: {state['immune']['n_activated']}")
    print(f"  Total kills: {state['immune']['total_kills']}")
    print(f"  Kill rate: {(state['immune']['total_kills'] / 48.0):.2f} kills/hour")
    print()
    
    # Verdict
    if state['cellular']['n_cancer'] < 3:
        print("✓ SUCCESS: Cancer largely eliminated!")
    elif state['cellular']['n_cancer'] < 7:
        print("⚠ PARTIAL: Cancer reduced but not eliminated")
    else:
        print("✗ FAILURE: Cancer persists despite treatment")
    
    print()
    return state


if __name__ == '__main__':
    run_immunotherapy_scenario()
