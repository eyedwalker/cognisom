#!/usr/bin/env python3
"""
Epigenetic Therapy Scenario
============================

Simulate DNA methyltransferase inhibitors (DNMTi).
"""

import sys
sys.path.insert(0, '..')

from core import SimulationEngine, SimulationConfig
from modules import CellularModule, EpigeneticModule


def run_epigenetic_therapy_scenario():
    """Run epigenetic therapy scenario"""
    print("=" * 70)
    print("SCENARIO: Epigenetic Therapy (DNMTi)")
    print("=" * 70)
    print()
    print("Intervention: DNA methyltransferase inhibitors")
    print("  - Demethylation rate: 0.005 → 0.05 (10x)")
    print("  - Reactivate silenced tumor suppressors")
    print("  - Duration: 24 hours")
    print()
    
    config = SimulationConfig(dt=0.01, duration=24.0)
    engine = SimulationEngine(config)
    
    engine.register_module('cellular', CellularModule, {
        'n_cancer_cells': 10
    })
    engine.register_module('epigenetic', EpigeneticModule, {
        'methylation_rate': 0.001,
        'demethylation_rate': 0.05  # BOOSTED!
    })
    
    engine.initialize()
    
    epigenetic = engine.modules['epigenetic']
    cellular = engine.modules['cellular']
    
    for cell_id, cell in cellular.cells.items():
        epigenetic.add_cell(cell_id, cell.cell_type)
    
    print("✓ Modules linked")
    print()
    print("Running simulation...")
    engine.run()
    
    state = engine.get_state()
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nAvg methylation: {state['epigenetic']['avg_methylation']:.3f}")
    print(f"Silenced genes: {state['epigenetic']['silenced_genes']}")
    print(f"Active genes: {state['epigenetic']['active_genes']}")
    
    if state['epigenetic']['avg_methylation'] < 0.3:
        print("\n✓ SUCCESS: Tumor suppressors reactivated!")
    else:
        print("\n⚠ PARTIAL: Some demethylation occurred")
    
    return state


if __name__ == '__main__':
    run_epigenetic_therapy_scenario()
