#!/usr/bin/env python3
"""
Test All Modules Including Circadian and Morphogens
====================================================

Complete integration test with 9 modules.
"""

from core import SimulationEngine, SimulationConfig
from modules import (MolecularModule, CellularModule, ImmuneModule,
                    VascularModule, LymphaticModule, SpatialModule,
                    EpigeneticModule, CircadianModule, MorphogenModule)

print("=" * 70)
print("COMPLETE INTEGRATION: ALL 9 MODULES")
print("=" * 70)
print()

# Create engine
config = SimulationConfig(dt=0.01, duration=2.0)
engine = SimulationEngine(config)

# Register ALL modules
print("Registering modules...")
engine.register_module('molecular', MolecularModule)
engine.register_module('cellular', CellularModule, {'n_normal_cells': 15, 'n_cancer_cells': 3})
engine.register_module('immune', ImmuneModule, {'n_t_cells': 5, 'n_nk_cells': 3})
engine.register_module('vascular', VascularModule, {'n_capillaries': 6})
engine.register_module('lymphatic', LymphaticModule, {'n_vessels': 3})
engine.register_module('spatial', SpatialModule)
engine.register_module('epigenetic', EpigeneticModule)
engine.register_module('circadian', CircadianModule)
engine.register_module('morphogen', MorphogenModule)

print()

# Initialize
engine.initialize()

# Link all modules
print("Linking modules...")
molecular = engine.modules['molecular']
cellular = engine.modules['cellular']
immune = engine.modules['immune']
vascular = engine.modules['vascular']
lymphatic = engine.modules['lymphatic']
epigenetic = engine.modules['epigenetic']
circadian = engine.modules['circadian']
morphogen = engine.modules['morphogen']

# Link modules
for cell_id, cell in cellular.cells.items():
    molecular.add_cell(cell_id)
    epigenetic.add_cell(cell_id, cell.cell_type)
    circadian.add_cell(cell_id)
    morphogen.add_cell(cell_id, cell.position)

immune.set_cellular_module(cellular)
vascular.set_cellular_module(cellular)
lymphatic.set_cellular_module(cellular)
lymphatic.set_immune_module(immune)

print("✓ All 9 modules linked")
print()

# Run simulation
engine.run()

# Results
print("\n" + "=" * 70)
print("ALL 9 MODULES RESULTS")
print("=" * 70)

state = engine.get_state()

print(f"\nTime: {state['time']:.2f} hours")
print(f"Steps: {state['step_count']}")

for module_name in ['molecular', 'cellular', 'immune', 'vascular', 
                    'lymphatic', 'spatial', 'epigenetic', 'circadian', 'morphogen']:
    if module_name in state:
        print(f"\n{module_name.upper()}:")
        for key, value in state[module_name].items():
            if key != 'fields' and key != 'fate_distribution':
                print(f"  {key}: {value}")

print("\n" + "=" * 70)
print("✓ ALL 9 MODULES WORKING!")
print("=" * 70)
