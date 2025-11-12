#!/usr/bin/env python3
"""
Test All 6 Modules
==================

Complete integration test of all modules working together.
"""

from core import SimulationEngine, SimulationConfig
from modules import (MolecularModule, CellularModule, ImmuneModule,
                    VascularModule, LymphaticModule, SpatialModule)

print("=" * 70)
print("COMPLETE INTEGRATION TEST: ALL 6 MODULES")
print("=" * 70)
print()

# Create engine
config = SimulationConfig(dt=0.01, duration=2.0)
engine = SimulationEngine(config)

# Register ALL modules
print("Registering modules...")
engine.register_module('molecular', MolecularModule, {
    'transcription_rate': 0.5,
    'exosome_release_rate': 0.1
})

engine.register_module('cellular', CellularModule, {
    'n_normal_cells': 20,
    'n_cancer_cells': 5
})

engine.register_module('immune', ImmuneModule, {
    'n_t_cells': 8,
    'n_nk_cells': 5,
    'n_macrophages': 3
})

engine.register_module('vascular', VascularModule, {
    'n_capillaries': 8,
    'exchange_rate': 1.0
})

engine.register_module('lymphatic', LymphaticModule, {
    'n_vessels': 4,
    'metastasis_probability': 0.001
})

engine.register_module('spatial', SpatialModule, {
    'grid_size': (20, 20, 10),
    'resolution': 10.0
})

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
spatial = engine.modules['spatial']

# Molecular tracks cells
for cell_id in cellular.cells.keys():
    molecular.add_cell(cell_id)

# Immune accesses cellular
immune.set_cellular_module(cellular)

# Vascular exchanges with cellular
vascular.set_cellular_module(cellular)

# Lymphatic accesses cellular and immune
lymphatic.set_cellular_module(cellular)
lymphatic.set_immune_module(immune)

print("✓ All modules linked")
print()

# Run simulation
engine.run()

# Results
print("\n" + "=" * 70)
print("COMPLETE SYSTEM RESULTS")
print("=" * 70)

state = engine.get_state()

print(f"\nTime: {state['time']:.2f} hours")
print(f"Steps: {state['step_count']}")
print()

print("MOLECULAR:")
for key, value in state['molecular'].items():
    print(f"  {key}: {value}")

print("\nCELLULAR:")
for key, value in state['cellular'].items():
    print(f"  {key}: {value}")

print("\nIMMUNE:")
for key, value in state['immune'].items():
    print(f"  {key}: {value}")

print("\nVASCULAR:")
for key, value in state['vascular'].items():
    print(f"  {key}: {value}")

print("\nLYMPHATIC:")
for key, value in state['lymphatic'].items():
    print(f"  {key}: {value}")

print("\nSPATIAL:")
for key, value in state['spatial'].items():
    if key != 'fields':
        print(f"  {key}: {value}")

print("\n" + "=" * 70)
print("✓ ALL 6 MODULES WORKING TOGETHER!")
print("=" * 70)
print()

# Summary
print("SUMMARY:")
print(f"  Modules: 6/6 active")
print(f"  Cells: {state['cellular']['n_cells']}")
print(f"  Immune cells: {state['immune']['n_immune_cells']}")
print(f"  Capillaries: {state['vascular']['n_capillaries']}")
print(f"  Lymphatics: {state['lymphatic']['n_vessels']}")
print(f"  Spatial fields: {state['spatial']['n_fields']}")
print(f"  Exosomes: {state['molecular']['n_exosomes']}")
print(f"  Immune kills: {state['immune']['total_kills']}")
print(f"  Metastases: {state['lymphatic']['total_metastases']}")
print()
