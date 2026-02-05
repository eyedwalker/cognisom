#!/usr/bin/env python3
"""
Test Full Integration
=====================

Test all three modules working together:
- Molecular (DNA/RNA/exosomes)
- Cellular (cells/metabolism/division)
- Immune (T cells/NK cells/killing)
"""

from core import SimulationEngine, SimulationConfig
from modules import MolecularModule, CellularModule, ImmuneModule

print("=" * 70)
print("FULL INTEGRATION TEST: Molecular + Cellular + Immune")
print("=" * 70)
print()

# Create engine
config = SimulationConfig(dt=0.01, duration=2.0)
engine = SimulationEngine(config)

# Register all modules
engine.register_module('molecular', MolecularModule, {
    'transcription_rate': 0.5,
    'exosome_release_rate': 0.1
})

engine.register_module('cellular', CellularModule, {
    'n_normal_cells': 20,
    'n_cancer_cells': 5,
    'division_time_cancer': 1.5  # Fast for testing
})

engine.register_module('immune', ImmuneModule, {
    'n_t_cells': 8,
    'n_nk_cells': 5,
    'n_macrophages': 3
})

print()

# Initialize
engine.initialize()

# Link modules
molecular = engine.modules['molecular']
cellular = engine.modules['cellular']
immune = engine.modules['immune']

print("Linking modules...")
# Molecular tracks all cells
for cell_id in cellular.cells.keys():
    molecular.add_cell(cell_id)

# Immune can access cellular
immune.set_cellular_module(cellular)

print("✓ Modules linked")
print()

# Run simulation
engine.run()

# Results
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

state = engine.get_state()

print("\nMolecular:")
for key, value in state['molecular'].items():
    print(f"  {key}: {value}")

print("\nCellular:")
for key, value in state['cellular'].items():
    print(f"  {key}: {value}")

print("\nImmune:")
for key, value in state['immune'].items():
    print(f"  {key}: {value}")

print("\n" + "=" * 70)
print("✓ FULL INTEGRATION WORKING!")
print("=" * 70)
print()

# Summary
print("Summary:")
print(f"  Time: {state['time']:.2f} hours")
print(f"  Total cells: {state['cellular']['n_cells']}")
print(f"  Cancer cells: {state['cellular']['n_cancer']}")
print(f"  Immune kills: {state['immune']['total_kills']}")
print(f"  Cell divisions: {state['cellular']['total_divisions']}")
print(f"  Exosomes: {state['molecular']['n_exosomes']}")
print()
