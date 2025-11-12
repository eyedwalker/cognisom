#!/usr/bin/env python3
"""
Platform Test Script
====================

Test all platform components non-interactively.
"""

print("=" * 70)
print("🧬 cognisom Platform Test")
print("=" * 70)
print()

# Test 1: Core modules
print("TEST 1: Core Modules")
print("-" * 70)

from core import SimulationEngine, SimulationConfig
from modules import (MolecularModule, CellularModule, ImmuneModule,
                    VascularModule, LymphaticModule, SpatialModule,
                    EpigeneticModule, CircadianModule, MorphogenModule)

print("✓ All modules imported successfully")

# Create engine
config = SimulationConfig(dt=0.01, duration=0.5)
engine = SimulationEngine(config)

# Register all modules
print("\nRegistering modules...")
engine.register_module('molecular', MolecularModule)
engine.register_module('cellular', CellularModule, {'n_normal_cells': 10, 'n_cancer_cells': 3})
engine.register_module('immune', ImmuneModule, {'n_t_cells': 5, 'n_nk_cells': 3})
engine.register_module('vascular', VascularModule, {'n_capillaries': 4})
engine.register_module('lymphatic', LymphaticModule, {'n_vessels': 2})
engine.register_module('spatial', SpatialModule)
engine.register_module('epigenetic', EpigeneticModule)
engine.register_module('circadian', CircadianModule)
engine.register_module('morphogen', MorphogenModule)

print("✓ All 9 modules registered")

# Initialize
engine.initialize()

# Link modules
print("\nLinking modules...")
molecular = engine.modules['molecular']
cellular = engine.modules['cellular']
immune = engine.modules['immune']
vascular = engine.modules['vascular']
lymphatic = engine.modules['lymphatic']
epigenetic = engine.modules['epigenetic']
circadian = engine.modules['circadian']
morphogen = engine.modules['morphogen']

for cell_id, cell in cellular.cells.items():
    molecular.add_cell(cell_id)
    epigenetic.add_cell(cell_id, cell.cell_type)
    circadian.add_cell(cell_id)
    morphogen.add_cell(cell_id, cell.position)

immune.set_cellular_module(cellular)
vascular.set_cellular_module(cellular)
lymphatic.set_cellular_module(cellular)
lymphatic.set_immune_module(immune)

print("✓ All modules linked")

# Run simulation
print("\nRunning simulation (0.5 hours)...")
engine.run()

state = engine.get_state()
print(f"✓ Simulation complete: {state['time']:.2f}h, {state['step_count']} steps")
print()

# Test 2: Data Export
print("TEST 2: Data Export")
print("-" * 70)

engine.export_to_csv('test_export.csv')
engine.export_to_json('test_export.json')
print("✓ Data export working")
print()

# Test 3: Scenarios
print("TEST 3: Scenarios")
print("-" * 70)

from scenarios import run_immunotherapy_scenario

print("Running immunotherapy scenario...")
result = run_immunotherapy_scenario()
print(f"✓ Scenario complete: {result['cellular']['n_cancer']} cancer cells remaining")
print()

# Test 4: Publisher
print("TEST 4: Publication System")
print("-" * 70)

from api.publisher import Publisher

publisher = Publisher(engine)
files = publisher.generate_all_formats('test_report')
print("✓ Reports generated:")
for format_type, filename in files.items():
    print(f"  - {format_type}: {filename}")
print()

# Test 5: Performance
print("TEST 5: Performance Optimizations")
print("-" * 70)

from core.performance import SpatialIndex, VectorizedOperations
import numpy as np

# Test spatial index
positions = np.random.rand(100, 3) * 200
ids = list(range(100))
index = SpatialIndex()
index.build(positions, ids)
neighbors = index.query_radius(np.array([100, 100, 50]), radius=50)
print(f"✓ Spatial index: Found {len(neighbors)} neighbors")

# Test vectorized operations
pos1 = np.random.rand(50, 3) * 200
pos2 = np.random.rand(30, 3) * 200
distances = VectorizedOperations.batch_distance(pos1, pos2)
print(f"✓ Vectorized operations: {distances.shape} distance matrix")
print()

# Summary
print("=" * 70)
print("✓ ALL TESTS PASSED!")
print("=" * 70)
print()
print("Platform Status:")
print("  ✓ 9 modules working")
print("  ✓ Data export functional")
print("  ✓ Scenarios running")
print("  ✓ Publication system ready")
print("  ✓ Performance optimizations active")
print()
print("Ready to use!")
print()
print("Next steps:")
print("  1. Run: python3 api/rest_server.py (for web dashboard)")
print("  2. Run: python3 ui/control_panel.py (for GUI)")
print("  3. Run: python3 visualize_complete.py (for visualization)")
print("  4. Or use programmatically in your own scripts")
print()
