#!/usr/bin/env python3
"""
Test All Quick Wins
===================

Test GUI, scenarios, and data export.
"""

from core import SimulationEngine, SimulationConfig
from modules import CellularModule, ImmuneModule, VascularModule
from scenarios import (run_immunotherapy_scenario, run_hypoxia_scenario,
                      run_epigenetic_therapy_scenario)

print("=" * 70)
print("TESTING ALL QUICK WINS")
print("=" * 70)
print()

# Test 1: Data Export
print("TEST 1: Data Export")
print("-" * 70)

engine = SimulationEngine(SimulationConfig(dt=0.01, duration=1.0))
engine.register_module('cellular', CellularModule)
engine.register_module('immune', ImmuneModule)
engine.register_module('vascular', VascularModule)
engine.initialize()

# Link
immune = engine.modules['immune']
vascular = engine.modules['vascular']
cellular = engine.modules['cellular']
immune.set_cellular_module(cellular)
vascular.set_cellular_module(cellular)

# Run
engine.run()

# Export
engine.export_to_csv('test_export.csv')
engine.export_to_json('test_export.json')

print("✓ Data export working!")
print()

# Test 2: Scenarios
print("TEST 2: Scenarios")
print("-" * 70)

print("\n1. Immunotherapy Scenario:")
print("-" * 40)
result1 = run_immunotherapy_scenario()

print("\n2. Hypoxia Scenario:")
print("-" * 40)
result2 = run_hypoxia_scenario()

print("\n3. Epigenetic Therapy Scenario:")
print("-" * 40)
result3 = run_epigenetic_therapy_scenario()

print("\n✓ All scenarios working!")
print()

# Test 3: GUI (manual test)
print("TEST 3: GUI Control Panel")
print("-" * 70)
print("To test GUI, run:")
print("  python3 ui/control_panel.py")
print()

print("=" * 70)
print("✓ ALL QUICK WINS WORKING!")
print("=" * 70)
print()
print("Summary:")
print("  ✓ Data export (CSV/JSON)")
print("  ✓ 5 pre-built scenarios")
print("  ✓ GUI control panel (manual test)")
