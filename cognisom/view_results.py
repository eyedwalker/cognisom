#!/usr/bin/env python3
"""
Quick results viewer
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
results_path = Path('output/basic_growth/results.json')
with open(results_path) as f:
    results = json.load(f)

# Print summary
print("=" * 60)
print("Simulation Results Summary")
print("=" * 60)
print(f"Duration: {results['duration']} hours")
print(f"Time step: {results['dt']} hours")
print(f"Final cells: {results['final_cells']}")
print(f"Total divisions: {results['events']['divisions']}")
print(f"Total deaths: {results['events']['deaths']}")
print()

# Show data points
history = results['history']
print(f"Data points collected: {len(history['time'])}")
print()
print("Time (h) | Cells | Total Proteins")
print("-" * 40)
for i in range(0, len(history['time']), 5):  # Every 5th point
    t = history['time'][i]
    cells = history['cell_count'][i]
    proteins = history['total_proteins'][i]
    print(f"{t:8.1f} | {cells:5d} | {proteins:14d}")

print()
print("=" * 60)
print("Opening visualization...")
print("=" * 60)

# Show the plot
img_path = Path('output/basic_growth/simulation_results.png')
if img_path.exists():
    import subprocess
    subprocess.run(['open', str(img_path)])
    print(f"✓ Opened: {img_path}")
else:
    print(f"⚠️  Image not found: {img_path}")
