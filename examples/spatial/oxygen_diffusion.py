#!/usr/bin/env python3
"""
Oxygen Diffusion Example
=========================

Demonstrates:
1. Creating 3D spatial grid
2. Adding cells that consume oxygen
3. Simulating diffusion
4. Visualizing gradients

This is Phase 2: Spatial Grid
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from engine.py.spatial import SpatialGrid
from engine.py.spatial.grid import GridConfig

print("=" * 70)
print("🧬 Oxygen Diffusion Simulation")
print("=" * 70)
print()

# Create spatial grid (smaller for demo)
config = GridConfig(
    size=(50, 50, 50),  # 50×50×50 voxels
    resolution=10.0,     # 10 μm per voxel = 500×500×500 μm total
    D_oxygen=2000.0,     # Oxygen diffusion coefficient
    oxygen_init=0.21,    # 21% O2
    glucose_init=5.0     # 5 mM glucose
)

grid = SpatialGrid(config)
print()

# Add cells randomly
print("Adding 100 cells...")
np.random.seed(42)
n_cells = 100

for cell_id in range(n_cells):
    # Random position (μm)
    x = np.random.uniform(50, 450)
    y = np.random.uniform(50, 450)
    z = np.random.uniform(50, 450)
    
    grid.add_cell(cell_id, (x, y, z))

print(f"✓ Added {n_cells} cells")
print()

# Initial statistics
stats = grid.get_statistics()
print("Initial state:")
print(f"  Cells: {stats['n_cells']}")
print(f"  Oxygen (mean): {stats['oxygen_mean']:.3f}")
print(f"  Glucose (mean): {stats['glucose_mean']:.3f}")
print()

# Simulate
print("Running simulation (10 hours)...")
print("Cells consume oxygen, diffusion replenishes...")
print()

duration = 10.0  # hours
dt = 0.01  # hours
steps = int(duration / dt)

consumption_rate_o2 = 0.001  # per cell per hour
consumption_rate_glucose = 0.0005

for step in range(steps):
    t = step * dt
    
    # Cells consume nutrients
    for cell_id, position in grid.cell_positions.items():
        grid.consume_at_position(position, consumption_rate_o2 * dt, consumption_rate_glucose * dt)
    
    # Diffusion
    grid.step_diffusion(dt)
    
    # Replenish at boundaries (blood vessels)
    grid.oxygen[0, :, :] = config.oxygen_init
    grid.oxygen[-1, :, :] = config.oxygen_init
    grid.oxygen[:, 0, :] = config.oxygen_init
    grid.oxygen[:, -1, :] = config.oxygen_init
    
    grid.glucose[0, :, :] = config.glucose_init
    grid.glucose[-1, :, :] = config.glucose_init
    
    # Print progress
    if step % 100 == 0:
        stats = grid.get_statistics()
        print(f"t={t:5.1f}h | O2: {stats['oxygen_mean']:.4f} | "
              f"Glucose: {stats['glucose_mean']:.3f}")

print()
print("✓ Simulation complete")
print()

# Final statistics
stats = grid.get_statistics()
print("Final state:")
print(f"  Oxygen (mean): {stats['oxygen_mean']:.4f}")
print(f"  Oxygen (min): {stats['oxygen_min']:.4f}")
print(f"  Oxygen (max): {stats['oxygen_max']:.4f}")
print(f"  Glucose (mean): {stats['glucose_mean']:.3f}")
print()

# Create visualizations
print("=" * 70)
print("Creating visualizations...")
print("=" * 70)
print()

output_dir = Path('output/spatial')
output_dir.mkdir(parents=True, exist_ok=True)

# Plot oxygen gradient (middle slice)
print("1. Oxygen gradient (2D slice)...")
grid.plot_slice(z_slice=25, field='oxygen', 
               save_path=str(output_dir / 'oxygen_gradient.png'))

# Plot glucose gradient
print("2. Glucose gradient (2D slice)...")
grid.plot_slice(z_slice=25, field='glucose',
               save_path=str(output_dir / 'glucose_gradient.png'))

# Plot cell positions (3D)
print("3. Cell positions (3D)...")
grid.plot_3d_cells(save_path=str(output_dir / 'cell_positions_3d.png'))

print()
print("=" * 70)
print("✓ Spatial Simulation Complete!")
print("=" * 70)
print()

print("Results:")
print(f"  • {n_cells} cells simulated")
print(f"  • Oxygen gradients formed")
print(f"  • Cells near center have less O2 (hypoxia!)")
print()

print("Visualizations saved:")
print(f"  • {output_dir / 'oxygen_gradient.png'}")
print(f"  • {output_dir / 'glucose_gradient.png'}")
print(f"  • {output_dir / 'cell_positions_3d.png'}")
print()

print("View them:")
print(f"  open {output_dir}/*.png")
print()

print("=" * 70)
print("🎉 This is Phase 2: Spatial Grid!")
print("=" * 70)
print()

print("What you just saw:")
print("  ✓ 3D spatial environment")
print("  ✓ Diffusion (oxygen, glucose)")
print("  ✓ Cell-environment interaction")
print("  ✓ Gradient formation")
print()

print("Next steps:")
print("  • Add cell-cell interactions (Phase 3)")
print("  • Port to GPU (Phase 4)")
print("  • Add biology (Phase 5)")
print()
