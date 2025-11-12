#!/usr/bin/env python3
"""
Colab-Friendly Visualization
=============================

Static visualizations that work in Google Colab.
"""

import matplotlib.pyplot as plt
import numpy as np
from core import SimulationEngine, SimulationConfig
from modules import CellularModule, ImmuneModule, VascularModule


def run_and_visualize(duration=24.0):
    """Run simulation and create visualizations"""
    
    print("🧬 cognisom - Colab Visualization")
    print("=" * 60)
    
    # Create simulation
    engine = SimulationEngine(SimulationConfig(dt=0.01, duration=duration))
    
    engine.register_module('cellular', CellularModule, {
        'n_normal_cells': 50,
        'n_cancer_cells': 10
    })
    engine.register_module('immune', ImmuneModule, {
        'n_t_cells': 15,
        'n_nk_cells': 10
    })
    engine.register_module('vascular', VascularModule)
    
    engine.initialize()
    
    # Link modules
    immune = engine.modules['immune']
    cellular = engine.modules['cellular']
    vascular = engine.modules['vascular']
    
    immune.set_cellular_module(cellular)
    vascular.set_cellular_module(cellular)
    
    # Track data
    time_points = []
    cancer_counts = []
    normal_counts = []
    immune_kills = []
    oxygen_levels = []
    
    print(f"\n🚀 Running {duration}-hour simulation...")
    
    # Run with tracking
    steps = int(duration / 0.01)
    record_interval = steps // 100  # Record 100 points
    
    for step in range(steps):
        engine.step()
        
        if step % record_interval == 0:
            state = engine.get_state()
            time_points.append(state['time'])
            cancer_counts.append(state['cellular']['n_cancer'])
            normal_counts.append(state['cellular']['n_normal'])
            immune_kills.append(state['immune']['total_kills'])
            oxygen_levels.append(state['vascular']['avg_cell_O2'])
    
    print("✅ Simulation complete!")
    
    # Create visualizations
    print("\n📊 Generating visualizations...")
    
    # Figure 1: Cell populations
    fig1 = plt.figure(figsize=(14, 10))
    
    # Plot 1: Cell counts
    plt.subplot(2, 2, 1)
    plt.plot(time_points, cancer_counts, 'r-', linewidth=2, label='Cancer Cells')
    plt.plot(time_points, normal_counts, 'g-', linewidth=2, label='Normal Cells')
    plt.xlabel('Time (hours)', fontsize=11)
    plt.ylabel('Cell Count', fontsize=11)
    plt.title('Cell Population Over Time', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Immune activity
    plt.subplot(2, 2, 2)
    plt.plot(time_points, immune_kills, 'b-', linewidth=2)
    plt.xlabel('Time (hours)', fontsize=11)
    plt.ylabel('Total Kills', fontsize=11)
    plt.title('Immune System Activity', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Oxygen levels
    plt.subplot(2, 2, 3)
    plt.plot(time_points, oxygen_levels, 'c-', linewidth=2)
    plt.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='Hypoxic')
    plt.xlabel('Time (hours)', fontsize=11)
    plt.ylabel('Oxygen Level', fontsize=11)
    plt.title('Average Cellular Oxygen', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Cell spatial distribution
    plt.subplot(2, 2, 4)
    cancer_positions = []
    normal_positions = []
    
    for cell_id, cell in cellular.cells.items():
        if cell.cell_type == 'cancer':
            cancer_positions.append(cell.position)
        else:
            normal_positions.append(cell.position)
    
    if normal_positions:
        normal_pos = np.array(normal_positions)
        plt.scatter(normal_pos[:, 0], normal_pos[:, 1], 
                   c='green', s=80, alpha=0.6, label='Normal')
    
    if cancer_positions:
        cancer_pos = np.array(cancer_positions)
        plt.scatter(cancer_pos[:, 0], cancer_pos[:, 1], 
                   c='red', s=120, alpha=0.8, label='Cancer', marker='*')
    
    plt.xlabel('X Position (μm)', fontsize=11)
    plt.ylabel('Y Position (μm)', fontsize=11)
    plt.title('Cell Spatial Distribution', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig('simulation_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)
    print(f"\nDuration: {duration} hours")
    print(f"\nFinal State:")
    print(f"  Cancer cells: {cancer_counts[-1]}")
    print(f"  Normal cells: {normal_counts[-1]}")
    print(f"  Immune kills: {immune_kills[-1]}")
    print(f"  Avg oxygen: {oxygen_levels[-1]:.3f}")
    
    print(f"\nStatistics:")
    print(f"  Max cancer: {max(cancer_counts)}")
    print(f"  Min cancer: {min(cancer_counts)}")
    print(f"  Avg oxygen: {np.mean(oxygen_levels):.3f}")
    print(f"  Min oxygen: {min(oxygen_levels):.3f}")
    
    print("\n✅ Visualization saved as 'simulation_results.png'")
    
    return engine


def compare_scenarios():
    """Compare multiple scenarios"""
    
    print("\n🔬 Running Scenario Comparison...")
    print("=" * 60)
    
    scenarios = {
        'Baseline': {'n_t_cells': 10},
        'Boosted Immune': {'n_t_cells': 30},
        'Suppressed Immune': {'n_t_cells': 5}
    }
    
    results = {}
    
    for name, params in scenarios.items():
        print(f"\nRunning {name}...")
        
        eng = SimulationEngine(SimulationConfig(dt=0.01, duration=24.0))
        eng.register_module('cellular', CellularModule, {'n_cancer_cells': 10})
        eng.register_module('immune', ImmuneModule, params)
        eng.initialize()
        
        imm = eng.modules['immune']
        cell = eng.modules['cellular']
        imm.set_cellular_module(cell)
        
        times = []
        cancers = []
        
        for step in range(2400):
            eng.step()
            if step % 100 == 0:
                state = eng.get_state()
                times.append(state['time'])
                cancers.append(state['cellular']['n_cancer'])
        
        results[name] = (times, cancers)
        print(f"  Final cancer cells: {cancers[-1]}")
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    colors = ['blue', 'green', 'red']
    
    for (name, (times, cancers)), color in zip(results.items(), colors):
        plt.plot(times, cancers, linewidth=2.5, label=name, color=color)
    
    plt.xlabel('Time (hours)', fontsize=12)
    plt.ylabel('Cancer Cell Count', fontsize=12)
    plt.title('Immunotherapy Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.savefig('scenario_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✅ Comparison saved as 'scenario_comparison.png'")


if __name__ == '__main__':
    # Run main visualization
    engine = run_and_visualize(duration=24.0)
    
    # Export data
    print("\n📁 Exporting data...")
    engine.export_to_csv('simulation_data.csv')
    engine.export_to_json('simulation_data.json')
    print("✅ Data exported!")
    
    # Compare scenarios
    compare_scenarios()
    
    print("\n" + "=" * 60)
    print("🎉 All visualizations complete!")
    print("=" * 60)
    print("\nFiles created:")
    print("  - simulation_results.png")
    print("  - scenario_comparison.png")
    print("  - simulation_data.csv")
    print("  - simulation_data.json")
