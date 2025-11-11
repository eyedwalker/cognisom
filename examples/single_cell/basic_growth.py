#!/usr/bin/env python3
"""
Example 1: Basic Cell Growth
=============================

Simulate a single cell growing and dividing over 24 hours.

This demonstrates:
- Cell creation
- Time stepping
- Division events
- Population growth
- Data visualization
"""

import sys
from pathlib import Path

# Add engine to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from engine.py.cell import Cell
from engine.py.simulation import Simulation


def main():
    print("=" * 60)
    print("cognisom Example 1: Basic Cell Growth")
    print("=" * 60)
    print()
    
    # Create initial cell
    initial_cell = Cell()
    print(f"Initial cell: {initial_cell}")
    print(f"  mRNA: {initial_cell.state.species_counts[0]}")
    print(f"  Proteins: {initial_cell.state.species_counts[1]}")
    print(f"  ATP: {initial_cell.state.species_counts[2]}")
    print()
    
    # Create simulation
    sim = Simulation(
        initial_cells=[initial_cell],
        duration=24.0,      # 24 hours
        dt=0.01,            # 0.01 hour time steps (36 seconds)
        output_dir='./output/basic_growth'
    )
    
    # Run simulation
    sim.run(verbose=True, save_interval=100)
    print()
    
    # Get results
    results = sim.get_results()
    
    # Calculate doubling time
    doubling_time = sim.get_doubling_time()
    if doubling_time:
        print(f"📊 Population doubling time: {doubling_time:.2f} hours")
    
    # Expected doubling time for mammalian cells: 18-36 hours
    # Our simplified model should be in this range
    
    # Save results
    sim.save_results()
    
    # Plot results
    try:
        sim.plot_results(save=True)
    except ImportError:
        print("\n⚠️  Install matplotlib to see plots: pip install matplotlib")
    
    print()
    print("=" * 60)
    print("✓ Example complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Check output/basic_growth/ for results")
    print("  2. Try: python examples/single_cell/stress_response.py")
    print("  3. Read: QUICKSTART.md for more examples")
    print()


if __name__ == '__main__':
    main()
