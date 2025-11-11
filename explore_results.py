#!/usr/bin/env python3
"""
Interactive results explorer with live plotting
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results(results_dir='output/basic_growth'):
    """Load simulation results"""
    results_path = Path(results_dir) / 'results.json'
    with open(results_path) as f:
        return json.load(f)

def plot_interactive(results):
    """Create interactive plot"""
    history = results['history']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('cognisom Cellular Simulation Results', fontsize=16, fontweight='bold')
    
    # Cell count
    ax = axes[0, 0]
    ax.plot(history['time'], history['cell_count'], 'o-', 
            linewidth=2, markersize=6, color='#2E86AB')
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Cell Count', fontsize=12)
    ax.set_title('Population Growth', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # Add doubling time annotation
    if results['final_cells'] >= 2:
        for i, count in enumerate(history['cell_count']):
            if count >= 2:
                doubling_time = history['time'][i]
                ax.axvline(doubling_time, color='red', linestyle='--', alpha=0.5)
                ax.text(doubling_time, max(history['cell_count']) * 0.9,
                       f'Doubling: {doubling_time:.1f}h',
                       rotation=90, verticalalignment='top')
                break
    
    # Total proteins
    ax = axes[0, 1]
    ax.plot(history['time'], history['total_proteins'], 'o-',
            linewidth=2, markersize=6, color='#06A77D')
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Total Proteins', fontsize=12)
    ax.set_title('Biomass Accumulation', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # MHC-I expression
    ax = axes[1, 0]
    ax.plot(history['time'], history['avg_mhc1'], 'o-',
            linewidth=2, markersize=6, color='#A23B72')
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('MHC-I Expression', fontsize=12)
    ax.set_title('Immune Recognition Marker', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.axhline(1.0, color='green', linestyle='--', alpha=0.3, label='Normal')
    ax.axhline(0.5, color='orange', linestyle='--', alpha=0.3, label='Reduced')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Stress level
    ax = axes[1, 1]
    ax.plot(history['time'], history['avg_stress'], 'o-',
            linewidth=2, markersize=6, color='#F18F01')
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Stress Level', fontsize=12)
    ax.set_title('Cellular Stress', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.axhline(0.5, color='orange', linestyle='--', alpha=0.3, label='Moderate')
    ax.axhline(0.9, color='red', linestyle='--', alpha=0.3, label='Critical')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def print_summary(results):
    """Print detailed summary"""
    print("\n" + "=" * 70)
    print("🧬 cognisom Cellular Simulation Results")
    print("=" * 70)
    
    print("\n📊 Simulation Parameters:")
    print(f"  Duration:     {results['duration']} hours")
    print(f"  Time step:    {results['dt']} hours")
    print(f"  Total steps:  {int(results['duration'] / results['dt'])}")
    
    print("\n📈 Population Dynamics:")
    print(f"  Initial cells:  1")
    print(f"  Final cells:    {results['final_cells']}")
    print(f"  Divisions:      {results['events']['divisions']}")
    print(f"  Deaths:         {results['events']['deaths']}")
    
    # Calculate doubling time
    history = results['history']
    for i, count in enumerate(history['cell_count']):
        if count >= 2:
            doubling_time = history['time'][i]
            print(f"  Doubling time:  {doubling_time:.2f} hours")
            
            # Compare to expected
            expected_min, expected_max = 18, 36
            if expected_min <= doubling_time <= expected_max:
                print(f"  ✓ Within expected range ({expected_min}-{expected_max}h)")
            break
    
    print("\n🔬 Molecular Content (Final):")
    final_proteins = history['total_proteins'][-1]
    print(f"  Total proteins: {final_proteins:,}")
    print(f"  Avg per cell:   {final_proteins // results['final_cells']:,}")
    
    print("\n🛡️  Immune Markers (Average):")
    avg_mhc1 = np.mean(history['avg_mhc1'])
    avg_stress = np.mean(history['avg_stress'])
    print(f"  MHC-I expression: {avg_mhc1:.3f}")
    print(f"  Stress level:     {avg_stress:.3f}")
    
    print("\n" + "=" * 70)
    print("✓ Analysis complete")
    print("=" * 70 + "\n")

def main():
    """Main function"""
    print("\n🔬 Loading simulation results...")
    
    try:
        results = load_results()
        print_summary(results)
        
        print("📊 Generating interactive plot...")
        fig = plot_interactive(results)
        
        # Save enhanced plot
        output_path = Path('output/basic_growth/enhanced_results.png')
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        
        # Show plot
        print("\n🖼️  Opening visualization...")
        plt.show()
        
    except FileNotFoundError:
        print("\n⚠️  No results found. Run a simulation first:")
        print("   python3 examples/single_cell/basic_growth.py")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
