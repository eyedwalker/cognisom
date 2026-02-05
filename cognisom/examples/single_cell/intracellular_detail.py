#!/usr/bin/env python3
"""
Detailed Intracellular Simulation
==================================

Simulate the internal workings of a single cell:
- DNA transcription (genes → mRNA)
- mRNA translation (mRNA → proteins)
- Ribosome dynamics
- Organelle function
- Metabolite levels (ATP, glucose)
- Beautiful visualizations

This shows what's happening INSIDE the cell at molecular level.
"""

import sys
from pathlib import Path
import numpy as np

# Add engine to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from engine.py.intracellular import IntracellularModel
from engine.py.visualize import CellVisualizer


def main():
    print("=" * 70)
    print("🧬 Detailed Intracellular Simulation")
    print("=" * 70)
    print()
    
    # Create intracellular model
    print("Creating cell with detailed internal structure...")
    cell = IntracellularModel()
    
    # Print initial state
    print("\n📊 Initial State:")
    print("-" * 70)
    state = cell.get_state_summary()
    print(f"  Genes in genome:     {state['genes']}")
    print(f"  Active genes:        {state['active_genes']}")
    print(f"  Ribosomes:           {state['ribosomes']}")
    print(f"  ATP:                 {state['atp']:,}")
    print(f"  Glucose:             {state['glucose']:,}")
    print()
    
    print("Genes in genome:")
    for gene_name, gene in cell.genes.items():
        status = "✓" if gene.is_active else "✗"
        print(f"  {status} {gene_name:12s} - {gene.sequence_length:,} bp, "
              f"rate: {gene.transcription_rate:.2f}/h")
    print()
    
    # Run simulation
    duration = 6.0  # 6 hours
    dt = 0.01
    steps = int(duration / dt)
    
    print(f"Running simulation for {duration} hours...")
    print("-" * 70)
    
    # Track history
    history = {
        'time': [],
        'total_mrna': [],
        'total_proteins': [],
        'atp': [],
        'glucose': [],
        'gene_expression': {},
    }
    
    # Initialize gene expression tracking
    for gene_name in cell.genes.keys():
        history['gene_expression'][gene_name] = []
    
    # Simulation loop
    for step in range(steps):
        t = step * dt
        
        # Step the model
        cell.step(dt)
        
        # Record every 0.5 hours
        if step % 50 == 0:
            state = cell.get_state_summary()
            expression = cell.get_gene_expression()
            
            history['time'].append(t)
            history['total_mrna'].append(state['total_mrna'])
            history['total_proteins'].append(state['total_proteins'])
            history['atp'].append(state['atp'])
            history['glucose'].append(state['glucose'])
            
            for gene_name in cell.genes.keys():
                history['gene_expression'][gene_name].append(
                    expression.get(gene_name, 0)
                )
            
            if step % 100 == 0:
                print(f"t={t:5.1f}h | mRNA: {state['total_mrna']:4d} | "
                      f"Proteins: {state['total_proteins']:6d} | "
                      f"ATP: {state['atp']:8,}")
    
    print("-" * 70)
    print("✓ Simulation complete")
    print()
    
    # Final state
    print("📊 Final State:")
    print("-" * 70)
    final_state = cell.get_state_summary()
    print(f"  mRNA species:        {final_state['mrna_species']}")
    print(f"  Total mRNA:          {final_state['total_mrna']}")
    print(f"  Protein species:     {final_state['protein_species']}")
    print(f"  Total proteins:      {final_state['total_proteins']:,}")
    print(f"  ATP remaining:       {final_state['atp']:,}")
    print()
    
    print("Gene Expression (mRNA counts):")
    expression = cell.get_gene_expression()
    for gene_name, count in sorted(expression.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {gene_name:12s}: {count:4d} mRNA")
    print()
    
    print("Protein Levels:")
    proteins = cell.get_protein_levels()
    for protein_name, count in sorted(proteins.items(), key=lambda x: -x[1]):
        print(f"  {protein_name:12s}: {count:6d} proteins")
    print()
    
    # Create visualizations
    print("=" * 70)
    print("🎨 Creating visualizations...")
    print("=" * 70)
    
    visualizer = CellVisualizer(cell)
    
    # Create comprehensive dashboard
    output_dir = Path('output/intracellular')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dashboard_path = output_dir / 'intracellular_dashboard.png'
    visualizer.create_dashboard(history, save_path=str(dashboard_path))
    
    print()
    print("=" * 70)
    print("✓ Intracellular simulation complete!")
    print("=" * 70)
    print()
    print("📁 Output saved to: output/intracellular/")
    print("   - intracellular_dashboard.png (comprehensive view)")
    print()
    print("🖼️  To view:")
    print("   open output/intracellular/intracellular_dashboard.png")
    print()
    print("=" * 70)
    print()
    print("What you're seeing:")
    print("  • Top-left: Cell structure with all organelles")
    print("  • Top-right: Molecular dynamics over time")
    print("  • Middle-right: Gene expression (mRNA levels)")
    print("  • Bottom: Protein abundance")
    print()
    print("This is what's happening INSIDE a single cell!")
    print("=" * 70)
    print()


if __name__ == '__main__':
    main()
