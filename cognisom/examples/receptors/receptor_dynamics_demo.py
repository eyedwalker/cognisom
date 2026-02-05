#!/usr/bin/env python3
"""
Receptor Dynamics Demonstration
================================

Shows how membrane receptors respond to ligands:
- Binding dynamics
- Internalization
- Desensitization
- Signal transduction

Visualizes the complete receptor lifecycle.
"""

import sys
sys.path.insert(0, '../..')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from engine.py.membrane.receptors import (
    ReceptorSystem,
    EGFReceptor,
    InsulinReceptor,
    CytokineReceptor
)


def create_receptor_visualization():
    """Create animated visualization of receptor dynamics"""
    
    print("=" * 60)
    print("Receptor Dynamics Visualization")
    print("=" * 60)
    print()
    print("Simulating:")
    print("  • EGFR (EGF receptor)")
    print("  • Insulin Receptor")
    print("  • IL-6 Receptor")
    print()
    print("Watch:")
    print("  • Ligand binding (equilibrium)")
    print("  • Receptor internalization")
    print("  • Signal strength over time")
    print("  • Desensitization")
    print("=" * 60)
    print()
    
    # Create receptor system
    system = ReceptorSystem()
    system.add_receptor('EGFR', EGFReceptor(50000))
    system.add_receptor('InsulinR', InsulinReceptor(100000))
    system.add_receptor('IL6R', CytokineReceptor('IL-6', 10000))
    
    # Ligand concentrations (will vary over time)
    base_ligands = {
        'EGF': 2e-9,
        'Insulin': 10e-9,
        'IL-6': 0.5e-9
    }
    
    # Simulation parameters
    dt = 0.01  # hours
    duration = 4.0  # hours
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Receptor Dynamics: Membrane Exchange in Action', 
                 fontsize=16, fontweight='bold')
    
    # Create subplots
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    ax_surface = fig.add_subplot(gs[0, :])  # Receptor numbers
    ax_bound = fig.add_subplot(gs[1, :])    # Bound fraction
    ax_signal = fig.add_subplot(gs[2, 0])   # Signal strength
    ax_deactivation = fig.add_subplot(gs[2, 1])  # Desensitization
    ax_pathways = fig.add_subplot(gs[2, 2])  # Pathway activity
    
    # Setup axes
    ax_surface.set_title('Receptor Numbers on Cell Surface', fontweight='bold')
    ax_surface.set_xlabel('Time (hours)')
    ax_surface.set_ylabel('Number of Receptors')
    ax_surface.grid(True, alpha=0.3)
    
    ax_bound.set_title('Bound Receptor Fraction', fontweight='bold')
    ax_bound.set_xlabel('Time (hours)')
    ax_bound.set_ylabel('Fraction Bound')
    ax_bound.set_ylim(0, 1)
    ax_bound.grid(True, alpha=0.3)
    
    ax_signal.set_title('Signal Strength', fontweight='bold')
    ax_signal.set_xlabel('Time (hours)')
    ax_signal.set_ylabel('Signal (AU)')
    ax_signal.grid(True, alpha=0.3)
    
    ax_deactivation.set_title('Desensitization', fontweight='bold')
    ax_deactivation.set_xlabel('Time (hours)')
    ax_deactivation.set_ylabel('Desensitization')
    ax_deactivation.set_ylim(0, 1)
    ax_deactivation.grid(True, alpha=0.3)
    
    ax_pathways.set_title('Signaling Pathways', fontweight='bold')
    ax_pathways.set_xlabel('Pathway')
    ax_pathways.set_ylabel('Activity (AU)')
    
    # Data storage
    time_data = []
    receptor_data = {
        'EGFR': {'surface': [], 'bound_frac': [], 'signal': [], 'desens': []},
        'InsulinR': {'surface': [], 'bound_frac': [], 'signal': [], 'desens': []},
        'IL6R': {'surface': [], 'bound_frac': [], 'signal': [], 'desens': []}
    }
    pathway_history = {'MAPK': [], 'PI3K-Akt': [], 'JAK-STAT': []}
    
    # Colors
    colors = {'EGFR': 'red', 'InsulinR': 'blue', 'IL6R': 'green'}
    
    # Lines (will be created in first frame)
    lines = {}
    
    def init():
        """Initialize animation"""
        return []
    
    def update(frame):
        """Update function for animation"""
        time = frame * dt
        
        # Vary ligand concentrations (simulate pulsatile release)
        ligands = base_ligands.copy()
        
        # EGF pulse at t=1h
        if 1.0 < time < 1.5:
            ligands['EGF'] = 5e-9
        
        # Insulin pulse at t=2h
        if 2.0 < time < 2.5:
            ligands['Insulin'] = 20e-9
        
        # Update system
        system.update(dt, ligands)
        
        # Collect data
        time_data.append(time)
        stats = system.get_receptor_stats()
        
        for name in ['EGFR', 'InsulinR', 'IL6R']:
            receptor_data[name]['surface'].append(stats[name]['surface'])
            receptor_data[name]['bound_frac'].append(stats[name]['bound_fraction'])
            receptor_data[name]['signal'].append(stats[name]['signal'])
            receptor_data[name]['desens'].append(stats[name]['desensitization'])
        
        # Pathway activity
        pathways = system.get_active_pathways()
        pathway_history['MAPK'].append(pathways.get('MAPK', 0))
        pathway_history['PI3K-Akt'].append(pathways.get('PI3K-Akt', 0))
        pathway_history['JAK-STAT'].append(pathways.get('JAK-STAT', 0))
        
        # Update plots
        if len(time_data) > 1:
            time_array = np.array(time_data)
            
            # Surface receptors
            ax_surface.clear()
            ax_surface.set_title('Receptor Numbers on Cell Surface', fontweight='bold')
            ax_surface.set_xlabel('Time (hours)')
            ax_surface.set_ylabel('Number of Receptors')
            ax_surface.grid(True, alpha=0.3)
            for name, color in colors.items():
                ax_surface.plot(time_array, receptor_data[name]['surface'],
                              label=name, color=color, linewidth=2)
            ax_surface.legend()
            
            # Bound fraction
            ax_bound.clear()
            ax_bound.set_title('Bound Receptor Fraction', fontweight='bold')
            ax_bound.set_xlabel('Time (hours)')
            ax_bound.set_ylabel('Fraction Bound')
            ax_bound.set_ylim(0, 1)
            ax_bound.grid(True, alpha=0.3)
            for name, color in colors.items():
                ax_bound.plot(time_array, receptor_data[name]['bound_frac'],
                            label=name, color=color, linewidth=2)
            ax_bound.legend()
            
            # Signal strength
            ax_signal.clear()
            ax_signal.set_title('Signal Strength', fontweight='bold')
            ax_signal.set_xlabel('Time (hours)')
            ax_signal.set_ylabel('Signal (AU)')
            ax_signal.grid(True, alpha=0.3)
            for name, color in colors.items():
                ax_signal.plot(time_array, receptor_data[name]['signal'],
                             label=name, color=color, linewidth=2)
            ax_signal.legend()
            
            # Desensitization
            ax_deactivation.clear()
            ax_deactivation.set_title('Desensitization', fontweight='bold')
            ax_deactivation.set_xlabel('Time (hours)')
            ax_deactivation.set_ylabel('Desensitization')
            ax_deactivation.set_ylim(0, 1)
            ax_deactivation.grid(True, alpha=0.3)
            for name, color in colors.items():
                ax_deactivation.plot(time_array, receptor_data[name]['desens'],
                                   label=name, color=color, linewidth=2)
            ax_deactivation.legend()
            
            # Pathway activity (bar chart)
            ax_pathways.clear()
            ax_pathways.set_title('Signaling Pathways (Current)', fontweight='bold')
            ax_pathways.set_xlabel('Pathway')
            ax_pathways.set_ylabel('Activity (AU)')
            
            pathway_names = list(pathways.keys())
            pathway_values = list(pathways.values())
            if pathway_names:
                bars = ax_pathways.bar(pathway_names, pathway_values,
                                      color=['red', 'blue', 'green'])
                ax_pathways.set_ylim(0, max(pathway_values) * 1.2 if pathway_values else 1000)
        
        # Update title with time
        fig.suptitle(
            f'Receptor Dynamics: Membrane Exchange in Action | Time: {time:.2f}h',
            fontsize=16, fontweight='bold'
        )
        
        return []
    
    # Create animation
    n_frames = int(duration / dt)
    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=n_frames,
        interval=50,  # 50ms = 20 FPS
        repeat=False,
        blit=False
    )
    
    plt.tight_layout()
    plt.show()
    
    return anim


if __name__ == '__main__':
    anim = create_receptor_visualization()
