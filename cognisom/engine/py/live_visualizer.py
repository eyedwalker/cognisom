#!/usr/bin/env python3
"""
Real-Time Interactive Cellular Visualization
============================================

Shows LIVE simulation with:
- Internal cellular processes (DNA, RNA, proteins, organelles)
- External cell-cell interactions
- Spatial environment (nutrients, signals)
- Real-time updates

Uses matplotlib animation for smooth, interactive display.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle, Wedge
import matplotlib.patches as mpatches
from typing import List, Dict, Any
import time


class LiveCellularVisualizer:
    """Real-time visualization of cellular simulation"""
    
    def __init__(self, figsize=(16, 10)):
        """Initialize the live visualizer"""
        self.fig = plt.figure(figsize=figsize)
        self.fig.suptitle('cognisom: Live Cellular Simulation', 
                         fontsize=16, fontweight='bold')
        
        # Create subplots
        # Top row: Spatial view + Single cell detail
        # Bottom row: Molecular counts + Environment
        gs = self.fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        self.ax_spatial = self.fig.add_subplot(gs[0, :2])  # Large spatial view
        self.ax_cell = self.fig.add_subplot(gs[0, 2])      # Single cell detail
        self.ax_molecules = self.fig.add_subplot(gs[1, 0]) # Molecular counts
        self.ax_environment = self.fig.add_subplot(gs[1, 1]) # Environment
        self.ax_signals = self.fig.add_subplot(gs[1, 2])   # Signaling
        
        # Setup axes
        self._setup_axes()
        
        # Animation state
        self.frame_count = 0
        self.start_time = time.time()
        
    def _setup_axes(self):
        """Setup all subplot axes"""
        # Spatial view
        self.ax_spatial.set_title('Spatial Environment (Multiple Cells)', 
                                 fontweight='bold')
        self.ax_spatial.set_xlabel('X Position (μm)')
        self.ax_spatial.set_ylabel('Y Position (μm)')
        self.ax_spatial.set_aspect('equal')
        self.ax_spatial.grid(True, alpha=0.3)
        
        # Single cell detail
        self.ax_cell.set_title('Single Cell Internal View', fontweight='bold')
        self.ax_cell.set_xlim(-15, 15)
        self.ax_cell.set_ylim(-15, 15)
        self.ax_cell.set_aspect('equal')
        self.ax_cell.axis('off')
        
        # Molecular counts
        self.ax_molecules.set_title('Molecular Counts (Live)', fontweight='bold')
        self.ax_molecules.set_xlabel('Time (hours)')
        self.ax_molecules.set_ylabel('Count')
        
        # Environment
        self.ax_environment.set_title('Environment Gradients', fontweight='bold')
        self.ax_environment.set_xlabel('X Position (μm)')
        self.ax_environment.set_ylabel('Y Position (μm)')
        self.ax_environment.set_aspect('equal')
        
        # Signals
        self.ax_signals.set_title('Cell Signaling Activity', fontweight='bold')
        self.ax_signals.set_xlabel('Time (hours)')
        self.ax_signals.set_ylabel('Signal Strength')
        
    def draw_cell_internal(self, cell_data: Dict[str, Any], ax):
        """Draw internal structure of a single cell"""
        ax.clear()
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Single Cell Internal View', fontweight='bold')
        
        # Cell membrane
        membrane = Circle((0, 0), 12, fill=False, edgecolor='black', 
                         linewidth=2, linestyle='-')
        ax.add_patch(membrane)
        
        # Nucleus
        nucleus = Circle((0, 0), 5, facecolor='lightblue', 
                        edgecolor='darkblue', linewidth=1.5, alpha=0.6)
        ax.add_patch(nucleus)
        ax.text(0, 0, 'Nucleus', ha='center', va='center', 
               fontsize=8, fontweight='bold')
        
        # DNA (chromatin)
        n_chromatin = 8
        for i in range(n_chromatin):
            angle = 2 * np.pi * i / n_chromatin
            x = 2.5 * np.cos(angle)
            y = 2.5 * np.sin(angle)
            dna = Circle((x, y), 0.8, facecolor='purple', 
                        edgecolor='darkviolet', alpha=0.7)
            ax.add_patch(dna)
        
        # Mitochondria (moving)
        n_mito = cell_data.get('n_mitochondria', 15)
        for i in range(n_mito):
            angle = 2 * np.pi * i / n_mito + self.frame_count * 0.05
            radius = 7 + 2 * np.sin(angle * 3)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            mito = Circle((x, y), 1.2, facecolor='red', 
                         edgecolor='darkred', alpha=0.6)
            ax.add_patch(mito)
        
        # Ribosomes (scattered, some moving)
        n_ribosomes = cell_data.get('n_ribosomes', 50)
        np.random.seed(42 + self.frame_count // 10)  # Slow movement
        for i in range(n_ribosomes):
            r = np.random.uniform(5.5, 11)
            theta = np.random.uniform(0, 2*np.pi)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            ribosome = Circle((x, y), 0.3, facecolor='green', alpha=0.8)
            ax.add_patch(ribosome)
        
        # mRNA (moving from nucleus to cytoplasm)
        n_mrna = cell_data.get('mrna_count', 20)
        for i in range(min(n_mrna, 10)):  # Show up to 10
            progress = (self.frame_count * 0.1 + i * 0.5) % 1.0
            start_r = 5
            end_r = 10
            r = start_r + (end_r - start_r) * progress
            theta = 2 * np.pi * i / 10
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            mrna = Circle((x, y), 0.5, facecolor='orange', 
                         edgecolor='darkorange', alpha=0.7)
            ax.add_patch(mrna)
        
        # Proteins (throughout cytoplasm)
        n_proteins = cell_data.get('protein_count', 100)
        np.random.seed(123 + self.frame_count // 5)
        for i in range(min(n_proteins, 30)):  # Show subset
            r = np.random.uniform(5.5, 11.5)
            theta = np.random.uniform(0, 2*np.pi)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            protein = Circle((x, y), 0.4, facecolor='cyan', alpha=0.6)
            ax.add_patch(protein)
        
        # Membrane receptors
        n_receptors = 12
        for i in range(n_receptors):
            angle = 2 * np.pi * i / n_receptors
            x = 12 * np.cos(angle)
            y = 12 * np.sin(angle)
            receptor = Rectangle((x-0.3, y-0.8), 0.6, 1.6, 
                                facecolor='yellow', edgecolor='orange',
                                angle=np.degrees(angle), alpha=0.8)
            ax.add_patch(receptor)
        
        # Legend
        legend_elements = [
            mpatches.Patch(color='purple', label='DNA'),
            mpatches.Patch(color='orange', label='mRNA'),
            mpatches.Patch(color='green', label='Ribosomes'),
            mpatches.Patch(color='cyan', label='Proteins'),
            mpatches.Patch(color='red', label='Mitochondria'),
            mpatches.Patch(color='yellow', label='Receptors')
        ]
        ax.legend(handles=legend_elements, loc='upper left', 
                 fontsize=7, framealpha=0.9)
        
    def draw_spatial_view(self, cells: List[Dict[str, Any]], 
                         environment: Dict[str, np.ndarray], ax):
        """Draw multiple cells in spatial environment"""
        ax.clear()
        ax.set_title('Spatial Environment (Multiple Cells)', fontweight='bold')
        ax.set_xlabel('X Position (μm)')
        ax.set_ylabel('Y Position (μm)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Draw environment gradient (e.g., oxygen)
        if 'oxygen' in environment:
            oxygen = environment['oxygen']
            extent = environment.get('extent', [0, 100, 0, 100])
            im = ax.imshow(oxygen.T, origin='lower', extent=extent,
                          cmap='Blues', alpha=0.3, vmin=0, vmax=1)
        
        # Draw cells
        for i, cell in enumerate(cells):
            x, y = cell['position']
            radius = cell.get('radius', 10)
            
            # Cell color based on state
            state = cell.get('state', 'alive')
            if state == 'dividing':
                color = 'lightgreen'
                edgecolor = 'green'
            elif state == 'stressed':
                color = 'yellow'
                edgecolor = 'orange'
            elif state == 'dead':
                color = 'gray'
                edgecolor = 'black'
            else:
                color = 'lightblue'
                edgecolor = 'blue'
            
            # Draw cell
            cell_circle = Circle((x, y), radius, facecolor=color,
                               edgecolor=edgecolor, linewidth=2, alpha=0.7)
            ax.add_patch(cell_circle)
            
            # Cell ID
            ax.text(x, y, f'{i}', ha='center', va='center',
                   fontsize=8, fontweight='bold')
            
            # Show signaling (if active)
            if cell.get('signaling', False):
                signal_circle = Circle((x, y), radius * 2, 
                                      fill=False, edgecolor='red',
                                      linewidth=1, linestyle='--', alpha=0.5)
                ax.add_patch(signal_circle)
        
        # Draw cell-cell interactions
        for i, cell1 in enumerate(cells):
            for j, cell2 in enumerate(cells):
                if i < j:  # Avoid duplicates
                    x1, y1 = cell1['position']
                    x2, y2 = cell2['position']
                    dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    
                    # Draw connection if close
                    if dist < 30:  # Interaction distance
                        alpha = 1.0 - dist / 30
                        ax.plot([x1, x2], [y1, y2], 'r--', 
                               alpha=alpha*0.5, linewidth=1)
        
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        
    def update_molecular_plot(self, time_data: np.ndarray, 
                             molecular_data: Dict[str, np.ndarray], ax):
        """Update molecular counts over time"""
        ax.clear()
        ax.set_title('Molecular Counts (Live)', fontweight='bold')
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Count')
        
        # Plot each molecular species
        if 'mrna' in molecular_data:
            ax.plot(time_data, molecular_data['mrna'], 
                   'o-', label='mRNA', color='orange', linewidth=2)
        if 'protein' in molecular_data:
            ax.plot(time_data, molecular_data['protein'], 
                   's-', label='Proteins', color='cyan', linewidth=2)
        if 'atp' in molecular_data:
            ax.plot(time_data, molecular_data['atp'], 
                   '^-', label='ATP', color='red', linewidth=2)
        
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
    def update_environment_plot(self, environment: Dict[str, np.ndarray], ax):
        """Update environment gradient visualization"""
        ax.clear()
        ax.set_title('Environment Gradients', fontweight='bold')
        ax.set_xlabel('X Position (μm)')
        ax.set_ylabel('Y Position (μm)')
        ax.set_aspect('equal')
        
        if 'oxygen' in environment:
            oxygen = environment['oxygen']
            extent = environment.get('extent', [0, 100, 0, 100])
            im = ax.imshow(oxygen.T, origin='lower', extent=extent,
                          cmap='RdYlBu_r', vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, label='Oxygen (normalized)')
            
            # Add contour lines
            X = np.linspace(extent[0], extent[1], oxygen.shape[0])
            Y = np.linspace(extent[2], extent[3], oxygen.shape[1])
            ax.contour(X, Y, oxygen.T, levels=5, colors='black', 
                      alpha=0.3, linewidths=0.5)
    
    def update_signals_plot(self, time_data: np.ndarray,
                           signal_data: Dict[str, np.ndarray], ax):
        """Update cell signaling activity"""
        ax.clear()
        ax.set_title('Cell Signaling Activity', fontweight='bold')
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Signal Strength')
        
        if 'growth_factor' in signal_data:
            ax.plot(time_data, signal_data['growth_factor'],
                   'g-', label='Growth Factor', linewidth=2)
        if 'stress_signal' in signal_data:
            ax.plot(time_data, signal_data['stress_signal'],
                   'r-', label='Stress Signal', linewidth=2)
        if 'contact_inhibition' in signal_data:
            ax.plot(time_data, signal_data['contact_inhibition'],
                   'b-', label='Contact Inhibition', linewidth=2)
        
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.2)


def create_live_simulation(duration_hours=2.0, dt=0.01, interval_ms=50):
    """
    Create and run live simulation visualization
    
    Parameters:
    -----------
    duration_hours : float
        Total simulation time in hours
    dt : float
        Time step in hours
    interval_ms : int
        Animation update interval in milliseconds
    """
    from engine.py.simulation import Simulation
    from engine.py.spatial.grid import SpatialGrid, GridConfig
    from engine.py.cell import Cell
    
    print("🎬 Starting live cellular simulation...")
    print(f"Duration: {duration_hours} hours")
    print(f"Time step: {dt} hours ({dt*60} minutes)")
    print(f"Update interval: {interval_ms} ms")
    print()
    
    # Create initial cells
    initial_cells = [Cell() for i in range(5)]
    
    # Create simulation
    sim = Simulation(initial_cells=initial_cells, duration=duration_hours, dt=dt)
    
    # Create spatial grid
    grid_config = GridConfig(
        size=(10, 10, 10),  # 10x10x10 voxels
        resolution=10.0      # 10 μm per voxel = 100x100x100 μm total
    )
    grid = SpatialGrid(config=grid_config)
    
    # Place cells in grid
    positions = [
        (25, 25, 50),
        (75, 25, 50),
        (50, 50, 50),
        (25, 75, 50),
        (75, 75, 50)
    ]
    for i, pos in enumerate(positions):
        grid.add_cell(i, pos)
    
    # Oxygen is already initialized in GridConfig
    
    # Create visualizer
    viz = LiveCellularVisualizer(figsize=(16, 10))
    
    # Data storage
    time_data = []
    molecular_data = {
        'mrna': [],
        'protein': [],
        'atp': []
    }
    signal_data = {
        'growth_factor': [],
        'stress_signal': [],
        'contact_inhibition': []
    }
    
    def update_frame(frame):
        """Update function for animation"""
        # Run simulation step
        sim.step()
        
        # Cells consume oxygen
        for i, cell in enumerate(sim.cells):
            if i < len(positions):
                pos = positions[i]
                grid.consume_at_position(pos, amount_o2=0.01, amount_glucose=0.01)
        
        # Diffuse oxygen
        grid.step_diffusion(dt)
        
        # Collect data
        current_time = sim.time
        time_data.append(current_time)
        
        # Average molecular counts across cells
        avg_mrna = np.mean([c.state.species_counts[0] for c in sim.cells])
        avg_protein = np.mean([c.state.species_counts[1] for c in sim.cells])
        avg_atp = np.mean([c.state.species_counts[2] for c in sim.cells])
        
        molecular_data['mrna'].append(avg_mrna)
        molecular_data['protein'].append(avg_protein)
        molecular_data['atp'].append(avg_atp)
        
        # Simulate signals (placeholder - will be real later)
        signal_data['growth_factor'].append(0.8 + 0.2 * np.sin(current_time * 2))
        signal_data['stress_signal'].append(0.3 + 0.1 * np.random.random())
        signal_data['contact_inhibition'].append(0.5 + 0.3 * np.cos(current_time))
        
        # Prepare cell data for visualization
        cells_viz = []
        for i, cell in enumerate(sim.cells):
            cell_data = {
                'position': positions[i][:2],  # 2D projection
                'radius': 10,
                'state': 'alive',
                'signaling': i == 0  # First cell signals
            }
            cells_viz.append(cell_data)
        
        # Prepare environment data
        oxygen_2d = grid.oxygen[:, :, 5]  # Middle slice
        environment = {
            'oxygen': oxygen_2d,
            'extent': [0, 100, 0, 100]
        }
        
        # Prepare single cell data
        cell_detail = {
            'n_mitochondria': 15,
            'n_ribosomes': 50,  # Fixed for visualization
            'mrna_count': int(sim.cells[0].state.species_counts[0]),
            'protein_count': int(sim.cells[0].state.species_counts[1])
        }
        
        # Update all plots
        viz.draw_spatial_view(cells_viz, environment, viz.ax_spatial)
        viz.draw_cell_internal(cell_detail, viz.ax_cell)
        
        if len(time_data) > 1:
            time_array = np.array(time_data)
            mol_data = {k: np.array(v) for k, v in molecular_data.items()}
            sig_data = {k: np.array(v) for k, v in signal_data.items()}
            
            viz.update_molecular_plot(time_array, mol_data, viz.ax_molecules)
            viz.update_environment_plot(environment, viz.ax_environment)
            viz.update_signals_plot(time_array, sig_data, viz.ax_signals)
        
        # Update frame counter
        viz.frame_count = frame
        
        # Update title with stats
        elapsed = time.time() - viz.start_time
        fps = frame / elapsed if elapsed > 0 else 0
        viz.fig.suptitle(
            f'cognisom: Live Cellular Simulation | '
            f'Time: {current_time:.2f}h | Frame: {frame} | FPS: {fps:.1f}',
            fontsize=16, fontweight='bold'
        )
        
        # Stop condition
        if current_time >= duration_hours:
            print(f"\n✓ Simulation complete: {current_time:.2f} hours")
            return
    
    # Create animation
    n_frames = int(duration_hours / dt)
    anim = FuncAnimation(
        viz.fig, 
        update_frame,
        frames=n_frames,
        interval=interval_ms,
        repeat=False,
        blit=False
    )
    
    plt.tight_layout()
    plt.show()
    
    return anim, viz


if __name__ == '__main__':
    # Run live simulation
    anim, viz = create_live_simulation(
        duration_hours=2.0,
        dt=0.01,
        interval_ms=50  # 20 FPS
    )
