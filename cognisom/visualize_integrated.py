#!/usr/bin/env python3
"""
Integrated Visualization
========================

Real-time 3D visualization of all 6 modules working together.

Features:
- 3D tissue view with all cell types
- Capillary and lymphatic networks
- Immune cell movement
- Oxygen gradients (heatmap)
- Real-time statistics
- Epigenetic states (methylation, histones)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

from core import SimulationEngine, SimulationConfig
from modules import (MolecularModule, CellularModule, ImmuneModule,
                    VascularModule, LymphaticModule, SpatialModule)


class IntegratedVisualizer:
    """
    Real-time visualization of complete simulation
    """
    
    def __init__(self, engine):
        self.engine = engine
        
        # Create figure with subplots
        self.fig = plt.figure(figsize=(18, 10))
        
        # 3D tissue view (main)
        self.ax_3d = self.fig.add_subplot(2, 3, (1, 4), projection='3d')
        
        # Statistics panel
        self.ax_stats = self.fig.add_subplot(2, 3, 2)
        self.ax_stats.axis('off')
        
        # Oxygen gradient
        self.ax_oxygen = self.fig.add_subplot(2, 3, 3)
        
        # Time series plots
        self.ax_cells = self.fig.add_subplot(2, 3, 5)
        self.ax_immune = self.fig.add_subplot(2, 3, 6)
        
        # Data collection
        self.time_data = []
        self.cancer_data = []
        self.normal_data = []
        self.immune_active_data = []
        self.kills_data = []
        
        # Setup
        self._setup_plots()
    
    def _setup_plots(self):
        """Setup plot properties"""
        # 3D view
        self.ax_3d.set_xlim(0, 200)
        self.ax_3d.set_ylim(0, 200)
        self.ax_3d.set_zlim(0, 100)
        self.ax_3d.set_xlabel('X (μm)')
        self.ax_3d.set_ylabel('Y (μm)')
        self.ax_3d.set_zlabel('Z (μm)')
        self.ax_3d.set_title('Multi-System Tissue View', fontweight='bold', fontsize=14)
        
        # Oxygen
        self.ax_oxygen.set_title('Oxygen Distribution', fontweight='bold')
        self.ax_oxygen.set_xlabel('X (μm)')
        self.ax_oxygen.set_ylabel('Y (μm)')
        
        # Cell counts
        self.ax_cells.set_title('Cell Population', fontweight='bold')
        self.ax_cells.set_xlabel('Time (hours)')
        self.ax_cells.set_ylabel('Count')
        self.ax_cells.grid(True, alpha=0.3)
        
        # Immune activity
        self.ax_immune.set_title('Immune Activity', fontweight='bold')
        self.ax_immune.set_xlabel('Time (hours)')
        self.ax_immune.set_ylabel('Count')
        self.ax_immune.grid(True, alpha=0.3)
    
    def update(self, frame):
        """Update visualization"""
        # Run simulation step
        self.engine.step()
        
        # Get current state
        state = self.engine.get_state()
        
        # Collect data
        self.time_data.append(state['time'])
        self.cancer_data.append(state['cellular']['n_cancer'])
        self.normal_data.append(state['cellular']['n_normal'])
        self.immune_active_data.append(state['immune']['n_activated'])
        self.kills_data.append(state['immune']['total_kills'])
        
        # Update 3D view
        self._update_3d_view(state)
        
        # Update statistics
        self._update_statistics(state)
        
        # Update oxygen heatmap
        self._update_oxygen_map(state)
        
        # Update time series
        self._update_time_series()
        
        return []
    
    def _update_3d_view(self, state):
        """Update 3D tissue view"""
        self.ax_3d.cla()
        self.ax_3d.set_xlim(0, 200)
        self.ax_3d.set_ylim(0, 200)
        self.ax_3d.set_zlim(0, 100)
        self.ax_3d.set_xlabel('X (μm)')
        self.ax_3d.set_ylabel('Y (μm)')
        self.ax_3d.set_zlabel('Z (μm)')
        self.ax_3d.set_title(f'Tissue View (t={state["time"]:.1f}h)', 
                            fontweight='bold', fontsize=14)
        
        # Get modules
        cellular = self.engine.modules['cellular']
        immune = self.engine.modules['immune']
        vascular = self.engine.modules['vascular']
        lymphatic = self.engine.modules['lymphatic']
        
        # Plot capillaries (red)
        for cap in vascular.capillaries.values():
            self.ax_3d.plot([cap.start[0], cap.end[0]],
                           [cap.start[1], cap.end[1]],
                           [cap.start[2], cap.end[2]],
                           'r-', linewidth=2, alpha=0.6)
        
        # Plot lymphatics (blue)
        for lymph in lymphatic.vessels.values():
            self.ax_3d.plot([lymph.start[0], lymph.end[0]],
                           [lymph.start[1], lymph.end[1]],
                           [lymph.start[2], lymph.end[2]],
                           'b-', linewidth=3, alpha=0.5)
        
        # Plot cells
        normal_cells = [c for c in cellular.cells.values() 
                       if c.alive and c.cell_type == 'normal']
        cancer_cells = [c for c in cellular.cells.values() 
                       if c.alive and c.cell_type == 'cancer']
        
        if normal_cells:
            pos = np.array([c.position for c in normal_cells])
            self.ax_3d.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                             c='green', s=30, alpha=0.6, label='Normal')
        
        if cancer_cells:
            pos = np.array([c.position for c in cancer_cells])
            # Color by epigenetic state (methylation)
            colors = ['red' if hasattr(c, 'methylation_level') and c.methylation_level > 0.5 
                     else 'orange' for c in cancer_cells]
            self.ax_3d.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                             c=colors, s=50, alpha=0.8, marker='*', label='Cancer')
        
        # Plot immune cells
        t_cells = [i for i in immune.immune_cells.values() 
                  if not i.in_blood and i.cell_type == 'T_cell']
        nk_cells = [i for i in immune.immune_cells.values() 
                   if not i.in_blood and i.cell_type == 'NK_cell']
        macrophages = [i for i in immune.immune_cells.values() 
                      if not i.in_blood and i.cell_type == 'macrophage']
        
        if t_cells:
            pos = np.array([i.position for i in t_cells])
            self.ax_3d.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                             c='cyan', s=40, alpha=0.7, marker='^', label='T cells')
        
        if nk_cells:
            pos = np.array([i.position for i in nk_cells])
            self.ax_3d.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                             c='magenta', s=40, alpha=0.7, marker='d', label='NK cells')
        
        if macrophages:
            pos = np.array([i.position for i in macrophages])
            self.ax_3d.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                             c='orange', s=40, alpha=0.7, marker='s', label='Macrophages')
        
        self.ax_3d.legend(loc='upper right', fontsize=8)
    
    def _update_statistics(self, state):
        """Update statistics panel"""
        self.ax_stats.cla()
        self.ax_stats.axis('off')
        
        stats_text = f"""
SIMULATION STATUS
═════════════════
Time: {state['time']:.1f} hours
Steps: {state['step_count']}

CELLS
─────────────────
Total: {state['cellular']['n_cells']}
Cancer: {state['cellular']['n_cancer']}
Normal: {state['cellular']['n_normal']}
Divisions: {state['cellular']['total_divisions']}
Deaths: {state['cellular']['total_deaths']}

IMMUNE
─────────────────
Active: {state['immune']['n_immune_cells']}
Activated: {state['immune']['n_activated']}
Kills: {state['immune']['total_kills']}

VASCULAR
─────────────────
Capillaries: {state['vascular']['n_capillaries']}
Avg O2: {state['vascular']['avg_cell_O2']:.3f}
Hypoxic: {state['vascular']['hypoxic_regions']}

LYMPHATIC
─────────────────
Vessels: {state['lymphatic']['n_vessels']}
Metastases: {state['lymphatic']['total_metastases']}

MOLECULAR
─────────────────
Transcriptions: {state['molecular']['total_transcriptions']}
Exosomes: {state['molecular']['n_exosomes']}
        """
        
        self.ax_stats.text(0.05, 0.5, stats_text, fontsize=9, family='monospace',
                          verticalalignment='center')
    
    def _update_oxygen_map(self, state):
        """Update oxygen heatmap"""
        self.ax_oxygen.cla()
        self.ax_oxygen.set_title('Oxygen Distribution', fontweight='bold')
        self.ax_oxygen.set_xlabel('X (μm)')
        self.ax_oxygen.set_ylabel('Y (μm)')
        
        # Create oxygen field
        cellular = self.engine.modules['cellular']
        
        x = np.linspace(0, 200, 50)
        y = np.linspace(0, 200, 50)
        X, Y = np.meshgrid(x, y)
        O2 = np.ones_like(X) * 0.21
        
        # Reduce O2 near cells
        for cell in cellular.cells.values():
            if cell.alive:
                dist = np.sqrt((X - cell.position[0])**2 + (Y - cell.position[1])**2)
                if cell.cell_type == 'cancer':
                    O2 -= 0.15 * np.exp(-dist / 30)
                else:
                    O2 -= 0.10 * np.exp(-dist / 30)
        
        O2 = np.clip(O2, 0, 0.21)
        
        im = self.ax_oxygen.contourf(X, Y, O2, levels=20, cmap='RdYlBu')
        self.ax_oxygen.contour(X, Y, O2, levels=[0.05], colors='red', linewidths=2)
        
        # Mark hypoxic regions
        if state['vascular']['hypoxic_regions'] > 0:
            self.ax_oxygen.text(10, 190, f"⚠️ {state['vascular']['hypoxic_regions']} hypoxic cells",
                              fontsize=10, color='red', fontweight='bold')
    
    def _update_time_series(self):
        """Update time series plots"""
        if len(self.time_data) < 2:
            return
        
        # Cell population
        self.ax_cells.cla()
        self.ax_cells.set_title('Cell Population', fontweight='bold')
        self.ax_cells.set_xlabel('Time (hours)')
        self.ax_cells.set_ylabel('Count')
        self.ax_cells.plot(self.time_data, self.cancer_data, 'r-', linewidth=2, label='Cancer')
        self.ax_cells.plot(self.time_data, self.normal_data, 'g-', linewidth=2, label='Normal')
        self.ax_cells.legend()
        self.ax_cells.grid(True, alpha=0.3)
        
        # Immune activity
        self.ax_immune.cla()
        self.ax_immune.set_title('Immune Activity', fontweight='bold')
        self.ax_immune.set_xlabel('Time (hours)')
        self.ax_immune.set_ylabel('Count')
        self.ax_immune.plot(self.time_data, self.immune_active_data, 'c-', 
                           linewidth=2, label='Activated')
        self.ax_immune.plot(self.time_data, self.kills_data, 'r--', 
                           linewidth=2, label='Kills')
        self.ax_immune.legend()
        self.ax_immune.grid(True, alpha=0.3)
    
    def run(self, duration=2.0, interval=50):
        """Run animated visualization"""
        # Calculate frames
        frames = int(duration / self.engine.config.dt)
        
        # Create animation
        anim = FuncAnimation(self.fig, self.update, frames=frames, 
                           interval=interval, blit=False, repeat=False)
        
        plt.tight_layout()
        plt.show()
        
        return anim


def main():
    """Run integrated visualization"""
    print("=" * 70)
    print("INTEGRATED VISUALIZATION: All 6 Modules")
    print("=" * 70)
    print()
    
    # Create engine
    config = SimulationConfig(dt=0.01, duration=2.0)
    engine = SimulationEngine(config)
    
    # Register modules
    print("Registering modules...")
    engine.register_module('molecular', MolecularModule, {
        'transcription_rate': 0.5
    })
    
    engine.register_module('cellular', CellularModule, {
        'n_normal_cells': 20,
        'n_cancer_cells': 5
    })
    
    engine.register_module('immune', ImmuneModule, {
        'n_t_cells': 8,
        'n_nk_cells': 5,
        'n_macrophages': 3
    })
    
    engine.register_module('vascular', VascularModule, {
        'n_capillaries': 8
    })
    
    engine.register_module('lymphatic', LymphaticModule, {
        'n_vessels': 4
    })
    
    engine.register_module('spatial', SpatialModule, {
        'grid_size': (20, 20, 10)
    })
    
    print()
    
    # Initialize
    engine.initialize()
    
    # Link modules
    print("Linking modules...")
    molecular = engine.modules['molecular']
    cellular = engine.modules['cellular']
    immune = engine.modules['immune']
    vascular = engine.modules['vascular']
    lymphatic = engine.modules['lymphatic']
    
    for cell_id in cellular.cells.keys():
        molecular.add_cell(cell_id)
    
    immune.set_cellular_module(cellular)
    vascular.set_cellular_module(cellular)
    lymphatic.set_cellular_module(cellular)
    lymphatic.set_immune_module(immune)
    
    print("✓ All modules linked")
    print()
    
    # Create visualizer
    print("Starting visualization...")
    print("(Close window to end)")
    print()
    
    viz = IntegratedVisualizer(engine)
    viz.run(duration=2.0, interval=50)
    
    print("\n✓ Visualization complete!")


if __name__ == '__main__':
    main()
