#!/usr/bin/env python3
"""
Complete Visualization - All 9 Modules
=======================================

Real-time 3D visualization of entire system.

Features:
- 3D tissue view with all cell types
- Capillary and lymphatic networks
- Immune cell movement
- Oxygen gradients
- Circadian rhythms (clock gene oscillations)
- Morphogen gradients (BMP, Shh, Wnt)
- Cell fates (anterior/middle/posterior)
- Epigenetic states (methylation levels)
- Real-time statistics (all 9 modules)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

from core import SimulationEngine, SimulationConfig
from modules import (MolecularModule, CellularModule, ImmuneModule,
                    VascularModule, LymphaticModule, SpatialModule,
                    EpigeneticModule, CircadianModule, MorphogenModule)


class CompleteVisualizer:
    """
    Complete visualization of all 9 modules
    """
    
    def __init__(self, engine):
        self.engine = engine
        
        # Create figure with 9 panels
        self.fig = plt.figure(figsize=(20, 12))
        
        # Main 3D tissue view (large, top-left)
        self.ax_3d = self.fig.add_subplot(3, 4, (1, 5), projection='3d')
        
        # Statistics panel (top-right)
        self.ax_stats = self.fig.add_subplot(3, 4, (2, 3))
        self.ax_stats.axis('off')
        
        # Oxygen gradient (middle-right)
        self.ax_oxygen = self.fig.add_subplot(3, 4, (6, 7))
        
        # Circadian rhythms (bottom-left)
        self.ax_circadian = self.fig.add_subplot(3, 4, 9)
        
        # Morphogen gradients (bottom-middle)
        self.ax_morphogen = self.fig.add_subplot(3, 4, 10)
        
        # Cell fates (bottom-right top)
        self.ax_fates = self.fig.add_subplot(3, 4, 4)
        
        # Epigenetic states (bottom-right middle)
        self.ax_epigenetic = self.fig.add_subplot(3, 4, 8)
        
        # Cell population (bottom-right bottom)
        self.ax_cells = self.fig.add_subplot(3, 4, 11)
        
        # Immune activity (far bottom-right)
        self.ax_immune = self.fig.add_subplot(3, 4, 12)
        
        # Data collection
        self.time_data = []
        self.cancer_data = []
        self.normal_data = []
        self.immune_active_data = []
        self.kills_data = []
        self.clock_data = []
        self.methylation_data = []
        
        # Setup
        self._setup_plots()
    
    def _setup_plots(self):
        """Setup plot properties"""
        # 3D view
        self.ax_3d.set_xlim(0, 200)
        self.ax_3d.set_ylim(0, 200)
        self.ax_3d.set_zlim(0, 100)
        self.ax_3d.set_xlabel('X (μm)', fontsize=8)
        self.ax_3d.set_ylabel('Y (μm)', fontsize=8)
        self.ax_3d.set_zlabel('Z (μm)', fontsize=8)
        self.ax_3d.set_title('Complete Tissue View (All 9 Modules)', 
                            fontweight='bold', fontsize=12)
        
        # Oxygen
        self.ax_oxygen.set_title('Oxygen Distribution', fontweight='bold', fontsize=10)
        self.ax_oxygen.set_xlabel('X (μm)', fontsize=8)
        self.ax_oxygen.set_ylabel('Y (μm)', fontsize=8)
        
        # Circadian
        self.ax_circadian.set_title('Circadian Rhythms', fontweight='bold', fontsize=10)
        self.ax_circadian.set_xlabel('Time (h)', fontsize=8)
        self.ax_circadian.set_ylabel('Clock Genes', fontsize=8)
        self.ax_circadian.grid(True, alpha=0.3)
        
        # Morphogen
        self.ax_morphogen.set_title('Morphogen Gradients', fontweight='bold', fontsize=10)
        self.ax_morphogen.set_xlabel('X (μm)', fontsize=8)
        self.ax_morphogen.set_ylabel('Y (μm)', fontsize=8)
        
        # Fates
        self.ax_fates.set_title('Cell Fates', fontweight='bold', fontsize=10)
        self.ax_fates.axis('off')
        
        # Epigenetic
        self.ax_epigenetic.set_title('Epigenetic States', fontweight='bold', fontsize=10)
        self.ax_epigenetic.set_xlabel('Time (h)', fontsize=8)
        self.ax_epigenetic.set_ylabel('Methylation', fontsize=8)
        self.ax_epigenetic.grid(True, alpha=0.3)
        
        # Cell counts
        self.ax_cells.set_title('Cell Population', fontweight='bold', fontsize=10)
        self.ax_cells.set_xlabel('Time (h)', fontsize=8)
        self.ax_cells.set_ylabel('Count', fontsize=8)
        self.ax_cells.grid(True, alpha=0.3)
        
        # Immune activity
        self.ax_immune.set_title('Immune Activity', fontweight='bold', fontsize=10)
        self.ax_immune.set_xlabel('Time (h)', fontsize=8)
        self.ax_immune.set_ylabel('Count', fontsize=8)
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
        
        if 'circadian' in state:
            self.clock_data.append(state['circadian']['master_phase'])
        
        if 'epigenetic' in state:
            self.methylation_data.append(state['epigenetic']['avg_methylation'])
        
        # Update all panels
        self._update_3d_view(state)
        self._update_statistics(state)
        self._update_oxygen_map(state)
        self._update_circadian_plot(state)
        self._update_morphogen_map(state)
        self._update_fate_chart(state)
        self._update_epigenetic_plot(state)
        self._update_time_series()
        
        return []
    
    def _update_3d_view(self, state):
        """Update 3D tissue view"""
        self.ax_3d.cla()
        self.ax_3d.set_xlim(0, 200)
        self.ax_3d.set_ylim(0, 200)
        self.ax_3d.set_zlim(0, 100)
        self.ax_3d.set_xlabel('X (μm)', fontsize=8)
        self.ax_3d.set_ylabel('Y (μm)', fontsize=8)
        self.ax_3d.set_zlabel('Z (μm)', fontsize=8)
        
        # Time of day indicator
        time_of_day = ""
        if 'circadian' in state:
            time_of_day = f" [{state['circadian']['master_time_of_day'].upper()}]"
        
        self.ax_3d.set_title(f'Tissue View (t={state["time"]:.1f}h){time_of_day}', 
                            fontweight='bold', fontsize=12)
        
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
        
        # Plot cells (colored by fate or methylation)
        morphogen = self.engine.modules.get('morphogen')
        epigenetic = self.engine.modules.get('epigenetic')
        
        for cell_id, cell in cellular.cells.items():
            if not cell.alive:
                continue
            
            # Determine color
            if cell.cell_type == 'cancer':
                # Color by methylation if available
                if epigenetic and cell_id in epigenetic.cell_identities:
                    methylation = epigenetic.cell_identities[cell_id].get('CDKN2A', {}).methylation_level
                    color = plt.cm.Reds(methylation)
                else:
                    color = 'red'
                marker = '*'
                size = 50
            else:
                # Color by fate if available
                if morphogen and cell_id in morphogen.cell_identities:
                    fate = morphogen.cell_identities[cell_id].cell_fate
                    if fate == 'posterior':
                        color = 'darkgreen'
                    elif fate == 'middle':
                        color = 'green'
                    else:
                        color = 'lightgreen'
                else:
                    color = 'green'
                marker = 'o'
                size = 30
            
            self.ax_3d.scatter([cell.position[0]], [cell.position[1]], [cell.position[2]],
                             c=[color], s=size, alpha=0.8, marker=marker)
        
        # Plot immune cells
        for immune_cell in immune.immune_cells.values():
            if immune_cell.in_blood:
                continue
            
            if immune_cell.cell_type == 'T_cell':
                color = 'cyan'
                marker = '^'
            elif immune_cell.cell_type == 'NK_cell':
                color = 'magenta'
                marker = 'd'
            else:
                color = 'orange'
                marker = 's'
            
            self.ax_3d.scatter([immune_cell.position[0]], 
                             [immune_cell.position[1]], 
                             [immune_cell.position[2]],
                             c=color, s=40, alpha=0.7, marker=marker)
    
    def _update_statistics(self, state):
        """Update statistics panel"""
        self.ax_stats.cla()
        self.ax_stats.axis('off')
        
        stats_text = f"""
╔═══════════════════════════╗
║   SIMULATION STATUS       ║
╚═══════════════════════════╝
Time: {state['time']:.1f}h  Steps: {state['step_count']}

CELLS
├─ Total: {state['cellular']['n_cells']}
├─ Cancer: {state['cellular']['n_cancer']}
├─ Normal: {state['cellular']['n_normal']}
└─ Deaths: {state['cellular']['total_deaths']}

IMMUNE
├─ Active: {state['immune']['n_immune_cells']}
├─ Activated: {state['immune']['n_activated']}
└─ Kills: {state['immune']['total_kills']}

VASCULAR
├─ Capillaries: {state['vascular']['n_capillaries']}
├─ Avg O2: {state['vascular']['avg_cell_O2']:.3f}
└─ Hypoxic: {state['vascular']['hypoxic_regions']}

LYMPHATIC
├─ Vessels: {state['lymphatic']['n_vessels']}
└─ Metastases: {state['lymphatic']['total_metastases']}
"""
        
        if 'epigenetic' in state:
            stats_text += f"""
EPIGENETIC
├─ Avg Methylation: {state['epigenetic']['avg_methylation']:.2f}
└─ Silenced Genes: {state['epigenetic']['silenced_genes']}
"""
        
        if 'circadian' in state:
            stats_text += f"""
CIRCADIAN
├─ Master Phase: {state['circadian']['master_phase']:.1f}h
├─ Time of Day: {state['circadian']['master_time_of_day']}
└─ Synchrony: {state['circadian']['synchrony']:.2f}
"""
        
        if 'morphogen' in state:
            stats_text += f"""
MORPHOGEN
├─ Gradients: {state['morphogen']['n_gradients']}
└─ Fates Set: {state['morphogen']['total_fates_determined']}
"""
        
        self.ax_stats.text(0.05, 0.5, stats_text, fontsize=8, family='monospace',
                          verticalalignment='center')
    
    def _update_oxygen_map(self, state):
        """Update oxygen heatmap"""
        self.ax_oxygen.cla()
        self.ax_oxygen.set_title('Oxygen Distribution', fontweight='bold', fontsize=10)
        self.ax_oxygen.set_xlabel('X (μm)', fontsize=8)
        self.ax_oxygen.set_ylabel('Y (μm)', fontsize=8)
        
        cellular = self.engine.modules['cellular']
        
        x = np.linspace(0, 200, 50)
        y = np.linspace(0, 200, 50)
        X, Y = np.meshgrid(x, y)
        O2 = np.ones_like(X) * 0.21
        
        for cell in cellular.cells.values():
            if cell.alive:
                dist = np.sqrt((X - cell.position[0])**2 + (Y - cell.position[1])**2)
                if cell.cell_type == 'cancer':
                    O2 -= 0.15 * np.exp(-dist / 30)
                else:
                    O2 -= 0.10 * np.exp(-dist / 30)
        
        O2 = np.clip(O2, 0, 0.21)
        
        self.ax_oxygen.contourf(X, Y, O2, levels=20, cmap='RdYlBu')
        self.ax_oxygen.contour(X, Y, O2, levels=[0.05], colors='red', linewidths=2)
    
    def _update_circadian_plot(self, state):
        """Update circadian rhythm plot"""
        self.ax_circadian.cla()
        self.ax_circadian.set_title('Circadian Rhythms', fontweight='bold', fontsize=10)
        self.ax_circadian.set_xlabel('Time (h)', fontsize=8)
        self.ax_circadian.set_ylabel('Phase', fontsize=8)
        self.ax_circadian.grid(True, alpha=0.3)
        
        if len(self.clock_data) > 1:
            self.ax_circadian.plot(self.time_data, self.clock_data, 'b-', 
                                  linewidth=2, label='Master Clock')
            self.ax_circadian.axhline(y=6, color='yellow', linestyle='--', 
                                     alpha=0.5, label='Dawn')
            self.ax_circadian.axhline(y=18, color='navy', linestyle='--', 
                                     alpha=0.5, label='Dusk')
            self.ax_circadian.set_ylim(0, 24)
            self.ax_circadian.legend(fontsize=7)
    
    def _update_morphogen_map(self, state):
        """Update morphogen gradient map"""
        self.ax_morphogen.cla()
        self.ax_morphogen.set_title('Morphogen: BMP Gradient', fontweight='bold', fontsize=10)
        self.ax_morphogen.set_xlabel('X (μm)', fontsize=8)
        self.ax_morphogen.set_ylabel('Y (μm)', fontsize=8)
        
        if 'morphogen' not in self.engine.modules:
            return
        
        morphogen = self.engine.modules['morphogen']
        
        # Create BMP gradient field
        x = np.linspace(0, 200, 50)
        y = np.linspace(0, 200, 50)
        X, Y = np.meshgrid(x, y)
        
        if 'BMP' in morphogen.gradients:
            gradient = morphogen.gradients['BMP']
            source = gradient.source_position
            
            # Calculate concentration at each point
            BMP = np.zeros_like(X)
            for i in range(len(x)):
                for j in range(len(y)):
                    pos = np.array([X[j, i], Y[j, i], 50])
                    dist = np.linalg.norm(pos - source)
                    BMP[j, i] = gradient.source_strength * np.exp(-dist / 50)
            
            self.ax_morphogen.contourf(X, Y, BMP, levels=20, cmap='viridis')
            self.ax_morphogen.plot(source[0], source[1], 'r*', markersize=15, 
                                  label='Source')
            self.ax_morphogen.legend(fontsize=7)
    
    def _update_fate_chart(self, state):
        """Update cell fate pie chart"""
        self.ax_fates.cla()
        self.ax_fates.set_title('Cell Fates', fontweight='bold', fontsize=10)
        self.ax_fates.axis('off')
        
        if 'morphogen' not in state or 'fate_distribution' not in state['morphogen']:
            return
        
        fates = state['morphogen']['fate_distribution']
        if fates:
            labels = list(fates.keys())
            sizes = list(fates.values())
            colors = ['lightcoral', 'lightyellow', 'lightgreen', 'lightblue']
            
            self.ax_fates.pie(sizes, labels=labels, colors=colors[:len(labels)],
                            autopct='%1.0f%%', startangle=90, textprops={'fontsize': 8})
    
    def _update_epigenetic_plot(self, state):
        """Update epigenetic methylation plot"""
        self.ax_epigenetic.cla()
        self.ax_epigenetic.set_title('Epigenetic States', fontweight='bold', fontsize=10)
        self.ax_epigenetic.set_xlabel('Time (h)', fontsize=8)
        self.ax_epigenetic.set_ylabel('Methylation', fontsize=8)
        self.ax_epigenetic.grid(True, alpha=0.3)
        
        if len(self.methylation_data) > 1:
            self.ax_epigenetic.plot(self.time_data, self.methylation_data, 'purple', 
                                   linewidth=2, label='Avg Methylation')
            self.ax_epigenetic.axhline(y=0.7, color='red', linestyle='--', 
                                      alpha=0.5, label='Silencing Threshold')
            self.ax_epigenetic.set_ylim(0, 1)
            self.ax_epigenetic.legend(fontsize=7)
    
    def _update_time_series(self):
        """Update time series plots"""
        if len(self.time_data) < 2:
            return
        
        # Cell population
        self.ax_cells.cla()
        self.ax_cells.set_title('Cell Population', fontweight='bold', fontsize=10)
        self.ax_cells.set_xlabel('Time (h)', fontsize=8)
        self.ax_cells.set_ylabel('Count', fontsize=8)
        self.ax_cells.plot(self.time_data, self.cancer_data, 'r-', linewidth=2, label='Cancer')
        self.ax_cells.plot(self.time_data, self.normal_data, 'g-', linewidth=2, label='Normal')
        self.ax_cells.legend(fontsize=7)
        self.ax_cells.grid(True, alpha=0.3)
        
        # Immune activity
        self.ax_immune.cla()
        self.ax_immune.set_title('Immune Activity', fontweight='bold', fontsize=10)
        self.ax_immune.set_xlabel('Time (h)', fontsize=8)
        self.ax_immune.set_ylabel('Count', fontsize=8)
        self.ax_immune.plot(self.time_data, self.immune_active_data, 'c-', 
                           linewidth=2, label='Activated')
        self.ax_immune.plot(self.time_data, self.kills_data, 'r--', 
                           linewidth=2, label='Kills')
        self.ax_immune.legend(fontsize=7)
        self.ax_immune.grid(True, alpha=0.3)
    
    def run(self, duration=2.0, interval=50):
        """Run animated visualization"""
        frames = int(duration / self.engine.config.dt)
        
        anim = FuncAnimation(self.fig, self.update, frames=frames, 
                           interval=interval, blit=False, repeat=False)
        
        plt.tight_layout()
        plt.show()
        
        return anim


def main():
    """Run complete visualization"""
    print("=" * 70)
    print("COMPLETE VISUALIZATION: All 9 Modules")
    print("=" * 70)
    print()
    
    # Create engine
    config = SimulationConfig(dt=0.01, duration=2.0)
    engine = SimulationEngine(config)
    
    # Register ALL modules
    print("Registering modules...")
    engine.register_module('molecular', MolecularModule)
    engine.register_module('cellular', CellularModule, {
        'n_normal_cells': 15,
        'n_cancer_cells': 3
    })
    engine.register_module('immune', ImmuneModule, {
        'n_t_cells': 5,
        'n_nk_cells': 3,
        'n_macrophages': 2
    })
    engine.register_module('vascular', VascularModule, {
        'n_capillaries': 6
    })
    engine.register_module('lymphatic', LymphaticModule, {
        'n_vessels': 3
    })
    engine.register_module('spatial', SpatialModule)
    engine.register_module('epigenetic', EpigeneticModule)
    engine.register_module('circadian', CircadianModule)
    engine.register_module('morphogen', MorphogenModule)
    
    print()
    
    # Initialize
    engine.initialize()
    
    # Link all modules
    print("Linking modules...")
    molecular = engine.modules['molecular']
    cellular = engine.modules['cellular']
    immune = engine.modules['immune']
    vascular = engine.modules['vascular']
    lymphatic = engine.modules['lymphatic']
    epigenetic = engine.modules['epigenetic']
    circadian = engine.modules['circadian']
    morphogen = engine.modules['morphogen']
    
    for cell_id, cell in cellular.cells.items():
        molecular.add_cell(cell_id)
        epigenetic.add_cell(cell_id, cell.cell_type)
        circadian.add_cell(cell_id)
        morphogen.add_cell(cell_id, cell.position)
    
    immune.set_cellular_module(cellular)
    vascular.set_cellular_module(cellular)
    lymphatic.set_cellular_module(cellular)
    lymphatic.set_immune_module(immune)
    
    print("✓ All 9 modules linked")
    print()
    
    # Create visualizer
    print("Starting complete visualization...")
    print("(Close window to end)")
    print()
    print("Panels:")
    print("  1. 3D Tissue View (all cells, vessels, immune)")
    print("  2. Statistics (all 9 modules)")
    print("  3. Oxygen Distribution")
    print("  4. Circadian Rhythms")
    print("  5. Morphogen Gradients")
    print("  6. Cell Fates")
    print("  7. Epigenetic States")
    print("  8. Cell Population")
    print("  9. Immune Activity")
    print()
    
    viz = CompleteVisualizer(engine)
    viz.run(duration=2.0, interval=50)
    
    print("\n✓ Visualization complete!")


if __name__ == '__main__':
    main()
