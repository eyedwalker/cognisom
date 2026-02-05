"""
Intracellular Visualization
============================

Beautiful visualizations of cell internals:
- Cell structure with organelles
- DNA, RNA, protein dynamics
- Membrane receptors
- Signaling pathways
- Real-time molecular counts
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, Wedge, Ellipse
import numpy as np
from typing import Dict, List, Optional
from .intracellular import IntracellularModel, Organelle


class CellVisualizer:
    """Visualize cell internals"""
    
    def __init__(self, cell_model: IntracellularModel):
        self.model = cell_model
        self.fig = None
        self.axes = None
    
    def plot_cell_structure(self, ax=None):
        """Draw cell structure with organelles"""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Cell membrane (outer circle)
        cell_radius = 10
        cell = Circle((0, 0), cell_radius, fill=False, 
                     edgecolor='#2E86AB', linewidth=3, label='Cell Membrane')
        ax.add_patch(cell)
        
        # Nucleus (large circle in center)
        nucleus_radius = 3.5
        nucleus = Circle((0, 0), nucleus_radius, 
                        facecolor='#E8F4F8', edgecolor='#1F4788', 
                        linewidth=2, alpha=0.7, label='Nucleus')
        ax.add_patch(nucleus)
        
        # Nuclear membrane (double line)
        nucleus_outer = Circle((0, 0), nucleus_radius, fill=False,
                              edgecolor='#1F4788', linewidth=1.5, linestyle='--')
        ax.add_patch(nucleus_outer)
        
        # DNA (squiggly lines in nucleus)
        theta = np.linspace(0, 4*np.pi, 100)
        r = nucleus_radius * 0.6
        x_dna = r * np.cos(theta) * (1 + 0.1*np.sin(10*theta))
        y_dna = r * np.sin(theta) * (1 + 0.1*np.cos(10*theta))
        ax.plot(x_dna, y_dna, color='#8B4789', linewidth=2, alpha=0.6, label='DNA')
        
        # Mitochondria (small ovals scattered)
        mito_count = 8
        for i in range(mito_count):
            angle = 2 * np.pi * i / mito_count
            distance = 6
            x = distance * np.cos(angle)
            y = distance * np.sin(angle)
            
            mito = Ellipse((x, y), width=1.2, height=0.6, angle=np.degrees(angle),
                          facecolor='#F18F01', edgecolor='#C73E1D', 
                          linewidth=1.5, alpha=0.7)
            ax.add_patch(mito)
            
            if i == 0:
                mito.set_label('Mitochondria')
        
        # Endoplasmic Reticulum (wavy network)
        er_angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
        for angle in er_angles:
            r_start = nucleus_radius + 0.5
            r_end = cell_radius - 1
            r_points = np.linspace(r_start, r_end, 20)
            
            x_er = r_points * np.cos(angle) + 0.3*np.sin(r_points*2)
            y_er = r_points * np.sin(angle) + 0.3*np.cos(r_points*2)
            
            ax.plot(x_er, y_er, color='#06A77D', linewidth=2, alpha=0.5)
        
        # Add label for ER
        ax.plot([], [], color='#06A77D', linewidth=2, alpha=0.5, label='Endoplasmic Reticulum')
        
        # Ribosomes (small dots on ER and cytoplasm)
        np.random.seed(42)
        ribosome_count = 30
        for i in range(ribosome_count):
            angle = np.random.uniform(0, 2*np.pi)
            distance = np.random.uniform(nucleus_radius + 1, cell_radius - 0.5)
            x = distance * np.cos(angle)
            y = distance * np.sin(angle)
            
            ribosome = Circle((x, y), 0.15, facecolor='#A23B72', 
                            edgecolor='none', alpha=0.8)
            ax.add_patch(ribosome)
            
            if i == 0:
                ribosome.set_label('Ribosomes')
        
        # Golgi apparatus (stacked crescents)
        golgi_x, golgi_y = 4, -4
        for i in range(4):
            golgi = Wedge((golgi_x, golgi_y + i*0.3), 1.5, 180, 360,
                         facecolor='#F4A261', edgecolor='#E76F51', 
                         linewidth=1, alpha=0.6)
            ax.add_patch(golgi)
            
            if i == 0:
                golgi.set_label('Golgi Apparatus')
        
        # Membrane receptors (small shapes on membrane)
        receptor_count = 12
        for i in range(receptor_count):
            angle = 2 * np.pi * i / receptor_count
            x = cell_radius * np.cos(angle)
            y = cell_radius * np.sin(angle)
            
            # Receptor as small rectangle
            receptor = Rectangle((x-0.2, y-0.4), 0.4, 0.8, 
                                angle=np.degrees(angle),
                                facecolor='#E63946', edgecolor='#9D0208',
                                linewidth=1, alpha=0.8)
            ax.add_patch(receptor)
            
            if i == 0:
                receptor.set_label('Membrane Receptors')
        
        # Vesicles (small circles)
        vesicle_count = 6
        for i in range(vesicle_count):
            angle = np.random.uniform(0, 2*np.pi)
            distance = np.random.uniform(nucleus_radius + 1, cell_radius - 1.5)
            x = distance * np.cos(angle)
            y = distance * np.sin(angle)
            
            vesicle = Circle((x, y), 0.4, facecolor='#90E0EF', 
                           edgecolor='#0077B6', linewidth=1, alpha=0.6)
            ax.add_patch(vesicle)
            
            if i == 0:
                vesicle.set_label('Vesicles')
        
        # Styling
        ax.set_xlim(-12, 12)
        ax.set_ylim(-12, 12)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)
        ax.set_title('Cell Structure & Organelles', fontsize=16, fontweight='bold', pad=20)
        
        return ax
    
    def plot_molecular_dynamics(self, history: Dict, ax=None):
        """Plot molecular species over time"""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        time = history['time']
        
        # Plot mRNA
        ax.plot(time, history['total_mrna'], 'o-', 
               label='Total mRNA', color='#8B4789', linewidth=2, markersize=4)
        
        # Plot proteins
        ax.plot(time, history['total_proteins'], 's-',
               label='Total Proteins', color='#06A77D', linewidth=2, markersize=4)
        
        # Plot ATP (scaled)
        atp_scaled = np.array(history['atp']) / 1000
        ax.plot(time, atp_scaled, '^-',
               label='ATP (×1000)', color='#F18F01', linewidth=2, markersize=4)
        
        ax.set_xlabel('Time (hours)', fontsize=12)
        ax.set_ylabel('Molecule Count', fontsize=12)
        ax.set_title('Molecular Dynamics', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_gene_expression(self, expression: Dict, ax=None):
        """Plot gene expression levels (mRNA counts)"""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        genes = list(expression.keys())
        counts = list(expression.values())
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', 
                 '#E63946', '#8B4789', '#F4A261', '#90E0EF']
        
        bars = ax.bar(genes, counts, color=colors[:len(genes)], 
                     edgecolor='black', linewidth=1.5, alpha=0.8)
        
        ax.set_xlabel('Gene', fontsize=12)
        ax.set_ylabel('mRNA Count', fontsize=12)
        ax.set_title('Gene Expression Profile', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=9)
        
        return ax
    
    def plot_protein_levels(self, proteins: Dict, ax=None):
        """Plot protein abundance"""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        if not proteins:
            ax.text(0.5, 0.5, 'No proteins yet', 
                   ha='center', va='center', fontsize=14)
            ax.set_title('Protein Levels', fontsize=14, fontweight='bold')
            return ax
        
        names = list(proteins.keys())
        counts = list(proteins.values())
        
        colors = ['#06A77D', '#F18F01', '#E63946', '#2E86AB', 
                 '#A23B72', '#8B4789', '#F4A261', '#90E0EF']
        
        bars = ax.barh(names, counts, color=colors[:len(names)],
                      edgecolor='black', linewidth=1.5, alpha=0.8)
        
        ax.set_xlabel('Protein Count', fontsize=12)
        ax.set_ylabel('Protein', fontsize=12)
        ax.set_title('Protein Abundance', fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            if width > 0:
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f' {int(width)}',
                       ha='left', va='center', fontsize=9)
        
        return ax
    
    def create_dashboard(self, history: Dict, save_path: Optional[str] = None):
        """Create comprehensive dashboard"""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Cell structure
        ax1 = fig.add_subplot(gs[0:2, 0])
        self.plot_cell_structure(ax1)
        
        # Molecular dynamics
        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_molecular_dynamics(history, ax2)
        
        # Gene expression (latest timepoint)
        ax3 = fig.add_subplot(gs[1, 1])
        expression = self.model.get_gene_expression()
        self.plot_gene_expression(expression, ax3)
        
        # Protein levels
        ax4 = fig.add_subplot(gs[2, :])
        proteins = self.model.get_protein_levels()
        self.plot_protein_levels(proteins, ax4)
        
        fig.suptitle('Intracellular Simulation Dashboard', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Dashboard saved: {save_path}")
        
        return fig
