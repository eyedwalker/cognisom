#!/usr/bin/env python3
"""
Prostate Tissue Multi-System Visualization
===========================================

Complete tissue simulation with:
- Prostate epithelial cells (normal + cancer)
- Capillary network (O2/nutrient exchange)
- Lymphatic vessels (drainage + immune trafficking)
- Immune cells (T cells, NK cells, macrophages)
- Molecular exchange (exosomes, chemokines)

Real-time 3D visualization showing all systems interacting!
"""

import sys
sys.path.insert(0, '../..')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle, FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Line3D
import matplotlib.patches as mpatches

from engine.py.molecular.nucleic_acids import Gene
from engine.py.molecular.exosomes import Exosome, ExosomeSystem


# ============================================
# Cell Classes
# ============================================

class ProstateCell:
    """Prostate epithelial cell"""
    
    def __init__(self, cell_id, position, cell_type="normal"):
        self.id = cell_id
        self.position = np.array(position, dtype=np.float32)
        self.cell_type = cell_type  # "normal" or "cancer"
        
        # Prostate-specific
        self.psa_production = 1.0 if cell_type == "normal" else 2.5
        self.androgen_receptor = 1.0
        
        # Metabolism
        self.oxygen = 0.21
        self.glucose = 5.0
        self.lactate = 0.0
        
        # Cancer properties
        self.mhc1_expression = 1.0 if cell_type == "normal" else 0.3
        self.mutations = []
        
        # State
        self.alive = True
        self.dividing = False
    
    def consume_nutrients(self, dt):
        """Consume O2 and glucose"""
        if self.cell_type == "cancer":
            # Cancer cells consume more glucose (Warburg effect)
            self.glucose -= 0.5 * dt
            self.oxygen -= 0.1 * dt
            self.lactate += 0.3 * dt
        else:
            self.glucose -= 0.2 * dt
            self.oxygen -= 0.15 * dt
            self.lactate += 0.1 * dt
    
    def check_hypoxia(self):
        """Check if cell is hypoxic"""
        return self.oxygen < 0.05


class ImmuneCell:
    """Base immune cell"""
    
    def __init__(self, cell_id, position, cell_type):
        self.id = cell_id
        self.position = np.array(position, dtype=np.float32)
        self.cell_type = cell_type  # "T_cell", "NK_cell", "macrophage"
        self.velocity = np.zeros(3)
        
        # State
        self.activated = False
        self.target_cell = None
        self.in_blood = False
    
    def migrate(self, direction, speed=10.0, dt=0.01):
        """Move in direction"""
        self.velocity = direction * speed
        self.position += self.velocity * dt
    
    def recognize_cancer(self, cell):
        """Check if can recognize cancer cell"""
        if self.cell_type == "T_cell":
            # T cells need MHC-I
            return cell.cell_type == "cancer" and cell.mhc1_expression > 0.2
        elif self.cell_type == "NK_cell":
            # NK cells detect LOW MHC-I
            return cell.cell_type == "cancer" and cell.mhc1_expression < 0.4
        elif self.cell_type == "macrophage":
            # Macrophages can recognize via other markers
            return cell.cell_type == "cancer"
        return False
    
    def kill_target(self, cell):
        """Kill target cell"""
        cell.alive = False
        self.target_cell = None


# ============================================
# Vascular System
# ============================================

class Capillary:
    """Blood capillary"""
    
    def __init__(self, start, end):
        self.start = np.array(start, dtype=np.float32)
        self.end = np.array(end, dtype=np.float32)
        self.length = np.linalg.norm(self.end - self.start)
        
        # Contents
        self.oxygen = 0.21
        self.glucose = 5.0
        self.immune_cells = []
        
        # Flow
        self.flow_rate = 0.5  # mm/s
    
    def get_position(self, t):
        """Get position along capillary (0-1)"""
        return self.start + t * (self.end - self.start)
    
    def exchange_with_tissue(self, cells, dt):
        """Exchange O2/glucose with nearby cells"""
        # Sample points along capillary
        for t in np.linspace(0, 1, 10):
            pos = self.get_position(t)
            
            # Find nearby cells
            for cell in cells:
                distance = np.linalg.norm(cell.position - pos)
                
                if distance < 50:  # Exchange radius
                    # O2 diffusion
                    gradient = (self.oxygen - cell.oxygen) / (distance + 1)
                    flux = 0.1 * gradient * dt
                    
                    self.oxygen -= flux
                    cell.oxygen += flux
                    
                    # Glucose diffusion
                    gradient = (self.glucose - cell.glucose) / (distance + 1)
                    flux = 0.05 * gradient * dt
                    
                    self.glucose -= flux
                    cell.glucose += flux
                    
                    # Lactate removal
                    gradient = (cell.lactate - 0) / (distance + 1)
                    flux = 0.05 * gradient * dt
                    
                    cell.lactate -= flux


class LymphaticVessel:
    """Lymphatic vessel"""
    
    def __init__(self, start, end):
        self.start = np.array(start, dtype=np.float32)
        self.end = np.array(end, dtype=np.float32)
        self.length = np.linalg.norm(self.end - self.start)
        
        # Contents
        self.immune_cells = []
        self.cancer_cells = []
        
        # Drainage
        self.drainage_rate = 0.01
    
    def get_position(self, t):
        """Get position along vessel (0-1)"""
        return self.start + t * (self.end - self.start)
    
    def collect_cells(self, cells, immune_cells, dt):
        """Collect immune and cancer cells"""
        # Sample points along vessel
        for t in np.linspace(0, 1, 5):
            pos = self.get_position(t)
            
            # Collect activated immune cells
            for immune in immune_cells:
                if not immune.in_blood:
                    distance = np.linalg.norm(immune.position - pos)
                    if distance < 20 and immune.activated:
                        # Immune cell enters lymphatic
                        if np.random.random() < 0.05 * dt:
                            self.immune_cells.append(immune)
                            immune.in_blood = True
            
            # Collect cancer cells (metastasis!)
            for cell in cells:
                if cell.cell_type == "cancer" and cell.alive:
                    distance = np.linalg.norm(cell.position - pos)
                    if distance < 15:
                        # Cancer cell can enter lymphatic
                        if np.random.random() < 0.001 * dt:
                            self.cancer_cells.append(cell)
                            print(f"⚠️  Cancer cell entered lymphatic at {pos}!")


# ============================================
# Tissue System
# ============================================

class ProstateTissue:
    """Complete prostate tissue with all systems"""
    
    def __init__(self, size=(200, 200, 100)):
        self.size = size
        
        # Cells
        self.epithelial_cells = []
        self.immune_cells = []
        
        # Vasculature
        self.capillaries = []
        self.lymphatics = []
        
        # Molecular
        self.exosome_system = ExosomeSystem()
        
        # Environment
        self.chemokine_field = np.zeros(size)
        
        # Statistics
        self.time = 0.0
        self.cancer_cells_killed = 0
        self.metastatic_events = 0
    
    def initialize(self):
        """Create initial tissue structure"""
        
        # Create epithelial cells (organized in acini)
        print("Creating epithelial cells...")
        n_normal = 80
        n_cancer = 20
        
        # Normal cells (clustered)
        for i in range(n_normal):
            angle = 2 * np.pi * i / n_normal
            radius = 50
            x = 100 + radius * np.cos(angle)
            y = 100 + radius * np.sin(angle)
            z = 50 + np.random.uniform(-10, 10)
            
            cell = ProstateCell(i, [x, y, z], "normal")
            self.epithelial_cells.append(cell)
        
        # Cancer cells (clustered in one region)
        for i in range(n_cancer):
            x = 120 + np.random.uniform(-15, 15)
            y = 120 + np.random.uniform(-15, 15)
            z = 50 + np.random.uniform(-5, 5)
            
            cell = ProstateCell(n_normal + i, [x, y, z], "cancer")
            cell.mutations = ["KRAS_G12D", "TP53_R175H"]
            self.epithelial_cells.append(cell)
        
        # Create capillary network (radial pattern)
        print("Creating capillary network...")
        for i in range(8):
            angle = 2 * np.pi * i / 8
            start = [100, 100, 50]
            end = [100 + 80 * np.cos(angle), 100 + 80 * np.sin(angle), 50]
            
            capillary = Capillary(start, end)
            self.capillaries.append(capillary)
        
        # Create lymphatic vessels (fewer, larger)
        print("Creating lymphatic vessels...")
        for i in range(4):
            angle = 2 * np.pi * i / 4 + np.pi/4
            start = [100 + 60 * np.cos(angle), 100 + 60 * np.sin(angle), 50]
            end = [100 + 90 * np.cos(angle), 100 + 90 * np.sin(angle), 50]
            
            lymphatic = LymphaticVessel(start, end)
            self.lymphatics.append(lymphatic)
        
        # Create immune cells
        print("Creating immune cells...")
        
        # T cells (patrol tissue)
        for i in range(15):
            x = np.random.uniform(50, 150)
            y = np.random.uniform(50, 150)
            z = np.random.uniform(40, 60)
            
            immune = ImmuneCell(i, [x, y, z], "T_cell")
            self.immune_cells.append(immune)
        
        # NK cells
        for i in range(10):
            x = np.random.uniform(50, 150)
            y = np.random.uniform(50, 150)
            z = np.random.uniform(40, 60)
            
            immune = ImmuneCell(15 + i, [x, y, z], "NK_cell")
            self.immune_cells.append(immune)
        
        # Macrophages
        for i in range(8):
            x = np.random.uniform(50, 150)
            y = np.random.uniform(50, 150)
            z = np.random.uniform(40, 60)
            
            immune = ImmuneCell(25 + i, [x, y, z], "macrophage")
            self.immune_cells.append(immune)
        
        print(f"✓ Tissue initialized:")
        print(f"  Epithelial cells: {len(self.epithelial_cells)}")
        print(f"  Immune cells: {len(self.immune_cells)}")
        print(f"  Capillaries: {len(self.capillaries)}")
        print(f"  Lymphatics: {len(self.lymphatics)}")
    
    def update(self, dt):
        """Update all systems"""
        self.time += dt
        
        # Update cells
        for cell in self.epithelial_cells:
            if cell.alive:
                cell.consume_nutrients(dt)
        
        # Capillary exchange
        for capillary in self.capillaries:
            capillary.exchange_with_tissue(self.epithelial_cells, dt)
        
        # Immune cell behavior
        for immune in self.immune_cells:
            if not immune.in_blood:
                # Random walk + chemotaxis
                direction = np.random.randn(3)
                direction = direction / (np.linalg.norm(direction) + 1e-6)
                immune.migrate(direction, speed=5.0, dt=dt)
                
                # Keep in bounds
                immune.position = np.clip(immune.position, 
                                         [20, 20, 20], 
                                         [180, 180, 80])
                
                # Look for cancer cells
                if not immune.activated:
                    for cell in self.epithelial_cells:
                        if cell.alive and cell.cell_type == "cancer":
                            distance = np.linalg.norm(immune.position - cell.position)
                            
                            if distance < 10:
                                if immune.recognize_cancer(cell):
                                    immune.activated = True
                                    immune.target_cell = cell
                                    break
                
                # Kill target
                if immune.activated and immune.target_cell:
                    if immune.target_cell.alive:
                        distance = np.linalg.norm(immune.position - immune.target_cell.position)
                        
                        if distance < 5:
                            immune.kill_target(immune.target_cell)
                            self.cancer_cells_killed += 1
                            print(f"t={self.time:.1f}h: {immune.cell_type} killed cancer cell!")
                    else:
                        immune.activated = False
                        immune.target_cell = None
        
        # Lymphatic collection
        for lymphatic in self.lymphatics:
            lymphatic.collect_cells(self.epithelial_cells, self.immune_cells, dt)
            
            if len(lymphatic.cancer_cells) > len(lymphatic.cancer_cells) - 1:
                self.metastatic_events += 1
        
        # Exosome system
        self.exosome_system.update(dt)
    
    def get_statistics(self):
        """Get tissue statistics"""
        alive_cells = [c for c in self.epithelial_cells if c.alive]
        cancer_cells = [c for c in alive_cells if c.cell_type == "cancer"]
        normal_cells = [c for c in alive_cells if c.cell_type == "normal"]
        
        active_immune = [i for i in self.immune_cells if not i.in_blood]
        activated_immune = [i for i in active_immune if i.activated]
        
        return {
            'time': self.time,
            'total_cells': len(alive_cells),
            'cancer_cells': len(cancer_cells),
            'normal_cells': len(normal_cells),
            'immune_cells': len(active_immune),
            'activated_immune': len(activated_immune),
            'cancer_killed': self.cancer_cells_killed,
            'metastatic_events': self.metastatic_events,
            'exosomes': len(self.exosome_system.exosomes)
        }


# ============================================
# Visualization
# ============================================

def create_tissue_visualization():
    """Create interactive 3D visualization"""
    
    # Create tissue
    tissue = ProstateTissue()
    tissue.initialize()
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    
    # 3D tissue view
    ax_3d = fig.add_subplot(2, 3, (1, 4), projection='3d')
    ax_3d.set_xlim(0, 200)
    ax_3d.set_ylim(0, 200)
    ax_3d.set_zlim(0, 100)
    ax_3d.set_xlabel('X (μm)')
    ax_3d.set_ylabel('Y (μm)')
    ax_3d.set_zlabel('Z (μm)')
    ax_3d.set_title('Prostate Tissue: Multi-System View', fontweight='bold', fontsize=14)
    
    # Statistics panels
    ax_stats = fig.add_subplot(2, 3, 2)
    ax_stats.axis('off')
    
    ax_immune = fig.add_subplot(2, 3, 3)
    ax_immune.set_title('Immune Activity', fontweight='bold')
    ax_immune.set_xlabel('Time (hours)')
    ax_immune.set_ylabel('Count')
    
    ax_oxygen = fig.add_subplot(2, 3, 5)
    ax_oxygen.set_title('Oxygen Distribution', fontweight='bold')
    ax_oxygen.set_xlabel('X (μm)')
    ax_oxygen.set_ylabel('Y (μm)')
    
    ax_cancer = fig.add_subplot(2, 3, 6)
    ax_cancer.set_title('Cancer vs Immune', fontweight='bold')
    ax_cancer.set_xlabel('Time (hours)')
    ax_cancer.set_ylabel('Count')
    
    # Data collection
    time_data = []
    cancer_count_data = []
    immune_active_data = []
    cancer_killed_data = []
    
    def update(frame):
        """Update visualization"""
        
        # Update tissue
        dt = 0.05  # hours
        tissue.update(dt)
        
        # Get statistics
        stats = tissue.get_statistics()
        
        # Collect data
        time_data.append(stats['time'])
        cancer_count_data.append(stats['cancer_cells'])
        immune_active_data.append(stats['activated_immune'])
        cancer_killed_data.append(stats['cancer_killed'])
        
        # Clear 3D plot
        ax_3d.cla()
        ax_3d.set_xlim(0, 200)
        ax_3d.set_ylim(0, 200)
        ax_3d.set_zlim(0, 100)
        ax_3d.set_xlabel('X (μm)')
        ax_3d.set_ylabel('Y (μm)')
        ax_3d.set_zlabel('Z (μm)')
        ax_3d.set_title(f'Prostate Tissue (t={stats["time"]:.1f}h)', 
                       fontweight='bold', fontsize=14)
        
        # Plot capillaries (red lines)
        for cap in tissue.capillaries:
            ax_3d.plot([cap.start[0], cap.end[0]],
                      [cap.start[1], cap.end[1]],
                      [cap.start[2], cap.end[2]],
                      'r-', linewidth=2, alpha=0.6, label='Capillary' if cap == tissue.capillaries[0] else '')
        
        # Plot lymphatics (blue lines)
        for lymph in tissue.lymphatics:
            ax_3d.plot([lymph.start[0], lymph.end[0]],
                      [lymph.start[1], lymph.end[1]],
                      [lymph.start[2], lymph.end[2]],
                      'b-', linewidth=3, alpha=0.5, label='Lymphatic' if lymph == tissue.lymphatics[0] else '')
        
        # Plot cells
        alive_cells = [c for c in tissue.epithelial_cells if c.alive]
        
        normal_cells = [c for c in alive_cells if c.cell_type == "normal"]
        cancer_cells = [c for c in alive_cells if c.cell_type == "cancer"]
        
        if normal_cells:
            normal_pos = np.array([c.position for c in normal_cells])
            ax_3d.scatter(normal_pos[:, 0], normal_pos[:, 1], normal_pos[:, 2],
                         c='green', s=30, alpha=0.6, label='Normal cells')
        
        if cancer_cells:
            cancer_pos = np.array([c.position for c in cancer_cells])
            ax_3d.scatter(cancer_pos[:, 0], cancer_pos[:, 1], cancer_pos[:, 2],
                         c='red', s=50, alpha=0.8, marker='*', label='Cancer cells')
        
        # Plot immune cells
        active_immune = [i for i in tissue.immune_cells if not i.in_blood]
        
        t_cells = [i for i in active_immune if i.cell_type == "T_cell"]
        nk_cells = [i for i in active_immune if i.cell_type == "NK_cell"]
        macrophages = [i for i in active_immune if i.cell_type == "macrophage"]
        
        if t_cells:
            t_pos = np.array([i.position for i in t_cells])
            ax_3d.scatter(t_pos[:, 0], t_pos[:, 1], t_pos[:, 2],
                         c='cyan', s=40, alpha=0.7, marker='^', label='T cells')
        
        if nk_cells:
            nk_pos = np.array([i.position for i in nk_cells])
            ax_3d.scatter(nk_pos[:, 0], nk_pos[:, 1], nk_pos[:, 2],
                         c='magenta', s=40, alpha=0.7, marker='d', label='NK cells')
        
        if macrophages:
            mac_pos = np.array([i.position for i in macrophages])
            ax_3d.scatter(mac_pos[:, 0], mac_pos[:, 1], mac_pos[:, 2],
                         c='orange', s=40, alpha=0.7, marker='s', label='Macrophages')
        
        ax_3d.legend(loc='upper right', fontsize=8)
        
        # Update statistics text
        ax_stats.cla()
        ax_stats.axis('off')
        stats_text = f"""
TISSUE STATISTICS
─────────────────
Time: {stats['time']:.1f} hours

CELLS:
  Total: {stats['total_cells']}
  Cancer: {stats['cancer_cells']}
  Normal: {stats['normal_cells']}

IMMUNE:
  Active: {stats['immune_cells']}
  Attacking: {stats['activated_immune']}
  Kills: {stats['cancer_killed']}

METASTASIS:
  Events: {stats['metastatic_events']}

EXOSOMES:
  Active: {stats['exosomes']}
        """
        ax_stats.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                     verticalalignment='center')
        
        # Update immune activity plot
        if len(time_data) > 1:
            ax_immune.cla()
            ax_immune.set_title('Immune Activity', fontweight='bold')
            ax_immune.set_xlabel('Time (hours)')
            ax_immune.set_ylabel('Count')
            ax_immune.plot(time_data, immune_active_data, 'c-', linewidth=2, label='Activated')
            ax_immune.plot(time_data, cancer_killed_data, 'r--', linewidth=2, label='Killed')
            ax_immune.legend()
            ax_immune.grid(True, alpha=0.3)
        
        # Update oxygen heatmap
        ax_oxygen.cla()
        ax_oxygen.set_title('Oxygen Distribution', fontweight='bold')
        ax_oxygen.set_xlabel('X (μm)')
        ax_oxygen.set_ylabel('Y (μm)')
        
        # Create oxygen field
        x = np.linspace(0, 200, 50)
        y = np.linspace(0, 200, 50)
        X, Y = np.meshgrid(x, y)
        O2 = np.ones_like(X) * 0.21
        
        # Reduce O2 near cancer cells
        for cell in cancer_cells:
            dist = np.sqrt((X - cell.position[0])**2 + (Y - cell.position[1])**2)
            O2 -= 0.15 * np.exp(-dist / 30)
        
        O2 = np.clip(O2, 0, 0.21)
        
        im = ax_oxygen.contourf(X, Y, O2, levels=20, cmap='RdYlBu')
        ax_oxygen.contour(X, Y, O2, levels=[0.05], colors='red', linewidths=2)
        
        # Update cancer vs immune plot
        if len(time_data) > 1:
            ax_cancer.cla()
            ax_cancer.set_title('Cancer vs Immune', fontweight='bold')
            ax_cancer.set_xlabel('Time (hours)')
            ax_cancer.set_ylabel('Count')
            ax_cancer.plot(time_data, cancer_count_data, 'r-', linewidth=2, label='Cancer cells')
            ax_cancer.plot(time_data, immune_active_data, 'c-', linewidth=2, label='Active immune')
            ax_cancer.legend()
            ax_cancer.grid(True, alpha=0.3)
        
        return []
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=200, interval=50, blit=False)
    
    plt.tight_layout()
    plt.show()
    
    return tissue


# ============================================
# Main
# ============================================

if __name__ == '__main__':
    print("=" * 70)
    print("PROSTATE TISSUE: Multi-System Visualization")
    print("=" * 70)
    print()
    print("Systems simulated:")
    print("  ✓ Prostate epithelial cells (normal + cancer)")
    print("  ✓ Capillary network (O2/nutrient exchange)")
    print("  ✓ Lymphatic vessels (drainage + metastasis)")
    print("  ✓ Immune cells (T cells, NK cells, macrophages)")
    print("  ✓ Immune surveillance and cancer killing")
    print("  ✓ Molecular exchange (exosomes)")
    print()
    print("=" * 70)
    print()
    
    tissue = create_tissue_visualization()
    
    print()
    print("=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    
    stats = tissue.get_statistics()
    print()
    print("Final Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
