"""
Spatial Grid
============

3D grid for cell positions and molecular diffusion.

Features:
- 3D voxel grid
- Diffusible fields (O2, glucose, cytokines)
- PDE solvers for diffusion
- Cell-environment interactions
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


@dataclass
class GridConfig:
    """Configuration for spatial grid"""
    size: Tuple[int, int, int] = (100, 100, 100)  # voxels
    resolution: float = 10.0  # μm per voxel
    
    # Diffusion coefficients (μm²/s)
    D_oxygen: float = 2000.0
    D_glucose: float = 600.0
    D_cytokine: float = 100.0
    
    # Initial concentrations
    oxygen_init: float = 0.21  # 21% O2
    glucose_init: float = 5.0  # 5 mM


class SpatialGrid:
    """
    3D spatial grid for cell simulation
    
    Manages:
    - Cell positions
    - Diffusible molecules (O2, glucose, cytokines)
    - Diffusion dynamics
    - Cell-environment interactions
    """
    
    def __init__(self, config: Optional[GridConfig] = None):
        if config is None:
            config = GridConfig()
        
        self.config = config
        self.size = config.size
        self.resolution = config.resolution
        
        # Physical dimensions (μm)
        self.dimensions = tuple(s * config.resolution for s in config.size)
        
        # Cell occupancy grid (cell IDs, -1 = empty)
        self.cells = np.full(config.size, -1, dtype=np.int32)
        
        # Diffusible fields
        self.oxygen = np.ones(config.size, dtype=np.float32) * config.oxygen_init
        self.glucose = np.ones(config.size, dtype=np.float32) * config.glucose_init
        self.growth_factors = np.zeros(config.size, dtype=np.float32)
        self.cytokines = np.zeros(config.size, dtype=np.float32)
        
        # Cell positions (for tracking)
        self.cell_positions = {}  # cell_id -> (x, y, z)
        
        print(f"Created spatial grid:")
        print(f"  Size: {config.size} voxels")
        print(f"  Resolution: {config.resolution} μm/voxel")
        print(f"  Physical size: {self.dimensions[0]:.0f} × {self.dimensions[1]:.0f} × {self.dimensions[2]:.0f} μm")
        print(f"  Volume: {self.dimensions[0] * self.dimensions[1] * self.dimensions[2] / 1e9:.2f} mm³")
    
    def add_cell(self, cell_id: int, position: Tuple[float, float, float]):
        """Add cell at physical position (μm)"""
        # Convert to voxel coordinates
        voxel = self.position_to_voxel(position)
        
        if self.is_valid_voxel(voxel):
            x, y, z = voxel
            self.cells[x, y, z] = cell_id
            self.cell_positions[cell_id] = position
    
    def remove_cell(self, cell_id: int):
        """Remove cell from grid"""
        if cell_id in self.cell_positions:
            position = self.cell_positions[cell_id]
            voxel = self.position_to_voxel(position)
            x, y, z = voxel
            self.cells[x, y, z] = -1
            del self.cell_positions[cell_id]
    
    def position_to_voxel(self, position: Tuple[float, float, float]) -> Tuple[int, int, int]:
        """Convert physical position (μm) to voxel indices"""
        return tuple(int(p / self.resolution) for p in position)
    
    def voxel_to_position(self, voxel: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Convert voxel indices to physical position (μm)"""
        return tuple((v + 0.5) * self.resolution for v in voxel)
    
    def is_valid_voxel(self, voxel: Tuple[int, int, int]) -> bool:
        """Check if voxel is within grid bounds"""
        return all(0 <= v < s for v, s in zip(voxel, self.size))
    
    def diffuse_field(self, field: np.ndarray, D: float, dt: float) -> np.ndarray:
        """
        Solve diffusion PDE: ∂C/∂t = D∇²C
        
        Uses explicit finite difference (3D Laplacian)
        """
        # Stability criterion: dt < dx²/(6D)
        # For D=2000 μm²/s, dx=10 μm: dt_max = 100/(6*2000) = 0.0083 s
        # Convert dt from hours to seconds
        dt_seconds = dt * 3600
        dx = self.resolution
        dt_stable = dx**2 / (6 * D)
        
        if dt_seconds > dt_stable:
            # Use stable time step
            dt_seconds = dt_stable * 0.5
        
        # 3D Laplacian using finite differences
        laplacian = (
            np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
            np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) +
            np.roll(field, 1, axis=2) + np.roll(field, -1, axis=2) -
            6 * field
        ) / (dx ** 2)
        
        # Update field
        field_new = field + D * laplacian * dt_seconds
        
        # Boundary conditions (Neumann - no flux)
        field_new[0, :, :] = field_new[1, :, :]
        field_new[-1, :, :] = field_new[-2, :, :]
        field_new[:, 0, :] = field_new[:, 1, :]
        field_new[:, -1, :] = field_new[:, -2, :]
        field_new[:, :, 0] = field_new[:, :, 1]
        field_new[:, :, -1] = field_new[:, :, -2]
        
        return field_new
    
    def step_diffusion(self, dt: float = 0.01):
        """Update all diffusible fields"""
        self.oxygen = self.diffuse_field(self.oxygen, self.config.D_oxygen, dt)
        self.glucose = self.diffuse_field(self.glucose, self.config.D_glucose, dt)
        self.cytokines = self.diffuse_field(self.cytokines, self.config.D_cytokine, dt)
        self.growth_factors = self.diffuse_field(self.growth_factors, self.config.D_cytokine, dt)
    
    def consume_at_position(self, position: Tuple[float, float, float], 
                           amount_o2: float, amount_glucose: float):
        """Cell consumes nutrients at position"""
        voxel = self.position_to_voxel(position)
        if self.is_valid_voxel(voxel):
            x, y, z = voxel
            self.oxygen[x, y, z] = max(0, self.oxygen[x, y, z] - amount_o2)
            self.glucose[x, y, z] = max(0, self.glucose[x, y, z] - amount_glucose)
    
    def secrete_at_position(self, position: Tuple[float, float, float],
                           cytokine_amount: float = 0.0,
                           growth_factor_amount: float = 0.0):
        """Cell secretes molecules at position"""
        voxel = self.position_to_voxel(position)
        if self.is_valid_voxel(voxel):
            x, y, z = voxel
            self.cytokines[x, y, z] += cytokine_amount
            self.growth_factors[x, y, z] += growth_factor_amount
    
    def get_concentration_at_position(self, position: Tuple[float, float, float],
                                     field: str = 'oxygen') -> float:
        """Get concentration of field at position"""
        voxel = self.position_to_voxel(position)
        if not self.is_valid_voxel(voxel):
            return 0.0
        
        x, y, z = voxel
        
        if field == 'oxygen':
            return self.oxygen[x, y, z]
        elif field == 'glucose':
            return self.glucose[x, y, z]
        elif field == 'cytokines':
            return self.cytokines[x, y, z]
        elif field == 'growth_factors':
            return self.growth_factors[x, y, z]
        else:
            return 0.0
    
    def get_statistics(self) -> dict:
        """Get grid statistics"""
        return {
            'n_cells': len(self.cell_positions),
            'oxygen_mean': float(np.mean(self.oxygen)),
            'oxygen_min': float(np.min(self.oxygen)),
            'oxygen_max': float(np.max(self.oxygen)),
            'glucose_mean': float(np.mean(self.glucose)),
            'glucose_min': float(np.min(self.glucose)),
            'cytokines_total': float(np.sum(self.cytokines)),
        }
    
    def plot_slice(self, z_slice: int, field: str = 'oxygen', save_path: Optional[str] = None):
        """Plot 2D slice of field"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if field == 'oxygen':
            data = self.oxygen[:, :, z_slice]
            title = f'Oxygen Concentration (z={z_slice})'
            cmap = 'viridis'
        elif field == 'glucose':
            data = self.glucose[:, :, z_slice]
            title = f'Glucose Concentration (z={z_slice})'
            cmap = 'plasma'
        elif field == 'cytokines':
            data = self.cytokines[:, :, z_slice]
            title = f'Cytokine Concentration (z={z_slice})'
            cmap = 'hot'
        else:
            data = self.cells[:, :, z_slice]
            title = f'Cell Occupancy (z={z_slice})'
            cmap = 'tab20'
        
        im = ax.imshow(data.T, origin='lower', cmap=cmap, interpolation='nearest')
        ax.set_xlabel('X (voxels)')
        ax.set_ylabel('Y (voxels)')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(field.capitalize())
        
        # Mark cell positions
        if field != 'cells':
            for cell_id, pos in self.cell_positions.items():
                voxel = self.position_to_voxel(pos)
                if voxel[2] == z_slice:
                    ax.plot(voxel[0], voxel[1], 'r.', markersize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        return fig
    
    def plot_3d_cells(self, save_path: Optional[str] = None):
        """Plot 3D visualization of cell positions"""
        if not self.cell_positions:
            print("No cells to plot")
            return None
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract positions
        positions = np.array(list(self.cell_positions.values()))
        
        # Plot cells
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                  c='red', marker='o', s=100, alpha=0.6)
        
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_zlabel('Z (μm)')
        ax.set_title(f'Cell Positions (n={len(self.cell_positions)})',
                    fontsize=14, fontweight='bold')
        
        # Set equal aspect ratio
        max_range = max(self.dimensions) / 2
        mid_x = self.dimensions[0] / 2
        mid_y = self.dimensions[1] / 2
        mid_z = self.dimensions[2] / 2
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        return fig
