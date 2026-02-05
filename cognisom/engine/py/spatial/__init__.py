"""
Spatial Module
==============

3D spatial grid for cell positions and diffusion.

Provides:
- SpatialGrid: 3D environment
- Diffusion solvers
- Cell-environment interactions
"""

from .grid import SpatialGrid

__all__ = ['SpatialGrid']
