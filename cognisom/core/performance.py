#!/usr/bin/env python3
"""
Performance Optimizations (CPU-based)
======================================

CPU-based optimizations for faster simulation without GPU.

Features:
- Vectorized operations (NumPy)
- Spatial indexing (KD-tree)
- Batch processing
- Caching
- Parallel processing (multiprocessing)
"""

import numpy as np
from scipy.spatial import cKDTree
from functools import lru_cache
from typing import List, Dict, Tuple
import multiprocessing as mp


class SpatialIndex:
    """
    Spatial indexing for fast neighbor queries
    Uses KD-tree for O(log n) lookups
    """
    
    def __init__(self):
        self.tree = None
        self.positions = None
        self.ids = None
    
    def build(self, positions: np.ndarray, ids: List[int]):
        """Build spatial index"""
        self.positions = np.array(positions)
        self.ids = np.array(ids)
        self.tree = cKDTree(self.positions)
    
    def query_radius(self, position: np.ndarray, radius: float) -> List[int]:
        """Find all objects within radius"""
        if self.tree is None:
            return []
        
        indices = self.tree.query_ball_point(position, radius)
        return [self.ids[i] for i in indices]
    
    def query_nearest(self, position: np.ndarray, k: int = 1) -> List[int]:
        """Find k nearest neighbors"""
        if self.tree is None:
            return []
        
        distances, indices = self.tree.query(position, k=k)
        if k == 1:
            return [self.ids[indices]]
        return [self.ids[i] for i in indices]


class VectorizedOperations:
    """
    Vectorized operations for batch processing
    """
    
    @staticmethod
    def batch_distance(positions1: np.ndarray, positions2: np.ndarray) -> np.ndarray:
        """
        Calculate pairwise distances between two sets of positions
        
        Args:
            positions1: (N, 3) array
            positions2: (M, 3) array
        
        Returns:
            (N, M) distance matrix
        """
        # Expand dimensions for broadcasting
        p1 = positions1[:, np.newaxis, :]  # (N, 1, 3)
        p2 = positions2[np.newaxis, :, :]  # (1, M, 3)
        
        # Calculate distances
        diff = p1 - p2  # (N, M, 3)
        distances = np.sqrt(np.sum(diff**2, axis=2))  # (N, M)
        
        return distances
    
    @staticmethod
    def batch_exponential_decay(distances: np.ndarray, 
                                decay_length: float,
                                strength: float = 1.0) -> np.ndarray:
        """
        Calculate exponential decay for all distances
        
        concentration = strength * exp(-distance / decay_length)
        """
        return strength * np.exp(-distances / decay_length)
    
    @staticmethod
    def batch_diffusion_step(concentration: np.ndarray,
                            diffusion_coeff: float,
                            dt: float,
                            dx: float) -> np.ndarray:
        """
        Vectorized diffusion step (3D Laplacian)
        
        ∂C/∂t = D∇²C
        """
        # Compute Laplacian using finite differences
        laplacian = np.zeros_like(concentration)
        
        # Interior points only
        laplacian[1:-1, 1:-1, 1:-1] = (
            concentration[2:, 1:-1, 1:-1] + concentration[:-2, 1:-1, 1:-1] +
            concentration[1:-1, 2:, 1:-1] + concentration[1:-1, :-2, 1:-1] +
            concentration[1:-1, 1:-1, 2:] + concentration[1:-1, 1:-1, :-2] -
            6 * concentration[1:-1, 1:-1, 1:-1]
        )
        
        # Update concentration
        dC = diffusion_coeff * laplacian * dt / (dx ** 2)
        return concentration + dC


class CacheManager:
    """
    Caching for expensive computations
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = {}
    
    @lru_cache(maxsize=1000)
    def cached_distance(self, pos1: Tuple[float, float, float],
                       pos2: Tuple[float, float, float]) -> float:
        """Cached distance calculation"""
        p1 = np.array(pos1)
        p2 = np.array(pos2)
        return np.linalg.norm(p1 - p2)
    
    @lru_cache(maxsize=1000)
    def cached_exponential(self, distance: float, decay_length: float) -> float:
        """Cached exponential decay"""
        return np.exp(-distance / decay_length)


class BatchProcessor:
    """
    Batch processing for multiple operations
    """
    
    @staticmethod
    def batch_cell_updates(cells: List, dt: float) -> None:
        """Update multiple cells in batch"""
        # Extract all cell data
        n_cells = len(cells)
        
        # Vectorized oxygen consumption
        oxygen_levels = np.array([c.oxygen for c in cells])
        glucose_levels = np.array([c.glucose for c in cells])
        
        # Batch metabolism calculation
        is_cancer = np.array([c.cell_type == 'cancer' for c in cells])
        
        # Oxygen consumption (vectorized)
        base_consumption = np.where(is_cancer, 0.02, 0.01)
        oxygen_consumed = base_consumption * dt
        oxygen_levels -= oxygen_consumed
        oxygen_levels = np.maximum(oxygen_levels, 0)
        
        # Glucose consumption (vectorized)
        glucose_consumed = base_consumption * 0.5 * dt
        glucose_levels -= glucose_consumed
        glucose_levels = np.maximum(glucose_levels, 0)
        
        # Update cells
        for i, cell in enumerate(cells):
            cell.oxygen = oxygen_levels[i]
            cell.glucose = glucose_levels[i]
    
    @staticmethod
    def batch_immune_patrol(immune_cells: List, dt: float) -> None:
        """Update immune cell positions in batch"""
        n_cells = len(immune_cells)
        
        # Extract positions
        positions = np.array([ic.position for ic in immune_cells])
        
        # Random movement (vectorized)
        speed = 10.0  # μm/hour
        directions = np.random.randn(n_cells, 3)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        
        # Update positions
        positions += directions * speed * dt
        
        # Boundary conditions
        positions = np.clip(positions, [0, 0, 0], [200, 200, 100])
        
        # Update immune cells
        for i, ic in enumerate(immune_cells):
            ic.position = positions[i]


class ParallelProcessor:
    """
    Parallel processing using multiprocessing
    """
    
    @staticmethod
    def parallel_cell_updates(cells: List, dt: float, n_workers: int = 4):
        """Update cells in parallel"""
        # Split cells into chunks
        chunk_size = len(cells) // n_workers
        chunks = [cells[i:i+chunk_size] for i in range(0, len(cells), chunk_size)]
        
        # Process in parallel
        with mp.Pool(n_workers) as pool:
            results = pool.starmap(_update_cell_chunk, 
                                  [(chunk, dt) for chunk in chunks])
        
        return results


def _update_cell_chunk(cells: List, dt: float):
    """Helper function for parallel processing"""
    for cell in cells:
        # Update cell
        cell.oxygen -= 0.01 * dt
        cell.glucose -= 0.005 * dt
    return cells


class PerformanceMonitor:
    """
    Monitor and report performance metrics
    """
    
    def __init__(self):
        self.timings = {}
        self.counts = {}
    
    def record(self, operation: str, time: float):
        """Record operation timing"""
        if operation not in self.timings:
            self.timings[operation] = []
            self.counts[operation] = 0
        
        self.timings[operation].append(time)
        self.counts[operation] += 1
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics"""
        stats = {}
        
        for op, times in self.timings.items():
            stats[op] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'total': np.sum(times),
                'count': self.counts[op]
            }
        
        return stats
    
    def print_report(self):
        """Print performance report"""
        print("\nPerformance Report:")
        print("=" * 70)
        
        stats = self.get_stats()
        
        for op, metrics in stats.items():
            print(f"\n{op}:")
            print(f"  Mean: {metrics['mean']*1000:.2f}ms")
            print(f"  Std:  {metrics['std']*1000:.2f}ms")
            print(f"  Min:  {metrics['min']*1000:.2f}ms")
            print(f"  Max:  {metrics['max']*1000:.2f}ms")
            print(f"  Total: {metrics['total']:.2f}s")
            print(f"  Count: {metrics['count']}")


# Test
if __name__ == '__main__':
    print("=" * 70)
    print("Performance Optimizations Test")
    print("=" * 70)
    print()
    
    # Test spatial indexing
    print("Testing Spatial Index...")
    positions = np.random.rand(1000, 3) * 200
    ids = list(range(1000))
    
    index = SpatialIndex()
    index.build(positions, ids)
    
    query_pos = np.array([100, 100, 50])
    neighbors = index.query_radius(query_pos, radius=50)
    print(f"  Found {len(neighbors)} neighbors within 50μm")
    
    nearest = index.query_nearest(query_pos, k=5)
    print(f"  Found {len(nearest)} nearest neighbors")
    print()
    
    # Test vectorized operations
    print("Testing Vectorized Operations...")
    pos1 = np.random.rand(100, 3) * 200
    pos2 = np.random.rand(50, 3) * 200
    
    distances = VectorizedOperations.batch_distance(pos1, pos2)
    print(f"  Calculated {distances.shape} distance matrix")
    
    concentrations = VectorizedOperations.batch_exponential_decay(distances, 50.0)
    print(f"  Calculated {concentrations.shape} concentration matrix")
    print()
    
    # Test diffusion
    print("Testing Vectorized Diffusion...")
    field = np.random.rand(20, 20, 10)
    field_new = VectorizedOperations.batch_diffusion_step(field, 100.0, 0.01, 10.0)
    print(f"  Diffusion step completed: {field_new.shape}")
    print()
    
    # Test performance monitor
    print("Testing Performance Monitor...")
    monitor = PerformanceMonitor()
    
    import time
    for i in range(10):
        start = time.time()
        _ = np.random.rand(1000, 3)
        monitor.record('random_generation', time.time() - start)
    
    monitor.print_report()
    
    print()
    print("=" * 70)
    print("✓ Performance optimizations working!")
    print("=" * 70)
