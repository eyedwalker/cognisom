"""
Simulation class - Orchestrates multi-cell simulations
"""

import numpy as np
from typing import List, Dict, Optional
import time
from pathlib import Path
import json

from .cell import Cell


class Simulation:
    """
    Multi-cell simulation orchestrator
    
    Manages:
    - Cell population
    - Time stepping
    - Event tracking (divisions, deaths)
    - Data collection
    """
    
    def __init__(
        self,
        initial_cells: List[Cell],
        duration: float = 24.0,
        dt: float = 0.01,
        max_cells: int = 100000,
        output_dir: Optional[str] = None
    ):
        """
        Initialize simulation
        
        Args:
            initial_cells: List of initial cells
            duration: Simulation duration in hours
            dt: Time step in hours
            max_cells: Maximum number of cells (safety limit)
            output_dir: Directory for output files
        """
        self.cells = initial_cells
        self.duration = duration
        self.dt = dt
        self.max_cells = max_cells
        self.output_dir = Path(output_dir) if output_dir else Path('./output')
        
        # Tracking
        self.current_time = 0.0
        self.step_count = 0
        self.events = {
            'divisions': 0,
            'deaths': 0,
        }
        
        # History
        self.history = {
            'time': [],
            'cell_count': [],
            'total_proteins': [],
            'avg_mhc1': [],
            'avg_stress': [],
        }
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def time(self):
        """Current simulation time (alias for current_time)"""
        return self.current_time
    
    def step(self):
        """
        Perform a single simulation step
        
        Returns:
            dict: Step statistics
        """
        self.current_time += self.dt
        self.step_count += 1
        
        # Update all cells
        new_cells = []
        dead_cells = []
        
        for cell in self.cells:
            # Step the cell
            daughter = cell.step(self.dt)
            
            # Check for division
            if daughter is not None:
                new_cells.append(daughter)
                self.events['divisions'] += 1
            
            # Check for death
            if not cell.is_alive():
                dead_cells.append(cell)
                self.events['deaths'] += 1
        
        # Add new cells
        self.cells.extend(new_cells)
        
        # Remove dead cells
        for dead_cell in dead_cells:
            self.cells.remove(dead_cell)
        
        # Safety check
        if len(self.cells) > self.max_cells:
            print(f"\n⚠️  Max cell limit reached ({self.max_cells})")
        
        return {
            'time': self.current_time,
            'n_cells': len(self.cells),
            'divisions': self.events['divisions'],
            'deaths': self.events['deaths']
        }
    
    def run(self, verbose: bool = True, save_interval: int = 100):
        """
        Run the simulation
        
        Args:
            verbose: Print progress
            save_interval: Save data every N steps
        """
        start_time = time.time()
        total_steps = int(self.duration / self.dt)
        
        if verbose:
            print(f"Starting simulation: {len(self.cells)} initial cells")
            print(f"Duration: {self.duration}h, dt: {self.dt}h, steps: {total_steps}")
            print("-" * 60)
        
        for step in range(total_steps):
            self.current_time = step * self.dt
            self.step_count = step
            
            # Update all cells
            new_cells = []
            dead_cells = []
            
            for cell in self.cells:
                # Step the cell
                daughter = cell.step(self.dt)
                
                # Check for division
                if daughter is not None:
                    new_cells.append(daughter)
                    self.events['divisions'] += 1
                
                # Check for death
                if not cell.is_alive():
                    dead_cells.append(cell)
                    self.events['deaths'] += 1
            
            # Add new cells
            self.cells.extend(new_cells)
            
            # Remove dead cells
            for dead_cell in dead_cells:
                self.cells.remove(dead_cell)
            
            # Safety check
            if len(self.cells) > self.max_cells:
                if verbose:
                    print(f"\n⚠️  Max cell limit reached ({self.max_cells})")
                break
            
            # Record history
            if step % save_interval == 0:
                self._record_state()
                
                if verbose and step % (save_interval * 10) == 0:
                    elapsed = time.time() - start_time
                    progress = (step / total_steps) * 100
                    print(f"t={self.current_time:6.1f}h | "
                          f"Cells: {len(self.cells):6d} | "
                          f"Divisions: {self.events['divisions']:4d} | "
                          f"Deaths: {self.events['deaths']:3d} | "
                          f"{progress:5.1f}% | "
                          f"{elapsed:.1f}s")
        
        # Final record
        self._record_state()
        
        elapsed = time.time() - start_time
        if verbose:
            print("-" * 60)
            print(f"✓ Simulation complete in {elapsed:.2f}s")
            print(f"  Final cells: {len(self.cells)}")
            print(f"  Total divisions: {self.events['divisions']}")
            print(f"  Total deaths: {self.events['deaths']}")
            print(f"  Steps/second: {total_steps/elapsed:.0f}")
    
    def _record_state(self):
        """Record current state to history"""
        if len(self.cells) == 0:
            return
        
        self.history['time'].append(self.current_time)
        self.history['cell_count'].append(len(self.cells))
        
        # Aggregate statistics
        total_proteins = sum(c.state.species_counts[1] for c in self.cells)
        avg_mhc1 = np.mean([c.state.mhc1_expression for c in self.cells])
        avg_stress = np.mean([c.state.stress_level for c in self.cells])
        
        self.history['total_proteins'].append(total_proteins)
        self.history['avg_mhc1'].append(avg_mhc1)
        self.history['avg_stress'].append(avg_stress)
    
    def get_results(self) -> Dict:
        """Get simulation results"""
        return {
            'history': self.history,
            'events': self.events,
            'final_cells': len(self.cells),
            'duration': self.duration,
            'dt': self.dt,
        }
    
    def save_results(self, filename: str = 'results.json'):
        """Save results to JSON file"""
        results = self.get_results()
        
        # Convert numpy types to native Python types for JSON
        def convert_to_native(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj
        
        results = convert_to_native(results)
        
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Results saved: {output_path}")
        return output_path
    
    def plot_results(self, save: bool = True):
        """Plot simulation results"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("⚠️  matplotlib not installed. Run: pip install matplotlib")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Cell count over time
        ax = axes[0, 0]
        ax.plot(self.history['time'], self.history['cell_count'], 'o-', markersize=3)
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Cell Count')
        ax.set_title('Population Growth')
        ax.grid(True, alpha=0.3)
        
        # Total proteins
        ax = axes[0, 1]
        ax.plot(self.history['time'], self.history['total_proteins'], 'o-', 
                markersize=3, color='green')
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Total Proteins')
        ax.set_title('Total Protein Content')
        ax.grid(True, alpha=0.3)
        
        # Average MHC-I expression
        ax = axes[1, 0]
        ax.plot(self.history['time'], self.history['avg_mhc1'], 'o-', 
                markersize=3, color='blue')
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('MHC-I Expression')
        ax.set_title('Average MHC-I Surface Expression')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3)
        
        # Average stress
        ax = axes[1, 1]
        ax.plot(self.history['time'], self.history['avg_stress'], 'o-', 
                markersize=3, color='red')
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Stress Level')
        ax.set_title('Average Cellular Stress')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / 'simulation_results.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Plot saved: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def get_doubling_time(self) -> Optional[float]:
        """Calculate population doubling time"""
        if len(self.history['cell_count']) < 2:
            return None
        
        # Find when population doubles from initial
        initial_count = self.history['cell_count'][0]
        target_count = initial_count * 2
        
        for i, count in enumerate(self.history['cell_count']):
            if count >= target_count:
                return self.history['time'][i]
        
        return None
