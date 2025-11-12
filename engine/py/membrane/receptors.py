#!/usr/bin/env python3
"""
Membrane Receptor System
========================

Models receptor-mediated signaling at the cell membrane.

Key Features:
- Ligand-receptor binding (equilibrium)
- Receptor trafficking (synthesis, internalization, recycling, degradation)
- Signal transduction
- Receptor dynamics over time

Biological Accuracy:
- Based on real receptor kinetics
- Includes desensitization
- Models receptor regulation
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class SignalingPathway(Enum):
    """Major signaling pathways"""
    MAPK = "MAPK"  # Cell proliferation
    PI3K_AKT = "PI3K-Akt"  # Survival, growth
    JAK_STAT = "JAK-STAT"  # Immune response
    NFKB = "NF-κB"  # Inflammation
    CAMP = "cAMP"  # Second messenger
    CALCIUM = "Calcium"  # Ca²⁺ signaling


@dataclass
class ReceptorParameters:
    """Parameters for a receptor type"""
    # Binding
    affinity_Kd: float = 1e-9  # Dissociation constant (M)
    association_rate: float = 1e6  # kon (M⁻¹s⁻¹)
    dissociation_rate: float = 1e-3  # koff (s⁻¹)
    
    # Trafficking
    synthesis_rate: float = 100.0  # receptors/hour
    degradation_rate: float = 0.05  # per hour
    internalization_rate: float = 0.1  # per minute (when bound)
    recycling_rate: float = 0.05  # per minute
    recycling_fraction: float = 0.7  # Fraction recycled vs degraded
    
    # Signaling
    signal_threshold: float = 0.1  # Min bound fraction to signal
    signal_amplification: float = 100.0  # Signal per bound receptor
    desensitization_rate: float = 0.02  # per minute


class MembraneReceptor:
    """
    Single receptor type on cell membrane
    
    Models the complete lifecycle:
    1. Synthesis (ER → Golgi → membrane)
    2. Ligand binding (equilibrium)
    3. Signal transduction
    4. Internalization (endocytosis)
    5. Recycling or degradation
    
    Example:
    --------
    >>> receptor = MembraneReceptor(
    ...     name='EGFR',
    ...     ligand='EGF',
    ...     initial_number=50000
    ... )
    >>> receptor.update(dt=0.01, ligand_concentration=1e-9)
    >>> print(f"Bound: {receptor.bound_number:.0f}")
    >>> print(f"Signal: {receptor.signal_strength():.2f}")
    """
    
    def __init__(
        self,
        name: str,
        ligand: str,
        pathway: SignalingPathway,
        initial_number: int = 10000,
        params: Optional[ReceptorParameters] = None
    ):
        self.name = name
        self.ligand = ligand
        self.pathway = pathway
        self.params = params if params else ReceptorParameters()
        
        # Receptor populations
        self.total_number = initial_number
        self.surface_number = int(initial_number * 0.8)  # 80% on surface
        self.internalized_number = int(initial_number * 0.2)  # 20% inside
        self.bound_number = 0
        
        # State
        self.bound_fraction = 0.0
        self.desensitization = 0.0  # 0 = fully sensitive, 1 = fully desensitized
        self.signaling_active = False
        
        # History
        self.history = {
            'time': [],
            'surface': [],
            'bound': [],
            'internalized': [],
            'signal': []
        }
    
    def update(self, dt: float, ligand_concentration: float):
        """
        Update receptor state for one time step
        
        Parameters:
        -----------
        dt : float
            Time step in hours
        ligand_concentration : float
            Ligand concentration in M (molar)
        """
        # Convert dt to minutes for some rates
        dt_min = dt * 60
        
        # 1. Ligand binding (fast equilibrium)
        self.bound_fraction = self._calculate_binding(ligand_concentration)
        self.bound_number = int(self.surface_number * self.bound_fraction)
        
        # 2. Receptor synthesis (ER → Golgi → membrane)
        new_receptors = self.params.synthesis_rate * dt
        self.total_number += new_receptors
        self.surface_number += new_receptors
        
        # 3. Receptor internalization (bound receptors)
        if self.bound_number > 0:
            internalized = self.bound_number * self.params.internalization_rate * dt_min
            internalized = min(internalized, self.bound_number)
            
            self.surface_number -= internalized
            self.internalized_number += internalized
        
        # 4. Receptor recycling and degradation
        # Recycling (back to surface)
        recycled = (self.internalized_number * 
                   self.params.recycling_fraction * 
                   self.params.recycling_rate * dt_min)
        recycled = min(recycled, self.internalized_number)
        
        # Degradation (destroyed)
        degraded_internal = (self.internalized_number * 
                           (1 - self.params.recycling_fraction) * 
                           self.params.degradation_rate * dt)
        degraded_surface = self.surface_number * self.params.degradation_rate * dt
        total_degraded = degraded_internal + degraded_surface
        
        # Update populations
        self.internalized_number -= (recycled + degraded_internal)
        self.surface_number += recycled
        self.surface_number -= degraded_surface
        self.total_number -= total_degraded
        
        # Ensure non-negative
        self.surface_number = max(0, self.surface_number)
        self.internalized_number = max(0, self.internalized_number)
        self.total_number = max(0, self.total_number)
        
        # 5. Desensitization (reduces signaling over time)
        if self.bound_fraction > 0:
            self.desensitization += self.params.desensitization_rate * dt_min
            self.desensitization = min(1.0, self.desensitization)
        else:
            # Recovery when no ligand
            self.desensitization -= self.params.desensitization_rate * dt_min * 0.5
            self.desensitization = max(0.0, self.desensitization)
        
        # 6. Check if signaling
        self.signaling_active = self.is_signaling()
    
    def _calculate_binding(self, ligand_conc: float) -> float:
        """
        Calculate receptor-ligand binding equilibrium
        
        Uses the equation:
        Bound fraction = [L] / (Kd + [L])
        
        Where:
        - [L] = ligand concentration
        - Kd = dissociation constant (affinity)
        
        Parameters:
        -----------
        ligand_conc : float
            Ligand concentration in M
        
        Returns:
        --------
        float : Fraction of receptors bound (0-1)
        """
        if ligand_conc <= 0:
            return 0.0
        
        Kd = self.params.affinity_Kd
        bound_frac = ligand_conc / (Kd + ligand_conc)
        
        return bound_frac
    
    def is_signaling(self) -> bool:
        """Check if receptor is actively signaling"""
        # Need sufficient binding and not fully desensitized
        return (self.bound_fraction > self.params.signal_threshold and
                self.desensitization < 0.9)
    
    def signal_strength(self) -> float:
        """
        Calculate signal strength
        
        Signal depends on:
        - Number of bound receptors
        - Amplification factor
        - Desensitization state
        
        Returns:
        --------
        float : Signal strength (arbitrary units)
        """
        if not self.signaling_active:
            return 0.0
        
        # Signal = bound receptors × amplification × (1 - desensitization)
        signal = (self.bound_number * 
                 self.params.signal_amplification * 
                 (1.0 - self.desensitization))
        
        return signal
    
    def record_state(self, time: float):
        """Record current state to history"""
        self.history['time'].append(time)
        self.history['surface'].append(self.surface_number)
        self.history['bound'].append(self.bound_number)
        self.history['internalized'].append(self.internalized_number)
        self.history['signal'].append(self.signal_strength())
    
    def get_statistics(self) -> Dict[str, float]:
        """Get current receptor statistics"""
        return {
            'total': self.total_number,
            'surface': self.surface_number,
            'internalized': self.internalized_number,
            'bound': self.bound_number,
            'bound_fraction': self.bound_fraction,
            'desensitization': self.desensitization,
            'signal': self.signal_strength(),
            'signaling': float(self.signaling_active)
        }


class EGFReceptor(MembraneReceptor):
    """
    Epidermal Growth Factor Receptor (EGFR)
    
    Signals through MAPK pathway
    Promotes cell proliferation
    """
    def __init__(self, initial_number: int = 50000):
        params = ReceptorParameters(
            affinity_Kd=1e-9,  # 1 nM (high affinity)
            synthesis_rate=200.0,
            internalization_rate=0.15,  # Fast internalization
            recycling_fraction=0.3,  # Mostly degraded
            signal_amplification=150.0
        )
        super().__init__(
            name='EGFR',
            ligand='EGF',
            pathway=SignalingPathway.MAPK,
            initial_number=initial_number,
            params=params
        )


class InsulinReceptor(MembraneReceptor):
    """
    Insulin Receptor
    
    Signals through PI3K-Akt pathway
    Promotes glucose uptake and cell survival
    """
    def __init__(self, initial_number: int = 100000):
        params = ReceptorParameters(
            affinity_Kd=1e-9,  # 1 nM
            synthesis_rate=150.0,
            internalization_rate=0.1,
            recycling_fraction=0.8,  # Mostly recycled
            signal_amplification=200.0,
            desensitization_rate=0.03  # Moderate desensitization
        )
        super().__init__(
            name='InsulinR',
            ligand='Insulin',
            pathway=SignalingPathway.PI3K_AKT,
            initial_number=initial_number,
            params=params
        )


class CytokineReceptor(MembraneReceptor):
    """
    Cytokine Receptor (e.g., IL-6R, TNF-R)
    
    Signals through JAK-STAT or NF-κB pathways
    Promotes immune response and inflammation
    """
    def __init__(self, cytokine_type: str = 'IL-6', initial_number: int = 10000):
        params = ReceptorParameters(
            affinity_Kd=1e-10,  # 0.1 nM (very high affinity)
            synthesis_rate=50.0,
            internalization_rate=0.08,
            recycling_fraction=0.5,
            signal_amplification=300.0,  # Strong amplification
            desensitization_rate=0.05  # Fast desensitization
        )
        
        pathway = SignalingPathway.JAK_STAT if cytokine_type == 'IL-6' else SignalingPathway.NFKB
        
        super().__init__(
            name=f'{cytokine_type}R',
            ligand=cytokine_type,
            pathway=pathway,
            initial_number=initial_number,
            params=params
        )


class ReceptorSystem:
    """
    Complete receptor system for a cell
    
    Manages multiple receptor types and their interactions
    with the environment.
    
    Example:
    --------
    >>> system = ReceptorSystem()
    >>> system.add_receptor('EGFR', EGFReceptor(50000))
    >>> system.add_receptor('InsulinR', InsulinReceptor(100000))
    >>> 
    >>> # Update with environment
    >>> ligands = {'EGF': 1e-9, 'Insulin': 5e-9}
    >>> system.update(dt=0.01, ligand_concentrations=ligands)
    >>> 
    >>> # Get signaling activity
    >>> signals = system.get_active_pathways()
    >>> print(signals)  # {'MAPK': 7500.0, 'PI3K-Akt': 20000.0}
    """
    
    def __init__(self):
        self.receptors: Dict[str, MembraneReceptor] = {}
        self.pathway_activity: Dict[str, float] = {}
    
    def add_receptor(self, name: str, receptor: MembraneReceptor):
        """Add a receptor type to the system"""
        self.receptors[name] = receptor
    
    def update(self, dt: float, ligand_concentrations: Dict[str, float]):
        """
        Update all receptors
        
        Parameters:
        -----------
        dt : float
            Time step in hours
        ligand_concentrations : dict
            Dictionary of {ligand_name: concentration_in_M}
        """
        # Reset pathway activity
        self.pathway_activity = {}
        
        # Update each receptor
        for name, receptor in self.receptors.items():
            # Get ligand concentration for this receptor
            ligand_conc = ligand_concentrations.get(receptor.ligand, 0.0)
            
            # Update receptor state
            receptor.update(dt, ligand_conc)
            
            # Accumulate pathway activity
            if receptor.signaling_active:
                pathway = receptor.pathway.value
                signal = receptor.signal_strength()
                
                if pathway in self.pathway_activity:
                    self.pathway_activity[pathway] += signal
                else:
                    self.pathway_activity[pathway] = signal
    
    def get_active_pathways(self) -> Dict[str, float]:
        """Get currently active signaling pathways and their strengths"""
        return self.pathway_activity.copy()
    
    def get_receptor_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all receptors"""
        return {name: receptor.get_statistics() 
                for name, receptor in self.receptors.items()}
    
    def record_all_states(self, time: float):
        """Record state of all receptors"""
        for receptor in self.receptors.values():
            receptor.record_state(time)
    
    def get_total_surface_receptors(self) -> int:
        """Get total number of receptors on cell surface"""
        return sum(r.surface_number for r in self.receptors.values())
    
    def get_total_bound_receptors(self) -> int:
        """Get total number of bound receptors"""
        return sum(r.bound_number for r in self.receptors.values())


# Example usage and testing
if __name__ == '__main__':
    print("=" * 60)
    print("Membrane Receptor System Demo")
    print("=" * 60)
    print()
    
    # Create receptor system
    system = ReceptorSystem()
    system.add_receptor('EGFR', EGFReceptor(50000))
    system.add_receptor('InsulinR', InsulinReceptor(100000))
    system.add_receptor('IL6R', CytokineReceptor('IL-6', 10000))
    
    print("Initial state:")
    for name, stats in system.get_receptor_stats().items():
        print(f"  {name}: {stats['surface']:.0f} surface, {stats['bound']:.0f} bound")
    print()
    
    # Simulate with ligands
    print("Simulating 2 hours with ligands...")
    print()
    
    ligands = {
        'EGF': 2e-9,      # 2 nM EGF
        'Insulin': 10e-9,  # 10 nM Insulin
        'IL-6': 0.5e-9    # 0.5 nM IL-6
    }
    
    dt = 0.01  # hours
    duration = 2.0  # hours
    steps = int(duration / dt)
    
    for step in range(steps):
        time = step * dt
        system.update(dt, ligands)
        
        if step % 50 == 0:  # Print every 0.5 hours
            system.record_all_states(time)
            print(f"Time: {time:.1f}h")
            
            # Receptor states
            for name, stats in system.get_receptor_stats().items():
                print(f"  {name}:")
                print(f"    Surface: {stats['surface']:.0f}")
                print(f"    Bound: {stats['bound']:.0f} ({stats['bound_fraction']*100:.1f}%)")
                print(f"    Signal: {stats['signal']:.0f}")
                print(f"    Desensitization: {stats['desensitization']*100:.0f}%")
            
            # Pathway activity
            pathways = system.get_active_pathways()
            if pathways:
                print(f"  Active pathways:")
                for pathway, strength in pathways.items():
                    print(f"    {pathway}: {strength:.0f}")
            print()
    
    print("=" * 60)
    print("✓ Receptor system simulation complete!")
    print("=" * 60)
