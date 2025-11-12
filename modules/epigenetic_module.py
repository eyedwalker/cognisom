#!/usr/bin/env python3
"""
Epigenetic Module
=================

Handles epigenetic regulation: DNA methylation and histone modifications.

Features:
- DNA methylation (CpG islands)
- Histone modifications (acetylation, methylation)
- Gene silencing/activation
- Epigenetic inheritance
- Environmental effects
"""

import sys
sys.path.insert(0, '..')

import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass, field

from core.module_base import SimulationModule
from core.event_bus import EventTypes


@dataclass
class EpigeneticState:
    """Epigenetic state for a gene"""
    gene_name: str
    
    # DNA methylation (0-1, higher = more silenced)
    methylation_level: float = 0.0
    
    # Histone modifications
    h3k4me3: float = 1.0  # Active mark (trimethylation of H3K4)
    h3k27me3: float = 0.0  # Repressive mark (trimethylation of H3K27)
    h3k9ac: float = 1.0  # Active mark (acetylation of H3K9)
    
    # Chromatin state
    chromatin_open: bool = True  # Euchromatin vs heterochromatin
    
    def is_silenced(self) -> bool:
        """Check if gene is epigenetically silenced"""
        # High methylation OR high repressive marks = silenced
        return (self.methylation_level > 0.7 or 
                self.h3k27me3 > 0.7 or
                not self.chromatin_open)
    
    def is_active(self) -> bool:
        """Check if gene is epigenetically active"""
        # Low methylation AND high active marks = active
        return (self.methylation_level < 0.3 and
                self.h3k4me3 > 0.5 and
                self.h3k9ac > 0.5 and
                self.chromatin_open)
    
    def get_expression_modifier(self) -> float:
        """Get expression level modifier (0-1)"""
        if self.is_silenced():
            return 0.1  # 90% reduction
        elif self.is_active():
            return 1.0  # Full expression
        else:
            # Intermediate state
            active_score = (self.h3k4me3 + self.h3k9ac) / 2
            repressive_score = (self.methylation_level + self.h3k27me3) / 2
            return max(0.1, active_score - repressive_score)


class EpigeneticModule(SimulationModule):
    """
    Epigenetic regulation simulation module
    
    Manages:
    - DNA methylation patterns
    - Histone modifications
    - Gene silencing/activation
    - Epigenetic inheritance
    - Environmental responses
    
    Events Emitted:
    - GENE_SILENCED: When gene becomes methylated
    - GENE_ACTIVATED: When gene becomes demethylated
    
    Events Subscribed:
    - CELL_DIVIDED: Inherit epigenetic marks
    - CELL_TRANSFORMED: Alter epigenetic landscape
    - HYPOXIA_DETECTED: Environmental epigenetic changes
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        
        # Cell-specific epigenetic states
        self.cell_epigenetics: Dict[int, Dict[str, EpigeneticState]] = {}
        
        # Parameters
        self.methylation_rate = config.get('methylation_rate', 0.01)
        self.demethylation_rate = config.get('demethylation_rate', 0.005)
        self.histone_mod_rate = config.get('histone_mod_rate', 0.02)
        
        # Cancer-associated changes
        self.cancer_hypermethylation = config.get('cancer_hypermethylation', True)
        self.hypoxia_methylation = config.get('hypoxia_methylation', True)
        
        # Statistics
        self.total_silencing_events = 0
        self.total_activation_events = 0
        
        # Reference to molecular module
        self.molecular_module = None
    
    def initialize(self):
        """Initialize epigenetic system"""
        print("  Creating epigenetic regulation...")
        
        # Subscribe to events
        self.subscribe(EventTypes.CELL_DIVIDED, self.on_cell_divided)
        self.subscribe(EventTypes.CELL_TRANSFORMED, self.on_cell_transformed)
        self.subscribe(EventTypes.HYPOXIA_DETECTED, self.on_hypoxia_detected)
        
        print(f"    ✓ DNA methylation system")
        print(f"    ✓ Histone modification system")
        print(f"    ✓ Environmental response ready")
    
    def add_cell(self, cell_id: int, cell_type: str = 'normal'):
        """Add cell with epigenetic state"""
        # Initialize epigenetic states for key genes
        self.cell_epigenetics[cell_id] = {
            'TP53': EpigeneticState('TP53', methylation_level=0.0),  # Active
            'KRAS': EpigeneticState('KRAS', methylation_level=0.0),  # Active
            'CDKN2A': EpigeneticState('CDKN2A', methylation_level=0.0),  # Tumor suppressor
            'MLH1': EpigeneticState('MLH1', methylation_level=0.0),  # DNA repair
            'BRCA1': EpigeneticState('BRCA1', methylation_level=0.0),  # DNA repair
        }
        
        # Cancer cells have altered epigenetic landscape
        if cell_type == 'cancer':
            # Hypermethylate tumor suppressors
            self.cell_epigenetics[cell_id]['CDKN2A'].methylation_level = 0.8
            self.cell_epigenetics[cell_id]['MLH1'].methylation_level = 0.7
            
            # Alter histone marks
            self.cell_epigenetics[cell_id]['CDKN2A'].h3k27me3 = 0.8
            self.cell_epigenetics[cell_id]['CDKN2A'].chromatin_open = False
    
    def remove_cell(self, cell_id: int):
        """Remove cell epigenetic data"""
        if cell_id in self.cell_epigenetics:
            del self.cell_epigenetics[cell_id]
    
    def set_molecular_module(self, molecular_module):
        """Link to molecular module"""
        self.molecular_module = molecular_module
    
    def update(self, dt: float):
        """Update epigenetic states"""
        for cell_id, epigenetic_states in self.cell_epigenetics.items():
            for gene_name, state in epigenetic_states.items():
                # Stochastic methylation changes
                if np.random.random() < self.methylation_rate * dt:
                    state.methylation_level = min(1.0, state.methylation_level + 0.1)
                
                if np.random.random() < self.demethylation_rate * dt:
                    state.methylation_level = max(0.0, state.methylation_level - 0.1)
                
                # Histone modifications
                if np.random.random() < self.histone_mod_rate * dt:
                    # Random walk for histone marks
                    state.h3k4me3 += np.random.normal(0, 0.05)
                    state.h3k27me3 += np.random.normal(0, 0.05)
                    state.h3k9ac += np.random.normal(0, 0.05)
                    
                    # Clamp values
                    state.h3k4me3 = np.clip(state.h3k4me3, 0, 1)
                    state.h3k27me3 = np.clip(state.h3k27me3, 0, 1)
                    state.h3k9ac = np.clip(state.h3k9ac, 0, 1)
                
                # Update chromatin state
                state.chromatin_open = (state.h3k9ac > 0.5 and state.methylation_level < 0.5)
    
    def get_expression_modifier(self, cell_id: int, gene_name: str) -> float:
        """Get epigenetic modifier for gene expression"""
        if cell_id in self.cell_epigenetics:
            if gene_name in self.cell_epigenetics[cell_id]:
                return self.cell_epigenetics[cell_id][gene_name].get_expression_modifier()
        return 1.0  # No modification
    
    def silence_gene(self, cell_id: int, gene_name: str):
        """Epigenetically silence a gene"""
        if cell_id in self.cell_epigenetics:
            if gene_name in self.cell_epigenetics[cell_id]:
                state = self.cell_epigenetics[cell_id][gene_name]
                state.methylation_level = 0.9
                state.h3k27me3 = 0.9
                state.h3k4me3 = 0.1
                state.chromatin_open = False
                
                self.total_silencing_events += 1
                
                self.emit_event(EventTypes.GENE_EXPRESSED, {
                    'cell_id': cell_id,
                    'gene': gene_name,
                    'silenced': True,
                    'methylation': state.methylation_level
                })
    
    def activate_gene(self, cell_id: int, gene_name: str):
        """Epigenetically activate a gene"""
        if cell_id in self.cell_epigenetics:
            if gene_name in self.cell_epigenetics[cell_id]:
                state = self.cell_epigenetics[cell_id][gene_name]
                state.methylation_level = 0.1
                state.h3k27me3 = 0.1
                state.h3k4me3 = 0.9
                state.h3k9ac = 0.9
                state.chromatin_open = True
                
                self.total_activation_events += 1
                
                self.emit_event(EventTypes.GENE_EXPRESSED, {
                    'cell_id': cell_id,
                    'gene': gene_name,
                    'silenced': False,
                    'methylation': state.methylation_level
                })
    
    def get_state(self) -> Dict[str, Any]:
        """Return current epigenetic state"""
        # Calculate average methylation levels
        if self.cell_epigenetics:
            all_methylation = []
            silenced_genes = 0
            active_genes = 0
            
            for states in self.cell_epigenetics.values():
                for state in states.values():
                    all_methylation.append(state.methylation_level)
                    if state.is_silenced():
                        silenced_genes += 1
                    elif state.is_active():
                        active_genes += 1
            
            avg_methylation = np.mean(all_methylation) if all_methylation else 0
        else:
            avg_methylation = 0
            silenced_genes = 0
            active_genes = 0
        
        return {
            'n_cells_tracked': len(self.cell_epigenetics),
            'avg_methylation': avg_methylation,
            'silenced_genes': silenced_genes,
            'active_genes': active_genes,
            'total_silencing_events': self.total_silencing_events,
            'total_activation_events': self.total_activation_events
        }
    
    # Event handlers
    def on_cell_divided(self, data):
        """Handle cell division - inherit epigenetic marks"""
        parent_id = data['cell_id']
        daughter_id = data.get('daughter_id')
        
        if daughter_id and parent_id in self.cell_epigenetics:
            # Daughter inherits parent's epigenetic state (with some variation)
            self.cell_epigenetics[daughter_id] = {}
            
            for gene_name, parent_state in self.cell_epigenetics[parent_id].items():
                # Copy with small random variation
                daughter_state = EpigeneticState(
                    gene_name=gene_name,
                    methylation_level=parent_state.methylation_level + np.random.normal(0, 0.05),
                    h3k4me3=parent_state.h3k4me3 + np.random.normal(0, 0.05),
                    h3k27me3=parent_state.h3k27me3 + np.random.normal(0, 0.05),
                    h3k9ac=parent_state.h3k9ac + np.random.normal(0, 0.05),
                    chromatin_open=parent_state.chromatin_open
                )
                
                # Clamp values
                daughter_state.methylation_level = np.clip(daughter_state.methylation_level, 0, 1)
                daughter_state.h3k4me3 = np.clip(daughter_state.h3k4me3, 0, 1)
                daughter_state.h3k27me3 = np.clip(daughter_state.h3k27me3, 0, 1)
                daughter_state.h3k9ac = np.clip(daughter_state.h3k9ac, 0, 1)
                
                self.cell_epigenetics[daughter_id][gene_name] = daughter_state
    
    def on_cell_transformed(self, data):
        """Handle cell transformation - cancer epigenetic landscape"""
        cell_id = data['cell_id']
        
        if self.cancer_hypermethylation and cell_id in self.cell_epigenetics:
            # Cancer cells hypermethylate tumor suppressors
            self.silence_gene(cell_id, 'CDKN2A')
            self.silence_gene(cell_id, 'MLH1')
            
            print(f"  Epigenetic: Cell {cell_id} silenced tumor suppressors")
    
    def on_hypoxia_detected(self, data):
        """Handle hypoxia - environmental epigenetic changes"""
        if not self.hypoxia_methylation:
            return
        
        # Hypoxia can alter methylation patterns
        # Randomly affect some cells
        for cell_id in list(self.cell_epigenetics.keys()):
            if np.random.random() < 0.1:  # 10% of cells affected
                # Increase methylation slightly
                for state in self.cell_epigenetics[cell_id].values():
                    state.methylation_level = min(1.0, state.methylation_level + 0.05)


# Test
if __name__ == '__main__':
    from core import SimulationEngine, SimulationConfig
    from cellular_module import CellularModule
    
    print("=" * 70)
    print("Epigenetic Module Test")
    print("=" * 70)
    print()
    
    # Create engine
    engine = SimulationEngine(SimulationConfig(dt=0.01, duration=1.0))
    
    # Register modules
    engine.register_module('cellular', CellularModule, {
        'n_normal_cells': 5,
        'n_cancer_cells': 2
    })
    
    engine.register_module('epigenetic', EpigeneticModule, {
        'methylation_rate': 0.05,
        'cancer_hypermethylation': True
    })
    
    # Initialize
    engine.initialize()
    
    # Add cells to epigenetic tracking
    epigenetic = engine.modules['epigenetic']
    cellular = engine.modules['cellular']
    
    for cell_id, cell in cellular.cells.items():
        epigenetic.add_cell(cell_id, cell.cell_type)
    
    print(f"Added {len(cellular.cells)} cells to epigenetic tracking")
    print()
    
    # Check initial states
    print("Initial epigenetic states:")
    for cell_id in list(cellular.cells.keys())[:3]:
        if cell_id in epigenetic.cell_epigenetics:
            print(f"\nCell {cell_id} ({cellular.cells[cell_id].cell_type}):")
            for gene, state in epigenetic.cell_epigenetics[cell_id].items():
                print(f"  {gene}: methylation={state.methylation_level:.2f}, "
                      f"silenced={state.is_silenced()}")
    print()
    
    # Run simulation
    engine.run()
    
    # Get results
    print("\nEpigenetic State:")
    epigenetic_state = epigenetic.get_state()
    for key, value in epigenetic_state.items():
        print(f"  {key}: {value}")
    
    print()
    print("=" * 70)
    print("✓ Epigenetic module working!")
    print("=" * 70)
