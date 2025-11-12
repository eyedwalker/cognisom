#!/usr/bin/env python3
"""
Molecular Module
================

Handles DNA/RNA dynamics, exosome transfer, and molecular communication.

Features:
- DNA/RNA with actual sequences
- Gene transcription and translation
- Mutations (oncogenic and normal)
- Exosome packaging and transfer
- Cell-to-cell molecular communication
"""

import sys
sys.path.insert(0, '..')

import numpy as np
from typing import Dict, List, Any

from core.module_base import SimulationModule
from core.event_bus import EventTypes
from engine.py.molecular.nucleic_acids import Gene, RNA, NucleicAcidType
from engine.py.molecular.exosomes import ExosomeSystem


class MolecularModule(SimulationModule):
    """
    Molecular simulation module
    
    Manages:
    - Genes (DNA)
    - Transcription (DNA → RNA)
    - Translation (RNA → Protein)
    - Exosomes (cell-to-cell transfer)
    - Mutations
    
    Events Emitted:
    - EXOSOME_RELEASED: When cell releases exosome
    - EXOSOME_UPTAKEN: When cell uptakes exosome
    - MUTATION_OCCURRED: When mutation happens
    - GENE_EXPRESSED: When gene is transcribed
    
    Events Subscribed:
    - CELL_DIVIDED: Create genes for new cell
    - CELL_TRANSFORMED: Create oncogenic exosomes
    - CELL_DIED: Clean up molecular data
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        
        # Genes database (shared across cells)
        self.genes: Dict[str, Gene] = {}
        
        # Exosome system
        self.exosome_system = None
        
        # Cell-specific molecular data
        self.cell_genes: Dict[int, Dict[str, Gene]] = {}  # {cell_id: {gene_name: Gene}}
        self.cell_mrnas: Dict[int, List[RNA]] = {}  # {cell_id: [RNA]}
        
        # Parameters (adjustable)
        self.transcription_rate = config.get('transcription_rate', 1.0)
        self.translation_rate = config.get('translation_rate', 0.5)
        self.exosome_release_rate = config.get('exosome_release_rate', 0.1)
        self.mutation_rate = config.get('mutation_rate', 0.001)
        
        # Statistics
        self.total_exosomes_released = 0
        self.total_exosomes_uptaken = 0
        self.total_mutations = 0
        self.total_transcriptions = 0
    
    def initialize(self):
        """Initialize molecular system"""
        print("  Creating gene library...")
        
        # Create gene library
        self._create_gene_library()
        
        # Create exosome system
        self.exosome_system = ExosomeSystem()
        
        # Subscribe to cellular events
        self.subscribe(EventTypes.CELL_DIVIDED, self.on_cell_divided)
        self.subscribe(EventTypes.CELL_TRANSFORMED, self.on_cell_transformed)
        self.subscribe(EventTypes.CELL_DIED, self.on_cell_died)
        
        print(f"    ✓ {len(self.genes)} genes in library")
        print(f"    ✓ Exosome system ready")
    
    def _create_gene_library(self):
        """Create library of genes"""
        # KRAS (oncogene)
        kras_sequence = (
            "ATGGACTGAATATAAACTTGTGGTAGTTGGAGCTGGTGGCGTAGGCAAGAGTGCCTTGACGATACAGC"
            "TAATTCAGAATCATTTTGTGGACGAATATGATCCAACAATAGAGGATTCCTACAGGAAGCAAGTAGTA"
        )
        self.genes['KRAS'] = Gene('KRAS', kras_sequence, 'protein_coding')
        self.genes['KRAS'].is_oncogene = False  # Normal initially
        
        # TP53 (tumor suppressor)
        tp53_sequence = (
            "ATGGAGGAGCCGCAGTCAGATCCTAGCGTCGAGCCCCCTCTGAGTCAGGAAACATTTTCAGACCTATG"
            "GAAACTACTTCCTGAAAACAACGTTCTGTCCCCCTTGCCGTCCCAAGCAATGGATGATTTGATGCTGT"
        )
        self.genes['TP53'] = Gene('TP53', tp53_sequence, 'protein_coding')
        self.genes['TP53'].is_tumor_suppressor = True
        
        # BRAF (oncogene)
        braf_sequence = (
            "ATGAAGACCTCACAGTAAAAATAGGTGATTTTGGTCTAGCTACAGTGAAATCTCGATGGAGTGGGTCC"
            "CATCAGTTTGAACAGTTGTCTGGATCCATTTTGTGGATGGAGTTGGAGCTATTTTTCCACTGATTAAA"
        )
        self.genes['BRAF'] = Gene('BRAF', braf_sequence, 'protein_coding')
        self.genes['BRAF'].is_oncogene = False  # Normal initially
    
    def add_cell(self, cell_id: int):
        """
        Add cell to molecular tracking
        
        Parameters:
        -----------
        cell_id : int
            Cell identifier
        """
        # Copy genes for this cell
        self.cell_genes[cell_id] = {}
        for name, gene in self.genes.items():
            # Each cell gets its own copy
            cell_gene = Gene(gene.name, gene.dna.sequence, gene.gene_type)
            cell_gene.is_oncogene = gene.is_oncogene
            cell_gene.is_tumor_suppressor = gene.is_tumor_suppressor
            self.cell_genes[cell_id][name] = cell_gene
        
        self.cell_mrnas[cell_id] = []
    
    def remove_cell(self, cell_id: int):
        """Remove cell from tracking"""
        if cell_id in self.cell_genes:
            del self.cell_genes[cell_id]
        if cell_id in self.cell_mrnas:
            del self.cell_mrnas[cell_id]
    
    def update(self, dt: float):
        """Update molecular dynamics"""
        # Update exosomes (diffusion)
        self.exosome_system.update(dt)
        
        # Check for exosome uptake events
        for exosome in self.exosome_system.exosomes:
            if exosome.uptaken and not hasattr(exosome, 'processed'):
                self.emit_event(EventTypes.EXOSOME_UPTAKEN, {
                    'exosome_id': exosome.id,
                    'cell_id': exosome.uptake_cell_id,
                    'cargo': {
                        'n_mrnas': len(exosome.cargo.mrnas),
                        'n_mirnas': len(exosome.cargo.mirnas),
                        'oncogenic': exosome.cargo.has_oncogenic_content()
                    }
                })
                exosome.processed = True
                self.total_exosomes_uptaken += 1
        
        # Transcription (stochastic)
        for cell_id, genes in self.cell_genes.items():
            for gene_name, gene in genes.items():
                if np.random.random() < self.transcription_rate * dt:
                    mrna = gene.transcribe()
                    if mrna:
                        self.cell_mrnas[cell_id].append(mrna)
                        self.total_transcriptions += 1
                        
                        # Emit event
                        self.emit_event(EventTypes.GENE_EXPRESSED, {
                            'cell_id': cell_id,
                            'gene': gene_name,
                            'oncogenic': gene.is_oncogene
                        })
    
    def introduce_mutation(self, cell_id: int, gene_name: str, mutation_name: str):
        """
        Introduce mutation in cell's gene
        
        Parameters:
        -----------
        cell_id : int
            Cell identifier
        gene_name : str
            Gene to mutate
        mutation_name : str
            Mutation name (e.g., 'G12D')
        """
        if cell_id in self.cell_genes and gene_name in self.cell_genes[cell_id]:
            gene = self.cell_genes[cell_id][gene_name]
            mutation = gene.introduce_oncogenic_mutation(mutation_name)
            
            if mutation:
                self.total_mutations += 1
                self.emit_event(EventTypes.MUTATION_OCCURRED, {
                    'cell_id': cell_id,
                    'gene': gene_name,
                    'mutation': mutation_name,
                    'oncogenic': mutation.oncogenic
                })
    
    def create_exosome(self, cell_id: int, oncogenic: bool = False):
        """
        Create exosome from cell
        
        Parameters:
        -----------
        cell_id : int
            Source cell
        oncogenic : bool
            Whether to package oncogenic content
        
        Returns:
        --------
        Exosome or None
        """
        if cell_id not in self.cell_genes:
            return None
        
        exosome = self.exosome_system.create_exosome(cell_id)
        
        if oncogenic:
            # Package oncogenic mRNA
            for gene_name, gene in self.cell_genes[cell_id].items():
                if gene.is_oncogene:
                    mrna = gene.transcribe()
                    if mrna:
                        exosome.package_mrna(mrna)
            
            # Package miRNA targeting tumor suppressors
            mirna = RNA("UAAGGCACGCGGUGAAUGCC", "miR-125b", NucleicAcidType.miRNA)
            mirna.target_genes = ["TP53"]
            exosome.package_mirna(mirna)
        else:
            # Package normal mRNA
            if self.cell_mrnas[cell_id]:
                mrna = self.cell_mrnas[cell_id][0]  # Take first available
                exosome.package_mrna(mrna)
        
        # Set surface markers
        exosome.set_surface_markers(['CD63', 'CD81', 'integrin_alpha_v'])
        
        return exosome
    
    def release_exosome(self, exosome, position):
        """Release exosome into environment"""
        self.exosome_system.release_exosome(exosome, position)
        self.total_exosomes_released += 1
        
        self.emit_event(EventTypes.EXOSOME_RELEASED, {
            'exosome_id': exosome.id,
            'source_cell': exosome.source_cell_id,
            'position': position.tolist() if hasattr(position, 'tolist') else position,
            'oncogenic': exosome.cargo.has_oncogenic_content()
        })
    
    def get_exosomes_near(self, position, radius: float = 15.0):
        """Get exosomes near position"""
        return self.exosome_system.get_exosomes_near(position, radius)
    
    def mark_exosome_uptaken(self, exosome, cell_id: int):
        """Mark exosome as uptaken"""
        self.exosome_system.mark_uptaken(exosome, cell_id)
    
    def get_state(self) -> Dict[str, Any]:
        """Return current molecular state"""
        return {
            'n_genes': len(self.genes),
            'n_cells_tracked': len(self.cell_genes),
            'n_exosomes': len(self.exosome_system.exosomes),
            'n_active_exosomes': len([e for e in self.exosome_system.exosomes 
                                     if not e.uptaken and not e.is_degraded()]),
            'total_released': self.total_exosomes_released,
            'total_uptaken': self.total_exosomes_uptaken,
            'total_mutations': self.total_mutations,
            'total_transcriptions': self.total_transcriptions,
            'exosome_stats': self.exosome_system.get_statistics()
        }
    
    # Event handlers
    def on_cell_divided(self, data):
        """Handle cell division - create genes for daughter cell"""
        parent_id = data['cell_id']
        daughter_id = data.get('daughter_id')
        
        if daughter_id and parent_id in self.cell_genes:
            # Daughter inherits parent's genes (with mutations)
            self.cell_genes[daughter_id] = {}
            for name, gene in self.cell_genes[parent_id].items():
                # Copy gene with mutations
                daughter_gene = Gene(gene.name, gene.dna.sequence, gene.gene_type)
                daughter_gene.is_oncogene = gene.is_oncogene
                daughter_gene.is_tumor_suppressor = gene.is_tumor_suppressor
                # Copy mutations
                for mutation in gene.dna.mutations:
                    daughter_gene.dna.mutations.append(mutation)
                
                self.cell_genes[daughter_id][name] = daughter_gene
            
            self.cell_mrnas[daughter_id] = []
    
    def on_cell_transformed(self, data):
        """Handle cell transformation - create oncogenic exosomes"""
        cell_id = data['cell_id']
        position = data['position']
        
        if cell_id in self.cell_genes:
            # Introduce oncogenic mutations
            self.introduce_mutation(cell_id, 'KRAS', 'G12D')
            
            # Create and release oncogenic exosome
            exosome = self.create_exosome(cell_id, oncogenic=True)
            if exosome:
                self.release_exosome(exosome, position)
    
    def on_cell_died(self, data):
        """Handle cell death - clean up molecular data"""
        cell_id = data['cell_id']
        self.remove_cell(cell_id)


# Test
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '..')
    
    from core import SimulationEngine, SimulationConfig
    
    print("=" * 70)
    print("Molecular Module Test")
    print("=" * 70)
    print()
    
    # Create engine
    engine = SimulationEngine(SimulationConfig(dt=0.01, duration=1.0))
    
    # Register molecular module
    engine.register_module('molecular', MolecularModule, {
        'transcription_rate': 0.5,
        'exosome_release_rate': 0.1
    })
    
    # Initialize
    engine.initialize()
    
    # Add some cells
    molecular = engine.modules['molecular']
    for i in range(5):
        molecular.add_cell(i)
    
    print(f"Added 5 cells to molecular tracking")
    print()
    
    # Simulate cell transformation
    print("Simulating cell transformation...")
    molecular.on_cell_transformed({
        'cell_id': 0,
        'position': np.array([50.0, 50.0, 50.0])
    })
    print()
    
    # Run simulation
    engine.run(duration=0.5)
    
    # Get results
    state = molecular.get_state()
    print("\nMolecular State:")
    for key, value in state.items():
        print(f"  {key}: {value}")
    print()
    
    print("=" * 70)
    print("✓ Molecular module working!")
    print("=" * 70)
