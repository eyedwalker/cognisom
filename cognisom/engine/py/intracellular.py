"""
Detailed Intracellular Model
=============================

Models the internal structure and dynamics of a single cell:
- DNA (genes, transcription)
- RNA (mRNA, tRNA, rRNA)
- Ribosomes (translation machinery)
- Proteins (enzymes, receptors, structural)
- Organelles (nucleus, ER, Golgi, mitochondria, vesicles)
- Membrane receptors
- Signaling pathways
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum


class Organelle(Enum):
    """Cell organelles"""
    NUCLEUS = "nucleus"
    CYTOPLASM = "cytoplasm"
    ENDOPLASMIC_RETICULUM = "ER"
    GOLGI = "golgi"
    MITOCHONDRIA = "mitochondria"
    LYSOSOME = "lysosome"
    PEROXISOME = "peroxisome"
    MEMBRANE = "membrane"


class MoleculeType(Enum):
    """Types of molecules"""
    DNA = "DNA"
    MRNA = "mRNA"
    TRNA = "tRNA"
    RRNA = "rRNA"
    PROTEIN = "protein"
    ENZYME = "enzyme"
    RECEPTOR = "receptor"
    LIGAND = "ligand"
    METABOLITE = "metabolite"
    ATP = "ATP"
    GTP = "GTP"


@dataclass
class Gene:
    """A gene on the DNA"""
    name: str
    sequence_length: int  # base pairs
    promoter_strength: float = 1.0
    transcription_rate: float = 0.1  # transcripts per hour
    is_active: bool = True
    chromosome: int = 1
    position: int = 0
    
    # Regulation
    transcription_factors: List[str] = field(default_factory=list)
    repressors: List[str] = field(default_factory=list)
    epigenetic_state: str = "open"  # open, closed, poised


@dataclass
class mRNA:
    """Messenger RNA molecule"""
    gene_name: str
    sequence_length: int
    copy_number: int = 1
    location: Organelle = Organelle.CYTOPLASM
    half_life: float = 2.0  # hours
    translation_rate: float = 10.0  # proteins per hour per mRNA
    age: float = 0.0


@dataclass
class Protein:
    """Protein molecule"""
    name: str
    gene_source: str
    copy_number: int = 0
    location: Organelle = Organelle.CYTOPLASM
    half_life: float = 10.0  # hours
    molecular_weight: float = 50.0  # kDa
    is_enzyme: bool = False
    is_receptor: bool = False
    is_structural: bool = False
    
    # For receptors
    ligand_bound: int = 0
    surface_expression: int = 0  # copies on membrane


@dataclass
class Ribosome:
    """Ribosome translation machinery"""
    ribosome_id: int
    location: Organelle = Organelle.CYTOPLASM
    is_translating: bool = False
    current_mrna: Optional[str] = None
    progress: float = 0.0  # 0-1, how far through translation


@dataclass
class Receptor:
    """Cell surface receptor"""
    name: str
    receptor_type: str  # GPCR, RTK, cytokine, etc.
    surface_count: int = 100
    ligand: Optional[str] = None
    bound_count: int = 0
    
    # Signaling
    downstream_pathway: Optional[str] = None
    activation_threshold: int = 10  # ligands needed
    is_activated: bool = False
    
    # Internalization
    internalization_rate: float = 0.1  # per hour
    recycling_rate: float = 0.5  # per hour


class IntracellularModel:
    """
    Detailed model of cell internals
    
    Tracks:
    - Genome (genes)
    - Transcriptome (mRNAs)
    - Proteome (proteins)
    - Organelles
    - Membrane receptors
    - Signaling pathways
    """
    
    def __init__(self):
        # Genome
        self.genes: Dict[str, Gene] = {}
        self.chromosomes: int = 23  # Human
        
        # Transcriptome
        self.mrnas: Dict[str, mRNA] = {}
        
        # Proteome
        self.proteins: Dict[str, Protein] = {}
        
        # Ribosomes
        self.ribosomes: List[Ribosome] = []
        self.num_ribosomes: int = 10000  # Typical mammalian cell
        
        # Organelles (counts/volumes)
        self.organelles = {
            Organelle.NUCLEUS: {'volume': 500.0, 'count': 1},
            Organelle.CYTOPLASM: {'volume': 1500.0, 'count': 1},
            Organelle.MITOCHONDRIA: {'volume': 50.0, 'count': 300},
            Organelle.ENDOPLASMIC_RETICULUM: {'volume': 200.0, 'count': 1},
            Organelle.GOLGI: {'volume': 100.0, 'count': 1},
        }
        
        # Membrane receptors
        self.receptors: Dict[str, Receptor] = {}
        
        # Metabolites
        self.metabolites = {
            'ATP': 5000000,  # ~5mM in typical cell
            'GTP': 500000,
            'amino_acids': 100000,
            'nucleotides': 50000,
            'glucose': 10000,
        }
        
        # Signaling pathways (active/inactive)
        self.pathways = {
            'MAPK': {'active': False, 'strength': 0.0},
            'PI3K_AKT': {'active': False, 'strength': 0.0},
            'p53': {'active': False, 'strength': 0.0},
            'NFkB': {'active': False, 'strength': 0.0},
        }
        
        # Initialize default genes
        self._initialize_default_genome()
    
    def _initialize_default_genome(self):
        """Create a minimal genome"""
        # Housekeeping genes
        self.add_gene(Gene(
            name='GAPDH',
            sequence_length=1200,
            promoter_strength=1.0,
            transcription_rate=0.5,
        ))
        
        self.add_gene(Gene(
            name='ACTB',  # Beta-actin
            sequence_length=1800,
            promoter_strength=1.0,
            transcription_rate=0.5,
        ))
        
        # Immune genes
        self.add_gene(Gene(
            name='HLA_A',  # MHC-I
            sequence_length=3500,
            promoter_strength=0.8,
            transcription_rate=0.2,
        ))
        
        self.add_gene(Gene(
            name='B2M',  # Beta-2-microglobulin (MHC-I component)
            sequence_length=600,
            promoter_strength=0.8,
            transcription_rate=0.3,
        ))
        
        # Stress response
        self.add_gene(Gene(
            name='TP53',  # p53 tumor suppressor
            sequence_length=1200,
            promoter_strength=0.5,
            transcription_rate=0.1,
        ))
        
        self.add_gene(Gene(
            name='HSP70',  # Heat shock protein
            sequence_length=1900,
            promoter_strength=0.3,
            transcription_rate=0.05,
        ))
        
        # Growth receptors
        self.add_gene(Gene(
            name='EGFR',  # Epidermal growth factor receptor
            sequence_length=3600,
            promoter_strength=0.6,
            transcription_rate=0.1,
        ))
        
        # Initialize ribosomes
        for i in range(100):  # Start with 100 active ribosomes
            self.ribosomes.append(Ribosome(
                ribosome_id=i,
                location=Organelle.CYTOPLASM
            ))
    
    def add_gene(self, gene: Gene):
        """Add a gene to the genome"""
        self.genes[gene.name] = gene
    
    def transcribe(self, gene_name: str, dt: float = 0.01) -> int:
        """
        Transcribe a gene to mRNA
        
        Returns number of new mRNA molecules created
        """
        if gene_name not in self.genes:
            return 0
        
        gene = self.genes[gene_name]
        
        if not gene.is_active:
            return 0
        
        # Check for ATP/GTP
        if self.metabolites['ATP'] < 100 or self.metabolites['GTP'] < 50:
            return 0
        
        # Stochastic transcription
        rate = gene.transcription_rate * gene.promoter_strength * dt
        new_transcripts = np.random.poisson(rate)
        
        if new_transcripts > 0:
            # Consume energy
            self.metabolites['ATP'] -= new_transcripts * 100
            self.metabolites['GTP'] -= new_transcripts * 50
            
            # Create or update mRNA
            if gene_name in self.mrnas:
                self.mrnas[gene_name].copy_number += new_transcripts
            else:
                self.mrnas[gene_name] = mRNA(
                    gene_name=gene_name,
                    sequence_length=gene.sequence_length,
                    copy_number=new_transcripts,
                    location=Organelle.CYTOPLASM,
                )
        
        return new_transcripts
    
    def translate(self, mrna_name: str, dt: float = 0.01) -> int:
        """
        Translate mRNA to protein
        
        Returns number of new protein molecules created
        """
        if mrna_name not in self.mrnas:
            return 0
        
        mrna = self.mrnas[mrna_name]
        
        # Check for resources
        if self.metabolites['ATP'] < 200 or self.metabolites['amino_acids'] < 100:
            return 0
        
        # Translation rate depends on mRNA count and ribosome availability
        available_ribosomes = sum(1 for r in self.ribosomes if not r.is_translating)
        active_translation = min(mrna.copy_number, available_ribosomes)
        
        rate = mrna.translation_rate * active_translation * dt
        new_proteins = np.random.poisson(rate)
        
        if new_proteins > 0:
            # Consume resources
            self.metabolites['ATP'] -= new_proteins * 200
            self.metabolites['amino_acids'] -= new_proteins * 100
            
            # Create or update protein
            if mrna_name in self.proteins:
                self.proteins[mrna_name].copy_number += new_proteins
            else:
                self.proteins[mrna_name] = Protein(
                    name=mrna_name,
                    gene_source=mrna_name,
                    copy_number=new_proteins,
                    location=Organelle.CYTOPLASM,
                )
        
        return new_proteins
    
    def degrade_mrna(self, dt: float = 0.01):
        """Degrade mRNA molecules"""
        to_remove = []
        
        for name, mrna in self.mrnas.items():
            mrna.age += dt
            
            # Exponential decay
            decay_rate = np.log(2) / mrna.half_life
            degraded = np.random.binomial(mrna.copy_number, decay_rate * dt)
            
            mrna.copy_number -= degraded
            
            if mrna.copy_number <= 0:
                to_remove.append(name)
        
        for name in to_remove:
            del self.mrnas[name]
    
    def degrade_protein(self, dt: float = 0.01):
        """Degrade protein molecules"""
        to_remove = []
        
        for name, protein in self.proteins.items():
            # Exponential decay
            decay_rate = np.log(2) / protein.half_life
            degraded = np.random.binomial(protein.copy_number, decay_rate * dt)
            
            protein.copy_number -= degraded
            
            if protein.copy_number <= 0:
                to_remove.append(name)
        
        for name in to_remove:
            del self.proteins[name]
    
    def regenerate_atp(self, dt: float = 0.01):
        """Regenerate ATP through metabolism"""
        # Simplified: glucose -> ATP
        if self.metabolites['glucose'] > 0:
            # Each glucose -> ~30 ATP
            glucose_consumed = min(10, self.metabolites['glucose'])
            atp_generated = glucose_consumed * 30
            
            self.metabolites['glucose'] -= glucose_consumed
            self.metabolites['ATP'] += atp_generated
            
            # Cap ATP
            self.metabolites['ATP'] = min(self.metabolites['ATP'], 10000000)
    
    def step(self, dt: float = 0.01):
        """
        Advance intracellular state by one time step
        """
        # 1. Transcription (DNA -> mRNA)
        for gene_name in self.genes.keys():
            self.transcribe(gene_name, dt)
        
        # 2. Translation (mRNA -> Protein)
        for mrna_name in list(self.mrnas.keys()):
            self.translate(mrna_name, dt)
        
        # 3. Degradation
        self.degrade_mrna(dt)
        self.degrade_protein(dt)
        
        # 4. Metabolism
        self.regenerate_atp(dt)
    
    def get_state_summary(self) -> Dict:
        """Get current state summary"""
        return {
            'genes': len(self.genes),
            'active_genes': sum(1 for g in self.genes.values() if g.is_active),
            'mrna_species': len(self.mrnas),
            'total_mrna': sum(m.copy_number for m in self.mrnas.values()),
            'protein_species': len(self.proteins),
            'total_proteins': sum(p.copy_number for p in self.proteins.values()),
            'ribosomes': len(self.ribosomes),
            'atp': self.metabolites['ATP'],
            'glucose': self.metabolites['glucose'],
        }
    
    def get_gene_expression(self) -> Dict[str, int]:
        """Get mRNA counts for all genes"""
        expression = {}
        for gene_name in self.genes.keys():
            expression[gene_name] = self.mrnas.get(gene_name, mRNA(gene_name, 0, 0)).copy_number
        return expression
    
    def get_protein_levels(self) -> Dict[str, int]:
        """Get protein counts"""
        return {name: p.copy_number for name, p in self.proteins.items()}
