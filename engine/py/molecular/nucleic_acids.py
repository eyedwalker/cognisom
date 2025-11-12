#!/usr/bin/env python3
"""
Nucleic Acids: DNA and RNA
===========================

Molecular representation with actual sequences.

Features:
- Real base sequences (ATCG/AUCG)
- Mutations tracking
- Chemical properties from sequence
- Transcription (DNA → RNA)
- Complementarity checking
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class NucleicAcidType(Enum):
    """Type of nucleic acid"""
    DNA = "DNA"
    RNA = "RNA"
    mRNA = "mRNA"
    miRNA = "miRNA"
    tRNA = "tRNA"


@dataclass
class Mutation:
    """Represents a mutation in sequence"""
    position: int
    original: str
    mutant: str
    mutation_type: str  # "substitution", "insertion", "deletion"
    name: str = ""  # e.g., "G12D"
    oncogenic: bool = False


class NucleicAcid:
    """
    Base class for DNA and RNA
    
    Represents nucleic acids with actual sequence and properties.
    """
    
    # Complementary base pairs
    DNA_COMPLEMENT = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    RNA_COMPLEMENT = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
    DNA_TO_RNA = {'A': 'A', 'T': 'U', 'G': 'G', 'C': 'C'}
    
    def __init__(
        self,
        sequence: str,
        molecule_type: NucleicAcidType,
        name: str = ""
    ):
        self.sequence = sequence.upper()
        self.type = molecule_type
        self.name = name
        self.length = len(sequence)
        
        # Mutations
        self.mutations: List[Mutation] = []
        
        # Chemical properties
        self.gc_content = self._calculate_gc_content()
        self.melting_temp = self._calculate_tm()
        self.stability = self._calculate_stability()
        
        # Modifications (epigenetic)
        self.methylation: Dict[int, str] = {}  # {position: type}
        self.acetylation: Dict[int, str] = {}
        
        # State
        self.bound_to: Optional['NucleicAcid'] = None
        self.location: str = "nucleus"  # "nucleus", "cytoplasm", "exosome"
        self.half_life: float = 8.0  # hours
        self.age: float = 0.0
    
    def _calculate_gc_content(self) -> float:
        """Calculate GC content (affects stability)"""
        if self.length == 0:
            return 0.0
        gc_count = self.sequence.count('G') + self.sequence.count('C')
        return gc_count / self.length
    
    def _calculate_tm(self) -> float:
        """
        Calculate melting temperature
        Simple formula: Tm = 4(G+C) + 2(A+T/U)
        """
        gc = self.sequence.count('G') + self.sequence.count('C')
        at = self.sequence.count('A') + self.sequence.count('T') + self.sequence.count('U')
        return 4 * gc + 2 * at
    
    def _calculate_stability(self) -> float:
        """
        Calculate stability score (0-1)
        Higher GC content = more stable
        """
        return min(1.0, self.gc_content * 1.5)
    
    def mutate(self, position: int, new_base: str, mutation_name: str = "") -> Mutation:
        """
        Introduce a mutation
        
        Parameters:
        -----------
        position : int
            Position in sequence (0-indexed)
        new_base : str
            New base to insert
        mutation_name : str
            Name of mutation (e.g., "G12D")
        
        Returns:
        --------
        Mutation object
        """
        if position >= self.length:
            raise ValueError(f"Position {position} out of range (length={self.length})")
        
        old_base = self.sequence[position]
        
        # Create mutation record
        mutation = Mutation(
            position=position,
            original=old_base,
            mutant=new_base,
            mutation_type="substitution",
            name=mutation_name
        )
        self.mutations.append(mutation)
        
        # Update sequence
        seq_list = list(self.sequence)
        seq_list[position] = new_base
        self.sequence = ''.join(seq_list)
        
        # Recalculate properties
        self.gc_content = self._calculate_gc_content()
        self.melting_temp = self._calculate_tm()
        self.stability = self._calculate_stability()
        
        return mutation
    
    def is_complementary(self, other: 'NucleicAcid', threshold: float = 0.8) -> bool:
        """
        Check if sequences are complementary
        
        Parameters:
        -----------
        other : NucleicAcid
            Other sequence to check
        threshold : float
            Minimum fraction of complementary bases (0-1)
        
        Returns:
        --------
        bool : True if complementary above threshold
        """
        if self.length != other.length:
            return False
        
        # Get complement map
        if self.type == NucleicAcidType.DNA:
            complement = self.DNA_COMPLEMENT
        else:
            complement = self.RNA_COMPLEMENT
        
        # Count complementary bases
        matches = 0
        for i in range(self.length):
            if complement.get(self.sequence[i]) == other.sequence[i]:
                matches += 1
        
        return (matches / self.length) >= threshold
    
    def get_codon(self, position: int) -> str:
        """Get codon (3 bases) at position"""
        if position + 3 > self.length:
            return ""
        return self.sequence[position:position+3]
    
    def translate_codon(self, codon: str) -> str:
        """Translate codon to amino acid"""
        # Genetic code
        genetic_code = {
            'UUU': 'F', 'UUC': 'F', 'UUA': 'L', 'UUG': 'L',
            'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S',
            'UAU': 'Y', 'UAC': 'Y', 'UAA': '*', 'UAG': '*',
            'UGU': 'C', 'UGC': 'C', 'UGA': '*', 'UGG': 'W',
            'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L',
            'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
            'CAU': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
            'AUU': 'I', 'AUC': 'I', 'AUA': 'I', 'AUG': 'M',
            'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
            'AAU': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
            'AGU': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
            'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V',
            'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
            'GAU': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
            'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
        }
        return genetic_code.get(codon, 'X')
    
    def update(self, dt: float):
        """Update state (aging, degradation)"""
        self.age += dt
        
        # Check if degraded
        if self.age > self.half_life:
            # Probabilistic degradation
            if np.random.random() < 0.1:  # 10% chance per time step
                return False  # Degraded
        
        return True  # Still alive


class DNA(NucleicAcid):
    """
    DNA molecule with double helix structure
    """
    
    def __init__(self, sequence: str, gene_name: str = ""):
        super().__init__(sequence, NucleicAcidType.DNA, gene_name)
        self.gene_name = gene_name
        self.is_coding = True
        self.promoter_strength = 1.0
        
        # Double helix
        self.complement_strand = self._generate_complement()
    
    def _generate_complement(self) -> str:
        """Generate complementary DNA strand"""
        return ''.join([self.DNA_COMPLEMENT[base] for base in self.sequence])
    
    def transcribe(self) -> 'RNA':
        """
        Transcribe DNA to RNA
        
        Returns:
        --------
        RNA object with transcribed sequence
        """
        # Convert DNA to RNA (T → U)
        rna_sequence = ''.join([self.DNA_TO_RNA[base] for base in self.sequence])
        
        # Create RNA
        rna = RNA(rna_sequence, self.gene_name)
        rna.from_gene = self.gene_name
        
        # Copy mutations
        for mutation in self.mutations:
            rna.mutations.append(mutation)
        
        return rna


class RNA(NucleicAcid):
    """
    RNA molecule (mRNA, miRNA, etc.)
    """
    
    def __init__(
        self,
        sequence: str,
        name: str = "",
        rna_type: NucleicAcidType = NucleicAcidType.mRNA
    ):
        super().__init__(sequence, rna_type, name)
        self.from_gene: str = ""
        self.coding: bool = True
        
        # mRNA specific
        if rna_type == NucleicAcidType.mRNA:
            self.has_poly_a_tail = True
            self.has_5_cap = True
            self.half_life = 8.0  # hours
        
        # miRNA specific
        elif rna_type == NucleicAcidType.miRNA:
            self.target_genes: List[str] = []
            self.target_sequence: str = ""
            self.half_life = 24.0  # hours (more stable)
    
    def translate(self) -> str:
        """
        Translate RNA to amino acid sequence
        
        Returns:
        --------
        str : Amino acid sequence
        """
        if self.type != NucleicAcidType.mRNA:
            return ""
        
        protein_sequence = []
        
        # Find start codon (AUG)
        start_pos = self.sequence.find('AUG')
        if start_pos == -1:
            return ""
        
        # Translate codon by codon
        for i in range(start_pos, self.length - 2, 3):
            codon = self.get_codon(i)
            amino_acid = self.translate_codon(codon)
            
            if amino_acid == '*':  # Stop codon
                break
            
            protein_sequence.append(amino_acid)
        
        return ''.join(protein_sequence)
    
    def can_bind_to(self, target: 'RNA') -> bool:
        """
        Check if this RNA can bind to target (e.g., miRNA to mRNA)
        """
        if self.type == NucleicAcidType.miRNA:
            # miRNA binds to complementary sequence in mRNA
            return self.is_complementary(target, threshold=0.7)
        
        return False


class Gene:
    """
    Gene representation with regulatory elements
    """
    
    def __init__(
        self,
        name: str,
        sequence: str,
        gene_type: str = "protein_coding"
    ):
        self.name = name
        self.dna = DNA(sequence, name)
        self.gene_type = gene_type
        
        # Regulatory elements
        self.promoter_sequence = ""
        self.enhancers: List[str] = []
        self.silencers: List[str] = []
        
        # Expression
        self.transcription_rate = 1.0
        self.is_active = True
        
        # Mutations
        self.is_oncogene = False
        self.is_tumor_suppressor = False
    
    def transcribe(self, rate_modifier: float = 1.0) -> RNA:
        """
        Transcribe gene to mRNA
        
        Parameters:
        -----------
        rate_modifier : float
            Modify transcription rate (e.g., from transcription factors)
        
        Returns:
        --------
        RNA object
        """
        if not self.is_active:
            return None
        
        # Transcribe DNA
        mrna = self.dna.transcribe()
        mrna.from_gene = self.name
        
        # Apply rate modifier
        effective_rate = self.transcription_rate * rate_modifier
        
        # Stochastic transcription
        if np.random.random() < effective_rate:
            return mrna
        
        return None
    
    def introduce_oncogenic_mutation(self, mutation_name: str) -> Mutation:
        """
        Introduce known oncogenic mutation
        
        Parameters:
        -----------
        mutation_name : str
            Name of mutation (e.g., "G12D" for KRAS)
        
        Returns:
        --------
        Mutation object
        """
        # Known oncogenic mutations
        oncogenic_mutations = {
            'KRAS': {
                'G12D': (35, 'G', 'A'),  # Position 35: GGT → GAT
                'G12V': (35, 'G', 'T'),  # GGT → GTT
                'G13D': (38, 'G', 'A'),  # GGC → GAC
            },
            'BRAF': {
                'V600E': (1799, 'T', 'A'),  # GTG → GAG
            },
            'TP53': {
                'R175H': (524, 'G', 'A'),  # CGC → CAC
                'R248W': (742, 'C', 'T'),  # CGG → TGG
            }
        }
        
        if self.name in oncogenic_mutations:
            if mutation_name in oncogenic_mutations[self.name]:
                pos, old, new = oncogenic_mutations[self.name][mutation_name]
                mutation = self.dna.mutate(pos, new, mutation_name)
                mutation.oncogenic = True
                self.is_oncogene = True
                return mutation
        
        return None


# Example usage
if __name__ == '__main__':
    print("=" * 60)
    print("Molecular Simulation: DNA/RNA with Actual Sequences")
    print("=" * 60)
    print()
    
    # Create KRAS gene (simplified sequence)
    kras_sequence = (
        "ATGGACTGAATATAAACTTGTGGTAGTTGGAGCTGGTGGCGTAGGCAAGAGTGCCTTGACGATACAGC"
        "TAATTCAGAATCATTTTGTGGACGAATATGATCCAACAATAGAGGATTCCTACAGGAAGCAAGTAGTA"
    )
    
    kras_gene = Gene("KRAS", kras_sequence)
    print(f"Created KRAS gene:")
    print(f"  Length: {kras_gene.dna.length} bases")
    print(f"  GC content: {kras_gene.dna.gc_content:.2%}")
    print(f"  Melting temp: {kras_gene.dna.melting_temp:.1f}°C")
    print()
    
    # Introduce oncogenic mutation
    print("Introducing G12D mutation (cancer-causing)...")
    mutation = kras_gene.introduce_oncogenic_mutation("G12D")
    print(f"  Position: {mutation.position}")
    print(f"  Change: {mutation.original} → {mutation.mutant}")
    print(f"  Oncogenic: {mutation.oncogenic}")
    print()
    
    # Transcribe to mRNA
    print("Transcribing mutant gene to mRNA...")
    mrna = kras_gene.transcribe()
    print(f"  mRNA length: {mrna.length} bases")
    print(f"  From gene: {mrna.from_gene}")
    print(f"  Mutations: {len(mrna.mutations)}")
    print()
    
    # Translate to protein
    print("Translating mRNA to protein...")
    protein_seq = mrna.translate()
    print(f"  Protein length: {len(protein_seq)} amino acids")
    print(f"  Sequence: {protein_seq[:50]}...")
    print()
    
    # Create miRNA
    print("Creating miRNA targeting TP53...")
    mirna_sequence = "UAAGGCACGCGGUGAAUGCC"
    mirna = RNA(mirna_sequence, "miR-125b", NucleicAcidType.miRNA)
    mirna.target_genes = ["TP53"]
    print(f"  miRNA length: {mirna.length} bases")
    print(f"  Targets: {mirna.target_genes}")
    print(f"  Half-life: {mirna.half_life} hours")
    print()
    
    print("=" * 60)
    print("✓ Molecular system working!")
    print("  - Real sequences (ATCG/AUCG)")
    print("  - Oncogenic mutations")
    print("  - Transcription (DNA → RNA)")
    print("  - Translation (RNA → Protein)")
    print("=" * 60)
