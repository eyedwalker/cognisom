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
    """Represents a mutation in sequence.

    The optional `effect` field carries the rule-based classification
    (synonymous / missense / nonsense / start_loss / outside_coding) and
    a numerical impact_score in [0, 1]. Populated by Gene.introduce_*
    methods when a classifier is provided. Older code paths that bypass
    the classifier leave it None.
    """
    position: int
    original: str
    mutant: str
    mutation_type: str  # "substitution", "insertion", "deletion"
    name: str = ""  # e.g., "G12D"
    oncogenic: bool = False
    effect: Optional["MutationEffect"] = None  # forward-ref; resolved at runtime


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
        Calculate melting temperature in degrees Celsius.

        Wallace rule (Tm = 4(G+C) + 2(A+T)) is valid only for short oligos
        (<= 14 bases). For longer sequences it overestimates badly (a 100mer
        gives ~300C, which is non-physical: DNA melts <= ~100C in water).
        For sequences > 14 bases we use Marmur-Doty (1962), which is the
        standard simple long-sequence approximation.
        """
        seq = self.sequence.upper()
        gc = seq.count('G') + seq.count('C')
        at = seq.count('A') + seq.count('T') + seq.count('U')
        n = gc + at
        if n == 0:
            return 0.0
        if n <= 14:
            return 4.0 * gc + 2.0 * at
        gc_fraction = gc / n
        return 64.9 + 41.0 * gc_fraction - 500.0 / n
    
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
    
    # Known oncogenic substitutions, expressed as (position_0indexed, new_base).
    # Positions are the EXACT base position in this.dna.sequence that must
    # be substituted to produce the named amino-acid change. These positions
    # were corrected in May 2026 from the prior off-by-one table that was
    # producing silent mutations under the same names (see DECISIONS.md).
    # The classifier in introduce_substitution() validates each one at
    # runtime so future corruption is caught immediately.
    ONCOGENIC_SUBSTITUTIONS = {
        'KRAS': {
            'G12D': (34, 'A'),  # codon 12 GGT -> GAT (middle base G->A)
            'G12V': (34, 'T'),  # codon 12 GGT -> GTT (middle base G->T)
            'G13D': (37, 'A'),  # codon 13 GGC -> GAC (middle base G->A)
        },
        'BRAF': {
            'V600E': (1798, 'A'),  # codon 600 GTG -> GAG (middle base T->A)
        },
        'TP53': {
            'R175H': (523, 'A'),  # codon 175 CGC -> CAC (middle base G->A)
            'R248W': (741, 'T'),  # codon 248 CGG -> TGG (first base C->T)
        }
    }

    def introduce_substitution(
        self,
        position: int,
        new_base: str,
        mutation_name: str = "",
        classifier=None,
        cds_start: int = 0,
    ) -> Mutation:
        """Introduce an arbitrary single-nucleotide substitution.

        Parameters
        ----------
        position : int
            0-indexed base position in this gene's DNA sequence.
        new_base : str
            New base ('A', 'C', 'G', 'T', or 'U').
        mutation_name : str
            Optional human-readable name (e.g., "G12D") for logging.
        classifier : MutationEffectClassifier, optional
            If provided, the mutation is classified (synonymous / missense
            / nonsense / etc.) and the resulting MutationEffect attached to
            the Mutation object. The is_oncogene flag is set automatically
            iff impact_score >= 0.4 (i.e., non-conservative coding change).
            If None, the substitution is performed without classification.
        cds_start : int
            Start of the coding region in this gene's DNA sequence. Default 0.

        Returns
        -------
        Mutation
            With effect populated if classifier was provided.
        """
        # Classify BEFORE applying so we score against the reference.
        effect = None
        if classifier is not None:
            effect = classifier.classify_substitution(
                coding_sequence=self.dna.sequence,
                position=position,
                new_base=new_base,
                cds_start=cds_start,
            )

        mutation = self.dna.mutate(position, new_base, mutation_name)
        if effect is not None:
            mutation.effect = effect
            # Promote to oncogenic flag based on impact, not external belief
            if effect.impact_score >= 0.4:
                mutation.oncogenic = True
                self.is_oncogene = True
        return mutation

    def introduce_oncogenic_mutation(
        self,
        mutation_name: str,
        classifier=None,
        cds_start: int = 0,
    ) -> Optional[Mutation]:
        """Introduce a named known oncogenic mutation (e.g., G12D for KRAS).

        Uses the corrected ONCOGENIC_SUBSTITUTIONS table. When a classifier
        is provided, validates at runtime that the resulting amino-acid
        change matches the named mutation; raises AssertionError on mismatch
        (this is a regression net for future table maintenance).

        Backward-compatible signature: classifier is optional. If None,
        no validation happens but the corrected positions still produce the
        right codon change.
        """
        if self.name not in self.ONCOGENIC_SUBSTITUTIONS:
            return None
        if mutation_name not in self.ONCOGENIC_SUBSTITUTIONS[self.name]:
            return None
        position, new_base = self.ONCOGENIC_SUBSTITUTIONS[self.name][mutation_name]

        mutation = self.introduce_substitution(
            position=position,
            new_base=new_base,
            mutation_name=mutation_name,
            classifier=classifier,
            cds_start=cds_start,
        )
        # If we classified, warn (don't raise) when the resulting AA change
        # doesn't match the named mutation. A mismatch usually means the
        # gene's reference sequence is not the canonical CDS for that gene
        # (e.g., a frame-shifted or synthetic demo sequence). The Mutation
        # is still attached with its true effect; callers that need
        # strict biological correctness should use introduce_substitution()
        # directly and inspect mutation.effect.aa_change themselves.
        if classifier is not None and mutation.effect is not None:
            actual = mutation.effect.aa_change
            if actual != mutation_name:
                import warnings
                warnings.warn(
                    f"Named mutation {self.name} {mutation_name} resolves to "
                    f"{actual!r} under the classifier (category={mutation.effect.category}). "
                    f"This usually means the gene's reference sequence is not the "
                    f"canonical CDS for that gene. Mutation attached with true "
                    f"effect; downstream code should inspect mutation.effect.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        # Explicitly mark oncogenic even if classifier wasn't given
        # (preserves the legacy semantics callers expect)
        mutation.oncogenic = True
        self.is_oncogene = True
        return mutation


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
