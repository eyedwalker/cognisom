#!/usr/bin/env python3
"""
Exosomes: Molecular Transfer Between Cells
===========================================

Exosomes are small vesicles (30-150 nm) that carry molecular cargo
between cells, including:
- mRNA (oncogenic or normal)
- miRNA (gene regulators)
- Proteins (functional or oncogenic)
- DNA fragments

This is a key mechanism for cancer spread!
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

try:
    from .nucleic_acids import RNA, DNA, NucleicAcidType
except ImportError:
    from nucleic_acids import RNA, DNA, NucleicAcidType


@dataclass
class ExosomeCargo:
    """Contents of an exosome"""
    mrnas: List[RNA] = field(default_factory=list)
    mirnas: List[RNA] = field(default_factory=list)
    proteins: List = field(default_factory=list)
    dna_fragments: List[DNA] = field(default_factory=list)
    
    def total_molecules(self) -> int:
        """Total number of molecules"""
        return (len(self.mrnas) + len(self.mirnas) + 
                len(self.proteins) + len(self.dna_fragments))
    
    def has_oncogenic_content(self) -> bool:
        """Check if cargo contains oncogenic molecules"""
        # Check mRNAs
        for mrna in self.mrnas:
            if any(m.oncogenic for m in mrna.mutations):
                return True
        
        # Check miRNAs targeting tumor suppressors
        for mirna in self.mirnas:
            if any(target in ['TP53', 'RB1', 'PTEN'] for target in mirna.target_genes):
                return True
        
        return False


class Exosome:
    """
    Extracellular vesicle for cell-cell communication
    
    Exosomes package and transport molecular cargo between cells.
    This is how cancer cells can transform normal cells!
    
    Example:
    --------
    >>> exosome = Exosome(source_cell_id=0)
    >>> exosome.package_mrna(oncogenic_mrna)
    >>> exosome.package_mirna(tumor_suppressor_mirna)
    >>> exosome.release(position=(10, 10, 10))
    """
    
    def __init__(
        self,
        source_cell_id: int,
        size: float = 100.0  # nm
    ):
        self.source_cell_id = source_cell_id
        self.size = size  # nanometers
        
        # Cargo
        self.cargo = ExosomeCargo()
        
        # Surface markers (determine which cells can take it up)
        self.surface_markers: List[str] = []
        
        # Spatial properties
        self.position: Optional[np.ndarray] = None
        self.velocity: np.ndarray = np.zeros(3)
        
        # Diffusion coefficient (μm²/s)
        # D = kT / (6πηr)
        # For 100nm particle in water: D ≈ 4 μm²/s
        self.diffusion_coeff = 4.0  # μm²/s
        
        # State
        self.released = False
        self.uptaken = False
        self.uptake_cell_id: Optional[int] = None
        self.age = 0.0
        self.lifetime = 24.0  # hours
    
    def package_mrna(self, mrna: RNA):
        """Add mRNA to cargo"""
        if mrna.type == NucleicAcidType.mRNA:
            # Mark for exosome packaging
            mrna.location = "exosome"
            self.cargo.mrnas.append(mrna)
    
    def package_mirna(self, mirna: RNA):
        """Add miRNA to cargo"""
        if mirna.type == NucleicAcidType.miRNA:
            mirna.location = "exosome"
            self.cargo.mirnas.append(mirna)
    
    def package_protein(self, protein):
        """Add protein to cargo"""
        protein.location = "exosome"
        self.cargo.proteins.append(protein)
    
    def package_dna_fragment(self, dna: DNA):
        """Add DNA fragment to cargo"""
        dna.location = "exosome"
        self.cargo.dna_fragments.append(dna)
    
    def set_surface_markers(self, markers: List[str]):
        """
        Set surface markers that determine uptake specificity
        
        Common markers:
        - CD63, CD81, CD9: General exosome markers
        - Integrins: Tissue-specific targeting
        - Tetraspanins: Cell-type specific
        """
        self.surface_markers = markers
    
    def release(self, position: np.ndarray):
        """
        Release exosome into extracellular space
        
        Parameters:
        -----------
        position : array
            3D position (x, y, z) in μm
        """
        self.position = np.array(position, dtype=np.float32)
        self.released = True
    
    def diffuse(self, dt: float):
        """
        Brownian motion diffusion
        
        Parameters:
        -----------
        dt : float
            Time step in hours
        """
        if not self.released or self.uptaken:
            return
        
        # Convert dt to seconds
        dt_sec = dt * 3600
        
        # Brownian motion: displacement ~ sqrt(2*D*dt)
        std = np.sqrt(2 * self.diffusion_coeff * dt_sec)
        displacement = np.random.normal(0, std, 3)
        
        self.position += displacement
        
        # Age
        self.age += dt
    
    def can_be_uptaken_by(self, cell_surface_receptors: List[str]) -> bool:
        """
        Check if cell has receptors for this exosome
        
        Parameters:
        -----------
        cell_surface_receptors : list
            List of receptor types on cell surface
        
        Returns:
        --------
        bool : True if cell can uptake this exosome
        """
        # Check for matching markers
        for marker in self.surface_markers:
            if marker in cell_surface_receptors:
                return True
        return False
    
    def is_degraded(self) -> bool:
        """Check if exosome has degraded"""
        return self.age > self.lifetime
    
    def get_statistics(self) -> Dict:
        """Get exosome statistics"""
        return {
            'source_cell': self.source_cell_id,
            'size': self.size,
            'n_mrnas': len(self.cargo.mrnas),
            'n_mirnas': len(self.cargo.mirnas),
            'n_proteins': len(self.cargo.proteins),
            'n_dna_fragments': len(self.cargo.dna_fragments),
            'total_cargo': self.cargo.total_molecules(),
            'oncogenic': self.cargo.has_oncogenic_content(),
            'released': self.released,
            'uptaken': self.uptaken,
            'age': self.age
        }


class ExosomeSystem:
    """
    Manages all exosomes in the simulation
    
    Handles:
    - Exosome release from cells
    - Diffusion in environment
    - Uptake by target cells
    - Degradation
    """
    
    def __init__(self):
        self.exosomes: List[Exosome] = []
        self.next_id = 0
        
        # Statistics
        self.total_released = 0
        self.total_uptaken = 0
        self.total_degraded = 0
    
    def create_exosome(self, source_cell_id: int) -> Exosome:
        """Create new exosome"""
        exosome = Exosome(source_cell_id)
        exosome.id = self.next_id
        self.next_id += 1
        return exosome
    
    def release_exosome(self, exosome: Exosome, position: np.ndarray):
        """Release exosome into environment"""
        exosome.release(position)
        self.exosomes.append(exosome)
        self.total_released += 1
    
    def update(self, dt: float):
        """
        Update all exosomes
        
        Parameters:
        -----------
        dt : float
            Time step in hours
        """
        # Diffuse all exosomes
        for exosome in self.exosomes:
            if not exosome.uptaken:
                exosome.diffuse(dt)
        
        # Remove degraded exosomes
        degraded = [ex for ex in self.exosomes if ex.is_degraded()]
        for ex in degraded:
            self.exosomes.remove(ex)
            self.total_degraded += 1
    
    def get_exosomes_near(
        self,
        position: np.ndarray,
        radius: float = 10.0
    ) -> List[Exosome]:
        """
        Get exosomes within radius of position
        
        Parameters:
        -----------
        position : array
            3D position (x, y, z)
        radius : float
            Search radius in μm
        
        Returns:
        --------
        list : Exosomes within radius
        """
        nearby = []
        for exosome in self.exosomes:
            if exosome.released and not exosome.uptaken:
                distance = np.linalg.norm(exosome.position - position)
                if distance <= radius:
                    nearby.append(exosome)
        return nearby
    
    def mark_uptaken(self, exosome: Exosome, cell_id: int):
        """Mark exosome as uptaken by cell"""
        exosome.uptaken = True
        exosome.uptake_cell_id = cell_id
        self.total_uptaken += 1
    
    def get_statistics(self) -> Dict:
        """Get system statistics"""
        active = [ex for ex in self.exosomes if not ex.uptaken and not ex.is_degraded()]
        oncogenic = [ex for ex in active if ex.cargo.has_oncogenic_content()]
        
        return {
            'total_exosomes': len(self.exosomes),
            'active': len(active),
            'uptaken': self.total_uptaken,
            'degraded': self.total_degraded,
            'oncogenic_active': len(oncogenic),
            'total_released': self.total_released
        }


# Example usage
if __name__ == '__main__':
    print("=" * 60)
    print("Exosome System: Molecular Transfer Between Cells")
    print("=" * 60)
    print()
    
    # Create exosome system
    exo_system = ExosomeSystem()
    
    # Cancer cell creates exosome
    print("Cancer cell (ID=0) creates exosome...")
    exosome = exo_system.create_exosome(source_cell_id=0)
    
    # Package oncogenic mRNA (KRAS G12D)
    try:
        from .nucleic_acids import Gene
    except ImportError:
        from nucleic_acids import Gene
    
    kras_sequence = (
        "ATGGACTGAATATAAACTTGTGGTAGTTGGAGCTGGTGGCGTAGGCAAGAGTGCCTTGACGATACAGC"
        "TAATTCAGAATCATTTTGTGGACGAATATGATCCAACAATAGAGGATTCCTACAGGAAGCAAGTAGTA"
    )
    kras_gene = Gene("KRAS", kras_sequence)
    kras_gene.introduce_oncogenic_mutation("G12D")
    oncogenic_mrna = kras_gene.transcribe()
    
    exosome.package_mrna(oncogenic_mrna)
    print(f"  Packaged oncogenic KRAS mRNA (G12D mutation)")
    
    # Package miRNA targeting TP53
    mirna = RNA("UAAGGCACGCGGUGAAUGCC", "miR-125b", NucleicAcidType.miRNA)
    mirna.target_genes = ["TP53"]
    exosome.package_mirna(mirna)
    print(f"  Packaged miR-125b (targets TP53 tumor suppressor)")
    
    # Set surface markers
    exosome.set_surface_markers(['CD63', 'CD81', 'integrin_alpha_v'])
    print(f"  Surface markers: {exosome.surface_markers}")
    print()
    
    # Release exosome
    print("Releasing exosome at position (0, 0, 0)...")
    exo_system.release_exosome(exosome, position=np.array([0.0, 0.0, 0.0]))
    print(f"  Released: {exosome.released}")
    print(f"  Cargo: {exosome.cargo.total_molecules()} molecules")
    print(f"  Oncogenic: {exosome.cargo.has_oncogenic_content()}")
    print()
    
    # Simulate diffusion
    print("Simulating diffusion for 1 hour...")
    dt = 0.01  # hours
    for step in range(100):
        exo_system.update(dt)
        
        if step % 25 == 0:
            time = step * dt
            pos = exosome.position
            print(f"  t={time:.2f}h: position = ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) μm")
    print()
    
    # Check for nearby cells
    print("Checking for cells near exosome...")
    normal_cell_position = np.array([5.0, 5.0, 5.0])
    distance = np.linalg.norm(exosome.position - normal_cell_position)
    print(f"  Distance to normal cell: {distance:.1f} μm")
    
    if distance < 10.0:
        print(f"  ✓ Exosome within uptake range!")
        
        # Check if cell can uptake
        cell_receptors = ['CD63', 'integrin_beta_1']
        can_uptake = exosome.can_be_uptaken_by(cell_receptors)
        print(f"  Cell receptors: {cell_receptors}")
        print(f"  Can uptake: {can_uptake}")
        
        if can_uptake:
            print(f"  → Normal cell uptakes exosome!")
            print(f"  → Receives oncogenic KRAS mRNA")
            print(f"  → Receives miRNA targeting TP53")
            print(f"  → Cell may transform to cancer!")
            exo_system.mark_uptaken(exosome, cell_id=1)
    print()
    
    # Statistics
    stats = exo_system.get_statistics()
    print("System statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
    
    print("=" * 60)
    print("✓ Exosome system working!")
    print("  - Package molecular cargo")
    print("  - Brownian motion diffusion")
    print("  - Cell-specific uptake")
    print("  - Oncogenic content tracking")
    print("=" * 60)
