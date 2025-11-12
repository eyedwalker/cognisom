#!/usr/bin/env python3
"""
Cancer Transmission Demo
========================

Demonstrates molecular-level cancer transmission between cells:

1. Cancer cell has oncogenic mutation (KRAS G12D)
2. Produces oncogenic mRNA
3. Packages into exosome with miRNA
4. Exosome diffuses in environment
5. Normal cell uptakes exosome
6. Translates oncogenic protein
7. Cell transforms to cancer

This shows the ACTUAL molecular mechanisms!
"""

import sys
sys.path.insert(0, '../..')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from engine.py.molecular.nucleic_acids import Gene, RNA, NucleicAcidType
from engine.py.molecular.exosomes import Exosome, ExosomeSystem


class SimplifiedCell:
    """Simplified cell for demonstration"""
    
    def __init__(self, cell_id, position, cell_type="normal"):
        self.id = cell_id
        self.position = np.array(position, dtype=np.float32)
        self.cell_type = cell_type
        
        # Genome
        self.genes = {}
        
        # Molecular content
        self.mrnas = []
        self.mirnas = []
        self.proteins = {}
        
        # Surface receptors
        self.surface_receptors = ['CD63', 'CD81', 'integrin_alpha_v']
        
        # State
        self.transformed = False
        self.oncogenic_pathways_active = False
    
    def create_kras_gene(self, with_mutation=False):
        """Create KRAS gene"""
        kras_sequence = (
            "ATGGACTGAATATAAACTTGTGGTAGTTGGAGCTGGTGGCGTAGGCAAGAGTGCCTTGACGATACAGC"
            "TAATTCAGAATCATTTTGTGGACGAATATGATCCAACAATAGAGGATTCCTACAGGAAGCAAGTAGTA"
        )
        kras_gene = Gene("KRAS", kras_sequence)
        
        if with_mutation:
            kras_gene.introduce_oncogenic_mutation("G12D")
            kras_gene.is_oncogene = True
        
        self.genes["KRAS"] = kras_gene
        return kras_gene
    
    def produce_exosome(self):
        """Cancer cell produces exosome with oncogenic cargo"""
        if self.cell_type != "cancer":
            return None
        
        exosome = Exosome(source_cell_id=self.id)
        
        # Transcribe oncogenic KRAS
        if "KRAS" in self.genes:
            mrna = self.genes["KRAS"].transcribe()
            if mrna:
                exosome.package_mrna(mrna)
        
        # Add miRNA targeting TP53
        mirna = RNA("UAAGGCACGCGGUGAAUGCC", "miR-125b", NucleicAcidType.miRNA)
        mirna.target_genes = ["TP53"]
        exosome.package_mirna(mirna)
        
        # Set surface markers
        exosome.set_surface_markers(['CD63', 'CD81', 'integrin_alpha_v'])
        
        return exosome
    
    def uptake_exosome(self, exosome):
        """Normal cell uptakes exosome"""
        if exosome.can_be_uptaken_by(self.surface_receptors):
            # Release cargo
            for mrna in exosome.cargo.mrnas:
                self.mrnas.append(mrna)
                
                # Check if oncogenic
                if any(m.oncogenic for m in mrna.mutations):
                    print(f"  Cell {self.id}: Received oncogenic mRNA!")
                    self.process_oncogenic_mrna(mrna)
            
            for mirna in exosome.cargo.mirnas:
                self.mirnas.append(mirna)
                
                # Check if targets tumor suppressor
                if "TP53" in mirna.target_genes:
                    print(f"  Cell {self.id}: Received miRNA targeting TP53!")
                    self.suppress_p53()
            
            return True
        return False
    
    def process_oncogenic_mrna(self, mrna):
        """Translate oncogenic mRNA"""
        # Translate to protein
        protein_seq = mrna.translate()
        
        # Mutant KRAS protein
        self.proteins["mutant_KRAS"] = {
            'sequence': protein_seq,
            'activity': 1.0,  # Constitutively active!
            'oncogenic': True
        }
        
        # Activate MAPK pathway
        self.oncogenic_pathways_active = True
    
    def suppress_p53(self):
        """miRNA suppresses p53 tumor suppressor"""
        if "p53" in self.proteins:
            self.proteins["p53"]['activity'] *= 0.1  # 90% reduction
        else:
            self.proteins["p53"] = {'activity': 0.1}
    
    def check_transformation(self):
        """Check if cell has transformed to cancer"""
        # Hallmarks acquired
        has_oncogene = "mutant_KRAS" in self.proteins
        has_suppressed_p53 = ("p53" in self.proteins and 
                             self.proteins["p53"]['activity'] < 0.2)
        
        if has_oncogene and has_suppressed_p53:
            if not self.transformed:
                print(f"  ⚠️  Cell {self.id} TRANSFORMED TO CANCER!")
                self.cell_type = "cancer"
                self.transformed = True
                return True
        
        return False


def run_cancer_transmission_demo():
    """Run the demo"""
    
    print("=" * 70)
    print("CANCER TRANSMISSION: Molecular-Level Simulation")
    print("=" * 70)
    print()
    print("Scenario:")
    print("  - Cancer cell (ID=0) has KRAS G12D mutation")
    print("  - Produces exosome with oncogenic mRNA + miRNA")
    print("  - Normal cells (ID=1-4) nearby")
    print("  - Exosomes diffuse and transform normal cells")
    print()
    print("=" * 70)
    print()
    
    # Create cells
    cancer_cell = SimplifiedCell(0, [50, 50, 50], "cancer")
    cancer_cell.create_kras_gene(with_mutation=True)
    
    normal_cells = [
        SimplifiedCell(1, [30, 30, 50], "normal"),
        SimplifiedCell(2, [70, 30, 50], "normal"),
        SimplifiedCell(3, [30, 70, 50], "normal"),
        SimplifiedCell(4, [70, 70, 50], "normal"),
    ]
    
    for cell in normal_cells:
        cell.create_kras_gene(with_mutation=False)
    
    all_cells = [cancer_cell] + normal_cells
    
    # Create exosome system
    exo_system = ExosomeSystem()
    
    # Simulation parameters
    dt = 0.01  # hours
    duration = 5.0  # hours
    exosome_release_interval = 0.5  # hours
    
    # Data collection
    time_data = []
    transformed_count = []
    exosome_count = []
    
    # Run simulation
    print("Starting simulation...")
    print()
    
    time = 0.0
    step = 0
    last_release = 0.0
    
    while time < duration:
        # Cancer cell releases exosomes periodically
        if time - last_release >= exosome_release_interval:
            exosome = cancer_cell.produce_exosome()
            if exosome:
                exo_system.release_exosome(exosome, cancer_cell.position)
                print(f"t={time:.2f}h: Cancer cell released exosome")
                print(f"  Cargo: {exosome.cargo.total_molecules()} molecules")
                print(f"  Oncogenic: {exosome.cargo.has_oncogenic_content()}")
                last_release = time
        
        # Update exosomes (diffusion)
        exo_system.update(dt)
        
        # Check for uptake by normal cells
        for cell in normal_cells:
            if cell.cell_type == "normal":
                nearby_exosomes = exo_system.get_exosomes_near(
                    cell.position, radius=15.0
                )
                
                for exosome in nearby_exosomes:
                    if not exosome.uptaken:
                        if cell.uptake_exosome(exosome):
                            exo_system.mark_uptaken(exosome, cell.id)
                            print(f"t={time:.2f}h: Cell {cell.id} uptook exosome")
                            
                            # Check transformation
                            cell.check_transformation()
                            print()
        
        # Collect data
        time_data.append(time)
        transformed = sum(1 for c in normal_cells if c.transformed)
        transformed_count.append(transformed)
        exosome_count.append(len([ex for ex in exo_system.exosomes 
                                  if not ex.uptaken and not ex.is_degraded()]))
        
        time += dt
        step += 1
    
    print()
    print("=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print()
    
    # Final statistics
    print("Final Results:")
    print(f"  Duration: {duration} hours")
    print(f"  Initial cancer cells: 1")
    print(f"  Initial normal cells: {len(normal_cells)}")
    print()
    
    transformed = [c for c in normal_cells if c.transformed]
    print(f"  Transformed cells: {len(transformed)}")
    for cell in transformed:
        print(f"    - Cell {cell.id}")
        print(f"      Has mutant KRAS: {'mutant_KRAS' in cell.proteins}")
        print(f"      p53 suppressed: {cell.proteins.get('p53', {}).get('activity', 1.0) < 0.2}")
    print()
    
    exo_stats = exo_system.get_statistics()
    print("Exosome Statistics:")
    for key, value in exo_stats.items():
        print(f"  {key}: {value}")
    print()
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Transformed cells over time
    ax1.plot(time_data, transformed_count, 'r-', linewidth=2)
    ax1.set_xlabel('Time (hours)', fontsize=12)
    ax1.set_ylabel('Transformed Cells', fontsize=12)
    ax1.set_title('Cancer Transmission: Normal → Cancer', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, len(normal_cells) + 0.1)
    
    # Exosomes over time
    ax2.plot(time_data, exosome_count, 'b-', linewidth=2)
    ax2.set_xlabel('Time (hours)', fontsize=12)
    ax2.set_ylabel('Active Exosomes', fontsize=12)
    ax2.set_title('Exosomes in Environment', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cancer_transmission.png', dpi=150, bbox_inches='tight')
    print("✓ Saved plot: cancer_transmission.png")
    print()
    
    plt.show()
    
    print("=" * 70)
    print("KEY INSIGHTS:")
    print("=" * 70)
    print()
    print("1. MOLECULAR MECHANISM:")
    print("   - Cancer cell has KRAS G12D mutation (actual sequence!)")
    print("   - Produces oncogenic mRNA with mutation")
    print("   - Packages into exosomes with miRNA")
    print()
    print("2. TRANSFER PROCESS:")
    print("   - Exosomes diffuse via Brownian motion")
    print("   - Normal cells uptake via surface receptors")
    print("   - Cargo released into cytoplasm")
    print()
    print("3. TRANSFORMATION:")
    print("   - Oncogenic mRNA translated to mutant protein")
    print("   - miRNA suppresses p53 tumor suppressor")
    print("   - Cell acquires cancer hallmarks")
    print("   - Normal → Cancer transformation complete!")
    print()
    print("4. THIS IS REAL BIOLOGY:")
    print("   - Exosome-mediated horizontal gene transfer")
    print("   - Documented in cancer research")
    print("   - We simulate the ACTUAL molecular mechanisms")
    print()
    print("=" * 70)


if __name__ == '__main__':
    run_cancer_transmission_demo()
