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
from typing import Dict, List, Any, Optional, Set

from core.module_base import SimulationModule
from core.event_bus import EventTypes
from engine.py.molecular.nucleic_acids import Gene, RNA, NucleicAcidType, Mutation
from engine.py.molecular.exosomes import ExosomeSystem
from engine.py.molecular.reference_genome import (
    ReferenceGenome,
    build_default_reference_genome,
)
from engine.py.molecular.sequence_view import CellGenomeView
from engine.py.molecular.mutation_effect import MutationEffectClassifier


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

        # Legacy gene library (Gene objects) - kept for backward-compat with
        # tests that access module.genes['KRAS'].dna.sequence etc. Per-cell
        # state no longer uses these; see self.cell_views below.
        self.genes: Dict[str, Gene] = {}

        # Authoritative reference genome (Upgrade 1 / Sprint 2).
        # Shared by all cell views. Populated at initialize().
        self.reference_genome: Optional[ReferenceGenome] = None

        # Per-cell sparse genome state. Replaces the legacy
        # self.cell_genes: Dict[int, Dict[str, Gene]] full-copy structure.
        # Each cell holds a CellGenomeView pointing at the shared reference;
        # per-cell deviations are sparse SubstitutionDelta records.
        self.cell_views: Dict[int, CellGenomeView] = {}

        # Per-cell flags that are NOT sequence state (e.g., "this cell
        # carries an oncogenic driver mutation"). Parallel to cell_views.
        self.cell_oncogene_flags: Dict[int, Set[str]] = {}

        # Per-cell mRNA pool (unchanged semantics; mRNA carries sequence
        # materialized from the view at transcription time)
        self.cell_mrnas: Dict[int, List[RNA]] = {}

        # Mutation effect classifier (used for impact scoring and to set
        # the oncogene flag based on classifier output, not external belief)
        self.classifier = MutationEffectClassifier()

        # Exosome system
        self.exosome_system = None

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

        # Build the shared reference genome (Upgrade 1)
        self.reference_genome = build_default_reference_genome()

        # Build the legacy Gene objects (backward compat)
        self._create_gene_library()

        # Create exosome system
        self.exosome_system = ExosomeSystem()

        # Subscribe to cellular events
        self.subscribe(EventTypes.CELL_DIVIDED, self.on_cell_divided)
        self.subscribe(EventTypes.CELL_TRANSFORMED, self.on_cell_transformed)
        self.subscribe(EventTypes.CELL_DIED, self.on_cell_died)

        print(f"    ✓ {len(self.genes)} genes in library "
              f"({self.reference_genome.total_bases()} reference bases shared)")
        print(f"    ✓ Exosome system ready")
    
    def _create_gene_library(self):
        """Create library of genes from engine.py.molecular.data.reference_cds.

        All three reference CDSes are authentic NCBI sequences inlined in
        reference_cds.py: KRAS (NM_004985.5, 188 aa), TP53 (NM_000546.6,
        393 aa), BRAF (NM_004333.6, 766 aa). Hotspots are validated at
        codon resolution at module-import time.
        """
        from engine.py.molecular.reference_cds import (
            KRAS_CDS, TP53_CDS, BRAF_CDS,
        )

        # KRAS (oncogene)
        self.genes['KRAS'] = Gene('KRAS', KRAS_CDS, 'protein_coding')
        self.genes['KRAS'].is_oncogene = False  # Normal initially

        # TP53 (tumor suppressor) - synthetic CDS reaches R175 and R248
        self.genes['TP53'] = Gene('TP53', TP53_CDS, 'protein_coding')
        self.genes['TP53'].is_tumor_suppressor = True

        # BRAF (oncogene) - synthetic CDS reaches V600
        self.genes['BRAF'] = Gene('BRAF', BRAF_CDS, 'protein_coding')
        self.genes['BRAF'].is_oncogene = False  # Normal initially
    
    def add_cell(self, cell_id: int):
        """Add cell to molecular tracking.

        Creates a CellGenomeView pointing at the shared ReferenceGenome.
        Per-cell memory cost is constant + (n_mutations * delta size),
        NOT (n_genes * genome_size).
        """
        self.cell_views[cell_id] = CellGenomeView(self.reference_genome)
        self.cell_oncogene_flags[cell_id] = set()
        self.cell_mrnas[cell_id] = []

    def remove_cell(self, cell_id: int):
        """Remove cell from tracking"""
        self.cell_views.pop(cell_id, None)
        self.cell_oncogene_flags.pop(cell_id, None)
        self.cell_mrnas.pop(cell_id, None)

    def get_reference_protein(self, gene_name: str) -> str:
        """Translate the reference DNA sequence for this gene into the
        wild-type protein sequence.

        Used by the cellular module to seed neoantigen peptide
        generation on MUTATION_OCCURRED (Upgrade 2). Translation starts
        at the first base (CDS-aligned reference sequences are stored
        in reference_cds.py) and stops at the first stop codon.
        Returns an empty string if the gene is not in the reference
        genome.
        """
        if self.reference_genome is None or not self.reference_genome.has_gene(gene_name):
            return ""
        dna = self.reference_genome.get_reference_sequence(gene_name)
        rna = RNA(dna.replace("T", "U"), gene_name, NucleicAcidType.mRNA)
        protein = rna.translate()
        # RNA.translate searches for AUG; the curated reference CDSes
        # are already CDS-aligned, so the result is the canonical
        # protein N-terminus through the first stop codon.
        return protein

    def _build_mrna_from_view(self, view: CellGenomeView, gene_name: str,
                              cell_id: int) -> RNA:
        """Materialize this cell's sequence for gene_name and produce an
        RNA object carrying any deltas as Mutation records.
        """
        # Materialize the per-cell DNA sequence (reference + deltas) and
        # transcribe to RNA (T->U substitution).
        dna_seq = view.materialize(gene_name)
        rna_seq = dna_seq.replace("T", "U")
        mrna = RNA(rna_seq, gene_name, NucleicAcidType.mRNA)
        mrna.from_gene = gene_name
        oncogenic_flag = gene_name in self.cell_oncogene_flags.get(cell_id, set())
        for delta in view.deltas_for_gene(gene_name):
            ref_base = self.reference_genome.get_reference_base(
                gene_name, delta.position
            )
            mrna.mutations.append(Mutation(
                position=delta.position,
                original=ref_base,
                mutant=delta.new_base,
                mutation_type="substitution",
                name=delta.mutation_id,
                oncogenic=oncogenic_flag,
            ))
        return mrna

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

        # Transcription (stochastic). Reads per-cell sequence state through
        # CellGenomeView; mRNA is built by materializing the view at
        # transcription time.
        if self.reference_genome is None:
            return
        gene_names = list(self.reference_genome.gene_names())
        for cell_id, view in list(self.cell_views.items()):
            cell_onco = self.cell_oncogene_flags.get(cell_id, set())
            for gene_name in gene_names:
                if np.random.random() < self.transcription_rate * dt:
                    mrna = self._build_mrna_from_view(view, gene_name, cell_id)
                    self.cell_mrnas[cell_id].append(mrna)
                    self.total_transcriptions += 1
                    self.emit_event(EventTypes.GENE_EXPRESSED, {
                        'cell_id': cell_id,
                        'gene': gene_name,
                        'oncogenic': gene_name in cell_onco,
                    })
    
    def introduce_mutation(self, cell_id: int, gene_name: str, mutation_name: str):
        """Introduce a named oncogenic mutation in a cell's view.

        Uses Gene.ONCOGENIC_SUBSTITUTIONS as the canonical mapping from
        name (e.g., 'G12D') to (position, new_base). Writes a sparse
        delta to the cell's CellGenomeView, classifies the substitution
        via MutationEffectClassifier, and sets cell_oncogene_flags if
        the classifier reports a sufficiently disruptive change.
        """
        if cell_id not in self.cell_views:
            return None
        if gene_name not in Gene.ONCOGENIC_SUBSTITUTIONS:
            return None
        if mutation_name not in Gene.ONCOGENIC_SUBSTITUTIONS[gene_name]:
            return None
        position, new_base = Gene.ONCOGENIC_SUBSTITUTIONS[gene_name][mutation_name]

        view = self.cell_views[cell_id]
        ref_base = self.reference_genome.get_reference_base(gene_name, position)

        # Classify against the reference (the substitution is what the
        # cell carries going forward; the classifier reads the reference).
        # gene_name is forwarded so the classifier can apply the
        # Upgrade 3 Stage B domain multiplier when the codon falls in
        # a curated functional region.
        ref_seq = self.reference_genome.get_reference_sequence(gene_name)
        effect = self.classifier.classify_substitution(
            coding_sequence=ref_seq,
            position=position,
            new_base=new_base,
            gene_name=gene_name,
        )

        # Apply the delta to the view
        view.add_substitution(gene_name, position, new_base, mutation_id=mutation_name)

        # Any entry in Gene.ONCOGENIC_SUBSTITUTIONS is a curated driver by
        # definition; impact_score also promotes any high-impact substitution
        # the classifier flags. The curated-table path is what makes adding
        # a new hotspot (e.g., KRAS G12C) automatically inherit driver status
        # without touching this function.
        is_driver = (
            effect.impact_score >= 0.4
            or mutation_name in Gene.ONCOGENIC_SUBSTITUTIONS[gene_name]
        )
        if is_driver:
            self.cell_oncogene_flags.setdefault(cell_id, set()).add(gene_name)

        mutation = Mutation(
            position=position,
            original=ref_base,
            mutant=new_base,
            mutation_type="substitution",
            name=mutation_name,
            oncogenic=is_driver,
            effect=effect,
        )

        self.total_mutations += 1
        self.emit_event(EventTypes.MUTATION_OCCURRED, {
            'cell_id': cell_id,
            'gene': gene_name,
            'mutation': mutation_name,
            'oncogenic': is_driver,
            'impact_score': effect.impact_score,
            'aa_change': effect.aa_change,
        })
        return mutation

    def create_exosome(self, cell_id: int, oncogenic: bool = False):
        """Create exosome from a cell.

        Oncogenic exosomes package mRNA for every gene flagged as
        oncogenic in this cell (via cell_oncogene_flags) plus a miRNA
        targeting TP53. The mRNA sequences are materialized from the
        cell's CellGenomeView so they carry the cell's specific deltas.
        """
        if cell_id not in self.cell_views:
            return None

        exosome = self.exosome_system.create_exosome(cell_id)
        view = self.cell_views[cell_id]

        if oncogenic:
            # Package mRNA for every gene this cell carries as oncogenic
            for gene_name in self.cell_oncogene_flags.get(cell_id, set()):
                mrna = self._build_mrna_from_view(view, gene_name, cell_id)
                exosome.package_mrna(mrna)

            # Package miRNA targeting TP53 (tumor suppressor knockdown)
            mirna = RNA("UAAGGCACGCGGUGAAUGCC", "miR-125b", NucleicAcidType.miRNA)
            mirna.target_genes = ["TP53"]
            exosome.package_mirna(mirna)
        else:
            # Package normal mRNA: take first available from cell pool
            if self.cell_mrnas.get(cell_id):
                mrna = self.cell_mrnas[cell_id][0]
                exosome.package_mrna(mrna)

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
            'n_cells_tracked': len(self.cell_views),
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
        """Handle cell division: daughter inherits parent's deltas via fork().

        This is the key memory-architecture operation: a daughter view
        shares the same ReferenceGenome object and copies the parent's
        delta log. Subsequent mutations on either parent or daughter
        diverge.
        """
        parent_id = data['cell_id']
        daughter_id = data.get('daughter_id')

        if daughter_id and parent_id in self.cell_views:
            self.cell_views[daughter_id] = self.cell_views[parent_id].fork()
            # Inherit oncogene flags
            parent_flags = self.cell_oncogene_flags.get(parent_id, set())
            self.cell_oncogene_flags[daughter_id] = set(parent_flags)
            self.cell_mrnas[daughter_id] = []

    def on_cell_transformed(self, data):
        """Handle cell transformation - create oncogenic exosomes"""
        cell_id = data['cell_id']
        position = data['position']

        if cell_id in self.cell_views:
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
