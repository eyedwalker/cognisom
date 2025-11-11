#!/usr/bin/env python3
"""
Test All Features - Comprehensive Demo
=======================================

Shows everything cognisom can do RIGHT NOW:
1. Basic cell simulation
2. Detailed intracellular dynamics
3. SBML model loading
4. Metabolic modeling (cobra)
5. Stochastic simulation (gillespy2)
6. Gene sequences (biopython)
"""

import sys
from pathlib import Path
import time

# Add engine to path
sys.path.insert(0, str(Path(__file__).parent))

def print_section(title):
    """Print section header"""
    print("\n" + "=" * 70)
    print(f"🧬 {title}")
    print("=" * 70 + "\n")

def test_basic_simulation():
    """Test 1: Basic cell growth and division"""
    print_section("TEST 1: Basic Cell Growth & Division")
    
    from engine.py.cell import Cell
    from engine.py.simulation import Simulation
    
    print("Creating a single cell...")
    cell = Cell()
    print(f"✓ Cell created: {cell}")
    print(f"  Initial proteins: {cell.state.species_counts[1]}")
    print()
    
    print("Running 6-hour simulation...")
    sim = Simulation(
        initial_cells=[cell],
        duration=6.0,
        dt=0.01,
        output_dir='./output/test_basic'
    )
    
    start = time.time()
    sim.run(verbose=False)
    elapsed = time.time() - start
    
    results = sim.get_results()
    print(f"✓ Simulation complete in {elapsed:.2f}s")
    print(f"  Final cells: {results['final_cells']}")
    print(f"  Divisions: {results['events']['divisions']}")
    print(f"  Deaths: {results['events']['deaths']}")
    print(f"  Speed: {int(6.0/0.01/elapsed):,} steps/second")
    
    return True

def test_intracellular():
    """Test 2: Detailed intracellular dynamics"""
    print_section("TEST 2: Detailed Intracellular Dynamics")
    
    from engine.py.intracellular import IntracellularModel
    
    print("Creating cell with detailed internals...")
    cell = IntracellularModel()
    
    state = cell.get_state_summary()
    print(f"✓ Cell created with:")
    print(f"  Genes: {state['genes']}")
    print(f"  Active genes: {state['active_genes']}")
    print(f"  Ribosomes: {state['ribosomes']}")
    print(f"  ATP: {state['atp']:,}")
    print()
    
    print("Genes in genome:")
    for gene_name in list(cell.genes.keys())[:5]:
        gene = cell.genes[gene_name]
        print(f"  • {gene_name:10s} - {gene.sequence_length:,} bp")
    print()
    
    print("Running 3-hour simulation...")
    for i in range(300):  # 3 hours
        cell.step(dt=0.01)
    
    final_state = cell.get_state_summary()
    print(f"✓ Simulation complete")
    print(f"  mRNA species: {final_state['mrna_species']}")
    print(f"  Total mRNA: {final_state['total_mrna']}")
    print(f"  Protein species: {final_state['protein_species']}")
    print(f"  Total proteins: {final_state['total_proteins']:,}")
    
    return True

def test_sbml():
    """Test 3: SBML model loading"""
    print_section("TEST 3: SBML Model Loading")
    
    try:
        import libsbml
        print("✓ libsbml available")
    except ImportError:
        print("✗ libsbml not installed")
        return False
    
    # Check if example SBML file exists
    sbml_file = Path('models/pathways/gene_expression.xml')
    if not sbml_file.exists():
        print(f"⚠️  SBML file not found: {sbml_file}")
        print("   Run: python3 examples/integration/load_sbml_model.py")
        return False
    
    print(f"Loading SBML model: {sbml_file}")
    reader = libsbml.SBMLReader()
    document = reader.readSBML(str(sbml_file))
    model = document.getModel()
    
    print(f"✓ Model loaded: {model.getName()}")
    print(f"  Species: {model.getNumSpecies()}")
    print(f"  Reactions: {model.getNumReactions()}")
    print(f"  Parameters: {model.getNumParameters()}")
    print()
    
    print("Species:")
    for i in range(model.getNumSpecies()):
        species = model.getSpecies(i)
        print(f"  • {species.getId():15s} = {species.getInitialAmount()}")
    
    return True

def test_cobra():
    """Test 4: Metabolic modeling"""
    print_section("TEST 4: Metabolic Modeling (COBRApy)")
    
    try:
        import cobra
        print(f"✓ cobra version {cobra.__version__}")
    except ImportError:
        print("✗ cobra not installed")
        return False
    
    print("\nLoading E. coli core metabolic model...")
    try:
        model = cobra.io.load_model("e_coli_core")
        print(f"✓ Model loaded: {model.id}")
        print(f"  Reactions: {len(model.reactions)}")
        print(f"  Metabolites: {len(model.metabolites)}")
        print(f"  Genes: {len(model.genes)}")
        print()
        
        print("Running flux balance analysis (FBA)...")
        solution = model.optimize()
        print(f"✓ Optimization complete")
        print(f"  Status: {solution.status}")
        print(f"  Growth rate: {solution.objective_value:.3f} /hour")
        print()
        
        print("Top 5 metabolic fluxes:")
        top_fluxes = solution.fluxes.abs().nlargest(5)
        for rxn_id, flux in top_fluxes.items():
            rxn = model.reactions.get_by_id(rxn_id)
            print(f"  • {rxn_id:20s}: {flux:8.2f} mmol/gDW/h")
            print(f"    {rxn.reaction}")
        
        return True
    except Exception as e:
        print(f"⚠️  Error loading model: {e}")
        return False

def test_gillespy2():
    """Test 5: Stochastic simulation"""
    print_section("TEST 5: Stochastic Simulation (GillesPy2)")
    
    try:
        import gillespy2
        print(f"✓ gillespy2 version {gillespy2.__version__}")
    except ImportError:
        print("✗ gillespy2 not installed")
        return False
    
    print("\nCreating simple gene expression model...")
    
    class SimpleGene(gillespy2.Model):
        def __init__(self):
            gillespy2.Model.__init__(self, name="simple_gene")
            
            # Species
            DNA = gillespy2.Species(name='DNA', initial_value=1)
            mRNA = gillespy2.Species(name='mRNA', initial_value=0)
            Protein = gillespy2.Species(name='Protein', initial_value=0)
            
            self.add_species([DNA, mRNA, Protein])
            
            # Reactions
            transcription = gillespy2.Reaction(
                name='transcription',
                reactants={'DNA': 1},
                products={'DNA': 1, 'mRNA': 1},
                rate=gillespy2.Parameter(name='k_tx', expression=0.1)
            )
            
            translation = gillespy2.Reaction(
                name='translation',
                reactants={'mRNA': 1},
                products={'mRNA': 1, 'Protein': 1},
                rate=gillespy2.Parameter(name='k_tl', expression=10.0)
            )
            
            mrna_decay = gillespy2.Reaction(
                name='mrna_decay',
                reactants={'mRNA': 1},
                products={},
                rate=gillespy2.Parameter(name='k_decay', expression=0.05)
            )
            
            self.add_reaction([transcription, translation, mrna_decay])
            self.timespan(range(0, 11))
    
    model = SimpleGene()
    print(f"✓ Model created: {model.name}")
    print(f"  Species: {len(model.listOfSpecies)}")
    print(f"  Reactions: {len(model.listOfReactions)}")
    print()
    
    print("Running stochastic simulation (10 hours)...")
    results = model.run()
    
    # Get final values
    final_mrna = results['mRNA'][-1]
    final_protein = results['Protein'][-1]
    
    print(f"✓ Simulation complete")
    print(f"  Final mRNA: {final_mrna}")
    print(f"  Final Protein: {final_protein}")
    
    return True

def test_biopython():
    """Test 6: Gene sequences"""
    print_section("TEST 6: Gene Sequences (BioPython)")
    
    try:
        from Bio.Seq import Seq
        from Bio import SeqIO
        import Bio
        print(f"✓ biopython version {Bio.__version__}")
    except ImportError:
        print("✗ biopython not installed")
        return False
    
    print("\nCreating DNA sequence...")
    dna = Seq("ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG")
    print(f"✓ DNA sequence: {dna}")
    print(f"  Length: {len(dna)} bp")
    print()
    
    print("Transcribing to RNA...")
    rna = dna.transcribe()
    print(f"✓ RNA sequence: {rna}")
    print()
    
    print("Translating to protein...")
    protein = dna.translate()
    print(f"✓ Protein sequence: {protein}")
    print(f"  Length: {len(protein)} amino acids")
    print()
    
    print("Sequence analysis:")
    print(f"  GC content: {(dna.count('G') + dna.count('C')) / len(dna) * 100:.1f}%")
    print(f"  Complement: {dna.complement()}")
    print(f"  Reverse complement: {dna.reverse_complement()}")
    
    return True

def test_visualization():
    """Test 7: Visualization"""
    print_section("TEST 7: Visualization")
    
    try:
        import matplotlib.pyplot as plt
        print("✓ matplotlib available")
    except ImportError:
        print("✗ matplotlib not installed")
        return False
    
    from engine.py.intracellular import IntracellularModel
    from engine.py.visualize import CellVisualizer
    
    print("\nCreating cell and visualizer...")
    cell = IntracellularModel()
    viz = CellVisualizer(cell)
    
    print("✓ Visualizer created")
    print("  Can generate:")
    print("  • Cell structure diagram")
    print("  • Molecular dynamics plots")
    print("  • Gene expression profiles")
    print("  • Protein abundance charts")
    print()
    
    print("Example visualization saved in:")
    print("  output/intracellular/intracellular_dashboard.png")
    
    return True

def main():
    """Run all tests"""
    print("=" * 70)
    print("🧬 cognisom Comprehensive Feature Test")
    print("=" * 70)
    print()
    print("Testing all capabilities...")
    print()
    
    tests = [
        ("Basic Cell Simulation", test_basic_simulation),
        ("Intracellular Dynamics", test_intracellular),
        ("SBML Model Loading", test_sbml),
        ("Metabolic Modeling", test_cobra),
        ("Stochastic Simulation", test_gillespy2),
        ("Gene Sequences", test_biopython),
        ("Visualization", test_visualization),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n⚠️  Error in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print_section("Summary")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}\n")
    
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {name}")
    
    print()
    print("=" * 70)
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
    else:
        print(f"⚠️  {total - passed} test(s) failed or skipped")
    
    print("=" * 70)
    print()
    
    print("What you can do now:")
    print("  1. Run simulations: python3 examples/single_cell/basic_growth.py")
    print("  2. View internals: python3 examples/single_cell/intracellular_detail.py")
    print("  3. Load SBML models: python3 examples/integration/load_sbml_model.py")
    print("  4. Explore metabolism: import cobra; cobra.io.load_model('e_coli_core')")
    print("  5. Read documentation: open README.md")
    print()

if __name__ == '__main__':
    main()
