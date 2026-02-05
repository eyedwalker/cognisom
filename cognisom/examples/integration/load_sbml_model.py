#!/usr/bin/env python3
"""
Example: Load SBML Model from BioModels
========================================

Shows how to:
1. Download a published model from BioModels database
2. Parse SBML file
3. Extract species, reactions, parameters
4. Integrate into cognisom

This example uses a simple gene expression model.
"""

import sys
from pathlib import Path

# Add engine to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

print("=" * 70)
print("🧬 SBML Model Integration Example")
print("=" * 70)
print()

# Check if libsbml is installed
try:
    import libsbml
    print("✓ python-libsbml is installed")
except ImportError:
    print("⚠️  python-libsbml not installed")
    print()
    print("To install:")
    print("  pip install python-libsbml")
    print()
    print("After installing, run this script again.")
    sys.exit(1)

print()
print("=" * 70)
print("📥 Downloading Model from BioModels Database")
print("=" * 70)
print()

# Simple gene expression model (SBML example)
# In practice, you'd download from: https://www.ebi.ac.uk/biomodels/

# Create a simple SBML model for demonstration
sbml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="gene_expression" name="Simple Gene Expression">
    <listOfCompartments>
      <compartment id="cell" spatialDimensions="3" size="1" constant="true"/>
    </listOfCompartments>
    
    <listOfSpecies>
      <species id="DNA" compartment="cell" initialAmount="1" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="mRNA" compartment="cell" initialAmount="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="Protein" compartment="cell" initialAmount="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    
    <listOfParameters>
      <parameter id="k_transcription" value="0.1" constant="true"/>
      <parameter id="k_translation" value="10.0" constant="true"/>
      <parameter id="k_mrna_decay" value="0.05" constant="true"/>
      <parameter id="k_protein_decay" value="0.01" constant="true"/>
    </listOfParameters>
    
    <listOfReactions>
      <reaction id="transcription" reversible="false">
        <listOfReactants>
          <speciesReference species="DNA" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="DNA" stoichiometry="1" constant="true"/>
          <speciesReference species="mRNA" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply>
              <times/>
              <ci> k_transcription </ci>
              <ci> DNA </ci>
            </apply>
          </math>
        </kineticLaw>
      </reaction>
      
      <reaction id="translation" reversible="false">
        <listOfReactants>
          <speciesReference species="mRNA" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="mRNA" stoichiometry="1" constant="true"/>
          <speciesReference species="Protein" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply>
              <times/>
              <ci> k_translation </ci>
              <ci> mRNA </ci>
            </apply>
          </math>
        </kineticLaw>
      </reaction>
      
      <reaction id="mrna_degradation" reversible="false">
        <listOfReactants>
          <speciesReference species="mRNA" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply>
              <times/>
              <ci> k_mrna_decay </ci>
              <ci> mRNA </ci>
            </apply>
          </math>
        </kineticLaw>
      </reaction>
      
      <reaction id="protein_degradation" reversible="false">
        <listOfReactants>
          <speciesReference species="Protein" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply>
              <times/>
              <ci> k_protein_decay </ci>
              <ci> Protein </ci>
            </apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>'''

# Save to file
sbml_file = Path('models/pathways/gene_expression.xml')
sbml_file.parent.mkdir(parents=True, exist_ok=True)
with open(sbml_file, 'w') as f:
    f.write(sbml_content)

print(f"✓ Created example SBML file: {sbml_file}")
print()

# Load SBML model
print("=" * 70)
print("📖 Parsing SBML Model")
print("=" * 70)
print()

reader = libsbml.SBMLReader()
document = reader.readSBML(str(sbml_file))

if document.getNumErrors() > 0:
    print("⚠️  Errors in SBML file:")
    document.printErrors()
    sys.exit(1)

model = document.getModel()
print(f"✓ Loaded model: {model.getName()}")
print(f"  Model ID: {model.getId()}")
print()

# Extract species
print("📊 Species in Model:")
print("-" * 70)
for i in range(model.getNumSpecies()):
    species = model.getSpecies(i)
    print(f"  {species.getId():15s} = {species.getInitialAmount():8.1f}")
print()

# Extract parameters
print("⚙️  Parameters:")
print("-" * 70)
for i in range(model.getNumParameters()):
    param = model.getParameter(i)
    print(f"  {param.getId():20s} = {param.getValue():8.3f}")
print()

# Extract reactions
print("⚡ Reactions:")
print("-" * 70)
for i in range(model.getNumReactions()):
    reaction = model.getReaction(i)
    
    # Get reactants
    reactants = []
    for j in range(reaction.getNumReactants()):
        ref = reaction.getReactant(j)
        reactants.append(ref.getSpecies())
    
    # Get products
    products = []
    for j in range(reaction.getNumProducts()):
        ref = reaction.getProduct(j)
        products.append(ref.getSpecies())
    
    reactants_str = " + ".join(reactants) if reactants else "∅"
    products_str = " + ".join(products) if products else "∅"
    
    print(f"  {reaction.getId():20s}: {reactants_str} → {products_str}")
print()

print("=" * 70)
print("🔗 Integration with cognisom")
print("=" * 70)
print()

print("This model can be integrated into cognisom by:")
print("  1. Creating Gene objects from species")
print("  2. Using parameters for transcription/translation rates")
print("  3. Implementing reactions as state transitions")
print()

print("Example integration code:")
print("-" * 70)
print("""
from engine.py.intracellular import IntracellularModel, Gene

# Create cell
cell = IntracellularModel()

# Add gene from SBML
cell.add_gene(Gene(
    name='GeneX',
    sequence_length=1200,
    transcription_rate=0.1,  # From k_transcription
    promoter_strength=1.0
))

# Set translation rate from SBML
if 'GeneX' in cell.mrnas:
    cell.mrnas['GeneX'].translation_rate = 10.0  # From k_translation

# Set degradation rates
if 'GeneX' in cell.mrnas:
    cell.mrnas['GeneX'].half_life = 0.693 / 0.05  # From k_mrna_decay

if 'GeneX' in cell.proteins:
    cell.proteins['GeneX'].half_life = 0.693 / 0.01  # From k_protein_decay
""")
print("-" * 70)
print()

print("=" * 70)
print("✓ SBML Integration Example Complete!")
print("=" * 70)
print()
print("Next steps:")
print("  1. Download real models from: https://www.ebi.ac.uk/biomodels/")
print("  2. Popular models:")
print("     - BIOMD0000000010: MAPK cascade")
print("     - BIOMD0000000028: p53 oscillations")
print("     - BIOMD0000000190: Cell cycle")
print("  3. Install more tools:")
print("     pip install cobra tellurium gillespy2")
print()
