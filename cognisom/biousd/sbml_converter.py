"""
SBML-to-USD Converter (Phase 6)
================================

Parses SBML XML (Systems Biology Markup Language) and converts biological
models to Bio-USD scene format (.usda).

SBML species -> BioMolecule / BioProtein prims
SBML reactions -> BioInteractionAPI relationships
SBML compartments -> BioTissue prims
SBML parameters -> custom data on prims

Supports SBML Level 2/3 using Python's xml.etree (no libsbml dependency).
If python-libsbml is available, it is used for richer parsing.

Usage::

    from cognisom.biousd.sbml_converter import SBMLToUSD

    converter = SBMLToUSD()
    scene = converter.from_file("model.xml")
    converter.export_usda("model.usda")

    # Or from string
    scene = converter.from_string(sbml_xml_string)
"""

from __future__ import annotations

import logging
import re
import textwrap
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .schema import (
    BioCell,
    BioGene,
    BioInteractionAPI,
    BioMetabolicAPI,
    BioMolecule,
    BioProtein,
    BioScene,
    BioTissue,
    CellType,
    GeneType,
)

log = logging.getLogger(__name__)

# SBML namespaces
SBML_NS = {
    "sbml2": "http://www.sbml.org/sbml/level2/version4",
    "sbml3": "http://www.sbml.org/sbml/level3/version2/core",
    "sbml3v1": "http://www.sbml.org/sbml/level3/version1/core",
    "mathml": "http://www.w3.org/1998/Math/MathML",
}


@dataclass
class SBMLSpecies:
    """Parsed SBML species."""
    id: str = ""
    name: str = ""
    compartment: str = ""
    initial_amount: float = 0.0
    initial_concentration: float = 0.0
    boundary_condition: bool = False
    constant: bool = False


@dataclass
class SBMLReaction:
    """Parsed SBML reaction."""
    id: str = ""
    name: str = ""
    reversible: bool = False
    reactants: List[Tuple[str, float]] = field(default_factory=list)  # (species_id, stoichiometry)
    products: List[Tuple[str, float]] = field(default_factory=list)
    modifiers: List[str] = field(default_factory=list)
    parameters: Dict[str, float] = field(default_factory=dict)


@dataclass
class SBMLCompartment:
    """Parsed SBML compartment."""
    id: str = ""
    name: str = ""
    spatial_dimensions: int = 3
    size: float = 1.0


@dataclass
class SBMLModel:
    """Complete parsed SBML model."""
    model_id: str = ""
    model_name: str = ""
    level: int = 3
    version: int = 2
    compartments: List[SBMLCompartment] = field(default_factory=list)
    species: List[SBMLSpecies] = field(default_factory=list)
    reactions: List[SBMLReaction] = field(default_factory=list)
    parameters: Dict[str, float] = field(default_factory=dict)


class SBMLToUSD:
    """Convert SBML models to Bio-USD scene format."""

    def __init__(self):
        self._model: Optional[SBMLModel] = None
        self._scene: Optional[BioScene] = None

    # ── Parsing ──────────────────────────────────────────────────────

    def from_file(self, filepath: str) -> BioScene:
        """Parse an SBML file and convert to BioScene."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"SBML file not found: {filepath}")
        return self.from_string(path.read_text())

    def from_string(self, sbml_xml: str) -> BioScene:
        """Parse an SBML XML string and convert to BioScene."""
        self._model = self._parse_sbml(sbml_xml)
        self._scene = self._convert_to_scene(self._model)
        return self._scene

    def _parse_sbml(self, xml_str: str) -> SBMLModel:
        """Parse SBML XML into intermediate SBMLModel."""
        root = ET.fromstring(xml_str)

        # Detect namespace
        ns = ""
        root_tag = root.tag
        if "}" in root_tag:
            ns = root_tag.split("}")[0] + "}"

        # Get level/version
        level = int(root.get("level", "3"))
        version = int(root.get("version", "2"))

        model_elem = root.find(f"{ns}model")
        if model_elem is None:
            raise ValueError("No <model> element found in SBML")

        model = SBMLModel(
            model_id=model_elem.get("id", ""),
            model_name=model_elem.get("name", ""),
            level=level,
            version=version,
        )

        # Parse compartments
        for comp in model_elem.findall(f".//{ns}compartment"):
            model.compartments.append(SBMLCompartment(
                id=comp.get("id", ""),
                name=comp.get("name", comp.get("id", "")),
                spatial_dimensions=int(comp.get("spatialDimensions", "3")),
                size=float(comp.get("size", "1.0")),
            ))

        # Parse species
        for spec in model_elem.findall(f".//{ns}species"):
            model.species.append(SBMLSpecies(
                id=spec.get("id", ""),
                name=spec.get("name", spec.get("id", "")),
                compartment=spec.get("compartment", ""),
                initial_amount=float(spec.get("initialAmount", "0")),
                initial_concentration=float(spec.get("initialConcentration", "0")),
                boundary_condition=spec.get("boundaryCondition", "false").lower() == "true",
                constant=spec.get("constant", "false").lower() == "true",
            ))

        # Parse global parameters
        for param in model_elem.findall(f".//{ns}listOfParameters/{ns}parameter"):
            pid = param.get("id", "")
            val = float(param.get("value", "0"))
            model.parameters[pid] = val

        # Parse reactions
        for rxn in model_elem.findall(f".//{ns}reaction"):
            reaction = SBMLReaction(
                id=rxn.get("id", ""),
                name=rxn.get("name", rxn.get("id", "")),
                reversible=rxn.get("reversible", "false").lower() == "true",
            )

            for ref in rxn.findall(f".//{ns}listOfReactants/{ns}speciesReference"):
                species_id = ref.get("species", "")
                stoich = float(ref.get("stoichiometry", "1"))
                reaction.reactants.append((species_id, stoich))

            for ref in rxn.findall(f".//{ns}listOfProducts/{ns}speciesReference"):
                species_id = ref.get("species", "")
                stoich = float(ref.get("stoichiometry", "1"))
                reaction.products.append((species_id, stoich))

            for mod in rxn.findall(f".//{ns}listOfModifiers/{ns}modifierSpeciesReference"):
                reaction.modifiers.append(mod.get("species", ""))

            # Extract local parameters from kinetic law
            for local_param in rxn.findall(f".//{ns}kineticLaw/{ns}listOfLocalParameters/{ns}localParameter"):
                reaction.parameters[local_param.get("id", "")] = float(local_param.get("value", "0"))

            # Also check older SBML format
            for local_param in rxn.findall(f".//{ns}kineticLaw/{ns}listOfParameters/{ns}parameter"):
                reaction.parameters[local_param.get("id", "")] = float(local_param.get("value", "0"))

            model.reactions.append(reaction)

        log.info(
            "Parsed SBML model '%s': %d compartments, %d species, %d reactions, %d parameters",
            model.model_name, len(model.compartments), len(model.species),
            len(model.reactions), len(model.parameters),
        )
        return model

    # ── Conversion to Bio-USD ────────────────────────────────────────

    def _convert_to_scene(self, model: SBMLModel) -> BioScene:
        """Convert SBMLModel to BioScene."""
        scene = BioScene()

        # Compartments -> BioTissue
        for comp in model.compartments:
            tissue = BioTissue(
                prim_path=f"/World/Compartments/{_safe_prim_name(comp.id)}",
                display_name=comp.name or comp.id,
                tissue_type=comp.id,
            )
            scene.tissues.append(tissue)

        # Species -> categorize and create prims
        for species in model.species:
            category = self._categorize_species(species)

            if category == "gene":
                gene = BioGene(
                    prim_path=f"/World/Genes/{_safe_prim_name(species.id)}",
                    display_name=species.name or species.id,
                    gene_name=species.id,
                    gene_type=GeneType.HOUSEKEEPING,
                    expression_level=species.initial_amount or species.initial_concentration,
                )
                scene.genes.append(gene)

            elif category == "protein":
                protein = BioProtein(
                    prim_path=f"/World/Proteins/{_safe_prim_name(species.id)}",
                    display_name=species.name or species.id,
                    protein_name=species.id,
                )
                scene.proteins.append(protein)

            else:
                # Default: BioMolecule
                mol = BioMolecule(
                    prim_path=f"/World/Molecules/{_safe_prim_name(species.id)}",
                    display_name=species.name or species.id,
                )
                scene.molecules.append(mol)

        # Reactions -> BioInteractionAPI relationships recorded as custom data
        # (relationships between prims are expressed via prim paths)

        return scene

    def _categorize_species(self, species: SBMLSpecies) -> str:
        """Categorize an SBML species as gene, protein, or molecule."""
        name = (species.name or species.id).lower()

        # Gene indicators
        gene_patterns = ["dna", "gene", "mrna", "m_rna", "transcript"]
        if any(p in name for p in gene_patterns):
            return "gene"

        # Protein indicators
        protein_patterns = ["protein", "enzyme", "kinase", "receptor", "phosphatase",
                           "ligase", "polymerase", "factor"]
        if any(p in name for p in protein_patterns):
            return "protein"

        # Default to molecule
        return "molecule"

    # ── Export ────────────────────────────────────────────────────────

    def export_usda(self, filepath: str) -> str:
        """Export the current scene to a .usda file."""
        if self._scene is None:
            raise ValueError("No scene to export. Call from_file() or from_string() first.")
        if self._model is None:
            raise ValueError("No model loaded")

        usda = self._render_usda(self._scene, self._model)
        Path(filepath).write_text(usda)
        log.info("Exported SBML-to-USD: %s", filepath)
        return usda

    def to_usda_string(self) -> str:
        """Return USDA string without writing to file."""
        if self._scene is None or self._model is None:
            raise ValueError("No scene/model loaded")
        return self._render_usda(self._scene, self._model)

    def _render_usda(self, scene: BioScene, model: SBMLModel) -> str:
        """Render BioScene to USDA text format."""
        lines = [
            '#usda 1.0',
            '(',
            f'    doc = "Bio-USD scene converted from SBML model: {model.model_name}"',
            f'    customLayerData = {{',
            f'        string source = "SBML"',
            f'        string sbml_model_id = "{model.model_id}"',
            f'        int sbml_level = {model.level}',
            f'        int sbml_version = {model.version}',
            f'    }}',
            '    metersPerUnit = 1e-6',
            '    upAxis = "Y"',
            ')',
            '',
            'def Xform "World" {',
        ]

        # Global parameters as custom data
        if model.parameters:
            lines.append('    custom dictionary sbml_parameters = {')
            for k, v in model.parameters.items():
                lines.append(f'        double {k} = {v}')
            lines.append('    }')
            lines.append('')

        # Compartments
        if scene.tissues:
            lines.append('    def Xform "Compartments" {')
            for tissue in scene.tissues:
                name = _safe_prim_name(tissue.tissue_type)
                comp = next((c for c in model.compartments if c.id == tissue.tissue_type), None)
                lines.append(f'        def BioTissue "{name}" {{')
                lines.append(f'            string bio:displayName = "{tissue.display_name}"')
                lines.append(f'            token bio:tissueType = "{tissue.tissue_type}"')
                if comp:
                    lines.append(f'            int bio:spatialDimensions = {comp.spatial_dimensions}')
                    lines.append(f'            double bio:size = {comp.size}')
                lines.append('        }')
            lines.append('    }')
            lines.append('')

        # Genes
        if scene.genes:
            lines.append('    def Xform "Genes" {')
            for gene in scene.genes:
                name = _safe_prim_name(gene.gene_name)
                lines.append(f'        def BioGene "{name}" {{')
                lines.append(f'            string bio:geneName = "{gene.gene_name}"')
                lines.append(f'            string bio:displayName = "{gene.display_name}"')
                lines.append(f'            float bio:expressionLevel = {gene.expression_level}')
                lines.append('        }')
            lines.append('    }')
            lines.append('')

        # Proteins
        if scene.proteins:
            lines.append('    def Xform "Proteins" {')
            for prot in scene.proteins:
                name = _safe_prim_name(prot.protein_name)
                lines.append(f'        def BioProtein "{name}" {{')
                lines.append(f'            string bio:proteinName = "{prot.protein_name}"')
                lines.append(f'            string bio:displayName = "{prot.display_name}"')
                lines.append('        }')
            lines.append('    }')
            lines.append('')

        # Molecules
        if scene.molecules:
            lines.append('    def Xform "Molecules" {')
            for mol in scene.molecules:
                name = _safe_prim_name(mol.display_name)
                lines.append(f'        def BioMolecule "{name}" {{')
                lines.append(f'            string bio:displayName = "{mol.display_name}"')
                lines.append('        }')
            lines.append('    }')
            lines.append('')

        # Reactions as relationship metadata
        if model.reactions:
            lines.append('    def Xform "Reactions" {')
            for rxn in model.reactions:
                rxn_name = _safe_prim_name(rxn.id)
                lines.append(f'        def Xform "{rxn_name}" {{')
                lines.append(f'            string bio:displayName = "{rxn.name}"')
                lines.append(f'            bool bio:reversible = {str(rxn.reversible).lower()}')

                # Reactant relationships
                for species_id, stoich in rxn.reactants:
                    prim = _safe_prim_name(species_id)
                    lines.append(f'            custom string bio:reactant_{prim} = "{species_id}"')
                    lines.append(f'            custom double bio:stoich_reactant_{prim} = {stoich}')

                # Product relationships
                for species_id, stoich in rxn.products:
                    prim = _safe_prim_name(species_id)
                    lines.append(f'            custom string bio:product_{prim} = "{species_id}"')
                    lines.append(f'            custom double bio:stoich_product_{prim} = {stoich}')

                # Local parameters
                for k, v in rxn.parameters.items():
                    lines.append(f'            custom double bio:param_{k} = {v}')

                lines.append('        }')
            lines.append('    }')
            lines.append('')

        lines.append('}')
        return '\n'.join(lines) + '\n'

    @property
    def model(self) -> Optional[SBMLModel]:
        return self._model

    @property
    def scene(self) -> Optional[BioScene]:
        return self._scene


def _safe_prim_name(name: str) -> str:
    """Convert a string to a valid USD prim name."""
    # Replace invalid characters with underscores
    safe = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    # Ensure it starts with a letter
    if safe and not safe[0].isalpha():
        safe = "N_" + safe
    return safe or "Unknown"
