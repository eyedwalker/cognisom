"""
Bio-USD Converter
=================

Exports Cognisom simulation state to OpenUSD-compliant .usda files.

Usage::

    from cognisom.biousd.converter import SimulationToUSD
    from cognisom.core.simulation_engine import SimulationEngine

    engine = SimulationEngine()
    engine.step(100)

    converter = SimulationToUSD(engine)
    converter.export("snapshot_t100.usda")

The converter can also produce a BioScene Python object for
programmatic inspection without writing to disk.
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path
from typing import List, Optional

import numpy as np

from .schema import (
    BioCapillary,
    BioCell,
    BioEpigeneticAPI,
    BioExosome,
    BioGene,
    BioGeneExpressionAPI,
    BioImmuneCell,
    BioMetabolicAPI,
    BioMolecule,
    BioProtein,
    BioScene,
    BioSpatialField,
    BioTissue,
    CellPhase,
    CellType,
    GeneType,
    ImmuneCellType,
    SpatialFieldType,
)

log = logging.getLogger(__name__)


class SimulationToUSD:
    """Convert Cognisom simulation engine state to Bio-USD format.

    This reads the current state of a SimulationEngine and produces
    either a BioScene object or a .usda text file.
    """

    def __init__(self, engine=None) -> None:
        """Initialise with an optional simulation engine.

        Args:
            engine: A cognisom.core.simulation_engine.SimulationEngine
                    instance. If None, you can still use manual methods.
        """
        self._engine = engine

    # ── High-level API ────────────────────────────────────────────────

    def to_scene(self) -> BioScene:
        """Convert current engine state to a BioScene object."""
        if self._engine is None:
            raise ValueError("No simulation engine provided")

        scene = BioScene(
            simulation_time=getattr(self._engine, "time", 0.0),
            time_step=getattr(self._engine, "dt", 0.1),
            step_count=getattr(self._engine, "step_count", 0),
        )

        # Extract cells from cellular module
        cellular = self._engine.modules.get("cellular")
        if cellular is not None:
            scene.cells = self._extract_cells(cellular)

        # Extract immune cells
        immune = self._engine.modules.get("immune")
        if immune is not None:
            scene.immune_cells = self._extract_immune_cells(immune)

        # Extract spatial fields
        spatial = self._engine.modules.get("spatial")
        if spatial is not None:
            scene.spatial_fields = self._extract_spatial_fields(spatial)

        # Extract vascular network
        vascular = self._engine.modules.get("vascular")
        if vascular is not None:
            scene.capillaries = self._extract_capillaries(vascular)

        # Extract molecular data
        molecular = self._engine.modules.get("molecular")
        if molecular is not None:
            scene.genes = self._extract_genes(molecular)

        return scene

    def export(self, output_path: str, scene: Optional[BioScene] = None) -> str:
        """Export to a .usda text file.

        Args:
            output_path: File path for the .usda output
            scene: BioScene to export. If None, generates from engine.

        Returns:
            The output file path.
        """
        if scene is None:
            scene = self.to_scene()

        usda_text = self.scene_to_usda(scene)

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(usda_text)
        log.info("Exported Bio-USD scene to %s (%d cells, %.1f hours)",
                 output_path, scene.total_cells, scene.simulation_time)
        return str(path)

    # ── USDA Serialisation ────────────────────────────────────────────

    def scene_to_usda(self, scene: BioScene) -> str:
        """Convert a BioScene to USDA text format."""
        lines = [
            '#usda 1.0',
            '(',
            '    """',
            '    Bio-USD Scene — Cognisom Simulation Snapshot',
            f'    Time: {scene.simulation_time:.2f} hours (step {scene.step_count})',
            f'    Cells: {scene.total_cells} ({scene.alive_cells} alive)',
            '    """',
            '    defaultPrim = "BioScene"',
            f'    customLayerData = {{',
            f'        float bio:simulationTime = {scene.simulation_time}',
            f'        float bio:timeStep = {scene.time_step}',
            f'        int bio:stepCount = {scene.step_count}',
            f'        string bio:version = "0.1.0"',
            f'        string bio:engine = "cognisom"',
            f'    }}',
            ')',
            '',
            'def Xform "BioScene" {',
        ]

        # Cells
        if scene.cells:
            lines.append('    def Scope "Cells" {')
            for cell in scene.cells:
                lines.extend(self._cell_to_usda(cell, indent=8))
            lines.append('    }')
            lines.append('')

        # Immune cells
        if scene.immune_cells:
            lines.append('    def Scope "ImmuneCells" {')
            for cell in scene.immune_cells:
                lines.extend(self._immune_cell_to_usda(cell, indent=8))
            lines.append('    }')
            lines.append('')

        # Spatial fields
        if scene.spatial_fields:
            lines.append('    def Scope "SpatialFields" {')
            for field_obj in scene.spatial_fields:
                lines.extend(self._spatial_field_to_usda(field_obj, indent=8))
            lines.append('    }')
            lines.append('')

        # Capillaries
        if scene.capillaries:
            lines.append('    def Scope "VascularNetwork" {')
            for cap in scene.capillaries:
                lines.extend(self._capillary_to_usda(cap, indent=8))
            lines.append('    }')
            lines.append('')

        # Genes
        if scene.genes:
            lines.append('    def Scope "Genes" {')
            for gene in scene.genes:
                lines.extend(self._gene_to_usda(gene, indent=8))
            lines.append('    }')
            lines.append('')

        # Proteins
        if scene.proteins:
            lines.append('    def Scope "Proteins" {')
            for protein in scene.proteins:
                lines.extend(self._protein_to_usda(protein, indent=8))
            lines.append('    }')
            lines.append('')

        # Molecules
        if scene.molecules:
            lines.append('    def Scope "Molecules" {')
            for mol in scene.molecules:
                lines.extend(self._molecule_to_usda(mol, indent=8))
            lines.append('    }')
            lines.append('')

        # Tissues
        if scene.tissues:
            lines.append('    def Scope "Tissues" {')
            for tissue in scene.tissues:
                lines.extend(self._tissue_to_usda(tissue, indent=8))
            lines.append('    }')
            lines.append('')

        lines.append('}')
        return '\n'.join(lines)

    # ── Per-prim USDA generators ──────────────────────────────────────

    def _cell_to_usda(self, cell: BioCell, indent: int = 8) -> List[str]:
        """Generate USDA lines for a BioCell."""
        pad = ' ' * indent
        name = f"Cell_{cell.cell_id}"
        pos = cell.position

        lines = [
            f'{pad}def BioCell "{name}" {{',
            f'{pad}    int bio:cell:id = {cell.cell_id}',
            f'{pad}    token bio:cell:type = "{cell.cell_type.value}"',
            f'{pad}    token bio:cell:phase = "{cell.phase.value}"',
            f'{pad}    float bio:cell:age = {cell.age:.2f}',
            f'{pad}    bool bio:cell:alive = {"true" if cell.alive else "false"}',
            f'{pad}    float bio:cell:volume = {cell.volume:.3f}',
            f'{pad}    float bio:cell:divisionTime = {cell.division_time:.1f}',
            f'{pad}    float3 xformOp:translate = ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})',
            f'{pad}    uniform token[] xformOpOrder = ["xformOp:translate"]',
        ]

        if cell.metabolic:
            m = cell.metabolic
            lines.append(f'{pad}    # BioMetabolicAPI')
            lines.append(f'{pad}    float bio:metabolic:oxygen = {m.oxygen:.4f}')
            lines.append(f'{pad}    float bio:metabolic:glucose = {m.glucose:.3f}')
            lines.append(f'{pad}    float bio:metabolic:atp = {m.atp:.1f}')
            lines.append(f'{pad}    float bio:metabolic:lactate = {m.lactate:.3f}')

        if cell.epigenetic:
            e = cell.epigenetic
            lines.append(f'{pad}    # BioEpigeneticAPI')
            lines.append(f'{pad}    float bio:epigenetic:methylationLevel = {e.methylation_level:.3f}')
            lines.append(f'{pad}    float bio:epigenetic:h3k4me3 = {e.h3k4me3:.3f}')
            lines.append(f'{pad}    float bio:epigenetic:h3k27me3 = {e.h3k27me3:.3f}')
            lines.append(f'{pad}    bool bio:epigenetic:chromatinOpen = {"true" if e.chromatin_open else "false"}')

        lines.append(f'{pad}}}')
        return lines

    def _immune_cell_to_usda(self, cell: BioImmuneCell, indent: int = 8) -> List[str]:
        """Generate USDA lines for a BioImmuneCell."""
        pad = ' ' * indent
        name = f"Immune_{cell.cell_id}"
        pos = cell.position

        lines = [
            f'{pad}def BioImmuneCell "{name}" {{',
            f'{pad}    int bio:cell:id = {cell.cell_id}',
            f'{pad}    token bio:cell:type = "immune"',
            f'{pad}    token bio:cell:phase = "{cell.phase.value}"',
            f'{pad}    float bio:cell:age = {cell.age:.2f}',
            f'{pad}    bool bio:cell:alive = {"true" if cell.alive else "false"}',
            f'{pad}    float3 xformOp:translate = ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})',
            f'{pad}    uniform token[] xformOpOrder = ["xformOp:translate"]',
            f'{pad}    token bio:immune:cellType = "{cell.immune_type.value}"',
            f'{pad}    bool bio:immune:activated = {"true" if cell.activated else "false"}',
            f'{pad}    int bio:immune:targetCellId = {cell.target_cell_id}',
            f'{pad}    float bio:immune:detectionRadius = {cell.detection_radius:.1f}',
            f'{pad}    float bio:immune:killRadius = {cell.kill_radius:.1f}',
            f'{pad}    float bio:immune:killProbability = {cell.kill_probability:.2f}',
            f'{pad}    float bio:immune:mhc1Expression = {cell.mhc1_expression:.3f}',
            f'{pad}}}',
        ]
        return lines

    def _spatial_field_to_usda(self, sf: BioSpatialField, indent: int = 8) -> List[str]:
        """Generate USDA lines for a BioSpatialField."""
        pad = ' ' * indent
        name = f"Field_{sf.field_type.value}"
        gs = sf.grid_shape

        lines = [
            f'{pad}def BioSpatialField "{name}" {{',
            f'{pad}    token bio:field:type = "{sf.field_type.value}"',
            f'{pad}    int3 bio:field:gridShape = ({gs[0]}, {gs[1]}, {gs[2]})',
            f'{pad}    float bio:field:voxelSize = {sf.voxel_size:.1f}',
            f'{pad}    float bio:field:diffusionCoeff = {sf.diffusion_coeff:.1f}',
            f'{pad}    float bio:field:minValue = {sf.min_value:.4f}',
            f'{pad}    float bio:field:maxValue = {sf.max_value:.4f}',
        ]
        if sf.data_ref:
            lines.append(f'{pad}    asset bio:field:dataRef = @{sf.data_ref}@')
        lines.append(f'{pad}}}')
        return lines

    def _capillary_to_usda(self, cap: BioCapillary, indent: int = 8) -> List[str]:
        """Generate USDA lines for a BioCapillary."""
        pad = ' ' * indent
        name = f"Capillary_{cap.capillary_id}"

        lines = [
            f'{pad}def BioCapillary "{name}" {{',
            f'{pad}    int bio:capillary:id = {cap.capillary_id}',
            f'{pad}    float3 bio:capillary:start = ({cap.start[0]:.2f}, {cap.start[1]:.2f}, {cap.start[2]:.2f})',
            f'{pad}    float3 bio:capillary:end = ({cap.end[0]:.2f}, {cap.end[1]:.2f}, {cap.end[2]:.2f})',
            f'{pad}    float bio:capillary:radius = {cap.radius:.2f}',
            f'{pad}    float bio:capillary:oxygenConc = {cap.oxygen_conc:.4f}',
            f'{pad}    float bio:capillary:glucoseConc = {cap.glucose_conc:.2f}',
            f'{pad}    float bio:capillary:flowRate = {cap.flow_rate:.3f}',
            f'{pad}}}',
        ]
        return lines

    def _gene_to_usda(self, gene: BioGene, indent: int = 8) -> List[str]:
        """Generate USDA lines for a BioGene."""
        pad = ' ' * indent
        name = gene.gene_name or f"Gene_{id(gene)}"
        safe_name = name.replace("-", "_").replace(" ", "_")

        mutations_str = ', '.join(f'"{m}"' for m in gene.mutations)

        lines = [
            f'{pad}def BioGene "{safe_name}" {{',
            f'{pad}    string bio:gene:name = "{gene.gene_name}"',
            f'{pad}    token bio:gene:type = "{gene.gene_type.value}"',
            f'{pad}    string bio:gene:chromosome = "{gene.chromosome}"',
            f'{pad}    int bio:gene:sequenceLength = {gene.sequence_length}',
            f'{pad}    float bio:gene:expressionLevel = {gene.expression_level:.3f}',
            f'{pad}    string[] bio:gene:mutations = [{mutations_str}]',
            f'{pad}}}',
        ]
        return lines

    def _protein_to_usda(self, protein: BioProtein, indent: int = 8) -> List[str]:
        """Generate USDA lines for a BioProtein."""
        pad = ' ' * indent
        name = protein.protein_name or f"Protein_{id(protein)}"
        safe_name = name.replace("-", "_").replace(" ", "_")

        sites_str = ', '.join(str(s) for s in protein.active_sites)

        lines = [
            f'{pad}def BioProtein "{safe_name}" {{',
            f'{pad}    string bio:protein:name = "{protein.protein_name}"',
            f'{pad}    string bio:protein:geneSource = "{protein.gene_source}"',
            f'{pad}    int bio:protein:aminoAcidLength = {protein.amino_acid_length}',
            f'{pad}    string bio:protein:pdbId = "{protein.pdb_id}"',
            f'{pad}    float bio:protein:bindingAffinity = {protein.binding_affinity:.3f}',
            f'{pad}    int[] bio:protein:activeSites = [{sites_str}]',
            f'{pad}}}',
        ]
        return lines

    def _molecule_to_usda(self, mol: BioMolecule, indent: int = 8) -> List[str]:
        """Generate USDA lines for a BioMolecule."""
        pad = ' ' * indent
        name = mol.display_name or f"Molecule_{id(mol)}"
        safe_name = name.replace("-", "_").replace(" ", "_").replace("(", "").replace(")", "")

        lines = [
            f'{pad}def BioMolecule "{safe_name}" {{',
            f'{pad}    string bio:molecule:smiles = "{mol.smiles}"',
            f'{pad}    string bio:molecule:inchi = "{mol.inchi}"',
            f'{pad}    float bio:molecule:molecularWeight = {mol.molecular_weight:.2f}',
            f'{pad}    float bio:molecule:logP = {mol.logp:.2f}',
            f'{pad}    int bio:molecule:charge = {mol.charge}',
            f'{pad}}}',
        ]
        return lines

    def _tissue_to_usda(self, tissue: BioTissue, indent: int = 8) -> List[str]:
        """Generate USDA lines for a BioTissue."""
        pad = ' ' * indent
        name = tissue.display_name or tissue.tissue_type
        safe_name = name.replace("-", "_").replace(" ", "_")

        cell_ids_str = ', '.join(str(i) for i in tissue.cell_ids[:100])
        if len(tissue.cell_ids) > 100:
            cell_ids_str += f" /* ... {len(tissue.cell_ids)} total */"

        ext = tissue.extent
        if len(ext) >= 6:
            ext_min = f"({ext[0]:.1f}, {ext[1]:.1f}, {ext[2]:.1f})"
            ext_max = f"({ext[3]:.1f}, {ext[4]:.1f}, {ext[5]:.1f})"
        else:
            ext_min = "(0.0, 0.0, 0.0)"
            ext_max = "(100.0, 100.0, 50.0)"

        lines = [
            f'{pad}def BioTissue "{safe_name}" {{',
            f'{pad}    token bio:tissue:type = "{tissue.tissue_type}"',
            f'{pad}    int[] bio:tissue:cellIds = [{cell_ids_str}]',
            f'{pad}    float3 bio:tissue:extentMin = {ext_min}',
            f'{pad}    float3 bio:tissue:extentMax = {ext_max}',
            f'{pad}    int bio:tissue:cellCount = {tissue.cell_count}',
            f'{pad}}}',
        ]
        return lines

    # ── Engine extraction helpers ─────────────────────────────────────

    def _extract_cells(self, cellular_module) -> List[BioCell]:
        """Extract BioCell objects from a cellular module."""
        cells = []
        cell_dict = getattr(cellular_module, "cells", {})
        for cid, cstate in cell_dict.items():
            pos = getattr(cstate, "position", np.zeros(3))
            ctype_str = getattr(cstate, "cell_type", "normal")
            phase_str = getattr(cstate, "phase", "G1")

            try:
                ctype = CellType(ctype_str)
            except ValueError:
                ctype = CellType.NORMAL

            try:
                phase = CellPhase(phase_str)
            except ValueError:
                phase = CellPhase.G1

            metabolic = None
            if hasattr(cstate, "oxygen"):
                metabolic = BioMetabolicAPI(
                    oxygen=getattr(cstate, "oxygen", 0.21),
                    glucose=getattr(cstate, "glucose", 5.0),
                    atp=getattr(cstate, "atp", 1000.0),
                    lactate=getattr(cstate, "lactate", 0.0),
                )

            cell = BioCell(
                prim_path=f"/BioScene/Cells/Cell_{cid}",
                cell_id=int(cid),
                position=(float(pos[0]), float(pos[1]), float(pos[2])),
                cell_type=ctype,
                phase=phase,
                age=getattr(cstate, "age", 0.0),
                alive=getattr(cstate, "alive", True),
                volume=getattr(cstate, "volume", 1.0),
                division_time=getattr(cstate, "division_time", 24.0),
                metabolic=metabolic,
            )
            cells.append(cell)

        return cells

    def _extract_immune_cells(self, immune_module) -> List[BioImmuneCell]:
        """Extract BioImmuneCell objects from an immune module."""
        cells = []
        immune_dict = getattr(immune_module, "immune_cells", {})
        for cid, icell in immune_dict.items():
            pos = getattr(icell, "position", np.zeros(3))
            itype_str = getattr(icell, "cell_type", "T_cell")

            try:
                itype = ImmuneCellType(itype_str)
            except ValueError:
                itype = ImmuneCellType.T_CELL

            cell = BioImmuneCell(
                prim_path=f"/BioScene/ImmuneCells/Immune_{cid}",
                cell_id=int(cid),
                position=(float(pos[0]), float(pos[1]), float(pos[2])),
                immune_type=itype,
                activated=getattr(icell, "activated", False),
                target_cell_id=getattr(icell, "target_cell_id", -1),
                detection_radius=getattr(icell, "detection_radius", 10.0),
                kill_radius=getattr(icell, "kill_radius", 5.0),
                kill_probability=getattr(icell, "kill_probability", 0.8),
                mhc1_expression=getattr(icell, "mhc1_expression", 1.0),
            )
            cells.append(cell)

        return cells

    def _extract_spatial_fields(self, spatial_module) -> List[BioSpatialField]:
        """Extract BioSpatialField objects from a spatial module."""
        fields = []
        field_dict = getattr(spatial_module, "fields", {})
        for fname, fobj in field_dict.items():
            concentration = getattr(fobj, "concentration", None)
            grid_shape = (200, 200, 100)
            min_val = 0.0
            max_val = 0.21
            if concentration is not None:
                grid_shape = concentration.shape
                min_val = float(np.min(concentration))
                max_val = float(np.max(concentration))

            try:
                ftype = SpatialFieldType(fname.lower())
            except ValueError:
                ftype = SpatialFieldType.OXYGEN

            sf = BioSpatialField(
                prim_path=f"/BioScene/SpatialFields/Field_{fname}",
                display_name=fname,
                field_type=ftype,
                grid_shape=grid_shape,
                voxel_size=getattr(fobj, "voxel_size", 10.0),
                diffusion_coeff=getattr(fobj, "diffusion_coeff", 2000.0),
                min_value=min_val,
                max_value=max_val,
            )
            fields.append(sf)

        return fields

    def _extract_capillaries(self, vascular_module) -> List[BioCapillary]:
        """Extract BioCapillary objects from a vascular module."""
        caps = []
        cap_dict = getattr(vascular_module, "capillaries", {})
        for cid, cobj in cap_dict.items():
            start = getattr(cobj, "start", np.zeros(3))
            end = getattr(cobj, "end", np.array([100.0, 0.0, 0.0]))

            cap = BioCapillary(
                prim_path=f"/BioScene/VascularNetwork/Capillary_{cid}",
                capillary_id=int(cid),
                start=(float(start[0]), float(start[1]), float(start[2])),
                end=(float(end[0]), float(end[1]), float(end[2])),
                radius=getattr(cobj, "radius", 5.0),
                oxygen_conc=getattr(cobj, "oxygen_conc", 0.21),
                glucose_conc=getattr(cobj, "glucose_conc", 5.0),
                flow_rate=getattr(cobj, "flow_rate", 0.5),
            )
            caps.append(cap)

        return caps

    def _extract_genes(self, molecular_module) -> List[BioGene]:
        """Extract BioGene objects from a molecular module."""
        genes = []
        gene_dict = getattr(molecular_module, "genes", {})
        for gname, gobj in gene_dict.items():
            gene = BioGene(
                prim_path=f"/BioScene/Genes/{gname}",
                gene_name=gname,
                gene_type=GeneType.HOUSEKEEPING,
                expression_level=getattr(gobj, "expression_level", 0.5),
                mutations=list(getattr(gobj, "mutations", [])),
            )
            genes.append(gene)

        return genes


# ── Standalone export function ────────────────────────────────────────

def export_demo_scene(output_path: str = "demo_biousd_scene.usda") -> str:
    """Generate a demo Bio-USD scene with sample biological data.

    Useful for testing the schema without a running simulation.
    """
    scene = BioScene(
        simulation_time=12.0,
        time_step=0.1,
        step_count=120,
    )

    # Sample cells
    for i in range(10):
        ctype = CellType.CANCER if i < 3 else CellType.NORMAL
        phase = [CellPhase.G1, CellPhase.S, CellPhase.G2, CellPhase.M, CellPhase.G0][i % 5]
        cell = BioCell(
            prim_path=f"/BioScene/Cells/Cell_{i}",
            cell_id=i,
            position=(float(i * 15), float((i % 3) * 20), 10.0),
            cell_type=ctype,
            phase=phase,
            age=float(i * 2.5),
            alive=True,
            volume=1.0 + (0.3 if ctype == CellType.CANCER else 0.0),
            division_time=12.0 if ctype == CellType.CANCER else 24.0,
            metabolic=BioMetabolicAPI(
                oxygen=0.15 - i * 0.01,
                glucose=4.0 + i * 0.2,
                atp=800.0 + i * 50,
                lactate=2.0 + (3.0 if ctype == CellType.CANCER else 0.0),
            ),
            epigenetic=BioEpigeneticAPI(
                methylation_level=0.3 if ctype == CellType.CANCER else 0.1,
                h3k4me3=0.5 if ctype == CellType.CANCER else 0.9,
                h3k27me3=0.6 if ctype == CellType.CANCER else 0.1,
                chromatin_open=ctype != CellType.CANCER,
            ),
        )
        scene.cells.append(cell)

    # Sample immune cells
    for i in range(3):
        itypes = [ImmuneCellType.T_CELL, ImmuneCellType.NK_CELL, ImmuneCellType.MACROPHAGE]
        icell = BioImmuneCell(
            prim_path=f"/BioScene/ImmuneCells/Immune_{100 + i}",
            cell_id=100 + i,
            position=(float(i * 30 + 10), 50.0, 10.0),
            immune_type=itypes[i],
            activated=i == 0,
            target_cell_id=0 if i == 0 else -1,
            detection_radius=15.0,
            kill_radius=5.0,
            kill_probability=0.85,
            mhc1_expression=1.0,
        )
        scene.immune_cells.append(icell)

    # Sample spatial fields
    for ftype, diff, maxv in [
        (SpatialFieldType.OXYGEN, 2000.0, 0.21),
        (SpatialFieldType.GLUCOSE, 600.0, 10.0),
        (SpatialFieldType.CYTOKINE, 100.0, 5.0),
    ]:
        sf = BioSpatialField(
            display_name=ftype.value,
            field_type=ftype,
            grid_shape=(200, 200, 100),
            voxel_size=10.0,
            diffusion_coeff=diff,
            min_value=0.0,
            max_value=maxv,
        )
        scene.spatial_fields.append(sf)

    # Sample capillaries
    for i in range(3):
        cap = BioCapillary(
            capillary_id=i,
            start=(0.0, float(i * 40), 10.0),
            end=(150.0, float(i * 40), 10.0),
            radius=5.0,
            oxygen_conc=0.21 - i * 0.03,
            glucose_conc=5.0 - i * 0.5,
            flow_rate=0.5 - i * 0.1,
        )
        scene.capillaries.append(cap)

    # Sample genes
    prostate_genes = [
        ("AR", GeneType.ONCOGENE, "Xq12", 2757, 0.8, ["T878A"]),
        ("TP53", GeneType.TUMOR_SUPPRESSOR, "17p13.1", 2629, 0.3, ["R175H"]),
        ("PTEN", GeneType.TUMOR_SUPPRESSOR, "10q23.31", 1212, 0.2, []),
        ("BRCA2", GeneType.TUMOR_SUPPRESSOR, "13q13.1", 10257, 0.6, []),
        ("MYC", GeneType.ONCOGENE, "8q24.21", 1365, 0.9, []),
    ]
    for gname, gtype, chrom, length, expr, muts in prostate_genes:
        gene = BioGene(
            gene_name=gname,
            gene_type=gtype,
            chromosome=chrom,
            sequence_length=length,
            expression_level=expr,
            mutations=muts,
        )
        scene.genes.append(gene)

    # Sample proteins
    scene.proteins.append(BioProtein(
        protein_name="Androgen_Receptor",
        gene_source="AR",
        amino_acid_length=919,
        pdb_id="1E3G",
        binding_affinity=0.5,
        active_sites=[701, 874, 878],
    ))
    scene.proteins.append(BioProtein(
        protein_name="p53",
        gene_source="TP53",
        amino_acid_length=393,
        pdb_id="1TSR",
        binding_affinity=0.3,
        active_sites=[175, 245, 248, 273],
    ))

    # Sample drug molecule (enzalutamide)
    scene.molecules.append(BioMolecule(
        display_name="Enzalutamide",
        smiles="CC1(C)C(=O)N(c2ccc(C#N)c(C(F)(F)F)c2)c2ccc(F)cc21",
        molecular_weight=464.44,
        logp=3.0,
        charge=0,
    ))

    converter = SimulationToUSD()
    return converter.export(output_path, scene=scene)
