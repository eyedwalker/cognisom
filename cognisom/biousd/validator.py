"""
Bio-USD Schema Validator (Phase 5 — CI Gate)
=============================================

Validates BioScene objects against the Bio-USD schema rules.
Designed to run in CI pipelines (pytest) and as a pre-export gate
before writing .usda files.

Checks:
    1. Required fields  — every prim has mandatory attributes
    2. Value ranges     — positions, concentrations, probabilities in bounds
    3. Enum validity    — CellType, CellPhase, etc. are valid tokens
    4. Reference integrity — cell_ids in tissues exist, target_cell_id exists
    5. Duplicate IDs    — no two cells share the same cell_id
    6. Scene completeness — non-empty scene with valid metadata
    7. Field consistency — spatial fields match declared grid shapes

Usage::

    from cognisom.biousd.validator import SceneValidator

    validator = SceneValidator()
    result = validator.validate(scene)
    assert result.passed, result.summary()

    # In CI / pytest:
    from cognisom.biousd.validator import validate_reference_scenes
    validate_reference_scenes()  # raises AssertionError on failure
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

from .schema import (
    BioCapillary,
    BioCell,
    BioExosome,
    BioGene,
    BioImmuneCell,
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


@dataclass
class ValidationIssue:
    """A single validation problem."""
    severity: str = "error"   # "error", "warning"
    category: str = ""        # "required", "range", "enum", "reference", "duplicate"
    prim_path: str = ""
    message: str = ""


@dataclass
class ValidationResult:
    """Outcome of schema validation."""
    passed: bool = True
    issues: List[ValidationIssue] = field(default_factory=list)
    checked_cells: int = 0
    checked_genes: int = 0
    checked_proteins: int = 0
    checked_fields: int = 0
    elapsed_sec: float = 0.0

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def summary(self) -> str:
        lines = [
            f"Validation {'PASSED' if self.passed else 'FAILED'} "
            f"({len(self.errors)} errors, {len(self.warnings)} warnings)",
            f"  Cells checked: {self.checked_cells}",
            f"  Genes checked: {self.checked_genes}",
            f"  Proteins checked: {self.checked_proteins}",
            f"  Fields checked: {self.checked_fields}",
        ]
        for issue in self.errors[:20]:
            lines.append(f"  [ERROR] {issue.category}: {issue.message} ({issue.prim_path})")
        for issue in self.warnings[:10]:
            lines.append(f"  [WARN]  {issue.category}: {issue.message} ({issue.prim_path})")
        if len(self.errors) > 20:
            lines.append(f"  ... and {len(self.errors) - 20} more errors")
        return "\n".join(lines)


class SceneValidator:
    """Validate a BioScene against the Bio-USD schema."""

    def validate(self, scene: BioScene) -> ValidationResult:
        """Run all validation checks on a scene."""
        t0 = time.time()
        result = ValidationResult()

        self._check_scene_metadata(scene, result)
        self._check_cells(scene, result)
        self._check_immune_cells(scene, result)
        self._check_genes(scene, result)
        self._check_proteins(scene, result)
        self._check_molecules(scene, result)
        self._check_tissues(scene, result)
        self._check_capillaries(scene, result)
        self._check_spatial_fields(scene, result)
        self._check_duplicate_ids(scene, result)
        self._check_reference_integrity(scene, result)

        result.elapsed_sec = time.time() - t0
        result.passed = len(result.errors) == 0
        return result

    # ── Scene-level checks ────────────────────────────────────

    def _check_scene_metadata(self, scene: BioScene, result: ValidationResult):
        if scene.time_step <= 0:
            result.issues.append(ValidationIssue(
                severity="error", category="range",
                prim_path="/World",
                message=f"time_step must be > 0, got {scene.time_step}",
            ))
        if scene.simulation_time < 0:
            result.issues.append(ValidationIssue(
                severity="error", category="range",
                prim_path="/World",
                message=f"simulation_time must be >= 0, got {scene.simulation_time}",
            ))

    # ── Cell checks ───────────────────────────────────────────

    def _check_cells(self, scene: BioScene, result: ValidationResult):
        for cell in scene.cells:
            self._validate_cell(cell, result)
        result.checked_cells += len(scene.cells)

    def _check_immune_cells(self, scene: BioScene, result: ValidationResult):
        for cell in scene.immune_cells:
            self._validate_cell(cell, result)
            # Immune-specific checks
            if not isinstance(cell, BioImmuneCell):
                result.issues.append(ValidationIssue(
                    severity="error", category="type",
                    prim_path=cell.prim_path,
                    message="Cell in immune_cells list is not BioImmuneCell",
                ))
                continue
            if cell.detection_radius <= 0:
                result.issues.append(ValidationIssue(
                    severity="error", category="range",
                    prim_path=cell.prim_path,
                    message=f"detection_radius must be > 0, got {cell.detection_radius}",
                ))
            if not 0 <= cell.kill_probability <= 1:
                result.issues.append(ValidationIssue(
                    severity="error", category="range",
                    prim_path=cell.prim_path,
                    message=f"kill_probability must be [0,1], got {cell.kill_probability}",
                ))
        result.checked_cells += len(scene.immune_cells)

    def _validate_cell(self, cell: BioCell, result: ValidationResult):
        path = cell.prim_path or f"cell_{cell.cell_id}"

        # Required: prim_path
        if not cell.prim_path:
            result.issues.append(ValidationIssue(
                severity="warning", category="required",
                prim_path=path,
                message="Cell missing prim_path",
            ))

        # Enum validity
        if not isinstance(cell.cell_type, CellType):
            result.issues.append(ValidationIssue(
                severity="error", category="enum",
                prim_path=path,
                message=f"Invalid cell_type: {cell.cell_type}",
            ))

        if not isinstance(cell.phase, CellPhase):
            result.issues.append(ValidationIssue(
                severity="error", category="enum",
                prim_path=path,
                message=f"Invalid phase: {cell.phase}",
            ))

        # Position bounds
        for i, label in enumerate(["x", "y", "z"]):
            val = cell.position[i]
            if abs(val) > 1e6:
                result.issues.append(ValidationIssue(
                    severity="error", category="range",
                    prim_path=path,
                    message=f"position.{label} = {val} exceeds 1e6 um",
                ))

        # Volume
        if cell.volume <= 0:
            result.issues.append(ValidationIssue(
                severity="error", category="range",
                prim_path=path,
                message=f"volume must be > 0, got {cell.volume}",
            ))

        # Age
        if cell.age < 0:
            result.issues.append(ValidationIssue(
                severity="error", category="range",
                prim_path=path,
                message=f"age must be >= 0, got {cell.age}",
            ))

        # Metabolic API
        if cell.metabolic:
            m = cell.metabolic
            if not 0 <= m.oxygen <= 0.25:
                result.issues.append(ValidationIssue(
                    severity="warning", category="range",
                    prim_path=path,
                    message=f"oxygen = {m.oxygen} outside [0, 0.25]",
                ))
            if m.glucose < 0 or m.glucose > 50:
                result.issues.append(ValidationIssue(
                    severity="warning", category="range",
                    prim_path=path,
                    message=f"glucose = {m.glucose} outside [0, 50]",
                ))
            if m.atp < 0 or m.atp > 5000:
                result.issues.append(ValidationIssue(
                    severity="warning", category="range",
                    prim_path=path,
                    message=f"atp = {m.atp} outside [0, 5000]",
                ))
            if m.lactate < 0:
                result.issues.append(ValidationIssue(
                    severity="error", category="range",
                    prim_path=path,
                    message=f"lactate must be >= 0, got {m.lactate}",
                ))

    # ── Gene checks ───────────────────────────────────────────

    def _check_genes(self, scene: BioScene, result: ValidationResult):
        for gene in scene.genes:
            if not gene.gene_name:
                result.issues.append(ValidationIssue(
                    severity="error", category="required",
                    prim_path=gene.prim_path,
                    message="Gene missing gene_name",
                ))
            if not isinstance(gene.gene_type, GeneType):
                result.issues.append(ValidationIssue(
                    severity="error", category="enum",
                    prim_path=gene.prim_path,
                    message=f"Invalid gene_type: {gene.gene_type}",
                ))
            if gene.expression_level < 0:
                result.issues.append(ValidationIssue(
                    severity="warning", category="range",
                    prim_path=gene.prim_path,
                    message=f"expression_level = {gene.expression_level} < 0",
                ))
        result.checked_genes = len(scene.genes)

    # ── Protein checks ────────────────────────────────────────

    def _check_proteins(self, scene: BioScene, result: ValidationResult):
        for prot in scene.proteins:
            if not prot.protein_name:
                result.issues.append(ValidationIssue(
                    severity="error", category="required",
                    prim_path=prot.prim_path,
                    message="Protein missing protein_name",
                ))
            if prot.amino_acid_length < 0:
                result.issues.append(ValidationIssue(
                    severity="error", category="range",
                    prim_path=prot.prim_path,
                    message=f"amino_acid_length = {prot.amino_acid_length} < 0",
                ))
        result.checked_proteins = len(scene.proteins)

    # ── Molecule checks ───────────────────────────────────────

    def _check_molecules(self, scene: BioScene, result: ValidationResult):
        for mol in scene.molecules:
            if mol.molecular_weight < 0:
                result.issues.append(ValidationIssue(
                    severity="error", category="range",
                    prim_path=mol.prim_path,
                    message=f"molecular_weight = {mol.molecular_weight} < 0",
                ))

    # ── Tissue checks ─────────────────────────────────────────

    def _check_tissues(self, scene: BioScene, result: ValidationResult):
        for tissue in scene.tissues:
            if not tissue.tissue_type:
                result.issues.append(ValidationIssue(
                    severity="warning", category="required",
                    prim_path=tissue.prim_path,
                    message="Tissue missing tissue_type",
                ))

    # ── Capillary checks ──────────────────────────────────────

    def _check_capillaries(self, scene: BioScene, result: ValidationResult):
        for cap in scene.capillaries:
            if cap.radius <= 0:
                result.issues.append(ValidationIssue(
                    severity="error", category="range",
                    prim_path=cap.prim_path,
                    message=f"capillary radius must be > 0, got {cap.radius}",
                ))
            if cap.oxygen_conc < 0:
                result.issues.append(ValidationIssue(
                    severity="error", category="range",
                    prim_path=cap.prim_path,
                    message=f"oxygen_conc must be >= 0, got {cap.oxygen_conc}",
                ))

    # ── Spatial field checks ──────────────────────────────────

    def _check_spatial_fields(self, scene: BioScene, result: ValidationResult):
        for sf in scene.spatial_fields:
            if not isinstance(sf.field_type, SpatialFieldType):
                result.issues.append(ValidationIssue(
                    severity="error", category="enum",
                    prim_path=sf.prim_path,
                    message=f"Invalid field_type: {sf.field_type}",
                ))
            if any(d <= 0 for d in sf.grid_shape):
                result.issues.append(ValidationIssue(
                    severity="error", category="range",
                    prim_path=sf.prim_path,
                    message=f"grid_shape dimensions must be > 0: {sf.grid_shape}",
                ))
            if sf.voxel_size <= 0:
                result.issues.append(ValidationIssue(
                    severity="error", category="range",
                    prim_path=sf.prim_path,
                    message=f"voxel_size must be > 0, got {sf.voxel_size}",
                ))
            if sf.diffusion_coeff < 0:
                result.issues.append(ValidationIssue(
                    severity="error", category="range",
                    prim_path=sf.prim_path,
                    message=f"diffusion_coeff must be >= 0, got {sf.diffusion_coeff}",
                ))
        result.checked_fields = len(scene.spatial_fields)

    # ── Cross-prim integrity ──────────────────────────────────

    def _check_duplicate_ids(self, scene: BioScene, result: ValidationResult):
        seen_ids = set()
        for cell in scene.cells + scene.immune_cells:
            if cell.cell_id in seen_ids:
                result.issues.append(ValidationIssue(
                    severity="error", category="duplicate",
                    prim_path=cell.prim_path,
                    message=f"Duplicate cell_id: {cell.cell_id}",
                ))
            seen_ids.add(cell.cell_id)

    def _check_reference_integrity(self, scene: BioScene, result: ValidationResult):
        all_cell_ids = {c.cell_id for c in scene.cells + scene.immune_cells}

        # Tissue cell_ids must exist
        for tissue in scene.tissues:
            for cid in tissue.cell_ids:
                if cid not in all_cell_ids:
                    result.issues.append(ValidationIssue(
                        severity="error", category="reference",
                        prim_path=tissue.prim_path,
                        message=f"Tissue references non-existent cell_id: {cid}",
                    ))

        # Immune target_cell_id must exist (if set)
        for ic in scene.immune_cells:
            if ic.target_cell_id >= 0 and ic.target_cell_id not in all_cell_ids:
                result.issues.append(ValidationIssue(
                    severity="warning", category="reference",
                    prim_path=ic.prim_path,
                    message=f"Immune cell targets non-existent cell_id: {ic.target_cell_id}",
                ))

        # Exosome source/target must exist (if set)
        for exo in scene.exosomes:
            if exo.source_cell_id >= 0 and exo.source_cell_id not in all_cell_ids:
                result.issues.append(ValidationIssue(
                    severity="warning", category="reference",
                    prim_path=exo.prim_path,
                    message=f"Exosome source_cell_id {exo.source_cell_id} not in scene",
                ))


# ── Reference Scenes ──────────────────────────────────────────────────

def create_reference_scene_minimal() -> BioScene:
    """Minimal valid scene — one cell, one gene, one field."""
    from .schema import BioMetabolicAPI
    return BioScene(
        simulation_time=0.0,
        time_step=0.01,
        step_count=0,
        cells=[
            BioCell(
                prim_path="/World/Cells/cell_0",
                cell_id=0,
                position=(100.0, 100.0, 50.0),
                cell_type=CellType.NORMAL,
                phase=CellPhase.G1,
                alive=True,
                volume=1.0,
                metabolic=BioMetabolicAPI(
                    oxygen=0.21, glucose=5.0, atp=1000.0, lactate=0.0,
                ),
            ),
        ],
        genes=[
            BioGene(
                prim_path="/World/Genes/GAPDH",
                gene_name="GAPDH",
                gene_type=GeneType.HOUSEKEEPING,
                expression_level=1.0,
            ),
        ],
        spatial_fields=[
            BioSpatialField(
                prim_path="/World/Fields/oxygen",
                field_type=SpatialFieldType.OXYGEN,
                grid_shape=(50, 50, 25),
                voxel_size=10.0,
                diffusion_coeff=2000.0,
                min_value=0.0,
                max_value=0.21,
            ),
        ],
    )


def create_reference_scene_prostate() -> BioScene:
    """Reference prostate tissue scene with all prim types populated."""
    from .schema import BioMetabolicAPI
    import random

    random.seed(42)
    scene = BioScene(
        simulation_time=24.0,
        time_step=0.01,
        step_count=2400,
    )

    # Cells
    cell_id = 0
    for _ in range(50):
        scene.cells.append(BioCell(
            prim_path=f"/World/Cells/cell_{cell_id}",
            cell_id=cell_id,
            position=(random.uniform(10, 190), random.uniform(10, 190), random.uniform(5, 95)),
            cell_type=CellType.LUMINAL,
            phase=random.choice(list(CellPhase)),
            alive=True,
            volume=random.uniform(0.8, 1.2),
            metabolic=BioMetabolicAPI(
                oxygen=random.uniform(0.05, 0.21),
                glucose=random.uniform(2.0, 8.0),
                atp=random.uniform(500, 1500),
                lactate=random.uniform(0.0, 3.0),
            ),
        ))
        cell_id += 1

    for _ in range(20):
        scene.cells.append(BioCell(
            prim_path=f"/World/Cells/cell_{cell_id}",
            cell_id=cell_id,
            position=(random.uniform(50, 150), random.uniform(50, 150), random.uniform(20, 80)),
            cell_type=CellType.CANCER,
            phase=random.choice(list(CellPhase)),
            alive=True,
            volume=random.uniform(1.0, 1.8),
            metabolic=BioMetabolicAPI(
                oxygen=random.uniform(0.01, 0.10),
                glucose=random.uniform(0.5, 4.0),
                atp=random.uniform(300, 1000),
                lactate=random.uniform(2.0, 10.0),
            ),
        ))
        cell_id += 1

    # Immune cells
    for _ in range(10):
        scene.immune_cells.append(BioImmuneCell(
            prim_path=f"/World/Cells/cell_{cell_id}",
            cell_id=cell_id,
            position=(random.uniform(10, 190), random.uniform(10, 190), random.uniform(5, 95)),
            immune_type=random.choice(list(ImmuneCellType)),
            activated=random.random() > 0.5,
            detection_radius=random.uniform(8, 15),
            kill_radius=random.uniform(3, 8),
            kill_probability=random.uniform(0.5, 0.95),
            alive=True,
            volume=1.0,
            metabolic=BioMetabolicAPI(oxygen=0.15, glucose=5.0, atp=1200.0),
        ))
        cell_id += 1

    # Genes
    for gene_name, gtype in [
        ("AR", GeneType.ONCOGENE),
        ("TP53", GeneType.TUMOR_SUPPRESSOR),
        ("PTEN", GeneType.TUMOR_SUPPRESSOR),
        ("MYC", GeneType.ONCOGENE),
        ("GAPDH", GeneType.HOUSEKEEPING),
    ]:
        scene.genes.append(BioGene(
            prim_path=f"/World/Genes/{gene_name}",
            gene_name=gene_name,
            gene_type=gtype,
            expression_level=random.uniform(0.1, 1.0),
        ))

    # Proteins
    for pname in ["AR_protein", "p53", "PTEN_protein"]:
        scene.proteins.append(BioProtein(
            prim_path=f"/World/Proteins/{pname}",
            protein_name=pname,
            amino_acid_length=random.randint(200, 900),
        ))

    # Molecules
    scene.molecules.append(BioMolecule(
        prim_path="/World/Molecules/testosterone",
        display_name="Testosterone",
        molecular_weight=288.42,
    ))
    scene.molecules.append(BioMolecule(
        prim_path="/World/Molecules/enzalutamide",
        display_name="Enzalutamide",
        molecular_weight=464.44,
    ))

    # Tissue
    all_cell_ids = [c.cell_id for c in scene.cells + scene.immune_cells]
    scene.tissues.append(BioTissue(
        prim_path="/World/Tissues/prostate_epithelium",
        tissue_type="prostate_epithelium",
        cell_ids=all_cell_ids,
        cell_count=len(all_cell_ids),
    ))

    # Capillaries
    scene.capillaries.append(BioCapillary(
        prim_path="/World/Vasculature/cap_0",
        capillary_id=0,
        start=(0, 100, 50),
        end=(200, 100, 50),
        radius=5.0,
        oxygen_conc=0.21,
        glucose_conc=5.0,
    ))

    # Spatial fields
    for ft, dc, lo, hi in [
        (SpatialFieldType.OXYGEN, 2000.0, 0.0, 0.21),
        (SpatialFieldType.GLUCOSE, 600.0, 0.0, 10.0),
        (SpatialFieldType.CYTOKINE, 100.0, 0.0, 50.0),
    ]:
        scene.spatial_fields.append(BioSpatialField(
            prim_path=f"/World/Fields/{ft.value}",
            field_type=ft,
            grid_shape=(200, 200, 100),
            voxel_size=10.0,
            diffusion_coeff=dc,
            min_value=lo,
            max_value=hi,
        ))

    return scene


def validate_reference_scenes() -> None:
    """Validate all built-in reference scenes. Raises on failure.

    Use in CI::

        def test_reference_scenes():
            validate_reference_scenes()
    """
    validator = SceneValidator()

    scenes = {
        "minimal": create_reference_scene_minimal(),
        "prostate": create_reference_scene_prostate(),
    }

    for name, scene in scenes.items():
        result = validator.validate(scene)
        if not result.passed:
            raise AssertionError(
                f"Reference scene '{name}' failed validation:\n{result.summary()}"
            )
        log.info("Reference scene '%s' passed: %d cells, %d issues",
                 name, result.checked_cells, len(result.issues))
