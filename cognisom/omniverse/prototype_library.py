"""
Biological Prototype Library (Phase B2)
========================================

Pre-defined prototypes for common biological structures.

Categories:
1. Molecules - ATP, proteins, lipids
2. Organelles - Mitochondria, ER, nucleus
3. Cells - Various cell types
4. Particles - Water, ions, signaling molecules
5. Structures - Membranes, filaments, vessels

Each prototype is designed for efficient instancing with:
- Optimized geometry for LOD
- Semantic variants for zoom levels
- Proper instancing flags
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# Import USD modules
try:
    from pxr import Usd, UsdGeom, UsdShade, Gf, Sdf
    USD_AVAILABLE = True
except ImportError:
    USD_AVAILABLE = False

try:
    from .instancing import (
        NestedInstancingManager,
        InstancingStrategy,
        PrototypeDefinition,
        create_sphere_prototype,
        create_capsule_prototype,
        create_ellipsoid_prototype,
    )
    INSTANCING_AVAILABLE = True
except ImportError:
    INSTANCING_AVAILABLE = False


class PrototypeCategory(str, Enum):
    """Categories of biological prototypes."""
    MOLECULE = "molecule"
    ORGANELLE = "organelle"
    CELL = "cell"
    PARTICLE = "particle"
    STRUCTURE = "structure"


@dataclass
class PrototypeSpec:
    """Specification for a biological prototype."""
    name: str
    category: PrototypeCategory
    description: str
    geometry_type: str  # sphere, capsule, mesh, etc.
    default_size: float  # In micrometers
    color: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    opacity: float = 1.0
    suggested_strategy: InstancingStrategy = InstancingStrategy.SCENEGRAPH
    max_instances: int = 100000
    geometry_params: Dict[str, Any] = field(default_factory=dict)
    lod_levels: int = 3


# ── Standard Prototype Specifications ────────────────────────────────────

MOLECULE_PROTOTYPES: Dict[str, PrototypeSpec] = {
    "atp": PrototypeSpec(
        name="atp",
        category=PrototypeCategory.MOLECULE,
        description="ATP molecule",
        geometry_type="sphere",
        default_size=0.001,  # ~1 nm
        color=(0.2, 0.9, 0.2),
        suggested_strategy=InstancingStrategy.POINT_INSTANCER,
        max_instances=10000000
    ),
    "glucose": PrototypeSpec(
        name="glucose",
        category=PrototypeCategory.MOLECULE,
        description="Glucose molecule",
        geometry_type="sphere",
        default_size=0.0008,
        color=(0.9, 0.8, 0.2),
        suggested_strategy=InstancingStrategy.POINT_INSTANCER,
        max_instances=10000000
    ),
    "oxygen": PrototypeSpec(
        name="oxygen",
        category=PrototypeCategory.MOLECULE,
        description="O2 molecule",
        geometry_type="capsule",
        default_size=0.0003,
        color=(0.9, 0.2, 0.2),
        suggested_strategy=InstancingStrategy.POINT_INSTANCER,
        max_instances=100000000,
        geometry_params={"radius": 0.15, "height": 0.3}
    ),
    "co2": PrototypeSpec(
        name="co2",
        category=PrototypeCategory.MOLECULE,
        description="CO2 molecule",
        geometry_type="sphere",
        default_size=0.0003,
        color=(0.5, 0.5, 0.5),
        suggested_strategy=InstancingStrategy.POINT_INSTANCER,
        max_instances=100000000
    ),
    "lipid": PrototypeSpec(
        name="lipid",
        category=PrototypeCategory.MOLECULE,
        description="Phospholipid molecule",
        geometry_type="capsule",
        default_size=0.002,
        color=(0.8, 0.7, 0.3),
        suggested_strategy=InstancingStrategy.POINT_INSTANCER,
        max_instances=1000000000,
        geometry_params={"radius": 0.3, "height": 2.0}
    ),
}

ORGANELLE_PROTOTYPES: Dict[str, PrototypeSpec] = {
    "mitochondrion": PrototypeSpec(
        name="mitochondrion",
        category=PrototypeCategory.ORGANELLE,
        description="Mitochondrion (powerhouse)",
        geometry_type="ellipsoid",
        default_size=2.0,  # ~2 μm
        color=(0.8, 0.4, 0.2),
        opacity=0.9,
        suggested_strategy=InstancingStrategy.SCENEGRAPH,
        max_instances=100000,
        geometry_params={"radii": (1.0, 0.3, 0.3)}
    ),
    "nucleus": PrototypeSpec(
        name="nucleus",
        category=PrototypeCategory.ORGANELLE,
        description="Cell nucleus",
        geometry_type="sphere",
        default_size=5.0,  # ~5 μm
        color=(0.3, 0.3, 0.8),
        opacity=0.8,
        suggested_strategy=InstancingStrategy.SCENEGRAPH,
        max_instances=100000
    ),
    "ribosome": PrototypeSpec(
        name="ribosome",
        category=PrototypeCategory.ORGANELLE,
        description="Ribosome",
        geometry_type="sphere",
        default_size=0.025,  # ~25 nm
        color=(0.6, 0.4, 0.7),
        suggested_strategy=InstancingStrategy.POINT_INSTANCER,
        max_instances=10000000
    ),
    "golgi": PrototypeSpec(
        name="golgi",
        category=PrototypeCategory.ORGANELLE,
        description="Golgi apparatus",
        geometry_type="ellipsoid",
        default_size=3.0,
        color=(0.7, 0.6, 0.3),
        suggested_strategy=InstancingStrategy.SCENEGRAPH,
        max_instances=50000,
        geometry_params={"radii": (1.0, 0.5, 0.2)}
    ),
    "lysosome": PrototypeSpec(
        name="lysosome",
        category=PrototypeCategory.ORGANELLE,
        description="Lysosome",
        geometry_type="sphere",
        default_size=0.5,
        color=(0.5, 0.8, 0.3),
        suggested_strategy=InstancingStrategy.SCENEGRAPH,
        max_instances=100000
    ),
    "vesicle": PrototypeSpec(
        name="vesicle",
        category=PrototypeCategory.ORGANELLE,
        description="Transport vesicle",
        geometry_type="sphere",
        default_size=0.1,
        color=(0.7, 0.7, 0.9),
        opacity=0.7,
        suggested_strategy=InstancingStrategy.POINT_INSTANCER,
        max_instances=1000000
    ),
    "er_segment": PrototypeSpec(
        name="er_segment",
        category=PrototypeCategory.ORGANELLE,
        description="Endoplasmic reticulum segment",
        geometry_type="capsule",
        default_size=1.0,
        color=(0.5, 0.6, 0.8),
        suggested_strategy=InstancingStrategy.SCENEGRAPH,
        max_instances=500000,
        geometry_params={"radius": 0.1, "height": 1.0}
    ),
}

CELL_PROTOTYPES: Dict[str, PrototypeSpec] = {
    "generic_cell": PrototypeSpec(
        name="generic_cell",
        category=PrototypeCategory.CELL,
        description="Generic eukaryotic cell",
        geometry_type="sphere",
        default_size=20.0,  # ~20 μm
        color=(0.9, 0.7, 0.6),
        suggested_strategy=InstancingStrategy.SCENEGRAPH,
        max_instances=100000
    ),
    "red_blood_cell": PrototypeSpec(
        name="red_blood_cell",
        category=PrototypeCategory.CELL,
        description="Erythrocyte (RBC)",
        geometry_type="ellipsoid",
        default_size=7.5,  # ~7.5 μm diameter
        color=(0.8, 0.1, 0.1),
        suggested_strategy=InstancingStrategy.POINT_INSTANCER,
        max_instances=100000000,  # Can have billions
        geometry_params={"radii": (1.0, 1.0, 0.3)}  # Biconcave
    ),
    "white_blood_cell": PrototypeSpec(
        name="white_blood_cell",
        category=PrototypeCategory.CELL,
        description="Leukocyte (WBC)",
        geometry_type="sphere",
        default_size=12.0,
        color=(0.95, 0.95, 0.9),
        suggested_strategy=InstancingStrategy.SCENEGRAPH,
        max_instances=100000
    ),
    "platelet": PrototypeSpec(
        name="platelet",
        category=PrototypeCategory.CELL,
        description="Thrombocyte (platelet)",
        geometry_type="ellipsoid",
        default_size=2.5,
        color=(0.9, 0.85, 0.7),
        suggested_strategy=InstancingStrategy.POINT_INSTANCER,
        max_instances=10000000,
        geometry_params={"radii": (1.0, 1.0, 0.4)}
    ),
    "neuron": PrototypeSpec(
        name="neuron",
        category=PrototypeCategory.CELL,
        description="Neuron (simplified)",
        geometry_type="sphere",
        default_size=20.0,
        color=(0.4, 0.6, 0.9),
        suggested_strategy=InstancingStrategy.SCENEGRAPH,
        max_instances=1000000
    ),
    "hepatocyte": PrototypeSpec(
        name="hepatocyte",
        category=PrototypeCategory.CELL,
        description="Liver cell",
        geometry_type="sphere",
        default_size=25.0,
        color=(0.7, 0.4, 0.3),
        suggested_strategy=InstancingStrategy.SCENEGRAPH,
        max_instances=100000000  # Liver has ~100 billion
    ),
    "tumor_cell": PrototypeSpec(
        name="tumor_cell",
        category=PrototypeCategory.CELL,
        description="Cancer/tumor cell",
        geometry_type="sphere",
        default_size=22.0,
        color=(0.8, 0.2, 0.2),
        suggested_strategy=InstancingStrategy.SCENEGRAPH,
        max_instances=10000000
    ),
    "immune_cell": PrototypeSpec(
        name="immune_cell",
        category=PrototypeCategory.CELL,
        description="Generic immune cell",
        geometry_type="sphere",
        default_size=10.0,
        color=(0.2, 0.6, 0.9),
        suggested_strategy=InstancingStrategy.SCENEGRAPH,
        max_instances=1000000
    ),
    "bacteria": PrototypeSpec(
        name="bacteria",
        category=PrototypeCategory.CELL,
        description="Rod-shaped bacterium",
        geometry_type="capsule",
        default_size=2.0,  # ~2 μm
        color=(0.3, 0.7, 0.3),
        suggested_strategy=InstancingStrategy.POINT_INSTANCER,
        max_instances=100000000,
        geometry_params={"radius": 0.4, "height": 1.5}
    ),
}

PARTICLE_PROTOTYPES: Dict[str, PrototypeSpec] = {
    "water": PrototypeSpec(
        name="water",
        category=PrototypeCategory.PARTICLE,
        description="Water molecule",
        geometry_type="sphere",
        default_size=0.000275,  # ~2.75 Å
        color=(0.3, 0.5, 0.9),
        opacity=0.6,
        suggested_strategy=InstancingStrategy.POINT_INSTANCER,
        max_instances=10000000000  # Can be trillions
    ),
    "sodium_ion": PrototypeSpec(
        name="sodium_ion",
        category=PrototypeCategory.PARTICLE,
        description="Na+ ion",
        geometry_type="sphere",
        default_size=0.000116,  # ~1.16 Å
        color=(0.9, 0.6, 0.2),
        suggested_strategy=InstancingStrategy.POINT_INSTANCER,
        max_instances=1000000000
    ),
    "potassium_ion": PrototypeSpec(
        name="potassium_ion",
        category=PrototypeCategory.PARTICLE,
        description="K+ ion",
        geometry_type="sphere",
        default_size=0.000152,
        color=(0.6, 0.2, 0.9),
        suggested_strategy=InstancingStrategy.POINT_INSTANCER,
        max_instances=1000000000
    ),
    "calcium_ion": PrototypeSpec(
        name="calcium_ion",
        category=PrototypeCategory.PARTICLE,
        description="Ca2+ ion",
        geometry_type="sphere",
        default_size=0.000114,
        color=(0.2, 0.9, 0.6),
        suggested_strategy=InstancingStrategy.POINT_INSTANCER,
        max_instances=1000000000
    ),
    "drug_particle": PrototypeSpec(
        name="drug_particle",
        category=PrototypeCategory.PARTICLE,
        description="Drug molecule (generic)",
        geometry_type="sphere",
        default_size=0.001,
        color=(0.2, 0.9, 0.3),
        suggested_strategy=InstancingStrategy.POINT_INSTANCER,
        max_instances=10000000
    ),
}

STRUCTURE_PROTOTYPES: Dict[str, PrototypeSpec] = {
    "microtubule_segment": PrototypeSpec(
        name="microtubule_segment",
        category=PrototypeCategory.STRUCTURE,
        description="Microtubule segment",
        geometry_type="capsule",
        default_size=0.5,
        color=(0.4, 0.7, 0.4),
        suggested_strategy=InstancingStrategy.SCENEGRAPH,
        max_instances=1000000,
        geometry_params={"radius": 0.012, "height": 0.5}
    ),
    "actin_filament": PrototypeSpec(
        name="actin_filament",
        category=PrototypeCategory.STRUCTURE,
        description="Actin filament segment",
        geometry_type="capsule",
        default_size=0.3,
        color=(0.9, 0.4, 0.4),
        suggested_strategy=InstancingStrategy.POINT_INSTANCER,
        max_instances=10000000,
        geometry_params={"radius": 0.004, "height": 0.3}
    ),
    "collagen_fiber": PrototypeSpec(
        name="collagen_fiber",
        category=PrototypeCategory.STRUCTURE,
        description="Collagen fiber segment",
        geometry_type="capsule",
        default_size=2.0,
        color=(0.9, 0.85, 0.7),
        suggested_strategy=InstancingStrategy.SCENEGRAPH,
        max_instances=1000000,
        geometry_params={"radius": 0.05, "height": 2.0}
    ),
    "blood_vessel_segment": PrototypeSpec(
        name="blood_vessel_segment",
        category=PrototypeCategory.STRUCTURE,
        description="Blood vessel segment",
        geometry_type="capsule",
        default_size=50.0,
        color=(0.8, 0.2, 0.2),
        suggested_strategy=InstancingStrategy.SCENEGRAPH,
        max_instances=100000,
        geometry_params={"radius": 5.0, "height": 50.0}
    ),
}


# ── Immune Cell Prototypes ────────────────────────────────────────────

IMMUNE_CELL_PROTOTYPES: Dict[str, PrototypeSpec] = {
    "t_cell_cd8": PrototypeSpec(
        name="t_cell_cd8",
        category=PrototypeCategory.CELL,
        description="CD8+ cytotoxic T cell",
        geometry_type="sphere",
        default_size=8.0,
        color=(0.1, 0.2, 0.8),  # Deep blue
        suggested_strategy=InstancingStrategy.SCENEGRAPH,
        max_instances=1000000
    ),
    "t_cell_cd4": PrototypeSpec(
        name="t_cell_cd4",
        category=PrototypeCategory.CELL,
        description="CD4+ helper T cell",
        geometry_type="sphere",
        default_size=8.0,
        color=(0.3, 0.4, 0.85),  # Medium blue
        suggested_strategy=InstancingStrategy.SCENEGRAPH,
        max_instances=1000000
    ),
    "b_cell": PrototypeSpec(
        name="b_cell",
        category=PrototypeCategory.CELL,
        description="B lymphocyte",
        geometry_type="sphere",
        default_size=7.0,
        color=(0.5, 0.3, 0.8),  # Purple-blue
        suggested_strategy=InstancingStrategy.SCENEGRAPH,
        max_instances=1000000
    ),
    "plasma_cell": PrototypeSpec(
        name="plasma_cell",
        category=PrototypeCategory.CELL,
        description="Antibody-secreting plasma cell",
        geometry_type="ellipsoid",
        default_size=15.0,
        color=(0.7, 0.2, 0.9),  # Bright purple
        suggested_strategy=InstancingStrategy.SCENEGRAPH,
        max_instances=500000,
        geometry_params={"radii": (1.0, 0.7, 0.7)}
    ),
    "dendritic_cell": PrototypeSpec(
        name="dendritic_cell",
        category=PrototypeCategory.CELL,
        description="Dendritic cell (antigen-presenting)",
        geometry_type="ellipsoid",
        default_size=15.0,
        color=(0.95, 0.7, 0.2),  # Yellow-orange
        suggested_strategy=InstancingStrategy.SCENEGRAPH,
        max_instances=500000,
        geometry_params={"radii": (1.0, 0.8, 0.6)}
    ),
    "macrophage_m1": PrototypeSpec(
        name="macrophage_m1",
        category=PrototypeCategory.CELL,
        description="M1 macrophage (pro-inflammatory)",
        geometry_type="sphere",
        default_size=20.0,
        color=(0.9, 0.35, 0.15),  # Red-orange
        suggested_strategy=InstancingStrategy.SCENEGRAPH,
        max_instances=500000
    ),
    "macrophage_m2": PrototypeSpec(
        name="macrophage_m2",
        category=PrototypeCategory.CELL,
        description="M2 macrophage (tissue repair)",
        geometry_type="sphere",
        default_size=20.0,
        color=(0.4, 0.75, 0.35),  # Green-tinted
        suggested_strategy=InstancingStrategy.SCENEGRAPH,
        max_instances=500000
    ),
    "neutrophil": PrototypeSpec(
        name="neutrophil",
        category=PrototypeCategory.CELL,
        description="Neutrophil granulocyte",
        geometry_type="sphere",
        default_size=12.0,
        color=(0.75, 0.8, 0.9),  # Light gray-blue
        suggested_strategy=InstancingStrategy.POINT_INSTANCER,
        max_instances=5000000
    ),
    "mast_cell": PrototypeSpec(
        name="mast_cell",
        category=PrototypeCategory.CELL,
        description="Mast cell (granule-filled)",
        geometry_type="sphere",
        default_size=12.0,
        color=(0.65, 0.2, 0.7),  # Purple
        suggested_strategy=InstancingStrategy.SCENEGRAPH,
        max_instances=500000
    ),
    "nk_cell": PrototypeSpec(
        name="nk_cell",
        category=PrototypeCategory.CELL,
        description="Natural killer cell",
        geometry_type="sphere",
        default_size=10.0,
        color=(0.1, 0.8, 0.8),  # Cyan
        suggested_strategy=InstancingStrategy.SCENEGRAPH,
        max_instances=1000000
    ),
}

IMMUNE_PARTICLE_PROTOTYPES: Dict[str, PrototypeSpec] = {
    "antibody_igg": PrototypeSpec(
        name="antibody_igg",
        category=PrototypeCategory.MOLECULE,
        description="IgG antibody (Y-shaped)",
        geometry_type="capsule",
        default_size=0.015,  # ~15 nm
        color=(0.95, 0.8, 0.2),  # Gold
        suggested_strategy=InstancingStrategy.POINT_INSTANCER,
        max_instances=100000000,
        geometry_params={"radius": 0.003, "height": 0.012}
    ),
    "antibody_igm": PrototypeSpec(
        name="antibody_igm",
        category=PrototypeCategory.MOLECULE,
        description="IgM pentamer",
        geometry_type="sphere",
        default_size=0.025,  # ~25 nm pentameric
        color=(0.95, 0.75, 0.1),  # Gold
        suggested_strategy=InstancingStrategy.POINT_INSTANCER,
        max_instances=10000000
    ),
    "virus_capsid": PrototypeSpec(
        name="virus_capsid",
        category=PrototypeCategory.PARTICLE,
        description="Viral capsid (generic)",
        geometry_type="sphere",
        default_size=0.1,  # ~100 nm
        color=(0.7, 0.15, 0.5),  # Red-purple
        suggested_strategy=InstancingStrategy.POINT_INSTANCER,
        max_instances=10000000
    ),
    "bacterium_rod": PrototypeSpec(
        name="bacterium_rod",
        category=PrototypeCategory.PARTICLE,
        description="Rod-shaped bacterium (bacillus)",
        geometry_type="capsule",
        default_size=2.0,
        color=(0.3, 0.7, 0.3),  # Green
        suggested_strategy=InstancingStrategy.POINT_INSTANCER,
        max_instances=100000000,
        geometry_params={"radius": 0.4, "height": 1.5}
    ),
    "bacterium_coccus": PrototypeSpec(
        name="bacterium_coccus",
        category=PrototypeCategory.PARTICLE,
        description="Spherical bacterium (coccus)",
        geometry_type="sphere",
        default_size=1.0,
        color=(0.35, 0.65, 0.3),  # Green
        suggested_strategy=InstancingStrategy.POINT_INSTANCER,
        max_instances=100000000
    ),
    "cytokine_particle": PrototypeSpec(
        name="cytokine_particle",
        category=PrototypeCategory.MOLECULE,
        description="Cytokine signaling molecule",
        geometry_type="sphere",
        default_size=0.005,  # ~5 nm
        color=(0.2, 0.9, 0.9),  # Cyan
        opacity=0.6,
        suggested_strategy=InstancingStrategy.POINT_INSTANCER,
        max_instances=100000000
    ),
    "complement_complex": PrototypeSpec(
        name="complement_complex",
        category=PrototypeCategory.MOLECULE,
        description="Complement protein complex",
        geometry_type="sphere",
        default_size=0.015,  # ~15 nm
        color=(0.95, 0.65, 0.15),  # Orange-yellow
        suggested_strategy=InstancingStrategy.POINT_INSTANCER,
        max_instances=10000000
    ),
    # Diapedesis-specific prototypes
    "endothelial_cell": PrototypeSpec(
        name="endothelial_cell",
        category=PrototypeCategory.CELL,
        description="Endothelial cell (flat squamous, vessel lining)",
        geometry_type="ellipsoid",
        default_size=25.0,  # ~25 μm
        color=(0.9, 0.75, 0.75),  # Light pink
        suggested_strategy=InstancingStrategy.SCENEGRAPH,
        max_instances=500000,
        geometry_params={"radii": (1.0, 1.0, 0.15)}  # Very flat (squamous)
    ),
    "selectin_molecule": PrototypeSpec(
        name="selectin_molecule",
        category=PrototypeCategory.MOLECULE,
        description="Selectin adhesion molecule (E/P/L-selectin)",
        geometry_type="sphere",
        default_size=0.02,  # ~20 nm
        color=(1.0, 0.9, 0.2),  # Yellow
        suggested_strategy=InstancingStrategy.POINT_INSTANCER,
        max_instances=100000000
    ),
    "integrin_low": PrototypeSpec(
        name="integrin_low",
        category=PrototypeCategory.MOLECULE,
        description="Integrin (low affinity / bent conformation)",
        geometry_type="sphere",
        default_size=0.015,  # ~15 nm
        color=(0.3, 0.6, 0.7),  # Dim cyan
        suggested_strategy=InstancingStrategy.POINT_INSTANCER,
        max_instances=100000000
    ),
    "integrin_high": PrototypeSpec(
        name="integrin_high",
        category=PrototypeCategory.MOLECULE,
        description="Integrin (high affinity / extended conformation)",
        geometry_type="sphere",
        default_size=0.015,  # ~15 nm
        color=(0.0, 1.0, 1.0),  # Bright cyan
        suggested_strategy=InstancingStrategy.POINT_INSTANCER,
        max_instances=100000000
    ),
    "ecm_fiber": PrototypeSpec(
        name="ecm_fiber",
        category=PrototypeCategory.STRUCTURE,
        description="Extracellular matrix fiber (collagen/fibronectin)",
        geometry_type="capsule",
        default_size=0.5,  # ~0.5 μm segment
        color=(0.8, 0.75, 0.6),  # Beige
        suggested_strategy=InstancingStrategy.POINT_INSTANCER,
        max_instances=10000000,
        geometry_params={"radius": 0.02, "height": 0.5}
    ),
}


# Combine all prototypes
ALL_PROTOTYPES: Dict[str, PrototypeSpec] = {
    **MOLECULE_PROTOTYPES,
    **ORGANELLE_PROTOTYPES,
    **CELL_PROTOTYPES,
    **PARTICLE_PROTOTYPES,
    **STRUCTURE_PROTOTYPES,
    **IMMUNE_CELL_PROTOTYPES,
    **IMMUNE_PARTICLE_PROTOTYPES,
}


class PrototypeLibrary:
    """
    Library of pre-defined biological prototypes.

    Provides:
    1. Pre-defined specs for common biological entities
    2. Registration helpers for custom prototypes
    3. Automatic material creation
    4. LOD variant generation
    """

    def __init__(self, instancing_manager: "NestedInstancingManager"):
        """
        Initialize the prototype library.

        Args:
            instancing_manager: NestedInstancingManager to register prototypes with
        """
        if not USD_AVAILABLE:
            raise RuntimeError("OpenUSD required for PrototypeLibrary")

        self.manager = instancing_manager
        self.stage = instancing_manager.stage
        self._registered: Dict[str, PrototypeSpec] = {}

    def register_standard_prototype(self, name: str) -> Optional[PrototypeDefinition]:
        """
        Register a standard prototype by name.

        Args:
            name: Name of prototype from ALL_PROTOTYPES

        Returns:
            PrototypeDefinition if successful
        """
        if name not in ALL_PROTOTYPES:
            log.error(f"Unknown prototype: {name}")
            return None

        spec = ALL_PROTOTYPES[name]
        return self._register_from_spec(spec)

    def register_all_standard(self, categories: Optional[List[PrototypeCategory]] = None) -> int:
        """
        Register all standard prototypes, optionally filtered by category.

        Args:
            categories: Categories to register (None = all)

        Returns:
            Number of prototypes registered
        """
        count = 0

        for name, spec in ALL_PROTOTYPES.items():
            if categories is None or spec.category in categories:
                if self._register_from_spec(spec):
                    count += 1

        log.info(f"Registered {count} standard prototypes")
        return count

    def _register_from_spec(self, spec: PrototypeSpec) -> Optional[PrototypeDefinition]:
        """Register a prototype from a spec."""
        if spec.name in self._registered:
            log.debug(f"Prototype '{spec.name}' already registered")
            return self.manager._prototypes.get(spec.name)

        # Create geometry creator based on type
        geometry_creator = self._get_geometry_creator(spec)

        definition = self.manager.register_prototype(
            name=spec.name,
            geometry_creator=geometry_creator,
            category=spec.category.value,
            suggested_strategy=spec.suggested_strategy,
            make_instanceable=True
        )

        if definition:
            self._registered[spec.name] = spec

            # Create material for prototype
            self._create_prototype_material(spec)

            # Create LOD variants if needed
            if spec.lod_levels > 1:
                self._create_lod_variants(spec)

        return definition

    def _get_geometry_creator(self, spec: PrototypeSpec) -> Callable:
        """Get a geometry creator function for a spec."""
        def creator(stage: "Usd.Stage", path: str) -> None:
            if spec.geometry_type == "sphere":
                create_sphere_prototype(stage, path, radius=spec.default_size / 2)

            elif spec.geometry_type == "capsule":
                params = spec.geometry_params
                create_capsule_prototype(
                    stage, path,
                    radius=params.get("radius", spec.default_size / 4),
                    height=params.get("height", spec.default_size)
                )

            elif spec.geometry_type == "ellipsoid":
                params = spec.geometry_params
                radii = params.get("radii", (1.0, 1.0, 1.0))
                # Scale radii by default_size
                scaled_radii = tuple(r * spec.default_size / 2 for r in radii)
                create_ellipsoid_prototype(stage, path, radii=scaled_radii)

            else:
                # Default to sphere
                create_sphere_prototype(stage, path, radius=spec.default_size / 2)

        return creator

    def _create_prototype_material(self, spec: PrototypeSpec) -> None:
        """Create a material for a prototype."""
        mat_path = f"/Prototypes/Materials/{spec.name}_Material"

        # Create material
        material = UsdShade.Material.Define(self.stage, mat_path)

        # Create shader
        shader = UsdShade.Shader.Define(self.stage, f"{mat_path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")

        # Set properties
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*spec.color)
        )
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(spec.opacity)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)

        # Connect to material
        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )

        # Bind to prototype
        proto_prim = self.manager._prototype_prims.get(spec.name)
        if proto_prim:
            UsdShade.MaterialBindingAPI(proto_prim).Bind(material)

    def _create_lod_variants(self, spec: PrototypeSpec) -> None:
        """Create LOD variants for a prototype."""
        proto_prim = self.manager._prototype_prims.get(spec.name)
        if not proto_prim:
            return

        # Add LOD variant set
        variant_sets = proto_prim.GetVariantSets()
        lod_set = variant_sets.AddVariantSet("LOD")

        for level in range(spec.lod_levels):
            variant_name = f"LOD{level}"
            lod_set.AddVariant(variant_name)

            with lod_set.GetVariantEditContext():
                # In a full implementation, we would:
                # - Create simplified geometry for higher LOD levels
                # - Reduce vertex count
                # - Simplify materials
                pass

        # Set default to highest quality
        lod_set.SetVariantSelection("LOD0")

    def get_spec(self, name: str) -> Optional[PrototypeSpec]:
        """Get the specification for a prototype."""
        return ALL_PROTOTYPES.get(name) or self._registered.get(name)

    def list_available(self, category: Optional[PrototypeCategory] = None) -> List[str]:
        """List available prototype names, optionally by category."""
        if category is None:
            return list(ALL_PROTOTYPES.keys())

        return [
            name for name, spec in ALL_PROTOTYPES.items()
            if spec.category == category
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get library statistics."""
        return {
            "registered_count": len(self._registered),
            "available_count": len(ALL_PROTOTYPES),
            "categories": [c.value for c in PrototypeCategory],
            "registered": list(self._registered.keys())
        }


# ── Convenience Functions ────────────────────────────────────────────────

def create_biological_scene(
    stage: "Usd.Stage",
    cell_count: int = 100,
    include_organelles: bool = True
) -> NestedInstancingManager:
    """
    Create a demo biological scene with cells and organelles.

    Args:
        stage: USD stage
        cell_count: Number of cells to create
        include_organelles: Whether to include organelles in cells

    Returns:
        NestedInstancingManager with populated scene
    """
    import numpy as np

    manager = NestedInstancingManager(stage)
    library = PrototypeLibrary(manager)

    # Register needed prototypes
    library.register_standard_prototype("generic_cell")
    library.register_standard_prototype("nucleus")

    if include_organelles:
        library.register_standard_prototype("mitochondrion")
        library.register_standard_prototype("ribosome")

    # Generate cell positions in a cluster
    np.random.seed(42)
    cell_positions = np.random.randn(cell_count, 3) * 100  # 100 μm spread

    from .instancing import InstanceData

    # Create cells
    cell_instances = [
        InstanceData(
            prototype_name="generic_cell",
            position=tuple(pos)
        )
        for pos in cell_positions
    ]

    manager.create_instances_batch("generic_cell", cell_instances)

    log.info(f"Created biological scene with {cell_count} cells")
    return manager
