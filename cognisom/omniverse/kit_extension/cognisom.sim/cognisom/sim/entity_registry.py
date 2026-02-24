"""
Biological Entity Registry
===========================

Configuration-driven registry mapping each biological entity to its mesh
source, material, biological scale, LOD levels, and instancing strategy.

Adding a new entity requires:
1. Add a mesh generator function in bio_mesh_gen.py
2. Add an EntitySpec entry in ENTITY_REGISTRY below
3. Run scripts/generate_bio_assets.py to create the .usd file
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class EntitySpec:
    """Specification for a biological entity's USD mesh asset."""

    name: str
    """Unique identifier, e.g. 'rbc', 'neutrophil_body'."""

    asset_filename: str
    """Filename in the assets/ directory, e.g. 'rbc.usd'."""

    default_material: str
    """Key into the MATERIALS dict for default material binding."""

    biological_diameter_um: float
    """Real biological diameter in micrometers (for documentation/scaling)."""

    mesh_generator: str
    """Function name in bio_mesh_gen module that generates this mesh."""

    instancing: str = "reference"
    """'reference' for individual prims, 'point_instancer' for PointInstancer."""

    subdiv_scheme: str = "catmullClark"
    """USD subdivision scheme. 'catmullClark' for organic, 'none' for flat."""

    subdiv_level: int = 1
    """Subdivision refinement level for RTX rendering."""

    sub_materials: Dict[str, str] = field(default_factory=dict)
    """Named sub-prim → material overrides.
    e.g. {"lectin_head": "selectin_head", "tip": "icam1_tip"}"""

    multi_part: bool = False
    """True if mesh generator returns Dict[str, MeshData] instead of MeshData."""


# ── Entity Registry ───────────────────────────────────────────────────

ENTITY_REGISTRY: Dict[str, EntitySpec] = {

    # ── Blood Cells ───────────────────────────────────────────────────

    "rbc": EntitySpec(
        name="rbc",
        asset_filename="rbc.usd",
        default_material="rbc",
        biological_diameter_um=7.5,
        mesh_generator="generate_rbc_mesh",
        instancing="point_instancer",
        subdiv_scheme="catmullClark",
        subdiv_level=1,
    ),

    "neutrophil_body": EntitySpec(
        name="neutrophil_body",
        asset_filename="neutrophil_body.usd",
        default_material="leuko_flowing",
        biological_diameter_um=12.0,
        mesh_generator="generate_neutrophil_body_mesh",
        subdiv_scheme="catmullClark",
        subdiv_level=1,
    ),

    "neutrophil_nucleus": EntitySpec(
        name="neutrophil_nucleus",
        asset_filename="neutrophil_nucleus.usd",
        default_material="nucleus",
        biological_diameter_um=5.0,
        mesh_generator="generate_neutrophil_nucleus_mesh",
        subdiv_scheme="catmullClark",
        subdiv_level=1,
    ),

    # ── Surface Molecules ─────────────────────────────────────────────

    "selectin": EntitySpec(
        name="selectin",
        asset_filename="selectin.usd",
        default_material="selectin_stalk",
        biological_diameter_um=0.03,  # 30nm exaggerated for visibility
        mesh_generator="generate_selectin_mesh",
        multi_part=True,
        sub_materials={"lectin_head": "selectin_head"},
        subdiv_scheme="catmullClark",
        subdiv_level=1,
    ),

    "icam1": EntitySpec(
        name="icam1",
        asset_filename="icam1.usd",
        default_material="icam1",
        biological_diameter_um=0.019,  # 19nm exaggerated
        mesh_generator="generate_icam1_mesh",
        multi_part=True,
        sub_materials={"tip": "icam1_tip"},
        subdiv_scheme="catmullClark",
        subdiv_level=0,
    ),

    "pecam1": EntitySpec(
        name="pecam1",
        asset_filename="pecam1.usd",
        default_material="pecam1",
        biological_diameter_um=0.015,
        mesh_generator="generate_pecam1_mesh",
        subdiv_scheme="catmullClark",
        subdiv_level=0,
    ),

    # ── Tissue Elements ───────────────────────────────────────────────

    "ecoli": EntitySpec(
        name="ecoli",
        asset_filename="ecoli.usd",
        default_material="bacterium_body",
        biological_diameter_um=2.0,
        mesh_generator="generate_ecoli_mesh",
        multi_part=True,
        sub_materials={
            "flagella": "bacterium_body",
            "complement": "complement_c3b",
        },
        subdiv_scheme="catmullClark",
        subdiv_level=1,
    ),

    "macrophage": EntitySpec(
        name="macrophage",
        asset_filename="macrophage.usd",
        default_material="macrophage",
        biological_diameter_um=20.0,
        mesh_generator="generate_macrophage_mesh",
        subdiv_scheme="catmullClark",
        subdiv_level=1,
    ),

    "endothelial_cell": EntitySpec(
        name="endothelial_cell",
        asset_filename="endothelial_cell.usd",
        default_material="endothelial_healthy",
        biological_diameter_um=15.0,
        mesh_generator="generate_endothelial_cell_mesh",
        subdiv_scheme="none",
        subdiv_level=0,
    ),

    # ── ECM / Fibers ──────────────────────────────────────────────────

    "fibrin_fiber": EntitySpec(
        name="fibrin_fiber",
        asset_filename="fibrin_fiber.usd",
        default_material="fibrin",
        biological_diameter_um=0.1,
        mesh_generator="generate_fibrin_fiber_mesh",
        subdiv_scheme="none",
        subdiv_level=0,
    ),

    "collagen_helix": EntitySpec(
        name="collagen_helix",
        asset_filename="collagen_helix.usd",
        default_material="collagen",
        biological_diameter_um=0.3,
        mesh_generator="generate_collagen_helix_mesh",
        subdiv_scheme="none",
        subdiv_level=0,
    ),
}


# ── Asset Path Resolution ─────────────────────────────────────────────

# Extension root: go up from cognisom/sim/ → cognisom/ → cognisom.sim/
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXTENSION_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_ASSETS_DIR = os.path.join(_EXTENSION_ROOT, "assets")


def get_asset_path(entity_name: str) -> str:
    """Resolve entity name to absolute .usd file path.

    In Docker: /exts/cognisom.sim/assets/rbc.usd
    Locally:   .../kit_extension/cognisom.sim/assets/rbc.usd
    """
    spec = ENTITY_REGISTRY.get(entity_name)
    if not spec:
        raise KeyError(f"Unknown entity: {entity_name}")
    return os.path.join(_ASSETS_DIR, spec.asset_filename)


def get_all_asset_paths() -> Dict[str, str]:
    """Return all entity → asset path mappings."""
    return {name: get_asset_path(name) for name in ENTITY_REGISTRY}


def verify_assets() -> Dict[str, bool]:
    """Check which assets exist on disk. Returns name → exists mapping."""
    return {
        name: os.path.isfile(get_asset_path(name))
        for name in ENTITY_REGISTRY
    }
