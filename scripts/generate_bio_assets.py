#!/usr/bin/env python3
"""
Generate Biological USD Mesh Assets
=====================================

Build-time script that generates high-fidelity .usd mesh files for every
biological entity in the diapedesis scene. Run once; commit the results.

Usage:
    python scripts/generate_bio_assets.py              # Generate all assets
    python scripts/generate_bio_assets.py --entity rbc # Generate one entity
    python scripts/generate_bio_assets.py --verify     # Verify existing assets

Output: cognisom/omniverse/kit_extension/cognisom.sim/assets/*.usd

Requirements: numpy, pxr (OpenUSD Python bindings via `pip install usd-core`)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Add project root to path so we can import from the extension
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
EXTENSION_SIM = os.path.join(
    PROJECT_ROOT,
    "cognisom", "omniverse", "kit_extension", "cognisom.sim", "cognisom", "sim",
)
sys.path.insert(0, EXTENSION_SIM)

import numpy as np

# Import mesh generators
from bio_mesh_gen import (
    generate_rbc_mesh,
    generate_neutrophil_body_mesh,
    generate_neutrophil_nucleus_mesh,
    generate_selectin_mesh,
    generate_icam1_mesh,
    generate_pecam1_mesh,
    generate_ecoli_mesh,
    generate_macrophage_mesh,
    generate_endothelial_cell_mesh,
    generate_fibrin_fiber_mesh,
    generate_collagen_helix_mesh,
)
from entity_registry import ENTITY_REGISTRY, EntitySpec

# USD imports (usd-core pip package)
try:
    from pxr import Usd, UsdGeom, Sdf, Vt, Gf
except ImportError:
    print("ERROR: pxr (OpenUSD) not available.")
    print("Install with: pip install usd-core")
    sys.exit(1)


# Map generator names to functions
GENERATORS = {
    "generate_rbc_mesh": generate_rbc_mesh,
    "generate_neutrophil_body_mesh": generate_neutrophil_body_mesh,
    "generate_neutrophil_nucleus_mesh": generate_neutrophil_nucleus_mesh,
    "generate_selectin_mesh": generate_selectin_mesh,
    "generate_icam1_mesh": generate_icam1_mesh,
    "generate_pecam1_mesh": generate_pecam1_mesh,
    "generate_ecoli_mesh": generate_ecoli_mesh,
    "generate_macrophage_mesh": generate_macrophage_mesh,
    "generate_endothelial_cell_mesh": generate_endothelial_cell_mesh,
    "generate_fibrin_fiber_mesh": generate_fibrin_fiber_mesh,
    "generate_collagen_helix_mesh": generate_collagen_helix_mesh,
}

ASSETS_DIR = os.path.join(
    PROJECT_ROOT,
    "cognisom", "omniverse", "kit_extension", "cognisom.sim", "assets",
)


def _write_mesh_to_prim(
    stage: Usd.Stage,
    prim_path: str,
    mesh_data: tuple,
    subdiv_scheme: str = "catmullClark",
) -> None:
    """Write mesh data tuple to a UsdGeom.Mesh prim."""
    vertices, face_counts, face_indices, normals = mesh_data

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(vertices))
    mesh.CreateFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(face_counts))
    mesh.CreateFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(face_indices))
    mesh.CreateNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals))
    mesh.SetNormalsInterpolation("vertex")
    mesh.CreateSubdivisionSchemeAttr().Set(subdiv_scheme)

    n_verts = len(vertices)
    n_faces = len(face_counts)
    print(f"    {prim_path}: {n_verts} verts, {n_faces} faces, subdiv={subdiv_scheme}")


def generate_single_mesh_asset(
    spec: EntitySpec, output_path: str
) -> None:
    """Generate a .usd file for a single-mesh entity."""
    gen_func = GENERATORS[spec.mesh_generator]

    stage = Usd.Stage.CreateNew(output_path)
    stage.SetMetadata("comment", f"Cognisom biological mesh: {spec.name}")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

    root = UsdGeom.Xform.Define(stage, "/mesh")

    # LOD0 (high detail) — default
    mesh_data_lod0 = gen_func(lod=0)
    _write_mesh_to_prim(stage, "/mesh/geo", mesh_data_lod0, spec.subdiv_scheme)

    # Add LOD VariantSet
    root_prim = root.GetPrim()
    vset = root_prim.GetVariantSets().AddVariantSet("LOD")
    vset.AddVariant("LOD0")
    vset.AddVariant("LOD1")

    # LOD0 is default (geometry already in place)
    vset.SetVariantSelection("LOD0")

    stage.GetRootLayer().Save()


def generate_multi_part_asset(
    spec: EntitySpec, output_path: str
) -> None:
    """Generate a .usd file for a multi-part entity (returns Dict[str, MeshData])."""
    gen_func = GENERATORS[spec.mesh_generator]

    stage = Usd.Stage.CreateNew(output_path)
    stage.SetMetadata("comment", f"Cognisom biological mesh: {spec.name}")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

    UsdGeom.Xform.Define(stage, "/mesh")

    # Generate LOD0 parts
    parts = gen_func(lod=0)
    for part_name, mesh_data in parts.items():
        prim_path = f"/mesh/{part_name}"
        _write_mesh_to_prim(stage, prim_path, mesh_data, spec.subdiv_scheme)

    stage.GetRootLayer().Save()


def generate_entity(name: str) -> str:
    """Generate .usd asset for one entity. Returns output path."""
    spec = ENTITY_REGISTRY[name]
    output_path = os.path.join(ASSETS_DIR, spec.asset_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"  Generating {spec.name} → {spec.asset_filename}")
    t0 = time.time()

    if spec.multi_part:
        generate_multi_part_asset(spec, output_path)
    else:
        generate_single_mesh_asset(spec, output_path)

    size_kb = os.path.getsize(output_path) / 1024
    elapsed = time.time() - t0
    print(f"    → {size_kb:.1f} KB in {elapsed:.2f}s")
    return output_path


def verify_assets() -> bool:
    """Verify all expected .usd files exist and are valid."""
    all_ok = True
    for name, spec in ENTITY_REGISTRY.items():
        path = os.path.join(ASSETS_DIR, spec.asset_filename)
        if not os.path.isfile(path):
            print(f"  MISSING: {spec.asset_filename}")
            all_ok = False
            continue

        # Try to open
        try:
            stage = Usd.Stage.Open(path)
            mesh_prim = stage.GetPrimAtPath("/mesh")
            if not mesh_prim:
                print(f"  INVALID: {spec.asset_filename} — no /mesh prim")
                all_ok = False
                continue

            # Count geometry
            n_meshes = 0
            for prim in Usd.PrimRange(mesh_prim):
                if prim.IsA(UsdGeom.Mesh):
                    n_meshes += 1

            size_kb = os.path.getsize(path) / 1024
            print(f"  OK: {spec.asset_filename} ({n_meshes} meshes, {size_kb:.1f} KB)")

        except Exception as e:
            print(f"  ERROR: {spec.asset_filename} — {e}")
            all_ok = False

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Generate biological USD mesh assets")
    parser.add_argument("--entity", help="Generate only this entity (by name)")
    parser.add_argument("--verify", action="store_true", help="Verify existing assets")
    args = parser.parse_args()

    print("=" * 60)
    print("Cognisom Biological Mesh Asset Generator")
    print("=" * 60)
    print(f"Output: {ASSETS_DIR}")
    print()

    if args.verify:
        print("Verifying assets...")
        ok = verify_assets()
        print()
        if ok:
            print("All assets valid.")
        else:
            print("Some assets missing or invalid!")
            sys.exit(1)
        return

    if args.entity:
        if args.entity not in ENTITY_REGISTRY:
            print(f"Unknown entity: {args.entity}")
            print(f"Available: {', '.join(ENTITY_REGISTRY.keys())}")
            sys.exit(1)
        generate_entity(args.entity)
    else:
        print(f"Generating {len(ENTITY_REGISTRY)} assets...")
        print()
        t_total = time.time()
        for name in ENTITY_REGISTRY:
            generate_entity(name)
        elapsed = time.time() - t_total
        print()
        print(f"All assets generated in {elapsed:.2f}s")

    # Verify
    print()
    print("Verification:")
    verify_assets()


if __name__ == "__main__":
    main()
