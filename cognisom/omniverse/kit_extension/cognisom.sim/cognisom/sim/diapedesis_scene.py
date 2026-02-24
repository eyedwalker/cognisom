"""
Diapedesis USD Scene Builder
=============================

Builds the diapedesis visualization as a USD scene for Omniverse Kit
with RTX-quality materials (subsurface scattering, volumetrics, PBR).

Loads frame snapshots from DiapedesisSim and creates/updates USD prims
for vessel, leukocytes, RBCs, endothelium, surface molecules, bacteria,
macrophages, fibrin mesh, collagen ECM, and chemokine gradient.

This is the Omniverse counterpart of the Three.js viewer in
cognisom/dashboard/pages/25_diapedesis.py — same data, RTX rendering.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

try:
    from pxr import Usd, UsdGeom, UsdShade, UsdLux, Gf, Sdf, Vt
    USD_AVAILABLE = True
except ImportError:
    USD_AVAILABLE = False

try:
    from .entity_registry import ENTITY_REGISTRY, get_asset_path, verify_assets
    from .mdl_materials import create_omnipbr_material
    _HAS_MESH_ASSETS = True
except ImportError:
    _HAS_MESH_ASSETS = False


# ── Constants ────────────────────────────────────────────────────────────

# Scene hierarchy paths
ROOT = "/World/Diapedesis"
VESSEL_PATH = f"{ROOT}/Vessel"
LUMEN_PATH = f"{ROOT}/Lumen"
LEUKOCYTES_PATH = f"{LUMEN_PATH}/Leukocytes"
RBCS_PATH = f"{LUMEN_PATH}/RBCs"
ENDOTHELIUM_PATH = f"{ROOT}/Endothelium"
MOLECULES_PATH = f"{ROOT}/Molecules"
SELECTINS_PATH = f"{MOLECULES_PATH}/Selectins"
ICAM1_PATH = f"{MOLECULES_PATH}/ICAM1"
PECAM1_PATH = f"{MOLECULES_PATH}/PECAM1"
TISSUE_PATH = f"{ROOT}/Tissue"
BACTERIA_PATH = f"{TISSUE_PATH}/Bacteria"
MACROPHAGES_PATH = f"{TISSUE_PATH}/Macrophages"
FIBRIN_PATH = f"{TISSUE_PATH}/Fibrin"
COLLAGEN_PATH = f"{TISSUE_PATH}/Collagen"
CHEMOKINE_PATH = f"{TISSUE_PATH}/Chemokines"
MATERIALS_PATH = f"{ROOT}/Materials"
LIGHTS_PATH = f"{ROOT}/Lights"
PROTOTYPES_PATH = f"{ROOT}/Prototypes"


# ── Material Definitions ────────────────────────────────────────────────

MATERIALS = {
    # Vessel & endothelium
    "vessel_wall": {
        "color": (0.91, 0.69, 0.69), "roughness": 0.6,
        "opacity": 0.18, "subsurface": 0.3,
    },
    "basement_membrane": {
        "color": (0.55, 0.45, 0.38), "roughness": 0.85,
        "opacity": 0.12,
    },
    "endothelial_healthy": {
        "color": (0.9, 0.75, 0.75), "roughness": 0.65,
        "subsurface": 0.4, "subsurface_color": (0.95, 0.6, 0.6),
    },
    "endothelial_inflamed": {
        "color": (0.9, 0.35, 0.35), "roughness": 0.55,
        "emissive": (0.1, 0.02, 0.02), "subsurface": 0.5,
    },

    # Leukocyte states
    "leuko_flowing": {
        "color": (0.92, 0.92, 0.95), "roughness": 0.7,
        "subsurface": 0.6, "subsurface_color": (0.85, 0.85, 0.9),
        "opacity": 0.85,
    },
    "leuko_rolling": {
        "color": (1.0, 0.85, 0.0), "roughness": 0.6,
        "subsurface": 0.5, "subsurface_color": (1.0, 0.7, 0.0),
    },
    "leuko_arrested": {
        "color": (1.0, 0.3, 0.0), "roughness": 0.55,
        "emissive": (0.08, 0.02, 0.0), "subsurface": 0.5,
    },
    "leuko_transmigrating": {
        "color": (0.8, 0.0, 0.5), "roughness": 0.5,
        "opacity": 0.6, "subsurface": 0.7,
    },
    "leuko_migrated": {
        "color": (0.2, 0.8, 0.2), "roughness": 0.6,
        "emissive": (0.02, 0.06, 0.02), "subsurface": 0.5,
    },

    # Nucleus (visible through translucent body)
    "nucleus": {
        "color": (0.35, 0.28, 0.65), "roughness": 0.8,
        "opacity": 0.7, "subsurface": 0.3,
    },

    # RBC
    "rbc": {
        "color": (0.72, 0.11, 0.11), "roughness": 0.5,
        "subsurface": 0.8, "subsurface_color": (0.9, 0.15, 0.1),
        "opacity": 0.88,
    },

    # Surface molecules
    "selectin_stalk": {
        "color": (0.8, 0.67, 0.13), "roughness": 0.6,
    },
    "selectin_head": {
        "color": (1.0, 0.67, 0.0), "roughness": 0.45,
        "emissive": (0.12, 0.06, 0.0),
    },
    "icam1": {
        "color": (0.27, 0.53, 0.93), "roughness": 0.55,
    },
    "icam1_tip": {
        "color": (0.4, 0.65, 1.0), "roughness": 0.45,
        "emissive": (0.05, 0.08, 0.15),
    },
    "pecam1": {
        "color": (0.2, 0.73, 0.53), "roughness": 0.6,
    },
    "integrin_bent": {
        "color": (0.27, 0.53, 0.6), "roughness": 0.6,
    },
    "integrin_extended": {
        "color": (0.0, 0.8, 0.87), "roughness": 0.45,
        "emissive": (0.0, 0.05, 0.06),
    },

    # Tissue
    "bacterium_body": {
        "color": (0.2, 0.4, 0.2), "roughness": 0.7,
        "subsurface": 0.3,
    },
    "complement_c3b": {
        "color": (0.8, 0.6, 0.2), "roughness": 0.5,
        "emissive": (0.06, 0.04, 0.01), "opacity": 0.85,
    },
    "macrophage": {
        "color": (0.85, 0.35, 0.15), "roughness": 0.6,
        "subsurface": 0.5, "subsurface_color": (0.9, 0.3, 0.1),
        "opacity": 0.82,
    },
    "fibrin": {
        "color": (0.95, 0.9, 0.7), "roughness": 0.8,
        "opacity": 0.4,
    },
    "collagen": {
        "color": (0.8, 0.75, 0.6), "roughness": 0.85,
        "opacity": 0.35,
    },
    "chemokine": {
        "color": (0.33, 0.73, 1.0), "roughness": 0.9,
        "opacity": 0.15, "emissive": (0.02, 0.06, 0.1),
    },
    "infection_glow": {
        "color": (1.0, 0.27, 0.13), "roughness": 0.9,
        "opacity": 0.08, "emissive": (0.3, 0.08, 0.04),
    },
}

# State index → material name
LEUKO_STATE_MATERIALS = {
    0: "leuko_flowing",      # FLOWING
    1: "leuko_flowing",      # MARGINATING
    2: "leuko_rolling",      # ROLLING
    3: "leuko_rolling",      # ACTIVATING
    4: "leuko_arrested",     # ARRESTED
    5: "leuko_arrested",     # CRAWLING
    6: "leuko_transmigrating",  # TRANSMIGRATING
    7: "leuko_migrated",     # MIGRATED
}


class DiapedesisSceneBuilder:
    """Builds and updates the diapedesis USD scene.

    Usage::

        builder = DiapedesisSceneBuilder(stage)
        builder.build_scene(first_frame)    # Create all geometry
        builder.apply_frame(frame_data)     # Update per-frame state
    """

    def __init__(self, stage: "Usd.Stage"):
        if not USD_AVAILABLE:
            raise RuntimeError("pxr (USD) not available")
        self._stage = stage
        self._materials: Dict[str, str] = {}  # name → prim path
        self._leuko_prims: List[str] = []
        self._rbc_instancer: Optional[str] = None
        self._endo_prims: List[str] = []
        self._selectin_prims: List[str] = []
        self._pecam_prims: List[Tuple[str, int]] = []  # (path, endo_idx)
        self._bacteria_prims: List[str] = []
        self._n_leuko = 0
        self._n_rbc = 0
        self._n_endo = 0
        self._n_bacteria = 0
        self._vessel_radius = 25.0
        self._vessel_length = 200.0

        # Cached USD attribute handles for per-frame updates
        # (avoids ClearXformOpOrder + AddTranslateOp schema churn)
        self._leuko_translate_attrs: List[Optional[Any]] = []
        self._leuko_scale_attrs: List[Optional[Any]] = []
        self._leuko_last_state: List[int] = []  # cache to skip redundant material binds
        self._endo_last_mat: List[str] = []  # cache to skip redundant endo material binds
        self._bacteria_scale_attrs: List[Optional[Any]] = []

        # Check if high-fidelity mesh assets are available
        self._use_mesh_assets = False
        if _HAS_MESH_ASSETS:
            asset_status = verify_assets()
            self._use_mesh_assets = all(asset_status.values())
            if self._use_mesh_assets:
                self._asset_paths = {
                    name: get_asset_path(name) for name in ENTITY_REGISTRY
                }
                log.info("[cognisom.sim] Using high-fidelity mesh assets")
            else:
                missing = [n for n, ok in asset_status.items() if not ok]
                log.warning(f"[cognisom.sim] Missing mesh assets: {missing}. "
                            f"Falling back to primitive geometry.")

    # ── Scene Construction ──────────────────────────────────────────────

    def build_scene(self, frame: Dict[str, Any]) -> None:
        """Build the complete diapedesis scene from first frame data.

        Note: build_scene does NOT use Sdf.ChangeBlock because DefinePrim
        fails inside ChangeBlock contexts. The per-frame apply_frame()
        method uses ChangeBlock for batching position/attribute updates.
        """
        self._vessel_radius = frame.get("vessel_radius", 25.0)
        self._vessel_length = frame.get("vessel_length", 200.0)
        R = self._vessel_radius
        L = self._vessel_length

        # Create hierarchy
        self._create_hierarchy()

        # Create all materials
        self._create_all_materials()

        # Create lighting
        self._create_lighting()

        # Build vessel geometry
        self._build_vessel()

        # Build endothelial cells
        self._build_endothelium(frame)

        # Build surface molecules (selectins, ICAM-1, PECAM-1)
        self._build_selectins(frame)
        self._build_icam1(frame)
        self._build_pecam1(frame)

        # Build leukocytes (individual prims with prototypes)
        self._build_leukocytes(frame)

        # Build RBCs (PointInstancer for 200 biconcave discs)
        self._build_rbcs(frame)

        # Build tissue scene
        self._build_bacteria(frame)
        self._build_macrophages(frame)
        self._build_fibrin()
        self._build_collagen()
        self._build_chemokine_cloud()

        # Infection site glow
        self._build_infection_glow(frame)

        log.info(f"DiapedesisSceneBuilder: scene built ({self._n_leuko} leukocytes, "
                 f"{self._n_rbc} RBCs, {self._n_endo} endo, {self._n_bacteria} bacteria)")

    def _create_hierarchy(self):
        """Create USD scene hierarchy."""
        paths = [
            ROOT, VESSEL_PATH, LUMEN_PATH, LEUKOCYTES_PATH, RBCS_PATH,
            ENDOTHELIUM_PATH, MOLECULES_PATH, SELECTINS_PATH, ICAM1_PATH,
            PECAM1_PATH, TISSUE_PATH, BACTERIA_PATH, MACROPHAGES_PATH,
            FIBRIN_PATH, COLLAGEN_PATH, CHEMOKINE_PATH,
            MATERIALS_PATH, LIGHTS_PATH, PROTOTYPES_PATH,
        ]
        for p in paths:
            if not self._stage.GetPrimAtPath(p):
                UsdGeom.Xform.Define(self._stage, p)

    def _create_all_materials(self):
        """Create all PBR materials (OmniPBR MDL + UsdPreviewSurface fallback)."""
        for name, spec in MATERIALS.items():
            path = f"{MATERIALS_PATH}/{name}"
            if _HAS_MESH_ASSETS:
                self._materials[name] = create_omnipbr_material(
                    self._stage, path, spec)
            else:
                self._materials[name] = self._create_pbr_material(path, spec)

    def _create_pbr_material(self, path: str, spec: Dict) -> str:
        """Create a UsdPreviewSurface material (legacy fallback)."""
        material = UsdShade.Material.Define(self._stage, path)
        shader = UsdShade.Shader.Define(self._stage, f"{path}/PBRShader")
        shader.CreateIdAttr("UsdPreviewSurface")

        color = spec.get("color", (0.5, 0.5, 0.5))
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
            spec.get("roughness", 0.7))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(
            spec.get("metallic", 0.0))
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(
            spec.get("opacity", 1.0))

        emissive = spec.get("emissive", (0, 0, 0))
        if any(e > 0 for e in emissive):
            shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*emissive))

        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface")

        return path

    def _bind_material(self, prim_path: str, mat_name: str,
                       override_descendants: bool = False):
        """Bind a material to a prim.

        Args:
            override_descendants: If True, use strongerThanDescendants
                binding strength to override materials in referenced meshes.
        """
        mat_path = self._materials.get(mat_name)
        if not mat_path:
            return
        prim = self._stage.GetPrimAtPath(prim_path)
        if not prim:
            return
        material = UsdShade.Material.Get(self._stage, mat_path)
        UsdShade.MaterialBindingAPI.Apply(prim)
        if override_descendants:
            UsdShade.MaterialBindingAPI(prim).Bind(
                material, UsdShade.Tokens.strongerThanDescendants)
        else:
            UsdShade.MaterialBindingAPI(prim).Bind(material)

    # ── Lighting ─────────────────────────────────────────────────────────

    def _create_lighting(self):
        """Create scene lighting for dramatic biological visualization."""
        R = self._vessel_radius
        L = self._vessel_length

        # Key light (warm, directional)
        key = UsdLux.DistantLight.Define(self._stage, f"{LIGHTS_PATH}/KeyLight")
        key.CreateIntensityAttr().Set(800.0)
        key.CreateColorAttr().Set(Gf.Vec3f(1.0, 0.96, 0.91))
        xf = UsdGeom.Xformable(key.GetPrim())
        xf.AddRotateXYZOp().Set(Gf.Vec3f(-40, 30, 0))

        # Fill light (cool blue)
        fill = UsdLux.DistantLight.Define(self._stage, f"{LIGHTS_PATH}/FillLight")
        fill.CreateIntensityAttr().Set(300.0)
        fill.CreateColorAttr().Set(Gf.Vec3f(0.7, 0.8, 1.0))
        xf = UsdGeom.Xformable(fill.GetPrim())
        xf.AddRotateXYZOp().Set(Gf.Vec3f(20, -150, 0))

        # Rim light (warm accent)
        rim = UsdLux.DistantLight.Define(self._stage, f"{LIGHTS_PATH}/RimLight")
        rim.CreateIntensityAttr().Set(200.0)
        rim.CreateColorAttr().Set(Gf.Vec3f(1.0, 0.7, 0.5))
        xf = UsdGeom.Xformable(rim.GetPrim())
        xf.AddRotateXYZOp().Set(Gf.Vec3f(60, 180, 0))

        # Dome light for ambient
        dome = UsdLux.DomeLight.Define(self._stage, f"{LIGHTS_PATH}/DomeLight")
        dome.CreateIntensityAttr().Set(100.0)
        dome.CreateColorAttr().Set(Gf.Vec3f(0.04, 0.04, 0.07))

    # ── Vessel Geometry ──────────────────────────────────────────────────

    def _build_vessel(self):
        """Build the vessel wall as a half-cylinder cutaway."""
        R = self._vessel_radius
        L = self._vessel_length

        # Vessel wall — half cylinder (cutaway)
        # USD doesn't have a native half-cylinder, so we use a full cylinder
        # with transform and rely on camera clipping for cutaway effect.
        # In Kit, a mesh-based half-cylinder would be better.
        vessel = UsdGeom.Cylinder.Define(self._stage, f"{VESSEL_PATH}/Wall")
        vessel.CreateRadiusAttr().Set(R)
        vessel.CreateHeightAttr().Set(L)
        vessel.CreateAxisAttr().Set("X")
        xf = UsdGeom.Xformable(vessel.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(L / 2, 0, 0))
        self._bind_material(vessel.GetPath().pathString, "vessel_wall")

        # Basement membrane (slightly larger cylinder)
        bm = UsdGeom.Cylinder.Define(self._stage, f"{VESSEL_PATH}/BasementMembrane")
        bm.CreateRadiusAttr().Set(R * 1.04)
        bm.CreateHeightAttr().Set(L * 0.95)
        bm.CreateAxisAttr().Set("X")
        xf = UsdGeom.Xformable(bm.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(L / 2, 0, 0))
        self._bind_material(bm.GetPath().pathString, "basement_membrane")

    # ── Endothelial Cells ────────────────────────────────────────────────

    def _build_endothelium(self, frame: Dict):
        """Build endothelial cells on vessel wall.

        With mesh assets: references endothelial_cell.usd (hexagonal tile with relief).
        Fallback: flattened spheres.
        """
        positions = frame.get("endo_positions", [])
        selectin_expr = frame.get("endo_selectin_expr", [])
        self._n_endo = len(positions)
        self._endo_prims = []
        self._endo_last_mat = []

        for i, pos in enumerate(positions):
            path = f"{ENDOTHELIUM_PATH}/endo_{i:03d}"

            if self._use_mesh_assets:
                xform = UsdGeom.Xform.Define(self._stage, path)
                xform.GetPrim().GetReferences().AddReference(
                    self._asset_paths["endothelial_cell"])
                xf = UsdGeom.Xformable(xform.GetPrim())
                xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
                # Scale hexagonal tile to ~10 unit radius, flatten slightly
                xf.AddScaleOp().Set(Gf.Vec3f(10.0, 10.0, 10.0))
                xf.AddRotateXYZOp().Set(Gf.Vec3f(0, 0, 0))
            else:
                sphere = UsdGeom.Sphere.Define(self._stage, path)
                sphere.CreateRadiusAttr().Set(10.0)
                xf = UsdGeom.Xformable(sphere.GetPrim())
                xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
                xf.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 0.12))
                xf.AddRotateXYZOp().Set(Gf.Vec3f(0, 0, 0))

            expr = selectin_expr[i] if i < len(selectin_expr) else 0.0
            mat = "endothelial_inflamed" if expr > 0.5 else "endothelial_healthy"
            self._bind_material(path, mat,
                                override_descendants=self._use_mesh_assets)
            self._endo_prims.append(path)
            self._endo_last_mat.append(mat)

    # ── Surface Molecules ────────────────────────────────────────────────

    def _build_selectins(self, frame: Dict):
        """Build selectin lollipop prototypes on endothelium."""
        positions = frame.get("endo_positions", [])
        selectin_expr = frame.get("endo_selectin_expr", [])
        self._selectin_prims = []

        # Prototype: selectin lollipop (5 SCR beads + EGF + lectin head)
        proto_path = f"{PROTOTYPES_PATH}/selectin"
        self._build_selectin_prototype(proto_path)

        idx = 0
        for i, pos in enumerate(positions):
            expr = selectin_expr[i] if i < len(selectin_expr) else 0.0
            if expr < 0.1:
                continue
            n_sel = max(1, int(expr * 3))
            for j in range(n_sel):
                path = f"{SELECTINS_PATH}/sel_{idx:04d}"
                ref = UsdGeom.Xform.Define(self._stage, path)
                # Reference the prototype
                ref.GetPrim().GetReferences().AddInternalReference(proto_path)

                offset = (j - n_sel / 2) * 2
                xf = UsdGeom.Xformable(ref.GetPrim())
                xf.AddTranslateOp().Set(Gf.Vec3d(pos[0] + offset, pos[1], pos[2]))
                xf.AddScaleOp().Set(Gf.Vec3f(1.2, 1.2, 1.2))
                self._selectin_prims.append(path)
                idx += 1

    def _build_selectin_prototype(self, path: str):
        """Build a selectin lollipop geometry: SCR stalk + EGF + lectin head."""
        UsdGeom.Xform.Define(self._stage, path)

        if self._use_mesh_assets:
            # Put mesh reference on a child Xform so the prototype's own
            # xformOpOrder stays clean for AddInternalReference instances
            child = UsdGeom.Xform.Define(self._stage, f"{path}/geo")
            child.GetPrim().GetReferences().AddReference(
                self._asset_paths["selectin"])
            xf = UsdGeom.Xformable(child.GetPrim())
            xf.AddScaleOp().Set(Gf.Vec3f(3.0, 3.0, 3.0))
            self._bind_material(path, "selectin_stalk",
                                override_descendants=True)
            self._bind_material(f"{path}/geo/mesh/lectin_head", "selectin_head",
                                override_descendants=True)
        else:
            # Fallback: primitive spheres
            for k in range(5):
                bead = UsdGeom.Sphere.Define(self._stage, f"{path}/scr_{k}")
                bead.CreateRadiusAttr().Set(0.25)
                xf = UsdGeom.Xformable(bead.GetPrim())
                xf.AddTranslateOp().Set(Gf.Vec3d(0, k * 0.45 + 0.2, 0))
                self._bind_material(bead.GetPath().pathString, "selectin_stalk")

            egf = UsdGeom.Sphere.Define(self._stage, f"{path}/egf")
            egf.CreateRadiusAttr().Set(0.3)
            xf = UsdGeom.Xformable(egf.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(0, 2.5, 0))
            self._bind_material(egf.GetPath().pathString, "selectin_stalk")

            head = UsdGeom.Sphere.Define(self._stage, f"{path}/lectin_head")
            head.CreateRadiusAttr().Set(0.55)
            xf = UsdGeom.Xformable(head.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(0, 3.1, 0))
            xf.AddScaleOp().Set(Gf.Vec3f(1.0, 0.7, 1.0))
            self._bind_material(head.GetPath().pathString, "selectin_head")

    def _build_icam1(self, frame: Dict):
        """Build ICAM-1 bead-rods on endothelium."""
        positions = frame.get("endo_positions", [])
        selectin_expr = frame.get("endo_selectin_expr", [])

        # ICAM-1 prototype (5 Ig domains with kink)
        proto = f"{PROTOTYPES_PATH}/icam1"
        UsdGeom.Xform.Define(self._stage, proto)
        if self._use_mesh_assets:
            # Child xform for mesh reference (keeps prototype xformOps clean)
            child = UsdGeom.Xform.Define(self._stage, f"{proto}/geo")
            child.GetPrim().GetReferences().AddReference(
                self._asset_paths["icam1"])
            xf = UsdGeom.Xformable(child.GetPrim())
            xf.AddScaleOp().Set(Gf.Vec3f(3.0, 3.0, 3.0))
            self._bind_material(proto, "icam1", override_descendants=True)
            self._bind_material(f"{proto}/geo/mesh/tip", "icam1_tip",
                                override_descendants=True)
        else:
            for k in range(5):
                is_d1 = (k == 4)
                domain = UsdGeom.Sphere.Define(self._stage, f"{proto}/d{k}")
                domain.CreateRadiusAttr().Set(0.22)
                y = k * 0.5
                x = 0.0
                if k >= 3:
                    kink_angle = 0.4
                    x = (k - 2) * 0.5 * math.sin(kink_angle)
                    y = 1.0 + (k - 2) * 0.5 * math.cos(kink_angle)
                xf = UsdGeom.Xformable(domain.GetPrim())
                xf.AddTranslateOp().Set(Gf.Vec3d(x, y, 0))
                xf.AddScaleOp().Set(Gf.Vec3f(1.0, 1.4, 1.0))
                self._bind_material(domain.GetPath().pathString,
                                    "icam1_tip" if is_d1 else "icam1")

        # Place on inflamed endothelium
        idx = 0
        for i, pos in enumerate(positions):
            expr = selectin_expr[i] if i < len(selectin_expr) else 0.0
            if expr < 0.2:
                continue
            path = f"{ICAM1_PATH}/icam_{idx:04d}"
            ref = UsdGeom.Xform.Define(self._stage, path)
            ref.GetPrim().GetReferences().AddInternalReference(proto)
            xf = UsdGeom.Xformable(ref.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(pos[0] + 3, pos[1], pos[2]))
            idx += 1

    def _build_pecam1(self, frame: Dict):
        """Build PECAM-1 dimer pairs at endothelial junctions.

        With mesh assets: uses a prototype referencing pecam1.usd.
        Fallback: inline sphere beads.
        """
        positions = frame.get("endo_positions", [])
        self._pecam_prims = []

        # Build prototype
        proto_path = f"{PROTOTYPES_PATH}/pecam1"
        self._build_pecam1_prototype(proto_path)

        idx = 0
        for i in range(len(positions) - 1):
            p1, p2 = positions[i], positions[i + 1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dz = p2[2] - p1[2]
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist > 20:
                continue

            path = f"{PECAM1_PATH}/pecam_{idx:03d}"
            ref = UsdGeom.Xform.Define(self._stage, path)
            ref.GetPrim().GetReferences().AddInternalReference(proto_path)

            mx = (p1[0] + p2[0]) / 2
            my = (p1[1] + p2[1]) / 2
            mz = (p1[2] + p2[2]) / 2
            xf = UsdGeom.Xformable(ref.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(mx, my, mz))
            xf.AddScaleOp().Set(Gf.Vec3f(0.8, 0.8, 0.8))
            self._pecam_prims.append((path, i))
            idx += 1

    def _build_pecam1_prototype(self, path: str):
        """Build PECAM-1 dimer prototype.

        With mesh assets: references pecam1.usd (antiparallel beaded rods).
        Fallback: two rows of 6 sphere beads.
        """
        UsdGeom.Xform.Define(self._stage, path)

        if self._use_mesh_assets:
            # Child xform for mesh reference (keeps prototype xformOps clean)
            child = UsdGeom.Xform.Define(self._stage, f"{path}/geo")
            child.GetPrim().GetReferences().AddReference(
                self._asset_paths["pecam1"])
            xf = UsdGeom.Xformable(child.GetPrim())
            xf.AddScaleOp().Set(Gf.Vec3f(3.0, 3.0, 3.0))
            self._bind_material(path, "pecam1", override_descendants=True)
        else:
            for side in (-1, 1):
                for k in range(6):
                    bead = UsdGeom.Sphere.Define(
                        self._stage, f"{path}/s{'+' if side > 0 else '-'}_d{k}")
                    bead.CreateRadiusAttr().Set(0.18)
                    xf = UsdGeom.Xformable(bead.GetPrim())
                    xf.AddTranslateOp().Set(Gf.Vec3d(0, side * (0.5 + k * 0.4), 0))
                    xf.AddScaleOp().Set(Gf.Vec3f(1.0, 1.3, 1.0))
                    self._bind_material(bead.GetPath().pathString, "pecam1")

    # ── Leukocytes ───────────────────────────────────────────────────────

    def _build_leukocytes(self, frame: Dict):
        """Build neutrophil leukocytes as individual prims.

        Caches translate and scale xform op attribute handles so that
        apply_frame() can .Set() values directly without ClearXformOpOrder.
        """
        positions = frame.get("leukocyte_positions", [])
        radii = frame.get("leukocyte_radii", [])
        states = frame.get("leukocyte_states", [])
        self._n_leuko = len(positions)
        self._leuko_prims = []
        self._leuko_translate_attrs = []
        self._leuko_scale_attrs = []
        self._leuko_last_state = []

        for i in range(self._n_leuko):
            path = f"{LEUKOCYTES_PATH}/neutrophil_{i:03d}"
            rad = radii[i] if i < len(radii) else 6.0
            self._build_neutrophil(path, rad)

            pos = positions[i]
            xf = UsdGeom.Xformable(self._stage.GetPrimAtPath(path))
            translate_op = xf.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(*pos))
            scale_op = xf.AddScaleOp()
            scale_op.Set(Gf.Vec3f(1.0, 1.0, 1.0))

            # Cache the underlying USD attributes for fast per-frame .Set()
            self._leuko_translate_attrs.append(
                self._stage.GetPrimAtPath(path).GetAttribute("xformOp:translate"))
            self._leuko_scale_attrs.append(
                self._stage.GetPrimAtPath(path).GetAttribute("xformOp:scale"))
            self._leuko_prims.append(path)
            state = states[i] if i < len(states) else 0
            self._leuko_last_state.append(state)

    def _build_neutrophil(self, path: str, radius: float):
        """Build a neutrophil: body + multi-lobed nucleus.

        Uses high-fidelity referenced meshes when available, with organic
        surface texture and microvilli. Falls back to primitive spheres.
        """
        xform = UsdGeom.Xform.Define(self._stage, path)

        if self._use_mesh_assets:
            # Body — reference external high-fidelity mesh
            body_xform = UsdGeom.Xform.Define(self._stage, f"{path}/body")
            body_xform.GetPrim().GetReferences().AddReference(
                self._asset_paths["neutrophil_body"])
            xf = UsdGeom.Xformable(body_xform.GetPrim())
            xf.AddScaleOp().Set(Gf.Vec3f(radius, radius, radius))
            self._bind_material(f"{path}/body", "leuko_flowing",
                                override_descendants=True)

            # Nucleus — single multi-lobe mesh replaces 4 separate spheres
            nuc_xform = UsdGeom.Xform.Define(self._stage, f"{path}/nucleus")
            nuc_xform.GetPrim().GetReferences().AddReference(
                self._asset_paths["neutrophil_nucleus"])
            xf = UsdGeom.Xformable(nuc_xform.GetPrim())
            nuc_scale = radius * 0.8
            xf.AddScaleOp().Set(Gf.Vec3f(nuc_scale, nuc_scale, nuc_scale))
            self._bind_material(f"{path}/nucleus", "nucleus",
                                override_descendants=True)
        else:
            # Fallback: primitive spheres
            body = UsdGeom.Sphere.Define(self._stage, f"{path}/body")
            body.CreateRadiusAttr().Set(radius)
            self._bind_material(body.GetPath().pathString, "leuko_flowing")

            lobe_positions = [
                (0.15, 0.1, 0), (-0.15, -0.05, 0.1),
                (0, -0.1, -0.15), (0.1, 0.15, -0.05),
            ]
            for k, lp in enumerate(lobe_positions):
                lobe = UsdGeom.Sphere.Define(
                    self._stage, f"{path}/nucleus_{k}")
                lobe.CreateRadiusAttr().Set(radius * 0.35)
                xf = UsdGeom.Xformable(lobe.GetPrim())
                xf.AddTranslateOp().Set(Gf.Vec3d(
                    lp[0] * radius, lp[1] * radius, lp[2] * radius))
                xf.AddScaleOp().Set(Gf.Vec3f(1.0, 0.9, 0.8))
                self._bind_material(lobe.GetPath().pathString, "nucleus")

    # ── RBCs ─────────────────────────────────────────────────────────────

    def _build_rbcs(self, frame: Dict):
        """Build RBCs using PointInstancer for efficiency.

        Uses high-fidelity Evans-Fung biconcave mesh when available.
        """
        positions = frame.get("rbc_positions", [])
        self._n_rbc = len(positions)

        if self._n_rbc == 0:
            return

        proto_path = f"{PROTOTYPES_PATH}/rbc"
        if self._use_mesh_assets:
            # High-fidelity biconcave mesh prototype
            proto = UsdGeom.Xform.Define(self._stage, proto_path)
            proto.GetPrim().GetReferences().AddReference(
                self._asset_paths["rbc"])
            xf = UsdGeom.Xformable(proto.GetPrim())
            xf.AddScaleOp().Set(Gf.Vec3f(3.75, 3.75, 3.75))
            self._bind_material(proto_path, "rbc",
                                override_descendants=True)
        else:
            # Fallback: flattened sphere
            rbc = UsdGeom.Sphere.Define(self._stage, proto_path)
            rbc.CreateRadiusAttr().Set(3.75)
            xf = UsdGeom.Xformable(rbc.GetPrim())
            xf.AddScaleOp().Set(Gf.Vec3f(1.0, 0.35, 1.0))
            self._bind_material(proto_path, "rbc")

        # PointInstancer
        instancer_path = f"{RBCS_PATH}/Instancer"
        instancer = UsdGeom.PointInstancer.Define(self._stage, instancer_path)
        self._rbc_instancer = instancer_path

        # Set prototype
        instancer.CreatePrototypesRel().SetTargets([proto_path])

        # Set positions
        pos_array = Vt.Vec3fArray(len(positions))
        idx_array = Vt.IntArray(len(positions))
        for i, p in enumerate(positions):
            pos_array[i] = Gf.Vec3f(*p)
            idx_array[i] = 0  # All reference prototype 0

        instancer.CreatePositionsAttr().Set(pos_array)
        instancer.CreateProtoIndicesAttr().Set(idx_array)

    # ── Bacteria ─────────────────────────────────────────────────────────

    def _build_bacteria(self, frame: Dict):
        """Build complement-opsonized bacteria at infection site."""
        positions = frame.get("bacteria_positions", [])
        self._n_bacteria = len(positions)
        self._bacteria_prims = []
        self._bacteria_scale_attrs = []

        for i, pos in enumerate(positions):
            path = f"{BACTERIA_PATH}/bact_{i:02d}"
            self._build_single_bacterium(path)

            xf = UsdGeom.Xformable(self._stage.GetPrimAtPath(path))
            xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
            xf.AddScaleOp().Set(Gf.Vec3f(1.5, 1.5, 1.5))
            self._bacteria_prims.append(path)
            self._bacteria_scale_attrs.append(
                self._stage.GetPrimAtPath(path).GetAttribute("xformOp:scale"))

    def _build_single_bacterium(self, path: str):
        """Build a rod-shaped bacterium with complement coating."""
        xform = UsdGeom.Xform.Define(self._stage, path)

        if self._use_mesh_assets:
            # High-fidelity E. coli with flagella and C3b coating baked in
            xform.GetPrim().GetReferences().AddReference(
                self._asset_paths["ecoli"])
            xf = UsdGeom.Xformable(xform.GetPrim())
            xf.AddScaleOp().Set(Gf.Vec3f(3.0, 3.0, 3.0))
            self._bind_material(path, "bacterium_body",
                                override_descendants=True)
            self._bind_material(f"{path}/mesh/complement", "complement_c3b",
                                override_descendants=True)
        else:
            # Fallback: capsule + scatter spheres
            body = UsdGeom.Capsule.Define(self._stage, f"{path}/body")
            body.CreateRadiusAttr().Set(0.8)
            body.CreateHeightAttr().Set(3.0)
            body.CreateAxisAttr().Set("Y")
            self._bind_material(body.GetPath().pathString, "bacterium_body")

            import random
            rng = random.Random(hash(path))
            for k in range(20):
                c3b = UsdGeom.Sphere.Define(
                    self._stage, f"{path}/c3b_{k:02d}")
                c3b.CreateRadiusAttr().Set(0.15)
                theta = rng.uniform(0, 2 * math.pi)
                y = rng.uniform(-1.5, 1.5)
                cr = 0.85 + rng.uniform(0, 0.1)
                xf = UsdGeom.Xformable(c3b.GetPrim())
                xf.AddTranslateOp().Set(Gf.Vec3d(
                    cr * math.cos(theta), y, cr * math.sin(theta)))
                self._bind_material(c3b.GetPath().pathString, "complement_c3b")

    # ── Macrophages ──────────────────────────────────────────────────────

    def _build_macrophages(self, frame: Dict):
        """Build resident macrophages near bacteria."""
        bacteria_pos = frame.get("bacteria_positions", [])
        if len(bacteria_pos) < 2:
            return

        for i in range(2):
            path = f"{MACROPHAGES_PATH}/mac_{i}"
            bp = bacteria_pos[i]
            offset_x = -5 if i == 0 else 5
            offset_z = 3 if i == 0 else -3

            if self._use_mesh_assets:
                # Amoeboid mesh with pseudopods
                mac_xform = UsdGeom.Xform.Define(self._stage, path)
                mac_xform.GetPrim().GetReferences().AddReference(
                    self._asset_paths["macrophage"])
                xf = UsdGeom.Xformable(mac_xform.GetPrim())
                xf.AddTranslateOp().Set(Gf.Vec3d(
                    bp[0] + offset_x, bp[1] - 3, bp[2] + offset_z))
                xf.AddScaleOp().Set(Gf.Vec3f(10.0, 7.0, 10.0))
                self._bind_material(path, "macrophage",
                                    override_descendants=True)
            else:
                body = UsdGeom.Sphere.Define(self._stage, path)
                body.CreateRadiusAttr().Set(10.0)
                self._bind_material(path, "macrophage")
                xf = UsdGeom.Xformable(body.GetPrim())
                xf.AddTranslateOp().Set(Gf.Vec3d(
                    bp[0] + offset_x, bp[1] - 3, bp[2] + offset_z))
                xf.AddScaleOp().Set(Gf.Vec3f(1.0, 0.7, 1.0))

    # ── Static Tissue Elements ───────────────────────────────────────────

    def _build_fibrin(self):
        """Build fibrin mesh fibers."""
        R = self._vessel_radius
        L = self._vessel_length
        import random
        rng = random.Random(42)

        for i in range(40):
            path = f"{FIBRIN_PATH}/fiber_{i:03d}"
            flen = 5 + rng.uniform(0, 12)

            if self._use_mesh_assets:
                fib_xform = UsdGeom.Xform.Define(self._stage, path)
                fib_xform.GetPrim().GetReferences().AddReference(
                    self._asset_paths["fibrin_fiber"])
                self._bind_material(path, "fibrin",
                                    override_descendants=True)
                xf = UsdGeom.Xformable(fib_xform.GetPrim())
                xf.AddTranslateOp().Set(Gf.Vec3d(
                    L * 0.2 + rng.uniform(0, L * 0.6),
                    -R * 1.5 - rng.uniform(0, R * 1.5),
                    (rng.uniform(0, 1) - 0.5) * R,
                ))
                xf.AddRotateXYZOp().Set(Gf.Vec3f(
                    rng.uniform(0, 180), rng.uniform(0, 180),
                    rng.uniform(0, 180)))
                xf.AddScaleOp().Set(Gf.Vec3f(1.0, flen, 1.0))
            else:
                fib = UsdGeom.Cylinder.Define(self._stage, path)
                fib.CreateRadiusAttr().Set(0.12)
                fib.CreateHeightAttr().Set(flen)
                fib.CreateAxisAttr().Set("Y")
                self._bind_material(path, "fibrin")
                xf = UsdGeom.Xformable(fib.GetPrim())
                xf.AddTranslateOp().Set(Gf.Vec3d(
                    L * 0.2 + rng.uniform(0, L * 0.6),
                    -R * 1.5 - rng.uniform(0, R * 1.5),
                    (rng.uniform(0, 1) - 0.5) * R,
                ))
                xf.AddRotateXYZOp().Set(Gf.Vec3f(
                    rng.uniform(0, 180), rng.uniform(0, 180),
                    rng.uniform(0, 180)))

    def _build_collagen(self):
        """Build ECM collagen fiber bundles."""
        R = self._vessel_radius
        L = self._vessel_length
        import random
        rng = random.Random(99)

        for i in range(20):
            path = f"{COLLAGEN_PATH}/col_{i:03d}"
            clen = 10 + rng.uniform(0, 15)

            if self._use_mesh_assets:
                col_xform = UsdGeom.Xform.Define(self._stage, path)
                col_xform.GetPrim().GetReferences().AddReference(
                    self._asset_paths["collagen_helix"])
                self._bind_material(path, "collagen",
                                    override_descendants=True)
                xf = UsdGeom.Xformable(col_xform.GetPrim())
                xf.AddTranslateOp().Set(Gf.Vec3d(
                    L * 0.1 + rng.uniform(0, L * 0.8),
                    -R * 1.3 - rng.uniform(0, R * 1.5),
                    (rng.uniform(0, 1) - 0.5) * R * 1.2,
                ))
                xf.AddRotateXYZOp().Set(Gf.Vec3f(
                    rng.uniform(-20, 20), rng.uniform(0, 180),
                    rng.uniform(-10, 10)))
                xf.AddScaleOp().Set(Gf.Vec3f(clen, clen, clen))
            else:
                col = UsdGeom.Cylinder.Define(self._stage, path)
                col.CreateRadiusAttr().Set(0.25 + rng.uniform(0, 0.15))
                col.CreateHeightAttr().Set(clen)
                col.CreateAxisAttr().Set("X")
                self._bind_material(path, "collagen")
                xf = UsdGeom.Xformable(col.GetPrim())
                xf.AddTranslateOp().Set(Gf.Vec3d(
                    L * 0.1 + rng.uniform(0, L * 0.8),
                    -R * 1.3 - rng.uniform(0, R * 1.5),
                    (rng.uniform(0, 1) - 0.5) * R * 1.2,
                ))
                xf.AddRotateXYZOp().Set(Gf.Vec3f(
                    rng.uniform(-20, 20), rng.uniform(0, 180),
                    rng.uniform(-10, 10)))

    def _build_chemokine_cloud(self):
        """Build chemokine gradient as scattered translucent spheres."""
        R = self._vessel_radius
        L = self._vessel_length
        import random
        rng = random.Random(77)

        for i in range(150):
            path = f"{CHEMOKINE_PATH}/ck_{i:03d}"
            t = rng.uniform(0, 1) ** 0.7  # Bias toward infection
            sphere = UsdGeom.Sphere.Define(self._stage, path)
            s = 0.3 + rng.uniform(0, 0.5)
            sphere.CreateRadiusAttr().Set(s)
            self._bind_material(path, "chemokine")

            xf = UsdGeom.Xformable(sphere.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(
                L * 0.25 + rng.uniform(0, L * 0.5),
                -R * 0.5 - t * R * 2.2,
                (rng.uniform(0, 1) - 0.5) * R * 0.8,
            ))

    def _build_infection_glow(self, frame: Dict):
        """Build infection site glow sphere and point light."""
        R = self._vessel_radius
        L = self._vessel_length

        # Glow sphere
        path = f"{TISSUE_PATH}/infection_glow"
        sphere = UsdGeom.Sphere.Define(self._stage, path)
        sphere.CreateRadiusAttr().Set(R * 0.8)
        xf = UsdGeom.Xformable(sphere.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(L * 0.5, -R * 1.75, 0))
        self._bind_material(path, "infection_glow")

        # Point light at infection
        light = UsdLux.SphereLight.Define(self._stage, f"{LIGHTS_PATH}/InfectionLight")
        light.CreateIntensityAttr().Set(500.0)
        light.CreateColorAttr().Set(Gf.Vec3f(1.0, 0.27, 0.13))
        light.CreateRadiusAttr().Set(R * 0.3)
        xf = UsdGeom.Xformable(light.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(L * 0.5, -R * 1.75, 0))

    # ── Frame Update ─────────────────────────────────────────────────────

    def apply_frame(self, frame: Dict[str, Any]) -> None:
        """Update scene from a simulation frame snapshot.

        Updates positions, colors/materials, visibility, and deformation
        for all dynamic elements. Uses Sdf.ChangeBlock to batch all USD
        modifications into a single notification, preventing per-change
        viewport update cascades that block the main thread in headless mode.
        """
        with Sdf.ChangeBlock():
            self._update_leukocytes(frame)
            self._update_rbcs(frame)
            self._update_endothelium(frame)
            self._update_pecam_opacity(frame)
            self._update_bacteria(frame)

    def _update_leukocytes(self, frame: Dict):
        """Update leukocyte positions, materials, and deformation.

        Uses cached xform attribute handles for fast .Set() without
        ClearXformOpOrder schema churn. Only rebinds material when
        state actually changes.
        """
        positions = frame.get("leukocyte_positions", [])
        states = frame.get("leukocyte_states", [])
        trans_prog = frame.get("transmigration_progress", [])

        for i, path in enumerate(self._leuko_prims):
            # Position — set directly on cached attribute (no schema change)
            if i < len(positions) and i < len(self._leuko_translate_attrs):
                attr = self._leuko_translate_attrs[i]
                if attr:
                    attr.Set(Gf.Vec3d(*positions[i]))

            # Scale — set directly on cached attribute
            if i < len(self._leuko_scale_attrs):
                state = states[i] if i < len(states) else 0
                prog = trans_prog[i] if i < len(trans_prog) else 0.0
                attr = self._leuko_scale_attrs[i]
                if attr:
                    if state == 6:  # TRANSMIGRATING
                        attr.Set(Gf.Vec3f(
                            1.0 - prog * 0.3, 1.0 + prog * 0.5, 1.0 - prog * 0.3))
                    elif 2 <= state <= 5:  # ROLLING-CRAWLING
                        attr.Set(Gf.Vec3f(1.1, 0.9, 1.1))
                    else:
                        attr.Set(Gf.Vec3f(1.0, 1.0, 1.0))

            # Material — only rebind if state changed
            if i < len(states):
                state = states[i]
                if i < len(self._leuko_last_state) and self._leuko_last_state[i] != state:
                    mat_name = LEUKO_STATE_MATERIALS.get(state, "leuko_flowing")
                    body_path = f"{path}/body"
                    self._bind_material(body_path, mat_name,
                                        override_descendants=self._use_mesh_assets)
                    self._leuko_last_state[i] = state

    def _update_rbcs(self, frame: Dict):
        """Update RBC positions via PointInstancer."""
        if not self._rbc_instancer:
            return

        positions = frame.get("rbc_positions", [])
        if not positions:
            return

        instancer = UsdGeom.PointInstancer.Get(
            self._stage, self._rbc_instancer)
        if not instancer:
            return

        pos_array = Vt.Vec3fArray(len(positions))
        for i, p in enumerate(positions):
            pos_array[i] = Gf.Vec3f(*p)
        instancer.CreatePositionsAttr().Set(pos_array)

    def _update_endothelium(self, frame: Dict):
        """Update endothelial cell colors based on inflammation.

        Only rebinds material when the threshold crossing changes.
        """
        selectin_expr = frame.get("endo_selectin_expr", [])
        for i, path in enumerate(self._endo_prims):
            expr = selectin_expr[i] if i < len(selectin_expr) else 0.0
            mat = "endothelial_inflamed" if expr > 0.5 else "endothelial_healthy"
            if i < len(self._endo_last_mat) and self._endo_last_mat[i] != mat:
                self._bind_material(path, mat,
                                    override_descendants=self._use_mesh_assets)
                self._endo_last_mat[i] = mat

    def _update_pecam_opacity(self, frame: Dict):
        """Update PECAM-1 visibility based on junction integrity."""
        junction_integrity = frame.get("endo_junction_integrity", [])
        for path, endo_idx in self._pecam_prims:
            integrity = junction_integrity[endo_idx] if endo_idx < len(
                junction_integrity) else 1.0
            # Toggle visibility based on integrity
            prim = self._stage.GetPrimAtPath(path)
            if prim:
                imageable = UsdGeom.Imageable(prim)
                if integrity < 0.3:
                    imageable.MakeInvisible()
                else:
                    imageable.MakeVisible()

    def _update_bacteria(self, frame: Dict):
        """Update bacteria visibility and scale based on phagocytosis.

        Uses cached scale attribute handles for fast updates.
        """
        alive = frame.get("bacteria_alive", [])
        phago = frame.get("bacteria_phagocytosis", [])

        for i, path in enumerate(self._bacteria_prims):
            prim = self._stage.GetPrimAtPath(path)
            if not prim:
                continue

            is_alive = alive[i] if i < len(alive) else True
            progress = phago[i] if i < len(phago) else 0.0

            imageable = UsdGeom.Imageable(prim)
            if not is_alive:
                imageable.MakeInvisible()
            else:
                imageable.MakeVisible()
                if progress > 0.01 and i < len(self._bacteria_scale_attrs):
                    s = 1.5 * (1.0 - progress * 0.8)
                    attr = self._bacteria_scale_attrs[i]
                    if attr:
                        attr.Set(Gf.Vec3f(s, s, s))
