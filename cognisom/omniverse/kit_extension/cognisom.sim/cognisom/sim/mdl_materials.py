"""
OmniPBR MDL Material System
=============================

Creates dual-output materials for the diapedesis scene:

1. **OmniPBR MDL** (primary): Full PBR with subsurface scattering (SSS),
   translucency, and emission for RTX rendering on L40S.

2. **UsdPreviewSurface** (fallback): Standard USD material for non-RTX
   contexts (usdview, Three.js exporter, etc.).

The OmniPBR material is connected via the "mdl" output token;
the UsdPreviewSurface via the default "" output token.
Kit's RTX renderer prefers the MDL output when available.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

log = logging.getLogger(__name__)

try:
    from pxr import Gf, Sdf, UsdShade
    USD_AVAILABLE = True
except ImportError:
    USD_AVAILABLE = False


# ── SSS Parameters per Material ───────────────────────────────────────
# These extend the base MATERIALS dict with OmniPBR-specific SSS properties.
# subsurface_radius is the mean free path (R, G, B) — red light scatters
# furthest in biological tissue (hemoglobin absorption).

MDL_EXTRAS: Dict[str, Dict[str, Any]] = {
    # Blood cells
    "rbc": {
        "subsurface_radius": (2.0, 0.4, 0.2),
        "thin_walled": True,
    },
    "leuko_flowing": {
        "subsurface_radius": (1.5, 0.8, 0.4),
    },
    "leuko_rolling": {
        "subsurface_radius": (1.5, 0.7, 0.3),
    },
    "leuko_arrested": {
        "subsurface_radius": (1.2, 0.5, 0.3),
    },
    "leuko_transmigrating": {
        "subsurface_radius": (1.8, 0.9, 0.5),
    },
    "leuko_migrated": {
        "subsurface_radius": (1.0, 0.6, 0.3),
    },
    "nucleus": {
        "subsurface_radius": (0.8, 0.5, 0.8),
    },
    # Vessel
    "vessel_wall": {
        "subsurface_radius": (1.5, 0.6, 0.4),
    },
    "endothelial_healthy": {
        "subsurface_radius": (1.2, 0.5, 0.4),
    },
    "endothelial_inflamed": {
        "subsurface_radius": (1.5, 0.4, 0.3),
    },
    # Tissue
    "bacterium_body": {
        "subsurface_radius": (0.5, 0.3, 0.2),
    },
    "macrophage": {
        "subsurface_radius": (1.8, 0.6, 0.3),
    },
}


def create_omnipbr_material(
    stage: Any, path: str, spec: Dict[str, Any]
) -> str:
    """Create a dual-output material: OmniPBR MDL + UsdPreviewSurface fallback.

    Args:
        stage: Usd.Stage
        path: Material prim path (e.g. "/World/Diapedesis/Materials/rbc")
        spec: Material specification dict with keys:
            - color: (r, g, b) tuple 0-1
            - roughness: float 0-1
            - opacity: float 0-1
            - metallic: float 0-1 (optional)
            - subsurface: float 0-1 (optional, SSS weight)
            - subsurface_color: (r, g, b) (optional)
            - emissive: (r, g, b) (optional)

    Returns:
        Material prim path.
    """
    if not USD_AVAILABLE:
        raise RuntimeError("pxr (USD) not available")

    material = UsdShade.Material.Define(stage, path)
    color = spec.get("color", (0.5, 0.5, 0.5))
    roughness = spec.get("roughness", 0.7)
    metallic = spec.get("metallic", 0.0)
    opacity = spec.get("opacity", 1.0)
    subsurface = spec.get("subsurface", 0.0)
    subsurface_color = spec.get("subsurface_color", color)
    emissive = spec.get("emissive", (0, 0, 0))

    # Look up material name from path for MDL extras
    mat_name = path.rsplit("/", 1)[-1] if "/" in path else path
    extras = MDL_EXTRAS.get(mat_name, {})

    # ── OmniPBR MDL Shader ────────────────────────────────────────────
    try:
        mdl_shader = UsdShade.Shader.Define(stage, f"{path}/OmniPBR")
        mdl_shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        mdl_shader.SetSourceAsset("OmniPBR.mdl", "mdl")
        mdl_shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")

        # Base color
        mdl_shader.CreateInput(
            "diffuse_color_constant", Sdf.ValueTypeNames.Color3f
        ).Set(Gf.Vec3f(*color))

        # Roughness
        mdl_shader.CreateInput(
            "reflection_roughness_constant", Sdf.ValueTypeNames.Float
        ).Set(roughness)

        # Metallic
        mdl_shader.CreateInput(
            "metallic_constant", Sdf.ValueTypeNames.Float
        ).Set(metallic)

        # Opacity / translucency
        if opacity < 1.0:
            mdl_shader.CreateInput(
                "enable_opacity", Sdf.ValueTypeNames.Bool
            ).Set(True)
            mdl_shader.CreateInput(
                "opacity_constant", Sdf.ValueTypeNames.Float
            ).Set(opacity)

        # Subsurface scattering (the key visual upgrade for biological tissue)
        if subsurface > 0:
            mdl_shader.CreateInput(
                "subsurface_weight", Sdf.ValueTypeNames.Float
            ).Set(subsurface)
            mdl_shader.CreateInput(
                "subsurface_color", Sdf.ValueTypeNames.Color3f
            ).Set(Gf.Vec3f(*subsurface_color))

            # Mean free path from extras or default
            ss_radius = extras.get("subsurface_radius", (1.0, 0.5, 0.25))
            mdl_shader.CreateInput(
                "subsurface_radius", Sdf.ValueTypeNames.Color3f
            ).Set(Gf.Vec3f(*ss_radius))

        # Emissive
        if any(e > 0 for e in emissive):
            mdl_shader.CreateInput(
                "enable_emission", Sdf.ValueTypeNames.Bool
            ).Set(True)
            mdl_shader.CreateInput(
                "emissive_color", Sdf.ValueTypeNames.Color3f
            ).Set(Gf.Vec3f(*emissive))
            mdl_shader.CreateInput(
                "emissive_intensity", Sdf.ValueTypeNames.Float
            ).Set(1.0)

        # Connect MDL surface output
        material.CreateSurfaceOutput("mdl").ConnectToSource(
            mdl_shader.ConnectableAPI(), "out"
        )

    except Exception as e:
        log.warning(f"OmniPBR MDL creation failed for {path}: {e}. "
                    f"Using UsdPreviewSurface only.")

    # ── UsdPreviewSurface Fallback ────────────────────────────────────
    fallback = UsdShade.Shader.Define(stage, f"{path}/PBRShader")
    fallback.CreateIdAttr("UsdPreviewSurface")

    fallback.CreateInput(
        "diffuseColor", Sdf.ValueTypeNames.Color3f
    ).Set(Gf.Vec3f(*color))
    fallback.CreateInput(
        "roughness", Sdf.ValueTypeNames.Float
    ).Set(roughness)
    fallback.CreateInput(
        "metallic", Sdf.ValueTypeNames.Float
    ).Set(metallic)
    fallback.CreateInput(
        "opacity", Sdf.ValueTypeNames.Float
    ).Set(opacity)

    if any(e > 0 for e in emissive):
        fallback.CreateInput(
            "emissiveColor", Sdf.ValueTypeNames.Color3f
        ).Set(Gf.Vec3f(*emissive))

    # Default surface output (non-MDL renderers use this)
    material.CreateSurfaceOutput().ConnectToSource(
        fallback.ConnectableAPI(), "surface"
    )

    return path
