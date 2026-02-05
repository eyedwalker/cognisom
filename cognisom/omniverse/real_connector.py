"""
Real Omniverse/USD Connector
============================

Actual USD stage operations using OpenUSD (pxr module).
No mocks - real USD files and real 3D scene manipulation.

This connector:
- Creates real USD stages (.usda/.usdc files)
- Manipulates actual 3D geometry (spheres, meshes)
- Exports to standard USD format viewable in any USD viewer
- Supports real-time updates for simulation visualization

Requirements:
    pip install usd-core
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from threading import Thread, Event
import math

log = logging.getLogger(__name__)

# Import real USD modules
try:
    from pxr import Usd, UsdGeom, UsdShade, Gf, Sdf, UsdLux
    USD_AVAILABLE = True
    log.info("OpenUSD (pxr) loaded successfully")
except ImportError as e:
    USD_AVAILABLE = False
    log.warning(f"OpenUSD not available: {e}. Install with: pip install usd-core")


class ConnectionStatus(str, Enum):
    """Connection status states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class CellVisualization:
    """Data for visualizing a single cell."""
    cell_id: str
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 1.0
    color: Tuple[float, float, float] = (0.8, 0.2, 0.2)  # RGB 0-1
    cell_type: str = "generic"
    metabolic_state: float = 1.0  # 0-1 scale
    gene_expression: Dict[str, float] = field(default_factory=dict)


@dataclass
class SimulationFrame:
    """Single frame of simulation data."""
    timestamp: float
    cells: List[CellVisualization]
    metadata: Dict[str, Any] = field(default_factory=dict)


class RealOmniverseConnector:
    """
    Real USD/Omniverse connector for 3D cell visualization.

    Creates actual USD files with real 3D geometry that can be:
    - Viewed in NVIDIA Omniverse
    - Opened in Pixar's usdview
    - Loaded into Blender, Maya, or any USD-compatible software
    - Streamed to web viewers
    """

    def __init__(self, output_dir: str = "data/simulation/usd") -> None:
        """Initialize the connector.

        Args:
            output_dir: Directory for USD output files
        """
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._status = ConnectionStatus.DISCONNECTED
        self._stage: Optional[Usd.Stage] = None
        self._stage_path: Optional[Path] = None
        self._frame_count = 0
        self._cells: Dict[str, Any] = {}  # prim references
        self._event_handlers: Dict[str, List[Callable]] = {}

        if not USD_AVAILABLE:
            log.error("USD not available - real connector cannot function")
            self._status = ConnectionStatus.ERROR

    @property
    def status(self) -> ConnectionStatus:
        """Current connection status."""
        return self._status

    @property
    def is_connected(self) -> bool:
        """Whether connected (stage is open)."""
        return self._status == ConnectionStatus.CONNECTED and self._stage is not None

    @property
    def stage_path(self) -> Optional[Path]:
        """Path to the current USD stage file."""
        return self._stage_path

    # ── Stage Management ────────────────────────────────────────────

    def create_stage(self, name: str = "cognisom_simulation") -> bool:
        """Create a new USD stage for simulation visualization.

        Args:
            name: Name for the USD file (without extension)

        Returns:
            True if stage created successfully
        """
        if not USD_AVAILABLE:
            log.error("Cannot create stage: USD not available")
            return False

        self._status = ConnectionStatus.CONNECTING

        try:
            # Create stage file path
            self._stage_path = self._output_dir / f"{name}.usda"

            # Create new USD stage
            self._stage = Usd.Stage.CreateNew(str(self._stage_path))

            # Set up stage metadata
            self._stage.SetMetadata("documentation", "Cognisom Cell Simulation")
            self._stage.SetStartTimeCode(0)
            self._stage.SetEndTimeCode(1000)
            self._stage.SetTimeCodesPerSecond(24)

            # Create scene hierarchy
            self._setup_scene_hierarchy()

            # Add default lighting
            self._setup_lighting()

            # Add camera
            self._setup_camera()

            # Save initial stage
            self._stage.Save()

            self._status = ConnectionStatus.CONNECTED
            self._emit_event("connected", f"Stage created: {self._stage_path}")
            log.info(f"Created USD stage: {self._stage_path}")

            return True

        except Exception as e:
            self._status = ConnectionStatus.ERROR
            log.error(f"Failed to create stage: {e}")
            return False

    def open_stage(self, path: str) -> bool:
        """Open an existing USD stage.

        Args:
            path: Path to USD file

        Returns:
            True if opened successfully
        """
        if not USD_AVAILABLE:
            return False

        try:
            self._stage = Usd.Stage.Open(path)
            self._stage_path = Path(path)
            self._status = ConnectionStatus.CONNECTED
            log.info(f"Opened USD stage: {path}")
            return True
        except Exception as e:
            log.error(f"Failed to open stage: {e}")
            return False

    def _setup_scene_hierarchy(self) -> None:
        """Create the standard scene hierarchy."""
        # Root transform
        UsdGeom.Xform.Define(self._stage, "/World")

        # Cell container
        UsdGeom.Xform.Define(self._stage, "/World/Cells")

        # Environment (substrate, boundaries)
        UsdGeom.Xform.Define(self._stage, "/World/Environment")

        # Create petri dish / substrate
        self._create_substrate()

        # Lights container
        UsdGeom.Xform.Define(self._stage, "/World/Lights")

        # Camera container
        UsdGeom.Xform.Define(self._stage, "/World/Cameras")

    def _create_substrate(self) -> None:
        """Create the simulation substrate (petri dish visualization)."""
        substrate = UsdGeom.Cylinder.Define(self._stage, "/World/Environment/Substrate")
        substrate.GetRadiusAttr().Set(50.0)
        substrate.GetHeightAttr().Set(2.0)
        substrate.GetAxisAttr().Set("Y")

        # Position it below the cells
        xform = UsdGeom.Xformable(substrate)
        xform.AddTranslateOp().Set(Gf.Vec3d(0, -1, 0))

        # Add material
        self._create_material(
            "/World/Environment/SubstrateMaterial",
            color=(0.9, 0.9, 0.95),
            opacity=0.3
        )

    def _setup_lighting(self) -> None:
        """Set up scene lighting."""
        # Dome light for ambient
        dome = UsdLux.DomeLight.Define(self._stage, "/World/Lights/DomeLight")
        dome.GetIntensityAttr().Set(500.0)
        dome.GetColorAttr().Set(Gf.Vec3f(1.0, 0.98, 0.95))

        # Key light
        key_light = UsdLux.DistantLight.Define(self._stage, "/World/Lights/KeyLight")
        key_light.GetIntensityAttr().Set(1000.0)
        key_light.GetAngleAttr().Set(1.0)
        xform = UsdGeom.Xformable(key_light)
        xform.AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))

    def _setup_camera(self) -> None:
        """Set up the default camera."""
        camera = UsdGeom.Camera.Define(self._stage, "/World/Cameras/MainCamera")
        camera.GetFocalLengthAttr().Set(50.0)

        xform = UsdGeom.Xformable(camera)
        xform.AddTranslateOp().Set(Gf.Vec3d(0, 50, 100))
        xform.AddRotateXYZOp().Set(Gf.Vec3f(-25, 0, 0))

    # ── Cell Operations ─────────────────────────────────────────────

    def add_cell(self, cell: CellVisualization, time_code: float = 0) -> bool:
        """Add a cell to the simulation.

        Args:
            cell: Cell visualization data
            time_code: USD time code for animation

        Returns:
            True if cell added successfully
        """
        if not self.is_connected:
            return False

        try:
            cell_path = f"/World/Cells/Cell_{cell.cell_id}"

            # Create sphere for cell body
            sphere = UsdGeom.Sphere.Define(self._stage, cell_path)
            sphere.GetRadiusAttr().Set(cell.radius, time_code)

            # Set position
            xform = UsdGeom.Xformable(sphere)
            xform.AddTranslateOp().Set(
                Gf.Vec3d(*cell.position),
                time_code
            )

            # Create and bind material
            mat_path = f"/World/Cells/Cell_{cell.cell_id}_Material"
            self._create_cell_material(mat_path, cell)

            # Bind material to sphere
            UsdShade.MaterialBindingAPI(sphere).Bind(
                UsdShade.Material.Get(self._stage, mat_path)
            )

            # Store reference
            self._cells[cell.cell_id] = {
                "prim": sphere,
                "xform": xform,
                "material_path": mat_path,
                "cell_data": cell
            }

            return True

        except Exception as e:
            log.error(f"Failed to add cell {cell.cell_id}: {e}")
            return False

    def update_cell(self, cell: CellVisualization, time_code: float = 0) -> bool:
        """Update an existing cell's properties.

        Args:
            cell: Updated cell data
            time_code: USD time code for animation keyframe

        Returns:
            True if updated successfully
        """
        if cell.cell_id not in self._cells:
            return self.add_cell(cell, time_code)

        try:
            cell_ref = self._cells[cell.cell_id]
            sphere = cell_ref["prim"]
            xform = cell_ref["xform"]

            # Update radius (animated)
            sphere.GetRadiusAttr().Set(cell.radius, time_code)

            # Update position (animated)
            translate_op = xform.GetOrderedXformOps()[0]  # First op is translate
            translate_op.Set(Gf.Vec3d(*cell.position), time_code)

            # Update material color based on metabolic state
            self._update_cell_material(cell_ref["material_path"], cell)

            cell_ref["cell_data"] = cell
            return True

        except Exception as e:
            log.error(f"Failed to update cell {cell.cell_id}: {e}")
            return False

    def remove_cell(self, cell_id: str) -> bool:
        """Remove a cell from the simulation."""
        if cell_id not in self._cells:
            return False

        try:
            cell_ref = self._cells[cell_id]
            self._stage.RemovePrim(cell_ref["prim"].GetPath())
            self._stage.RemovePrim(cell_ref["material_path"])
            del self._cells[cell_id]
            return True
        except Exception as e:
            log.error(f"Failed to remove cell {cell_id}: {e}")
            return False

    def _create_cell_material(self, path: str, cell: CellVisualization) -> None:
        """Create a material for a cell based on its properties."""
        # Color based on cell type and metabolic state
        base_color = cell.color

        # Modulate by metabolic state (brighter = more active)
        intensity = 0.5 + (cell.metabolic_state * 0.5)
        color = tuple(c * intensity for c in base_color)

        self._create_material(path, color, opacity=0.9, metallic=0.1, roughness=0.6)

    def _update_cell_material(self, path: str, cell: CellVisualization) -> None:
        """Update cell material based on current state."""
        material = UsdShade.Material.Get(self._stage, path)
        if not material:
            return

        # Find the shader
        shader_path = f"{path}/Shader"
        shader = UsdShade.Shader.Get(self._stage, shader_path)
        if shader:
            # Update color based on metabolic state
            intensity = 0.5 + (cell.metabolic_state * 0.5)
            color = tuple(c * intensity for c in cell.color)
            shader.GetInput("diffuseColor").Set(Gf.Vec3f(*color))

    def _create_material(
        self,
        path: str,
        color: Tuple[float, float, float],
        opacity: float = 1.0,
        metallic: float = 0.0,
        roughness: float = 0.5
    ) -> None:
        """Create a USD Preview Surface material."""
        material = UsdShade.Material.Define(self._stage, path)

        # Create shader
        shader = UsdShade.Shader.Define(self._stage, f"{path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")

        # Set shader inputs
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)

        # Connect shader to material surface output
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    # ── Simulation Frame Operations ─────────────────────────────────

    def render_frame(self, frame: SimulationFrame) -> bool:
        """Render a complete simulation frame.

        Args:
            frame: Simulation frame data with all cells

        Returns:
            True if frame rendered successfully
        """
        if not self.is_connected:
            return False

        time_code = self._frame_count

        # Track which cells are in this frame
        frame_cell_ids = {c.cell_id for c in frame.cells}

        # Update or add cells
        for cell in frame.cells:
            if cell.cell_id in self._cells:
                self.update_cell(cell, time_code)
            else:
                self.add_cell(cell, time_code)

        # Remove cells not in this frame (cell death)
        dead_cells = set(self._cells.keys()) - frame_cell_ids
        for cell_id in dead_cells:
            self.remove_cell(cell_id)

        # Update end time
        self._stage.SetEndTimeCode(time_code)
        self._frame_count += 1

        return True

    def save(self) -> bool:
        """Save the current stage."""
        if self._stage:
            try:
                self._stage.Save()
                log.info(f"Saved USD stage: {self._stage_path}")
                return True
            except Exception as e:
                log.error(f"Failed to save stage: {e}")
        return False

    def export(self, path: str, format: str = "usda") -> bool:
        """Export the stage to a different format.

        Args:
            path: Output path
            format: Format (usda, usdc, usdz)
        """
        if self._stage:
            try:
                self._stage.Export(path)
                log.info(f"Exported stage to: {path}")
                return True
            except Exception as e:
                log.error(f"Failed to export: {e}")
        return False

    # ── Event System ────────────────────────────────────────────────

    def on(self, event_type: str, handler: Callable) -> None:
        """Register event handler."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    def _emit_event(self, event_type: str, message: str = "") -> None:
        """Emit an event."""
        for handler in self._event_handlers.get(event_type, []):
            try:
                handler({"type": event_type, "message": message})
            except Exception as e:
                log.error(f"Event handler error: {e}")

    # ── Status ──────────────────────────────────────────────────────

    def get_info(self) -> Dict[str, Any]:
        """Get connector information."""
        return {
            "status": self._status.value,
            "stage_path": str(self._stage_path) if self._stage_path else None,
            "usd_available": USD_AVAILABLE,
            "cell_count": len(self._cells),
            "frame_count": self._frame_count,
            "is_real": True,  # This is the REAL connector
            "simulated": False,
        }

    def close(self) -> None:
        """Close the stage and clean up."""
        if self._stage:
            self.save()
            self._stage = None
        self._status = ConnectionStatus.DISCONNECTED
        self._cells.clear()


# ── Helper Functions ────────────────────────────────────────────────

def create_demo_simulation(output_dir: str = "data/simulation/usd") -> RealOmniverseConnector:
    """Create a demo simulation with sample cells.

    Returns:
        Configured connector with demo scene
    """
    connector = RealOmniverseConnector(output_dir)

    if not connector.create_stage("cognisom_demo"):
        raise RuntimeError("Failed to create USD stage")

    # Create sample cells in a cluster
    import random
    random.seed(42)

    cell_types = {
        "stem": (0.2, 0.8, 0.2),      # Green
        "progenitor": (0.2, 0.6, 0.8), # Cyan
        "differentiated": (0.8, 0.4, 0.2), # Orange
        "dividing": (0.8, 0.2, 0.8),   # Magenta
    }

    cells = []
    for i in range(50):
        cell_type = random.choice(list(cell_types.keys()))

        # Position in a sphere cluster
        theta = random.uniform(0, 2 * math.pi)
        phi = random.uniform(0, math.pi)
        r = random.uniform(5, 25)

        x = r * math.sin(phi) * math.cos(theta)
        y = r * math.sin(phi) * math.sin(theta) + 10  # Offset up
        z = r * math.cos(phi)

        cell = CellVisualization(
            cell_id=f"cell_{i:04d}",
            position=(x, y, z),
            radius=random.uniform(0.8, 2.0),
            color=cell_types[cell_type],
            cell_type=cell_type,
            metabolic_state=random.uniform(0.3, 1.0)
        )
        cells.append(cell)
        connector.add_cell(cell)

    connector.save()

    return connector


if __name__ == "__main__":
    # Demo
    logging.basicConfig(level=logging.INFO)

    print("Creating real USD simulation...")
    connector = create_demo_simulation()

    print(f"\nConnector info: {connector.get_info()}")
    print(f"\nUSD file created: {connector.stage_path}")
    print("\nView this file in:")
    print("  - NVIDIA Omniverse")
    print("  - usdview (from OpenUSD)")
    print("  - Blender (with USD import)")
    print("  - Any USD-compatible application")
