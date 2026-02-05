"""
Simulation Manager for Omniverse Kit Extension
===============================================

Manages the biological simulation within Omniverse, handling:
- Cell creation and updates
- Physics integration
- State synchronization with Cognisom backend
"""

import asyncio
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import carb
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf


class SimulationManager:
    """Manages biological simulation in Omniverse scene."""

    # Scene hierarchy
    ROOT_PATH = "/World/CognisomSimulation"
    CELLS_PATH = "/World/CognisomSimulation/Cells"
    ENVIRONMENT_PATH = "/World/CognisomSimulation/Environment"

    def __init__(self):
        self._stage = None
        self._is_running = False
        self._is_paused = False
        self._cells: Dict[str, Dict] = {}
        self._time = 0.0
        self._step_count = 0

        # Simulation parameters
        self._params = {
            "cell_count": 100,
            "division_rate": 0.1,
            "death_rate": 0.05,
            "migration_speed": 2.0,
            "interaction_radius": 20.0,
        }

        # Statistics
        self._stats = {
            "total_cells": 0,
            "dividing": 0,
            "apoptotic": 0,
            "fps": 0.0,
            "step_time_ms": 0.0,
        }

        # Materials
        self._materials = {}

        # Initialize stage
        self._init_stage()

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    @property
    def params(self) -> Dict[str, Any]:
        return self._params

    @property
    def stats(self) -> Dict[str, Any]:
        return self._stats

    def _init_stage(self):
        """Initialize USD stage structure."""
        context = omni.usd.get_context()
        self._stage = context.get_stage()

        if not self._stage:
            carb.log_warn("[cognisom.sim] No USD stage available")
            return

        # Create hierarchy
        for path in [self.ROOT_PATH, self.CELLS_PATH, self.ENVIRONMENT_PATH]:
            if not self._stage.GetPrimAtPath(path):
                UsdGeom.Xform.Define(self._stage, path)

        # Create materials
        self._create_materials()

        carb.log_info("[cognisom.sim] Stage initialized")

    def _create_materials(self):
        """Create cell materials."""
        if not self._stage:
            return

        mat_path = "/World/CognisomSimulation/Materials"
        if not self._stage.GetPrimAtPath(mat_path):
            UsdGeom.Xform.Define(self._stage, mat_path)

        # Normal cell material (pink/tan)
        self._materials["normal"] = self._create_material(
            f"{mat_path}/NormalCell",
            color=(0.9, 0.7, 0.6),
            roughness=0.8
        )

        # Tumor cell material (red)
        self._materials["tumor"] = self._create_material(
            f"{mat_path}/TumorCell",
            color=(0.8, 0.2, 0.2),
            roughness=0.6,
            emissive=(0.1, 0.0, 0.0)
        )

        # Dividing cell material (bright)
        self._materials["dividing"] = self._create_material(
            f"{mat_path}/DividingCell",
            color=(1.0, 0.8, 0.3),
            roughness=0.5,
            emissive=(0.2, 0.1, 0.0)
        )

        # Apoptotic cell material (dark)
        self._materials["apoptotic"] = self._create_material(
            f"{mat_path}/ApoptoticCell",
            color=(0.4, 0.3, 0.3),
            roughness=0.9,
            opacity=0.6
        )

        # Immune cell material (blue)
        self._materials["immune"] = self._create_material(
            f"{mat_path}/ImmuneCell",
            color=(0.2, 0.5, 0.9),
            roughness=0.7
        )

    def _create_material(
        self,
        path: str,
        color: Tuple[float, float, float],
        roughness: float = 0.7,
        emissive: Tuple[float, float, float] = (0, 0, 0),
        opacity: float = 1.0
    ):
        """Create a USD preview material."""
        try:
            from pxr import UsdShade

            material = UsdShade.Material.Define(self._stage, path)
            shader = UsdShade.Shader.Define(self._stage, f"{path}/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")

            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
            shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)

            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

            return material.GetPath().pathString

        except Exception as e:
            carb.log_warn(f"[cognisom.sim] Material creation failed: {e}")
            return None

    # ── Simulation Control ──────────────────────────────────────────────

    def start(self):
        """Start the simulation."""
        if self._is_running:
            return

        self._is_running = True
        self._is_paused = False
        self._time = 0.0
        self._step_count = 0

        # Initialize cells
        self._create_initial_cells()

        carb.log_info("[cognisom.sim] Simulation started")

    def stop(self):
        """Stop the simulation."""
        self._is_running = False
        self._is_paused = False

        # Clear cells
        self._clear_cells()

        carb.log_info("[cognisom.sim] Simulation stopped")

    def pause(self):
        """Pause the simulation."""
        self._is_paused = True
        carb.log_info("[cognisom.sim] Simulation paused")

    def resume(self):
        """Resume the simulation."""
        self._is_paused = False
        carb.log_info("[cognisom.sim] Simulation resumed")

    def reset(self):
        """Reset simulation to initial state."""
        self.stop()
        self.start()

    # ── Update Loop ─────────────────────────────────────────────────────

    def update(self, dt: float):
        """Update simulation state."""
        if not self._is_running or self._is_paused:
            return

        t0 = time.perf_counter()

        # Update cell states
        self._update_cells(dt)

        # Handle cell division
        self._handle_division()

        # Handle cell death
        self._handle_death()

        # Update statistics
        self._time += dt
        self._step_count += 1
        self._stats["step_time_ms"] = (time.perf_counter() - t0) * 1000
        self._stats["fps"] = 1.0 / dt if dt > 0 else 0
        self._stats["total_cells"] = len(self._cells)

    def _update_cells(self, dt: float):
        """Update all cell positions and states."""
        for cell_id, cell in list(self._cells.items()):
            # Random migration
            if cell["state"] == "normal":
                speed = self._params["migration_speed"]
                angle = cell.get("direction", 0) + (hash(cell_id) % 100 - 50) * 0.01

                cell["position"][0] += math.cos(angle) * speed * dt
                cell["position"][1] += math.sin(angle) * speed * dt * 0.5
                cell["position"][2] += math.sin(angle * 1.3) * speed * dt

                cell["direction"] = angle

            # Update USD prim
            self._update_cell_prim(cell_id, cell)

    def _update_cell_prim(self, cell_id: str, cell: Dict):
        """Update USD prim for a cell."""
        prim_path = f"{self.CELLS_PATH}/{cell_id}"
        prim = self._stage.GetPrimAtPath(prim_path)

        if not prim:
            return

        # Update transform
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()

        translate_op = xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(*cell["position"]))

        scale = cell.get("radius", 5.0) / 5.0
        scale_op = xform.AddScaleOp()
        scale_op.Set(Gf.Vec3f(scale, scale, scale))

    # ── Cell Management ─────────────────────────────────────────────────

    def _create_initial_cells(self):
        """Create initial cell population."""
        import random

        count = self._params["cell_count"]

        for i in range(count):
            cell_id = f"cell_{i:04d}"

            # Random position in a sphere
            theta = random.uniform(0, 2 * math.pi)
            phi = random.uniform(0, math.pi)
            r = random.uniform(0, 100)

            x = r * math.sin(phi) * math.cos(theta)
            y = r * math.sin(phi) * math.sin(theta) * 0.3  # Flatten Y
            z = r * math.cos(phi)

            cell = {
                "id": cell_id,
                "position": [x, y, z],
                "radius": random.uniform(4.0, 6.0),
                "state": "normal",
                "cell_type": "tumor" if random.random() < 0.7 else "immune",
                "age": 0.0,
                "direction": random.uniform(0, 2 * math.pi),
            }

            self._cells[cell_id] = cell
            self._create_cell_prim(cell_id, cell)

        carb.log_info(f"[cognisom.sim] Created {count} cells")

    def _create_cell_prim(self, cell_id: str, cell: Dict):
        """Create USD prim for a cell."""
        prim_path = f"{self.CELLS_PATH}/{cell_id}"

        # Create sphere
        sphere = UsdGeom.Sphere.Define(self._stage, prim_path)
        sphere.GetRadiusAttr().Set(cell["radius"])

        # Set transform
        xform = UsdGeom.Xformable(sphere.GetPrim())
        translate_op = xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(*cell["position"]))

        # Bind material
        mat_type = cell.get("cell_type", "tumor")
        mat_path = self._materials.get(mat_type)
        if mat_path:
            try:
                from pxr import UsdShade
                material = UsdShade.Material.Get(self._stage, mat_path)
                UsdShade.MaterialBindingAPI(sphere.GetPrim()).Bind(material)
            except Exception:
                pass

        # Add physics collision (optional)
        try:
            UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
        except Exception:
            pass

    def _handle_division(self):
        """Handle cell division events."""
        import random

        dividing = []
        for cell_id, cell in self._cells.items():
            if cell["state"] == "normal" and random.random() < self._params["division_rate"] * 0.001:
                dividing.append(cell_id)

        self._stats["dividing"] = len(dividing)

        # Limit divisions per frame
        for cell_id in dividing[:5]:
            self._divide_cell(cell_id)

    def _divide_cell(self, parent_id: str):
        """Divide a cell into two."""
        import random

        parent = self._cells.get(parent_id)
        if not parent:
            return

        # Create daughter cell
        new_id = f"cell_{len(self._cells):04d}"
        offset = [random.uniform(-5, 5) for _ in range(3)]

        daughter = {
            "id": new_id,
            "position": [
                parent["position"][0] + offset[0],
                parent["position"][1] + offset[1],
                parent["position"][2] + offset[2],
            ],
            "radius": parent["radius"] * 0.8,
            "state": "normal",
            "cell_type": parent["cell_type"],
            "age": 0.0,
            "direction": random.uniform(0, 2 * math.pi),
        }

        self._cells[new_id] = daughter
        self._create_cell_prim(new_id, daughter)

    def _handle_death(self):
        """Handle cell death events."""
        import random

        dying = []
        for cell_id, cell in self._cells.items():
            if cell["state"] == "normal" and random.random() < self._params["death_rate"] * 0.0001:
                dying.append(cell_id)

        self._stats["apoptotic"] = len(dying)

        for cell_id in dying:
            self._kill_cell(cell_id)

    def _kill_cell(self, cell_id: str):
        """Kill a cell (apoptosis)."""
        cell = self._cells.get(cell_id)
        if not cell:
            return

        cell["state"] = "apoptotic"

        # Update material
        prim_path = f"{self.CELLS_PATH}/{cell_id}"
        prim = self._stage.GetPrimAtPath(prim_path)
        if prim:
            mat_path = self._materials.get("apoptotic")
            if mat_path:
                try:
                    from pxr import UsdShade
                    material = UsdShade.Material.Get(self._stage, mat_path)
                    UsdShade.MaterialBindingAPI(prim).Bind(material)
                except Exception:
                    pass

        # Schedule removal
        # In real impl, would fade out over time

    def _clear_cells(self):
        """Remove all cells from scene."""
        for cell_id in list(self._cells.keys()):
            prim_path = f"{self.CELLS_PATH}/{cell_id}"
            prim = self._stage.GetPrimAtPath(prim_path)
            if prim:
                self._stage.RemovePrim(prim_path)

        self._cells.clear()
        carb.log_info("[cognisom.sim] Cleared all cells")

    # ── Parameter Control ───────────────────────────────────────────────

    def set_param(self, name: str, value: Any):
        """Set a simulation parameter."""
        if name in self._params:
            self._params[name] = value
            carb.log_info(f"[cognisom.sim] Set {name} = {value}")

    def get_param(self, name: str) -> Any:
        """Get a simulation parameter."""
        return self._params.get(name)
