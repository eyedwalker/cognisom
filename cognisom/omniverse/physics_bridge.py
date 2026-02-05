"""
Physics Bridge (Phase 8)
========================

Bridge between Cognisom physics and PhysX/Isaac Sim.

Enables:
- Physical interactions in Omniverse environment
- Collision detection using PhysX
- Soft body physics for cells (optional)
- Force field integration
- Fluid dynamics coupling

Usage::

    from cognisom.omniverse import PhysicsBridge

    bridge = PhysicsBridge(connector, engine)
    bridge.initialize()

    # Enable physics simulation
    bridge.enable_collisions(True)
    bridge.set_gravity([0, -9.81, 0])

    # Update loop syncs physics
    bridge.sync()
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


class PhysicsMode(str, Enum):
    """Physics simulation modes."""
    DISABLED = "disabled"         # No physics
    BASIC = "basic"              # Basic collision detection
    FULL = "full"                # Full PhysX simulation
    SOFT_BODY = "soft_body"      # Soft body for cells
    FLUID = "fluid"              # Fluid dynamics


class CollisionType(str, Enum):
    """Types of collision responses."""
    NONE = "none"
    BOUNCE = "bounce"
    SLIDE = "slide"
    ABSORB = "absorb"           # For cell-drug interactions
    FUSE = "fuse"               # For cell division/fusion


@dataclass
class PhysicsConfig:
    """Physics simulation configuration."""
    mode: PhysicsMode = PhysicsMode.BASIC
    gravity: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Biological - no gravity
    time_step: float = 1.0 / 60.0
    substeps: int = 4
    collision_enabled: bool = True
    soft_body_enabled: bool = False
    fluid_enabled: bool = False
    cell_stiffness: float = 100.0    # Cell membrane stiffness
    cell_damping: float = 0.5
    adhesion_strength: float = 10.0   # Cell-cell adhesion
    friction: float = 0.3
    restitution: float = 0.2         # Bounciness
    max_velocity: float = 100.0      # Clamp max velocity


@dataclass
class CollisionEvent:
    """A collision between two entities."""
    entity_a: str = ""
    entity_b: str = ""
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    normal: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    impulse: float = 0.0
    timestamp: float = 0.0
    collision_type: CollisionType = CollisionType.BOUNCE

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


@dataclass
class PhysicsBody:
    """Physics body representation."""
    entity_id: str = ""
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    acceleration: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    angular_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    mass: float = 1.0
    radius: float = 5.0
    is_static: bool = False
    collision_type: CollisionType = CollisionType.BOUNCE
    force_accumulator: Tuple[float, float, float] = (0.0, 0.0, 0.0)


class PhysicsBridge:
    """Bridge between Cognisom physics and PhysX/Isaac Sim.

    Synchronizes physical simulation between the biological model
    and Omniverse's physics engine.
    """

    def __init__(
        self,
        connector=None,
        engine=None,
        config: Optional[PhysicsConfig] = None
    ) -> None:
        """Initialize physics bridge.

        Args:
            connector: OmniverseConnector instance
            engine: SimulationEngine instance
            config: Physics configuration
        """
        self._connector = connector
        self._engine = engine
        self._config = config or PhysicsConfig()

        self._stage = None
        self._physics_scene = None
        self._bodies: Dict[str, PhysicsBody] = {}

        self._collision_handlers: List[Callable[[CollisionEvent], None]] = []
        self._collision_history: List[CollisionEvent] = []

        # Force fields
        self._force_fields: List[Dict] = []

        # Performance
        self._last_sync_time = 0.0
        self._sync_count = 0
        self._collision_count = 0

        # Isaac Sim references (lazy loaded)
        self._isaac_core = None
        self._physx = None

    @property
    def mode(self) -> PhysicsMode:
        """Current physics mode."""
        return self._config.mode

    @property
    def is_active(self) -> bool:
        """Whether physics is active."""
        return self._config.mode != PhysicsMode.DISABLED

    def initialize(self) -> bool:
        """Initialize physics simulation.

        Returns:
            True if initialization successful
        """
        if not self._connector:
            log.warning("No connector provided")
            return self._init_standalone()

        if not self._connector.is_connected:
            log.warning("Connector not connected")
            return self._init_standalone()

        self._stage = self._connector.get_stage()

        # Try to initialize Isaac Sim physics
        try:
            return self._init_isaac_physics()
        except ImportError:
            log.info("Isaac Sim not available, using built-in physics")
            return self._init_standalone()

    def _init_isaac_physics(self) -> bool:
        """Initialize Isaac Sim physics."""
        try:
            from omni.isaac.core import World
            from pxr import UsdPhysics, PhysxSchema

            self._isaac_core = World
            self._physx = PhysxSchema

            # Create physics scene
            physics_path = "/World/PhysicsScene"
            if not self._stage.GetPrimAtPath(physics_path):
                physics_scene = UsdPhysics.Scene.Define(self._stage, physics_path)

                # Configure physics
                physics_scene.CreateGravityDirectionAttr().Set(self._config.gravity)
                physics_scene.CreateGravityMagnitudeAttr().Set(
                    math.sqrt(sum(g**2 for g in self._config.gravity))
                )

                # PhysX specific settings
                physx_scene = PhysxSchema.PhysxSceneAPI.Apply(physics_scene.GetPrim())
                physx_scene.CreateTimeStepsPerSecondAttr().Set(
                    int(1.0 / self._config.time_step)
                )

                self._physics_scene = physics_scene

            log.info("Isaac Sim physics initialized")
            return True

        except Exception as e:
            log.warning("Isaac physics init failed: %s", e)
            return self._init_standalone()

    def _init_standalone(self) -> bool:
        """Initialize standalone physics (no Isaac Sim)."""
        log.info("Using standalone physics simulation")
        self._physics_scene = None
        return True

    # ── Body Management ─────────────────────────────────────────────────

    def add_body(
        self,
        entity_id: str,
        position: Tuple[float, float, float],
        velocity: Tuple[float, float, float] = (0, 0, 0),
        mass: float = 1.0,
        radius: float = 5.0,
        is_static: bool = False,
        collision_type: CollisionType = CollisionType.BOUNCE
    ) -> PhysicsBody:
        """Add a physics body.

        Args:
            entity_id: Unique identifier
            position: Initial position
            velocity: Initial velocity
            mass: Body mass
            radius: Collision radius
            is_static: Whether body is static (immovable)
            collision_type: How collisions are handled

        Returns:
            The created PhysicsBody
        """
        body = PhysicsBody(
            entity_id=entity_id,
            position=position,
            velocity=velocity,
            mass=mass,
            radius=radius,
            is_static=is_static,
            collision_type=collision_type,
        )

        self._bodies[entity_id] = body

        # Create physics collider in USD if available
        if self._stage and self._physx:
            self._create_physx_body(body)

        return body

    def remove_body(self, entity_id: str) -> bool:
        """Remove a physics body."""
        if entity_id not in self._bodies:
            return False

        del self._bodies[entity_id]
        return True

    def update_body(
        self,
        entity_id: str,
        position: Optional[Tuple[float, float, float]] = None,
        velocity: Optional[Tuple[float, float, float]] = None,
        mass: Optional[float] = None,
        radius: Optional[float] = None
    ) -> Optional[PhysicsBody]:
        """Update an existing physics body."""
        if entity_id not in self._bodies:
            return None

        body = self._bodies[entity_id]

        if position is not None:
            body.position = position
        if velocity is not None:
            body.velocity = velocity
        if mass is not None:
            body.mass = mass
        if radius is not None:
            body.radius = radius

        return body

    def _create_physx_body(self, body: PhysicsBody) -> None:
        """Create PhysX rigid body in USD."""
        if not self._stage:
            return

        try:
            from pxr import UsdPhysics, UsdGeom

            prim_path = f"/World/Physics/Body_{body.entity_id}"

            # Create sphere geometry
            sphere = UsdGeom.Sphere.Define(self._stage, prim_path)
            sphere.GetRadiusAttr().Set(body.radius)

            # Add rigid body API
            rigid_body = UsdPhysics.RigidBodyAPI.Apply(sphere.GetPrim())
            if body.is_static:
                rigid_body.CreateKinematicEnabledAttr().Set(True)

            # Add collision
            UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())

            # Set mass
            mass_api = UsdPhysics.MassAPI.Apply(sphere.GetPrim())
            mass_api.CreateMassAttr().Set(body.mass)

        except Exception as e:
            log.debug("PhysX body creation: %s", e)

    # ── Force Application ───────────────────────────────────────────────

    def apply_force(
        self,
        entity_id: str,
        force: Tuple[float, float, float],
        point: Optional[Tuple[float, float, float]] = None
    ) -> bool:
        """Apply force to a body.

        Args:
            entity_id: Target body
            force: Force vector (N)
            point: Point of application (uses center if None)

        Returns:
            True if force was applied
        """
        if entity_id not in self._bodies:
            return False

        body = self._bodies[entity_id]
        if body.is_static:
            return False

        # Accumulate force
        body.force_accumulator = (
            body.force_accumulator[0] + force[0],
            body.force_accumulator[1] + force[1],
            body.force_accumulator[2] + force[2],
        )

        return True

    def apply_impulse(
        self,
        entity_id: str,
        impulse: Tuple[float, float, float]
    ) -> bool:
        """Apply instant velocity change.

        Args:
            entity_id: Target body
            impulse: Impulse vector

        Returns:
            True if impulse was applied
        """
        if entity_id not in self._bodies:
            return False

        body = self._bodies[entity_id]
        if body.is_static:
            return False

        # Apply impulse: Δv = impulse / mass
        dv = (
            impulse[0] / body.mass,
            impulse[1] / body.mass,
            impulse[2] / body.mass,
        )

        body.velocity = (
            body.velocity[0] + dv[0],
            body.velocity[1] + dv[1],
            body.velocity[2] + dv[2],
        )

        return True

    # ── Force Fields ────────────────────────────────────────────────────

    def add_force_field(
        self,
        field_type: str,
        center: Tuple[float, float, float],
        strength: float,
        radius: float = 100.0,
        falloff: str = "linear"
    ) -> int:
        """Add a force field.

        Args:
            field_type: "attraction", "repulsion", "vortex", "flow"
            center: Field center position
            strength: Field strength
            radius: Effective radius
            falloff: "linear", "quadratic", "constant"

        Returns:
            Field index
        """
        field = {
            "type": field_type,
            "center": center,
            "strength": strength,
            "radius": radius,
            "falloff": falloff,
            "enabled": True,
        }
        self._force_fields.append(field)
        return len(self._force_fields) - 1

    def remove_force_field(self, index: int) -> bool:
        """Remove a force field."""
        if 0 <= index < len(self._force_fields):
            del self._force_fields[index]
            return True
        return False

    def _apply_force_fields(self, body: PhysicsBody) -> Tuple[float, float, float]:
        """Calculate force from all force fields."""
        total_force = [0.0, 0.0, 0.0]

        for field in self._force_fields:
            if not field.get("enabled", True):
                continue

            center = field["center"]
            strength = field["strength"]
            radius = field["radius"]
            falloff = field["falloff"]
            field_type = field["type"]

            # Vector from body to field center
            dx = center[0] - body.position[0]
            dy = center[1] - body.position[1]
            dz = center[2] - body.position[2]

            dist = math.sqrt(dx**2 + dy**2 + dz**2)

            if dist > radius or dist < 0.001:
                continue

            # Normalize direction
            dx /= dist
            dy /= dist
            dz /= dist

            # Calculate falloff
            if falloff == "linear":
                factor = 1.0 - dist / radius
            elif falloff == "quadratic":
                factor = 1.0 - (dist / radius) ** 2
            else:  # constant
                factor = 1.0

            force_mag = strength * factor

            # Apply based on field type
            if field_type == "attraction":
                total_force[0] += dx * force_mag
                total_force[1] += dy * force_mag
                total_force[2] += dz * force_mag
            elif field_type == "repulsion":
                total_force[0] -= dx * force_mag
                total_force[1] -= dy * force_mag
                total_force[2] -= dz * force_mag
            elif field_type == "vortex":
                # Perpendicular force (in XZ plane)
                total_force[0] -= dz * force_mag
                total_force[2] += dx * force_mag
            elif field_type == "flow":
                # Directional flow (use direction stored in field)
                flow_dir = field.get("direction", (1, 0, 0))
                total_force[0] += flow_dir[0] * force_mag
                total_force[1] += flow_dir[1] * force_mag
                total_force[2] += flow_dir[2] * force_mag

        return tuple(total_force)

    # ── Collision Detection ─────────────────────────────────────────────

    def _detect_collisions(self) -> List[CollisionEvent]:
        """Detect collisions between bodies."""
        collisions = []
        bodies = list(self._bodies.values())

        for i, body_a in enumerate(bodies):
            for body_b in bodies[i + 1:]:
                # Distance between centers
                dx = body_b.position[0] - body_a.position[0]
                dy = body_b.position[1] - body_a.position[1]
                dz = body_b.position[2] - body_a.position[2]
                dist = math.sqrt(dx**2 + dy**2 + dz**2)

                # Check overlap
                min_dist = body_a.radius + body_b.radius
                if dist < min_dist:
                    # Collision detected
                    if dist > 0:
                        normal = (dx/dist, dy/dist, dz/dist)
                    else:
                        normal = (1, 0, 0)

                    contact_point = (
                        body_a.position[0] + normal[0] * body_a.radius,
                        body_a.position[1] + normal[1] * body_a.radius,
                        body_a.position[2] + normal[2] * body_a.radius,
                    )

                    # Calculate impulse
                    rel_vel = (
                        body_b.velocity[0] - body_a.velocity[0],
                        body_b.velocity[1] - body_a.velocity[1],
                        body_b.velocity[2] - body_a.velocity[2],
                    )
                    vel_along_normal = (
                        rel_vel[0] * normal[0] +
                        rel_vel[1] * normal[1] +
                        rel_vel[2] * normal[2]
                    )

                    impulse = abs(vel_along_normal * (body_a.mass + body_b.mass) / 2)

                    # Determine collision type
                    ctype = CollisionType.BOUNCE
                    if body_a.collision_type == CollisionType.ABSORB or \
                       body_b.collision_type == CollisionType.ABSORB:
                        ctype = CollisionType.ABSORB

                    event = CollisionEvent(
                        entity_a=body_a.entity_id,
                        entity_b=body_b.entity_id,
                        position=contact_point,
                        normal=normal,
                        impulse=impulse,
                        collision_type=ctype,
                    )
                    collisions.append(event)

        return collisions

    def _resolve_collision(self, event: CollisionEvent) -> None:
        """Resolve a collision between two bodies."""
        body_a = self._bodies.get(event.entity_a)
        body_b = self._bodies.get(event.entity_b)

        if not body_a or not body_b:
            return

        if event.collision_type == CollisionType.NONE:
            return

        if event.collision_type == CollisionType.ABSORB:
            # Special handling for cell-drug absorption
            for handler in self._collision_handlers:
                handler(event)
            return

        # Calculate collision response
        normal = event.normal

        # Relative velocity
        rel_vel = (
            body_a.velocity[0] - body_b.velocity[0],
            body_a.velocity[1] - body_b.velocity[1],
            body_a.velocity[2] - body_b.velocity[2],
        )

        vel_along_normal = (
            rel_vel[0] * normal[0] +
            rel_vel[1] * normal[1] +
            rel_vel[2] * normal[2]
        )

        # Don't resolve if separating
        if vel_along_normal > 0:
            return

        # Restitution
        e = self._config.restitution

        # Calculate impulse scalar
        if body_a.is_static:
            j = -(1 + e) * vel_along_normal / (1 / body_b.mass)
        elif body_b.is_static:
            j = -(1 + e) * vel_along_normal / (1 / body_a.mass)
        else:
            j = -(1 + e) * vel_along_normal / (1 / body_a.mass + 1 / body_b.mass)

        # Apply impulse
        impulse = (j * normal[0], j * normal[1], j * normal[2])

        if not body_a.is_static:
            body_a.velocity = (
                body_a.velocity[0] + impulse[0] / body_a.mass,
                body_a.velocity[1] + impulse[1] / body_a.mass,
                body_a.velocity[2] + impulse[2] / body_a.mass,
            )

        if not body_b.is_static:
            body_b.velocity = (
                body_b.velocity[0] - impulse[0] / body_b.mass,
                body_b.velocity[1] - impulse[1] / body_b.mass,
                body_b.velocity[2] - impulse[2] / body_b.mass,
            )

        # Separate overlapping bodies
        dx = body_b.position[0] - body_a.position[0]
        dy = body_b.position[1] - body_a.position[1]
        dz = body_b.position[2] - body_a.position[2]
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        overlap = (body_a.radius + body_b.radius) - dist

        if overlap > 0 and dist > 0:
            sep = overlap / 2 + 0.01
            if not body_a.is_static:
                body_a.position = (
                    body_a.position[0] - normal[0] * sep,
                    body_a.position[1] - normal[1] * sep,
                    body_a.position[2] - normal[2] * sep,
                )
            if not body_b.is_static:
                body_b.position = (
                    body_b.position[0] + normal[0] * sep,
                    body_b.position[1] + normal[1] * sep,
                    body_b.position[2] + normal[2] * sep,
                )

    # ── Physics Step ────────────────────────────────────────────────────

    def step(self, dt: Optional[float] = None) -> None:
        """Step physics simulation.

        Args:
            dt: Time step (uses config if None)
        """
        if self._config.mode == PhysicsMode.DISABLED:
            return

        dt = dt or self._config.time_step

        # Substep integration
        sub_dt = dt / self._config.substeps

        for _ in range(self._config.substeps):
            # Apply force fields
            for body in self._bodies.values():
                if body.is_static:
                    continue

                # Get force field contribution
                field_force = self._apply_force_fields(body)

                # Total force = accumulated + field forces
                total_force = (
                    body.force_accumulator[0] + field_force[0],
                    body.force_accumulator[1] + field_force[1],
                    body.force_accumulator[2] + field_force[2],
                )

                # Apply gravity
                grav = self._config.gravity
                total_force = (
                    total_force[0] + grav[0] * body.mass,
                    total_force[1] + grav[1] * body.mass,
                    total_force[2] + grav[2] * body.mass,
                )

                # Calculate acceleration: a = F / m
                body.acceleration = (
                    total_force[0] / body.mass,
                    total_force[1] / body.mass,
                    total_force[2] / body.mass,
                )

                # Update velocity: v += a * dt
                body.velocity = (
                    body.velocity[0] + body.acceleration[0] * sub_dt,
                    body.velocity[1] + body.acceleration[1] * sub_dt,
                    body.velocity[2] + body.acceleration[2] * sub_dt,
                )

                # Apply damping
                damping = 1.0 - self._config.cell_damping * sub_dt
                body.velocity = (
                    body.velocity[0] * damping,
                    body.velocity[1] * damping,
                    body.velocity[2] * damping,
                )

                # Clamp velocity
                speed = math.sqrt(
                    body.velocity[0]**2 +
                    body.velocity[1]**2 +
                    body.velocity[2]**2
                )
                if speed > self._config.max_velocity:
                    scale = self._config.max_velocity / speed
                    body.velocity = (
                        body.velocity[0] * scale,
                        body.velocity[1] * scale,
                        body.velocity[2] * scale,
                    )

                # Update position: x += v * dt
                body.position = (
                    body.position[0] + body.velocity[0] * sub_dt,
                    body.position[1] + body.velocity[1] * sub_dt,
                    body.position[2] + body.velocity[2] * sub_dt,
                )

            # Clear force accumulators
            for body in self._bodies.values():
                body.force_accumulator = (0.0, 0.0, 0.0)

            # Detect and resolve collisions
            if self._config.collision_enabled:
                collisions = self._detect_collisions()
                for collision in collisions:
                    self._resolve_collision(collision)
                    self._collision_history.append(collision)
                    self._collision_count += 1

                    # Notify handlers
                    for handler in self._collision_handlers:
                        handler(collision)

        # Trim collision history
        if len(self._collision_history) > 1000:
            self._collision_history = self._collision_history[-500:]

    # ── Synchronization ─────────────────────────────────────────────────

    def sync(self) -> None:
        """Synchronize physics state with simulation engine."""
        t0 = time.time()

        if self._engine:
            # Update from engine
            cells = getattr(self._engine, "cells", [])
            for cell in cells:
                cell_id = str(getattr(cell, "cell_id", id(cell)))
                pos = tuple(getattr(cell, "position", [0, 0, 0]))
                vel = tuple(getattr(cell, "velocity", [0, 0, 0]))
                radius = getattr(cell, "radius", 5.0)

                if cell_id in self._bodies:
                    self.update_body(cell_id, position=pos, velocity=vel, radius=radius)
                else:
                    self.add_body(cell_id, position=pos, velocity=vel, radius=radius)

            # Remove bodies for deleted cells
            cell_ids = {str(getattr(c, "cell_id", id(c))) for c in cells}
            for body_id in list(self._bodies.keys()):
                if body_id not in cell_ids:
                    self.remove_body(body_id)

        self._last_sync_time = time.time() - t0
        self._sync_count += 1

    def sync_to_engine(self) -> None:
        """Push physics state back to simulation engine."""
        if not self._engine:
            return

        cells = getattr(self._engine, "cells", [])
        for cell in cells:
            cell_id = str(getattr(cell, "cell_id", id(cell)))
            if cell_id in self._bodies:
                body = self._bodies[cell_id]
                if hasattr(cell, "position"):
                    cell.position = list(body.position)
                if hasattr(cell, "velocity"):
                    cell.velocity = list(body.velocity)

    # ── Callbacks ───────────────────────────────────────────────────────

    def on_collision(self, handler: Callable[[CollisionEvent], None]) -> None:
        """Register collision callback."""
        self._collision_handlers.append(handler)

    # ── Configuration ───────────────────────────────────────────────────

    def set_gravity(self, gravity: Tuple[float, float, float]) -> None:
        """Set gravity vector."""
        self._config.gravity = gravity

    def enable_collisions(self, enabled: bool) -> None:
        """Enable/disable collision detection."""
        self._config.collision_enabled = enabled

    def set_mode(self, mode: PhysicsMode) -> None:
        """Set physics mode."""
        self._config.mode = mode

    # ── Status ──────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Get physics statistics."""
        return {
            "mode": self._config.mode.value,
            "body_count": len(self._bodies),
            "force_field_count": len(self._force_fields),
            "collision_count": self._collision_count,
            "sync_count": self._sync_count,
            "last_sync_ms": self._last_sync_time * 1000,
            "collision_enabled": self._config.collision_enabled,
            "gravity": self._config.gravity,
            "has_isaac": self._isaac_core is not None,
        }

    def get_recent_collisions(self, limit: int = 50) -> List[CollisionEvent]:
        """Get recent collision events."""
        return self._collision_history[-limit:]
