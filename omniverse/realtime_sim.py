"""
Real-Time Simulation Loop (Phase 8)
===================================

Real-time simulation integration with Omniverse/Isaac Sim.

Provides:
- Fixed-timestep simulation loop with real-time synchronization
- Live state updates to USD scene
- Bidirectional parameter editing
- Performance monitoring and adaptive timesteps

Usage::

    from cognisom.omniverse import RealtimeSimulation, SimulationMode
    from cognisom.core import SimulationEngine

    engine = SimulationEngine(config)
    connector = OmniverseConnector()
    connector.connect()

    sim = RealtimeSimulation(engine, connector)
    sim.start()  # Begins real-time loop

    # Later
    sim.pause()
    sim.set_speed(2.0)  # 2x speed
    sim.resume()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from threading import Thread, Event, Lock
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger(__name__)


class SimulationMode(str, Enum):
    """Simulation execution modes."""
    REALTIME = "realtime"       # 1:1 with wall clock
    FAST = "fast"               # As fast as possible
    STEPPED = "stepped"         # Manual step control
    SYNCHRONIZED = "synchronized"  # Sync with Omniverse physics


@dataclass
class SimulationStats:
    """Real-time simulation statistics."""
    total_steps: int = 0
    total_time_sec: float = 0.0
    sim_time_sec: float = 0.0
    avg_step_ms: float = 0.0
    max_step_ms: float = 0.0
    min_step_ms: float = 999999.0
    updates_sent: int = 0
    updates_received: int = 0
    frame_drops: int = 0
    current_speed: float = 1.0

    def reset(self) -> None:
        """Reset statistics."""
        self.total_steps = 0
        self.total_time_sec = 0.0
        self.sim_time_sec = 0.0
        self.avg_step_ms = 0.0
        self.max_step_ms = 0.0
        self.min_step_ms = 999999.0
        self.updates_sent = 0
        self.updates_received = 0
        self.frame_drops = 0


@dataclass
class SimulationConfig:
    """Configuration for real-time simulation."""
    target_fps: float = 60.0
    sim_dt: float = 0.01         # Simulation timestep (seconds)
    max_steps_per_frame: int = 10  # Prevent runaway
    speed_multiplier: float = 1.0
    adaptive_timestep: bool = True
    sync_threshold_ms: float = 16.0  # Max allowed frame time
    buffer_size: int = 100       # State buffer for interpolation


class RealtimeSimulation:
    """Real-time simulation loop with Omniverse synchronization.

    Runs the Cognisom simulation engine in real-time, updating
    the USD scene for live visualization.
    """

    def __init__(
        self,
        engine=None,
        connector=None,
        config: Optional[SimulationConfig] = None
    ) -> None:
        """Initialize real-time simulation.

        Args:
            engine: SimulationEngine instance
            connector: OmniverseConnector instance
            config: Real-time simulation configuration
        """
        self._engine = engine
        self._connector = connector
        self._config = config or SimulationConfig()

        self._mode = SimulationMode.REALTIME
        self._running = False
        self._paused = False
        self._stop_event = Event()
        self._pause_event = Event()
        self._lock = Lock()

        self._thread: Optional[Thread] = None
        self._stats = SimulationStats()
        self._state_buffer: List[Dict] = []

        # Scene manager for USD updates
        self._scene_manager = None

        # Callbacks
        self._on_step: List[Callable] = []
        self._on_state_change: List[Callable] = []

        # Timing
        self._last_frame_time = 0.0
        self._accumulator = 0.0
        self._frame_count = 0

    @property
    def is_running(self) -> bool:
        """Whether simulation is actively running."""
        return self._running and not self._paused

    @property
    def is_paused(self) -> bool:
        """Whether simulation is paused."""
        return self._paused

    @property
    def mode(self) -> SimulationMode:
        """Current simulation mode."""
        return self._mode

    @property
    def stats(self) -> SimulationStats:
        """Current statistics."""
        return self._stats

    @property
    def sim_time(self) -> float:
        """Current simulation time."""
        if self._engine:
            return getattr(self._engine, "time", self._stats.sim_time_sec)
        return self._stats.sim_time_sec

    # ── Control ─────────────────────────────────────────────────────────

    def start(self, mode: SimulationMode = SimulationMode.REALTIME) -> None:
        """Start the real-time simulation loop.

        Args:
            mode: Simulation execution mode
        """
        if self._running:
            log.warning("Simulation already running")
            return

        self._mode = mode
        self._running = True
        self._paused = False
        self._stop_event.clear()
        self._pause_event.clear()
        self._stats.reset()

        # Initialize scene manager if we have a connector
        if self._connector and self._connector.is_connected:
            from .scene_manager import SceneManager
            self._scene_manager = SceneManager(self._connector)
            self._scene_manager.initialize()

        # Start simulation thread
        self._thread = Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        log.info("Real-time simulation started in %s mode", mode.value)

    def stop(self) -> None:
        """Stop the simulation loop."""
        self._stop_event.set()
        self._running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        log.info("Real-time simulation stopped")

    def pause(self) -> None:
        """Pause the simulation."""
        self._paused = True
        self._pause_event.set()
        log.info("Simulation paused at t=%.3f", self.sim_time)

    def resume(self) -> None:
        """Resume paused simulation."""
        self._paused = False
        self._pause_event.clear()
        self._last_frame_time = time.perf_counter()  # Reset timing
        log.info("Simulation resumed")

    def step(self) -> None:
        """Execute a single simulation step (for STEPPED mode)."""
        if self._mode != SimulationMode.STEPPED:
            log.warning("Manual step only available in STEPPED mode")
            return

        self._execute_step()

    def set_speed(self, multiplier: float) -> None:
        """Set simulation speed multiplier.

        Args:
            multiplier: Speed factor (1.0 = real-time, 2.0 = 2x speed)
        """
        with self._lock:
            self._config.speed_multiplier = max(0.1, min(100.0, multiplier))
            self._stats.current_speed = self._config.speed_multiplier
        log.info("Simulation speed set to %.1fx", multiplier)

    def set_mode(self, mode: SimulationMode) -> None:
        """Change simulation mode."""
        self._mode = mode
        log.info("Simulation mode changed to %s", mode.value)

    # ── Main Loop ───────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        """Main simulation loop."""
        self._last_frame_time = time.perf_counter()
        target_frame_time = 1.0 / self._config.target_fps

        while not self._stop_event.is_set():
            # Handle pause
            if self._paused:
                self._pause_event.wait(timeout=0.1)
                if self._paused:
                    continue
                self._last_frame_time = time.perf_counter()

            # Calculate delta time
            current_time = time.perf_counter()
            frame_dt = current_time - self._last_frame_time
            self._last_frame_time = current_time

            # Execute based on mode
            if self._mode == SimulationMode.REALTIME:
                self._update_realtime(frame_dt)
            elif self._mode == SimulationMode.FAST:
                self._update_fast()
            elif self._mode == SimulationMode.SYNCHRONIZED:
                self._update_synchronized(frame_dt)
            # STEPPED mode doesn't auto-update

            # Sync to USD scene
            self._sync_to_scene()

            # Frame rate limiting for REALTIME mode
            if self._mode == SimulationMode.REALTIME:
                elapsed = time.perf_counter() - current_time
                sleep_time = target_frame_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    self._stats.frame_drops += 1

            self._frame_count += 1

    def _update_realtime(self, frame_dt: float) -> None:
        """Update simulation in real-time mode."""
        # Apply speed multiplier
        scaled_dt = frame_dt * self._config.speed_multiplier

        # Use fixed timestep with accumulator
        self._accumulator += scaled_dt
        steps = 0

        while self._accumulator >= self._config.sim_dt:
            if steps >= self._config.max_steps_per_frame:
                # Prevent spiral of death
                self._accumulator = 0
                log.warning("Max steps per frame exceeded, resetting accumulator")
                break

            self._execute_step()
            self._accumulator -= self._config.sim_dt
            steps += 1

    def _update_fast(self) -> None:
        """Update simulation as fast as possible."""
        # Run multiple steps per frame
        for _ in range(self._config.max_steps_per_frame):
            self._execute_step()

    def _update_synchronized(self, frame_dt: float) -> None:
        """Update simulation synchronized with Omniverse physics."""
        # Match Omniverse physics timestep
        if self._connector and self._connector.is_connected:
            # Query Omniverse physics time step
            omni_dt = self._get_omniverse_dt()
            steps = max(1, int(frame_dt / omni_dt))
            for _ in range(steps):
                self._execute_step()
        else:
            self._update_realtime(frame_dt)

    def _execute_step(self) -> None:
        """Execute a single simulation step."""
        t0 = time.perf_counter()

        with self._lock:
            if self._engine:
                try:
                    self._engine.step()
                    self._stats.sim_time_sec = self._engine.time
                except Exception as e:
                    log.error("Simulation step error: %s", e)
            else:
                # No engine, just advance time
                self._stats.sim_time_sec += self._config.sim_dt

        step_time_ms = (time.perf_counter() - t0) * 1000

        # Update stats
        self._stats.total_steps += 1
        self._stats.max_step_ms = max(self._stats.max_step_ms, step_time_ms)
        self._stats.min_step_ms = min(self._stats.min_step_ms, step_time_ms)

        # Running average
        n = self._stats.total_steps
        self._stats.avg_step_ms = (
            self._stats.avg_step_ms * (n - 1) + step_time_ms
        ) / n

        # Buffer state for interpolation
        self._buffer_state()

        # Notify callbacks
        for callback in self._on_step:
            try:
                callback(self._stats)
            except Exception as e:
                log.warning("Step callback error: %s", e)

    def _buffer_state(self) -> None:
        """Buffer current state for interpolation."""
        if len(self._state_buffer) >= self._config.buffer_size:
            self._state_buffer.pop(0)

        state = {
            "time": self._stats.sim_time_sec,
            "step": self._stats.total_steps,
        }

        if self._engine:
            # Extract relevant state from engine
            state["entities"] = self._extract_entity_states()

        self._state_buffer.append(state)

    def _extract_entity_states(self) -> List[Dict]:
        """Extract entity states from simulation engine."""
        entities = []

        if not self._engine:
            return entities

        # Get cells/entities from engine
        cells = getattr(self._engine, "cells", [])
        for cell in cells[:1000]:  # Limit for performance
            entities.append({
                "id": getattr(cell, "cell_id", id(cell)),
                "position": list(getattr(cell, "position", [0, 0, 0])),
                "radius": getattr(cell, "radius", 5.0),
                "state": getattr(cell, "state", "normal"),
                "type": getattr(cell, "cell_type", "generic"),
            })

        return entities

    # ── Scene Synchronization ───────────────────────────────────────────

    def _sync_to_scene(self) -> None:
        """Synchronize simulation state to USD scene."""
        if not self._scene_manager:
            return

        try:
            if self._state_buffer:
                current_state = self._state_buffer[-1]
                entities = current_state.get("entities", [])

                for entity in entities:
                    self._scene_manager.update_entity(
                        entity_id=str(entity["id"]),
                        position=entity["position"],
                        radius=entity.get("radius", 5.0),
                        state=entity.get("state", "normal"),
                        entity_type=entity.get("type", "cell"),
                    )

                self._stats.updates_sent += 1

        except Exception as e:
            log.warning("Scene sync error: %s", e)

    def _get_omniverse_dt(self) -> float:
        """Get Omniverse physics timestep."""
        # Default Isaac Sim physics timestep
        return 1.0 / 60.0

    # ── Parameter Editing ───────────────────────────────────────────────

    def set_parameter(self, param_name: str, value: Any) -> bool:
        """Set simulation parameter (bidirectional editing).

        Args:
            param_name: Parameter path (e.g., "cell.division_rate")
            value: New parameter value

        Returns:
            True if parameter was set successfully
        """
        with self._lock:
            if not self._engine:
                log.warning("No engine connected")
                return False

            try:
                parts = param_name.split(".")
                target = self._engine

                for part in parts[:-1]:
                    target = getattr(target, part)

                setattr(target, parts[-1], value)
                log.info("Set parameter %s = %s", param_name, value)

                # Notify state change
                for callback in self._on_state_change:
                    callback(param_name, value)

                return True

            except Exception as e:
                log.error("Failed to set parameter %s: %s", param_name, e)
                return False

    def get_parameter(self, param_name: str) -> Any:
        """Get current parameter value."""
        with self._lock:
            if not self._engine:
                return None

            try:
                parts = param_name.split(".")
                target = self._engine

                for part in parts:
                    target = getattr(target, part)

                return target

            except Exception:
                return None

    # ── Callbacks ───────────────────────────────────────────────────────

    def on_step(self, callback: Callable[[SimulationStats], None]) -> None:
        """Register step callback."""
        self._on_step.append(callback)

    def on_state_change(self, callback: Callable[[str, Any], None]) -> None:
        """Register state change callback."""
        self._on_state_change.append(callback)

    # ── Status ──────────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Get current simulation status."""
        return {
            "running": self._running,
            "paused": self._paused,
            "mode": self._mode.value,
            "sim_time": self.sim_time,
            "speed": self._config.speed_multiplier,
            "total_steps": self._stats.total_steps,
            "avg_step_ms": self._stats.avg_step_ms,
            "fps": self._config.target_fps,
            "updates_sent": self._stats.updates_sent,
            "frame_drops": self._stats.frame_drops,
            "has_engine": self._engine is not None,
            "has_connector": self._connector is not None,
        }

    def get_interpolated_state(self, time_offset: float = 0.0) -> Optional[Dict]:
        """Get interpolated state at given time offset.

        Args:
            time_offset: Offset from current time (negative = past)

        Returns:
            Interpolated state dictionary
        """
        if not self._state_buffer:
            return None

        target_time = self._stats.sim_time_sec + time_offset

        # Find bracketing states
        prev_state = None
        next_state = None

        for state in self._state_buffer:
            if state["time"] <= target_time:
                prev_state = state
            else:
                next_state = state
                break

        if prev_state is None:
            return self._state_buffer[0] if self._state_buffer else None

        if next_state is None:
            return self._state_buffer[-1]

        # Linear interpolation factor
        dt = next_state["time"] - prev_state["time"]
        if dt <= 0:
            return prev_state

        t = (target_time - prev_state["time"]) / dt

        # Interpolate entity positions
        interpolated = {
            "time": target_time,
            "step": prev_state["step"],
            "entities": [],
        }

        prev_entities = {e["id"]: e for e in prev_state.get("entities", [])}
        next_entities = {e["id"]: e for e in next_state.get("entities", [])}

        for eid, prev_e in prev_entities.items():
            if eid in next_entities:
                next_e = next_entities[eid]
                interp_e = {
                    "id": eid,
                    "position": [
                        prev_e["position"][i] + t * (next_e["position"][i] - prev_e["position"][i])
                        for i in range(3)
                    ],
                    "radius": prev_e["radius"] + t * (next_e["radius"] - prev_e["radius"]),
                    "state": next_e["state"],
                    "type": prev_e["type"],
                }
                interpolated["entities"].append(interp_e)
            else:
                interpolated["entities"].append(prev_e)

        return interpolated
