"""
Diapedesis Simulation Manager for Omniverse Kit
=================================================

Bridges the DiapedesisSim engine with the DiapedesisSceneBuilder,
providing frame-based playback of the leukocyte extravasation cascade
in Omniverse Kit with RTX rendering.

Usage from extension.py::

    mgr = DiapedesisManager()
    mgr.load_preset("inflammation")   # Run simulation, collect frames
    mgr.build_scene()                 # Build USD scene from first frame
    mgr.play()                        # Start playback
    # per frame update:
    mgr.update(dt)                    # Advance and apply next frame
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import carb
import omni.usd

try:
    from pxr import Usd
    USD_AVAILABLE = True
except ImportError:
    USD_AVAILABLE = False

from .diapedesis_scene import DiapedesisSceneBuilder


# ── Preset Configurations ────────────────────────────────────────────────

PRESETS = {
    "healthy": {
        "label": "Healthy Vessel",
        "description": "Normal blood flow, no inflammation",
        "params": {
            "initial_inflammation": 0.0,
            "tnf_alpha": 0.0,
            "il1_beta": 0.0,
        },
    },
    "inflammation": {
        "label": "Acute Inflammation (Normal)",
        "description": "Full 7-step diapedesis cascade",
        "params": {
            "initial_inflammation": 0.8,
            "tnf_alpha": 50.0,
            "il1_beta": 40.0,
        },
    },
    "lad1": {
        "label": "LAD-1 (Selectin Deficiency)",
        "description": "Missing selectin ligands - rolling failure",
        "params": {
            "initial_inflammation": 0.8,
            "tnf_alpha": 50.0,
            "selectin_ligand_expr": 0.0,
        },
    },
    "lad2": {
        "label": "LAD-2 (Fucose Deficiency)",
        "description": "Missing sLex - slow rolling only",
        "params": {
            "initial_inflammation": 0.8,
            "tnf_alpha": 50.0,
            "slow_rolling_factor": 0.1,
        },
    },
    "lad3": {
        "label": "LAD-3 (Integrin Activation Defect)",
        "description": "Integrins can't activate - no firm adhesion",
        "params": {
            "initial_inflammation": 0.8,
            "tnf_alpha": 50.0,
            "integrin_activation_rate": 0.0,
        },
    },
}


class DiapedesisManager:
    """Manages diapedesis simulation playback in Omniverse Kit.

    Runs the DiapedesisSim to pre-compute frames, then plays them back
    through the DiapedesisSceneBuilder for RTX rendering.
    """

    def __init__(self):
        self._stage: Optional[Usd.Stage] = None
        self._scene_builder: Optional[DiapedesisSceneBuilder] = None
        self._frames: List[Dict[str, Any]] = []
        self._current_frame: int = 0
        self._is_playing: bool = False
        self._is_paused: bool = False
        self._playback_speed: float = 1.0
        self._accumulated_time: float = 0.0
        self._frame_interval: float = 1.0 / 30.0  # 30 fps playback
        self._preset_name: str = ""
        self._scene_built: bool = False
        self._build_requested: bool = False  # queued for main thread

        # Stats
        self._stats = {
            "total_frames": 0,
            "current_frame": 0,
            "time": 0.0,
            "preset": "",
            "fps": 0.0,
            "step_time_ms": 0.0,
            "leukocytes": 0,
            "rbcs": 0,
            "bacteria_alive": 0,
        }

        self._init_stage()

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    @property
    def stats(self) -> Dict[str, Any]:
        return self._stats

    @property
    def total_frames(self) -> int:
        return len(self._frames)

    @property
    def current_frame(self) -> int:
        return self._current_frame

    @property
    def preset_name(self) -> str:
        return self._preset_name

    @property
    def playback_speed(self) -> float:
        return self._playback_speed

    @playback_speed.setter
    def playback_speed(self, value: float):
        self._playback_speed = max(0.1, min(5.0, value))

    def _init_stage(self):
        """Get or create a USD stage for scene building."""
        # Try getting the existing stage from Kit's USD context
        try:
            context = omni.usd.get_context()
            self._stage = context.get_stage()
        except Exception:
            self._stage = None

        # If no stage available (headless mode), create one in memory
        if not self._stage:
            try:
                self._stage = Usd.Stage.CreateInMemory("diapedesis.usda")
                carb.log_info("[diapedesis] Created in-memory USD stage")
            except Exception as e:
                carb.log_error(f"[diapedesis] Failed to create stage: {e}")
                self._stage = None

    # ── Simulation / Frame Loading ──────────────────────────────────────

    def load_preset(self, preset_name: str, duration: float = 120.0,
                    fps: float = 30.0) -> bool:
        """Run a simulation preset and collect frames.

        Args:
            preset_name: Key from PRESETS dict
            duration: Simulation duration in seconds
            fps: Frames per second to capture

        Returns:
            True if frames were collected successfully
        """
        if preset_name not in PRESETS:
            carb.log_warn(f"[diapedesis] Unknown preset: {preset_name}")
            return False

        self._preset_name = preset_name
        preset = PRESETS[preset_name]

        carb.log_info(f"[diapedesis] Running preset '{preset_name}': {preset['label']}")

        try:
            frames = self._run_simulation(preset["params"], duration, fps)
        except Exception as e:
            carb.log_error(f"[diapedesis] Simulation failed: {e}")
            return False

        if not frames:
            carb.log_warn("[diapedesis] No frames generated")
            return False

        self._frames = frames
        self._current_frame = 0
        self._frame_interval = 1.0 / fps
        self._scene_built = False

        self._stats["total_frames"] = len(frames)
        self._stats["preset"] = preset_name

        carb.log_info(f"[diapedesis] Collected {len(frames)} frames")
        return True

    def load_frames(self, frames: List[Dict[str, Any]], fps: float = 30.0):
        """Load pre-computed frames directly (for streaming from backend)."""
        self._frames = frames
        self._current_frame = 0
        self._frame_interval = 1.0 / fps
        self._scene_built = False
        self._stats["total_frames"] = len(frames)
        carb.log_info(f"[diapedesis] Loaded {len(frames)} pre-computed frames")

    def _run_simulation(self, params: Dict, duration: float,
                        fps: float) -> List[Dict]:
        """Run DiapedesisSim and collect frame snapshots."""
        # Import here to avoid circular imports and allow Kit to load
        # even if the simulation engine isn't available
        try:
            from cognisom.simulations.diapedesis import DiapedesisSim, DiapedesisConfig
        except ImportError:
            carb.log_warn("[diapedesis] DiapedesisSim not available, "
                          "using mock frames")
            return self._generate_mock_frames(duration, fps)

        # Build config from preset params
        config = DiapedesisConfig()
        for key, value in params.items():
            if hasattr(config, key):
                setattr(config, key, value)

        sim = DiapedesisSim(config)

        dt = config.dt
        capture_interval = 1.0 / fps
        frames = []
        sim_time = 0.0
        next_capture = 0.0

        while sim_time < duration:
            sim.step()
            sim_time += dt

            if sim_time >= next_capture:
                frames.append(sim.get_snapshot())
                next_capture += capture_interval

        carb.log_info(f"[diapedesis] Simulation complete: {len(frames)} frames "
                      f"over {duration}s")
        return frames

    def _generate_mock_frames(self, duration: float,
                              fps: float) -> List[Dict]:
        """Generate mock frames for testing without simulation engine."""
        import math
        import random

        rng = random.Random(42)
        n_frames = int(duration * fps)
        R = 25.0
        L = 200.0
        n_leuko = 15
        n_rbc = 100
        n_endo = 30
        n_bacteria = 6

        # Static positions
        endo_positions = []
        for i in range(n_endo):
            theta = (i / n_endo) * math.pi * 0.8 + math.pi * 0.1
            x = (i % 10) * (L / 10) + L * 0.05
            y = -R * math.cos(theta)
            z = R * math.sin(theta)
            endo_positions.append([x, y, z])

        bacteria_positions = []
        for i in range(n_bacteria):
            bacteria_positions.append([
                L * 0.3 + rng.uniform(0, L * 0.4),
                -R * 1.5 - rng.uniform(0, R * 0.5),
                rng.uniform(-R * 0.3, R * 0.3),
            ])

        frames = []
        for f in range(n_frames):
            t = f / fps
            progress = t / duration

            # Leukocyte positions — flowing right with some dropping to wall
            leuko_pos = []
            leuko_states = []
            integrin_act = []
            trans_prog = []
            radii = []
            for i in range(n_leuko):
                phase = (t + i * 3) % (duration * 0.8)
                x = (phase / duration) * L
                if phase < duration * 0.3:
                    y = rng.uniform(-R * 0.5, R * 0.5)
                    state = 0  # FLOWING
                elif phase < duration * 0.5:
                    y = -R * 0.85 + rng.uniform(-2, 2)
                    state = 2  # ROLLING
                else:
                    y = -R * 0.95
                    state = 4  # ARRESTED
                z = rng.uniform(-R * 0.3, R * 0.3)
                leuko_pos.append([x, y, z])
                leuko_states.append(state)
                integrin_act.append(min(1.0, progress * 2) if state >= 3 else 0.0)
                trans_prog.append(0.0)
                radii.append(6.0)

            # RBC positions
            rbc_pos = []
            for i in range(n_rbc):
                x = ((t * 30 + i * 5) % L)
                y = rng.uniform(-R * 0.7, R * 0.7)
                z = rng.uniform(-R * 0.7, R * 0.7)
                rbc_pos.append([x, y, z])

            # Selectin expression ramps up with inflammation
            selectin_expr = [min(1.0, progress * 3) for _ in range(n_endo)]
            junction_integrity = [max(0.0, 1.0 - progress * 1.5)
                                  for _ in range(n_endo)]

            bacteria_alive = [True] * n_bacteria
            bacteria_phago = [0.0] * n_bacteria
            # Kill bacteria as simulation progresses
            kills = int(progress * n_bacteria * 0.8)
            for bi in range(min(kills, n_bacteria)):
                bacteria_alive[bi] = False
                bacteria_phago[bi] = 1.0

            frames.append({
                "time": t,
                "step": f,
                "leukocyte_positions": leuko_pos,
                "leukocyte_radii": radii,
                "leukocyte_states": leuko_states,
                "integrin_activation": integrin_act,
                "transmigration_progress": trans_prog,
                "rbc_positions": rbc_pos,
                "endo_positions": endo_positions,
                "endo_selectin_expr": selectin_expr,
                "endo_junction_integrity": junction_integrity,
                "vessel_radius": R,
                "vessel_length": L,
                "bacteria_positions": bacteria_positions,
                "bacteria_alive": bacteria_alive,
                "bacteria_phagocytosis": bacteria_phago,
                "leukocyte_target": [-1] * n_leuko,
                "metrics": {
                    "state_counts": {"flowing": n_leuko},
                    "bacteria_alive": sum(bacteria_alive),
                    "bacteria_total": n_bacteria,
                },
            })

        return frames

    # ── Scene Building ──────────────────────────────────────────────────

    def request_build_scene(self):
        """Queue a scene build for the main thread (thread-safe)."""
        self._build_requested = True

    def process_pending(self):
        """Process queued actions — MUST be called from Kit main thread."""
        if self._build_requested:
            self._build_requested = False
            self.build_scene()

    def build_scene(self) -> bool:
        """Build the USD scene from the first frame."""
        if not self._frames:
            carb.log_warn("[diapedesis] No frames loaded")
            return False

        if not self._stage:
            self._init_stage()
        if not self._stage:
            carb.log_warn("[diapedesis] No USD stage available")
            return False

        self._scene_builder = DiapedesisSceneBuilder(self._stage)
        self._scene_builder.build_scene(self._frames[0])
        self._scene_built = True
        self._current_frame = 0

        carb.log_info("[diapedesis] Scene built from first frame")
        return True

    # ── Playback Control ────────────────────────────────────────────────

    def play(self):
        """Start or resume playback."""
        if not self._scene_built:
            if not self.build_scene():
                return
        self._is_playing = True
        self._is_paused = False
        carb.log_info("[diapedesis] Playback started")

    def pause(self):
        """Pause playback."""
        self._is_paused = True
        carb.log_info("[diapedesis] Playback paused")

    def resume(self):
        """Resume playback."""
        self._is_paused = False
        carb.log_info("[diapedesis] Playback resumed")

    def stop(self):
        """Stop playback and reset to frame 0."""
        self._is_playing = False
        self._is_paused = False
        self._current_frame = 0
        self._accumulated_time = 0.0
        carb.log_info("[diapedesis] Playback stopped")

    def seek(self, frame_index: int):
        """Jump to a specific frame."""
        if not self._frames:
            return
        self._current_frame = max(0, min(frame_index, len(self._frames) - 1))
        self._accumulated_time = 0.0
        if self._scene_builder and self._scene_built:
            self._scene_builder.apply_frame(self._frames[self._current_frame])
        self._update_stats()

    def step_forward(self):
        """Advance one frame."""
        if self._current_frame < len(self._frames) - 1:
            self._current_frame += 1
            if self._scene_builder and self._scene_built:
                self._scene_builder.apply_frame(self._frames[self._current_frame])
            self._update_stats()

    def step_backward(self):
        """Go back one frame."""
        if self._current_frame > 0:
            self._current_frame -= 1
            if self._scene_builder and self._scene_built:
                self._scene_builder.apply_frame(self._frames[self._current_frame])
            self._update_stats()

    # ── Per-Frame Update ────────────────────────────────────────────────

    def update(self, dt: float):
        """Called every Kit frame. Advances playback based on elapsed time."""
        if not self._is_playing or self._is_paused:
            return
        if not self._frames or not self._scene_builder:
            return

        t0 = time.perf_counter()

        self._accumulated_time += dt * self._playback_speed

        # Advance frames based on accumulated time
        if self._accumulated_time >= self._frame_interval:
            frames_to_advance = int(self._accumulated_time / self._frame_interval)
            self._accumulated_time -= frames_to_advance * self._frame_interval

            self._current_frame += frames_to_advance

            if self._current_frame >= len(self._frames):
                # Loop or stop at end
                self._current_frame = len(self._frames) - 1
                self._is_playing = False
                carb.log_info("[diapedesis] Playback complete")

            # Apply current frame to scene
            self._scene_builder.apply_frame(self._frames[self._current_frame])

        elapsed = (time.perf_counter() - t0) * 1000
        self._stats["step_time_ms"] = elapsed
        self._stats["fps"] = 1.0 / dt if dt > 0 else 0.0
        self._update_stats()

    def _update_stats(self):
        """Update stats dict from current frame."""
        self._stats["current_frame"] = self._current_frame
        if self._frames and self._current_frame < len(self._frames):
            frame = self._frames[self._current_frame]
            self._stats["time"] = frame.get("time", 0.0)
            self._stats["leukocytes"] = len(frame.get("leukocyte_positions", []))
            self._stats["rbcs"] = len(frame.get("rbc_positions", []))
            metrics = frame.get("metrics", {})
            self._stats["bacteria_alive"] = metrics.get("bacteria_alive",
                sum(1 for a in frame.get("bacteria_alive", []) if a))

    # ── Cleanup ─────────────────────────────────────────────────────────

    def clear(self):
        """Remove scene and free frames."""
        self.stop()
        if self._stage and self._scene_built:
            prim = self._stage.GetPrimAtPath("/World/Diapedesis")
            if prim:
                self._stage.RemovePrim("/World/Diapedesis")
        self._frames = []
        self._scene_builder = None
        self._scene_built = False
        carb.log_info("[diapedesis] Scene cleared")
