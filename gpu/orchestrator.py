"""
GPU Orchestrator
================

Wraps simulation module update loops with GPU-accelerated replacements.
Drop-in enhancement: call `accelerate(engine)` after engine.initialize()
and the orchestrator patches the update methods of compatible modules.

Usage:
    from cognisom.gpu.orchestrator import GPUOrchestrator

    engine = SimulationEngine(config)
    engine.register_module("cellular", CellularModule, {...})
    engine.register_module("spatial", SpatialModule, {...})
    engine.register_module("immune", ImmuneModule, {...})
    engine.initialize()

    gpu = GPUOrchestrator()
    gpu.accelerate(engine)  # Patches modules in-place

    engine.run()  # Now runs with GPU acceleration
"""

import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

import numpy as np

from .backend import get_backend, GPUBackend
from .diffusion import diffuse_field
from .cell_ops import CellArrays, update_metabolism_vectorized, detect_death_candidates, detect_division_candidates
from .spatial_ops import pairwise_distances, find_neighbors, immune_detection_batch

logger = logging.getLogger(__name__)


@dataclass
class AccelerationReport:
    """Report on which modules were accelerated."""

    backend: str
    modules_accelerated: List[str] = field(default_factory=list)
    modules_skipped: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"GPU Orchestrator: {self.backend}"]
        if self.modules_accelerated:
            lines.append(f"  Accelerated: {', '.join(self.modules_accelerated)}")
        if self.modules_skipped:
            lines.append(f"  CPU fallback: {', '.join(self.modules_skipped)}")
        return "\n".join(lines)


class GPUOrchestrator:
    """Patches simulation modules with GPU-accelerated update methods.

    The orchestrator inspects the engine's registered modules and
    replaces their `update()` methods with vectorized/GPU versions
    where possible. Original methods are preserved as `_original_update`.
    """

    def __init__(self, force_cpu: bool = False):
        self.backend = get_backend(force_cpu=force_cpu)
        self._patched_modules: Dict[str, Any] = {}

    def accelerate(self, engine) -> AccelerationReport:
        """Patch engine modules with GPU-accelerated versions.

        Args:
            engine: SimulationEngine instance (already initialized).

        Returns:
            AccelerationReport describing what was accelerated.
        """
        report = AccelerationReport(backend=self.backend.summary())

        for name, module in engine.modules.items():
            module_type = type(module).__name__

            if module_type == "SpatialModule":
                self._patch_spatial(module)
                report.modules_accelerated.append(f"spatial ({module_type})")

            elif module_type == "CellularModule":
                self._patch_cellular(module)
                report.modules_accelerated.append(f"cellular ({module_type})")

            elif module_type == "ImmuneModule":
                self._patch_immune(module, engine)
                report.modules_accelerated.append(f"immune ({module_type})")

            else:
                report.modules_skipped.append(f"{name} ({module_type})")

        logger.info(report.summary())
        return report

    # ── Spatial Module Patch ─────────────────────────────────────

    def _patch_spatial(self, module):
        """Replace SpatialField.update() with GPU diffusion solver."""
        module._original_update = module.update

        def gpu_update(dt):
            module.time_since_update += dt
            if module.time_since_update >= module.update_interval:
                for field_obj in module.fields.values():
                    field_obj.concentration = self.backend.to_numpy(
                        diffuse_field(
                            concentration=field_obj.concentration,
                            diffusion_coeff=field_obj.diffusion_coeff,
                            dt=module.time_since_update,
                            resolution=field_obj.resolution,
                            sources=field_obj.sources,
                            sinks=field_obj.sinks,
                        )
                    )
                module.time_since_update = 0.0

        module.update = gpu_update
        self._patched_modules["spatial"] = module

    # ── Cellular Module Patch ────────────────────────────────────

    def _patch_cellular(self, module):
        """Replace per-cell loop with vectorized metabolism."""
        module._original_update = module.update

        def gpu_update(dt):
            if not module.cells:
                return

            # Pack into arrays
            arrays = CellArrays.from_cell_dict(module.cells)

            # Vectorized metabolism
            update_metabolism_vectorized(
                arrays, dt,
                glucose_rate_normal=module.glucose_consumption_normal,
                glucose_rate_cancer=module.glucose_consumption_cancer,
            )

            # Write back metabolic state
            arrays.write_back(module.cells)

            # Division detection (vectorized)
            div_ids = detect_division_candidates(
                arrays,
                division_time_normal=module.division_time_normal,
                division_time_cancer=module.division_time_cancer,
            )
            for cid in div_ids:
                module._divide_cell(int(cid))

            # Death detection (vectorized)
            normal_die, cancer_die = detect_death_candidates(arrays)
            for cid in normal_die:
                module._kill_cell(int(cid), cause="hypoxia")
            for cid in cancer_die:
                module._kill_cell(int(cid), cause="hypoxia")

        module.update = gpu_update
        self._patched_modules["cellular"] = module

    # ── Immune Module Patch ──────────────────────────────────────

    def _patch_immune(self, module, engine):
        """Replace nested detection loop with batch distance computation."""
        module._original_update = module.update

        def gpu_update(dt):
            if not module.cellular_module:
                return

            cancer_cells = {
                cid: cell for cid, cell in module.cellular_module.cells.items()
                if cell.cell_type == "cancer" and cell.alive
            }

            if not cancer_cells:
                # Just patrol
                for ic in module.immune_cells.values():
                    if not ic.in_blood and not ic.activated:
                        module._patrol(ic, dt)
                    ic.position = np.clip(ic.position, [20, 20, 20], [180, 180, 80])
                return

            # Build cancer position / mhc arrays
            cancer_ids = list(cancer_cells.keys())
            cancer_pos = np.array([cancer_cells[cid].position for cid in cancer_ids], dtype=np.float32)
            cancer_mhc = np.array([cancer_cells[cid].mhc1_expression for cid in cancer_ids], dtype=np.float32)

            # Process each immune cell
            inactive = []
            active = []
            for iid, ic in module.immune_cells.items():
                if ic.in_blood:
                    continue
                if ic.activated:
                    active.append((iid, ic))
                else:
                    inactive.append((iid, ic))

            # ── Inactive: batch detection ────────────────────────
            if inactive:
                immune_pos = np.array([ic.position for _, ic in inactive], dtype=np.float32)
                type_map = {"T_cell": 0, "NK_cell": 1, "macrophage": 2}
                immune_types = np.array([type_map.get(ic.cell_type, 2) for _, ic in inactive], dtype=np.int8)

                detections = immune_detection_batch(
                    immune_pos, cancer_pos,
                    detection_radius=10.0,
                    cancer_mhc1=cancer_mhc,
                    immune_types=immune_types,
                )

                for imm_idx, can_idx, dist in detections:
                    iid, ic = inactive[imm_idx]
                    ic.activated = True
                    ic.target_cell_id = cancer_ids[can_idx]
                    module.total_activations += 1
                    module.emit_event("immune_activated", {
                        "immune_id": iid,
                        "immune_type": ic.cell_type,
                        "target_id": cancer_ids[can_idx],
                        "position": ic.position.tolist(),
                    })

                # Patrol non-detected
                detected_set = {inactive[d[0]][0] for d in detections}
                for iid, ic in inactive:
                    if iid not in detected_set:
                        module._patrol(ic, dt)

            # ── Active: chase and kill ───────────────────────────
            for iid, ic in active:
                if ic.target_cell_id in cancer_cells:
                    target = cancer_cells[ic.target_cell_id]
                    direction = target.position - ic.position
                    distance = np.linalg.norm(direction)
                    if distance > 0:
                        direction = direction / distance
                        ic.velocity = direction * ic.speed
                        ic.position += ic.velocity * dt * 0.01
                    if distance < ic.kill_radius:
                        if np.random.random() < module.kill_probability:
                            module._kill_target(ic, target)
                else:
                    ic.activated = False
                    ic.target_cell_id = None

            # Bounds
            for ic in module.immune_cells.values():
                ic.position = np.clip(ic.position, [20, 20, 20], [180, 180, 80])

        module.update = gpu_update
        self._patched_modules["immune"] = module

    # ── Restore original methods ─────────────────────────────────

    def restore(self, engine):
        """Undo acceleration patches, restoring original update methods."""
        for name, module in self._patched_modules.items():
            if hasattr(module, "_original_update"):
                module.update = module._original_update
                delattr(module, "_original_update")
        self._patched_modules.clear()
        logger.info("GPU orchestrator: patches removed")

    def get_report(self) -> Dict[str, Any]:
        """Get current acceleration status."""
        return {
            "backend": self.backend.summary(),
            "has_gpu": self.backend.has_gpu,
            "device": self.backend.device_name,
            "memory_gb": self.backend.device_memory_gb,
            "patched_modules": list(self._patched_modules.keys()),
        }
