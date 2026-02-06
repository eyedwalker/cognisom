"""
Dynamic Payload Manager
======================

Phase B3: Proximity-based payload streaming for massive USD stages.

Handles datasets larger than RAM by:
1. Opening stages with LoadNone (skeleton only)
2. Requiring extent (bounding box) for placeholder rendering
3. Camera-proximity algorithm loads/unloads payloads with hysteresis

The hysteresis buffer prevents load/unload thrashing when the camera
is near the threshold distance.

Usage::

    from cognisom.omniverse.payload_manager import DynamicPayloadManager

    # Create manager with custom distances
    manager = DynamicPayloadManager(
        stage=stage,
        load_distance=1000.0,    # Load when within 1km
        unload_distance=1500.0,  # Unload when beyond 1.5km (hysteresis)
    )

    # Each frame, update with camera position
    manager.update(camera_position)

    # Get loading statistics
    stats = manager.get_stats()
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Try to import USD modules
try:
    from pxr import Gf, Sdf, Usd, UsdGeom
    USD_AVAILABLE = True
except ImportError:
    USD_AVAILABLE = False
    log.warning("USD (pxr) not available - payload manager will use mock mode")


class PayloadState(str, Enum):
    """State of a payload-bearing prim."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"
    ERROR = "error"


class LoadPriority(str, Enum):
    """Priority levels for payload loading."""
    CRITICAL = "critical"   # Must load immediately (in frustum, close)
    HIGH = "high"           # Load soon (approaching)
    NORMAL = "normal"       # Standard queue position
    LOW = "low"             # Background loading
    DEFERRED = "deferred"   # Load only when idle


@dataclass
class PayloadInfo:
    """Information about a payload-bearing prim."""
    path: str
    state: PayloadState = PayloadState.UNLOADED
    priority: LoadPriority = LoadPriority.NORMAL

    # Bounding box (world space)
    bbox_min: Optional[Tuple[float, float, float]] = None
    bbox_max: Optional[Tuple[float, float, float]] = None
    bbox_center: Optional[Tuple[float, float, float]] = None

    # Distance tracking
    last_distance: float = float('inf')
    last_update_time: float = 0.0

    # Loading metrics
    load_start_time: Optional[float] = None
    load_duration: Optional[float] = None
    memory_estimate_mb: float = 0.0

    # Error tracking
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class PayloadManagerConfig:
    """Configuration for the Dynamic Payload Manager."""

    # Distance thresholds (in scene units, typically meters)
    load_distance: float = 1000.0      # Load when camera is within this distance
    unload_distance: float = 1500.0    # Unload when camera exceeds this distance

    # Hysteresis ensures unload_distance > load_distance to prevent thrashing
    # Default provides 50% buffer

    # Priority distance multipliers
    critical_distance_factor: float = 0.3   # Within 30% of load_distance
    high_priority_factor: float = 0.6       # Within 60% of load_distance

    # Performance limits
    max_concurrent_loads: int = 4           # Parallel load operations
    max_loads_per_frame: int = 2            # Limit to prevent stalls
    max_unloads_per_frame: int = 4          # Unloading is cheaper

    # Memory management
    target_memory_mb: float = 4096.0        # Target GPU/CPU memory budget
    emergency_unload_threshold: float = 0.9  # Unload aggressively at 90%

    # Timing
    update_interval_ms: float = 16.67       # ~60fps update rate
    stale_check_interval_s: float = 1.0     # Check for stale payloads

    # Async loading
    enable_async_loading: bool = True
    loading_thread_count: int = 2

    # Debug
    enable_placeholder_rendering: bool = True
    log_load_events: bool = False


@dataclass
class PayloadManagerStats:
    """Statistics for payload manager monitoring."""
    total_payloads: int = 0
    loaded_payloads: int = 0
    loading_payloads: int = 0
    unloaded_payloads: int = 0

    # Memory
    estimated_memory_mb: float = 0.0
    memory_budget_mb: float = 0.0
    memory_utilization: float = 0.0

    # Performance
    loads_this_frame: int = 0
    unloads_this_frame: int = 0
    total_loads: int = 0
    total_unloads: int = 0

    # Timing
    avg_load_time_ms: float = 0.0
    last_update_time_ms: float = 0.0

    # Queue
    pending_loads: int = 0
    pending_unloads: int = 0


class DynamicPayloadManager:
    """
    Proximity-based payload streaming for massive USD stages.

    Manages loading and unloading of USD payloads based on camera distance
    to enable visualization of datasets larger than available memory.

    Key features:
    - Hysteresis buffer to prevent load/unload thrashing
    - Priority-based loading queue
    - Async loading to prevent render stalls
    - Memory budget tracking
    - Placeholder rendering for unloaded payloads

    Parameters
    ----------
    stage : Usd.Stage
        The USD stage to manage (should be opened with LoadNone)
    config : PayloadManagerConfig, optional
        Configuration options
    load_distance : float, optional
        Override config load distance
    unload_distance : float, optional
        Override config unload distance

    Examples
    --------
    >>> stage = Usd.Stage.Open("massive_scene.usda", Usd.Stage.LoadNone)
    >>> manager = DynamicPayloadManager(stage, load_distance=100.0)
    >>>
    >>> # In render loop:
    >>> while rendering:
    ...     camera_pos = get_camera_position()
    ...     manager.update(camera_pos)
    ...     render_frame()
    """

    def __init__(
        self,
        stage: Optional[Any] = None,
        config: Optional[PayloadManagerConfig] = None,
        load_distance: Optional[float] = None,
        unload_distance: Optional[float] = None,
    ):
        self.stage = stage
        self.config = config or PayloadManagerConfig()

        # Override config with explicit parameters
        if load_distance is not None:
            self.config.load_distance = load_distance
        if unload_distance is not None:
            self.config.unload_distance = unload_distance

        # Validate hysteresis
        if self.config.unload_distance <= self.config.load_distance:
            log.warning(
                f"unload_distance ({self.config.unload_distance}) should be > "
                f"load_distance ({self.config.load_distance}) to prevent thrashing. "
                f"Adjusting to {self.config.load_distance * 1.5}"
            )
            self.config.unload_distance = self.config.load_distance * 1.5

        # Payload tracking
        self._payloads: Dict[str, PayloadInfo] = {}
        self._loaded_paths: Set[str] = set()

        # Loading queues (priority-ordered)
        self._load_queue: deque = deque()
        self._unload_queue: deque = deque()

        # Async loading
        self._loading_lock = threading.Lock()
        self._loading_in_progress: Set[str] = set()

        # Statistics
        self._stats = PayloadManagerStats()
        self._load_times: deque = deque(maxlen=100)  # Rolling average

        # Timing
        self._last_update_time = 0.0
        self._last_stale_check = 0.0

        # Camera state
        self._camera_position: Optional[Tuple[float, float, float]] = None
        self._camera_velocity: Optional[Tuple[float, float, float]] = None

        # Initialize if stage provided
        if stage is not None:
            self._discover_payloads()

    def _discover_payloads(self) -> None:
        """Discover all payload-bearing prims in the stage."""
        if not USD_AVAILABLE or self.stage is None:
            return

        self._payloads.clear()

        for prim in self.stage.Traverse():
            # Check if prim has payloads
            if prim.HasPayload():
                path = str(prim.GetPath())

                # Get bounding box if available
                bbox_min, bbox_max, bbox_center = self._get_prim_bounds(prim)

                # Estimate memory based on type/metadata
                memory_est = self._estimate_payload_memory(prim)

                self._payloads[path] = PayloadInfo(
                    path=path,
                    state=PayloadState.UNLOADED,
                    bbox_min=bbox_min,
                    bbox_max=bbox_max,
                    bbox_center=bbox_center,
                    memory_estimate_mb=memory_est,
                )

        self._stats.total_payloads = len(self._payloads)
        log.info(f"Discovered {len(self._payloads)} payload-bearing prims")

    def _get_prim_bounds(
        self,
        prim: Any
    ) -> Tuple[Optional[Tuple[float, float, float]], ...]:
        """Get world-space bounding box for a prim."""
        if not USD_AVAILABLE:
            return None, None, None

        try:
            imageable = UsdGeom.Imageable(prim)
            bbox = imageable.ComputeWorldBound(
                Usd.TimeCode.Default(),
                UsdGeom.Tokens.default_
            )

            if bbox.IsEmpty():
                return None, None, None

            range_box = bbox.ComputeAlignedRange()
            min_pt = range_box.GetMin()
            max_pt = range_box.GetMax()
            center = (min_pt + max_pt) / 2.0

            return (
                (min_pt[0], min_pt[1], min_pt[2]),
                (max_pt[0], max_pt[1], max_pt[2]),
                (center[0], center[1], center[2]),
            )
        except Exception as e:
            log.debug(f"Could not compute bounds for {prim.GetPath()}: {e}")
            return None, None, None

    def _estimate_payload_memory(self, prim: Any) -> float:
        """Estimate memory usage of a payload in MB."""
        # Default estimate based on prim type
        prim_type = prim.GetTypeName()

        # Base estimates by type
        estimates = {
            "Mesh": 10.0,
            "PointInstancer": 50.0,
            "Scope": 1.0,
            "Xform": 1.0,
        }

        base = estimates.get(prim_type, 5.0)

        # Check for custom metadata hints
        if prim.HasCustomDataKey("cognisom:memoryEstimateMB"):
            return float(prim.GetCustomDataByKey("cognisom:memoryEstimateMB"))

        # Scale by child count if available
        try:
            child_count = len(list(prim.GetChildren()))
            base *= (1 + child_count * 0.1)
        except Exception:
            pass

        return base

    def _compute_distance(
        self,
        camera_pos: Tuple[float, float, float],
        payload_info: PayloadInfo
    ) -> float:
        """Compute distance from camera to payload bounding box center."""
        if payload_info.bbox_center is None:
            return float('inf')

        dx = camera_pos[0] - payload_info.bbox_center[0]
        dy = camera_pos[1] - payload_info.bbox_center[1]
        dz = camera_pos[2] - payload_info.bbox_center[2]

        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def _compute_priority(self, distance: float) -> LoadPriority:
        """Determine load priority based on distance."""
        critical_dist = self.config.load_distance * self.config.critical_distance_factor
        high_dist = self.config.load_distance * self.config.high_priority_factor

        if distance < critical_dist:
            return LoadPriority.CRITICAL
        elif distance < high_dist:
            return LoadPriority.HIGH
        elif distance < self.config.load_distance:
            return LoadPriority.NORMAL
        else:
            return LoadPriority.LOW

    def update(
        self,
        camera_position: Tuple[float, float, float],
        camera_direction: Optional[Tuple[float, float, float]] = None,
        dt: Optional[float] = None,
    ) -> None:
        """
        Update payload loading based on camera position.

        Should be called each frame to manage payload streaming.

        Parameters
        ----------
        camera_position : Tuple[float, float, float]
            Current camera world position (x, y, z)
        camera_direction : Tuple[float, float, float], optional
            Camera look direction for frustum culling (not yet implemented)
        dt : float, optional
            Delta time since last update (for velocity prediction)
        """
        current_time = time.time()

        # Update camera state
        if self._camera_position is not None and dt is not None and dt > 0:
            self._camera_velocity = (
                (camera_position[0] - self._camera_position[0]) / dt,
                (camera_position[1] - self._camera_position[1]) / dt,
                (camera_position[2] - self._camera_position[2]) / dt,
            )
        self._camera_position = camera_position

        # Reset per-frame stats
        self._stats.loads_this_frame = 0
        self._stats.unloads_this_frame = 0

        # Process each payload
        loads_needed = []
        unloads_needed = []

        for path, info in self._payloads.items():
            distance = self._compute_distance(camera_position, info)
            info.last_distance = distance
            info.last_update_time = current_time

            if info.state == PayloadState.LOADED:
                # Check if should unload (beyond unload distance)
                if distance > self.config.unload_distance:
                    unloads_needed.append((path, distance))

            elif info.state == PayloadState.UNLOADED:
                # Check if should load (within load distance)
                if distance < self.config.load_distance:
                    priority = self._compute_priority(distance)
                    info.priority = priority
                    loads_needed.append((path, distance, priority))

        # Sort loads by priority then distance
        priority_order = {
            LoadPriority.CRITICAL: 0,
            LoadPriority.HIGH: 1,
            LoadPriority.NORMAL: 2,
            LoadPriority.LOW: 3,
            LoadPriority.DEFERRED: 4,
        }
        loads_needed.sort(key=lambda x: (priority_order[x[2]], x[1]))

        # Sort unloads by distance (furthest first)
        unloads_needed.sort(key=lambda x: -x[1])

        # Process unloads first (free memory)
        for path, _ in unloads_needed[:self.config.max_unloads_per_frame]:
            self._unload_payload(path)

        # Process loads
        for path, _, _ in loads_needed[:self.config.max_loads_per_frame]:
            if len(self._loading_in_progress) < self.config.max_concurrent_loads:
                self._load_payload(path)

        # Update statistics
        self._update_stats()

        # Log timing
        update_duration = (time.time() - current_time) * 1000
        self._stats.last_update_time_ms = update_duration

        if self.config.log_load_events and (self._stats.loads_this_frame > 0 or
                                             self._stats.unloads_this_frame > 0):
            log.info(
                f"Payload update: loaded={self._stats.loads_this_frame}, "
                f"unloaded={self._stats.unloads_this_frame}, "
                f"time={update_duration:.2f}ms"
            )

    def _load_payload(self, path: str) -> bool:
        """Load a single payload."""
        info = self._payloads.get(path)
        if info is None or info.state != PayloadState.UNLOADED:
            return False

        info.state = PayloadState.LOADING
        info.load_start_time = time.time()

        with self._loading_lock:
            self._loading_in_progress.add(path)

        try:
            if USD_AVAILABLE and self.stage is not None:
                sdf_path = Sdf.Path(path)
                self.stage.Load(sdf_path)

            # Mark as loaded
            info.state = PayloadState.LOADED
            info.load_duration = time.time() - info.load_start_time
            self._loaded_paths.add(path)
            self._load_times.append(info.load_duration * 1000)

            self._stats.loads_this_frame += 1
            self._stats.total_loads += 1

            if self.config.log_load_events:
                log.debug(f"Loaded payload: {path} ({info.load_duration*1000:.1f}ms)")

            return True

        except Exception as e:
            info.state = PayloadState.ERROR
            info.error_message = str(e)
            info.retry_count += 1
            log.error(f"Failed to load payload {path}: {e}")
            return False

        finally:
            with self._loading_lock:
                self._loading_in_progress.discard(path)

    def _unload_payload(self, path: str) -> bool:
        """Unload a single payload."""
        info = self._payloads.get(path)
        if info is None or info.state != PayloadState.LOADED:
            return False

        info.state = PayloadState.UNLOADING

        try:
            if USD_AVAILABLE and self.stage is not None:
                sdf_path = Sdf.Path(path)
                self.stage.Unload(sdf_path)

            # Mark as unloaded
            info.state = PayloadState.UNLOADED
            self._loaded_paths.discard(path)

            self._stats.unloads_this_frame += 1
            self._stats.total_unloads += 1

            if self.config.log_load_events:
                log.debug(f"Unloaded payload: {path}")

            return True

        except Exception as e:
            info.state = PayloadState.ERROR
            info.error_message = str(e)
            log.error(f"Failed to unload payload {path}: {e}")
            return False

    def _update_stats(self) -> None:
        """Update statistics."""
        loaded = 0
        loading = 0
        unloaded = 0
        memory = 0.0

        for info in self._payloads.values():
            if info.state == PayloadState.LOADED:
                loaded += 1
                memory += info.memory_estimate_mb
            elif info.state == PayloadState.LOADING:
                loading += 1
            elif info.state == PayloadState.UNLOADED:
                unloaded += 1

        self._stats.loaded_payloads = loaded
        self._stats.loading_payloads = loading
        self._stats.unloaded_payloads = unloaded
        self._stats.estimated_memory_mb = memory
        self._stats.memory_budget_mb = self.config.target_memory_mb
        self._stats.memory_utilization = memory / self.config.target_memory_mb

        if self._load_times:
            self._stats.avg_load_time_ms = sum(self._load_times) / len(self._load_times)

        self._stats.pending_loads = len(self._load_queue)
        self._stats.pending_unloads = len(self._unload_queue)

    def load_all(self) -> None:
        """Force load all payloads (use for small scenes)."""
        for path, info in self._payloads.items():
            if info.state == PayloadState.UNLOADED:
                self._load_payload(path)

    def unload_all(self) -> None:
        """Unload all payloads."""
        for path, info in self._payloads.items():
            if info.state == PayloadState.LOADED:
                self._unload_payload(path)

    def load_within_radius(
        self,
        center: Tuple[float, float, float],
        radius: float
    ) -> int:
        """Load all payloads within a radius of a point."""
        loaded = 0
        for path, info in self._payloads.items():
            if info.state == PayloadState.UNLOADED:
                distance = self._compute_distance(center, info)
                if distance < radius:
                    if self._load_payload(path):
                        loaded += 1
        return loaded

    def preload_path(
        self,
        path_points: List[Tuple[float, float, float]],
        lookahead: float = 100.0
    ) -> None:
        """
        Preload payloads along a camera path.

        Useful for flythrough sequences where camera path is known.

        Parameters
        ----------
        path_points : List[Tuple[float, float, float]]
            Points along the camera path
        lookahead : float
            Distance ahead to preload
        """
        for point in path_points:
            # Create expanded radius for lookahead
            expanded_radius = self.config.load_distance + lookahead

            for path, info in self._payloads.items():
                if info.state == PayloadState.UNLOADED:
                    distance = self._compute_distance(point, info)
                    if distance < expanded_radius:
                        info.priority = LoadPriority.HIGH
                        self._load_queue.append(path)

    def get_stats(self) -> PayloadManagerStats:
        """Get current statistics."""
        return self._stats

    def get_payload_info(self, path: str) -> Optional[PayloadInfo]:
        """Get information about a specific payload."""
        return self._payloads.get(path)

    def get_loaded_paths(self) -> Set[str]:
        """Get set of currently loaded payload paths."""
        return self._loaded_paths.copy()

    def get_nearby_payloads(
        self,
        position: Tuple[float, float, float],
        radius: float
    ) -> List[PayloadInfo]:
        """Get all payloads within a radius of a position."""
        nearby = []
        for info in self._payloads.values():
            distance = self._compute_distance(position, info)
            if distance < radius:
                nearby.append(info)
        return sorted(nearby, key=lambda x: x.last_distance)

    def set_load_distance(self, distance: float) -> None:
        """Update the load distance threshold."""
        self.config.load_distance = distance
        if self.config.unload_distance <= distance:
            self.config.unload_distance = distance * 1.5

    def set_memory_budget(self, budget_mb: float) -> None:
        """Update the memory budget."""
        self.config.target_memory_mb = budget_mb

    def force_load(self, path: str) -> bool:
        """Force load a specific payload regardless of distance."""
        return self._load_payload(path)

    def force_unload(self, path: str) -> bool:
        """Force unload a specific payload regardless of distance."""
        return self._unload_payload(path)

    def register_payload(
        self,
        path: str,
        bbox_center: Tuple[float, float, float],
        bbox_min: Optional[Tuple[float, float, float]] = None,
        bbox_max: Optional[Tuple[float, float, float]] = None,
        memory_estimate_mb: float = 10.0,
    ) -> None:
        """
        Manually register a payload (for custom/dynamic payloads).

        Parameters
        ----------
        path : str
            USD prim path
        bbox_center : Tuple[float, float, float]
            Center of bounding box (for distance calculations)
        bbox_min, bbox_max : Tuple[float, float, float], optional
            Bounding box extents
        memory_estimate_mb : float
            Estimated memory usage
        """
        self._payloads[path] = PayloadInfo(
            path=path,
            bbox_center=bbox_center,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            memory_estimate_mb=memory_estimate_mb,
        )
        self._stats.total_payloads = len(self._payloads)

    def refresh_payloads(self) -> None:
        """Re-scan stage for payloads."""
        self._discover_payloads()


def create_payload_manager(
    stage: Any,
    load_distance: float = 1000.0,
    unload_distance: Optional[float] = None,
    memory_budget_mb: float = 4096.0,
) -> DynamicPayloadManager:
    """
    Convenience function to create a configured payload manager.

    Parameters
    ----------
    stage : Usd.Stage
        USD stage (should be opened with LoadNone)
    load_distance : float
        Distance at which to load payloads
    unload_distance : float, optional
        Distance at which to unload (default: 1.5x load_distance)
    memory_budget_mb : float
        Target memory budget

    Returns
    -------
    DynamicPayloadManager
        Configured manager
    """
    config = PayloadManagerConfig(
        load_distance=load_distance,
        unload_distance=unload_distance or load_distance * 1.5,
        target_memory_mb=memory_budget_mb,
    )
    return DynamicPayloadManager(stage=stage, config=config)


def open_stage_for_streaming(
    usd_path: str,
    load_distance: float = 1000.0,
) -> Tuple[Any, DynamicPayloadManager]:
    """
    Open a USD stage configured for payload streaming.

    Opens the stage with LoadNone and creates a payload manager.

    Parameters
    ----------
    usd_path : str
        Path to USD file
    load_distance : float
        Distance threshold for loading

    Returns
    -------
    Tuple[Usd.Stage, DynamicPayloadManager]
        The stage and its payload manager
    """
    if not USD_AVAILABLE:
        log.warning("USD not available - returning None")
        return None, DynamicPayloadManager()

    # Open with LoadNone for streaming
    stage = Usd.Stage.Open(usd_path, Usd.Stage.LoadNone)

    if stage is None:
        raise ValueError(f"Failed to open stage: {usd_path}")

    manager = create_payload_manager(stage, load_distance=load_distance)

    log.info(
        f"Opened stage for streaming: {usd_path} "
        f"({manager._stats.total_payloads} payloads)"
    )

    return stage, manager
